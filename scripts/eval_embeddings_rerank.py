from __future__ import annotations

"""
Evaluate retrieval quality with cross-encoder reranking enabled (Recall/MRR@k).

This script mirrors scripts/eval_embeddings.py but applies the two-stage
retrieval (vector Top-N then BGE reranker to Top-K) similar to scripts/query_kb.py.

Usage (Windows cmd):
  set TRANSFORMERS_NO_TORCHVISION=1
  C:\BU_SUPERAGENT\.venv\Scripts\python.exe scripts\eval_embeddings_rerank.py \
    --gold data\gold\gold_qa_eval.jsonl \
    --sig hf:intfloat/multilingual-e5-large-instruct:norm \
    -k 5 --top-n 10 --top-k 5
"""

# ruff: noqa: E402 (allow sys.path bootstrap before imports)
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

# Bootstrap repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.settings import AppSettings, Settings  # noqa: E402
from src.infra.embeddings.factory import build_embeddings  # noqa: E402
from src.infra.rerankers.bge import RerankItem  # noqa: E402
from src.infra.rerankers.factory import get_reranker  # noqa: E402
from src.infra.retrieval.citations import make_doc_short  # noqa: E402
from src.infra.vectorstores.chroma_store import (  # noqa: E402
    ChromaStore,
    collection_name_for,
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


ess_aliases: dict[str, str] = {
    "hf": "huggingface",
    "huggingface": "huggingface",
    "openai": "openai",
    "dummy": "dummy",
}


def _sig_to_config(sig: str) -> EmbeddingConfig:
    from src.core.settings import EmbeddingConfig  # local import to avoid E402

    parts = sig.split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid signature: {sig}")
    prov_key = parts[0].strip().lower()
    provider = ess_aliases.get(prov_key, prov_key)
    if len(parts) == 2:
        model_part = parts[1]
        norm_part = "norm"
    else:
        model_part = ":".join(parts[1:-1])
        norm_part = parts[-1]
    model = model_part.strip()
    if provider == "huggingface" and "/" not in model:
        model = f"sentence-transformers/{model}"
    normalize = str(norm_part).lower().startswith("norm")
    prov_lit: Literal["huggingface", "openai", "dummy"]
    if provider not in ("huggingface", "openai", "dummy"):
        prov_lit = "huggingface"
    else:
        prov_lit = provider  # type: ignore[assignment]
    return EmbeddingConfig(provider=prov_lit, model_name=model, normalize_embeddings=normalize)


def hit_gold(candidate: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    md = candidate.get("metadata", {}) or {}
    cand_doc: str = make_doc_short(md.get("source", "Unknown"))
    try:
        cand_page: int = int(md.get("page", 0) or 0)
    except Exception:
        cand_page = 0
    cand_section: str = str(md.get("section") or "").strip()
    for g in gold:
        g_doc = g["doc"]
        g_page = int(g["page"])
        g_section: str = str(g.get("section") or "").strip().lower()

        # If gold has no section specified, match on doc + page with ±1 tolerance
        if not g_section:
            if cand_doc == g_doc and abs(cand_page - g_page) <= 1:
                return True
        else:
            # When gold specifies a section, require all three to match
            if cand_doc == g_doc and cand_page == g_page and cand_section.lower() == g_section:
                return True
    return False


def recall_at_k(ranked: list[dict[str, Any]], gold: list[dict[str, Any]], k: int) -> float:
    topk = ranked[:k]
    return 1.0 if any(hit_gold(r, gold) for r in topk) else 0.0


def mrr_at_k(ranked: list[dict[str, Any]], gold: list[dict[str, Any]], k: int) -> float:
    for i, r in enumerate(ranked[:k], start=1):
        if hit_gold(r, gold):
            return 1.0 / i
    return 0.0


from src.core.settings import EmbeddingConfig  # safe import after sys.path


def retrieve_with_rerank(
    query: str, emb_cfg: EmbeddingConfig, top_n: int, top_k: int
) -> list[dict[str, Any]]:
    app = AppSettings()
    emb = build_embeddings(emb_cfg)

    # Collection name scoped by signature
    collection = collection_name_for(app.kb.collection_base, emb_cfg.signature)

    # E5 dedicated persist dir to avoid collisions with legacy stores
    persist_dir = (
        Path("vector_store/e5_large")
        if "intfloat/multilingual-e5" in (emb_cfg.model_name or "").lower()
        else Path(app.kb.persist_directory)
    )

    store = ChromaStore(collection, persist_dir, emb)
    md_filter: dict[str, Any] = {"embedding_sig": emb_cfg.signature}

    # initial vector recall
    docs_scores = store.query_with_scores(query, k=top_n, filter=md_filter)
    items: list[RerankItem] = []
    for d, sc in docs_scores:
        meta = dict(getattr(d, "metadata", {}) or {})
        meta["vector_score"] = float(sc)
        items.append(RerankItem(text=(getattr(d, "page_content", "") or ""), metadata=meta))

    if not items:
        return []

    # build reranker
    reranker = get_reranker()
    if reranker is None:
        # fail open to vector order
        items.sort(key=lambda x: x.metadata.get("vector_score", 0.0), reverse=True)
        kept = items[:top_k]
    else:
        kept = reranker.rerank(query=query, items=items, top_k=top_k)

    # normalize to dicts for metric helpers
    ranked: list[dict[str, Any]] = []
    for it in kept:
        md = dict(it.metadata)
        # ensure source/page/section for matching
        ranked.append(
            {
                "text": it.text,
                "metadata": {
                    "source": md.get("source") or md.get("title") or "Unknown",
                    "page": (
                        md.get("page")
                        if md.get("page") is not None
                        else (int(md.get("page_index", 0)) + 1)
                    ),
                    "section": md.get("section") or "",
                },
            }
        )
    return ranked


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate reranked retrieval against gold JSONL")
    ap.add_argument("--gold", default="data/gold/gold_qa_eval.jsonl")
    ap.add_argument("--sig", default="hf:intfloat/multilingual-e5-large-instruct:norm")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()

    rows = _read_jsonl(args.gold)
    emb_cfg = _sig_to_config(args.sig)

    rcfg = Settings().reranker
    top_n = int(args.top_n if args.top_n is not None else getattr(rcfg, "initial_top_n", 10))
    top_k = int(args.top_k if args.top_k is not None else getattr(rcfg, "final_top_k", args.k))

    recalls: list[float] = []
    mrrs: list[float] = []

    for row in rows:
        q = row["question"]
        gold = row["gold"]
        ranked = retrieve_with_rerank(q, emb_cfg, top_n, top_k)
        recalls.append(recall_at_k(ranked, gold, args.k))
        mrrs.append(mrr_at_k(ranked, gold, args.k))

    out = {
        f"recall@{args.k}": sum(recalls) / max(1, len(recalls)),
        f"mrr@{args.k}": sum(mrrs) / max(1, len(mrrs)),
    }

    print("\n=== RERANK RESULTS ===")
    print(out)


if __name__ == "__main__":
    main()
