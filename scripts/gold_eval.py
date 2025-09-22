from __future__ import annotations

# ruff: noqa: E402  (allow sys.path bootstrap before imports)
import argparse
import json
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Bootstrap repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document

from src.core.settings import AppSettings, EmbeddingConfig
from src.infra.embeddings.factory import build_embeddings
from src.infra.utils.citations import make_doc_short
from src.infra.vectorstores.chroma_store import (
    ChromaStore,
    collection_name_for,
)


@dataclass
class Metrics:
    recall_at_k: float
    mrr: float
    ndcg: float


def dcg(scores: list[int]) -> float:
    return sum(s / math.log2(i + 2) for i, s in enumerate(scores))


def ndcg_at_k(relevances: list[int], k: int) -> float:
    rel_k = relevances[:k]
    ideal = sorted(rel_k, reverse=True)
    idcg = dcg(ideal)
    if idcg == 0:
        return 0.0
    return dcg(rel_k) / idcg


def mrr_at_k(hits: list[bool]) -> float:
    for i, ok in enumerate(hits, start=1):
        if ok:
            return 1.0 / i
    return 0.0


def ensure_store(cfg_app: AppSettings, cfg_emb: EmbeddingConfig) -> ChromaStore:
    emb = build_embeddings(cfg_emb)
    collection = collection_name_for(cfg_app.kb.collection_base, cfg_emb.signature)
    return ChromaStore(collection, Path(cfg_app.kb.persist_directory), emb)


def ingest_docs(store: ChromaStore, signature: str, docs: Iterable[Document]) -> None:
    payload: list[Document] = []
    for d in docs:
        d.metadata = {**(d.metadata or {}), "embedding_sig": signature, "source": "gold", "page": 1}
        payload.append(d)
    store.add_documents(payload)


def eval_embeddings(
    models: list[EmbeddingConfig], gold: list[dict], k: int
) -> list[tuple[str, Metrics]]:
    app = AppSettings()
    results: list[tuple[str, Metrics]] = []

    # Build docs corpus from gold answers
    corpus_docs = [Document(page_content=item["answer"], metadata={}) for item in gold]

    for emb_cfg in models:
        store = ensure_store(app, emb_cfg)
        ingest_docs(store, emb_cfg.signature, corpus_docs)

        recalls = 0
        mrrs = 0.0
        ndcgs = 0.0
        total = 0

        for item in gold:
            q = item["question"]
            # Gold references: list of acceptable (doc, page, section)
            refs = item.get("gold") or []
            # Query
            docs = store.query(q, k=k, filter={"embedding_sig": emb_cfg.signature})
            # Map retrieved docs to citation-like tuples for matching
            retrieved = []
            for d in docs:
                md: dict[str, Any] = getattr(d, "metadata", {}) or {}
                doc_short = make_doc_short(md.get("source"), md.get("title"))
                page_val: Any = (
                    md.get("page") if md.get("page") is not None else md.get("page_number")
                )
                page: int | None
                if page_val is not None:
                    try:
                        page = int(page_val)
                    except Exception:
                        page = None
                else:
                    pi = md.get("page_index")
                    page = (int(pi) + 1) if isinstance(pi, int) else None
                section = str(md.get("section") or "")
                retrieved.append((str(doc_short), page, section))

            # Build set of acceptable gold tuples (doc_short, page, section) with flexible matching:
            # - page match exact when provided, else ignore
            # - section match substring-insensitive if provided
            def _is_match(ret: tuple[str, int | None, str], g: dict) -> bool:
                r_doc, r_page, r_sec = ret
                g_doc = str(g.get("doc") or "").strip()
                g_page = g.get("page")
                g_sec = str(g.get("section") or "").strip()
                if r_doc != g_doc:
                    return False
                if g_page is not None and r_page != int(g_page):
                    return False
                if g_sec and g_sec.lower() not in r_sec.lower():
                    return False
                return True

            hit_flags = [any(_is_match(r, g) for g in refs) for r in retrieved]
            recalls += 1 if any(hit_flags) else 0

            # MRR@k
            mrrs += mrr_at_k(hit_flags)

            # NDCG@k with binary relevance
            ndcgs += ndcg_at_k([1 if h else 0 for h in hit_flags], k)

            total += 1

        metrics = Metrics(
            recall_at_k=recalls / max(1, total),
            mrr=mrrs / max(1, total),
            ndcg=ndcgs / max(1, total),
        )
        results.append((emb_cfg.model_name, metrics))

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Gold-Standard Evaluation (Recall@k/MRR/NDCG)")
    ap.add_argument(
        "--gold",
        required=True,
        help="Path to JSON gold file: [{question, answer, expect_substr?}, ...]",
    )
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument(
        "--accept-recall", type=float, default=0.85, help="Acceptance threshold for Recall@k"
    )
    args = ap.parse_args()

    def _read_gold(path: Path) -> list[dict[str, Any]]:
        if str(path).lower().endswith(".jsonl"):
            items: list[dict[str, Any]] = []
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        items.append(obj)
            return items
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
            if not isinstance(data, list):
                raise ValueError("Gold file must be a list (JSON) or JSONL file")
            out: list[dict[str, Any]] = []
            for obj in data:
                if isinstance(obj, dict):
                    out.append(obj)
            return out

    gold = _read_gold(Path(args.gold))

    # Define the two HF models
    a = EmbeddingConfig(provider="huggingface", model_name="sentence-transformers/all-MiniLM-L6-v2")
    b = EmbeddingConfig(
        provider="huggingface",
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )

    results = eval_embeddings([a, b], gold, args.k)

    print("\n=== Gold Eval (k=%d) ===" % args.k)
    for name, m in results:
        print(
            f"{name:<55} | Recall@{args.k}={m.recall_at_k:.2%} | MRR={m.mrr:.3f} | NDCG={m.ndcg:.3f}"
        )

    # Acceptance: mpnet must meet recall threshold
    mpnet = next((m for name, m in results if "mpnet" in name), None)
    if mpnet is None:
        print("\n[WARN] mpnet result missing; acceptance not evaluated")
        sys.exit(0)

    if mpnet.recall_at_k < args.accept_recall:
        print(
            f"\n[FAIL] mpnet Recall@{args.k} {mpnet.recall_at_k:.2%} < {args.accept_recall:.0%}. Consider tuning chunking or model."
        )
        sys.exit(1)

    print("\n[OK] Acceptance met for mpnet.")
    sys.exit(0)


if __name__ == "__main__":
    main()
