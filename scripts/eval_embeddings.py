from __future__ import annotations

import argparse
import json

# Ensure repository root on path when run directly
import sys
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.settings import EmbeddingConfig  # noqa: E402
from src.infra.retrieval.citations import make_doc_short  # noqa: E402
from src.infra.retrieval.retriever import retrieve  # noqa: E402


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
}


def _sig_to_config(sig: str) -> EmbeddingConfig:
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
    cand = {
        "doc": make_doc_short(md.get("source", "Unknown")),
        "page": int(md.get("page", 0) or 0),
        "section": (md.get("section") or "").strip(),
    }
    for g in gold:
        if (
            cand["doc"] == g["doc"]
            and cand["page"] == int(g["page"])
            and str(cand.get("section") or "").lower()
            == str(g.get("section") or "").strip().lower()
        ):
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


def eval_model(
    sig: str, gold_path: str, k_values: tuple[int, ...] = (3, 5, 10)
) -> dict[str, float]:
    recalls: dict[int, list[float]] = {k: [] for k in k_values}
    mrrs: dict[int, list[float]] = {k: [] for k in k_values}

    rows = _read_jsonl(gold_path)
    emb_cfg = _sig_to_config(sig)

    for row in rows:
        q = row["question"]
        gold = row["gold"]
        # retrieve must choose the correct collection/embedding under the hood.
        ranked = retrieve(q, k=max(k_values), embeddings=emb_cfg)

        for k in k_values:
            recalls[k].append(recall_at_k(ranked, gold, k))
            mrrs[k].append(mrr_at_k(ranked, gold, k))

    out: dict[str, float] = {}
    for k in k_values:
        out[f"recall@{k}"] = sum(recalls[k]) / max(1, len(recalls[k]))
        out[f"mrr@{k}"] = sum(mrrs[k]) / max(1, len(mrrs[k]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate embeddings against gold JSONL (Recall/MRR)")
    ap.add_argument("--gold", default="data/gold/gold_qa.jsonl")
    ap.add_argument("--sig-mpnet", default="hf:paraphrase-multilingual-mpnet-base-v2:norm")
    ap.add_argument("--sig-minilm", default="hf:all-MiniLM-L6-v2:norm")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    mpnet = eval_model(args.sig_mpnet, args.gold, k_values=(3, 5, 10))
    minilm = eval_model(args.sig_minilm, args.gold, k_values=(3, 5, 10))

    print("\n=== RESULTS ===")
    print("mpnet :", mpnet)
    print("MiniLM:", minilm)

    # Acceptance gate: mpnet recall@5 >= 0.85
    if mpnet.get("recall@5", 0.0) < 0.85:
        print("\n[WARN] mpnet recall@5 below 0.85 — consider improving embeddings/data.")
    else:
        print("\n[OK] mpnet recall@5 meets >= 0.85")

    # Optional: Markdown snippet for quick sharing
    def md_table(results: dict, label: str) -> str:
        cols = ["recall@3", "recall@5", "recall@10", "mrr@3", "mrr@5", "mrr@10"]
        row = " | ".join(f"{results.get(c, 0):.3f}" for c in cols)
        sep = "|".join(["---"] * len(cols))
        return f"### {label}\n\n| {' | '.join(cols)} |\n|{sep}|\n| {row} |\n"

    print("\n--- Markdown Summary ---\n")
    print(md_table(mpnet, "mpnet"))
    print(md_table(minilm, "MiniLM"))


if __name__ == "__main__":
    main()
