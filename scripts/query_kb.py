from __future__ import annotations

import argparse
import math
from pathlib import Path

from src.core.settings import AppSettings
from src.infra.embeddings.factory import build_embeddings
from src.infra.vectorstores.chroma_store import (
    ChromaStore,
    collection_name_for,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query KB with metadata filters.")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--category")
    parser.add_argument("--section")
    parser.add_argument("--no-dedup", action="store_true", help="Disable retrieval-time dedup")
    parser.add_argument(
        "--dedup-threshold", type=float, help="Cosine similarity threshold for dedup (0..1)"
    )
    # Embedding overrides to align with a specific collection/signature
    parser.add_argument(
        "--provider",
        choices=["huggingface", "openai", "dummy"],
        help="Embeddings provider override (default from AppSettings)",
    )
    parser.add_argument("--model", help="Embedding model name override")
    parser.add_argument(
        "--device",
        help='Device override, e.g. "cpu", "cuda", "cuda:0", "mps" (default from AppSettings)',
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable embedding vector normalization (defaults to enabled)",
    )
    args = parser.parse_args()

    cfg = AppSettings()  # defaults; wire pydantic-settings if you want .env loading
    # Apply optional overrides to embeddings config
    emb_cfg = cfg.embeddings.model_copy()
    if args.provider:
        emb_cfg.provider = args.provider
    if args.model:
        emb_cfg.model_name = args.model
    if args.device:
        emb_cfg.device = args.device
    if args.no_normalize:
        emb_cfg.normalize_embeddings = False

    # Apply dedup flags
    if args.no_dedup:
        cfg.dedup_query.enabled = False
    if args.dedup_threshold:
        cfg.dedup_query.similarity_threshold = float(args.dedup_threshold)

    emb = build_embeddings(emb_cfg)
    collection = collection_name_for(cfg.kb.collection_base, emb_cfg.signature)

    store = ChromaStore(
        collection,
        Path(cfg.kb.persist_directory),
        emb,
    )

    # Build filter dict
    md_filter: dict[str, object] = {"embedding_sig": emb_cfg.signature}
    if args.category:
        md_filter["category"] = args.category
    if args.section:
        md_filter["section"] = args.section

    docs = store.query(args.query, k=args.k, filter=md_filter)

    # Optional retrieval-time dedup (cosine default; falls back to exact if embedding unavailable)
    if getattr(cfg.dedup_query, "enabled", True) and len(docs) > 1:
        method = str(getattr(cfg.dedup_query, "method", "cosine")).lower()
        thr = float(getattr(cfg.dedup_query, "similarity_threshold", 0.95))

        def _cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b, strict=False))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(y * y for y in b)) or 1.0
            return dot / (na * nb)

        kept = []
        kept_vecs: list[list[float]] = []
        seen_norms: set[str] = set()
        for d in docs:
            txt = (d.page_content or "").strip()
            if not txt:
                kept.append(d)
                continue

            if method == "exact":
                norm = txt.lower().replace("\n", " ").strip()
                if norm in seen_norms:
                    continue
                seen_norms.add(norm)
                kept.append(d)
                continue

            vec = None
            try:
                vec = emb.embed_query(txt)
            except Exception:
                vec = None
            if vec is None:
                norm = txt.lower().replace("\n", " ").strip()
                if norm in seen_norms:
                    continue
                seen_norms.add(norm)
                kept.append(d)
                continue

            if any(_cosine(vec, kv) >= thr for kv in kept_vecs):
                continue
            kept.append(d)
            kept_vecs.append(vec)

        docs = kept[: args.k]
    for i, d in enumerate(docs, 1):
        m = d.metadata or {}
        title = m.get("title", "<no-title>")
        section = m.get("section", "")
        page = m.get("page", m.get("page_number", "?"))
        src = m.get("source", "")
        print(f"[{i}] {title} | {section} (p.{page}) | {src}")
        print("     ", (d.page_content or "").replace("\n", " ")[:200], "…")


if __name__ == "__main__":
    main()
