from __future__ import annotations

import argparse
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
