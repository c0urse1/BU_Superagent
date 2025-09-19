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
    args = parser.parse_args()

    cfg = AppSettings()  # defaults; wire pydantic-settings if you want .env loading
    emb = build_embeddings(cfg.embeddings)
    collection = collection_name_for(cfg.kb.collection_base, cfg.embeddings.signature)

    store = ChromaStore(
        collection,
        Path(cfg.kb.persist_directory),
        emb,
    )

    # Build filter dict
    md_filter: dict[str, object] = {"embedding_sig": cfg.embeddings.signature}
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
