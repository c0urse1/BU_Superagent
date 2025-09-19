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
    parser = argparse.ArgumentParser(description="Query the KB with configured embeddings.")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=4)
    args = parser.parse_args()

    cfg = AppSettings()  # defaults; wire pydantic-settings if you want .env loading
    emb = build_embeddings(cfg.embeddings)
    collection = collection_name_for(cfg.kb.collection_base, cfg.embeddings.signature)

    store = ChromaStore(
        collection,
        Path(cfg.kb.persist_directory),
        emb,
    )

    docs = store.query(
        args.query,
        k=args.k,
        filter={"embedding_sig": cfg.embeddings.signature},
    )
    for i, d in enumerate(docs, 1):
        meta = getattr(d, "metadata", None) or {}
        src = meta.get("source", "<unknown>")
        page = meta.get("page", meta.get("page_number", ""))
        content = (d.page_content or "").replace("\n", " ")
        print(f"[{i}] {src}:{page} -> {content[:200]}")


if __name__ == "__main__":
    main()
