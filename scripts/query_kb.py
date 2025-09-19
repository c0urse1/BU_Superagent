from __future__ import annotations

import argparse
from pathlib import Path

try:
    from chromadb.config import Settings as ChromaSettings
except Exception:  # pragma: no cover
    ChromaSettings = None

# Embeddings
try:
    # prefer modern package
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:  # pragma: no cover
    try:
        # community fallback
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception:  # pragma: no cover
        # older langchain
        from langchain.embeddings import HuggingFaceEmbeddings

# Adjust import path to match your project structure:
from bu_kb.ingest.store import ChromaStore  # existing store wrapper in this repo


def main() -> None:
    p = argparse.ArgumentParser(description="Query the BU knowledge base (Chroma).")
    p.add_argument("--persist-dir", required=True, help="Path to Chroma persist directory")
    p.add_argument("--collection", default="bu_knowledge", help="Chroma collection name")
    p.add_argument(
        "--model", default="sentence-transformers/all-MiniLM-L6-v2", help="HF embedding model"
    )
    p.add_argument("--device", default="cpu", help="cpu|cuda")
    p.add_argument("-k", type=int, default=4, help="number of hits")
    p.add_argument("query", help="natural-language query")
    args = p.parse_args()

    emb = HuggingFaceEmbeddings(
        model_name=args.model,
        model_kwargs={"device": args.device},
        encode_kwargs={"normalize_embeddings": True},
    )

    # settings is unused in our wrapper, but kept for parity with other setups
    _settings = ChromaSettings(anonymized_telemetry=False) if ChromaSettings else None

    store = ChromaStore(
        collection=args.collection,
        persist_dir=Path(args.persist_dir).expanduser(),
        embedder=emb,
    )

    docs = store.query(args.query, k=args.k)
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source", "<unknown>")
        page = meta.get("page", meta.get("page_number", ""))
        snippet = (d.page_content or "").replace("\n", " ")[:200]
        print(f"[{i}] {src}:{page}  {snippet}")


if __name__ == "__main__":
    main()
