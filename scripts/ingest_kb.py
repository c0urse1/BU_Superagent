from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main() -> None:
    # Ensure the repository root (containing the 'src' package) is importable when run directly
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # Local imports after path bootstrap to keep module-level imports clean (fixes E402)
    from src.bu_kb.ingest.loaders import PdfLoader
    from src.bu_kb.ingest.pipeline import IngestionPipeline
    from src.bu_kb.ingest.splitters import TextSplitterAdapter
    from src.core.settings import AppSettings
    from src.infra.embeddings.factory import build_embeddings
    from src.infra.splitting.factory import build_splitter
    from src.infra.vectorstores.chroma_store import ChromaStore, collection_name_for

    ap = argparse.ArgumentParser(description="Ingest PDFs into Chroma with enriched metadata.")
    ap.add_argument("--source", default="data/pdfs", help="Folder with PDFs (recursive)")
    # Embedding overrides to avoid heavyweight downloads locally
    ap.add_argument(
        "--provider",
        choices=["huggingface", "openai", "dummy"],
        help="Embeddings provider override (default from AppSettings)",
    )
    ap.add_argument("--model", help="Embedding model name override")
    ap.add_argument(
        "--device",
        help='Device override, e.g. "cpu", "cuda", "cuda:0", "mps" (default from AppSettings)',
    )
    ap.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable embedding vector normalization (defaults to enabled)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Build embeddings and collection name from core settings
    cfg = AppSettings()
    # Apply optional overrides
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
    persist_dir = Path(cfg.kb.persist_directory)

    # Vector store (write path)
    store = ChromaStore(collection, persist_dir, emb)

    # Build splitter from chunking config (defaults to sentence-aware)
    mode = getattr(cfg.chunking, "mode", "sentence_aware")
    splitter_impl = build_splitter(
        mode=mode,  # type: ignore[arg-type]
        chunk_size=int(cfg.chunking.chunk_size),
        chunk_overlap=int(cfg.chunking.chunk_overlap),
        max_overflow=int(cfg.chunking.chunk_max_overflow),
        min_merge_char_len=int(cfg.chunking.chunk_min_merge_char_len),
    )

    # Files to ingest
    src_dir = Path(args.source)
    if not src_dir.exists():
        raise SystemExit(f"source directory not found: {src_dir}")
    pdfs = sorted([p for p in src_dir.rglob("*.pdf") if p.is_file()])
    if not pdfs:
        print(f"No PDFs found under {src_dir}")
        return

    pipeline = IngestionPipeline(
        loader=PdfLoader(),
        splitter=TextSplitterAdapter(splitter_impl),
        store=store,
        embedding_signature=emb_cfg.signature,
    )

    total = pipeline.run(pdfs)
    print(
        f"OK: {total} chunks → {persist_dir} (collection={collection}, embedding_sig={emb_cfg.signature})"
    )


if __name__ == "__main__":
    main()
