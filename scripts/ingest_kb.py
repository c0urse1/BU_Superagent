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
    # Adaptive chunking & context flags (non-breaking; defaults taken from AppSettings)
    ap.add_argument("--chunk-target", type=int, default=None, help="Target chunk length in chars")
    ap.add_argument("--chunk-min", type=int, default=None, help="Minimum chunk length in chars")
    ap.add_argument("--chunk-max", type=int, default=None, help="Maximum chunk length in chars")
    ap.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Overlap between chunks in chars (overrides AppSettings.chunking.chunk_overlap)",
    )

    ap.add_argument("--section-inject", action="store_true", default=None)
    ap.add_argument("--no-section-inject", dest="section_inject", action="store_false")

    ap.add_argument("--cross-page-merge", action="store_true", default=None)
    ap.add_argument("--no-cross-page-merge", dest="cross_page_merge", action="store_false")

    ap.add_argument("--sentence-boundaries", action="store_true", default=None)
    ap.add_argument("--no-sentence-boundaries", dest="sentence_boundaries", action="store_false")
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

    # Apply adaptive chunking/context overrides to settings (if provided)
    if args.chunk_target is not None:
        cfg.adaptive_chunking.chunk_target_chars = int(args.chunk_target)
        # For legacy splitter, also map to chunk_size
        cfg.chunking.chunk_size = int(args.chunk_target)
    if args.chunk_min is not None:
        cfg.adaptive_chunking.chunk_min_chars = int(args.chunk_min)
    if args.chunk_max is not None:
        cfg.adaptive_chunking.chunk_max_chars = int(args.chunk_max)
    if args.chunk_overlap is not None:
        cfg.adaptive_chunking.chunk_overlap_chars = int(args.chunk_overlap)
        cfg.chunking.chunk_overlap = int(args.chunk_overlap)
    if args.section_inject is not None:
        cfg.section_context.inject_section_title = bool(args.section_inject)
    if args.cross_page_merge is not None:
        cfg.section_context.cross_page_merge = bool(args.cross_page_merge)
    if getattr(args, "sentence_boundaries", None) is not None:
        cfg.adaptive_chunking.enforce_sentence_boundaries = bool(args.sentence_boundaries)

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
