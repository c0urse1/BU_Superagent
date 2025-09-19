#!/usr/bin/env python3
"""Ingest PDFs using the infra/core layer (device-aware embeddings)."""

from __future__ import annotations

import logging
from pathlib import Path

from src.core.settings import AppSettings
from src.infra.embeddings.factory import build_embeddings
from src.infra.splitting.factory import build_splitter
from src.infra.vectorstores.chroma_store import ChromaStore, collection_name_for
from src.bu_kb.ingest.loaders import PdfLoader
from src.bu_kb.ingest.pipeline import IngestionPipeline
from src.bu_kb.ingest.splitters import TextSplitterAdapter
from src.bu_kb.logging_setup import setup_logging


def main() -> None:
    """Run ingestion using AppSettings (supports env vars like EMBEDDINGS__DEVICE=cuda)."""
    setup_logging(logging.INFO)
    log = logging.getLogger(__name__)

    # Load settings (supports .env and env vars with proper prefixes)
    cfg = AppSettings()
    
    # Build device-aware embeddings
    emb = build_embeddings(cfg.embeddings)
    
    # Resolve device for logging (similar to CLI approach)
    try:
        from src.infra.embeddings.device import resolve_device
        resolved_device = resolve_device(cfg.embeddings.device)
    except Exception:
        resolved_device = cfg.embeddings.device

    log.info(
        "[ingest] embeddings=%s provider=%s device=%s batch_size=%s", 
        cfg.embeddings.model_name,
        cfg.embeddings.provider,
        resolved_device,
        cfg.embeddings.batch_size,
    )
    
    # Create model-scoped collection name
    collection = collection_name_for(cfg.kb.collection_base, cfg.embeddings.signature)
    
    # Set up store
    persist_dir = Path(cfg.kb.persist_directory)
    store = ChromaStore(collection, persist_dir, emb)
    
    # Build splitter
    splitter_impl = build_splitter(
        mode=cfg.chunking.mode,  # type: ignore[arg-type] # pydantic validates this
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
        max_overflow=cfg.chunking.chunk_max_overflow,
        min_merge_char_len=cfg.chunking.chunk_min_merge_char_len,
    )
    
    # Set up pipeline with embedding signature for metadata stamping
    pipeline = IngestionPipeline(
        loader=PdfLoader(),
        splitter=TextSplitterAdapter(splitter_impl),
        store=store,
        embedding_signature=cfg.embeddings.signature,
    )
    
    # Find PDFs (default: data/pdfs)
    source_dir = Path("data/pdfs")
    if not source_dir.exists():
        log.error("PDF source directory not found: %s", source_dir)
        return
        
    pdfs = sorted([p for p in source_dir.rglob("*.pdf") if p.is_file()])
    if not pdfs:
        log.warning("No PDFs found in %s", source_dir)
        return
        
    log.info("Found %d PDF files in %s", len(pdfs), source_dir)
    
    # Run ingestion
    count = pipeline.run(pdfs)
    log.info("✅ Ingested %d chunks → %s (collection=%s)", count, persist_dir, collection)


if __name__ == "__main__":
    main()