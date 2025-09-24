from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.core.settings import AppSettings, EmbeddingConfig
from src.infra.embeddings.factory import build_embeddings
from src.infra.loaders.pdf_smart_loader import load_pdf_with_metadata
from src.infra.splitting.factory import build_splitter
from src.infra.vectorstores.chroma_store import (
    ChromaStore,
    collection_name_for,
    get_collection_for_sig,
)

log = logging.getLogger(__name__)


def normalize_metadata(md: dict) -> dict:
    """Normalize per-chunk metadata for citation-ready fields.

    Ensures:
    - source: prefer metadata.source, fall back to title, else "Unknown"
    - page: 1-based integer (if only page_index is present, convert to 1-based)
    - section: string (may be empty)
    - category: string (may be empty)
    """
    md = (md or {}).copy()
    source = md.get("source") or md.get("title") or "Unknown"
    if md.get("page") is not None:
        try:
            raw = md.get("page")
            page = int(raw if raw is not None else 0)
        except Exception:
            page = 0
    else:
        try:
            rawi = md.get("page_index", 0)
            page = int(rawi if rawi is not None else 0) + 1
        except Exception:
            page = 1
    section = md.get("section") or ""
    category = md.get("category") or ""
    return {
        "source": source,
        "page": page,
        "section": section,
        "category": category,
    }


def retrieve(
    query: str,
    *,
    k: int = 5,
    category: str | None = None,
    section: str | None = None,
    embeddings: EmbeddingConfig | None = None,
    embedding_sig: str | None = None,
) -> list[dict]:
    """Retrieve chunks as dicts with text + normalized metadata, ready for prompting.

    Returns a list like: [{"text": str, "metadata": {source, page, section, category}}, ...]
    """
    app = AppSettings()
    emb_cfg = embeddings or app.embeddings
    emb = build_embeddings(emb_cfg)
    # Route to collection by explicit signature if provided, else by emb_cfg.signature
    if embedding_sig:
        collection = get_collection_for_sig(embedding_sig)
        md_sig = embedding_sig
    else:
        collection = collection_name_for(app.kb.collection_base, emb_cfg.signature)
        md_sig = emb_cfg.signature

    # Use dedicated persist dir for E5 embeddings to avoid index clashes (1024-dim)
    persist_dir = (
        Path("vector_store/e5_large")
        if "intfloat/multilingual-e5" in (emb_cfg.model_name or "").lower()
        else Path(app.kb.persist_directory)
    )
    store = ChromaStore(collection, persist_dir, emb)

    md_filter: dict[str, Any] = {"embedding_sig": md_sig}
    if category:
        md_filter["category"] = category
    if section:
        md_filter["section"] = section

    docs = store.query(query, k=k, filter=md_filter)
    try:
        log.info("retrieval.stage1.vector_hits=%d", len(docs))
    except Exception:
        pass
    chunks: list[dict] = []
    for d in docs:
        chunks.append(
            {
                "text": (getattr(d, "page_content", "") or ""),
                "metadata": normalize_metadata(getattr(d, "metadata", {}) or {}),
            }
        )
    try:
        # No dedup here; mirror vector count
        log.info("retrieval.stage1.after_dedup=%d", len(docs))
        log.info("retrieval.final_hits=%d", len(chunks))
    except Exception:
        pass
    return chunks


def load_pdf_and_chunk(path: str | Path) -> list[dict]:
    """Load a PDF and produce sentence-aware chunks as dicts.

    Returns a list like: [{"text": str, "metadata": {...}}, ...]
    Uses AppSettings().chunking to configure the sentence-aware chunker.
    """
    # Normalize path and load per-page base docs with enriched metadata
    p = Path(path)
    docs = load_pdf_with_metadata(p)

    # Build splitter from core settings (sentence-aware defaults)
    app = AppSettings()
    splitter = build_splitter(
        mode=getattr(app.chunking, "mode", "sentence_aware"),
        chunk_size=int(getattr(app.chunking, "chunk_size", 1000)),
        chunk_overlap=int(getattr(app.chunking, "chunk_overlap", 150)),
        max_overflow=int(getattr(app.chunking, "chunk_max_overflow", 200)),
        min_merge_char_len=int(getattr(app.chunking, "chunk_min_merge_char_len", 500)),
    )

    out_docs = splitter.split_documents(docs)
    return [
        {
            "text": (getattr(d, "page_content", "") or ""),
            "metadata": (getattr(d, "metadata", {}) or {}),
        }
        for d in out_docs
    ]
