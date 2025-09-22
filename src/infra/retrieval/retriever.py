from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.settings import AppSettings, EmbeddingConfig
from src.infra.embeddings.factory import build_embeddings
from src.infra.vectorstores.chroma_store import ChromaStore, collection_name_for


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
) -> list[dict]:
    """Retrieve chunks as dicts with text + normalized metadata, ready for prompting.

    Returns a list like: [{"text": str, "metadata": {source, page, section, category}}, ...]
    """
    app = AppSettings()
    emb_cfg = embeddings or app.embeddings
    emb = build_embeddings(emb_cfg)
    collection = collection_name_for(app.kb.collection_base, emb_cfg.signature)
    store = ChromaStore(collection, Path(app.kb.persist_directory), emb)

    md_filter: dict[str, Any] = {"embedding_sig": emb_cfg.signature}
    if category:
        md_filter["category"] = category
    if section:
        md_filter["section"] = section

    docs = store.query(query, k=k, filter=md_filter)
    chunks: list[dict] = []
    for d in docs:
        chunks.append(
            {
                "text": (getattr(d, "page_content", "") or ""),
                "metadata": normalize_metadata(getattr(d, "metadata", {}) or {}),
            }
        )
    return chunks
