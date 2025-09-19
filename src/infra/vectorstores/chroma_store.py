"""Chroma vector store wiring for the infra layer.

Provides a small helper to derive a model-scoped collection name and re-exports
the existing ChromaStore implementation from the ingestion package for reuse.
"""

from __future__ import annotations

import re
from typing import Final

from bu_kb.ingest.store import ChromaStore as _ChromaStore

# Re-export for callers to import from infra.vectorstores consistently
ChromaStore: Final = _ChromaStore


def collection_name_for(base: str, signature: str) -> str:
    """Build a filesystem-friendly, model-scoped collection name.

    Examples
    --------
    >>> collection_name_for("bu_knowledge", "huggingface:all-MiniLM:norm")
    'bu_knowledge__huggingface_all_minilm_norm'
    """

    slug = re.sub(r"[^a-z0-9]+", "_", signature.lower())
    # Chroma sqlite filenames end up under the persist directory; keep it modest
    return f"{base}__{slug}"[:63]
