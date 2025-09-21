"""Chroma vector store wiring for the infra layer.

Provides a small helper to derive a model-scoped collection name and extends
the ingestion-layer ChromaStore with an optional existence check by content hash.
"""

from __future__ import annotations

import re

from bu_kb.ingest.store import ChromaStore as _ChromaStore


class ChromaStore(_ChromaStore):
    def exists_by_hash(self, content_hash: str) -> bool:
        """Best-effort: check if a doc with metadata.content_hash exists.

        Tries to access the underlying Chroma collection if exposed by the
        langchain-chroma wrapper; returns False if not available or on error.
        """
        try:  # langchain-chroma exposes the raw chroma collection on _collection
            raw = getattr(self._db, "_collection", None)
            if raw is None:
                return False
            got = raw.get(where={"content_hash": content_hash}, limit=1)
            ids = (got or {}).get("ids") or []
            return len(ids) > 0
        except Exception:
            return False


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
