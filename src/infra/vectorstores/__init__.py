"""Vector store adapters for the infra layer.

Currently re-exports the ChromaStore wrapper from the existing ingest path
and provides small utilities that help compose collections safely.
"""

from __future__ import annotations

__all__ = [
    "ChromaStore",
    "collection_name_for",
]

# Re-export for a stable import point within the infra namespace
from bu_kb.ingest.store import ChromaStore  # noqa: F401

from .chroma_store import collection_name_for  # noqa: F401
