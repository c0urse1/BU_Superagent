from __future__ import annotations

from typing import Any

# Reuse the implementation from ingest, but provide a stable top-level import path
from .ingest.embeddings import build_embedder as _build_embedder


def build_embedder(model_name: str) -> Any:
    """Baut einen HuggingFace-Embedder (CPU, normalisierte Embeddings)."""
    return _build_embedder(model_name)
