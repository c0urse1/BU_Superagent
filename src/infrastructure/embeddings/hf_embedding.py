from __future__ import annotations

"""HuggingFace Embeddings adapter (alias module).

Re-exports HuggingFaceEmbeddingProvider which implements EmbeddingsPort and is compatible
with LangChain's Embeddings interface.
"""
from .providers import HuggingFaceEmbeddingProvider  # noqa: E402

__all__ = ["HuggingFaceEmbeddingProvider"]
