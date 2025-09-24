from __future__ import annotations

"""OpenAI Embeddings adapter (alias module).

Re-exports OpenAIEmbeddingProvider which implements EmbeddingsPort.
"""
from .providers import OpenAIEmbeddingProvider  # noqa: E402

__all__ = ["OpenAIEmbeddingProvider"]
