from __future__ import annotations

# ruff: noqa: E402

"""Infrastructure-level builder for concrete embedding providers.

This complements src/infra/embeddings/factory.py which returns LangChain embeddings. This
builder returns our EmbeddingsPort implementations for use by SAM application/infrastructure
adapters that prefer not to depend on LangChain types directly.
"""

from collections.abc import Sequence

from src.application.ports.embeddings_port import EmbeddingsPort
from src.core.settings import EmbeddingConfig
from src.infrastructure.embeddings.providers import (
    HuggingFaceEmbeddingProvider,
    OpenAIEmbeddingProvider,
)


def build_embeddings_provider(cfg: EmbeddingConfig) -> EmbeddingsPort:
    p = (cfg.provider or "").lower()
    if p == "huggingface":
        return HuggingFaceEmbeddingProvider(cfg)
    if p == "openai":
        return OpenAIEmbeddingProvider(cfg)
    # For tests or unknown providers, fall back to the legacy LangChain builder which has a dummy
    # provider option, but wrap it in a simple adapter to match our port.
    from src.infra.embeddings.factory import build_embeddings as _build_lc

    lc = _build_lc(cfg)

    class _LCAdapter(EmbeddingsPort):
        def embed_documents(self, texts: list[str]) -> list[Sequence[float]]:
            return lc.embed_documents(texts)

        def embed_query(self, text: str) -> list[float]:
            return lc.embed_query(text)

    return _LCAdapter()


__all__ = ["build_embeddings_provider"]
