from __future__ import annotations

from typing import Any

# Version-tolerant imports
try:
    from langchain_core.embeddings import Embeddings
except Exception:  # pragma: no cover
    from langchain.embeddings.base import Embeddings

# Providers
try:
    # prefer modern package name
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:  # pragma: no cover
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception:  # pragma: no cover
        from langchain.embeddings import HuggingFaceEmbeddings

try:
    from langchain_openai import OpenAIEmbeddings
except Exception:  # pragma: no cover
    OpenAIEmbeddings = None

from src.core.settings import EmbeddingConfig
from src.infra.embeddings.device import resolve_device


def build_embeddings(cfg: EmbeddingConfig) -> Embeddings:
    provider = cfg.provider.lower()
    device = resolve_device(cfg.device)

    if provider == "huggingface":
        # Keep model_kwargs minimal for broad version compatibility
        return HuggingFaceEmbeddings(
            model_name=cfg.model_name,
            model_kwargs={"device": device},
            encode_kwargs={
                "normalize_embeddings": cfg.normalize_embeddings,
                "batch_size": cfg.batch_size,
            },
        )

    if provider == "openai":
        if OpenAIEmbeddings is None:
            raise RuntimeError(
                "OpenAI embeddings not installed. pip install langchain-openai openai"
            )
        kwargs: dict[str, Any] = {
            "model": cfg.model_name,  # e.g. text-embedding-3-small / large
        }
        if cfg.openai_api_key:
            kwargs["api_key"] = cfg.openai_api_key
        if cfg.openai_base_url:
            kwargs["base_url"] = cfg.openai_base_url
        # OpenAI vectors are L2-normalized on retrieval; keep normalize off on LC side
        return OpenAIEmbeddings(**kwargs)

    if provider == "dummy":
        # Small, local, test-only embedding to keep unit tests offline
        class DummyEmbeddings(Embeddings):
            def _embed(self, text: str) -> list[float]:
                import math

                dim = 16
                vec = [0.0] * dim
                for i, ch in enumerate(text.lower()):
                    vec[(i + ord(ch)) % dim] += 1.0
                norm = math.sqrt(sum(v * v for v in vec)) or 1.0
                return [v / norm for v in vec]

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [self._embed(t) for t in texts]

            def embed_query(self, text: str) -> list[float]:
                return self._embed(text)

        return DummyEmbeddings()

    raise ValueError(f"Unsupported embeddings provider: {cfg.provider}")


def build_embeddings_with_signature(cfg: EmbeddingConfig) -> tuple[Embeddings, str]:
    """Return an embeddings encoder together with its stable signature.

    The signature is used to namespace collections and as a metadata filter
    (metadata["embedding_sig"]) to avoid cross-model bleed.
    """
    emb = build_embeddings(cfg)
    return emb, cfg.signature
