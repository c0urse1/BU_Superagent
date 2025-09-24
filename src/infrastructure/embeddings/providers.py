from __future__ import annotations

"""
Concrete embedding providers for the SAM 'infrastructure' layer.

These classes implement our EmbeddingsPort (application layer) and also match the
LangChain Embeddings duck-typed interface (embed_documents / embed_query) so they can be
plugged into existing vector store wrappers expecting an LC Embeddings object.

HuggingFaceEmbeddingProvider: uses sentence-transformers directly (no LangChain wrapper).
OpenAIEmbeddingProvider: delegates to langchain_openai.OpenAIEmbeddings for simplicity.

Both providers optionally apply E5 instruction/prefixing and L2 normalization according to
EmbeddingConfig flags to preserve behavior parity with the current codebase.
"""

from collections.abc import Sequence  # noqa: E402

import numpy as np  # noqa: E402

from src.application.ports.embeddings_port import EmbeddingsPort  # noqa: E402
from src.core.settings import EmbeddingConfig  # noqa: E402
from src.infra.embeddings.device import resolve_device  # noqa: E402


def _is_e5_model(name: str | None) -> bool:
    try:
        return name is not None and "e5" in name.lower()
    except Exception:
        return False


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        denom = float(np.linalg.norm(arr) + 1e-12)
        return arr / denom
    denom = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / denom


class HuggingFaceEmbeddingProvider(EmbeddingsPort):
    """SentenceTransformers-backed embeddings with optional E5 prefix + L2 norm.

    Implements both our EmbeddingsPort and the LangChain-compatible interface.
    """

    def __init__(self, cfg: EmbeddingConfig) -> None:
        # Lazy import to avoid hard dependency when not used
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:  # pragma: no cover - import path
            raise RuntimeError(
                "sentence-transformers is required for HuggingFaceEmbeddingProvider.\n"
                "Install with: pip install sentence-transformers"
            ) from e

        device = resolve_device(cfg.device)
        self._model = SentenceTransformer(cfg.model_name, device=device)
        # Config flags
        self._normalize = bool(cfg.normalize_embeddings)
        self._batch = int(getattr(cfg, "batch_size", 32) or 32)
        self._e5 = _is_e5_model(cfg.model_name) and bool(getattr(cfg, "e5_enable_prefix", True))
        # E5 text decorations
        self._instr = str(
            getattr(
                cfg,
                "e5_query_instruction",
                "Instruct: Given a web search query, retrieve relevant passages that answer the query",
            )
        )
        self._q = str(getattr(cfg, "e5_query_prefix", "Query: "))
        self._p = str(getattr(cfg, "e5_passage_prefix", "Passage: "))

    # LangChain-compatible + EmbeddingsPort
    def embed_documents(self, texts: list[str]) -> list[Sequence[float]]:
        if not texts:
            return []
        t2 = [f"{self._p}{t}" if self._e5 and self._p else t for t in texts]
        vecs = self._model.encode(
            t2,
            batch_size=self._batch,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we'll handle normalization explicitly
        )
        if self._normalize:
            vecs = _l2_normalize(vecs)
        # Ensure plain Python lists
        return vecs.astype(np.float32).tolist()

    def embed_query(self, text: str) -> Sequence[float]:
        q = f"{self._instr}\n{self._q}{text}" if self._e5 else text
        vec = self._model.encode(
            [q],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )[0]
        if self._normalize:
            vec = _l2_normalize(vec)
        return vec.astype(np.float32).tolist()


class OpenAIEmbeddingProvider(EmbeddingsPort):
    """OpenAI embeddings provider, delegating to langchain_openai.OpenAIEmbeddings.

    We post-normalize when requested for parity; OpenAI vectors are typically suitable for
    cosine similarity without additional normalization, but normalization can improve
    numerical stability for some vectorstores.
    """

    def __init__(self, cfg: EmbeddingConfig) -> None:
        try:
            from langchain_openai import OpenAIEmbeddings
        except Exception as e:  # pragma: no cover - import path
            raise RuntimeError(
                "langchain-openai is required for OpenAIEmbeddingProvider.\n"
                "Install with: pip install langchain-openai openai"
            ) from e

        kwargs: dict[str, object] = {"model": cfg.model_name}
        if cfg.openai_api_key:
            kwargs["api_key"] = cfg.openai_api_key
        if cfg.openai_base_url:
            kwargs["base_url"] = cfg.openai_base_url
        self._inner = OpenAIEmbeddings(**kwargs)
        self._normalize = bool(cfg.normalize_embeddings)

    # LangChain-compatible + EmbeddingsPort
    def embed_documents(self, texts: list[str]) -> list[Sequence[float]]:
        if not texts:
            return []
        vecs = self._inner.embed_documents(texts)
        if not self._normalize:
            return vecs
        X = np.asarray(vecs, dtype=np.float32)
        return _l2_normalize(X).tolist()

    def embed_query(self, text: str) -> Sequence[float]:
        vec = self._inner.embed_query(text)
        if not self._normalize:
            return vec
        X = np.asarray(vec, dtype=np.float32)
        return _l2_normalize(X).tolist()


__all__ = [
    "HuggingFaceEmbeddingProvider",
    "OpenAIEmbeddingProvider",
]
