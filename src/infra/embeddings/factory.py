from __future__ import annotations

from typing import Any

import numpy as np

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
        # E5 models benefit from explicit instruction/prefixing and cosine normalization.
        def _is_e5(name: str) -> bool:
            return "intfloat/multilingual-e5" in (name or "").lower()

        # Base embedder with normalization disabled; we'll post-normalize ourselves when required
        base_hf = HuggingFaceEmbeddings(
            model_name=cfg.model_name,
            model_kwargs={"device": device},
            encode_kwargs={
                # disable internal normalization so wrappers can control it explicitly
                "normalize_embeddings": False,
                "batch_size": cfg.batch_size,
            },
        )

        if _is_e5(cfg.model_name) and getattr(cfg, "e5_enable_prefix", True):
            # Lightweight wrapper adding instruction/prefix and L2 normalization
            class _E5Wrapped(Embeddings):
                def __init__(self, base: Embeddings) -> None:
                    self._base = base
                    self._normalize = bool(cfg.normalize_embeddings)
                    self._instr = str(
                        getattr(
                            cfg,
                            "e5_query_instruction",
                            "Instruct: Given a web search query, retrieve relevant passages that answer the query",
                        )
                    )
                    self._q = str(getattr(cfg, "e5_query_prefix", "Query: "))
                    self._p = str(getattr(cfg, "e5_passage_prefix", "Passage: "))

                def _post(self, arrs: list[list[float]]) -> list[list[float]]:
                    if not self._normalize or not arrs:
                        return arrs
                    X = np.asarray(arrs, dtype=np.float32)
                    denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
                    X = X / denom
                    return X.tolist()

                def embed_documents(self, texts: list[str]) -> list[list[float]]:
                    # Prefix passages when configured
                    t2 = [f"{self._p}{t}" if self._p else t for t in texts]
                    vecs = self._base.embed_documents(t2)
                    return self._post(vecs)

                def embed_query(self, text: str) -> list[float]:
                    q = f"{self._instr}\n{self._q}{text}"
                    vec = self._base.embed_query(q)
                    if not self._normalize or not vec:
                        return vec
                    X = np.asarray([vec], dtype=np.float32)
                    denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
                    X = X / denom
                    return X[0].tolist()

            return _E5Wrapped(base_hf)

        # Non-E5 path: optionally normalize via LC encode_kwargs to keep parity with prior behavior
        if cfg.normalize_embeddings:
            # When no wrapper is used, re-enable normalization through LC to keep behavior
            base_hf = HuggingFaceEmbeddings(
                model_name=cfg.model_name,
                model_kwargs={"device": device},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": cfg.batch_size,
                },
            )
        return base_hf

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


def get_embedder() -> Embeddings:
    """Build embeddings from the application-wide AppSettings.

    Convenience for scripts and CLIs that don't need to construct an EmbeddingConfig
    manually. Respects provider/model/device/normalization including E5 wrapping.
    """
    # Local import to avoid cycles at module import time
    from src.core.settings import AppSettings

    cfg = AppSettings().embeddings
    return build_embeddings(cfg)
