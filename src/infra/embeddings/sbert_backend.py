"""SBERT embeddings backend with optional E5-style prefixing and normalization.

This backend wraps `sentence_transformers.SentenceTransformer` and:
- Defaults to the E5 instruct model when no model is provided.
- Applies E5 prefixes ("query:" / "passage:") when the model name indicates E5.
- Returns float32 numpy arrays with cosine normalization enabled.
"""

from __future__ import annotations

import os

import numpy as np
from sentence_transformers import SentenceTransformer


class SbertEmbeddings:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        # Prefer SBERT_MODEL, fallback to EMBEDDING_MODEL_NAME, then default to E5 instruct
        model_name = model_name or os.getenv(
            "SBERT_MODEL",
            os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-large-instruct"),
        )
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name, device=device)
        # Heuristic: enable E5 prefixing when model name includes "e5"
        name_lc = self.model_name.lower() if isinstance(self.model_name, str) else ""
        self._use_e5_prefix = "e5" in name_lc

    def encode(self, texts: list[str], batch_size: int = 32, mode: str = "passage") -> np.ndarray:
        """Encode a batch of texts.

        Args:
            texts: List of input texts.
            batch_size: Batch size for encoding.
            mode: "query" or "passage"; controls E5 prefixing when enabled.

        Returns:
            np.ndarray: (N, D) float32 array of normalized embeddings.
        """
        if self._use_e5_prefix:
            prefix = "query: " if mode == "query" else "passage: "
            texts = [prefix + t for t in texts]

        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vecs.astype(np.float32)
