from __future__ import annotations

import pytest

from src.core.settings import EmbeddingConfig
from src.infra.embeddings.factory import build_embeddings


def test_hf_cpu_builds() -> None:
    cfg = EmbeddingConfig(
        provider="huggingface",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    try:
        emb = build_embeddings(cfg)
    except Exception as e:
        pytest.skip(f"Skipping HF CPU build test due to environment/model availability: {e}")

    v = emb.embed_query("hello")
    assert isinstance(v, list) and len(v) > 0
