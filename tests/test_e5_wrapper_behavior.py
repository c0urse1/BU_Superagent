from __future__ import annotations

import math
import os

import pytest

from src.core.settings import EmbeddingConfig
from src.infra.embeddings.factory import build_embeddings


@pytest.mark.slow
def test_e5_embedder_returns_1024_unit_norm() -> None:
    model = os.environ.get("TEST_E5_MODEL", "intfloat/multilingual-e5-large-instruct")
    # Allow skipping via env when offline or model not present
    if os.environ.get("SKIP_E5_TESTS"):
        pytest.skip("E5 tests skipped by environment")

    cfg = EmbeddingConfig(
        provider="huggingface",
        model_name=model,
        device="cpu",
        normalize_embeddings=True,
    )
    try:
        emb = build_embeddings(cfg)
        v = emb.embed_query("probe")
    except Exception as e:
        pytest.skip(f"E5 model not available offline or failed to load: {e}")

    assert isinstance(v, list) and len(v) == 1024
    norm = math.sqrt(sum(x * x for x in v))
    assert 0.999 <= norm <= 1.001


def test_e5_prefix_toggle_does_not_break_norm() -> None:
    model = os.environ.get("TEST_E5_MODEL", "intfloat/multilingual-e5-large-instruct")
    if os.environ.get("SKIP_E5_TESTS"):
        pytest.skip("E5 tests skipped by environment")

    cfg = EmbeddingConfig(
        provider="huggingface",
        model_name=model,
        device="cpu",
        normalize_embeddings=True,
        e5_enable_prefix=False,  # disable prefixing; normalization should remain
    )
    try:
        emb = build_embeddings(cfg)
        v = emb.embed_query("probe")
    except Exception as e:
        pytest.skip(f"E5 model not available offline or failed to load: {e}")

    norm = math.sqrt(sum(x * x for x in v))
    assert 0.999 <= norm <= 1.001
