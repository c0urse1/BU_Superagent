from __future__ import annotations

from src.core.settings import EmbeddingConfig
from src.infra.embeddings.factory import build_embeddings


def test_dummy_provider_embeds_deterministically() -> None:
    cfg = EmbeddingConfig(
        provider="dummy",
        model_name="ignored-for-dummy",
        device="cpu",
        normalize_embeddings=True,
    )

    emb = build_embeddings(cfg)
    v1 = emb.embed_query("Hello World")
    v2 = emb.embed_query("Hello World")

    assert isinstance(v1, list)
    assert len(v1) > 0
    # Deterministic for same input
    assert v1 == v2
