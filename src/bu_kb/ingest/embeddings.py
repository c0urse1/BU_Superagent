from __future__ import annotations

from typing import Any

from src.core.settings import AppSettings
from src.infra.embeddings.factory import build_embeddings


def build_embedder(model_name: str) -> Any:
    """Build embeddings via infra factory to ensure E5 prefixing and normalization.

    The CLI passes a model_name; we adapt the AppSettings embedding config accordingly
    and delegate to the shared factory.
    """
    cfg = AppSettings().embeddings
    if model_name:
        cfg = cfg.model_copy(update={"model_name": model_name})
    return build_embeddings(cfg)
