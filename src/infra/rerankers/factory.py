from __future__ import annotations

from src.core.settings import Settings
from src.infra.rerankers.bge import BGEReranker


def get_reranker() -> BGEReranker | None:
    cfg = Settings().reranker
    if not cfg.enabled or cfg.provider == "none":
        return None
    if cfg.provider == "bge":
        return BGEReranker(
            model_name=cfg.bge_model_name,
            device=cfg.bge_device,
            max_length=cfg.bge_max_length,
            batch_size=cfg.bge_batch_size,
        )
    raise ValueError(f"Unknown reranker provider: {cfg.provider}")
