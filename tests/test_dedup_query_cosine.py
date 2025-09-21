from __future__ import annotations

import math
from typing import Any

from langchain_core.documents import Document

from src.core.settings import AppSettings
from src.infra.embeddings.factory import build_embeddings


class DummyQueryService:
    def __init__(self, emb: Any) -> None:
        self._embedding = emb

    def _search(self, query: str, k: int, **_: object) -> list[Document]:
        return [
            Document(
                page_content="Definition der Berufsunfähigkeit gemäß §172 VVG.",
                metadata={"source": "A"},
            ),
            Document(
                page_content="Definition der Berufsunfähigkeit gemaess §172 VVG.",
                metadata={"source": "B"},
            ),
        ]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def test_query_dedup_cosine() -> None:
    cfg = AppSettings()
    cfg.dedup_query.enabled = True
    cfg.dedup_query.method = "cosine"
    cfg.dedup_query.similarity_threshold = 0.95

    # Dummy embeddings provider to stay offline
    cfg.embeddings.provider = "dummy"
    emb = build_embeddings(cfg.embeddings)

    qs = DummyQueryService(emb)
    hits = qs._search("BU Definition", k=5)

    kept: list[Document] = []
    kept_vecs: list[list[float]] = []
    for h in hits:
        vec = emb.embed_query(h.page_content or "")
        if any(_cosine(vec, kv) >= cfg.dedup_query.similarity_threshold for kv in kept_vecs):
            continue
        kept.append(h)
        kept_vecs.append(vec)
    assert len(kept) == 1
