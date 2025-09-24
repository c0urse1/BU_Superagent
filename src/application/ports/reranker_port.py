from __future__ import annotations

from typing import Protocol

from src.domain.document import Document


class RerankerPort(Protocol):  # pragma: no cover - interface
    """Re-rank retrieved document chunks against a query.

    Implementations may use local cross-encoders (e.g., BGE) or remote APIs. The function
    returns a list of (Document, score) pairs sorted by descending score.
    """

    def rerank(
        self, query: str, docs: list[Document], top_k: int
    ) -> list[tuple[Document, float]]: ...


__all__ = ["RerankerPort"]
