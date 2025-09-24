from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class EmbeddingsPort(Protocol):
    def embed_documents(self, texts: list[str]) -> list[Sequence[float]]:  # pragma: no cover
        ...

    def embed_query(self, text: str) -> Sequence[float]:  # pragma: no cover
        ...
