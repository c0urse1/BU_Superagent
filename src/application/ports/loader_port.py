from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from src.domain.document import Document


class DocumentLoaderPort(Protocol):
    def load(self, path: str) -> Iterable[Document]:  # pragma: no cover - interface
        ...
