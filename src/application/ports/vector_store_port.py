from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from src.domain.document import Document


class VectorStorePort(Protocol):
    """Abstract vector store operations for ingest and query."""

    # Ingest path
    def add_documents(self, docs: list[Document]) -> None:  # pragma: no cover - interface
        ...

    def persist(self) -> None:  # pragma: no cover - interface
        ...

    # Query path
    def search(
        self, query: str, k: int, *, metadata_filter: Mapping[str, object] | None = None
    ) -> list[Document]:  # pragma: no cover - interface
        ...

    # Optional: scores / MMR specialized methods; adapters can implement as needed
    def search_with_scores(
        self, query: str, k: int, *, metadata_filter: Mapping[str, object] | None = None
    ) -> list[tuple[Document, float]]:  # pragma: no cover - interface
        ...
