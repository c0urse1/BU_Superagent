from __future__ import annotations

from collections.abc import Iterable, Mapping

from src.infra.vectorstores.chroma_store import ChromaStore


class ChromaVectorStoreAdapter:
    """Adapter implementing VectorStorePort on top of existing ChromaStore wrapper."""

    def __init__(self, store: ChromaStore) -> None:
        self._store = store

    def query(
        self, query: str, *, k: int, metadata_filter: Mapping[str, object] | None = None
    ) -> Iterable[tuple[str, Mapping[str, object]]]:
        docs = self._store.query(query, k=k, filter=dict(metadata_filter or {}))
        for d in docs:
            yield (getattr(d, "page_content", "") or ""), (d.metadata or {})
