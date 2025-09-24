from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.application.ports.vector_store_port import VectorStorePort
from src.bu_kb.ingest.store import ChromaStore as _ChromaStore
from src.domain.document import Document as DomainDocument


class ChromaVectorStore(VectorStorePort):
    """VectorStorePort adapter backed by LangChain-Chroma via existing wrapper.

    This adapter keeps all external dependencies in Infrastructure and converts to/from
    domain Document objects.
    """

    def __init__(self, collection: str, persist_dir: Path, embedder: Any) -> None:
        self._store = _ChromaStore(collection, persist_dir, embedder)

    @staticmethod
    def _to_lc(d: DomainDocument) -> Any:
        from langchain_core.documents import Document as LCDocument

        return LCDocument(page_content=d.content, metadata=dict(d.metadata or {}))

    @staticmethod
    def _to_domain(d: Any) -> DomainDocument:
        # LC Document has .page_content and .metadata; tolerate simple dicts
        txt = getattr(d, "page_content", None)
        md = getattr(d, "metadata", None)
        if txt is None and isinstance(d, dict):  # fallback tolerance
            txt = d.get("text", "")
            md = d.get("metadata", {})
        return DomainDocument(content=(txt or ""), metadata=dict(md or {}))

    def add_documents(self, docs: list[DomainDocument]) -> None:
        if not docs:
            return
        lcs = [self._to_lc(d) for d in docs]
        self._store.add_documents(lcs)

    def persist(self) -> None:
        self._store.persist()

    def search(
        self,
        query: str,
        k: int,
        *,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> list[DomainDocument]:
        docs = self._store.query(query, k=k, filter=dict(metadata_filter or {}))
        return [self._to_domain(d) for d in docs]

    def search_with_scores(
        self,
        query: str,
        k: int,
        *,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> list[tuple[DomainDocument, float]]:
        pairs = self._store.query_with_scores(query, k=k, filter=dict(metadata_filter or {}))
        return [(self._to_domain(d), float(s)) for (d, s) in pairs]

    # Optional helper (not part of the base port): cross-run hash existence check
    def exists_by_hash(self, content_hash: str) -> bool:
        """Best-effort: check if a document with metadata.content_hash exists."""
        try:
            return bool(self._store.exists_by_hash(content_hash))
        except Exception:
            return False

    # Diversified retrieval (MMR): surface underlying capability for use-cases
    def max_marginal_relevance_search(
        self,
        query: str,
        *,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.7,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> list[DomainDocument]:
        docs = self._store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=dict(metadata_filter or {}),
        )
        return [self._to_domain(d) for d in docs]
