from __future__ import annotations

from typing import Any

try:
    # langchain >= 0.2
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    # older langchain
    from langchain.schema import Document

from .ingest.store import ChromaStore


def answer_candidates(
    store: ChromaStore,
    q: str,
    *,
    k: int = 10,
    score_threshold: float = 0.2,
    filter: dict[str, Any] | None = None,
) -> list[Document]:
    """Return top-k answer candidates with safer defaults.

    - Uses a retriever with score pruning (similarity_score_threshold) by default.
    - Accepts an optional metadata filter; omitted when None to satisfy Chroma.
    """
    retriever = store.get_retriever(k=k, score_threshold=score_threshold, filter=filter)
    return retriever.get_relevant_documents(q)
