from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

try:  # Prefer core interface; tolerate older installs
    from langchain_core.embeddings import Embeddings as LCEmbeddings
except Exception:  # pragma: no cover
    try:
        from langchain.embeddings.base import Embeddings as LCEmbeddings
    except Exception:  # pragma: no cover
        from typing import Any as LCEmbeddings


class ChromaStore:
    def __init__(self, collection: str, persist_dir: Path, embedder: LCEmbeddings) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._db = Chroma(
            collection_name=collection,
            persist_directory=str(persist_dir),
            embedding_function=embedder,
        )

    def add_documents(self, docs: list[Document]) -> None:
        if docs:
            self._db.add_documents(docs)

    def persist(self) -> None:
        # In some versions of langchain-chroma, the wrapper has no persist();
        # the underlying client does. Safely call it if present.
        client = getattr(self._db, "_client", None)
        if client and hasattr(client, "persist"):
            client.persist()

    # --- Query convenience APIs ---
    def query(
        self,
        text: str,
        k: int = 4,
        *,
        filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Semantic search over stored chunks; returns top-k Documents.

        Parameters
        ----------
        text : str
            Natural-language query.
        k : int
            Maximum number of hits to return.
        filter : dict | None
            Optional metadata filter (e.g., {"category": "rules"}).
        """
        return self._db.similarity_search(text, k=k, filter=filter)

    def query_with_scores(
        self,
        text: str,
        k: int = 4,
        *,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Same as `query` but also returns similarity scores.

        Handles both modern and older LangChain method names.
        """
        if hasattr(self._db, "similarity_search_with_relevance_scores"):
            return self._db.similarity_search_with_relevance_scores(text, k=k, filter=filter)
        if hasattr(self._db, "similarity_search_with_score"):
            return self._db.similarity_search_with_score(text, k=k, filter=filter)
        # Fallback to scoreless API; attach neutral score 0.0
        docs: list[Document] = self._db.similarity_search(text, k=k, filter=filter)
        return [(d, 0.0) for d in docs]

    def get_retriever(
        self,
        k: int = 4,
        *,
        score_threshold: float | None = None,
        filter: dict[str, Any] | None = None,
    ) -> Any:
        """Return a LangChain retriever. If score_threshold is set, prunes low-relevance hits."""
        if score_threshold is not None:
            kwargs: dict[str, Any] = {
                "k": max(k, 10),
                "score_threshold": float(score_threshold),
            }
            if filter is not None:
                kwargs["filter"] = filter
            return self._db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs=kwargs,
            )

        kwargs2: dict[str, Any] = {"k": k}
        if filter is not None:
            kwargs2["filter"] = filter
        return self._db.as_retriever(search_kwargs=kwargs2)
