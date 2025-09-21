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
        # Keep a direct reference for components that need to embed ad-hoc text
        self._embedder = embedder
        self._db = Chroma(
            collection_name=collection,
            persist_directory=str(persist_dir),
            embedding_function=embedder,
        )

    @staticmethod
    def _normalize_filter(f: dict[str, Any] | None) -> dict[str, Any] | None:
        """Normalize a metadata filter for Chroma.

        Recent Chroma versions expect a top-level operator (e.g., {"$and": [...]}) in the
        "where" clause. LangChain passes our "filter" through as "where". For simple, single-key
        filters we can pass {"field": value} directly; for multi-key dicts we wrap them in $and.
        If the caller already provided an operator (key starts with "$"), pass it through.
        """
        if not f:
            return f
        # If already an operator at top-level, leave as-is
        if any(str(k).startswith("$") for k in f.keys()):
            return f
        if len(f) <= 1:
            return f
        # Wrap multi-field filters into an explicit $and
        return {"$and": [{k: v} for k, v in f.items()]}

    def add_documents(self, docs: list[Document]) -> None:
        if docs:
            self._db.add_documents(docs)

    def persist(self) -> None:
        # In some versions of langchain-chroma, the wrapper has no persist();
        # the underlying client does. Safely call it if present.
        client = getattr(self._db, "_client", None)
        if client and hasattr(client, "persist"):
            client.persist()

    # Optional cross-run dedup helper (best-effort)
    def exists_by_hash(self, content_hash: str) -> bool:  # pragma: no cover - passthrough util
        """Check if a document with metadata.content_hash exists in the collection.

        Uses the underlying chroma collection when available; returns False on errors.
        """
        try:
            raw = getattr(self._db, "_collection", None)
            if raw is None:
                return False
            got = raw.get(where={"content_hash": content_hash}, limit=1)
            ids = (got or {}).get("ids") or []
            return len(ids) > 0
        except Exception:
            return False

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
        return self._db.similarity_search(text, k=k, filter=self._normalize_filter(filter))

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
            return self._db.similarity_search_with_relevance_scores(
                text, k=k, filter=self._normalize_filter(filter)
            )
        if hasattr(self._db, "similarity_search_with_score"):
            return self._db.similarity_search_with_score(
                text, k=k, filter=self._normalize_filter(filter)
            )
        # Fallback to scoreless API; attach neutral score 0.0
        docs: list[Document] = self._db.similarity_search(
            text, k=k, filter=self._normalize_filter(filter)
        )
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
                kwargs["filter"] = self._normalize_filter(filter)
            return self._db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs=kwargs,
            )

        kwargs2: dict[str, Any] = {"k": k}
        if filter is not None:
            kwargs2["filter"] = self._normalize_filter(filter)
        return self._db.as_retriever(search_kwargs=kwargs2)
