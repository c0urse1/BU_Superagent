from __future__ import annotations

from dataclasses import dataclass

from .vectorstore import load_vectorstore


@dataclass
class QueryHit:
    content: str
    source: str
    page: int | None
    score: float | None  # 0..1 bei relevance_score; bei Fallback evtl. anderer Range
    metadata: dict


class QueryService:
    def __init__(self) -> None:
        self.vs = load_vectorstore()

    def _safe_meta(self, meta: dict) -> tuple[str, int | None]:
        source = meta.get("source") or meta.get("file_path") or meta.get("path") or "unknown"
        page = meta.get("page") or meta.get("page_number")
        try:
            page = int(page) if page is not None else None
        except Exception:  # noqa: BLE001
            page = None
        return str(source), page

    def top_k(
        self,
        query: str,
        k: int = 5,
        *,
        use_mmr: bool = False,
        fetch_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[QueryHit]:
        """
        Führt eine semantische Suche durch.
        - use_mmr=True nutzt Maximal Marginal Relevance (diversere Treffer).
        - score_threshold filtert (nur wenn relevance_scores verfügbar sind).
        """
        hits: list[QueryHit] = []

        try:
            if use_mmr:
                # Diversität (MMR). fetch_k kann >k sein, um Auswahl zu verbessern.
                docs = self.vs.max_marginal_relevance_search(
                    query, k=k, fetch_k=fetch_k or max(k * 4, 20)
                )
                for d in docs:
                    src, page = self._safe_meta(d.metadata or {})
                    hits.append(QueryHit(d.page_content, src, page, None, d.metadata or {}))
                return hits

            # Bevorzugt: Relevanz-Scores (0..1)
            if hasattr(self.vs, "similarity_search_with_relevance_scores"):
                docs_scores = self.vs.similarity_search_with_relevance_scores(query, k=k)
                for d, score in docs_scores:
                    src, page = self._safe_meta(d.metadata or {})
                    if score_threshold is not None and score < score_threshold:
                        continue
                    hits.append(QueryHit(d.page_content, src, page, float(score), d.metadata or {}))
                return hits

            # Fallback: ältere API liefert Scores anders (meist cosine-distanzähnlich)
            if hasattr(self.vs, "similarity_search_with_score"):
                docs_scores = self.vs.similarity_search_with_score(query, k=k)
                for d, score in docs_scores:
                    src, page = self._safe_meta(d.metadata or {})
                    # score-Skala kann je nach Backend variieren; kein threshold-Filter per Default
                    hits.append(QueryHit(d.page_content, src, page, float(score), d.metadata or {}))
                return hits

            # Letzter Fallback: ohne Scores
            docs = self.vs.similarity_search(query, k=k)
            for d in docs:
                src, page = self._safe_meta(d.metadata or {})
                hits.append(QueryHit(d.page_content, src, page, None, d.metadata or {}))
            return hits

        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Query fehlgeschlagen: {e}") from e
