from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.core.settings import AppSettings

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
        # Best-effort access to embedding function for cosine dedup
        self._embedding: Any = getattr(self.vs, "embedding_function", None)

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
        # Guardrail: constrain queries to current embedding signature to avoid cross-model bleed
        try:
            sig = AppSettings().embeddings.signature
            md_filter: dict[str, Any] | None = {"embedding_sig": sig}
        except Exception:  # noqa: BLE001
            md_filter = None

        try:
            if use_mmr:
                # Diversität (MMR). fetch_k kann >k sein, um Auswahl zu verbessern.
                docs = self.vs.max_marginal_relevance_search(
                    query, k=k, fetch_k=fetch_k or max(k * 4, 20), filter=md_filter
                )
                for d in docs:
                    src, page = self._safe_meta(d.metadata or {})
                    hits.append(QueryHit(d.page_content, src, page, None, d.metadata or {}))
                # fallthrough to optional dedup below

            # Bevorzugt: Relevanz-Scores (0..1)
            if hasattr(self.vs, "similarity_search_with_relevance_scores"):
                docs_scores = self.vs.similarity_search_with_relevance_scores(
                    query, k=k, filter=md_filter
                )
                for d, score in docs_scores:
                    src, page = self._safe_meta(d.metadata or {})
                    if score_threshold is not None and score < score_threshold:
                        continue
                    hits.append(QueryHit(d.page_content, src, page, float(score), d.metadata or {}))
                # fallthrough to optional dedup below

            # Fallback: ältere API liefert Scores anders (meist cosine-distanzähnlich)
            if hasattr(self.vs, "similarity_search_with_score"):
                docs_scores = self.vs.similarity_search_with_score(query, k=k, filter=md_filter)
                for d, score in docs_scores:
                    src, page = self._safe_meta(d.metadata or {})
                    # score-Skala kann je nach Backend variieren; kein threshold-Filter per Default
                    hits.append(QueryHit(d.page_content, src, page, float(score), d.metadata or {}))
                # fallthrough to optional dedup below

            # Letzter Fallback: ohne Scores
            docs = self.vs.similarity_search(query, k=k, filter=md_filter)
            for d in docs:
                src, page = self._safe_meta(d.metadata or {})
                hits.append(QueryHit(d.page_content, src, page, None, d.metadata or {}))
            # fallthrough to optional dedup below

        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Query fehlgeschlagen: {e}") from e

        # Retrieval-time dedup (post-MMR/similarity, pre-return)
        # - controlled by AppSettings().dedup_query
        # - method: "cosine" or "exact"
        def _cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b, strict=False))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(y * y for y in b)) or 1.0
            return dot / (na * nb)

        try:
            cfg = AppSettings()
        except Exception:  # be resilient in legacy contexts
            cfg = type("_Cfg", (), {"dedup_query": type("_DQ", (), {"enabled": False})()})()

        if getattr(cfg.dedup_query, "enabled", False) and len(hits) > 1:
            kept: list[QueryHit] = []
            kept_vecs: list[list[float]] = []
            seen_norms: set[str] = set()
            thr = float(getattr(cfg.dedup_query, "similarity_threshold", 0.95))
            method = str(getattr(cfg.dedup_query, "method", "cosine")).lower()

            for h in hits:
                txt = (h.content or "").strip()
                if not txt:
                    kept.append(h)
                    continue

                if method == "exact":
                    norm = txt.lower().replace("\n", " ").strip()
                    if norm in seen_norms:
                        continue
                    seen_norms.add(norm)
                    kept.append(h)
                    continue

                # cosine: compare to already-accepted results
                vec: list[float] | None
                try:
                    vec = self._embedding.embed_query(txt) if self._embedding else None
                except Exception:
                    vec = None

                if vec is None:
                    # Fallback to exact on embedding failure
                    norm = txt.lower().replace("\n", " ").strip()
                    if norm in seen_norms:
                        continue
                    seen_norms.add(norm)
                    kept.append(h)
                    continue

                if any(_cosine(vec, kv) >= thr for kv in kept_vecs):
                    continue

                kept.append(h)
                kept_vecs.append(vec)

            hits = kept

        # Return truncated to k after dedup (unique top-k)
        return hits[:k]
