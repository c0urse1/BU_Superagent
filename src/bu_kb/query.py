from __future__ import annotations

import warnings
from dataclasses import dataclass

from src.application.use_cases.query_kb import QueryUseCase
from src.config.providers import build_chroma_vectorstore_with_provider
from src.core.settings import AppSettings


@dataclass
class QueryHit:
    content: str
    source: str
    page: int | None
    score: float | None
    metadata: dict


class QueryService:
    """Deprecated: use QueryUseCase instead.

    This thin shim delegates to QueryUseCase to preserve backward compatibility
    with legacy CLI/tests while the new SAM-based layers are adopted.
    """

    def __init__(self) -> None:  # pragma: no cover - compatibility shim
        warnings.warn(
            "QueryService is deprecated; use QueryUseCase via config composition.",
            DeprecationWarning,
            stacklevel=2,
        )
        store = build_chroma_vectorstore_with_provider()
        self._uc = QueryUseCase(store=store, llm=None, dedup=None)

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
    ) -> list[QueryHit]:  # pragma: no cover - compatibility shim
        # Constrain by embedding signature to avoid cross-model bleed
        try:
            sig = AppSettings().embeddings.signature
        except Exception:
            sig = None

        if not use_mmr and score_threshold is not None:
            pairs = self._uc.retrieve_top_k_with_scores(
                query,
                k=k,
                embedding_sig=sig,
                score_threshold=score_threshold,
            )
            out: list[QueryHit] = []
            for d, s in pairs:
                src, page = self._safe_meta(dict(d.metadata or {}))
                out.append(
                    QueryHit(
                        d.content,
                        src,
                        page,
                        (float(s) if s is not None else None),
                        dict(d.metadata or {}),
                    )
                )
            return out

        docs = self._uc.get_top_k(
            query,
            k=k,
            embedding_sig=sig,
            use_mmr=use_mmr,
            fetch_k=fetch_k,
        )
        out2: list[QueryHit] = []
        for d in docs:
            src, page = self._safe_meta(dict(d.metadata or {}))
            out2.append(QueryHit(d.content, src, page, None, dict(d.metadata or {})))
        return out2
