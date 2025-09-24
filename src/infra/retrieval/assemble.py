from __future__ import annotations

import warnings
from collections.abc import Iterable

from src.domain.context import assemble_context as _assemble_domain
from src.domain.document import Document


def _to_documents(chunks: Iterable[dict]) -> list[Document]:
    docs: list[Document] = []
    for c in chunks:
        md = (c or {}).get("metadata", {}) or {}
        txt = (c or {}).get("text", "") or ""
        docs.append(Document(content=txt, metadata=md))
    return docs


def assemble_context(chunks: list[dict], k: int = 5) -> str:
    """Deprecated: use src.domain.context.assemble_context instead.

    This shim converts legacy chunk dicts to domain Documents and delegates to
    the domain implementation so CLI and API share the same formatting logic.
    """
    warnings.warn(
        "src.infra.retrieval.assemble.assemble_context is deprecated; use src.domain.context.assemble_context",
        DeprecationWarning,
        stacklevel=2,
    )
    k = max(0, int(k))
    docs = _to_documents((chunks or [])[:k])
    return _assemble_domain(docs, k=len(docs))
