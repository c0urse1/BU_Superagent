from __future__ import annotations

from typing import Protocol

try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    from langchain.schema import Document


class TextSplitterLike(Protocol):
    def split_documents(self, docs: list[Document]) -> list[Document]: ...
