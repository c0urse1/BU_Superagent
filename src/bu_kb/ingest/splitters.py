from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Adapter to bridge a TextSplitter-like object (with split_documents) to our Splitter Protocol
class TextSplitterAdapter:
    def __init__(self, splitter: object) -> None:
        self._splitter = splitter

    def split(self, docs: list[Document]) -> list[Document]:
        fn = getattr(self._splitter, "split_documents", None)
        if callable(fn):
            return fn(docs)
        # Fallback: if underlying offers .split, delegate; otherwise raise
        alt = getattr(self._splitter, "split", None)
        if callable(alt):
            return alt(docs)
        raise AttributeError("Provided splitter does not support split_documents or split")


class RecursiveSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def split(self, docs: list[Document]) -> list[Document]:
        return self._splitter.split_documents(docs)
