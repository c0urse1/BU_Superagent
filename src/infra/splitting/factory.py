from __future__ import annotations

from typing import Literal

try:
    # Preferred modern package
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    # Fallback for older LangChain
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from .sentence_chunker import SentenceAwareChunker, SentenceAwareParams
from .types import TextSplitterLike


def build_splitter(
    mode: Literal["sentence_aware", "recursive"] = "sentence_aware",
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    max_overflow: int = 200,
    min_merge_char_len: int = 500,
) -> TextSplitterLike:
    if mode == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],  # mild preference for sentence-ish cuts
        )
    # default: sentence-aware
    return SentenceAwareChunker(
        SentenceAwareParams(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_overflow=max_overflow,
            min_merge_char_len=min_merge_char_len,
        )
    )
