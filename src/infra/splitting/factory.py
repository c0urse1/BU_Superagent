from __future__ import annotations

from typing import Any, Literal, TypedDict

try:
    # Preferred modern package
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    # Fallback for older LangChain
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from .sentence_chunker import SentenceAwareChunker, SentenceAwareParams, SentenceChunker
from .types import TextSplitterLike


class ChunkerOpts(TypedDict):
    target_chars: int
    min_chars: int
    max_chars: int
    overlap_chars: int
    enforce_sentence_boundaries: bool
    inject_section_titles: bool
    cross_page_title_merge: bool


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


def make_chunker(**opts: Any) -> SentenceChunker:
    """Construct the adaptive SentenceChunker from explicit options.

    Expected keys in opts:
      - target_chars, min_chars, max_chars, overlap_chars
      - enforce_sentence_boundaries, inject_section_titles, cross_page_title_merge
    """
    # Extract with explicit casts
    t = int(opts["target_chars"])
    mn = int(opts["min_chars"])
    mx = int(opts["max_chars"])
    ol = int(opts["overlap_chars"])
    esb = bool(opts["enforce_sentence_boundaries"])
    inj = bool(opts["inject_section_titles"])
    cpm = bool(opts["cross_page_title_merge"])
    return SentenceChunker(
        target_chars=t,
        min_chars=mn,
        max_chars=mx,
        overlap_chars=ol,
        enforce_sentence_boundaries=esb,
        inject_section_titles=inj,
        cross_page_title_merge=cpm,
    )
