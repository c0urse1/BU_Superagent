from __future__ import annotations

import os
from argparse import Namespace

from src.core.settings import Settings


def resolve_chunking_config(args: Namespace) -> dict[str, object]:
    s = Settings().chunking
    return {
        "target_chars": getattr(args, "chunk_target", None) or s.chunk_target_chars,
        "min_chars": getattr(args, "chunk_min", None) or s.chunk_min_chars,
        "max_chars": getattr(args, "chunk_max", None) or s.chunk_max_chars,
        "overlap_chars": getattr(args, "chunk_overlap", None) or s.chunk_overlap_chars,
        "inject_section_titles": (
            s.inject_section_titles
            if getattr(args, "section_inject", None) is None
            else bool(args.section_inject)
        ),
        "cross_page_title_merge": (
            s.cross_page_title_merge
            if getattr(args, "cross_page_merge", None) is None
            else bool(args.cross_page_merge)
        ),
        "enforce_sentence_boundaries": (
            s.enforce_sentence_boundaries
            if getattr(args, "sentence_boundaries", None) is None
            else bool(args.sentence_boundaries)
        ),
    }


def windowed_chunk_text(
    text: str, target: int | None = None, overlap: int | None = None
) -> list[str]:
    """Slice text into overlapping windows by characters.

    Defaults are environment-backed to align with CLI/config:
    - CHUNK_TARGET_CHARS (default 500)
    - CHUNK_OVERLAP_CHARS (default 150)

    Example:
        chunks = windowed_chunk_text(long_text)
    """
    # Coerce to str for mypy compatibility, then to int
    t_env = os.getenv("CHUNK_TARGET_CHARS", "500")
    o_env = os.getenv("CHUNK_OVERLAP_CHARS", "150")
    t = int(str(target) if target is not None else str(t_env))
    o = int(str(overlap) if overlap is not None else str(o_env))
    stride = max(1, t - o)

    chunks: list[str] = []
    for start in range(0, len(text), stride):
        end = start + t
        chunks.append(text[start:end])
        if end >= len(text):
            break
    return chunks
