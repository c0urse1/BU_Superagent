from __future__ import annotations

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
