from __future__ import annotations

from .context_blocks import chunk_to_block


def assemble_context(chunks: list[dict], k: int = 5) -> str:
    blocks = [chunk_to_block(c) for c in (chunks or [])[: max(0, int(k))]]
    return "\n\n---\n\n".join(blocks)
