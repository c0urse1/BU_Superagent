from __future__ import annotations

import re
from dataclasses import dataclass

from src.domain.document import Document

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9])")


@dataclass
class ChunkingConfig:
    chunk_size: int = 500
    chunk_overlap: int = 150
    chunk_max_overflow: int = 80
    chunk_min_merge_char_len: int = 300


class SentenceChunker:
    """Sentence-aware chunker with overlap and tiny-merge.

    Pure implementation; does not depend on external splitters. For production,
    infra can provide adapters (e.g., syntok) that pre-split sentences and feed here.
    """

    def __init__(self, cfg: ChunkingConfig | None = None) -> None:
        self.cfg = cfg or ChunkingConfig()

    def split(self, text: str, *, source: str = "", page: int | None = None) -> list[Document]:
        t = (text or "").strip()
        if not t:
            return []
        # First, split into coarse sentences
        sents = [s.strip() for s in _SENTENCE_SPLIT_RE.split(t) if s.strip()]
        if not sents:
            return [Document(content=t, metadata={"source": source, "page": page})]

        target = int(self.cfg.chunk_size)
        overlap = max(0, int(self.cfg.chunk_overlap))
        max_over = max(0, int(self.cfg.chunk_max_overflow))
        min_merge = max(0, int(self.cfg.chunk_min_merge_char_len))

        chunks: list[str] = []
        cur = ""
        for s in sents:
            if not cur:
                cur = s
                continue
            # Would adding this sentence exceed target?
            if len(cur) + 1 + len(s) <= target + max_over:
                cur = f"{cur} {s}"
            else:
                # finalize current and start a new chunk
                chunks.append(cur)
                cur = s
        if cur:
            chunks.append(cur)

        # Merge tiny neighbors
        merged: list[str] = []
        i = 0
        while i < len(chunks):
            a = chunks[i]
            if i + 1 < len(chunks) and (len(a) < min_merge or len(chunks[i + 1]) < min_merge):
                merged.append(a + " " + chunks[i + 1])
                i += 2
            else:
                merged.append(a)
                i += 1

        # Apply overlap (characters) between adjacent chunks by duplicating suffix/prefix
        if overlap > 0 and len(merged) > 1:
            with_ovl: list[str] = []
            for j, c in enumerate(merged):
                if j > 0:
                    prev = merged[j - 1]
                    ovl = prev[-overlap:]
                    c = (ovl + " " + c).strip()
                with_ovl.append(c)
            merged = with_ovl

        return [Document(content=c, metadata={"source": source, "page": page}) for c in merged]
