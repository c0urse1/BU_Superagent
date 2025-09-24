from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from src.domain.document import Document, content_hash


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


@dataclass
class DedupConfig:
    enabled: bool = True
    hash_enabled: bool = True
    semantic_enabled: bool = True
    min_chars_for_hash: int = 20
    similarity_threshold: float = 0.95
    keep_strategy: str = "first"  # "first" | "longest"


class DuplicateDetector:
    """Deduplicate a sequence of Documents.

    Embeddings are injected via a callable to keep the domain pure.
    """

    def __init__(
        self, cfg: DedupConfig | None = None, embed: Callable[[str], Sequence[float]] | None = None
    ) -> None:
        self.cfg = cfg or DedupConfig()
        self._embed = embed

    def unique(self, docs: Iterable[Document]) -> tuple[list[Document], list[Document]]:
        if not self.cfg.enabled:
            ds = list(docs)
            return ds, []

        kept: list[Document] = []
        skipped: list[Document] = []

        seen_hash: set[str] = set()
        kept_vecs: list[Sequence[float]] = []

        for d in docs:
            txt = (d.content or "").strip()
            if not txt:
                kept.append(d)
                continue

            # Hash-based exact dedup
            if self.cfg.hash_enabled and len(txt) >= int(self.cfg.min_chars_for_hash):
                h = content_hash(txt)
                if h in seen_hash:
                    skipped.append(d)
                    continue
                seen_hash.add(h)

            # Semantic near-dup
            if self.cfg.semantic_enabled and self._embed is not None:
                vec: Sequence[float] | None = None
                try:
                    vec = self._embed(txt)
                except Exception:
                    vec = None
                if vec is not None and any(
                    cosine(vec, kv) >= float(self.cfg.similarity_threshold) for kv in kept_vecs
                ):
                    skipped.append(d)
                    continue

            kept.append(d)
            if self.cfg.semantic_enabled and self._embed is not None:
                try:
                    kv = self._embed(txt)
                    kept_vecs.append(kv)
                except Exception:
                    pass

        if self.cfg.keep_strategy == "longest":
            # Optional: keep longest variant per normalized text (approximate)
            def _norm(s: str) -> str:
                return " ".join(s.lower().split())

            by_norm: dict[str, Document] = {}
            for d in kept:
                key = _norm(d.content)
                cur = by_norm.get(key)
                if cur is None or len(d.content) > len(cur.content):
                    by_norm[key] = d
            kept = list(by_norm.values())

        return kept, skipped
