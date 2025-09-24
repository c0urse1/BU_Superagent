from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    metadata: Mapping[str, object]

    def short(self, limit: int = 200) -> str:
        t = (self.text or "").replace("\n", " ")
        return (t[:limit] + ("…" if len(t) > limit else "")) if t else ""


def ensure_sig(metadata: Mapping[str, object], sig: str) -> bool:
    """Return True if metadata carries the expected embedding signature."""
    return str(metadata.get("embedding_sig", "")) == str(sig)
