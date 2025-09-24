from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


def _normalize_text(s: str) -> str:
    return " ".join((s or "").replace("\r", " ").replace("\n", " ").split()).strip()


def content_hash(text: str) -> str:
    norm = _normalize_text(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Document:
    """Domain document chunk with metadata.

    Pure structure independent from external libraries. Used across domain logic
    to avoid LangChain/Chroma coupling.
    """

    content: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # Convenience getters (do not throw)
    @property
    def source(self) -> str:
        return str(self.metadata.get("source", ""))

    @property
    def page(self) -> int | str:
        return self.metadata.get("page", self.metadata.get("page_number", "?"))

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", "<no-title>"))

    @property
    def section(self) -> str:
        return str(self.metadata.get("section", ""))

    def norm_text(self) -> str:
        return _normalize_text(self.content)

    def hash(self) -> str:
        return content_hash(self.content)
