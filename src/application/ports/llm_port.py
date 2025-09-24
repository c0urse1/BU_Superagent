from __future__ import annotations

from typing import Protocol


class LLMPort(Protocol):
    """Abstract language model interface for Q&A prompts."""

    def ask(self, question: str, context: str) -> str:  # pragma: no cover - interface
        ...
