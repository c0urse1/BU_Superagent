"""Minimal local LLM provider stubs.

This module provides a tiny, dependency-free fallback implementation so that
imports like `from src.services.llm.provider import llm` resolve during
development, linting, and tests. In production, replace these with real
clients (OpenAI, Azure OpenAI, HF Inference, etc.) wired via settings/DI.
"""

from __future__ import annotations

from typing import Any


class _EchoLLM:
    """A trivial LLM-like client that simply echoes the user prompt.

    Contract: exposes `.invoke(system: str, user: str) -> Any` and returns an
    object with a `.text` attribute for compatibility with `answer_with_citations`.
    """

    def invoke(self, *, system: str, user: str) -> Any:  # noqa: D401 - simple stub
        class _R:
            def __init__(self, t: str):
                self.text = t

        return _R(user)


# Default export used by API/scripts as a simple placeholder
llm = _EchoLLM()


def answer_plain(_llm: Any, question: str, context: str) -> str:
    """Very simple non-cited QA function for baseline/manual testing.

    Real implementations should prompt an LLM; this fallback just concatenates
    the question and context so downstream code paths remain functional without
    external services.
    """

    _ = _llm  # intentionally unused in the stub
    return f"{question}\n\n{context}"
