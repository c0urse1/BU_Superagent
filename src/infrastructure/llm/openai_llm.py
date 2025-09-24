from __future__ import annotations

"""OpenAI Chat LLM adapter (alias module).

Re-exports OpenAIChatLLM which implements LLMPort.
"""
from .providers import OpenAIChatLLM  # noqa: E402

__all__ = ["OpenAIChatLLM"]
