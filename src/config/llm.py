from __future__ import annotations

"""Composition: construct LLMPort implementations from settings.

Defaults to DummyLLM to keep tests/offline flows working. When OPENAI_* env is set (or
extended settings are provided), builds an OpenAIChatLLM.
"""

import os  # noqa: E402

from src.application.ports.llm_port import LLMPort  # noqa: E402
from src.infrastructure.llm.providers import DummyLLM, OpenAIChatLLM  # noqa: E402


def build_llm_from_env(
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMPort:
    prov = (provider or os.getenv("LLM_PROVIDER") or "").lower().strip()
    if prov in ("openai", "azure-openai"):
        mdl = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        key = api_key or os.getenv("OPENAI_API_KEY")
        base = base_url or os.getenv("OPENAI_BASE_URL")
        return OpenAIChatLLM(model=str(mdl), api_key=key, base_url=base)

    # default
    return DummyLLM()


__all__ = ["build_llm_from_env"]
