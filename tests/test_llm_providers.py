from __future__ import annotations

import os

import pytest

from src.infrastructure.llm.providers import DummyLLM


def test_dummy_llm_ask_concat() -> None:
    llm = DummyLLM()
    out = llm.ask("What?", "Here is context.")
    assert out.startswith("What?") and "Here is context." in out


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="OpenAI dependency may not be installed in CI"
)
def test_openai_chat_llm_import_only() -> None:
    # Ensure the class can be imported; do not network-call
    try:
        from src.infrastructure.llm.providers import OpenAIChatLLM

        # Construction may fail if dependency missing; tolerate via skip
        try:
            _ = OpenAIChatLLM(model="gpt-4o-mini", api_key="sk-TEST")
        except RuntimeError:
            pytest.skip("langchain-openai not installed")
    except Exception as e:  # pragma: no cover - defensive
        pytest.fail(f"Unexpected import failure: {e}")
