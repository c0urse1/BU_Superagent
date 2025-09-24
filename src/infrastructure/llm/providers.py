from __future__ import annotations

"""Infrastructure LLM providers implementing the LLMPort contract.

Adapters:
- DummyLLM: dependency-free echo/concat for testing and offline use.
- OpenAIChatLLM: wraps langchain-openai ChatOpenAI to produce answers from a system+user prompt.

These providers keep external dependencies in the infrastructure layer and present a
minimal ask(question, context) API to the application layer.
"""

from dataclasses import dataclass  # noqa: E402

from src.application.ports.llm_port import LLMPort  # noqa: E402


@dataclass
class DummyLLM(LLMPort):
    """Simple test double: returns question + two newlines + context."""

    def ask(self, question: str, context: str) -> str:  # pragma: no cover - trivial
        return f"{question}\n\n{context}"


@dataclass
class OpenAIChatLLM(LLMPort):
    """OpenAI Chat-based LLM using langchain-openai.

    The provider accepts model and credentials via constructor args. It uses a simple
    two-message prompt (system, user). The system content is a concise instruction for
    QA with citations; callers may adapt as needed.
    """

    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.1
    max_tokens: int | None = 512

    def __post_init__(self) -> None:  # lazy import and instantiate client
        try:
            from langchain_openai import ChatOpenAI
        except Exception as e:  # pragma: no cover - import guarded
            raise RuntimeError(
                "langchain-openai is required for OpenAIChatLLM.\n"
                "Install with: pip install langchain-openai openai"
            ) from e

        kwargs: dict[str, object] = {
            "model": self.model,
            "temperature": float(self.temperature),
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.max_tokens is not None:
            kwargs["max_tokens"] = int(self.max_tokens)
        self._chat = ChatOpenAI(**kwargs)

    def ask(self, question: str, context: str) -> str:
        # Compose system+user prompts similar to services.llm.chain
        system = (
            "You are a helpful assistant. Answer clearly and include at least one citation "
            'formatted as (DocShort, S. <page>, "Section Title").'
        )
        user = f"Kontext:\n{context}\n\nFrage:\n{question}"

        # Avoid hard runtime dependency on LC message classes by using invoke() with dicts
        try:
            # Many versions of LC ChatOpenAI accept [ {"role":..., "content":...}, ... ] via messages
            resp = self._chat.invoke(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )
            # Return text content; LC messages have .content
            text = getattr(resp, "content", None)
            if isinstance(text, str) and text:
                return text
            # Fallback to string conversion
            return str(resp)
        except Exception:
            # Fallback formatting in case of signature mismatch; try older .predict_messages or .predict
            try:
                text = self._chat.predict_messages(system=system, messages=[("user", user)])
                return getattr(text, "content", str(text))
            except Exception:
                try:
                    return self._chat.predict(f"{system}\n\n{user}")
                except Exception as e:
                    # Propagate a useful error
                    raise RuntimeError(f"OpenAIChatLLM failed to generate: {e}") from e


__all__ = ["DummyLLM", "OpenAIChatLLM"]
