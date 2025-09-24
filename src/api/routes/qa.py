from __future__ import annotations

from typing import Any

from src.api.services import query_use_case


def post_qa(req: Any) -> dict:
    question = (getattr(req, "json", {}) or {}).get("question", "")
    # Delegate to pre-configured use-case with LLM injected via config composition
    answer = query_use_case.execute(question, use_llm=True)
    # Ensure a string return (uc.execute returns str when use_llm=True)
    return {"answer": str(answer)}
