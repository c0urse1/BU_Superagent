from __future__ import annotations

from typing import Any

from src.infra.retrieval.assemble import assemble_context
from src.infra.retrieval.retriever import retrieve
from src.services.llm.chain import answer_with_citations


class _FeatureFlags:
    enforce_citations: bool = False


feature_flags = _FeatureFlags()


def post_qa(req: Any) -> dict:
    question = (req.json.get("question") if hasattr(req, "json") else None) or ""

    chunks = retrieve(question, k=5)
    ctx = assemble_context(chunks)

    # Obtain an llm client from your DI/container; placeholder below
    try:
        from src.services.llm.provider import llm
    except Exception:

        class _EchoLLM:
            def invoke(self, *, system: str, user: str) -> Any:
                class R:
                    def __init__(self, t: str):
                        self.text = t

                return R(user)

        llm = _EchoLLM()

    if feature_flags.enforce_citations:
        return {"answer": answer_with_citations(llm, question, ctx)}

    # Fallback path; implement your existing plain answer function
    try:
        from src.services.llm.provider import answer_plain
    except Exception:

        def answer_plain(_llm: Any, q: str, c: str) -> str:  # minimal placeholder
            return q + "\n\n" + c

    return {"answer": answer_plain(llm, question, ctx)}
