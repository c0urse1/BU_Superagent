from __future__ import annotations

from typing import Any

from src.core.settings import Settings
from src.infra.retrieval.assemble import assemble_context
from src.infra.retrieval.retriever import retrieve
from src.services.llm.chain import answer_with_citations


class _FeatureFlags:
    enforce_citations: bool = False


feature_flags = _FeatureFlags()


def post_qa(req: Any) -> dict:
    question = (req.json.get("question") if hasattr(req, "json") else None) or ""

    # Two-stage: retrieve Top-N vector hits, optionally cross-encode rerank to Top-K
    rcfg = Settings().reranker
    initial_n = int(getattr(rcfg, "initial_top_n", 10)) if getattr(rcfg, "enabled", False) else 10
    final_k = int(getattr(rcfg, "final_top_k", 5))

    chunks = retrieve(question, k=initial_n)

    # Optional reranking using local BGE cross-encoder
    if getattr(rcfg, "enabled", False):
        try:
            from src.infra.rerankers.bge import RerankItem
            from src.infra.rerankers.factory import get_reranker

            reranker = get_reranker()
            if reranker is not None and chunks:
                items = [
                    RerankItem(text=(c.get("text") or ""), metadata=(c.get("metadata") or {}))
                    for c in chunks
                ]
                reranked = reranker.rerank(query=question, items=items, top_k=final_k)
                chunks = [{"text": it.text, "metadata": it.metadata} for it in reranked]
        except Exception:
            # Fail open: if model not available or any error, keep vector order
            pass

    ctx = assemble_context(chunks)

    # Obtain an llm client from your DI/container; placeholder below
    llm: Any
    try:
        from src.services.llm.provider import llm as _provider_llm

        llm = _provider_llm
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

        def answer_plain(_llm: Any, question: str, context: str) -> str:  # minimal placeholder
            return question + "\n\n" + context

    return {"answer": answer_plain(llm, question, ctx)}
