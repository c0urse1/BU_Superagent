from __future__ import annotations

import re
from typing import Any, Protocol, TypedDict

from src.infra.observability.citations_log import log_citation_event
from src.infra.prompting.parsers import has_min_citation, why_failed
from src.infra.prompting.templates import SYSTEM_TEMPLATE, USER_QA_TEMPLATE

MAX_RETRIES = 2


class _InvokeResult(TypedDict, total=False):
    text: str


class SupportsInvoke(Protocol):
    def invoke(self, *, system: str, user: str) -> Any: ...


def answer_with_citations(llm: SupportsInvoke, question: str, context: str) -> str:
    """Ask the LLM and enforce that result contains at least one properly formatted citation.

    The function will retry up to MAX_RETRIES with a corrective instruction
    if validation fails. Returns the final text (with a warning suffix if all retries fail).
    """
    sys = SYSTEM_TEMPLATE
    user = USER_QA_TEMPLATE.format(context=context, question=question)

    response = llm.invoke(system=sys, user=user)
    text = response.text if hasattr(response, "text") else str(response)

    if has_min_citation(text):
        cited = _extract_citations(text)
        log_citation_event(question, text, retries=0, cited_docs=cited)
        return text

    # Retry with corrective instruction
    for _ in range(MAX_RETRIES):
        reason = why_failed(text)
        corrective = (
            f"\n\nREVISE:\nYour previous answer failed validation: {reason}\n"
            "Revise the answer. Append at least one citation using the required format."
        )
        response = llm.invoke(system=sys, user=user + corrective)
        text = response.text if hasattr(response, "text") else str(response)
        if has_min_citation(text):
            cited = _extract_citations(text)
            log_citation_event(question, text, retries=_ + 1, cited_docs=cited)
            return text

    # Last resort: surface explicit failure so tests can catch
    warn = text + "\n\n[WARNING] Citation validation failed."
    log_citation_event(question, warn, retries=MAX_RETRIES, cited_docs=[])
    return warn


_CIT_RE = re.compile(r"\((?P<doc>[^,]+),\s*S\.\s*(?P<page>\d+),\s*\"(?P<section>[^\"\n]+)\"\)")


def _extract_citations(answer: str) -> list[dict]:
    """Parse citations into structured dicts for logging/metrics."""
    cited = []
    for m in _CIT_RE.finditer(answer or ""):
        try:
            cited.append(
                {
                    "doc_short": (m.group("doc") or "").strip(),
                    "page": int(m.group("page") or 0),
                    "section": (m.group("section") or "").strip(),
                }
            )
        except Exception:
            continue
    return cited
