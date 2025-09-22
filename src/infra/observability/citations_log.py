from __future__ import annotations

import logging
from typing import Any

from src.infra.prompting.parsers import has_min_citation

_log = logging.getLogger(__name__)


def log_citation_event(
    question: str, answer: str, retries: int, cited_docs: list[dict[str, Any]]
) -> None:
    """Log citation enforcement outcome.

    Parameters
    ----------
    question : str
        The input question asked.
    answer : str
        The final model answer text.
    retries : int
        How many corrective retries were needed (0 when first attempt passed).
    cited_docs : list[dict]
        Parsed citations from the answer, e.g. [{"doc_short": ..., "page": 12, "section": ...}].
    """
    passed = has_min_citation(answer or "")
    try:
        _log.info(
            "citation_event: passed=%s retries=%d cited=%d docs=%s",
            passed,
            int(retries),
            len(cited_docs or []),
            cited_docs,
        )
    except Exception:
        # Be non-fatal for logging issues
        pass
