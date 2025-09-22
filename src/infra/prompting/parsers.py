from __future__ import annotations

import re

CITATION_RE = re.compile(r"\([^)]+,\s*S\.\s*\d+,\s*\"[^\"\n]+\"\)", re.UNICODE)


def has_min_citation(text: str) -> bool:
    return bool(CITATION_RE.search(text or ""))


def why_failed(text: str) -> str:
    if not has_min_citation(text or ""):
        return 'Missing at least one citation in format (<doc>, S.<page>, "<section>").'
    return ""
