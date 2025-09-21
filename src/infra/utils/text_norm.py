from __future__ import annotations

import hashlib
import re

_WS = re.compile(r"\s+", re.UNICODE)


def normalize_text(s: str) -> str:
    """Lowercase, collapse whitespace, trim."""
    return _WS.sub(" ", (s or "").strip().lower())


def sha256_of(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()
