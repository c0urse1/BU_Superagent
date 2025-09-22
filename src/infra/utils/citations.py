from __future__ import annotations

import os
from pathlib import Path


def make_doc_short(source: str | None = None, title: str | None = None) -> str:
    """Return a short, human-readable document label for citations.

    Preference order:
    1) Non-empty title
    2) Basename of source path without extension
    3) "Unknown"
    """
    t = (title or "").strip()
    if t:
        return t
    s = (source or "").strip()
    if s:
        base = os.path.basename(Path(s).as_posix())
        # drop common extensions
        name, _ext = os.path.splitext(base)
        return name or base
    return "Unknown"
