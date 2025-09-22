from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Citation:
    doc_short: str  # short, human-friendly name (e.g., "BU Manual")
    page: int  # 1-based
    section: str  # section heading

    def render(self) -> str:
        """Render a canonical citation string.

        Format: (<doc_short>, S.<page>, "<section>")
        - page is coerced to int and treated as 1-based
        - section is stripped; empty becomes ""
        """
        sec = (self.section or "").strip()
        return f'({self.doc_short}, S.{int(self.page)}, "{sec}")'


def make_doc_short(source: str) -> str:
    """
    Turn a file path/title into a short label, e.g.:
    'docs/BU_Handbuch_v2.pdf' -> 'BU Handbuch' (then mapped to 'BU Manual')
    """
    import os
    import re

    base = os.path.basename(source or "").strip()
    # drop version suffixes like _v2 and .pdf
    base = re.sub(r"(_v\d+|\.pdf)$", "", base, flags=re.IGNORECASE)
    # Normalize separators to spaces (treat '+', '_', '-' alike)
    base = base.replace("_", " ").replace("-", " ").replace("+", " ").strip()
    # Collapse multiple spaces
    base = re.sub(r"\s+", " ", base)
    # Project-specific mapping if needed
    mapping = {
        "BU Handbuch": "BU Manual",
        "BU Superagent Technische Analyse": "BU Superagent",
    }
    for k, v in mapping.items():
        if base.lower().startswith(k.lower()):
            return v
    return base or "Unknown"
