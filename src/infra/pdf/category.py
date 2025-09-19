from __future__ import annotations

from pathlib import Path

# Map parent folders → category label (extend as needed)
CATEGORY_RULES = {
    "Rechtliches": "Rechtliches",
    "Rules": "Regelwerk",
    "Guides": "Leitfäden",
    "Medizin": "Medizin",
    "Underwriting": "Underwriting",
}


def infer_category_from_path(file_path: str) -> str | None:
    p = Path(file_path)
    # Check all parents, nearest-first
    for parent in [p.parent] + list(p.parents):
        label = CATEGORY_RULES.get(parent.name)
        if label:
            return label
    return None
