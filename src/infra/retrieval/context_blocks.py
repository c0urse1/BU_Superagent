from __future__ import annotations

from .citations import Citation, make_doc_short


def chunk_to_block(chunk: dict) -> str:
    """
    chunk: {"text": str, "metadata": {"source": str, "page": int, "section": str}}
    """
    md = (chunk or {}).get("metadata", {}) or {}
    cit = Citation(
        doc_short=make_doc_short(md.get("source", "Unknown")),
        page=md.get("page", 0) or 0,
        section=md.get("section", "") or "",
    ).render()

    # Bracketed header the model can re-use verbatim:
    header = f"[Source {cit}]\n"
    text = (chunk or {}).get("text", "") or ""
    return header + text.strip()
