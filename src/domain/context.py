from __future__ import annotations

from collections.abc import Iterable

from src.domain.document import Document


def assemble_context(docs: Iterable[Document], k: int | None = None) -> str:
    """Assemble a prompt-ready context with lightweight citations.

    Format per line:
    [i] Title | Section (p.Page) | Source\nSnippet...
    """
    out: list[str] = []
    max_docs = float("inf") if k is None else int(k)
    for i, d in enumerate(docs, 1):
        if i > max_docs:
            break
        title = d.title
        section = d.section
        page = d.page
        src = d.source
        header = f"[{i}] {title} | {section} (p.{page}) | {src}".rstrip()
        out.append(header)
        out.append((d.content or "").strip())
    return "\n\n".join(out)
