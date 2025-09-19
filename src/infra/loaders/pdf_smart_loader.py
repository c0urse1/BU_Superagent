from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    from langchain.schema import Document

from langchain_community.document_loaders import PyMuPDFLoader

from src.infra.pdf.category import infer_category_from_path
from src.infra.pdf.pdf_metadata import extract_pdf_docinfo


def load_pdf_with_metadata(path: str | Path) -> list[Document]:
    """
    Loads PDF pages as LangChain Documents and enriches metadata:
      - 'source': full file path
      - 'title': PDF title / first TOC entry / filename
      - 'page': 1-based page number (also keep 0-based if you prefer)
      - 'section': current section from TOC for that page ('' if none)
      - 'category': inferred from folder name, if any
    """
    path = str(Path(path).resolve())
    info = extract_pdf_docinfo(path)
    category = infer_category_from_path(path)

    # Use LangChain's PyMuPDFLoader to get per-page Documents
    base_docs = PyMuPDFLoader(path).load()  # one Document per page

    enriched: list[Document] = []
    for d in base_docs:
        md = dict(d.metadata or {})
        # LangChain commonly sets 'source' and 'page' already; enforce our values
        page_1 = (
            (md.get("page", md.get("page_number", 0)) or 0) + 1
            if isinstance(md.get("page"), int)
            else md.get("page_number", 1)
        )
        page_0 = (page_1 - 1) if isinstance(page_1, int) else 0
        # Build metadata
        new_md: dict[str, Any] = {
            **md,
            "source": path,
            "page": page_1,
            "page_index": page_0,
            "title": info.title,
            "section": info.page_to_section.get(page_0, ""),
        }
        if category:
            new_md["category"] = category
        enriched.append(Document(page_content=d.page_content, metadata=new_md))
    return enriched
