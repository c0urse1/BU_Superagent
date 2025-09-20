from __future__ import annotations

from langchain_core.documents import Document

from src.infra.loaders import load_pdf_with_metadata


class PdfLoader:
    """PyMuPDF-based loader; returns one Document per page with metadata."""

    def load(self, path: str) -> list[Document]:
        # Use the smart loader to enrich metadata (title/section/category) per page
        docs = load_pdf_with_metadata(path)
        # Ensure 'source' is always set (smart loader already sets it)
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata.setdefault("source", path)
        return docs
