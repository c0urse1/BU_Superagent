from __future__ import annotations

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


class PdfLoader:
    """PyMuPDF-based loader; returns one Document per page with metadata."""

    def load(self, path: str) -> list[Document]:
        loader = PyMuPDFLoader(path)
        docs = loader.load()
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata.setdefault("source", path)
        return docs
