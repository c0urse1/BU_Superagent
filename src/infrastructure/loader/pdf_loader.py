from __future__ import annotations

"""PDF loader adapter implementing DocumentLoaderPort.

Uses the existing infra PDF helpers to read pages and metadata, then wraps results into
pure domain Document objects for use by the application layer.
"""

from collections.abc import Iterable  # noqa: E402
from pathlib import Path  # noqa: E402

from src.application.ports.loader_port import DocumentLoaderPort  # noqa: E402
from src.domain.document import Document as DomainDocument  # noqa: E402
from src.infra.loaders.pdf_smart_loader import load_pdf_with_metadata  # noqa: E402


class PdfDocumentLoader(DocumentLoaderPort):
    def load(self, path: str) -> Iterable[DomainDocument]:
        docs = load_pdf_with_metadata(Path(path))
        # Convert LangChain Documents to domain documents, preserving metadata
        out: list[DomainDocument] = []
        for d in docs:
            text = getattr(d, "page_content", "")
            md = getattr(d, "metadata", {}) or {}
            out.append(DomainDocument(content=text, metadata=dict(md)))
        return out


__all__ = ["PdfDocumentLoader"]
