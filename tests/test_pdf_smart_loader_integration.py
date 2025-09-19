from __future__ import annotations

from pathlib import Path

from src.bu_kb.ingest.loaders import PdfLoader
from src.bu_kb.ingest.splitters import TextSplitterAdapter
from src.infra.splitting.factory import build_splitter


def test_pdf_loader_enriches_metadata() -> None:
    pdf = Path("data/pdfs/Allianz_test.pdf")
    assert pdf.exists(), "Sample PDF missing for test"

    loader = PdfLoader()
    docs = loader.load(str(pdf))
    assert len(docs) >= 1
    md = docs[0].metadata or {}
    # Source and page info (normalize path separators)
    src = md.get("source", "")
    assert Path(src).name == "Allianz_test.pdf"
    assert isinstance(md.get("page"), int)
    # Enriched metadata
    assert "title" in md
    assert "section" in md  # may be empty string
    # category is optional (depends on folder naming); if present, it's a string or None
    if "category" in md:
        v = md["category"]
        assert (v is None) or isinstance(v, str)


def test_splitters_propagate_metadata_sentence_aware() -> None:
    pdf = Path("data/pdfs/Allianz_test.pdf")
    loader = PdfLoader()
    docs = loader.load(str(pdf))
    splitter = TextSplitterAdapter(
        build_splitter(mode="sentence_aware", chunk_size=300, chunk_overlap=50)
    )
    chunks = splitter.split(docs)
    assert chunks, "No chunks produced"
    md = chunks[0].metadata or {}
    for key in ("source", "page", "title", "section"):
        assert key in md


def test_splitters_propagate_metadata_recursive() -> None:
    pdf = Path("data/pdfs/Allianz_test.pdf")
    loader = PdfLoader()
    docs = loader.load(str(pdf))
    splitter = TextSplitterAdapter(
        build_splitter(mode="recursive", chunk_size=500, chunk_overlap=50)
    )
    chunks = splitter.split(docs)
    assert chunks, "No chunks produced"
    md = chunks[0].metadata or {}
    for key in ("source", "page", "title", "section"):
        assert key in md
