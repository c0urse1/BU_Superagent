from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class PdfDocInfo:
    title: str | None
    toc: list[tuple[int, str, int]]  # (level, heading, page_index_1_based)
    page_to_section: dict[int, str]  # 0-based page index -> most relevant section title


def _build_page_to_section(toc: list[tuple[int, str, int]], page_count: int) -> dict[int, str]:
    """
    Convert TOC entries to a 'current section' label for each page.
    - toc page indices are 1-based; internal mapping is 0-based.
    """
    # Sort TOC by page
    toc_sorted = sorted(toc, key=lambda x: x[2])
    page_to_sec: dict[int, str] = {}
    cur_title: str = ""
    cur_ptr = 0
    for p in range(page_count):
        # advance current section if next TOC entry starts at/before this page
        while cur_ptr < len(toc_sorted) and (toc_sorted[cur_ptr][2] - 1) <= p:
            cur_title = toc_sorted[cur_ptr][1].strip()
            cur_ptr += 1
        page_to_sec[p] = cur_title
    return page_to_sec


def extract_pdf_docinfo(path: str | Path) -> PdfDocInfo:
    """
    Open a PDF and extract:
      - document title (PDF metadata or first TOC entry fallback)
      - full TOC (level, title, page)
      - per-page 'section' label derived from TOC
    """
    path = str(Path(path))
    with fitz.open(path) as doc:
        meta = doc.metadata or {}
        title = meta.get("title") or meta.get("Title")
        toc = doc.get_toc(simple=True) or []  # [(level, title, page_1based), ...]
        # Fallback title: first TOC heading or basename
        if not title:
            if toc:
                title = toc[0][1]
            else:
                title = Path(path).stem
        page_to_section = _build_page_to_section(toc, doc.page_count)
        return PdfDocInfo(title=title, toc=toc, page_to_section=page_to_section)
