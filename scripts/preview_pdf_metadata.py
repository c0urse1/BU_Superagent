from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.infra.pdf.category import infer_category_from_path
from src.infra.pdf.pdf_metadata import extract_pdf_docinfo


def main() -> None:
    ap = argparse.ArgumentParser(description="Preview PDF title/TOC/section mapping.")
    ap.add_argument("pdf_path")
    args = ap.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        sys.stderr.write(f"Error: file not found: {pdf_path}\n")
        sys.exit(2)

    info = extract_pdf_docinfo(args.pdf_path)
    cat = infer_category_from_path(args.pdf_path)

    print("Title:", info.title)
    print("Category:", cat or "<none>")
    print("TOC entries (level, title, page):")
    for lvl, head, pg in info.toc[:20]:
        print(f"  {lvl}  {head}  @ p.{pg}")
    # Use ASCII arrow for Windows consoles
    print("Page->Section (first 10 pages):")
    for p in range(10):
        print(f"  p.{p+1}: {info.page_to_section.get(p,'')}")


if __name__ == "__main__":
    main()
