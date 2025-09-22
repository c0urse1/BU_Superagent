from __future__ import annotations

import argparse

from langchain_core.documents import Document

from src.infra.splitting.factory import build_splitter


def main() -> None:
    ap = argparse.ArgumentParser(description="Preview sentence-aware chunking.")
    ap.add_argument("--mode", default="sentence_aware")
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--chunk-overlap", type=int, default=150)
    ap.add_argument("--max-overflow", type=int, default=200)
    ap.add_argument("--min-merge", type=int, default=500)
    ap.add_argument("--source", default="stdin", help="display-only source name")
    ap.add_argument("--page", type=int, default=1)
    args = ap.parse_args()

    text = "".join(__import__("sys").stdin.readlines())
    doc = Document(page_content=text, metadata={"source": args.source, "page": args.page})

    splitter = build_splitter(
        mode=args.mode,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_overflow=args.max_overflow,
        min_merge_char_len=args.min_merge,
    )
    chunks = splitter.split_documents([doc])

    for i, d in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} | chars={len(d.page_content)} | page={d.metadata.get('page')} ---")
        print(d.page_content)
        # after printing chunk content, print helpful markers if present
        print(
            "     meta:",
            "section_injected=" + str(d.metadata.get("section_injected")),
            "title_merged=" + str(d.metadata.get("title_merged")),
            "title_from_page=" + str(d.metadata.get("title_merged_from_page")),
        )


if __name__ == "__main__":
    main()
