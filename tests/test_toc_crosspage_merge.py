from __future__ import annotations

try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    from langchain.schema import Document

from src.infra.splitting.sentence_chunker import (
    SentenceAwareChunker,
    SentenceAwareParams,
)


def test_title_only_merges_into_next_page() -> None:
    chunker = SentenceAwareChunker(SentenceAwareParams(chunk_size=1000, chunk_overlap=0))
    # simulate two docs: page 5 title-only, page 6 content
    d5 = Document(
        page_content="Section 2: Definitions",
        metadata={"source": "X", "page": 5, "num_sentences": 0, "char_len": 23},
    )
    d6 = Document(
        page_content="This insurance covers ...",
        metadata={"source": "X", "page": 6, "num_sentences": 2, "char_len": 28},
    )
    # bypass split_documents; call private flow by emulating final passes
    docs = [d5, d6]
    # ensure our merge function exists; use config defaults
    merged = chunker._merge_title_into_next_page(docs, title_max_chars=100, enable=True)
    assert len(merged) == 1
    assert merged[0].metadata.get("title_merged") is True
    assert "Section 2: Definitions" in merged[0].page_content
