from __future__ import annotations

try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    from langchain.schema import Document

from src.infra.splitting.sentence_chunker import (
    SentenceAwareChunker,
    SentenceAwareParams,
)


def test_section_injection_once() -> None:
    chunker = SentenceAwareChunker(SentenceAwareParams())
    d1 = Document(
        page_content="Intro text.",
        metadata={
            "source": "X",
            "page": 1,
            "section": "Definitions",
            "num_sentences": 1,
            "char_len": 11,
        },
    )
    d2 = Document(
        page_content="More details here.",
        metadata={
            "source": "X",
            "page": 2,
            "section": "Definitions",
            "num_sentences": 2,
            "char_len": 19,
        },
    )
    docs = [d1, d2]
    injected = chunker._inject_section_titles(
        docs, enabled=True, inject_once=True, fmt="{section}: {text}"
    )
    assert injected[0].metadata.get("section_injected") is True
    assert injected[0].page_content.startswith("Definitions:")
    # second chunk in same section should not be re-prefixed
    assert not injected[1].page_content.startswith("Definitions:")
