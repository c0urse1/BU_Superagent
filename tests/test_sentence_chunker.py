from __future__ import annotations

from langchain_core.documents import Document

from src.infra.splitting.sentence_chunker import (
    SentenceAwareChunker,
    SentenceAwareParams,
)


def test_no_mid_sentence_cut_and_overlap_merge() -> None:
    long_sentence = (
        "Dies ist ein sehr langer Satz, der nicht mitten drin getrennt werden sollte, "
        "auch dann nicht, wenn die Chunk-Grenze erreicht ist."
    )
    short = "Kurz."
    text = f"{long_sentence} {short} Noch ein kurzer Satz."

    params = SentenceAwareParams(
        chunk_size=80, chunk_overlap=20, max_overflow=120, min_merge_char_len=60
    )
    chunker = SentenceAwareChunker(params)
    docs = [Document(page_content=text, metadata={"source": "t", "page": 1})]
    chunks = chunker.split_documents(docs)

    # Ensure no chunk ends mid-word (proxy for mid-sentence)
    for c in chunks:
        s = c.page_content
        # very lenient: last char should be not a bare alphabetic (should end with punctuation)
        assert not (s and s[-1].isalpha()), f"Chunk appears to end mid-sentence: {s[-40:]}"

    # Ensure tiny neighbor chunks were merged where possible (expect <= 2 chunks)
    assert len(chunks) <= 2
    # Page/source retained
    assert all(c.metadata.get("page") == 1 for c in chunks)
    assert all(c.metadata.get("source") == "t" for c in chunks)
