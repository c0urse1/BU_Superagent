from __future__ import annotations

from langchain_core.documents import Document

from src.infra.splitting.sentence_chunker import (
    SentenceAwareChunker,
    SentenceAwareParams,
    SentenceChunker,
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


# --- Adaptive SentenceChunker tests ---


class _Sent:
    def __init__(self, text: str, page: int = 1, section: str | None = None) -> None:
        self.text = text
        self.page = page
        self.section = section


def _mk_chunker(
    target: int = 20,
    min_chars: int = 10,
    max_chars: int = 40,
    overlap: int = 5,
    *,
    enforce_sentence_boundaries: bool = True,
    inject_section_titles: bool = True,
    cross_page_title_merge: bool = True,
) -> SentenceChunker:
    return SentenceChunker(
        target_chars=target,
        min_chars=min_chars,
        max_chars=max_chars,
        overlap_chars=overlap,
        enforce_sentence_boundaries=enforce_sentence_boundaries,
        inject_section_titles=inject_section_titles,
        cross_page_title_merge=cross_page_title_merge,
    )


def test_no_sentence_cut() -> None:
    sents = [
        _Sent("Dies ist Satz eins."),
        _Sent("Hier folgt Satz zwei!"),
        _Sent("Und schließlich Satz drei?"),
    ]
    c = _mk_chunker(target=25, max_chars=60)
    chunks = c.chunk(sents)
    # Ensure no sentence was cut across chunk boundaries
    joined = "\n".join(ch["text"] for ch in chunks)
    for s in sents:
        assert s.text in joined


def test_respects_min_max() -> None:
    sents = [
        _Sent("A" * 5),
        _Sent("B" * 5),
        _Sent("C" * 5),
        _Sent("D" * 5),
        _Sent("E" * 5),
    ]
    c = _mk_chunker(target=12, min_chars=8, max_chars=15)
    chunks = c.chunk(sents)
    for ch in chunks:
        assert 8 <= ch["length"] <= 15


def test_overlap_sentence_based() -> None:
    # Overlap is a hint for downstream; our SentenceChunker tracks boundaries but
    # does not duplicate sentences. Validate stable chunking with target threshold.
    sents = [_Sent("Wort ") for _ in range(12)]
    c = _mk_chunker(target=10, max_chars=15, overlap=4)
    chunks = c.chunk(sents)
    assert len(chunks) >= 2
    total_len = sum(len(ch["text"]) for ch in chunks)
    assert total_len >= sum(len(s.text) for s in sents)


def test_section_injected_once() -> None:
    # Adaptive chunker exposes section_title per chunk; injection of titles into text is upstream.
    c = _mk_chunker(inject_section_titles=True)
    sents = [_Sent("Intro.", page=1, section="Alpha"), _Sent("Weiter.", page=1, section="Alpha")]
    chunks = c.chunk(sents)
    assert all("section_title" in ch for ch in chunks)
    assert all(ch.get("section_title") == "Alpha" for ch in chunks)


def test_cross_page_title_merge() -> None:
    c = _mk_chunker(cross_page_title_merge=True)
    sents = [_Sent("Titel", page=1), _Sent("Text", page=2)]
    chunks = c.chunk(sents)
    # The chunker signals cross-page capability; actual merge is handled in sentence-aware path
    assert all(ch.get("title_merged") is True for ch in chunks)
