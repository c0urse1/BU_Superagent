from __future__ import annotations

from langchain_core.documents import Document

from src.infra.splitting.factory import build_splitter
from src.infra.splitting.sentence_chunker import SentenceAwareChunker
from src.infra.splitting.types import TextSplitterLike


def test_factory_builds_sentence_aware_splitter() -> None:
    """Test that factory creates sentence-aware splitter with correct params."""
    splitter = build_splitter(
        mode="sentence_aware",
        chunk_size=800,
        chunk_overlap=100,
        max_overflow=150,
        min_merge_char_len=400
    )
    
    assert isinstance(splitter, SentenceAwareChunker)
    assert splitter.p.chunk_size == 800
    assert splitter.p.chunk_overlap == 100
    assert splitter.p.max_overflow == 150
    assert splitter.p.min_merge_char_len == 400


def test_factory_builds_recursive_splitter() -> None:
    """Test that factory creates recursive splitter."""
    splitter = build_splitter(
        mode="recursive",
        chunk_size=600,
        chunk_overlap=50
    )
    
    # Should implement the TextSplitterLike protocol
    assert hasattr(splitter, 'split_documents')
    assert callable(splitter.split_documents)
    
    # Basic functionality test
    docs = [Document(page_content="Test text. Another sentence.")]
    result = splitter.split_documents(docs)
    assert isinstance(result, list)


def test_factory_default_sentence_aware() -> None:
    """Test that factory defaults to sentence-aware mode."""
    # No mode specified should default to sentence_aware
    splitter = build_splitter(chunk_size=1000, chunk_overlap=150)
    
    assert isinstance(splitter, SentenceAwareChunker)
    assert splitter.p.chunk_size == 1000
    assert splitter.p.chunk_overlap == 150


def test_splitter_protocol_compliance() -> None:
    """Test that both splitter types implement TextSplitterLike protocol."""
    sentence_aware = build_splitter(mode="sentence_aware")
    recursive = build_splitter(mode="recursive")
    
    # Both should have split_documents method
    assert hasattr(sentence_aware, 'split_documents')
    assert hasattr(recursive, 'split_documents')
    
    # Test with sample document
    test_doc = Document(
        page_content="Erste Aussage. Zweite längere Aussage mit mehr Details. Dritte kurze Aussage.",
        metadata={"source": "test.pdf", "page": 1}
    )
    
    # Both should process the document
    sa_result = sentence_aware.split_documents([test_doc])
    rec_result = recursive.split_documents([test_doc])
    
    assert isinstance(sa_result, list)
    assert isinstance(rec_result, list)
    assert all(isinstance(doc, Document) for doc in sa_result)
    assert all(isinstance(doc, Document) for doc in rec_result)


def test_sentence_aware_vs_recursive_behavior() -> None:
    """Test that sentence-aware and recursive splitters behave differently."""
    text = (
        "Dies ist ein sehr langer Satz mit vielen Details, der demonstrieren soll, "
        "wie sich die beiden Splitting-Modi unterscheiden. "
        "Ein zweiter Satz folgt hier. "
        "Und noch ein dritter kurzer Satz."
    )
    
    doc = Document(page_content=text, metadata={"source": "comparison.pdf"})
    
    # Create splitters with same basic params
    sentence_aware = build_splitter(mode="sentence_aware", chunk_size=100, chunk_overlap=20)
    recursive = build_splitter(mode="recursive", chunk_size=100, chunk_overlap=20)
    
    sa_chunks = sentence_aware.split_documents([doc])
    rec_chunks = recursive.split_documents([doc])
    
    # Sentence-aware should preserve sentence boundaries better
    for chunk in sa_chunks:
        content = chunk.page_content.strip()
        if content:
            # Should typically end with sentence punctuation
            assert content[-1] in '.!?', f"Sentence-aware chunk doesn't end properly: '{content[-20:]}'"
    
    # Results might differ in structure (though both should work)
    assert len(sa_chunks) >= 1
    assert len(rec_chunks) >= 1