from __future__ import annotations

from langchain_core.documents import Document

from src.bu_kb.ingest.splitters import TextSplitterAdapter
from src.infra.splitting.factory import build_splitter
from src.infra.splitting.sentence_chunker import SentenceAwareChunker


def test_text_splitter_adapter_with_sentence_aware() -> None:
    """Test that TextSplitterAdapter works with sentence-aware splitter."""
    # Create a sentence-aware splitter via factory
    underlying_splitter = build_splitter(
        mode="sentence_aware",
        chunk_size=200,
        chunk_overlap=50
    )
    
    # Wrap it with the adapter
    adapter = TextSplitterAdapter(underlying_splitter)
    
    # Test that adapter implements the expected interface
    assert hasattr(adapter, 'split')
    assert callable(adapter.split)
    
    # Test functionality
    docs = [Document(
        page_content="Erste wichtige Aussage. Zweite längere Aussage mit vielen Details. Dritte abschließende Aussage.",
        metadata={"source": "adapter_test.pdf", "page": 1}
    )]
    
    chunks = adapter.split(docs)
    
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    assert all(isinstance(chunk, Document) for chunk in chunks)
    
    # Metadata should be preserved
    for chunk in chunks:
        assert chunk.metadata.get("source") == "adapter_test.pdf"
        assert chunk.metadata.get("page") == 1


def test_text_splitter_adapter_with_recursive() -> None:
    """Test that TextSplitterAdapter works with recursive splitter."""
    underlying_splitter = build_splitter(
        mode="recursive",
        chunk_size=150,
        chunk_overlap=30
    )
    
    adapter = TextSplitterAdapter(underlying_splitter)
    
    docs = [Document(
        page_content="Text for recursive splitting test. Should work with the adapter pattern.",
        metadata={"source": "recursive_test.pdf"}
    )]
    
    chunks = adapter.split(docs)
    
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    assert all(isinstance(chunk, Document) for chunk in chunks)


def test_adapter_fallback_behavior() -> None:
    """Test adapter fallback when splitter doesn't have split_documents."""
    # Mock splitter that only has split method
    class MockSplitterWithSplit:
        def split(self, docs):
            return docs  # Just return input as-is for testing
    
    mock_splitter = MockSplitterWithSplit()
    adapter = TextSplitterAdapter(mock_splitter)
    
    docs = [Document(page_content="Test content")]
    result = adapter.split(docs)
    
    assert result == docs


def test_adapter_error_handling() -> None:
    """Test that adapter raises appropriate error for unsupported splitters."""
    # Mock splitter that has neither split_documents nor split
    class UnsupportedSplitter:
        def some_other_method(self):
            pass
    
    unsupported = UnsupportedSplitter()
    adapter = TextSplitterAdapter(unsupported)
    
    docs = [Document(page_content="Test")]
    
    try:
        adapter.split(docs)
        assert False, "Should have raised AttributeError"
    except AttributeError as e:
        assert "split_documents or split" in str(e)


def test_cli_integration_compatibility() -> None:
    """Test that the CLI integration pattern works end-to-end."""
    # Simulate the CLI pattern: factory -> adapter -> pipeline usage
    
    # 1. Build splitter via factory (as CLI does)
    splitter_impl = build_splitter(
        mode="sentence_aware",
        chunk_size=300,
        chunk_overlap=75,
        max_overflow=100,
        min_merge_char_len=200
    )
    
    # 2. Wrap with adapter (as CLI does)
    splitter = TextSplitterAdapter(splitter_impl)
    
    # 3. Use in pipeline context
    test_docs = [
        Document(
            page_content=(
                "Die Berufsunfähigkeitsversicherung ist ein wichtiger Baustein der Absicherung. "
                "Sie zahlt eine monatliche Rente, wenn der Versicherte seinen Beruf nicht mehr ausüben kann. "
                "Die Gesundheitsprüfung ist dabei ein zentraler Punkt. "
                "Vorerkrankungen müssen wahrheitsgemäß angegeben werden."
            ),
            metadata={"source": "bu_guide.pdf", "page": 15}
        )
    ]
    
    # Process through the adapter
    chunks = splitter.split(test_docs)
    
    # Verify results
    assert len(chunks) >= 1
    
    for chunk in chunks:
        # Should preserve source metadata
        assert chunk.metadata.get("source") == "bu_guide.pdf"
        assert chunk.metadata.get("page") == 15
        
        # Should have sentence-aware chunking metadata
        assert "chunk_index" in chunk.metadata
        assert "char_len" in chunk.metadata
        
        # Content should not end mid-sentence
        content = chunk.page_content.strip()
        if content:
            assert not content[-1].isalpha(), f"Chunk may end mid-sentence: '{content[-30:]}'"