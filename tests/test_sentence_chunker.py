from __future__ import annotations

from langchain_core.documents import Document

from src.infra.splitting.sentence_chunker import (
    SentenceAwareChunker,
    SentenceAwareParams,
)
from src.infra.splitting.sentence_splitter import split_to_sentences


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


def test_sentence_splitting_abbreviations() -> None:
    """Test that common German abbreviations are handled correctly."""
    text = "Die BU-Versicherung ist wichtig z.B. für Selbständige. Das VVG regelt u.a. die Anzeigepflicht."
    sentences = split_to_sentences(text)
    
    # Should not split on abbreviations like z.B. and u.a.
    assert len(sentences) == 2
    assert "z.B. für Selbständige" in sentences[0]
    assert "u.a. die Anzeigepflicht" in sentences[1]


def test_metadata_preservation() -> None:
    """Test that all metadata fields are preserved through chunking."""
    text = "Erster Satz. Zweiter sehr langer Satz mit vielen Details. Dritter Satz."
    metadata = {
        "source": "bu_guide.pdf", 
        "page": 42,
        "custom_field": "test_value",
        "embedding_sig": "test:model:norm"
    }
    
    params = SentenceAwareParams(chunk_size=50, chunk_overlap=10, max_overflow=20)
    chunker = SentenceAwareChunker(params)
    docs = [Document(page_content=text, metadata=metadata)]
    chunks = chunker.split_documents(docs)
    
    # All chunks should preserve original metadata
    for chunk in chunks:
        assert chunk.metadata["source"] == "bu_guide.pdf"
        assert chunk.metadata["page"] == 42
        assert chunk.metadata["custom_field"] == "test_value"
        assert chunk.metadata["embedding_sig"] == "test:model:norm"
        # Should also have computed metadata
        assert "chunk_index" in chunk.metadata
        assert "char_len" in chunk.metadata
        assert "num_sentences" in chunk.metadata


def test_tiny_chunk_merging() -> None:
    """Test that tiny adjacent chunks from same source are merged."""
    # Create text that will produce tiny chunks
    text = "Kurz. Auch kurz. Winzig. Lang genug um nicht winzig zu sein und einen echten Chunk zu bilden."
    
    params = SentenceAwareParams(
        chunk_size=30, 
        chunk_overlap=5, 
        max_overflow=10, 
        min_merge_char_len=40
    )
    chunker = SentenceAwareChunker(params)
    docs = [Document(page_content=text, metadata={"source": "test", "page": 1})]
    chunks = chunker.split_documents(docs)
    
    # Should merge the tiny chunks at the beginning
    assert len(chunks) >= 1
    
    # Check that tiny chunks were indeed merged
    first_chunk = chunks[0]
    assert len(first_chunk.page_content) >= params.min_merge_char_len or len(chunks) == 1


def test_empty_document_handling() -> None:
    """Test that empty documents are handled gracefully."""
    chunker = SentenceAwareChunker()
    
    # Test empty document
    empty_docs = [Document(page_content="", metadata={"source": "empty"})]
    chunks = chunker.split_documents(empty_docs)
    assert len(chunks) == 0
    
    # Test whitespace-only document  
    whitespace_docs = [Document(page_content="   \n\n  \t  ", metadata={"source": "whitespace"})]
    chunks = chunker.split_documents(whitespace_docs)
    assert len(chunks) == 0


def test_chunk_statistics() -> None:
    """Test that chunk statistics are computed correctly."""
    text = "Erste Frage! Zweite wichtige Aussage? Dritte normale Feststellung."
    
    chunker = SentenceAwareChunker()
    docs = [Document(page_content=text, metadata={"source": "stats_test"})]
    chunks = chunker.split_documents(docs)
    
    assert len(chunks) == 1
    chunk = chunks[0]
    
    # Check computed statistics
    assert chunk.metadata["char_len"] == len(chunk.page_content)
    assert chunk.metadata["num_sentences"] == 3  # 1 '!' + 1 '?' + 1 '.'
    assert chunk.metadata["chunk_index"] == 0
