from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from src.application.ports.loader_port import DocumentLoaderPort
from src.application.ports.vector_store_port import VectorStorePort
from src.application.use_cases.import_documents import ImportDocumentsUseCase
from src.application.use_cases.query_kb import QueryUseCase
from src.domain.document import Document

# --- Fakes -------------------------------------------------------------------


class FakeLoader(DocumentLoaderPort):
    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs

    def load(self, path: str) -> Iterable[Document]:
        # Ignore path; return predefined docs
        return list(self.docs)


class FakeStore(VectorStorePort):
    def __init__(self) -> None:
        self.added: list[Document] = []
        self.persist_called = 0
        self._data: list[Document] = []

    # Ingest
    def add_documents(self, docs: list[Document]) -> None:
        self.added.extend(docs)
        self._data.extend(docs)

    def persist(self) -> None:
        self.persist_called += 1

    # Query
    def search(
        self, query: str, k: int, *, metadata_filter: Mapping[str, object] | None = None
    ) -> list[Document]:
        # Very small simulation: return first k docs regardless of query
        return list(self._data)[:k]

    def search_with_scores(
        self, query: str, k: int, *, metadata_filter: Mapping[str, object] | None = None
    ) -> list[tuple[Document, float]]:
        # Return pairs with descending dummy scores
        docs = list(self._data)[:k]
        scores = [1.0 - (i * 0.1) for i in range(len(docs))]
        return list(zip(docs, scores, strict=False))


# --- Tests -------------------------------------------------------------------


def test_import_documents_use_case_ingests_and_persists(tmp_path: Path) -> None:
    # Prepare a fake PDF file structure (the loader ignores path contents in this fake)
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "doc1.pdf").write_text("dummy")

    # Fake loader returns one big Document which will be chunked into 1+ pieces depending on chunk_size
    base_doc = Document(
        content=("A." * 600), metadata={"source": str(pdf_dir / "doc1.pdf"), "page": 1}
    )
    loader = FakeLoader([base_doc])
    store = FakeStore()

    # Use default chunker/dedup from the composition in ImportDocumentsUseCase
    from src.domain.chunking import ChunkingConfig, SentenceChunker
    from src.domain.dedup import DedupConfig, DuplicateDetector

    chunker = SentenceChunker(ChunkingConfig(chunk_size=500, chunk_overlap=150))
    dedup = DuplicateDetector(DedupConfig(enabled=True, hash_enabled=True, semantic_enabled=False))

    uc = ImportDocumentsUseCase(loader=loader, vector_store=store, chunker=chunker, dedup=dedup)
    n = uc.execute(pdf_dir)

    assert n == len(store.added) > 0
    assert store.persist_called == 1


def test_query_use_case_retrieve_with_scores_and_threshold() -> None:
    # Prepare documents and store
    docs = [Document(content=f"Doc {i}", metadata={"source": f"s{i}", "page": i}) for i in range(5)]
    store = FakeStore()
    store.add_documents(docs)

    uc = QueryUseCase(store=store, llm=None, dedup=None)
    # Threshold is inclusive (>=). Use 0.85 so only 1.0 and 0.9 remain.
    pairs = uc.retrieve_top_k_with_scores("q", k=3, score_threshold=0.85)

    # Only the top two scores 1.0 and 0.9 should pass the 0.85 threshold
    assert len(pairs) == 2
    s0, s1 = pairs[0][1], pairs[1][1]
    assert isinstance(s0, float) and isinstance(s1, float)
    assert s0 >= s1 and s1 >= 0.85


def test_query_use_case_ask_without_llm_returns_context() -> None:
    docs = [Document(content="Hello", metadata={"source": "s", "page": 1})]
    store = FakeStore()
    store.add_documents(docs)

    uc = QueryUseCase(store=store, llm=None, dedup=None)
    out = uc.ask("Q?", k=1)

    # Without LLM, .ask returns the assembled context string containing snippet
    assert "Hello" in out
