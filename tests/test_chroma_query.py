from __future__ import annotations

import math
from pathlib import Path

try:
    from chromadb.config import Settings as ChromaSettings
except Exception:  # pragma: no cover
    ChromaSettings = None

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
except Exception:  # pragma: no cover
    from langchain.embeddings.base import Embeddings
    from langchain.schema import Document

# Adjust to this project's import
from bu_kb.ingest.store import ChromaStore


class DummyEmbeddings(Embeddings):
    """Deterministic tiny embedding for tests (no external model)."""

    dim: int = 16

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        # simple hashed character-based features
        for i, ch in enumerate(text.lower()):
            vec[(i + ord(ch)) % self.dim] += 1.0
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def test_query_returns_relevant_chunk(tmp_path: Path) -> None:
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir(parents=True, exist_ok=True)

    _settings = ChromaSettings(anonymized_telemetry=False) if ChromaSettings else None

    store = ChromaStore(
        collection="test_collection",
        persist_dir=persist_dir,
        embedder=DummyEmbeddings(),
    )

    docs = [
        Document(page_content="Definition der Berufsunfähigkeit nach VVG."),
        Document(
            page_content=(
                "Gesundheitsprüfung: Fragen zu Vorerkrankungen und Behandlungen in den letzten 5 Jahren."
            )
        ),
    ]
    store.add_documents(docs)
    store.persist()

    hits = store.query("Vorerkrankungen", k=1)
    assert hits, "Expected at least one result"
    assert "Vorerkrankungen" in hits[0].page_content


def test_get_retriever_returns_docs(tmp_path: Path) -> None:
    persist_dir = tmp_path / "chroma2"
    persist_dir.mkdir(parents=True, exist_ok=True)

    store = ChromaStore(
        collection="test_collection_2",
        persist_dir=persist_dir,
        embedder=DummyEmbeddings(),
    )

    docs = [
        Document(page_content="Leitfaden zur Gesundheitsprüfung in der BU."),
        Document(page_content="Ausschlüsse bei Hochrisiko-Hobbys."),
    ]
    store.add_documents(docs)
    store.persist()

    retriever = store.get_retriever(k=2)
    # Use modern Retriever API: invoke(input) instead of deprecated get_relevant_documents
    results = retriever.invoke("Gesundheitsprüfung")
    assert results and "Gesundheitsprüfung" in results[0].page_content
