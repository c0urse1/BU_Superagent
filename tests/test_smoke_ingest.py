from __future__ import annotations

from pathlib import Path

import pytest
from langchain.schema import Document

from bu_kb.ingest.embeddings import build_embedder
from bu_kb.ingest.interfaces import Loader, Splitter, VectorStore
from bu_kb.ingest.pipeline import IngestionPipeline
from bu_kb.ingest.store import ChromaStore


class FakeLoader(Loader):
    def load(self, path: str) -> list[Document]:
        text = "BU ist wichtig.\nLeistungsfall, Gesundheitsfragen, VVG, Anzeigepflicht."
        return [Document(page_content=text, metadata={"source": path})]


class TinySplitter(Splitter):
    def split(self, docs: list[Document]) -> list[Document]:
        out: list[Document] = []
        for d in docs:
            mid = max(1, len(d.page_content) // 2)
            out.append(Document(page_content=d.page_content[:mid], metadata=d.metadata))
            out.append(Document(page_content=d.page_content[mid:], metadata=d.metadata))
        return out


@pytest.mark.slow
def test_pipeline_persists_chroma(tmp_path: Path) -> None:
    persist_dir = tmp_path / "chroma"
    try:
        embedder = build_embedder("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:  # noqa: BLE001
        # In offline environments or without cached models, skip this smoke test
        import pytest

        pytest.skip(f"Skipping smoke test: cannot load embedding model ({e})")
    store: VectorStore = ChromaStore("test_collection", persist_dir, embedder)
    pipeline = IngestionPipeline(loader=FakeLoader(), splitter=TinySplitter(), store=store)

    count = pipeline.run(files=[tmp_path / "dummy.pdf"])  # path only for metadata
    assert count == 2
    assert persist_dir.exists()
    assert any(persist_dir.iterdir())
