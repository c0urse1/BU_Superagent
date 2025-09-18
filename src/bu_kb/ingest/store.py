from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class ChromaStore:
    def __init__(self, collection: str, persist_dir: Path, embedder: HuggingFaceEmbeddings) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._db = Chroma(
            collection_name=collection,
            persist_directory=str(persist_dir),
            embedding_function=embedder,
        )

    def add_documents(self, docs: list[Document]) -> None:
        if docs:
            self._db.add_documents(docs)

    def persist(self) -> None:
        self._db.persist()
