from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document


class Loader(Protocol):
    def load(self, path: str) -> list[Document]: ...


class Splitter(Protocol):
    def split(self, docs: list[Document]) -> list[Document]: ...


class VectorStore(Protocol):
    def add_documents(self, docs: list[Document]) -> None: ...
    def persist(self) -> None: ...
