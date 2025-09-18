from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from langchain_core.documents import Document

from ..exceptions import IngestionError
from .interfaces import Loader, Splitter, VectorStore

log = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, loader: Loader, splitter: Splitter, store: VectorStore) -> None:
        self.loader = loader
        self.splitter = splitter
        self.store = store

    def run(self, files: Iterable[Path]) -> int:
        chunks_total = 0
        for pdf in files:
            try:
                docs: list[Document] = self.loader.load(str(pdf))
                chunks = self.splitter.split(docs)
                self.store.add_documents(chunks)
                chunks_total += len(chunks)
                log.info("ingested %s -> %d chunks", pdf.name, len(chunks))
            except Exception as e:  # noqa: BLE001
                log.exception("failed to ingest %s", pdf)
                raise IngestionError(str(e)) from e
        self.store.persist()
        return chunks_total
