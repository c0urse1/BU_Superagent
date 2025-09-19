from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from langchain_core.documents import Document

from ..exceptions import IngestionError
from .interfaces import Loader, Splitter, VectorStore

log = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        loader: Loader,
        splitter: Splitter,
        store: VectorStore,
        embedding_signature: str | None = None,
    ) -> None:
        self.loader = loader
        self.splitter = splitter
        self.store = store
        # When provided, stamp each chunk with the embedding signature to enable
        # filtered retrieval across multiple collections/models.
        self.embedding_signature = embedding_signature

    def run(self, files: Iterable[Path]) -> int:
        chunks_total = 0
        for pdf in files:
            try:
                docs: list[Document] = self.loader.load(str(pdf))
                chunks = self.splitter.split(docs)
                # Lightweight quality checks: summarize chunk stats per file
                try:
                    # Local import to avoid hard dependency at module import time
                    from src.core.settings import AppSettings

                    _cfg = AppSettings()
                    _min_merge = int(getattr(_cfg.chunking, "min_merge_char_len", 500))
                    _mode = str(getattr(_cfg.chunking, "mode", "sentence_aware"))
                except Exception:  # noqa: BLE001 - be resilient in CLI/legacy contexts
                    _min_merge = 500
                    _mode = "unknown"

                total_chars = sum(len(d.page_content or "") for d in chunks)
                avg_len = int(total_chars / max(len(chunks), 1))
                tiny = [d for d in chunks if len(d.page_content or "") < _min_merge]
                log.info(
                    "[ingest] chunks=%d avg_chars=%d tiny_chunks=%d mode=%s",
                    len(chunks),
                    avg_len,
                    len(tiny),
                    _mode,
                )
                # Optionally tag chunks with the embedding signature for later filtering
                if self.embedding_signature:
                    for d in chunks:
                        d.metadata = {
                            **(getattr(d, "metadata", None) or {}),
                            "embedding_sig": self.embedding_signature,
                        }
                self.store.add_documents(chunks)
                chunks_total += len(chunks)
                log.info("ingested %s -> %d chunks", pdf.name, len(chunks))
            except Exception as e:  # noqa: BLE001
                log.exception("failed to ingest %s", pdf)
                raise IngestionError(str(e)) from e
        self.store.persist()
        return chunks_total
