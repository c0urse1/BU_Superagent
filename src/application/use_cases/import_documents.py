from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.application.ports.loader_port import DocumentLoaderPort
from src.application.ports.vector_store_port import VectorStorePort
from src.domain.chunking import SentenceChunker
from src.domain.dedup import DuplicateDetector
from src.domain.document import Document


@dataclass
class ImportDocumentsUseCase:
    loader: DocumentLoaderPort
    vector_store: VectorStorePort
    chunker: SentenceChunker
    dedup: DuplicateDetector

    def execute(self, source_dir: Path) -> int:
        """Ingest PDFs found recursively under source_dir.

        1) Load documents via loader port
        2) Chunk with domain chunker
        3) Deduplicate via domain detector
        4) Add to vector store and persist
        Returns number of stored chunks.
        """
        paths = sorted([p for p in Path(source_dir).rglob("*.pdf") if p.is_file()])
        if not paths:
            return 0

        loaded: list[Document] = []
        for p in paths:
            for d in self.loader.load(str(p)):
                loaded.append(d)

        # Chunk
        chunks: list[Document] = []
        for d in loaded:
            # Attach upstream metadata fields onto each chunk
            for ch in self.chunker.split(
                d.content,
                source=d.source,
                page=int(d.page) if isinstance(d.page, int) or str(d.page).isdigit() else None,
            ):
                # merge metadata while preserving existing fields
                meta = dict(d.metadata)
                meta.update(ch.metadata)
                chunks.append(Document(content=ch.content, metadata=meta))

        # Dedup
        kept, _skipped = self.dedup.unique(chunks)

        if not kept:
            return 0

        self.vector_store.add_documents(kept)
        self.vector_store.persist()
        return len(kept)
