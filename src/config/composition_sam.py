from __future__ import annotations

"""SAM composition helpers building use-cases with configured adapters.

Keeps environment/settings handling out of the interface layer.
"""

from src.application.use_cases import ImportDocumentsUseCase, QueryUseCase  # noqa: E402
from src.config.llm import build_llm_from_env  # noqa: E402
from src.config.providers import (  # noqa: E402
    build_chroma_vectorstore_with_provider,
    resolve_embedding_config,
)
from src.core.settings import AppSettings  # noqa: E402
from src.domain.chunking import ChunkingConfig, SentenceChunker  # noqa: E402
from src.domain.dedup import DedupConfig, DuplicateDetector  # noqa: E402
from src.infrastructure.embeddings import build_embeddings_provider  # noqa: E402
from src.infrastructure.loader.pdf_loader import PdfDocumentLoader  # noqa: E402


def build_import_use_case() -> ImportDocumentsUseCase:
    app = AppSettings()
    # Embeddings provider chosen by config (huggingface/openai) and model/device options with env overlay
    emb_cfg = resolve_embedding_config(app)
    emb = build_embeddings_provider(emb_cfg)
    # Vector store adapter (E5-aware persist dir + namespaced collection)
    store = build_chroma_vectorstore_with_provider()
    # Chunking mapped from AppSettings
    c = app.chunking
    chunker_cfg = ChunkingConfig(
        chunk_size=int(c.chunk_size),
        chunk_overlap=int(c.chunk_overlap),
        chunk_max_overflow=int(c.chunk_max_overflow),
        chunk_min_merge_char_len=int(c.chunk_min_merge_char_len),
    )
    chunker = SentenceChunker(chunker_cfg)
    # Dedup config from settings
    di = app.dedup_ingest
    dedup_cfg = DedupConfig(
        enabled=bool(di.enabled),
        hash_enabled=bool(di.hash_enabled),
        semantic_enabled=bool(di.semantic_enabled),
        min_chars_for_hash=int(di.min_chars_for_hash),
        similarity_threshold=float(di.similarity_threshold),
        keep_strategy=str(di.keep_strategy or "first"),
    )
    dedup = DuplicateDetector(
        dedup_cfg, embed=(lambda s: emb.embed_query(s)) if dedup_cfg.semantic_enabled else None
    )
    return ImportDocumentsUseCase(
        loader=PdfDocumentLoader(), vector_store=store, chunker=chunker, dedup=dedup
    )


def build_query_use_case() -> QueryUseCase:
    app = AppSettings()
    store = build_chroma_vectorstore_with_provider()
    llm = build_llm_from_env()
    # Dedup configuration for retrieval-time deduplication
    dq = app.dedup_query
    dedup_cfg = DedupConfig(
        enabled=bool(dq.enabled),
        hash_enabled=(str(dq.method).lower() == "exact"),
        semantic_enabled=(str(dq.method).lower() == "cosine"),
        similarity_threshold=float(dq.similarity_threshold),
    )
    # Inject embedder for MMR fallback and semantic dedup if configured
    emb_cfg = resolve_embedding_config(app)
    emb = build_embeddings_provider(emb_cfg)
    dedup = DuplicateDetector(
        dedup_cfg, embed=(lambda s: emb.embed_query(s)) if dedup_cfg.semantic_enabled else None
    )
    return QueryUseCase(store=store, llm=llm, dedup=dedup, embedder=emb)


__all__ = ["build_import_use_case", "build_query_use_case"]
