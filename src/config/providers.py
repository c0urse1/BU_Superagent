from __future__ import annotations

"""Composition helpers to build infrastructure adapters with providers.

Uses AppSettings to decide on embeddings provider and model, and wires the concrete
EmbeddingsPort into the ChromaVectorStore adapter.
"""

from pathlib import Path  # noqa: E402

from src.core.settings import AppSettings, EmbeddingConfig, Settings  # noqa: E402
from src.infra.vectorstores.chroma_store import collection_name_for  # noqa: E402
from src.infrastructure.embeddings import build_embeddings_provider  # noqa: E402
from src.infrastructure.vectorstore.chroma_vectorstore import ChromaVectorStore  # noqa: E402


def resolve_embedding_config(app: AppSettings | None = None) -> EmbeddingConfig:
    """Resolve final EmbeddingConfig using AppSettings with environment overlay from Settings().

    This centralizes the previous CLI-side merging (env → AppSettings) so all composition
    uses the same embedding provider/model/device/normalization configuration.
    """
    app = app or AppSettings()
    cfg = app.embeddings.model_copy()
    try:
        s_emb = Settings().embeddings
        if getattr(s_emb, "provider", None):
            cfg.provider = str(s_emb.provider)
        if getattr(s_emb, "model_name", None):
            cfg.model_name = str(s_emb.model_name)
        if getattr(s_emb, "device", None):
            cfg.device = str(s_emb.device)
        if getattr(s_emb, "normalize", None) is not None:
            cfg.normalize_embeddings = bool(s_emb.normalize)
        # E5 prefixing toggles (optional env)
        if getattr(s_emb, "e5_enable_prefix", None) is not None:
            cfg.e5_enable_prefix = bool(s_emb.e5_enable_prefix)
        if getattr(s_emb, "e5_query_instruction", None):
            cfg.e5_query_instruction = str(s_emb.e5_query_instruction)
        if getattr(s_emb, "e5_query_prefix", None):
            cfg.e5_query_prefix = str(s_emb.e5_query_prefix)
        if getattr(s_emb, "e5_passage_prefix", None):
            cfg.e5_passage_prefix = str(s_emb.e5_passage_prefix)
    except Exception:
        # Fail open: keep AppSettings defaults if env overlay fails
        pass
    return cfg


def _choose_persist_dir(cfg: EmbeddingConfig, override: Path | None = None) -> Path:
    if override is not None:
        return override
    # Optional: mirror legacy behavior of separate dir for E5 dimensionality
    # Default remains a single persist root for general cases.
    model = (cfg.model_name or "").lower()
    if "e5" in model:
        return Path("vector_store") / "e5_large"
    return Path("vector_store")


def build_chroma_vectorstore_with_provider(persist_dir: Path | None = None) -> ChromaVectorStore:
    app = AppSettings()
    cfg = resolve_embedding_config(app)
    embedder = build_embeddings_provider(cfg)
    collection = collection_name_for(app.kb.collection_base, cfg.signature)
    pdir = _choose_persist_dir(cfg, persist_dir)
    return ChromaVectorStore(collection, pdir, embedder)


def build_chroma_vectorstore_with_embedder(
    embedder: object, persist_dir: Path | None = None
) -> ChromaVectorStore:
    app = AppSettings()
    cfg = app.embeddings
    collection = collection_name_for(app.kb.collection_base, cfg.signature)
    pdir = _choose_persist_dir(cfg, persist_dir)
    return ChromaVectorStore(collection, pdir, embedder)


__all__ = [
    "resolve_embedding_config",
    "build_chroma_vectorstore_with_provider",
    "build_chroma_vectorstore_with_embedder",
]
