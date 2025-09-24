from __future__ import annotations

from pathlib import Path

from src.core.settings import AppSettings
from src.infra.embeddings.factory import get_embedder
from src.infra.vectorstores.chroma_store import ChromaStore, collection_name_for


def get_embedding_signature() -> str:
    """Return the current embedding signature from AppSettings.

    Kept in config layer to avoid leaking settings throughout the app.
    """
    return AppSettings().embeddings.signature


def make_chroma_store() -> ChromaStore:
    """Build a ChromaStore using current settings and default persist dir.

    - Single persist dir strategy (vector_store)
    - Collection namespaced by embedding signature
    - Embeddings built via get_embedder() respecting env overlays
    """
    app = AppSettings()
    emb = get_embedder()
    collection = collection_name_for(app.kb.collection_base, app.embeddings.signature)
    persist_dir = Path("vector_store")
    return ChromaStore(collection, persist_dir, emb)
