from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from src.core.settings import AppSettings, EmbeddingConfig
from src.infra.embeddings.factory import build_embeddings
from src.infra.vectorstores.chroma_store import ChromaStore, collection_name_for


def test_per_model_collection(tmp_path: Path) -> None:
    app = AppSettings()
    app.kb.persist_directory = str(tmp_path)

    a = EmbeddingConfig(provider="dummy", model_name="A")
    b = EmbeddingConfig(provider="dummy", model_name="B")

    for cfg in (a, b):
        emb = build_embeddings(cfg)
        coll = collection_name_for(app.kb.collection_base, cfg.signature)
        store = ChromaStore(coll, tmp_path, emb)
        d = Document(
            page_content=f"model={cfg.model_name}",
            metadata={"embedding_sig": cfg.signature},
        )
        store.add_documents([d])

    # Query with model A should not see docs from model B
    embA = build_embeddings(a)
    collA = collection_name_for(app.kb.collection_base, a.signature)
    storeA = ChromaStore(collA, tmp_path, embA)
    hits = storeA.query("model=A", k=2, filter={"embedding_sig": a.signature})
    assert hits and "model=A" in hits[0].page_content
