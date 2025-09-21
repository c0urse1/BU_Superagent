from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from src.bu_kb.ingest.pipeline import _dedup_chunks_for_ingest
from src.core.settings import AppSettings
from src.infra.embeddings.factory import build_embeddings
from src.infra.vectorstores.chroma_store import ChromaStore


def test_ingest_dedup_hash(tmp_path: Path) -> None:
    cfg = AppSettings()
    cfg.dedup_ingest.enabled = True
    cfg.dedup_ingest.hash_enabled = True
    cfg.dedup_ingest.semantic_enabled = False

    # Use dummy embeddings to avoid external dependencies
    cfg.embeddings.provider = "dummy"
    emb = build_embeddings(cfg.embeddings)

    store = ChromaStore(
        collection="test_dedup_hash",
        persist_dir=tmp_path,
        embedder=emb,
    )

    d1 = Document(
        page_content="Ausschlüsse: Klettern im Outdoor-Bereich.",
        metadata={"source": "docA", "page": 1},
    )
    d2 = Document(
        page_content="Ausschlüsse:  Klettern  im   Outdoor-Bereich.",
        metadata={"source": "docB", "page": 2},
    )

    kept, skipped = _dedup_chunks_for_ingest([d1, d2], cfg, store, emb)
    assert len(kept) == 1 and len(skipped) == 1
    assert skipped[0].metadata.get("duplicate_reason") == "hash"
