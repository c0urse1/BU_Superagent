from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import get_settings
from .embeddings import build_embedder


def _import_chroma() -> Any:
    try:
        # neuer Split
        from langchain_chroma import Chroma
    except Exception:  # noqa: BLE001
        # Fallback (ältere LangChain-Versionen)
        from langchain_community.vectorstores import Chroma
    return Chroma


class VectorStoreLoaderError(RuntimeError):
    pass


def load_vectorstore(persist_dir: Path | None = None, collection_name: str | None = None) -> Any:
    """
    Lädt eine bestehende Chroma-Collection für READ-ONLY Queries.
    Wir nutzen denselben Embedder wie beim Ingest.
    """
    s = get_settings()
    persist = str((persist_dir or s.persist_dir).resolve())
    collection = collection_name or s.collection_name

    # Sanity-Check: Existiert der Pfad & hat Chroma dort Daten?
    if not Path(persist).exists():
        raise VectorStoreLoaderError(
            f"Persist-Verzeichnis '{persist}' nicht gefunden. Bitte zuerst ingest laufen lassen."
        )

    Chroma = _import_chroma()
    embedder = build_embedder(s.embed_model)

    # Wichtig: embedding_function setzen, damit die Query richtig eingebettet wird.
    vs = Chroma(
        collection_name=collection,
        persist_directory=persist,
        embedding_function=embedder,
    )
    # Keine persist() oder add_documents() Aufrufe hier -> read-only Nutzung
    return vs
