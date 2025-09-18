from __future__ import annotations

from typing import Any


def build_embedder(model_name: str) -> Any:
    """
    Baut einen HuggingFace-Embedder. Normalisierte Embeddings verbessern die Suche.
    Versucht zuerst den neuen LangChain-Split, fällt sonst auf den älteren Pfad zurück.
    """
    try:
        # Neuer Split (empfohlen)
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception:  # noqa: BLE001
        # Älterer Pfad (Fallback)
        from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
