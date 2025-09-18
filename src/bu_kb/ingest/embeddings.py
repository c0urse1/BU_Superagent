from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings


def build_embedder(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # add CUDA detection later if desired
        encode_kwargs={"normalize_embeddings": True},
    )
