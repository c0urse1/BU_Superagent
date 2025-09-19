from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Provider = Literal["huggingface", "openai", "dummy"]


class EmbeddingConfig(BaseModel):
    provider: Provider = Field(default="huggingface")
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    device: str = Field(default="auto")  # "auto" | "cpu" | "cuda"
    normalize_embeddings: bool = True
    # OpenAI specific
    openai_api_key: str | None = None
    openai_base_url: str | None = None  # e.g., Azure endpoint
    # Operational
    batch_size: int = 32

    @property
    def signature(self) -> str:
        # Stable signature to tag collections / metadata
        return f"{self.provider}:{self.model_name}:{'norm' if self.normalize_embeddings else 'raw'}"


class KBSettings(BaseModel):
    collection_base: str = "bu_knowledge"
    persist_directory: str = ".vector_store/chroma"


class AppSettings(BaseModel):
    embeddings: EmbeddingConfig = EmbeddingConfig()
    kb: KBSettings = KBSettings()
