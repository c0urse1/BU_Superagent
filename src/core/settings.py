from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Provider = Literal["huggingface", "openai", "dummy"]


class EmbeddingConfig(BaseModel):
    provider: Provider = Field(default="huggingface")
    # Default multilingual model for DE-heavy corpora; override to
    # "sentence-transformers/all-MiniLM-L6-v2" for a smaller English-focused model.
    model_name: str = Field(default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
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


class ChunkingConfig(BaseModel):
    """Configuration for text chunking modes and parameters."""
    mode: str = Field(default="sentence_aware")  # or "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    max_overflow: int = 200  # renamed for consistency with factory
    min_merge_char_len: int = 500  # renamed for consistency with factory


class AppSettings(BaseModel):
    embeddings: EmbeddingConfig = EmbeddingConfig()
    kb: KBSettings = KBSettings()
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
