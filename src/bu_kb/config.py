from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    source_dir: Path = Field(default=Path("data/pdfs"))
    persist_dir: Path = Field(default=Path(".vector_store/chroma"))
    collection_name: str = Field(default="bu_knowledge")
    # Primary name as requested: embed_model; accept common env aliases
    embed_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias=AliasChoices("KB_EMBED_MODEL", "KB_EMBEDDING_MODEL"),
    )
    # Keep ingestion parameters for existing pipeline components
    chunk_size: int = 1000
    chunk_overlap: int = 150

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="KB_",  # supports KB_PERSIST_DIR, KB_COLLECTION_NAME, KB_EMBED_MODEL, etc.
    )

    # Back-compat attribute name used elsewhere in the repo
    @property
    def embedding_model(self) -> str:  # pragma: no cover - simple alias
        return self.embed_model


# Cached accessor as requested (shared between ingest and query without re-parsing env)
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Preserve original module-level instance for existing imports
settings = get_settings()
