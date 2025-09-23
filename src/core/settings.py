from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# TODO: Add chunking config with adaptive target length and overlap.
# Goals:
# - Provide CHUNK_TARGET_CHARS (default 500), CHUNK_MIN_CHARS (e.g. 350), CHUNK_MAX_CHARS (e.g. 700)
# - Provide CHUNK_OVERLAP_CHARS (default 120)
# - Keep sentence-aware splitting; do not cut sentences.
# - Ensure section_injection and cross_page_title_merge stay supported via existing flags.
# - Expose via environment variables and safe defaults.


class ChunkingSettings(BaseSettings):
    chunk_target_chars: int = Field(500, alias="CHUNK_TARGET_CHARS")
    chunk_min_chars: int = Field(350, alias="CHUNK_MIN_CHARS")
    chunk_max_chars: int = Field(700, alias="CHUNK_MAX_CHARS")
    chunk_overlap_chars: int = Field(120, alias="CHUNK_OVERLAP_CHARS")
    enforce_sentence_boundaries: bool = Field(True, alias="CHUNK_ENFORCE_SENTENCE_BOUNDARIES")
    cross_page_title_merge: bool = Field(True, alias="CHUNK_CROSS_PAGE_TITLE_MERGE")
    inject_section_titles: bool = Field(True, alias="CHUNK_INJECT_SECTION_TITLES")


Provider = Literal["huggingface", "openai", "dummy"]


class EmbeddingConfig(BaseModel):
    provider: Provider = Field(default="huggingface")
    # Default multilingual model for DE-heavy corpora; override to
    # "sentence-transformers/all-MiniLM-L6-v2" for a smaller English-focused model.
    model_name: str = Field(default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    # Device selection for embeddings runtime: "auto" | "cpu" | "cuda" | "cuda:0" | "mps"
    device: str = Field(default="auto")
    normalize_embeddings: bool = True
    # OpenAI specific
    openai_api_key: str | None = None
    openai_base_url: str | None = None  # e.g., Azure endpoint
    # Operational
    batch_size: int = 32

    @property
    def signature(self) -> str:
        # Stable signature to tag collections / metadata, includes device and normalization
        return (
            f"{self.provider}:{self.model_name}:{self.device}"
            f":{'norm' if self.normalize_embeddings else 'raw'}"
        )


class KBSettings(BaseModel):
    collection_base: str = "bu_knowledge"
    persist_directory: str = ".vector_store/chroma"


# Deduplication settings
class DedupIngestConfig(BaseModel):
    enabled: bool = True
    hash_enabled: bool = True
    semantic_enabled: bool = True  # cosine-based near-dup check (same doc)
    min_chars_for_hash: int = 20
    similarity_threshold: float = 0.95  # 0..1 cosine
    keep_strategy: str = "first"  # "first" | "longest" | "authoritative"


class DedupQueryConfig(BaseModel):
    enabled: bool = True
    method: str = "cosine"  # "cosine" | "exact"
    similarity_threshold: float = 0.95


# Section/title context handling for chunking
class SectionContextConfig(BaseModel):
    enabled: bool = True
    # consider a chunk "title-only" when 0 sentences and few chars
    title_only_max_chars: int = 100
    # allow cross-page merge of title-only -> next page content
    cross_page_merge: bool = True
    # inject the section name into the first chunk of each section
    inject_section_title: bool = True
    # how to prefix
    section_prefix_format: str = "{section}: {text}"
    # only inject on first chunk per section (recommended)
    inject_once_per_section: bool = True


class AppSettings(BaseModel):
    embeddings: EmbeddingConfig = EmbeddingConfig()
    kb: KBSettings = KBSettings()

    # Chunking configuration (used by sentence-aware or recursive splitters)
    class ChunkingConfig(BaseModel):
        mode: str = Field(default="sentence_aware")  # or "recursive"
        chunk_size: int = 1000
        chunk_overlap: int = 150
        chunk_max_overflow: int = 200
        chunk_min_merge_char_len: int = 500

    chunking: AppSettings.ChunkingConfig = Field(
        default_factory=lambda: AppSettings.ChunkingConfig()
    )

    # New: Environment-backed adaptive chunking settings (non-breaking; parallel to legacy fields)
    adaptive_chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)

    # Deduplication configuration blocks (restored for backward compatibility)
    dedup_ingest: DedupIngestConfig = DedupIngestConfig()
    dedup_query: DedupQueryConfig = DedupQueryConfig()
    # Section/TOC context controls (restored)
    section_context: SectionContextConfig = SectionContextConfig()


# Minimal environment-backed settings facade used by pipeline helpers
class Settings(BaseSettings):
    # Only expose chunking knobs needed by resolve_chunking_config; safe defaults via env
    chunking: ChunkingSettings = ChunkingSettings()

    # New: Deduplication configuration blocks
    dedup_ingest: DedupIngestConfig = DedupIngestConfig()
    dedup_query: DedupQueryConfig = DedupQueryConfig()
    # New: Section/TOC context controls
    section_context: SectionContextConfig = SectionContextConfig()
