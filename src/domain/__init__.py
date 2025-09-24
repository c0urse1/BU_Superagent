"""Domain layer: pure types and logic (no I/O, no external libs).

Keep this layer free of side-effects. Define entities and small pure helpers only.
"""

from .chunking import ChunkingConfig, SentenceChunker
from .context import assemble_context
from .dedup import DedupConfig, DuplicateDetector
from .document import Document, content_hash
from .retrieval import Chunk, ensure_sig
from .scoring import mmr, normalize_scores

__all__ = [
    "Document",
    "content_hash",
    "Chunk",
    "ensure_sig",
    "SentenceChunker",
    "ChunkingConfig",
    "DuplicateDetector",
    "DedupConfig",
    "normalize_scores",
    "mmr",
    "assemble_context",
]
