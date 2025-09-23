from __future__ import annotations

import os

from pydantic import Field
from pydantic_settings import BaseSettings


class VectorDatabaseConfig(BaseSettings):
    """Central embedding settings for vector DB wiring.

    Defaults to multilingual E5 instruct (1024-d). Honors EMBEDDING_MODEL_NAME and
    falls back to EMBEDDING_MODEL for legacy environments.
    """

    embedding_model: str = Field(
        default=os.getenv(
            "EMBEDDING_MODEL_NAME",
            os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct"),
        ),
        description="Embedding Model Name (default: E5-instruct)",
    )
    embedding_dim: int = Field(default=1024, description="Embedding dimension (E5-instruct = 1024)")
