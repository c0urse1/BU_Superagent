from __future__ import annotations

"""Composition root: assemble and expose application use-cases.

Provides cached getters to avoid re-building adapters repeatedly in long-lived
processes (API/CLI). Keeps environment/settings handling inside the config layer.
"""

from functools import lru_cache  # noqa: E402

from src.application.use_cases import ImportDocumentsUseCase, QueryUseCase  # noqa: E402
from src.config.composition_sam import build_import_use_case, build_query_use_case  # noqa: E402


@lru_cache(maxsize=1)
def get_import_use_case() -> ImportDocumentsUseCase:
    return build_import_use_case()


@lru_cache(maxsize=1)
def get_query_use_case() -> QueryUseCase:
    return build_query_use_case()


def configure_app() -> tuple[ImportDocumentsUseCase, QueryUseCase]:
    """Convenience function returning both primary use-cases.

    Returns:
        (import_use_case, query_use_case)
    """
    return get_import_use_case(), get_query_use_case()


__all__ = [
    "get_import_use_case",
    "get_query_use_case",
    "configure_app",
]
