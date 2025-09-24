from __future__ import annotations

"""API services: configured, cached use-case instances.

These are constructed via the composition root and imported by routes.
"""

from src.config.configure_app import get_import_use_case, get_query_use_case  # noqa: E402

import_use_case = get_import_use_case()
query_use_case = get_query_use_case()

__all__ = ["import_use_case", "query_use_case"]
