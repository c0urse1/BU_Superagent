from __future__ import annotations


class IngestionError(Exception):
    """Raised when ingestion of a single file fails."""


class ConfigurationError(Exception):
    """Raised for invalid or missing configuration."""
