from __future__ import annotations

# Thin entrypoint that delegates to the SAM-based interface CLI
from src.interface.cli import app

if __name__ == "__main__":
    app()
