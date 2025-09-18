"""
Force UTF-8 encoding for stdio early in process startup.

This helps avoid UnicodeEncodeError on Windows consoles (cp1252) and ensures
consistent behavior across tools and hooks.
"""

from __future__ import annotations

import os
import sys

# Prefer Python's UTF-8 mode when not explicitly set
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

try:
    # Reconfigure stdio streams to UTF-8 if supported (Python 3.7+)
    if sys.stdout:
        getattr(sys.stdout, "reconfigure", lambda **_: None)(encoding="utf-8", errors="replace")
    if sys.stderr:
        getattr(sys.stderr, "reconfigure", lambda **_: None)(encoding="utf-8", errors="replace")
except Exception:
    # Never fail app startup due to terminal encoding quirks
    pass
