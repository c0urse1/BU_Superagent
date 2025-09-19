from __future__ import annotations


def auto_device() -> str:
    """Best-effort device detection.

    Preference order: CUDA (NVIDIA) → MPS (Apple Silicon) → CPU.
    Never raises; falls back to "cpu" on any error or when no accelerator is available.
    """
    # CUDA first (NVIDIA)
    try:
        import torch  # noqa: F401

        tmod = __import__("torch")
        if hasattr(tmod, "cuda") and getattr(tmod.cuda, "is_available", None):
            if tmod.cuda.is_available():
                return "cuda"
    except Exception:
        pass

    # Apple Silicon MPS
    try:
        import torch  # noqa: F401

        tmod = __import__("torch")
        backends = getattr(tmod, "backends", None)
        mps = getattr(backends, "mps", None) if backends is not None else None
        if mps is not None and getattr(mps, "is_available", None):
            if mps.is_available():
                return "mps"
    except Exception:
        pass

    return "cpu"


def resolve_device(requested: str) -> str:
    """Return the requested device or resolve automatically when set to "auto".

    Examples:
    - resolve_device("cpu") -> "cpu"
    - resolve_device("cuda:0") -> "cuda:0"
    - resolve_device("auto") -> auto_device() (e.g., "cuda" | "mps" | "cpu")
    - resolve_device("") -> auto_device()
    """
    try:
        if requested and requested.lower() != "auto":
            return requested
    except Exception:
        # Defensive: if requested isn't a str-like, fall back to auto
        pass
    return auto_device()
