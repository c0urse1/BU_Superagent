from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Ensure repository root on path to import infra factory
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Avoid optional torchvision import path in transformers for text-only flows
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

from src.core.settings import AppSettings  # noqa: E402
from src.infra.embeddings.factory import build_embeddings  # noqa: E402


def main() -> None:
    cfg = AppSettings().embeddings.model_copy()
    # Force E5 defaults explicitly for this smoke test
    cfg.provider = "huggingface"
    cfg.model_name = "intfloat/multilingual-e5-large-instruct"
    cfg.normalize_embeddings = True

    emb = build_embeddings(cfg)
    v = np.asarray(emb.embed_query("ping"), dtype=np.float32)
    dim = v.shape[0]
    l2 = float(np.linalg.norm(v))
    print(f"dim: {dim}")
    print(f"L2 norm: {l2:.6f}")


if __name__ == "__main__":
    main()
