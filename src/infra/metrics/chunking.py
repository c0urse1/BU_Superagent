from __future__ import annotations

import json
import statistics
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol


class _LoggerLike(Protocol):
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None: ...


def log_chunking_stats(
    chunks: Iterable[dict[str, Any]],
    doc_meta: dict[str, Any],
    logger: _LoggerLike | None = None,
) -> None:
    """Write per-document chunking stats to .logs/*.jsonl and optionally log a summary.

    Expected chunk shape: {"length": int}. Any extra keys are ignored.
    doc_meta should contain: {"name": str, "min_chars": int, "max_chars": int}.
    """
    lengths = [int(c.get("length", 0)) for c in chunks]
    if not lengths:
        return
    stats = {
        "doc": doc_meta.get("name"),
        "chunks": len(lengths),
        "mean": (
            statistics.fmean(lengths) if hasattr(statistics, "fmean") else statistics.mean(lengths)
        ),
        "p50": statistics.median(lengths),
        "p90": statistics.quantiles(lengths, n=10)[8] if len(lengths) >= 10 else max(lengths),
        "p99": max(lengths),
        "within_min_max": (
            sum(
                int(doc_meta.get("min_chars", 0)) <= val <= int(doc_meta.get("max_chars", 10**9))
                for val in lengths
            )
            / max(1, len(lengths))
        ),
        "ts": int(time.time()),
    }
    path = Path(".logs") / f"chunking_stats_{stats['ts']}.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(stats, ensure_ascii=False) + "\n")
    except Exception:
        # Ignore filesystem failures silently to avoid breaking ingestion
        pass
    if logger:
        try:
            logger.info("Chunking stats: %s", stats)
        except Exception:
            pass
