from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from ..exceptions import IngestionError
from .interfaces import Loader, Splitter, VectorStore

log = logging.getLogger(__name__)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _dedup_chunks_for_ingest(
    chunks: list[Document], settings: Any, store: VectorStore, embedding: Any
) -> tuple[list[Document], list[Document]]:
    # Lazy imports to avoid hard deps at module import time
    try:
        from src.infra.utils.text_norm import normalize_text, sha256_of
    except Exception:  # noqa: BLE001
        # If utils are unavailable, skip dedup entirely
        return chunks, []

    seen_hashes: set[str] = set()
    kept: list[Document] = []
    skipped: list[Document] = []

    do_hash = bool(getattr(settings.dedup_ingest, "enabled", True)) and bool(
        getattr(settings.dedup_ingest, "hash_enabled", True)
    )
    do_sem = bool(getattr(settings.dedup_ingest, "enabled", True)) and bool(
        getattr(settings.dedup_ingest, "semantic_enabled", True)
    )
    thr = float(getattr(settings.dedup_ingest, "similarity_threshold", 0.95))
    min_len = int(getattr(settings.dedup_ingest, "min_chars_for_hash", 20))

    # For semantic near-dup within the same document
    kept_embeds: list[tuple[int, list[float]]] = []  # (index in kept, embedding)

    # Best-effort: cross-run hash existence check
    def _exists_by_hash(h: str) -> bool:
        fn = getattr(store, "exists_by_hash", None)
        if callable(fn):
            try:
                return bool(fn(h))
            except Exception:
                return False
        return False

    for d in chunks:
        # Ensure metadata is a dict
        d.metadata = dict(getattr(d, "metadata", {}) or {})
        text = d.page_content or ""
        if not text.strip():
            kept.append(d)
            continue

        # --- HASH DEDUP ---
        if do_hash and len(text) >= min_len:
            norm = normalize_text(text)
            h = sha256_of(norm)
            d.metadata["content_hash"] = h

            if _exists_by_hash(h) or h in seen_hashes:
                d.metadata["is_duplicate"] = True
                d.metadata["duplicate_reason"] = "hash"
                skipped.append(d)
                continue
            seen_hashes.add(h)

        # --- SEMANTIC DEDUP (same source only) ---
        if do_sem and embedding is not None:
            try:
                emb = embedding.embed_query(text)
            except Exception:  # noqa: BLE001
                emb = None
            if emb:
                is_near_dup = False
                for kept_idx, kept_vec in kept_embeds:
                    same_src = (kept[kept_idx].metadata or {}).get("source") == d.metadata.get(
                        "source"
                    )
                    if not same_src:
                        continue
                    if _cosine(emb, kept_vec) >= thr:
                        d.metadata["is_duplicate"] = True
                        d.metadata["duplicate_reason"] = "cosine"
                        d.metadata["duplicate_of"] = (
                            kept[kept_idx].metadata.get("content_hash") or kept_idx
                        )
                        is_near_dup = True
                        break
                if is_near_dup:
                    skipped.append(d)
                    continue
                else:
                    kept_embeds.append((len(kept), emb))

        kept.append(d)

    return kept, skipped


class IngestionPipeline:
    def __init__(
        self,
        loader: Loader,
        splitter: Splitter,
        store: VectorStore,
        embedding_signature: str | None = None,
    ) -> None:
        self.loader = loader
        self.splitter = splitter
        self.store = store
        # When provided, stamp each chunk with the embedding signature to enable
        # filtered retrieval across multiple collections/models.
        self.embedding_signature = embedding_signature

    def run(self, files: Iterable[Path]) -> int:
        chunks_total = 0
        for pdf in files:
            try:
                docs: list[Document] = self.loader.load(str(pdf))
                chunks = self.splitter.split(docs)
                # Metrics: per-document chunking stats (JSONL + log)
                try:
                    from src.infra.metrics.chunking import log_chunking_stats

                    # Build doc_meta and per-chunk length list in a tolerant way
                    doc_meta = {
                        "name": pdf.name,
                        "min_chars": int(
                            getattr(getattr(self, "settings", None), "chunk_min_chars", 350)
                        ),
                        "max_chars": int(
                            getattr(getattr(self, "settings", None), "chunk_max_chars", 700)
                        ),
                    }
                    # Prefer explicit char_len metadata; fallback to text length
                    safe_chunks = []
                    for d in chunks:
                        md = dict(getattr(d, "metadata", None) or {})
                        length = int(md.get("char_len", len(getattr(d, "page_content", "") or "")))
                        safe_chunks.append({"length": length})
                    log_chunking_stats(safe_chunks, doc_meta, logger=log)
                except Exception:
                    # best-effort: do not block ingestion on metrics
                    pass
                # Lightweight quality checks: summarize chunk stats per file
                try:
                    # Local import to avoid hard dependency at module import time
                    from src.core.settings import AppSettings

                    _cfg = AppSettings()
                    _min_merge = int(getattr(_cfg.chunking, "chunk_min_merge_char_len", 500))
                    _mode = str(getattr(_cfg.chunking, "mode", "sentence_aware"))
                except Exception:  # noqa: BLE001 - be resilient in CLI/legacy contexts
                    _min_merge = 500
                    _mode = "unknown"

                total_chars = sum(len(d.page_content or "") for d in chunks)
                avg_len = int(total_chars / max(len(chunks), 1))
                tiny = [d for d in chunks if len(d.page_content or "") < _min_merge]
                log.info(
                    "[ingest] chunks=%d avg_chars=%d tiny_chunks=%d mode=%s",
                    len(chunks),
                    avg_len,
                    len(tiny),
                    _mode,
                )
                # Optionally tag chunks with the embedding signature for later filtering
                if self.embedding_signature:
                    for d in chunks:
                        d.metadata = {
                            **(getattr(d, "metadata", None) or {}),
                            "embedding_sig": self.embedding_signature,
                        }
                # Dedup pass (hash + optional semantic)
                try:
                    from src.core.settings import AppSettings

                    cfg = AppSettings()
                except Exception:  # noqa: BLE001
                    cfg = type("_Cfg", (), {"dedup_ingest": type("_DI", (), {})()})()

                # Retrieve embedder (best-effort) from store
                embedding = getattr(self.store, "_embedder", None)
                if embedding is None:
                    try:
                        embedding = getattr(
                            getattr(self.store, "_db", None), "embedding_function", None
                        )
                    except Exception:
                        embedding = None

                unique_chunks, duplicates = _dedup_chunks_for_ingest(
                    chunks, cfg, self.store, embedding
                )
                hash_skips = sum(
                    1 for x in duplicates if (x.metadata or {}).get("duplicate_reason") == "hash"
                )
                cos_skips = sum(
                    1 for x in duplicates if (x.metadata or {}).get("duplicate_reason") == "cosine"
                )
                log.info(
                    "[ingest/dedup] kept=%d skipped_dups=%d (hash=%d, cosine=%d)",
                    len(unique_chunks),
                    len(duplicates),
                    hash_skips,
                    cos_skips,
                )

                # Timing: embedding + indexing path
                t0 = time.perf_counter()
                log.info("[ingest] timing started ...")
                self.store.add_documents(unique_chunks)
                dt_ms = int((time.perf_counter() - t0) * 1000)
                log.info("[ingest] embedding+indexing took %d ms", dt_ms)
                chunks_total += len(unique_chunks)
                log.info("ingested %s -> %d chunks", pdf.name, len(unique_chunks))
            except Exception as e:  # noqa: BLE001
                log.exception("failed to ingest %s", pdf)
                raise IngestionError(str(e)) from e
        self.store.persist()
        return chunks_total
