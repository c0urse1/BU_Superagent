## Strict Architecture Mode (SAM) — MUST FOLLOW

- Please apply Strict Architecture Mode (SAM) when generating code or refactors.
- Always structure proposals by layer: domain → application → infrastructure → interface → config.
- Domain: pure functions/classes, no I/O, no globals, no environment access.
- Application: use-cases + ports; orchestrates domain; no direct infra.
- Infrastructure: adapters implementing ports; wrap external libs (Chroma, OpenAI, DB, FS).
- Interface: CLI/HTTP; only parse/format + delegate to use-cases.
- Errors: typed classes or Result objects, never silent fails.
- Config: only in composition root.
- Prefer correctness & architecture over speed or pragmatism.

# Copilot instructions for BU_Superagent

Purpose: Make agents productive with PDF → Chroma ingestion and read‑only retrieval. Two paths: classic CLI (`src/bu_kb/cli.py`) and the infra/core stack (preferred). Compose existing helpers—don’t re‑wire basics.

## Architecture (+ SAM placement)
- Ingest (infrastructure): PDF → sentence‑aware splitter → Chroma. Use `src/infra/embeddings/factory.py::build_embeddings` and stamp every chunk with `metadata["embedding_sig"]`. Ingest de‑dup: hash + optional cosine (`AppSettings().dedup_ingest`).
- Query (infrastructure): `scripts/query_kb.py` uses `src/infra/vectorstores/chroma_store.py::ChromaStore` with `collection_name_for(...)`/`filter_for_signature(...)` to avoid cross‑model bleed; supports MMR and retrieval‑time dedup.
- API (interface): `src/api/routes/qa.py` composes `retrieve → assemble_context → answer_with_citations`; LLM provider in `src/services/llm/provider.py` exposes `.invoke()`.
- Domain/Application: keep pure/coordination logic here if you add new use‑cases. No I/O in domain; app orchestrates domain via ports. Infra wraps external libs (Chroma, transformers). Config only in `src/core/settings.py`.
- Reranking (infrastructure): optional local cross‑encoder (`BAAI/bge-reranker-v2-m3`) after vector Top‑N → final Top‑K via `src/infra/rerankers/factory.py::get_reranker()`.

## Conventions and gotchas
- Embeddings default: E5 instruct `intfloat/multilingual-e5-large-instruct` with instruction+prefixing and L2‑norm. Env overlay via `Settings().embeddings` (EMBEDDING_PROVIDER, EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE, NORMALIZE_EMBEDDINGS).
- Persisting: Single persist dir `vector_store`; collections are namespaced by embedding signature: `collection_name_for(app.kb.collection_base, emb_cfg.signature)`. Always filter queries by `{ "embedding_sig": emb_cfg.signature }`.
- Metadata to keep: `source` (abs path), `page` (1‑based), `title`, `section`, `category`, plus `content_hash`; duplicates may be marked `is_duplicate`.
- Read‑only queries: never call `persist()`/`add_documents()`; use the VectorStore adapter (`src/infra/vectorstores/chroma_store.py::ChromaStore`) via config/composition. Avoid raw `chromadb`—use tolerant wrappers.
- Chunking defaults (sentence‑aware): chunk_size 500, overlap 150, soft overflow 80, merge if tiny neighbor ≤300. Section/TOC toggles in `AppSettings.section_context`. Ingest script flags: `--section-inject|--no-section-inject`, `--cross-page-merge|--no-cross-page-merge`.

## Dev workflows (cmd.exe)
- VS Code tasks: Install deps “Install deps (pip)”; run tests “Run tests (pytest)”.
- Ingest (infra, E5 default): `python scripts\ingest_kb.py` (overrides: `--provider/--model/--device/--no-normalize`; chunking flags like `--chunk-overlap`).
- Query (infra): `python scripts\query_kb.py "Welche Gesundheitsfragen sind relevant?" -k 5 --rerank --top-n 20 --top-k 5` (supports `--mmr`, `--no-rerank`, `--category`, `--section`, `--no-dedup`, `--dedup-threshold`).
- Ingest (CLI classic): `python -m bu_kb.cli ingest --source data\pdfs --persist .vector_store\chroma --collection bu_knowledge --chunk-size 500 --chunk-overlap 150`.
- Tests: `pytest -q` (offline friendly; dummy embeddings available).

## Reranker controls
- CLI (`scripts/query_kb.py`): `--rerank|--no-rerank`, `--top-n`, `--top-k`, `--reranker-model BAAI/bge-reranker-v2-m3`.
- Env (`src/core/settings.py`): `RERANKER_ENABLED`, `RERANKER_PROVIDER`, `RERANKER_INITIAL_TOP_N`, `RERANKER_FINAL_TOP_K`, `BGE_RERANKER_MODEL`, `BGE_RERANKER_DEVICE`, `BGE_RERANKER_MAX_LEN`, `BGE_RERANKER_BATCH`.

## Notes
- Windows/transformers: `TRANSFORMERS_NO_TORCHVISION=1` is set by scripts for text‑only flows.
- Model switches: different dims require re‑ingest; namespacing prevents cross‑model bleed when filtering by `embedding_sig`.
- Utilities: `scripts/preview_pdf_metadata.py`, `scripts/preview_chunking.py`, `scripts/eval_embeddings.py`, `scripts/prefetch_models.py`.
