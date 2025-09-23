# Copilot instructions for BU_Superagent

Purpose: Help agents work productively on PDF → Chroma ingestion and read‑only retrieval. Two paths: classic CLI (`src/bu_kb/cli.py`) and the infra/core stack (preferred). Compose existing helpers; don’t re‑wire basics.

## Architecture
- Ingest: PDF → sentence‑aware splitter → Chroma. Use infra embeddings via `src/infra/embeddings/factory.py::build_embeddings`. Stamp every chunk with `metadata["embedding_sig"]`. Ingest de‑dup: hash + optional cosine (see `AppSettings().dedup_ingest`).
- Query (infra): `scripts/query_kb.py` uses `src/infra/vectorstores/chroma_store.py::ChromaStore` and namespacing helpers `collection_name_for(...)`/`filter_for_signature(...)` to avoid cross‑model bleed. Vector store loaders use tolerant imports (avoid raw `chromadb`).
- Query (CLI classic): `src/bu_kb/query.py::QueryService` supports MMR, score thresholding, normalization, and JSON/outfile.
- API: `src/api/routes/qa.py` composes `retrieve → assemble_context → answer_with_citations`; LLM provider lives in `src/services/llm/provider.py` (`.invoke()` contract).
- Reranking: optional local cross‑encoder (`BAAI/bge-reranker-v2-m3`) after vector Top‑N → final Top‑K via `src/infra/rerankers/factory.py::get_reranker()`.

## Key conventions
- Default embeddings: E5 instruct `intfloat/multilingual-e5-large-instruct` with instruction+prefixing and L2 normalization. Use persist dir `vector_store/e5_large` (1024‑dim). Filter queries by `{ "embedding_sig": emb_cfg.signature }`.
- Namespacing: Collections via `collection_name_for(app.kb.collection_base, emb_cfg.signature)`. Always include `embedding_sig` (plus optional `category`, `section`) in Chroma filters.
- Metadata: keep `source` (abs path), `page` (1‑based), `title`, `section`, `category`. Ingest sets `content_hash` and may mark `is_duplicate`.
- Read‑only queries: never call `persist()` or `add_documents()`; use `src/bu_kb/vectorstore.py::load_vectorstore` or `ChromaStore` (infra) with an embedding function.
- Chunking defaults (sentence‑aware): chunk_size 1000, overlap 150, soft overflow, merge tiny neighbors. Section context toggles in `AppSettings.section_context`.

## Dev workflows (cmd.exe)
- Install deps (VS Code task “Install deps (pip)”): python -m pip install -e .[dev,nlp,openai]
- Ingest (infra, E5 default): python scripts\ingest_kb.py
- Ingest (CLI): python -m bu_kb.cli ingest --source data\pdfs --persist .vector_store\chroma --collection bu_knowledge --chunk-size 1000 --chunk-overlap 150
- Query (infra): python scripts\query_kb.py "Welche Gesundheitsfragen sind relevant?" -k 5 --rerank --top-n 10 --top-k 5
- Tests (task “Run tests (pytest)”; offline/dummy embeddings): pytest -q

## Reranker controls
- CLI flags (`scripts/query_kb.py`): `--rerank|--no-rerank`, `--top-n`, `--top-k`, `--reranker-model BAAI/bge-reranker-v2-m3`.
- Settings (`src/core/settings.py` env‑backed): `RERANKER_ENABLED`, `RERANKER_PROVIDER`, `RERANKER_INITIAL_TOP_N`, `RERANKER_FINAL_TOP_K`, `BGE_RERANKER_MODEL`, `BGE_RERANKER_DEVICE`, `BGE_RERANKER_MAX_LEN`, `BGE_RERANKER_BATCH`.

## Notes
- Windows/transformers: set `TRANSFORMERS_NO_TORCHVISION=1` for text‑only flows (scripts already do this).
- Migration (E5 switch): re‑ingest into `vector_store/e5_large` and always filter by `embedding_sig` to prevent cross‑model bleed.
- Utilities: `scripts/preview_pdf_metadata.py` (TOC/title/page/category), `scripts/preview_chunking.py` (boundaries), `scripts/eval_embeddings.py` (A/B), `scripts/prefetch_models.py` (cache E5/BGE).
