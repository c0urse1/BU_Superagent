# Copilot instructions for BU_Superagent

Purpose: Make agents productive on PDF → Chroma ingestion and read‑only retrieval. Two paths exist: a simple CLI (`src/bu_kb/cli.py`) and a configurable infra/core layer—compose helpers, don’t re‑wire basics.

## Architecture at a glance
- CLI ingest: `PdfLoader → TextSplitterAdapter (sentence‑aware via infra splitter) → bu_kb.ingest.store.ChromaStore`; embeddings from infra `build_embeddings(AppSettings().embeddings)`. Each chunk gets `metadata["embedding_sig"]`; ingest de‑dup = hash + optional cosine (`AppSettings().dedup_ingest`).
- CLI query: `src/bu_kb/query.py::QueryService` supports MMR (`--mmr`, `--fetch-k`), score thresholding, optional min‑max score normalization, JSON/outfile. Retrieval‑time de‑dup (`cosine|exact`) via `AppSettings().dedup_query`. Vectorstore loads read‑only via `src/bu_kb/vectorstore.py` using tolerant `_import_chroma`.
- CLI config `src/bu_kb/config.py`: env‑backed (prefix `KB_`), defaults: `data/pdfs`, `.vector_store/chroma`, `bu_knowledge`, model `sentence-transformers/all-MiniLM-L6-v2`, `chunk_size=1000`, `chunk_overlap=150`.
- Infra/Core: settings in `src/core/settings.py` (providers `huggingface|openai|dummy`, device auto/CPU/CUDA/MPS); embeddings factory `src/infra/embeddings/factory.py` (device via `resolve_device`); vector store + namespacing helpers `src/infra/vectorstores/chroma_store.py::{collection_name_for, filter_for_signature}`; PDF metadata `src/infra/loaders/pdf_smart_loader.py` (adds `source/page/title/section/category`).
- Embeddings (default): E5 instruct `intfloat/multilingual-e5-large-instruct` with instruction+prefixing and L2 normalization. Uses dedicated persist dir `vector_store/e5_large` (1024‑dim) to avoid clashes with older indexes.
- API path: `src/api/routes/qa.py` composes infra retrieval (`retrieve → assemble_context`) with `src/services/llm/chain.py::answer_with_citations`; LLM provider lives under `src/services/llm/provider.py` (simple `.invoke()` contract).
 - Reranker (two‑stage retrieval): optional local cross‑encoder rerank after vector recall. Implementation in `src/infra/rerankers/bge.py` (model: `BAAI/bge-reranker-v2-m3`), factory in `src/infra/rerankers/factory.py`. Wired into `scripts/query_kb.py` (CLI) and `src/api/routes/qa.py` (API) to rerank Top‑N to final Top‑K.
 - Version‑tolerance: prefer provided factories/wrappers (`langchain_*`, `chromadb`) and the tolerant loader `src/bu_kb/vectorstore.py::load_vectorstore`—avoid raw `chromadb` instantiation.

## Dev workflows (cmd.exe examples)
- Install deps (VS Code task: “Install deps (pip)”): python -m pip install -e .[dev,nlp,openai]
- Ingest (infra/core, E5 default): python scripts\ingest_kb.py
- Ingest (CLI classic): python -m bu_kb.cli ingest --source data\pdfs --persist .vector_store\chroma --collection bu_knowledge [--model ... --device ... --batch-size ... --chunk-size 1000 --chunk-overlap 150]
- Query (CLI): python -m bu_kb.cli query "Welche Gesundheitsfragen sind relevant?" -k 5 --mmr --json --outfile results.json --normalize-scores
- Query (infra/core) with namespacing + optional rerank:
  - python scripts\query_kb.py "frage" -k 5 [--category X --section Y --provider huggingface --model ... --device ... --no-normalize --enforce-citations --rerank --top-n 10 --top-k 5 --reranker-model BAAI/bge-reranker-v2-m3]
  - Use: `collection_name_for(app.kb.collection_base, emb_cfg.signature)` and a filter like `{"embedding_sig": emb_cfg.signature, "category": "...", "section": "..."}` to avoid cross‑model bleed.
  - E5 note: When `model_name` contains `intfloat/multilingual-e5`, `scripts/query_kb.py` automatically targets `vector_store/e5_large` and prefixes queries/passages appropriately.
- Tests (VS Code task: “Run tests (pytest)”; offline; dummy embeddings): pytest -q
  - Keep tests green; they encode behavior for dedup, namespacing, PDF metadata, sentence chunking, and API citations.
  - Reranker smoke test: `tests/test_bge_reranker.py` exercises the BGE reranker on CPU (skips when HF hub is offline).

## Conventions and project‑specific rules
- Always propagate metadata. Loader adds `source` (abs path), `page` (1‑based), `title`, `section` (TOC), and `category` (folder). Ingest stamps `embedding_sig`; de‑dup sets `metadata.content_hash` and `is_duplicate`.
- Namespacing for multi‑model corpora: always scope by embedding signature
  - Collection naming: `collection_name_for(app.kb.collection_base, emb_cfg.signature)` (capped to 63 chars)
  - Query filter: `{"embedding_sig": emb_cfg.signature}` (optionally add `category`, `section`)
- Read‑only query flows: never call `persist()` or `add_documents()`; use `src/bu_kb/vectorstore.py::load_vectorstore` (tolerant `_import_chroma`).
- Chunking defaults (sentence‑aware): size 1000, overlap 150, soft overflow, merge tiny neighbors. Section context flags in `AppSettings.section_context` control TOC/title injection and cross‑page merge.
- Retrieval‑time de‑dup: `AppSettings().dedup_query` supports `method=cosine|exact`, `similarity_threshold` (defaults 0.95). Script `scripts/query_kb.py` shows a simple cosine implementation with fallback to exact when embeddings unavailable.
- Chroma filter shape: multiple fields in one dict become `$and` via `bu_kb.ingest.store.ChromaStore._normalize_filter`.
 - Debug/Utilities: `scripts/preview_pdf_metadata.py` (inspect TOC/title/page/category), `scripts/preview_chunking.py` (visualize chunk boundaries), `scripts/eval_embeddings.py` (A/B embeddings). See README for pinned versions and model A/B evaluation.

### Reranker controls and settings
- CLI flags (scripts/query_kb.py):
  - `--rerank` to explicitly enable, `--no-rerank` to disable (overrides settings)
  - `--top-n` initial vector results to consider (default from settings)
  - `--top-k` final results to keep after rerank (default k)
  - `--reranker-model` to override model name (default `BAAI/bge-reranker-v2-m3`)
- Settings (env‑backed via `src/core/settings.py`): `reranker.enabled`, `reranker.provider`, `reranker.initial_top_n`, `reranker.final_top_k`, `reranker.bge_model_name`, `reranker.bge_device`, `reranker.bge_max_length`, `reranker.bge_batch_size`. Environment variables typically map as `RERANKER_*` (e.g., `RERANKER_ENABLED=true`).
- API: `src/api/routes/qa.py` optionally applies reranking before `assemble_context()` when enabled in settings.

### Windows/transformers note
- For text‑only workflows, avoid torchvision import issues by setting `TRANSFORMERS_NO_TORCHVISION=1` early. The ingest/query scripts set this automatically; you can also export it in your shell if needed.

Migration note (E5 switch): Re‑ingest into `vector_store/e5_large` due to 1024‑dim vectors. Collections remain namespaced by `embedding_sig` to prevent cross‑model bleed. Queries should filter with `{"embedding_sig": emb_cfg.signature}`.

Tip (PowerShell): avoid `&&/||` on older PS; use `$LASTEXITCODE` as shown in README. For device/batch overrides, prefer environment or `AppSettings().embeddings`.
