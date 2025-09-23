# Copilot instructions for BU_Superagent

Purpose: Make agents productive on PDF → Chroma ingestion and read‑only retrieval. Two paths exist: a simple CLI (`src/bu_kb/cli.py`) and a configurable infra/core layer—compose helpers, don’t re‑wire basics.

## Architecture at a glance
- CLI ingest: `PdfLoader → TextSplitterAdapter (sentence‑aware via infra splitter) → bu_kb.ingest.store.ChromaStore`; embeddings from infra `build_embeddings(AppSettings().embeddings)`. Each chunk gets `metadata["embedding_sig"]`; ingest de‑dup = hash + optional cosine (`AppSettings().dedup_ingest`).
- CLI query: `src/bu_kb/query.py::QueryService` supports MMR (`--mmr`, `--fetch-k`), score thresholding, optional min‑max score normalization, JSON/outfile. Retrieval‑time de‑dup (`cosine|exact`) via `AppSettings().dedup_query`. Vectorstore loads read‑only via `src/bu_kb/vectorstore.py` using tolerant `_import_chroma`.
- CLI config `src/bu_kb/config.py`: env‑backed (prefix `KB_`), defaults: `data/pdfs`, `.vector_store/chroma`, `bu_knowledge`, model `sentence-transformers/all-MiniLM-L6-v2`, `chunk_size=1000`, `chunk_overlap=150`.
- Infra/Core: settings in `src/core/settings.py` (providers `huggingface|openai|dummy`, device auto/CPU/CUDA/MPS); embeddings factory `src/infra/embeddings/factory.py` (device via `resolve_device`); vector store + namespacing helpers `src/infra/vectorstores/chroma_store.py::{collection_name_for, filter_for_signature}`; PDF metadata `src/infra/loaders/pdf_smart_loader.py` (adds `source/page/title/section/category`).
- API path: `src/api/routes/qa.py` composes infra retrieval (`retrieve → assemble_context`) with `src/services/llm/chain.py::answer_with_citations`; LLM provider lives under `src/services/llm/provider.py` (simple `.invoke()` contract).

## Dev workflows (cmd.exe examples)
- Install deps (VS Code task preferred): python -m pip install -e .[dev,nlp,openai]
- Ingest: python -m bu_kb.cli ingest --source data\pdfs --persist .vector_store\chroma --collection bu_knowledge [--model ... --device ... --batch-size ... --chunk-size 1000 --chunk-overlap 150]
- Query (CLI): python -m bu_kb.cli query "Welche Gesundheitsfragen sind relevant?" -k 5 --mmr --json --outfile results.json --normalize-scores
- Query (infra/core) with namespacing: python scripts\query_kb.py "frage" -k 5 [--category X --section Y --provider huggingface --model ... --device ... --no-normalize --enforce-citations]
  - Use: `collection_name_for(app.kb.collection_base, emb_cfg.signature)` and a filter like `{"embedding_sig": emb_cfg.signature, "category": "...", "section": "..."}` to avoid cross‑model bleed.
- Tests (offline; dummy embeddings): pytest -q

## Conventions and project‑specific rules
- Always propagate metadata. Loader adds `source` (abs path), `page` (1‑based), `title`, `section` (TOC), and `category` (folder). Ingest stamps `embedding_sig`; de‑dup sets `metadata.content_hash` and `is_duplicate`.
- Namespacing for multi‑model corpora: always scope by embedding signature
  - Collection naming: `collection_name_for(app.kb.collection_base, emb_cfg.signature)` (capped to 63 chars)
  - Query filter: `{"embedding_sig": emb_cfg.signature}` (optionally add `category`, `section`)
- Read‑only query flows: never call `persist()` or `add_documents()`; use `src/bu_kb/vectorstore.py::load_vectorstore` (tolerant `_import_chroma`).
- Chunking defaults (sentence‑aware): size 1000, overlap 150, soft overflow, merge tiny neighbors. Section context flags in `AppSettings.section_context` control TOC/title injection and cross‑page merge.
- Retrieval‑time de‑dup: `AppSettings().dedup_query` supports `method=cosine|exact`, `similarity_threshold` (defaults 0.95). Script `scripts/query_kb.py` shows a simple cosine implementation with fallback to exact when embeddings unavailable.
- Chroma filter shape: multiple fields in one dict become `$and` via `bu_kb.ingest.store.ChromaStore._normalize_filter`.
- Version‑tolerance: use provided factories/wrappers (`langchain_*`, `chromadb`). See README for pinned versions and model A/B evaluation.

Tip (PowerShell): avoid `&&/||` on older PS; use `$LASTEXITCODE` as shown in README. For device/batch overrides, prefer environment or `AppSettings().embeddings`.
