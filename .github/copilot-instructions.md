# Copilot instructions for BU_Superagent

Purpose: Quickly ingest PDFs into a Chroma vector store and run read‑only similarity queries. The repo exposes two paths: a friendly `bu_kb` CLI and a configurable infra/core layer. Prefer composing the provided helpers over ad‑hoc glue.

## Architecture at a glance
- CLI (Typer) `src/bu_kb/cli.py`
  - Ingest: `PdfLoader → TextSplitterAdapter(sentence‑aware by default) → ChromaStore`, embeddings via infra `build_embeddings(AppSettings().embeddings)`; chunks stamped with `metadata["embedding_sig"]`.
  - Query: read‑only Chroma through `QueryService` with MMR, `--fetch-k`, optional score threshold, per‑result min‑max normalization, JSON/outfile modes.
- CLI config `src/bu_kb/config.py`: BaseSettings (env prefix `KB_`). Defaults: `source_dir=data/pdfs`, `persist_dir=.vector_store/chroma`, `collection_name=bu_knowledge`, `embed_model=sentence-transformers/all-MiniLM-L6-v2`, `chunk_size=1000`, `chunk_overlap=150`.
- Infra/Core
  - Settings `src/core/settings.py`: `EmbeddingConfig` (provider `huggingface|openai|dummy`, device, normalize, batch) and `AppSettings` (KB + chunking mode/params).
  - Embeddings `src/infra/embeddings/factory.py::build_embeddings()`: version‑tolerant imports; supports HF/OpenAI/Dummy; device auto‑resolve.
  - Vector store `src/infra/vectorstores/chroma_store.py`: `collection_name_for(base, signature)` for per‑model namespacing and re‑exports `ChromaStore`.
  - PDF loader `src/infra/loaders/pdf_smart_loader.py`: enriches metadata (`title`, `section` via TOC, `category` from path, `source`, 1‑based `page`).
  - Splitters `src/infra/splitting/factory.py`: `SentenceAwareChunker` (default) or `RecursiveCharacterTextSplitter`.
- Query utilities: `src/bu_kb/vectorstore.py::_import_chroma()` for tolerant imports; `src/bu_kb/query.py::QueryService` prefers `similarity_search_with_relevance_scores`.

## Workflows (Windows cmd examples)
- Install: `python -m pip install -e .[dev,nlp,openai]`
- Ingest (sentence‑aware by default):
  - `python -m bu_kb.cli ingest --source data\pdfs --persist .vector_store\chroma --collection bu_knowledge [--model ... --device ... --batch-size ... --chunk-size 1000 --chunk-overlap 150]`
- Query (CLI):
  - `python -m bu_kb.cli query "Welche Gesundheitsfragen sind relevant?" -k 5 --mmr --json --outfile results.json --normalize-scores`
- Query (infra/core) with metadata filters: `python scripts\query_kb.py "frage" -k 5 [--category X --section Y --provider huggingface --model ... --device ...]`
  - Uses `AppSettings` + `build_embeddings()`; collection is `collection_name_for(cfg.kb.collection_base, emb_cfg.signature)` and queries filter `{"embedding_sig": emb_cfg.signature, ...}`.
- Peek stored chunks: `python tools\peek_chunks.py` (args: path, collection, limit)
- Tests: `pytest -q` (includes dummy embeddings and namespacing checks).

## Conventions and gotchas
- Always set/propagate metadata on chunks: at minimum `source`; when ingesting model‑scoped corpora, stamp `embedding_sig` (pipeline does this when `embedding_signature` is provided).
- Namespacing: Use `collection_name_for(base, cfg.embeddings.signature)` and filter queries with `{"embedding_sig": cfg.embeddings.signature}`. See `scripts/query_kb.py` for a concrete example.
- Read‑only query path: never call `persist()` or `add_documents()` during query. Use `vectorstore._import_chroma()` for version‑tolerant loads.
- Chunking: default is sentence‑aware with `chunk_size=1000`, `chunk_overlap=150`, plus `chunk_max_overflow` and `chunk_min_merge_char_len` knobs wired in CLI ingest via `AppSettings().chunking`.
- Version compatibility: imports are defensive across `langchain_*` packages and `chromadb`; prefer the provided factories/helpers instead of direct constructors.
- PowerShell shell chaining: older PS may not support `||`/`&&`. Use `$LASTEXITCODE` checks instead. Example (create branch or switch if it exists):
  - `git checkout -b feat/dedup-ingest-and-query; if ($LASTEXITCODE -ne 0) { git checkout feat/dedup-ingest-and-query }`

See `README.md` for pinned dependency versions and quick A/B evaluation of embedding models.
