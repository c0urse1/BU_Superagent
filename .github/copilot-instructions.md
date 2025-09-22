# Copilot instructions for BU_Superagent

Purpose: Help agents ingest PDFs into Chroma and run read‑only similarity queries fast. Two paths exist: a simple `bu_kb` CLI and a configurable infra/core layer—compose the provided helpers, don’t reinvent wiring.

## Architecture (what lives where)
- CLI `src/bu_kb/cli.py`
  - Ingest: `PdfLoader → TextSplitterAdapter (sentence‑aware by default) → bu_kb.ingest.store.ChromaStore`; embeddings via infra `build_embeddings(AppSettings().embeddings)`. Each chunk gets `metadata["embedding_sig"]`; ingest de‑dup does hash + optional cosine per `AppSettings().dedup_ingest`.
  - Query: `src/bu_kb/query.py::QueryService` with MMR (`--mmr`, `--fetch-k`), optional score threshold, min‑max score normalization, JSON/outfile modes. Retrieval‑time de‑dup (`cosine|exact`) via `AppSettings().dedup_query`. Uses `src/bu_kb/vectorstore.py` for a read‑only Chroma load (version‑tolerant `_import_chroma`).
- CLI config `src/bu_kb/config.py`: env‑backed (prefix `KB_`), defaults: `data/pdfs`, `.vector_store/chroma`, `bu_knowledge`, model `sentence-transformers/all-MiniLM-L6-v2`, `chunk_size=1000`, `chunk_overlap=150`.
- Infra/Core
  - Settings `src/core/settings.py`: `EmbeddingConfig` (provider `huggingface|openai|dummy`, device auto/CPU/CUDA/MPS, normalize, batch) and `AppSettings` (KB paths, chunking knobs, ingest/query de‑dup, section context).
  - Embeddings `src/infra/embeddings/factory.py::build_embeddings()`: tolerant imports across `langchain_*`, OpenAI optional, device via `infra/embeddings/device.resolve_device`.
  - Vector store `src/infra/vectorstores/chroma_store.py`: `ChromaStore` wrapper and `collection_name_for(base, signature)` to namespace per embedding signature.
  - PDF loader `src/infra/loaders/pdf_smart_loader.py`: enriches `metadata` (`source`, 1‑based `page`, `title`, `section` via TOC, `category` from path).
  - Splitters `src/infra/splitting/factory.py`: `SentenceAwareChunker` (default; optional `syntok`) or `RecursiveCharacterTextSplitter` with chunking knobs.

## Workflows (Windows cmd examples)
- Install deps: use VS Code task “Install deps (pip)” or run `python -m pip install -e .[dev,nlp,openai]`.
- Ingest (sentence‑aware by default):
  - `python -m bu_kb.cli ingest --source data\pdfs --persist .vector_store\chroma --collection bu_knowledge [--model ... --device ... --batch-size ... --chunk-size 1000 --chunk-overlap 150]`
- Query (CLI):
  - `python -m bu_kb.cli query "Welche Gesundheitsfragen sind relevant?" -k 5 --mmr --json --outfile results.json --normalize-scores`
- Query (infra/core) with namespacing and filters: `python scripts\query_kb.py "frage" -k 5 [--category X --section Y --provider huggingface --model ... --device ...]`.
  - Use `collection_name_for(app.kb.collection_base, emb_cfg.signature)` and filter `{"embedding_sig": emb_cfg.signature}` to avoid cross‑model bleed.
- Utilities: peek chunks `python tools\peek_chunks.py`; eval models `python scripts\eval_embeddings.py`; tests `pytest -q` (uses `dummy` embeddings; checks namespacing/dedup).

## Conventions and gotchas (project‑specific)
- Always propagate metadata: at least `source`; loader adds `page/title/section/category`. Ingest stamps `embedding_sig`; hash de‑dup sets `metadata.content_hash` and `is_duplicate`.
- Namespacing: if you ingest multiple models, prefer infra path and `collection_name_for(...)` + query filter `embedding_sig`. Chroma filenames are capped (function truncates to 63 chars).
- Read‑only query path: do not call `persist()` or `add_documents()` in query flows. Use `vectorstore._import_chroma()` for tolerant loads.
- Chunking defaults: sentence‑aware (`chunk_size=1000`, `chunk_overlap=150`, `chunk_max_overflow`, `chunk_min_merge_char_len`). Section context can inject TOC section into first chunk; see `AppSettings.section_context`.
- Chroma filter shape: multi‑field dicts become `$and` in `bu_kb.ingest.store.ChromaStore` to match newer Chroma expectations.
- Version compatibility: rely on provided factories/wrappers instead of raw constructors (`langchain_*`/`chromadb` have breaking changes).
- PowerShell tip: older PS may not support `&&/||`. Use `$LASTEXITCODE` (see README example) when chaining.

See `README.md` for pinned versions and a quick A/B model evaluation flow.
