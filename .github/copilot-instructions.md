# Copilot instructions for BU_Superagent

Purpose: Ingest PDFs into a Chroma vector store (LangChain) and run read-only similarity queries. The code is intentionally small and modular—compose the provided interfaces instead of adding ad‑hoc glue.

## Architecture (where things live)
- CLI: `src/bu_kb/cli.py` (Typer)
  - `ingest`: PdfLoader → RecursiveSplitter → ChromaStore (HuggingFaceEmbeddings)
  - `query`: read-only search over persisted Chroma; options: `--mmr`, `--fetch-k`, `--threshold`, `--json`, `--outfile` (JSON or outfile suppresses logs)
- Config: `src/bu_kb/config.py` (Pydantic Settings)
  - Defaults: `source_dir=data/pdfs`, `persist_dir=.vector_store/chroma`, `collection_name=bu_knowledge`, `embed_model=sentence-transformers/all-MiniLM-L6-v2`, `chunk_size=1000`, `chunk_overlap=150`
  - Env prefix `KB_` for all fields. Accepts both `KB_EMBED_MODEL` and `KB_EMBEDDING_MODEL`. `settings.embedding_model` remains as alias.
- Ingestion pipeline: `src/bu_kb/ingest/pipeline.py` orchestrates `Loader | Splitter | VectorStore` Protocols (`ingest/interfaces.py`).
- Loader: `ingest/loaders.py::PdfLoader` via PyMuPDF; ensures `metadata["source"]` is set.
- Splitter: `ingest/splitters.py::RecursiveSplitter` uses `RecursiveCharacterTextSplitter` with separators `["\n\n", "\n", " ", ""]`.
- Store (ingest): `ingest/store.py::ChromaStore` wraps `langchain_chroma.Chroma`; safe `persist()` via underlying client if needed.
- Query path:
  - `src/bu_kb/vectorstore.py::load_vectorstore()` loads existing collection (read-only), imports Chroma from `langchain_chroma` or community fallback, and wires the same embedder.
  - `src/bu_kb/query.py::QueryService` prefers `similarity_search_with_relevance_scores` (0..1), falls back to score variants or scoreless; supports MMR; normalizes metadata (`source`, `page`).
- Embeddings: `ingest/embeddings.py::build_embedder()` (CPU, normalize_embeddings=True). Top-level `src/bu_kb/embeddings.py` re-exports for a stable import.
- Logging/UTF‑8: `src/bu_kb/logging_setup.py` (rich); `sitecustomize.py` enforces UTF‑8 on Windows.

## Developer workflows (Windows cmd)
- Install (dev): create venv → `pip install -e .[dev]`
- Ingest: `python -m bu_kb.cli ingest --source data\pdfs --persist .vector_store\chroma --collection bu_knowledge`
- Query (top‑k): `python -m bu_kb.cli query "your question" --k 5` (respects `KB_*` env: persist dir, collection, model)
  - Examples: `--mmr` for diverse results; `--json` or `--outfile results.json`; optional `--threshold 0.3` when relevance scores are available
- Peek stored chunks: `python tools\peek_chunks.py` (args: `path`, `collection`, `limit`)
- Tests: `pytest` (quiet + coverage). `tests/test_smoke_ingest.py` skips if the HF model isn’t available/cached.

## Conventions and extension points
- Use `IngestionPipeline.run(Iterable[Path])` for end‑to‑end ingest; it logs per file and persists once at the end.
- New components: implement the `interfaces.py` Protocols (e.g., a TXT `Loader` sets `metadata["source"]`) and wire them in `cli.py`.
- Vector store differences across LangChain versions are handled (persist, import path) — follow the `ChromaStore.persist()` and `vectorstore._import_chroma()` patterns.
- Always build embeddings via `build_embedder(model_name)`; GPU is not assumed.
- Config comes from `.env` or `KB_*` env vars; prefer `KB_EMBED_MODEL` (alias to `KB_EMBEDDING_MODEL`).

## Integration notes
- Versions are pinned in `pyproject.toml`; minor bumps can change Chroma/LC behavior (APIs and persist). Test after upgrades.
- Chroma data lives under `persist_dir` per collection. `vectorstore.load_vectorstore()` raises if the directory doesn’t exist (ingest first).
