# Copilot instructions for BU_Superagent

This repo ingests PDFs into a Chroma vector store using LangChain + sentence-transformer embeddings. The code is small but intentionally modular; favor composing the provided interfaces over adding new ad‑hoc code.

## Architecture overview
- CLI entrypoint: `src/bu_kb/cli.py` (Typer). Command `ingest` wires the pipeline:
  1) `PdfLoader` → 2) `RecursiveSplitter` → 3) `ChromaStore` with `HuggingFaceEmbeddings`.
  - Command `query` provides a read-only similarity search over the persisted Chroma store.
- Config: `src/bu_kb/config.py` via Pydantic Settings. Defaults:
  - `source_dir=./data/pdfs`, `persist_dir=./.vector_store/chroma`, `collection_name="bu_knowledge"`
  - `embedding_model="sentence-transformers/all-MiniLM-L6-v2"`, `chunk_size=1000`, `chunk_overlap=150`
  - Env prefix `KB_` supported for all fields (e.g., `KB_PERSIST_DIR`, `KB_COLLECTION_NAME`, `KB_EMBED_MODEL`).
- Pipeline: `src/bu_kb/ingest/pipeline.py` coordinates `Loader`, `Splitter`, `VectorStore` protocols (`src/bu_kb/ingest/interfaces.py`).
- Storage: `src/bu_kb/ingest/store.py` wraps `langchain_chroma.Chroma`. Calls `client.persist()` if available.
- Embeddings: `src/bu_kb/ingest/embeddings.py` returns `HuggingFaceEmbeddings` (CPU by default, normalized embeddings).
- Split: `src/bu_kb/ingest/splitters.py` uses `RecursiveCharacterTextSplitter` with explicit separators.
- Logging: `src/bu_kb/logging_setup.py` configures `rich` logging.
- Errors: `src/bu_kb/exceptions.py` defines `ConfigurationError` and `IngestionError`.
- Windows UTF‑8: `sitecustomize.py` enforces UTF‑8 stdio to avoid cp1252 issues.

## Typical workflows (Windows cmd)
- Install (dev):
  - Create venv, then: `pip install -e .[dev]`
- Run tests:
  - `pytest` (see `pyproject.toml` for quiet mode + coverage)
  - The smoke test may skip if the embedding model can't be downloaded (offline)
- Ingest PDFs:
  - `python -m bu_kb.cli ingest --source data\pdfs --persist .vector_store\chroma --collection bu_knowledge`
  - Override defaults via options or `.env` (see `Settings` fields)
- Inspect stored chunks (quick peek tool):
  - `python tools\peek_chunks.py` (optional args: `path`, `collection`, `limit`)
 - Read-only query:
   - `python -m bu_kb.cli query "your question" --k 5` (respects `KB_*` env vars for persist dir, collection, and model)

## Conventions and patterns
- Keep the ingestion pipeline modular using the `interfaces.py` Protocols; add new loaders/splitters/stores by implementing those contracts and wiring them in `cli.py`.
- Use `IngestionPipeline.run(Iterable[Path])` exclusively for end-to-end ingestion; it logs per-file and returns total chunk count; it persists at the end.
- When adding a new store, ensure a safe `.persist()` call; mirror the `ChromaStore.persist()` pattern to handle wrapper differences across versions.
- Embeddings should be created via `build_embedder(model_name)`. Prefer CPU by default; add CUDA detection only if necessary.
- Tests: prefer small smoke tests like `tests/test_smoke_ingest.py` that avoid network when offline; skip gracefully if models can't load.
- Logging: call `setup_logging()` early in CLIs/tools for consistent rich output.

## External dependencies and nuances
- LangChain packages are version-pinned in `pyproject.toml`; update thoughtfully because `persist()` behavior and APIs can differ across minor versions.
- Embedding model (HuggingFace) requires either network access or a pre-cached model. The test already skips when unavailable.
- Chroma persistence lives under `persist_dir`; collection name defaults to `bu_knowledge`.

## Examples
- Adding a TXT loader:
  - Implement `Loader.load(path: str) -> list[Document]` that fills `metadata["source"]`.
  - Wire it in `cli.ingest()` as the `loader` argument.
- Changing chunking:
  - Provide a new `Splitter` (e.g., token-based) and swap in `cli.py` with config‑driven sizes/overlaps.

If anything above is inaccurate or unclear (e.g., preferred GPU settings, alternative stores, or CI commands), please comment and I’ll refine this file.
