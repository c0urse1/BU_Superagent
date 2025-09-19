# Copilot instructions for BU_Superagent

Purpose: Ingest PDFs into a Chroma vector store and run read‑only similarity queries. The repo has two complementary paths: a simple bu_kb CLI and a configurable infra/core layer. Prefer composing the provided interfaces over ad‑hoc glue.

## Architecture (what lives where)
- bu_kb CLI (Typer): `src/bu_kb/cli.py`
  - ingest: `PdfLoader → RecursiveSplitter → ChromaStore` with HuggingFace embeddings
  - query: read‑only search over persisted Chroma; options: `--mmr`, `--fetch-k`, `--threshold`, `--json`, `--outfile`, `--normalize-scores`
- Config for CLI: `src/bu_kb/config.py` (Pydantic BaseSettings)
  - Defaults: `source_dir=data/pdfs`, `persist_dir=.vector_store/chroma`, `collection_name=bu_knowledge`, `embed_model=sentence-transformers/all-MiniLM-L6-v2`, `chunk_size=1000`, `chunk_overlap=150`
  - Env prefix `KB_`; accepts `KB_EMBED_MODEL` and `KB_EMBEDDING_MODEL` (alias `settings.embedding_model`)
- Ingestion building blocks (compose via Protocols):
  - Loader: `src/bu_kb/ingest/loaders.py::PdfLoader` (PyMuPDF); ensures `metadata["source"]`
  - Splitter: `src/bu_kb/ingest/splitters.py::RecursiveSplitter` (separators: `\n\n`, `\n`, space, empty)
  - Store: `src/bu_kb/ingest/store.py::ChromaStore` wraps `langchain_chroma.Chroma`; safe `persist()` via underlying client
  - Pipeline: `src/bu_kb/ingest/pipeline.py` with optional `embedding_signature` stamping as `metadata["embedding_sig"]`
- Query path:
  - `src/bu_kb/vectorstore.py::load_vectorstore()` read‑only load; tolerant import of Chroma; wires same embedder
  - `src/bu_kb/query.py::QueryService` prefers `similarity_search_with_relevance_scores` (0..1), falls back to older score APIs or no‑score; supports MMR and an optional threshold
- Embeddings (CLI): `src/bu_kb/ingest/embeddings.py::build_embedder()` (CPU, normalize_embeddings=True). Re‑exported at `src/bu_kb/embeddings.py`.
- Configurable infra/core layer:
  - Settings: `src/core/settings.py::EmbeddingConfig, AppSettings` (programmatic defaults; not env‑backed by default)
  - Embeddings factory: `src/infra/embeddings/factory.py::build_embeddings()` supports providers `huggingface | openai | dummy`
  - Per‑model namespacing: `src/infra/vectorstores/chroma_store.py::collection_name_for(base, signature)` and re‑exports `ChromaStore`

## Developer workflows (Windows cmd)
- Install (dev): create venv → run: pip install -e .[dev]
- Ingest (CLI): python -m bu_kb.cli ingest --source data\pdfs --persist .vector_store\chroma --collection bu_knowledge
- Query (CLI): python -m bu_kb.cli query "your question" --k 5 --mmr --json --outfile results.json
- Query via infra/core script: python scripts\query_kb.py "Welche Gesundheitsfragen sind relevant?" -k 5
  - Uses `AppSettings` + `build_embeddings()` and filters by `embedding_sig`
- Peek stored chunks: python tools\peek_chunks.py  (args: path, collection, limit)
- Tests: pytest -q  — includes smoke ingest (skips if HF model missing), dummy embeddings, and collection namespacing

## Conventions and extension points
- Always set document metadata: at minimum `source`; optionally stamp `embedding_sig` when ingesting model‑scoped corpora
- Build embeddings via provided helpers: CLI path `build_embedder(model)`, infra path `build_embeddings(EmbeddingConfig)`
- Vector store compatibility: use `vectorstore._import_chroma()` and `ChromaStore.persist()` patterns to tolerate LC/Chroma version changes
- Read‑only queries: never mutate/persist in the query path
- Namespacing pattern: collection = `collection_name_for(base, cfg.embeddings.signature)` and filter queries with `{ "embedding_sig": cfg.embeddings.signature }`

## External integrations
- LangChain: langchain_core/langchain_community, langchain_chroma, langchain_huggingface; optional langchain_openai
- Chroma persists under `persist_dir` per collection; ingest first — `load_vectorstore()` raises if missing
- Logging: `src/bu_kb/logging_setup.py` (rich); Windows UTF‑8 handled in `cli.py`
