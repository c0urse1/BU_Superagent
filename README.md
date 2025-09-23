# BU_Superagent

## Dependency alignment

This repo pins a compatible set of core libraries to avoid version drift:

- langchain 0.3.27
- langchain-community 0.3.29
- langchain-text-splitters 0.3.11
- langchain-huggingface 0.3.1
- langchain-chroma 0.2.6
- chromadb 1.1.0
- sentence-transformers 3.0.1
- PyMuPDF 1.26.4
- pydantic 2.11.9, pydantic-settings 2.10.1

Optional extras:

- nlp extra: syntok (for better sentence segmentation if available)
- openai extra: langchain-openai 0.3.33, openai 1.x (opt-in provider)

Install (editable) with dev tools and optional extras:

```cmd
REM cmd.exe
python -m pip install -e .[dev,nlp,openai]
```

```powershell
# PowerShell
python -m pip install -e .[dev,nlp,openai]
```

If you previously installed mismatched versions, re-run the command to reconcile.

PowerShell chaining tip: Prefer separate lines or use `$LASTEXITCODE` checks; avoid `&&`/`||` in older PS versions. Example (create branch or switch if it exists):

```powershell
git checkout -b feat/dedup-ingest-and-query; if ($LASTEXITCODE -ne 0) { git checkout feat/dedup-ingest-and-query }
```

## Embeddings configuration

Embeddings are configurable via settings (see `src/core/settings.py`) and environment variables.

- Default provider (infra/core and CLI): HuggingFace with `intfloat/multilingual-e5-large-instruct`.
	- Behavior: E5 instruction + prefixes are applied automatically ("Instruct: ...", `Query: `, `Passage: `) and vectors are L2-normalized. Output dim is 1024.
	- Storage: Option B — a single persist directory with signature-suffixed collection names. Default persist dir is `vector_store` for infra scripts and `.vector_store/chroma` for the classic CLI, but both use the same collection namespacing to prevent cross-model bleed.
- Optional providers: OpenAI (opt-in), Dummy (for tests)

See `.env.example` for a ready-to-copy configuration file. Examples (flat keys are preferred):

```
# Default HF (E5 multilingual instruct)
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-large-instruct
EMBEDDING_DEVICE=auto           # or "cuda", "cuda:0", "cpu", "mps"
NORMALIZE_EMBEDDINGS=true       # defaults to true unless explicitly set to a falsy value
EMBEDDINGS__BATCH_SIZE=64       # tune for throughput vs. memory
# E5 prefixing (defaults shown)
E5_ENABLE_PREFIX=true
E5_QUERY_INSTRUCTION=Instruct: Given a web search query, retrieve relevant passages that answer the query
E5_QUERY_PREFIX=Query: 
E5_PASSAGE_PREFIX=Passage: 

# OpenAI (opt-in)
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL_NAME=text-embedding-3-small
OPENAI_API_KEY=sk-...
# Optional for Azure/OpenAI-compatible endpoints
# EMBEDDINGS__OPENAI_BASE_URL=https://<your-azure-endpoint>
```

Compliance note: Only use OpenAI embeddings if your data processing and cross-border transfer are allowed and documented; keep PII to a minimum and obtain user consent where required. These are explicit requirements in your security & compliance principles (data minimization, legal basis, transparency, human-in-the-loop).

## Choose a model via .env

Configure the embedding provider/model through environment variables (see `.env.example`).

- Default (multilingual E5):
	- `EMBEDDINGS__PROVIDER=huggingface`
	- `EMBEDDINGS__MODEL_NAME=intfloat/multilingual-e5-large-instruct`
- Alternative (smaller):
	- `EMBEDDINGS__MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2`
- OpenAI (opt-in):
	- `EMBEDDINGS__PROVIDER=openai`
	- `EMBEDDINGS__MODEL_NAME=text-embedding-3-small`
	- `OPENAI_API_KEY=...`

## Quick evaluation (A/B)

Compare models offline with a tiny DE-centric fixture:

```
python scripts/eval_embeddings.py
```

You’ll get a small scoreboard per model:

```
provider    | model                                                    | R@3=0.92 | ingest=210 ms | q_avg=38 ms
```

- Recall@k: fraction of queries where an expected snippet appears in the top-k results (here k=3). Higher is better.
- ingest: milliseconds to add the (tiny) fixture; a proxy for ingestion performance.
- q_avg: average query latency in milliseconds; lower is better.

## Rollout and migration notes

We switched the defaults to `intfloat/multilingual-e5-large-instruct` everywhere.

- Re-ingest recommended: E5 has 1024-dim vectors, incompatible with older indexes. With Option B we use a single persist directory and isolate via signature-suffixed collection names. Run:
	- `python scripts/ingest_kb.py` (sentence-aware splitter by default)
- Namespacing: Collections are namespaced by embedding signature; each chunk is stamped with `metadata["embedding_sig"]` and queries filter on the same signature.
- Query: `python scripts/query_kb.py --q "Welche Gesundheitsfragen sind relevant?" -k 5` will use the configured E5 embeddings and the namespaced collection under `vector_store`.

If you prefer previous models, override via env or flags and (re)ingest into a separate collection.

## Compliance (OpenAI)

OpenAI embeddings are opt-in and must follow your “Sicherheit, Datenschutz, Compliance (EU/DACH)” principles:

- Data minimization: only send what is needed (no raw PII if avoidable).
- Legal basis and transparency: document purpose and lawful basis; notify users.
- Consent where required: get explicit consent for processing and cross-border transfer.
- Human-in-the-loop: include review/override for critical decisions.

Set `EMBEDDINGS__PROVIDER=openai` and `OPENAI_API_KEY`. For Azure/OpenAI-compatible endpoints, set `EMBEDDINGS__OPENAI_BASE_URL`.

## Sentence-aware chunking (default)

Ingestion now uses a sentence-aware splitter by default for development. The goal is to produce semantically coherent chunks and reduce “empty” mini-chunks that can hurt retrieval quality.

- Pre-segmentation: Text is split into sentences first. If the optional `syntok` package is available, it’s used; otherwise a robust regex heuristic is applied.
- Greedy packing: Sentences are packed into chunks up to `chunk_size` characters without cutting inside a sentence. A small soft overflow (`chunk_max_overflow`) is allowed to avoid leaving a tiny remainder.
- Merge tiny neighbors: After packing, adjacent small chunks from the same source and page are merged if they’re individually tiny or if their combined size is below a threshold (`chunk_min_merge_char_len`). This reduces low-signal fragments.

Defaults (development):

- mode: `sentence_aware`
- chunk_size: `1000`
- chunk_overlap: `150`
- chunk_max_overflow: `200`
- chunk_min_merge_char_len: `500`

Configuration:

- Infra/Core path: Adjust `AppSettings().chunking` in `src/core/settings.py` (programmatic defaults). Example:

	```python
	from src.core.settings import AppSettings

	cfg = AppSettings()
	cfg.chunking.mode = "sentence_aware"  # or "recursive"
	cfg.chunking.chunk_size = 1000
	cfg.chunking.chunk_overlap = 150
	```

- CLI path: The `bu_kb` CLI uses sentence-aware by default. You can control the size/overlap via flags during ingestion:

	```powershell
	python -m bu_kb.cli ingest --source data\pdfs --persist .vector_store\chroma --collection bu_knowledge --chunk-size 1000 --chunk-overlap 150
	```

Optional preview: Visualize chunk boundaries with the included helper script:

```powershell
type data\pdfs\Allianz_test.pdf | python scripts\preview_chunking.py --mode sentence_aware --chunk-size 1000 --chunk-overlap 150
```

### Device override and batch size (one-off runs)

```powershell
# Windows PowerShell example (adjust model as needed)
$env:EMBEDDINGS__DEVICE = "cuda"; $env:EMBEDDINGS__BATCH_SIZE = "128"; python -m bu_kb.cli ingest --source data\pdfs --persist .vector_store\chroma --collection bu_knowledge

# cmd.exe example
set EMBEDDINGS__DEVICE=cuda && set EMBEDDINGS__BATCH_SIZE=128 && python -m bu_kb.cli ingest --source data\pdfs --persist .vector_store\chroma --collection bu_knowledge
```

### Minimal device usage cheatsheet

```bash
# CPU (default)
python scripts/ingest_kb.py

# Force CUDA (if available)
EMBEDDING_DEVICE=cuda python scripts/ingest_kb.py

# Apple Silicon MPS
EMBEDDINGS__DEVICE=mps python scripts/ingest_kb.py
```

Windows equivalents:

```powershell
# PowerShell
$env:EMBEDDING_DEVICE = "cuda"; python scripts\ingest_kb.py
$env:EMBEDDING_DEVICE = "mps"; python scripts\ingest_kb.py
```

```cmd
# cmd.exe
set EMBEDDING_DEVICE=cuda && python scripts\ingest_kb.py
set EMBEDDING_DEVICE=mps && python scripts\ingest_kb.py
```

## Quick usage cheatsheet

```powershell
# Ingest with multilingual HF model (config via .env); sentence-aware chunking by default
python -m bu_kb.cli ingest --source data\pdfs --persist .vector_store\chroma --collection bu_knowledge

# Query via infra/core path (auto-uses E5 persist dir when configured)
python scripts\query_kb.py "Welche Gesundheitsfragen sind bei der BU relevant?" -k 5

# Evaluate models head-to-head
python scripts\eval_embeddings.py
```

### Quick usage cheatsheet (Windows cmd)

1) Configure (optional)

Sentence-aware mode is the default. If you’ve wired environment-backed `AppSettings` for the infra/core path, you can set chunking via env vars (cmd syntax shown):

```cmd
set CHUNKING__MODE=sentence_aware
set CHUNKING__CHUNK_SIZE=1000
set CHUNKING__CHUNK_OVERLAP=150
set CHUNKING__CHUNK_MAX_OVERFLOW=200
set CHUNKING__CHUNK_MIN_MERGE_CHAR_LEN=500
```

2) Ingest your PDFs (uses the sentence-aware splitter by default)

```cmd
python -m bu_kb.cli ingest --source data\pdfs --persist .vector_store\chroma --collection bu_knowledge
```

3) Sanity-check a text sample (preview chunking)

```cmd
echo Satz A. Sehr langer Satz B ... Satz C! | python scripts\preview_chunking.py --chunk-size 80 --chunk-overlap 20
```

4) Run tests

```cmd
pytest -q
```
