# BU_Superagent

## Embeddings configuration

Embeddings are configurable via settings (see `src/core/settings.py`) and environment variables.

- Default provider: HuggingFace, model `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (good for DE-heavy corpora)
- Easy override to `sentence-transformers/all-MiniLM-L6-v2` for a smaller model
- Optional providers: OpenAI (opt-in), Dummy (for tests)

See `.env.example` for a ready-to-copy configuration file. Examples:

```
# Default HF (multilingual)
EMBEDDINGS__PROVIDER=huggingface
EMBEDDINGS__MODEL_NAME=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
EMBEDDINGS__DEVICE=auto           # or "cuda", "cuda:0", "cpu", "mps"
EMBEDDINGS__BATCH_SIZE=64         # tune for throughput vs. memory

# OpenAI (opt-in)
EMBEDDINGS__PROVIDER=openai
EMBEDDINGS__MODEL_NAME=text-embedding-3-small
OPENAI_API_KEY=sk-...
# Optional for Azure/OpenAI-compatible endpoints
# EMBEDDINGS__OPENAI_BASE_URL=https://<your-azure-endpoint>
```

Compliance note: Only use OpenAI embeddings if your data processing and cross-border transfer are allowed and documented; keep PII to a minimum and obtain user consent where required. These are explicit requirements in your security & compliance principles (data minimization, legal basis, transparency, human-in-the-loop).

## Choose a model via .env

Configure the embedding provider/model through environment variables (see `.env.example`).

- Default (multilingual):
	- `EMBEDDINGS__PROVIDER=huggingface`
	- `EMBEDDINGS__MODEL_NAME=sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
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

## Rollout plan

1. Keep MiniLM as the default initially.
2. Run the A/B evaluation locally against the multilingual MPNet model.
3. If Recall@k improves on your DE corpus (and latency remains acceptable), flip the default model.
4. Re-ingest the corpus into a model-scoped collection (we namespace per embedding signature):
	 - Collection name = `collection_base` + `__` + slugged `embedding.signature`
	 - Chunks are stamped with `embedding_sig` metadata; queries filter on the same signature.

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

## Quick usage cheatsheet

```powershell
# Ingest with multilingual HF model (config via .env); sentence-aware chunking by default
python -m bu_kb.cli ingest --source data\pdfs --persist .vector_store\chroma --collection bu_knowledge

# Query via infra/core path
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
