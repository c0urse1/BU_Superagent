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
