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
