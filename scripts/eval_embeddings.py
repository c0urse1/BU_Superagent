from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from src.core.settings import AppSettings, EmbeddingConfig
from src.infra.embeddings.factory import build_embeddings
from src.infra.vectorstores.chroma_store import ChromaStore, collection_name_for

# Tiny DE-centric fixture
FIXTURE: list[Document] = [
    Document(page_content="Definition der Berufsunfähigkeit gemäß VVG §172."),
    Document(
        page_content="Gesundheitsprüfung umfasst Vorerkrankungen und Behandlungen der letzten 5 Jahre."
    ),
    Document(page_content="Ausschlüsse: Hochrisiko-Hobbys wie Klettern im Outdoor-Bereich."),
    Document(page_content="Leistungsprüfung: Nachweise, Stabilität, berufliche Tätigkeit."),
]

QUERIES_EXPECT: dict[str, str] = {
    "Was zählt zur Gesundheitsprüfung?": "Gesundheitsprüfung",
    "Was sind typische Ausschlüsse?": "Ausschlüsse",
    "Was ist Berufsunfähigkeit?": "Berufsunfähigkeit",
}


@dataclass
class RunResult:
    provider: str
    model: str
    recall_at_k: float
    ingest_ms: int
    avg_query_ms: int


def _ensure_store(cfg_app: AppSettings, cfg_emb: EmbeddingConfig) -> ChromaStore:
    emb = build_embeddings(cfg_emb)
    collection = collection_name_for(cfg_app.kb.collection_base, cfg_emb.signature)
    # Use positional args to match the repo's ChromaStore signature
    store = ChromaStore(collection, Path(cfg_app.kb.persist_directory), emb)
    return store


def ingest(store: ChromaStore, signature: str) -> int:
    # tag signature to allow filtered retrieval
    docs: list[Document] = []
    for d in FIXTURE:
        d.metadata = {
            **(d.metadata or {}),
            "embedding_sig": signature,
            "source": "fixture",
            "page": 1,
        }
        docs.append(d)
    t0 = time.perf_counter()
    store.add_documents(docs)
    return int((time.perf_counter() - t0) * 1000)


def eval_model(cfg_app: AppSettings, cfg_emb: EmbeddingConfig, k: int = 3) -> RunResult:
    store = _ensure_store(cfg_app, cfg_emb)
    ingest_ms = ingest(store, cfg_emb.signature)

    # queries
    t_total = 0.0
    hits = 0
    total = 0
    for q, expect_sub in QUERIES_EXPECT.items():
        t0 = time.perf_counter()
        docs = store.query(q, k=k, filter={"embedding_sig": cfg_emb.signature})
        t_total += time.perf_counter() - t0
        total += 1
        joined = " ".join(d.page_content for d in docs)
        if expect_sub.lower() in joined.lower():
            hits += 1

    return RunResult(
        provider=cfg_emb.provider,
        model=cfg_emb.model_name,
        recall_at_k=hits / max(total, 1),
        ingest_ms=ingest_ms,
        avg_query_ms=int((t_total / max(total, 1)) * 1000),
    )


def main() -> None:
    base = AppSettings()

    # A) baseline (MiniLM)
    a = EmbeddingConfig(
        provider="huggingface",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    # B) multilingual (mpnet)
    b = EmbeddingConfig(
        provider="huggingface",
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    # C) optional OpenAI (only runs if key present)
    c = EmbeddingConfig(
        provider="openai",
        model_name="text-embedding-3-small",
        openai_api_key=base.embeddings.openai_api_key,
        openai_base_url=base.embeddings.openai_base_url,
    )

    results: list[RunResult] = []
    for cfg in (a, b):
        results.append(eval_model(base, cfg))
    if c.openai_api_key:
        results.append(eval_model(base, c))

    print("\n=== Embedding A/B Results ===")
    for r in results:
        print(
            f"{r.provider:<10} | {r.model:<55} | R@3={r.recall_at_k:.2f} | "
            f"ingest={r.ingest_ms} ms | q_avg={r.avg_query_ms} ms"
        )


if __name__ == "__main__":
    main()
