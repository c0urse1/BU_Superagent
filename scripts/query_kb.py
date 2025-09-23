from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path


def main() -> None:
    # Ensure repository root is importable when running this script directly.
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # Avoid optional torchvision import path in transformers for text-only flows
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

    # Import after path bootstrap (fixes E402)
    from src.core.settings import AppSettings, Settings
    from src.infra.embeddings.factory import build_embeddings
    from src.infra.rerankers.bge import RerankItem
    from src.infra.rerankers.factory import get_reranker
    from src.infra.retrieval.assemble import assemble_context
    from src.infra.retrieval.retriever import normalize_metadata, retrieve
    from src.infra.vectorstores.chroma_store import (
        ChromaStore,
        collection_name_for,
    )
    from src.services.llm.chain import answer_with_citations

    parser = argparse.ArgumentParser(description="Query KB with metadata filters.")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--q", dest="q", help="Query text (alternative to positional)")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--category")
    parser.add_argument("--section")
    parser.add_argument("--no-dedup", action="store_true", help="Disable retrieval-time dedup")
    parser.add_argument(
        "--dedup-threshold", type=float, help="Cosine similarity threshold for dedup (0..1)"
    )
    # Embedding overrides to align with a specific collection/signature
    parser.add_argument(
        "--provider",
        choices=["huggingface", "openai", "dummy"],
        help="Embeddings provider override (default from AppSettings)",
    )
    parser.add_argument("--model", help="Embedding model name override")
    parser.add_argument(
        "--device",
        help='Device override, e.g. "cpu", "cuda", "cuda:0", "mps" (default from AppSettings)',
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable embedding vector normalization (defaults to enabled)",
    )
    parser.add_argument(
        "--enforce-citations",
        action="store_true",
        help="Validate/auto-retry to ensure at least one citation in the answer",
    )
    # Reranker flags
    parser.add_argument(
        "--rerank", action="store_true", help="Enable cross-encoder reranking (BGE)"
    )
    parser.add_argument(
        "--no-rerank", action="store_true", help="Disable reranking (override settings)"
    )
    parser.add_argument(
        "--reranker-model", help="Override reranker model name (e.g., BAAI/bge-reranker-v2-m3)"
    )
    parser.add_argument("--top-n", type=int, help="Initial vector Top-N to consider for reranker")
    parser.add_argument("--top-k", type=int, help="Final Top-K to keep after reranking")
    args = parser.parse_args()

    cfg = AppSettings()  # defaults; wire pydantic-settings if you want .env loading
    # Apply optional overrides to embeddings config
    emb_cfg = cfg.embeddings.model_copy()
    if args.provider:
        emb_cfg.provider = args.provider
    if args.model:
        emb_cfg.model_name = args.model
    if args.device:
        emb_cfg.device = args.device
    if args.no_normalize:
        emb_cfg.normalize_embeddings = False

    # Apply dedup flags
    if args.no_dedup:
        cfg.dedup_query.enabled = False
    if args.dedup_threshold:
        cfg.dedup_query.similarity_threshold = float(args.dedup_threshold)

    # Quick dev toggles for reranker settings
    # These override the in-memory Settings() instance used below
    try:
        s = Settings().reranker
        if args.no_rerank:
            s.enabled = False
        if args.top_n is not None:
            s.initial_top_n = int(args.top_n)
        if args.top_k is not None:
            s.final_top_k = int(args.top_k)
    except Exception:
        # Fail open if Settings is not available for some reason
        pass

    # Resolve query text
    query_text = args.q if getattr(args, "q", None) else args.query
    if not query_text:
        parser.error("missing query text (positional or --q)")

    emb = build_embeddings(emb_cfg)
    collection = collection_name_for(cfg.kb.collection_base, emb_cfg.signature)

    # Use a dedicated index path for E5 embeddings (1024-dim)
    persist_dir = (
        Path("vector_store/e5_large")
        if "intfloat/multilingual-e5" in (emb_cfg.model_name or "").lower()
        else Path(cfg.kb.persist_directory)
    )

    store = ChromaStore(
        collection,
        persist_dir,
        emb,
    )

    # Build filter dict
    md_filter: dict[str, object] = {"embedding_sig": emb_cfg.signature}
    if args.category:
        md_filter["category"] = args.category
    if args.section:
        md_filter["section"] = args.section

    # Helper: vector Top-N + optional BGE rerank to Top-K
    def retrieve_with_bge_rerank() -> list[RerankItem]:
        rcfg = Settings().reranker
        initial_n = int(getattr(rcfg, "initial_top_n", 10))
        final_k = int(getattr(rcfg, "final_top_k", args.k))

        # initial recall-oriented retrieval with vector scores
        docs_scores: list[tuple] = store.query_with_scores(
            query_text, k=initial_n, filter=md_filter
        )
        items: list[RerankItem] = []
        for d, sc in docs_scores:
            meta = dict(getattr(d, "metadata", {}) or {})
            meta["vector_score"] = float(sc)
            items.append(RerankItem(text=(getattr(d, "page_content", "") or ""), metadata=meta))

        # decide rerank enablement
        rerank_enabled: bool
        if args.rerank:
            rerank_enabled = True
        elif args.no_rerank:
            rerank_enabled = False
        else:
            rerank_enabled = bool(getattr(rcfg, "enabled", False))

        if not items:
            return []
        if not rerank_enabled:
            items.sort(key=lambda x: x.metadata.get("vector_score", 0.0), reverse=True)
            return items[:final_k]

        # Build reranker (allow model override via flag)
        reranker = None
        if args.reranker_model:
            from src.infra.rerankers.bge import (
                BGEReranker,  # local import to avoid heavy import when unused
            )

            reranker = BGEReranker(
                model_name=args.reranker_model,
                device=str(getattr(rcfg, "bge_device", "auto")),
                max_length=int(getattr(rcfg, "bge_max_length", 512)),
                batch_size=int(getattr(rcfg, "bge_batch_size", 16)),
            )
        else:
            reranker = get_reranker()

        if reranker is None:
            items.sort(key=lambda x: x.metadata.get("vector_score", 0.0), reverse=True)
            return items[:final_k]

        return reranker.rerank(query=query_text, items=items, top_k=final_k)

    # Default path (no LLM) → use reranker if enabled
    reranked_items = retrieve_with_bge_rerank()
    # Fallback to plain vector retrieval when reranker returns nothing
    if not reranked_items:
        docs = store.query(query_text, k=args.k, filter=md_filter)
    else:
        docs = []  # we will print from reranked_items

    # Optional retrieval-time dedup (cosine default) only for plain vector results
    if getattr(cfg.dedup_query, "enabled", True) and len(docs) > 1:
        method = str(getattr(cfg.dedup_query, "method", "cosine")).lower()
        thr = float(getattr(cfg.dedup_query, "similarity_threshold", 0.95))

        def _cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b, strict=False))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(y * y for y in b)) or 1.0
            return dot / (na * nb)

        kept = []
        kept_vecs: list[list[float]] = []
        seen_norms: set[str] = set()
        for d in docs:
            txt = (d.page_content or "").strip()
            if not txt:
                kept.append(d)
                continue

            if method == "exact":
                norm = txt.lower().replace("\n", " ").strip()
                if norm in seen_norms:
                    continue
                seen_norms.add(norm)
                kept.append(d)
                continue

            vec = None
            try:
                vec = emb.embed_query(txt)
            except Exception:
                vec = None
            if vec is None:
                norm = txt.lower().replace("\n", " ").strip()
                if norm in seen_norms:
                    continue
                seen_norms.add(norm)
                kept.append(d)
                continue

            if any(_cosine(vec, kv) >= thr for kv in kept_vecs):
                continue
            kept.append(d)
            kept_vecs.append(vec)

        docs = kept[: args.k]
    # Developer testing: optionally run through prompting path
    if args.enforce_citations:
        # Very small stub LLM; in your environment, supply a real client with .invoke()
        from typing import Any as _Any

        llm: _Any
        try:
            from src.services.llm.provider import llm as _provider_llm

            llm = _provider_llm
        except Exception:

            class _EchoLLM:
                def invoke(self, *, system: str, user: str) -> object:
                    class R:
                        def __init__(self, t: str):
                            self.text: str = t

                    return R(user)

            llm = _EchoLLM()

        # Prefer reranked items for context if available; else fall back to infra retrieve()
        if reranked_items:
            chunks = [
                {
                    "text": it.text,
                    "metadata": normalize_metadata(it.metadata),
                }
                for it in reranked_items
            ]
        else:
            chunks = retrieve(query_text, k=args.k, embeddings=emb_cfg)
        ctx = assemble_context(chunks, k=min(args.k, len(chunks)))
        ans = answer_with_citations(llm, query_text, ctx)
        print(ans)
        return

    if reranked_items:
        for i, it in enumerate(reranked_items, 1):
            m = normalize_metadata(it.metadata)
            title = m.get("source", "<no-title>")
            section = m.get("section", "")
            page = m.get("page", "?")
            src = it.metadata.get("source", "")
            print(f"[{i}] {title} | {section} (p.{page}) | {src}")
            print("     ", (it.text or "").replace("\n", " ")[:200], "…")
    else:
        for i, d in enumerate(docs, 1):
            m = d.metadata or {}
            title = m.get("title", "<no-title>")
            section = m.get("section", "")
            page = m.get("page", m.get("page_number", "?"))
            src = m.get("source", "")
            print(f"[{i}] {title} | {section} (p.{page}) | {src}")
            print("     ", (d.page_content or "").replace("\n", " ")[:200], "…")


if __name__ == "__main__":
    main()
