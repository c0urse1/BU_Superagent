from __future__ import annotations

import argparse
import logging
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
    parser.add_argument("--question", "--query", dest="q", help="Alias for --q")
    parser.add_argument("-k", "--k", type=int, default=5)
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
    parser.add_argument("--embed-model", dest="model", help="Alias for --model")
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
    # Diversified retrieval
    parser.add_argument(
        "--mmr",
        action="store_true",
        help="Use Maximal Marginal Relevance (MMR) for diversified retrieval (fetch_k≈20, λ=0.7)",
    )
    args = parser.parse_args()

    # Simple console logging (INFO); keep message-only format like ingest
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = AppSettings()  # defaults
    # Start from app defaults, then overlay env-backed Settings(), then CLI overrides
    emb_cfg = cfg.embeddings.model_copy()
    try:
        s_emb = Settings().embeddings
        if getattr(s_emb, "provider", None):
            emb_cfg.provider = str(s_emb.provider)
        if getattr(s_emb, "model_name", None):
            emb_cfg.model_name = str(s_emb.model_name)
        if getattr(s_emb, "device", None):
            emb_cfg.device = str(s_emb.device)
        if getattr(s_emb, "normalize", None) is not None:
            emb_cfg.normalize_embeddings = bool(s_emb.normalize)
    except Exception:
        pass
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
    # Log resolved embedding configuration at startup
    logging.getLogger(__name__).info(
        "embeddings_config: model=%s e5_prefix=%s normalized=%s",
        getattr(
            emb, "model_name", getattr(getattr(emb, "client", None), "model_name", "<unknown>")
        ),
        getattr(emb, "_use_e5_prefix", False),
        True,
    )
    collection = collection_name_for(cfg.kb.collection_base, emb_cfg.signature)
    # Option B: use a single persist dir and separate models by collection name
    persist_dir = Path("vector_store")

    store = ChromaStore(
        collection,
        persist_dir,
        emb,
    )

    # Log query configuration for verification (mirrors ingest log)
    logging.getLogger(__name__).info(
        "[query] embeddings=%s provider=%s normalize=%s sig=%s persist_dir=%s collection=%s k=%d mmr=%s",
        emb_cfg.model_name,
        emb_cfg.provider,
        getattr(emb_cfg, "normalize_embeddings", True),
        emb_cfg.signature,
        str(persist_dir),
        collection,
        int(args.k),
        bool(getattr(args, "mmr", False)),
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
        try:
            logging.getLogger(__name__).info("retrieval.stage1.vector_hits=%d", len(docs_scores))
        except Exception:
            pass
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
            # vector-only default raised to 10 when reranker is disabled
            return items[: max(final_k, 10)]

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

    # Retrieval paths
    reranked_items: list[RerankItem] = []
    from langchain_core.documents import Document as _LCDocument  # local import for typing

    docs: list[_LCDocument] = []
    used_mmr = False
    if getattr(args, "mmr", False):
        # Diversified retrieval via MMR; align fetch_k with reranker pool (default 20)
        fetch_k = int(getattr(Settings().reranker, "initial_top_n", 20))
        docs = store.max_marginal_relevance_search(
            query_text, k=int(args.k), fetch_k=fetch_k, lambda_mult=0.7, filter=md_filter
        )
        try:
            logging.getLogger(__name__).info("retrieval.stage1.vector_hits=%d", len(docs))
        except Exception:
            pass
        used_mmr = True
    else:
        # Default path (no LLM) → use reranker if enabled
        reranked_items = retrieve_with_bge_rerank()
        # Fallback to plain vector retrieval when reranker returns nothing
        if not reranked_items:
            # Use vector-only default of 10 when reranker not used
            k_vec_only = (
                10 if bool(getattr(Settings().reranker, "enabled", False)) is False else args.k
            )
            docs = store.query(query_text, k=k_vec_only, filter=md_filter)

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
                # Prefer SBERT-style API with explicit PASSAGE mode if available
                if hasattr(emb, "encode"):
                    enc = emb.encode([txt], mode="passage")
                    # enc can be list[list[float]] or np.ndarray; extract first row
                    if enc is not None:
                        v0 = enc[0]
                        # normalize to list[float]
                        vec = [float(x) for x in (v0.tolist() if hasattr(v0, "tolist") else v0)]
                # Fallback to document/passages embedding API
                if vec is None and hasattr(emb, "embed_documents"):
                    ed = emb.embed_documents([txt])
                    if ed:
                        v0 = ed[0]
                        vec = [float(x) for x in (v0.tolist() if hasattr(v0, "tolist") else v0)]
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
        try:
            logging.getLogger(__name__).info("retrieval.stage1.after_dedup=%d", len(docs))
        except Exception:
            pass
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

        # Prefer reranked items for context if available; else prefer MMR docs; else fall back to infra retrieve()
        if reranked_items:
            chunks = [
                {
                    "text": it.text,
                    "metadata": normalize_metadata(it.metadata),
                }
                for it in reranked_items
            ]
        elif used_mmr and docs:
            chunks = [
                {
                    "text": (d.page_content or ""),
                    "metadata": normalize_metadata(d.metadata or {}),
                }
                for d in docs[: args.k]
            ]
        else:
            chunks = retrieve(query_text, k=args.k, embeddings=emb_cfg)
        ctx = assemble_context(chunks, k=min(args.k, len(chunks)))
        ans = answer_with_citations(llm, query_text, ctx)
        print(ans)
        return

    # Final hits count (post-rerank or direct vector/MMR)
    try:
        logging.getLogger(__name__).info(
            "retrieval.final_hits=%d", len(reranked_items) if reranked_items else len(docs)
        )
    except Exception:
        pass

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
