from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.application.ports.embeddings_port import EmbeddingsPort
from src.application.ports.llm_port import LLMPort
from src.application.ports.vector_store_port import VectorStorePort
from src.domain.context import assemble_context
from src.domain.dedup import DuplicateDetector
from src.domain.document import Document
from src.domain.scoring import mmr as _mmr


@dataclass
class QueryUseCase:
    store: VectorStorePort
    llm: LLMPort | None = None
    dedup: DuplicateDetector | None = None
    embedder: EmbeddingsPort | None = None

    def _build_filter(
        self,
        metadata_filter: Mapping[str, object] | None,
        embedding_sig: str | None,
    ) -> dict[str, object] | None:
        md: dict[str, object] = {}
        if metadata_filter:
            md.update(metadata_filter)
        if embedding_sig:
            md.setdefault("embedding_sig", embedding_sig)
        return md or None

    def retrieve_top_k(
        self,
        query: str,
        *,
        k: int = 5,
        metadata_filter: Mapping[str, object] | None = None,
        embedding_sig: str | None = None,
        score_threshold: float | None = None,
        deduplicate: bool = True,
        use_mmr: bool = False,
        fetch_k: int | None = None,
        lambda_mult: float = 0.7,
    ) -> list[Document]:
        """Retrieve top-k documents with optional score thresholding and dedup.

        - Attempts to use search_with_scores if available to apply threshold.
        - Falls back to plain search when scores are not provided by adapter.
        - Applies post-retrieval deduplication when a DuplicateDetector is injected.
        """
        md_filter = self._build_filter(metadata_filter, embedding_sig)

        docs: list[Document] = []

        # MMR path first when requested
        if use_mmr:
            # Prefer vectorstore-native MMR if available
            mmr_fn = getattr(self.store, "max_marginal_relevance_search", None)
            if callable(mmr_fn):
                docs = mmr_fn(
                    query,
                    k=k,
                    fetch_k=(fetch_k or max(k * 4, 20)),
                    lambda_mult=lambda_mult,
                    metadata_filter=md_filter,
                )
            else:
                # Fallback: retrieve a candidate pool, then apply domain MMR when we can embed
                pool = fetch_k or max(k * 4, 20)
                base_docs = self.store.search(query, k=pool, metadata_filter=md_filter)
                if self.embedder is not None and base_docs:
                    qv = self.embedder.embed_query(query)
                    dvs = self.embedder.embed_documents([d.content for d in base_docs])
                    idxs = _mmr(qv, dvs, top_k=k, lambda_mult=lambda_mult)
                    docs = [base_docs[i] for i in idxs]
                else:
                    docs = base_docs[:k]
        else:
            # Try scores first if the adapter provides them
            sws = getattr(self.store, "search_with_scores", None)
            if callable(sws):
                pairs = sws(query, k=k, metadata_filter=md_filter)
                if score_threshold is not None:
                    pairs = [(d, s) for (d, s) in pairs if s >= float(score_threshold)]
                docs = [d for (d, _s) in pairs]
            else:
                docs = self.store.search(query, k=k, metadata_filter=md_filter)

        if deduplicate and self.dedup is not None and docs:
            kept, _skipped = self.dedup.unique(docs)
            docs = kept
        return docs[:k]

    # Alias to align with CLI naming in examples
    def get_top_k(
        self,
        query: str,
        *,
        k: int = 5,
        metadata_filter: Mapping[str, object] | None = None,
        embedding_sig: str | None = None,
        score_threshold: float | None = None,
        deduplicate: bool = True,
        use_mmr: bool = False,
        fetch_k: int | None = None,
        lambda_mult: float = 0.7,
    ) -> list[Document]:
        return self.retrieve_top_k(
            query,
            k=k,
            metadata_filter=metadata_filter,
            embedding_sig=embedding_sig,
            score_threshold=score_threshold,
            deduplicate=deduplicate,
            use_mmr=use_mmr,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )

    def retrieve_top_k_with_scores(
        self,
        query: str,
        *,
        k: int = 5,
        metadata_filter: Mapping[str, object] | None = None,
        embedding_sig: str | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[Document, float | None]]:
        md_filter = self._build_filter(metadata_filter, embedding_sig)

        sws = getattr(self.store, "search_with_scores", None)
        if callable(sws):
            pairs = sws(query, k=k, metadata_filter=md_filter)
            if score_threshold is not None:
                pairs = [(d, s) for (d, s) in pairs if s >= float(score_threshold)]
            return [(d, float(s)) for (d, s) in pairs]
        # Fallback: scoreless search
        docs = self.store.search(query, k=k, metadata_filter=md_filter)
        return [(d, None) for d in docs]

    # Back-compat thin wrapper
    def retrieve(
        self, query: str, *, k: int = 5, metadata_filter: Mapping[str, object] | None = None
    ) -> list[Document]:
        return self.retrieve_top_k(query, k=k, metadata_filter=metadata_filter)

    def ask(
        self,
        question: str,
        *,
        k: int = 5,
        metadata_filter: Mapping[str, object] | None = None,
        embedding_sig: str | None = None,
        score_threshold: float | None = None,
        deduplicate: bool = True,
    ) -> str:
        docs = self.retrieve_top_k(
            question,
            k=k,
            metadata_filter=metadata_filter,
            embedding_sig=embedding_sig,
            score_threshold=score_threshold,
            deduplicate=deduplicate,
        )
        ctx = assemble_context(docs, k=min(k, len(docs)))
        if self.llm is None:
            return ctx  # minimal fallback: return context when no LLM wired
        return self.llm.ask(question, ctx)

    # Back-compat thin wrapper
    def answer(
        self, question: str, *, k: int = 5, metadata_filter: Mapping[str, object] | None = None
    ) -> str:
        return self.ask(question, k=k, metadata_filter=metadata_filter)

    def execute(
        self,
        question: str,
        *,
        k: int = 5,
        use_llm: bool = False,
        metadata_filter: Mapping[str, object] | None = None,
        embedding_sig: str | None = None,
        score_threshold: float | None = None,
        deduplicate: bool = True,
    ) -> list[Document] | str:
        """Unified entry for CLI/API flows.

        - When use_llm=False: returns top-k documents (CLI flow)
        - When use_llm=True: returns answer string (API flow)
        """
        if not use_llm:
            return self.retrieve_top_k(
                question,
                k=k,
                metadata_filter=metadata_filter,
                embedding_sig=embedding_sig,
                score_threshold=score_threshold,
                deduplicate=deduplicate,
            )
        return self.ask(
            question,
            k=k,
            metadata_filter=metadata_filter,
            embedding_sig=embedding_sig,
            score_threshold=score_threshold,
            deduplicate=deduplicate,
        )
