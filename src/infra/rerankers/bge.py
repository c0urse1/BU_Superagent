from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class RerankItem:
    def __init__(self, text: str, metadata: dict[str, Any], score: float = 0.0):
        self.text = text
        self.metadata = metadata
        self.score = score


class BGEReranker:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 16,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.batch_size = batch_size

    @torch.no_grad()
    def _score_batch(self, query: str, texts: Sequence[str]) -> list[float]:
        inputs = self.tokenizer(
            [query] * len(texts),
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits  # shape: [B, 1] or [B]
        # For BGE reranker heads, first logit is relevance
        scores = logits.view(-1).tolist()
        return scores

    def rerank(self, query: str, items: list[RerankItem], top_k: int) -> list[RerankItem]:
        if not items:
            return []
        scores: list[float] = []
        # batched scoring
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            batch_scores = self._score_batch(query, [b.text for b in batch])
            scores.extend(batch_scores)
        # attach & sort
        for it, s in zip(items, scores, strict=False):
            it.score = float(s)
        items.sort(key=lambda x: x.score, reverse=True)
        return items[:top_k]
