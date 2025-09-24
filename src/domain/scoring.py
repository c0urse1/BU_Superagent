from __future__ import annotations

from collections.abc import Iterable, Sequence


def normalize_scores(scores: Iterable[float]) -> list[float]:
    vals = list(scores)
    if not vals:
        return []
    lo = min(vals)
    hi = max(vals)
    if hi <= lo:
        return [0.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def mmr(
    query_vec: Sequence[float],
    doc_vecs: Sequence[Sequence[float]],
    *,
    top_k: int,
    lambda_mult: float = 0.7,
) -> list[int]:
    """Return indices of selected docs using Maximal Marginal Relevance.

    Expects vectors (already embedded) and returns the indices in selection order.
    Uses cosine similarity internally.
    """
    import math

    def _cos(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    n = len(doc_vecs)
    if n == 0 or top_k <= 0:
        return []
    selected: list[int] = []
    candidates = set(range(n))
    while len(selected) < min(top_k, n):
        best_i = -1
        best_score = float("-inf")
        for i in list(candidates):
            rel = _cos(query_vec, doc_vecs[i])
            div = 0.0 if not selected else max(_cos(doc_vecs[i], doc_vecs[j]) for j in selected)
            score = lambda_mult * rel - (1 - lambda_mult) * div
            if score > best_score:
                best_score = score
                best_i = i
        if best_i == -1:
            break
        selected.append(best_i)
        candidates.remove(best_i)
    return selected
