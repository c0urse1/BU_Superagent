from scripts.eval_embeddings import mrr_at_k, recall_at_k


def test_metrics_basics() -> None:
    gold = [{"doc": "BU Manual", "page": 12, "section": "Basics"}]
    ranked = [
        {"metadata": {"source": "BU_Manual.pdf", "page": 12, "section": "Basics"}},
        {"metadata": {"source": "Other.pdf", "page": 1, "section": "Intro"}},
    ]
    assert recall_at_k(ranked, gold, 1) == 1.0
    assert mrr_at_k(ranked, gold, 5) == 1.0  # hit at rank 1

    ranked2 = [
        {"metadata": {"source": "Other.pdf", "page": 1, "section": "Intro"}},
        {"metadata": {"source": "BU_Manual.pdf", "page": 12, "section": "Basics"}},
    ]
    assert recall_at_k(ranked2, gold, 1) == 0.0
    assert recall_at_k(ranked2, gold, 2) == 1.0
    assert mrr_at_k(ranked2, gold, 5) == 0.5  # hit at rank 2
