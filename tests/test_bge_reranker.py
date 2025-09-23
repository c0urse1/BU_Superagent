import os

import pytest

from src.infra.rerankers.bge import BGEReranker, RerankItem


def _internet_ok() -> bool:
    # Heuristic: HF_HOME or offline test environments may block downloads
    # Also skip on CI without network
    return not os.environ.get("HF_HUB_OFFLINE")


@pytest.mark.skipif(not _internet_ok(), reason="HF hub offline; skipping reranker smoke test")
def test_bge_reranker_smoke() -> None:
    # Keep things light for CI: cpu device, short sequence, tiny batch
    rr = BGEReranker("BAAI/bge-reranker-v2-m3", device="cpu", max_length=128, batch_size=2)
    items = [
        RerankItem(
            "Leistungen gelten bei Auslandsaufenthalt bis 6 Monate.", {"doc": "A", "page": 10}
        ),
        RerankItem("Unrelated text about pets.", {"doc": "B", "page": 2}),
    ]
    out = rr.rerank("Gilt BU-Schutz im Ausland?", items, top_k=1)
    assert len(out) == 1
    assert ("Ausland" in out[0].text) or ("Auslandsaufenthalt" in out[0].text)
