from src.infra.embeddings.device import resolve_device


def test_resolve_device_respects_explicit() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:0") == "cuda:0"


def test_auto_device_returns_known_value() -> None:
    # We don't enforce a specific accelerator in CI; ensure it returns one of the known labels
    dev = resolve_device("auto")
    assert dev in {"cpu", "cuda", "mps"}
