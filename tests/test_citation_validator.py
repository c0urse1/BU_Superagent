from src.infra.prompting.parsers import has_min_citation


def test_accepts_valid_format() -> None:
    assert has_min_citation('Claim. (BU Manual, S.12, "Exclusions")')


def test_rejects_missing_page() -> None:
    assert not has_min_citation('Claim. (BU Manual, "Exclusions")')


def test_rejects_wrong_quotes() -> None:
    assert not has_min_citation("Claim. (BU Manual, S.12, Exclusions)")
