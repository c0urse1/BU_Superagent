from src.infra.pdf.category import infer_category_from_path


def test_infer_category_from_path() -> None:
    assert infer_category_from_path("/x/Rechtliches/BU.pdf") == "Rechtliches"
    assert infer_category_from_path("/x/Underwriting/Handbuch.pdf") == "Underwriting"
    assert infer_category_from_path("/x/Unknown/Doc.pdf") is None
