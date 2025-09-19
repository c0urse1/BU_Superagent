from src.infra.pdf.pdf_metadata import _build_page_to_section


def test_build_page_to_section_simple() -> None:
    # TOC: (level, title, page_1based)
    toc = [
        (1, "Einleitung", 1),
        (1, "Gesundheitsprüfung", 3),
        (2, "Vorerkrankungen", 4),
    ]
    m = _build_page_to_section(toc, page_count=6)
    # p1-2 -> Einleitung
    assert m[0] == "Einleitung" and m[1] == "Einleitung"
    # p3 -> Gesundheitsprüfung
    assert m[2] == "Gesundheitsprüfung"
    # p4-6 -> Vorerkrankungen (last section carries on)
    assert m[3] == "Vorerkrankungen" and m[4] == "Vorerkrankungen" and m[5] == "Vorerkrankungen"
