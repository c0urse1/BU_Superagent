from collections.abc import Iterable, Iterator

from src.services.llm.chain import answer_with_citations


class MockLLM:
    def __init__(self, replies: Iterable[str]):
        self.replies: Iterator[str] = iter(replies)

    def invoke(self, *, system: str, user: str) -> object:
        class R:
            def __init__(self, t: str):
                self.text: str = t

        return R(next(self.replies))


def test_autoretry_fixes_missing_citation() -> None:
    llm = MockLLM(
        [
            "Here is an answer without citation.",
            'Corrected. (BU Manual, S.3, "Basics")',
        ]
    )
    out = answer_with_citations(llm, "Q", "CTX")
    assert '(BU Manual, S.3, "Basics")' in out
