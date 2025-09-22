from __future__ import annotations

try:
    from src.infra.prompting.parsers import has_min_citation
except ModuleNotFoundError:
    # Allow running as a script: add project root to sys.path
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.infra.prompting.parsers import has_min_citation


def eval_pass_rate(answers: list[str]) -> float:
    good = sum(1 for a in answers if has_min_citation(a))
    return good / max(1, len(answers))


if __name__ == "__main__":
    sample = [
        'A. (BU Manual, S.1, "Intro")',
        "B.",
        'C. (BU Manual, S.3, "Basics")',
    ]
    print(f"pass-rate: {eval_pass_rate(sample):.2%}")
