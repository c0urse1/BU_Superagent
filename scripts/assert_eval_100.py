from __future__ import annotations

import os
import sys

# Ensure project root on path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.eval_citations import eval_pass_rate


def main() -> None:
    answers = [
        'A. (BU Manual, S.1, "Intro")',
        'B. (BU Manual, S.2, "Basic")',
        'C. (BU Manual, S.3, "Eligibility")',
    ]
    rate = eval_pass_rate(answers)
    assert rate == 1.0, f"expected 1.0, got {rate}"
    print(f"pass-rate: {rate:.2%}")


if __name__ == "__main__":
    main()
