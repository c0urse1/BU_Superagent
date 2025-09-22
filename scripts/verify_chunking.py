import argparse
import json
import sys
from pathlib import Path

# Ensure repository root is importable when running this script directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infra.retrieval.retriever import load_pdf_and_chunk  # noqa: E402
from src.infra.splitting.sentence_chunker import compute_chunk_diagnostics  # noqa: E402

ACCEPTANCE_SENTENCE_BOUNDARY = 90.0  # percent
ALLOWABLE_SECTION_INJECT_PER_SECTION = 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify chunking quality and section handling.")
    ap.add_argument("--pdf", required=True, help="Path to PDF with known edge-cases")
    ap.add_argument("--dump-json", help="Optional path to write raw chunks JSON")
    args = ap.parse_args()

    chunks = load_pdf_and_chunk(args.pdf)  # returns [{"text":..., "metadata":{...}}, ...]

    if args.dump_json:
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

    diag = compute_chunk_diagnostics(chunks)

    # Print human-friendly report
    print(f"Total chunks: {diag.total_chunks}")
    print(
        f"Ends on sentence boundary: {diag.sentence_boundary_ends} ({diag.percent_sentence_boundary:.1f}%)"
    )
    print(f"Title-merged chunks: {diag.title_merged_count}")
    print(f"Orphan minis (0-sentence/too short): {diag.orphan_minis}")

    # Section injection check
    bad_sections = {
        k: v
        for k, v in diag.section_injected_counts.items()
        if v != ALLOWABLE_SECTION_INJECT_PER_SECTION
    }
    if bad_sections:
        print("\n[FAIL] section_injected appears != 1 for some sections:")
        for k, v in bad_sections.items():
            print(f"  {k} -> {v} times")

    # Acceptance gates
    failed = False
    if diag.percent_sentence_boundary < ACCEPTANCE_SENTENCE_BOUNDARY:
        print(
            f"\n[FAIL] Sentence-boundary rate {diag.percent_sentence_boundary:.1f}% < {ACCEPTANCE_SENTENCE_BOUNDARY}%"
        )
        failed = True
    if bad_sections:
        failed = True

    # Heuristic on orphan minis: should be 0 after merge; tolerate a tiny number if needed
    if diag.orphan_minis > 0:
        print(
            f"\n[WARN] Found {diag.orphan_minis} orphan mini-chunk(s). Inspect dump and thresholds."
        )

    if failed:
        sys.exit(1)
    print("\n[OK] Chunking edge-cases passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
