from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> tuple[int, str, str]:
    """Run a command and capture (returncode, stdout, stderr)."""
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def parse_metrics(text: str) -> tuple[float, float]:
    """Parse recall@5 and mrr@5 from eval_embeddings.py output.

    Accepts both the dict-print form and the markdown table emitted by eval_embeddings.py.
    Falls back to scanning for keys in JSON-looking lines.
    """
    recall = None
    mrr = None
    # Try to find a JSON-ish dict line like: {'recall@5': 0.88, 'mrr@5': 0.67, ...}
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("{") and s.endswith("}"):
            try:
                # Replace single quotes with double quotes for JSON
                js = s.replace("'", '"')
                obj = json.loads(js)
                recall = float(obj.get("recall@5")) if obj.get("recall@5") is not None else recall
                mrr = float(obj.get("mrr@5")) if obj.get("mrr@5") is not None else mrr
            except Exception:
                pass
        # Markdown table line parsing is omitted; eval script prints dicts first.
    if recall is None or mrr is None:
        # Best-effort: regex-free scan of key/value tokens
        for line in text.splitlines():
            if "recall@5" in line and recall is None:
                try:
                    recall = float(line.split("recall@5")[-1].split()[-1].strip(" ,|:"))
                except Exception:
                    pass
            if "mrr@5" in line and mrr is None:
                try:
                    mrr = float(line.split("mrr@5")[-1].split()[-1].strip(" ,|:"))
                except Exception:
                    pass
    return float(recall or 0.0), float(mrr or 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-evaluate chunking params and record metrics")
    parser.add_argument("--targets", nargs="*", type=int, default=[350, 500, 650])
    parser.add_argument("--overlaps", nargs="*", type=int, default=[80, 120, 160])
    parser.add_argument("--gold", default=str(Path("data/gold/gold_qa.jsonl")))
    parser.add_argument(
        "--outfile",
        default=str(
            Path(".reports") / f"chunk_grid_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        ),
    )
    parser.add_argument(
        "--sig",
        default="hf:paraphrase-multilingual-mpnet-base-v2:norm",
        help="Embedding signature used by eval script for retrieval",
    )
    args = parser.parse_args()

    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "overlap", "recall5", "mrr5"])  # header

        for t, o in itertools.product(args.targets, args.overlaps):
            print(f"\n[GRID] target={t} overlap={o}")

            # Ingest with overrides (use our ingest_kb script)
            code, out, err = _run(
                [
                    sys.executable,
                    "scripts/ingest_kb.py",
                    "--source",
                    "data/pdfs",
                    "--chunk-target",
                    str(t),
                    "--chunk-overlap",
                    str(o),
                ]
            )
            if code != 0:
                print("[WARN] ingest failed:", err or out)

            # Evaluate via existing eval_embeddings
            code, out, err = _run(
                [
                    sys.executable,
                    "scripts/eval_embeddings.py",
                    "--gold",
                    args.gold,
                    "--only",
                    "mpnet",
                    "--sig-mpnet",
                    args.sig,
                    "--k",
                    "5",
                ]
            )
            if code != 0:
                print("[WARN] eval failed:", err or out)
            recall, mrr = parse_metrics(out)
            writer.writerow([t, o, f"{recall:.4f}", f"{mrr:.4f}"])
            f.flush()


if __name__ == "__main__":
    main()
