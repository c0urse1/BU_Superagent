from __future__ import annotations

# ruff: noqa: E402, B008

"""Thin CLI that delegates to use-cases.

Commands:
- ingest: recursively load PDFs and ingest into Chroma via ImportDocumentsUseCase
- query: retrieve top-k chunks or ask the LLM for an answer
"""

import sys
from pathlib import Path

import typer

from src.config.configure_app import get_import_use_case, get_query_use_case

app = typer.Typer(add_completion=False)


from src.application.use_cases import ImportDocumentsUseCase, QueryUseCase


def _build_import_use_case() -> ImportDocumentsUseCase:
    return get_import_use_case()


def _build_query_use_case() -> QueryUseCase:
    return get_query_use_case()


@app.command("ingest")
def ingest_cmd(
    source: Path | None = typer.Argument(None, help="Directory with PDFs (default: data/pdfs)")
) -> None:
    if source is None:
        source = Path("data/pdfs")
    uc = _build_import_use_case()
    count = uc.execute(source)
    typer.echo(f"Imported {count} chunks from {source}.")


@app.command("query")
def query_cmd(
    query: str = typer.Argument(..., help="Suchfrage / Query-Text"),
    k: int = typer.Option(5, "--top-k", help="Anzahl Treffer"),
    mmr: bool = typer.Option(False, "--mmr/--no-mmr", help="Maximal Marginal Relevance"),
    fetch_k: int = typer.Option(0, help="MMR: Kandidatenpool (0 = auto)"),
    threshold: float = typer.Option(-1.0, help="Score-Schwelle (0..1) – nur bei relevance_scores"),
    as_json: bool = typer.Option(False, "--json", help="Ergebnis als JSON ausgeben"),
    normalize_scores: bool = typer.Option(
        False,
        "--normalize-scores/--no-normalize-scores",
        help="Scores (pro Ergebnisliste) via Min-Max auf 0..1 normieren",
    ),
) -> None:
    uc = _build_query_use_case()
    # Try to get scores when not using MMR and a threshold/normalization is desired
    if not mmr and (threshold >= 0 or normalize_scores or as_json):
        pairs = uc.retrieve_top_k_with_scores(
            query,
            k=k,
            score_threshold=(None if threshold < 0 else threshold),
        )
        docs = [d for (d, _s) in pairs]
        scores = [s for (_d, s) in pairs]
    else:
        docs = uc.get_top_k(query, k=k, use_mmr=mmr, fetch_k=(None if fetch_k <= 0 else fetch_k))
        scores = [None] * len(docs)

    if not docs:
        typer.echo("No results.")
        raise typer.Exit()

    # Optional normalization (only for present scores)
    if normalize_scores:
        from src.domain.scoring import normalize_scores as _norm

        present = [s for s in scores if isinstance(s, (int | float)) and s is not None]
        if present:
            normed = _norm([float(s) for s in present])
            it = iter(normed)
            scores = [next(it) if s is not None else None for s in scores]

    if as_json:
        import json

        payload = []
        for i, (d, s) in enumerate(zip(docs, scores, strict=False), 1):
            m = d.metadata
            payload.append(
                {
                    "index": i,
                    "score": s,
                    "source": m.get("source"),
                    "page": m.get("page"),
                    "content": d.content,
                    "metadata": m,
                }
            )
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        raise typer.Exit()

    for i, (d, s) in enumerate(zip(docs, scores, strict=False), 1):
        m = d.metadata
        prefix = f"[{i}]" if s is None else f"[{i}] score={float(s):.3f}"
        src = m.get("source", "")
        page = m.get("page")
        title = m.get("title", "")
        section = m.get("section", "")
        head = f"{prefix} {title} | {section} (p.{page}) | {src}"
        text = (d.content or "").strip().replace("\n", " ")
        if len(text) > 300:
            text = text[:300] + "..."
        typer.echo(head)
        typer.echo("     " + text)


def main() -> int:
    try:
        app()
        return 0
    except Exception as e:  # noqa: BLE001
        typer.secho(f"Error: {e}", fg=typer.colors.RED)
        return 1


if __name__ == "__main__":
    sys.exit(main())
