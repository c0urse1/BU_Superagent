from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Literal, cast

import typer

from src.infra.splitting.factory import build_splitter

from .config import settings
from .exceptions import ConfigurationError
from .ingest.embeddings import build_embedder
from .ingest.loaders import PdfLoader
from .ingest.pipeline import IngestionPipeline
from .ingest.splitters import TextSplitterAdapter
from .ingest.store import ChromaStore
from .logging_setup import setup_logging
from .query import QueryService

app = typer.Typer(add_completion=False, no_args_is_help=True, help="KB Tools")


@app.command("ingest")
def ingest(
    source: str = typer.Option(None, help="Directory with PDFs (recursive)."),
    persist: str = typer.Option(None, help="Directory to persist Chroma DB."),
    collection: str = typer.Option(None, help="Chroma collection name."),
    model: str = typer.Option(None, help="HuggingFace sentence-transformer model."),
    chunk_size: int = typer.Option(None, help="Chunk size (chars)."),
    chunk_overlap: int = typer.Option(None, help="Chunk overlap (chars)."),
) -> None:
    setup_logging(logging.INFO)

    cfg_source = Path(source) if source else settings.source_dir
    cfg_persist = Path(persist) if persist else settings.persist_dir
    cfg_collection = collection or settings.collection_name
    cfg_model = model or settings.embedding_model
    cfg_chunk_size = chunk_size or settings.chunk_size
    cfg_chunk_overlap = chunk_overlap or settings.chunk_overlap

    if not cfg_source.exists():
        raise ConfigurationError(f"source_dir not found: {cfg_source}")

    pdfs: list[Path] = sorted([p for p in cfg_source.rglob("*.pdf") if p.is_file()])
    if not pdfs:
        typer.echo(f"No PDFs found under {cfg_source}")
        raise typer.Exit(code=0)

    embedder = build_embedder(cfg_model)
    store = ChromaStore(cfg_collection, cfg_persist, embedder)

    # Build configurable splitter (default: sentence-aware); fall back via getattr defaults
    # Prefer core AppSettings.chunking if present, else fall back to legacy attributes
    chunking = getattr(settings, "chunking", None)
    chunking_mode = getattr(chunking, "mode", getattr(settings, "chunking_mode", "sentence_aware"))
    chunk_max_overflow = getattr(
        chunking, "max_overflow", getattr(settings, "chunk_max_overflow", 200)
    )
    chunk_min_merge_char_len = getattr(
        chunking, "min_merge_char_len", getattr(settings, "chunk_min_merge_char_len", 500)
    )
    _mode_str = str(chunking_mode)
    if _mode_str not in ("sentence_aware", "recursive"):
        _mode_str = "sentence_aware"
    splitter_impl = build_splitter(
        mode=cast(Literal["sentence_aware", "recursive"], _mode_str),
        chunk_size=int(cfg_chunk_size),
        chunk_overlap=int(cfg_chunk_overlap),
        max_overflow=int(chunk_max_overflow),
        min_merge_char_len=int(chunk_min_merge_char_len),
    )

    pipeline = IngestionPipeline(
        loader=PdfLoader(),
        splitter=TextSplitterAdapter(splitter_impl),
        store=store,
    )

    count = pipeline.run(pdfs)
    typer.echo(f"OK: {count} chunks → {cfg_persist} (collection={cfg_collection})")


@app.command("query")
def query_cmd(
    text: str = typer.Argument(..., help="Suchfrage / Query-Text"),
    k: int = typer.Option(5, help="Anzahl Treffer"),
    mmr: bool = typer.Option(False, "--mmr/--no-mmr", help="Maximal Marginal Relevance"),
    fetch_k: int = typer.Option(0, help="MMR: Kandidatenpool (0 = auto)"),
    threshold: float = typer.Option(-1.0, help="Score-Schwelle (0..1) – nur bei relevance_scores"),
    as_json: bool = typer.Option(False, "--json", help="Ergebnis als JSON ausgeben"),
    normalize_scores: bool = typer.Option(
        False,
        "--normalize-scores/--no-normalize-scores",
        help="Scores (pro Ergebnisliste) via Min-Max auf 0..1 normieren",
    ),
    outfile: Path | None = typer.Option(  # noqa: B008 - Typer keeps options in signature
        None,
        "--outfile",
        "-o",
        help="JSON direkt in Datei (UTF-8) schreiben; Logs werden unterdrückt",
    ),
) -> None:
    """Read-only Top-k Abfrage gegen die Chroma-Collection."""
    # Quiet logs for machine-readable output; otherwise, show INFO
    quiet = bool(as_json or outfile)
    log_level = logging.ERROR if quiet else logging.INFO
    setup_logging(log_level)
    if quiet:
        warnings.filterwarnings("ignore")

    svc = QueryService()
    hits = svc.top_k(
        text,
        k=k,
        use_mmr=mmr,
        fetch_k=(fetch_k or None),
        score_threshold=(None if threshold < 0 else threshold),
    )

    # Optional Min-Max Normalisierung der Scores für Ausgabe
    if normalize_scores and hits:
        numeric_scores = [
            h.score for h in hits if isinstance(h.score, int | float) and h.score is not None
        ]
        if numeric_scores:
            min_s = min(numeric_scores)
            max_s = max(numeric_scores)
            if max_s == min_s:
                for h in hits:
                    if h.score is not None:
                        h.score = 1.0
            else:
                rng = max_s - min_s
                for h in hits:
                    if h.score is not None:
                        h.score = float((h.score - min_s) / rng)

    # JSON- oder Datei-Ausgabe (UTF-8, ohne Log-Spam)
    if as_json or outfile:
        payload = [
            {
                "index": i + 1,
                "score": h.score,
                "source": h.source,
                "page": h.page,
                "content": h.content,
                "metadata": h.metadata,
            }
            for i, h in enumerate(hits)
        ]
        data = json.dumps(payload, ensure_ascii=False, indent=2)

        if outfile:
            try:
                outfile.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            with open(outfile, "w", encoding="utf-8", newline="\n") as f:
                f.write(data + "\n")
        else:
            # Robust UTF-8 output for Windows consoles and redirection
            try:
                getattr(sys.stdout, "reconfigure", lambda **_: None)(
                    encoding="utf-8", errors="replace"
                )
                sys.stdout.write(data + "\n")
            except Exception:  # noqa: BLE001
                try:
                    sys.stdout.buffer.write((data + "\n").encode("utf-8"))
                except Exception:  # last resort
                    print(data)
        raise typer.Exit()

    # Menschlich lesbare Ausgabe
    if not hits:
        typer.echo("Keine Treffer.")
        return

    for i, h in enumerate(hits, start=1):
        if h.score is not None:
            header = f"[{i}] score={h.score:.3f}"
        else:
            header = f"[{i}]"
        src = h.source + (f":{h.page}" if h.page is not None else "")
        typer.echo(f"{header}  {src}")
        content_line = (h.content or "").strip().replace("\n", " ")
        if len(content_line) > 600:
            content_line = content_line[:600] + "..."
        typer.echo(content_line)
        typer.echo("-" * 80)


def main() -> int:
    try:
        app()
        return 0
    except ConfigurationError as ce:
        typer.secho(f"Config error: {ce}", fg=typer.colors.RED)
        return 2
    except Exception as e:  # noqa: BLE001
        typer.secho(f"Unexpected error: {e}", fg=typer.colors.RED)
        return 1


if __name__ == "__main__":
    sys.exit(main())
