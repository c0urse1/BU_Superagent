from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

from .config import settings
from .exceptions import ConfigurationError
from .ingest.embeddings import build_embedder
from .ingest.loaders import PdfLoader
from .ingest.pipeline import IngestionPipeline
from .ingest.splitters import RecursiveSplitter
from .ingest.store import ChromaStore
from .logging_setup import setup_logging

app = typer.Typer(add_completion=False, no_args_is_help=True)


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
    pipeline = IngestionPipeline(
        loader=PdfLoader(),
        splitter=RecursiveSplitter(cfg_chunk_size, cfg_chunk_overlap),
        store=store,
    )

    count = pipeline.run(pdfs)
    typer.echo(f"OK: {count} chunks → {cfg_persist} (collection={cfg_collection})")


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
