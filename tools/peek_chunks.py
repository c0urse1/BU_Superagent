from __future__ import annotations

import itertools
from pathlib import Path

import chromadb


def main(
    path: str = ".vector_store/chroma", collection: str = "bu_knowledge", limit: int = 3
) -> None:
    p = Path(path)
    if not p.exists():
        print(f"No persistence dir found at: {p}")
        return

    client = chromadb.PersistentClient(path=str(p))
    try:
        col = client.get_collection(collection)
    except Exception as e:  # noqa: BLE001
        print(f"Collection not found ({collection}): {e}")
        return

    total = col.count()  # count of items in the collection
    res = col.get(include=["documents", "metadatas"])  # request allowed fields
    ids = list(range(len(res.get("documents", []) or [])))
    docs = res.get("documents", []) or []
    metas = res.get("metadatas", []) or []

    print("total chunks:", total)
    for i, (id_, doc, meta) in enumerate(
        itertools.islice(zip(ids, docs, metas, strict=False), limit)
    ):
        source = (meta or {}).get("source")
        preview = (doc or "")[:120].replace("\n", " ")
        print(f"[{i}] id={id_} source={source} doc[:120]={preview!r}")


if __name__ == "__main__":
    main()
