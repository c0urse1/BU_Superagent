from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    from langchain.schema import Document

from .sentence_splitter import split_to_sentences
from .types import TextSplitterLike


@dataclass
class SentenceAwareParams:
    chunk_size: int = 1000  # soft target in characters
    chunk_overlap: int = 150  # desired overlap (approx) in characters
    max_overflow: int = 200  # allow last sentence to exceed chunk_size up to this
    min_merge_char_len: int = 500  # post-merge: join adjacent tiny chunks if combined < this
    joiner: str = "\n"  # join sentences with newline


class SentenceAwareChunker(TextSplitterLike):
    def __init__(self, params: SentenceAwareParams | None = None) -> None:
        self.p = params or SentenceAwareParams()

    def split_documents(self, docs: list[Document]) -> list[Document]:
        out: list[Document] = []
        for doc in docs:
            meta = dict(getattr(doc, "metadata", None) or {})
            # keep track of original markers commonly used downstream
            source = meta.get("source")
            page = meta.get("page", meta.get("page_number"))
            sentences = split_to_sentences(getattr(doc, "page_content", "") or "")
            chunks = self._pack_sentences(sentences)
            # attach metadata + computed stats
            for idx, text in enumerate(chunks):
                md: dict[str, Any] = {
                    **meta,
                    "chunk_index": idx,
                    "char_len": len(text),
                    # cheap proxy for number of sentences
                    "num_sentences": text.count(".") + text.count("!") + text.count("?"),
                }
                if source is not None:
                    md["source"] = source
                if page is not None:
                    md["page"] = page
                out.append(Document(page_content=text, metadata=md))
        # post-process: merge adjacent tiny chunks from same page/source
        return self._merge_small_neighbors(out)

    def _pack_sentences(self, sentences: list[str]) -> list[str]:
        chunks: list[str] = []
        if not sentences:
            return chunks
        cur: list[str] = []
        cur_len = 0

        i = 0
        while i < len(sentences):
            s = sentences[i]
            s_len = len(s) + len(self.p.joiner)
            if cur_len + s_len <= self.p.chunk_size or not cur:
                cur.append(s)
                cur_len += s_len
                i += 1
            else:
                # soft overflow: if adding this one would exceed by <= max_overflow, still include
                if (cur_len + s_len - self.p.chunk_size) <= self.p.max_overflow:
                    cur.append(s)
                    cur_len += s_len
                    i += 1
                # finalize chunk and start a new one with overlap
                chunks.append(self.p.joiner.join(cur).strip())
                # build overlap by reusing tail sentences to approx overlap length
                overlap: list[str] = []
                ol_len = 0
                for s_rev in reversed(cur):
                    ol_len += len(s_rev) + len(self.p.joiner)
                    overlap.insert(0, s_rev)
                    if ol_len >= self.p.chunk_overlap:
                        break
                cur = overlap[:]  # start next chunk with sentence-level overlap
                cur_len = sum(len(x) + len(self.p.joiner) for x in cur)

        if cur:
            chunks.append(self.p.joiner.join(cur).strip())
        return chunks

    def _merge_small_neighbors(self, docs: list[Document]) -> list[Document]:
        """Merge adjacent tiny chunks from same source & page until stable."""
        if not docs:
            return docs
        merged: list[Document] = []
        i = 0
        while i < len(docs):
            cur = docs[i]
            if i + 1 < len(docs):
                nxt = docs[i + 1]
                same_src = (cur.metadata or {}).get("source") == (nxt.metadata or {}).get("source")
                same_page = (cur.metadata or {}).get("page") == (nxt.metadata or {}).get("page")
                if same_src and same_page:
                    cur_len = len(cur.page_content or "")
                    nxt_len = len(nxt.page_content or "")
                    combined_len = cur_len + nxt_len
                    if (
                        combined_len < self.p.min_merge_char_len
                        or cur_len < self.p.min_merge_char_len
                        or nxt_len < self.p.min_merge_char_len
                    ):
                        text = ((cur.page_content or "") + "\n" + (nxt.page_content or "")).strip()
                        md = {**(cur.metadata or {})}
                        md["char_len"] = combined_len
                        md["num_sentences"] = text.count(".") + text.count("!") + text.count("?")
                        merged.append(Document(page_content=text, metadata=md))
                        i += 2
                        continue
            merged.append(cur)
            i += 1
        # If a single pass still leaves tiny pairs, run once more (idempotent in practice)
        if len(merged) < len(docs):
            return self._merge_small_neighbors(merged)
        return merged
