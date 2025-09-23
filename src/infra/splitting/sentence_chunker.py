from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    from langchain.schema import Document

from .sentence_splitter import split_to_sentences
from .types import TextSplitterLike


def _is_title_only_chunk(doc: Document, max_chars: int) -> bool:
    """
    A chunk is considered 'title-only' if it has no detected sentences and is very short.
    Requires chunker to set num_sentences & char_len metadata. Falls back to text length.
    """
    md = getattr(doc, "metadata", None) or {}
    try:
        num_sentences = int(md.get("num_sentences", 0))
    except Exception:
        num_sentences = 0
    try:
        char_len = int(md.get("char_len", len(getattr(doc, "page_content", "") or "")))
    except Exception:
        char_len = len(getattr(doc, "page_content", "") or "")
    return num_sentences == 0 and char_len <= int(max_chars)


def _same_source(doc_a: Document, doc_b: Document) -> bool:
    a = getattr(doc_a, "metadata", None) or {}
    b = getattr(doc_b, "metadata", None) or {}
    return a.get("source") == b.get("source")


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
                # Ensure key metadata are retained even if not present in 'meta'
                for k in ("source", "page", "title", "section", "category"):
                    if k not in md and k in (meta or {}):
                        md[k] = meta[k]
                if source is not None:
                    md["source"] = source
                if page is not None:
                    md["page"] = page
                out.append(Document(page_content=text, metadata=md))
        # post-process: merge adjacent tiny chunks from same page/source
        docs = self._merge_small_neighbors(out)

        # NEW: cross-page merge of title-only chunks
        try:
            from src.core.settings import AppSettings

            cfg = AppSettings().section_context
            docs = self._merge_title_into_next_page(
                docs,
                title_max_chars=cfg.title_only_max_chars,
                enable=cfg.cross_page_merge,
            )
        except Exception:
            # settings not importable -> skip gracefully
            pass

        # NEW: inject section titles into the first chunk of each section
        try:
            from src.core.settings import AppSettings as _AS

            cfg2 = _AS().section_context
            docs = self._inject_section_titles(
                docs,
                enabled=cfg2.inject_section_title,
                inject_once=cfg2.inject_once_per_section,
                fmt=cfg2.section_prefix_format,
            )
        except Exception:
            pass

        return docs

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
                # If the upcoming sentence still wouldn't fit even with overlap,
                # clear overlap to ensure progress (force-start a new chunk with s).
                overlap_len = sum(len(x) + len(self.p.joiner) for x in overlap)
                max_total = self.p.chunk_size + self.p.max_overflow
                if overlap_len + s_len > max_total:
                    # If a single sentence is too long for any chunk, emit it as its own chunk
                    if s_len > max_total:
                        chunks.append(s)
                        i += 1
                        cur = []
                        cur_len = 0
                    else:
                        # Start fresh so the next loop will add s (cur is empty)
                        cur = []
                        cur_len = 0
                else:
                    cur = overlap[:]  # start next chunk with sentence-level overlap
                    cur_len = overlap_len

        if cur:
            chunks.append(self.p.joiner.join(cur).strip())
        return chunks

    def _merge_small_neighbors(self, docs: list[Document], _pass: int = 0) -> list[Document]:
        """Merge adjacent tiny chunks from same source & page.

        Runs at most two passes (initial + one extra) to avoid quadratic behavior
        on very long sequences. This matches the intention in the comment and is
        sufficient in practice to eliminate tiny pairs.
        """
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
        # If a pass performed merges, run at most one additional pass
        if len(merged) < len(docs) and _pass < 1:
            return self._merge_small_neighbors(merged, _pass=_pass + 1)
        return merged

    def _merge_title_into_next_page(
        self, docs: list, *, title_max_chars: int, enable: bool
    ) -> list:
        """
        If a chunk is a 'title-only' piece (tiny, 0 sentences) and the next chunk
        is from the next page of the same source, prepend title into that next chunk and drop the title chunk.
        """
        if not enable or not docs:
            return docs

        merged: list = []
        i = 0
        while i < len(docs):
            cur = docs[i]
            if i + 1 < len(docs):
                nxt = docs[i + 1]
                cur_page = (cur.metadata or {}).get("page")
                nxt_page = (nxt.metadata or {}).get("page")
                if (
                    _same_source(cur, nxt)
                    and isinstance(cur_page, int)
                    and isinstance(nxt_page, int)
                    and nxt_page == cur_page + 1
                    and _is_title_only_chunk(cur, title_max_chars)
                ):
                    # Prepend title text to next chunk
                    title_text = (cur.page_content or "").strip()
                    if title_text:
                        sep = "\n" if not (nxt.page_content or "").startswith(title_text) else " "
                        new_text = f"{title_text}{sep}{nxt.page_content or ''}".strip()
                        # copy next chunk and modify text + metadata markers
                        try:
                            from langchain_core.documents import Document as _Doc  # prefer core
                        except Exception:  # pragma: no cover
                            from langchain.schema import Document as _Doc
                        md = dict(nxt.metadata or {})
                        md["title_merged_from_page"] = cur_page
                        md["title_merged"] = True
                        nxt = _Doc(page_content=new_text, metadata=md)
                    # drop the title-only chunk, keep modified next
                    merged.append(nxt)
                    i += 2
                    continue
            # default: keep current
            merged.append(cur)
            i += 1
        return merged

    def _inject_section_titles(
        self, docs: list, *, enabled: bool, inject_once: bool, fmt: str
    ) -> list:
        """
        Prepend section title into the first chunk of each new section (once),
        unless the text already starts with that title (avoid duplication).
        """
        if not enabled or not docs:
            return docs

        out: list = []
        last_section_by_source: dict[str | None, str] = {}

        for d in docs:
            md = getattr(d, "metadata", None) or {}
            source = md.get("source")
            section = (md.get("section") or "").strip()
            text = getattr(d, "page_content", "") or ""

            should_inject = bool(section)
            if (
                inject_once
                and source in last_section_by_source
                and last_section_by_source.get(source) == section
            ):
                should_inject = False

            if should_inject:
                # avoid double prefix if already starts with the section
                normalized_head = (text[: len(section)]).strip().lower()
                if normalized_head != section.lower():
                    injected = fmt.format(section=section, text=text)
                    try:
                        from langchain_core.documents import Document as _Doc
                    except Exception:  # pragma: no cover
                        from langchain.schema import Document as _Doc
                    md2 = dict(md)
                    md2["section_injected"] = True
                    d = _Doc(page_content=injected, metadata=md2)

                last_section_by_source[source] = section

            out.append(d)

        return out


# --- Light, non-intrusive instrumentation helpers ---

SENTENCE_END_RE = re.compile(r"[.!?…]\s*$", re.UNICODE)


@dataclass
class ChunkDiagnostics:
    total_chunks: int
    sentence_boundary_ends: int
    percent_sentence_boundary: float
    orphan_minis: int
    section_injected_counts: dict[str, int]
    title_merged_count: int


def ends_on_sentence_boundary(text: str) -> bool:
    """Heuristic check whether a chunk ends on a sentence boundary.

    Strips trailing whitespace/footnotes-like content and matches basic punctuation.
    """
    return bool(SENTENCE_END_RE.search((text or "").strip()))


def compute_chunk_diagnostics(chunks: list[dict[str, Any]]) -> ChunkDiagnostics:
    """Compute light diagnostics over a list of chunk dicts.

    Expected chunk shape: {"text": str, "metadata": dict}
    Uses existing metadata fields when available and does not mutate inputs.
    """
    total = len(chunks)
    sent_end = sum(1 for c in chunks if ends_on_sentence_boundary(c.get("text", "")))
    orphan_minis = sum(
        1
        for c in chunks
        if (c.get("metadata", {}) or {}).get("num_sentences", 0) == 0
        or len((c.get("text", "") or "").strip()) < 60  # align with title-only heuristic
    )

    # Count section_injected per (source, section)
    sec_counts: dict[str, int] = {}
    for c in chunks:
        md = c.get("metadata", {}) or {}
        key = f"{md.get('source', '')}|{md.get('section', '')}"
        if md.get("section_injected"):
            sec_counts[key] = sec_counts.get(key, 0) + 1

    title_merged = sum(1 for c in chunks if (c.get("metadata", {}) or {}).get("title_merged"))

    pct = (sent_end / max(1, total)) * 100.0
    return ChunkDiagnostics(
        total_chunks=total,
        sentence_boundary_ends=sent_end,
        percent_sentence_boundary=pct,
        orphan_minis=orphan_minis,
        section_injected_counts=sec_counts,
        title_merged_count=title_merged,
    )


# --- Adaptive Chunker (target/min/max/overlap) ---


class SentenceChunker:
    """Adaptive sentence chunker with target/min/max length and overlap hints.

    This class operates on pre-tokenized sentence objects with attributes:
    - text: str (sentence text)
    - page: int (1-based page index)
    - section: Optional[str] (title/TOC section)

    It does not modify metadata beyond emitting start/end page and section markers
    for downstream assemblers. Title merge & injection are signaled via flags only;
    actual text manipulation remains up to higher-level processors.
    """

    def __init__(
        self,
        target_chars: int,
        min_chars: int,
        max_chars: int,
        overlap_chars: int,
        enforce_sentence_boundaries: bool = True,
        inject_section_titles: bool = True,
        cross_page_title_merge: bool = True,
    ) -> None:
        self.target_chars = int(target_chars)
        self.min_chars = int(min_chars)
        self.max_chars = int(max_chars)
        self.overlap_chars = int(overlap_chars)
        self.enforce_sentence_boundaries = bool(enforce_sentence_boundaries)
        self.inject_section_titles = bool(inject_section_titles)
        self.cross_page_title_merge = bool(cross_page_title_merge)

    def chunk(self, sentences: list) -> list[dict]:
        # Build chunks as lists of sentence objects first
        chunks_sents: list[list] = []
        current: list = []
        length = 0  # sum of sentence text lengths (preserve original)

        for sent in sentences:
            s_text = getattr(sent, "text", "") or ""
            s_len = len(s_text)

            # If adding would exceed max and we already have content, finalize current
            if length + s_len > self.max_chars and current:
                chunks_sents.append(current)
                current = []
                length = 0

            # Append sentence and update length
            current.append(sent)
            length += s_len

            # When target reached or exceeded, finalize current
            if length >= self.target_chars:
                chunks_sents.append(current)
                current = []
                length = 0

        if current:
            chunks_sents.append(current)

        # Enforce minimum length by merging a tiny tail into the previous chunk
        if len(chunks_sents) >= 2:

            def _chunk_len(sent_list: list) -> int:
                return sum(len(getattr(s, "text", "") or "") for s in sent_list)

            if _chunk_len(chunks_sents[-1]) < self.min_chars:
                chunks_sents[-2].extend(chunks_sents[-1])
                chunks_sents.pop()

        return [self._finalize(slist) for slist in chunks_sents]

    def _finalize(self, sents: list) -> dict:
        # Preserve original sentence text verbatim and concatenate
        text = "".join((getattr(s, "text", "") or "") for s in sents)
        start_page = getattr(sents[0], "page", None)
        end_page = getattr(sents[-1], "page", None)
        section = getattr(sents[0], "section", None)
        return {
            "text": text,
            "start_page": start_page,
            "end_page": end_page,
            "section_title": section,
            "title_merged": self.cross_page_title_merge,
            "length": len(text),
        }
