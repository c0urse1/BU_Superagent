from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Optional: use syntok at runtime if present (better German segmentation)
_seg: Any | None = None
try:  # pragma: no cover - exercised indirectly in integration
    import importlib

    _seg = importlib.import_module("syntok.segmenter")
    _HAS_SYNTOK = True
except Exception:  # pragma: no cover
    _HAS_SYNTOK = False


# Common German abbreviations to avoid false splits after '.'
_ABBR = {
    "z.b.",
    "u.a.",
    "u.u.",
    "bzw.",
    "dr.",
    "prof.",
    "abs.",
    "vgl.",
    "etc.",
    "ca.",
    "nr.",
    "art.",
    "s.",
    "bd.",
    "kap.",
    "ggf.",
    "o.ä.",
    "i.d.r.",
    "i.s.d.",
}


def _split_with_syntok(text: str) -> list[str]:
    assert _seg is not None
    sents: list[str] = []
    process = getattr(_seg, "process", None)
    if not callable(process):
        return _split_with_regex(text)
    for paragraph in process(text):
        for sentence in paragraph:
            # reconstruct surface form with original spacing
            s = "".join([t.spacing + t.value for t in sentence]).strip()
            if s:
                sents.append(s)
    return sents


def _split_with_regex(text: str) -> list[str]:
    """Zero-dep heuristic German splitter with abbreviation guards and paragraph breaks.

    Heuristics:
    - Treat double newlines as hard paragraph boundaries
    - Avoid splitting on common abbreviations (e.g., "z.B.")
    - Avoid numeric patterns like "1.2" or list items like "1."
    - Allow trailing quotes/brackets after sentence-final punctuation
    """
    import re

    if not text:
        return []
    # Normalize spaces around newlines; treat double newlines as paragraph breaks (hard boundary)
    norm = text.replace("\r\n", "\n")
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", norm) if p.strip()]

    out: list[str] = []
    for p in paragraphs:
        buf: list[str] = []
        i = 0
        while i < len(p):
            ch = p[i]
            buf.append(ch)
            # Sentence end candidates
            if ch in ".!?…":
                nxt = p[i + 1] if i + 1 < len(p) else ""
                prev_token = "".join(reversed(_take_while_reversed(buf[:-1], str.isalnum))).lower()
                # Skip splits for abbreviations or numbers like "1.2" or "1." in lists
                if prev_token in _ABBR or (prev_token.isdigit() and nxt.isdigit()):
                    i += 1
                    continue
                # finalize sentence when next is space/quote/end
                # allow quotes or brackets to follow end punctuation
                j = i + 1
                while j < len(p) and p[j] in "\")”»']":
                    buf.append(p[j])
                    j += 1
                # consume trailing spaces
                while j < len(p) and p[j] in " \t":
                    j += 1
                sent = "".join(buf).strip()
                if sent:
                    out.append(sent)
                buf = []
                i = j - 1
            i += 1
        tail = "".join(buf).strip()
        if tail:
            out.append(tail)
    return out


def _take_while_reversed(chars: list[str], pred: Callable[[str], bool]) -> list[str]:
    out: list[str] = []
    for ch in reversed(chars):
        if pred(ch):
            out.append(ch)
        else:
            break
    return out


def split_to_sentences(text: str) -> list[str]:
    if _HAS_SYNTOK and _seg is not None:
        try:
            return _split_with_syntok(text)
        except Exception:  # pragma: no cover - fall back on any failure
            pass
    return _split_with_regex(text)
