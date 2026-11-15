from __future__ import annotations

import functools
import importlib
import math
import re
from dataclasses import dataclass

DEFAULT_RATIO = 1.35
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Fitted:
    text: str
    tokens: int
    limit: int
    dropped: str = ""
    exact: bool = False

    @property
    def trimmed(self) -> bool:
        return bool(self.dropped)


@functools.lru_cache(maxsize=8)
def _tokenizer(name: str):
    try:
        transformers = importlib.import_module("transformers")
    except ImportError:
        return None
    try:
        return transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:  # noqa: BLE001
        return None


def count_tokens(
    text: str, tokenizer: str | None = None, ratio: float = DEFAULT_RATIO
) -> tuple[int, bool]:
    """Token count of `text` and whether it is exact.

    With a tokenizer that is already on disk the count is exact; otherwise it is
    estimated from words and punctuation with a per-tokenizer ratio, rounded up.
    """
    if tokenizer:
        tok = _tokenizer(tokenizer)
        if tok is not None:
            return len(tok(text, add_special_tokens=False)["input_ids"]), True
    words = len(_WORD.findall(text))
    punct = len(re.findall(r"[,;:()\"]", text))
    return math.ceil(words * ratio + punct * 0.5), False


def fit_tokens(
    text: str,
    limit: int,
    tokenizer: str | None = None,
    ratio: float = DEFAULT_RATIO,
    tags: bool = False,
) -> Fitted:
    """Shorten `text` to at most `limit` tokens without cutting mid-sentence or mid-tag.

    Sentences (or comma-separated tags) are kept from the start while they fit. Only
    when the first one alone is too long are words cut.
    """
    text = " ".join(text.split())
    total, exact = count_tokens(text, tokenizer, ratio)
    if total <= limit:
        return Fitted(text, total, limit, exact=exact)

    pieces = [p.strip() for p in text.split(",")] if tags else _SENTENCE.split(text)
    joiner = ", " if tags else " "
    kept: list[str] = []
    for piece in pieces:
        candidate = joiner.join([*kept, piece])
        if count_tokens(candidate, tokenizer, ratio)[0] > limit:
            break
        kept.append(piece)
    if not kept:
        words = text.split()
        lo, hi = 0, len(words)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count_tokens(" ".join(words[:mid]), tokenizer, ratio)[0] <= limit:
                lo = mid
            else:
                hi = mid - 1
        kept = [" ".join(words[:lo])]
    out = joiner.join(kept)
    dropped = text[len(out) :].lstrip(" ,") if text.startswith(out) else text
    tokens, exact = count_tokens(out, tokenizer, ratio)
    return Fitted(out, tokens, limit, dropped=dropped, exact=exact)
