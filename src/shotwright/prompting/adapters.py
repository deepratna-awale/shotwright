from __future__ import annotations

from dataclasses import dataclass

from shotwright.plan.spec import Look, PromptParts
from shotwright.profiles.schema import I2VPrompt, ModelProfile
from shotwright.prompting.budget import count_tokens, fit_tokens

ANIME_STYLE = "anime style, 2D cel shading"
_MODES = ("t2v", "i2v")


@dataclass(frozen=True)
class AdaptedPrompt:
    text: str
    negative: str | None
    tokens: int
    limit: int
    exact: bool
    dropped: str
    notes: tuple[str, ...]


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def _sentence(text: str | None) -> str:
    text = _clean(text).rstrip(" ,;:")
    if not text:
        return ""
    return text if text[-1] in ".!?" else text + "."


def _join(sentences: list[str]) -> str:
    return " ".join(s for s in (_sentence(x) for x in sentences) if s)


def _contains(text: str, phrase: str) -> bool:
    return _clean(phrase).rstrip(".").lower() in _clean(text).lower()


class _Builder:
    """Ordered sentences, with the free prompt placed after the subject/action sentences."""

    def __init__(self, free: str = ""):
        self.items: list[tuple[str, str]] = []
        self.free = _clean(free)

    def add(self, key: str, text: str | None) -> None:
        if _clean(text):
            self.items.append((key, text or ""))

    def text(self) -> str:
        sentences = [t for _, t in self.items]
        if self.free:
            keys = [k for k, _ in self.items]
            anchors = [i for i, k in enumerate(keys) if k in ("subject", "action")]
            at = anchors[-1] + 1 if anchors else sum(1 for k in keys if k == "prefix")
            sentences.insert(at, self.free)
        return _join(sentences)


def _t2v(family: str, p: PromptParts, style: str, look: Look, free: str) -> str:
    b = _Builder(free)
    if family == "wan":
        b.add("prefix", style)
        anime = look == Look.ANIME
        if not anime:
            tags = ", ".join(_clean(x).rstrip(".") for x in (p.lighting, p.camera) if _clean(x))
            b.add("prefix", tags)
        b.add("subject", p.subject)
        b.add("action", p.action)
        b.add("setting", p.setting)
        if anime:
            b.add("lighting", p.lighting)
            b.add("camera", p.camera)
    elif family == "hunyuanvideo":
        b.add("subject", p.subject)
        b.add("action", p.action)
        b.add("setting", p.setting)
        b.add("camera", p.camera)
        b.add("lighting", p.lighting)
        b.add("style", style)
    elif family == "ltx":
        if _clean(style):
            b.add("prefix", f"Style: {_clean(style).rstrip('.')}")
        b.add("action", p.action)
        b.add("subject", p.subject)
        b.add("setting", p.setting)
        b.add("camera", p.camera)
        b.add("lighting", p.lighting)
        b.add("audio", p.audio)
    else:
        b.add("subject", p.subject)
        b.add("action", p.action)
        b.add("setting", p.setting)
        b.add("lighting", p.lighting)
        b.add("camera", p.camera)
        b.add("style", style)
    return b.text()


def _i2v(
    family: str,
    p: PromptParts,
    action: str | None,
    rule: I2VPrompt,
    start: str | None,
    free: str,
) -> str:
    b = _Builder(free)
    if rule == I2VPrompt.DESCRIBE_FIRST:
        if _clean(start):
            b.add("prefix", start)
        else:
            b.add("subject", p.subject)
            b.add("setting", p.setting)
            b.add("lighting", p.lighting)
    b.add("action", action)
    b.add("camera", p.camera)
    if family == "ltx":
        b.add("audio", p.audio)
    return b.text()


def _style_with_free(family: str, text: str, style: str) -> str:
    if not _clean(style) or _contains(text, style):
        return _join([text])
    return _join([style, text] if family == "wan" else [text, style])


def _words(text: str) -> int:
    return len(text.split())


def adapt_prompt(
    profile: ModelProfile,
    *,
    mode: str,
    prompt: str = "",
    parts: PromptParts | None = None,
    motion: str | None = None,
    start: str | None = None,
    style: str | None = None,
    look: Look = Look.REALISTIC,
    negative: str | None = None,
) -> AdaptedPrompt:
    """Turn prompt pieces into the text a model's official prompting guide asks for.

    Each family orders the pieces differently; image-to-video prompts describe only
    the motion, unless the profile says the model wants the first frame described.
    The result is fitted to the text encoder's token limit.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    rules = profile.prompt
    family = profile.family
    look = Look(look)
    notes: list[str] = []
    free = _clean(prompt)
    if parts is not None and parts.is_empty():
        parts = None

    style_phrase = _clean((parts.style if parts else None) or style)
    if not style_phrase and look == Look.ANIME:
        style_phrase = ANIME_STYLE
    if style_phrase and free and _contains(free, style_phrase):
        style_phrase = ""

    if parts is not None:
        if mode == "i2v" and _clean(motion or parts.action):
            text = _i2v(family, parts, motion or parts.action, rules.i2v, start, free)
        else:
            text = _t2v(family, parts, style_phrase, look, free)
    elif mode == "i2v":
        if _clean(motion):
            first = start if _clean(start) else free
            if rules.i2v == I2VPrompt.DESCRIBE_FIRST:
                text = _join([first, motion or ""])
            else:
                text = _join([motion or ""])
                if free:
                    notes.append(f"free prompt left out: {family} i2v prompts describe motion only")
        else:
            text = _join([free])
    else:
        text = _style_with_free(family, free, style_phrase)

    before, _ = count_tokens(text, rules.tokenizer, rules.tokens_per_word)
    fitted = fit_tokens(
        text, rules.max_tokens, tokenizer=rules.tokenizer, ratio=rules.tokens_per_word
    )
    if fitted.trimmed:
        notes.append(
            f"trimmed {before - fitted.tokens} tokens to fit {rules.encoder}'s {rules.max_tokens}"
        )
    upper = (rules.i2v_words if mode == "i2v" else rules.t2v_words)[1]
    words = _words(fitted.text)
    if words > upper:
        notes.append(f"{words} words is more than the {upper} recommended for {family} {mode}")

    if not rules.uses_negative:
        neg = None
        if _clean(negative):
            notes.append(f"negative prompt ignored: {profile.id} does not use one")
    else:
        neg = negative if negative is not None else rules.negative

    return AdaptedPrompt(
        text=fitted.text,
        negative=neg,
        tokens=fitted.tokens,
        limit=fitted.limit,
        exact=fitted.exact,
        dropped=fitted.dropped,
        notes=tuple(notes),
    )
