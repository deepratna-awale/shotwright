from __future__ import annotations

import re
import unicodedata

_SEXUAL = (
    r"sex|sexy|sexual\w*|erotic\w*|porn\w*|nsfw|explicit|nude|nudity|naked|topless|bottomless"
    r"|undress\w*|strip(?:s|ped|ping|tease)?|lingerie|genital\w*|breasts?|nipples?|penis|vagina"
    r"|vulva|orgasm\w*|masturbat\w*|intercourse|aroused|arousal|fondl\w*|grop\w*|seductive\w*"
    r"|lewd|xxx|fetish\w*|bdsm|hentai|cum|blowjob|handjob|thrust\w*"
)
_MINOR = (
    r"child|children|childlike|kids?|minors?|underage|under-age|pre-?teens?|tweens?|toddlers?"
    r"|infants?|bab(?:y|ies)|schoolgirls?|schoolboys?|lol[i1]|shota|jailbait|young teens?"
    r"|early teens|little (?:girl|boy)s?|(?:elementary|middle|primary|junior high) school\w*"
    r"|(?:[1-9]|1[0-7])[- ]?(?:yo|y/o|years?[- ]old)"
)
_NON_CONSENT = (
    r"rap(?:e|ed|es|ing|ist)|non-?consensual\w*|without (?:her|his|their|any)? ?consent"
    r"|against (?:her|his|their) will|forced|forcing|coerc\w*|unconscious|drugged|passed out"
    r"|asleep|sleeping|hidden camera|upskirt|voyeur\w*|molest\w*|assault\w*|revenge porn"
)

SEXUAL = re.compile(rf"\b(?:{_SEXUAL})\b")
MINOR = re.compile(rf"\b(?:{_MINOR})\b")
NON_CONSENT = re.compile(rf"\b(?:{_NON_CONSENT})\b")


class PolicyViolation(ValueError):
    pass


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower().replace("_", " ")
    return re.sub(r"(?<=\w)[.*](?=\w)", "", text)


def violation(*texts: str | None) -> str | None:
    """Return why the combined text is not allowed, or None."""
    text = _normalise(" ".join(t for t in texts if t))
    if not SEXUAL.search(text):
        return None
    if MINOR.search(text):
        return "sexual content involving minors"
    if NON_CONSENT.search(text):
        return "non-consensual sexual content"
    return None


def check(*texts: str | None, where: str = "prompt") -> None:
    reason = violation(*texts)
    if reason:
        raise PolicyViolation(f"{where} is not allowed: {reason}")
