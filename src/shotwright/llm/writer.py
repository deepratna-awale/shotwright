from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..plan.spec import ShotList, Transition
from ..profiles import Registry, UnknownModelError, default_registry
from .client import LLMClient, LLMError, image_part

TEMPLATE_VERSION = "v1"


def load_template(name: str, version: str = TEMPLATE_VERSION) -> str:
    return (resources.files("shotwright.llm") / "prompts" / f"{name}.{version}.md").read_text(
        encoding="utf-8"
    )


def family_guide(family: str | None) -> str:
    folder = resources.files("shotwright.llm") / "prompts" / "families"
    entry = folder / f"{family}.md" if family else None
    if entry is None or not entry.is_file():
        entry = folder / "default.md"
    return entry.read_text(encoding="utf-8").strip()


@dataclass
class Entry:
    name: str
    description: str


@dataclass
class StoryBible:
    characters: list[Entry] = field(default_factory=list)
    locations: list[Entry] = field(default_factory=list)
    props: list[Entry] = field(default_factory=list)
    style: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StoryBible:
        def entries(key: str) -> list[Entry]:
            out = []
            for item in data.get(key) or []:
                if isinstance(item, dict) and item.get("name") and item.get("description"):
                    out.append(Entry(str(item["name"]), str(item["description"])))
            return out

        style = data.get("style")
        return cls(
            entries("characters"),
            entries("locations"),
            entries("props"),
            str(style) if style else None,
        )

    @classmethod
    def load(cls, path: str | Path) -> StoryBible:
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True, width=100),
            encoding="utf-8",
        )

    def to_text(self) -> str:
        lines = []
        for title, items in (
            ("Characters", self.characters),
            ("Locations", self.locations),
            ("Props", self.props),
        ):
            if items:
                lines.append(f"{title}:")
                lines += [f"- {e.name}: {e.description}" for e in items]
        if self.style:
            lines.append(f"Style: {self.style}")
        return "\n".join(lines) or "(empty)"


def _brief(shot) -> str:
    return shot.brief or shot.prompt


def _length(shot, native_fps: float | None) -> str:
    if shot.duration is not None:
        return f"{shot.duration:g} seconds"
    if shot.frames is not None and native_fps:
        return f"about {shot.frames / native_fps:.1f} seconds"
    return "a few seconds"


def build_bible(spec: ShotList, client: LLMClient) -> StoryBible:
    shots = [
        {
            "id": s.id,
            "brief": _brief(s),
            "transition": s.transition.value,
            **({"duration": s.duration} if s.duration else {}),
        }
        for s in spec.shots
    ]
    user = yaml.safe_dump(
        {"project": spec.project, "style": spec.style, "shots": shots},
        sort_keys=False,
        allow_unicode=True,
    )
    data = client.chat_json(load_template("story_bible"), user)
    if not isinstance(data, dict):
        raise LLMError("the story bible reply was not a JSON object")
    return StoryBible.from_dict(data)


def write_prompts(
    spec: ShotList,
    client: LLMClient,
    bible: StoryBible | None = None,
    registry: Registry | None = None,
) -> ShotList:
    """Return a copy of `spec` with every shot's prompt written out in full.

    The original text is kept as each shot's `brief`, so running this again starts
    from the brief rather than from a previous expansion.
    """
    registry = registry or default_registry()
    bible = bible or StoryBible()
    out = spec.model_copy(deep=True)
    if bible.style and not out.style:
        out.style = bible.style
    previous: str | None = None
    for shot in out.shots:
        model_id = shot.model or spec.defaults.model
        family = native = None
        if model_id:
            try:
                profile = registry.get(model_id)
                family, native = profile.family, profile.native_fps
            except UnknownModelError:
                pass
        system = load_template("prompt_writer").replace("{family_guide}", family_guide(family))
        parts = [
            f"Story bible:\n{bible.to_text()}",
            f"Shot {shot.id}, {_length(shot, native)} long.",
        ]
        if shot.transition == Transition.CONTINUOUS and previous:
            parts.append(f"This shot continues directly from the previous shot:\n{previous}")
        else:
            parts.append("This shot starts with a cut.")
        if out.style:
            parts.append(f"Film style: {out.style}")
        parts.append(f"Brief:\n{_brief(shot)}")
        data = client.chat_json(system, "\n\n".join(parts))
        prompt = data.get("prompt") if isinstance(data, dict) else None
        if not isinstance(prompt, str) or not prompt.strip():
            raise LLMError(f"no prompt returned for shot {shot.id}")
        shot.brief = _brief(shot)
        shot.prompt = prompt.strip()
        previous = shot.prompt
    return out


def describe_frame(image: np.ndarray, client: LLMClient) -> str:
    data = client.chat_json(
        load_template("frame_describer"),
        [{"type": "text", "text": "Describe this frame."}, image_part(image)],
    )
    description = data.get("description") if isinstance(data, dict) else None
    if not isinstance(description, str) or not description.strip():
        raise LLMError("no frame description returned")
    return description.strip()


def continue_prompt(prompt: str, description: str, client: LLMClient) -> str:
    data = client.chat_json(
        load_template("continuation"),
        f"Starting frame:\n{description}\n\nPlanned prompt:\n{prompt}",
    )
    rewritten = data.get("prompt") if isinstance(data, dict) else None
    if not isinstance(rewritten, str) or not rewritten.strip():
        raise LLMError("no continuation prompt returned")
    return rewritten.strip()
