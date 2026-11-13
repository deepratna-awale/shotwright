from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Transition(str, Enum):
    CUT = "cut"
    CONTINUOUS = "continuous"


class OutputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)


class Defaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    seed: int = 0
    negative: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class Line(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    speaker: str | None = None
    at: float = Field(default=0.0, ge=0)


class ContinuitySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_hops: int = Field(default=6, ge=1)


class ShotSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    prompt: str
    brief: str | None = None
    duration: float | None = Field(default=None, gt=0)
    frames: int | None = Field(default=None, gt=0)
    transition: Transition = Transition.CUT
    model: str | None = None
    seed: int | None = None
    negative: str | None = None
    init_image: str | None = None
    end_image: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    lines: list[Line] = Field(default_factory=list)

    @model_validator(mode="after")
    def _length(self) -> ShotSpec:
        if self.duration is not None and self.frames is not None:
            raise ValueError(f"shot {self.id}: give duration or frames, not both")
        return self


class ShotList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    output: OutputSpec
    defaults: Defaults = Field(default_factory=Defaults)
    style: str | None = None
    continuity: ContinuitySettings = Field(default_factory=ContinuitySettings)
    voices: dict[str, str] = Field(default_factory=dict)
    shots: list[ShotSpec] = Field(min_length=1)
    base_dir: Path | None = Field(default=None, exclude=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ShotList:
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        spec = cls(**data)
        spec.base_dir = path.parent.resolve()
        return spec

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: str | Path | None = None) -> ShotList:
        spec = cls(**data)
        spec.base_dir = Path(base_dir).resolve() if base_dir else None
        return spec

    def to_yaml(self, path: str | Path) -> None:
        data = self.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
        data = {"project": self.project, "output": self.output.model_dump(mode="json"), **data}
        Path(path).write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100),
            encoding="utf-8",
        )

    def resolve_path(self, value: str) -> str:
        p = Path(value).expanduser()
        if not p.is_absolute() and self.base_dir is not None:
            p = self.base_dir / p
        return str(p)
