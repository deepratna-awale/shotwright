from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Conditioning(str, Enum):
    T2V = "t2v"
    I2V = "i2v"
    FLF2V = "flf2v"
    V2V = "v2v"
    EXTEND = "extend"
    REF = "ref"
    AUDIO = "audio"
    MULTISHOT = "multishot"


class Continuity(str, Enum):
    NATIVE_EXTENSION = "native_extension"
    OVERLAP_CONTEXT = "overlap_context"
    LAST_FRAME_I2V = "last_frame_i2v"
    KEYFRAME_FILL = "keyframe_fill"


class AudioCapability(str, Enum):
    NONE = "none"
    AMBIENT = "ambient"
    SPEECH = "speech"


class Resolution(BaseModel):
    model_config = ConfigDict(frozen=True)

    width: int = Field(gt=0)
    height: int = Field(gt=0)

    @classmethod
    def parse(cls, value: str | Resolution | dict | tuple) -> Resolution:
        if isinstance(value, Resolution):
            return value
        if isinstance(value, dict):
            return cls(**value)
        if isinstance(value, tuple):
            return cls(width=value[0], height=value[1])
        w, sep, h = str(value).lower().partition("x")
        if not sep:
            raise ValueError(f"resolution must look like 1280x720, got {value!r}")
        return cls(width=int(w), height=int(h))

    @property
    def aspect(self) -> float:
        return self.width / self.height

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"


class FrameRange(BaseModel):
    min: int = Field(ge=1)
    default: int = Field(ge=1)
    max: int = Field(ge=1)
    recommended: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> FrameRange:
        if not self.min <= self.default <= self.max:
            raise ValueError("frames must satisfy min <= default <= max")
        if self.recommended is not None and not self.min <= self.recommended <= self.max:
            raise ValueError("frames.recommended must lie between min and max")
        return self

    @property
    def segment(self) -> int:
        """Longest segment the planner makes by default: the trained length, not the hard max."""
        return self.recommended or self.max


class I2VPrompt(str, Enum):
    MOTION = "motion"
    DESCRIBE_FIRST = "describe_first"


class PromptRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encoder: str = "unknown"
    tokenizer: str | None = None
    max_tokens: int = Field(default=512, gt=0)
    tokens_per_word: float = Field(default=1.35, gt=0)
    t2v_words: tuple[int, int] = (60, 150)
    i2v_words: tuple[int, int] = (20, 100)
    i2v: I2VPrompt = I2VPrompt.MOTION
    negative: str | None = None
    anime_negative: str | None = None
    uses_negative: bool = True


class VramPreset(BaseModel):
    min_gb: float = Field(gt=0)
    recipe: str


class ComfyFile(BaseModel):
    repo: str
    file: str
    folder: str


class Backends(BaseModel):
    diffusers: str | None = None
    diffusers_repo: str | None = None
    diffusers_i2v: str | None = None
    diffusers_repo_i2v: str | None = None
    diffusers_condition: str | None = None
    comfyui: str | None = None
    comfyui_files: list[ComfyFile] = Field(default_factory=list)
    hosted: str | None = None


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    family: str
    variant: str
    base: str | None = None
    params: str | None = None
    temporal_factor: int = Field(ge=1)
    spatial_multiple: int = Field(ge=1)
    native_fps: float = Field(gt=0)
    frames: FrameRange
    resolution_buckets: list[Resolution] = Field(default_factory=list)
    buckets_only: bool = False
    pixel_budget: int = Field(gt=0)
    conditioning: list[Conditioning]
    continuity: list[Continuity] = Field(default_factory=list)
    audio: AudioCapability = AudioCapability.NONE
    min_vram_gb: float = Field(gt=0)
    vram_presets: list[VramPreset] = Field(default_factory=list)
    backends: Backends = Field(default_factory=Backends)
    prompt: PromptRules = Field(default_factory=PromptRules)
    call_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    handoff_skip: int | None = Field(default=None, ge=0)
    license: str
    sources: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("resolution_buckets", mode="before")
    @classmethod
    def _parse_buckets(cls, value):
        return [Resolution.parse(v) for v in value or []]

    @model_validator(mode="after")
    def _consistent(self) -> ModelProfile:
        for name in ("min", "default", "max"):
            n = getattr(self.frames, name)
            if not self.is_legal_frames(n):
                raise ValueError(
                    f"{self.id}: frames.{name}={n} is not of the form {self.temporal_factor}n+1"
                )
        for b in self.resolution_buckets:
            if b.width % self.spatial_multiple or b.height % self.spatial_multiple:
                raise ValueError(
                    f"{self.id}: bucket {b} is not a multiple of {self.spatial_multiple}"
                )
        if self.buckets_only and not self.resolution_buckets:
            raise ValueError(f"{self.id}: buckets_only requires resolution_buckets")
        if self.frames.recommended is not None and not self.is_legal_frames(
            self.frames.recommended
        ):
            raise ValueError(
                f"{self.id}: frames.recommended={self.frames.recommended} is not of the form "
                f"{self.temporal_factor}n+1"
            )
        self.vram_presets.sort(key=lambda p: p.min_gb, reverse=True)
        return self

    @property
    def skip_last(self) -> int:
        """Frames at the end of a clip not used as a hand-off frame: the last latent group."""
        return self.temporal_factor if self.handoff_skip is None else self.handoff_skip

    def is_legal_frames(self, n: int) -> bool:
        return n >= 1 and (n - 1) % self.temporal_factor == 0

    def is_legal_size(self, width: int, height: int) -> bool:
        m = self.spatial_multiple
        if width % m or height % m:
            return False
        if self.buckets_only:
            return Resolution(width=width, height=height) in self.resolution_buckets
        return True

    def preset_for(self, vram_gb: float) -> VramPreset | None:
        for preset in self.vram_presets:
            if vram_gb >= preset.min_gb:
                return preset
        return None

    def supports(self, conditioning: Conditioning | str) -> bool:
        return Conditioning(conditioning) in self.conditioning
