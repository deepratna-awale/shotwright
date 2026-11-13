from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..plan import Job
from ..profiles import ModelProfile, VramPreset


class BackendError(RuntimeError):
    pass


class OutOfMemory(BackendError):
    pass


class TransientError(BackendError):
    pass


@dataclass
class JobInputs:
    init_image: np.ndarray | None = None
    init_image_path: Path | None = None
    end_image: np.ndarray | None = None
    end_image_path: Path | None = None


@dataclass
class ClipResult:
    frames: np.ndarray
    fps: float
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    name: str

    def supports(self, profile: ModelProfile) -> bool: ...

    def prepare(self, profile: ModelProfile, preset: VramPreset | None) -> None: ...

    def run(self, job: Job, inputs: JobInputs) -> ClipResult: ...

    def release(self) -> None: ...
