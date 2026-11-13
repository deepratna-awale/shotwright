from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..plan import Job
from ..profiles import ModelProfile, VramPreset
from .backend import BackendError, ClipResult, JobInputs, OutOfMemory, TransientError


class FakeBackend:
    """Deterministic synthetic frames for tests and dry runs.

    With an init image the clip starts exactly on it and drifts toward a
    seed-dependent colour, which mimics last-frame image-to-video chaining.
    """

    name = "fake"

    def __init__(
        self,
        fail: Iterable[str] = (),
        oom_until_recipe: str | None = None,
        transient: dict[str, int] | None = None,
        audio: bool = False,
    ):
        self.fail = set(fail)
        self.oom_until_recipe = oom_until_recipe
        self.transient = dict(transient or {})
        self.audio = audio
        self.loaded: tuple[str, str | None] | None = None
        self.loads: list[tuple[str, str | None]] = []
        self.calls: list[str] = []

    def supports(self, profile: ModelProfile) -> bool:
        return True

    def prepare(self, profile: ModelProfile, preset: VramPreset | None) -> None:
        key = (profile.id, preset.recipe if preset else None)
        if self.loaded != key:
            self.loaded = key
            self.loads.append(key)

    def run(self, job: Job, inputs: JobInputs) -> ClipResult:
        self.calls.append(job.id)
        if job.id in self.fail:
            raise BackendError(f"configured to fail on {job.id}")
        if self.transient.get(job.id, 0) > 0:
            self.transient[job.id] -= 1
            raise TransientError(f"temporary failure on {job.id}")
        if self.oom_until_recipe and self.loaded and self.loaded[1] != self.oom_until_recipe:
            raise OutOfMemory(f"not enough memory with recipe {self.loaded[1]}")

        rng = np.random.default_rng(job.seed)
        target = rng.integers(0, 256, size=3).astype(np.float32)
        h, w = job.height, job.width
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        texture = ((xx / max(w - 1, 1)) * 40 + (yy / max(h - 1, 1)) * 20)[..., None]

        if inputs.init_image is not None:
            start = inputs.init_image.astype(np.float32)
        else:
            start = np.clip(target * 0.5 + texture, 0, 255)

        end = (
            inputs.end_image.astype(np.float32)
            if inputs.end_image is not None
            else np.clip(target + texture, 0, 255)
        )
        frames = np.empty((job.frames, h, w, 3), dtype=np.uint8)
        for i in range(job.frames):
            t = i / max(job.frames - 1, 1)
            frames[i] = np.clip(start * (1 - t) + end * t, 0, 255).astype(np.uint8)
        sound = None
        if self.audio:
            rate = 16000
            t = np.arange(int(job.frames / job.fps * rate)) / rate
            sound = (0.2 * np.sin(2 * np.pi * (200 + job.seed % 300) * t)).astype(np.float32)
        return ClipResult(
            frames=frames,
            fps=job.fps,
            meta={"backend": self.name},
            audio=sound,
            audio_rate=16000 if sound is not None else None,
        )

    def release(self) -> None:
        self.loaded = None
