from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..profiles import Continuity


class InitKind(str, Enum):
    IMAGE = "image"
    LAST_FRAME = "last_frame"
    OVERLAP = "overlap"


class JobInit(BaseModel):
    kind: InitKind
    path: str | None = None
    job: str | None = None
    strategy: Continuity | None = None
    frames: int = 1


class Job(BaseModel):
    id: str
    shot_id: str
    segment: int
    segment_count: int
    model: str
    prompt: str
    negative: str | None = None
    seed: int
    frames: int
    width: int
    height: int
    fps: float
    init: JobInit | None = None
    end_image: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    resize_to: tuple[int, int] | None = None
    crop: tuple[int, int, int, int] | None = None

    @property
    def depends_on(self) -> str | None:
        if self.init is not None and self.init.kind in (InitKind.LAST_FRAME, InitKind.OVERLAP):
            return self.init.job
        return None

    @property
    def shared_frames(self) -> int:
        """Leading frames that repeat the end of the job this one continues from."""
        return self.init.frames if self.depends_on is not None else 0


class PlannedLine(BaseModel):
    shot_id: str
    text: str
    speaker: str | None = None
    voice: str | None = None
    at: float = 0.0


class Plan(BaseModel):
    version: int = 1
    project: str
    output_width: int
    output_height: int
    output_fps: float
    jobs: list[Job]
    lines: list[PlannedLine] = Field(default_factory=list)

    def job(self, job_id: str) -> Job:
        for job in self.jobs:
            if job.id == job_id:
                return job
        raise KeyError(job_id)

    def dependants(self, job_id: str) -> list[str]:
        out: list[str] = []
        frontier = [job_id]
        while frontier:
            current = frontier.pop()
            for job in self.jobs:
                if job.depends_on == current:
                    out.append(job.id)
                    frontier.append(job.id)
        return out

    def chains(self) -> list[list[str]]:
        chains: list[list[str]] = []
        for job in self.jobs:
            if job.depends_on is None:
                chains.append([job.id])
            else:
                for chain in chains:
                    if chain[-1] == job.depends_on:
                        chain.append(job.id)
                        break
        return chains

    @property
    def total_frames(self) -> int:
        return sum(j.frames - j.shared_frames for j in self.jobs)

    def summary(self) -> str:
        lines = [
            f"project   {self.project}",
            f"output    {self.output_width}x{self.output_height} at {self.output_fps:g} fps",
            f"jobs      {len(self.jobs)} in {len(self.chains())} independent chain(s)",
            "",
        ]
        width = max(len(j.id) for j in self.jobs)
        for job in self.jobs:
            link = f"<- {job.init.job}" if job.depends_on else ("init image" if job.init else "cut")
            lines.append(
                f"  {job.id.ljust(width)}  {job.model}  {job.frames}f @ {job.fps:g}  "
                f"{job.width}x{job.height}  seed {job.seed}  {link}"
            )
        seconds = sum((j.frames - j.shared_frames) / j.fps for j in self.jobs)
        lines.append("")
        lines.append(f"duration  about {seconds:.1f} s before resampling")
        return "\n".join(lines)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Plan:
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
