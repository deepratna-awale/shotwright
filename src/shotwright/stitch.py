from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .media import VideoWriter, read_video, resize_crop
from .run.rundir import RunDir

METHODS = ("resample", "blend")


class StitchError(RuntimeError):
    pass


@dataclass
class Seam:
    before: str
    after: str
    kind: str
    mean_abs_diff: float


@dataclass
class StitchResult:
    path: Path
    frames: int
    fps: float
    seams: list[Seam] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.frames / self.fps


def stitch(
    run_dir: str | Path,
    out: str | Path | None = None,
    fps: float | None = None,
    method: str = "resample",
    crf: int = 16,
) -> StitchResult:
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    run = RunDir(run_dir)
    if not run.exists():
        raise StitchError(f"no plan.json in {run.root}")
    plan = run.load_plan()

    missing = [
        j.id for j in plan.jobs if not run.clip_path(j.id).is_file() or run.read_meta(j.id) is None
    ]
    if missing:
        raise StitchError(
            f"{len(missing)} job(s) have not been rendered: {', '.join(missing)}. "
            "Run `shotwright resume` first."
        )

    out_path = Path(out) if out else run.root / "final.mp4"
    out_fps = float(fps or plan.output_fps)
    step = 1.0 / out_fps
    seams: list[Seam] = []
    previous: tuple[str, np.ndarray] | None = None

    with VideoWriter(out_path, plan.output_width, plan.output_height, out_fps, crf) as writer:
        start = 0.0
        k = 0
        for job in plan.jobs:
            frames, _ = read_video(run.clip_path(job.id))
            chained = job.depends_on is not None
            if chained:
                frames = frames[1:]
            if len(frames) == 0:
                continue
            frames = np.stack([resize_crop(f, job.resize_to, job.crop) for f in frames])
            if frames.shape[1:3] != (plan.output_height, plan.output_width):
                raise StitchError(
                    f"{job.id}: frames are {frames.shape[2]}x{frames.shape[1]} after resize, "
                    f"expected {plan.output_width}x{plan.output_height}"
                )

            if previous is not None:
                diff = np.abs(previous[1].astype(np.int16) - frames[0].astype(np.int16)).mean()
                seams.append(
                    Seam(previous[0], job.id, "chain" if chained else "cut", round(float(diff), 3))
                )

            n = len(frames)
            end = start + n / job.fps
            while k * step < end - 1e-9:
                pos = (k * step - start) * job.fps
                i = min(max(math.floor(pos + 1e-9), 0), n - 1)
                frac = pos - i
                if method == "blend" and i + 1 < n and frac > 1e-6:
                    frame = (frames[i] * (1 - frac) + frames[i + 1] * frac).round().astype(np.uint8)
                else:
                    frame = frames[i]
                writer.write(frame)
                k += 1
            start = end
            previous = (job.id, frames[-1])
        written = writer.frames_written

    result = StitchResult(path=out_path, frames=written, fps=out_fps, seams=seams)
    report = {
        "output": str(out_path),
        "frames": written,
        "fps": out_fps,
        "duration": round(result.duration, 4),
        "method": method,
        "seams": [asdict(s) for s in seams],
    }
    (run.root / "stitch.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    run.log("stitched", output=str(out_path), frames=written, fps=out_fps)
    return result
