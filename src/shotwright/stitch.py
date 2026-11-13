from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .color import LabStats, match_to
from .media import (
    SAMPLE_RATE,
    VideoWriter,
    mux,
    read_video,
    read_wav,
    resample,
    resize_crop,
    wav_bytes,
)
from .run.rundir import RunDir

METHODS = ("resample", "blend")
COLOR_MODES = ("none", "seam", "anchor")


class StitchError(RuntimeError):
    pass


@dataclass
class Seam:
    before: str
    after: str
    kind: str
    mean_abs_diff: float
    color_delta: float = 0.0
    jump: float = 0.0
    flagged: bool = False


JUMP_LIMIT = 2.5


def _motion(frames: np.ndarray) -> float:
    if len(frames) < 2:
        return 0.0
    sample = frames[:, ::2, ::2].astype(np.int16)
    return float(np.abs(np.diff(sample, axis=0)).mean())


def _seam(before: str, last: np.ndarray, after: str, frames: np.ndarray, chained: bool) -> Seam:
    diff = float(np.abs(last.astype(np.int16) - frames[0].astype(np.int16)).mean())
    delta = float(np.linalg.norm(LabStats.of(last).mean - LabStats.of(frames[0]).mean))
    jump = diff / max(_motion(frames), 0.5)
    return Seam(
        before,
        after,
        "chain" if chained else "cut",
        round(diff, 3),
        round(delta, 3),
        round(jump, 3),
        chained and jump > JUMP_LIMIT,
    )


@dataclass
class StitchResult:
    path: Path
    frames: int
    fps: float
    seams: list[Seam] = field(default_factory=list)
    timeline: list[dict] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.frames / self.fps


def stitch(
    run_dir: str | Path,
    out: str | Path | None = None,
    fps: float | None = None,
    method: str = "resample",
    crf: int = 16,
    color: str = "seam",
    anchor_strength: float = 0.5,
) -> StitchResult:
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    if color not in COLOR_MODES:
        raise ValueError(f"color must be one of {', '.join(COLOR_MODES)}")
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
    timeline: list[dict] = []
    sound: list[tuple[float, np.ndarray]] = []
    previous: tuple[str, np.ndarray] | None = None
    root: LabStats | None = None

    with VideoWriter(out_path, plan.output_width, plan.output_height, out_fps, crf) as writer:
        start = 0.0
        k = 0
        for job in plan.jobs:
            frames, _ = read_video(run.clip_path(job.id))
            frames = np.stack([resize_crop(f, job.resize_to, job.crop) for f in frames])
            if frames.shape[1:3] != (plan.output_height, plan.output_width):
                raise StitchError(
                    f"{job.id}: frames are {frames.shape[2]}x{frames.shape[1]} after resize, "
                    f"expected {plan.output_width}x{plan.output_height}"
                )
            chained = job.depends_on is not None and previous is not None
            if not chained:
                root = LabStats.of(frames) if color == "anchor" else None
            elif color != "none":
                frames = match_to(frames, LabStats.of(frames[0]), LabStats.of(previous[1]))
                if root is not None:
                    frames = match_to(frames, LabStats.of(frames), root, ramp=anchor_strength)
            if job.depends_on is not None:
                frames = frames[1:]
            if len(frames) == 0:
                continue

            if previous is not None:
                seams.append(_seam(previous[0], previous[1], job.id, frames, chained))

            n = len(frames)
            end = start + n / job.fps
            piece = _job_audio(run, job, n)
            if piece is not None:
                sound.append((start, piece))
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
            timeline.append(
                {"job": job.id, "shot": job.shot_id, "start": round(start, 4), "end": round(end, 4)}
            )
            start = end
            previous = (job.id, frames[-1])
        written = writer.frames_written

    has_audio = bool(sound)
    if has_audio:
        track = np.zeros(int(round(written / out_fps * SAMPLE_RATE)), dtype=np.float32)
        for begin, piece in sound:
            first = int(round(begin * SAMPLE_RATE))
            stop = min(first + len(piece), len(track))
            if stop > first:
                track[first:stop] = piece[: stop - first]
        audio_path = run.root / "native_audio.wav"
        audio_path.write_bytes(wav_bytes(track, SAMPLE_RATE))
        muxed = out_path.with_name(f".{out_path.stem}.muxing{out_path.suffix}")
        mux(out_path, audio_path, muxed)
        muxed.replace(out_path)

    result = StitchResult(
        path=out_path, frames=written, fps=out_fps, seams=seams, timeline=timeline
    )
    report = {
        "output": str(out_path),
        "frames": written,
        "fps": out_fps,
        "duration": round(result.duration, 4),
        "method": method,
        "color": color,
        "audio": has_audio,
        "seams": [asdict(s) for s in seams],
        "timeline": timeline,
    }
    (run.root / "stitch.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    run.log("stitched", output=str(out_path), frames=written, fps=out_fps)
    return result


def _job_audio(run: RunDir, job, frames_kept: int) -> np.ndarray | None:
    """The job's own audio, minus the shared first frame of a chain, fitted to its frames."""
    path = run.audio_path(job.id)
    if not path.is_file():
        return None
    samples, rate = read_wav(path.read_bytes())
    samples = resample(samples, rate)
    if job.depends_on is not None:
        samples = samples[int(round(SAMPLE_RATE / job.fps)) :]
    length = int(round(frames_kept / job.fps * SAMPLE_RATE))
    if len(samples) < length:
        samples = np.pad(samples, (0, length - len(samples)))
    return samples[:length]
