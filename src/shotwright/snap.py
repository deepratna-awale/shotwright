from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from .profiles import ModelProfile, Resolution, get_profile

ASPECT_TOLERANCE = 0.02


@dataclass(frozen=True)
class Adjustment:
    field: str
    requested: Any
    chosen: Any
    reason: str

    def __str__(self) -> str:
        return f"{self.field}: {_fmt(self.requested)} -> {_fmt(self.chosen)} ({self.reason})"


def _fmt(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


@dataclass(frozen=True)
class Crop:
    left: int
    top: int
    right: int
    bottom: int

    def __str__(self) -> str:
        return f"left {self.left}, top {self.top}, right {self.right}, bottom {self.bottom}"


@dataclass(frozen=True)
class Postprocess:
    resize_to: Resolution | None = None
    crop: Crop | None = None
    fps_from: float | None = None
    fps_to: float | None = None
    fps_method: str | None = None

    @property
    def is_noop(self) -> bool:
        return self.resize_to is None and self.crop is None and self.fps_to is None


@dataclass(frozen=True)
class SnapResult:
    model: str
    width: int
    height: int
    fps: float
    segments: tuple[int, ...]
    overlap: int
    output_width: int
    output_height: int
    output_fps: float
    adjustments: tuple[Adjustment, ...] = field(default_factory=tuple)
    postprocess: Postprocess = field(default_factory=Postprocess)

    @property
    def frames(self) -> int:
        return sum(self.segments) - self.overlap * (len(self.segments) - 1)

    @property
    def duration(self) -> float:
        return self.frames / self.fps

    @property
    def is_split(self) -> bool:
        return len(self.segments) > 1

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["frames"] = self.frames
        data["duration"] = round(self.duration, 4)
        data["segments"] = list(self.segments)
        return data


def legal_frames_at_least(n: int, temporal_factor: int) -> int:
    n = max(n, 1)
    return 1 + temporal_factor * math.ceil((n - 1) / temporal_factor)


def legal_frames_at_most(n: int, temporal_factor: int) -> int:
    n = max(n, 1)
    return 1 + temporal_factor * ((n - 1) // temporal_factor)


def parse_aspect(value: str | float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    a, sep, b = str(value).partition(":")
    return float(a) / float(b) if sep else float(a)


def _ceil_to(value: float, multiple: int) -> int:
    return max(multiple, multiple * math.ceil(value / multiple - 1e-9))


def _round_to(value: float, multiple: int) -> int:
    return max(multiple, multiple * round(value / multiple))


def _aspect_close(a: float, b: float, tolerance: float = ASPECT_TOLERANCE) -> bool:
    return abs(math.log(a / b)) <= math.log1p(tolerance)


def _fit_budget(aspect: float, profile: ModelProfile) -> Resolution:
    m = profile.spatial_multiple
    height = math.sqrt(profile.pixel_budget / aspect)
    w, h = _round_to(height * aspect, m), _round_to(height, m)
    while w * h > profile.pixel_budget and min(w, h) > m:
        if w >= h:
            w -= m
        else:
            h -= m
    return Resolution(width=w, height=h)


def _closest_bucket(target: Resolution, profile: ModelProfile) -> Resolution | None:
    matching = [b for b in profile.resolution_buckets if _aspect_close(b.aspect, target.aspect)]
    if matching:
        return min(matching, key=lambda b: abs(math.log(b.pixels / target.pixels)))
    if profile.buckets_only:
        return min(
            profile.resolution_buckets,
            key=lambda b: (
                abs(math.log(b.aspect / target.aspect)),
                abs(math.log(b.pixels / target.pixels)),
            ),
        )
    return None


def _snap_size(profile: ModelProfile, requested: Resolution) -> tuple[Resolution, str | None]:
    m = profile.spatial_multiple
    if profile.is_legal_size(requested.width, requested.height) and (
        requested.pixels <= profile.pixel_budget
    ):
        return requested, None

    bucket = _closest_bucket(requested, profile)
    if bucket is not None:
        return bucket, "nearest trained resolution"

    if requested.pixels <= profile.pixel_budget:
        size = Resolution(width=_ceil_to(requested.width, m), height=_ceil_to(requested.height, m))
        if size.pixels <= profile.pixel_budget:
            return size, f"sides must be multiples of {m}"
    return _fit_budget(requested.aspect, profile), (
        f"scaled to the {profile.pixel_budget:,} pixel budget, sides multiples of {m}"
    )


def _postprocess_size(
    generated: Resolution, wanted: Resolution
) -> tuple[Resolution | None, Crop | None]:
    if generated == wanted:
        return None, None
    scale = max(wanted.width / generated.width, wanted.height / generated.height)
    resized = Resolution(
        width=max(wanted.width, round(generated.width * scale)),
        height=max(wanted.height, round(generated.height * scale)),
    )
    dx, dy = resized.width - wanted.width, resized.height - wanted.height
    crop = Crop(dx // 2, dy // 2, dx - dx // 2, dy - dy // 2) if dx or dy else None
    return (resized if resized != generated else None), crop


def _split(want: int, profile: ModelProfile, overlap: int) -> tuple[int, ...]:
    k = profile.temporal_factor
    longest = profile.frames.max
    if overlap >= longest:
        raise ValueError(f"overlap {overlap} must be shorter than the {longest}-frame clip limit")
    if want <= longest:
        return (max(legal_frames_at_least(want, k), profile.frames.min),)
    count = 1 + math.ceil((want - longest) / (longest - overlap))
    while True:
        per = legal_frames_at_least(math.ceil((want + (count - 1) * overlap) / count), k)
        per = max(min(per, longest), profile.frames.min)
        if count * per - (count - 1) * overlap >= want:
            return (per,) * count
        count += 1


def snap(
    model: str | ModelProfile,
    *,
    width: int | None = None,
    height: int | None = None,
    size: str | None = None,
    aspect: str | float | None = None,
    duration: float | None = None,
    frames: int | None = None,
    fps: float | None = None,
    overlap: int = 0,
) -> SnapResult:
    profile = model if isinstance(model, ModelProfile) else get_profile(model)
    adjustments: list[Adjustment] = []

    if size is not None:
        if width is not None or height is not None:
            raise ValueError("pass either size or width/height, not both")
        parsed = Resolution.parse(size)
        width, height = parsed.width, parsed.height
    if (width is None) != (height is None):
        raise ValueError("width and height must be given together")
    if duration is not None and frames is not None:
        raise ValueError("pass either duration or frames, not both")
    if overlap < 0:
        raise ValueError("overlap must be zero or positive")

    native = profile.native_fps
    output_fps = float(fps) if fps else native

    if frames is None and duration is None:
        want = profile.frames.default
    elif frames is not None:
        want = int(frames)
        if output_fps != native:
            want = math.ceil(want * native / output_fps - 1e-9)
    else:
        want = math.ceil(duration * native - 1e-9)
    want = max(want, 1)

    segments = _split(want, profile, overlap)
    total = sum(segments) - overlap * (len(segments) - 1)
    if len(segments) > 1:
        adjustments.append(
            Adjustment(
                "frames",
                want,
                f"{len(segments)} x {segments[0]}",
                f"longer than the {profile.frames.max}-frame clip limit; split into segments"
                + (
                    f" with {overlap} overlapping frame{'s' if overlap > 1 else ''}"
                    if overlap
                    else ""
                ),
            )
        )
    elif total != want:
        reason = (
            f"raised to the {profile.frames.min}-frame minimum"
            if total == profile.frames.min and want < profile.frames.min
            else f"frame count must be {profile.temporal_factor}n+1"
        )
        adjustments.append(Adjustment("frames", want, total, reason))

    if width is not None:
        requested = Resolution(width=width, height=height)
        generated, reason = _snap_size(profile, requested)
        if reason:
            adjustments.append(Adjustment("size", str(requested), str(generated), reason))
        resize_to, crop = _postprocess_size(generated, requested)
        output = requested
    else:
        if aspect is not None:
            target = parse_aspect(aspect)
            bucket = _closest_bucket(Resolution(width=round(1000 * target), height=1000), profile)
            generated = bucket or _fit_budget(target, profile)
        elif profile.resolution_buckets:
            generated = profile.resolution_buckets[0]
        else:
            generated = _fit_budget(16 / 9, profile)
        resize_to, crop = None, None
        output = generated

    fps_method = None
    if output_fps != native:
        fps_method = "interpolate" if output_fps > native else "resample"
        adjustments.append(
            Adjustment(
                "fps",
                output_fps,
                native,
                f"generated at the model's native {native:g} fps, "
                f"then {fps_method}d to {output_fps:g}",
            )
        )

    if resize_to or crop:
        detail = []
        if resize_to:
            detail.append(f"resize to {resize_to}")
        if crop:
            detail.append(f"crop {crop}")
        adjustments.append(Adjustment("output", str(generated), str(output), "; ".join(detail)))

    return SnapResult(
        model=profile.id,
        width=generated.width,
        height=generated.height,
        fps=native,
        segments=segments,
        overlap=overlap if len(segments) > 1 else 0,
        output_width=output.width,
        output_height=output.height,
        output_fps=output_fps,
        adjustments=tuple(adjustments),
        postprocess=Postprocess(
            resize_to=resize_to,
            crop=crop,
            fps_from=native if fps_method else None,
            fps_to=output_fps if fps_method else None,
            fps_method=fps_method,
        ),
    )
