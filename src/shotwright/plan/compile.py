from __future__ import annotations

import hashlib
from pathlib import Path

from ..profiles import Conditioning, Continuity, Registry, UnknownModelError, default_registry
from ..snap import snap
from .model import InitKind, Job, JobInit, Plan
from .spec import ShotList, Transition

SEED_MAX = 2**31 - 1


class PlanError(ValueError):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("\n".join(f"- {p}" for p in problems))


def derive_seed(project_seed: int, shot_id: str, segment: int) -> int:
    digest = hashlib.sha256(f"{project_seed}:{shot_id}:{segment}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & SEED_MAX


def _prompt(prompt: str, style: str | None) -> str:
    prompt = prompt.strip()
    if not style:
        return prompt
    sep = "" if prompt.endswith((".", "!", "?")) else "."
    return f"{prompt}{sep} {style.strip()}"


def compile_plan(
    spec: ShotList,
    registry: Registry | None = None,
    check_files: bool = True,
) -> Plan:
    registry = registry or default_registry()
    problems: list[str] = []
    jobs: list[Job] = []

    seen: set[str] = set()
    for shot in spec.shots:
        if shot.id in seen:
            problems.append(f"shot id {shot.id!r} is used more than once")
        seen.add(shot.id)
    if spec.shots[0].transition == Transition.CONTINUOUS:
        problems.append(f"first shot {spec.shots[0].id!r} cannot be continuous")

    out = spec.output
    previous_job: Job | None = None
    hops = 0

    for shot in spec.shots:
        model_id = shot.model or spec.defaults.model
        if not model_id:
            problems.append(f"shot {shot.id!r} has no model and there is no default model")
            previous_job = None
            continue
        try:
            profile = registry.get(model_id)
        except UnknownModelError as exc:
            problems.append(f"shot {shot.id!r}: {exc}")
            previous_job = None
            continue

        continuous = shot.transition == Transition.CONTINUOUS and previous_job is not None
        native = profile.native_fps

        if shot.duration is not None:
            seconds = shot.duration
        elif shot.frames is not None:
            seconds = shot.frames / out.fps
        else:
            seconds = profile.frames.default / native
        if continuous:
            seconds += 1 / native

        try:
            result = snap(
                profile,
                width=out.width,
                height=out.height,
                duration=seconds,
                fps=out.fps,
                overlap=1,
            )
        except ValueError as exc:
            problems.append(f"shot {shot.id!r}: {exc}")
            previous_job = None
            continue

        needs_i2v = continuous or result.is_split
        if needs_i2v and not profile.supports(Conditioning.I2V):
            reason = "is continuous" if continuous else "is split into segments"
            problems.append(
                f"shot {shot.id!r} {reason}, which chains from the previous frame, "
                f"but {profile.id} has no image conditioning; use an i2v model"
            )
        init_image = None
        if shot.init_image:
            if continuous:
                problems.append(
                    f"shot {shot.id!r} is continuous and also has an init_image; pick one"
                )
            elif not profile.supports(Conditioning.I2V):
                problems.append(f"shot {shot.id!r} has an init_image but {profile.id} is text-only")
            init_image = spec.resolve_path(shot.init_image)
            if check_files and not Path(init_image).is_file():
                problems.append(f"shot {shot.id!r}: init_image not found: {init_image}")

        resize = result.postprocess.resize_to
        crop = result.postprocess.crop
        count = len(result.segments)
        for index, frames in enumerate(result.segments):
            job_id = shot.id if count == 1 else f"{shot.id}.{index + 1:02d}"
            if index > 0:
                init = JobInit(
                    kind=InitKind.LAST_FRAME, job=jobs[-1].id, strategy=Continuity.LAST_FRAME_I2V
                )
                hops += 1
            elif continuous:
                init = JobInit(
                    kind=InitKind.LAST_FRAME,
                    job=previous_job.id,
                    strategy=Continuity.LAST_FRAME_I2V,
                )
                hops += 1
            else:
                init = JobInit(kind=InitKind.IMAGE, path=init_image) if init_image else None
                hops = 0
            if hops > spec.continuity.max_hops:
                problems.append(
                    f"job {job_id!r} is hop {hops} of an unbroken last-frame chain; "
                    f"the limit is {spec.continuity.max_hops}. Add a cut, shorten the run, "
                    f"or raise continuity.max_hops"
                )
            jobs.append(
                Job(
                    id=job_id,
                    shot_id=shot.id,
                    segment=index,
                    segment_count=count,
                    model=profile.id,
                    prompt=_prompt(shot.prompt, spec.style),
                    negative=shot.negative if shot.negative is not None else spec.defaults.negative,
                    seed=shot.seed + index
                    if shot.seed is not None
                    else derive_seed(spec.defaults.seed, shot.id, index),
                    frames=frames,
                    width=result.width,
                    height=result.height,
                    fps=native,
                    init=init,
                    params={**spec.defaults.params, **shot.params},
                    resize_to=(resize.width, resize.height) if resize else None,
                    crop=(crop.left, crop.top, crop.right, crop.bottom) if crop else None,
                )
            )
        previous_job = jobs[-1]

    if problems:
        raise PlanError(problems)
    return Plan(
        project=spec.project,
        output_width=out.width,
        output_height=out.height,
        output_fps=out.fps,
        jobs=jobs,
    )
