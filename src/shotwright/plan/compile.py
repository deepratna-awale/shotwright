from __future__ import annotations

import hashlib
from pathlib import Path

from ..policy import violation
from ..profiles import (
    Conditioning,
    Continuity,
    ModelProfile,
    Registry,
    UnknownModelError,
    default_registry,
)
from ..snap import snap
from .model import FrameRefine, InitKind, Job, JobInit, Plan, PlannedLine
from .spec import Look, ShotList, ShotSpec, Transition

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


def _end_of_segment(shot: ShotSpec, index: int, count: int) -> tuple[str | None, str | None]:
    """What the last frame of segment `index` shows, if the shot says so."""
    if index == count - 1 and (shot.end_prompt or shot.end_tags):
        return shot.end_prompt, shot.end_tags
    if index < len(shot.beats):
        beat = shot.beats[index]
        return beat.end, beat.end_tags
    return None, None


def _handoff_skip(spec: ShotList, registry: Registry, parent: Job) -> int:
    if spec.continuity.handoff_skip is not None:
        return spec.continuity.handoff_skip
    return registry.get(parent.model).skip_last


def _still_description(shot: ShotSpec, style: str | None) -> str:
    """A still-image prompt for a frame of `shot` when nobody described its end."""
    parts = shot.parts
    pieces = (
        [parts.subject, parts.setting, parts.lighting, parts.style or style]
        if parts is not None
        else []
    )
    text = " ".join(_sentence(p) for p in pieces if p)
    return text or _prompt(shot.prompt, style)


def _sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _refine_for(
    spec: ShotList,
    parent: Job,
    parent_shot: ShotSpec,
    shot: ShotSpec,
    seed: int,
    problems: list[str],
) -> FrameRefine | None:
    settings = spec.continuity.refine
    wanted = settings.enabled if shot.refine is None else shot.refine
    if not wanted:
        return None
    from ..refine import LOOK_DEFAULTS, REFINERS

    model = settings.model or LOOK_DEFAULTS[spec.look.value]
    if model not in REFINERS:
        problems.append(
            f"shot {shot.id!r}: unknown refiner {model!r}; known: {', '.join(sorted(REFINERS))}"
        )
        return None
    if parent.end_prompt or parent.end_tags:
        prompt = parent.end_prompt or parent.end_tags
        source = "end_prompt"
    else:
        prompt = _still_description(parent_shot, spec.style)
        source = "shot"
    return FrameRefine(
        model=model,
        prompt=prompt,
        tags=parent.end_tags,
        strength=settings.strength,
        steps=settings.steps,
        guidance=settings.guidance,
        seed=seed,
        color_match=settings.color_match,
        prompt_source=source,
    )


def _job_prompt(
    profile: ModelProfile,
    spec: ShotList,
    shot: ShotSpec,
    index: int,
    image_start: bool,
    start: str | None,
):
    from ..prompting import adapt_prompt

    beat = shot.beat(index)
    mode = "i2v" if image_start else "t2v"
    motion = beat.action if beat is not None and mode == "i2v" else None
    prompt = shot.prompt
    if beat is not None and mode == "t2v":
        prompt = f"{_sentence(prompt)} {_sentence(beat.action)}" if prompt.strip() else beat.action
    return mode, adapt_prompt(
        profile,
        mode=mode,
        prompt=prompt,
        parts=shot.parts,
        motion=motion,
        start=start,
        style=spec.style,
        look=spec.look if isinstance(spec.look, Look) else Look(spec.look),
        negative=shot.negative if shot.negative is not None else spec.defaults.negative,
    )


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
        texts = [
            shot.prompt,
            shot.end_prompt or "",
            shot.end_tags or "",
            *(v for v in (shot.parts.model_dump().values() if shot.parts else ()) if v),
            *(t for b in shot.beats for t in (b.action, b.end or "", b.end_tags or "")),
        ]
        reason = violation(" ".join(t for t in texts if t), spec.style)
        if reason:
            problems.append(f"shot {shot.id!r} is not allowed: {reason}")
    if spec.shots[0].transition == Transition.CONTINUOUS:
        problems.append(f"first shot {spec.shots[0].id!r} cannot be continuous")

    out = spec.output
    previous_job: Job | None = None
    previous_shot: ShotSpec | None = None
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
        shared = (
            spec.continuity.overlap
            if spec.continuity.overlap > 1 and Continuity.OVERLAP_CONTEXT in profile.continuity
            else 1
        )
        if continuous:
            seconds += shared / native

        try:
            result = snap(
                profile,
                width=out.width,
                height=out.height,
                duration=seconds,
                fps=out.fps,
                overlap=shared,
                segment_frames=spec.continuity.segment_frames,
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
        elif not continuous and not profile.supports(Conditioning.T2V):
            problems.append(
                f"shot {shot.id!r} starts from nothing but {profile.id} needs an init_image"
            )

        end_image = None
        if shot.end_image:
            if not profile.supports(Conditioning.FLF2V):
                problems.append(
                    f"shot {shot.id!r} has an end_image but {profile.id} cannot end on a "
                    f"given frame; use a first-last-frame model such as wan2.1-flf2v-14b-720p"
                )
            elif not (continuous or init_image or result.is_split):
                problems.append(
                    f"shot {shot.id!r} has an end_image but nothing to start from; "
                    f"add an init_image or make it continuous"
                )
            end_image = spec.resolve_path(shot.end_image)
            if check_files and not Path(end_image).is_file():
                problems.append(f"shot {shot.id!r}: end_image not found: {end_image}")

        resize = result.postprocess.resize_to
        crop = result.postprocess.crop
        count = len(result.segments)
        for index, frames in enumerate(result.segments):
            job_id = shot.id if count == 1 else f"{shot.id}.{index + 1:02d}"
            seed = (
                shot.seed + index
                if shot.seed is not None
                else derive_seed(spec.defaults.seed, shot.id, index)
            )
            parent_job: Job | None = None
            if index > 0 or continuous:
                parent_job = jobs[-1] if index > 0 else previous_job
                parent = parent_job.id
                init = (
                    JobInit(
                        kind=InitKind.OVERLAP,
                        job=parent,
                        strategy=Continuity.OVERLAP_CONTEXT,
                        frames=shared,
                    )
                    if shared > 1
                    else JobInit(
                        kind=InitKind.LAST_FRAME,
                        job=parent,
                        strategy=Continuity.LAST_FRAME_I2V,
                        skip=_handoff_skip(spec, registry, parent_job),
                        window=spec.continuity.handoff_window,
                        refine=_refine_for(
                            spec,
                            parent_job,
                            shot if index > 0 else previous_shot,
                            shot,
                            seed,
                            problems,
                        ),
                    )
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
            start_text = None
            if parent_job is not None:
                start_text = parent_job.end_prompt or (
                    init.refine.prompt if init.refine is not None else None
                )
            mode, adapted = _job_prompt(
                profile, spec, shot, index, init is not None, start_text
            )
            end_prompt, end_tags = _end_of_segment(shot, index, count)
            jobs.append(
                Job(
                    id=job_id,
                    shot_id=shot.id,
                    segment=index,
                    segment_count=count,
                    model=profile.id,
                    prompt=adapted.text,
                    negative=adapted.negative,
                    seed=seed,
                    frames=frames,
                    width=result.width,
                    height=result.height,
                    fps=native,
                    init=init,
                    end_image=end_image if index == count - 1 else None,
                    params={
                        **profile.call_params.get(mode, {}),
                        **spec.defaults.params,
                        **shot.params,
                    },
                    end_prompt=end_prompt,
                    end_tags=end_tags,
                    prompt_tokens=adapted.tokens,
                    prompt_notes=list(adapted.notes),
                    resize_to=(resize.width, resize.height) if resize else None,
                    crop=(crop.left, crop.top, crop.right, crop.bottom) if crop else None,
                )
            )
        previous_job = jobs[-1]
        previous_shot = shot
        if end_image:
            hops = 0

    lines = []
    for shot in spec.shots:
        for line in shot.lines:
            reason = violation(line.text)
            if reason:
                problems.append(f"a line in shot {shot.id!r} is not allowed: {reason}")
            voice = spec.voices.get(line.speaker or "", spec.voices.get("default"))
            lines.append(
                PlannedLine(
                    shot_id=shot.id, text=line.text, speaker=line.speaker, voice=voice, at=line.at
                )
            )

    if problems:
        raise PlanError(problems)
    return Plan(
        lines=lines,
        project=spec.project,
        output_width=out.width,
        output_height=out.height,
        output_fps=out.fps,
        jobs=jobs,
    )
