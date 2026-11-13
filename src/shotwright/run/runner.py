from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..media import file_sha256, load_image, save_image, write_video
from ..plan import InitKind, Job, Plan
from ..profiles import ModelProfile, Registry, VramPreset, default_registry
from .backend import Backend, JobInputs, OutOfMemory, TransientError
from .rundir import RunDir, job_cache_key


class RunError(RuntimeError):
    pass


@dataclass
class RunResult:
    run_dir: Path
    done: list[str] = field(default_factory=list)
    cached: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed and not self.skipped


@dataclass
class Event:
    kind: str
    job_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


Listener = Callable[[Event], None]
PromptHook = Callable[[Job, JobInputs], str]
PROMPT_FILE = "prompt.json"


class Runner:
    def __init__(
        self,
        backend: Backend,
        runs_root: str | Path = "runs",
        registry: Registry | None = None,
        vram_gb: float | None = None,
        transient_retries: int = 2,
        prompt_hook: PromptHook | None = None,
    ):
        self.backend = backend
        self.runs_root = Path(runs_root)
        self.registry = registry or default_registry()
        self.vram_gb = vram_gb
        self.transient_retries = transient_retries
        self.prompt_hook = prompt_hook
        self._listeners: dict[str, list[Listener]] = defaultdict(list)
        self._preset: dict[str, VramPreset | None] = {}

    def on(self, kind: str, listener: Listener) -> Runner:
        self._listeners[kind].append(listener)
        return self

    def _emit(self, kind: str, job_id: str | None = None, **detail: Any) -> None:
        event = Event(kind, job_id, detail)
        for listener in [*self._listeners[kind], *self._listeners["*"]]:
            listener(event)

    def start(self, plan: Plan, name: str | None = None) -> RunResult:
        run = RunDir.create(self.runs_root, plan, name)
        return self._execute(run, plan)

    def resume(self, run_dir: str | Path) -> RunResult:
        run = RunDir(run_dir)
        if not run.exists():
            raise RunError(f"no plan.json in {run.root}")
        return self._execute(run, run.load_plan())

    def rerun(self, run_dir: str | Path, job_id: str, seed: int | None = None) -> RunResult:
        run = RunDir(run_dir)
        plan = run.load_plan()
        job = plan.job(job_id)
        job.seed = seed if seed is not None else job.seed + 1
        run.save_plan(plan)
        run.log("rerun", job_id, seed=job.seed, dependants=plan.dependants(job_id))
        return self._execute(run, plan)

    def _initial_preset(self, profile: ModelProfile) -> VramPreset | None:
        if profile.id in self._preset:
            return self._preset[profile.id]
        if not profile.vram_presets:
            return None
        if self.vram_gb is None:
            return profile.vram_presets[-1]
        return profile.preset_for(self.vram_gb) or profile.vram_presets[-1]

    def _lower_preset(self, profile: ModelProfile, current: VramPreset | None) -> VramPreset | None:
        if current is None:
            return None
        presets = profile.vram_presets
        index = next((i for i, p in enumerate(presets) if p.recipe == current.recipe), None)
        if index is None or index + 1 >= len(presets):
            return None
        return presets[index + 1]

    def _inputs(self, run: RunDir, job: Job) -> tuple[JobInputs, str | None]:
        inputs = JobInputs()
        hashes = []
        if job.init is not None:
            if job.init.kind == InitKind.IMAGE:
                path = Path(job.init.path)
            else:
                path = run.last_frame_path(job.init.job)
            if not path.is_file():
                raise RunError(f"{job.id}: init image missing at {path}")
            inputs.init_image = load_image(path, job.width, job.height)
            inputs.init_image_path = path
            hashes.append(file_sha256(path))
        if job.end_image is not None:
            path = Path(job.end_image)
            if not path.is_file():
                raise RunError(f"{job.id}: end image missing at {path}")
            inputs.end_image = load_image(path, job.width, job.height)
            inputs.end_image_path = path
            hashes.append("end:" + file_sha256(path))
        if not hashes:
            return inputs, None
        return inputs, hashes[0] if len(hashes) == 1 else "+".join(hashes)

    def _is_done(self, run: RunDir, job: Job, key: str) -> bool:
        meta = run.read_meta(job.id)
        if not meta or meta.get("key") != key:
            return False
        clip = run.clip_path(job.id)
        return (
            clip.is_file()
            and run.last_frame_path(job.id).is_file()
            and file_sha256(clip) == meta.get("clip_sha256")
        )

    def _order(self, plan: Plan) -> list[Job]:
        pending = list(plan.jobs)
        ordered: list[Job] = []
        placed: set[str] = set()
        current_model: str | None = None
        while pending:
            ready = [j for j in pending if j.depends_on is None or j.depends_on in placed]
            if not ready:
                raise RunError("plan has a dependency cycle or a missing job")
            pick = next((j for j in ready if j.model == current_model), ready[0])
            ordered.append(pick)
            placed.add(pick.id)
            pending.remove(pick)
            current_model = pick.model
        return ordered

    def _execute(self, run: RunDir, plan: Plan) -> RunResult:
        result = RunResult(run_dir=run.root)
        blocked: set[str] = set()
        run.log("run_started", jobs=len(plan.jobs), backend=self.backend.name)
        self._emit("run_started", jobs=len(plan.jobs))

        for job in self._order(plan):
            if job.depends_on in blocked:
                blocked.add(job.id)
                result.skipped.append(job.id)
                run.log("skipped", job.id, reason=f"depends on {job.depends_on}")
                self._emit("job_skipped", job.id)
                continue

            try:
                inputs, input_hash = self._inputs(run, job)
                job = self._adapt_prompt(run, job, inputs, input_hash)
                key = job_cache_key(job, input_hash)
                if self._is_done(run, job, key):
                    result.cached.append(job.id)
                    self._emit("job_cached", job.id)
                    continue
                seconds = self._run_job(run, job, inputs, key)
            except Exception as exc:  # noqa: BLE001
                blocked.add(job.id)
                result.failed[job.id] = f"{type(exc).__name__}: {exc}"
                run.log("failed", job.id, error=result.failed[job.id])
                self._emit("job_failed", job.id, error=result.failed[job.id])
                continue
            result.done.append(job.id)
            self._emit("job_done", job.id, seconds=seconds)

        run.log(
            "run_finished",
            done=len(result.done),
            cached=len(result.cached),
            failed=len(result.failed),
            skipped=len(result.skipped),
        )
        self._emit("run_finished", ok=result.ok)
        return result

    def _adapt_prompt(
        self, run: RunDir, job: Job, inputs: JobInputs, input_hash: str | None
    ) -> Job:
        """Let the prompt hook rewrite a chained job's prompt from its starting frame.

        The rewrite is stored per job and keyed by the planned prompt and the input
        frame, so a resume reuses it instead of calling the hook again.
        """
        if self.prompt_hook is None or job.depends_on is None or inputs.init_image is None:
            return job
        source = hashlib.sha256(f"{job.prompt}\0{input_hash}".encode()).hexdigest()
        path = run.job_dir(job.id) / PROMPT_FILE
        if path.is_file():
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored.get("source") == source:
                return job.model_copy(update={"prompt": stored["prompt"]})
        prompt = self.prompt_hook(job, inputs).strip()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"source": source, "planned": job.prompt, "prompt": prompt}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        run.log("prompt_adapted", job.id)
        return job.model_copy(update={"prompt": prompt})

    def _run_job(self, run: RunDir, job: Job, inputs: JobInputs, key: str) -> float:
        profile = self.registry.get(job.model)
        if not self.backend.supports(profile):
            raise RunError(f"backend {self.backend.name} cannot run {profile.id}")
        preset = self._initial_preset(profile)
        transient_left = self.transient_retries

        while True:
            self.backend.prepare(profile, preset)
            self._preset[profile.id] = preset
            run.log("started", job.id, recipe=preset.recipe if preset else None)
            self._emit("job_started", job.id)
            began = time.monotonic()
            try:
                clip = self.backend.run(job, inputs)
                break
            except OutOfMemory as exc:
                lower = self._lower_preset(profile, preset)
                if lower is None:
                    raise
                run.log("oom", job.id, recipe=preset.recipe, next=lower.recipe, error=str(exc))
                self.backend.release()
                preset = lower
            except TransientError as exc:
                if transient_left <= 0:
                    raise
                transient_left -= 1
                run.log("retry", job.id, error=str(exc))

        seconds = time.monotonic() - began
        frames = clip.frames
        if frames.shape[0] != job.frames:
            raise RunError(
                f"{job.id}: backend returned {frames.shape[0]} frames, expected {job.frames}"
            )
        if frames.shape[1:3] != (job.height, job.width):
            raise RunError(
                f"{job.id}: backend returned {frames.shape[2]}x{frames.shape[1]}, "
                f"expected {job.width}x{job.height}"
            )

        job_dir = run.job_dir(job.id)
        job_dir.mkdir(parents=True, exist_ok=True)
        clip_path = run.clip_path(job.id)
        write_video(clip_path, frames, job.fps)
        save_image(run.last_frame_path(job.id), frames[-1])
        run.write_meta(
            job.id,
            {
                "key": key,
                "clip_sha256": file_sha256(clip_path),
                "job": job.model_dump(mode="json"),
                "backend": self.backend.name,
                "recipe": preset.recipe if preset else None,
                "seconds": round(seconds, 3),
                "backend_meta": clip.meta,
            },
        )
        run.log("done", job.id, seconds=round(seconds, 3))
        return seconds
