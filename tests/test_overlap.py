import numpy as np
import pytest

from shotwright.media import read_video
from shotwright.plan import InitKind, compile_plan
from shotwright.profiles import Registry
from shotwright.run import (
    BackendError,
    ComfyUIBackend,
    FakeBackend,
    HostedBackend,
    JobInputs,
    RunDir,
    Runner,
)
from shotwright.stitch import stitch
from tests.conftest import TINY_PROFILE, tiny_spec

VACE = (
    TINY_PROFILE
    + """
  - id: tiny-vace
    base: tiny-i2v
    continuity: [overlap_context, last_frame_i2v]
"""
)

SHOTS = [
    {"id": "a", "prompt": "first", "duration": 1},
    {"id": "b", "prompt": "second", "duration": 1, "transition": "continuous"},
    {"id": "c", "prompt": "long", "duration": 4},
]


@pytest.fixture
def vace_registry(tmp_path_factory):
    d = tmp_path_factory.mktemp("profiles")
    (d / "tiny.yaml").write_text(VACE)
    return Registry.load(extra_dirs=[d])


def plan(registry, model="tiny-vace", overlap=4):
    spec = tiny_spec(SHOTS, defaults={"model": model, "seed": 3}, continuity={"overlap": overlap})
    return compile_plan(spec, registry=registry)


def test_overlap_jobs_carry_context(vace_registry):
    p = plan(vace_registry)
    b = p.job("b")
    assert b.init.kind == InitKind.OVERLAP and b.init.frames == 4
    assert b.shared_frames == 4 and b.depends_on == "a"
    segments = [j for j in p.jobs if j.shot_id == "c"]
    assert len(segments) > 1
    assert all(j.init.kind == InitKind.OVERLAP for j in segments[1:])


def test_models_without_overlap_fall_back(vace_registry):
    p = plan(vace_registry, model="tiny-i2v")
    assert p.job("b").init.kind == InitKind.LAST_FRAME
    assert p.job("b").shared_frames == 1


def test_overlap_keeps_requested_length(vace_registry):
    requested = sum(s["duration"] for s in SHOTS) * 8
    for p in (plan(vace_registry), plan(vace_registry, overlap=0)):
        assert requested <= p.total_frames <= requested + 4 * len(p.jobs)


def test_run_and_stitch_with_context(tmp_path, vace_registry):
    p = plan(vace_registry)
    result = Runner(FakeBackend(), runs_root=tmp_path / "runs", registry=vace_registry).start(
        p, name="r"
    )
    assert result.ok, result.failed
    run = RunDir(result.run_dir)
    a, _ = read_video(run.clip_path("a"))
    b, _ = read_video(run.clip_path("b"))
    assert np.abs(b[:4].astype(int) - a[-4:].astype(int)).mean() < 4
    stitched = stitch(run.root)
    assert stitched.frames == p.total_frames
    assert not any(s.flagged for s in stitched.seams)


def test_resume_reuses_overlap_jobs(tmp_path, vace_registry):
    p = plan(vace_registry)
    first = Runner(FakeBackend(), runs_root=tmp_path / "runs", registry=vace_registry).start(
        p, name="r"
    )
    backend = FakeBackend()
    again = Runner(backend, registry=vace_registry).resume(first.run_dir)
    assert sorted(again.cached) == sorted(j.id for j in p.jobs)
    assert backend.calls == []


def test_diffusers_builds_video_and_mask(vace_registry):
    from tests.test_diffusers_backend import StubPipeline, backend

    class VacePipeline(StubPipeline):
        def __call__(self, video, mask, prompt, num_frames, height, width, generator, output_type):
            out = super().__call__(prompt, num_frames, height, width, generator, output_type)
            self.calls[-1]["video"] = len(video)
            self.calls[-1]["mask"] = [int(np.asarray(m).max()) for m in mask]
            return out

    loaded = []

    def loader(class_name, repo, dtype):
        pipe = (VacePipeline if class_name == "VacePipeline" else StubPipeline)(repo, dtype)
        loaded.append(pipe)
        return pipe

    profile = vace_registry.get("tiny-vace")
    job = plan(vace_registry).job("b")
    context = np.zeros((4, job.height, job.width, 3), dtype=np.uint8)
    inputs = JobInputs(init_image=context[-1], context_frames=context)
    for cls, ok in (("VacePipeline", True), ("StubPipeline", False)):
        prof = profile.model_copy(
            update={
                "backends": profile.backends.model_copy(
                    update={"diffusers": cls, "diffusers_repo": "t/v"}
                )
            }
        )
        b = backend(loader)
        b.prepare(prof, None)
        if ok:
            b.run(job, inputs)
            call = loaded[-1].calls[0]
            assert call["video"] == job.frames
            assert call["mask"] == [0] * 4 + [255] * (job.frames - 4)
        else:
            with pytest.raises(BackendError, match="context frames"):
                b.run(job, inputs)


@pytest.mark.parametrize("cls", [ComfyUIBackend, HostedBackend])
def test_other_backends_refuse_context(cls, vace_registry, tmp_path):
    b = cls(transport=object()) if cls is ComfyUIBackend else cls(transport=object())
    b.prepare(vace_registry.get("tiny-vace"), None)
    job = plan(vace_registry).job("b")
    context = np.zeros((4, job.height, job.width, 3), dtype=np.uint8)
    with pytest.raises(BackendError, match="overlap"):
        b.run(job, JobInputs(init_image=context[-1], context_frames=context))
