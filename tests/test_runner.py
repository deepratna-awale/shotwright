import numpy as np
import pytest

from shotwright.media import file_sha256, load_image, read_video
from shotwright.run import FakeBackend, RunDir, Runner

SHOTS = [
    {"id": "a", "prompt": "first", "duration": 1},
    {"id": "b", "prompt": "second", "duration": 1, "transition": "continuous"},
    {"id": "c", "prompt": "third", "duration": 3},
    {"id": "d", "prompt": "fourth", "duration": 1},
]


def make_runner(tmp_path, registry, backend=None, **kw):
    return Runner(backend or FakeBackend(), runs_root=tmp_path / "runs", registry=registry, **kw)


def test_full_run_writes_every_job(tmp_path, registry, tiny_plan):
    plan = tiny_plan(SHOTS)
    result = make_runner(tmp_path, registry).start(plan, name="r")
    assert result.ok
    assert sorted(result.done) == sorted(j.id for j in plan.jobs)
    run = RunDir(result.run_dir)
    for job in plan.jobs:
        frames, fps = read_video(run.clip_path(job.id))
        assert frames.shape == (job.frames, job.height, job.width, 3)
        assert fps == job.fps
        assert run.read_meta(job.id)["job"]["seed"] == job.seed


def test_chained_job_starts_on_previous_last_frame(tmp_path, registry, tiny_plan):
    plan = tiny_plan(SHOTS)
    backend = FakeBackend()
    result = make_runner(tmp_path, registry, backend).start(plan, name="r")
    run = RunDir(result.run_dir)
    last_a = load_image(run.last_frame_path("a"))
    frames_b, _ = read_video(run.clip_path("b"))
    assert np.abs(frames_b[0].astype(int) - last_a.astype(int)).mean() < 4


def test_resume_reuses_everything(tmp_path, registry, tiny_plan):
    plan = tiny_plan(SHOTS)
    runner = make_runner(tmp_path, registry)
    first = runner.start(plan, name="r")
    hashes = {j.id: file_sha256(RunDir(first.run_dir).clip_path(j.id)) for j in plan.jobs}
    backend = FakeBackend()
    second = make_runner(tmp_path, registry, backend).resume(first.run_dir)
    assert sorted(second.cached) == sorted(j.id for j in plan.jobs)
    assert backend.calls == []
    assert hashes == {j.id: file_sha256(RunDir(first.run_dir).clip_path(j.id)) for j in plan.jobs}


def test_crash_then_resume_matches_clean_run(tmp_path, registry, tiny_plan):
    plan = tiny_plan(SHOTS)
    clean = make_runner(tmp_path, registry).start(plan, name="clean")

    crashing = FakeBackend(fail={"c.01"})
    partial = make_runner(tmp_path, registry, crashing).start(plan, name="crash")
    assert "c.01" in partial.failed
    assert "c.02" in partial.skipped
    assert "d" in partial.done

    resumed = make_runner(tmp_path, registry).resume(partial.run_dir)
    assert resumed.ok
    assert set(resumed.done) == {"c.01", "c.02"}
    for job in plan.jobs:
        assert file_sha256(RunDir(clean.run_dir).clip_path(job.id)) == file_sha256(
            RunDir(partial.run_dir).clip_path(job.id)
        )


def test_corrupted_clip_is_rendered_again(tmp_path, registry, tiny_plan):
    plan = tiny_plan(SHOTS)
    first = make_runner(tmp_path, registry).start(plan, name="r")
    RunDir(first.run_dir).clip_path("d").write_bytes(b"broken")
    backend = FakeBackend()
    make_runner(tmp_path, registry, backend).resume(first.run_dir)
    assert backend.calls == ["d"]


def test_rerun_invalidates_only_the_chain(tmp_path, registry, tiny_plan):
    plan = tiny_plan(SHOTS)
    first = make_runner(tmp_path, registry).start(plan, name="r")
    backend = FakeBackend()
    result = make_runner(tmp_path, registry, backend).rerun(first.run_dir, "a", seed=99)
    assert set(result.done) == {"a", "b"}
    assert sorted(backend.calls) == ["a", "b"]
    assert RunDir(first.run_dir).load_plan().job("a").seed == 99


def test_edited_plan_rerenders_changed_jobs(tmp_path, registry, tiny_plan):
    plan = tiny_plan(SHOTS)
    first = make_runner(tmp_path, registry).start(plan, name="r")
    run = RunDir(first.run_dir)

    edited = plan.model_copy(deep=True)
    edited.job("c.01").prompt = "something else"
    run.save_plan(edited)
    backend = FakeBackend()
    make_runner(tmp_path, registry, backend).resume(first.run_dir)
    # the fake ignores prompts, so c.01's last frame is byte-identical and c.02 is reused
    assert backend.calls == ["c.01"]

    edited.job("c.01").seed += 1
    run.save_plan(edited)
    backend = FakeBackend()
    make_runner(tmp_path, registry, backend).resume(first.run_dir)
    assert backend.calls == ["c.01", "c.02"]


def test_oom_steps_down_presets(tmp_path, registry, tiny_plan):
    plan = tiny_plan(SHOTS[:1])
    backend = FakeBackend(oom_until_recipe="tiny")
    result = make_runner(tmp_path, registry, backend, vram_gb=24).start(plan, name="r")
    assert result.ok
    assert [r for _, r in backend.loads] == ["full", "offload", "tiny"]
    events = [e["event"] for e in RunDir(result.run_dir).events()]
    assert events.count("oom") == 2


def test_oom_with_no_lower_preset_fails(tmp_path, registry, tiny_plan):
    plan = tiny_plan(SHOTS[:1])
    backend = FakeBackend(oom_until_recipe="never")
    result = make_runner(tmp_path, registry, backend, vram_gb=24).start(plan, name="r")
    assert "OutOfMemory" in result.failed["a"]


def test_transient_errors_retry(tmp_path, registry, tiny_plan):
    plan = tiny_plan(SHOTS[:1])
    ok = make_runner(tmp_path, registry, FakeBackend(transient={"a": 2})).start(plan, name="ok")
    assert ok.ok
    bad = make_runner(tmp_path, registry, FakeBackend(transient={"a": 3})).start(plan, name="bad")
    assert "TransientError" in bad.failed["a"]


def test_missing_init_image_fails_job_not_run(tmp_path, registry, tiny_plan):
    (tmp_path / "ref.png").write_bytes(b"x")
    from shotwright.plan import compile_plan

    from .conftest import tiny_spec

    spec = tiny_spec(
        [
            {"id": "a", "prompt": "x", "init_image": str(tmp_path / "ref.png")},
            {"id": "b", "prompt": "y"},
        ]
    )
    plan = compile_plan(spec, registry=registry)
    (tmp_path / "ref.png").unlink()
    result = make_runner(tmp_path, registry).start(plan, name="r")
    assert "a" in result.failed
    assert "b" in result.done


def test_jobs_grouped_by_model(tmp_path, registry, tiny_plan):
    plan = tiny_plan(SHOTS)
    runner = make_runner(tmp_path, registry)
    order = [j.id for j in runner._order(plan)]
    assert order.index("a") < order.index("b")
    assert order.index("c.01") < order.index("c.02")


def test_jobs_grouped_by_conditioning(tmp_path, registry, tiny_plan):
    plan = tiny_plan(SHOTS)
    order = list(make_runner(tmp_path, registry)._order(plan))
    modes = [j.init is not None for j in order]
    assert sum(a != b for a, b in zip(modes, modes[1:], strict=False)) == 1


def test_events(tmp_path, registry, tiny_plan):
    seen = []
    runner = make_runner(tmp_path, registry).on("*", lambda e: seen.append(e.kind))
    runner.start(tiny_plan(SHOTS[:1]), name="r")
    assert seen[0] == "run_started" and seen[-1] == "run_finished"
    assert "job_done" in seen


def test_resume_without_plan(tmp_path, registry):
    from shotwright.run import RunError

    with pytest.raises(RunError):
        make_runner(tmp_path, registry).resume(tmp_path)
