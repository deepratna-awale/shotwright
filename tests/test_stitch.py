import json

import numpy as np
import pytest
from click.testing import CliRunner

from shotwright.cli import main
from shotwright.media import read_video
from shotwright.run import FakeBackend, RunDir, Runner
from shotwright.stitch import StitchError, stitch

SHOTS = [
    {"id": "a", "prompt": "first", "duration": 1},
    {"id": "b", "prompt": "second", "duration": 1, "transition": "continuous"},
    {"id": "c", "prompt": "third", "duration": 2},
]


@pytest.fixture
def rendered(tmp_path, registry, tiny_plan):
    plan = tiny_plan(SHOTS)
    result = Runner(FakeBackend(), runs_root=tmp_path / "runs", registry=registry).start(
        plan, name="r"
    )
    assert result.ok
    return plan, result.run_dir


def test_stitch_drops_shared_frames(rendered):
    plan, run_dir = rendered
    result = stitch(run_dir)
    frames, fps = read_video(result.path)
    expected = sum(j.frames - (1 if j.depends_on else 0) for j in plan.jobs)
    assert fps == 8
    assert len(frames) == expected == result.frames
    assert frames.shape[1:3] == (48, 64)


def test_chain_seams_are_smooth_and_cuts_are_recorded(rendered):
    _, run_dir = rendered
    result = stitch(run_dir)
    kinds = {(s.before, s.after): s for s in result.seams}
    assert kinds[("a", "b")].kind == "chain"
    assert kinds[("b", "c")].kind == "cut"
    assert kinds[("a", "b")].mean_abs_diff < kinds[("b", "c")].mean_abs_diff
    report = json.loads((RunDir(run_dir).root / "stitch.json").read_text())
    assert report["frames"] == result.frames


@pytest.mark.parametrize("method", ["resample", "blend"])
def test_fps_conversion_keeps_duration(rendered, method):
    plan, run_dir = rendered
    native = stitch(run_dir, out=RunDir(run_dir).root / "native.mp4")
    up = stitch(run_dir, out=RunDir(run_dir).root / f"up-{method}.mp4", fps=24, method=method)
    assert up.fps == 24
    assert abs(up.duration - native.duration) <= 1 / 24
    frames, _ = read_video(up.path)
    assert len(frames) == up.frames


def test_blend_produces_intermediate_frames(rendered):
    _, run_dir = rendered
    a = stitch(run_dir, out=RunDir(run_dir).root / "r.mp4", fps=24, method="resample")
    b = stitch(run_dir, out=RunDir(run_dir).root / "b.mp4", fps=24, method="blend")
    fa, _ = read_video(a.path)
    fb, _ = read_video(b.path)
    repeats_a = sum(np.array_equal(fa[i], fa[i + 1]) for i in range(len(fa) - 1))
    repeats_b = sum(np.array_equal(fb[i], fb[i + 1]) for i in range(len(fb) - 1))
    assert repeats_b < repeats_a


def test_resize_and_crop_to_output(tmp_path, registry, tiny_plan):
    plan = tiny_plan(
        [{"id": "a", "prompt": "x", "duration": 1}], output={"width": 96, "height": 64, "fps": 8}
    )
    job = plan.job("a")
    assert (job.width, job.height) == (64, 48)
    run = Runner(FakeBackend(), runs_root=tmp_path / "runs", registry=registry).start(
        plan, name="r"
    )
    result = stitch(run.run_dir)
    frames, _ = read_video(result.path)
    assert frames.shape[1:3] == (64, 96)


def test_refuses_incomplete_run(tmp_path, registry, tiny_plan):
    plan = tiny_plan(SHOTS)
    run = Runner(FakeBackend(fail={"c"}), runs_root=tmp_path / "runs", registry=registry).start(
        plan, name="r"
    )
    with pytest.raises(StitchError, match="c"):
        stitch(run.run_dir)


def test_bad_method(rendered):
    with pytest.raises(ValueError):
        stitch(rendered[1], method="magic")


def test_cli_end_to_end(tmp_path, monkeypatch):
    shots = tmp_path / "shots.yaml"
    shots.write_text(
        "project: e2e\n"
        "output: {width: 320, height: 176, fps: 24}\n"
        "defaults: {model: wan2.2-ti2v-5b, seed: 1}\n"
        "shots:\n"
        "  - {id: one, prompt: a, frames: 9}\n"
        "  - {id: two, prompt: b, frames: 9, transition: continuous}\n"
    )
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    r = runner.invoke(main, ["run", str(shots), "--name", "e2e"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(main, ["stitch", "runs/e2e"])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "runs/e2e/final.mp4").is_file()
    r = runner.invoke(main, ["resume", "runs/e2e"])
    assert "2 reused" in r.output
