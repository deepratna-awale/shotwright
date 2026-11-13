import json

import yaml
from click.testing import CliRunner

from shotwright.cli import main
from shotwright.llm import continuation_hook
from shotwright.run import FakeBackend, RunDir, Runner
from tests.test_llm import Scripted, client

SHOTS = [
    {"id": "a", "prompt": "first", "duration": 1},
    {"id": "b", "prompt": "second", "duration": 1, "transition": "continuous"},
    {"id": "c", "prompt": "third", "duration": 1},
]


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, job, inputs):
        self.calls.append((job.id, inputs.init_image.shape))
        return f"{job.prompt}, adapted"


def test_hook_rewrites_only_chained_jobs(tmp_path, registry, tiny_plan):
    hook = Recorder()
    plan = tiny_plan(SHOTS)
    result = Runner(
        FakeBackend(), runs_root=tmp_path / "runs", registry=registry, prompt_hook=hook
    ).start(plan, name="r")
    assert result.ok
    assert [c[0] for c in hook.calls] == ["b"]
    run = RunDir(result.run_dir)
    assert run.read_meta("b")["job"]["prompt"].endswith(", adapted")
    assert run.read_meta("a")["job"]["prompt"] == plan.job("a").prompt
    stored = json.loads((run.job_dir("b") / "prompt.json").read_text())
    assert stored["planned"] == plan.job("b").prompt


def test_resume_reuses_adapted_prompt(tmp_path, registry, tiny_plan):
    plan = tiny_plan(SHOTS)
    first = Runner(
        FakeBackend(), runs_root=tmp_path / "runs", registry=registry, prompt_hook=Recorder()
    ).start(plan, name="r")
    hook = Recorder()
    backend = FakeBackend()
    again = Runner(backend, registry=registry, prompt_hook=hook).resume(first.run_dir)
    assert sorted(again.cached) == sorted(j.id for j in plan.jobs)
    assert hook.calls == []
    assert backend.calls == []


def test_new_input_frame_asks_again(tmp_path, registry, tiny_plan):
    plan = tiny_plan(SHOTS)
    hook = Recorder()
    runner = Runner(FakeBackend(), runs_root=tmp_path / "runs", registry=registry, prompt_hook=hook)
    first = runner.start(plan, name="r")
    runner.rerun(first.run_dir, "a", seed=99)
    assert [c[0] for c in hook.calls] == ["b", "b"]


def test_continuation_hook_uses_vision_then_rewrite(tmp_path, registry, tiny_plan):
    c = client([{"description": "red boat at left"}, {"prompt": "the red boat drifts"}])
    plan = tiny_plan(SHOTS)
    result = Runner(
        FakeBackend(),
        runs_root=tmp_path / "runs",
        registry=registry,
        prompt_hook=continuation_hook(c),
    ).start(plan, name="r")
    assert result.ok
    assert RunDir(result.run_dir).read_meta("b")["job"]["prompt"] == "the red boat drifts"
    vision, rewrite = (r[1]["messages"][1]["content"] for r in c.transport.requests)
    assert vision[1]["type"] == "image_url"
    assert "red boat at left" in rewrite and plan.job("b").prompt in rewrite


def test_cli_llm_continuity(tmp_path, monkeypatch):
    shots = tmp_path / "shots.yaml"
    shots.write_text(
        yaml.safe_dump(
            {
                "project": "p",
                "output": {"width": 320, "height": 176, "fps": 24},
                "defaults": {"model": "wan2.2-ti2v-5b", "seed": 1},
                "shots": [
                    {"id": "one", "prompt": "a", "frames": 9},
                    {"id": "two", "prompt": "b", "frames": 9, "transition": "continuous"},
                ],
            }
        )
    )
    scripted = Scripted([{"description": "d"}, {"prompt": "b, continued"}])
    monkeypatch.setattr("shotwright.llm.client.HttpTransport", lambda: scripted)
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(
        main, ["run", str(shots), "--name", "x", "--llm-continuity", "--llm-model", "m"]
    )
    assert r.exit_code == 0, r.output
    meta = RunDir(tmp_path / "runs/x").read_meta("two")
    assert meta["job"]["prompt"] == "b, continued"
    assert len(scripted.requests) == 2
