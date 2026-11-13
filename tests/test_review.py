import json

import numpy as np
from click.testing import CliRunner
from PIL import Image

from shotwright.cli import main
from shotwright.media import write_video
from shotwright.review import contact_sheet
from shotwright.run import FakeBackend, RunDir, Runner
from shotwright.stitch import stitch

SHOTS = [
    {"id": "a", "prompt": "first", "duration": 1},
    {"id": "b", "prompt": "second", "duration": 1, "transition": "continuous"},
    {"id": "c", "prompt": "third", "duration": 1},
]


def render(tmp_path, registry, tiny_plan, **kw):
    plan = tiny_plan(SHOTS)
    result = Runner(FakeBackend(**kw), runs_root=tmp_path / "runs", registry=registry).start(
        plan, name="r"
    )
    return plan, RunDir(result.run_dir)


def test_smooth_chain_is_not_flagged(tmp_path, registry, tiny_plan):
    _, run = render(tmp_path, registry, tiny_plan)
    seams = {s.after: s for s in stitch(run.root, color="none").seams}
    assert not seams["b"].flagged
    assert seams["b"].kind == "chain"
    assert seams["c"].kind == "cut"
    assert not seams["c"].flagged


def test_broken_chain_is_flagged(tmp_path, registry, tiny_plan):
    plan, run = render(tmp_path, registry, tiny_plan)
    job = plan.job("b")
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (job.frames, job.height, job.width, 3), dtype=np.uint8)
    frames[1:] = frames[1]
    write_video(run.clip_path("b"), frames, job.fps)
    result = stitch(run.root, color="none")
    seam = next(s for s in result.seams if s.after == "b")
    assert seam.flagged
    assert seam.jump > 2.5
    assert seam.color_delta >= 0
    report = json.loads((run.root / "stitch.json").read_text())
    assert next(s for s in report["seams"] if s["after"] == "b")["flagged"] is True


def test_contact_sheet_layout(tmp_path, registry, tiny_plan):
    plan, run = render(tmp_path, registry, tiny_plan)
    stitch(run.root)
    path = contact_sheet(run.root, per_job=4, thumb_width=64)
    image = Image.open(path)
    assert path == run.root / "contact_sheet.png"
    assert image.height == len(plan.jobs) * (48 + 12)
    assert image.width == 220 + 4 * (64 + 6) + 6


def test_contact_sheet_marks_missing_jobs(tmp_path, registry, tiny_plan):
    plan, run = render(tmp_path, registry, tiny_plan, fail={"c"})
    path = contact_sheet(run.root, out=tmp_path / "sheet.png")
    assert path.is_file()


def test_cli_sheet(tmp_path, registry, tiny_plan):
    _, run = render(tmp_path, registry, tiny_plan)
    r = CliRunner().invoke(main, ["sheet", str(run.root), "--frames", "3"])
    assert r.exit_code == 0, r.output
    assert (run.root / "contact_sheet.png").is_file()


def test_cli_stitch_reports_flagged_seams(tmp_path, registry, tiny_plan):
    plan, run = render(tmp_path, registry, tiny_plan)
    job = plan.job("b")
    frames = np.zeros((job.frames, job.height, job.width, 3), dtype=np.uint8)
    frames[1:] = 255
    write_video(run.clip_path("b"), frames, job.fps)
    r = CliRunner().invoke(main, ["stitch", str(run.root), "--color", "none"])
    assert r.exit_code == 0, r.output
    assert "a -> b" in r.output
