import json

import numpy as np
import pytest
from click.testing import CliRunner

from shotwright.cli import main
from shotwright.color import LabStats, lab_to_rgb, match_to, rgb_to_lab
from shotwright.media import read_video, write_video
from shotwright.run import FakeBackend, RunDir, Runner
from shotwright.stitch import stitch

CHAIN = [
    {"id": "a", "prompt": "first", "duration": 1},
    {"id": "b", "prompt": "second", "duration": 1, "transition": "continuous"},
    {"id": "c", "prompt": "third", "duration": 1, "transition": "continuous"},
    {"id": "d", "prompt": "fourth", "duration": 1, "transition": "continuous"},
]


def texture(height=48, width=64, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = np.stack(
        [100 + 60 * np.sin(x / 9), 120 + 50 * np.cos(y / 7), 90 + 40 * np.sin((x + y) / 11)],
        axis=-1,
    )
    return np.clip(base + rng.normal(0, 4, base.shape), 0, 255)


def tint(image, hops):
    shift = np.array([14.0, 4.0, -12.0]) * hops
    return np.clip(image * (1 + 0.04 * hops) + shift, 0, 255).round().astype(np.uint8)


def render_chain(tmp_path, registry, tiny_plan, tint_at):
    plan = tiny_plan(CHAIN)
    result = Runner(FakeBackend(), runs_root=tmp_path / "runs", registry=registry).start(
        plan, name="r"
    )
    run = RunDir(result.run_dir)
    base = texture()
    offset = 0
    for hop, job in enumerate(plan.jobs):
        if hop:
            offset -= 1
        frames = np.stack(
            [
                tint(np.roll(base, offset + i, axis=1), tint_at(hop, i / (job.frames - 1)))
                for i in range(job.frames)
            ]
        )
        write_video(run.clip_path(job.id), frames, job.fps)
        offset += job.frames
    return run


@pytest.fixture
def drifting(tmp_path, registry, tiny_plan):
    """Every hop jumps the colour a little further at the seam."""
    return render_chain(tmp_path, registry, tiny_plan, lambda hop, t: hop)


@pytest.fixture
def creeping(tmp_path, registry, tiny_plan):
    """Colour creeps within each clip but every seam is continuous."""
    return render_chain(tmp_path, registry, tiny_plan, lambda hop, t: (hop + t) * 0.8)


def test_lab_round_trip():
    rgb = np.random.default_rng(1).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    back = lab_to_rgb(rgb_to_lab(rgb))
    assert np.abs(back.astype(int) - rgb.astype(int)).max() <= 1


def test_known_lab_values():
    lab = rgb_to_lab(np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0]], dtype=np.uint8))
    assert lab[0] == pytest.approx([100, 0, 0], abs=0.05)
    assert lab[1] == pytest.approx([0, 0, 0], abs=0.05)
    assert lab[2] == pytest.approx([53.24, 80.09, 67.20], abs=0.05)


def test_match_moves_statistics_to_target():
    source = tint(texture(), 3)[None]
    target = tint(texture(), 0)[None]
    moved = match_to(source, LabStats.of(source), LabStats.of(target))
    before = np.abs(LabStats.of(source).mean - LabStats.of(target).mean).max()
    after = np.abs(LabStats.of(moved).mean - LabStats.of(target).mean).max()
    assert after < before / 5


def test_ramp_leaves_first_frame_alone():
    frames = np.repeat(tint(texture(), 3)[None], 5, axis=0)
    target = LabStats.of(texture().astype(np.uint8))
    moved = match_to(frames, LabStats.of(frames), target, ramp=1.0)
    assert np.abs(moved[0].astype(int) - frames[0].astype(int)).max() <= 1
    assert np.abs(moved[-1].astype(int) - frames[-1].astype(int)).mean() > 5


def seam_diffs(result):
    return [s.mean_abs_diff for s in result.seams]


def test_seam_mode_removes_colour_jumps(drifting):
    raw = stitch(drifting.root, out=drifting.root / "raw.mp4", color="none")
    fixed = stitch(drifting.root, out=drifting.root / "seam.mp4", color="seam")
    assert all(f < r / 2 for f, r in zip(seam_diffs(fixed), seam_diffs(raw), strict=True))
    assert raw.frames == fixed.frames


def test_anchor_mode_limits_long_term_drift(creeping):
    first = read_video(stitch(creeping.root, out=creeping.root / "n.mp4", color="none").path)[0]
    start = LabStats.of(first[:2]).mean

    def end_drift(mode):
        out = creeping.root / f"{mode}.mp4"
        frames, _ = read_video(stitch(creeping.root, out=out, color=mode).path)
        return np.abs(LabStats.of(frames[-2:]).mean - start).max()

    assert end_drift("anchor") < end_drift("seam") * 0.7


def test_cuts_are_not_colour_matched(tmp_path, registry, tiny_plan):
    plan = tiny_plan(
        [{"id": "a", "prompt": "x", "duration": 1}, {"id": "b", "prompt": "y", "duration": 1}]
    )
    run = Runner(FakeBackend(), runs_root=tmp_path / "runs", registry=registry).start(
        plan, name="r"
    )
    raw = stitch(run.run_dir, out=tmp_path / "raw.mp4", color="none")
    fixed = stitch(run.run_dir, out=tmp_path / "fixed.mp4", color="anchor")
    assert np.array_equal(read_video(raw.path)[0], read_video(fixed.path)[0])


def test_report_and_cli(drifting):
    r = CliRunner().invoke(main, ["stitch", str(drifting.root), "--color", "anchor"])
    assert r.exit_code == 0, r.output
    report = json.loads((drifting.root / "stitch.json").read_text())
    assert report["color"] == "anchor"


def test_bad_color_mode(drifting):
    with pytest.raises(ValueError):
        stitch(drifting.root, color="sepia")
