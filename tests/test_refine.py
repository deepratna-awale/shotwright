import json

import numpy as np
import pytest

from shotwright.media import read_video
from shotwright.plan import FrameRefine, PlanError
from shotwright.profiles import get_profile
from shotwright.refine import (
    LOOK_DEFAULTS,
    FakeRefiner,
    match_color,
    pick_start_frame,
    refine_prompt,
    refiner_model,
    sharpness,
    working_size,
)
from shotwright.run import FakeBackend, Runner
from shotwright.snap import snap
from shotwright.stitch import stitch

CHAIN = [
    {"id": "a", "prompt": "A red kite over a beach.", "end_prompt": "A red kite high in the sky."},
    {"id": "b", "prompt": "The kite dives.", "transition": "continuous"},
]


def _textured(seed: int, blur: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(48, 64, 3)).astype(np.float32)
    for _ in range(blur):
        img = (img + np.roll(img, 1, 0) + np.roll(img, 1, 1) + np.roll(img, -1, 0)) / 4
    return img.astype(np.uint8)


def test_sharpest_frame_is_picked_from_the_window():
    frames = np.stack([_textured(1, 3), _textured(2, 0), _textured(3, 4), _textured(4, 5)])
    assert sharpness(frames[1]) > sharpness(frames[0])
    assert pick_start_frame(frames, 4) == 2
    assert pick_start_frame(frames, 1) == 0


def test_working_size_keeps_aspect_and_multiples():
    w, h = working_size(832, 480, 1024 * 1024, 16)
    assert w % 16 == 0 and h % 16 == 0
    assert abs(w / h - 832 / 480) < 0.03
    assert 0.9 < w * h / (1024 * 1024) < 1.1


def test_match_color_restores_source_statistics():
    source = _textured(5)
    shifted = np.clip(source.astype(int) + 40, 0, 255).astype(np.uint8)
    fixed = match_color(shifted, source)
    assert abs(fixed.mean() - source.mean()) < abs(shifted.mean() - source.mean()) / 4
    assert np.array_equal(match_color(shifted, source, 0), shifted)


def test_tag_prompts_keep_quality_tags_within_clip_limit():
    model = refiner_model("animagine-xl-4.0")
    tags = ", ".join(f"tag{i} word" for i in range(60))
    prompt, negative = refine_prompt(FrameRefine(model=model.id, prompt="unused", tags=tags), model)
    assert prompt.endswith(model.quality_tags)
    assert "tag0 word" in prompt and "tag59 word" not in prompt
    assert negative == model.negative


def test_unknown_refiner_is_rejected():
    with pytest.raises(Exception, match="unknown refiner"):
        refiner_model("nope")


def test_plan_attaches_refine_to_chained_jobs(tiny_plan):
    plan = tiny_plan(CHAIN, continuity={"refine": {"enabled": True, "strength": 0.4}})
    refine = plan.job("b").init.refine
    assert refine.model == LOOK_DEFAULTS["realistic"]
    assert refine.prompt == "A red kite high in the sky."
    assert refine.prompt_source == "end_prompt"
    assert refine.strength == 0.4
    assert plan.job("a").init is None


def test_anime_look_picks_anime_refiner_and_falls_back_to_shot(tiny_plan):
    shots = [{**CHAIN[0], "end_prompt": None}, CHAIN[1]]
    plan = tiny_plan(shots, look="anime", continuity={"refine": {"enabled": True}})
    refine = plan.job("b").init.refine
    assert refine.model == LOOK_DEFAULTS["anime"]
    assert refine.prompt_source == "shot"
    assert "kite" in refine.prompt


def test_shot_can_turn_refine_off_or_on(tiny_plan):
    off = tiny_plan(
        [CHAIN[0], {**CHAIN[1], "refine": False}], continuity={"refine": {"enabled": True}}
    )
    assert off.job("b").init.refine is None
    on = tiny_plan([CHAIN[0], {**CHAIN[1], "refine": True}])
    assert on.job("b").init.refine is not None


def test_unknown_refiner_in_spec_is_a_plan_problem(tiny_plan):
    with pytest.raises(PlanError, match="unknown refiner"):
        tiny_plan(CHAIN, continuity={"refine": {"enabled": True, "model": "nope"}})


def test_beats_give_each_segment_its_own_action_and_end(tiny_plan):
    shot = {
        "id": "long",
        "prompt": "A dog on a lawn.",
        "frames": 40,
        "beats": [
            {"action": "The dog runs to the gate.", "end": "A dog at a white gate."},
            "The dog jumps the gate.",
        ],
    }
    plan = tiny_plan([shot], continuity={"refine": {"enabled": True}})
    first, second = plan.jobs[0], plan.jobs[1]
    assert "runs to the gate" in first.prompt
    assert "jumps the gate" in second.prompt
    assert first.end_prompt == "A dog at a white gate."
    assert second.init.refine.prompt == "A dog at a white gate."


def test_runner_refines_handoff_and_stitch_drops_tail(tmp_path, registry, tiny_plan):
    plan = tiny_plan(
        CHAIN,
        continuity={"refine": {"enabled": True}, "handoff_skip": 2, "handoff_window": 1},
    )
    refiner = FakeRefiner()
    runner = Runner(FakeBackend(), runs_root=tmp_path, registry=registry, refiner=refiner)
    result = runner.start(plan, name="r")
    assert result.ok, result.failed
    assert len(refiner.calls) == 1
    job_dir = result.run_dir / "jobs" / "b"
    assert (job_dir / "start.png").is_file() and (job_dir / "start_source.png").is_file()
    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["handoff"]["drop_last"] == 2 and meta["handoff"]["refined"]

    again = Runner(FakeBackend(), runs_root=tmp_path, registry=registry, refiner=refiner)
    assert again.resume(result.run_dir).cached == ["a", "b"]
    assert len(refiner.calls) == 1

    out = stitch(result.run_dir)
    frames, _ = read_video(out.path)
    a, b = plan.job("a"), plan.job("b")
    assert len(frames) == a.frames - 2 + b.frames - 1


def test_refine_without_refiner_fails_the_job(tmp_path, registry, tiny_plan):
    plan = tiny_plan(CHAIN, continuity={"refine": {"enabled": True}})
    result = Runner(FakeBackend(), runs_root=tmp_path, registry=registry).start(plan, name="r")
    assert "no refiner" in result.failed["b"]


def test_long_shots_split_at_the_trained_length():
    ltx = get_profile("ltx-2.3")
    assert ltx.frames.max == 481 and ltx.frames.segment == 121
    result = snap(ltx, duration=15)
    assert result.is_split and max(result.segments) <= 121
    assert not snap(ltx, duration=15, segment_frames=481).is_split


def test_profiles_carry_prompt_rules():
    wan = get_profile("wan2.2-ti2v-5b")
    assert wan.prompt.max_tokens == 512 and wan.prompt.negative
    assert get_profile("ltx-2.5").prompt.i2v.value == "describe_first"
    assert get_profile("hunyuanvideo-1.5-480p").prompt.max_tokens == 1000
