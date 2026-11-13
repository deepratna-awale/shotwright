from pathlib import Path

import pytest
from click.testing import CliRunner

from shotwright.cli import main
from shotwright.plan import InitKind, Plan, PlanError, ShotList, compile_plan, derive_seed

EXAMPLE = Path(__file__).parent.parent / "examples" / "harbour.yaml"


def shot_list(shots, **extra):
    data = {
        "project": "test",
        "output": {"width": 1280, "height": 704, "fps": 24},
        "defaults": {"model": "wan2.2-ti2v-5b", "seed": 7},
        "shots": shots,
    }
    data.update(extra)
    return ShotList.from_dict(data)


def test_example_compiles():
    plan = compile_plan(ShotList.from_yaml(EXAMPLE))
    assert plan.project == "harbour-dawn"
    assert len(plan.chains()) == 3
    for job in plan.jobs:
        assert (job.frames - 1) % 4 == 0
        assert (job.width, job.height) == (1280, 704)


def test_cut_shots_are_independent():
    plan = compile_plan(
        shot_list(
            [
                {"id": "a", "prompt": "one", "duration": 4},
                {"id": "b", "prompt": "two", "duration": 4},
            ]
        )
    )
    assert [j.depends_on for j in plan.jobs] == [None, None]
    assert len(plan.chains()) == 2


def test_continuous_shot_chains_from_previous_last_frame():
    plan = compile_plan(
        shot_list(
            [
                {"id": "a", "prompt": "one", "duration": 4},
                {"id": "b", "prompt": "two", "duration": 4, "transition": "continuous"},
            ]
        )
    )
    b = plan.job("b")
    assert b.init.kind == InitKind.LAST_FRAME
    assert b.init.job == "a"
    assert (b.frames - 1) / b.fps >= 4


def test_long_shot_splits_into_chained_segments():
    plan = compile_plan(shot_list([{"id": "long", "prompt": "x", "duration": 12}]))
    ids = [j.id for j in plan.jobs]
    assert ids[0] == "long.01" and len(ids) >= 3
    for prev, job in zip(plan.jobs, plan.jobs[1:], strict=False):
        assert job.depends_on == prev.id
    assert plan.total_frames >= 12 * 24


def test_seeds_are_stable_and_independent():
    a = compile_plan(
        shot_list(
            [
                {"id": "a", "prompt": "one"},
                {"id": "b", "prompt": "two"},
            ]
        )
    )
    b = compile_plan(
        shot_list(
            [
                {"id": "a", "prompt": "changed prompt"},
                {"id": "b", "prompt": "two"},
                {"id": "c", "prompt": "new"},
            ]
        )
    )
    assert a.job("a").seed == b.job("a").seed
    assert a.job("b").seed == b.job("b").seed
    assert a.job("a").seed == derive_seed(7, "a", 0)
    assert a.job("a").seed != a.job("b").seed


def test_pinned_seed():
    plan = compile_plan(shot_list([{"id": "a", "prompt": "x", "seed": 42}]))
    assert plan.job("a").seed == 42


def test_style_and_negative_merge():
    plan = compile_plan(
        shot_list(
            [
                {"id": "a", "prompt": "A harbour"},
                {"id": "b", "prompt": "Boats.", "negative": "fog"},
            ],
            style="film grain",
            defaults={"model": "wan2.2-ti2v-5b", "negative": "blur", "params": {"steps": 30}},
        )
    )
    assert plan.job("a").prompt == "A harbour. film grain"
    assert plan.job("b").prompt == "Boats. film grain"
    assert plan.job("a").negative == "blur"
    assert plan.job("b").negative == "fog"
    assert plan.job("a").params == {"steps": 30}


def test_output_postprocess_recorded():
    plan = compile_plan(
        shot_list(
            [{"id": "a", "prompt": "x", "duration": 2}],
            output={"width": 1920, "height": 1080, "fps": 24},
        )
    )
    job = plan.job("a")
    assert (job.width, job.height) == (1280, 704)
    assert job.resize_to is not None


def problems_of(shots, **extra):
    with pytest.raises(PlanError) as exc:
        compile_plan(shot_list(shots, **extra))
    return exc.value.problems


def test_validation_collects_every_problem():
    problems = problems_of(
        [
            {"id": "a", "prompt": "x", "transition": "continuous"},
            {"id": "a", "prompt": "y"},
            {"id": "c", "prompt": "z", "model": "sora"},
        ]
    )
    text = " ".join(problems)
    assert "more than once" in text
    assert "cannot be continuous" in text
    assert "unknown model" in text


def test_continuous_into_text_only_model_rejected():
    problems = problems_of(
        [
            {"id": "a", "prompt": "x", "model": "wan2.1-t2v-14b"},
            {"id": "b", "prompt": "y", "model": "wan2.1-t2v-14b", "transition": "continuous"},
        ]
    )
    assert any("no image conditioning" in p for p in problems)


def test_init_image_checks(tmp_path):
    problems = problems_of(
        [
            {"id": "a", "prompt": "x", "init_image": "missing.png"},
            {"id": "b", "prompt": "y", "model": "wan2.1-t2v-1.3b", "init_image": "missing.png"},
        ]
    )
    assert any("not found" in p for p in problems)
    assert any("text-only" in p for p in problems)

    (tmp_path / "ref.png").write_bytes(b"x")
    spec = ShotList.from_dict(
        {
            "project": "t",
            "output": {"width": 1280, "height": 704, "fps": 24},
            "defaults": {"model": "wan2.2-ti2v-5b"},
            "shots": [{"id": "a", "prompt": "x", "init_image": "ref.png"}],
        },
        base_dir=tmp_path,
    )
    job = compile_plan(spec).job("a")
    assert job.init.kind == InitKind.IMAGE
    assert job.init.path == str(tmp_path / "ref.png")


def test_hop_limit():
    shots = [{"id": "s0", "prompt": "x", "duration": 4}] + [
        {"id": f"s{i}", "prompt": "x", "duration": 4, "transition": "continuous"}
        for i in range(1, 5)
    ]
    problems = problems_of(shots, continuity={"max_hops": 3})
    assert any("hop 4" in p for p in problems)
    compile_plan(shot_list(shots, continuity={"max_hops": 4}))


def test_plan_round_trip(tmp_path):
    plan = compile_plan(ShotList.from_yaml(EXAMPLE))
    path = tmp_path / "plan.json"
    plan.save(path)
    assert Plan.load(path) == plan


def test_dependants():
    plan = compile_plan(ShotList.from_yaml(EXAMPLE))
    assert plan.dependants("hands") == ["face"]
    assert set(plan.dependants("wide.01")) == {"wide.02", "deck.01", "deck.02"}


def test_cli_plan(tmp_path):
    out = tmp_path / "plan.json"
    result = CliRunner().invoke(main, ["plan", str(EXAMPLE), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert "harbour-dawn" in result.output
    assert Plan.load(out).project == "harbour-dawn"


def test_cli_plan_reports_problems(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "project: bad\noutput: {width: 1280, height: 704, fps: 24}\n"
        "shots:\n  - {id: a, prompt: x}\n"
    )
    result = CliRunner().invoke(main, ["plan", str(bad)])
    assert result.exit_code != 0
    assert "no default model" in result.output
