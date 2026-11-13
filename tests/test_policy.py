import pytest

from shotwright.plan import PlanError
from shotwright.policy import PolicyViolation, check, violation
from shotwright.run import FakeBackend, Runner

ALLOWED = [
    "a child flies a kite on the beach",
    "two adults kiss on a balcony at sunset",
    "a nude marble statue in a museum hall",
    "a baby elephant splashes in a river",
    "the boxer is forced back against the ropes",
    "a family having breakfast, kids laughing",
]
BLOCKED_MINOR = [
    "a nude child",
    "sexy schoolgirl in a classroom",
    "explicit scene with a 15 year old",
    "N.U.D.E toddler",
    "erotic photo, 16yo",
]
BLOCKED_CONSENT = [
    "she is drugged and undressed",
    "hidden camera nude footage in a changing room",
    "a non-consensual sex scene",
]


@pytest.mark.parametrize("text", ALLOWED)
def test_allowed(text):
    assert violation(text) is None


@pytest.mark.parametrize("text", BLOCKED_MINOR)
def test_blocks_minors(text):
    assert violation(text) == "sexual content involving minors"


@pytest.mark.parametrize("text", BLOCKED_CONSENT)
def test_blocks_non_consent(text):
    assert violation(text) == "non-consensual sexual content"


def test_style_counts_as_part_of_the_prompt():
    assert violation("a child in a park", "erotic") is not None


def test_check_raises():
    with pytest.raises(PolicyViolation, match="shot x"):
        check("nude child", where="shot x")


def test_planner_rejects(tiny_plan):
    with pytest.raises(PlanError, match="not allowed"):
        tiny_plan([{"id": "a", "prompt": "a naked kid", "duration": 1}])


def test_negative_prompt_is_ignored(tiny_plan):
    tiny_plan(
        [{"id": "a", "prompt": "a child reading", "duration": 1, "negative": "nude, explicit"}]
    )


def test_runner_rejects_adapted_prompt(tmp_path, registry, tiny_plan):
    plan = tiny_plan(
        [
            {"id": "a", "prompt": "a park", "duration": 1},
            {"id": "b", "prompt": "a park", "duration": 1, "transition": "continuous"},
        ]
    )
    runner = Runner(
        FakeBackend(),
        runs_root=tmp_path / "runs",
        registry=registry,
        prompt_hook=lambda job, inputs: "a naked child in a park",
    )
    result = runner.start(plan, name="r")
    assert "b" in result.failed
    assert "not allowed" in result.failed["b"]
