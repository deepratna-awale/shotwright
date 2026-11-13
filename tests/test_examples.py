from pathlib import Path

import pytest

from shotwright.audio import FakeVoice, voice_over
from shotwright.plan import ShotList, compile_plan
from shotwright.run import FakeBackend, Runner
from shotwright.stitch import stitch

EXAMPLES = sorted((Path(__file__).parent.parent / "examples").glob("*.yaml"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_examples_compile(path):
    plan = compile_plan(ShotList.from_yaml(path))
    assert plan.jobs


def test_harbour_end_to_end(tmp_path):
    spec = ShotList.from_yaml(Path(__file__).parent.parent / "examples" / "harbour.yaml")
    spec = spec.model_copy(
        update={"output": spec.output.model_copy(update={"width": 160, "height": 88})}
    )
    spec.shots = [s.model_copy(update={"duration": 0.5}) for s in spec.shots]
    plan = compile_plan(spec)
    result = Runner(FakeBackend(), runs_root=tmp_path).start(plan, name="harbour")
    assert result.ok, result.failed
    stitched = stitch(result.run_dir, color="anchor")
    assert stitched.frames > 0
    voiced = voice_over(result.run_dir, FakeVoice(seconds_per_char=0.01))
    assert len(voiced.lines) == 2
