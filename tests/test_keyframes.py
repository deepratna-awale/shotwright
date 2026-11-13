import numpy as np
import pytest

from shotwright.media import load_image, read_video, save_image
from shotwright.plan import PlanError, compile_plan
from shotwright.profiles import Registry
from shotwright.run import BackendError, FakeBackend, JobInputs, RunDir, Runner
from shotwright.stitch import stitch
from tests.conftest import TINY_PROFILE, tiny_spec

FLF = (
    TINY_PROFILE
    + """
  - id: tiny-flf
    base: tiny-i2v
    conditioning: [t2v, i2v, flf2v]
    continuity: [keyframe_fill, last_frame_i2v]
"""
)


@pytest.fixture
def flf_registry(tmp_path_factory):
    d = tmp_path_factory.mktemp("profiles")
    (d / "tiny.yaml").write_text(FLF)
    return Registry.load(extra_dirs=[d])


@pytest.fixture
def keyframes(tmp_path):
    paths = {}
    for name, value in (("start", 30), ("mid", 140), ("end", 230)):
        path = tmp_path / f"{name}.png"
        save_image(path, np.full((48, 64, 3), value, dtype=np.uint8))
        paths[name] = str(path)
    return paths


def plan(registry, shots, **extra):
    return compile_plan(
        tiny_spec(shots, defaults={"model": "tiny-flf", "seed": 3}, **extra), registry=registry
    )


def test_end_image_goes_on_last_segment_only(flf_registry, keyframes):
    p = plan(
        flf_registry,
        [
            {
                "id": "a",
                "prompt": "x",
                "duration": 3,
                "init_image": keyframes["start"],
                "end_image": keyframes["end"],
            }
        ],
    )
    assert len(p.jobs) > 1
    assert [j.end_image for j in p.jobs[:-1]] == [None] * (len(p.jobs) - 1)
    assert p.jobs[-1].end_image == keyframes["end"]


def test_end_image_needs_flf_model(registry, keyframes):
    with pytest.raises(PlanError, match="first-last-frame"):
        compile_plan(
            tiny_spec(
                [
                    {
                        "id": "a",
                        "prompt": "x",
                        "duration": 1,
                        "init_image": keyframes["start"],
                        "end_image": keyframes["end"],
                    }
                ]
            ),
            registry=registry,
        )


def test_end_image_needs_a_start(flf_registry, keyframes):
    with pytest.raises(PlanError, match="nothing to start from"):
        plan(
            flf_registry, [{"id": "a", "prompt": "x", "duration": 1, "end_image": keyframes["end"]}]
        )


def test_missing_end_image_file(flf_registry, keyframes, tmp_path):
    with pytest.raises(PlanError, match="end_image not found"):
        plan(
            flf_registry,
            [
                {
                    "id": "a",
                    "prompt": "x",
                    "duration": 1,
                    "init_image": keyframes["start"],
                    "end_image": str(tmp_path / "nope.png"),
                }
            ],
        )


def test_keyframe_resets_hop_count(flf_registry, keyframes):
    shots = [{"id": "s0", "prompt": "x", "frames": 9}] + [
        {"id": f"s{i}", "prompt": "x", "frames": 9, "transition": "continuous"} for i in range(1, 6)
    ]
    with pytest.raises(PlanError, match="hop"):
        plan(flf_registry, shots, continuity={"max_hops": 3})
    shots[2]["end_image"] = keyframes["mid"]
    p = plan(flf_registry, shots, continuity={"max_hops": 3})
    assert p.job("s2").end_image == keyframes["mid"]


def test_run_lands_on_keyframes_and_chains_through_them(tmp_path, flf_registry, keyframes):
    p = plan(
        flf_registry,
        [
            {
                "id": "a",
                "prompt": "x",
                "duration": 1,
                "init_image": keyframes["start"],
                "end_image": keyframes["mid"],
            },
            {
                "id": "b",
                "prompt": "y",
                "duration": 1,
                "transition": "continuous",
                "end_image": keyframes["end"],
            },
        ],
    )
    result = Runner(FakeBackend(), runs_root=tmp_path / "runs", registry=flf_registry).start(
        p, name="r"
    )
    assert result.ok, result.failed
    run = RunDir(result.run_dir)
    a, _ = read_video(run.clip_path("a"))
    b, _ = read_video(run.clip_path("b"))
    assert abs(float(a[0].mean()) - 30) < 3
    assert abs(float(a[-1].mean()) - 140) < 3
    assert abs(float(b[0].mean()) - 140) < 3
    assert abs(float(b[-1].mean()) - 230) < 3
    stitched = stitch(run.root)
    assert not any(s.flagged for s in stitched.seams)


def test_changing_end_image_invalidates_cache(tmp_path, flf_registry, keyframes):
    shots = [
        {
            "id": "a",
            "prompt": "x",
            "duration": 1,
            "init_image": keyframes["start"],
            "end_image": keyframes["end"],
        }
    ]
    runner = Runner(FakeBackend(), runs_root=tmp_path / "runs", registry=flf_registry)
    first = runner.start(plan(flf_registry, shots), name="r")
    save_image(keyframes["end"], np.full((48, 64, 3), 10, dtype=np.uint8))
    backend = FakeBackend()
    again = Runner(backend, runs_root=tmp_path / "runs", registry=flf_registry).resume(
        first.run_dir
    )
    assert again.done == ["a"]
    last = load_image(RunDir(first.run_dir).last_frame_path("a"))
    assert abs(float(last.mean()) - 10) < 3


def _flf_job(flf_registry, keyframes):
    return plan(
        flf_registry,
        [
            {
                "id": "a",
                "prompt": "x",
                "duration": 1,
                "init_image": keyframes["start"],
                "end_image": keyframes["end"],
            }
        ],
    ).jobs[0]


def test_diffusers_passes_last_image(flf_registry, keyframes):
    from tests.test_diffusers_backend import StubImageToVideoPipeline, backend

    class FlfPipeline(StubImageToVideoPipeline):
        def __call__(
            self, image, last_image, prompt, num_frames, height, width, generator, output_type
        ):
            out = super().__call__(image, prompt, num_frames, height, width, generator, output_type)
            self.calls[-1]["last_image"] = last_image.size
            return out

    loaded = []

    def loader(class_name, repo, dtype):
        pipe = (FlfPipeline if repo == "t/flf" else StubImageToVideoPipeline)(repo, dtype)
        loaded.append(pipe)
        return pipe

    base = flf_registry.get("tiny-flf")
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    job = _flf_job(flf_registry, keyframes)

    for repo, ok in (("t/flf", True), ("t/plain", False)):
        profile = base.model_copy(
            update={
                "backends": base.backends.model_copy(
                    update={"diffusers": "StubImageToVideoPipeline", "diffusers_repo": repo}
                )
            }
        )
        b = backend(loader)
        b.prepare(profile, None)
        if ok:
            b.run(job, JobInputs(init_image=image, end_image=image))
            assert loaded[-1].calls[0]["last_image"] == (64, 48)
        else:
            with pytest.raises(BackendError, match="end image"):
                b.run(job, JobInputs(init_image=image, end_image=image))


def test_comfyui_uses_flf2v_workflow(tmp_path, flf_registry, keyframes):
    import json

    from tests.test_comfyui_backend import I2V, FakeComfy, make

    d = tmp_path / "wf"
    d.mkdir()
    flf = {**I2V, "6": {"class_type": "LoadImage", "inputs": {"image": "{{end_image}}"}}}
    (d / "tiny-flf.flf2v.json").write_text(json.dumps(flf))
    server = FakeComfy()
    b = make(d, server)
    b.prepare(flf_registry.get("tiny-flf"), None)
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    b.run(_flf_job(flf_registry, keyframes), JobInputs(init_image=image, end_image=image))
    graph = server.prompts[0]
    assert len(server.uploads) == 2
    assert graph["6"]["inputs"]["image"] in server.uploads
    assert graph["6"]["inputs"]["image"] != graph["5"]["inputs"]["image"]
    with pytest.raises(BackendError, match="image"):
        b.run(_flf_job(flf_registry, keyframes), JobInputs(end_image=image))
