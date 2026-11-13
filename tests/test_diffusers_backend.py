import sys
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from shotwright.cli import main
from shotwright.plan import PlanError, compile_plan
from shotwright.profiles import Registry
from shotwright.profiles.schema import Backends, VramPreset
from shotwright.run import BackendError, DiffusersBackend, JobInputs, OutOfMemory, RunDir, Runner
from shotwright.run.diffusers_backend import to_uint8_frames
from tests.conftest import TINY_PROFILE, tiny_spec

PROFILES = (
    TINY_PROFILE.replace(
        "    license: test",
        "    backends:\n"
        "      diffusers: StubPipeline\n"
        "      diffusers_repo: test/stub\n"
        "      diffusers_i2v: StubImageToVideoPipeline\n"
        "    license: test",
    )
    + """
  - id: tiny-i2v-only
    base: tiny-i2v
    conditioning: [i2v]
    backends:
      diffusers: StubImageToVideoPipeline
      diffusers_repo: test/stub-i2v
      diffusers_i2v: null
"""
)


class FakeOOM(Exception):
    pass


class FakeGenerator:
    def __init__(self, device="cpu"):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


def fake_torch(cuda=False):
    return SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
        float8_e4m3fn="fp8",
        Generator=FakeGenerator,
        OutOfMemoryError=FakeOOM,
        cuda=SimpleNamespace(is_available=lambda: cuda, empty_cache=lambda: None),
    )


class Transformer:
    def __init__(self):
        self.casting = None

    def enable_layerwise_casting(self, storage_dtype, compute_dtype):
        self.casting = (storage_dtype, compute_dtype)


class Vae:
    def __init__(self):
        self.tiled = False

    def enable_tiling(self):
        self.tiled = True


class StubPipeline:
    def __init__(self, repo, dtype, scale=1, oom=False):
        self.repo = repo
        self.dtype = dtype
        self.scale = scale
        self.oom = oom
        self.transformer = Transformer()
        self.vae = Vae()
        self.placement = None
        self.calls = []

    def to(self, device):
        self.placement = ("to", device)
        return self

    def enable_model_cpu_offload(self):
        self.placement = ("offload",)

    def enable_sequential_cpu_offload(self):
        self.placement = ("sequential",)

    def __call__(
        self,
        prompt,
        num_frames,
        height,
        width,
        generator,
        output_type,
        negative_prompt=None,
        guidance_scale=5.0,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "seed": generator.seed,
                "output_type": output_type,
                "negative_prompt": negative_prompt,
                "guidance_scale": guidance_scale,
            }
        )
        if self.oom:
            raise FakeOOM("out of memory")
        rng = np.random.default_rng(generator.seed)
        video = rng.random((num_frames, height * self.scale, width * self.scale, 3))
        return SimpleNamespace(frames=[video.astype(np.float32)])


class StubImageToVideoPipeline(StubPipeline):
    def __call__(
        self,
        image,
        prompt,
        num_frames,
        height,
        width,
        generator,
        output_type,
        negative_prompt=None,
        guidance_scale=5.0,
    ):
        out = super().__call__(
            prompt, num_frames, height, width, generator, output_type, negative_prompt
        )
        self.calls[-1]["image"] = image.size
        first = np.asarray(image.resize((width * self.scale, height * self.scale)))
        out.frames[0][0] = first.astype(np.float32) / 255
        return out


class Loader:
    def __init__(self, **options):
        self.options = options
        self.loaded = []

    def __call__(self, class_name, repo, dtype):
        cls = {"StubPipeline": StubPipeline, "StubImageToVideoPipeline": StubImageToVideoPipeline}[
            class_name
        ]
        pipe = cls(repo, dtype, **self.options)
        self.loaded.append(pipe)
        return pipe


@pytest.fixture
def stub_registry(tmp_path_factory) -> Registry:
    d = tmp_path_factory.mktemp("profiles")
    (d / "tiny.yaml").write_text(PROFILES)
    return Registry.load(extra_dirs=[d])


def backend(loader=None, **kw):
    return DiffusersBackend(loader=loader or Loader(), torch_module=fake_torch(), **kw)


def plan_for(registry, shots, **extra):
    return compile_plan(tiny_spec(shots, **extra), registry=registry)


def test_supports_needs_class_and_repo(stub_registry):
    profile = stub_registry.get("tiny-i2v")
    assert backend().supports(profile)
    assert not backend().supports(profile.model_copy(update={"backends": Backends()}))


def test_repo_override_enables_support(stub_registry):
    profile = stub_registry.get("tiny-i2v")
    bare = profile.model_copy(
        update={"backends": profile.backends.model_copy(update={"diffusers_repo": None})}
    )
    assert not backend().supports(bare)
    assert backend(repos={"tiny-i2v": "me/tiny"}).supports(bare)


def test_text_to_video_passes_job_settings(stub_registry):
    plan = plan_for(stub_registry, [{"id": "a", "prompt": "a boat", "duration": 1}])
    job = plan.jobs[0]
    loader = Loader()
    b = backend(loader)
    profile = stub_registry.get("tiny-i2v")
    b.prepare(profile, profile.preset_for(24))
    result = b.run(job, JobInputs())

    assert result.frames.shape == (job.frames, job.height, job.width, 3)
    assert result.frames.dtype == np.uint8
    assert result.fps == job.fps
    assert result.meta["pipeline"] == "StubPipeline"
    (pipe,) = loader.loaded
    assert pipe.repo == "test/stub"
    assert pipe.dtype == "bf16"
    assert pipe.placement == ("to", "cuda")
    call = pipe.calls[0]
    assert call["prompt"] == job.prompt
    assert call["seed"] == job.seed
    assert (call["num_frames"], call["height"], call["width"]) == (
        job.frames,
        job.height,
        job.width,
    )
    assert call["output_type"] == "np"


def test_unknown_kwargs_are_dropped_and_params_pass_through(stub_registry):
    plan = plan_for(
        stub_registry,
        [{"id": "a", "prompt": "p", "duration": 1, "params": {"guidance_scale": 3.5, "junk": 1}}],
    )
    loader = Loader()
    b = backend(loader)
    b.prepare(stub_registry.get("tiny-i2v"), None)
    b.run(plan.jobs[0], JobInputs())
    assert loader.loaded[0].calls[0]["guidance_scale"] == 3.5


def test_init_image_uses_i2v_pipeline_and_caches_it(stub_registry):
    plan = plan_for(stub_registry, [{"id": "a", "prompt": "p", "duration": 1}])
    job = plan.jobs[0]
    loader = Loader()
    b = backend(loader)
    b.prepare(stub_registry.get("tiny-i2v"), None)
    image = np.full((job.height, job.width, 3), 200, dtype=np.uint8)
    first = b.run(job, JobInputs(init_image=image))
    b.run(job, JobInputs(init_image=image))

    assert [type(p).__name__ for p in loader.loaded] == ["StubImageToVideoPipeline"]
    assert loader.loaded[0].calls[0]["image"] == (job.width, job.height)
    assert np.abs(first.frames[0].astype(int) - 200).max() <= 1


def test_image_only_model_refuses_text_to_video(stub_registry):
    b = backend()
    b.prepare(stub_registry.get("tiny-i2v-only"), None)
    plan = plan_for(stub_registry, [{"id": "a", "prompt": "p", "duration": 1}])
    with pytest.raises(BackendError, match="needs an init image"):
        b.run(plan.jobs[0], JobInputs())


def test_planner_rejects_cut_shot_without_image_on_image_only_model(stub_registry):
    with pytest.raises(PlanError, match="needs an init_image"):
        plan_for(
            stub_registry,
            [{"id": "a", "prompt": "p", "duration": 1, "model": "tiny-i2v-only"}],
        )


@pytest.mark.parametrize(
    ("recipe", "placement", "fp8", "tiled"),
    [
        ("full", ("to", "cuda"), False, False),
        ("offload", ("offload",), False, False),
        ("fp8+offload+vae_tiling", ("offload",), True, True),
        ("fp8+sequential", ("sequential",), True, False),
    ],
)
def test_recipes(stub_registry, recipe, placement, fp8, tiled):
    loader = Loader()
    b = backend(loader)
    b.prepare(stub_registry.get("tiny-i2v"), VramPreset(min_gb=1, recipe=recipe))
    plan = plan_for(stub_registry, [{"id": "a", "prompt": "p", "duration": 1}])
    b.run(plan.jobs[0], JobInputs())
    pipe = loader.loaded[0]
    assert pipe.placement == placement
    assert (pipe.transformer.casting == ("fp8", "bf16")) is fp8
    assert pipe.vae.tiled is tiled


def test_prepare_reloads_only_when_model_or_recipe_changes(stub_registry):
    loader = Loader()
    b = backend(loader)
    profile = stub_registry.get("tiny-i2v")
    job = plan_for(stub_registry, [{"id": "a", "prompt": "p", "duration": 1}]).jobs[0]
    b.prepare(profile, VramPreset(min_gb=24, recipe="full"))
    b.run(job, JobInputs())
    b.prepare(profile, VramPreset(min_gb=30, recipe="full"))
    b.run(job, JobInputs())
    assert len(loader.loaded) == 1
    b.prepare(profile, VramPreset(min_gb=8, recipe="offload"))
    b.run(job, JobInputs())
    assert len(loader.loaded) == 2


def test_run_before_prepare_fails(stub_registry):
    job = plan_for(stub_registry, [{"id": "a", "prompt": "p", "duration": 1}]).jobs[0]
    with pytest.raises(BackendError, match="prepare"):
        backend().run(job, JobInputs())


def test_torch_oom_becomes_out_of_memory(stub_registry):
    b = backend(Loader(oom=True))
    b.prepare(stub_registry.get("tiny-i2v"), None)
    job = plan_for(stub_registry, [{"id": "a", "prompt": "p", "duration": 1}]).jobs[0]
    with pytest.raises(OutOfMemory):
        b.run(job, JobInputs())


def test_wrong_output_size_is_resized(stub_registry):
    b = backend(Loader(scale=2))
    b.prepare(stub_registry.get("tiny-i2v"), None)
    job = plan_for(stub_registry, [{"id": "a", "prompt": "p", "duration": 1}]).jobs[0]
    result = b.run(job, JobInputs())
    assert result.frames.shape[1:3] == (job.height, job.width)
    assert result.meta["resized_from"] == [job.width * 2, job.height * 2]


def test_runner_end_to_end_with_chaining(tmp_path, stub_registry):
    plan = plan_for(
        stub_registry,
        [
            {"id": "a", "prompt": "first", "duration": 1},
            {"id": "b", "prompt": "second", "duration": 1, "transition": "continuous"},
            {"id": "c", "prompt": "third", "duration": 3},
        ],
    )
    loader = Loader()
    runner = Runner(
        backend(loader), runs_root=tmp_path / "runs", registry=stub_registry, vram_gb=24
    )
    result = runner.start(plan, name="r")
    assert result.ok, result.failed
    assert [type(p).__name__ for p in loader.loaded] == ["StubPipeline", "StubImageToVideoPipeline"]
    run = RunDir(result.run_dir)
    assert all(run.read_meta(j.id) for j in plan.jobs)


def test_to_uint8_frames_handles_layouts():
    signed = np.full((2, 3, 4, 5), -1.0, dtype=np.float32)
    out = to_uint8_frames(signed)
    assert out.shape == (2, 4, 5, 3)
    assert out.max() == 0
    grey = to_uint8_frames([np.ones((4, 5, 1), dtype=np.float32)] * 2)
    assert grey.shape == (2, 4, 5, 3)
    assert grey.min() == 255


def test_missing_torch_is_a_clear_error(monkeypatch):
    import importlib

    real = importlib.import_module

    def fake_import(name, *a, **kw):
        if name == "torch":
            raise ImportError("no torch")
        return real(name, *a, **kw)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(BackendError, match=r"shotwright\[diffusers\]"):
        _ = DiffusersBackend().torch


def test_cli_lists_diffusers_backend():
    out = CliRunner().invoke(main, ["run", "--help"])
    assert out.exit_code == 0
    assert "diffusers" in out.output
    assert "--device" in out.output


def test_i2v_from_base_pipe_keeps_dtype(stub_registry, monkeypatch):
    shared = {}

    class Base(StubPipeline):
        @classmethod
        def from_pretrained(cls, repo, torch_dtype):
            return cls(repo, torch_dtype)

    class FromPipe(StubImageToVideoPipeline):
        @classmethod
        def from_pipe(cls, base, **kwargs):
            shared["kwargs"] = kwargs
            return cls(base.repo, kwargs.get("torch_dtype"))

    fake = SimpleNamespace(__version__="0", StubPipeline=Base, StubImageToVideoPipeline=FromPipe)
    monkeypatch.setitem(sys.modules, "diffusers", fake)
    plan = plan_for(stub_registry, [{"id": "a", "prompt": "p", "duration": 1}])
    job = plan.jobs[0]
    b = DiffusersBackend(torch_module=fake_torch())
    b.prepare(stub_registry.get("tiny-i2v"), None)
    b.run(job, JobInputs(init_image=np.zeros((job.height, job.width, 3), dtype=np.uint8)))

    assert shared["kwargs"] == {"torch_dtype": "bf16"}


def test_separately_loaded_pipelines_are_not_kept_together(stub_registry):
    plan = plan_for(stub_registry, [{"id": "a", "prompt": "p", "duration": 1}])
    job = plan.jobs[0]
    loader = Loader()
    b = backend(loader)
    b.prepare(stub_registry.get("tiny-i2v"), None)
    image = np.zeros((job.height, job.width, 3), dtype=np.uint8)
    b.run(job, JobInputs())
    b.run(job, JobInputs(init_image=image))
    assert list(b._pipes) == ["i2v"]
    b.run(job, JobInputs())

    assert [type(p).__name__ for p in loader.loaded] == [
        "StubPipeline",
        "StubImageToVideoPipeline",
        "StubPipeline",
    ]
    assert list(b._pipes) == ["base"]
