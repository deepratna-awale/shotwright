import io
import json
import urllib.parse

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from shotwright.cli import main
from shotwright.media import write_video
from shotwright.profiles import default_registry
from shotwright.profiles.schema import VramPreset
from shotwright.run import (
    BackendError,
    ComfyUIBackend,
    JobInputs,
    OutOfMemory,
    RunDir,
    Runner,
    TransientError,
)
from shotwright.run.comfyui_backend import HttpTransport, fill

T2V = {
    "1": {"class_type": "Text", "inputs": {"text": "{{prompt}}", "neg": "{{negative}}"}},
    "2": {
        "class_type": "Latent",
        "inputs": {"width": "{{width}}", "height": "{{height}}", "length": "{{frames}}"},
    },
    "3": {"class_type": "Sampler", "inputs": {"seed": "{{seed}}", "steps": "{{steps|4}}"}},
    "4": {"class_type": "SaveImage", "inputs": {"filename_prefix": "{{prefix}}"}},
}
I2V = {**T2V, "5": {"class_type": "LoadImage", "inputs": {"image": "{{image}}"}}}


def png(frame):
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeComfy:
    """Mimics the parts of the ComfyUI HTTP API the backend uses."""

    def __init__(self, polls=1, error=None, video=False, scale=1, tmp_path=None):
        self.polls = polls
        self.error = error
        self.video = video
        self.scale = scale
        self.tmp_path = tmp_path
        self.prompts = []
        self.uploads = {}
        self.files = {}
        self.history = {}
        self.freed = 0
        self.history_calls = 0

    def upload(self, path, filename, data):
        assert path == "/upload/image"
        self.uploads[filename] = data
        return {"name": filename, "subfolder": "", "type": "input"}

    def post_json(self, path, payload):
        if path == "/free":
            self.freed += 1
            return {}
        assert path == "/prompt"
        graph = payload["prompt"]
        self.prompts.append(graph)
        prompt_id = f"p{len(self.prompts)}"
        self._render(prompt_id, graph)
        return {"prompt_id": prompt_id, "number": len(self.prompts), "node_errors": {}}

    def _render(self, prompt_id, graph):
        latent = graph["2"]["inputs"]
        w, h, n = latent["width"] * self.scale, latent["height"] * self.scale, latent["length"]
        rng = np.random.default_rng(graph["3"]["inputs"]["seed"])
        frames = rng.integers(0, 256, (n, h, w, 3), dtype=np.uint8)
        if "5" in graph:
            start = Image.open(io.BytesIO(self.uploads[graph["5"]["inputs"]["image"]]))
            frames[0] = np.asarray(start.convert("RGB").resize((w, h)))
        prefix = graph["4"]["inputs"]["filename_prefix"]
        if self.error:
            outputs = {}
            status = {
                "status_str": "error",
                "completed": False,
                "messages": [
                    ["execution_start", {}],
                    [
                        "execution_error",
                        {"node_type": "KSampler", **self.error},
                    ],
                ],
            }
        elif self.video:
            path = self.tmp_path / f"{prompt_id}.mp4"
            write_video(path, frames, 24)
            name = f"{prefix}_00001.mp4"
            self.files[name] = path.read_bytes()
            outputs = {"9": {"gifs": [{"filename": name, "subfolder": "", "type": "output"}]}}
            status = {"status_str": "success", "completed": True, "messages": []}
        else:
            images = []
            for i, frame in enumerate(frames):
                name = f"{prefix}_{i:05d}_.png"
                self.files[name] = png(frame)
                images.append({"filename": name, "subfolder": "", "type": "output"})
            preview = {"filename": "preview.png", "subfolder": "", "type": "temp"}
            self.files["preview.png"] = png(frames[0])
            outputs = {"4": {"images": images}, "8": {"images": [preview]}}
            status = {"status_str": "success", "completed": True, "messages": []}
        self.history[prompt_id] = {"outputs": outputs, "status": status, "_left": self.polls}
        self.frames = frames

    def get(self, path):
        if path.startswith("/history/"):
            self.history_calls += 1
            prompt_id = path.rsplit("/", 1)[1]
            entry = self.history[prompt_id]
            if entry["_left"] > 0:
                entry["_left"] -= 1
                return b"{}"
            return json.dumps({prompt_id: entry}).encode()
        if path.startswith("/view?"):
            query = urllib.parse.parse_qs(path.split("?", 1)[1])
            return self.files[query["filename"][0]]
        raise AssertionError(path)


@pytest.fixture
def workflows(tmp_path):
    d = tmp_path / "workflows"
    d.mkdir()
    (d / "tiny-i2v.t2v.json").write_text(json.dumps(T2V))
    (d / "tiny-i2v.i2v.json").write_text(json.dumps(I2V))
    return d


def make(workflows, server, **kw):
    return ComfyUIBackend(workflow_dirs=[workflows], transport=server, sleep=lambda s: None, **kw)


def first_job(tiny_plan, **shot):
    return tiny_plan([{"id": "a", "prompt": "a boat", "duration": 1, **shot}]).jobs[0]


def test_fill_types_and_defaults():
    graph = {"a": "{{n}}", "b": "x {{n}} y", "c": ['{{s|"uni_pc"}}', "{{k|3.5}}"], "d": 1}
    assert fill(graph, {"n": 7}) == {"a": 7, "b": "x 7 y", "c": ["uni_pc", 3.5], "d": 1}
    assert fill({"k": "{{k|3.5}}"}, {"k": 9}) == {"k": 9}
    with pytest.raises(BackendError, match="missing"):
        fill({"a": "{{missing}}"}, {})


def test_text_to_video_round_trip(tiny_plan, registry, workflows):
    server = FakeComfy(polls=2)
    b = make(workflows, server)
    profile = registry.get("tiny-i2v")
    assert b.supports(profile)
    b.prepare(profile, None)
    job = first_job(tiny_plan, params={"steps": 12})
    result = b.run(job, JobInputs())

    graph = server.prompts[0]
    assert graph["1"]["inputs"] == {"text": job.prompt, "neg": job.negative or ""}
    assert graph["2"]["inputs"] == {"width": job.width, "height": job.height, "length": job.frames}
    assert graph["3"]["inputs"] == {"seed": job.seed, "steps": 12}
    assert graph["4"]["inputs"]["filename_prefix"] == "shotwright/a"
    assert server.history_calls == 3
    assert result.frames.shape == (job.frames, job.height, job.width, 3)
    assert np.array_equal(result.frames, server.frames)
    assert result.meta["prompt_id"] == "p1"


def test_image_to_video_uploads_init_frame(tiny_plan, registry, workflows):
    server = FakeComfy()
    b = make(workflows, server)
    b.prepare(registry.get("tiny-i2v"), None)
    job = first_job(tiny_plan)
    image = np.full((job.height, job.width, 3), 77, dtype=np.uint8)
    result = b.run(job, JobInputs(init_image=image))
    (name,) = server.uploads
    assert server.prompts[0]["5"]["inputs"]["image"] == name
    assert np.abs(result.frames[0].astype(int) - 77).max() <= 1


def test_video_output_is_decoded(tiny_plan, registry, workflows, tmp_path):
    server = FakeComfy(video=True, tmp_path=tmp_path)
    b = make(workflows, server)
    b.prepare(registry.get("tiny-i2v"), None)
    job = first_job(tiny_plan)
    result = b.run(job, JobInputs())
    assert result.frames.shape == (job.frames, job.height, job.width, 3)
    assert result.meta["output"].endswith(".mp4")


def test_wrong_size_is_resized(tiny_plan, registry, workflows):
    b = make(workflows, FakeComfy(scale=2))
    b.prepare(registry.get("tiny-i2v"), None)
    job = first_job(tiny_plan)
    result = b.run(job, JobInputs())
    assert result.frames.shape[1:3] == (job.height, job.width)
    assert result.meta["resized_from"] == [job.width * 2, job.height * 2]


def test_oom_and_other_errors(tiny_plan, registry, workflows):
    job = first_job(tiny_plan)
    oom = make(
        workflows,
        FakeComfy(error={"exception_type": "torch.OutOfMemoryError", "exception_message": "CUDA"}),
    )
    oom.prepare(registry.get("tiny-i2v"), None)
    with pytest.raises(OutOfMemory, match="KSampler"):
        oom.run(job, JobInputs())
    broken = make(
        workflows, FakeComfy(error={"exception_type": "ValueError", "exception_message": "bad"})
    )
    broken.prepare(registry.get("tiny-i2v"), None)
    with pytest.raises(BackendError, match="bad") as info:
        broken.run(job, JobInputs())
    assert not isinstance(info.value, OutOfMemory)


def test_timeout_is_transient(tiny_plan, registry, workflows):
    b = make(workflows, FakeComfy(polls=10**9), timeout=0)
    b.prepare(registry.get("tiny-i2v"), None)
    with pytest.raises(TransientError):
        b.run(first_job(tiny_plan), JobInputs())


def test_recipe_specific_workflow_wins(tiny_plan, registry, workflows):
    special = json.loads(json.dumps(T2V))
    special["3"]["inputs"]["steps"] = "{{steps|99}}"
    (workflows / "tiny-i2v+offload.t2v.json").write_text(json.dumps(special))
    server = FakeComfy()
    b = make(workflows, server)
    b.prepare(registry.get("tiny-i2v"), VramPreset(min_gb=8, recipe="offload"))
    b.run(first_job(tiny_plan), JobInputs())
    assert server.prompts[0]["3"]["inputs"]["steps"] == 99


def test_unsupported_without_workflow(registry, tmp_path):
    b = ComfyUIBackend(workflow_dirs=[tmp_path], transport=FakeComfy())
    assert not b.supports(registry.get("tiny-i2v"))


def test_bundled_wan_workflows_are_complete():
    b = ComfyUIBackend(transport=FakeComfy())
    profile = default_registry().get("wan2.2-ti2v-5b")
    assert b.supports(profile)
    b.prepare(profile, None)
    values = {
        "prompt": "p",
        "negative": "n",
        "seed": 1,
        "frames": 121,
        "width": 1280,
        "height": 704,
        "fps": 24,
        "prefix": "x",
        "image": "in.png",
    }
    t2v = fill(b._find(profile, "t2v"), values)
    i2v = fill(b._find(profile, "i2v"), values)
    assert "{{" not in json.dumps(t2v) + json.dumps(i2v)
    assert i2v["7"]["inputs"]["start_image"] == ["10", 0]
    assert "start_image" not in t2v["7"]["inputs"]
    for graph in (t2v, i2v):
        for node in graph.values():
            for value in node["inputs"].values():
                if isinstance(value, list):
                    assert value[0] in graph


def test_release_frees_memory(workflows):
    server = FakeComfy()
    make(workflows, server).release()
    assert server.freed == 1


def test_runner_end_to_end(tmp_path, tiny_plan, registry, workflows):
    plan = tiny_plan(
        [
            {"id": "a", "prompt": "first", "duration": 1},
            {"id": "b", "prompt": "second", "duration": 1, "transition": "continuous"},
        ]
    )
    server = FakeComfy()
    result = Runner(make(workflows, server), runs_root=tmp_path / "runs", registry=registry).start(
        plan, name="r"
    )
    assert result.ok, result.failed
    assert len(server.uploads) == 1
    assert RunDir(result.run_dir).read_meta("b")


def test_unreachable_server_is_transient():
    transport = HttpTransport("http://127.0.0.1:9", timeout=1)
    with pytest.raises(TransientError):
        transport.get("/history/x")


def test_cli_lists_comfyui_backend():
    out = CliRunner().invoke(main, ["run", "--help"])
    assert "comfyui" in out.output
    assert "--comfy-url" in out.output
