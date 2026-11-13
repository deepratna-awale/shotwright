import json

import numpy as np
import pytest

from shotwright.media import write_video
from shotwright.run import BackendError, HostedBackend, JobInputs, RunDir, Runner, TransientError
from shotwright.run.hosted_backend import conform, dig

FAL_SPEC = {
    "provider": "fal",
    "app": "fal-ai/some-model/text-to-video",
    "input": {
        "prompt": "{{prompt}}",
        "negative_prompt": "{{negative}}",
        "num_frames": "{{frames}}",
        "seed": "{{seed}}",
        "aspect_ratio": "{{aspect_ratio}}",
        "steps": "{{steps|30}}",
    },
    "video": "video.url",
}
REPLICATE_SPEC = {
    "provider": "replicate",
    "app": "owner/model",
    "input": {"prompt": "{{prompt}}", "image": "{{image}}", "seconds": "{{seconds}}"},
    "video": "output",
}


class FakeApis:
    def __init__(self, tmp_path, frames, width, height, polls=1, fail=None):
        self.tmp_path = tmp_path
        self.shape = (frames, height, width)
        self.polls = polls
        self.fail = fail
        self.posts = []
        self.gets = []

    def _video(self):
        n, h, w = self.shape
        rng = np.random.default_rng(0)
        path = self.tmp_path / "out.mp4"
        write_video(path, rng.integers(0, 256, (n, h, w, 3), dtype=np.uint8), 24)
        return path.read_bytes()

    def post(self, url, payload, headers):
        self.posts.append((url, payload, headers))
        if url.startswith("https://queue.fal.run/"):
            return {
                "request_id": "r1",
                "status_url": "https://queue.fal.run/x/requests/r1/status",
                "response_url": "https://queue.fal.run/x/requests/r1",
            }
        return {"id": "p1", "status": "starting", "urls": {"get": "https://api.replicate.com/p1"}}

    def get(self, url, headers):
        self.gets.append((url, headers))
        if url.endswith("/status"):
            if self.polls > 0:
                self.polls -= 1
                return json.dumps({"status": "IN_PROGRESS"}).encode()
            if self.fail:
                return json.dumps({"status": "FAILED", "error": self.fail}).encode()
            return json.dumps({"status": "COMPLETED"}).encode()
        if url == "https://queue.fal.run/x/requests/r1":
            return json.dumps({"video": {"url": "https://cdn/v.mp4"}}).encode()
        if url == "https://api.replicate.com/p1":
            if self.polls > 0:
                self.polls -= 1
                return json.dumps(
                    {"status": "processing", "urls": {"get": "https://api.replicate.com/p1"}}
                ).encode()
            if self.fail:
                return json.dumps({"status": "failed", "error": self.fail}).encode()
            return json.dumps({"status": "succeeded", "output": "https://cdn/v.mp4"}).encode()
        if url == "https://cdn/v.mp4":
            assert headers == {}
            return self._video()
        raise AssertionError(url)


@pytest.fixture
def specs(tmp_path):
    d = tmp_path / "specs"
    d.mkdir()
    (d / "tiny-i2v.t2v.json").write_text(json.dumps(FAL_SPEC))
    (d / "tiny-i2v.i2v.json").write_text(json.dumps(REPLICATE_SPEC))
    return d


def make(specs, apis, **kw):
    return HostedBackend(
        spec_dirs=[specs],
        keys={"fal": "fk", "replicate": "rk"},
        transport=apis,
        sleep=lambda s: None,
        **kw,
    )


def job_for(tiny_plan):
    return tiny_plan([{"id": "a", "prompt": "a boat", "duration": 1}]).jobs[0]


def test_dig():
    assert dig({"a": [{"b": 1}]}, "a.0.b") == 1
    assert dig("x", "") == "x"


def test_fal_text_to_video(tmp_path, specs, registry, tiny_plan):
    job = job_for(tiny_plan)
    apis = FakeApis(tmp_path, job.frames, job.width, job.height, polls=2)
    b = make(specs, apis)
    profile = registry.get("tiny-i2v")
    assert b.supports(profile)
    b.prepare(profile, None)
    result = b.run(job, JobInputs())
    url, payload, headers = apis.posts[0]
    assert url == "https://queue.fal.run/fal-ai/some-model/text-to-video"
    assert headers == {"Authorization": "Key fk"}
    assert payload["num_frames"] == job.frames and payload["seed"] == job.seed
    assert payload["steps"] == 30 and payload["aspect_ratio"] == "4:3"
    assert result.frames.shape == (job.frames, job.height, job.width, 3)
    assert result.meta["provider"] == "fal"


def test_replicate_image_to_video_with_conform(tmp_path, specs, registry, tiny_plan):
    job = job_for(tiny_plan)
    apis = FakeApis(tmp_path, job.frames + 7, job.width * 2, job.height * 2 + 32, polls=1)
    b = make(specs, apis)
    b.prepare(registry.get("tiny-i2v"), None)
    image = np.full((job.height, job.width, 3), 9, dtype=np.uint8)
    result = b.run(job, JobInputs(init_image=image))
    url, body, headers = apis.posts[0]
    assert url == "https://api.replicate.com/v1/models/owner/model/predictions"
    assert headers == {"Authorization": "Bearer rk"}
    assert body["input"]["image"].startswith("data:image/png;base64,")
    assert body["input"]["seconds"] == round(job.frames / job.fps, 3)
    assert result.frames.shape == (job.frames, job.height, job.width, 3)
    assert result.meta["frames_returned"] == job.frames + 7
    assert "resized_from" in result.meta


def test_replicate_version_ids(tmp_path, specs, registry, tiny_plan):
    spec = {**REPLICATE_SPEC, "app": "owner/model:abc123"}
    (specs / "tiny-i2v.i2v.json").write_text(json.dumps(spec))
    job = job_for(tiny_plan)
    apis = FakeApis(tmp_path, job.frames, job.width, job.height)
    b = make(specs, apis)
    b.prepare(registry.get("tiny-i2v"), None)
    b.run(job, JobInputs(init_image=np.zeros((job.height, job.width, 3), dtype=np.uint8)))
    url, body, _ = apis.posts[0]
    assert url == "https://api.replicate.com/v1/predictions"
    assert body["version"] == "abc123"


@pytest.mark.parametrize("mode", ["t2v", "i2v"])
def test_provider_failures(tmp_path, specs, registry, tiny_plan, mode):
    job = job_for(tiny_plan)
    apis = FakeApis(tmp_path, job.frames, job.width, job.height, fail="nsfw filter")
    b = make(specs, apis)
    b.prepare(registry.get("tiny-i2v"), None)
    image = np.zeros((job.height, job.width, 3), dtype=np.uint8) if mode == "i2v" else None
    with pytest.raises(BackendError, match="nsfw filter"):
        b.run(job, JobInputs(init_image=image))


def test_timeout_and_missing_key(tmp_path, specs, registry, tiny_plan, monkeypatch):
    job = job_for(tiny_plan)
    b = make(specs, FakeApis(tmp_path, 9, 64, 48, polls=10**9), timeout=0)
    b.prepare(registry.get("tiny-i2v"), None)
    with pytest.raises(TransientError):
        b.run(job, JobInputs())
    monkeypatch.delenv("FAL_KEY", raising=False)
    nokey = HostedBackend(spec_dirs=[specs], transport=FakeApis(tmp_path, 9, 64, 48))
    nokey.prepare(registry.get("tiny-i2v"), None)
    with pytest.raises(BackendError, match="FAL_KEY"):
        nokey.run(job, JobInputs())


def test_bad_spec_and_missing_mode(tmp_path, registry, tiny_plan):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "tiny-i2v.t2v.json").write_text(json.dumps({**FAL_SPEC, "provider": "elsewhere"}))
    b = HostedBackend(spec_dirs=[d], transport=FakeApis(tmp_path, 9, 64, 48))
    with pytest.raises(BackendError, match="provider"):
        b.supports(registry.get("tiny-i2v"))
    assert not HostedBackend(spec_dirs=[tmp_path]).supports(registry.get("tiny-i2v"))


def test_conform_crops_to_aspect(tiny_plan):
    job = job_for(tiny_plan)
    wide = np.zeros((job.frames, job.height, job.width * 2, 3), dtype=np.uint8)
    wide[:, :, job.width // 2 : job.width // 2 + job.width] = 200
    meta = {}
    out = conform(wide, job, meta)
    assert out.shape == (job.frames, job.height, job.width, 3)
    assert out.mean() > 190


def test_runner_end_to_end(tmp_path, specs, registry, tiny_plan):
    plan = tiny_plan(
        [
            {"id": "a", "prompt": "first", "duration": 1},
            {"id": "b", "prompt": "second", "duration": 1, "transition": "continuous"},
        ]
    )
    apis = FakeApis(tmp_path, plan.jobs[0].frames, plan.jobs[0].width, plan.jobs[0].height)
    result = Runner(make(specs, apis), runs_root=tmp_path / "runs", registry=registry).start(
        plan, name="r"
    )
    assert result.ok, result.failed
    assert RunDir(result.run_dir).read_meta("b")["backend_meta"]["provider"] == "replicate"
