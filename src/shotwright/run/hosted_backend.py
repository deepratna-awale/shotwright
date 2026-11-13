from __future__ import annotations

import base64
import copy
import io
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from ..media import read_video
from ..plan import Job
from ..profiles import ModelProfile, VramPreset
from .backend import BackendError, ClipResult, JobInputs, TransientError
from .comfyui_backend import fill

SPEC_PATH_ENV = "SHOTWRIGHT_HOSTED_PATH"
PROVIDERS = ("fal", "replicate")
KEY_ENV = {"fal": "FAL_KEY", "replicate": "REPLICATE_API_TOKEN"}


class Transport(Protocol):
    def get(self, url: str, headers: dict[str, str]) -> bytes: ...

    def post(
        self, url: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]: ...


class HttpTransport:
    def __init__(self, timeout: float = 120.0):
        self.timeout = timeout

    def _open(self, request: urllib.request.Request) -> bytes:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:1000]
            if exc.code in (408, 429) or exc.code >= 500:
                raise TransientError(f"{request.full_url} returned {exc.code}: {body}") from exc
            raise BackendError(f"{request.full_url} returned {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise TransientError(f"cannot reach {request.full_url}: {exc}") from exc

    def get(self, url: str, headers: dict[str, str]) -> bytes:
        return self._open(urllib.request.Request(url, headers=headers))

    def post(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", **headers},
        )
        return json.loads(self._open(request))


def data_uri(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def dig(value: Any, path: str) -> Any:
    for part in path.split(".") if path else []:
        if isinstance(value, list):
            value = value[int(part)]
        elif isinstance(value, dict):
            value = value[part]
        else:
            raise KeyError(part)
    return value


class HostedBackend:
    """Runs jobs on hosted inference APIs that serve open-weight models.

    Each model and mode needs a spec file `<profile id>.<t2v|i2v|flf2v>.json`:

        {"provider": "fal", "app": "<app id>", "input": {...}, "video": "video.url"}

    `input` takes the same {{placeholders}} as ComfyUI workflows, plus {{image}} and
    {{end_image}} as PNG data URIs. `video` is the dotted path to the video URL in the
    finished result.
    """

    name = "hosted"

    def __init__(
        self,
        spec_dirs: Iterable[str | Path] = (),
        keys: dict[str, str] | None = None,
        transport: Transport | None = None,
        poll_interval: float = 3.0,
        timeout: float = 3600.0,
        sleep=time.sleep,
    ):
        dirs = [Path(d) for d in spec_dirs]
        env = os.environ.get(SPEC_PATH_ENV)
        if env:
            dirs += [Path(p) for p in env.split(os.pathsep) if p]
        self.spec_dirs = dirs
        self.keys = {p: (keys or {}).get(p) or os.environ.get(KEY_ENV[p]) for p in PROVIDERS}
        self.transport = transport or HttpTransport()
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._sleep = sleep
        self._profile: ModelProfile | None = None

    def _spec(self, profile: ModelProfile, mode: str) -> dict[str, Any] | None:
        ids = [profile.id] + ([profile.base] if profile.base else [])
        for model_id in ids:
            for directory in self.spec_dirs:
                path = directory / f"{model_id}.{mode}.json"
                if path.is_file():
                    spec = json.loads(path.read_text(encoding="utf-8"))
                    if spec.get("provider") not in PROVIDERS:
                        raise BackendError(f"{path}: provider must be one of {PROVIDERS}")
                    return spec
        return None

    def supports(self, profile: ModelProfile) -> bool:
        return any(self._spec(profile, m) is not None for m in ("t2v", "i2v", "flf2v"))

    def prepare(self, profile: ModelProfile, preset: VramPreset | None) -> None:
        self._profile = profile

    def release(self) -> None:
        pass

    def _headers(self, provider: str) -> dict[str, str]:
        key = self.keys.get(provider)
        if not key:
            raise BackendError(f"set {KEY_ENV[provider]} to use {provider}")
        return {"Authorization": f"Key {key}" if provider == "fal" else f"Bearer {key}"}

    def _deadline_passed(self, deadline: float, what: str) -> None:
        if time.monotonic() > deadline:
            raise TransientError(f"{what} did not finish in time")
        self._sleep(self.poll_interval)

    def _fal(self, app: str, payload: dict[str, Any], headers: dict[str, str]) -> Any:
        queued = self.transport.post(f"https://queue.fal.run/{app}", payload, headers)
        status_url, response_url = queued.get("status_url"), queued.get("response_url")
        if not status_url or not response_url:
            raise BackendError(f"fal did not queue the request: {queued}")
        deadline = time.monotonic() + self.timeout
        while True:
            status = json.loads(self.transport.get(status_url, headers))
            state = status.get("status")
            if state == "COMPLETED":
                if status.get("error"):
                    raise BackendError(f"fal request failed: {status['error']}")
                return json.loads(self.transport.get(response_url, headers))
            if state not in ("IN_QUEUE", "IN_PROGRESS"):
                raise BackendError(f"fal request failed: {status}")
            self._deadline_passed(deadline, "fal request")

    def _replicate(self, app: str, payload: dict[str, Any], headers: dict[str, str]) -> Any:
        if ":" in app:
            url, body = (
                "https://api.replicate.com/v1/predictions",
                {
                    "version": app.split(":", 1)[1],
                    "input": payload,
                },
            )
        else:
            url, body = f"https://api.replicate.com/v1/models/{app}/predictions", {"input": payload}
        prediction = self.transport.post(url, body, headers)
        deadline = time.monotonic() + self.timeout
        while True:
            state = prediction.get("status")
            if state == "succeeded":
                return prediction
            if state in ("failed", "canceled"):
                raise BackendError(f"replicate prediction {state}: {prediction.get('error')}")
            get_url = (prediction.get("urls") or {}).get("get")
            if not get_url:
                raise BackendError(f"replicate did not return a prediction URL: {prediction}")
            self._deadline_passed(deadline, "replicate prediction")
            prediction = json.loads(self.transport.get(get_url, headers))

    def run(self, job: Job, inputs: JobInputs) -> ClipResult:
        profile = self._profile
        if profile is None:
            raise BackendError("prepare() must be called before run()")
        if inputs.context_frames is not None:
            raise BackendError(
                "the hosted backend does not support overlap context yet; "
                "set continuity.overlap to 0"
            )
        if inputs.end_image is not None:
            mode = "flf2v"
        elif inputs.init_image is not None:
            mode = "i2v"
        else:
            mode = "t2v"
        spec = self._spec(profile, mode)
        if spec is None:
            raise BackendError(f"no hosted {mode} spec for {profile.id}")
        provider = spec["provider"]
        headers = self._headers(provider)

        values: dict[str, Any] = {
            "prompt": job.prompt,
            "negative": job.negative or "",
            "seed": job.seed,
            "frames": job.frames,
            "width": job.width,
            "height": job.height,
            "fps": job.fps,
            "seconds": round(job.frames / job.fps, 3),
            "resolution": f"{min(job.width, job.height)}p",
            "aspect_ratio": _aspect(job.width, job.height),
        }
        values.update(job.params)
        if inputs.init_image is not None:
            values["image"] = data_uri(inputs.init_image)
        if inputs.end_image is not None:
            values["end_image"] = data_uri(inputs.end_image)
        payload = fill(copy.deepcopy(spec.get("input", {})), values)

        call = self._fal if provider == "fal" else self._replicate
        result = call(spec["app"], payload, headers)
        try:
            url = dig(result, spec.get("video", "video.url"))
        except (KeyError, IndexError, ValueError) as exc:
            raise BackendError(f"no video at {spec.get('video')!r} in {str(result)[:300]}") from exc

        path = Path(tempfile.gettempdir()) / f"hosted-{uuid.uuid4().hex}.mp4"
        path.write_bytes(self.transport.get(url, {}))
        try:
            frames, _ = read_video(path)
        finally:
            path.unlink(missing_ok=True)
        meta: dict[str, Any] = {"provider": provider, "app": spec["app"], "video_url": url}
        return ClipResult(frames=conform(frames, job, meta), fps=job.fps, meta=meta)


def _aspect(width: int, height: int) -> str:
    ratio = width / height
    options = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4}
    return min(options, key=lambda k: abs(options[k] - ratio))


def conform(frames: np.ndarray, job: Job, meta: dict[str, Any]) -> np.ndarray:
    """Fit a hosted result to the job's size and frame count, noting what changed."""
    if frames.shape[1:3] != (job.height, job.width):
        meta["resized_from"] = [int(frames.shape[2]), int(frames.shape[1])]
        source_ratio = frames.shape[2] / frames.shape[1]
        target_ratio = job.width / job.height
        h, w = frames.shape[1:3]
        if source_ratio > target_ratio:
            new_w = round(h * target_ratio)
            left = (w - new_w) // 2
            frames = frames[:, :, left : left + new_w]
        elif source_ratio < target_ratio:
            new_h = round(w / target_ratio)
            top = (h - new_h) // 2
            frames = frames[:, top : top + new_h]
        frames = np.stack(
            [
                np.asarray(
                    Image.fromarray(f).resize((job.width, job.height), Image.Resampling.LANCZOS)
                )
                for f in frames
            ]
        )
    if len(frames) != job.frames:
        meta["frames_returned"] = int(len(frames))
        picks = np.linspace(0, len(frames) - 1, job.frames).round().astype(int)
        frames = frames[picks]
    return frames
