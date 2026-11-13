from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Iterable
from importlib import resources
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from ..media import read_video
from ..plan import Job
from ..profiles import ModelProfile, VramPreset
from .backend import BackendError, ClipResult, JobInputs, OutOfMemory, TransientError

WORKFLOW_PATH_ENV = "SHOTWRIGHT_WORKFLOW_PATH"
PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*(?:\|\s*(.*?)\s*)?\}\}")
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".gif", ".webp"}
OOM_MARKERS = ("OutOfMemoryError", "out of memory", "Allocation on device")


class Transport(Protocol):
    def get(self, path: str) -> bytes: ...

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]: ...

    def upload(self, path: str, filename: str, data: bytes) -> dict[str, Any]: ...


class HttpTransport:
    def __init__(self, url: str, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _open(self, request: urllib.request.Request) -> bytes:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:2000]
            if exc.code >= 500:
                raise TransientError(f"ComfyUI returned {exc.code}: {body}") from exc
            raise BackendError(f"ComfyUI rejected the request ({exc.code}): {body}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise TransientError(f"cannot reach ComfyUI at {self.url}: {exc}") from exc

    def get(self, path: str) -> bytes:
        return self._open(urllib.request.Request(self.url + path))

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        return json.loads(self._open(request) or b"{}")

    def upload(self, path: str, filename: str, data: bytes) -> dict[str, Any]:
        boundary = uuid.uuid4().hex
        body = b"".join(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode(),
                b"Content-Type: image/png\r\n\r\n",
                data,
                f"\r\n--{boundary}\r\n".encode(),
                b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue',
                f"\r\n--{boundary}--\r\n".encode(),
            ]
        )
        request = urllib.request.Request(
            self.url + path,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        return json.loads(self._open(request))


def _value(match: re.Match, values: dict[str, Any]) -> Any:
    name, default = match.group(1), match.group(2)
    if name in values:
        return values[name]
    if default is None:
        raise BackendError(f"workflow placeholder {{{{{name}}}}} has no value")
    try:
        return json.loads(default)
    except json.JSONDecodeError:
        return default


def fill(graph: Any, values: dict[str, Any]) -> Any:
    """Replace {{name}} or {{name|default}} placeholders.

    A string that is only a placeholder takes the value's type; defaults are JSON.
    """
    if isinstance(graph, dict):
        return {k: fill(v, values) for k, v in graph.items()}
    if isinstance(graph, list):
        return [fill(v, values) for v in graph]
    if isinstance(graph, str):
        whole = PLACEHOLDER.fullmatch(graph.strip())
        if whole:
            return _value(whole, values)
        return PLACEHOLDER.sub(lambda m: str(_value(m, values)), graph)
    return graph


class ComfyUIBackend:
    """Runs jobs by queueing API-format workflows on a ComfyUI server.

    Workflows are looked up as `<profile id>.<t2v|i2v>.json`, falling back to the
    profile's base, in user directories first and then the bundled set. A
    `<profile id>+<recipe>.<mode>.json` file takes precedence for that VRAM recipe.
    """

    name = "comfyui"

    def __init__(
        self,
        url: str = "http://127.0.0.1:8188",
        workflow_dirs: Iterable[str | Path] = (),
        transport: Transport | None = None,
        poll_interval: float = 1.0,
        timeout: float = 3600.0,
        sleep=time.sleep,
    ):
        self.transport = transport or HttpTransport(url)
        dirs = [Path(d) for d in workflow_dirs]
        env = os.environ.get(WORKFLOW_PATH_ENV)
        if env:
            dirs += [Path(p) for p in env.split(os.pathsep) if p]
        self.workflow_dirs = dirs
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._sleep = sleep
        self.client_id = uuid.uuid4().hex
        self._profile: ModelProfile | None = None
        self._preset: VramPreset | None = None

    def _candidates(self, profile: ModelProfile, mode: str) -> list[str]:
        ids = [profile.id] + ([profile.base] if profile.base else [])
        recipe = self._preset.recipe if self._preset and profile is self._profile else None
        names = []
        for model_id in ids:
            if recipe:
                names.append(f"{model_id}+{recipe}.{mode}.json")
            names.append(f"{model_id}.{mode}.json")
        return names

    def _find(self, profile: ModelProfile, mode: str) -> dict[str, Any] | None:
        bundled = resources.files("shotwright") / "workflows"
        for name in self._candidates(profile, mode):
            for directory in self.workflow_dirs:
                path = directory / name
                if path.is_file():
                    return json.loads(path.read_text(encoding="utf-8"))
            entry = bundled / name
            if entry.is_file():
                return json.loads(entry.read_text(encoding="utf-8"))
        return None

    def supports(self, profile: ModelProfile) -> bool:
        return any(self._find(profile, mode) is not None for mode in ("t2v", "i2v"))

    def prepare(self, profile: ModelProfile, preset: VramPreset | None) -> None:
        self._profile = profile
        self._preset = preset

    def release(self) -> None:
        with contextlib.suppress(BackendError):
            self.transport.post_json("/free", {"unload_models": True, "free_memory": True})

    def _upload(self, job: Job, image: np.ndarray) -> str:
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        name = f"shotwright-{job.id}-{uuid.uuid4().hex[:8]}.png"
        info = self.transport.upload("/upload/image", name, buffer.getvalue())
        subfolder = info.get("subfolder") or ""
        return f"{subfolder}/{info['name']}" if subfolder else info["name"]

    def _wait(self, prompt_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        while True:
            history = json.loads(self.transport.get(f"/history/{prompt_id}") or b"{}")
            entry = history.get(prompt_id)
            if entry:
                status = entry.get("status") or {}
                if status.get("status_str") == "error" or status.get("completed") is False:
                    raise _execution_error(status)
                if entry.get("outputs") or status.get("completed"):
                    return entry
            if time.monotonic() > deadline:
                raise TransientError(f"ComfyUI did not finish prompt {prompt_id} in time")
            self._sleep(self.poll_interval)

    def _download(self, item: dict[str, Any]) -> bytes:
        query = urllib.parse.urlencode(
            {
                "filename": item["filename"],
                "subfolder": item.get("subfolder", ""),
                "type": item.get("type", "output"),
            }
        )
        return self.transport.get(f"/view?{query}")

    def _frames(self, entry: dict[str, Any], scratch: Path) -> tuple[np.ndarray, dict[str, Any]]:
        images: list[dict[str, Any]] = []
        videos: list[dict[str, Any]] = []
        for node_id, output in sorted(entry.get("outputs", {}).items()):
            for key, items in output.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict) or "filename" not in item:
                        continue
                    if item.get("type") == "temp":
                        continue
                    suffix = Path(item["filename"]).suffix.lower()
                    tagged = {**item, "_node": node_id}
                    if suffix in VIDEO_SUFFIXES or key in ("gifs", "videos"):
                        videos.append(tagged)
                    else:
                        images.append(tagged)
        if videos:
            item = videos[0]
            path = scratch / f"comfy-{uuid.uuid4().hex}{Path(item['filename']).suffix}"
            path.write_bytes(self._download(item))
            try:
                frames, _ = read_video(path)
            finally:
                path.unlink(missing_ok=True)
            return frames, {"output": item["filename"], "node": item["_node"]}
        if images:
            node = images[0]["_node"]
            chosen = [i for i in images if i["_node"] == node]
            frames = np.stack(
                [
                    np.asarray(Image.open(io.BytesIO(self._download(i))).convert("RGB"))
                    for i in chosen
                ]
            )
            return frames, {"output": f"{len(chosen)} images", "node": node}
        raise BackendError("the workflow finished without saving any images or video")

    def run(self, job: Job, inputs: JobInputs) -> ClipResult:
        profile = self._profile
        if profile is None:
            raise BackendError("prepare() must be called before run()")
        mode = "i2v" if inputs.init_image is not None else "t2v"
        template = self._find(profile, mode)
        if template is None:
            raise BackendError(f"no ComfyUI {mode} workflow for {profile.id}")

        values: dict[str, Any] = {
            "prompt": job.prompt,
            "negative": job.negative or "",
            "seed": job.seed,
            "frames": job.frames,
            "width": job.width,
            "height": job.height,
            "fps": job.fps,
            "prefix": f"shotwright/{job.id}",
        }
        values.update(job.params)
        if inputs.init_image is not None:
            values["image"] = self._upload(job, inputs.init_image)
        graph = fill(copy.deepcopy(template), values)

        queued = self.transport.post_json("/prompt", {"prompt": graph, "client_id": self.client_id})
        if queued.get("node_errors"):
            raise BackendError(f"ComfyUI rejected the workflow: {queued['node_errors']}")
        prompt_id = queued.get("prompt_id")
        if not prompt_id:
            raise BackendError(f"ComfyUI did not queue the workflow: {queued}")
        entry = self._wait(prompt_id)

        frames, meta = self._frames(entry, Path(tempfile.gettempdir()))
        if frames.shape[1:3] != (job.height, job.width):
            meta["resized_from"] = [int(frames.shape[2]), int(frames.shape[1])]
            frames = np.stack(
                [
                    np.asarray(
                        Image.fromarray(f).resize((job.width, job.height), Image.Resampling.LANCZOS)
                    )
                    for f in frames
                ]
            )
        meta["prompt_id"] = prompt_id
        return ClipResult(frames=frames, fps=job.fps, meta=meta)


def _execution_error(status: dict[str, Any]) -> BackendError:
    for kind, data in status.get("messages") or []:
        if kind == "execution_error":
            text = f"{data.get('exception_type', '')}: {data.get('exception_message', '')}"
            node = data.get("node_type") or data.get("node_id")
            message = f"ComfyUI failed in {node}: {text.strip()}"
            if any(marker in text for marker in OOM_MARKERS):
                return OutOfMemory(message)
            return BackendError(message)
    return BackendError(f"ComfyUI reported an error: {status}")
