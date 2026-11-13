"""RunPod serverless entry point: render a shot list and stitch it in one request."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import yaml

from .plan import PlanError, ShotList, compile_plan
from .run import ComfyUIBackend, DiffusersBackend, FakeBackend, HostedBackend, Runner
from .stitch import stitch

RUNS_DIR = os.environ.get("SHOTWRIGHT_RUNS_DIR", "/runpod-volume/shotwright/runs")
MAX_INLINE_BYTES = 15 * 1024 * 1024

_backends: dict[str, Any] = {}


def _backend(name: str, options: dict[str, Any]):
    if name not in _backends:
        if name == "diffusers":
            _backends[name] = DiffusersBackend(**options)
        elif name == "comfyui":
            _backends[name] = ComfyUIBackend(**options)
        elif name == "hosted":
            _backends[name] = HostedBackend(**options)
        elif name == "fake":
            _backends[name] = FakeBackend()
        else:
            raise ValueError(f"unknown backend {name!r}")
    return _backends[name]


def handler(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("input") or {}
    shots = data.get("shot_list")
    if isinstance(shots, str):
        shots = yaml.safe_load(shots)
    if not isinstance(shots, dict):
        return {"error": "input.shot_list must be a shot list object or YAML string"}
    try:
        plan = compile_plan(ShotList.from_dict(shots, base_dir=data.get("base_dir")))
    except PlanError as exc:
        return {"error": "the shot list has problems", "problems": exc.problems}

    backend = _backend(data.get("backend", "diffusers"), data.get("backend_options") or {})
    runner = Runner(
        backend,
        runs_root=data.get("runs_dir", RUNS_DIR),
        vram_gb=data.get("vram_gb"),
    )
    run_dir = data.get("resume")
    result = runner.resume(run_dir) if run_dir else runner.start(plan, name=data.get("name"))
    response: dict[str, Any] = {
        "run_dir": str(result.run_dir),
        "done": result.done,
        "cached": result.cached,
        "failed": result.failed,
        "skipped": result.skipped,
    }
    if not result.ok:
        return response

    options = data.get("stitch") or {}
    stitched = stitch(result.run_dir, **options)
    response.update(
        video=str(stitched.path),
        frames=stitched.frames,
        fps=stitched.fps,
        flagged_seams=[f"{s.before}->{s.after}" for s in stitched.seams if s.flagged],
    )
    path = Path(stitched.path)
    if data.get("return_video") and path.stat().st_size <= MAX_INLINE_BYTES:
        response["video_base64"] = base64.b64encode(path.read_bytes()).decode()
    return response


def main() -> None:
    import runpod

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
