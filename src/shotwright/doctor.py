from __future__ import annotations

import importlib
import os
import shutil
import urllib.request
from dataclasses import dataclass

import imageio_ffmpeg


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = False


def _module(name: str) -> object | None:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _version(name: str) -> str | None:
    module = _module(name)
    return getattr(module, "__version__", "installed") if module else None


def run_checks(comfy_url: str | None = None) -> list[Check]:
    checks: list[Check] = []
    try:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        checks.append(Check("ffmpeg", True, ffmpeg, required=True))
    except RuntimeError as exc:
        checks.append(Check("ffmpeg", False, str(exc), required=True))

    torch = _module("torch")
    if torch is None:
        checks.append(Check("torch", False, "not installed (needed for --backend diffusers)"))
    else:
        cuda = torch.cuda.is_available()
        if cuda:
            props = torch.cuda.get_device_properties(0)
            detail = f"{torch.__version__}, {props.name}, {props.total_memory / 2**30:.0f} GB"
        else:
            detail = f"{torch.__version__}, no CUDA device"
        checks.append(Check("torch", cuda, detail))
    for name in ("diffusers", "transformers", "accelerate"):
        version = _version(name)
        checks.append(Check(name, version is not None, version or "not installed"))

    if comfy_url:
        try:
            with urllib.request.urlopen(comfy_url.rstrip("/") + "/system_stats", timeout=5) as r:
                ok = r.status == 200
            checks.append(Check("comfyui", ok, comfy_url))
        except OSError as exc:
            checks.append(Check("comfyui", False, f"{comfy_url}: {exc}"))

    for label, names in (
        ("llm key", ("SHOTWRIGHT_LLM_API_KEY", "OPENAI_API_KEY")),
        ("llm model", ("SHOTWRIGHT_LLM_MODEL",)),
        ("tts model", ("SHOTWRIGHT_TTS_MODEL",)),
        ("fal key", ("FAL_KEY",)),
        ("replicate key", ("REPLICATE_API_TOKEN",)),
    ):
        found = next((n for n in names if os.environ.get(n)), None)
        checks.append(Check(label, found is not None, f"${found}" if found else "not set"))

    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    usage = shutil.disk_usage(hf_home if os.path.isdir(hf_home) else os.path.expanduser("~"))
    free = usage.free / 2**30
    checks.append(Check("disk", free > 50, f"{free:.0f} GB free for weights at {hf_home}"))
    return checks
