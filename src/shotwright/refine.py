from __future__ import annotations

import gc
import importlib
import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from PIL import Image

from .color import LabStats, lab_to_rgb, rgb_to_lab, transfer
from .plan.model import FrameRefine


class RefineError(RuntimeError):
    pass


@dataclass(frozen=True)
class RefinerModel:
    id: str
    pipeline: str
    repo: str
    strength: float
    steps: int
    guidance: float
    prompt_style: str = "natural"
    max_tokens: int = 512
    multiple: int = 16
    pixels: int = 1024 * 1024
    negative: str | None = None
    quality_tags: str | None = None
    scheduler: str | None = None
    guidance_arg: str = "guidance_scale"
    license: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


REFINERS: dict[str, RefinerModel] = {
    m.id: m
    for m in (
        RefinerModel(
            id="z-image-turbo",
            pipeline="ZImageImg2ImgPipeline",
            repo="Tongyi-MAI/Z-Image-Turbo",
            strength=0.2,
            steps=12,
            guidance=0.0,
            license="apache-2.0",
        ),
        RefinerModel(
            id="animagine-xl-4.0",
            pipeline="StableDiffusionXLImg2ImgPipeline",
            repo="cagliostrolab/animagine-xl-4.0",
            strength=0.35,
            steps=28,
            guidance=5.0,
            prompt_style="tags",
            max_tokens=77,
            multiple=8,
            negative=(
                "lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, "
                "fewer digits, cropped, worst quality, low quality, low score, bad score, "
                "average score, signature, watermark, username, blurry"
            ),
            quality_tags="masterpiece, high score, great score, absurdres",
            scheduler="EulerDiscreteScheduler",
            license="openrail++",
        ),
        RefinerModel(
            id="flux.1-dev",
            pipeline="FluxImg2ImgPipeline",
            repo="black-forest-labs/FLUX.1-dev",
            strength=0.4,
            steps=28,
            guidance=3.5,
            license="flux-1-dev-non-commercial",
        ),
        RefinerModel(
            id="qwen-image",
            pipeline="QwenImageImg2ImgPipeline",
            repo="Qwen/Qwen-Image",
            strength=0.35,
            steps=40,
            guidance=4.0,
            guidance_arg="true_cfg_scale",
            negative=" ",
            license="apache-2.0",
        ),
    )
}

LOOK_DEFAULTS = {"realistic": "z-image-turbo", "anime": "animagine-xl-4.0"}


def refiner_model(model_id: str) -> RefinerModel:
    try:
        return REFINERS[model_id]
    except KeyError:
        raise RefineError(
            f"unknown refiner {model_id!r}; known refiners: {', '.join(sorted(REFINERS))}"
        ) from None


def sharpness(frame: np.ndarray) -> float:
    """Variance of the Laplacian of the luma channel; higher is sharper."""
    y = frame[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lap = -4 * y[1:-1, 1:-1] + y[:-2, 1:-1] + y[2:, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:]
    return float(lap.var())


def pick_start_frame(frames: np.ndarray, window: int) -> int:
    """How many frames from the end the sharpest of the last `window` frames sits (0 = last).

    Ties and near-ties go to the later frame, so motion is not cut short for nothing.
    """
    window = max(1, min(window, len(frames)))
    scores = [sharpness(frames[-1 - k]) for k in range(window)]
    best = max(scores)
    for k, score in enumerate(scores):
        if score >= best * 0.97:
            return k
    return 0


def working_size(width: int, height: int, pixels: int, multiple: int) -> tuple[int, int]:
    """Size the image model works at: the frame scaled to about `pixels`, sides on `multiple`."""
    scale = math.sqrt(pixels / (width * height))
    w = max(multiple, round(width * scale / multiple) * multiple)
    h = max(multiple, round(height * scale / multiple) * multiple)
    return w, h


def match_color(refined: np.ndarray, source: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Move the refined frame's Lab statistics back to the source frame's."""
    if strength <= 0:
        return refined
    lab = rgb_to_lab(refined)
    moved = transfer(lab, LabStats.of(refined, stride=1), LabStats.of(source, stride=1))
    return lab_to_rgb(lab + (moved - lab) * min(strength, 1.0))


def refine_prompt(spec: FrameRefine, model: RefinerModel) -> tuple[str, str | None]:
    """The prompt and negative the image model gets, in its own style."""
    from .prompting.budget import count_tokens, fit_tokens

    negative = spec.negative if spec.negative is not None else model.negative
    limit = model.max_tokens - 2
    if model.prompt_style != "tags":
        return fit_tokens(spec.prompt, limit).text, negative
    suffix = model.quality_tags or ""
    reserve = count_tokens(suffix)[0] + 1 if suffix else 0
    text = fit_tokens(spec.tags or spec.prompt, limit - reserve, tags=True).text
    return (f"{text.rstrip(', ')}, {suffix}" if suffix else text), negative


@runtime_checkable
class FrameRefiner(Protocol):
    name: str

    def refine(self, frame: np.ndarray, spec: FrameRefine) -> np.ndarray: ...

    def release(self) -> None: ...


class FakeRefiner:
    """Brightens fine detail a little; for tests and dry runs."""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[FrameRefine] = []

    def refine(self, frame: np.ndarray, spec: FrameRefine) -> np.ndarray:
        self.calls.append(spec)
        shifted = frame.astype(np.int16) + int(round(20 * (spec.strength or 0.35)))
        return np.clip(shifted, 0, 255).astype(np.uint8)

    def release(self) -> None:
        pass


class DiffusersRefiner:
    """Re-renders a frame with an image model through diffusers img2img."""

    name = "diffusers"

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "bfloat16",
        offload: str = "none",
        repos: dict[str, str] | None = None,
        loader: Callable[[RefinerModel, Any], Any] | None = None,
        torch_module: Any = None,
    ):
        self.device = device
        self.dtype_name = dtype
        self.offload = offload
        self.repos = dict(repos or {})
        self._loader = loader
        self._torch = torch_module
        self._model: RefinerModel | None = None
        self._pipe: Any = None

    @property
    def torch(self):
        if self._torch is None:
            try:
                self._torch = importlib.import_module("torch")
            except ImportError as exc:
                raise RefineError("frame refinement needs torch and diffusers") from exc
        return self._torch

    def _load(self, model: RefinerModel) -> Any:
        dtype = getattr(self.torch, self.dtype_name)
        if self._loader is not None:
            return self._loader(model, dtype)
        diffusers = importlib.import_module("diffusers")
        cls = getattr(diffusers, model.pipeline, None)
        if cls is None:
            raise RefineError(f"diffusers {diffusers.__version__} has no {model.pipeline}")
        repo = self.repos.get(model.id, model.repo)
        if repo.endswith(".safetensors"):
            pipe = cls.from_single_file(repo, torch_dtype=dtype)
        else:
            pipe = cls.from_pretrained(repo, torch_dtype=dtype)
        if model.scheduler:
            sched = getattr(diffusers, model.scheduler)
            pipe.scheduler = sched.from_config(pipe.scheduler.config)
        if self.offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif self.offload == "model":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        return pipe

    def pipe_for(self, model: RefinerModel) -> Any:
        if self._model is not None and self._model.id == model.id:
            return self._pipe
        self.release()
        self._pipe = self._load(model)
        self._model = model
        return self._pipe

    def refine(self, frame: np.ndarray, spec: FrameRefine) -> np.ndarray:
        model = refiner_model(spec.model)
        pipe = self.pipe_for(model)
        height, width = frame.shape[:2]
        work_w, work_h = working_size(width, height, model.pixels, model.multiple)
        image = Image.fromarray(frame).resize((work_w, work_h), Image.Resampling.LANCZOS)
        prompt, negative = refine_prompt(spec, model)
        wanted: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative,
            "image": image,
            "strength": spec.strength if spec.strength is not None else model.strength,
            "num_inference_steps": spec.steps or model.steps,
            model.guidance_arg: spec.guidance if spec.guidance is not None else model.guidance,
            "width": work_w,
            "height": work_h,
            "generator": self.torch.Generator(device="cpu").manual_seed(spec.seed),
            "output_type": "pil",
            **model.extra,
        }
        try:
            params = inspect.signature(pipe.__call__).parameters
            accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
            kwargs = {
                k: v for k, v in wanted.items() if v is not None and (accepts_any or k in params)
            }
        except (TypeError, ValueError):
            kwargs = {k: v for k, v in wanted.items() if v is not None}
        out = pipe(**kwargs).images[0]
        out = out.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        return np.asarray(out, dtype=np.uint8)

    def release(self) -> None:
        if self._pipe is None:
            return
        self._pipe = None
        self._model = None
        gc.collect()
        torch = self._torch
        if torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available():
            torch.cuda.empty_cache()
