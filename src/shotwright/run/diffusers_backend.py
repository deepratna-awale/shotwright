from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image

from ..plan import Job
from ..profiles import ModelProfile, VramPreset
from .backend import BackendError, ClipResult, JobInputs, OutOfMemory


def is_image_to_video(class_name: str) -> bool:
    return class_name.endswith("ImageToVideoPipeline")


class DiffusersBackend:
    """Runs jobs in-process through Hugging Face diffusers pipelines."""

    name = "diffusers"

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "bfloat16",
        repos: dict[str, str] | None = None,
        loader: Callable[[str, str, Any], Any] | None = None,
        torch_module: Any = None,
    ):
        self.device = device
        self.dtype_name = dtype
        self.repos = dict(repos or {})
        self._loader = loader
        self._torch = torch_module
        self._profile: ModelProfile | None = None
        self._preset: VramPreset | None = None
        self._pipes: dict[str, Any] = {}

    @property
    def torch(self):
        if self._torch is None:
            try:
                self._torch = importlib.import_module("torch")
            except ImportError as exc:
                raise BackendError(
                    "the diffusers backend needs torch and diffusers: "
                    "pip install 'shotwright[diffusers]'"
                ) from exc
        return self._torch

    def _dtype(self):
        return getattr(self.torch, self.dtype_name)

    def _load(self, class_name: str, repo: str):
        if self._loader is not None:
            return self._loader(class_name, repo, self._dtype())
        try:
            diffusers = importlib.import_module("diffusers")
        except ImportError as exc:
            raise BackendError(
                "diffusers is not installed: pip install 'shotwright[diffusers]'"
            ) from exc
        cls = getattr(diffusers, class_name, None)
        if cls is None:
            raise BackendError(f"diffusers {diffusers.__version__} has no {class_name}")
        return cls.from_pretrained(repo, torch_dtype=self._dtype())

    def supports(self, profile: ModelProfile) -> bool:
        return bool(profile.backends.diffusers) and bool(
            self.repos.get(profile.id) or profile.backends.diffusers_repo
        )

    def prepare(self, profile: ModelProfile, preset: VramPreset | None) -> None:
        same_model = self._profile is not None and self._profile.id == profile.id
        same_recipe = (self._preset.recipe if self._preset else None) == (
            preset.recipe if preset else None
        )
        if same_model and same_recipe:
            return
        self.release()
        self._profile = profile
        self._preset = preset

    def release(self) -> None:
        self._pipes.clear()
        self._profile = None
        self._preset = None
        torch = self._torch
        if torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _recipe_parts(self) -> set[str]:
        return set((self._preset.recipe if self._preset else "full").split("+"))

    def _apply_recipe(self, pipe: Any) -> Any:
        parts = self._recipe_parts()
        if "fp8" in parts:
            transformer = getattr(pipe, "transformer", None)
            if transformer is not None and hasattr(transformer, "enable_layerwise_casting"):
                transformer.enable_layerwise_casting(
                    storage_dtype=self.torch.float8_e4m3fn, compute_dtype=self._dtype()
                )
        if "vae_tiling" in parts and hasattr(getattr(pipe, "vae", None), "enable_tiling"):
            pipe.vae.enable_tiling()
        return self._place(pipe)

    def _place(self, pipe: Any) -> Any:
        parts = self._recipe_parts()
        if "sequential" in parts:
            pipe.enable_sequential_cpu_offload()
        elif "offload" in parts:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        return pipe

    def _pipe(self, image_to_video: bool) -> Any:
        profile = self._profile
        if profile is None:
            raise BackendError("prepare() must be called before run()")
        backends = profile.backends
        base_class = backends.diffusers
        base_is_i2v = is_image_to_video(base_class)
        if image_to_video and not base_is_i2v:
            if not backends.diffusers_i2v:
                raise BackendError(f"{profile.id} has no image-to-video pipeline for diffusers")
            key, class_name = "i2v", backends.diffusers_i2v
        elif not image_to_video and base_is_i2v:
            raise BackendError(f"{profile.id} needs an init image")
        else:
            key, class_name = "base", base_class

        if key in self._pipes:
            return self._pipes[key]

        repo = self.repos.get(profile.id) or backends.diffusers_repo
        if key == "i2v":
            i2v_repo = self.repos.get(f"{profile.id}:i2v") or backends.diffusers_repo_i2v
            if i2v_repo or self._loader is not None:
                pipe = self._apply_recipe(self._load(class_name, i2v_repo or repo))
            else:
                base = self._pipe(image_to_video=False)
                cls = getattr(importlib.import_module("diffusers"), class_name)
                pipe = self._place(cls.from_pipe(base))
        else:
            pipe = self._apply_recipe(self._load(class_name, repo))
        self._pipes[key] = pipe
        return pipe

    def _kwargs(self, pipe: Any, job: Job, inputs: JobInputs) -> dict[str, Any]:
        torch = self.torch
        generator = torch.Generator(device="cpu").manual_seed(job.seed)
        wanted: dict[str, Any] = {
            "prompt": job.prompt,
            "negative_prompt": job.negative,
            "num_frames": job.frames,
            "height": job.height,
            "width": job.width,
            "frame_rate": job.fps,
            "generator": generator,
            "output_type": "np",
        }
        if inputs.context_frames is not None:
            k = len(inputs.context_frames)
            blank = np.full_like(inputs.context_frames[0], 127)
            known = np.zeros_like(inputs.context_frames[0])
            unknown = np.full_like(inputs.context_frames[0], 255)
            wanted["video"] = [Image.fromarray(f) for f in inputs.context_frames] + [
                Image.fromarray(blank)
            ] * (job.frames - k)
            wanted["mask"] = [Image.fromarray(known)] * k + [Image.fromarray(unknown)] * (
                job.frames - k
            )
        elif inputs.init_image is not None:
            wanted["image"] = Image.fromarray(inputs.init_image)
        if inputs.end_image is not None:
            wanted["last_image"] = Image.fromarray(inputs.end_image)
        wanted.update(job.params)
        try:
            params = inspect.signature(pipe.__call__).parameters
        except (TypeError, ValueError):
            return {k: v for k, v in wanted.items() if v is not None}
        accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        return {k: v for k, v in wanted.items() if v is not None and (accepts_any or k in params)}

    def run(self, job: Job, inputs: JobInputs) -> ClipResult:
        overlap = inputs.context_frames is not None
        pipe = self._pipe(image_to_video=inputs.init_image is not None and not overlap)
        kwargs = self._kwargs(pipe, job, inputs)
        if overlap and not {"video", "mask"} <= kwargs.keys():
            raise BackendError(
                f"{type(pipe).__name__} cannot take context frames; use a VACE model "
                "or set continuity.overlap to 0"
            )
        if inputs.end_image is not None and "last_image" not in kwargs:
            raise BackendError(f"{type(pipe).__name__} cannot take an end image")
        oom_types = tuple(
            t
            for t in (
                getattr(self.torch, "OutOfMemoryError", None),
                getattr(getattr(self.torch, "cuda", None), "OutOfMemoryError", None),
            )
            if isinstance(t, type)
        )
        try:
            output = pipe(**kwargs)
        except oom_types as exc:
            raise OutOfMemory(str(exc)) from exc

        frames = to_uint8_frames(output.frames[0] if hasattr(output, "frames") else output[0])
        meta: dict[str, Any] = {"pipeline": type(pipe).__name__}
        audio, rate = native_audio(pipe, output)
        if audio is not None:
            meta["audio_rate"] = rate
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
        if frames.shape[0] != job.frames:
            meta["frames_returned"] = int(frames.shape[0])
        return ClipResult(frames=frames, fps=job.fps, meta=meta, audio=audio, audio_rate=rate)


def to_uint8_frames(video: Any) -> np.ndarray:
    if isinstance(video, list):
        video = np.stack([np.asarray(v) for v in video])
    if hasattr(video, "detach"):
        video = video.detach().float().cpu().numpy()
    arr = np.asarray(video)
    if arr.ndim == 4 and arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = arr.transpose(0, 2, 3, 1)
    if arr.dtype != np.uint8:
        if arr.min() < 0:
            arr = (arr + 1) / 2
        arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def native_audio(pipe: Any, output: Any) -> tuple[np.ndarray | None, int | None]:
    """Mono float audio and its sample rate from pipelines that generate sound, such as LTX-2."""
    audio = getattr(output, "audio", None)
    if audio is None:
        return None, None
    if isinstance(audio, (list, tuple)) or getattr(audio, "ndim", 0) == 3:
        audio = audio[0]
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=0) if arr.shape[0] <= 8 else arr.mean(axis=1)
    config = getattr(getattr(pipe, "vocoder", None), "config", None)
    rate = getattr(config, "output_sampling_rate", None) or getattr(
        pipe, "audio_sampling_rate", None
    )
    return arr.reshape(-1), int(rate or 24000)
