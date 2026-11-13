from __future__ import annotations

import hashlib
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image


def write_video(path: str | Path, frames: np.ndarray, fps: float, crf: int = 12) -> None:
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected frames shaped (T, H, W, 3), got {frames.shape}")
    _, height, width, _ = frames.shape
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_out="yuv420p" if width % 2 == 0 and height % 2 == 0 else "yuv444p",
        output_params=["-crf", str(crf), "-preset", "medium"],
        macro_block_size=1,
    )
    writer.send(None)
    for frame in frames:
        writer.send(frame)
    writer.close()


def read_video(path: str | Path) -> tuple[np.ndarray, float]:
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = reader.__next__()
    width, height = meta["size"]
    frames = [np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3) for chunk in reader]
    return np.stack(frames), float(meta["fps"])


def save_image(path: str | Path, frame: np.ndarray) -> None:
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(path)


def load_image(path: str | Path, width: int | None = None, height: int | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if width and height and image.size != (width, height):
        image = fit_image(image, width, height)
    return np.asarray(image, dtype=np.uint8)


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (max(width, round(image.width * scale)), max(height, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
