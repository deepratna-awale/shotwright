from __future__ import annotations

import hashlib
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image


class VideoWriter:
    def __init__(self, path: str | Path, width: int, height: int, fps: float, crf: int = 12):
        self.width, self.height = width, height
        self._gen = imageio_ffmpeg.write_frames(
            str(path),
            (width, height),
            fps=fps,
            codec="libx264",
            pix_fmt_out="yuv420p" if width % 2 == 0 and height % 2 == 0 else "yuv444p",
            output_params=["-crf", str(crf), "-preset", "medium"],
            macro_block_size=1,
        )
        self._gen.send(None)
        self.frames_written = 0

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"frame {frame.shape} does not match {self.width}x{self.height}")
        self._gen.send(np.ascontiguousarray(frame, dtype=np.uint8))
        self.frames_written += 1

    def close(self) -> None:
        self._gen.close()

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_video(path: str | Path, frames: np.ndarray, fps: float, crf: int = 12) -> None:
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected frames shaped (T, H, W, 3), got {frames.shape}")
    _, height, width, _ = frames.shape
    with VideoWriter(path, width, height, fps, crf) as writer:
        for frame in frames:
            writer.write(frame)


def read_video(path: str | Path) -> tuple[np.ndarray, float]:
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = reader.__next__()
    width, height = meta["size"]
    frames = [np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3) for chunk in reader]
    return np.stack(frames), float(meta["fps"])


def resize_crop(
    frame: np.ndarray,
    resize_to: tuple[int, int] | None,
    crop: tuple[int, int, int, int] | None,
) -> np.ndarray:
    if resize_to is None and crop is None:
        return frame
    image = Image.fromarray(frame)
    if resize_to is not None and image.size != tuple(resize_to):
        image = image.resize(tuple(resize_to), Image.Resampling.LANCZOS)
    if crop is not None:
        left, top, right, bottom = crop
        image = image.crop((left, top, image.width - right, image.height - bottom))
    return np.asarray(image, dtype=np.uint8)


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
