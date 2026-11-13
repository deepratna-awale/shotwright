from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .media import SAMPLE_RATE, AudioError, mux, read_wav, resample, wav_bytes
from .run.rundir import RunDir


class Voice(Protocol):
    name: str

    def synthesize(self, text: str, voice: str | None) -> tuple[np.ndarray, int]: ...


class FakeVoice:
    """A tone whose length follows the text, for tests and dry runs."""

    name = "fake"

    def __init__(self, seconds_per_char: float = 0.05, rate: int = 22050):
        self.seconds_per_char = seconds_per_char
        self.rate = rate
        self.calls: list[tuple[str, str | None]] = []

    def synthesize(self, text: str, voice: str | None) -> tuple[np.ndarray, int]:
        self.calls.append((text, voice))
        n = max(1, round(len(text) * self.seconds_per_char * self.rate))
        pitch = 180 + (int(hashlib.sha256((voice or "").encode()).hexdigest(), 16) % 200)
        t = np.arange(n) / self.rate
        return (0.3 * np.sin(2 * np.pi * pitch * t)).astype(np.float32), self.rate


class OpenAISpeech:
    """Any server that implements the OpenAI-style /audio/speech endpoint."""

    name = "openai-speech"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        default_voice: str | None = None,
        timeout: float = 120.0,
    ):
        self.model = model or os.environ.get("SHOTWRIGHT_TTS_MODEL")
        self.base_url = (
            base_url or os.environ.get("SHOTWRIGHT_TTS_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.api_key = (
            api_key or os.environ.get("SHOTWRIGHT_TTS_API_KEY") or os.environ.get("OPENAI_API_KEY")
        )
        self.default_voice = default_voice or os.environ.get("SHOTWRIGHT_TTS_VOICE") or "alloy"
        self.timeout = timeout

    def _post(self, payload: dict[str, Any]) -> bytes:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/audio/speech", data=json.dumps(payload).encode(), headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:500]
            raise AudioError(f"speech request failed ({exc.code}): {body}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise AudioError(f"cannot reach {self.base_url}: {exc}") from exc

    def synthesize(self, text: str, voice: str | None) -> tuple[np.ndarray, int]:
        if not self.model:
            raise AudioError("no TTS model set; pass --tts-model or set SHOTWRIGHT_TTS_MODEL")
        payload = {
            "model": self.model,
            "input": text,
            "voice": voice or self.default_voice,
            "response_format": "wav",
        }
        return read_wav(self._post(payload))


@dataclass
class PlacedLine:
    shot: str
    text: str
    voice: str | None
    start: float
    end: float
    clipped: bool = False


@dataclass
class VoiceResult:
    path: Path
    audio: Path
    lines: list[PlacedLine] = field(default_factory=list)
    overlaps: list[tuple[str, str]] = field(default_factory=list)


def _synth_cached(run: RunDir, voice: Voice, text: str, name: str | None) -> np.ndarray:
    key = hashlib.sha256(json.dumps([voice.name, name, text]).encode()).hexdigest()[:24]
    path = run.root / "voice" / f"{key}.wav"
    if path.is_file():
        samples, rate = read_wav(path.read_bytes())
    else:
        samples, rate = voice.synthesize(text, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(wav_bytes(samples, rate))
    return resample(samples, rate)


def voice_over(
    run_dir: str | Path,
    voice: Voice,
    out: str | Path | None = None,
    gap: float = 0.15,
    duck: float = 0.35,
) -> VoiceResult:
    """Speak the plan's lines at their shots' times and mux them into the stitched video.

    Lines in the same shot without an explicit offset follow each other with a short gap.
    When the video has its own generated sound, it is kept and lowered to `duck` under speech.
    """
    run = RunDir(run_dir)
    report_path = run.root / "stitch.json"
    if not report_path.is_file():
        raise AudioError("stitch the run first; voice-over needs stitch.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    video = Path(report["output"])
    duration = float(report["duration"])
    starts: dict[str, float] = {}
    for item in report.get("timeline", []):
        starts.setdefault(item["shot"], item["start"])
    plan = run.load_plan()

    track = np.zeros(int(round(duration * SAMPLE_RATE)) + 1, dtype=np.float32)
    placed: list[PlacedLine] = []
    cursor: dict[str, float] = {}
    for line in plan.lines:
        if line.shot_id not in starts:
            raise AudioError(f"shot {line.shot_id!r} is not in the stitched timeline")
        samples = _synth_cached(run, voice, line.text, line.voice)
        begin = starts[line.shot_id] + line.at
        if line.at == 0 and line.shot_id in cursor:
            begin = max(begin, cursor[line.shot_id] + gap)
        first = int(round(begin * SAMPLE_RATE))
        end_index = first + len(samples)
        clipped = end_index > len(track)
        if first < len(track):
            stop = min(end_index, len(track))
            track[first:stop] += samples[: stop - first]
        end = end_index / SAMPLE_RATE
        cursor[line.shot_id] = end
        placed.append(PlacedLine(line.shot_id, line.text, line.voice, begin, end, clipped))

    overlaps = []
    ordered = sorted(placed, key=lambda p: p.start)
    for a, b in zip(ordered, ordered[1:], strict=False):
        if b.start < a.end:
            overlaps.append((a.text, b.text))

    native_path = run.root / "native_audio.wav"
    if report.get("audio") and native_path.is_file():
        native, rate = read_wav(native_path.read_bytes())
        native = resample(native, rate)[: len(track)]
        native = np.pad(native, (0, len(track) - len(native)))
        speaking = _envelope(track) > 0.01
        track = track + native * np.where(speaking, duck, 1.0).astype(np.float32)

    peak = float(np.abs(track).max()) if len(track) else 0.0
    if peak > 0.89:
        track *= 0.89 / peak
    audio_path = run.root / "voice.wav"
    audio_path.write_bytes(wav_bytes(track, SAMPLE_RATE))
    out_path = Path(out) if out else video.with_name(f"{video.stem}.voiced{video.suffix}")
    mux(video, audio_path, out_path)
    run.log("voiced", output=str(out_path), lines=len(placed))
    return VoiceResult(out_path, audio_path, placed, overlaps)


def _envelope(track: np.ndarray, window: float = 0.05) -> np.ndarray:
    size = max(1, int(window * SAMPLE_RATE))
    kernel = np.ones(size, dtype=np.float32) / size
    return np.convolve(np.abs(track), kernel, mode="same")
