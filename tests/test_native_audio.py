import json
import subprocess
from types import SimpleNamespace

import imageio_ffmpeg
import numpy as np

from shotwright.audio import FakeVoice, voice_over
from shotwright.media import SAMPLE_RATE, read_wav
from shotwright.run import FakeBackend, RunDir, Runner
from shotwright.run.diffusers_backend import native_audio
from shotwright.stitch import stitch

SHOTS = [
    {"id": "a", "prompt": "first", "duration": 1, "lines": [{"text": "Hello there."}]},
    {"id": "b", "prompt": "second", "duration": 1, "transition": "continuous"},
    {"id": "c", "prompt": "third", "duration": 2},
]


def has_audio(path):
    probe = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(path)], capture_output=True, text=True
    )
    return "Audio:" in probe.stderr


def render(tmp_path, registry, tiny_plan, audio=True):
    plan = tiny_plan(SHOTS)
    result = Runner(FakeBackend(audio=audio), runs_root=tmp_path / "runs", registry=registry).start(
        plan, name="r"
    )
    assert result.ok
    return plan, RunDir(result.run_dir)


def test_runner_saves_job_audio(tmp_path, registry, tiny_plan):
    plan, run = render(tmp_path, registry, tiny_plan)
    for job in plan.jobs:
        samples, rate = read_wav(run.audio_path(job.id).read_bytes())
        assert rate == 16000
        assert abs(len(samples) / rate - job.frames / job.fps) < 0.01


def test_stitch_muxes_native_audio_aligned_to_video(tmp_path, registry, tiny_plan):
    _, run = render(tmp_path, registry, tiny_plan)
    result = stitch(run.root)
    assert has_audio(result.path)
    report = json.loads((run.root / "stitch.json").read_text())
    assert report["audio"] is True
    track, rate = read_wav((run.root / "native_audio.wav").read_bytes())
    assert rate == SAMPLE_RATE
    assert abs(len(track) / rate - result.duration) < 0.01


def test_silent_runs_have_no_audio_track(tmp_path, registry, tiny_plan):
    _, run = render(tmp_path, registry, tiny_plan, audio=False)
    result = stitch(run.root)
    assert not has_audio(result.path)
    assert json.loads((run.root / "stitch.json").read_text())["audio"] is False


def test_voice_over_ducks_native_audio(tmp_path, registry, tiny_plan):
    _, run = render(tmp_path, registry, tiny_plan)
    stitch(run.root)
    result = voice_over(run.root, FakeVoice())
    assert has_audio(result.path)
    mixed, _ = read_wav(result.audio.read_bytes())
    native, _ = read_wav((run.root / "native_audio.wav").read_bytes())
    end_of_line = int(result.lines[0].end * SAMPLE_RATE)
    tail = slice(end_of_line + SAMPLE_RATE // 10, len(native))
    assert np.abs(mixed[tail] - native[tail]).max() < 0.05


def test_rerender_without_audio_drops_stale_file(tmp_path, registry, tiny_plan):
    plan, run = render(tmp_path, registry, tiny_plan)
    Runner(FakeBackend(audio=False), registry=registry).rerun(run.root, "c", seed=5)
    assert not run.audio_path("c").exists()
    assert run.audio_path("a").exists()


def test_native_audio_from_ltx_style_output():
    pipe = SimpleNamespace(
        vocoder=SimpleNamespace(config=SimpleNamespace(output_sampling_rate=24000))
    )
    stereo = np.stack([np.ones(480), -np.ones(480) * 0.5])[None].astype(np.float32)
    audio, rate = native_audio(pipe, SimpleNamespace(audio=stereo))
    assert rate == 24000
    assert audio.shape == (480,)
    assert np.allclose(audio, 0.25)
    assert native_audio(pipe, SimpleNamespace(frames=[1])) == (None, None)
