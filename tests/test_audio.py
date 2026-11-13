import json
import subprocess

import imageio_ffmpeg
import numpy as np
import pytest
from click.testing import CliRunner

from shotwright.audio import (
    AudioError,
    FakeVoice,
    OpenAISpeech,
    read_wav,
    resample,
    voice_over,
    wav_bytes,
)
from shotwright.cli import main
from shotwright.plan import PlanError
from shotwright.run import FakeBackend, RunDir, Runner
from shotwright.stitch import stitch

SHOTS = [
    {
        "id": "a",
        "prompt": "harbour",
        "duration": 2,
        "lines": [
            {"text": "Morning.", "speaker": "old"},
            {"text": "Nets are dry.", "speaker": "young"},
        ],
    },
    {"id": "b", "prompt": "deck", "duration": 1, "transition": "continuous"},
    {"id": "c", "prompt": "sea", "duration": 1, "lines": [{"text": "Go.", "at": 0.5}]},
]


@pytest.fixture
def stitched(tmp_path, registry, tiny_plan):
    plan = tiny_plan(SHOTS, voices={"old": "onyx", "default": "nova"})
    result = Runner(FakeBackend(), runs_root=tmp_path / "runs", registry=registry).start(
        plan, name="r"
    )
    stitch(result.run_dir)
    return plan, RunDir(result.run_dir)


def has_audio(path):
    probe = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(path)], capture_output=True, text=True
    )
    return "Audio:" in probe.stderr


def test_wav_round_trip_and_resample():
    samples = np.sin(np.linspace(0, 20, 1000)).astype(np.float32) * 0.5
    back, rate = read_wav(wav_bytes(samples, 16000))
    assert rate == 16000
    assert np.abs(back - samples).max() < 1e-3
    assert len(resample(samples, 16000, 48000)) == 3000
    with pytest.raises(AudioError):
        read_wav(b"nope")


def test_plan_carries_lines_with_voices(tiny_plan):
    plan = tiny_plan(SHOTS, voices={"old": "onyx", "default": "nova"})
    assert [(x.shot_id, x.voice) for x in plan.lines] == [
        ("a", "onyx"),
        ("a", "nova"),
        ("c", "nova"),
    ]


def test_lines_are_policy_checked(tiny_plan):
    with pytest.raises(PlanError, match="line"):
        tiny_plan([{"id": "a", "prompt": "x", "duration": 1, "lines": [{"text": "a naked child"}]}])


def test_stitch_records_timeline(stitched):
    _, run = stitched
    report = json.loads((run.root / "stitch.json").read_text())
    shots = [t["shot"] for t in report["timeline"]]
    assert shots[0] == "a" and shots[-1] == "c"
    assert report["timeline"][0]["start"] == 0


def test_voice_over_places_and_muxes(stitched):
    plan, run = stitched
    voice = FakeVoice()
    result = voice_over(run.root, voice)
    assert result.path.name == "final.voiced.mp4"
    assert has_audio(result.path)
    assert [c[1] for c in voice.calls] == ["onyx", "nova", "nova"]
    first, second, third = result.lines
    assert first.start == 0
    assert second.start >= first.end
    timeline = json.loads((run.root / "stitch.json").read_text())["timeline"]
    c_start = next(t["start"] for t in timeline if t["shot"] == "c")
    assert third.start == pytest.approx(c_start + 0.5)


def test_voice_lines_are_cached(stitched):
    _, run = stitched
    voice_over(run.root, FakeVoice())
    again = FakeVoice()
    voice_over(run.root, again)
    assert again.calls == []


def test_long_line_is_reported(tmp_path, registry, tiny_plan):
    plan = tiny_plan([{"id": "a", "prompt": "x", "duration": 1, "lines": [{"text": "x" * 200}]}])
    result = Runner(FakeBackend(), runs_root=tmp_path / "runs", registry=registry).start(
        plan, name="r"
    )
    stitch(result.run_dir)
    voiced = voice_over(result.run_dir, FakeVoice())
    assert voiced.lines[0].clipped


def test_needs_stitch_first(tmp_path, registry, tiny_plan):
    result = Runner(FakeBackend(), runs_root=tmp_path / "runs", registry=registry).start(
        tiny_plan(SHOTS), name="r"
    )
    with pytest.raises(AudioError, match="stitch"):
        voice_over(result.run_dir, FakeVoice())


def test_openai_speech_request(monkeypatch):
    sent = {}

    class Response:
        def __init__(self, data):
            self.data = data

        def read(self):
            return self.data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(request, timeout):
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data)
        sent["auth"] = request.get_header("Authorization")
        return Response(wav_bytes(np.zeros(100, dtype=np.float32), 24000))

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    speech = OpenAISpeech(model="tts-x", base_url="http://tts.local/v1", api_key="k")
    samples, rate = speech.synthesize("hello", "nova")
    assert rate == 24000 and len(samples) == 100
    assert sent["url"] == "http://tts.local/v1/audio/speech"
    assert sent["body"] == {
        "model": "tts-x",
        "input": "hello",
        "voice": "nova",
        "response_format": "wav",
    }
    assert sent["auth"] == "Bearer k"


def test_speech_needs_model(monkeypatch):
    monkeypatch.delenv("SHOTWRIGHT_TTS_MODEL", raising=False)
    with pytest.raises(AudioError, match="SHOTWRIGHT_TTS_MODEL"):
        OpenAISpeech().synthesize("x", None)


def test_cli_voice(stitched):
    _, run = stitched
    r = CliRunner().invoke(main, ["voice", str(run.root), "--fake"])
    assert r.exit_code == 0, r.output
    assert "3 lines" in r.output
