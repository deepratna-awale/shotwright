from click.testing import CliRunner

from shotwright.cli import main
from shotwright.doctor import run_checks


def test_checks_cover_the_basics(monkeypatch):
    monkeypatch.setenv("FAL_KEY", "x")
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    checks = {c.name: c for c in run_checks()}
    assert checks["ffmpeg"].ok and checks["ffmpeg"].required
    assert checks["fal key"].ok and checks["fal key"].detail == "$FAL_KEY"
    assert not checks["replicate key"].ok
    assert "torch" in checks and "disk" in checks


def test_unreachable_comfyui_is_reported():
    checks = {c.name: c for c in run_checks("http://127.0.0.1:9")}
    assert not checks["comfyui"].ok


def test_cli_doctor():
    r = CliRunner().invoke(main, ["doctor"])
    assert r.exit_code == 0, r.output
    assert "ffmpeg" in r.output
