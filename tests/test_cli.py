import json

from click.testing import CliRunner

from shotwright.cli import main


def run(*args):
    return CliRunner().invoke(main, list(args))


def test_snap_text():
    result = run("snap", "ltx-2", "--size", "1920x1080", "--duration", "5")
    assert result.exit_code == 0, result.output
    assert "frames    121 at 24 fps" in result.output
    assert "generate  1920x1088" in result.output
    assert "crop" in result.output


def test_snap_json():
    result = run("snap", "wan2.2-ti2v-5b", "--duration", "20", "--overlap", "1", "--json")
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["segments"] == [121, 121, 121, 121]
    assert data["frames"] == 481


def test_snap_unknown_model():
    result = run("snap", "nope")
    assert result.exit_code != 0
    assert "unknown model" in result.output


def test_models_filter():
    result = run("models", "--conditioning", "flf2v")
    assert result.exit_code == 0
    assert "wan2.1-flf2v-14b-720p" in result.output
    assert "ltx-2" not in result.output


def test_models_json():
    result = run("models", "--family", "hunyuanvideo", "--json")
    data = json.loads(result.output)
    assert {p["id"] for p in data} == {"hunyuanvideo-1.5-480p", "hunyuanvideo-1.5-720p"}


def test_version():
    result = run("--version")
    assert result.exit_code == 0
    assert "shotwright" in result.output
