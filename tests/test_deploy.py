import base64
from pathlib import Path

import pytest
from click.testing import CliRunner

from shotwright import serverless
from shotwright.cli import main
from shotwright.download import DownloadError, download_comfyui, download_diffusers
from shotwright.profiles import default_registry

SHOTS = {
    "project": "srv",
    "output": {"width": 320, "height": 176, "fps": 24},
    "defaults": {"model": "wan2.2-ti2v-5b", "seed": 1},
    "shots": [
        {"id": "one", "prompt": "a", "frames": 9},
        {"id": "two", "prompt": "b", "frames": 9, "transition": "continuous"},
    ],
}


def test_download_diffusers_repos():
    seen = []
    profile = default_registry().get("hunyuanvideo-1.5-480p")
    download_diffusers(profile, snapshot=lambda repo_id: seen.append(repo_id) or f"/c/{repo_id}")
    assert seen == [
        "hunyuanvideo-community/HunyuanVideo-1.5-480p_t2v",
        "hunyuanvideo-community/HunyuanVideo-1.5-480p_i2v",
    ]


def test_download_comfyui_links_into_model_folders(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    calls = []

    def fetch(repo_id, filename):
        calls.append((repo_id, filename))
        path = cache / Path(filename).name
        path.write_bytes(b"w")
        return str(path)

    profile = default_registry().get("wan2.2-ti2v-5b")
    models = tmp_path / "models"
    written = download_comfyui(profile, models, fetch=fetch)
    assert {p.relative_to(models).as_posix() for p in written} == {
        "diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
        "vae/wan2.2_vae.safetensors",
        "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    }
    assert all(p.read_bytes() == b"w" for p in written)
    download_comfyui(profile, models, fetch=fetch)
    assert len(calls) == 3


def test_bundled_workflow_names_match_comfyui_files():
    from importlib import resources

    profile = default_registry().get("wan2.2-ti2v-5b")
    names = {Path(f.file).name for f in profile.backends.comfyui_files}
    graph = (resources.files("shotwright") / "workflows" / "wan2.2-ti2v-5b.t2v.json").read_text()
    for name in names:
        assert name in graph


def test_download_errors():
    profile = default_registry().get("ltx-2")
    with pytest.raises(DownloadError, match="ComfyUI files"):
        download_comfyui(profile, "x", fetch=lambda **k: "")


def test_cli_download_unknown_model():
    r = CliRunner().invoke(main, ["download", "no-such-model"])
    assert r.exit_code != 0


@pytest.fixture(autouse=True)
def fresh_backends():
    serverless._backends.clear()
    yield
    serverless._backends.clear()


def test_serverless_renders_and_stitches(tmp_path):
    response = serverless.handler(
        {
            "input": {
                "shot_list": SHOTS,
                "backend": "fake",
                "runs_dir": str(tmp_path),
                "name": "job",
                "return_video": True,
            }
        }
    )
    assert response["failed"] == {}
    assert response["done"] == ["one", "two"]
    assert Path(response["video"]).is_file()
    assert response["frames"] > 9
    assert base64.b64decode(response["video_base64"])[4:8] == b"ftyp"


def test_serverless_accepts_yaml_and_resume(tmp_path):
    import yaml

    first = serverless.handler(
        {
            "input": {
                "shot_list": yaml.safe_dump(SHOTS),
                "backend": "fake",
                "runs_dir": str(tmp_path),
            }
        }
    )
    again = serverless.handler(
        {
            "input": {
                "shot_list": SHOTS,
                "backend": "fake",
                "resume": first["run_dir"],
            }
        }
    )
    assert again["cached"] == ["one", "two"]


def test_serverless_reports_plan_problems():
    bad = {**SHOTS, "shots": [{"id": "x", "prompt": "a", "frames": 9, "transition": "continuous"}]}
    response = serverless.handler({"input": {"shot_list": bad, "backend": "fake"}})
    assert response["error"] == "the shot list has problems"
    assert response["problems"]
    assert serverless.handler({"input": {}})["error"]
