import pytest
from pydantic import ValidationError

from shotwright.profiles import (
    Conditioning,
    ModelProfile,
    Registry,
    UnknownModelError,
    default_registry,
    list_models,
)

ALL_IDS = [p.id for p in default_registry()]


def test_bundled_profiles_load():
    ids = set(ALL_IDS)
    for expected in (
        "wan2.1-t2v-1.3b",
        "wan2.1-flf2v-14b-720p",
        "wan2.2-ti2v-5b",
        "ltx-2",
        "ltx-2.3",
        "hunyuanvideo-1.5-720p",
    ):
        assert expected in ids


@pytest.mark.parametrize("model_id", ALL_IDS)
def test_profile_defaults_are_legal(model_id):
    p = default_registry().get(model_id)
    for n in (p.frames.min, p.frames.default, p.frames.max):
        assert p.is_legal_frames(n)
    for b in p.resolution_buckets:
        assert p.is_legal_size(b.width, b.height)
        assert b.pixels <= p.pixel_budget
    assert p.license
    assert p.sources


def test_inheritance_merges_backends():
    p = default_registry().get("ltx-2.3")
    assert p.backends.diffusers == "LTX2Pipeline"
    assert p.backends.diffusers_repo == "Lightricks/LTX-2.3"
    assert p.temporal_factor == 8


def test_documented_rules():
    reg = default_registry()
    wan21 = reg.get("wan2.1-t2v-14b")
    assert (wan21.temporal_factor, wan21.native_fps, wan21.frames.default) == (4, 16, 81)
    ltx = reg.get("ltx-2")
    assert (ltx.temporal_factor, ltx.spatial_multiple) == (8, 32)
    hy = reg.get("hunyuanvideo-1.5-480p")
    assert (hy.temporal_factor, hy.native_fps) == (4, 24)
    five_b = reg.get("wan2.2-ti2v-5b")
    assert (five_b.spatial_multiple, five_b.native_fps) == (32, 24)


def test_filter_by_conditioning_and_vram():
    flf = list_models(conditioning=["flf2v"])
    assert [p.id for p in flf] == ["wan2.1-flf2v-14b-720p"]
    small = list_models(max_vram_gb=8)
    assert {p.id for p in small} == {"wan2.1-t2v-1.3b"}
    assert all(p.family == "ltx" for p in list_models(family="ltx"))
    assert all(p.supports(Conditioning.I2V) for p in list_models(conditioning=["i2v"]))


def test_unknown_model_lists_known():
    with pytest.raises(UnknownModelError) as exc:
        default_registry().get("sora")
    assert "wan2.2-ti2v-5b" in str(exc.value)


def test_preset_selection():
    p = default_registry().get("wan2.1-t2v-14b")
    assert p.preset_for(80).recipe == "full"
    assert p.preset_for(48).recipe == "fp8"
    assert p.preset_for(24).recipe == "fp8+offload"
    assert p.preset_for(12) is None


def test_invalid_profile_rejected():
    with pytest.raises(ValidationError):
        ModelProfile(
            id="bad",
            family="x",
            variant="x",
            temporal_factor=4,
            spatial_multiple=16,
            native_fps=16,
            frames={"min": 5, "default": 80, "max": 81},
            pixel_budget=1000,
            conditioning=["t2v"],
            min_vram_gb=8,
            license="x",
        )
    with pytest.raises(ValidationError):
        ModelProfile(
            id="bad",
            family="x",
            variant="x",
            temporal_factor=4,
            spatial_multiple=16,
            native_fps=16,
            frames={"min": 5, "default": 81, "max": 81},
            resolution_buckets=["1280x721"],
            pixel_budget=10**6,
            conditioning=["t2v"],
            min_vram_gb=8,
            license="x",
        )


def test_user_profile_directory(tmp_path):
    (tmp_path / "mine.yaml").write_text(
        "profiles:\n"
        "  - id: my-finetune\n"
        "    base: wan2.2-ti2v-5b\n"
        "    variant: custom\n"
        "    license: other\n"
    )
    reg = Registry.load(extra_dirs=[tmp_path])
    p = reg.get("my-finetune")
    assert p.spatial_multiple == 32
    assert p.license == "other"
    assert "wan2.2-ti2v-5b" in reg


def test_inheritance_cycle_rejected(tmp_path):
    (tmp_path / "loop.yaml").write_text("profiles:\n  - {id: a, base: b}\n  - {id: b, base: a}\n")
    with pytest.raises(ValueError, match="cycle"):
        Registry.load(extra_dirs=[tmp_path], bundled=False)
