import math

import pytest

from shotwright import snap
from shotwright.profiles import default_registry
from shotwright.snap import legal_frames_at_least, legal_frames_at_most

ALL_IDS = [p.id for p in default_registry()]


def fields(result):
    return {a.field for a in result.adjustments}


def test_ltx_full_hd_example():
    r = snap("ltx-2", size="1920x1080", duration=5)
    assert (r.frames, r.width, r.height) == (121, 1920, 1088)
    assert (r.output_width, r.output_height) == (1920, 1080)
    assert r.postprocess.resize_to is None
    assert (r.postprocess.crop.top, r.postprocess.crop.bottom) == (4, 4)
    assert fields(r) == {"frames", "size", "output"}


def test_wan_five_seconds_is_81_frames():
    r = snap("wan2.1-t2v-14b", size="1280x720", duration=5)
    assert r.frames == 81
    assert (r.width, r.height) == (1280, 720)
    assert fields(r) == {"frames"}


def test_never_shorter_than_requested():
    r = snap("wan2.1-t2v-1.3b", duration=3)
    assert r.frames == 49
    assert r.duration >= 3


def test_minimum_frames():
    r = snap("ltx-2", frames=2)
    assert r.frames == 9
    assert "minimum" in r.adjustments[0].reason


def test_default_request_uses_profile_defaults():
    r = snap("wan2.2-ti2v-5b")
    assert (r.frames, r.width, r.height, r.fps) == (121, 1280, 704, 24)
    assert not r.adjustments


def test_bucket_with_upscale():
    r = snap("wan2.1-i2v-14b-720p", size="1920x1080", duration=5, fps=24)
    assert (r.width, r.height, r.frames) == (1280, 720, 81)
    assert r.postprocess.resize_to.width == 1920
    assert r.postprocess.crop is None
    assert r.postprocess.fps_method == "interpolate"


def test_portrait_picks_portrait_bucket():
    r = snap("hunyuanvideo-1.5-720p", size="1080x1920")
    assert (r.width, r.height) == (720, 1280)


def test_odd_aspect_without_bucket_scales_to_budget():
    p = default_registry().get("wan2.1-t2v-14b")
    r = snap(p, size="2560x1080")
    assert r.width % 16 == 0 and r.height % 16 == 0
    assert r.width * r.height <= p.pixel_budget
    assert r.postprocess.crop is not None or r.postprocess.resize_to is not None


def test_aspect_only():
    r = snap("ltx-2", aspect="9:16")
    assert r.width < r.height
    assert r.width % 32 == 0 and r.height % 32 == 0
    assert (r.output_width, r.output_height) == (r.width, r.height)


def test_split_long_request():
    r = snap("wan2.1-t2v-14b", duration=20, overlap=1)
    assert r.is_split
    assert all(n <= 81 and (n - 1) % 4 == 0 for n in r.segments)
    assert r.frames >= 20 * 16
    assert r.overlap == 1


def test_split_without_overlap():
    r = snap("wan2.2-ti2v-5b", duration=10)
    assert r.segments == (121, 121)
    assert r.frames == 242


def test_fps_resample_down():
    r = snap("hunyuanvideo-1.5-480p", duration=2, fps=12)
    assert r.postprocess.fps_method == "resample"
    assert r.fps == 24


def test_frames_given_at_output_fps():
    r = snap("wan2.1-t2v-14b", frames=120, fps=24)
    assert r.frames == 81
    assert not r.is_split


def test_bad_arguments():
    with pytest.raises(ValueError):
        snap("ltx-2", size="1280x720", width=1280, height=720)
    with pytest.raises(ValueError):
        snap("ltx-2", width=1280)
    with pytest.raises(ValueError):
        snap("ltx-2", duration=2, frames=49)
    with pytest.raises(ValueError):
        snap("ltx-2", duration=30, overlap=300)


def test_frame_helpers():
    assert legal_frames_at_least(80, 4) == 81
    assert legal_frames_at_least(81, 4) == 81
    assert legal_frames_at_most(80, 8) == 73
    assert legal_frames_at_least(0, 8) == 1


@pytest.mark.parametrize("model_id", ALL_IDS)
@pytest.mark.parametrize("duration", [0.3, 1, 2.5, 4, 5, 7.3, 12, 31])
@pytest.mark.parametrize("size", ["1920x1080", "1080x1920", "1000x1000", "640x360", "3840x2160"])
def test_snapped_output_is_always_legal(model_id, duration, size):
    p = default_registry().get(model_id)
    r = snap(p, size=size, duration=duration, overlap=1)
    assert p.is_legal_size(r.width, r.height)
    assert r.width * r.height <= p.pixel_budget
    for n in r.segments:
        assert p.is_legal_frames(n)
        assert p.frames.min <= n <= p.frames.max
    assert r.frames >= math.ceil(duration * p.native_fps - 1e-9)
    w, h = (int(x) for x in size.split("x"))
    assert (r.output_width, r.output_height) == (w, h)
    pp = r.postprocess
    final_w = pp.resize_to.width if pp.resize_to else r.width
    final_h = pp.resize_to.height if pp.resize_to else r.height
    if pp.crop:
        final_w -= pp.crop.left + pp.crop.right
        final_h -= pp.crop.top + pp.crop.bottom
    assert (final_w, final_h) == (w, h)
