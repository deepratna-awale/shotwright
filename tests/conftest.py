from pathlib import Path

import pytest

from shotwright.plan import ShotList, compile_plan
from shotwright.profiles import Registry

TINY_PROFILE = """
profiles:
  - id: tiny-i2v
    family: test
    variant: tiny
    temporal_factor: 4
    spatial_multiple: 16
    native_fps: 8
    frames: {min: 5, default: 9, max: 17}
    resolution_buckets: [64x48]
    pixel_budget: 3072
    conditioning: [t2v, i2v]
    continuity: [last_frame_i2v]
    min_vram_gb: 4
    vram_presets:
      - {min_gb: 24, recipe: full}
      - {min_gb: 8, recipe: offload}
      - {min_gb: 4, recipe: tiny}
    license: test
    sources: [test]
"""


@pytest.fixture
def registry(tmp_path_factory) -> Registry:
    d = tmp_path_factory.mktemp("profiles")
    (d / "tiny.yaml").write_text(TINY_PROFILE)
    return Registry.load(extra_dirs=[d])


def tiny_spec(shots, **extra) -> ShotList:
    data = {
        "project": "tiny",
        "output": {"width": 64, "height": 48, "fps": 8},
        "defaults": {"model": "tiny-i2v", "seed": 3},
        "shots": shots,
    }
    data.update(extra)
    return ShotList.from_dict(data, base_dir=Path.cwd())


@pytest.fixture
def tiny_plan(registry):
    def build(shots, **extra):
        return compile_plan(tiny_spec(shots, **extra), registry=registry)

    return build
