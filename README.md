# shotwright

Plan, run and stitch long videos across open-weight video diffusion models.

Every video model makes clips of a few seconds under its own rules. Wan 2.1 wants 4n+1 frames at 16 fps, LTX-2 wants 8n+1 frames with sides divisible by 32, HunyuanVideo 1.5 wants 4n+1 frames at 24 fps. Anything longer than one clip means chaining shots, carrying continuity across them and fixing drift at the seams. shotwright handles that layer so you don't rebuild it by hand for every model.

> Status: early development. The constraint table and `snap` are usable today; the shot planner, runners (ComfyUI, diffusers, hosted APIs) and stitcher are next.

## Install

```bash
pip install -e .
```

## Snap a request to legal values

```bash
$ shotwright snap ltx-2 --size 1920x1080 --duration 5
model     ltx-2
frames    121 at 24 fps (5.04 s)
generate  1920x1088
output    1920x1080 at 24 fps
adjustments
  - frames: 120 -> 121 (frame count must be 8n+1)
  - size: 1920x1080 -> 1920x1088 (sides must be multiples of 32)
  - output: 1920x1088 -> 1920x1080 (crop left 0, top 4, right 0, bottom 4)
```

Requests longer than a model's clip limit come back as segments:

```bash
shotwright snap wan2.2-ti2v-5b --duration 20 --overlap 1
```

From Python:

```python
import shotwright

job = shotwright.snap("wan2.1-i2v-14b-720p", size="1920x1080", duration=5, fps=24)
job.frames, job.width, job.height  # 81, 1280, 720
for adjustment in job.adjustments:
    print(adjustment)
```

## List models

```bash
shotwright models --conditioning flf2v
shotwright models --vram 24 --family wan
```

Bundled profiles cover Wan 2.1, Wan 2.2, LTX-2, LTX-2.3, LTX-2.5 and HunyuanVideo 1.5. Each profile records its source links and license; shotwright ships no model weights, so each model's license applies to whoever downloads it.

## Custom profiles

Profiles are YAML. Put your own in a directory and point `SHOTWRIGHT_PROFILE_PATH` at it, or ship them in a package through the `shotwright.profiles` entry point. A fine-tune only needs the fields that differ from its base:

```yaml
profiles:
  - id: my-wan-finetune
    base: wan2.2-ti2v-5b
    variant: "my fine-tune"
    license: apache-2.0
```

## License

Apache-2.0
