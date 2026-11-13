# shotwright

Plan, run and stitch long videos across open-weight video diffusion models.

Every video model makes clips of a few seconds under its own rules. Wan 2.1 wants 4n+1 frames at 16 fps, LTX-2 wants 8n+1 frames with sides divisible by 32, HunyuanVideo 1.5 wants 4n+1 frames at 24 fps. Anything longer than one clip means chaining shots, carrying continuity across them and fixing drift at the seams. shotwright handles that layer so you don't rebuild it by hand for every model.

> Status: early development. The constraint table, shot planner, diffusers and ComfyUI runners, and stitcher work today; hosted runners are next.

## Install

```bash
pip install -e .                # planning and stitching
pip install -e ".[diffusers]"   # plus local rendering through diffusers
```

## Render a shot list

Write the film as shots. Each shot is snapped to its model's rules, long shots are split into chained segments, and `continuous` shots start from the last frame of the shot before.

```bash
shotwright plan examples/harbour.yaml                       # see the job graph
shotwright run examples/harbour.yaml --backend diffusers --vram 24
shotwright resume runs/harbour-dawn-20261113T120000         # after a crash, reuses finished clips
shotwright rerun runs/harbour-dawn-20261113T120000 deck --seed 7
shotwright stitch runs/harbour-dawn-20261113T120000 --fps 24
```

The stitcher drops the duplicated frame at each chained seam, resizes and crops to the output size, converts frame rates, and corrects colour drift across chains. `--color seam` matches each clip to the one before it. `--color anchor` also pulls long chains back toward the look of their first shot.

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

## Render through ComfyUI

Point shotwright at a running ComfyUI server and it queues one workflow per job, uploading the init frame for chained shots:

```bash
shotwright run examples/harbour.yaml --backend comfyui --comfy-url http://127.0.0.1:8188
```

Workflows are ComfyUI API-format JSON files named `<model id>.t2v.json` and `<model id>.i2v.json`. A bundled pair covers `wan2.2-ti2v-5b` using ComfyUI's native Wan nodes. For other models, export your own workflow with "Save (API)" and replace the values shotwright should control with placeholders:

| Placeholder | Value |
|---|---|
| `{{prompt}}`, `{{negative}}` | shot prompt and negative prompt |
| `{{seed}}`, `{{frames}}`, `{{width}}`, `{{height}}`, `{{fps}}` | the snapped job settings |
| `{{image}}` | uploaded init frame (i2v workflows only) |
| `{{prefix}}` | output filename prefix for SaveImage or a video save node |
| `{{steps\|20}}` | anything else, with a default; a shot's `params` override it |

Put the files in a directory and pass `--workflows DIR`, or set `SHOTWRIGHT_WORKFLOW_PATH`. A `<model id>+<recipe>.t2v.json` file is used for that VRAM recipe, so a low-memory variant can swap in GGUF or block-swap nodes.

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
