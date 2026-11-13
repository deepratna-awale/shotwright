# Container images

One Dockerfile with three targets. Each one is built on CUDA 12.8 with PyTorch and `shotwright[diffusers]` already installed.

| Target | Image | What it runs |
|---|---|---|
| `base` | `ghcr.io/deepratna-awale/shotwright:<version>` | A pod with SSH and the `shotwright` CLI; renders through diffusers |
| `comfyui` | `ghcr.io/deepratna-awale/shotwright:<version>-comfyui` | The same, plus ComfyUI on port 8188 |
| `serverless` | `ghcr.io/deepratna-awale/shotwright:<version>-serverless` | A RunPod serverless worker that renders and stitches a shot list per request |

Weights are not baked in. They go on a network volume mounted at `/workspace`, so pods start fast and every pod shares one download:

- diffusers weights live in `/workspace/hf` (`HF_HOME`).
- ComfyUI models live in `/workspace/models`.
- Runs are written to `/workspace/runs`.

Set `SHOTWRIGHT_PRELOAD=wan2.2-ti2v-5b,ltx-2` to download models on first boot. Downloads are skipped when the files are already on the volume.

Build locally:

```bash
docker build -f docker/Dockerfile --target comfyui -t shotwright:comfyui .
```

## Pods

Start a pod from the `comfyui` image with a network volume and expose port 8188. Then, from the pod's terminal or over SSH:

```bash
shotwright run shots.yaml --backend comfyui --vram 24
shotwright stitch runs/<run>
```

Or skip ComfyUI and use `--backend diffusers`.

## Serverless

Create an endpoint from the `serverless` image with the network volume attached (it's mounted at `/runpod-volume`), and send:

```json
{
  "input": {
    "shot_list": { "project": "demo", "output": {"width": 1280, "height": 704, "fps": 24},
                   "defaults": {"model": "wan2.2-ti2v-5b", "seed": 1},
                   "shots": [{"id": "a", "prompt": "...", "duration": 5}] },
    "vram_gb": 48,
    "stitch": {"color": "anchor"},
    "return_video": true
  }
}
```

The response includes the run directory, the per-job status, the stitched video's path, any seams flagged for review and, when `return_video` is set and the video is under 15 MB, the video as base64. To continue a run after a worker timeout, send `"resume": "<run_dir>"`.

On serverless, weights are read from `/runpod-volume/hf` and runs are written to `/runpod-volume/shotwright/runs`.
