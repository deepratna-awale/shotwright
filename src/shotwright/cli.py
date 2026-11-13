from __future__ import annotations

import functools
import json

import click

from . import __version__
from .plan import PlanError, ShotList, compile_plan
from .profiles import Conditioning, UnknownModelError, list_models
from .review import ReviewError, contact_sheet
from .run import ComfyUIBackend, DiffusersBackend, FakeBackend, HostedBackend, Runner, RunResult
from .snap import snap as snap_request
from .stitch import COLOR_MODES, METHODS, StitchError
from .stitch import stitch as stitch_run

CONDITIONING_CHOICES = [c.value for c in Conditioning]


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="shotwright")
def main() -> None:
    """Plan, run and stitch long videos across open-weight video models."""


@main.command()
@click.argument("model")
@click.option("--size", help="Requested output size, e.g. 1920x1080.")
@click.option("--aspect", help="Aspect ratio when no size is given, e.g. 16:9.")
@click.option("--duration", type=float, help="Requested duration in seconds.")
@click.option("--frames", type=int, help="Requested frame count at the output fps.")
@click.option("--fps", type=float, help="Output frame rate. Defaults to the model's native fps.")
@click.option(
    "--overlap",
    type=int,
    default=0,
    show_default=True,
    help="Frames shared between segments when a request is split.",
)
@click.option("--json", "as_json", is_flag=True, help="Print machine-readable JSON.")
def snap(model, size, aspect, duration, frames, fps, overlap, as_json):
    """Print legal generation settings for MODEL and every adjustment made."""
    try:
        result = snap_request(
            model,
            size=size,
            aspect=aspect,
            duration=duration,
            frames=frames,
            fps=fps,
            overlap=overlap,
        )
    except (UnknownModelError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        click.echo(json.dumps(result.to_dict(), indent=2, default=str))
        return

    click.echo(f"model     {result.model}")
    if result.is_split:
        click.echo(
            f"segments  {len(result.segments)} x {result.segments[0]} frames"
            + (f", {result.overlap} overlapping" if result.overlap else "")
        )
    click.echo(f"frames    {result.frames} at {result.fps:g} fps ({result.duration:.2f} s)")
    click.echo(f"generate  {result.width}x{result.height}")
    click.echo(
        f"output    {result.output_width}x{result.output_height} at {result.output_fps:g} fps"
    )
    if result.adjustments:
        click.echo("adjustments")
        for adj in result.adjustments:
            click.echo(f"  - {adj}")


@main.command()
@click.option(
    "--conditioning",
    "-c",
    multiple=True,
    type=click.Choice(CONDITIONING_CHOICES),
    help="Only models that accept this conditioning. Repeatable.",
)
@click.option("--vram", type=float, help="Only models that run within this many GB of VRAM.")
@click.option("--family", help="Only models from this family, e.g. wan.")
@click.option("--json", "as_json", is_flag=True, help="Print machine-readable JSON.")
def models(conditioning, vram, family, as_json):
    """List known model profiles."""
    found = list_models(conditioning=conditioning, max_vram_gb=vram, family=family)
    if as_json:
        click.echo(json.dumps([p.model_dump(mode="json") for p in found], indent=2))
        return
    if not found:
        click.echo("no models match")
        return
    rows = [
        (
            p.id,
            f"{p.temporal_factor}n+1",
            f"{p.native_fps:g}",
            f"/{p.spatial_multiple}",
            str(p.frames.max),
            ",".join(c.value for c in p.conditioning),
            f"{p.min_vram_gb:g}",
            p.license,
        )
        for p in found
    ]
    header = ("model", "frames", "fps", "size", "max", "conditioning", "vram", "license")
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    for row in [header, *rows]:
        click.echo("  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip())


@main.command()
@click.argument("shot_list", type=click.Path(exists=True, dir_okay=False))
@click.option("--json", "as_json", is_flag=True, help="Print the compiled plan as JSON.")
@click.option("-o", "--output", type=click.Path(dir_okay=False), help="Write the plan JSON here.")
def plan(shot_list, as_json, output):
    """Compile and validate SHOT_LIST without running anything."""
    try:
        compiled = compile_plan(ShotList.from_yaml(shot_list))
    except PlanError as exc:
        raise click.ClickException(f"the shot list has problems:\n{exc}") from exc
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if output:
        compiled.save(output)
    click.echo(compiled.to_json() if as_json else compiled.summary())


def _llm_options(f):
    f = click.option("--llm-model", help="Chat model. Defaults to $SHOTWRIGHT_LLM_MODEL.")(f)
    f = click.option(
        "--llm-url", help="OpenAI-compatible base URL. Defaults to $SHOTWRIGHT_LLM_BASE_URL."
    )(f)
    return f


@main.command()
@click.argument("shot_list", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o", "--output", type=click.Path(dir_okay=False), help="Defaults to <name>.enhanced.yaml."
)
@click.option(
    "--bible", "bible_path", type=click.Path(dir_okay=False), help="Use or write this story bible."
)
@click.option("--no-bible", is_flag=True, help="Skip the story bible.")
@_llm_options
def enhance(shot_list, output, bible_path, no_bible, llm_model, llm_url):
    """Expand each shot's brief into a full prompt for its model with an LLM.

    A story bible of characters, locations and style is written first so every shot
    describes them the same way. Uses any OpenAI-compatible API; set
    SHOTWRIGHT_LLM_API_KEY or OPENAI_API_KEY.
    """
    from pathlib import Path

    from .llm import LLMClient, LLMError, StoryBible, build_bible, write_prompts

    source = Path(shot_list)
    out = Path(output) if output else source.with_name(f"{source.stem}.enhanced.yaml")
    spec = ShotList.from_yaml(source)
    client = LLMClient(model=llm_model, base_url=llm_url)
    try:
        bible = None
        if not no_bible:
            path = Path(bible_path) if bible_path else source.with_name(f"{source.stem}.bible.yaml")
            if bible_path and path.is_file():
                bible = StoryBible.load(path)
                click.echo(f"bible     {path} (existing)")
            else:
                bible = build_bible(spec, client)
                bible.save(path)
                click.echo(f"bible     {path}")
        enhanced = write_prompts(spec, client, bible)
    except LLMError as exc:
        raise click.ClickException(str(exc)) from exc
    enhanced.to_yaml(out)
    click.echo(f"wrote     {out} ({len(enhanced.shots)} shots, {client.calls} LLM calls)")


@main.command()
@click.argument("models", nargs=-1, required=True)
@click.option(
    "--for",
    "target",
    type=click.Choice(["diffusers", "comfyui"]),
    default="diffusers",
    show_default=True,
)
@click.option(
    "--comfy-models",
    type=click.Path(file_okay=False),
    default="ComfyUI/models",
    show_default=True,
    help="ComfyUI models folder, used with --for comfyui.",
)
def download(models, target, comfy_models):
    """Download weights for MODELS into the Hugging Face cache or a ComfyUI models folder."""
    from .download import DownloadError, download_comfyui, download_diffusers
    from .profiles import get_profile

    for model in models:
        try:
            profile = get_profile(model)
            if target == "diffusers":
                paths = download_diffusers(profile)
            else:
                paths = download_comfyui(profile, comfy_models)
        except (DownloadError, UnknownModelError) as exc:
            raise click.ClickException(str(exc)) from exc
        for path in paths:
            click.echo(f"{model:<24} {path}")


@main.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-o", "--output", type=click.Path(dir_okay=False), help="Defaults to <video>.voiced.mp4."
)
@click.option("--tts-model", help="Speech model. Defaults to $SHOTWRIGHT_TTS_MODEL.")
@click.option(
    "--tts-url", help="OpenAI-compatible speech server. Defaults to $SHOTWRIGHT_TTS_BASE_URL."
)
@click.option("--fake", is_flag=True, help="Use placeholder tones instead of a speech server.")
def voice(run_dir, output, tts_model, tts_url, fake):
    """Speak the shot list's lines over the stitched video in RUN_DIR."""
    from .audio import AudioError, FakeVoice, OpenAISpeech, voice_over

    speaker = FakeVoice() if fake else OpenAISpeech(model=tts_model, base_url=tts_url)
    try:
        result = voice_over(run_dir, speaker, out=output)
    except AudioError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"wrote     {result.path} ({len(result.lines)} lines)")
    for line in result.lines:
        if line.clipped:
            click.echo(f"  check   line in {line.shot} runs past the end of the video")
    for first, second in result.overlaps:
        click.echo(f"  check   lines overlap: {first[:30]!r} and {second[:30]!r}")


@main.command()
@click.option("--comfy-url", help="Also check that this ComfyUI server answers.")
def doctor(comfy_url):
    """Check what this machine can run."""
    from .doctor import run_checks

    checks = run_checks(comfy_url)
    width = max(len(c.name) for c in checks)
    for check in checks:
        mark = "ok  " if check.ok else ("FAIL" if check.required else "--  ")
        click.echo(f"{mark}  {check.name.ljust(width)}  {check.detail}")
    if any(c.required and not c.ok for c in checks):
        raise SystemExit(1)


BACKENDS = ["fake", "diffusers", "comfyui", "hosted"]


def _backend(
    name: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
    comfy_url: str = "http://127.0.0.1:8188",
    workflows: tuple[str, ...] = (),
):
    if name == "fake":
        return FakeBackend()
    if name == "diffusers":
        return DiffusersBackend(device=device, dtype=dtype)
    if name == "comfyui":
        return ComfyUIBackend(url=comfy_url, workflow_dirs=workflows)
    if name == "hosted":
        return HostedBackend(spec_dirs=workflows)
    raise click.ClickException(f"unknown backend {name!r}")


def _report(result: RunResult) -> None:
    click.echo(f"run       {result.run_dir}")
    click.echo(
        f"jobs      {len(result.done)} rendered, {len(result.cached)} reused, "
        f"{len(result.failed)} failed, {len(result.skipped)} skipped"
    )
    for job_id, error in result.failed.items():
        click.echo(f"  failed  {job_id}: {error}")
    for job_id in result.skipped:
        click.echo(f"  skipped {job_id}")
    if not result.ok:
        raise SystemExit(1)


def _runner_options(f):
    @functools.wraps(f)
    def wrapper(
        *args,
        backend,
        device,
        dtype,
        comfy_url,
        workflows,
        llm_continuity,
        llm_model,
        llm_url,
        **kwargs,
    ):
        hook = None
        if llm_continuity:
            from .llm import LLMClient, continuation_hook

            hook = continuation_hook(LLMClient(model=llm_model, base_url=llm_url))
        return f(
            *args,
            backend=_backend(backend, device, dtype, comfy_url, workflows),
            prompt_hook=hook,
            **kwargs,
        )

    wrapper = _llm_options(wrapper)
    wrapper = click.option(
        "--llm-continuity",
        is_flag=True,
        help="Describe each chained job's starting frame with a vision LLM and adapt its prompt.",
    )(wrapper)

    wrapper = click.option(
        "--workflows",
        multiple=True,
        type=click.Path(exists=True, file_okay=False),
        help="Extra directory of ComfyUI workflows or hosted API specs. Repeatable.",
    )(wrapper)
    wrapper = click.option(
        "--comfy-url", default="http://127.0.0.1:8188", show_default=True, help="ComfyUI server."
    )(wrapper)
    wrapper = click.option("--dtype", default="bfloat16", show_default=True)(wrapper)
    wrapper = click.option("--device", default="cuda", show_default=True)(wrapper)
    wrapper = click.option(
        "--backend", type=click.Choice(BACKENDS), default="fake", show_default=True
    )(wrapper)
    return click.option("--vram", type=float, help="GPU memory in GB, used to pick presets.")(
        wrapper
    )


@main.command()
@click.argument("shot_list", type=click.Path(exists=True, dir_okay=False))
@_runner_options
@click.option("--runs-dir", default="runs", show_default=True, type=click.Path(file_okay=False))
@click.option("--name", help="Run directory name. Defaults to project name plus a timestamp.")
def run(shot_list, backend, prompt_hook, vram, runs_dir, name):
    """Compile SHOT_LIST and render every job into a run directory."""
    try:
        compiled = compile_plan(ShotList.from_yaml(shot_list))
    except PlanError as exc:
        raise click.ClickException(f"the shot list has problems:\n{exc}") from exc
    runner = Runner(backend, runs_root=runs_dir, vram_gb=vram, prompt_hook=prompt_hook)
    runner.on(
        "job_done", lambda e: click.echo(f"  done    {e.job_id} ({e.detail['seconds']:.1f} s)")
    )
    _report(runner.start(compiled, name=name))


@main.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@_runner_options
def resume(run_dir, backend, prompt_hook, vram):
    """Continue RUN_DIR, rendering only jobs that are missing or out of date."""
    runner = Runner(backend, vram_gb=vram, prompt_hook=prompt_hook)
    runner.on(
        "job_done", lambda e: click.echo(f"  done    {e.job_id} ({e.detail['seconds']:.1f} s)")
    )
    _report(runner.resume(run_dir))


@main.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("job_id")
@click.option("--seed", type=int, help="New seed. Defaults to the current seed plus one.")
@_runner_options
def rerun(run_dir, job_id, seed, backend, prompt_hook, vram):
    """Re-render JOB_ID in RUN_DIR with a new seed, plus every job chained from it."""
    runner = Runner(backend, vram_gb=vram, prompt_hook=prompt_hook)
    try:
        _report(runner.rerun(run_dir, job_id, seed=seed))
    except KeyError as exc:
        raise click.ClickException(f"no job {job_id!r} in this run") from exc


@main.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option("-o", "--output", type=click.Path(dir_okay=False), help="Defaults to final.mp4.")
@click.option("--fps", type=float, help="Output fps. Defaults to the plan's output fps.")
@click.option(
    "--method",
    type=click.Choice(METHODS),
    default="resample",
    show_default=True,
    help="How to convert from each model's native fps.",
)
@click.option("--crf", type=int, default=16, show_default=True, help="x264 quality.")
@click.option(
    "--color",
    type=click.Choice(COLOR_MODES),
    default="seam",
    show_default=True,
    help="Colour drift correction across chained clips.",
)
@click.option(
    "--anchor-strength",
    type=click.FloatRange(0, 1),
    default=0.5,
    show_default=True,
    help="How far --color anchor pulls each clip back toward the start of its chain.",
)
def stitch(run_dir, output, fps, method, crf, color, anchor_strength):
    """Join the rendered clips in RUN_DIR into one video."""
    try:
        result = stitch_run(
            run_dir,
            out=output,
            fps=fps,
            method=method,
            crf=crf,
            color=color,
            anchor_strength=anchor_strength,
        )
    except StitchError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"wrote     {result.path}")
    click.echo(f"length    {result.frames} frames at {result.fps:g} fps ({result.duration:.2f} s)")
    for seam in result.seams:
        if seam.flagged:
            click.echo(
                f"  check   {seam.before} -> {seam.after}: visible jump "
                f"({seam.jump:.1f}x normal motion). Try `shotwright rerun` with a new seed."
            )


@main.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-o", "--output", type=click.Path(dir_okay=False), help="Defaults to contact_sheet.png."
)
@click.option("--frames", "per_job", type=click.IntRange(1, 24), default=6, show_default=True)
@click.option(
    "--width", "thumb_width", type=click.IntRange(32, 1024), default=192, show_default=True
)
def sheet(run_dir, output, per_job, thumb_width):
    """Write a contact sheet of every job in RUN_DIR, with seam scores after a stitch."""
    try:
        path = contact_sheet(run_dir, out=output, per_job=per_job, thumb_width=thumb_width)
    except ReviewError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"wrote     {path}")


if __name__ == "__main__":
    main()
