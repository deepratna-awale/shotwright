from __future__ import annotations

import json

import click

from . import __version__
from .plan import PlanError, ShotList, compile_plan
from .profiles import Conditioning, UnknownModelError, list_models
from .snap import snap as snap_request

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


if __name__ == "__main__":
    main()
