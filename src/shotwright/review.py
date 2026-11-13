from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .media import read_video, resize_crop
from .run.rundir import RunDir

LABEL_WIDTH = 220
PAD = 6
BG = (24, 24, 24)
TEXT = (230, 230, 230)
MUTED = (150, 150, 150)
FLAG = (230, 70, 60)
OK = (90, 190, 110)


class ReviewError(RuntimeError):
    pass


def contact_sheet(
    run_dir: str | Path,
    out: str | Path | None = None,
    per_job: int = 6,
    thumb_width: int = 192,
) -> Path:
    """Write a PNG with evenly spaced frames from every job and each seam's score."""
    run = RunDir(run_dir)
    if not run.exists():
        raise ReviewError(f"no plan.json in {run.root}")
    plan = run.load_plan()
    seams = {}
    report = run.root / "stitch.json"
    if report.is_file():
        seams = {s["after"]: s for s in json.loads(report.read_text())["seams"]}

    thumb_height = max(1, round(thumb_width * plan.output_height / plan.output_width))
    row_height = thumb_height + 2 * PAD
    width = LABEL_WIDTH + per_job * (thumb_width + PAD) + PAD
    sheet = Image.new("RGB", (width, row_height * len(plan.jobs)), BG)
    draw = ImageDraw.Draw(sheet)

    for row, job in enumerate(plan.jobs):
        top = row * row_height
        seam = seams.get(job.id)
        lines = [
            (job.id, TEXT),
            (f"{job.model}", MUTED),
            (f"{job.frames} f @ {job.fps:g} fps, seed {job.seed}", MUTED),
        ]
        if seam:
            colour = FLAG if seam.get("flagged") else OK if seam["kind"] == "chain" else MUTED
            lines.append((f"{seam['kind']} from {seam['before']}: jump {seam['jump']:.2f}", colour))
        for i, (text, colour) in enumerate(lines):
            draw.text((PAD, top + PAD + i * 14), text, fill=colour)

        clip = run.clip_path(job.id)
        if not clip.is_file():
            draw.text((LABEL_WIDTH, top + row_height // 2 - 6), "not rendered", fill=FLAG)
            continue
        frames, _ = read_video(clip)
        picks = np.unique(np.linspace(0, len(frames) - 1, per_job).round().astype(int))
        for col, index in enumerate(picks):
            frame = resize_crop(frames[index], job.resize_to, job.crop)
            thumb = Image.fromarray(frame).resize((thumb_width, thumb_height))
            left = LABEL_WIDTH + col * (thumb_width + PAD)
            sheet.paste(thumb, (left, top + PAD))
            if col == 0 and seam and seam.get("flagged"):
                draw.rectangle(
                    (left - 2, top + PAD - 2, left + thumb_width + 1, top + PAD + thumb_height + 1),
                    outline=FLAG,
                    width=2,
                )
            draw.text((left + 3, top + PAD + 2), str(index), fill=TEXT)

    out_path = Path(out) if out else run.root / "contact_sheet.png"
    sheet.save(out_path)
    return out_path
