from __future__ import annotations

import importlib
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .profiles import ModelProfile


class DownloadError(RuntimeError):
    pass


def _hub() -> Any:
    try:
        return importlib.import_module("huggingface_hub")
    except ImportError as exc:
        raise DownloadError(
            "downloading needs huggingface_hub: pip install 'shotwright[diffusers]'"
        ) from exc


def diffusers_repos(profile: ModelProfile) -> list[str]:
    repos = [profile.backends.diffusers_repo, profile.backends.diffusers_repo_i2v]
    return [r for r in dict.fromkeys(repos) if r]


def download_diffusers(
    profile: ModelProfile, snapshot: Callable[..., str] | None = None
) -> list[str]:
    repos = diffusers_repos(profile)
    if not repos:
        raise DownloadError(f"{profile.id} has no diffusers weights listed")
    snapshot = snapshot or _hub().snapshot_download
    return [snapshot(repo_id=repo) for repo in repos]


def download_comfyui(
    profile: ModelProfile,
    models_dir: str | Path,
    fetch: Callable[..., str] | None = None,
) -> list[Path]:
    """Fetch the files a profile's ComfyUI workflows load into ComfyUI's models folders."""
    files = profile.backends.comfyui_files
    if not files:
        raise DownloadError(f"{profile.id} has no ComfyUI files listed")
    fetch = fetch or _hub().hf_hub_download
    written = []
    for item in files:
        target = Path(models_dir) / item.folder / Path(item.file).name
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            cached = Path(fetch(repo_id=item.repo, filename=item.file))
            try:
                target.symlink_to(cached)
            except OSError:
                shutil.copyfile(cached, target)
        written.append(target)
    return written
