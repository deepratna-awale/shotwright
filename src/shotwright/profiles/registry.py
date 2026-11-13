from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from importlib import resources
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import yaml

from .schema import Conditioning, ModelProfile

PROFILE_PATH_ENV = "SHOTWRIGHT_PROFILE_PATH"
ENTRY_POINT_GROUP = "shotwright.profiles"


class UnknownModelError(KeyError):
    def __init__(self, model_id: str, known: Iterable[str]):
        self.model_id = model_id
        self.known = sorted(known)
        super().__init__(model_id)

    def __str__(self) -> str:
        return f"unknown model {self.model_id!r}; known models: {', '.join(self.known)}"


def _read_yaml(text: str, origin: str) -> list[dict[str, Any]]:
    data = yaml.safe_load(text)
    if data is None:
        return []
    if isinstance(data, dict):
        data = data.get("profiles", [data])
    if not isinstance(data, list):
        raise ValueError(f"{origin}: expected a profile mapping or a list of profiles")
    return data


def _bundled_documents() -> Iterator[tuple[str, str]]:
    root = resources.files("shotwright.profiles") / "data"
    for item in sorted(root.iterdir(), key=lambda p: p.name):
        if item.name.endswith((".yaml", ".yml")):
            yield f"shotwright:{item.name}", item.read_text(encoding="utf-8")


def _directory_documents(directory: Path) -> Iterator[tuple[str, str]]:
    for path in sorted(directory.glob("*.y*ml")):
        yield str(path), path.read_text(encoding="utf-8")


def _external_directories() -> list[Path]:
    dirs: list[Path] = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        target = ep.load()
        dirs.append(Path(target() if callable(target) else target))
    env = os.environ.get(PROFILE_PATH_ENV, "")
    dirs.extend(Path(p) for p in env.split(os.pathsep) if p)
    return dirs


def _resolve(raw: dict[str, dict[str, Any]]) -> dict[str, ModelProfile]:
    resolved: dict[str, dict[str, Any]] = {}

    def merged(model_id: str, chain: tuple[str, ...]) -> dict[str, Any]:
        if model_id in resolved:
            return resolved[model_id]
        if model_id in chain:
            raise ValueError(f"profile inheritance cycle: {' -> '.join(chain + (model_id,))}")
        if model_id not in raw:
            raise UnknownModelError(model_id, raw)
        entry = raw[model_id]
        base_id = entry.get("base")
        data = dict(merged(base_id, chain + (model_id,))) if base_id else {}
        for key, value in entry.items():
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                data[key] = {**data[key], **value}
            else:
                data[key] = value
        resolved[model_id] = data
        return data

    return {model_id: ModelProfile(**merged(model_id, ())) for model_id in raw}


class Registry:
    def __init__(self, profiles: dict[str, ModelProfile]):
        self._profiles = profiles

    @classmethod
    def load(cls, extra_dirs: Iterable[Path | str] = (), bundled: bool = True) -> Registry:
        raw: dict[str, dict[str, Any]] = {}
        documents: list[tuple[str, str]] = list(_bundled_documents()) if bundled else []
        for directory in [*_external_directories(), *map(Path, extra_dirs)]:
            documents.extend(_directory_documents(directory))
        for origin, text in documents:
            for entry in _read_yaml(text, origin):
                if "id" not in entry:
                    raise ValueError(f"{origin}: profile without an id")
                raw[entry["id"]] = entry
        return cls(_resolve(raw))

    def get(self, model_id: str) -> ModelProfile:
        try:
            return self._profiles[model_id]
        except KeyError:
            raise UnknownModelError(model_id, self._profiles) from None

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._profiles

    def __iter__(self) -> Iterator[ModelProfile]:
        return iter(sorted(self._profiles.values(), key=lambda p: p.id))

    def __len__(self) -> int:
        return len(self._profiles)

    def filter(
        self,
        conditioning: Iterable[Conditioning | str] = (),
        max_vram_gb: float | None = None,
        family: str | None = None,
    ) -> list[ModelProfile]:
        wanted = [Conditioning(c) for c in conditioning]
        out = []
        for profile in self:
            if family and profile.family != family:
                continue
            if any(not profile.supports(c) for c in wanted):
                continue
            if max_vram_gb is not None and profile.min_vram_gb > max_vram_gb:
                continue
            out.append(profile)
        return out


_default: Registry | None = None


def default_registry() -> Registry:
    global _default
    if _default is None:
        _default = Registry.load()
    return _default


def reset_default_registry() -> None:
    global _default
    _default = None


def get_profile(model_id: str) -> ModelProfile:
    return default_registry().get(model_id)


def list_models(
    conditioning: Iterable[Conditioning | str] = (),
    max_vram_gb: float | None = None,
    family: str | None = None,
) -> list[ModelProfile]:
    return default_registry().filter(conditioning, max_vram_gb, family)
