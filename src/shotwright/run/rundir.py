from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..plan import Job, Plan

PLAN_FILE = "plan.json"
MANIFEST_FILE = "manifest.jsonl"
JOBS_DIR = "jobs"
CLIP_FILE = "clip.mp4"
LAST_FRAME_FILE = "last.png"
META_FILE = "meta.json"
AUDIO_FILE = "audio.wav"


def new_run_name(project: str) -> str:
    return f"{project}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"


class RunDir:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    @classmethod
    def create(cls, parent: str | Path, plan: Plan, name: str | None = None) -> RunDir:
        run = cls(Path(parent) / (name or new_run_name(plan.project)))
        run.root.mkdir(parents=True, exist_ok=True)
        run.save_plan(plan)
        return run

    @property
    def plan_path(self) -> Path:
        return self.root / PLAN_FILE

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_FILE

    def exists(self) -> bool:
        return self.plan_path.is_file()

    def load_plan(self) -> Plan:
        return Plan.load(self.plan_path)

    def save_plan(self, plan: Plan) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.plan_path.with_suffix(".json.tmp")
        tmp.write_text(plan.to_json() + "\n", encoding="utf-8")
        tmp.replace(self.plan_path)

    def job_dir(self, job_id: str) -> Path:
        return self.root / JOBS_DIR / job_id

    def clip_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / CLIP_FILE

    def last_frame_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / LAST_FRAME_FILE

    def audio_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / AUDIO_FILE

    def meta_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / META_FILE

    def read_meta(self, job_id: str) -> dict[str, Any] | None:
        path = self.meta_path(job_id)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def write_meta(self, job_id: str, meta: dict[str, Any]) -> None:
        path = self.meta_path(job_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)

    def log(self, event: str, job: str | None = None, **detail: Any) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        record = {"ts": round(time.time(), 3), "event": event}
        if job is not None:
            record["job"] = job
        record.update(detail)
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def events(self) -> list[dict[str, Any]]:
        if not self.manifest_path.is_file():
            return []
        out = []
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out


def job_cache_key(job: Job, input_hash: str | None) -> str:
    payload = job.model_dump(mode="json")
    payload.pop("id", None)
    payload.pop("shot_id", None)
    payload["_input"] = input_hash
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()
