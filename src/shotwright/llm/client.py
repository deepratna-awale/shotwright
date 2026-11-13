from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class LLMError(RuntimeError):
    pass


class Transport(Protocol):
    def post(
        self, url: str, payload: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]: ...


class HttpTransport:
    def __init__(self, timeout: float = 120.0):
        self.timeout = timeout

    def post(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", **headers},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:1000]
            raise LLMError(f"LLM request failed ({exc.code}): {body}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise LLMError(f"cannot reach {url}: {exc}") from exc


def image_part(image: np.ndarray, max_side: int = 768) -> dict[str, Any]:
    picture = Image.fromarray(np.asarray(image, dtype=np.uint8))
    picture.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    picture.save(buffer, format="JPEG", quality=90)
    data = base64.b64encode(buffer.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}


def parse_json(text: str) -> Any:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
    if start < 0:
        raise LLMError(f"expected JSON from the model, got: {text[:200]}")
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise LLMError(f"the model returned invalid JSON: {exc}") from exc
    return value


class LLMClient:
    """Minimal client for any OpenAI-compatible chat completions endpoint.

    Responses are cached on disk by request content, so re-running a plan does not
    re-bill identical calls and results stay reproducible.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        cache_dir: str | Path | None = ".shotwright-cache/llm",
        temperature: float = 0.4,
        transport: Transport | None = None,
    ):
        self.model = model or os.environ.get("SHOTWRIGHT_LLM_MODEL")
        self.base_url = (
            base_url or os.environ.get("SHOTWRIGHT_LLM_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.api_key = (
            api_key or os.environ.get("SHOTWRIGHT_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        )
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.temperature = temperature
        self.transport = transport or HttpTransport()
        self.calls = 0

    def _key(self, payload: dict[str, Any]) -> str:
        blob = json.dumps({"base": self.base_url, **payload}, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    def chat(self, system: str, user: str | list[dict[str, Any]]) -> str:
        if not self.model:
            raise LLMError("no LLM model set; pass --llm-model or set SHOTWRIGHT_LLM_MODEL")
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        cached = self.cache_dir / f"{self._key(payload)}.json" if self.cache_dir else None
        if cached is not None and cached.is_file():
            return json.loads(cached.read_text(encoding="utf-8"))["content"]

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = self.transport.post(f"{self.base_url}/chat/completions", payload, headers)
        self.calls += 1
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response from the model: {str(response)[:300]}") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("the model returned an empty reply")
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps({"content": content}), encoding="utf-8")
        return content

    def chat_json(self, system: str, user: str | list[dict[str, Any]]) -> Any:
        return parse_json(self.chat(system, user))
