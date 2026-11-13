import json

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from shotwright.cli import main
from shotwright.llm import (
    LLMClient,
    LLMError,
    StoryBible,
    build_bible,
    continue_prompt,
    describe_frame,
    family_guide,
    load_template,
    parse_json,
    write_prompts,
)
from shotwright.plan import ShotList, compile_plan

SPEC = {
    "project": "harbour",
    "output": {"width": 1280, "height": 704, "fps": 24},
    "defaults": {"model": "wan2.2-ti2v-5b", "seed": 1},
    "shots": [
        {"id": "wide", "prompt": "boats in a misty harbour", "duration": 4},
        {"id": "deck", "prompt": "move onto the deck", "duration": 4, "transition": "continuous"},
        {"id": "face", "prompt": "the fisherman smiles", "duration": 3, "model": "ltx-2"},
    ],
}

BIBLE = {
    "characters": [{"name": "Fisherman", "description": "weathered man in his sixties"}],
    "locations": [{"name": "Harbour", "description": "stone quay, wooden boats"}],
    "props": [],
    "style": "35mm film, soft dawn light",
}


class Scripted:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def post(self, url, payload, headers):
        self.requests.append((url, payload, headers))
        reply = self.replies.pop(0)
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return {"choices": [{"message": {"content": content}}]}


def client(replies, tmp_path=None, **kw):
    return LLMClient(
        model="test-model",
        base_url="http://llm.local/v1",
        api_key="k",
        cache_dir=tmp_path,
        transport=Scripted(replies),
        **kw,
    )


def test_parse_json_variants():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('Sure!\n```json\n{"a": 2}\n```') == {"a": 2}
    assert parse_json('here: {"a": 3} thanks') == {"a": 3}
    with pytest.raises(LLMError):
        parse_json("no json here")


def test_templates_and_guides_exist():
    for name in ("story_bible", "prompt_writer", "frame_describer", "continuation"):
        assert "JSON" in load_template(name)
    assert "Wan" in family_guide("wan")
    assert "LTX" in family_guide("ltx")
    assert family_guide("unknown-family") == family_guide(None)


def test_request_shape_and_auth():
    c = client(["hello"])
    assert c.chat("sys", "user") == "hello"
    url, payload, headers = c.transport.requests[0]
    assert url == "http://llm.local/v1/chat/completions"
    assert payload["model"] == "test-model"
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    assert headers == {"Authorization": "Bearer k"}


def test_missing_model_is_clear(monkeypatch):
    monkeypatch.delenv("SHOTWRIGHT_LLM_MODEL", raising=False)
    c = LLMClient(cache_dir=None, transport=Scripted(["x"]))
    with pytest.raises(LLMError, match="SHOTWRIGHT_LLM_MODEL"):
        c.chat("s", "u")


def test_responses_are_cached(tmp_path):
    first = client(["one"], tmp_path)
    assert first.chat("s", "u") == "one"
    second = client([], tmp_path)
    assert second.chat("s", "u") == "one"
    assert second.calls == 0


def test_bad_replies_raise():
    c = client([{"unexpected": True}])
    c.transport.post = lambda *a: {"error": "nope"}
    with pytest.raises(LLMError, match="unexpected response"):
        c.chat("s", "u")
    with pytest.raises(LLMError, match="empty"):
        client(["  "]).chat("s", "u")


def test_build_bible():
    spec = ShotList.from_dict(SPEC)
    c = client([BIBLE])
    bible = build_bible(spec, c)
    assert bible.characters[0].name == "Fisherman"
    assert bible.style == "35mm film, soft dawn light"
    sent = c.transport.requests[0][1]["messages"][1]["content"]
    assert "boats in a misty harbour" in sent


def test_write_prompts_keeps_briefs_and_uses_family_guides():
    spec = ShotList.from_dict(SPEC)
    replies = [{"prompt": f"full prompt {i}"} for i in range(3)]
    c = client(replies)
    out = write_prompts(spec, c, StoryBible.from_dict(BIBLE))

    assert [s.prompt for s in out.shots] == ["full prompt 0", "full prompt 1", "full prompt 2"]
    assert [s.brief for s in out.shots] == [s.prompt for s in spec.shots]
    assert out.style == "35mm film, soft dawn light"
    assert spec.shots[0].brief is None

    systems = [r[1]["messages"][0]["content"] for r in c.transport.requests]
    users = [r[1]["messages"][1]["content"] for r in c.transport.requests]
    assert "Wan responds best" in systems[0]
    assert "LTX responds best" in systems[2]
    assert "Fisherman: weathered man" in users[0]
    assert "continues directly from the previous shot:\nfull prompt 0" in users[1]
    assert "starts with a cut" in users[2]
    assert "4 seconds" in users[0]


def test_write_prompts_restarts_from_brief():
    spec = ShotList.from_dict(SPEC)
    once = write_prompts(spec, client([{"prompt": f"a{i}"} for i in range(3)]))
    c = client([{"prompt": f"b{i}"} for i in range(3)])
    twice = write_prompts(once, c)
    assert [s.brief for s in twice.shots] == [s.prompt for s in spec.shots]
    assert (
        "Brief:\nboats in a misty harbour" in c.transport.requests[0][1]["messages"][1]["content"]
    )


def test_enhanced_spec_round_trips_and_compiles(tmp_path):
    spec = ShotList.from_dict(SPEC)
    out = write_prompts(spec, client([{"prompt": f"p{i}"} for i in range(3)]))
    path = tmp_path / "e.yaml"
    out.to_yaml(path)
    again = ShotList.from_yaml(path)
    assert again.shots[1].brief == "move onto the deck"
    assert again.shots[1].transition.value == "continuous"
    plan = compile_plan(again)
    assert plan.jobs[0].prompt.startswith("p0")


def test_describe_and_continue():
    c = client([{"description": "a man on a deck"}, {"prompt": "rewritten"}])
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    assert describe_frame(image, c) == "a man on a deck"
    content = c.transport.requests[0][1]["messages"][1]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert continue_prompt("planned", "a man on a deck", c) == "rewritten"


def test_cli_enhance(tmp_path, monkeypatch):
    source = tmp_path / "shots.yaml"
    source.write_text(yaml.safe_dump(SPEC))
    scripted = Scripted([BIBLE] + [{"prompt": f"p{i}"} for i in range(3)])
    monkeypatch.setattr("shotwright.llm.client.HttpTransport", lambda: scripted)
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(main, ["enhance", str(source), "--llm-model", "m"])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "shots.bible.yaml").is_file()
    enhanced = ShotList.from_yaml(tmp_path / "shots.enhanced.yaml")
    assert enhanced.shots[2].prompt == "p2"
    assert "4 LLM calls" in r.output
