import pytest

from shotwright.plan.spec import Look, PromptParts
from shotwright.profiles import get_profile
from shotwright.profiles.schema import I2VPrompt, PromptRules
from shotwright.prompting import AdaptedPrompt, adapt_prompt, count_tokens, fit_tokens

PARTS = PromptParts(
    subject="A red fox",
    action="The fox trots across the snow",
    setting="A birch forest at dawn",
    lighting="soft golden backlight",
    camera="slow dolly in",
    audio="crunching snow and distant birds",
)


def profile(model_id: str, **rules):
    base = {"encoder": "test-encoder", "max_tokens": 512, "t2v_words": (5, 200)}
    base.update(rules)
    return get_profile(model_id).model_copy(update={"prompt": PromptRules(**base)})


WAN = "wan2.2-ti2v-5b"
HY = "hunyuanvideo-1.5-720p"
LTX = "ltx-2"


def order(text: str, *needles: str) -> list[int]:
    idx = [text.index(n) for n in needles]
    assert idx == sorted(idx), text
    return idx


def test_free_text_t2v_appends_style_last():
    out = adapt_prompt(profile(HY), mode="t2v", prompt="a cat  sleeps", style="film grain")
    assert isinstance(out, AdaptedPrompt)
    assert out.text == "a cat sleeps. film grain."
    assert out.notes == ()


def test_free_text_t2v_wan_puts_style_first():
    out = adapt_prompt(profile(WAN), mode="t2v", prompt="a cat sleeps", style="film grain")
    assert out.text == "film grain. a cat sleeps."


def test_style_not_repeated_when_already_in_prompt():
    out = adapt_prompt(
        profile(LTX), mode="t2v", prompt="A cat, Film Grain look.", style="film grain"
    )
    assert out.text == "A cat, Film Grain look."


def test_wan_t2v_order():
    out = adapt_prompt(profile(WAN), mode="t2v", parts=PARTS, style="cinematic")
    assert out.text.startswith("cinematic. soft golden backlight, slow dolly in. A red fox.")
    order(out.text, "cinematic", "golden", "fox.", "trots", "birch")
    assert "crunching" not in out.text


def test_wan_t2v_anime_drops_cinematic_tags():
    out = adapt_prompt(profile(WAN), mode="t2v", parts=PARTS, look=Look.ANIME)
    assert out.text.startswith("anime style, 2D cel shading. A red fox.")
    order(out.text, "anime", "fox.", "trots", "birch", "golden", "dolly")
    assert "backlight, slow" not in out.text


def test_free_prompt_goes_after_subject_and_action():
    out = adapt_prompt(profile(WAN), mode="t2v", parts=PARTS, prompt="Its breath steams")
    order(out.text, "trots", "Its breath steams.", "birch")


def test_wan_i2v_motion_only():
    out = adapt_prompt(profile(WAN), mode="i2v", parts=PARTS, motion="The fox leaps", style="x")
    assert out.text == "The fox leaps. slow dolly in."


def test_wan_i2v_without_motion_falls_back_to_t2v():
    parts = PARTS.model_copy(update={"action": None})
    out = adapt_prompt(profile(WAN), mode="i2v", parts=parts)
    assert "A red fox." in out.text and "birch" in out.text


def test_hunyuan_t2v_order_style_last():
    out = adapt_prompt(profile(HY), mode="t2v", parts=PARTS, style="35mm film")
    order(out.text, "fox.", "trots", "birch", "dolly", "golden", "35mm film")
    assert out.text.endswith("35mm film.")


def test_hunyuan_i2v_action_and_camera_only():
    out = adapt_prompt(profile(HY), mode="i2v", parts=PARTS)
    assert out.text == "The fox trots across the snow. slow dolly in."
    no_cam = PARTS.model_copy(update={"camera": None})
    assert adapt_prompt(profile(HY), mode="i2v", parts=no_cam).text == (
        "The fox trots across the snow."
    )


def test_ltx_t2v_action_first_with_style_prefix():
    out = adapt_prompt(profile(LTX), mode="t2v", parts=PARTS, style="documentary")
    assert out.text.startswith("Style: documentary. The fox trots")
    order(out.text, "trots", "fox.", "birch", "dolly", "golden", "crunching")


def test_ltx_i2v_motion_describes_changes_only():
    out = adapt_prompt(profile(LTX), mode="i2v", parts=PARTS, motion="The fox turns")
    assert out.text == "The fox turns. slow dolly in. crunching snow and distant birds."


def test_ltx_i2v_describe_first_uses_parts():
    p = profile(LTX, i2v=I2VPrompt.DESCRIBE_FIRST)
    out = adapt_prompt(p, mode="i2v", parts=PARTS, motion="The fox turns")
    order(out.text, "fox.", "birch", "golden", "turns", "dolly", "crunching")


def test_ltx_i2v_describe_first_prefers_start():
    p = profile(LTX, i2v=I2VPrompt.DESCRIBE_FIRST)
    out = adapt_prompt(p, mode="i2v", parts=PARTS, motion="The fox turns", start="A fox stands")
    assert out.text.startswith("A fox stands. The fox turns.")
    assert "birch" not in out.text


def test_free_text_i2v_describe_first():
    p = profile(LTX, i2v=I2VPrompt.DESCRIBE_FIRST)
    out = adapt_prompt(p, mode="i2v", prompt="A fox in snow", motion="it runs")
    assert out.text == "A fox in snow. it runs."


def test_free_text_i2v_motion_model_keeps_motion_only():
    out = adapt_prompt(profile(WAN), mode="i2v", prompt="A fox in snow", motion="it runs")
    assert out.text == "it runs."
    assert out.notes


def test_free_text_i2v_without_motion_uses_prompt():
    out = adapt_prompt(profile(WAN), mode="i2v", prompt="A fox runs")
    assert out.text == "A fox runs."


def test_default_family_order():
    p = profile(WAN).model_copy(update={"family": "other"})
    out = adapt_prompt(p, mode="t2v", parts=PARTS, style="noir")
    order(out.text, "fox.", "trots", "birch", "golden", "dolly", "noir")


def test_anime_style_used_only_without_explicit_style():
    out = adapt_prompt(profile(HY), mode="t2v", prompt="a girl waves", look=Look.ANIME)
    assert out.text.endswith("anime style, 2D cel shading.")
    out = adapt_prompt(profile(HY), mode="t2v", prompt="a girl waves", look="anime", style="ghibli")
    assert "cel shading" not in out.text and out.text.endswith("ghibli.")


def test_negative_handling():
    p = profile(WAN, negative="blurry")
    assert adapt_prompt(p, mode="t2v", prompt="x").negative == "blurry"
    assert adapt_prompt(p, mode="t2v", prompt="x", negative="ugly").negative == "ugly"
    off = profile(LTX, negative="blurry", uses_negative=False)
    out = adapt_prompt(off, mode="t2v", prompt="x", negative="ugly")
    assert out.negative is None
    assert any("negative" in n for n in out.notes)
    assert adapt_prompt(off, mode="t2v", prompt="x").notes == ()


def test_trims_to_token_limit_with_note():
    p = profile(WAN, max_tokens=12, tokens_per_word=1.0, encoder="umT5-XXL")
    text = "One two three four five. Six seven eight nine ten. Eleven twelve thirteen."
    out = adapt_prompt(p, mode="t2v", prompt=text)
    assert out.text == "One two three four five. Six seven eight nine ten."
    assert out.dropped == "Eleven twelve thirteen."
    assert out.tokens <= out.limit == 12
    assert not out.exact
    assert any("umT5-XXL's 12" in n and "trimmed 3 tokens" in n for n in out.notes)


def test_word_range_note_without_trimming():
    p = profile(HY, t2v_words=(1, 3))
    out = adapt_prompt(p, mode="t2v", prompt="one two three four five")
    assert out.text == "one two three four five."
    assert any("more than the 3" in n for n in out.notes)


def test_rejects_unknown_mode():
    with pytest.raises(ValueError):
        adapt_prompt(profile(WAN), mode="v2v", prompt="x")


def test_fit_tokens_keeps_whole_sentences():
    f = fit_tokens("A b c. D e f. G h i.", 6, ratio=1.0)
    assert (f.text, f.dropped, f.tokens, f.trimmed) == ("A b c. D e f.", "G h i.", 6, True)


def test_fit_tokens_untouched_when_it_fits():
    f = fit_tokens("A  b c.", 10, ratio=1.0)
    assert (f.text, f.dropped, f.trimmed) == ("A b c.", "", False)


def test_fit_tokens_trims_tags():
    f = fit_tokens("red, green, blue, yellow", 3, ratio=1.0, tags=True)
    assert f.text == "red, green"
    assert f.dropped == "blue, yellow"
    assert f.tokens == count_tokens("red, green", ratio=1.0)[0] <= 3


def test_fit_tokens_cuts_words_of_single_long_sentence():
    f = fit_tokens("one two three four five six seven eight. nine.", 5, ratio=1.0)
    assert f.text == "one two three four five"
    assert f.dropped.startswith("six")


def test_wan_anime_look_drops_style_terms_from_negative():
    from shotwright.profiles import get_profile

    profile = get_profile("wan2.2-ti2v-5b")
    real = adapt_prompt(profile, mode="t2v", prompt="A girl on a roof.")
    anime = adapt_prompt(profile, mode="t2v", prompt="A girl on a roof.", look=Look.ANIME)
    assert "画作" in real.negative
    assert "画作" not in anime.negative and "风格" not in anime.negative
