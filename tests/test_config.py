"""The packaged configuration read with a user's on top: field by field, forms only added."""
import json

import pytest

from llm2decision import ConfigError, DecisionClient, Noul
from llm2decision import config
from llm2decision.client import side_of
from llm_server import LLMServer


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM2DECISION_HOME", str(tmp_path))
    return tmp_path


def put(home, name, obj):
    (home / name).write_text(json.dumps(obj))


def test_merge_is_field_by_field():
    base = {"a": {"x": 1, "y": {"p": 1, "q": 2}}, "b": 2}
    assert config.merge(base, {"a": {"y": {"q": 3}}}) == {"a": {"x": 1, "y": {"p": 1, "q": 3}}, "b": 2}
    assert config.merge(base, {"a": {"x": None}, "b": None}) == {"a": {"y": {"p": 1, "q": 2}}}
    assert config.merge(base, {"c": [1]})["c"] == [1] and base == {"a": {"x": 1, "y": {"p": 1, "q": 2}}, "b": 2}


def test_one_temperature_changed_the_rest_kept(home):
    packaged = config.resolve("qwen3.8-27b@nebius")
    put(home, "models.json", {"qwen3.8-27b@nebius": {"calibration": {"choice": 1.5}}})
    b = config.resolve("qwen3.8-27b@nebius")
    assert b.calibration["choice"] == 1.5 and b.calibration["verdict"] == packaged.calibration["verdict"]
    assert b.model == packaged.model and b.transport == "completions" and b.options == packaged.options


def test_calibration_added_to_a_binding_without_one(home):
    name = next(k for k, v in config.bindings().items() if "calibration" not in v)
    put(home, "models.json", {name: {"calibration": {"verdict": 0.9}}})
    assert config.resolve(name).calibration == {"verdict": 0.9}


def test_calibration_removed_with_null(home):
    put(home, "models.json", {"qwen3.8-27b@nebius": {"calibration": None}})
    b = config.resolve("qwen3.8-27b@nebius")
    assert b.calibration == {} and b.fitted is None


def test_fitted_is_kept_apart_from_temperatures():
    b = config.resolve("qwen3.8-27b@nebius")
    assert "fitted" not in b.calibration and b.fitted == "2026-10-07"


@pytest.mark.parametrize("cal", [{"verdict": 0}, {"verdict": "hot"}, {"vibes": 1.0}])
def test_bad_calibration_is_refused(home, cal):
    put(home, "models.json", {"qwen3.8-27b@nebius": {"calibration": cal}})
    with pytest.raises(ConfigError, match="calibration"):
        config.resolve("qwen3.8-27b@nebius")


def test_packaged_dates_are_measurements_not_checks():
    for name in config.bindings():
        b = config.resolve(name)
        assert b.checked is None, name


def test_new_binding_from_the_user(home):
    put(home, "models.json", {"mine@local": {"provider": "local", "model": "m", "digits": "grouped"}})
    b = config.resolve("mine@local")
    assert b.model == "m" and b.digits == "grouped" and b.forms == config.forms()


def test_provider_changed_by_one_field(home):
    put(home, "providers.json", {"nebius": {"top_k": 7}})
    b = config.resolve("qwen3.8-27b@nebius")
    assert b.top_k == 7 and b.base_url.startswith("https://")


def test_broken_user_file_is_named(home):
    (home / "models.json").write_text("{nope")
    with pytest.raises(ConfigError, match="models.json"):
        config.bindings()
    put(home, "models.json", ["not", "an", "object"])
    with pytest.raises(ConfigError, match="object"):
        config.bindings()


def test_forms_only_add(home):
    put(home, "forms.json", {"true": ["Si", "yes"], "false": ["non"]})
    f = config.forms()
    assert f["true"].count("yes") == 1 and "si" in f["true"] and "non" in f["false"] and "unsure" in f["unsure"]
    assert side_of("[Si", f) == "true" and side_of("Non", f) == "false"


def test_forms_on_a_side_that_does_not_exist(home):
    put(home, "forms.json", {"maybe": ["perhaps"]})
    with pytest.raises(ConfigError, match="maybe"):
        config.forms()


def test_binding_forms_extend_the_packaged_ones(home):
    put(home, "models.json", {"mine@local": {"provider": "local", "model": "m", "forms": {"true": ["oui"]}}})
    b = config.resolve("mine@local")
    assert "oui" in b.forms["true"] and "yes" in b.forms["true"]
    assert "oui" not in config.resolve("qwen3.8-27b@nebius").forms["true"]


def test_binding_forms_reach_the_reading(home, monkeypatch):
    """A model that answers in its own word is read once the word is in its binding."""
    from llm2decision import transports
    from llm2decision.transports import Step
    put(home, "models.json", {"fr@local": {"provider": "local", "model": "m", "forms": {"true": ["oui"]}}})
    monkeypatch.setattr(transports.ChatContinue, "step",
                        lambda self, prompt, written: Step([("Oui", -0.1), ("Non", -2.4)], 20))
    s = LLMServer()
    try:
        with DecisionClient("fr@local", base_url=s.url) as c:
            a = c.system_one("x", {"q": Noul(instructions="?")}).nouls["q"]
        assert a.meta.answered and a.noul > 0.5
        with DecisionClient("m", base_url=s.url) as c:             # without the binding: not an answer
            assert c.system_one("x", {"q": Noul(instructions="?")}).nouls["q"].meta.answered is False
    finally:
        s.shutdown()
