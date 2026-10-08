"""`check()` against the fake model: a model that answers right passes; each way of going wrong is
named by its own point."""
import json

import pytest

from llm2decision import DecisionClient
from llm2decision import transports
from llm2decision.__main__ import main
from llm2decision.check import EASY
from llm_server import LLMServer

ORACLE = {"paid in full": "truth: yes", "cancelled the day before": "truth: no",
          "left the warehouse": "truth: unsure", "charged twice": "pick: money", "one kiwi": "pick: kiwi",
          "UNACCEPTABLE": "pick: furious"}


@pytest.fixture()
def server():
    s = LLMServer()
    s.oracle = dict(ORACLE)
    yield s
    s.shutdown()


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM2DECISION_HOME", str(tmp_path))
    monkeypatch.setattr(transports.time, "sleep", lambda s: None)
    return tmp_path


def points(rep):
    return {p.name: p.ok for p in rep.points}


def test_oracle_covers_the_easy_set():
    assert len(ORACLE) == len(EASY) and all(any(k in st for k in ORACLE) for st, *_ in EASY)


def test_a_good_model_passes_and_is_marked_checked(server):
    with DecisionClient("tiny-model", base_url=server.url) as c:
        rep = c.check()
        assert rep.ok, str(rep)
        assert c.binding.checked and c.binding.digits == "single"
        assert c.system_one("truth: yes", {"q": EASY[0][2]}).nouls["q"].meta.checked
    p = points(rep)
    assert p["prefix"] and p["logprobs"] and p["mass"] and p["digits"] and p["easy"] and p["reasoning"]
    assert "passed" in str(rep)


@pytest.mark.parametrize("how", [True, "bracket"], ids=["Answer", "own-bracket"])
def test_prefix_ignored_is_named(server, how):
    server.ignore_prefix = how
    with DecisionClient("tiny-model", base_url=server.url) as c:
        rep = c.check()
    assert not rep.ok and points(rep)["prefix"] is False and "completions" in str(rep)
    assert c.binding.checked is None


def test_wrong_answers_fail_easy(server):
    server.oracle = {}
    server.oracle["one kiwi"] = "pick: apple"
    with DecisionClient("tiny-model", base_url=server.url) as c:
        rep = c.check()
    assert points(rep)["easy"] is False and "fruit" in str(rep)


def test_reasoning_reported_fails(server):
    server.reasoning = 12
    with DecisionClient("m", provider="openrouter", base_url=server.url, api_key="sk-or-v1-t") as c:
        rep = c.check()
    assert points(rep)["reasoning"] is False and "prefix" not in points(rep)


def test_ask_mode_passes(server):
    with DecisionClient("m", provider="openrouter", base_url=server.url, api_key="sk-or-v1-t") as c:
        rep = c.check()
    assert rep.ok, str(rep)


def test_text_only_passes_without_logprob_points(server):
    with DecisionClient("m", provider="mistral", base_url=server.url, api_key="m" * 32) as c:
        rep = c.check()
    assert rep.ok, str(rep)
    assert points(rep)["logprobs"] is None and points(rep)["digits"] is None


def test_grouped_digits_against_a_single_binding(server, home):
    (home / "models.json").write_text(json.dumps({"g@local": {"provider": "local", "model": "g", "digits": "single"}}))
    server.grouped = True
    with DecisionClient("g@local", base_url=server.url) as c:
        rep = c.check()
    assert points(rep)["digits"] is False and rep.digits == "grouped"


def test_unreachable_is_one_point():
    with DecisionClient("tiny-model", base_url="http://127.0.0.1:9/v1", retries=0) as c:
        rep = c.check()
    assert not rep.ok and [p.name for p in rep.points] == ["reach"]


def test_save_writes_the_user_binding(server, home):
    with DecisionClient("tiny-model", base_url=server.url) as c:
        assert c.check(save=True).ok
    saved = json.loads((home / "models.json").read_text())["tiny-model@local"]
    assert saved["checked"] and saved["digits"] == "single" and saved["model"] == "tiny-model"
    with DecisionClient("tiny-model@local", base_url=server.url) as c:
        assert c.binding.checked == saved["checked"]


def test_cli(server, capsys):
    assert main(["models", "--provider", "nebius"]) == 0
    assert "@nebius" in capsys.readouterr().out
    assert main(["check", "tiny-model", "--base-url", server.url]) == 0
    assert "passed" in capsys.readouterr().out
    server.oracle = {}
    assert main(["check", "tiny-model", "--base-url", server.url]) == 1
    assert main(["check", "nope@nebius"]) == 2


def test_one_certain_candidate_is_not_a_failure(server, monkeypatch):
    from llm2decision.check import _first_token, CheckReport
    from llm2decision.transports import Step
    with DecisionClient("tiny-model", base_url=server.url) as c:
        monkeypatch.setattr(c.transport, "step", lambda prompt, written: Step([("True", -0.0001)], 5))
        rep = CheckReport("x")
        _first_token(c, rep)
        assert points(rep)["logprobs"] is None and "near 1 and 0" in str(rep)
        monkeypatch.setattr(c.transport, "step", lambda prompt, written: Step([("True", -0.7)], 5))
        rep = CheckReport("x")
        _first_token(c, rep)
        assert points(rep)["logprobs"] is False
