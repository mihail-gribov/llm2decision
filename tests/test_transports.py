"""The two transports that need an optional dependency, with stand-ins for it: `completions` (a chat
template rendered locally)."""
import sys
import types

import pytest

from llm2decision import DecisionClient, UnreadableAnswer
from llm2decision import transports
from llm_server import P, FakeTokenizer, LLMServer

from test_client import QUESTIONS, ask_all


@pytest.fixture()
def server():
    s = LLMServer()
    yield s
    s.shutdown()


@pytest.fixture()
def home(tmp_path):
    return tmp_path


@pytest.fixture(autouse=True)
def stand_ins(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM2DECISION_HOME", str(tmp_path))
    monkeypatch.setattr(transports.time, "sleep", lambda s: None)
    tf = types.ModuleType("transformers")
    tf.AutoTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "transformers", tf)


def test_completions_binding_needs_no_transformers(server, monkeypatch):
    """A binding with its rendered `chat_template` works with `transformers` absent."""
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setenv("NEBIUS_API_KEY", "nb-test")
    with DecisionClient("qwen3.8-27b@nebius", base_url=server.url, key_env="NEBIUS_API_KEY",
                        calibrated=False) as c:
        assert type(c.transport).__name__ == "Completions" and c.transport.tok is None
        r = ask_all(c)
    assert server.requests[-1]["body"]["prompt"].startswith("<|im_start|>system\n")
    assert r["pay"].choices["pay"].choice == "paypal"
    assert r["pay"].choices["pay"].probabilities["paypal"] == pytest.approx(P)
    assert r["shipped"].nouls["shipped"].noul > 0.5 and r["decided"].tfus["decided"].tfu == "unknown"
    assert r["urgency"].scores["urgency"].probabilities[2] == pytest.approx(P)
    req = server.requests[-1]
    assert req["path"].endswith("/completions") and req["body"]["prompt"].endswith("Answer: [")
    assert req["body"]["logprobs"] == c.binding.top_k


def test_completions_template_from_transformers(server, home):
    """A binding without a rendered template falls back to the model's own, through `transformers`."""
    import json
    (home / "models.json").write_text(json.dumps({"mine@nebius": {"provider": "nebius", "model": "m",
        "transport": "completions", "tokenizer": "x/y", "template_kwargs": {"enable_thinking": False}}}))
    with DecisionClient("mine@nebius", base_url=server.url, api_key="k") as c:
        assert c.transport.tok.name == "x/y"
        r = c.system_one("pick: paypal", {"q": QUESTIONS["pay"]})
        assert c.transport.tok.kwargs == {"enable_thinking": False}
    assert r.choices["q"].choice == "paypal"
    assert server.requests[-1]["body"]["prompt"].startswith("<|system|>")


def test_completions_suffix_goes_before_the_answer(server, monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "nb-test")
    with DecisionClient("gpt-oss-120b@nebius", base_url=server.url, key_env="NEBIUS_API_KEY") as c:
        c.system_one("pick: cash", {"q": QUESTIONS["pay"]})
    prompt = server.requests[-1]["body"]["prompt"]
    assert prompt.endswith(c.binding.options["suffix"] + "Answer: [") and c.binding.options["suffix"]


def test_completions_numbers_beyond_26(server):
    crit = {f"o{i}": f"Option{i} item" for i in range(40)}
    with DecisionClient("m", base_url=server.url) as c:
        c.transport = transports.Completions(server.url, "m", {}, top_k=20, tokenizer="x")
        r = c.system_one("pick: option33", {"q": QUESTIONS["pay"].__class__(criteria=crit)})
    assert r.choices["q"].choice == "o33" and r.choices["q"].confidence == pytest.approx(P)


def test_completions_without_transformers(monkeypatch, server):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(transports.TransportError, match="template"):
        transports.Completions(server.url, "m", {}, tokenizer="x")
