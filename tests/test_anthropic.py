"""The Anthropic Messages API over plain HTTP: the prefill for models that take it, `Answer: [..]` for
those that do not, no temperature where it is refused, thinking switched off by the binding."""
import json

import pytest

from llm2decision import DecisionClient, UnreadableAnswer
from llm2decision import transports
from llm_server import LLMServer

from test_client import QUESTIONS, ask_all


@pytest.fixture()
def server():
    s = LLMServer()
    yield s
    s.shutdown()


@pytest.fixture(autouse=True)
def quick(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM2DECISION_HOME", str(tmp_path))
    monkeypatch.setattr(transports.time, "sleep", lambda s: None)
    return tmp_path


def client(server, name, **kw):
    return DecisionClient(name, base_url=server.root, api_key="sk-ant-test", **kw)


def last(server):
    return server.requests[-1]


def test_prefill_reads_text(server):
    with client(server, "claude-haiku-4.5@anthropic") as c:
        r = ask_all(c)
    assert r["pay"].choices["pay"].choice == "paypal" and r["pay"].choices["pay"].meta.logprobs is False
    assert r["shipped"].nouls["shipped"].noul == 1.0 and r["refund"].nouls["refund"].noul == 0.0
    assert r["decided"].tfus["decided"].tfu == "unknown" and r["urgency"].scores["urgency"].score == 2
    req = last(server)
    assert req["path"] == "/v1/messages"
    assert req["body"]["messages"][-1] == {"role": "assistant", "content": "Answer: ["}
    assert req["body"]["temperature"] == 0
    h = req["headers"]
    assert h["x-api-key"] == "sk-ant-test" and h["anthropic-version"] and "authorization" not in h


def test_per_call_options(server):
    with client(server, "claude-haiku-4.5@anthropic") as c:
        c.system_one("pick: cash", {"q": QUESTIONS["pay"]}, extra_headers={"X-T": "1"},
                     extra_body={"metadata": {"user_id": "u"}})
    req = last(server)
    assert req["headers"]["x-t"] == "1" and req["body"]["metadata"] == {"user_id": "u"}


def test_prose_is_unanswered(server):
    with client(server, "claude-haiku-4.5@anthropic") as c:
        a = c.system_one("prose", {"q": QUESTIONS["shipped"]}).nouls["q"]
        assert a.meta.answered is False and a.noul == pytest.approx(0.5)
    with client(server, "claude-haiku-4.5@anthropic", strict=True) as c:
        with pytest.raises(UnreadableAnswer, match="cannot tell"):
            c.system_one("prose", {"q": QUESTIONS["pay"]})


@pytest.mark.parametrize("name", ["claude-sonnet-5.5@anthropic", "claude-haiku-5.5@anthropic"])
def test_newer_models_are_asked_without_prefill(server, name):
    with client(server, name) as c:
        r = ask_all(c)
    assert r["pay"].choices["pay"].choice == "paypal" and r["pay"].choices["pay"].meta.answered
    assert r["shipped"].nouls["shipped"].noul == 1.0 and r["refund"].nouls["refund"].noul == 0.0
    assert r["decided"].tfus["decided"].tfu == "unknown" and r["urgency"].scores["urgency"].score == 2
    body = last(server)["body"]
    assert body["messages"][-1]["role"] == "user" and "Answer: [<your answer>]" in body["messages"][-1]["content"]
    assert "temperature" not in body and body["thinking"]["type"] in ("disabled", "between_tools")
    assert body["output_config"] == {"effort": "low"}


def test_thinking_left_on_is_reported_and_fails_check(server, quick):
    (quick / "models.json").write_text(json.dumps({"claude-sonnet-5.5@anthropic": {"extra": None}}))
    with client(server, "claude-sonnet-5.5@anthropic") as c:
        a = c.system_one("pick: cash", {"q": QUESTIONS["pay"]}).choices["q"]
        assert a.meta.answered is False and a.meta.reasoning_tokens > 0
        rep = c.check()
    assert not rep.ok and any(p.name == "reasoning" and p.ok is False for p in rep.points)


def test_overloaded_is_retried(server):
    server.fail = [529, 529]
    with client(server, "claude-haiku-4.5@anthropic") as c:
        assert c.system_one("pick: cash", {"q": QUESTIONS["pay"]}).choices["q"].choice == "cash"
    assert len(server.requests) == 3


def test_refusal_is_named(server):
    with client(server, "claude-sonnet-5.5@anthropic") as c:
        c.transport.extra.pop("thinking")
        c.transport.temperature = True
        with pytest.raises(transports.TransportError, match="400"):
            c.system_one("pick: cash", {"q": QUESTIONS["pay"]})
