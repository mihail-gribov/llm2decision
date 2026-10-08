"""What the caller sees when the provider or the network goes wrong: one error type, the question
named where one is at fault, nothing retried that waiting will not fix, no key in any message."""
import pytest

from llm2decision import DecisionClient, LLM2DecisionError, TransportError
from llm2decision import transports
from llm_server import LLMServer

from test_client import QUESTIONS


@pytest.fixture()
def server():
    s = LLMServer()
    yield s
    s.shutdown()


@pytest.fixture(autouse=True)
def quick(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM2DECISION_HOME", str(tmp_path))
    monkeypatch.setattr(transports.time, "sleep", lambda s: None)


@pytest.mark.parametrize("body", [{}, {"choices": []}, {"choices": [{}]}, {"choices": [{"message": None}]}])
@pytest.mark.parametrize("how", [{}, {"provider": "openrouter", "api_key": "sk-or-v1-x"},
                                 {"provider": "mistral", "api_key": "m" * 32}], ids=["continue", "ask", "text"])
def test_malformed_response_is_a_transport_error(server, body, how):
    server.broken = body
    with DecisionClient("m", base_url=server.url, **how) as c:
        with pytest.raises(TransportError, match="'q'.*(unexpected shape|no log-probabilities)"):
            c.system_one("pick: cash", {"q": QUESTIONS["pay"]})


@pytest.mark.parametrize("code", [400, 401, 403, 404, 402, 422])
def test_client_errors_fail_at_once(server, code):
    server.fail = [code]
    with DecisionClient("m", base_url=server.url, api_key="secret-key-123") as c:
        with pytest.raises(TransportError) as e:
            c.system_one("pick: cash", {"q": QUESTIONS["pay"]})
    assert str(code) in str(e.value) and "secret-key-123" not in str(e.value)
    assert len(server.requests) == 1


def test_gives_up_after_the_retries(server):
    server.fail = [503] * 10
    with DecisionClient("m", base_url=server.url, retries=2) as c:
        with pytest.raises(TransportError, match="gave up after 3 attempts"):
            c.system_one("pick: cash", {"q": QUESTIONS["pay"]})
    assert len(server.requests) == 3


def test_connection_refused_is_retried_then_named():
    with DecisionClient("m", base_url="http://127.0.0.1:9/v1", retries=1) as c:
        with pytest.raises(TransportError, match="Connection"):
            c.system_one("pick: cash", {"q": QUESTIONS["pay"]})


def test_every_error_is_one_type(server):
    server.fail = [400]
    with DecisionClient("m", base_url=server.url) as c:
        with pytest.raises(LLM2DecisionError):
            c.system_one("pick: cash", {"q": QUESTIONS["pay"]})
        with pytest.raises(LLM2DecisionError):
            c.system_one("x", {"q": {"type": "essay"}})
    with pytest.raises(LLM2DecisionError):
        DecisionClient("nope@nebius")
    with pytest.raises(LLM2DecisionError):
        DecisionClient("m", provider="nebius", key_env="NO_SUCH_VAR_HERE")


def test_one_failing_question_fails_the_call(server):
    server.fail = [400]
    with DecisionClient("m", base_url=server.url, workers=1) as c:
        with pytest.raises(TransportError):
            c.system_one("pick: cash", {"a": QUESTIONS["pay"], "b": QUESTIONS["shipped"]})


@pytest.mark.parametrize("header,low,high", [("7", 7, 7 * 1.25), (None, 1, 1.25), ("600", 120, 150)])
def test_retry_after_is_honoured(server, monkeypatch, header, low, high):
    slept = []
    monkeypatch.setattr(transports.time, "sleep", slept.append)
    server.fail, server.retry_after = [429], header
    with DecisionClient("m", base_url=server.url) as c:
        c.system_one("pick: cash", {"q": QUESTIONS["pay"]})
    assert len(slept) == 1 and low <= slept[0] <= high


def test_retry_after_as_a_date():
    from email.utils import format_datetime
    import datetime as dt
    when = format_datetime(dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=30), usegmt=True)
    assert 25 <= transports._retry_after(when) <= 31
    assert transports._retry_after("soon") is None and transports._retry_after(None) is None
