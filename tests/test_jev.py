"""The `typesafe` provider: questions go to a server shaped like the Jev API (`POST /v1/systemone`), all
of a call in one request, and its answers come back as the package's answers."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import llm2decision as l2d
from llm2decision import transports
from llm2decision.check import run


class JevServer:
    """Answers every question with fixed probabilities and keeps the bodies it got."""

    def __init__(self, status: int = 200):
        self.requests, self.status = [], status
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, self.headers.get("Authorization"), body))
                if outer.status != 200:
                    self.send_response(outer.status)
                    self.end_headers()
                    self.wfile.write(b'{"detail": "refused"}')
                    return
                answers = {k: outer.answer(q) for k, q in body["questions"].items()}
                out = json.dumps({"model": body["model"], "answers": answers,
                                  "usage": {"input_tokens": 100, "output_tokens": 20}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/"

    @staticmethod
    def answer(q):
        if q["type"] == "noul":
            return {"type": "noul", "noul": 0.9}
        labels = [str(i) for i in range(len(q["criteria"]))] if q["type"] == "score" else list(q["criteria"])
        p = {k: (0.7 if i == len(labels) - 1 else 0.3 / (len(labels) - 1)) for i, k in enumerate(labels)}
        return {"type": q["type"], "probabilities": p}

    def shutdown(self):
        self.httpd.shutdown()


@pytest.fixture()
def server():
    s = JevServer()
    yield s
    s.shutdown()


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM2DECISION_HOME", str(tmp_path))
    monkeypatch.setattr(transports.time, "sleep", lambda s: None)


def client(server, **kw):
    return l2d.DecisionClient("jev-1.13.0@typesafe", base_url=server.url, api_key="test-key", **kw)


QUESTIONS = {
    "refund": l2d.Noul(instructions="Does the customer ask for money back?"),
    "team": l2d.Choice(instructions="Which team?", criteria={"billing": "Payments", "tech": "Bugs"}),
    "anger": l2d.Score(instructions="How angry?", criteria=["Calm", "Annoyed", "Furious"]),
    "card": l2d.Tfu(instructions="Paid by card?", criteria={"unknown": "the text does not say"}),
}


def test_one_request_per_call_in_the_jev_form(server):
    r = client(server).system_one({"note": "charged twice"}, QUESTIONS)
    assert len(server.requests) == 1
    path, auth, body = server.requests[0]
    assert path == "/v1/systemone" and auth == "Bearer test-key"
    assert body["model"] == "jev-1.13.0" and body["state"] == '{"note": "charged twice"}'
    q = body["questions"]
    assert q["refund"] == {"type": "noul", "instructions": "Does the customer ask for money back?"}
    assert q["anger"] == {"type": "score", "criteria": ["Calm", "Annoyed", "Furious"], "instructions": "How angry?"}
    # no true/false/unknown type at Jev: a choice over the three outcomes, the package's wording where left out
    assert q["card"]["type"] == "choice" and set(q["card"]["criteria"]) == {"true", "false", "unknown"}
    assert q["card"]["criteria"]["unknown"] == "the text does not say"
    assert r.usage.input_tokens == 100 and r.usage.output_tokens == 20


def test_answers_come_back_as_the_package_answers(server):
    r = client(server).system_one("charged twice", QUESTIONS)
    assert r.nouls["refund"].noul == pytest.approx(0.9)
    assert r.choices["team"].choice == "tech" and r.choices["team"].confidence == pytest.approx(0.7)
    assert r.scores["anger"].probabilities == pytest.approx({0: 0.15, 1: 0.15, 2: 0.7})
    assert r.scores["anger"].score == pytest.approx(0.15 + 1.4)
    assert r.tfus["card"].tfu == "unknown"
    meta = r.nouls["refund"].meta
    assert meta.answered and not meta.calibrated


def test_one_option_is_answered_without_a_request(server):
    r = client(server).system_one("x", {"only": l2d.Choice(criteria={"a": "The only one"})})
    assert r.choices["only"].choice == "a" and not server.requests


def test_a_bare_yes_no_question_fails_before_any_request(server):
    with pytest.raises(l2d.QuestionError):
        client(server).system_one("x", {"q": l2d.Noul()})
    assert not server.requests


def test_a_refused_request_is_a_transport_error():
    s = JevServer(status=400)
    try:
        with pytest.raises(l2d.TransportError):
            client(s).system_one("x", {"q": l2d.Noul(instructions="Is it so?")})
        assert len(s.requests) == 1                      # 400 is not worth a retry
    finally:
        s.shutdown()


def test_check_skips_the_token_points(server):
    rep = run(client(server))
    names = [p.name for p in rep.points]
    assert "digits" not in names and "mass" not in names
    assert next(p for p in rep.points if p.name == "logprobs").ok is None
