"""`bench`: a Decision Questions file read through the fake model, answers saved and resumed, and the report's
numbers on rows whose answers are known."""
import json

import pytest

from llm2decision import bench
from llm2decision import transports
from llm2decision.__main__ import main
from llm_server import LLMServer

ITEMS = [
    {"id": "d00001", "type": "noul", "state": "truth: yes", "state_format": "text", "answer": "yes",
     "question": {"type": "noul", "instructions": "Is it so?", "criteria": "null"}},
    {"id": "d00002", "type": "noul", "state": "truth: no", "state_format": "text", "answer": "yes",
     "question": {"type": "noul", "instructions": "Is it so?", "criteria": "null"}},
    {"id": "d00003", "type": "choice", "state": json.dumps({"note": "pick: money"}), "state_format": "json",
     "answer": "money", "question": {"type": "choice", "instructions": "Which team?",
                                     "criteria": json.dumps({"money": "Payments and money", "post": "Delivery by post"})}},
    {"id": "d00004", "type": "tfu", "state": "truth: unsure", "state_format": "text", "answer": "unknown",
     "question": {"type": "tfu", "instructions": "Is it so?", "criteria": "null"}},
    {"id": "d00005", "type": "score", "state": "pick: serious", "state_format": "text", "answer": 2,
     "question": {"type": "score", "instructions": "How bad is it?",
                  "criteria": json.dumps(["fine", "minor", "serious"])}},
]


@pytest.fixture()
def server():
    s = LLMServer()
    yield s
    s.shutdown()


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM2DECISION_HOME", str(tmp_path))
    monkeypatch.setattr(transports.time, "sleep", lambda s: None)
    return tmp_path


@pytest.fixture()
def data(tmp_path):
    p = tmp_path / "set.jsonl"
    p.write_text("".join(json.dumps(x) + "\n" for x in ITEMS))
    return p


def test_question_and_state_come_back_as_the_models_were_given_them():
    assert bench.question(ITEMS[2])["criteria"] == {"money": "Payments and money", "post": "Delivery by post"}
    assert bench.question(ITEMS[0])["criteria"] is None
    assert bench.state(ITEMS[2]) == {"note": "pick: money"}
    assert bench.state(ITEMS[0]) == "truth: yes"


def test_cli_measures_saves_and_resumes(server, data, tmp_path, capsys):
    out = tmp_path / "answers.jsonl"
    args = ["bench", "tiny-model", "--base-url", server.url, "--data", str(data), "--out", str(out)]
    assert main(args) == 0
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert sorted(r["id"] for r in rows) == ["d00001", "d00002", "d00003", "d00004", "d00005"]
    for r in rows:                                       # what benchmark/irt.py reads
        assert {"id", "type", "answer", "probs", "answered", "logprobs"} <= r.keys()
        assert r["answer"] in r["probs"]
    printed = capsys.readouterr().out
    assert "questions        5" in printed and "accuracy" in printed
    asked = len(server.requests)
    assert main(args) == 0                               # everything on disk: nothing asked again
    assert len(server.requests) == asked
    assert len(out.read_text().splitlines()) == 5


def test_published_set_is_downloaded_once_per_revision(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    urls = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return "".join(json.dumps(x) + "\n" for x in ITEMS).encode()

    def urlopen(url, timeout):
        urls.append(url)
        return Reply()

    monkeypatch.setattr(bench.urllib.request, "urlopen", urlopen)
    assert [x["id"] for x in bench.load()] == [x["id"] for x in ITEMS]
    assert [x["id"] for x in bench.load()] == [x["id"] for x in ITEMS]
    assert urls == [bench.DATA_URL] and bench.DATA_REVISION in bench.DATA_URL
    assert [f.name for f in (tmp_path / "llm2decision").iterdir()] == \
        [f"decision_questions_test_{bench.DATA_REVISION[:12]}.jsonl"]


def test_limit_takes_the_first_questions(server, data, tmp_path):
    out = tmp_path / "answers.jsonl"
    assert main(["bench", "tiny-model", "--base-url", server.url, "--data", str(data), "--out", str(out),
                 "--limit", "2"]) == 0
    assert sorted(json.loads(x)["id"] for x in out.read_text().splitlines()) == ["d00001", "d00002"]


def row(i, t, answer, probs, answered=True, logprobs=True):
    return {"id": str(i), "type": t, "answer": answer, "probs": probs, "answered": answered, "logprobs": logprobs}


def test_report_counts_accuracy_over_answered_and_undecided_apart():
    rows = [row(1, "noul", "yes", {"yes": 0.9, "no": 0.1}),
            row(2, "noul", "yes", {"yes": 0.2, "no": 0.8}),
            row(3, "choice", "a", {"a": 0.7, "b": 0.3}),
            row(4, "choice", "a", {"a": 0.5, "b": 0.5}, answered=False)]
    rep = bench.report(rows)
    assert rep.n == 4
    assert rep.accuracy == pytest.approx(2 / 3)
    assert rep.by_type == {"noul": 0.5, "choice": 1.0}
    assert rep.undecided == pytest.approx(0.25)


def test_calibration_error_per_type_weighted():
    # yes/no: one row at 0.9 that is right → |1 - 0.9| = 0.1; choice: one at 0.6 that is wrong → 0.6
    rows = [row(1, "noul", "yes", {"yes": 0.9, "no": 0.1}), row(2, "choice", "a", {"a": 0.4, "b": 0.6})]
    assert bench.ece(rows) == pytest.approx((0.1 + 0.6) / 2)


def test_text_answers_have_no_calibration_error():
    rows = [row(1, "noul", "yes", {"yes": 1.0, "no": 0.0}, logprobs=False)]
    rep = bench.report(rows)
    assert rep.calibration_error is None and "no probabilities" in str(rep)
