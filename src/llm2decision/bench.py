"""Measure a model on Decision Questions, the set behind the README's model table.

    llm2decision bench qwen3.8-27b@nebius                     # the published set, downloaded once
    llm2decision bench my-qwen@local --data test.jsonl --limit 200

Every question goes out on its own, as in the table. Answers are appended to `--out` as they come, so an
interrupted run continues where it stopped when started again. The report gives accuracy over the
answered questions (all and per type), the share left undecided, and the calibration error when the
model returns probabilities.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

from .errors import LLM2DecisionError, TransportError

# the revision the README's model table was measured on; the cache is per revision, so a newer one is fetched anew
DATA_REVISION = "04da4c4131f23424b5d41e219459fa1ce5a8fa69"
DATA_URL = f"https://huggingface.co/datasets/mihailgribov/decision-questions/resolve/{DATA_REVISION}/data/test.jsonl"
TYPES = ("noul", "tfu", "choice", "score")
NAMES = {"noul": "yes/no", "tfu": "yes/no/unknown", "choice": "choice", "score": "score"}


def cache_dir() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "llm2decision"


def load(path: str | os.PathLike | None = None) -> list[dict]:
    """The items of a Decision Questions file; without a path, the published set (downloaded once and cached)."""
    if path is None:
        path = cache_dir() / f"decision_questions_test_{DATA_REVISION[:12]}.jsonl"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading Decision Questions to {path}", file=sys.stderr, flush=True)
            tmp = path.with_suffix(".part")
            with urllib.request.urlopen(DATA_URL, timeout=120) as r, open(tmp, "wb") as f:
                f.write(r.read())
            tmp.replace(path)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def question(item: dict) -> dict:
    q = dict(item["question"])
    if isinstance(q.get("criteria"), str):
        q["criteria"] = json.loads(q["criteria"])
    return q


def state(item: dict):
    return json.loads(item["state"]) if item.get("state_format") == "json" else item["state"]


def ask(client, item: dict) -> dict:
    """One item, one call; the answer as probabilities over the item's answer labels."""
    a = client.system_one(state(item), {"q": question(item)}).answers["q"]
    if item["type"] == "noul":
        probs = {"yes": a.noul, "no": 1 - a.noul}
    else:
        probs = {str(k): v for k, v in a.probabilities.items()}
    return {"id": item["id"], "type": item["type"], "answer": str(item["answer"]),
            "probs": {k: round(v, 6) for k, v in probs.items()},
            "answered": a.meta.answered, "logprobs": a.meta.logprobs}


def _done(out: Path) -> dict[str, dict]:
    if not out.exists():
        return {}
    rows = {}
    for line in out.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def run(client, items: list[dict], out: str | os.PathLike, *, workers: int = 8, progress=print) -> dict[str, dict]:
    """Ask what `out` does not hold yet, appending each answer as it comes; returns every row of `out`
    for the given items. Ctrl-C stops after the requests in flight; a refused request stops the run."""
    out = Path(out)
    rows = _done(out)
    todo = [x for x in items if x["id"] not in rows]
    progress(f"{len(items) - len(todo)}/{len(items)} already in {out}, {len(todo)} to ask")
    stop = threading.Event()
    previous = None
    if threading.current_thread() is threading.main_thread():
        def on_signal(sig, _):
            stop.set()
            progress("stopping: finishing the requests in flight…")
        previous = signal.signal(signal.SIGINT, on_signal)
    errors, t0, n0 = 0, time.time(), len(rows)
    try:
        with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
            it, running = iter(todo), {}
            while True:
                while not stop.is_set() and len(running) < workers:
                    x = next(it, None)
                    if x is None:
                        break
                    running[ex.submit(ask, client, x)] = x
                if not running:
                    break
                for fut in wait(running, timeout=1, return_when=FIRST_COMPLETED)[0]:
                    x = running.pop(fut)
                    try:
                        r = fut.result()
                    except LLM2DecisionError as e:
                        errors += 1
                        progress(f"  {x['id']}: {str(e)[:160]}")
                        if isinstance(e, TransportError) and "HTTP 4" in str(e) and "429" not in str(e):
                            stop.set()                  # a refusal (key, balance, model): the rest would fail too
                        continue
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    rows[r["id"]] = r
                    done = len(rows) - n0
                    if done % 50 == 0:
                        rate = done / max(time.time() - t0, 1e-9)
                        progress(f"  {len(rows)}/{len(items)}  {rate:.1f}/s  errors {errors}")
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
    wanted = {x["id"] for x in items}
    rows = {k: v for k, v in rows.items() if k in wanted}
    progress(f"{len(rows)}/{len(items)} answered and saved, errors {errors}")
    if len(rows) < len(items):
        progress("to continue: run the same command again")
    return rows


def _right(r: dict) -> bool:
    return max(r["probs"], key=r["probs"].get) == r["answer"]


def _ece(rows: list[dict], bins: int) -> float:
    e = 0.0
    for b in range(bins):
        inside = [r for r in rows if b / bins < max(r["probs"].values()) <= (b + 1) / bins]
        if inside:
            conf = sum(max(r["probs"].values()) for r in inside) / len(inside)
            acc = sum(map(_right, inside)) / len(inside)
            e += len(inside) / len(rows) * abs(acc - conf)
    return e


def ece(rows: list[dict], bins: int = 10) -> float:
    """Expected calibration error of the top answer's probability over the answered rows: per question
    type, then weighted by the type's share, so that easy types do not hide a badly calibrated one."""
    rows = [r for r in rows if r["answered"]]
    by = [[r for r in rows if r["type"] == t] for t in TYPES]
    return sum(len(g) * _ece(g, bins) for g in by if g) / len(rows) if rows else 0.0


@dataclass
class Report:
    n: int
    accuracy: float | None
    by_type: dict[str, float | None] = field(default_factory=dict)
    undecided: float = 0.0
    calibration_error: float | None = None       # None: the model answers in text

    def __str__(self) -> str:
        f = lambda v: "-" if v is None else f"{v:.3f}"
        lines = [f"questions        {self.n}", f"accuracy         {f(self.accuracy)}  (over answered questions)"]
        lines += [f"  {NAMES[t]:<15}{f(v)}" for t, v in self.by_type.items()]
        lines.append(f"undecided        {100 * self.undecided:.1f}%")
        lines.append("calibration err  " + ("no probabilities (text answers)" if self.calibration_error is None
                                            else f(self.calibration_error)))
        return "\n".join(lines)


def report(rows) -> Report:
    rows = list(rows.values() if isinstance(rows, dict) else rows)
    answered = [r for r in rows if r["answered"]]
    acc = lambda rs: sum(map(_right, rs)) / len(rs) if rs else None
    by = {t: acc([r for r in answered if r["type"] == t]) for t in TYPES if any(r["type"] == t for r in rows)}
    probs = answered and all(r["logprobs"] for r in answered)
    return Report(n=len(rows), accuracy=acc(answered), by_type=by,
                  undecided=(1 - len(answered) / len(rows)) if rows else 0.0,
                  calibration_error=ece(rows) if probs else None)
