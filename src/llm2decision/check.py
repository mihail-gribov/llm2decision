"""`check()`: does this model, through this provider, answer the way the package reads it?

About a dozen requests on six easy questions, then a verdict per point:

  reach        the provider answers with this key
  format       every answer is one of the options (nothing written past them, no prose)
  easy         the six easy questions are answered right — a model that fails them is not read right
  prefix       (prefilled transports) the answer continues the opened `[`, rather than starting anew
  logprobs     (models read by log-probabilities) candidates arrive, more than one per position
  mass         the options take most of the probability of the first answer token
  reasoning    no reasoning tokens are reported
  digits       how the tokenizer writes numbers 100–999, against the binding's `digits`
  noise        the same request twice gives the same probabilities (informational)

A passed check sets `binding.checked` to today for this client; `save=True` also writes the date and
the digits found into the user's `models.json`, so later clients of the binding carry it.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass, field, replace

from .config import bindings, home, merge
from .marks import normalise
from .prompt import build
from .types import Choice, Noul, Score, Tfu

FRUITS = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew", "jackfruit", "kiwi",
          "lemon", "lime", "mango", "nectarine", "orange", "papaya", "peach", "pear", "pineapple", "plum",
          "pomegranate", "quince", "raspberry", "strawberry", "tangerine", "watermelon", "apricot", "blueberry",
          "coconut", "guava"]

EASY = [  # (state, name, question, right answer)
    ("The invoice was paid in full on 3 March.", "paid",
     Noul(instructions="Was the invoice paid?"), True),
    ("The meeting was cancelled the day before and nobody came.", "met",
     Noul(instructions="Did the meeting take place?"), False),
    ("The package left the warehouse on Monday. Nothing else is known about it.", "signed",
     Tfu(instructions="Did the recipient sign for the package?"), "unknown"),
    ("Hello, I was charged twice for my order, please return the extra payment to my card.", "topic",
     Choice(instructions="What does the customer want?",
            criteria={"refund": "Money back", "status": "Where the delivery is", "password": "A password reset",
                      "staff": "To complain about staff"}), "refund"),
    ("Please add one kiwi to my basket, nothing else.", "fruit",
     Choice(instructions="Which fruit does the customer order?", criteria={f: f.capitalize() for f in FRUITS}),
     "kiwi"),
    ("THIS IS UNACCEPTABLE!!! Third time the order is lost. I am furious. Cancel everything NOW.", "anger",
     Score(instructions="How angry is the customer?", criteria=["Calm", "Annoyed", "Furious"]), 2),
]


@dataclass
class Point:
    name: str
    ok: bool | None                      # None: does not apply here, or informational
    detail: str


@dataclass
class CheckReport:
    binding: str
    points: list[Point] = field(default_factory=list)
    digits: str | None = None            # what the probe found: "single", "grouped" or None

    @property
    def ok(self) -> bool:
        return all(p.ok is not False for p in self.points)

    def __str__(self) -> str:
        mark = {True: "ok  ", False: "FAIL", None: "--  "}
        lines = [f"{self.binding}: {'passed' if self.ok else 'FAILED'}"]
        lines += [f"  {mark[p.ok]} {p.name:<10} {p.detail}" for p in self.points]
        return "\n".join(lines)


def run(client, save: bool = False) -> CheckReport:
    b, t = client.binding, client.transport
    rep = CheckReport(b.name)
    add = rep.points.append
    by_logprobs = t.logprobs and b.logprobs
    try:
        answers = {name: client.system_one(state, {name: q}).answers[name] for state, name, q, _ in EASY}
    except Exception as e:                                       # noqa: BLE001 - the report names it
        add(Point("reach", False, f"{type(e).__name__}: {str(e)[:200]}"))
        return rep
    add(Point("reach", True, f"{len(EASY)} questions answered"))

    unread = [n for n, a in answers.items() if not a.meta.answered]
    add(Point("format", not unread, "every answer is an option" if not unread else f"no option read: {unread}"))

    wrong = []
    for _, name, q, right in EASY:
        a = answers[name]
        got = (a.noul > 0.5) if isinstance(q, Noul) else a.tfu if isinstance(q, Tfu) else \
            a.choice if isinstance(q, Choice) else max(a.probabilities, key=a.probabilities.get)
        if got != right:
            wrong.append(f"{name}: {got!r}, not {right!r}")
    add(Point("easy", not wrong, "all right" if not wrong else "; ".join(wrong)))

    reasoning = sum(a.meta.reasoning_tokens or 0 for a in answers.values())
    add(Point("reasoning", reasoning == 0, "none reported" if reasoning == 0 else f"{reasoning} tokens reported"))

    if not by_logprobs:
        add(Point("logprobs", None, "read from text: probabilities are 1 and 0"))
    else:
        _first_token(client, rep)
    _digits(client, rep, by_logprobs)

    if rep.ok:
        today = dt.date.today().isoformat()
        client.binding = replace(b, checked=today, digits=b.digits or rep.digits)
        if save:
            fields = {"checked": today, **({"digits": rep.digits} if rep.digits and not b.digits else {})}
            if b.name not in bindings():                     # a model id at a provider: the binding is new
                fields = {"provider": b.provider, "model": b.model, **fields}
            _save(b.name, fields)
    return rep


def _first_token(client, rep: CheckReport) -> None:
    """Prefix, candidates, mass and noise, from the first answer token of the easy yes/no question."""
    from .client import side_of
    b, t = client.binding, client.transport
    state, _, q, _ = EASY[0]
    prompt = build(state, q)
    try:
        s1 = t.step(prompt, "")
        t.forget(prompt)
        s2 = t.step(prompt, "")
        t.forget(prompt)
    except Exception as e:                                       # noqa: BLE001
        rep.points.append(Point("logprobs", False, f"{type(e).__name__}: {str(e)[:200]}"))
        return
    c = s1.candidates
    if not c:
        rep.points.append(Point("logprobs", False, "no candidates came back"
                                + (f"; the model wrote {s1.text[:60]!r}" if s1.text else "")))
        return
    if len(c) == 1 and c[0][1] > math.log(0.99):
        # one candidate at p≈1: the provider drops the rest (OpenAI does), not a fault — but confidence is near 1/0
        rep.points.append(Point("logprobs", None, "1 candidate at p≈1: probabilities come out near 1 and 0"))
    else:
        rep.points.append(Point("logprobs", len(c) > 1, f"{len(c)} candidates (top-k asked {b.top_k})"))
    top = c[0][0].replace("Ġ", " ").replace("▁", " ").strip()
    if b.transport in ("chat_continue", "completions"):
        # after the opened `[` the answer itself comes; `Answer` or a `[` of its own means the model started anew
        dropped = top.lower().startswith("answer") or top.startswith("[")
        rep.points.append(Point("prefix", not dropped, "the answer continues `[`" if not dropped else
                                f"the model starts anew with {c[0][0]!r}: the server ignores the opened answer "
                                "(try transport `completions`)"))
    mass = sum(math.exp(lp) for tok, lp in c if side_of(tok, b.forms))
    verdict = True if mass >= 0.9 else None if mass >= 0.5 else False      # between: worth a look, not a fault
    rep.points.append(Point("mass", verdict, f"{mass:.3f} of the first token on True/False/Unsure"))
    d2 = dict(s2.candidates)
    diff = max((abs(math.exp(lp) - math.exp(d2.get(tok, -math.inf))) for tok, lp in c[:5]), default=0.0)
    rep.points.append(Point("noise", None, f"same request twice: top-5 probabilities differ by up to {diff:.3f}"))


def _digits(client, rep: CheckReport, by_logprobs: bool) -> None:
    """How the model writes a number from 100 on: one token per digit, or more digits in a token."""
    b, t = client.binding, client.transport
    if not by_logprobs:
        rep.points.append(Point("digits", None, "read from text: the tokenizer does not matter"))
        return
    state, _, q, _ = EASY[4]
    prompt = build(state, q)
    try:
        s = t.step(prompt, "")
        t.forget(prompt)
    except Exception as e:                                       # noqa: BLE001
        rep.points.append(Point("digits", False, f"{type(e).__name__}: {str(e)[:200]}"))
        return
    nums = [normalise(tok).strip() for tok, _ in s.candidates]
    nums = [n for n in nums if n.isdigit()]
    if not nums:
        rep.points.append(Point("digits", None, "no number among the candidates: kind not found"))
        return
    found = "grouped" if any(len(n) > 1 for n in nums) else "single"
    rep.digits = found
    if b.digits and b.digits != found:
        rep.points.append(Point("digits", False, f"the binding says {b.digits!r}, the model writes {found!r} "
                                                 f"(candidates {nums[:5]})"))
    else:
        rep.points.append(Point("digits", True, f"{found}" + ("" if b.digits else " (not in the binding)")))


def _save(name: str, fields: dict) -> None:
    path = home() / "models.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    mine = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    mine = merge(mine, {name: fields})
    path.write_text(json.dumps(mine, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
