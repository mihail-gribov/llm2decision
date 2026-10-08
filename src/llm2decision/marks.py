"""Marks: what the model sees in place of the caller's option names, and how its answer is read.

The caller's names never reach the prompt. Each option is shown as `[mark] description`, the answer
is opened with `[`, and the model's next tokens are read back into the caller's names.

  up to 26 options      letters A … Z
  27 … 900 options      three-digit numbers from 100 (100 … 999): every mark the same length
  a score whose question names its own levels ("0 = none, 1 = one …") keeps those levels as marks,
                        so the question and the marks agree

Reading a number depends on how the model's tokenizer cuts digits (`digits`):

  "single"   one digit a token (Qwen, Nemotron, Mistral): the digits all marks share are written by
             us, the rest is read a token at a time
  "grouped"  up to three digits a token (DeepSeek, MiniMax, Llama, gpt-oss): a mark is one token
  None       not known: nothing is written; a token every candidate mark starts with is stepped over

Several requests may be needed (`read` follows the likeliest branch); a mark's log-probability is
the sum along its path. When the provider returns only its top candidates, a mark never seen gets
an upper bound — the lowest log-probability the provider did return there — and is listed in
`bounded`.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FIRST_NUMBER = 100
MAX_OPTIONS = 900                       # 100 … 999
END = "]"

ASK = {
    ("choice", "letters"): "Answer with one letter in square brackets, for example [A].",
    ("score", "letters"): "Answer with the letter of the level that rates it, in square brackets, for example [A].",
    ("choice", "numbers"): "Answer with the number of one option, in square brackets.",
    ("score", "numbers"): "Answer with the number of the level that rates it, in square brackets.",
    ("score", "own"): "Answer with the level that rates it, in square brackets.",
}

# What a provider returns for one position: (token, log-probability) pairs in any order, possibly
# with the same token twice, and how many it returns at most (0 = no limit is known).
Candidates = list[tuple[str, float]]
NextTokens = Callable[[str], tuple[Candidates, int]]


from .errors import MarksError  # noqa: E402  re-exported


def referenced(keys: Iterable[str], instructions: str) -> bool:
    """Does the question name its own levels, as in "(0 = none, 1 = one …)"?"""
    return any(re.search(rf"(?<![\w.]){re.escape(str(k))}\s*=", instructions or "") for k in keys)


@dataclass(frozen=True)
class Marks:
    keys: tuple[str, ...]               # the caller's names, in the order given
    marks: tuple[str, ...]              # what the model sees, one per key
    scheme: str                         # "letters", "numbers" or "own"
    kind: str = "choice"

    def __post_init__(self):
        n = len(self.marks)
        if len(self.keys) != n:
            raise MarksError("one mark per option")
        if self.scheme == "letters" and n > len(LETTERS):
            raise MarksError(f"{n} options: letters mark at most {len(LETTERS)}")
        if self.scheme == "numbers" and n <= len(LETTERS):
            raise MarksError(f"{n} options: up to {len(LETTERS)} are marked with letters, never numbers")
        if self.scheme not in ("letters", "numbers", "own"):
            raise MarksError(f"unknown marks scheme {self.scheme!r}")

    def lines(self, descriptions: Iterable[str]) -> str:
        """The options as the model reads them, one `[mark] description` a line."""
        return "\n".join(f"[{m}] {d}" for m, d in zip(self.marks, descriptions))

    def ask(self) -> str:
        return ASK[(self.kind if self.kind == "score" else "choice", self.scheme)]

    def key_of(self, mark: str) -> str:
        return self.keys[self.marks.index(mark)]

    def prefill(self, digits: str | None) -> str:
        """What we write after `[` ourselves: the start every mark shares, when the tokenizer is
        known to cut there. Letters and own levels: nothing."""
        if self.scheme != "numbers" or digits != "single":
            return ""
        shared = self.marks[0]
        for m in self.marks[1:]:
            while not m.startswith(shared):
                shared = shared[:-1]
        return shared[: len(self.marks[0]) - 1]         # never the whole mark


def assign(keys: Iterable, kind: str = "choice", instructions: str = "") -> Marks:
    """Marks for the caller's option names (`keys`, in order): letters up to 26 options, numbers
    beyond — the count alone decides; a score whose question names its levels keeps them."""
    keys = tuple(str(k) for k in keys)
    if len(set(keys)) != len(keys):
        raise MarksError("option names must be distinct")
    n = len(keys)
    if n < 2:
        raise MarksError(f"{n} option(s): a {kind} needs at least two")
    if n > MAX_OPTIONS:
        raise MarksError(f"{n} options: at most {MAX_OPTIONS} can be marked (numbers {FIRST_NUMBER}…"
                         f"{FIRST_NUMBER + MAX_OPTIONS - 1})")
    if kind == "score" and referenced(keys, instructions) and all(k.strip() == k and k and END not in k for k in keys):
        return Marks(keys, keys, "own", kind)
    if n <= len(LETTERS):
        return Marks(keys, tuple(LETTERS[:n]), "letters", kind)
    return Marks(keys, tuple(str(FIRST_NUMBER + i) for i in range(n)), "numbers", kind)


# -- reading ------------------------------------------------------------------------------------

def normalise(token: str) -> str:
    """A candidate token as answer text: spaces dropped, an opening bracket dropped; a closing
    bracket stays, because it ends a mark (`1]` is the mark 1, not the start of 10)."""
    t = token.strip()
    while t.startswith("["):
        t = t[1:].lstrip()
    return t


def _merge(cands: Candidates) -> dict[str, float]:
    """Continuations and their log-probabilities. The same token twice counts once (a server
    quirk); different tokens that mean the same continuation (`D`, ` D`, `[D`) are different
    events and their probabilities add up."""
    best: dict[str, float] = {}
    for tok, lp in cands:
        best[tok] = max(lp, best.get(tok, -math.inf))
    out: dict[str, list[float]] = {}
    for tok, lp in best.items():
        out.setdefault(normalise(tok), []).append(lp)
    return {t: _lse(v) for t, v in out.items()}


def _lse(xs: list[float]) -> float:
    m = max(xs)
    return -math.inf if m == -math.inf else m + math.log(sum(math.exp(x - m) for x in xs))


@dataclass
class Reading:
    key: str | None                     # the caller's name of the likeliest option; None if no mark was read
    logprobs: dict[str, float]          # per key: exact, or an upper bound for keys in `bounded`; -inf: not seen
    bounded: set[str] = field(default_factory=set)
    written: str = ""                   # what followed `[` when the answer was read
    requests: int = 0
    mass: dict[str, float] = field(default_factory=dict)   # per key: probability as read (estimates for `bounded`)

    @property
    def probabilities(self) -> dict[str, float]:
        """Over the options, summing to one: the probability the model put on answers, shared
        out as read. A branch not followed is known as a whole and split evenly among its marks;
        a mark past the provider's top-k counts as nothing."""
        total = sum(self.mass.values())
        if total <= 0:
            return {k: 0.0 for k in self.logprobs}
        return {k: self.mass.get(k, 0.0) / total for k in self.logprobs}


def read(marks: Marks, next_tokens: NextTokens, digits: str | None = None, max_requests: int = 4) -> Reading:
    """Read the model's answer into the caller's names.

    `next_tokens(written)` asks the provider for the candidates of the next token after
    `[` + `written`. The likeliest continuation is followed: when it completes a mark the answer is
    read; when it is the start of several marks it is written and the next token is asked for.
    """
    written = marks.prefill(digits)
    chain = 0.0                          # log P(written | prompt), beyond what we wrote ourselves
    exact: dict[str, float] = {}
    bound: dict[str, float] = {}         # past the top-k: an upper bound, no mass
    left: dict[str, float] = {}          # branches not followed: their whole mass, known exactly
    requests = 0
    while requests < max_requests:
        cands, limit = next_tokens(written)
        requests += 1
        cont = _merge(cands)
        floor = min(cont.values()) if cont and limit and len(cands) >= limit else -math.inf
        open_marks = [m for m in marks.marks if m.startswith(written)]
        branches: dict[str, float] = {}
        for t, lp in cont.items():
            body, closed = (t[:-1], True) if t.endswith(END) else (t, False)
            if not body and closed and written in open_marks:            # `]` right after a whole mark
                exact[written] = max(exact.get(written, -math.inf), chain + lp)
                continue
            if not body:
                continue
            whole = written + body
            if whole in open_marks and (closed or not any(m != whole and m.startswith(whole) for m in open_marks)):
                exact[whole] = _lse([exact.get(whole, -math.inf), chain + lp])
            elif any(m.startswith(whole) and m != whole for m in open_marks):
                branches[whole] = _lse([branches.get(whole, -math.inf), chain + lp])
        best_mark = max(exact.items(), key=lambda kv: kv[1], default=(None, -math.inf))
        best_branch = max(branches.items(), key=lambda kv: kv[1], default=(None, -math.inf))
        for m in open_marks:                                              # what this position leaves unseen
            if m not in exact and not any(m.startswith(b) for b in branches):
                bound[m] = max(bound.get(m, -math.inf), chain + floor)
        follow = best_branch[0] is not None and best_branch[1] > best_mark[1]
        for b, lp in branches.items():
            if not (follow and b == best_branch[0]):
                left[b] = lp
        if not follow:
            break
        written, chain = best_branch[0], best_branch[1]
    logprobs, bounded, mass = {}, set(), {}
    for k, m in zip(marks.keys, marks.marks):
        if m in exact:
            logprobs[k], mass[k] = exact[m], math.exp(exact[m])
            continue
        b = next((b for b in left if m.startswith(b)), None)
        if b is not None:                                    # in a branch not followed
            share = sum(1 for x in marks.marks if x.startswith(b) and x not in exact)
            logprobs[k], mass[k] = left[b], math.exp(left[b]) / share
            bounded.add(k)
        elif bound.get(m, -math.inf) > -math.inf:            # past the top-k
            logprobs[k], mass[k] = bound[m], 0.0
            bounded.add(k)
        else:
            logprobs[k], mass[k] = -math.inf, 0.0
    key = max(exact, key=exact.get) if exact else None
    return Reading(marks.key_of(key) if key else None, logprobs, bounded, written, requests, mass)


def read_text(marks: Marks, text: str) -> str | None:
    """The caller's name for a mark written as text, for providers that give no log-probabilities:
    `[C]`, `C`, ` [c] ` — letters in either case."""
    m = re.match(r"\s*\[?\s*([^\]\s]+)\s*\]?", text or "")
    if not m:
        return None
    got = m.group(1)
    if marks.scheme == "letters":
        got = got.upper()
    return marks.key_of(got) if got in marks.marks else None
