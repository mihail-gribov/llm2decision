"""The wording: one prompt per question, the answer opened with `[`.

Every answer is asked for in square brackets, so the first token after `[` is the answer. A yes/no question names its two
outcomes and words the third ("Unsure") itself unless a `Tfu` describes it; a choice or a score lists
its options under marks (`marks.py`), so the caller's labels reach the model only as the description
of an option that has none.
"""
from __future__ import annotations

from dataclasses import dataclass

from .marks import Marks, assign
from .types import Choice, Noul, Score, Tfu, as_text

YESNO_SYSTEM = ("You are a decision function inside a program. You judge the state you are given against one "
                "yes/no question and answer with a single word. You never explain.")
YESNO_FORMAT = "Answer format: one word in square brackets, either [True], [False] or [Unsure]."
YESNO_MEANS = "True means: {true}\nFalse means: {false}\nUnsure means: {unsure}."
YESNO_UNSURE = ("the material leaves the question unclear, unanswered or self-contradictory, or the task itself "
                "is ill-posed — there is nothing here to judge")
YESNO_DEFAULT = {"true": "the answer to the question is yes", "false": "the answer to the question is no"}
MARKS_SYSTEM = {"letters": "You answer with exactly one letter from the given list.",
                "numbers": "You answer with exactly one number from the given list.",
                "own": "You answer with exactly one level from the given list."}
TAIL = "Answer: ["


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str
    marks: Marks | None = None                   # None for a yes/no question
    tail: str = TAIL                             # what opens the answer


def yes_no(state, q: Noul | Tfu) -> Prompt:
    crit = dict(q.criteria or {})
    true = as_text(crit.get("true")) or YESNO_DEFAULT["true"]
    false = as_text(crit.get("false")) or YESNO_DEFAULT["false"]
    unsure = as_text(crit.get("unknown")).rstrip(".") or YESNO_UNSURE
    user = (YESNO_FORMAT + f"\n\n<state>\n{as_text(state)}\n</state>\n\nQuestion: {as_text(q.instructions)}\n"
            + YESNO_MEANS.format(true=true, false=false, unsure=unsure))
    return Prompt(YESNO_SYSTEM, user)


def options(state, q: Choice | Score) -> Prompt:
    if isinstance(q, Choice):
        keys, descs = list(q.criteria), [as_text(v) or k for k, v in q.criteria.items()]
        m = assign(keys, "choice")
    else:
        keys = [str(i) for i in range(len(q.criteria))]
        descs = [as_text(v) for v in q.criteria]
        m = assign(keys, "score", as_text(q.instructions))
    user = (f"<state>\n{as_text(state)}\n</state>\n\n{as_text(q.instructions)}\n\n"
            + m.lines(descs) + f"\n\n{m.ask()}")
    return Prompt(MARKS_SYSTEM[m.scheme], user, m)


def build(state, q) -> Prompt:
    return yes_no(state, q) if isinstance(q, (Noul, Tfu)) else options(state, q)
