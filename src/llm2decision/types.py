"""Questions and answers, in the shapes of the TypeSafe (Jev) Python SDK.

Code written against `typesafe_sdk` moves over by changing the import and the client: the question
classes take the same fields, a plain dict with a `type` key is accepted wherever a question is, and
the response groups its answers the same way (`nouls`, `choices`, `scores`). `Tfu` — yes, no, or
the material does not decide — is ours; Jev has no such type.

Every answer also carries `meta`: how it was read (log-probabilities or a text answer), whether a
calibration was applied, which options got only a bound, how many requests it took.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence, Union

JSONContent = Union[str, Mapping[str, Any], Sequence[Any]]


def as_text(value: Any) -> str:
    """Text for the prompt: a string as is, an object or an array as JSON, nothing as ""."""
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


# -- questions ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Noul:
    """A yes/no question; `criteria` may describe either outcome (`{"true": …, "false": …}`)."""
    instructions: JSONContent | None = None
    criteria: Mapping[str, JSONContent | None] | None = None
    type: str = "noul"


@dataclass(frozen=True)
class Tfu:
    """Yes, no, or the material does not decide. `criteria` may describe all three outcomes
    (`{"true": …, "false": …, "unknown": …}`); an undescribed third outcome gets the package's wording."""
    instructions: JSONContent | None = None
    criteria: Mapping[str, JSONContent | None] | None = None
    type: str = "tfu"


@dataclass(frozen=True)
class Choice:
    """One of named alternatives: `criteria` maps each label to a description. The model sees the
    descriptions under marks, not the labels — except a label with no description (None), which is
    then shown as its own description."""
    criteria: Mapping[str, JSONContent | None]
    instructions: JSONContent | None = None
    type: str = "choice"


@dataclass(frozen=True)
class Score:
    """A level of an ordered rubric: `criteria` lists the descriptions, one per level from zero."""
    criteria: Sequence[JSONContent]
    instructions: JSONContent | None = None
    type: str = "score"


Question = Union[Noul, Tfu, Choice, Score, Mapping[str, Any]]
_KINDS = {"noul": Noul, "tfu": Tfu, "choice": Choice, "score": Score}


from .errors import QuestionError  # noqa: E402  re-exported


def coerce(key: str, q: Question) -> Noul | Tfu | Choice | Score:
    """A question object from an object or a dict, checked the way the SDK checks it."""
    if isinstance(q, (Noul, Tfu, Choice, Score)):
        obj = q
    elif isinstance(q, Mapping):
        kind = q.get("type")
        if kind not in _KINDS:
            raise QuestionError(key, f"type must be one of {sorted(_KINDS)}, got {kind!r}")
        extra = set(q) - {"type", "instructions", "criteria"}
        if extra:
            raise QuestionError(key, f"unknown fields {sorted(extra)}")
        args = {k: v for k, v in q.items() if k != "type"}
        try:
            obj = _KINDS[kind](**args)
        except TypeError as e:
            raise QuestionError(key, str(e)) from None
    else:
        raise QuestionError(key, "a question is a Noul, Tfu, Choice, Score or a dict with `type`")
    if isinstance(obj, (Noul, Tfu)):
        c = obj.criteria
        allowed = {"true", "false"} | ({"unknown"} if isinstance(obj, Tfu) else set())
        if c is not None and (not isinstance(c, Mapping) or set(c) - allowed):
            raise QuestionError(key, f"criteria of a {obj.type} question are {sorted(allowed)}")
    elif isinstance(obj, Choice):
        if not isinstance(obj.criteria, Mapping) or not obj.criteria:
            raise QuestionError(key, "a choice needs at least one labelled option in `criteria`")
        if any(not isinstance(k, str) or not k.strip() for k in obj.criteria):
            raise QuestionError(key, "choice labels must be non-empty strings")
    elif isinstance(obj, Score):
        if isinstance(obj.criteria, (str, bytes, Mapping)) or not obj.criteria:
            raise QuestionError(key, "a score needs a non-empty list of levels in `criteria`")
    return obj


# -- answers --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Meta:
    """How an answer was read.

    `answered`: False when the options take less than half of the answer token's probability — the
    model wrote something else (prose, a format of its own, `Unsure` to a yes/no question); the
    probabilities then mean little (even, when nothing was read); `strict=True` on the client raises
    instead. `logprobs`: read from the model's token probabilities (False: from the text
    it wrote, probabilities 1 and 0). `bounded`: options past the provider's top-k, given only an
    upper bound. `calibrated`: a fitted temperature was applied. `checked`: the binding passed
    `check()`. `reasoning_tokens`: what the provider reported (0 or None with reasoning off)."""
    answered: bool = True
    logprobs: bool = True
    calibrated: bool = False
    bounded: tuple[str, ...] = ()
    requests: int = 1
    checked: bool = False
    reasoning_tokens: int | None = None


@dataclass(frozen=True)
class NoulAnswer:
    noul: float                                    # P(yes), from 0 to 1
    meta: Meta = field(default_factory=Meta)
    type: str = "noul"


@dataclass(frozen=True)
class TfuAnswer:
    tfu: str                                       # "true", "false" or "unknown"
    probabilities: dict[str, float]
    confidence: float
    meta: Meta = field(default_factory=Meta)
    type: str = "tfu"


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str                                    # the label with the highest probability
    confidence: float
    probabilities: dict[str, float]
    meta: Meta = field(default_factory=Meta)
    type: str = "choice"


@dataclass(frozen=True)
class ScoreAnswer:
    score: float                                   # the expected level, may fall between levels
    confidence: float
    legend: dict[int, JSONContent]
    probabilities: dict[int, float]
    meta: Meta = field(default_factory=Meta)
    type: str = "score"


Answer = Union[NoulAnswer, TfuAnswer, ChoiceAnswer, ScoreAnswer]


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class SystemOneResponse:
    """Answers keyed by question name, grouped by type as the SDK groups them."""
    model: str
    usage: Usage
    answers: dict[str, Answer]

    def to_dict(self) -> dict[str, Any]:
        """Plain data for JSON: `{"model", "usage", "answers": {name: {..., "meta": {...}}}}`. Score
        levels become string keys, as JSON has no others."""
        def plain(a):
            d = asdict(a)
            if isinstance(a, ScoreAnswer):
                d["legend"] = {str(k): v for k, v in a.legend.items()}
                d["probabilities"] = {str(k): v for k, v in a.probabilities.items()}
            d["meta"]["bounded"] = list(a.meta.bounded)
            return d
        return {"model": self.model, "usage": asdict(self.usage),
                "answers": {k: plain(a) for k, a in self.answers.items()}}

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        return {k: a for k, a in self.answers.items() if isinstance(a, NoulAnswer)}

    @property
    def tfus(self) -> dict[str, TfuAnswer]:
        return {k: a for k, a in self.answers.items() if isinstance(a, TfuAnswer)}

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        return {k: a for k, a in self.answers.items() if isinstance(a, ChoiceAnswer)}

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        return {k: a for k, a in self.answers.items() if isinstance(a, ScoreAnswer)}
