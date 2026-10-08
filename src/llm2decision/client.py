"""`DecisionClient`: a hosted LLM answering Jev-shaped questions, used like the TypeSafe SDK's client.

    from llm2decision import DecisionClient, Noul, Choice, Score

    with DecisionClient("qwen3.8-27b@nebius") as client:            # a known binding
        r = client.system_one(state, {"team": Choice(criteria={...}), "billing": Noul(instructions="…")})
        r.choices["team"].choice, r.nouls["billing"].noul

    DecisionClient(model="Qwen/Qwen3-8B", base_url="http://localhost:8000/v1")   # any OpenAI-compatible server
    DecisionClient(model="google/gemini-2.5-flash", provider="openrouter")      # any model of a known provider

One request a question (more when an answer spans tokens). A yes/no answer is the probability of
its side summed over the side's forms; a choice or a score is read through marks (`marks.py`).
Version 1 asks with reasoning switched off; a model that cannot switch it off is refused by its
provider's settings rather than read after a hidden chain of thought.
"""
from __future__ import annotations

import asyncio
import contextvars
import math
import threading
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from typing import Mapping

from . import auth
from .config import Binding, providers, resolve
from .config import forms as config_forms
from .errors import ConfigError, LLM2DecisionError, MarksError, QuestionError, TransportError, UnreadableAnswer
from .marks import normalise, read, read_text
from .prompt import Prompt, build
from .transports import CALL, KINDS, CallOptions, Step, Transport
from .types import (Choice, ChoiceAnswer, Meta, Noul, NoulAnswer, Score, ScoreAnswer, SystemOneResponse, Tfu,
                    TfuAnswer, Usage, as_text, coerce)

SIDES = ("true", "false", "unsure")
MIN_MASS = 0.5      # an answer is read when the options take at least this much of the answer token


def side_of(token: str, forms: Mapping[str, list[str]] | None = None) -> str | None:
    """Which yes/no side a token starts (`True`, ` [T`, `Yes` → true), if exactly one. `forms`:
    the words of each side (`forms.json` plus the binding's)."""
    forms = forms or config_forms()
    t = normalise(token).rstrip("]").strip().lower()
    if not t:
        return None
    hits = [s for s in SIDES if t in forms[s] or s.startswith(t)]
    return hits[0] if len(hits) == 1 else None


def soften(logits: dict[str, float], temperature: float) -> dict[str, float]:
    finite = {k: v for k, v in logits.items() if v > -math.inf}
    if not finite:
        return {k: 1 / len(logits) for k in logits}
    m = max(finite.values())
    e = {k: (math.exp((v - m) / temperature) if v > -math.inf else 0.0) for k, v in logits.items()}
    s = sum(e.values())
    return {k: v / s for k, v in e.items()}


class DecisionClient:
    def __init__(self, model: str, *, provider: str | None = None, base_url: str | None = None,
                 api_key=None, key_env: str | None = None, env_file=None, timeout: float = 120.0,
                 retries: int = 6, workers: int = 8, calibrated: bool = True, strict: bool = False):
        self._args = dict(provider=provider, base_url=base_url, api_key=api_key, key_env=key_env, env_file=env_file,
                          timeout=timeout, retries=retries, workers=workers, calibrated=calibrated, strict=strict)
        self._others: dict[str, DecisionClient] = {}
        self._lock = threading.Lock()
        self.binding: Binding = resolve(model, provider, base_url, api_key if isinstance(api_key, str) else None)
        b = self.binding
        # the provider's own address names the host its environment key may go to
        spec = auth.AuthSpec.from_config(b.auth, providers()[b.provider]["base_url"])
        self.credential = auth.resolve(spec, key=api_key, key_env=key_env, env_file=env_file, base_url=b.base_url)
        headers = auth.headers(spec, self.credential)
        kind = KINDS[b.transport]
        kw = dict(timeout=timeout, retries=retries, top_k=b.top_k, extra=b.extra)
        if b.transport == "chat_ask":
            kw.update(logprobs=b.logprobs, max_tokens_field=b.max_tokens_field,
                      temperature=b.options.get("temperature", True))
        if b.transport == "completions":
            kw.update(b.options)
        if b.transport == "anthropic":
            kw.update(b.options)
        self.transport: Transport = kind(b.base_url, b.model, headers, **kw)
        self.workers, self.calibrated, self.strict = workers, calibrated, strict

    # -- the SDK's surface ----------------------------------------------------------------------

    def system_one(self, state, questions: Mapping, *, model: str | None = None, timeout: float | None = None,
                   extra_headers: Mapping[str, str] | None = None,
                   extra_body: Mapping | None = None) -> SystemOneResponse:
        """Answer named questions about one state.

        `model`: another model for this call, reached with this client's provider, address and key
        (a client for it is made once and kept). `timeout`, `extra_headers`, `extra_body`: for this
        call's requests only; `extra_body` is merged over the provider's fields."""
        if model is not None and model not in (self.binding.name, self.binding.model):
            return self._other(model).system_one(state, questions, timeout=timeout, extra_headers=extra_headers,
                                                 extra_body=extra_body)
        if not questions:
            raise ValueError("no questions")
        qs = {k: coerce(k, q) for k, q in questions.items()}
        prompts = {}
        for k, q in qs.items():                       # every prompt before the first request: a bad question
            try:                                      # fails the call without spending anything
                prompts[k] = None if isinstance(q, (Choice, Score)) and len(q.criteria) == 1 else build(state, q)
            except MarksError as e:
                raise QuestionError(k, str(e)) from None
        token = CALL.set(CallOptions(timeout, dict(extra_headers or {}), dict(extra_body or {})))
        try:                                          # each worker runs in a copy of this call's context
            with ThreadPoolExecutor(max(1, min(self.workers, len(qs)))) as ex:
                futures = {k: ex.submit(contextvars.copy_context().run, self._answer, k, qs[k], prompts[k])
                           for k in qs}
                done = {k: f.result() for k, f in futures.items()}
        finally:
            CALL.reset(token)
        answers = {k: a for k, (a, _, _) in done.items()}
        tin = sum(t for _, t, _ in done.values())
        tout = sum(t for _, _, t in done.values())
        return SystemOneResponse(model=self.binding.name, usage=Usage(tin, tout), answers=answers)

    def check(self, save: bool = False):
        """Probe this binding with a dozen requests on easy questions (`check.py`): a `CheckReport`,
        printable, with `.ok`. A pass marks this client's answers `meta.checked`; `save=True` writes
        the date (and the digits found) into the user's `models.json`."""
        from .check import run
        return run(self, save=save)

    def _other(self, model: str) -> "DecisionClient":
        with self._lock:
            if model not in self._others:
                args = dict(self._args)
                if "@" not in model and args["provider"] is None and not args["base_url"]:
                    try:                              # a binding's short name, else a model id at this provider
                        resolve(model)
                    except ConfigError:
                        args["provider"] = self.binding.provider
                self._others[model] = DecisionClient(model, **args)
            return self._others[model]

    def close(self) -> None:
        self.transport.close()
        for other in self._others.values():
            other.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self) -> str:
        return f"DecisionClient({self.binding.name!r}, key={self.credential.source})"

    # -- reading --------------------------------------------------------------------------------

    def _temp(self, mode: str) -> float:
        return float(self.binding.calibration.get(mode, 1.0)) if self.calibrated else 1.0

    def _meta(self, **kw) -> Meta:
        return Meta(logprobs=self.transport.logprobs and self.binding.logprobs,
                    calibrated=self.calibrated and bool(self.binding.calibration),
                    checked=self.binding.checked is not None, **kw)

    def _answer(self, key: str, q, prompt: Prompt | None):
        if prompt is None:
            return self._single(q), 0, 0
        steps: list[Step] = []

        def step(written: str) -> Step:
            try:
                s = self.transport.step(prompt, written)
            except TransportError as e:
                raise TransportError(f"question {key!r}: {e}") from e
            except (KeyError, IndexError, TypeError, AttributeError, ValueError) as e:
                if isinstance(e, LLM2DecisionError):
                    raise
                raise TransportError(f"question {key!r}: the provider's response has an unexpected shape "
                                     f"({type(e).__name__}: {str(e)[:120]})") from e
            steps.append(s)
            return s

        try:
            ans = self._yes_no(q, prompt, step) if isinstance(q, (Noul, Tfu)) else self._options(q, prompt, step)
        finally:
            self.transport.forget(prompt)
        reasoning = [s.reasoning_tokens for s in steps if s.reasoning_tokens is not None]
        ans = replace(ans, meta=replace(ans.meta, reasoning_tokens=sum(reasoning) if reasoning else None))
        if self.strict and not ans.meta.answered:
            said = next((s.text for s in steps if s.text), None)
            raise UnreadableAnswer(key, "the model gave none of the options"
                                   + (f"; it wrote {said[:80]!r}" if said else ""))
        return ans, sum(s.input_tokens for s in steps), sum(s.output_tokens for s in steps)

    def _yes_no(self, q, prompt: Prompt, step):
        f = self.binding.forms
        s = step("")
        n = 1
        if s.candidates and side_of(s.candidates[0][0], f) is None and not normalise(s.candidates[0][0]).strip("] ") \
                and s.candidates[0][1] > math.log(0.5) and self.binding.transport != "chat_ask":
            s = step(s.candidates[0][0])                     # a bare bracket first: read the token after it
            n = 2
        if s.text is not None or not s.candidates:           # text only
            side = side_of(s.text or "", f) if s.text is not None else None
            logits = {k: (0.0 if k == side else -math.inf) for k in SIDES}
            bounded: tuple = ()
        else:
            mass = {k: 0.0 for k in SIDES}
            for tok, lp in s.candidates:
                side = side_of(tok, f)
                if side:
                    mass[side] += math.exp(lp)
            floor = s.candidates[-1][1] if s.limit and len(s.candidates) >= s.limit else -math.inf
            logits = {k: (math.log(v) if v > 0 else floor) for k, v in mass.items()}
            bounded = tuple(k for k, v in mass.items() if v == 0 and floor > -math.inf)
        offered = ("true", "false") if isinstance(q, Noul) else SIDES
        if s.text is not None or not s.candidates:
            read_any = any(logits[k] > -math.inf for k in offered)
        else:
            read_any = sum(math.exp(lp) for tok, lp in s.candidates if side_of(tok, f) in offered) >= MIN_MASS
        meta = self._meta(bounded=bounded, requests=n, answered=read_any)
        if isinstance(q, Noul):
            two = soften({k: logits[k] for k in ("true", "false")}, self._temp("verdict"))
            return NoulAnswer(noul=two["true"], meta=meta)
        three = soften(logits, self._temp("three_answers"))
        lead = max(three, key=three.get)
        return TfuAnswer(tfu={"unsure": "unknown"}.get(lead, lead),
                         probabilities={"true": three["true"], "false": three["false"], "unknown": three["unsure"]},
                         confidence=three[lead], meta=meta)

    def _options(self, q, prompt: Prompt, step):
        m = prompt.marks
        mode = "scale" if isinstance(q, Score) else "choice"
        if not (self.transport.logprobs and self.binding.logprobs):
            s = step("")
            key = read_text(m, s.text or "")
            p = {k: (1.0 if k == key else 0.0) for k in m.keys} if key else {k: 1 / len(m.keys) for k in m.keys}
            meta = self._meta(requests=1, answered=key is not None)
        else:
            r = read(m, lambda w: (lambda s: (s.candidates, s.limit))(step(w)), digits=self.binding.digits)
            raw = r.probabilities
            t = self._temp(mode)
            p = {k: (v ** (1 / t) if v > 0 else 0.0) for k, v in raw.items()}
            z = sum(p.values())
            p = {k: v / z for k, v in p.items()} if z else {k: 1 / len(raw) for k in raw}
            meta = self._meta(bounded=tuple(sorted(r.bounded)), requests=r.requests,
                              answered=sum(r.mass.values()) >= MIN_MASS)
        lead = max(p, key=p.get)
        if isinstance(q, Choice):
            return ChoiceAnswer(choice=lead, confidence=p[lead], probabilities=p, meta=meta)
        probs = {int(k): v for k, v in p.items()}
        return ScoreAnswer(score=sum(i * v for i, v in probs.items()), confidence=p[lead],
                           legend={i: d for i, d in enumerate(q.criteria)}, probabilities=probs, meta=meta)

    def _single(self, q):
        meta = Meta(logprobs=False, requests=0, checked=self.binding.checked is not None)
        if isinstance(q, Choice):
            label = next(iter(q.criteria))
            return ChoiceAnswer(choice=label, confidence=1.0, probabilities={label: 1.0}, meta=meta)
        return ScoreAnswer(score=0.0, confidence=1.0, legend={0: q.criteria[0]}, probabilities={0: 1.0}, meta=meta)


class AsyncDecisionClient:
    """The same client for `asyncio` code: `await client.system_one(...)`. Requests run in a thread
    each (the questions of one call side by side, as in `DecisionClient`), so the event loop is not
    blocked; `asyncio.gather` over several calls runs them together."""

    def __init__(self, model: str, **kw):
        self.sync = DecisionClient(model, **kw)
        self.binding = self.sync.binding

    async def system_one(self, state, questions: Mapping, **kw) -> SystemOneResponse:
        return await asyncio.to_thread(self.sync.system_one, state, questions, **kw)

    async def check(self, **kw):
        return await asyncio.to_thread(self.sync.check, **kw)

    async def aclose(self) -> None:
        self.sync.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    def __repr__(self) -> str:
        return "Async" + repr(self.sync)
