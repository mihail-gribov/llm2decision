"""How a prompt reaches a model and what comes back: the candidates for the next token after `[`.

One operation, five ways to do it, and a decision API that answers whole questions:

  chat_continue  chat with the answer opened by an assistant message the server continues
                 (`continue_final_message`) — vLLM, SGLang, Nebius; top-k log-probabilities
  chat_ask       chat with no prefix (OpenAI, OpenRouter): the model is asked to reply
                 `Answer: [<answer>]`, the tokens after its own `[` are read; later positions come
                 from the same generation, along the model's own path
  completions    the prompt rendered locally with the model's chat template (`transformers`), for
                 servers that drop an assistant prefix in chat
  anthropic      Messages API, the answer prefilled where the model takes it; text only
  mistral        chat with `prefix: true`; text only
  jev            TypeSafe's decision API: whole questions in, probabilities out — no token to read

Whatever the way, the result is a `Step`: candidates most probable first, one per token string (servers
return them unsorted, and the same string twice), how many the provider returns at most, and — for
text-only ways — the text written after `[`. Reasoning is switched off as the provider takes it,
from configuration; nothing here is specific to one provider.
"""
from __future__ import annotations

import math
import random
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from ._http import NetworkError, Session

from .prompt import Prompt

ASK_FORMAT = "\n\nReply with exactly `Answer: [<your answer>]` and nothing else."


from .errors import TransportError  # noqa: E402  re-exported


@dataclass(frozen=True)
class CallOptions:
    """What one `system_one` call changes for its requests: a timeout, extra headers, extra body fields
    (merged over the provider's, a nested object key by key)."""
    timeout: float | None = None
    headers: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] = field(default_factory=dict)


CALL: ContextVar[CallOptions] = ContextVar("llm2decision_call", default=CallOptions())


def _over(base: dict, more: dict) -> dict:
    out = dict(base)
    for k, v in more.items():
        out[k] = _over(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@dataclass
class Step:
    candidates: list[tuple[str, float]] = field(default_factory=list)   # most probable first
    limit: int = 0                     # top-k the provider returns at most (0: unknown or text only)
    text: str | None = None            # text-only providers: what was written after `[`
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int | None = None


def tidy(cands) -> list[tuple[str, float]]:
    """Sorted, most probable first; the same token string once (its best log-probability)."""
    best: dict[str, float] = {}
    for tok, lp in cands:
        if lp is None:
            continue
        best[tok] = max(float(lp), best.get(tok, -math.inf))
    return sorted(best.items(), key=lambda kv: -kv[1])


class Transport:
    """Base: an HTTP session with retries. Subclasses implement `step`."""

    logprobs = True

    def __init__(self, base_url: str, model: str, headers: dict[str, str], *, timeout: float = 120.0,
                 retries: int = 6, top_k: int = 20, extra: dict[str, Any] | None = None):
        self.base_url = base_url.rstrip("/") + "/"
        self.model, self.timeout, self.retries, self.top_k = model, timeout, retries, top_k
        self.extra = dict(extra or {})     # provider fields: switching reasoning off, routing, …
        self.s = Session(headers)

    def close(self) -> None:
        self.s.close()

    def forget(self, prompt) -> None:
        """Drop whatever is kept for this prompt (nothing, unless the transport caches)."""

    def post(self, path: str, body: dict) -> dict:
        call = CALL.get()
        body = _over(body, call.body)
        timeout = call.timeout or self.timeout
        last, wait_for = None, None
        for attempt in range(self.retries + 1):
            wait_for = None
            try:
                r = self.s.post(self.base_url + path, body, timeout=timeout, headers=call.headers or None)
            except NetworkError as e:                       # dropped or slow connection
                last = str(e)
            else:
                if r.status == 200:
                    try:
                        return r.json()
                    except ValueError:
                        raise TransportError(f"the response is not JSON: {r.text[:200]}") from None
                last = f"HTTP {r.status}: {r.text[:300]}"
                if r.status not in RETRY or _out_of_credit(r.text):
                    raise TransportError(last)
                wait_for = _retry_after(r.headers.get("Retry-After"))
            if attempt < self.retries:
                # the provider's own wait when it names one (capped), else exponential backoff with jitter
                pause = min(wait_for, 120.0) if wait_for is not None else min(2 ** attempt, 30)
                time.sleep(pause * (1 + random.random() * 0.25))
        raise TransportError(f"gave up after {self.retries + 1} attempts: {last}")

    def step(self, prompt: Prompt, written: str) -> Step:
        raise NotImplementedError


RETRY = (408, 409, 425, 429, 500, 502, 503, 504, 529)     # worth waiting for; 529: Anthropic overloaded


def _retry_after(value: str | None) -> float | None:
    """Seconds from a `Retry-After` header: a number, or an HTTP date."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime
        import datetime as dt
        return max(0.0, (parsedate_to_datetime(value) - dt.datetime.now(dt.timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return None


def _out_of_credit(text: str) -> bool:
    """A 429 that will not pass by waiting: the account has no money (OpenAI says so with a 429)."""
    t = text.lower()
    return any(s in t for s in ("insufficient_quota", "no credits", "exhausted your budget", "add funds"))


class ChatContinue(Transport):
    def step(self, prompt, written):
        d = self.post("chat/completions", {
            "model": self.model, "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": self.top_k,
            "messages": [{"role": "system", "content": prompt.system}, {"role": "user", "content": prompt.user},
                         {"role": "assistant", "content": prompt.tail + written}],
            "continue_final_message": True, "add_generation_prompt": False, **self.extra})
        c = d["choices"][0]
        if c.get("logprobs") is None:
            raise TransportError("the response carries no log-probabilities: this model or server does not "
                                 "return them (a binding with \"logprobs\": false reads the text instead)")
        content = c["logprobs"].get("content") or []
        cands = [(x["token"], x["logprob"]) for x in content[0]["top_logprobs"]] if content else []
        u = d.get("usage") or {}
        return Step(tidy(cands), self.top_k, None, u.get("prompt_tokens", 0), u.get("completion_tokens", 0),
                    (u.get("completion_tokens_details") or {}).get("reasoning_tokens"))


class ChatAsk(Transport):
    """No prefix: one generation per prompt, cached; `written` walks the model's own path in it."""

    def __init__(self, *a, max_tokens: int = 16, logprobs: bool = True, max_tokens_field: str = "max_tokens",
                 temperature: bool = True, **kw):
        super().__init__(*a, **kw)
        self.max_tokens, self.logprobs, self.max_tokens_field = max_tokens, logprobs, max_tokens_field
        self.temperature = temperature           # False: the model refuses the field
        self._cache: dict[tuple[str, str], tuple[list, str, dict]] = {}

    def _generate(self, prompt):
        """The generation for this prompt and whether it was made now (False: from the cache, so
        its tokens are not counted again)."""
        key = (prompt.system, prompt.user)
        fresh = key not in self._cache
        if fresh:
            body = {"model": self.model, self.max_tokens_field: self.max_tokens,
                    "messages": [{"role": "system", "content": prompt.system},
                                 {"role": "user", "content": prompt.user + ASK_FORMAT}], **self.extra}
            if self.temperature:                 # a text answer must not be a sample
                body["temperature"] = 0
            if self.logprobs:
                body.update(logprobs=True, top_logprobs=self.top_k)
            d = self.post("chat/completions", body)
            c = d["choices"][0]
            self._cache[key] = (((c.get("logprobs") or {}).get("content") or []),
                                c["message"].get("content") or "", d.get("usage") or {})
        return self._cache[key], fresh

    def forget(self, prompt) -> None:
        """Drop the generation kept for this prompt; the client calls it once a question is read."""
        self._cache.pop((prompt.system, prompt.user), None)

    def step(self, prompt, written):
        (positions, text, u), fresh = self._generate(prompt)
        usage = dict(input_tokens=u.get("prompt_tokens", 0) if fresh else 0,
                     output_tokens=u.get("completion_tokens", 0) if fresh else 0,
                     reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens") if fresh else None)
        bracket = any("[" in p["token"] for p in positions)
        if not self.logprobs or not positions or not bracket:   # no logprobs, or no `[` written: read the text
            after = text.split("[", 1)[1] if "[" in text else text
            return Step([], 0, after.split("]")[0].strip(), **usage)
        so_far, opened = "", False
        for pos in positions:                     # past the model's own `[`, follow what is already read
            if opened and so_far.split("[", 1)[1].strip() == written:
                return Step(tidy((x["token"], x["logprob"]) for x in pos["top_logprobs"]), self.top_k, None, **usage)
            so_far += pos["token"]
            opened = opened or "[" in so_far
        return Step([], self.top_k, None, **usage)  # off the model's own path: nothing to read there


class Completions(Transport):
    """The chat template rendered here; the server only continues text.

    A binding's `chat_template` is the rendered form with `{system}`, `{user}` and `{date}` in place
    (checked against the model's own template), so no extra package is needed. Without one, the
    model's template is rendered with `transformers` (`pip install 'llm2decision[template]'`)."""

    def __init__(self, *a, tokenizer: str = "", template_kwargs: dict | None = None, suffix: str = "",
                 chat_template: str = "", **kw):
        super().__init__(*a, **kw)
        self.template_kwargs, self.suffix, self.chat_template = dict(template_kwargs or {}), suffix, chat_template
        self.tok = None
        if chat_template:
            return
        try:
            from transformers import AutoTokenizer
        except ImportError as e:                 # pragma: no cover - optional dependency
            raise TransportError("the completions transport renders the chat template locally: "
                                 "pip install 'llm2decision[template]'") from e
        self.tok = AutoTokenizer.from_pretrained(tokenizer or self.model)

    def render(self, prompt: Prompt) -> str:
        if self.chat_template:
            import datetime as dt
            text = (self.chat_template.replace("{date}", dt.date.today().isoformat())
                    .replace("{system}", prompt.system).replace("{user}", prompt.user))
            return text + self.suffix
        msgs = [{"role": "system", "content": prompt.system}, {"role": "user", "content": prompt.user}]
        try:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **self.template_kwargs)
        except TypeError:
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text + self.suffix

    def step(self, prompt, written):
        d = self.post("completions", {"model": self.model, "prompt": self.render(prompt) + prompt.tail + written,
                                      "max_tokens": 1, "temperature": 0, "logprobs": self.top_k, **self.extra})
        c = d["choices"][0]
        tops = ((c.get("logprobs") or {}).get("top_logprobs") or [{}])[0] or {}
        u = d.get("usage") or {}
        return Step(tidy(tops.items()), self.top_k, None, u.get("prompt_tokens", 0), u.get("completion_tokens", 0))


class TextOnly(Transport):
    """Providers that give no log-probabilities: the answer is the text written after `[`."""
    logprobs = False


class Anthropic(TextOnly):
    """The Messages API over HTTP. Models that still take an assistant prefill (Claude Haiku 4.5 and
    earlier) get the answer opened with `[`; newer ones (`prefill: false` in the binding) are asked to
    reply `Answer: [<answer>]` and the text after their `[` is read. `temperature: false` leaves sampling
    to the model, for models that refuse the field. A thinking block in the reply counts as reasoning."""

    VERSION = "2023-06-01"

    def __init__(self, base_url, model, headers, *, prefill: bool = True, temperature: bool = True,
                 ask_format: str = ASK_FORMAT, **kw):
        super().__init__(base_url, model, {**headers, "anthropic-version": self.VERSION}, **kw)
        self.prefill, self.temperature, self.ask_format = prefill, temperature, ask_format

    def step(self, prompt, written):
        if self.prefill:
            messages = [{"role": "user", "content": prompt.user},
                        {"role": "assistant", "content": prompt.tail + written}]
        else:
            messages = [{"role": "user", "content": prompt.user + self.ask_format}]
        body = {"model": self.model, "max_tokens": 8 if self.prefill else 24, "system": prompt.system,
                "messages": messages, **({"temperature": 0} if self.temperature else {}), **self.extra}
        m = self.post("v1/messages", body)
        blocks = m.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        u = m.get("usage") or {}
        thought = any(b.get("type") in ("thinking", "redacted_thinking") for b in blocks)
        reasoning = u.get("output_tokens", 0) if thought else 0
        if not self.prefill:                     # `Answer: [C]`: what follows the model's own bracket
            text = text.split("[", 1)[1] if "[" in text else text
        return Step([], 0, written + text.split("]")[0].strip(), u.get("input_tokens", 0), u.get("output_tokens", 0),
                    reasoning)


class Mistral(TextOnly):
    def step(self, prompt, written):
        tail = prompt.tail + written
        d = self.post("chat/completions", {
            "model": self.model, "max_tokens": 8, "temperature": 0,
            "messages": [{"role": "system", "content": prompt.system}, {"role": "user", "content": prompt.user},
                         {"role": "assistant", "content": tail, "prefix": True}], **self.extra})
        text = d["choices"][0]["message"].get("content") or ""
        text = text[len(tail):] if text.startswith(tail) else text
        u = d.get("usage") or {}
        return Step([], 0, written + text.split("]")[0].strip(), u.get("prompt_tokens", 0), u.get("completion_tokens", 0))


class Jev(Transport):
    """TypeSafe's decision API (`POST /v1/systemone`): the questions of a call go in one request and
    come back as probabilities, so `step` does not apply."""

    def ask(self, state: str, questions: dict) -> dict:
        return self.post("systemone", {"model": self.model, "state": state, "questions": questions, **self.extra})

    def step(self, prompt, written):
        raise TransportError("the Jev API answers whole questions, not tokens")


KINDS = {"chat_continue": ChatContinue, "chat_ask": ChatAsk, "completions": Completions,
         "anthropic": Anthropic, "mistral": Mistral, "jev": Jev}
