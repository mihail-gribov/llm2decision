"""An OpenAI-compatible server that behaves like a model, for end-to-end tests without a network.

The state says what the right answer is — `pick: <description word>` for a choice or a score, `truth:
yes|no|unsure` for a yes/no question. The server finds the matching option in the prompt and puts
`p` on it (the rest spread evenly), cutting the answer into tokens one digit or letter at a time.

  continue mode   (`continue_final_message`): the distribution of the next token after the
                  assistant's text, top-k, shuffled, with a duplicate — the way real servers return it
  ask mode        (no prefix): it writes `Answer: [<mark>]` with per-token logprobs
  mistral mode    (`prefix: true`): it writes the rest of the answer as text
  completions     (`/completions`, a prompt rendered by `FakeTokenizer`): the next-token distribution
                  in the legacy `top_logprobs` form, one dict per position
  messages        (`/v1/messages`, Anthropic): text, with the prefill or asked for `Answer: [..]`

Every request body and its headers are kept in `requests` for the tests to look at. `fail` makes the
next responses fail with the given codes (or `(code, message)` pairs). A state with `prose` makes
the model talk instead of answering: no option, no bracket. `oracle` maps a piece of a state to the
marker it stands for (so plain-language states can be answered); `ignore_prefix` makes a continuing
server start the answer anew; `grouped` writes a number in one token; `reasoning` is reported.
"""
from __future__ import annotations

import json
import math
import random
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

P = 0.8


def answer_marks(system: str, user: str, state: str) -> tuple[dict[str, float], bool]:
    """The distribution over marks (or yes/no words) this prompt and state call for."""
    if "yes/no question" in system:
        m = re.search(r"truth:\s*(yes|no|unsure)", state)
        if not m:                                        # no opinion: even odds
            return {"True": 0.45, "False": 0.45, "Unsure": 0.1}, True
        target ={"yes": "True", "no": "False", "unsure": "Unsure"}[m.group(1)]
        words = ["True", "False", "Unsure"]
        return {w: (P if w == target else (1 - P) / 2) for w in words}, True
    marks = re.findall(r"^\[([^\]]+)\] (.*)$", user, re.M)
    m = re.search(r"pick:\s*([A-Za-z0-9]+)", state)
    if not m:                                            # no opinion: even odds
        return {mk: 1 / len(marks) for mk, _ in marks}, False
    target = next(mk for mk, desc in marks if m.group(1).lower() in desc.lower())
    rest = (1 - P) / (len(marks) - 1)
    return {mk: (P if mk == target else rest) for mk, _ in marks}, False


GROUPED = threading.local()


def tokens_of(answer: str, yes_no: bool) -> list[str]:
    if yes_no or answer.isalpha() or getattr(GROUPED, "on", False):
        return [answer]
    return list(answer)


def next_dist(dist: dict[str, float], written: str, yes_no: bool) -> dict[str, float]:
    out: dict[str, float] = {}
    total = sum(p for a, p in dist.items() if a.startswith(written))
    for a, p in dist.items():
        if not a.startswith(written):
            continue
        rest = a[len(written):]
        tok = tokens_of(rest, yes_no)[0] if rest else "]"
        out[tok] = out.get(tok, 0.0) + p / total
    return out


class Handler(BaseHTTPRequestHandler):
    server: "LLMServer"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append({"path": self.path, "body": body,
                                     "headers": {k.lower(): v for k, v in self.headers.items()}})
        if self.server.fail:
            code = self.server.fail.pop(0)
            code, message = code if isinstance(code, tuple) else (code, f"fail {code}")
            self.send_response(code)
            if self.server.retry_after is not None:
                self.send_header("Retry-After", self.server.retry_after)
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": message}}).encode())
            return
        if self.server.broken is not None:
            self._send(self.server.broken)
            return
        if self.path.endswith("/messages"):
            self._anthropic(body)
            return
        if self.path.endswith("/completions") and "prompt" in body:
            self._completions(body)
            return
        msgs = body["messages"]
        system, user = msgs[0]["content"], msgs[1]["content"]
        state = re.search(r"<state>\n(.*?)\n</state>", user, re.S).group(1)
        state = next((v for k, v in self.server.oracle.items() if k in state), state)
        GROUPED.on = self.server.grouped
        usage = {"prompt_tokens": len(system + user) // 4, "completion_tokens": 1}
        if "prose" in state:
            self._prose(body, msgs, usage)
            return
        dist, yes_no = answer_marks(system, user, state)
        rng = random.Random(len(self.server.requests))
        if body.get("continue_final_message") and self.server.ignore_prefix:
            first = "Answer" if self.server.ignore_prefix is True else "["     # "bracket": a `[` of its own
            top = [{"token": first, "logprob": math.log(0.9)}, {"token": "(", "logprob": math.log(0.1)}]
            content = [{"token": first, "logprob": top[0]["logprob"], "top_logprobs": top}]
            self._send({"choices": [{"message": {"role": "assistant", "content": ""},
                                     "logprobs": {"content": content}}], "usage": usage})
            return
        if body.get("continue_final_message"):
            written = msgs[-1]["content"].split("[", 1)[1]
            nxt = next_dist(dist, written, yes_no)
            k = body.get("top_logprobs", 20)
            top = sorted(((t, math.log(p)) for t, p in nxt.items()), key=lambda x: -x[1])[:k]
            top = [{"token": t, "logprob": lp} for t, lp in top]
            if top:
                top.append(dict(top[0]))                 # the same token twice, as some servers send it
            rng.shuffle(top)                             # and unsorted
            content = [{"token": top[0]["token"] if top else "", "logprob": 0, "top_logprobs": top}]
            msg = {"role": "assistant", "content": ""}
            self._send({"choices": [{"message": msg, "logprobs": {"content": content}}], "usage": usage})
            return
        if msgs[-1].get("prefix"):                       # Mistral: write the rest as text
            written = msgs[-1]["content"].split("[", 1)[1]
            best = max((a for a in dist if a.startswith(written)), key=dist.get)
            text = msgs[-1]["content"] + best[len(written):] + "]"
            self._send({"choices": [{"message": {"role": "assistant", "content": text}}], "usage": usage})
            return
        best = max(dist, key=dist.get)                   # ask mode: write `Answer: [<best>]`
        toks = ["Answer", ":", " ["] + tokens_of(best, yes_no) + ["]"]
        content, written = [], ""
        for i, t in enumerate(toks):
            if i < 3 or t == "]":
                tops = [{"token": t, "logprob": 0.0}]
            else:
                nxt = next_dist(dist, written, yes_no)
                tops = sorted(({"token": a, "logprob": math.log(p)} for a, p in nxt.items()), key=lambda x: -x["logprob"])
                tops = tops[: body.get("top_logprobs", 20)]
                written += t
            content.append({"token": t, "logprob": tops[0]["logprob"], "top_logprobs": tops})
        msg = {"role": "assistant", "content": "".join(toks)}
        choice = {"message": msg}
        if body.get("logprobs"):
            choice["logprobs"] = {"content": content}
        self._send({"choices": [choice], "usage": dict(usage, completion_tokens=len(toks),
                                                        completion_tokens_details={"reasoning_tokens": self.server.reasoning})})

    def _anthropic(self, body):
        """The Messages API: a Claude 5.x model (`-5-` in its id) takes no prefill and no temperature
        and thinks unless told not to; older ones take the prefill."""
        model, system, messages = body["model"], body.get("system", ""), body["messages"]
        newer = "-5-" in model
        if newer and (messages[-1]["role"] == "assistant" or "temperature" in body):
            self._send({"type": "error", "error": {"message": "prefill or temperature not supported"}}, 400)
            return
        user = messages[0]["content"]
        opened = messages[-1]["content"] if messages[-1]["role"] == "assistant" else "Answer: ["
        state = re.search(r"<state>\n(.*?)\n</state>", user, re.S).group(1)
        if "prose" in state:
            text = "I cannot tell from this."
        else:
            dist, _ = answer_marks(system, user, state)
            written = opened.split("[", 1)[1]
            best = max((a for a in dist if a.startswith(written)), key=dist.get)
            text = best[len(written):] + "]"
            if messages[-1]["role"] == "user":
                text = "Answer: [" + text
        content = [{"type": "text", "text": text}]
        if newer and (body.get("thinking") or {}).get("type") not in ("disabled", "between_tools"):
            content = [{"type": "thinking", "thinking": "…"}]       # thinks by default, runs out of tokens
        self._send({"content": content, "usage": {"input_tokens": len(user) // 4, "output_tokens": 2}})

    def _completions(self, body):
        forms = (r"<\|system\|>(.*)<\|user\|>(.*)<\|assistant\|>(.*)$",                       # FakeTokenizer
                 r"<\|im_start\|>system\n(.*)<\|im_end\|>\n<\|im_start\|>user\n(.*)<\|im_end\|>\n"
                 r"<\|im_start\|>assistant\n<think>\n\n</think>\n\n(.*)$",                              # Qwen
                 r"# Instructions\n\n(.*)\n\n<\|end\|><\|start\|>user<\|message\|>(.*)<\|end\|>"
                 r"<\|start\|>assistant(?:<\|channel\|>final<\|message\|>)?(.*)$")                 # harmony
        system, user, opened = next(m.groups() for f in forms if (m := re.match(f, body["prompt"], re.S)
                                                                  or re.search(f, body["prompt"], re.S)))
        state = re.search(r"<state>\n(.*?)\n</state>", user, re.S).group(1)
        state = next((v for k, v in self.server.oracle.items() if k in state), state)
        GROUPED.on = self.server.grouped
        dist, yes_no = answer_marks(system, user, state)
        nxt = next_dist(dist, opened.split("[", 1)[1], yes_no)
        top = dict(sorted(((t, math.log(p)) for t, p in nxt.items()), key=lambda x: -x[1])[: body.get("logprobs", 5)])
        self._send({"choices": [{"text": next(iter(top), ""), "logprobs": {"top_logprobs": [top]}}],
                    "usage": {"prompt_tokens": len(body["prompt"]) // 4, "completion_tokens": 1}})

    def _prose(self, body, msgs, usage):
        words = ["I", " cannot", " tell", " from", " this", "."]
        if body.get("continue_final_message"):
            top = [{"token": "I", "logprob": math.log(0.7)}, {"token": "It", "logprob": math.log(0.3)}]
            content = [{"token": "I", "logprob": top[0]["logprob"], "top_logprobs": top}]
            self._send({"choices": [{"message": {"role": "assistant", "content": ""},
                                     "logprobs": {"content": content}}], "usage": usage})
            return
        text = "".join(words)
        if msgs[-1].get("prefix"):
            text = msgs[-1]["content"] + text
        choice = {"message": {"role": "assistant", "content": text}}
        if body.get("logprobs"):
            choice["logprobs"] = {"content": [{"token": w, "logprob": -0.1, "top_logprobs": [{"token": w, "logprob": -0.1}]}
                                              for w in words]}
        self._send({"choices": [choice], "usage": usage})

    def _send(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class LLMServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), Handler)
        self.requests: list[dict] = []
        self.fail: list = []
        self.oracle: dict[str, str] = {}
        self.ignore_prefix = False
        self.grouped = False
        self.reasoning = 0
        self.broken: dict | None = None
        self.retry_after: str | None = None
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}/v1"

    @property
    def root(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"


class FakeTokenizer:
    """Just enough of a `transformers` tokenizer for the completions transport: a plain chat template."""

    def __init__(self, name):
        self.name = name

    @classmethod
    def from_pretrained(cls, name):
        return cls(name)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        self.kwargs = kw
        out = "".join(f"<|{m['role']}|>{m['content']}" for m in msgs)
        return out + ("<|assistant|>" if add_generation_prompt else "")
