"""DecisionClient end to end against a server that behaves like a model (tests/llm_server.py):
every question type, both ways of asking, text-only providers, calibration, keys, errors."""
import json
import math

import pytest

from llm2decision import (Choice, DecisionClient, LLM2DecisionError, Noul, QuestionError, Score, SystemOneResponse, Tfu,
                          TransportError, UnreadableAnswer)
from llm2decision import auth as auth_mod
from llm2decision import transports
from llm_server import LLMServer, P


@pytest.fixture()
def server():
    s = LLMServer()
    yield s
    s.shutdown()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(transports.time, "sleep", lambda s: None)


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM2DECISION_HOME", str(tmp_path))
    return tmp_path


QUESTIONS = {
    "shipped": Noul(instructions="Has the order shipped?", criteria={"true": "it has", "false": "it has not"}),
    "refund": Noul(instructions="Was a refund issued?"),
    "decided": Tfu(instructions="Is the delivery date set?"),
    "pay": Choice(instructions="Which payment method?",
                  criteria={"bank": "Bank transfer", "cash": "Cash", "card": "Card", "paypal": "PayPal"}),
    "urgency": Score(instructions="How urgent?", criteria=["Can wait", "Within days", "Today"]),
}
STATE = "Order shipped. pick: paypal truth: yes"


def state_for(q: str) -> str:
    return {"shipped": "truth: yes", "refund": "truth: no", "decided": "truth: unsure",
            "pay": "pick: paypal", "urgency": "pick: today"}[q]


def ask_all(c):
    return {k: c.system_one(state_for(k), {k: q}) for k, q in QUESTIONS.items()}


# -- a local server: chat with the answer continued ------------------------------------------------

def test_local_server_answers_every_type(server):
    with DecisionClient("tiny-model", base_url=server.url) as c:
        r = ask_all(c)
    assert r["shipped"].nouls["shipped"].noul == pytest.approx(P / (P + (1 - P) / 2))
    assert r["refund"].nouls["refund"].noul == pytest.approx(((1 - P) / 2) / ((1 - P) / 2 + P))
    t = r["decided"].tfus["decided"]
    assert t.tfu == "unknown" and t.probabilities["unknown"] == pytest.approx(P)
    ch = r["pay"].choices["pay"]
    assert ch.choice == "paypal" and ch.probabilities["paypal"] == pytest.approx(P) and ch.confidence == pytest.approx(P)
    sc = r["urgency"].scores["urgency"]
    assert sc.probabilities[2] == pytest.approx(P) and sc.legend[2] == "Today"
    assert sc.score == pytest.approx(0 * 0.1 + 1 * 0.1 + 2 * 0.8)
    assert all(not a.meta.checked for x in r.values() for a in x.answers.values())   # not a checked binding


def test_request_shape_on_a_local_server(server):
    with DecisionClient("tiny-model", base_url=server.url) as c:
        c.system_one("pick: card", {"q": QUESTIONS["pay"]})
    body = server.requests[-1]["body"]
    assert body["model"] == "tiny-model" and body["max_tokens"] == 1 and body["continue_final_message"] is True
    assert body["messages"][-1] == {"role": "assistant", "content": "Answer: ["}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "authorization" not in server.requests[-1]["headers"]          # local: no key
    user = body["messages"][1]["content"]
    assert "[A] Bank transfer" in user and "bank" not in user.replace("Bank", "")   # labels never shown


def test_many_questions_one_call(server):
    with DecisionClient("tiny-model", base_url=server.url) as c:
        r = c.system_one("pick: card truth: yes", {"a": QUESTIONS["shipped"], "b": QUESTIONS["pay"]})
    assert isinstance(r, SystemOneResponse) and set(r.answers) == {"a", "b"}
    assert r.nouls["a"].noul > 0.5 and r.choices["b"].choice == "card"
    assert r.usage.input_tokens > 0 and r.model == "tiny-model@local"


def test_dict_questions_like_the_sdk(server):
    with DecisionClient("tiny-model", base_url=server.url) as c:
        r = c.system_one({"note": "pick: cash"}, {
            "pay": {"type": "choice", "instructions": "Method?", "criteria": {"cash": "Cash", "card": "Card", "x": None}},
            "ok": {"type": "noul", "instructions": "Fine?"}})
    assert r.choices["pay"].choice == "cash" and "ok" in r.nouls


def test_numbers_beyond_26_options(server):
    crit = {f"o{i}": f"Option{i} item" for i in range(40)}
    with DecisionClient("tiny-model", base_url=server.url) as c:
        r = c.system_one("pick: option33", {"q": Choice(criteria=crit)})
    a = r.choices["q"]
    assert a.choice == "o33" and a.probabilities["o33"] == pytest.approx(P) and a.meta.requests >= 2


def test_one_option_needs_no_request(server):
    with DecisionClient("tiny-model", base_url=server.url) as c:
        r = c.system_one("x", {"c": Choice(criteria={"only": None}), "s": Score(criteria=["just one"])})
    assert r.choices["c"].choice == "only" and r.scores["s"].score == 0 and server.requests == []


def test_score_keeps_levels_the_question_names(server):
    q = Score(instructions="How many? (0 = none, 1 = one, 2 = two)", criteria=["None", "One", "Two"])
    with DecisionClient("tiny-model", base_url=server.url) as c:
        r = c.system_one("pick: two", {"q": q})
    assert "[2] Two" in server.requests[-1]["body"]["messages"][1]["content"]
    assert r.scores["q"].probabilities[2] == pytest.approx(P)


# -- asking without a prefix (OpenRouter, OpenAI) --------------------------------------------------

def test_ask_mode_reads_after_the_models_own_bracket(server):
    with DecisionClient("some/model", provider="openrouter", base_url=server.url, api_key="sk-or-v1-test") as c:
        r = ask_all(c)
    assert r["pay"].choices["pay"].choice == "paypal"
    assert r["pay"].choices["pay"].probabilities["paypal"] == pytest.approx(P)
    assert r["shipped"].nouls["shipped"].noul > 0.5 and r["decided"].tfus["decided"].tfu == "unknown"
    body = server.requests[-1]["body"]
    assert body["reasoning"] == {"enabled": False} and body["logprobs"] is True
    assert body["messages"][-1]["role"] == "user" and "Answer: [<your answer>]" in body["messages"][-1]["content"]
    assert server.requests[-1]["headers"]["authorization"] == "Bearer sk-or-v1-test"


def test_ask_mode_numbers_walk_the_models_path_with_one_request(server):
    crit = {f"o{i}": f"Option{i} item" for i in range(40)}
    with DecisionClient("some/model", provider="openrouter", base_url=server.url, api_key="sk-or-v1-test") as c:
        r = c.system_one("pick: option33", {"q": Choice(criteria=crit)})
    assert r.choices["q"].choice == "o33" and len(server.requests) == 1


def test_openai_uses_max_completion_tokens(server):
    with DecisionClient("gpt-x", provider="openai", base_url=server.url, api_key="sk-proj-test") as c:
        c.system_one("pick: cash", {"q": QUESTIONS["pay"]})
    body = server.requests[-1]["body"]
    assert "max_completion_tokens" in body and "max_tokens" not in body
    assert body["reasoning_effort"] == "none" and body["top_logprobs"] == 5


# -- text only (Mistral) ---------------------------------------------------------------------------

def test_text_only_provider(server):
    with DecisionClient("mini", provider="mistral", base_url=server.url, api_key="m" * 32) as c:
        r = ask_all(c)
    assert r["pay"].choices["pay"].choice == "paypal" and r["pay"].choices["pay"].probabilities["paypal"] == 1.0
    assert r["pay"].choices["pay"].meta.logprobs is False
    assert r["shipped"].nouls["shipped"].noul == 1.0 and r["refund"].nouls["refund"].noul == 0.0
    assert server.requests[-1]["body"]["messages"][-1]["prefix"] is True


# -- calibration from a binding --------------------------------------------------------------------

def test_calibration_changes_confidence_not_the_answer(server, home):
    (home / "models.json").write_text(json.dumps({
        "tiny@local": {"provider": "local", "model": "tiny-model", "checked": "2026-10-07",
                       "calibration": {"verdict": 2.0, "choice": 2.0, "three_answers": 2.0, "scale": 2.0}}}))
    with DecisionClient("tiny@local", base_url=server.url) as c:
        hot = c.system_one("pick: paypal", {"q": QUESTIONS["pay"]}).choices["q"]
    with DecisionClient("tiny@local", base_url=server.url, calibrated=False) as c:
        raw = c.system_one("pick: paypal", {"q": QUESTIONS["pay"]}).choices["q"]
    assert hot.choice == raw.choice == "paypal"
    assert hot.confidence < raw.confidence and hot.meta.calibrated and hot.meta.checked
    w = {k: math.sqrt(v) for k, v in raw.probabilities.items()}
    assert hot.probabilities["paypal"] == pytest.approx(w["paypal"] / sum(w.values()))


# -- keys ------------------------------------------------------------------------------------------

def test_environment_key_not_sent_to_another_host(server, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-from-env")
    with pytest.raises(auth_mod.AuthError):
        DecisionClient("some/model", provider="openrouter", base_url=server.url)


@pytest.mark.parametrize("key,provider", [("sk-or-v1-abc", "openrouter"), ("sk-ant-xyz", "anthropic"),
                                          ("sk-proj-xyz", "openai")])
def test_provider_from_key_prefix(key, provider):
    from llm2decision.config import resolve
    assert resolve("some/model", api_key=key).provider == provider


def test_base_url_means_a_local_server_whatever_the_key(server):
    c = DecisionClient("some/model", api_key="sk-or-v1-abc", base_url=server.url)
    assert c.binding.provider == "local"


def test_ambiguous_key_needs_a_provider():
    from llm2decision import ConfigError
    with pytest.raises(ConfigError, match="provider"):
        DecisionClient("some/model", api_key="sk-plain-key")


# -- errors ----------------------------------------------------------------------------------------

def test_bad_question_is_named():
    with DecisionClient("tiny-model", base_url="http://127.0.0.1:9/v1") as c:
        with pytest.raises(QuestionError, match="'q'"):
            c.system_one("x", {"q": {"type": "essay"}})
        with pytest.raises(QuestionError, match="'s'"):
            c.system_one("x", {"s": Score(criteria=[])})


def test_retries_then_succeeds(server):
    server.fail = [503, 429]
    with DecisionClient("tiny-model", base_url=server.url) as c:
        r = c.system_one("pick: cash", {"q": QUESTIONS["pay"]})
    assert r.choices["q"].choice == "cash"


def test_refusal_is_not_retried(server):
    server.fail = [400]
    with DecisionClient("tiny-model", base_url=server.url) as c:
        with pytest.raises(TransportError, match="400"):
            c.system_one("pick: cash", {"q": QUESTIONS["pay"]})
    assert len(server.requests) == 1


def test_unknown_binding():
    from llm2decision import ConfigError
    with pytest.raises(ConfigError, match="no binding"):
        DecisionClient("nope@nebius")


# -- a binding name and provider= that disagree ----------------------------------------------------

def test_provider_argument_wins_when_the_model_has_a_binding_there(home):
    from llm2decision import LLM2DecisionWarning
    from llm2decision.config import resolve
    (home / "models.json").write_text(json.dumps({"qwen3.8-27b@openrouter": {"provider": "openrouter", "model": "qwen/qwen3.8-27b"}}))
    with pytest.warns(LLM2DecisionWarning, match="overrides"):
        b = resolve("qwen3.8-27b@nebius", provider="openrouter")
    assert (b.provider, b.model, b.name) == ("openrouter", "qwen/qwen3.8-27b", "qwen3.8-27b@openrouter")


def test_named_binding_stays_when_the_other_provider_has_none():
    from llm2decision import LLM2DecisionWarning
    from llm2decision.config import resolve
    with pytest.warns(LLM2DecisionWarning, match="ignored"):
        b = resolve("qwen3.8-27b@nebius", provider="openrouter")
    assert (b.provider, b.model) == ("nebius", "Qwen/Qwen3.8-27B")


def test_agreeing_provider_gives_no_warning():
    import warnings
    from llm2decision.config import resolve
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve("qwen3.8-27b@nebius", provider="nebius").provider == "nebius"


# -- an answer that is not one ---------------------------------------------------------------------

ASK = dict(provider="openrouter", api_key="sk-or-v1-test")
TEXT = dict(provider="mistral", api_key="m" * 32)


@pytest.mark.parametrize("how", [{}, ASK, TEXT], ids=["continue", "ask", "text"])
@pytest.mark.parametrize("q", ["shipped", "decided", "pay", "urgency"])
def test_prose_is_marked_unanswered(server, how, q):
    with DecisionClient("tiny-model", base_url=server.url, **how) as c:
        a = c.system_one("prose", {q: QUESTIONS[q]}).answers[q]
    assert a.meta.answered is False
    probs = [a.noul, 1 - a.noul] if q == "shipped" else list(a.probabilities.values())
    assert probs == pytest.approx([1 / len(probs)] * len(probs))


@pytest.mark.parametrize("how", [{}, ASK, TEXT], ids=["continue", "ask", "text"])
def test_strict_raises_with_the_key(server, how):
    with DecisionClient("tiny-model", base_url=server.url, strict=True, **how) as c:
        with pytest.raises(UnreadableAnswer, match="'pay'") as e:
            c.system_one("prose", {"pay": QUESTIONS["pay"]})
    assert e.value.key == "pay" and isinstance(e.value, LLM2DecisionError)


def test_a_trace_of_an_option_is_not_an_answer(server, monkeypatch):
    """A model that starts anew puts e^-8 on an option: something was read, but not an answer."""
    monkeypatch.setattr(transports.ChatContinue, "step",
                        lambda self, prompt, written: transports.Step([("(", -0.001), ("A", -8.0), ("True", -8.0)], 20))
    with DecisionClient("tiny-model", base_url=server.url) as c:
        r = c.system_one("x", {"n": QUESTIONS["shipped"], "c": QUESTIONS["pay"]})
    assert r.nouls["n"].meta.answered is False and r.choices["c"].meta.answered is False


def test_strict_passes_real_answers(server):
    with DecisionClient("tiny-model", base_url=server.url, strict=True) as c:
        r = ask_all(c)
    assert all(a.meta.answered for x in r.values() for a in x.answers.values())


def test_unsure_to_a_noul_is_not_an_answer_but_to_a_tfu_it_is(server):
    with DecisionClient("tiny-model", base_url=server.url, **TEXT) as c:
        r = c.system_one("truth: unsure", {"n": QUESTIONS["shipped"], "t": QUESTIONS["decided"]})
    assert r.nouls["n"].meta.answered is False and r.tfus["t"].meta.answered is True


def test_ask_mode_reports_reasoning_tokens_once(server):
    with DecisionClient("some/model", base_url=server.url, **ASK) as c:
        r = c.system_one("pick: option33", {"q": Choice(criteria={f"o{i}": f"Option{i} item" for i in range(40)})})
    assert r.choices["q"].meta.reasoning_tokens == 0


def test_ask_mode_forgets_a_generation_once_read(server):
    with DecisionClient("some/model", base_url=server.url, **ASK) as c:
        c.system_one("pick: cash", {"q": QUESTIONS["pay"]})
        c.system_one("pick: cash", {"q": QUESTIONS["pay"]})
        assert c.transport._cache == {}
    assert len(server.requests) == 2


def test_tfu_describes_its_third_outcome(server):
    q = Tfu(instructions="Set?", criteria={"true": "set", "false": "not set", "unknown": "the thread never says."})
    with DecisionClient("tiny-model", base_url=server.url) as c:
        c.system_one("truth: yes", {"q": q})
    user = server.requests[0]["body"]["messages"][1]["content"]
    assert "Unsure means: the thread never says.\n" not in user and user.endswith("Unsure means: the thread never says.")


def test_noul_cannot_describe_a_third_outcome():
    with DecisionClient("tiny-model", base_url="http://127.0.0.1:9/v1") as c:
        with pytest.raises(QuestionError, match="'n'"):
            c.system_one("x", {"n": Noul(criteria={"unknown": "?"})})


def test_marks_error_names_the_question_before_any_request(server):
    with DecisionClient("tiny-model", base_url=server.url) as c:
        with pytest.raises(QuestionError, match="'huge'"):
            c.system_one("pick: o1", {"ok": QUESTIONS["pay"],
                                      "huge": Choice(criteria={f"o{i}": f"x{i}" for i in range(1000)})})
    assert server.requests == []


def test_exhausted_credit_is_not_retried(server):
    server.fail = [(429, "You exceeded your current quota: insufficient_quota")]
    with DecisionClient("tiny-model", base_url=server.url) as c:
        with pytest.raises(TransportError, match="429"):
            c.system_one("pick: cash", {"q": QUESTIONS["pay"]})
    assert len(server.requests) == 1


# -- the interface around a call -------------------------------------------------------------------

def test_per_call_options_reach_the_request_and_only_that_call(server):
    with DecisionClient("tiny-model", base_url=server.url) as c:
        c.system_one("pick: cash", {"a": QUESTIONS["pay"], "b": QUESTIONS["shipped"]},
                     extra_headers={"X-Trace": "t1"}, extra_body={"priority": 3, "chat_template_kwargs": {"x": 1}})
        n = len(server.requests)
        c.system_one("pick: cash", {"q": QUESTIONS["pay"]})
    for r in server.requests[:n]:
        assert r["headers"]["x-trace"] == "t1" and r["body"]["priority"] == 3
        assert r["body"]["chat_template_kwargs"] == {"enable_thinking": False, "x": 1}   # merged, not replaced
    assert all("x-trace" not in r["headers"] and "priority" not in r["body"] for r in server.requests[n:])


def test_per_call_timeout(server, monkeypatch):
    seen = []
    real = transports.Session.post
    monkeypatch.setattr(transports.Session, "post",
                        lambda self, url, body, **kw: seen.append(kw["timeout"]) or real(self, url, body, **kw))
    with DecisionClient("tiny-model", base_url=server.url, timeout=50) as c:
        c.system_one("pick: cash", {"q": QUESTIONS["pay"]}, timeout=3)
        c.system_one("pick: cash", {"q": QUESTIONS["pay"]})
    assert seen[0] == 3 and seen[-1] == 50


def test_model_per_call_uses_the_same_address_and_key(server):
    with DecisionClient("tiny-model", base_url=server.url, api_key="k-local") as c:
        r = c.system_one("pick: cash", {"q": QUESTIONS["pay"]}, model="other-model")
        again = c.system_one("pick: cash", {"q": QUESTIONS["pay"]}, model="other-model")
        same = c.system_one("pick: cash", {"q": QUESTIONS["pay"]}, model="tiny-model")
        assert len(c._others) == 1
    assert r.model == "other-model@local" and again.model == r.model and same.model == "tiny-model@local"
    assert server.requests[0]["body"]["model"] == "other-model"
    assert server.requests[0]["headers"]["authorization"] == "Bearer k-local"


def test_model_per_call_stays_at_the_provider(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "nb-test")
    base = DecisionClient("Qwen/Qwen3-8B", provider="nebius")
    assert base._other("deepseek-ai/Some").binding.provider == "nebius"
    assert base._other("qwen3-235b").binding.name == "qwen3-235b@nebius"


def test_short_binding_name():
    from llm2decision import ConfigError, config
    assert config.resolve("qwen3.8-27b").name == "qwen3.8-27b@nebius"
    fams = {}
    for k in config.bindings():
        fams.setdefault(k.split("@")[0], []).append(k)
    multi = [f for f, v in fams.items() if len(v) > 1]
    for f in multi:
        with pytest.raises(ConfigError, match="several providers"):
            config.resolve(f)
    with pytest.raises(ConfigError, match="not a known binding"):
        config.resolve("no-such-model")


def test_list_models():
    from llm2decision import list_models
    all_ = list_models()
    assert {m.name for m in all_} == set(__import__("llm2decision").bindings())
    neb = list_models("nebius")
    assert neb and all(m.provider == "nebius" for m in neb)
    q = next(m for m in all_ if m.name == "qwen3.8-27b@nebius")
    assert q.transport == "completions" and "verdict" in q.calibrated and q.checked is None and q.measured


def test_to_dict_is_json(server):
    with DecisionClient("tiny-model", base_url=server.url) as c:
        r = c.system_one("pick: today truth: yes", {k: QUESTIONS[k] for k in ("shipped", "decided", "urgency")})
    d = json.loads(json.dumps(r.to_dict()))
    assert d["model"] == "tiny-model@local" and d["usage"]["input_tokens"] > 0
    assert d["answers"]["urgency"]["probabilities"]["2"] == pytest.approx(r.scores["urgency"].probabilities[2])
    assert d["answers"]["shipped"]["meta"]["answered"] is True and d["answers"]["decided"]["type"] == "tfu"


def test_async_client(server):
    import asyncio
    from llm2decision import AsyncDecisionClient

    async def go():
        async with AsyncDecisionClient("tiny-model", base_url=server.url) as c:
            return await asyncio.gather(*(c.system_one(state_for(k), {k: q}) for k, q in QUESTIONS.items()))
    rs = asyncio.run(go())
    assert rs[3].choices["pay"].choice == "paypal" and rs[0].nouls["shipped"].noul > 0.5


def test_ask_mode_sends_temperature_zero(server):
    with DecisionClient("some/model", base_url=server.url, **ASK) as c:
        c.system_one("pick: cash", {"q": QUESTIONS["pay"]})
    assert server.requests[-1]["body"]["temperature"] == 0


def test_openai_bindings_read_text(server, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-test")
    with DecisionClient("gpt-5.4@openai", base_url=server.url, key_env="OPENAI_API_KEY") as c:
        a = c.system_one("pick: cash", {"q": QUESTIONS["pay"]}).choices["q"]
    body = server.requests[-1]["body"]
    assert "logprobs" not in body and body["temperature"] == 0
    assert a.choice == "cash" and a.meta.logprobs is False and a.probabilities["cash"] == 1.0
