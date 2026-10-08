# llm2decision

**Turn any LLM into a decision model.** You ask a question about a piece of text and get back a
probability for every possible answer instead of a paragraph to parse, so a program can act on it
directly. Any hosted model or your own server works through the same Python interface.

```python
from llm2decision import DecisionClient, Noul, Choice

with DecisionClient("qwen3.8-27b@nebius") as client:
    r = client.system_one(
        "Hi, I was charged twice for order #4411, please refund the extra payment.",
        {
            "refund": Noul(instructions="Does the customer ask for money back?"),
            "team": Choice(instructions="Which team should handle this?",
                           criteria={"billing": "Payments and refunds", "shipping": "Delivery",
                                     "tech": "Bugs and errors"}),
        },
    )

r.nouls["refund"].noul            # 0.996 — P(yes)
r.choices["team"].choice          # "billing"
r.choices["team"].probabilities   # {"billing": 0.999, "shipping": 0.0, "tech": 0.001}
```

It fits wherever a program decides something about text: routing a message to a team, checking a
reply against a policy, grading an answer on your own rubric, reading a fact from a document
(with "the text does not say" as an answer of its own), choosing an agent's next step. Because
each answer carries a probability, you decide what happens to the uncertain ones — act when the
model is sure and send the rest to a person, for example.

## 1. Choose a model

Any model of a supported provider works by its id, but the models below are tested with the
package and, where the provider returns probabilities, calibrated for it.

| model | decision set | JevBench | yes/no | yes/no/unknown | choice | score | no decision | calibration error | seconds |
|---|---|---|---|---|---|---|---|---|---|
| `claude-sonnet-5.5@anthropic` | **0.920** | 0.954 | 0.964 | 0.865 | 0.984 | 0.690 | 7.1% | 1 / 0 only | 1.16 |
| `gpt-5.4@openai` | **0.902** | 0.896 | 0.940 | 0.837 | 0.977 | 0.685 | 1.3% | 1 / 0 only | 0.92 |
| `kimi-k2.6@nebius` | **0.882** | 0.882 | 0.931 | 0.766 | 0.974 | 0.730 | 1.2% | 0.035¹ | 0.45 |
| `claude-haiku-5.5@anthropic` | **0.880** | 0.862 | 0.919 | 0.813 | 0.966 | 0.634 | 5.9% | 1 / 0 only | 0.66 |
| `qwen3.8-27b@nebius` | **0.875** | 0.885 | 0.900 | 0.804 | 0.957 | 0.719 | 0.9% | 0.034 | 0.33 |
| `qwen3.5-397b@nebius` | **0.875** | 0.860 | 0.910 | 0.780 | 0.968 | 0.713 | 0.9% | 0.036 | 0.65 |
| `deepseek-v4-pro@nebius` | **0.873** | 0.864 | 0.912 | 0.785 | 0.953 | 0.725 | 1.7% | 0.052, not calibrated | 0.59 |
| `claude-haiku-4.5@anthropic` | **0.870** | 0.855 | 0.904 | 0.797 | 0.959 | 0.652 | 1.9% | 1 / 0 only | 0.53 |
| `minimax-m3@nebius` | **0.868** | 0.850 | 0.912 | 0.793 | 0.939 | 0.674 | 1.8% | 0.034 | 0.40 |
| `gemini-2.5-flash@openrouter` | **0.864** | 0.878 | 0.911 | 0.750 | 0.966 | 0.680 | 1.1% | 1 / 0 only | 0.82 |
| `nemotron-3-ultra@nebius` | **0.862** | 0.855 | 0.910 | 0.763 | 0.953 | 0.663 | 0.9% | 0.037¹ | 0.49 |
| `hermes-4-405b@nebius` | **0.861** | 0.857 | 0.921 | 0.744 | 0.942 | 0.708 | 0.5% | 0.029¹ | 0.36 |
| `qwen3-235b@nebius` | **0.855** | 0.838 | 0.896 | 0.738 | 0.957 | 0.713 | 0.9% | 0.026 | 0.44 |
| `gpt-5.4-mini@openai` | **0.843** | 0.840 | 0.900 | 0.710 | 0.951 | 0.657 | 0.5% | 1 / 0 only | 0.62 |
| `nemotron-3-super@nebius` | **0.805** | 0.804 | 0.867 | 0.647 | 0.921 | 0.652 | 0.2% | 0.029 | 0.51 |
| `qwen3-30b-a3b@nebius` | **0.796** | 0.800 | 0.865 | 0.658 | 0.898 | 0.596 | 0.7% | 0.035 | 0.27 |
| `ministral-14b@mistral` | **0.795** | 0.830 | 0.843 | 0.678 | 0.890 | 0.640 | 0.7% | 1 / 0 only | 0.41 |
| `gpt-oss-120b@nebius` | **0.782** | 0.779 | 0.827 | 0.664 | 0.887 | 0.602 | 0.7% | 0.032 | 0.16 |
| `gemma-3-27b@nebius` | **0.775** | 0.730 | 0.850 | 0.634 | 0.863 | 0.601 | 0.5% | 0.082 | 0.16 |
| `nemotron-3-nano@nebius` | **0.713** | 0.709 | 0.754 | 0.564 | 0.843 | 0.579 | 1.4% | 0.043 | 0.33 |
| `nemotron-3.5-lightning@nebius` | **0.704** | 0.753 | 0.779 | 0.505 | 0.849 | 0.528 | 0.1% | 0.048 | 0.35 |

- **decision set** — accuracy on 2 315 held-out decision questions in English across many
  domains; the next four columns split it by question type.
- **JevBench** — accuracy on the 231 public tasks of [JevBench](https://github.com/fstandhartinger/jevbench),
  a community benchmark for decision models. This is plain accuracy rather than the benchmark's own
  score, and with 231 tasks differences under ~5 points are noise.
- **no decision** — share of questions the model **refused to decide**, over both sets: instead of
  one of the options it wrote something else, usually a calculation in prose. Your program gets no
  decision on these questions, and accuracy is counted without them, so a model that declines the
  hard questions looks more accurate on the rest. Read the two columns together.
- **calibration error** — how far the stated probabilities are from how often the model is right
  (lower is better; at 0.03 they are off by about 3 points on average). *1 / 0 only*: the provider
  gives the answer as text, so it comes with probability 1 and no confidence. ¹ calibrated on
  yes/no/unknown only.
- **seconds** — median time per question with 8 requests in parallel, which depends on the
  provider's load.

How to pick:

- **The most accurate:** `claude-sonnet-5.5`, but it refuses to decide about 7% of questions, so
  you need a fallback for those; `gpt-5.4` decides nearly everything and comes next.
- **A confidence you can set thresholds on:** a model with a low calibration error. `qwen3.8-27b`,
  `kimi-k2.6` and `qwen3.5-397b` are within 3 points of `gpt-5.4`, and `qwen3.8-27b` is the fastest
  of them.
- **Volume at a lower price:** `qwen3-30b-a3b`, `gpt-oss-120b` or `nemotron-3-super` are 7–9 points
  less accurate but still calibrated; compare your provider's prices.
- **Data that must not leave your hardware:** any model on your own OpenAI-compatible server,
  checked as in step 4.

Whatever the model, yes/no/unknown and rubric scores are the hardest question types, so where a
decision matters, a yes/no or a choice question gets better answers. A question costs a few
hundred tokens plus your text and one or a few output tokens at the provider's prices.

## 2. Install

```
pip install llm2decision
```

The package needs nothing beyond the Python standard library (Python 3.10+).

## 3. Connect

Name a model from the table, or a provider and any of its models:

```python
DecisionClient("qwen3.8-27b@nebius")                                   # a model from the table
DecisionClient("qwen3.8-27b")                                          # the same, when the name is unique
DecisionClient("google/gemini-2.5-flash", provider="openrouter")       # any model of a provider
DecisionClient("Qwen/Qwen3-8B", base_url="http://localhost:8000/v1")   # your own server, no key
```

| provider | `provider=` | key variable | answers |
|---|---|---|---|
| your own server (vLLM, SGLang, llama.cpp, Ollama) | `"local"`, or just `base_url=` | — | with probabilities |
| Nebius | `"nebius"` | `NEBIUS_API_KEY` | with probabilities |
| OpenRouter | `"openrouter"` | `OPENROUTER_API_KEY` | with probabilities where the model gives them |
| OpenAI | `"openai"` | `OPENAI_API_KEY` | text |
| Anthropic | `"anthropic"` | `ANTHROPIC_API_KEY` | text |
| Mistral | `"mistral"` | `MISTRAL_API_KEY` | text |

The key is taken from the first of these that has it: `api_key=` in code (a string, or a function
that returns one from a keyring or a vault), `key_env=` naming an environment variable of your
choice, the provider's variable from the table, and an env file — the last only when you name it
with `env_file=".env"` or `LLM2DECISION_ENV_FILE`. A key found in the environment or an env file is
sent only to its own provider, so pointing `base_url` elsewhere by mistake does not leak it; to send
a key to another address, pass it with `api_key=` or `key_env=`. Keys never appear in errors or
logs.

## 4. Check the model

Before relying on a model that is not in the table, or on your own server, run:

```
llm2decision check Qwen/Qwen3-8B --base-url http://localhost:8000/v1
```

It sends about a dozen easy questions and reports, point by point, whether the model is reachable,
answers in the expected form, gets the easy questions right, returns probabilities and keeps
reasoning off. A failed point names the reason. `client.check()` does the same in code, and
`--save` remembers a pass so that later answers carry `meta.checked`.

Then try the model on a few dozen questions of your own and look at `meta.answered`: a model that
passes the easy questions can still refuse to decide on hard ones.

## 5. Ask questions

| type | use it when | answer |
|---|---|---|
| `Noul(instructions, criteria={"true": …, "false": …})` | the answer is yes or no | `.noul` — P(yes) |
| `Tfu(instructions, criteria={"true": …, "false": …, "unknown": …})` | the text may not settle it, and "unknown" should be an answer rather than a guess | `.tfu` — `"true"`, `"false"` or `"unknown"`, with `.probabilities` |
| `Choice(criteria={label: description, …}, instructions)` | one of several named options | `.choice`, `.confidence`, `.probabilities` |
| `Score(criteria=[level 0, level 1, …], instructions)` | a level on an ordered scale | `.score` — the expected level, `.probabilities` per level |

`criteria` are optional for yes/no questions; write them when "yes" needs defining. For a `Choice`,
describe every option, because the description is what the model judges by. Many questions about
one text go in a single call and run in parallel, and the text can be a string or any
JSON-serialisable object. Questions in the TypeSafe (Jev) API shape — plain dicts with a `type` key —
are accepted too, so code written for its Python SDK moves over by changing the import and the
client.

Every answer carries `meta`, and `meta.answered` is the field to act on: it is False when the
model refused to decide, and the probabilities of such an answer mean nothing. A typical handler
acts on confident answers and passes the rest on:

```python
a = r.choices["team"]
if a.meta.answered and a.confidence >= 0.9:
    route(message, a.choice)
else:
    send_to_person(message)       # undecided or unsure
```

With `DecisionClient(..., strict=True)` an undecided answer raises `UnreadableAnswer` instead. The
other `meta` fields say how the answer was obtained: `logprobs` (False for text answers, whose
probability is always 1), `calibrated`, `checked`, `bounded` (options too unlikely for the provider
to report), `requests` and `reasoning_tokens`. `r.to_dict()` gives the whole response as plain data.

## 6. Tune

```python
DecisionClient(model, *, provider=None, base_url=None, api_key=None, key_env=None, env_file=None,
               timeout=120.0, retries=6, workers=8, calibrated=True, strict=False)
```

`workers` is how many questions of one call go out in parallel. `timeout` and `retries` apply per
request: rate limits, server errors and dropped connections are retried with backoff, while a
refused request or an exhausted balance fails at once. `calibrated=False` returns the model's own
probabilities.

A single call can override the model, the timeout and the request:

```python
client.system_one(text, questions,
                  model="qwen3-235b",                 # same provider and key
                  timeout=10,
                  extra_headers={"X-Trace": "…"},
                  extra_body={"priority": 1})         # merged into the request

from llm2decision import AsyncDecisionClient
async with AsyncDecisionClient("qwen3.8-27b@nebius") as client:
    r = await client.system_one(text, questions)
```

All errors derive from `LLM2DecisionError`: `QuestionError` names a malformed question before any
request is sent, `AuthError` and `ConfigError` come from the setup, `TransportError` means the
provider refused or kept failing, and `UnreadableAnswer` is raised in strict mode.

## 7. Your own models

Files named `models.json`, `providers.json` and `forms.json` in `$LLM2DECISION_HOME` (default
`~/.config/llm2decision`) are read on top of the package's own, field by field. A model of your own
gets a short name, and a model that answers in another language learns its words for yes and no:

```json
{
  "my-qwen@local": {"provider": "local", "model": "Qwen/Qwen3-8B"},
  "my-french@local": {"provider": "local", "model": "…", "forms": {"true": ["oui"], "false": ["non"]}}
}
```

`llm2decision models` (or `list_models()`) lists every model the package knows, yours included.

## Limits

- Version 1 answers without reasoning, so models that cannot switch reasoning off, such as
  Gemini 3, are not supported.
- OpenAI, Anthropic, Mistral and Gemini through OpenRouter answer as text, without a confidence.
- Text only: images are not supported yet.
- Calibration was done on English decision questions and may be off on very different material.

## License

Apache-2.0
