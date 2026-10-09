# llm2decision

[![tests](https://github.com/mihail-gribov/llm2decision/actions/workflows/tests.yml/badge.svg)](https://github.com/mihail-gribov/llm2decision/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/llm2decision)](https://pypi.org/project/llm2decision/)
[![Python](https://img.shields.io/pypi/pyversions/llm2decision)](https://pypi.org/project/llm2decision/)
[![required dependencies](https://img.shields.io/badge/required%20dependencies-0-brightgreen)](https://github.com/mihail-gribov/llm2decision/blob/main/pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/mihail-gribov/llm2decision/blob/main/LICENSE)

**Turn any LLM into a decision model.** Ask a question about a text and get a probability for
each predefined option, from the model you already use, hosted or on your own server.

```
Text:      I love the headphones, but the left ear cushion already came off after a week.
Question:  Is the customer complaining?

Qwen3.8-27B in a chat:
  Yes, the customer is complaining. While they start with a positive statement ("I love the
  headphones"), the core of their message is reporting a significant defect or failure of the
  product (the left ear cushion detaching after just one week). This indicates dissatisfaction
  with the quality, durability, or performance of the item, which constitutes a complaint that
  likely requires customer service intervention, such as a repair or replacement.

The same model through llm2decision:
  0.96   (the probability of "yes")
```

- **Your choice of model.** Use the models you already have access to, or the one that does best
  on your tasks: large or small, through an API (Nebius, OpenAI, Anthropic, OpenRouter, Mistral)
  or on your own server (vLLM, SGLang, llama.cpp, Ollama), or TypeSafe's Jev decision model. The
  code stays the same, and switching is one string. Code written for the Jev SDK moves over by
  changing the import.
- **Chosen on your own questions.** `llm2decision bench` runs a file of your questions through any
  model and reports accuracy, undecided answers and calibration, so you compare candidates on
  your task.
- **Probabilities you can set thresholds on.** Where the provider returns token probabilities, they
  are calibrated per model, so an answer given with 0.9 is right about 90% of the time. Act on
  confident answers and send the rest to a person.
- **Nothing to parse.** The answer is always one of your options, or explicitly marked undecided.
- **Measured.** Every tested model has its accuracy, undecided share and calibration error on the
  same 3 358 questions (section 4), a common reference for your candidates.
- **Zero required dependencies**, only the standard library of Python 3.10+.

## Quick start

```
pip install llm2decision
export NEBIUS_API_KEY=...
```

```python
import llm2decision as l2d

client = l2d.DecisionClient("qwen3.8-27b@nebius")

document = "I love the headphones, but the left ear cushion already came off after a week."
questions = {"complaint": l2d.Noul(instructions="Is the customer complaining?")}
answers = client.system_one(document, questions)

print(answers.nouls["complaint"].noul)   # the probability of "yes": 0.96
```

`Noul` is a yes/no question; section 3 lists the other types. The numbers in the examples come
from real runs: a rerun can give slightly different probabilities, and another model gives other
numbers.

## 1. Connect

Name a tested model or a provider and any of its models. `qwen3.8-27b@nebius`, used in the
examples, and `gpt-5.4@openai` are good to start with; section 4 compares all 21.

```python
import llm2decision as l2d

l2d.DecisionClient("qwen3.8-27b@nebius")                                   # a tested model (section 4)
l2d.DecisionClient("qwen3.8-27b")                                          # the same, when the name is unique
l2d.DecisionClient("google/gemini-2.5-flash", provider="openrouter")       # any model of a provider
l2d.DecisionClient("Qwen/Qwen3-8B", base_url="http://localhost:8000/v1")   # your own server, no key
l2d.DecisionClient("jev-1.13.0@typesafe")                                  # TypeSafe's Jev
```

| provider | `provider=` | key variable | answers |
|---|---|---|---|
| your own server (vLLM, SGLang, llama.cpp, Ollama) | `"local"`, or just `base_url=` | — | with probabilities |
| Nebius | `"nebius"` | `NEBIUS_API_KEY` | with probabilities |
| OpenRouter | `"openrouter"` | `OPENROUTER_API_KEY` | with probabilities where the model gives them |
| OpenAI | `"openai"` | `OPENAI_API_KEY` | text |
| Anthropic | `"anthropic"` | `ANTHROPIC_API_KEY` | text |
| Mistral | `"mistral"` | `MISTRAL_API_KEY` | text |
| TypeSafe (Jev) | `"typesafe"` | `TYPESAFE_API_KEY` | with Jev's own probabilities; all questions of a call in one request |

Keys: the provider's variable, `api_key=` (a string or a function), `key_env=` or `env_file=`.

## 2. Check the model

Before relying on a model that is not in the table of section 4, run the check with the same
arguments you connect with — a provider, or the `base_url` of your own server:

```
llm2decision check google/gemini-2.5-flash --provider openrouter
llm2decision check Qwen/Qwen3-8B --base-url http://localhost:8000/v1
```

It sends about a dozen easy questions and reports, point by point, whether the model is reachable,
answers in the expected form, gets the easy questions right, returns probabilities and keeps
reasoning off. A failed point names the reason. `client.check()` does the same in code, and
`--save` remembers a pass so that later answers carry `meta.checked`.

Then try the model on a few dozen questions of your own and look at `meta.answered`: a model that
passes the easy questions can still refuse to decide on hard ones.

To compare a model with the table of section 4, measure it on the same
[Decision Questions](https://huggingface.co/datasets/mihailgribov/decision-questions) set:

```
llm2decision bench Qwen/Qwen3-8B --base-url http://localhost:8000/v1
```

It asks the 3 358 questions one by one (on a hosted model, at your provider's prices; `--limit`
takes the first N), saves the answers as they come, continues where it stopped when run again, and
prints accuracy by question type, the share of undecided questions and the calibration error.
`--data my_questions.jsonl` runs your own questions instead, written in the same format as
Decision Questions.

## 3. Ask questions

Several questions about one text go in a single call:

```python
import llm2decision as l2d

message = "Hi, I was charged twice for order #4411, please refund the extra payment."

with l2d.DecisionClient("qwen3.8-27b@nebius") as client:
    answers = client.system_one(
        message,
        {
            "refund": l2d.Noul(instructions="Does the customer ask for money back?"),
            "team": l2d.Choice(instructions="Which team should handle this?",
                               criteria={"billing": "Payments and refunds", "shipping": "Delivery",
                                         "tech": "Bugs and errors"}),
        },
    )

answers.nouls["refund"].noul            # 0.998 — P(yes)
answers.choices["team"].choice          # "billing"
answers.choices["team"].probabilities   # {"billing": 0.999, "shipping": 0.0, "tech": 0.001}
```

| type | question | use it when | answer |
|---|---|---|---|
| `l2d.Noul(instructions=…, criteria={"true": …, "false": …})` | yes / no | the text settles it either way | `.noul` — P(yes) |
| `l2d.Tfu(instructions=…, criteria={"true": …, "false": …, "unknown": …})` | **t**rue / **f**alse / **u**nknown | the text may not settle it, and "unknown" should be an answer rather than a guess | `.tfu` — `"true"`, `"false"` or `"unknown"`, with `.probabilities` |
| `l2d.Choice(instructions=…, criteria={label: description, …})` | choice | one of several named options | `.choice`, `.confidence`, `.probabilities` |
| `l2d.Score(instructions=…, criteria=[level 0, level 1, …])` | score | a level on an ordered scale | `.score` — the expected level, `.probabilities` per level |

The answers come back grouped by type, under the names you gave the questions: `answers.nouls`,
`.tfus`, `.choices` and `.scores`.

When the text may not settle the question, `Tfu` makes "unknown" an answer of its own:

```python
import llm2decision as l2d

client = l2d.DecisionClient("qwen3.8-27b@nebius")
document = "The parcel came a week late and the box was dented, but everything inside works fine."
questions = {"satisfied": l2d.Tfu(instructions="Is the customer satisfied with the order?")}

client.system_one(document, questions).tfus["satisfied"].probabilities
# {"true": 0.065, "false": 0.086, "unknown": 0.849}
```

`criteria` are optional for yes/no questions; write them when "yes" needs defining. For a `Choice`,
describe every option, because the description is what the model judges by. The questions of a
call run in parallel, and the text can be a string or any JSON-serialisable object.

Every answer carries `meta`, and `meta.answered` is the field to act on: it is False when the
model refused to decide, and the probabilities of such an answer mean nothing. A typical handler
acts on confident answers and passes the rest on:

```python
answer = answers.choices["team"]
if answer.meta.answered and answer.confidence >= 0.9:
    route(message, answer.choice)
else:
    send_to_person(message)       # undecided or unsure
```

With `strict=True` an undecided answer raises `l2d.UnreadableAnswer` instead. The other `meta`
fields say how the answer was obtained:

| field | meaning |
|---|---|
| `logprobs` | True: read from the model's token probabilities; False: from the text it wrote, so the probabilities are 1 and 0 and a threshold does nothing |
| `calibrated` | the probabilities were calibrated for this model (section 5) |
| `checked` | the model passed `check()` and the pass was saved (section 2) |

`answers.to_dict()` gives the whole response as plain data.

## 4. Choose a model

The models tested with the package, by accuracy on Decision Questions:

| model | Decision Questions | JevBench | undecided | calibration error |
|---|---|---|---|---|
| `claude-sonnet-5.5@anthropic` | **0.948** | 0.954 | 5.7% | no probabilities |
| `gpt-5.4@openai` | **0.936** | 0.896 | 1.4% | no probabilities |
| `kimi-k2.6@nebius` | **0.917** | 0.882 | 1.1% | 0.017 |
| `claude-haiku-5.5@anthropic` | **0.915** | 0.862 | 5.7% | no probabilities |
| `qwen3.8-27b@nebius` | **0.913** | 0.885 | 0.8% | 0.023 |
| `minimax-m3@nebius` | **0.912** | 0.850 | 1.7% | 0.024 |
| `deepseek-v4-pro@nebius` | **0.912** | 0.864 | 1.5% | 0.021 |
| `qwen3.5-397b@nebius` | **0.907** | 0.860 | 0.8% | 0.024 |
| `claude-haiku-4.5@anthropic` | **0.905** | 0.855 | 1.7% | no probabilities |
| `nemotron-3-ultra@nebius` | **0.901** | 0.855 | 0.8% | 0.020 |
| `gemini-2.5-flash@openrouter` | **0.896** | 0.878 | 1.0% | no probabilities |
| `hermes-4-405b@nebius` | **0.888** | 0.857 | 0.4% | 0.022 |
| `qwen3-235b@nebius` | **0.884** | 0.838 | 0.8% | 0.024 |
| `gpt-5.4-mini@openai` | **0.875** | 0.840 | 0.5% | no probabilities |
| `nemotron-3-super@nebius` | **0.842** | 0.804 | 0.2% | 0.028 |
| `ministral-14b@mistral` | **0.836** | 0.830 | 0.7% | no probabilities |
| `qwen3-30b-a3b@nebius` | **0.833** | 0.800 | 0.5% | 0.034 |
| `gpt-oss-120b@nebius` | **0.830** | 0.779 | 0.7% | 0.031 |
| `gemma-3-27b@nebius` | **0.818** | 0.730 | 0.4% | 0.064 |
| `nemotron-3-nano@nebius` | **0.756** | 0.709 | 1.1% | 0.051 |
| `nemotron-3.5-lightning@nebius` | **0.734** | 0.753 | 0.1% | 0.051 |

- **Decision Questions** — accuracy on
  [Decision Questions](https://huggingface.co/datasets/mihailgribov/decision-questions), 3 358
  questions across many domains; `llm2decision bench` measures it for any model (section 2), and
  [`benchmark/`](https://github.com/mihail-gribov/llm2decision/tree/main/benchmark) has every
  model's answers.
- **JevBench** — accuracy on the 231 public tasks of
  [JevBench](https://github.com/fstandhartinger/jevbench), not its own score; differences under
  ~5 points are noise.
- **undecided** — share of questions the model refused to decide. Accuracy leaves them out, so read
  the two columns together.
- **calibration error** — how far the stated probabilities are from how often the model is right
  (lower is better). *no probabilities*: the provider answers in text.

How to pick:

- **The most accurate:** `claude-sonnet-5.5`, but it refuses to decide about 6% of questions, so
  you need a fallback for those; `gpt-5.4` decides nearly everything and comes next.
- **A confidence you can set thresholds on:** a model with a low calibration error. `qwen3.8-27b`,
  `kimi-k2.6` and `qwen3.5-397b` are within 3 points of `gpt-5.4`.
- **Volume at a lower price:** `qwen3-30b-a3b` (0.833) and `gpt-oss-120b` (0.830) are calibrated and
  cost $0.10 and $0.15 per million input tokens on Nebius, against $0.45 for `qwen3.8-27b`.
- **The lowest price, when accuracy around 0.75 is enough:** `nemotron-3-nano` and
  `nemotron-3.5-lightning`, $0.06 per million input tokens on Nebius.
- **Data that must not leave your hardware:** any model on your own OpenAI-compatible server,
  checked as in section 2.

Prices are Nebius list prices of October 2026.

Whatever the model, yes/no/unknown and rubric scores are the hardest question types (`bench`
prints accuracy per type), so where a decision matters, a yes/no or a choice question gets better
answers.

## 5. Tune

```python
l2d.DecisionClient(model, *, provider=None, base_url=None, api_key=None, key_env=None, env_file=None,
                   timeout=120.0, retries=6, workers=8, calibrated=True, strict=False)
```

`workers` is how many questions of one call go out in parallel. `timeout` and `retries` apply per
request: rate limits, server errors and dropped connections are retried with backoff, while a
refused request or an exhausted balance fails at once. The connection arguments (`provider`,
`base_url`, `api_key`, `key_env`, `env_file`) are described in section 1, and `strict` in section 3.

By default (`calibrated=True`) the probabilities of a model from the table are calibrated: the
model's own ones are rescaled so that an answer given with 0.8 is right about 80% of the time. The
rescaling makes them sharper or softer but never changes which answer is the most likely. A model
without a calibration (your own, or one that answers in text) is returned as it is, and
`meta.calibrated` says which case you got. `calibrated=False` returns the model's own probabilities.

A single call can override the model, the timeout and the request:

```python
client.system_one(text, questions,
                  model="qwen3-235b",                 # same provider and key
                  timeout=10,
                  extra_headers={"X-Trace": "…"},
                  extra_body={"priority": 1})         # merged into the request

async with l2d.AsyncDecisionClient("qwen3.8-27b@nebius") as client:
    answers = await client.system_one(text, questions)
```

All errors derive from `l2d.LLM2DecisionError`: `QuestionError` names a malformed question before
any request is sent, `AuthError` and `ConfigError` come from the setup, `TransportError` means the
provider refused or kept failing, and `UnreadableAnswer` means the model refused to decide
(`meta.answered` is False) and is raised only with `strict=True`.

## 6. Your own models

Files named `models.json`, `providers.json` and `forms.json` in `$LLM2DECISION_HOME` (default
`~/.config/llm2decision`) are read on top of the package's own, field by field. A model of your own
gets a short name, and a model that answers in another language learns its words for yes and no:

```json
{
  "my-qwen@local": {"provider": "local", "model": "Qwen/Qwen3-8B"},
  "my-french@local": {"provider": "local", "model": "…", "forms": {"true": ["oui"], "false": ["non"]}}
}
```

`llm2decision models` and `l2d.list_models()` include your models.

Some servers cannot continue an answer that the package has started. For your own model on such a
server, the package renders the model's chat template itself, which needs the only optional extra:
`pip install "llm2decision[template]"` (transformers and jinja2).

## Limits

- This version answers without reasoning, so models that cannot switch reasoning off, such as
  Gemini 3, are not supported.
- Text only: images are not supported yet.
- Calibration was done on English decision questions and may be off on very different material.

## License

Apache-2.0. The base install has no third-party dependencies to license.
