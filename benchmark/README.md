# Benchmark

The answers behind the model table in the [package README](https://github.com/mihail-gribov/llm2decision#readme), on
[Decision Questions](https://huggingface.co/datasets/mihailgribov/decision-questions), and an item
response theory fit over them. Besides the 21 models run through llm2decision, the benchmark includes
[Jev](https://typesafe.ai) 1.13.0, TypeSafe's decision model, asked through its own API.

## Add a model

```
llm2decision bench Qwen/Qwen3-8B --base-url http://localhost:8000/v1 --out qwen3-8b.jsonl
uv run --with numpy --with scipy python benchmark/irt.py --score qwen3-8b.jsonl
```

The first command asks a model the whole set, here one on your own server (any model the package
connects to works the same way); the second, from the repository root, places it on the same scale
as the table below without moving anything in it.

## Abilities

Ability comes from a two-parameter item response model fitted on the answers below. The scale is
set by fixed anchor questions (`anchors.json`): their difficulty has mean 0 and spread 1, so adding a
model does not move it. ± is the standard deviation over resamples of the questions.

An answer is right when its most probable option is the correct one. A reply that is none of the
allowed answers counts as undecided; both ability and accuracy, over all 3 358 questions, leave
undecided questions out.

| system | ability | accuracy | undecided | measured through |
|---|---|---|---|---|
| `qwen3.8-27b@nebius` | +5.39 ± 0.18 | 0.913 | 0.8% | llm2decision |
| `gpt-5.4@openai` | +5.26 ± 0.15 | 0.936 | 1.4% | llm2decision |
| `kimi-k2.6@nebius` | +5.24 ± 0.17 | 0.917 | 1.1% | llm2decision |
| `qwen3.5-397b@nebius` | +5.23 ± 0.20 | 0.907 | 0.8% | llm2decision |
| `deepseek-v4-pro@nebius` | +5.04 ± 0.19 | 0.912 | 1.5% | llm2decision |
| `claude-haiku-4.5@anthropic` | +4.82 ± 0.15 | 0.905 | 1.7% | llm2decision |
| `claude-sonnet-5.5@anthropic` | +4.71 ± 0.13 | 0.948 | 5.7% | llm2decision |
| `claude-haiku-5.5@anthropic` | +4.68 ± 0.12 | 0.915 | 5.7% | llm2decision |
| `jev-1.13.0` | +4.66 ± 0.15 | 0.920 | 0.0% | its own API |
| `nemotron-3-ultra@nebius` | +4.63 ± 0.16 | 0.901 | 0.8% | llm2decision |
| `gemini-2.5-flash@openrouter` | +4.29 ± 0.12 | 0.896 | 1.0% | llm2decision |
| `minimax-m3@nebius` | +4.25 ± 0.12 | 0.912 | 1.7% | llm2decision |
| `hermes-4-405b@nebius` | +3.80 ± 0.10 | 0.888 | 0.4% | llm2decision |
| `qwen3-235b@nebius` | +3.61 ± 0.09 | 0.884 | 0.8% | llm2decision |
| `gpt-5.4-mini@openai` | +3.51 ± 0.10 | 0.875 | 0.5% | llm2decision |
| `nemotron-3-super@nebius` | +2.76 ± 0.08 | 0.842 | 0.2% | llm2decision |
| `ministral-14b@mistral` | +2.44 ± 0.07 | 0.836 | 0.7% | llm2decision |
| `gpt-oss-120b@nebius` | +2.36 ± 0.07 | 0.830 | 0.7% | llm2decision |
| `qwen3-30b-a3b@nebius` | +2.17 ± 0.07 | 0.833 | 0.5% | llm2decision |
| `gemma-3-27b@nebius` | +1.88 ± 0.06 | 0.818 | 0.4% | llm2decision |
| `nemotron-3.5-lightning@nebius` | +0.48 ± 0.06 | 0.734 | 0.1% | llm2decision |
| `nemotron-3-nano@nebius` | +0.46 ± 0.08 | 0.756 | 1.1% | llm2decision |
