"""Which model, through which provider, sent how — from the packaged configuration and the caller.

  providers.json   a service: address, how the key is sent, the transport, how reasoning is
                   switched off, its top-k, the key prefixes that name it
  models.json      bindings `name@provider`: the provider's model id and what is known about the pair —
                   a transport other than the provider's, how its tokenizer cuts digits, fitted
                   temperatures, extra answer forms, request fields of its own (`extra`, over the
                   provider's), the date `check()` last passed
  forms.json       the words that count as each side of a yes/no answer (`true`, `false`, `unsure`)

A user directory (`$LLM2DECISION_HOME`, else `~/.config/llm2decision`) may hold its own copies of
these files, read on top of the packaged ones field by field: a binding that adds `calibration`
keeps its packaged model id and transport, `{"calibration": {"choice": 1.1}}` changes one
temperature and keeps the rest, `null` removes a field. Answer forms only add: a user's
`forms.json` and a binding's `forms` extend the packaged lists, so a model that answers `[Si]` is
taught that by one line.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any


from .errors import ConfigError  # noqa: E402  re-exported


MODES = ("verdict", "three_answers", "choice", "scale")      # what a temperature is fitted for


class LLM2DecisionWarning(UserWarning):
    """Something was resolved differently from how it was asked for; the message says which way."""


def merge(base: dict, over: dict) -> dict:
    """`over` on top of `base`: objects merge key by key, `null` removes, anything else replaces."""
    out = dict(base)
    for k, v in over.items():
        if v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def home() -> Path:
    return Path(os.environ.get("LLM2DECISION_HOME", Path.home() / ".config" / "llm2decision"))


def _load(name: str) -> dict:
    data = json.loads(resources.files(__package__).joinpath("configs", name).read_text(encoding="utf-8"))
    user = home() / name
    if user.is_file():
        try:
            mine = json.loads(user.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ConfigError(f"{user}: {e}") from None
        if not isinstance(mine, dict):
            raise ConfigError(f"{user}: expected an object of entries by name")
        if name == "forms.json":
            return add_forms(data, mine)
        data = merge(data, mine)
    return data


def add_forms(base: dict, more: dict | None) -> dict[str, list[str]]:
    """Answer forms: `more` only adds words to a side, lower-cased; an unknown side is an error."""
    out = {s: list(base.get(s, [])) for s in ("true", "false", "unsure")}
    for side, words in (more or {}).items():
        if side not in out or not isinstance(words, list):
            raise ConfigError(f"answer forms are lists under 'true', 'false', 'unsure'; got {side!r}")
        out[side] += [w.strip().lower() for w in words if w.strip().lower() not in out[side]]
    return out


def forms() -> dict[str, list[str]]:
    return _load("forms.json")


def providers() -> dict[str, dict]:
    return _load("providers.json")


def bindings() -> dict[str, dict]:
    return _load("models.json")


@dataclass
class Binding:
    """Everything needed to talk to one model."""
    name: str
    provider: str
    model: str
    base_url: str
    auth: dict
    transport: str
    top_k: int = 20
    extra: dict = field(default_factory=dict)
    max_tokens_field: str = "max_tokens"
    digits: str | None = None
    calibration: dict[str, float] = field(default_factory=dict)
    fitted: str | None = None                                    # where the temperatures come from
    checked: str | None = None                                   # the date `check()` last passed
    measured: str | None = None                                  # the date of a measurement (not a check)
    forms: dict[str, list[str]] = field(default_factory=dict)    # yes/no forms, packaged plus the binding's
    options: dict[str, Any] = field(default_factory=dict)       # transport options: tokenizer, suffix, …
    logprobs: bool = True


def provider_from_key(key: str) -> str:
    """The provider a key's prefix names, if exactly one does; never a guess."""
    hits = [name for name, p in providers().items()
            if any(key.startswith(pre) for pre in p.get("key_prefixes", []))]
    if len(hits) != 1:
        raise ConfigError("the key's prefix does not name one provider; pass provider=… "
                          f"(known: {', '.join(sorted(providers()))})")
    return hits[0]


def resolve(model: str, provider: str | None = None, base_url: str | None = None,
            api_key: str | None = None) -> Binding:
    """A binding by name (`qwen3.8-27b@nebius`, or `qwen3.8-27b` with `provider=`), else the
    provider's defaults with `model` as the provider's model id."""
    known = bindings()
    if "@" in model and provider and model.split("@", 1)[1] != provider:
        # the name says one provider, `provider=` another: the argument wins when the same model has a
        # binding there; otherwise the named binding stays, since its model id means nothing elsewhere
        family, named = model.split("@", 1)
        other = f"{family}@{provider}"
        if other in known:
            warnings.warn(f"provider={provider!r} overrides the provider in {model!r}: using binding {other!r}",
                          LLM2DecisionWarning, stacklevel=3)
            model = other
        else:
            warnings.warn(f"provider={provider!r} ignored: {model!r} names {named!r}, and there is no binding "
                          f"{other!r}", LLM2DecisionWarning, stacklevel=3)
            provider = None
    if "@" not in model and provider is None and not base_url:
        # a short name: the binding of that model, when only one provider has it (or the key names one)
        same = sorted(k for k in known if k.split("@", 1)[0] == model)
        if len(same) > 1 and api_key:
            try:
                same = [k for k in same if k.endswith("@" + provider_from_key(api_key))] or same
            except ConfigError:
                pass
        if len(same) == 1:
            model = same[0]
        elif same:
            raise ConfigError(f"{model!r} is bound at several providers: {', '.join(same)}; name one")
    name = model if "@" in model else (f"{model}@{provider}" if provider else None)
    if name and name in known:
        b = dict(known[name])
        provider = b.pop("provider")
    else:
        if "@" in model:
            raise ConfigError(f"no binding {model!r}; known: {', '.join(sorted(known))}")
        b = {"model": model}
        if provider is None:
            if base_url:
                provider = "local"
            elif api_key:
                provider = provider_from_key(api_key)
            else:
                raise ConfigError(f"{model!r} is not a known binding; pass provider=… or base_url=…")
    if provider not in providers():
        raise ConfigError(f"unknown provider {provider!r}; known: {', '.join(sorted(providers()))}")
    p = providers()[provider]
    options = {k: b.pop(k) for k in ("tokenizer", "template_kwargs", "suffix", "chat_template", "prefill", "temperature",
                                     "ask_format") if k in b}
    b.pop("note", None)
    calibration = dict(b.get("calibration") or {})
    fitted = calibration.pop("fitted", None)
    bad = {k: v for k, v in calibration.items() if k not in MODES or not isinstance(v, (int, float)) or v <= 0}
    if bad:
        raise ConfigError(f"{model}: calibration takes positive temperatures for {', '.join(MODES)}; got {bad}")
    return Binding(
        name=name if name in known else f"{model}@{provider}", provider=provider, model=b["model"],
        base_url=base_url or p["base_url"], auth=dict(p.get("auth", {})),
        transport=b.get("transport", p["transport"]), top_k=p.get("top_k", 20),
        extra=merge(dict(p.get("extra", {})), dict(b.get("extra") or {})),     # the binding's fields over the provider's
        max_tokens_field=p.get("max_tokens_field", "max_tokens"), digits=b.get("digits"),
        calibration=calibration, fitted=fitted, checked=b.get("checked"), measured=b.get("measured"),
        forms=add_forms(forms(), b.get("forms")), options=options, logprobs=b.get("logprobs", True))


@dataclass(frozen=True)
class ModelInfo:
    """One known binding, as `list_models()` shows it."""
    name: str
    provider: str
    model: str
    transport: str
    calibrated: tuple[str, ...]          # the modes with a fitted temperature
    logprobs: bool
    checked: str | None
    measured: str | None
    note: str = ""


def list_models(provider: str | None = None) -> list[ModelInfo]:
    """The known bindings (packaged and the user's), optionally of one provider. Any other model of a
    known provider works too, by its id with `provider=`; a binding only says what is known about it."""
    ps, out = providers(), []
    for name, b in sorted(bindings().items()):
        if provider and b.get("provider") != provider:
            continue
        cal = tuple(m for m in MODES if m in (b.get("calibration") or {}))
        out.append(ModelInfo(name, b["provider"], b["model"],
                             b.get("transport") or ps.get(b["provider"], {}).get("transport", ""), cal,
                             b.get("logprobs", True), b.get("checked"), b.get("measured"), b.get("note", "")))
    return out
