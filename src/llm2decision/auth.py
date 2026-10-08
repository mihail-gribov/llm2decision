"""Where a provider's key comes from and how it is sent — with nothing printed and nothing sent
where it does not belong.

Sources, first found wins:

  1. `key=` in code — a string, or a callable returning one (a keyring, a vault, a secrets manager)
  2. `key_env=` — the name of an environment variable chosen by the caller
  3. the provider's own variable from its configuration (`NEBIUS_API_KEY`, `OPENROUTER_API_KEY`, …)
  4. an env file — only when named: `env_file=` or `LLM2DECISION_ENV_FILE`; never picked up silently,
     and it never overrides a variable already set in the environment

A provider configured with `auth: none` (a vLLM, llama-server or Ollama of your own) needs no key.

A key found in the environment or an env file (3, 4) is sent only to the provider's own host: with
`base_url` pointed elsewhere it is withheld, and only an explicit `key=` or `key_env=` is sent there.
A typo in an address must not hand a key to a stranger.

How it is sent comes from the provider: `Authorization: Bearer <key>` (OpenAI-style), `x-api-key`
(Anthropic), `api-key` (Azure) or any header with an optional scheme.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping
from urllib.parse import urlparse

ENV_FILE_VAR = "LLM2DECISION_ENV_FILE"


from .errors import AuthError  # noqa: E402  re-exported


@dataclass(frozen=True)
class AuthSpec:
    """How a provider takes its key (from the provider's configuration)."""
    mode: str = "header"                 # "header" or "none"
    header: str = "Authorization"
    scheme: str = "Bearer"               # "" for a bare key, as `x-api-key` takes it
    env: str = ""                        # the provider's own variable, e.g. NEBIUS_API_KEY
    host: str = ""                       # the provider's own host: environment keys go only here

    @classmethod
    def from_config(cls, cfg: Mapping | None, base_url: str = "") -> "AuthSpec":
        cfg = dict(cfg or {})
        if cfg.get("mode") == "none" or cfg.get("type") == "none":
            return cls(mode="none", host=_host(base_url))
        return cls(mode="header", header=cfg.get("header", "Authorization"),
                   scheme=cfg.get("scheme", "Bearer"), env=cfg.get("env", ""),
                   host=cfg.get("host") or _host(base_url))


@dataclass(frozen=True)
class Credential:
    """A resolved key and where it came from. `repr` and `str` never show the key."""
    value: str | None
    source: str                          # "code", "env:NAME", "env_file:PATH:NAME", "none"
    explicit: bool                       # given by the caller (code, key_env) rather than found

    def __repr__(self) -> str:
        return f"Credential(source={self.source!r}, set={self.value is not None})"

    __str__ = __repr__


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def parse_env_file(path: str | Path) -> dict[str, str]:
    """`NAME=value` lines: `export ` allowed, `#` comments and blank lines skipped, one level of
    matching quotes removed, an inline ` #` comment cut from unquoted values."""
    out: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if name:
            out[name] = value
    return out


def resolve(spec: AuthSpec, *, key: str | Callable[[], str] | None = None, key_env: str | None = None,
            env_file: str | Path | None = None, base_url: str = "",
            environ: Mapping[str, str] | None = None) -> Credential:
    """The key for one provider, by the order in the module docstring."""
    environ = os.environ if environ is None else environ
    if spec.mode == "none" and key is None and key_env is None:
        return Credential(None, "none", False)
    if key is not None:
        value = key() if callable(key) else key
        if not isinstance(value, str) or not value.strip():
            raise AuthError("the key given in code is empty")
        return Credential(value.strip(), "code", True)
    if key_env:
        value = environ.get(key_env, "").strip()
        if not value:
            raise AuthError(f"key_env names {key_env!r}, which is not set or is empty")
        return Credential(value, f"env:{key_env}", True)
    looked = []
    if spec.env:
        looked.append(f"the environment variable {spec.env}")
        value = environ.get(spec.env, "").strip()
        if value:
            return _for_host(Credential(value, f"env:{spec.env}", False), spec, base_url)
    file = env_file or environ.get(ENV_FILE_VAR)
    if file and spec.env:
        looked.append(f"{spec.env} in the env file {file}")
        p = Path(file)
        if not p.is_file():
            raise AuthError(f"env file {file} does not exist")
        value = parse_env_file(p).get(spec.env, "").strip()
        if value:
            return _for_host(Credential(value, f"env_file:{file}:{spec.env}", False), spec, base_url)
    where = ", ".join(looked) or "nowhere: the provider names no variable"
    raise AuthError(f"no key for this provider; looked in {where}. Pass key=…, key_env=…, or "
                    f"env_file=… (or set {ENV_FILE_VAR})")


def _for_host(cred: Credential, spec: AuthSpec, base_url: str) -> Credential:
    """A key found rather than given goes only to the provider's own host."""
    target = _host(base_url) or spec.host
    if spec.host and target and target != spec.host:
        raise AuthError(f"a key from {cred.source.split(':')[0]} is meant for {spec.host}, not {target}; "
                        "pass key=… or key_env=… to send a key to another address")
    return cred


def headers(spec: AuthSpec, cred: Credential) -> dict[str, str]:
    """The request headers that carry the key."""
    if cred.value is None:
        return {}
    return {spec.header: f"{spec.scheme} {cred.value}" if spec.scheme else cred.value}
