"""`python -m llm2decision` (or `llm2decision`): list the known models, or check one.

    llm2decision models [--provider nebius]
    llm2decision check qwen3.8-27b@nebius [--save]
    llm2decision check Qwen/Qwen3-8B --base-url http://localhost:8000/v1
"""
from __future__ import annotations

import argparse
import sys

from .client import DecisionClient
from .config import list_models
from .errors import LLM2DecisionError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="llm2decision")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("models", help="the known bindings")
    m.add_argument("--provider")
    c = sub.add_parser("check", help="probe a model through its provider")
    c.add_argument("model")
    c.add_argument("--provider")
    c.add_argument("--base-url")
    c.add_argument("--key-env", help="the environment variable holding the key")
    c.add_argument("--env-file")
    c.add_argument("--save", action="store_true", help="record a pass in the user's models.json")
    a = ap.parse_args(argv)
    if a.cmd == "models":
        for x in list_models(a.provider):
            cal = ",".join(x.calibrated) or "-"
            print(f"{x.name:<34} {x.transport:<14} calibrated={cal:<34} checked={x.checked or '-'}")
        return 0
    try:
        with DecisionClient(a.model, provider=a.provider, base_url=a.base_url, key_env=a.key_env,
                            env_file=a.env_file) as client:
            rep = client.check(save=a.save)
    except LLM2DecisionError as e:
        print(f"llm2decision: {e}", file=sys.stderr)
        return 2
    print(rep)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
