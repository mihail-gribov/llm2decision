"""`python -m llm2decision` (or `llm2decision`): list the known models, check one, or measure it.

    llm2decision models [--provider nebius]
    llm2decision check qwen3.8-27b@nebius [--save]
    llm2decision check Qwen/Qwen3-8B --base-url http://localhost:8000/v1
    llm2decision bench qwen3.8-27b@nebius [--data test.jsonl] [--limit 200] [--out answers.jsonl]
"""
from __future__ import annotations

import argparse
import re
import sys

from . import bench
from .client import DecisionClient
from .config import list_models
from .errors import LLM2DecisionError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="llm2decision")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("models", help="the known bindings")
    m.add_argument("--provider")
    c = sub.add_parser("check", help="probe a model through its provider")
    b = sub.add_parser("bench", help="measure a model on Decision Questions")
    for p in (c, b):
        p.add_argument("model")
        p.add_argument("--provider")
        p.add_argument("--base-url")
        p.add_argument("--key-env", help="the environment variable holding the key")
        p.add_argument("--env-file")
    c.add_argument("--save", action="store_true", help="record a pass in the user's models.json")
    b.add_argument("--data", help="a Decision Questions file (default: the published set, downloaded once)")
    b.add_argument("--out", help="where answers are kept and resumed from (default: bench_<model>.jsonl)")
    b.add_argument("--limit", type=int, help="the first N questions only")
    b.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    if a.cmd == "models":
        for x in list_models(a.provider):
            cal = ",".join(x.calibrated) or "-"
            print(f"{x.name:<34} {x.transport:<14} calibrated={cal:<34} checked={x.checked or '-'}")
        return 0
    try:
        with DecisionClient(a.model, provider=a.provider, base_url=a.base_url, key_env=a.key_env,
                            env_file=a.env_file, **({"workers": 1} if a.cmd == "bench" else {})) as client:
            if a.cmd == "check":
                rep = client.check(save=a.save)
                print(rep)
                return 0 if rep.ok else 1
            items = bench.load(a.data)[: a.limit]
            out = a.out or f"bench_{re.sub(r'[^A-Za-z0-9._@-]+', '_', a.model)}.jsonl"
            rows = bench.run(client, items, out, workers=a.workers,
                             progress=lambda s: print(s, file=sys.stderr, flush=True))
            print(bench.report(rows))
            return 0 if len(rows) == len(items) else 1
    except LLM2DecisionError as e:
        print(f"llm2decision: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
