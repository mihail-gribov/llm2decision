"""Item response theory on Decision Questions: a difficulty and a discrimination per question, fitted once
on the answers in `answers/` and then frozen, and one ability per system scored against them.

    uv run --with numpy --with scipy python benchmark/irt.py [--boot 200]      # fit the questions, score all
    uv run --with numpy --with scipy python benchmark/irt.py --score FILE      # score one more system

The model is 2PL: P(system i answers item j right) = sigmoid(a_j * (theta_i - b_j)). Questions a system
left undecided are missing, not wrong. Weak priors keep the fit finite: theta ~ N(0, 1), b ~ N(0, 2),
a ~ N(1, 1), with the sign of `a` left free — a question that stronger systems miss more often than
weaker ones gets a < 0.

The scale is tied to the questions, not to the systems. The anchor questions, listed in `anchors.json`,
are fixed: over them the difficulty has mean 0 and standard deviation 1 in every fit, so a refit with
more systems stays on the same scale. They were chosen once, at the first fit, as the questions that
separated the systems (some right, some wrong) with a difficulty between the 10th and 90th percentile. Ability 0 is the level at which a
question of the pool's average difficulty is answered right half the time, and one unit is the spread
of difficulty across the pool. Adding a system does not move the scale or the other systems: `--score`
uses the frozen questions.

A system's ability is its maximum-likelihood estimate on the separating questions with the question
parameters fixed; ± is the spread over resamples of those questions. A question every system answers
right (or wrong) says nothing about the systems; its estimates come from the priors, `status` marks it,
and it is not used for scoring.

Writes `items.jsonl` (per question) and prints the system table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit

HERE = Path(__file__).resolve().parent


def read(f: Path) -> list[dict]:
    return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]


def right(r: dict) -> float:
    return float(max(r["probs"], key=r["probs"].get) == r["answer"]) if r["answered"] else np.nan


def load(folder: Path):
    files = sorted(folder.glob("*.jsonl"))
    names = [f.stem for f in files]
    rows = [read(f) for f in files]
    ids = sorted({r["id"] for rs in rows for r in rs})
    col = {k: j for j, k in enumerate(ids)}
    kind = [""] * len(ids)
    Y = np.full((len(files), len(ids)), np.nan)
    for i, rs in enumerate(rows):
        for r in rs:
            kind[col[r["id"]]] = r["type"]
            Y[i, col[r["id"]]] = right(r)
    return names, ids, kind, Y


def fit(Y: np.ndarray):
    """Joint MAP fit; returns theta, b, a on the fit's own (arbitrary) scale."""
    n, m = Y.shape
    seen = ~np.isnan(Y)
    y = np.nan_to_num(Y)

    def loss(x):
        th, b, a = x[:n], x[n:n + m], x[n + m:]
        z = a[None, :] * (th[:, None] - b[None, :])
        nll = np.where(seen, y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z), 0.0).sum()
        f = nll + 0.5 * (th ** 2).sum() + 0.125 * (b ** 2).sum() + 0.5 * ((a - 1) ** 2).sum()
        g = np.where(seen, expit(z) - y, 0.0)
        dth = (g * a[None, :]).sum(1) + th
        db = -(g * a[None, :]).sum(0) + b / 4
        da = (g * (th[:, None] - b[None, :])).sum(0) + (a - 1)
        return f, np.concatenate([dth, db, da])

    x0 = np.concatenate([np.zeros(n), np.zeros(m), np.ones(m)])
    x = minimize(loss, x0, jac=True, method="L-BFGS-B", options={"maxiter": 5000}).x
    return x[:n], x[n:n + m], x[n + m:]


ANCHORS = HERE / "anchors.json"
TRIM = (10, 90)        # percentiles of difficulty that bounded the anchor questions when they were chosen


def anchors(ids: list[str], b: np.ndarray, sep: np.ndarray) -> np.ndarray:
    """The fixed anchor questions as a mask over `ids`; chosen and written once, when `anchors.json` is absent."""
    if not ANCHORS.exists():
        lo, hi = np.percentile(b[sep], TRIM)
        chosen = [k for k, s, x in zip(ids, sep, b) if s and lo <= x <= hi]
        ANCHORS.write_text(json.dumps({"rule": f"separating questions, difficulty between the {TRIM[0]}th and "
                                               f"{TRIM[1]}th percentile, at the first fit", "ids": chosen}, indent=0) + "\n")
    fixed = set(json.loads(ANCHORS.read_text())["ids"])
    pool = np.array([k in fixed for k in ids])
    assert pool.sum() == len(fixed), "anchors.json lists questions that are not in the answers"
    return pool


def anchor(b: np.ndarray, a: np.ndarray, pool: np.ndarray):
    """Shift and unit from the anchor questions: mean b = 0, sd b = 1 over them."""
    mu, sd = b[pool].mean(), b[pool].std()
    return (b - mu) / sd, a * sd


def score(y: np.ndarray, b: np.ndarray, a: np.ndarray) -> float:
    """Maximum-likelihood ability of one system on frozen questions (NaN = undecided)."""
    s = ~np.isnan(y)
    y, b, a = y[s], b[s], a[s]

    def nll(t):
        z = a * (t - b)
        return (y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z)).sum()

    return float(minimize_scalar(nll, bounds=(-15, 15), method="bounded").x)


def scored(y, b, a, boot: int, rng) -> tuple[float, float]:
    t = score(y, b, a)
    m = len(y)
    ts = [score(y[idx], b[idx], a[idx]) for idx in (rng.integers(0, m, m) for _ in range(boot))]
    return t, float(np.std(ts))


def frozen():
    items = read(HERE / "items.jsonl")
    sep = [x for x in items if x["status"] == "mixed"]
    return ({x["id"]: j for j, x in enumerate(sep)}, np.array([x["difficulty"] for x in sep]),
            np.array([x["discrimination"] for x in sep]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default=str(HERE / "answers"))
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--score", help="an answer file to score against the frozen questions in items.jsonl")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    if args.score:
        col, b, a = frozen()
        y = np.full(len(col), np.nan)
        rows = read(Path(args.score))
        for r in rows:
            if r["id"] in col:
                y[col[r["id"]]] = right(r)
        t, se = scored(y, b, a, args.boot, rng)
        every = np.array([right(r) for r in rows])
        print(f"{Path(args.score).stem}: ability {t:+.2f} ± {se:.2f}, accuracy {np.nanmean(every):.3f}, "
              f"undecided {np.isnan(every).mean():.1%} of {len(rows)} questions")
        if (~np.isnan(y)).sum() < 300:
            print(f"only {int((~np.isnan(y)).sum())} of the {len(col)} separating questions are answered: the ability "
                  f"is rough; run the whole set for a number comparable with the table")
        return

    names, ids, kind, Y = load(Path(args.answers))
    n, m = Y.shape
    seen = ~np.isnan(Y)
    hits = np.nansum(Y, 0)
    status = np.where(hits == seen.sum(0), "all right", np.where(hits == 0, "all wrong", "mixed"))
    sep = status == "mixed"
    _, b, a = fit(Y)
    pool = anchors(ids, b, sep)
    b, a = anchor(b, a, pool)

    B, A = [], []                                         # question uncertainty: refits on resampled systems
    for k in range(args.boot):
        idx = rng.integers(0, n, n)
        _, b_, a_ = fit(Y[idx])
        b_, a_ = anchor(b_, a_, pool)
        B.append(b_), A.append(a_)
        if (k + 1) % 50 == 0:
            print(f"  refit {k + 1}/{args.boot}", flush=True)
    b_se, a_se = np.array(B).std(0), np.array(A).std(0)

    with open(HERE / "items.jsonl", "w", encoding="utf-8") as f:
        for j in range(m):
            f.write(json.dumps({"id": ids[j], "type": kind[j], "status": str(status[j]), "anchor": bool(pool[j]),
                                "models_right": int(hits[j]), "models_answered": int(seen[:, j].sum()),
                                "difficulty": round(float(b[j]), 2), "difficulty_se": round(float(b_se[j]), 2),
                                "discrimination": round(float(a[j]), 2),
                                "discrimination_se": round(float(a_se[j]), 2)}) + "\n")

    col, bf, af = frozen()                                # score everyone exactly as a new system is scored
    js = [ids.index(k) for k in col]
    res = [scored(Y[i, js], bf, af, args.boot, rng) for i in range(n)]
    acc = np.nanmean(Y, 1)
    print("\n| model | ability | accuracy |\n|---|---|---|")
    for i in sorted(range(n), key=lambda i: -res[i][0]):
        print(f"| `{names[i]}` | {res[i][0]:+.2f} ± {res[i][1]:.2f} | {acc[i]:.3f} |")
    print(f"\nquestions: {int((status == 'all right').sum())} all right, {int((status == 'all wrong').sum())} "
          f"all wrong, {int(sep.sum())} separating, {int(pool.sum())} anchors")
    print("\n| type | separating | median difficulty | median discrimination | a < 0 beyond ± |\n|---|---|---|---|---|")
    for t in ("noul", "tfu", "choice", "score"):
        s = sep & (np.array(kind) == t)
        print(f"| {t} | {int(s.sum())} | {np.median(b[s]):+.2f} | {np.median(a[s]):.2f} | {int((s & (a + a_se < 0)).sum())} |")


if __name__ == "__main__":
    main()
