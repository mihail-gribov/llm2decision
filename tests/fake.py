"""A stand-in for a provider: given the answer distribution over marks and how the tokenizer cuts
them, it returns what a server would — the candidates for the next token after `[` + what is
already written — with the server quirks seen in practice switchable on."""
from __future__ import annotations

import math
import random


class FakeModel:
    def __init__(self, dist: dict[str, float], tokenizer: str = "single", limit: int = 20,
                 unsorted: bool = False, duplicates: bool = False, forms: bool = False,
                 glued: bool = False, noise: float = 0.0, seed: int = 0):
        """`dist`: mark -> probability (sums to ≤ 1; the rest goes to a non-answer token).
        `tokenizer`: "letters" (a mark is one token), "single" (one digit a token), "grouped"
        (a run of up to three digits is one token). `forms`: each continuation also comes as
        ` X` and `X]`, splitting its probability. `glued`: the first token carries the bracket,
        `[X`. `noise`: probability on a non-answer word."""
        self.dist, self.tokenizer, self.limit = dist, tokenizer, limit
        self.unsorted, self.duplicates, self.forms, self.glued, self.noise = unsorted, duplicates, forms, glued, noise
        self.rng = random.Random(seed)
        self.calls: list[str] = []

    def _tokens(self, rest: str) -> list[str]:
        """How the tokenizer cuts the rest of an answer (`rest` + `]`)."""
        if self.tokenizer == "single" and rest.isdigit():
            return list(rest) + ["]"]
        if self.tokenizer == "grouped" and rest.isdigit():
            return [rest[i:i + 3] for i in range(0, len(rest), 3)] + ["]"]
        return [rest + "]"] if self.tokenizer == "letters" else [rest, "]"]

    def __call__(self, written: str):
        self.calls.append(written)
        nxt: dict[str, float] = {}
        total = sum(p for m, p in self.dist.items() if m.startswith(written))
        for m, p in self.dist.items():
            if not m.startswith(written) or p <= 0:
                continue
            first = self._tokens(m[len(written):])[0] if m != written else "]"
            nxt[first] = nxt.get(first, 0.0) + p / total
        scale = 1.0 - self.noise
        cands: list[tuple[str, float]] = []
        for t, p in nxt.items():
            p *= scale
            shown = ("[" + t) if (self.glued and written == "") else t
            if self.forms and t != "]":
                closes = (written + t.rstrip("]")) in self.dist and not t.endswith("]")
                if closes:     # the same continuation as ` X` and `X]` too
                    cands += [(shown, p * 0.6), (" " + shown, p * 0.3), (shown + "]", p * 0.1)]
                else:
                    cands += [(shown, p * 0.7), (" " + shown, p * 0.3)]
            else:
                cands.append((shown, p))
        if self.noise:
            cands.append((" The", self.noise))
        cands = [(t, math.log(p)) for t, p in cands if p > 0]
        cands.sort(key=lambda x: -x[1])
        cands = cands[: self.limit] if self.limit else cands
        if self.duplicates and cands:
            cands = cands + [cands[0]]
        if self.unsorted:
            self.rng.shuffle(cands)
        return cands, self.limit
