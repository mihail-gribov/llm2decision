"""Marks: assignment at every boundary, the prompt text, prefill, and reading the answer back
through each kind of tokenizer and each server quirk."""
import math

import pytest

from llm2decision.marks import (FIRST_NUMBER, LETTERS, MAX_OPTIONS, Marks, MarksError, assign, read,
                                read_text, referenced)

from fake import FakeModel


def keys(n):
    return [f"opt_{i}" for i in range(n)]


# -- assignment -----------------------------------------------------------------------------------

@pytest.mark.parametrize("n,scheme,first,last", [
    (2, "letters", "A", "B"),
    (10, "letters", "A", "J"),
    (11, "letters", "A", "K"),
    (26, "letters", "A", "Z"),
    (27, "numbers", "100", "126"),
    (99, "numbers", "100", "198"),
    (100, "numbers", "100", "199"),
    (101, "numbers", "100", "200"),
    (899, "numbers", "100", "998"),
    (900, "numbers", "100", "999"),
])
def test_assign_boundaries(n, scheme, first, last):
    m = assign(keys(n))
    assert m.scheme == scheme
    assert (m.marks[0], m.marks[-1]) == (first, last)
    assert len(m.marks) == len(set(m.marks)) == n
    assert len({len(x) for x in m.marks}) == 1                 # every mark the same length


@pytest.mark.parametrize("n", [0, 1, MAX_OPTIONS + 1, 1000])
def test_assign_refuses(n):
    with pytest.raises(MarksError):
        assign(keys(n))


def test_assign_refuses_duplicates():
    with pytest.raises(MarksError):
        assign(["a", "b", "a"])


def test_letters_and_numbers_meet_at_26():
    assert assign(keys(26)).marks[-1] == LETTERS[-1]
    assert assign(keys(27)).marks[0] == str(FIRST_NUMBER)


def test_score_keeps_own_levels_when_named():
    q = "How many violations? (0 = none, 1 = one, 2 = two, 3 = three or more)"
    m = assign(["0", "1", "2", "3"], "score", q)
    assert m.scheme == "own" and m.marks == ("0", "1", "2", "3")


def test_score_without_named_levels_gets_letters():
    m = assign(["0", "1", "2", "3"], "score", "Rate how complete the response is.")
    assert m.scheme == "letters" and m.marks == ("A", "B", "C", "D")


def test_choice_never_keeps_own_names_even_if_named():
    m = assign(["0", "1"], "choice", "0 = no, 1 = yes")
    assert m.scheme == "letters"


def test_referenced_needs_the_equals_form():
    assert referenced(["2"], "level 2 = two")
    assert not referenced(["2"], "there are 2 items")
    assert not referenced(["1"], "version 1.5 = new")          # part of a number is not a level


# -- the prompt -----------------------------------------------------------------------------------

def test_lines_hide_the_callers_names():
    m = assign(["bank_transfer", "cash", "paypal"])
    text = m.lines(["Bank transfer", "Cash", "PayPal"])
    assert text == "[A] Bank transfer\n[B] Cash\n[C] PayPal"
    assert "bank_transfer" not in text and "paypal" not in text


def test_lines_with_numbers_keep_order():
    m = assign(keys(30))
    lines = m.lines([f"d{i}" for i in range(30)]).split("\n")
    assert lines[0] == "[100] d0" and lines[26] == "[126] d26" and lines[29] == "[129] d29"


@pytest.mark.parametrize("kind,n,instr,phrase", [
    ("choice", 4, "", "one letter"),
    ("score", 4, "", "letter of the level"),
    ("choice", 40, "", "number of one option"),
    ("score", 40, "", "number of the level"),
    ("score", 4, "(0 = none, 1 = one)", "level that rates it"),
])
def test_ask_matches_the_scheme(kind, n, instr, phrase):
    k = [str(i) for i in range(n)] if instr else keys(n)
    assert phrase in assign(k, kind, instr).ask()


@pytest.mark.parametrize("n,digits,prefill", [
    (26, "single", ""), (26, "grouped", ""), (26, None, ""),             # letters: nothing written
    (27, "single", "1"), (100, "single", "1"), (101, "single", ""),      # numbers, one digit a token
    (27, "grouped", ""), (900, "grouped", ""),                          # a mark is one token
    (27, None, ""), (900, None, ""),                                    # unknown: nothing written
])
def test_prefill(n, digits, prefill):
    assert assign(keys(n)).prefill(digits) == prefill


def test_prefill_never_writes_a_whole_mark():
    m = assign(keys(27))
    assert all(len(m.prefill(d)) < 3 for d in ("single", "grouped", None))


@pytest.mark.parametrize("n", [2, 10, 26])
def test_numbers_never_for_26_or_fewer(n):
    with pytest.raises(MarksError):
        Marks(tuple(keys(n)), tuple(str(100 + i) for i in range(n)), "numbers")


def test_letters_never_for_27_or_more():
    with pytest.raises(MarksError):
        Marks(tuple(keys(27)), tuple(LETTERS) + ("AA",), "letters")


# -- reading --------------------------------------------------------------------------------------

def dist_on(marks: Marks, target: str, p: float = 0.8) -> dict:
    """`p` on the target mark, the rest spread over the others."""
    rest = (1 - p) / (len(marks.marks) - 1)
    return {m: (p if m == target else rest) for m in marks.marks}


CASES = [(n, t) for n, ts in {
    2: ["A", "B"], 26: ["A", "M", "Z"], 27: ["100", "109", "110", "126"],
    99: ["100", "150", "198"], 100: ["100", "199"], 101: ["100", "199", "200"],
    900: ["100", "109", "110", "199", "200", "555", "999"],
}.items() for t in ts]


@pytest.mark.parametrize("tokenizer", ["single", "grouped"])
@pytest.mark.parametrize("known", [True, False])
@pytest.mark.parametrize("n,target", CASES)
def test_read_round_trip(n, target, tokenizer, known):
    m = assign(keys(n))
    tok = "letters" if m.scheme == "letters" else tokenizer
    fake = FakeModel(dist_on(m, target), tok, limit=0)               # no top-k limit: exact numbers
    r = read(m, fake, digits=tokenizer if known else None)
    assert r.key == m.key_of(target)
    assert r.probabilities[r.key] == pytest.approx(0.8, abs=1e-9)
    assert r.key not in r.bounded
    assert r.logprobs[r.key] == pytest.approx(math.log(0.8))
    if tok in ("letters", "grouped"):                          # one token a mark: everything read exactly
        assert not r.bounded


@pytest.mark.parametrize("n,tokenizer,known,requests", [
    (26, "letters", True, 1),
    (27, "single", True, 2),        # `[1` written: tens, then units
    (27, "single", False, 3),       # `1` stepped over, then tens, then units
    (27, "grouped", True, 1),       # one token is the whole number
    (27, "grouped", False, 1),
    (900, "single", True, 3),       # nothing shared: hundreds, tens, units
    (900, "grouped", True, 1),
])
def test_read_request_count(n, tokenizer, known, requests):
    m = assign(keys(n))
    target = m.marks[-1]
    fake = FakeModel(dist_on(m, target), tokenizer, limit=0)
    r = read(m, fake, digits=(tokenizer if tokenizer != "letters" else None) if known else None)
    assert r.key == m.key_of(target) and r.requests == requests


@pytest.mark.parametrize("quirk", ["unsorted", "duplicates", "forms", "glued"])
@pytest.mark.parametrize("n", [4, 26, 27])
def test_read_survives_server_quirks(quirk, n):
    m = assign(keys(n))
    target = m.marks[len(m.marks) // 2]
    tok = "letters" if m.scheme == "letters" else "single"
    fake = FakeModel(dist_on(m, target, 0.7), tok, limit=0, **{quirk: True})
    r = read(m, fake, digits="single")
    assert r.key == m.key_of(target)
    assert r.probabilities[r.key] == pytest.approx(0.7, abs=1e-9)    # duplicates once, forms summed


def test_read_with_a_top_k_limit_bounds_the_unseen():
    m = assign(keys(26))
    fake = FakeModel(dist_on(m, "C", 0.9), "letters", limit=5)
    r = read(m, fake)
    assert r.key == m.key_of("C")
    assert len(r.bounded) == 26 - 5
    floor = min(lp for k, lp in r.logprobs.items() if k not in r.bounded)
    assert all(r.logprobs[k] == pytest.approx(floor) for k in r.bounded)


def test_read_ignores_a_non_answer_word():
    m = assign(keys(4))
    fake = FakeModel(dist_on(m, "B", 0.5), "letters", limit=0, noise=0.4)
    r = read(m, fake)
    assert r.key == m.key_of("B")
    assert r.logprobs[r.key] == pytest.approx(math.log(0.5 * 0.6))      # absolute: the word took 0.4
    assert r.probabilities[r.key] == pytest.approx(0.5)                 # among answers, as before the word


def test_read_nothing_readable():
    m = assign(keys(4))
    r = read(m, lambda w: ([(" The", -0.1), (" I", -2.0)], 20))
    assert r.key is None and r.requests == 1


def test_read_own_levels_of_unequal_length():
    """Levels 0 … 11: `1` starts both 1 and 10, 11 — the closing bracket decides."""
    levels = [str(i) for i in range(12)]
    q = "Rate it: " + ", ".join(f"{k} = level {k}" for k in levels)
    m = assign(levels, "score", q)
    assert m.scheme == "own"
    for target in ["0", "1", "9", "10", "11"]:
        fake = FakeModel(dist_on(m, target, 0.85), "single", limit=0)
        r = read(m, fake, digits="single")
        assert r.key == target, target
        assert r.probabilities[target] == pytest.approx(0.85, abs=1e-9)


def test_read_follows_the_likeliest_branch_and_bounds_the_rest():
    m = assign(keys(200))                        # 100 … 299, nothing shared: hundreds first
    dist = {x: 0.0 for x in m.marks}
    dist["250"], dist["130"], dist["131"] = 0.6, 0.25, 0.15
    fake = FakeModel(dist, "single", limit=0)
    r = read(m, fake, digits="single")
    assert r.key == m.key_of("250") and r.requests == 3
    assert r.logprobs[m.key_of("250")] == pytest.approx(math.log(0.6))
    assert r.probabilities[m.key_of("250")] == pytest.approx(0.6)
    assert m.key_of("130") in r.bounded                      # branch `1` not followed: bounded by its mass
    assert r.logprobs[m.key_of("130")] == pytest.approx(math.log(0.4))
    assert sum(r.probabilities.values()) == pytest.approx(1.0)


# -- text answers (providers without log-probabilities) -------------------------------------------

@pytest.mark.parametrize("text,want", [("[C]", "c"), ("C", "c"), (" [c] ", "c"), ("C]", "c"), ("[Z]", None), ("", None)])
def test_read_text_letters(text, want):
    m = assign(["a", "b", "c", "d"])
    assert read_text(m, text) == want


def test_read_text_numbers():
    m = assign(keys(40))
    assert read_text(m, "[139]") == "opt_39" and read_text(m, "140") is None


# -- numbers at every count: prefill, requests, reading -------------------------------------------

def expected(n, tokenizer, known):
    """Prefill and requests for numbers 100 … 99+n."""
    if tokenizer == "grouped":
        return "", 1                                   # the whole number is one token
    if not known:
        return "", 3                                   # every digit read, shared ones stepped over
    if n <= 100:
        return "1", 2                                  # 100 … 199: tens, then units
    return "", 3                                       # 100 … 999: hundreds, tens, units


def targets(n):
    last = 99 + n
    ts = {100, last, 100 + n // 2}
    ts |= {t for t in (109, 110, 199, 200) if t <= last}
    return sorted(str(t) for t in ts)


GRID = [(n, t) for n in [27, 28, 99, 100, 101, 102, 899, 900] for t in targets(n)]


@pytest.mark.parametrize("tokenizer", ["single", "grouped"])
@pytest.mark.parametrize("known", [True, False])
@pytest.mark.parametrize("n,target", GRID)
def test_numbers_every_count(n, target, tokenizer, known):
    m = assign(keys(n))
    assert m.scheme == "numbers" and m.marks[0] == "100" and m.marks[-1] == str(99 + n)
    prefill, requests = expected(n, tokenizer, known)
    digits = tokenizer if known else None
    assert m.prefill(digits) == prefill
    fake = FakeModel(dist_on(m, target, 0.8), tokenizer, limit=0)
    r = read(m, fake, digits=digits)
    assert fake.calls[0] == prefill                    # the first request is written as planned
    assert r.requests == requests
    assert r.key == m.key_of(target)
    assert r.probabilities[r.key] == pytest.approx(0.8, abs=1e-9)
    assert r.logprobs[r.key] == pytest.approx(math.log(0.8))
    assert r.written == (target[:-1] if tokenizer == "single" else "")   # read up to the last digit


@pytest.mark.parametrize("n", [2, 9, 10, 11, 25, 26])
def test_up_to_26_always_letters(n):
    m = assign(keys(n))
    assert m.scheme == "letters" and m.prefill("single") == "" and m.prefill(None) == ""


@pytest.mark.parametrize("n", [27, 100, 101, 900])
def test_numbers_with_a_top_k_limit(n):
    """Twenty candidates a position: the target is still read; nothing is read as more likely."""
    m = assign(keys(n))
    target = m.marks[-1]
    fake = FakeModel(dist_on(m, target, 0.7), "single", limit=20)
    r = read(m, fake, digits="single")
    assert r.key == m.key_of(target)
    assert r.probabilities[r.key] == max(r.probabilities.values())
