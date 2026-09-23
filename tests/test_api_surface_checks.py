"""aliased_return_tuple: distinct promised outputs must be distinct
objects (the night3 `return mu_all, sigma_all, nu_all, mu_all`)."""

import ast
import json

import pytest
import re
from pathlib import Path

from scripts.api_surface_checks import find_aliased_return_tuples


def _hits(source: str):
    return find_aliased_return_tuples(ast.parse(source))


def test_the_night3_decode_shape_flags():
    hits = _hits(
        "def autoregressive_decode(model, history):\n"
        "    mu_all = compute(history)\n"
        "    sigma_all = spread(history)\n"
        "    nu_all = tails(history)\n"
        "    return mu_all, sigma_all, nu_all, mu_all\n"
    )
    assert hits == [(5, "mu_all")]


def test_distinct_returns_are_silent():
    assert _hits(
        "def decode(model, history):\n"
        "    return mu, sigma, nu, samples\n"
    ) == []


def test_non_tuple_returns_are_silent():
    assert _hits(
        "def f(x):\n"
        "    return x\n"
    ) == []


def test_expressions_are_not_names_and_stay_silent():
    """`return x, x + 0` computes distinct objects; only NAME aliasing is
    the certain shape."""
    assert _hits(
        "def f(x):\n"
        "    return x, x + 0\n"
    ) == []


@pytest.mark.manual_only
def test_the_delivered_fleet_flags_no_new_aliased_return():
    """Noise floor: nothing outside the one known bad (the night3
    decode). Pinned on the defect class, not the dir name, surviving
    archival renames and the delete-before-reroll default. Corpus: the
    live fleet plus the committed example_runs/ floor; empty corpus skips
    instead of passing vacuously (W3 verdicts §5)."""
    scanned = 0
    flagged: set[str] = set()
    for root in (Path("r2c_runs"), Path("example_runs")):
        for py in sorted(root.glob("*/method/*.py")):
            if "tests" in py.parts:
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, ValueError):
                continue
            scanned += 1
            for _, name in find_aliased_return_tuples(tree):
                slug = re.sub(r"_\d+$", "", py.parent.parent.name)
                flagged.add(f"{slug}/{py.name}:{name}")

    if scanned == 0:
        pytest.skip("live fleet absent")
    known_bad = {
        "probabilistic-demand-forecasting-with-graph-neural-networks"
        "/method.py:mu_all"
    }
    assert flagged <= known_bad, flagged
