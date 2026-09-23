"""Does the API surface return what it names? (aliased_return_tuple)

The night3 pdfgnn delivery's `autoregressive_decode` ends with
`return mu_all, sigma_all, nu_all, mu_all`: the fourth slot is the API's
promised `predicted_means`, and it is the SAME object as the first slot's
`mu`. The sampled path that slot was supposed to carry was computed and
discarded, every downstream consumer silently got the conditional mean,
and the shipped test for the sampling behavior became tautological
(asserting `predicted_means == mu`, which the aliasing makes always
true). Three independent reviewers called the resulting uncertainty
quantification bogus by four orders of magnitude.

A return tuple whose slots are distinct named quantities must return
distinct objects. Returning the same NAME twice is the certain, cheap
signature of the defect: it is visible without type inference, and the
delivered fleet's noise floor is exactly the one known-bad (measured at
landing, pinned by the fleet test).

This module is the home for deterministic API-surface primitives shared
by the coder validators; the params-code binding arms decided under
R2C-044 land here when implemented.
"""

from __future__ import annotations

import ast


def find_aliased_return_tuples(tree: ast.Module) -> list[tuple[int, str]]:
    """Return statements whose tuple carries the same name twice, as
    (line, name). One entry per (return, name)."""
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return) \
                or not isinstance(node.value, ast.Tuple):
            continue
        names = [e.id for e in node.value.elts if isinstance(e, ast.Name)]
        for name in sorted({n for n in names if names.count(n) > 1}):
            hits.append((node.lineno, name))
    return hits
