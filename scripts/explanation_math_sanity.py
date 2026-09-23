"""Explanation math-sanity pass — extract claims, then refute (queue item 3).

METHOD.md explanation entries make MECHANISM claims about formulas — "maximizing
the minimum log-probability is mathematically equivalent to maximizing the
minimum distance", "the factorization introduces diminishing returns as the
batch grows", "log p is non-positive on (0,1] so the term is always a penalty".
Two live defects of this genus shipped or nearly shipped (the archived GBALD
"Maximized Weighted Likelihood" invented-mechanism claim, and a fresh roll's
inverted core-set equivalence that survived the 07-06 prompt fixes). Prompt
rules are mitigation; this pass is the verification.

Design shape, with a hard trust boundary between the two halves:

1. **Claim extraction (LLM, Think tier), quote-anchored.** A dispatch over each
   explanation entry emits structured claims (see CLAIM_SCHEMA). The extractor
   NEVER judges. A deterministic floor (`anchor_claims`) drops any claim whose
   `quote` is not a byte-substring of the source entry, or whose formula labels
   do not appear in the entry or spec — the anti-hallucination rule, same genus
   as validate_method_explanations' fabricated-quote/label rejection.

2. **Refutation checker (pure Python, no LLM, no new heavy deps).** A numeric
   counterexample search with refutation-only semantics: one reproducible
   counterexample refutes; sampling that finds nothing is `consistent` (NOT
   proof); an unbindable/unsupported/over-budget claim is `not_checkable`.

**Refutation-only is the honesty contract:** this pass can DEMOTE and never
certifies. `consistent` adds no verification evidence anywhere — it just fails
to object. Only `refuted` produces a finding (a stage-1x rejection); the other
two verdicts are counted for REPORT.md disclosure and never block.

The numeric core evaluates the extractor's `formal.lhs`/`formal.rhs` expression
strings with `_safe_eval`, an AST-restricted evaluator over a whitelist of math
functions — NOT Python `eval`. A string that reaches for an attribute, an
unknown name, or a non-whitelisted call is unbindable, hence `not_checkable`,
never a code-execution surface.
"""

from __future__ import annotations

import ast
import json
import math
import random
import time
from dataclasses import dataclass, field

# --- tunables (deterministic; no wall-clock in the sampling itself) ---------

DEFAULT_SEED = 1729
N_SAMPLES = 400          # sample points per claim
MAX_VARS = 4             # more free variables than this ⇒ not_checkable (budget)
PARAM_DRAWS = 8          # fixed-parameter draws for cross-point comparisons
PER_CLAIM_BUDGET_S = 2.0  # wall-clock guard; expiry ⇒ not_checkable (test 6)
# A refutation must clear this margin at re-evaluation so float noise never
# manufactures a counterexample (the "re-evaluated at higher precision" guard).
REFUTE_EPS = 1e-6

# Verdicts.
REFUTED = "refuted"
CONSISTENT = "consistent"
NOT_CHECKABLE = "not_checkable"

# Relations the checker understands, grouped by refutation strategy. The
# relation NAME carries the claim's polarity, so the checker needs no separate
# "is this asserting X or not-X" flag.
_ORDER_EQUIV = {"argmax_equivalent", "argmin_equivalent"}
_IDENTITY = {"algebraic_identity"}
_NOVELTY = {"introduces_new_behavior"}
_MONOTONE = {"monotone_increasing", "monotone_decreasing"}
_SIGN = {"nonpositive", "nonnegative"}
KNOWN_RELATIONS = _ORDER_EQUIV | _IDENTITY | _NOVELTY | _MONOTONE | _SIGN

CLAIM_TYPES = {"equivalence", "monotonicity", "domain", "optimum", "novelty"}

# The JSON schema the extractor dispatch is held to. Kept here so the extractor
# prompt and the checker's expectations cannot drift apart.
CLAIM_SCHEMA = {
    "type": "object",
    "required": ["claims"],
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["claim_type", "quote", "formulas", "formal"],
                "properties": {
                    "claim_type": {"enum": sorted(CLAIM_TYPES)},
                    "quote": {"type": "string"},
                    "formulas": {"type": "array", "items": {"type": "string"}},
                    "formal": {
                        "type": "object",
                        "required": ["lhs", "relation", "variables"],
                        "properties": {
                            "lhs": {"type": "string"},
                            "rhs": {"type": "string"},
                            "relation": {"enum": sorted(KNOWN_RELATIONS)},
                            "variables": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["name", "domain"],
                                    "properties": {
                                        "name": {"type": "string"},
                                        "domain": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                            "minItems": 2,
                                            "maxItems": 2,
                                        },
                                        # A shared constant (batch size, dataset
                                        # size, a fixed hyperparameter) is held
                                        # FIXED while candidates vary; comparing
                                        # points across different constant values
                                        # would refute true claims like
                                        # "min s ⟺ min s/n".
                                        "role": {"enum": ["variable",
                                                          "parameter"]},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }
    },
}


# --- safe expression evaluation ---------------------------------------------


class _SafeEvalError(Exception):
    """The expression cannot be bound/evaluated — the claim is not_checkable."""


# Whitelisted unary math functions available inside an expression. All total on
# their guarded domains; domain errors (log of <=0) raise and mark not_checkable
# for that point rather than crashing.
_FUNCS = {
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "min": min,
    "max": max,
    "pow": pow,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def _safe_eval(node: ast.AST, env: dict[str, float]) -> float:
    """Evaluate a restricted arithmetic AST. Raises _SafeEvalError on anything
    outside {numbers, whitelisted names/consts, +-*/ ** %, unary +/-, and calls
    to _FUNCS}. No attribute access, subscripting, comprehensions, or names not
    in env/_CONSTS — so an extractor string can never reach Python internals."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body, env)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise _SafeEvalError(f"non-numeric constant {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in env:
            return float(env[node.id])
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise _SafeEvalError(f"unbound name {node.id!r}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _safe_eval(node.operand, env)
        return +val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left, env)
        right = _safe_eval(node.right, env)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.Pow):
            return left ** right
        if isinstance(op, ast.Mod):
            return left % right
        raise _SafeEvalError(f"unsupported operator {type(op).__name__}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise _SafeEvalError("only whitelisted math functions may be called")
        if node.keywords:
            raise _SafeEvalError("keyword arguments are not supported")
        args = [_safe_eval(a, env) for a in node.args]
        return float(_FUNCS[node.func.id](*args))
    raise _SafeEvalError(f"unsupported syntax {type(node).__name__}")


# The complete set of AST node types an expression may contain. A whitelist
# (not a blacklist) is the safe posture: any node outside this set — attribute
# access, subscripting, comprehensions, lambdas, string constants — makes the
# claim not_checkable rather than a code-execution surface.
_ALLOWED_NODES = (
    ast.Expression, ast.Constant, ast.Name, ast.Load,
    ast.UnaryOp, ast.UAdd, ast.USub,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.Call,
)


def _compile(expr: str) -> ast.Expression:
    """Parse and STATICALLY validate an expression before any evaluation, so an
    unsafe string is rejected as not_checkable up front (never evaluated). Raises
    _SafeEvalError on a syntax error or any disallowed node."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise _SafeEvalError(f"unparseable expression {expr!r}: {e}") from e
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise _SafeEvalError(f"disallowed syntax {type(node).__name__} in {expr!r}")
        if isinstance(node, ast.Constant) and (
                isinstance(node.value, bool)
                or not isinstance(node.value, (int, float))):
            raise _SafeEvalError(f"non-numeric constant {node.value!r} in {expr!r}")
        if isinstance(node, ast.Call) and (
                not isinstance(node.func, ast.Name)
                or node.func.id not in _FUNCS
                or node.keywords):
            raise _SafeEvalError(f"only whitelisted math functions may be called "
                                 f"in {expr!r}")
    return tree


def _free_names(tree: ast.AST) -> set[str]:
    """Variable names an expression references (excludes function names + consts)."""
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    return {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and n.id not in called
            and n.id not in _CONSTS}


# --- claim + verdict data model ---------------------------------------------


@dataclass
class ClaimVerdict:
    claim_type: str
    quote: str
    relation: str
    verdict: str
    detail: str
    counterexample: dict | None = None


@dataclass
class EntryResult:
    element_id: str
    dropped: list[dict] = field(default_factory=list)   # anti-hallucination floor
    verdicts: list[ClaimVerdict] = field(default_factory=list)

    @property
    def refuted(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if v.verdict == REFUTED]

    @property
    def consistent_count(self) -> int:
        return sum(v.verdict == CONSISTENT for v in self.verdicts)

    @property
    def not_checkable_count(self) -> int:
        return sum(v.verdict == NOT_CHECKABLE for v in self.verdicts)


# --- 1. quote-anchoring floor (deterministic, anti-hallucination) -----------


def anchor_claims(claims: list[dict], entry_prose: str,
                  spec_labels: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    """Split extractor claims into (kept, dropped).

    A claim is dropped when its `quote` is not a byte-substring of the source
    entry prose, or when any of its `formulas` labels appears neither in the
    entry prose nor in the spec label set. This is the hard anti-hallucination
    floor: the extractor may only make claims anchored to text that is actually
    there. `dropped` carries a reason so the caller can register the drop."""
    spec_labels = spec_labels or set()
    kept: list[dict] = []
    dropped: list[dict] = []
    for claim in claims:
        quote = claim.get("quote", "")
        if not quote or quote not in entry_prose:
            dropped.append({"claim": claim, "reason": "quote_not_in_entry",
                            "quote": quote[:120]})
            continue
        labels = claim.get("formulas") or []
        bad = [lab for lab in labels
               if lab and lab not in entry_prose and lab not in spec_labels]
        if bad:
            dropped.append({"claim": claim, "reason": "formula_label_absent",
                            "labels": bad})
            continue
        kept.append(claim)
    return kept, dropped


# --- 2. refutation checker (pure Python, refutation-only) -------------------


def _bindings(variables: list[dict]) -> tuple[
        list[str], list[tuple[float, float]], list[str]]:
    names: list[str] = []
    domains: list[tuple[float, float]] = []
    roles: list[str] = []
    for v in variables:
        name = v.get("name")
        dom = v.get("domain")
        role = v.get("role", "variable")
        if not name or not isinstance(dom, (list, tuple)) or len(dom) != 2:
            raise _SafeEvalError(f"bad variable spec {v!r}")
        if role not in ("variable", "parameter"):
            raise _SafeEvalError(f"unknown role {role!r} for {name}")
        lo, hi = float(dom[0]), float(dom[1])
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi < lo:
            raise _SafeEvalError(f"bad domain for {name}: {dom!r}")
        names.append(name)
        domains.append((lo, hi))
        roles.append(role)
    return names, domains, roles


def _samples(names, domains, rng, n):
    for _ in range(n):
        yield {name: rng.uniform(lo, hi) for name, (lo, hi) in zip(names, domains)}


def _eval_at(tree, env):
    """Evaluate; ValueError/ZeroDivisionError (e.g. log of a sampled <=0) means
    this point is out of the function's domain — skip it, don't crash."""
    try:
        return _safe_eval(tree, env)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _round_pt(pt: dict) -> dict:
    return {k: round(v, 4) for k, v in pt.items()}


def _collect(trees, names, domains, rng, n, deadline, fixed=None):
    """Sample up to n points; at each, evaluate every tree in `trees`. Keep only
    points inside every expression's domain (no None). Returns (rows, expired)
    where each row is (env, [values...]) and `expired` is True if the per-claim
    time budget cut the sampling short — the caller turns an unfinished search
    into not_checkable, never a false 'consistent'. `fixed` (parameter values
    held constant for this whole collection) is merged into every env so
    counterexample details show the constants a refutation was found at."""
    rows = []
    expired = False
    for env in _samples(names, domains, rng, n):
        if time.monotonic() > deadline:
            expired = True
            break
        if fixed:
            env = {**fixed, **env}
        vals = [_eval_at(t, env) for t in trees]
        if any(v is None for v in vals):
            continue
        rows.append((env, vals))
    return rows, expired


def _collect_grouped(trees, names, domains, roles, rng, n, deadline):
    """Like _collect, but role-aware: role=parameter names (shared constants)
    are drawn ONCE per group and held fixed while role=variable names vary.
    Cross-point checks (ranking equivalence, monotonicity) compare only within
    a group, so a shared constant can never manufacture a counterexample —
    "minimizing s" IS "minimizing s/n" when n is the same on both sides.
    PARAM_DRAWS groups keep refutation power across the constants' range.
    No parameters → one group, byte-identical sampling to the ungrouped path."""
    var_idx = [i for i, r in enumerate(roles) if r == "variable"]
    par_idx = [i for i, r in enumerate(roles) if r == "parameter"]
    if not par_idx:
        rows, expired = _collect(trees, names, domains, rng, n, deadline)
        return [rows], expired
    var_names = [names[i] for i in var_idx]
    var_domains = [domains[i] for i in var_idx]
    groups = []
    expired = False
    per_draw = max(2, n // PARAM_DRAWS)
    for _ in range(PARAM_DRAWS):
        fixed = {names[i]: rng.uniform(*domains[i]) for i in par_idx}
        rows, expired = _collect(trees, var_names, var_domains, rng,
                                 per_draw, deadline, fixed=fixed)
        if rows:
            groups.append(rows)
        if expired:
            break
    return groups, expired


def check_claim(claim: dict, *, seed: int = DEFAULT_SEED,
                n_samples: int = N_SAMPLES,
                budget_s: float = PER_CLAIM_BUDGET_S) -> ClaimVerdict:
    """Return a three-valued verdict (refuted / consistent / not_checkable) for
    one anchored claim, by numeric counterexample search over the declared
    variable domains with a fixed seed."""
    formal = claim.get("formal") or {}
    relation = formal.get("relation")
    claim_type = claim.get("claim_type", "?")
    quote = claim.get("quote", "")

    def nc(detail: str) -> ClaimVerdict:
        return ClaimVerdict(claim_type, quote, relation or "?", NOT_CHECKABLE, detail)

    if relation not in KNOWN_RELATIONS:
        return nc(f"unsupported relation {relation!r}")
    try:
        names, domains, roles = _bindings(formal.get("variables") or [])
    except _SafeEvalError as e:
        return nc(str(e))
    # A combinatorially huge sample space (too many free variables to search) is
    # a budget miss, not a guess — the design's "pathological sample space" case.
    if len(names) > MAX_VARS:
        return nc(f"{len(names)} free variables exceeds the search budget "
                  f"of {MAX_VARS}")

    needs_rhs = relation in (_ORDER_EQUIV | _IDENTITY | _NOVELTY)
    try:
        lhs = _compile(formal["lhs"])
        rhs = _compile(formal["rhs"]) if needs_rhs else None
    except (KeyError, _SafeEvalError) as e:
        return nc(f"expression binding failed: {e}")

    # Every free name in the expressions must have a declared domain, or we
    # cannot sample it — not_checkable, never a guess.
    declared = set(names)
    used = _free_names(lhs) | (_free_names(rhs) if rhs is not None else set())
    missing = used - declared
    if missing:
        return nc(f"no declared domain for variable(s): {', '.join(sorted(missing))}")

    rng = random.Random(seed)
    deadline = time.monotonic() + budget_s
    trees = [lhs, rhs] if needs_rhs else [lhs]
    groups, expired = _collect_grouped(trees, names, domains, roles, rng,
                                       n_samples, deadline)
    # Point-wise checks (both sides at the SAME point) are indifferent to
    # whether a name varies or stays fixed; only cross-point comparisons
    # (ranking, monotonicity) must never straddle two parameter values.
    rows = [row for g in groups for row in g]

    if relation in _ORDER_EQUIV:
        return _check_order_equiv(claim, groups, rows, expired, nc)
    if relation in _IDENTITY:
        return _check_identity(claim, rows, expired, nc)
    if relation in _NOVELTY:
        return _check_novelty(claim, rows, expired, nc)
    if relation in _MONOTONE:
        var_names = [n for n, r in zip(names, roles) if r == "variable"]
        return _check_monotone(claim, groups, rows, expired, var_names, nc)
    return _check_sign(claim, rows, expired, nc)


def _v(claim, verdict, detail, ce=None):
    return ClaimVerdict(claim.get("claim_type", "?"), claim.get("quote", ""),
                        claim["formal"]["relation"], verdict, detail, ce)


def _no_decision(rows, expired, nc):
    """Shared tail for the 'found no counterexample' case: distinguish an
    exhausted-but-clean search (consistent, caller supplies the message) from an
    empty or budget-truncated one (not_checkable)."""
    if not rows:
        return nc("no sampled point lay inside the expression domain(s)")
    if expired:
        return nc("per-claim search budget expired before a verdict")
    return None


def _check_order_equiv(claim, groups, rows, expired, nc):
    """argmax/argmin equivalence ⟺ lhs and rhs rank points identically. A pair
    whose relative order flips between the two forms refutes it (the two argmax
    forms would then select different points) — the core-set false-equivalence
    case lands exactly here. Pairs are compared only WITHIN a group (shared
    constants fixed), so a parameter difference never fakes an ordering flip."""
    for grp in groups:
        for i in range(len(grp)):
            ei, (li, ri) = grp[i]
            for j in range(i + 1, len(grp)):
                ej, (lj, rj) = grp[j]
                dl, dr = li - lj, ri - rj
                if abs(dl) > REFUTE_EPS and abs(dr) > REFUTE_EPS and (dl > 0) != (dr > 0):
                    detail = (f"at {_round_pt(ei)} the two forms rank one way "
                              f"(lhs={li:.4g}, rhs={ri:.4g}) but at {_round_pt(ej)} "
                              f"they rank the other (lhs={lj:.4g}, rhs={rj:.4g}) — "
                              f"the argmax/argmin forms select different points, so "
                              f"they are not equivalent")
                    return _v(claim, REFUTED, detail,
                              {"point_a": _round_pt(ei), "point_b": _round_pt(ej),
                               "lhs_a": li, "rhs_a": ri, "lhs_b": lj, "rhs_b": rj})
    undecided = _no_decision(rows, expired, nc)
    return undecided or _v(claim, CONSISTENT,
                           f"no ordering disagreement over {len(rows)} in-domain samples")


def _check_identity(claim, rows, expired, nc):
    """Asserts lhs == rhs everywhere. One sample where they differ refutes."""
    for env, (a, b) in rows:
        if abs(a - b) > REFUTE_EPS * max(1.0, abs(a), abs(b)):
            detail = (f"at {_round_pt(env)} the two sides differ "
                      f"(lhs={a:.6g}, rhs={b:.6g}) — the claimed identity does "
                      f"not hold")
            return _v(claim, REFUTED, detail,
                      {"point": _round_pt(env), "lhs": a, "rhs": b})
    undecided = _no_decision(rows, expired, nc)
    return undecided or _v(claim, CONSISTENT,
                           f"sides agreed over {len(rows)} in-domain samples")


def _check_novelty(claim, rows, expired, nc):
    """A novelty claim ('introduces new behavior') is an equivalence claim in
    disguise: if the two forms are algebraically identical on every sample, the
    'new mechanism' assertion is refuted (it is a plain identity). If they ever
    differ, we do NOT certify the difference is 'new' — we just fail to object
    (consistent). The honest 'direct consequence, no new behavior' dual is an
    identity claim and passes through _check_identity."""
    for env, (a, b) in rows:
        if abs(a - b) > REFUTE_EPS * max(1.0, abs(a), abs(b)):
            return _v(claim, CONSISTENT,
                      f"the two forms differ (e.g. at {_round_pt(env)}: "
                      f"lhs={a:.6g}, rhs={b:.6g}); not objecting")
    # No sample distinguished them. Only claim identity (and thus refute the
    # novelty) if the search actually completed over real samples.
    undecided = _no_decision(rows, expired, nc)
    if undecided is not None:
        return undecided
    detail = (f"the two forms are algebraically identical over {len(rows)} "
              f"samples, so the factorization introduces no new behavior — the "
              f"'{claim.get('claim_type')}' mechanism claim is unsupported")
    return _v(claim, REFUTED, detail, {"identical_over_samples": len(rows)})


def _check_monotone(claim, groups, rows, expired, var_names, nc):
    """Asserts lhs increases/decreases in its single role=variable name. A
    consecutive pair (sorted by that variable, shared constants fixed within
    the group) violating the direction refutes."""
    if len(var_names) != 1:
        return nc("monotonicity needs exactly one non-parameter variable")
    increasing = claim["formal"]["relation"] == "monotone_increasing"
    var = var_names[0]
    total = 0
    for grp in groups:
        pts = sorted(((env[var], vals[0], env) for env, vals in grp),
                     key=lambda t: t[0])
        if len(pts) < 2:
            continue
        total += len(pts)
        for (x0, y0, e0), (x1, y1, e1) in zip(pts, pts[1:]):
            if x1 - x0 <= REFUTE_EPS or abs(y1 - y0) <= REFUTE_EPS:
                continue
            if ((y1 - y0) > 0) != increasing:
                direction = "increasing" if increasing else "decreasing"
                detail = (f"claimed {direction} in {var}, but at {var}={x0:.4g} "
                          f"value is {y0:.4g} and at {var}={x1:.4g} value is "
                          f"{y1:.4g} — the wrong direction")
                return _v(claim, REFUTED, detail,
                          {"point_a": _round_pt(e0), "point_b": _round_pt(e1)})
    if total < 2:
        return nc("fewer than two in-domain samples")
    if expired:
        return nc("per-claim search budget expired before a verdict")
    return _v(claim, CONSISTENT, f"monotone over {total} sorted samples")


def _check_sign(claim, rows, expired, nc):
    """Asserts lhs is nonpositive/nonnegative over the domain. One sample of the
    wrong sign refutes."""
    nonpositive = claim["formal"]["relation"] == "nonpositive"
    for env, (y,) in rows:
        violates = (y > REFUTE_EPS) if nonpositive else (y < -REFUTE_EPS)
        if violates:
            want = "non-positive" if nonpositive else "non-negative"
            detail = (f"claimed {want} on the domain, but at {_round_pt(env)} "
                      f"the value is {y:.6g}")
            return _v(claim, REFUTED, detail, {"point": _round_pt(env), "value": y})
    undecided = _no_decision(rows, expired, nc)
    want = "non-positive" if nonpositive else "non-negative"
    return undecided or _v(claim, CONSISTENT,
                           f"stayed {want} over {len(rows)} in-domain samples")


# --- entry-level orchestration ----------------------------------------------


def evaluate_entry(element_id: str, entry: dict, claims: list[dict], *,
                   spec_labels: set[str] | None = None,
                   seed: int = DEFAULT_SEED,
                   n_samples: int = N_SAMPLES) -> EntryResult:
    """Anchor then check every claim for one explanation entry.

    `entry` is the sidecar explanation dict (what/why_novel/intuition); its
    concatenated prose is the byte-substring authority for quote anchoring."""
    prose = "\n".join(str(entry.get(f, "")) for f in ("what", "why_novel", "intuition"))
    kept, dropped = anchor_claims(claims, prose, spec_labels)
    result = EntryResult(element_id=element_id, dropped=dropped)
    for claim in kept:
        result.verdicts.append(check_claim(claim, seed=seed, n_samples=n_samples))
    return result


def findings_from_result(result: EntryResult) -> list[dict]:
    """Refuted claims become check_explanations-shaped findings (one rejection
    per refuted claim), so the stage-1x loop rejects the entry and carries the
    counterexample into the retry prompt. consistent/not_checkable produce no
    finding — the honesty contract (this pass only demotes)."""
    out: list[dict] = []
    for v in result.refuted:
        out.append({
            "severity": "error",
            "check": "math_sanity_refuted",
            "element_id": result.element_id,
            "message": (f"{result.element_id}: a mechanism claim is refuted by a "
                        f"numeric counterexample — {v.detail}. Claimed: "
                        f"\"{v.quote[:160]}\". Fix the explanation or remove the "
                        f"claim."),
            "counterexample": v.counterexample,
        })
    return out


# --- claim extraction: prompt + response parsing (LLM boundary) -------------
#
# The extractor runs on the Think tier and NEVER judges — it only surfaces the
# mechanism claims already present in an entry, anchored to verbatim quotes, for
# the deterministic checker to refute. It follows the judge/reviewer pattern:
# the dispatched agent WRITES its structured output to a scratch file, which the
# driver reads and feeds to `parse_extractor_response`. The specific Think agent
# is chosen at the run_stage_1x wiring site (reuse, no new agent surface).

_EXTRACTOR_PROMPT = """\
You are extracting the MECHANISM CLAIMS an explanation makes about a formula, so
a separate deterministic checker can test them. You do NOT judge whether any
claim is true — extract faithfully and let the checker decide.

Explanation entry id: {element_id}

The entry's prose (this is the ONLY text you may quote from, verbatim):
--- entry prose ---
{entry_prose}
--- end entry prose ---

The formulas this entry explains (for binding claims to expressions):
--- formulas ---
{formulas_context}
--- end formulas ---

Emit ONLY claims that assert one of these MECHANISM relations about the formula:
- equivalence / optimum: two forms share an argmax/argmin, or are algebraically
  identical (relation: argmax_equivalent | argmin_equivalent | algebraic_identity)
- novelty: a rewriting "introduces new behavior" / "a new mechanism"
  (relation: introduces_new_behavior)
- monotonicity: a quantity increases/decreases in a variable
  (relation: monotone_increasing | monotone_decreasing)
- domain/sign: an expression is non-positive / non-negative on a domain
  (relation: nonpositive | nonnegative)

Skip pure intuition or motivation with no formal, checkable content.

For each claim provide `formal` so the checker can sample it: `lhs` (and `rhs`
for equivalence/identity/novelty) as plain arithmetic expressions using the
formula's variables and the functions log, exp, sqrt, sin, cos, abs, min, max,
pow; and `variables` with a numeric `domain` [low, high] for EVERY variable the
expressions use. Mark a shared constant — a batch size, dataset size, or fixed
hyperparameter that stays the SAME while candidates are compared — with
"role": "parameter"; the quantity the mechanism actually varies over keeps the
default "role": "variable". If you cannot bind a claim to expressions with
domains, omit it — a missing claim is safe, an unbindable one wastes the
checker's budget.

`quote` MUST be copied verbatim (byte-for-byte) from the entry prose above; any
claim whose quote is not found in the entry is discarded.

Write EXACTLY ONE file at: {output_path}
It must be a JSON object matching:
{schema}
"""


def build_extractor_prompt(element_id: str, entry: dict, formulas_context: str,
                           output_path: str) -> str:
    """The dispatch prompt for the claim extractor (Think tier). Inlines the
    entry prose (the sole quotable source) and the formulas, and holds the agent
    to CLAIM_SCHEMA written at output_path — the write-to-scratch pattern the
    judge/reviewer dispatches already use."""
    prose = "\n".join(str(entry.get(f, "")) for f in ("what", "why_novel", "intuition"))
    schema = (
        '{"claims": [{"claim_type": "equivalence|monotonicity|domain|optimum|'
        'novelty", "quote": "<verbatim sentence>", "formulas": ["<eq label>"], '
        '"formal": {"lhs": "<expr>", "rhs": "<expr, if needed>", "relation": '
        '"<one relation above>", "variables": [{"name": "x", "domain": [lo, hi], '
        '"role": "variable|parameter"}]'
        "}}]}"
    )
    return _EXTRACTOR_PROMPT.format(
        element_id=element_id, entry_prose=prose,
        formulas_context=formulas_context or "(none provided)",
        output_path=output_path, schema=schema)


def _strip_to_json(raw: str) -> str:
    """Best-effort: pull the JSON object out of an agent response that may wrap
    it in a ```json fence or surround it with prose."""
    text = raw.strip()
    if "```" in text:
        # take the content of the first fenced block
        after = text.split("```", 1)[1]
        if after[:4].lower() == "json":
            after = after[4:]
        text = after.split("```", 1)[0].strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def parse_extractor_response(raw) -> list[dict]:
    """Parse the extractor's output (a dict already decoded, or a JSON string
    possibly fenced/prose-wrapped) into a list of well-formed claim dicts.

    Malformed claims are DROPPED, not raised — a bad extraction must never crash
    the best-effort stage 1x. A claim is well-formed when it has a known
    claim_type, a string quote, and a formal object with a relation. The
    deterministic anchoring floor + checker apply the deeper rules."""
    if isinstance(raw, dict):
        obj = raw
    elif isinstance(raw, str):
        try:
            obj = json.loads(_strip_to_json(raw))
        except (ValueError, TypeError):   # any parse failure ⇒ no claims
            return []
    else:
        return []
    claims = obj.get("claims") if isinstance(obj, dict) else None
    if not isinstance(claims, list):
        return []
    out: list[dict] = []
    for c in claims:
        if not isinstance(c, dict):
            continue
        if c.get("claim_type") not in CLAIM_TYPES:
            continue
        if not isinstance(c.get("quote"), str) or not c["quote"]:
            continue
        formal = c.get("formal")
        if not isinstance(formal, dict) or not formal.get("relation"):
            continue
        out.append(c)
    return out


# --- cross-entry orchestration (what the stage-1x loop calls) ---------------


def math_sanity_findings(sidecar: dict, extractions: dict[str, object], *,
                         spec_labels: set[str] | None = None,
                         seed: int = DEFAULT_SEED,
                         n_samples: int = N_SAMPLES) -> tuple[list[dict], dict]:
    """Run the pass over a merged sidecar given per-entry extractor outputs.

    `extractions` maps element_id -> the extractor's raw response for that entry
    (dict or string; parsed here). Returns (findings, metadata):
      - findings: check_explanations-shaped rejections (refuted claims only),
        ready to fold into the stage-1x findings list.
      - metadata: per-entry and total counts of refuted / consistent /
        not_checkable / dropped, for the REPORT.md disclosure ("N mechanism
        claims were not machine-checkable") — never blocks anything.
    """
    explanations = sidecar.get("explanations")
    if not isinstance(explanations, dict):
        return [], {"entries": {}, "totals": {}}
    findings: list[dict] = []
    per_entry: dict[str, dict] = {}
    totals = {"refuted": 0, "consistent": 0, "not_checkable": 0, "dropped": 0}
    for eid, entry in explanations.items():
        if eid not in extractions or not isinstance(entry, dict):
            continue
        claims = parse_extractor_response(extractions[eid])
        result = evaluate_entry(eid, entry, claims, spec_labels=spec_labels,
                                seed=seed, n_samples=n_samples)
        findings.extend(findings_from_result(result))
        counts = {
            "refuted": len(result.refuted),
            "consistent": result.consistent_count,
            "not_checkable": result.not_checkable_count,
            "dropped": len(result.dropped),
        }
        per_entry[eid] = counts
        for k in totals:
            totals[k] += counts[k]
    # Coverage counters (ADAM 2026-07-14 blind spot: 6 of 10 equations
    # carried no claims and no surface said so). Disclosure only.
    totals["entries_total"] = sum(
        1 for entry in explanations.values() if isinstance(entry, dict))
    totals["entries_examined"] = len(per_entry)
    totals["entries_claim_free"] = sum(
        1 for counts in per_entry.values()
        if counts["refuted"] + counts["consistent"] + counts["not_checkable"] == 0
    )
    return findings, {"entries": per_entry, "totals": totals}
