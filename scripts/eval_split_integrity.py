"""Does the generated code evaluate on data it did not train on? (R2C-066)

Nothing in the pipeline compared what the generated code trains on with what
it reports metrics on. The provenance battery binds parameters to the paper,
the smoke gate checks execution, and the fidelity review checks mechanisms,
so a demo that fits its own test window and reports the fit as accuracy
passed every existing gate as long as it ran. Two delivered runs did exactly
that, and a researcher reviewer called the second one the single
disqualifying finding for a forecasting paper.

## The two shapes, and what their diff fixed

- 2026-08-04: the split was computed, the validation loss aliased the
  training windows, and the test window was trained on.
- 2026-08-05: the split was computed and wholly ignored. The training target
  IS the test horizon (the code's own comment says "the test set"), early
  stopping watches the same tensor as a "validation" loss, and the metric
  function reports on it again.

Per the two-concrete-cases rule, the diff between them determines the
abstraction: checking that a split is COMPUTED proves nothing, so the check
has to look at what the code actually CONSUMES in each role.

## The primitive

One idea, applied twice. Every ground-truth expression the code consumes has
a role — supervision, model selection, or reported evaluation — and those
roles must read disjoint data. So: collect the ground-truth expressions per
role, and any expression that turns up under two roles is the finding. It is
alias analysis on expression text, which is mechanical and exact for the
shape that matters (the same variable handed to two consumers), and
deliberately silent about roles that merely look similar.

Reported as a producer-fixable finding at the coder's own validation seam,
per the maintainer's 2026-08-05 policy call: the existing fix loop repairs it, and the
delivery degrades with the finding disclosed only if the loop exhausts.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

# Callee-name tokens that mark a call as consuming data in a given role. The
# verbs are near-universal in ML code, which is what makes the join work
# across paradigms without a per-paper registry.
_TRAIN_VERBS = frozenset({"train", "fit", "optimize"})
_EVAL_VERBS = frozenset({"eval", "evaluate", "score", "metrics", "test",
                         "benchmark"})
_VALIDATION_TOKENS = frozenset({"val", "valid", "validation", "dev", "holdout"})
# A loss function IS supervision, whatever it is named. Without this, a
# `score_loss_fn(preds, targets)` reads as an evaluation call because of the
# `score` token and its own targets look aliased against themselves (the ms3d
# delivery, verified as a false positive before this guard existed).
_LOSS_TOKENS = frozenset({"loss", "criterion", "nll", "objective", "cost",
                          "penalty"})

# An expression only counts when it names ground truth. Predictions, models,
# and features are shared between roles by design; the observed targets are
# what must not be.
#
# Split in two because `y` alone is ambiguous. A strong marker settles it
# whatever else the name carries, so `forecast_targets` is ground truth. A
# weak marker needs the absence of a prediction word, so `y_pred` is not.
_STRONG_TARGET_TOKENS = frozenset({
    "target", "targets", "label", "labels", "truth", "gt", "actual",
    "actuals", "observed", "ytrue", "ygt",
})
_WEAK_TARGET_TOKENS = frozenset({"y"})
_PREDICTION_TOKENS = frozenset({
    "pred", "preds", "predicted", "prediction", "predictions", "output",
    "outputs", "logits", "mu", "sigma", "hat", "yhat", "est", "estimate",
    "estimates", "forecast", "forecasts", "score", "scores",
})


@dataclass(frozen=True)
class SplitFinding:
    line: int
    expression: str
    first_role: str
    second_role: str
    detail: str

    def message(self, file_label: str) -> str:
        return (
            f"{file_label}:{self.line}: evaluation-split integrity is not "
            f"established: the {self.first_role} data range and the "
            f"{self.second_role} data range cannot be shown disjoint "
            f"({self.detail}). Required protocol-axis overlap: empty. Change "
            f"the split arithmetic or slice bounds until the two consumed "
            f"ranges are disjoint (eval_split_aliasing)."
        )


def _tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", name.lower()) if t}


def _root_name(node: ast.AST) -> str:
    """The base identifier an expression reads from.

    `target[a:b].detach()` is `target`. Keying on the root rather than every
    identifier in the expression is what keeps a SLICE BOUND from deciding
    what the expression is: `pred_scores[:, :n_gt]` read as ground truth
    under a whole-text match, because the bound `n_gt` carries a
    ground-truth token while the tensor is a prediction (found on the ms3d
    delivery)."""
    while True:
        if isinstance(node, (ast.Subscript, ast.Attribute)):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        else:
            break
    return node.id if isinstance(node, ast.Name) else ""


def _is_target_expression(node: ast.AST) -> bool:
    tokens = _tokens(_root_name(node))
    if tokens & _STRONG_TARGET_TOKENS:
        return True
    return bool(tokens & _WEAK_TARGET_TOKENS) and not (
        tokens & _PREDICTION_TOKENS)


def _callee_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _call_role(node: ast.Call) -> str | None:
    tokens = _tokens(_callee_name(node))
    if not tokens or tokens & _LOSS_TOKENS:
        return None
    if tokens & _VALIDATION_TOKENS and tokens & _TRAIN_VERBS:
        return None  # `train_val_split` and friends produce splits, not roles
    if tokens & _EVAL_VERBS:
        return "the reported evaluation"
    if tokens & _TRAIN_VERBS:
        return "training supervision"
    return None


def _target_arguments(node: ast.Call) -> list[tuple[str, int]]:
    """Ground-truth expressions this call consumes, as (text, line)."""
    out: list[tuple[str, int]] = []
    for arg in list(node.args) + [kw.value for kw in node.keywords]:
        if isinstance(arg, (ast.Constant, ast.Lambda)):
            continue
        if _is_target_expression(arg):
            out.append((ast.unparse(arg),
                        getattr(arg, "lineno", node.lineno)))
    return out


def _loss_role(target_names: list[str]) -> str | None:
    """The role a loss assignment plays, read from what it is called.

    `val_loss = criterion(pred, target)` is model selection;
    `loss = criterion(pred, target)` is supervision. When both read the same
    ground truth, early stopping is watching the training objective."""
    for name in target_names:
        tokens = _tokens(name)
        if not tokens & {"loss", "objective", "nll", "error", "cost"}:
            continue
        return ("model selection" if tokens & _VALIDATION_TOKENS
                else "training supervision")
    return None


def _assigned_names(node: ast.Assign) -> list[str]:
    names: list[str] = []
    for target in node.targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Tuple):
            names.extend(e.id for e in target.elts if isinstance(e, ast.Name))
    return names


def _roles_in_function(func: ast.AST) -> list[tuple[str, str, int, str]]:
    """(expression, role, line, detail) for every ground-truth consumption.

    A loss assignment claims the calls inside it, so one consumption is
    counted once: the variable's own name decides whether that loss is
    supervision or model selection, and its callee verb is not consulted."""
    seen: list[tuple[str, str, int, str]] = []
    claimed: set[int] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        role = _loss_role(_assigned_names(node))
        if role is None:
            continue
        for call in ast.walk(node.value):
            if not isinstance(call, ast.Call):
                continue
            claimed.add(id(call))
            for text, line in _target_arguments(call):
                seen.append((text, role, line, f"as the {role} loss"))
    for node in ast.walk(func):
        if not isinstance(node, ast.Call) or id(node) in claimed:
            continue
        role = _call_role(node)
        if role is None:
            continue
        for text, line in _target_arguments(node):
            seen.append((text, role, line,
                         f"passed to `{_callee_name(node)}(...)`"))
    return seen


def find_split_aliasing(tree: ast.Module) -> list[SplitFinding]:
    """Ground-truth expressions consumed under two roles that must be
    disjoint. One finding per expression, naming the two roles."""
    findings: list[SplitFinding] = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        by_expression: dict[str, list[tuple[str, int, str]]] = {}
        for text, role, line, detail in _roles_in_function(func):
            by_expression.setdefault(text, []).append((role, line, detail))
        for text, uses in by_expression.items():
            roles = {role for role, _, _ in uses}
            if len(roles) < 2:
                continue
            first, second = sorted(roles)
            detail = "; ".join(dict.fromkeys(d for _, _, d in uses))
            findings.append(SplitFinding(
                line=min(line for _, line, _ in uses), expression=text,
                first_role=first, second_role=second, detail=detail))
    return sorted(findings, key=lambda f: (f.line, f.expression))
