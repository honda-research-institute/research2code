"""Do the demo's evaluation windows overlap what the model saw? (R2C-066)

The runtime arm's shape, made static. The alias arm in
`eval_split_integrity` catches the same EXPRESSION consumed under two
roles; it is silent by construction when the expressions differ and the
index WINDOWS overlap — the 2026-08-04 shape, and again the night3
notebook (2026-08-05): training consumed the full demand history, the
forecast call conditioned on the full demand history, and the metrics
cell scored predictions against `demand_history[:, T_total-K:T_total]`,
a slice of the very window the model trained on and conditioned on.

## The primitive

Interval arithmetic over slice bounds, with names as symbols. Slice
bounds in generated notebooks are small affine expressions over a
handful of integers (`test_start = T_total - K`), where some of the
integers are known from params.json (`K = cfg["forecast_horizon"]`) and
some are opaque (`T_total` from a shape). An affine form over opaque
symbols still proves the two facts that matter: that `[T_total-K,
T_total)` is non-empty (K is known positive), and that a full-range read
therefore overlaps it.

Roles at the notebook seam:

- training: a tensor argument of a call whose callee carries a train
  verb (`train_model`, `fit`, ...).
- conditioning: a tensor argument of a call whose callee carries an
  inference verb (`forecast`, `predict`, ...). A held-out protocol must
  not condition the inference on the window it scores.
- evaluation: a tensor read inside a metric-named assignment (`rmse =
  ...`, `wmape = ...`) or an argument of an eval-verb call.

A finding is a root tensor read by (training or conditioning) and by
evaluation, where the two reads PROVABLY overlap and at least one of
them is range-constrained. Both conditions are load-bearing:

- provable overlap keeps symbolic windows honest — a training window
  `[t, t+P)` over a loop variable proves nothing against `[T-K, T)`
  and stays silent;
- the constrained-side requirement keeps `evaluate_model(model, X)`
  silent when the split lives inside the callee, which this seam
  cannot see. That boundary is pinned by a negative fixture.

Reported at the stage 3a notebook validation seam, before execution,
routed as a producer-fixable finding per the maintainer's 2026-08-05 severity
call for this class (fix-finding routing, degrade with disclosure only
when the loop exhausts).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from fractions import Fraction

_TRAIN_VERBS = frozenset({"train", "fit", "optimize"})
# Split-declaring tokens in a METRIC name: `train_acc` announces it is
# computed on the training split, so reading fitted data there is the
# point, not a leak. Deliberately narrow — ambiguous names (`warmup_acc`)
# stay classified as evaluation.
_TRAIN_SPLIT_TOKENS = frozenset({"train", "training", "insample", "tr"})
_INFERENCE_VERBS = frozenset({"forecast", "predict", "infer", "inference"})
_EVAL_VERBS = frozenset({"eval", "evaluate", "score", "metrics",
                         "benchmark"})
_METRIC_TOKENS = frozenset({
    "rmse", "mse", "mae", "mape", "wmape", "smape", "nrmse", "crps",
    "accuracy", "acc", "f1", "auc", "precision", "recall", "r2",
    "error", "errors",
})
# Arguments that are shared plumbing rather than data: the model itself,
# configuration, and training machinery appear in several roles by design.
_PLUMBING_TOKENS = frozenset({
    "model", "net", "network", "estimator", "clf", "cfg", "config",
    "params", "optimizer", "criterion", "scheduler", "rng", "seed",
    "device",
})
# Calls and methods that return their tensor argument's data unchanged
# for range purposes.
_WRAP_CALLS = frozenset({"tensor", "asarray", "array", "from_numpy",
                         "as_tensor"})
_WRAP_METHODS = frozenset({"astype", "copy", "clone", "float", "double",
                           "detach", "cpu", "numpy", "to", "contiguous"})


# ---------------------------------------------------------------------------
# Affine integer expressions: {symbol: coefficient} plus a constant term
# under the empty-string key. Enough arithmetic to fold generated slice
# bounds; anything richer resolves to None and the range reads as unknown.

Affine = dict[str, Fraction]


def _const(value: int | Fraction) -> Affine:
    return {"": Fraction(value)}


def _add(a: Affine, b: Affine, sign: int = 1) -> Affine:
    out = dict(a)
    for sym, coeff in b.items():
        out[sym] = out.get(sym, Fraction(0)) + sign * coeff
    return {s: c for s, c in out.items() if c or s == ""} or _const(0)


def _as_const(a: Affine | None) -> Fraction | None:
    if a is None:
        return None
    if any(c for s, c in a.items() if s):
        return None
    return a.get("", Fraction(0))


def _resolve_int(node: ast.AST, env: dict[str, Affine],
                 params: dict[str, int]) -> Affine | None:
    """Fold an expression to an affine form, opaque names as symbols."""
    if isinstance(node, ast.Constant):
        return _const(node.value) if isinstance(node.value, int) \
            and not isinstance(node.value, bool) else None
    if isinstance(node, ast.Name):
        return env.get(node.id, {node.id: Fraction(1)})
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _resolve_int(node.operand, env, params)
        return None if inner is None else _add(_const(0), inner, -1)
    if isinstance(node, ast.BinOp):
        left = _resolve_int(node.left, env, params)
        right = _resolve_int(node.right, env, params)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return _add(left, right)
        if isinstance(node.op, ast.Sub):
            return _add(left, right, -1)
        if isinstance(node.op, ast.Mult):
            for scalar, other in ((left, right), (right, left)):
                value = _as_const(scalar)
                if value is not None:
                    return {s: c * value for s, c in other.items()}
            return None
        if isinstance(node.op, (ast.FloorDiv, ast.Div)):
            lc, rc = _as_const(left), _as_const(right)
            if lc is None or rc in (None, 0):
                return None
            return _const(lc // rc if isinstance(node.op, ast.FloorDiv)
                          else lc / rc)
        return None
    if isinstance(node, ast.Subscript):
        # cfg["forecast_horizon"] and friends, from params.json.
        if isinstance(node.slice, ast.Constant) \
                and isinstance(node.slice.value, str) \
                and node.slice.value in params:
            value = params[node.slice.value]
            if isinstance(value, int) and not isinstance(value, bool):
                return _const(value)
        return None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in ("min", "max") and node.args \
            and not node.keywords:
        resolved = [_resolve_int(a, env, params) for a in node.args]
        if any(r is None for r in resolved):
            return None
        consts = [_as_const(r) for r in resolved]
        if all(c is not None for c in consts):
            picked = (min if node.func.id == "min" else max)(consts)
            return _const(picked)
        first = resolved[0]
        return first if all(r == first for r in resolved[1:]) else None
    return None


def _provably_positive(a: Affine | None) -> bool:
    value = _as_const(a)
    return value is not None and value > 0


def _provably_nonneg(a: Affine | None) -> bool:
    value = _as_const(a)
    return value is not None and value >= 0


# ---------------------------------------------------------------------------
# Ranges. One axis is (lower, upper) affine bounds; None means unbounded
# (a bare `:` or an axis never subscripted). UNKNOWN marks a bound the
# folder could not resolve — such an axis proves nothing either way.

UNKNOWN = object()
Axis = tuple  # (lower: Affine | None | UNKNOWN, upper: Affine | None | UNKNOWN)
FULL_AXIS: Axis = (None, None)


def _axis_from_slice(node: ast.AST, env: dict[str, Affine],
                     params: dict[str, int]) -> Axis:
    if isinstance(node, ast.Slice):
        if node.step is not None:
            return (UNKNOWN, UNKNOWN)
        lower = None if node.lower is None \
            else (_resolve_int(node.lower, env, params) or UNKNOWN)
        upper = None if node.upper is None \
            else (_resolve_int(node.upper, env, params) or UNKNOWN)
        return (lower, upper)
    index = _resolve_int(node, env, params)
    if index is None:
        return (UNKNOWN, UNKNOWN)
    return (index, _add(index, _const(1)))


def _axis_constrained(axis: Axis) -> bool:
    lower, upper = axis
    return (lower not in (None, UNKNOWN) and _as_const(lower) != 0) \
        or upper not in (None, UNKNOWN)


def _axis_nonempty(axis: Axis) -> bool:
    """Is [lower, upper) provably non-empty? Unbounded ends count as
    non-empty: arrays are non-empty on every axis a real read touches,
    and negative-index slices stay UNKNOWN rather than reaching here."""
    lower, upper = axis
    if lower is UNKNOWN or upper is UNKNOWN:
        return False
    if lower is None or upper is None:
        return True
    return _provably_positive(_add(upper, lower, -1))


def _axes_overlap(a: Axis, b: Axis) -> bool:
    """Provable non-empty intersection of two [lower, upper) axes."""
    if not (_axis_nonempty(a) and _axis_nonempty(b)):
        return False

    def _lt(x: Affine | None, y: Affine | None) -> bool:
        # x < y provably, with None as the matching infinity.
        if x is None or y is None:
            return True
        return _provably_positive(_add(y, x, -1))

    (a_lo, a_hi), (b_lo, b_hi) = a, b
    # Slices are in-bounds reads, so an unbounded lower is 0 and an
    # unbounded upper is the axis length; a bounded lower against an
    # unbounded upper needs the bound to be provably in range, which a
    # nonnegative lower bound of an in-bounds read is.
    lower_ok = _lt(a_lo, b_hi) if a_lo is not None or b_hi is not None \
        else True
    upper_ok = _lt(b_lo, a_hi) if b_lo is not None or a_hi is not None \
        else True
    if a_lo is not None and b_hi is None and not _provably_nonneg(a_lo):
        lower_ok = False
    if b_lo is not None and a_hi is None and not _provably_nonneg(b_lo):
        upper_ok = False
    return lower_ok and upper_ok


@dataclass(frozen=True)
class TensorRead:
    root: str
    axes: tuple[Axis, ...]  # () means the whole tensor, all axes full
    line: int

    def constrained(self) -> bool:
        return any(_axis_constrained(axis) for axis in self.axes)


def _reads_overlap(a: TensorRead, b: TensorRead) -> bool:
    if a.root != b.root:
        return False
    if not a.axes or not b.axes:
        # A whole-tensor read overlaps any provably non-empty read.
        other = b if not a.axes else a
        return not other.axes or all(_axis_nonempty(axis) or axis == FULL_AXIS
                                     for axis in other.axes)
    for axis_a, axis_b in zip(a.axes, b.axes):
        if axis_a == FULL_AXIS or axis_b == FULL_AXIS:
            continue
        if not _axes_overlap(axis_a, axis_b):
            return False
    return True


# ---------------------------------------------------------------------------
# Reading the notebook module: integer bindings, tensor bindings (a name
# standing for a possibly-sliced view of a root), and role consumptions.


def _tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", name.lower()) if t}


def _callee_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _tensor_read(node: ast.AST, tensors: dict[str, TensorRead],
                 env: dict[str, Affine],
                 params: dict[str, int]) -> TensorRead | None:
    """Resolve an expression to (root, axes) through wraps and slices."""
    if isinstance(node, ast.Name):
        known = tensors.get(node.id)
        if known is not None:
            return TensorRead(known.root, known.axes, node.lineno)
        return TensorRead(node.id, (), node.lineno)
    if isinstance(node, ast.Call):
        callee = _callee_name(node)
        if callee in _WRAP_CALLS and node.args:
            return _tensor_read(node.args[0], tensors, env, params)
        if callee in _WRAP_METHODS and isinstance(node.func, ast.Attribute):
            return _tensor_read(node.func.value, tensors, env, params)
        return None
    if isinstance(node, ast.Subscript):
        base = _tensor_read(node.value, tensors, env, params)
        if base is None:
            return None
        if base.axes:
            # Composing relative indexing is out of scope; a doubly-sliced
            # read proves nothing and stays silent.
            return TensorRead(base.root,
                              ((UNKNOWN, UNKNOWN),), node.lineno)
        elts = node.slice.elts if isinstance(node.slice, ast.Tuple) \
            else [node.slice]
        axes = tuple(_axis_from_slice(e, env, params) for e in elts)
        return TensorRead(base.root, axes, node.lineno)
    return None


def _iter_statements(body: list[ast.stmt]):
    for stmt in body:
        yield stmt
        for attr in ("body", "orelse", "finalbody"):
            inner = getattr(stmt, attr, None)
            if inner:
                yield from _iter_statements(inner)
        for handler in getattr(stmt, "handlers", []) or []:
            yield from _iter_statements(handler.body)


def _is_plumbing(read: TensorRead) -> bool:
    return bool(_tokens(read.root) & _PLUMBING_TOKENS)


@dataclass(frozen=True)
class LeakageFinding:
    root: str
    fit_role: str
    fit_line: int
    fit_text: str
    eval_line: int
    eval_text: str

    def message(self) -> str:
        return (
            f"line {self.eval_line}: the evaluation reads "
            f"`{self.eval_text}` while {self.fit_role} consumed "
            f"`{self.fit_text}` (line {self.fit_line}) — the windows "
            f"provably overlap along the protocol axis, so the reported "
            f"metrics describe data the model already saw. Hold the "
            f"evaluation window out: slice `{self.root}` so training and "
            f"the inference call's context end BEFORE the window the "
            f"metrics are computed on (eval_split_range_overlap)."
        )


def find_notebook_split_leakage(
        tree: ast.Module, params: dict[str, int] | None = None,
) -> list[LeakageFinding]:
    """Training/conditioning reads vs evaluation reads, joined by root
    tensor, flagged on provable window overlap with at least one side
    range-constrained. One finding per (root, fitting-role)."""
    params = {k: v for k, v in (params or {}).items()
              if isinstance(v, int) and not isinstance(v, bool)}
    env: dict[str, Affine] = {}
    tensors: dict[str, TensorRead] = {}
    fit_reads: list[tuple[str, TensorRead, str]] = []
    eval_reads: list[tuple[TensorRead, str]] = []

    def _iter_maximal_reads(node: ast.AST):
        """Tensor reads at their outermost resolvable expression, so
        `series[:, a:b]` yields once and its inner `series` never does."""
        read = _tensor_read(node, tensors, env, params)
        if read is not None:
            yield read, ast.unparse(node)
            return
        for child in ast.iter_child_nodes(node):
            yield from _iter_maximal_reads(child)

    def _collect_call(call: ast.Call) -> None:
        callee_tokens = _tokens(_callee_name(call))
        if "split" in callee_tokens:
            return  # split helpers produce the split, they consume no role
        if callee_tokens & _TRAIN_VERBS:
            role = "training"
        elif callee_tokens & _INFERENCE_VERBS:
            role = "the inference call's conditioning context"
        elif callee_tokens & _EVAL_VERBS:
            role = None  # evaluation consumer
        else:
            return
        args = list(call.args) + [kw.value for kw in call.keywords]
        for arg in args:
            read = _tensor_read(arg, tensors, env, params)
            if read is None or _is_plumbing(read):
                continue
            text = ast.unparse(arg)
            if role is None:
                eval_reads.append((read, text))
            else:
                fit_reads.append((role, read, text))

    for stmt in _iter_statements(tree.body):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                and isinstance(stmt.targets[0], ast.Name):
            name = stmt.targets[0].id
            value = _resolve_int(stmt.value, env, params)
            if value is not None:
                env[name] = value
            read = _tensor_read(stmt.value, tensors, env, params)
            if read is not None and read.root != name and all(
                    UNKNOWN not in axis for axis in read.axes):
                # A binding with an unresolvable range would poison every
                # later read of the name (the night3 notebook rebinds
                # demand_history through a list-indexed subscript, and
                # composition onto constrained axes always bails). Leave
                # the name as its own root instead: reads of the SAME name
                # still join, which is the observed leakage class.
                tensors[name] = read
            # A metric-named assignment's tensor reads are evaluation —
            # unless the name declares the TRAIN split. A train-labeled
            # metric (`warmup_train_acc`, `training_accuracy`) reads trained
            # data by definition; that is a fit diagnostic, not a leak
            # (GBALD 2026-09-01: the train-to-99%-accuracy protocol's
            # train-acc report halted stage 3c as eval_split_range_overlap
            # while the actual test metric sat on held-out x_test one line
            # below). Names claiming the eval split (`test_acc`, `val_acc`)
            # still register as evaluation.
            if _tokens(name) & _METRIC_TOKENS \
                    and not _tokens(name) & _TRAIN_SPLIT_TOKENS:
                for inner, text in _iter_maximal_reads(stmt.value):
                    if not _is_plumbing(inner):
                        eval_reads.append((inner, text))
        for node in ast.walk(stmt) if not isinstance(
                stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) else ():
            if isinstance(node, ast.Call):
                _collect_call(node)

    findings: list[LeakageFinding] = []
    seen: set[tuple[str, str]] = set()
    for role, fit_read, fit_text in fit_reads:
        key = (fit_read.root, role)
        if key in seen:
            continue
        for eval_read, eval_text in eval_reads:
            if eval_read.root != fit_read.root:
                continue
            if not (fit_read.constrained() or eval_read.constrained()):
                continue  # a split inside a callee is invisible here
            if _reads_overlap(fit_read, eval_read):
                seen.add(key)
                findings.append(LeakageFinding(
                    root=fit_read.root, fit_role=role,
                    fit_line=fit_read.line, fit_text=fit_text,
                    eval_line=eval_read.line, eval_text=eval_text))
                break
    # Dedupe nested Subscript reads that resolved to the same finding:
    # `demand[:, a:b]` walked twice (outer and inner nodes).
    unique: dict[tuple[str, str, int], LeakageFinding] = {}
    for finding in findings:
        unique.setdefault(
            (finding.root, finding.fit_role, finding.eval_line), finding)
    return sorted(unique.values(),
                  key=lambda f: (f.eval_line, f.root, f.fit_role))
