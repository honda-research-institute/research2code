"""Report-only split lineage analysis for R2C-077 piece 3.

This module reduces the two observed train-and-evaluate shapes into one small
abstract interpreter.  It records what protocol-axis range each consumer role
reads for a scored model.  It deliberately has no finding type, message, or
validator integration.  The reductions and fleet controls must establish the
safe abstraction before a later block decides enforcement.

The supported transfer set is intentionally narrow and evidence-led:

* source-ordered assignments and scoped reassignments;
* integer/rational arithmetic, ``int``, ``min``, and ``max``;
* ``range`` and ``randperm`` induction domains;
* monotone ``while t + K <= bound`` loops with a positive increment;
* tensor wrappers, aliases, protocol-axis slices, and boolean masks;
* conventional model-builder calls used by generated notebooks;
* one-level trainer summaries instantiated at notebook call sites; and
* model identity through training, forecasting, and metric expressions.

Unsupported syntax is retained as unresolved analysis state.  It never turns
into a clean or failing enforcement verdict here.
"""

from __future__ import annotations

import ast
import builtins
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping, Sequence, Union


FITTING = "fitting"
MODEL_SELECTION = "model_selection"
INFERENCE_CONDITIONING = "inference_conditioning"
REPORTED_EVALUATION = "reported_evaluation"

_FORECAST_TOKENS = frozenset({"forecast", "predict", "infer", "inference"})
_LOSS_TOKENS = frozenset({
    "loss", "criterion", "nll", "objective", "cost", "penalty",
})
_METRIC_TOKENS = frozenset({
    "rmse", "mse", "mae", "mape", "wmape", "smape", "nrmse", "crps",
    "accuracy", "acc", "f1", "auc", "precision", "recall", "r2",
    "error", "errors", "score", "scoring", "metric", "metrics",
    "evaluate", "evaluation", "eval",
})
_TRANSPARENT_CALLS = frozenset({
    "tensor", "as_tensor", "asarray", "array", "from_numpy", "log1p",
    "nan_to_num", "mean", "sum", "sqrt", "abs", "norm",
})
_TRANSPARENT_METHODS = frozenset({
    "astype", "copy", "clone", "float", "double", "detach", "cpu", "numpy",
    "to", "contiguous", "item",
})
_NON_EVALUATION_CALL_TOKENS = frozenset({
    "append", "display", "extend", "fill", "hist", "imshow", "log",
    "plot", "print", "render", "scatter", "show", "visualize", "write",
})


def _tokens(name: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", name.lower()) if token}


def _callee_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _exception_type_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Call):
        return _exception_type_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _handler_exception_names(node: ast.AST | None) -> set[str] | None:
    if node is None:
        return None
    if isinstance(node, ast.Tuple):
        names = {
            name for item in node.elts
            if (name := _exception_type_name(item)) is not None
        }
        return names or None
    name = _exception_type_name(node)
    return {name} if name is not None else None


def _handler_matches(
    handler: ast.ExceptHandler, exception_name: str | None,
) -> bool:
    names = _handler_exception_names(handler.type)
    if names is None or exception_name is None:
        return True
    raised_class = getattr(builtins, exception_name, None)
    for name in names:
        handler_class = getattr(builtins, name, None)
        if isinstance(raised_class, type) \
                and isinstance(handler_class, type) \
                and issubclass(raised_class, BaseException) \
                and issubclass(handler_class, BaseException):
            if issubclass(raised_class, handler_class):
                return True
        elif name == exception_name:
            return True
        else:
            # Custom exception inheritance is unavailable from the narrow
            # source slice.  Retain the handler path conservatively.
            return True
    return False


def _handler_builtin_exception_classes(
    handler: ast.ExceptHandler,
) -> tuple[type[BaseException], ...] | None:
    if handler.type is None:
        return None
    names = _handler_exception_names(handler.type)
    if names is None:
        return ()
    classes: list[type[BaseException]] = []
    for name in names:
        candidate = getattr(builtins, name, None)
        if not isinstance(candidate, type) \
                or not issubclass(candidate, BaseException):
            return ()
        classes.append(candidate)
    return tuple(classes)


def _implicit_handler_fully_covered(
    handler: ast.ExceptHandler,
    prior: Sequence[type[BaseException]],
) -> bool:
    classes = _handler_builtin_exception_classes(handler)
    if classes is None:
        return bool(prior) and any(item is BaseException for item in prior)
    if not classes:
        return False
    return all(
        any(issubclass(candidate, earlier) for earlier in prior)
        for candidate in classes
    )


def _handlers_catch_all_implicit(
    handlers: Sequence[ast.ExceptHandler],
) -> bool:
    return any(
        handler.type is None
        or BaseException in (
            _handler_builtin_exception_classes(handler) or ()
        )
        for handler in handlers
    )


def _target_evaluation_expressions(
    target: ast.AST,
) -> tuple[ast.AST, ...]:
    """Expressions evaluated when Python binds or deletes a target."""
    if isinstance(target, ast.Name):
        return ()
    if isinstance(target, ast.Attribute):
        return (target.value,)
    if isinstance(target, ast.Subscript):
        return (target.value, target.slice)
    if isinstance(target, (ast.Tuple, ast.List)):
        return tuple(
            expression
            for item in target.elts
            for expression in _target_evaluation_expressions(item)
        )
    if isinstance(target, ast.Starred):
        return _target_evaluation_expressions(target.value)
    return ()


def _literal_iterable_nonempty(expression: ast.AST) -> bool | None:
    if isinstance(expression, (ast.List, ast.Tuple, ast.Set)):
        return bool(expression.elts)
    if isinstance(expression, ast.Call) \
            and isinstance(expression.func, ast.Name) \
            and expression.func.id == "range" \
            and not expression.keywords:
        try:
            values = [ast.literal_eval(item) for item in expression.args]
            if all(isinstance(item, int) for item in values):
                return bool(range(*values))
        except (TypeError, ValueError):
            pass
    return None


def _unpack_target_accepts_length(
    target: ast.AST, value_length: int,
) -> bool:
    """Whether a concrete sequence length can bind one tuple/list target."""
    if not isinstance(target, (ast.Tuple, ast.List)):
        return True
    if any(isinstance(item, ast.Starred) for item in target.elts):
        return value_length >= len(target.elts) - 1
    return value_length == len(target.elts)


def _literal_unpack_outcome(
    target: ast.AST, value: ast.AST,
) -> tuple[bool, str | None]:
    """Return ``(known, exception)`` for one literal target binding."""
    if not isinstance(target, (ast.Tuple, ast.List)):
        return True, None
    if isinstance(value, (ast.Tuple, ast.List)):
        return (
            True,
            None if _unpack_target_accepts_length(
                target, len(value.elts)
            ) else "ValueError",
        )
    if isinstance(value, ast.Constant) and isinstance(
            value.value, (bool, int, float, complex)):
        return True, "TypeError"
    return False, None


def _known_eager_comprehension_failure(
    expression: ast.AST | None,
) -> str | None:
    """First deterministic target failure in a simple eager comprehension."""
    if not isinstance(expression, (ast.ListComp, ast.SetComp, ast.DictComp)) \
            or len(expression.generators) != 1:
        return None
    generator = expression.generators[0]
    if not isinstance(generator.iter, (ast.List, ast.Tuple)):
        return None
    for value in generator.iter.elts:
        known, exception = _literal_unpack_outcome(
            generator.target, value
        )
        if not known:
            return None
        if exception is not None:
            return exception
    return None


def _definition_evaluated_expressions(
    statement: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    *,
    annotations_deferred: bool = False,
) -> tuple[ast.AST, ...]:
    """Expressions evaluated before a definition statement binds its name."""
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        annotations: tuple[ast.AST, ...] = ()
        if not annotations_deferred:
            annotations = tuple(
                item.annotation
                for item in (
                    list(statement.args.posonlyargs)
                    + list(statement.args.args)
                    + list(statement.args.kwonlyargs)
                )
                if item.annotation is not None
            )
            if statement.args.vararg is not None \
                    and statement.args.vararg.annotation is not None:
                annotations += (statement.args.vararg.annotation,)
            if statement.args.kwarg is not None \
                    and statement.args.kwarg.annotation is not None:
                annotations += (statement.args.kwarg.annotation,)
            if statement.returns is not None:
                annotations += (statement.returns,)
        return (
            tuple(statement.decorator_list)
            + tuple(statement.args.defaults)
            + tuple(
                item for item in statement.args.kw_defaults
                if item is not None
            )
            + annotations
        )
    return (
        tuple(statement.decorator_list)
        + tuple(statement.bases)
        + tuple(item.value for item in statement.keywords)
    )


def _future_annotations_enabled(tree: ast.Module) -> bool:
    return any(
        isinstance(statement, ast.ImportFrom)
        and statement.module == "__future__"
        and any(alias.name == "annotations" for alias in statement.names)
        for statement in tree.body
    )


def _guaranteed_namedexpr_bindings(expression: ast.AST) -> set[str]:
    """Names certainly bound while one proven-safe expression evaluates."""
    if isinstance(expression, ast.NamedExpr):
        names = _guaranteed_namedexpr_bindings(expression.value)
        if isinstance(expression.target, ast.Name):
            names.add(expression.target.id)
        return names
    if isinstance(expression, (ast.Tuple, ast.List)):
        return set().union(*(
            _guaranteed_namedexpr_bindings(item)
            for item in expression.elts
        )) if expression.elts else set()
    return set()


def _scope_nonlocal_names(statements: Sequence[ast.stmt]) -> set[str]:
    """Collect nonlocal declarations without entering nested scopes."""
    names: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
            names.update(node.names)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

    visitor = Visitor()
    for statement in statements:
        visitor.visit(statement)
    return names


def _expression_is_definitely_non_raising(
    expression: ast.AST | None, bound_names: set[str],
) -> bool:
    if expression is None or isinstance(expression, ast.Constant):
        return True
    if isinstance(expression, ast.Name):
        return expression.id in bound_names
    if isinstance(expression, ast.NamedExpr):
        return _expression_is_definitely_non_raising(
            expression.value, bound_names
        )
    if isinstance(expression, ast.Lambda):
        defaults = tuple(expression.args.defaults) + tuple(
            item for item in expression.args.kw_defaults
            if item is not None
        )
        return all(
            _expression_is_definitely_non_raising(item, bound_names)
            for item in defaults
        )
    if isinstance(expression, (ast.Tuple, ast.List)):
        return all(
            _expression_is_definitely_non_raising(item, bound_names)
            for item in expression.elts
        )
    if isinstance(expression, ast.Starred):
        return _expression_is_definitely_non_raising(
            expression.value, bound_names
        )
    if isinstance(expression, ast.Call) \
            and isinstance(expression.func, ast.Name) \
            and expression.func.id == "range" \
            and not expression.keywords:
        return all(
            isinstance(item, ast.Constant)
            and isinstance(item.value, int)
            for item in expression.args
        )
    if isinstance(expression, ast.Call) \
            and isinstance(expression.func, ast.Name):
        candidate = getattr(builtins, expression.func.id, None)
        if isinstance(candidate, type) \
                and issubclass(candidate, BaseException):
            return all(
                _expression_is_definitely_non_raising(item, bound_names)
                for item in (
                    tuple(expression.args)
                    + tuple(keyword.value for keyword in expression.keywords)
                )
            )
    if isinstance(expression, ast.GeneratorExp):
        if not expression.generators:
            return True
        outer = expression.generators[0].iter
        return (
            isinstance(outer, (ast.Tuple, ast.List))
            and all(
                _expression_is_definitely_non_raising(item, bound_names)
                for item in outer.elts
            )
            or isinstance(outer, ast.Call)
            and _expression_is_definitely_non_raising(outer, bound_names)
        )
    if isinstance(expression, (ast.ListComp, ast.SetComp, ast.DictComp)):
        comprehension_names = set(bound_names)
        for generator in expression.generators:
            if not _expression_is_definitely_non_raising(
                    generator.iter, comprehension_names):
                return False
            iterator_nonempty = _literal_iterable_nonempty(generator.iter)
            if iterator_nonempty is False:
                return True
            target_expressions = _target_evaluation_expressions(
                generator.target
            )
            if not all(
                    _expression_is_definitely_non_raising(
                        target_expression, comprehension_names
                    )
                    for target_expression
                    in target_expressions):
                return False
            if target_expressions:
                # Attribute/subscript target binding invokes descriptor or
                # item-assignment behavior after its base reads succeed.
                return False
            if isinstance(generator.target, (ast.Tuple, ast.List)):
                if not isinstance(generator.iter, (ast.List, ast.Tuple)):
                    return False
                outcomes = [
                    _literal_unpack_outcome(generator.target, value)
                    for value in generator.iter.elts
                ]
                if not all(known and exception is None
                           for known, exception in outcomes):
                    return False
            comprehension_names.update(
                item.id for item in ast.walk(generator.target)
                if isinstance(item, ast.Name)
                and isinstance(item.ctx, ast.Store)
            )
            for condition in generator.ifs:
                if not _expression_is_definitely_non_raising(
                        condition, comprehension_names):
                    return False
                if isinstance(condition, ast.Constant) \
                        and not bool(condition.value):
                    return True
        results: tuple[ast.AST, ...] = (
            (expression.key, expression.value)
            if isinstance(expression, ast.DictComp)
            else (expression.elt,)
        )
        return all(
            _expression_is_definitely_non_raising(
                result, comprehension_names
            )
            for result in results
        )
    if isinstance(expression, ast.IfExp) \
            and isinstance(expression.test, ast.Constant):
        branch = expression.body if bool(expression.test.value) \
            else expression.orelse
        return _expression_is_definitely_non_raising(
            branch, bound_names
        )
    if isinstance(expression, ast.BoolOp):
        for item in expression.values:
            if not _expression_is_definitely_non_raising(
                    item, bound_names):
                return False
            if isinstance(item, ast.Constant):
                if isinstance(expression.op, ast.And) \
                        and not bool(item.value):
                    return True
                if isinstance(expression.op, ast.Or) \
                        and bool(item.value):
                    return True
            else:
                return False
        return True
    return False


def _expression_operation_may_raise(
    expression: ast.AST, bound_names: set[str],
) -> bool:
    """Whether the node's own operation may fail after children succeed."""
    if isinstance(expression, ast.Call):
        return not _expression_is_definitely_non_raising(
            expression, bound_names
        )
    return isinstance(expression, (
        ast.Attribute,
        ast.Subscript,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Dict,
        ast.Set,
    ))


def _statement_may_raise_implicitly(
    statement: ast.stmt,
    bound_names: set[str] | None = None,
    *,
    annotations_deferred: bool = False,
) -> bool:
    """Retain handler paths unless a statement is narrowly proven safe.

    Calls are only one source of implicit exceptions: subscription, attribute
    access, operators, iteration, and descriptor-backed assignment can all
    transfer control to an ``except`` suite.  Simple local/name copies are the
    useful non-raising case for lineage propagation; everything else stays
    conservative.
    """
    bound_names = bound_names or set()
    if isinstance(statement, (ast.Raise, ast.Assert, ast.Try)):
        return False
    if isinstance(statement, (ast.Pass, ast.Break, ast.Continue)):
        return False
    if isinstance(statement, ast.Assign):
        return not _expression_is_definitely_non_raising(
            statement.value, bound_names
        )
    if isinstance(statement, ast.AnnAssign):
        if statement.value is None:
            if isinstance(statement.target, ast.Name):
                return False
            if isinstance(statement.target, ast.Attribute):
                return not _expression_is_definitely_non_raising(
                    statement.target.value, bound_names
                )
            return True
        return not _expression_is_definitely_non_raising(
            statement.value, bound_names
        )
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if statement.decorator_list:
            return True
        definition_bound = set(bound_names)
        for expression in _definition_evaluated_expressions(
                statement,
                annotations_deferred=annotations_deferred):
            if not _expression_is_definitely_non_raising(
                    expression, definition_bound):
                return True
            definition_bound.update(
                _guaranteed_namedexpr_bindings(expression)
            )
        return False
    if isinstance(statement, ast.ClassDef):
        return bool(
            statement.decorator_list
            or statement.bases
            or statement.keywords
        ) or any(
            _statement_may_raise_implicitly(
                item,
                bound_names,
                annotations_deferred=annotations_deferred,
            )
            for item in statement.body
        )
    if isinstance(statement, ast.Expr):
        return not _expression_is_definitely_non_raising(
            statement.value, bound_names
        )
    if isinstance(statement, ast.Return):
        return not _expression_is_definitely_non_raising(
            statement.value, bound_names
        )
    if isinstance(statement, (ast.If, ast.While)):
        return not _expression_is_definitely_non_raising(
            statement.test, bound_names
        )
    if isinstance(statement, (ast.For, ast.AsyncFor)):
        return not _expression_is_definitely_non_raising(
            statement.iter, bound_names
        )
    return True


def _loop_body_may_use_partial_domain(statements: Sequence[ast.stmt]) -> bool:
    """Whether one abstract loop domain may overstate visited iterations."""
    class Visitor(ast.NodeVisitor):
        found = False
        nested_loop_depth = 0

        def visit_Break(self, node: ast.Break) -> None:
            if self.nested_loop_depth == 0:
                self.found = True

        def visit_Continue(self, node: ast.Continue) -> None:
            if self.nested_loop_depth == 0:
                self.found = True

        def visit_Return(self, node: ast.Return) -> None:
            self.found = True

        def visit_Raise(self, node: ast.Raise) -> None:
            self.found = True

        def _visit_nested_loop(self, node: ast.AST) -> None:
            self.nested_loop_depth += 1
            self.generic_visit(node)
            self.nested_loop_depth -= 1

        def visit_For(self, node: ast.For) -> None:
            self._visit_nested_loop(node)

        def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
            self._visit_nested_loop(node)

        def visit_While(self, node: ast.While) -> None:
            self._visit_nested_loop(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_ListComp(self, node: ast.ListComp) -> None:
            return

        def visit_SetComp(self, node: ast.SetComp) -> None:
            return

        def visit_DictComp(self, node: ast.DictComp) -> None:
            return

        def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

    visitor = Visitor()
    for statement in statements:
        visitor.visit(statement)
        if visitor.found:
            return True
    return False


def _match_pattern_status(pattern: ast.pattern, subject: object) -> bool | None:
    if isinstance(pattern, ast.MatchAs) and pattern.pattern is None:
        return True
    if isinstance(pattern, ast.MatchSingleton):
        if isinstance(subject, _NumberRange):
            exact = subject.exact_int()
            return exact is not None and exact == pattern.value
        return None
    if isinstance(pattern, ast.MatchValue) \
            and isinstance(pattern.value, ast.Constant):
        if isinstance(subject, _NumberRange):
            exact = subject.exact_int()
            if exact is not None:
                return exact == pattern.value.value
        return None
    if isinstance(pattern, ast.MatchOr):
        statuses = [
            _match_pattern_status(item, subject)
            for item in pattern.patterns
        ]
        if any(item is True for item in statuses):
            return True
        if all(item is False for item in statuses):
            return False
    return None


def _bind_match_pattern(
    pattern: ast.pattern, value: _Value, env: dict[str, _Value],
) -> None:
    if isinstance(pattern, ast.MatchAs):
        if pattern.name is not None:
            env[pattern.name] = value
        if pattern.pattern is not None:
            _bind_match_pattern(pattern.pattern, value, env)
    elif isinstance(pattern, ast.MatchStar) and pattern.name is not None:
        env[pattern.name] = value
    elif isinstance(pattern, (ast.MatchSequence, ast.MatchOr)):
        for item in pattern.patterns:
            _bind_match_pattern(item, value, env)
    elif isinstance(pattern, ast.MatchMapping):
        for item in pattern.patterns:
            _bind_match_pattern(item, value, env)
        if pattern.rest is not None:
            env[pattern.rest] = value
    elif isinstance(pattern, ast.MatchClass):
        for item in pattern.patterns + pattern.kwd_patterns:
            _bind_match_pattern(item, value, env)


def _is_model_builder_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    tokens = _tokens(_callee_name(node))
    return bool(
        tokens & {"build", "create", "construct", "make"}
        and tokens & {"model", "net", "network", "estimator"}
    )


def _source(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__


@dataclass(frozen=True, order=True)
class ProtocolRange:
    """A concrete half-open interval on the protocol axis."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.stop < self.start:
            raise ValueError("protocol range stop precedes start")

    def intersection(self, other: "ProtocolRange") -> "ProtocolRange | None":
        start = max(self.start, other.start)
        stop = min(self.stop, other.stop)
        return ProtocolRange(start, stop) if start < stop else None


@dataclass(frozen=True)
class RoleObservation:
    role: str
    read_kind: str
    root: str
    protocol_range: ProtocolRange
    subset: str | None
    model_id: str
    line: int
    source: str
    expression: str


@dataclass(frozen=True)
class RangeRelation:
    model_id: str
    root: str
    first_role: str
    first_read_kind: str
    second_role: str
    second_read_kind: str
    first_range: ProtocolRange
    second_range: ProtocolRange
    status: str
    overlap: ProtocolRange | None
    certainty: str


@dataclass(frozen=True)
class FormalBinding:
    formal: str
    root: str
    protocol_range: ProtocolRange | None


@dataclass(frozen=True)
class ModelBinding:
    formal: str
    actual: str


@dataclass(frozen=True)
class CallSiteSummary:
    caller: str
    callee: str
    line: int
    model_id: str
    model_formal: str | None
    model_bindings: tuple[ModelBinding, ...]
    bindings: tuple[FormalBinding, ...]
    observations: tuple[RoleObservation, ...]


@dataclass(frozen=True)
class UnresolvedLineage:
    source: str
    line: int
    expression: str
    reason: str
    role: str | None = None
    read_kind: str | None = None
    root: str | None = None
    model_id: str | None = None
    protocol_range: ProtocolRange | None = None
    subset: str | None = None


@dataclass(frozen=True)
class SplitLineageReport:
    observations: tuple[RoleObservation, ...]
    relations: tuple[RangeRelation, ...]
    calls: tuple[CallSiteSummary, ...]
    unresolved: tuple[UnresolvedLineage, ...]


@dataclass(frozen=True)
class _NumberRange:
    """Inclusive bounds for possible scalar values."""

    lower: Fraction
    upper: Fraction
    support_envelope: bool = False
    symbolic_id: str | None = None

    @classmethod
    def exact(cls, value: Union[int, float, Fraction]) -> "_NumberRange":
        if isinstance(value, float):
            value = Fraction(str(value))
        return cls(Fraction(value), Fraction(value))

    def exact_int(self) -> int | None:
        if self.lower == self.upper and self.lower.denominator == 1:
            return int(self.lower)
        return None


@dataclass(frozen=True)
class _RangeIterableValue(_NumberRange):
    """A concrete ``range`` with numeric-domain and iterable identity."""

    range_start: int = 0
    range_step: int = 1
    iterable_length: int = 0

    def item(self, index: int) -> _NumberRange:
        return _NumberRange.exact(
            self.range_start + index * self.range_step
        )


@dataclass(frozen=True)
class _TensorValue:
    root: str
    protocol_range: ProtocolRange | None
    subset: str | None = None
    lineage_id: str | None = None
    ancestor_lineage_ids: frozenset[str] = frozenset()

    @property
    def protocol_length(self) -> int | None:
        if self.protocol_range is None:
            return None
        return self.protocol_range.stop - self.protocol_range.start

    @property
    def lineage(self) -> frozenset[str]:
        if self.lineage_id is None:
            return self.ancestor_lineage_ids
        return self.ancestor_lineage_ids | {self.lineage_id}


@dataclass(frozen=True)
class _TensorAlternativesValue:
    tensors: tuple[_TensorValue, ...]


@dataclass(frozen=True)
class _ShapeValue:
    protocol_length: int | None


@dataclass(frozen=True)
class _MaskValue:
    parent: _TensorValue


@dataclass(frozen=True)
class _ModelValue:
    identity: str
    attrs: tuple[tuple[str, _NumberRange], ...]

    def attr(self, name: str) -> _NumberRange | None:
        return dict(self.attrs).get(name)


@dataclass(frozen=True)
class _NamespaceValue:
    attrs: tuple[tuple[str, object], ...]

    def attr(self, name: str) -> object:
        return dict(self.attrs).get(name, UNKNOWN)


@dataclass(frozen=True)
class _PredictionValue:
    model_id: str
    conditioning: tuple[_TensorValue, ...] = ()


@dataclass(frozen=True)
class _UnresolvedPredictionValue:
    model_id: str | None
    reason: str


@dataclass(frozen=True)
class _MetricExpressionValue:
    tensors: tuple[_TensorValue, ...]
    predictions: tuple[_PredictionValue, ...]
    unresolved_predictions: tuple[_UnresolvedPredictionValue, ...]


@dataclass(frozen=True)
class _CallableValue:
    name: str


@dataclass(frozen=True)
class _LossValue:
    tensors: tuple[_TensorValue, ...]
    model_ids: tuple[str, ...]


@dataclass(frozen=True)
class _SequenceValue:
    values: tuple[object, ...]


class _Unknown:
    pass


UNKNOWN = _Unknown()


class _Unbound:
    pass


UNBOUND = _Unbound()


@dataclass(frozen=True)
class _PossiblyUnboundValue:
    value: object


_UNEVALUATED = object()
_Value = Union[
    _NumberRange, _TensorValue, _TensorAlternativesValue, _ShapeValue,
    _MaskValue, _ModelValue,
    _NamespaceValue,
    _PredictionValue, _UnresolvedPredictionValue, _MetricExpressionValue,
    _CallableValue, _LossValue, _SequenceValue, _Unknown, _Unbound,
    _PossiblyUnboundValue,
]


@dataclass(frozen=True)
class _ImplicitRaiseState:
    environment: dict[str, _Value]
    excluded_classes: tuple[type[BaseException], ...] = ()


def _branch_support_tensor(tensor: _TensorValue) -> _TensorValue:
    return _TensorValue(
        root=tensor.root,
        protocol_range=tensor.protocol_range,
        subset=tensor.subset or "branch_support_envelope",
        lineage_id=tensor.lineage_id,
        ancestor_lineage_ids=tensor.ancestor_lineage_ids,
    )


@dataclass(frozen=True)
class _FlowOutcome:
    falls_through: bool = True
    fallthrough_environments: tuple[dict[str, _Value], ...] = ()
    returned: bool = False
    broke: bool = False
    continued: bool = False
    raised: bool = False
    raise_types: frozenset[str | None] = frozenset()
    return_completions: tuple[
        tuple[_Value | None, dict[str, _Value]], ...
    ] = ()
    implicit_raise_environments: tuple[_ImplicitRaiseState, ...] = ()
    break_environments: tuple[dict[str, _Value], ...] = ()
    continue_environments: tuple[dict[str, _Value], ...] = ()
    raise_environments: tuple[
        tuple[str | None, dict[str, _Value]], ...
    ] = ()


def _union_flow(
    *outcomes: _FlowOutcome,
    falls_through: bool | None = None,
) -> _FlowOutcome:
    return _FlowOutcome(
        falls_through=(
            any(item.falls_through for item in outcomes)
            if falls_through is None else falls_through
        ),
        fallthrough_environments=tuple(
            environment
            for outcome in outcomes
            for environment in outcome.fallthrough_environments
        ),
        returned=any(item.returned for item in outcomes),
        broke=any(item.broke for item in outcomes),
        continued=any(item.continued for item in outcomes),
        raised=any(item.raised for item in outcomes),
        raise_types=frozenset(
            item
            for outcome in outcomes
            for item in outcome.raise_types
        ),
        return_completions=tuple(
            item
            for outcome in outcomes
            for item in outcome.return_completions
        ),
        implicit_raise_environments=tuple(
            environment
            for outcome in outcomes
            for environment in outcome.implicit_raise_environments
        ),
        break_environments=tuple(
            environment
            for outcome in outcomes
            for environment in outcome.break_environments
        ),
        continue_environments=tuple(
            environment
            for outcome in outcomes
            for environment in outcome.continue_environments
        ),
        raise_environments=tuple(
            item
            for outcome in outcomes
            for item in outcome.raise_environments
        ),
    )


def _flow_with_unbound_name(
    outcome: _FlowOutcome, name: str,
) -> _FlowOutcome:
    def unbound(environment: dict[str, _Value]) -> dict[str, _Value]:
        result = dict(environment)
        result[name] = UNBOUND
        return result

    return _FlowOutcome(
        falls_through=outcome.falls_through,
        fallthrough_environments=tuple(
            unbound(environment)
            for environment in outcome.fallthrough_environments
        ),
        returned=outcome.returned,
        broke=outcome.broke,
        continued=outcome.continued,
        raised=outcome.raised,
        raise_types=outcome.raise_types,
        return_completions=tuple(
            (value, unbound(environment))
            for value, environment in outcome.return_completions
        ),
        implicit_raise_environments=tuple(
            _ImplicitRaiseState(
                unbound(state.environment), state.excluded_classes,
            )
            for state in outcome.implicit_raise_environments
        ),
        break_environments=tuple(
            unbound(environment)
            for environment in outcome.break_environments
        ),
        continue_environments=tuple(
            unbound(environment)
            for environment in outcome.continue_environments
        ),
        raise_environments=tuple(
            (exception_name, unbound(environment))
            for exception_name, environment in outcome.raise_environments
        ),
    )


def _merge_branch_values(left: _Value, right: _Value) -> _Value:
    """Join values from two reachable control-flow branches.

    Equality preserves exact facts.  A mixed metric/non-metric result keeps
    the target lineage but marks the scored value unresolved: dropping either
    side would respectively fail open or turn a possible metric into a false
    exact verdict.
    """
    if left == right:
        return left

    if isinstance(left, _Unbound):
        return right if isinstance(right, _PossiblyUnboundValue) \
            else _PossiblyUnboundValue(right)
    if isinstance(right, _Unbound):
        return left if isinstance(left, _PossiblyUnboundValue) \
            else _PossiblyUnboundValue(left)
    if isinstance(left, _PossiblyUnboundValue) \
            and isinstance(right, _PossiblyUnboundValue):
        return _PossiblyUnboundValue(_merge_branch_values(
            left.value, right.value  # type: ignore[arg-type]
        ))
    if isinstance(left, _PossiblyUnboundValue):
        return _PossiblyUnboundValue(_merge_branch_values(
            left.value, right  # type: ignore[arg-type]
        ))
    if isinstance(right, _PossiblyUnboundValue):
        return _PossiblyUnboundValue(_merge_branch_values(
            left, right.value  # type: ignore[arg-type]
        ))

    if isinstance(left, _MetricExpressionValue) \
            and isinstance(right, _MetricExpressionValue):
        if len(left.tensors) == len(right.tensors) == 1:
            target_join = _merge_branch_values(
                left.tensors[0], right.tensors[0]
            )
            if isinstance(target_join, _TensorValue):
                tensors = (target_join,)
            elif isinstance(target_join, _TensorAlternativesValue):
                tensors = target_join.tensors
            else:
                tensors = tuple(
                    _branch_support_tensor(item)
                    for item in left.tensors + right.tensors
                )
        else:
            tensors = tuple(
                _branch_support_tensor(item)
                for item in left.tensors + right.tensors
            )
        predictions = tuple(dict.fromkeys(
            left.predictions + right.predictions
        ))
        unresolved_predictions = tuple(dict.fromkeys(
            left.unresolved_predictions + right.unresolved_predictions
        ))
        return _MetricExpressionValue(
            tensors, predictions, unresolved_predictions,
        )

    if isinstance(left, _MetricExpressionValue) \
            or isinstance(right, _MetricExpressionValue):
        merged = _merge_metric_expression_values(left, right)
        if merged is not None:
            model_ids = {
                item.model_id for item in merged.predictions
            } | {
                item.model_id for item in merged.unresolved_predictions
                if item.model_id is not None
            }
            model_id = next(iter(model_ids)) if len(model_ids) == 1 else None
            return _MetricExpressionValue(
                merged.tensors,
                merged.predictions,
                merged.unresolved_predictions + (
                    _UnresolvedPredictionValue(
                        model_id,
                        "conditional return can select a non-metric value",
                    ),
                ),
            )

    prediction_values = [
        value for value in (left, right)
        if isinstance(value, (_PredictionValue, _UnresolvedPredictionValue))
    ]
    if prediction_values:
        model_ids = {
            value.model_id for value in prediction_values
            if value.model_id is not None
        }
        return _UnresolvedPredictionValue(
            next(iter(model_ids)) if len(model_ids) == 1 else None,
            "conditional return prediction lineage is unresolved",
        )

    if isinstance(left, _NumberRange) and isinstance(right, _NumberRange):
        return _NumberRange(
            min(left.lower, right.lower),
            max(left.upper, right.upper),
            True,
        )

    if isinstance(left, _TensorValue) and isinstance(right, _TensorValue) \
            and left.root == right.root:
        protocol_range = None
        if left.protocol_range is not None and right.protocol_range is not None:
            protocol_range = ProtocolRange(
                min(left.protocol_range.start, right.protocol_range.start),
                max(left.protocol_range.stop, right.protocol_range.stop),
            )
        return _TensorValue(
            root=left.root,
            protocol_range=protocol_range,
            subset="branch_support_envelope",
            ancestor_lineage_ids=left.lineage & right.lineage,
        )

    def tensor_alternatives(value: _Value) -> tuple[_TensorValue, ...]:
        if isinstance(value, _TensorValue):
            return (value,)
        if isinstance(value, _TensorAlternativesValue):
            return value.tensors
        return ()

    left_tensors = tensor_alternatives(left)
    right_tensors = tensor_alternatives(right)
    if left_tensors and right_tensors:
        unique = {
            (item.root, item.protocol_range, item.lineage_id):
            _branch_support_tensor(item)
            for item in left_tensors + right_tensors
        }
        return _TensorAlternativesValue(tuple(unique.values()))
    retained_tensors = left_tensors or right_tensors
    if retained_tensors:
        unresolved = tuple(
            _TensorValue(
                root=item.root,
                protocol_range=None,
                subset="branch_support_envelope",
                ancestor_lineage_ids=item.lineage,
            )
            for item in retained_tensors
        )
        return unresolved[0] if len(unresolved) == 1 \
            else _TensorAlternativesValue(unresolved)

    if isinstance(left, _SequenceValue) and isinstance(right, _SequenceValue) \
            and len(left.values) == len(right.values):
        return _SequenceValue(tuple(
            _merge_branch_values(first, second)
            if isinstance(first, (
                _NumberRange, _TensorValue, _TensorAlternativesValue,
                _ShapeValue, _MaskValue,
                _ModelValue, _PredictionValue, _UnresolvedPredictionValue,
                _MetricExpressionValue, _CallableValue, _LossValue,
                _SequenceValue, _Unknown, _Unbound,
                _PossiblyUnboundValue,
            )) and isinstance(second, (
                _NumberRange, _TensorValue, _TensorAlternativesValue,
                _ShapeValue, _MaskValue,
                _ModelValue, _PredictionValue, _UnresolvedPredictionValue,
                _MetricExpressionValue, _CallableValue, _LossValue,
                _SequenceValue, _Unknown, _Unbound,
                _PossiblyUnboundValue,
            )) else UNKNOWN
            for first, second in zip(left.values, right.values)
        ))

    return UNKNOWN


def _number(value: object) -> _NumberRange | None:
    return value if isinstance(value, _NumberRange) else None


def _loop_escape_support(value: _Value) -> _Value:
    """Downgrade a loop-union value that escapes its iteration context."""
    if isinstance(value, _PossiblyUnboundValue):
        return _PossiblyUnboundValue(_loop_escape_support(
            value.value  # type: ignore[arg-type]
        ))
    if isinstance(value, _Unbound):
        return value
    if isinstance(value, _NumberRange):
        return _NumberRange(
            value.lower, value.upper, True, value.symbolic_id,
        )
    if isinstance(value, _TensorValue):
        return _TensorValue(
            root=value.root,
            protocol_range=value.protocol_range,
            subset=value.subset or "loop_escape_envelope",
            lineage_id=value.lineage_id,
            ancestor_lineage_ids=value.ancestor_lineage_ids,
        )
    if isinstance(value, _TensorAlternativesValue):
        return _TensorAlternativesValue(tuple(
            _loop_escape_support(item)
            for item in value.tensors
        ))
    if isinstance(value, _MaskValue):
        parent = _loop_escape_support(value.parent)
        return _MaskValue(parent) \
            if isinstance(parent, _TensorValue) else value
    if isinstance(value, _PredictionValue):
        conditioning = tuple(
            item for item in (
                _loop_escape_support(tensor)
                for tensor in value.conditioning
            ) if isinstance(item, _TensorValue)
        )
        return _PredictionValue(value.model_id, conditioning)
    if isinstance(value, (_MetricExpressionValue, _LossValue)):
        # These values aggregate every reached loop contribution.  Their
        # constituent tensors already carry support when the iteration domain
        # itself may terminate early; a normal full loop remains exact.
        return value
    if isinstance(value, _SequenceValue):
        return _SequenceValue(tuple(
            _loop_escape_support(item)
            if isinstance(item, (
                _NumberRange, _TensorValue, _TensorAlternativesValue,
                _MaskValue, _PredictionValue, _MetricExpressionValue,
                _LossValue, _SequenceValue,
            )) else item
            for item in value.values
        ))
    return value


def _loop_escape_environment(
    entry: Mapping[str, _Value], candidate: Mapping[str, _Value],
    *, multi_valued: bool,
) -> dict[str, _Value]:
    escaped = dict(candidate)
    if not multi_valued:
        return escaped
    missing = object()
    for name, value in tuple(escaped.items()):
        if entry.get(name, missing) != value:
            escaped[name] = _loop_escape_support(value)
    return escaped


def _add_numbers(
    left: _NumberRange, right: _NumberRange, *, subtract: bool = False,
) -> _NumberRange:
    same_symbolic_value = (
        left.symbolic_id is not None
        and left.symbolic_id == right.symbolic_id
    )
    same_exact_value = (
        left.lower == left.upper == right.lower == right.upper
    )
    if subtract:
        if same_symbolic_value or same_exact_value:
            return _NumberRange.exact(0)
        if right.lower == right.upper == 0:
            return left
        return _NumberRange(
            left.lower - right.upper,
            left.upper - right.lower,
            left.support_envelope or right.support_envelope,
        )
    if right.lower == right.upper == 0:
        return left
    if left.lower == left.upper == 0:
        return right
    correlated_sparsity = (
        same_symbolic_value and left.lower != left.upper
    )
    return _NumberRange(
        left.lower + right.lower,
        left.upper + right.upper,
        left.support_envelope or right.support_envelope
        or correlated_sparsity,
    )


def _multiply_numbers(
    left: _NumberRange, right: _NumberRange,
) -> _NumberRange:
    if right.lower == right.upper == 1:
        return left
    if left.lower == left.upper == 1:
        return right
    if right.lower == right.upper == 0 \
            or left.lower == left.upper == 0:
        return _NumberRange.exact(0)
    products = (
        left.lower * right.lower,
        left.lower * right.upper,
        left.upper * right.lower,
        left.upper * right.upper,
    )
    left_varies = left.lower != left.upper
    right_varies = right.lower != right.upper
    left_scale = left.lower if not left_varies else None
    right_scale = right.lower if not right_varies else None
    affine_sparsity = (
        left_varies and right_scale not in (None, -1, 0, 1)
        or right_varies and left_scale not in (None, -1, 0, 1)
        or left_varies and right_varies
    )
    return _NumberRange(
        min(products), max(products),
        left.support_envelope or right.support_envelope or affine_sparsity,
    )


def _divide_numbers(
    left: _NumberRange, right: _NumberRange,
) -> _NumberRange | None:
    if right.lower == right.upper == 1:
        return left
    if right.lower <= 0 <= right.upper:
        return None
    reciprocals = _NumberRange(
        1 / right.upper, 1 / right.lower, right.support_envelope,
    )
    return _multiply_numbers(left, reciprocals)


def _merge_losses(left: object, right: object) -> _LossValue | None:
    losses = [value for value in (left, right) if isinstance(value, _LossValue)]
    if not losses:
        return None
    tensors: list[_TensorValue] = []
    for loss in losses:
        tensors.extend(loss.tensors)
    model_ids = {
        model_id
        for loss in losses
        for model_id in loss.model_ids
    }
    return _LossValue(tuple(tensors), tuple(sorted(model_ids)))


def _merge_metric_expression_values(
    *values: object,
) -> _MetricExpressionValue | None:
    tensors: list[_TensorValue] = []
    predictions: list[_PredictionValue] = []
    unresolved_predictions: list[_UnresolvedPredictionValue] = []
    for value in values:
        if isinstance(value, _TensorValue):
            tensors.append(value)
        elif isinstance(value, _TensorAlternativesValue):
            tensors.extend(value.tensors)
        elif isinstance(value, _PredictionValue):
            predictions.append(value)
        elif isinstance(value, _UnresolvedPredictionValue):
            unresolved_predictions.append(value)
        elif isinstance(value, _MetricExpressionValue):
            tensors.extend(value.tensors)
            predictions.extend(value.predictions)
            unresolved_predictions.extend(value.unresolved_predictions)
    if not tensors and not predictions and not unresolved_predictions:
        return None
    return _MetricExpressionValue(
        tuple(tensors), tuple(predictions), tuple(unresolved_predictions),
    )


def _parse(source: Union[str, ast.Module]) -> ast.Module:
    return source if isinstance(source, ast.Module) else ast.parse(source)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return node  # type: ignore[return-value]
    return None


def _function_local_names(function: ast.FunctionDef) -> set[str]:
    names = {
        argument.arg
        for argument in (
            list(function.args.posonlyargs)
            + list(function.args.args)
            + list(function.args.kwonlyargs)
        )
    }

    class Visitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                names.add(node.id)

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            if node.name is not None:
                names.add(node.name)
            for statement in node.body:
                self.visit(statement)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            names.add(node.name)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            names.add(node.name)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            names.add(node.name)

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_ListComp(self, node: ast.ListComp) -> None:
            return

        def visit_SetComp(self, node: ast.SetComp) -> None:
            return

        def visit_DictComp(self, node: ast.DictComp) -> None:
            return

        def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
            return

    visitor = Visitor()
    for statement in function.body:
        visitor.visit(statement)
    return names

def _captures_model_state(statements: Sequence[ast.stmt]) -> bool:
    """Whether a branch snapshots or restores model parameters."""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"state_dict", "load_state_dict"}
        for statement in statements
        for node in ast.walk(statement)
    )


class _Analyzer:
    def __init__(
        self,
        *,
        trainers: Mapping[str, ast.Module],
        constants: Mapping[str, object],
        observations: list[RoleObservation] | None = None,
        calls: list[CallSiteSummary] | None = None,
        unresolved: list[UnresolvedLineage] | None = None,
        observe_reported_metrics: bool = False,
        observe_returned_metrics: bool = False,
        project_child_observations: bool = False,
        conditioning_values: set[tuple[str, str, str]] | None = None,
        lineage_context: str = "root",
    ) -> None:
        self.trainers = trainers
        self.constants = constants
        self.observations = observations if observations is not None else []
        self.calls = calls if calls is not None else []
        self.unresolved = unresolved if unresolved is not None else []
        self.observe_reported_metrics = observe_reported_metrics
        self.observe_returned_metrics = observe_returned_metrics
        self.project_child_observations = project_child_observations
        self.conditioning_values = (
            conditioning_values
            if conditioning_values is not None else set()
        )
        self.lineage_context = lineage_context
        self.return_values: list[_Value] = []
        self._pending_expression_raises: list[
            tuple[str, dict[str, _Value]]
        ] = []
        self._pending_expression_implicit: list[_ImplicitRaiseState] = []
        self.local_names: set[str] = set()
        self.scope = "<module>"
        self.source_label = "notebook"
        self.active_model = "model"
        self.future_annotations = False

    def _model_from_constants(self, name: str) -> _ModelValue:
        attrs: dict[str, _NumberRange] = {}
        # Generic ``model.*`` facts describe a conventional builder result.
        # Binding-specific facts override them when independently constructed
        # models have distinct configuration.
        prefixes = ("model.", f"{name}.") if name != "model" else ("model.",)
        for prefix in prefixes:
            for key, raw in self.constants.items():
                if key.startswith(prefix) \
                        and key != f"{prefix}protocol_length" \
                        and isinstance(raw, (int, float, Fraction)) \
                        and not isinstance(raw, bool):
                    attrs[key[len(prefix):]] = _NumberRange.exact(raw)
        return _ModelValue(name, tuple(sorted(attrs.items())))

    def _initial_value(self, name: str) -> _Value:
        raw = self.constants.get(name)
        if isinstance(raw, (int, float, Fraction)) and not isinstance(raw, bool):
            return _NumberRange.exact(raw)
        length = self.constants.get(f"{name}.protocol_length")
        if isinstance(length, int) and not isinstance(length, bool) and length >= 0:
            return _TensorValue(
                name,
                ProtocolRange(0, length),
                lineage_id=f"root:{name}",
            )
        if name in self.trainers or name in {"forecast", "predict", "infer"}:
            return _CallableValue(name)
        if _tokens(name) & {"model", "net", "network", "estimator"}:
            return self._model_from_constants(name)
        return UNKNOWN

    def _entry_env(self, func: ast.FunctionDef) -> dict[str, _Value]:
        env: dict[str, _Value] = {}
        positional = list(func.args.posonlyargs) + list(func.args.args)
        keyword_only = list(func.args.kwonlyargs)
        for arg in positional + keyword_only:
            env[arg.arg] = self._initial_value(arg.arg)
        default_start = len(positional) - len(func.args.defaults)
        for index, default in enumerate(func.args.defaults, start=default_start):
            name = positional[index].arg
            if env[name] is UNKNOWN:
                env[name] = self._eval(default, env)
        for arg, default in zip(keyword_only, func.args.kw_defaults):
            if default is not None and env[arg.arg] is UNKNOWN:
                env[arg.arg] = self._eval(default, env)
        return env

    def _record_unresolved(
        self,
        node: ast.AST,
        reason: str,
        *,
        role: str | None = None,
        read_kind: str | None = None,
        root: str | None = None,
        model_id: str | None = None,
        protocol_range: ProtocolRange | None = None,
        subset: str | None = None,
    ) -> None:
        self.unresolved.append(UnresolvedLineage(
            source=self.source_label,
            line=getattr(node, "lineno", 0),
            expression=_source(node),
            reason=reason,
            role=role,
            read_kind=read_kind,
            root=root,
            model_id=model_id,
            protocol_range=protocol_range,
            subset=subset,
        ))

    def _eval(self, node: ast.AST, env: dict[str, _Value]) -> _Value:
        if isinstance(node, ast.NamedExpr):
            value = self._eval(node.value, env)
            self._assign_target(node.target, value, env)
            return value
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return _NumberRange.exact(int(node.value))
            if isinstance(node.value, (int, float, Fraction)) \
                    and not isinstance(node.value, bool):
                return _NumberRange.exact(node.value)
            return UNKNOWN
        if isinstance(node, ast.Name):
            value = env.get(node.id, self._initial_value(node.id))
            if isinstance(value, _PossiblyUnboundValue):
                return value.value  # type: ignore[return-value]
            if isinstance(value, _Unbound):
                return UNKNOWN
            return value
        if isinstance(node, ast.Tuple):
            return _SequenceValue(tuple(self._eval(item, env) for item in node.elts))
        if isinstance(node, ast.List):
            return _SequenceValue(tuple(self._eval(item, env) for item in node.elts))
        if isinstance(node, ast.Attribute):
            base = self._eval(node.value, env)
            if isinstance(base, _NamespaceValue):
                return base.attr(node.attr)  # type: ignore[return-value]
            if isinstance(base, _TensorValue) and node.attr == "shape":
                return _ShapeValue(base.protocol_length)
            if isinstance(base, _ModelValue):
                return base.attr(node.attr) or UNKNOWN
            if isinstance(base, (
                    _PredictionValue, _UnresolvedPredictionValue,
                    _MetricExpressionValue)):
                return base
            if isinstance(base, (_TensorValue, _LossValue)) \
                    and node.attr in _TRANSPARENT_METHODS:
                return base
            if isinstance(base, _TensorAlternativesValue) \
                    and node.attr in _TRANSPARENT_METHODS:
                return base
            return UNKNOWN
        if isinstance(node, ast.Subscript):
            base = self._eval(node.value, env)
            if isinstance(base, _ShapeValue):
                index = self._eval(node.slice, env)
                idx = index.exact_int() if isinstance(index, _NumberRange) else None
                if idx in (-1, 1):
                    return (_NumberRange.exact(base.protocol_length)
                            if base.protocol_length is not None else UNKNOWN)
                return UNKNOWN
            if isinstance(base, _TensorValue):
                return self._slice_tensor(base, node.slice, env)
            if isinstance(base, _TensorAlternativesValue):
                return _TensorAlternativesValue(tuple(
                    self._slice_tensor(item, node.slice, env)
                    for item in base.tensors
                ))
            if isinstance(base, (
                    _PredictionValue, _UnresolvedPredictionValue,
                    _MetricExpressionValue)):
                return base
            return UNKNOWN
        if isinstance(node, ast.UnaryOp):
            value = _number(self._eval(node.operand, env))
            if value is None:
                return UNKNOWN
            if isinstance(node.op, ast.USub):
                return _NumberRange(
                    -value.upper,
                    -value.lower,
                    value.support_envelope,
                    (
                        f"neg({value.symbolic_id})"
                        if value.symbolic_id is not None else None
                    ),
                )
            if isinstance(node.op, ast.UAdd):
                return value
            return UNKNOWN
        if isinstance(node, ast.IfExp):
            condition = self._condition(node.test, env)
            if condition is True:
                normal, _ = self._unbound_paths(node.body, env)
                return self._eval(node.body, env) if normal else UNKNOWN
            if condition is False:
                normal, _ = self._unbound_paths(node.orelse, env)
                return self._eval(node.orelse, env) if normal else UNKNOWN
            values = [
                self._eval(branch, dict(env))
                for branch in (node.body, node.orelse)
                if self._unbound_paths(branch, env)[0]
            ]
            if not values:
                return UNKNOWN
            result = values[0]
            for value in values[1:]:
                result = _merge_branch_values(result, value)
            return result
        if isinstance(node, ast.BoolOp):
            def evaluate_from(index: int) -> _Value:
                normal, _ = self._unbound_paths(
                    node.values[index], env
                )
                if not normal:
                    return UNKNOWN
                value = self._eval(node.values[index], env)
                if index == len(node.values) - 1:
                    return value
                condition = self._condition(node.values[index], env)
                short_circuits = (
                    isinstance(node.op, ast.And) and condition is False
                    or isinstance(node.op, ast.Or) and condition is True
                )
                if short_circuits:
                    return value
                continues = (
                    isinstance(node.op, ast.And) and condition is True
                    or isinstance(node.op, ast.Or) and condition is False
                )
                remainder = evaluate_from(index + 1)
                if continues:
                    return remainder
                remainder_normal, _ = self._unbound_paths(
                    ast.BoolOp(
                        op=node.op,
                        values=node.values[index + 1:],
                    ),
                    env,
                )
                return _merge_branch_values(value, remainder) \
                    if remainder_normal else value

            return evaluate_from(0) if node.values else UNKNOWN
        if isinstance(node, ast.BinOp):
            left = self._eval(node.left, env)
            right = self._eval(node.right, env)
            loss = _merge_losses(left, right)
            if loss is not None:
                return loss
            metric_expression = _merge_metric_expression_values(left, right)
            if metric_expression is not None:
                return metric_expression
            left_num, right_num = _number(left), _number(right)
            if left_num is None or right_num is None:
                return UNKNOWN
            if isinstance(node.op, ast.Add):
                return _add_numbers(left_num, right_num)
            if isinstance(node.op, ast.Sub):
                return _add_numbers(left_num, right_num, subtract=True)
            if isinstance(node.op, ast.Mult):
                return _multiply_numbers(left_num, right_num)
            if isinstance(node.op, (ast.Div, ast.FloorDiv)):
                divided = _divide_numbers(left_num, right_num)
                if divided is None:
                    return UNKNOWN
                if isinstance(node.op, ast.FloorDiv):
                    return _NumberRange(
                        Fraction(divided.lower.numerator // divided.lower.denominator),
                        Fraction(divided.upper.numerator // divided.upper.denominator),
                        divided.support_envelope,
                    )
                return divided
            return UNKNOWN
        if isinstance(node, ast.Call):
            return self._eval_call(node, env)
        return UNKNOWN

    def _slice_tensor(
        self, base: _TensorValue, slice_node: ast.AST,
        env: dict[str, _Value],
    ) -> _TensorValue:
        derived_id = (
            f"{self.lineage_context}:{base.root}:{base.lineage_id}:"
            f"{self.source_label}:{self.scope}:"
            f"{getattr(slice_node, 'lineno', 0)}:"
            f"{getattr(slice_node, 'col_offset', 0)}:{_source(slice_node)}"
        )

        def derived(
            protocol_range: ProtocolRange | None,
            subset: str | None,
        ) -> _TensorValue:
            return _TensorValue(
                base.root,
                protocol_range,
                subset,
                lineage_id=derived_id,
                ancestor_lineage_ids=base.lineage,
            )

        parts = list(slice_node.elts) if isinstance(slice_node, ast.Tuple) \
            else [slice_node]
        protocol_part = parts[-1]
        protocol_index = self._eval(protocol_part, env)
        if isinstance(protocol_index, _MaskValue):
            return derived(
                base.protocol_range, base.subset or "boolean_mask",
            )

        protocol_slice = protocol_part \
            if isinstance(protocol_part, ast.Slice) else None
        if protocol_slice is None:
            # A tuple's last selector is the protocol axis.  Preserve row-only
            # gathers such as ``x[idx]`` and ``x[idx, :]``.  A concrete
            # protocol-axis gather becomes its inclusive support envelope;
            # unsupported protocol indices remain unresolved.
            if len(parts) == 1:
                return base
            if isinstance(protocol_index, _NumberRange) \
                    and base.protocol_range is not None:
                base_length = base.protocol_length
                if base_length is None:
                    return derived(None, base.subset)
                if protocol_index.upper < 0:
                    lower = Fraction(base_length) + protocol_index.lower
                    upper = Fraction(base_length) + protocol_index.upper
                elif protocol_index.lower >= 0:
                    lower = protocol_index.lower
                    upper = protocol_index.upper
                else:
                    return derived(None, base.subset)
                if lower < 0 or upper >= base_length:
                    return derived(None, base.subset)
                start = (
                    base.protocol_range.start + int(lower.__floor__())
                )
                stop = (
                    base.protocol_range.start + int(upper.__ceil__()) + 1
                )
                subset = base.subset
                if protocol_index.support_envelope:
                    subset = subset or "sparse_index_envelope"
                return derived(
                    ProtocolRange(start, max(start, stop)), subset,
                )
            return derived(None, base.subset)

        if protocol_slice.step is not None:
            return derived(None, base.subset)
        if base.protocol_range is None:
            return derived(None, base.subset)

        if protocol_slice.lower is None:
            lower = _NumberRange.exact(0)
        else:
            lower = _number(self._eval(protocol_slice.lower, env))
        if protocol_slice.upper is None:
            upper = _NumberRange.exact(base.protocol_length or 0)
        else:
            upper = _number(self._eval(protocol_slice.upper, env))
        if lower is None or upper is None:
            return derived(None, base.subset)
        base_length = base.protocol_length
        if base_length is None:
            return derived(None, base.subset)

        def relative_bound(
            value: _NumberRange,
        ) -> tuple[Fraction, Fraction] | None:
            if value.upper < 0:
                return (
                    Fraction(base_length) + value.lower,
                    Fraction(base_length) + value.upper,
                )
            if value.lower >= 0:
                return value.lower, value.upper
            # A support envelope spanning negative and non-negative indices
            # is discontinuous under Python's relative-index semantics.  Do
            # not manufacture a precise interval for it.
            return None

        lower_bounds = relative_bound(lower)
        upper_bounds = relative_bound(upper)
        if lower_bounds is None or upper_bounds is None:
            return derived(None, base.subset)
        relative_start = max(
            0, min(base_length, int(lower_bounds[0].__floor__()))
        )
        relative_stop = max(
            0, min(base_length, int(upper_bounds[1].__ceil__()))
        )
        start = base.protocol_range.start + relative_start
        stop = min(base.protocol_range.start + relative_stop,
                   base.protocol_range.stop)
        if stop < start:
            return derived(None, base.subset)
        subset = base.subset
        if lower.support_envelope or upper.support_envelope:
            subset = subset or "sparse_index_envelope"
        return derived(ProtocolRange(start, stop), subset)

    def _range_call(self, node: ast.Call, env: dict[str, _Value]) -> _Value:
        values = [_number(self._eval(arg, env)) for arg in node.args]
        if any(value is None or value.exact_int() is None for value in values):
            return UNKNOWN
        ints = [value.exact_int() for value in values if value is not None]
        if len(ints) == 1:
            start, stop, step = 0, ints[0], 1
        elif len(ints) == 2:
            start, stop, step = ints[0], ints[1], 1
        elif len(ints) == 3:
            start, stop, step = ints
        else:
            return UNKNOWN
        if step is None or step == 0 or start is None or stop is None:
            return UNKNOWN
        values_range = range(start, stop, step)
        if len(values_range) == 0:
            return _SequenceValue(())
        return _RangeIterableValue(
            Fraction(min(values_range[0], values_range[-1])),
            Fraction(max(values_range[0], values_range[-1])),
            abs(step) != 1,
            None,
            start,
            step,
            len(values_range),
        )

    def _loss_value(
        self,
        node: ast.Call,
        env: dict[str, _Value],
        model_id: str | None = None,
    ) -> _LossValue:
        tensors: list[_TensorValue] = []
        prediction_model_ids: set[str] = set()
        args = list(node.args) + [kw.value for kw in node.keywords]
        for arg in args:
            value = self._eval(arg, env)
            # A loss call is already the semantic sink.  Direct tensor
            # arguments are targets regardless of their local spelling;
            # prediction arguments retain their separate value type.
            if isinstance(value, _TensorValue):
                tensors.append(value)
            elif isinstance(value, _TensorAlternativesValue):
                tensors.extend(value.tensors)
            elif isinstance(value, _PredictionValue):
                prediction_model_ids.add(value.model_id)
            elif isinstance(value, _MetricExpressionValue):
                prediction_model_ids.update(
                    item.model_id for item in value.predictions
                )
        model_ids = (
            {model_id}
            if model_id is not None else prediction_model_ids
        )
        if not model_ids:
            model_ids = {self.active_model}
        return _LossValue(
            tuple(tensors), tuple(sorted(model_ids)),
        )

    def _model_prediction_call(
        self,
        node: ast.Call,
        env: dict[str, _Value],
        model: _ModelValue,
    ) -> _PredictionValue:
        """Retain inputs so a prediction-returning helper can expose them."""
        arguments = list(node.args) + [item.value for item in node.keywords]
        conditioning: list[_TensorValue] = []
        for argument in arguments:
            value = self._eval(argument, env)
            if isinstance(value, _TensorValue):
                conditioning.append(value)
            elif isinstance(value, _TensorAlternativesValue):
                conditioning.extend(value.tensors)
        return _PredictionValue(model.identity, tuple(conditioning))

    def _eval_call(self, node: ast.Call, env: dict[str, _Value]) -> _Value:
        name = _callee_name(node)
        tokens = _tokens(name)
        if _is_model_builder_call(node):
            # Generated notebooks conventionally bind ``model =
            # build_model(...)`` before calling a trainer.  Treat the builder
            # result as the model itself, rather than misclassifying the
            # callable name ``build_model`` as a prediction-bearing model.
            return self._model_from_constants(f"model_at_line_{node.lineno}")
        if name == "int" and node.args:
            value = _number(self._eval(node.args[0], env))
            if value is None:
                return UNKNOWN
            return _NumberRange(
                Fraction(int(value.lower)), Fraction(int(value.upper)),
                value.support_envelope,
                value.symbolic_id,
            )
        if name in {"min", "max"} and node.args:
            values = [_number(self._eval(arg, env)) for arg in node.args]
            if any(value is None for value in values):
                return UNKNOWN
            concrete = [value for value in values if value is not None]
            pick = min if name == "min" else max
            return _NumberRange(
                pick(value.lower for value in concrete),
                pick(value.upper for value in concrete),
                any(value.support_envelope for value in concrete),
            )
        if name == "range":
            return self._range_call(node, env)
        if name == "arange":
            return self._range_call(node, env)
        if name == "randperm" and node.args:
            stop = _number(self._eval(node.args[0], env))
            stop_int = stop.exact_int() if stop is not None else None
            if stop_int is not None and stop_int > 0:
                return _NumberRange(
                    Fraction(0), Fraction(stop_int - 1),
                    stop.support_envelope if stop is not None else False,
                )
            return UNKNOWN
        if name == "len" and node.args:
            value = self._eval(node.args[0], env)
            if isinstance(value, _TensorValue) and value.subset is None \
                    and value.protocol_length is not None:
                return _NumberRange.exact(value.protocol_length)
            return UNKNOWN
        if name == "isfinite" and node.args:
            value = self._eval(node.args[0], env)
            return _MaskValue(value) if isinstance(value, _TensorValue) else UNKNOWN
        if name in _TRANSPARENT_CALLS and node.args:
            return self._eval(node.args[0], env)
        if isinstance(node.func, ast.Attribute):
            base = self._eval(node.func.value, env)
            if isinstance(base, _ModelValue) and name in {
                    "to", "cpu", "cuda", "train", "eval"}:
                return base
            if isinstance(base, _LossValue) and name in {"mean", "sum", "item"}:
                return base
            if name in _TRANSPARENT_METHODS and isinstance(
                    base, (_TensorValue, _TensorAlternativesValue,
                           _LossValue, _PredictionValue,
                           _UnresolvedPredictionValue,
                           _MetricExpressionValue)):
                return base
            if isinstance(base, _ModelValue) and tokens & _LOSS_TOKENS:
                return self._loss_value(node, env, base.identity)
            if isinstance(base, _ModelValue) and name in {"forward", "__call__"}:
                return self._model_prediction_call(node, env, base)
        if isinstance(node.func, ast.Name):
            callee_value = env.get(node.func.id, UNKNOWN)
            if callee_value is UNKNOWN \
                    and not _tokens(node.func.id) & _METRIC_TOKENS:
                callee_value = self._initial_value(node.func.id)
            if isinstance(callee_value, _ModelValue):
                return self._model_prediction_call(node, env, callee_value)
            if node.func.id in self.trainers or isinstance(callee_value, _CallableValue) \
                    and callee_value.name in self.trainers:
                return self._call_trainer(node, env, callee_value)
        if tokens & _LOSS_TOKENS:
            return self._loss_value(node, env)
        if tokens & _FORECAST_TOKENS:
            return self._forecast_call(node, env)
        argument_values = [self._eval(arg, env) for arg in node.args]
        argument_values.extend(
            self._eval(keyword.value, env) for keyword in node.keywords
        )
        non_evaluation_call = bool(tokens & _NON_EVALUATION_CALL_TOKENS)
        models = [value for value in argument_values
                  if isinstance(value, _ModelValue)]
        if models:
            model_ids = {model.identity for model in models}
            unresolved_prediction = _UnresolvedPredictionValue(
                next(iter(model_ids)) if len(model_ids) == 1 else None,
                f"unsupported model-consuming call `{name or '<call>'}`",
            )
            if tokens & _METRIC_TOKENS:
                metric_expression = _merge_metric_expression_values(
                    *argument_values, unresolved_prediction,
                )
                if metric_expression is not None \
                        and metric_expression.tensors:
                    return metric_expression
            return unresolved_prediction
        metric_expression = _merge_metric_expression_values(*argument_values)
        if metric_expression is not None \
                and metric_expression.tensors \
                and (metric_expression.predictions
                     or metric_expression.unresolved_predictions) \
                and not non_evaluation_call:
            # A neutral scalar helper such as ``compare(prediction, actual)``
            # must retain the same metric lineage after a rename.  Known
            # display/mutation calls are deliberately excluded: rendering a
            # conditioning history beside a forecast is not evaluation.
            return metric_expression
        return UNKNOWN

    def _actual_arguments(
        self, call: ast.Call, func: ast.FunctionDef,
        env: dict[str, _Value],
    ) -> dict[str, _Value]:
        positional = list(func.args.posonlyargs) + list(func.args.args)
        bound: dict[str, _Value] = {}
        for formal, actual in zip(positional, call.args):
            value = self._eval(actual, env)
            if value is UNKNOWN:
                value = self._initial_value(formal.arg)
                if value is not UNKNOWN and isinstance(actual, ast.Name):
                    # The producer contract identifies an otherwise opaque
                    # caller binding as the trainer's temporal formal.  Keep
                    # that alias for later reported-evaluation reads.
                    env[actual.id] = value
            bound[formal.arg] = value
        for keyword in call.keywords:
            if keyword.arg:
                value = self._eval(keyword.value, env)
                if value is UNKNOWN:
                    value = self._initial_value(keyword.arg)
                    if value is not UNKNOWN \
                            and isinstance(keyword.value, ast.Name):
                        env[keyword.value.id] = value
                bound[keyword.arg] = value

        positional_defaults = list(func.args.defaults)
        default_start = len(positional) - len(positional_defaults)
        for index, formal in enumerate(positional):
            if formal.arg in bound:
                continue
            if index >= default_start:
                bound[formal.arg] = self._eval(
                    positional_defaults[index - default_start], env)
        for formal, default in zip(func.args.kwonlyargs, func.args.kw_defaults):
            if formal.arg not in bound and default is not None:
                bound[formal.arg] = self._eval(default, env)
        for formal in positional + list(func.args.kwonlyargs):
            bound.setdefault(formal.arg, self._initial_value(formal.arg))
        return bound

    def _call_trainer(
        self, node: ast.Call, env: dict[str, _Value],
        callee_value: _Value,
    ) -> _Value:
        name = (callee_value.name if isinstance(callee_value, _CallableValue)
                else _callee_name(node))
        tree = self.trainers.get(name)
        func = _function(tree, name) if tree is not None else None
        if func is None:
            self._record_unresolved(node, f"trainer `{name}` has no function body")
            return UNKNOWN
        bound = self._actual_arguments(node, func, env)
        formal_names = {
            arg.arg
            for arg in (list(func.args.posonlyargs) + list(func.args.args)
                        + list(func.args.kwonlyargs))
        }
        bound_models = [
            (formal, value)
            for formal, value in bound.items()
            if isinstance(value, _ModelValue)
        ]
        model_binding = bound_models[0] if bound_models else None
        model = model_binding[1] if model_binding is not None else None
        if model is None:
            self._record_unresolved(node, f"trainer `{name}` has no model binding")
            model = self._model_from_constants("model")
        before = len(self.observations)
        child_observations = [] if self.project_child_observations \
            else self.observations
        child_unresolved = [] if self.project_child_observations \
            else self.unresolved
        child = _Analyzer(
            trainers=self.trainers,
            constants=self.constants,
            observations=child_observations,
            calls=self.calls,
            unresolved=child_unresolved,
            # A helper computes a metric value; its reachable caller decides
            # whether that value is reported evaluation or training progress.
            observe_reported_metrics=False,
            observe_returned_metrics=False,
            conditioning_values=self.conditioning_values,
            lineage_context=(
                f"{self.lineage_context}>{self.source_label}:"
                f"{self.scope}:{node.lineno}:"
                f"{getattr(node, 'col_offset', 0)}:{name}"
            ),
        )
        child.scope = name
        child.source_label = name
        child.active_model = model.identity
        child.future_annotations = _future_annotations_enabled(tree)
        child.local_names = _function_local_names(func)
        child_flow = child._run_block(func.body, bound)
        produced = _dedupe_observations(
            child_observations
            if self.project_child_observations
            else self.observations[before:]
        )
        if self.project_child_observations:
            projected = tuple(
                RoleObservation(
                    role=item.role,
                    read_kind=item.read_kind,
                    root=item.root,
                    protocol_range=item.protocol_range,
                    subset=item.subset,
                    model_id=item.model_id,
                    line=node.lineno,
                    source=self.source_label,
                    expression=_source(node),
                )
                for item in produced
            )
            self.observations.extend(projected)
            for item in child_unresolved:
                self.unresolved.append(UnresolvedLineage(
                    source=self.source_label,
                    line=node.lineno,
                    expression=_source(node),
                    reason=f"{name}: {item.reason}",
                    role=item.role,
                    read_kind=item.read_kind,
                    root=item.root,
                    model_id=item.model_id,
                    protocol_range=item.protocol_range,
                    subset=item.subset,
                ))
            produced = projected
        bindings = tuple(
            FormalBinding(formal, value.root, value.protocol_range)
            for formal, value in sorted(bound.items())
            if formal in formal_names and isinstance(value, _TensorValue)
        )
        self.calls.append(CallSiteSummary(
            caller=self.scope,
            callee=name,
            line=node.lineno,
            model_id=model.identity,
            model_formal=(
                model_binding[0] if model_binding is not None else None
            ),
            model_bindings=tuple(
                ModelBinding(formal, value.identity)
                for formal, value in bound_models
            ),
            bindings=bindings,
            observations=produced,
        ))
        returned_value: _Value | None = None
        return_candidates: list[_Value] = [
            value if value is not None else UNKNOWN
            for value, _ in child_flow.return_completions
        ]
        if child_flow.falls_through and return_candidates:
            # A reachable function-end is an implicit ``return None`` and is
            # therefore a non-metric branch of any explicit metric return.
            return_candidates.append(UNKNOWN)
        for value in return_candidates:
            returned_value = value if returned_value is None \
                else _merge_branch_values(returned_value, value)
        if isinstance(returned_value, _PredictionValue):
            for tensor in returned_value.conditioning:
                self._observe(
                    role=INFERENCE_CONDITIONING,
                    read_kind="conditioning",
                    tensor=tensor,
                    model_id=returned_value.model_id,
                    node=node,
                )
                if tensor.lineage_id is not None:
                    self.conditioning_values.add(
                        (tensor.lineage_id, tensor.root,
                         returned_value.model_id)
                    )
        if isinstance(returned_value, (
                _MetricExpressionValue, _PredictionValue,
                _UnresolvedPredictionValue, _ModelValue)):
            return returned_value
        return model

    def _forecast_call(self, node: ast.Call, env: dict[str, _Value]) -> _Value:
        values = [self._eval(arg, env) for arg in node.args]
        values.extend(self._eval(keyword.value, env) for keyword in node.keywords)
        if isinstance(node.func, ast.Attribute):
            values.append(self._eval(node.func.value, env))
        model = next((value for value in values if isinstance(value, _ModelValue)), None)
        model_id = model.identity if model is not None else self.active_model
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            value = self._eval(arg, env)
            tensors = (
                value.tensors
                if isinstance(value, _TensorAlternativesValue)
                else (value,) if isinstance(value, _TensorValue) else ()
            )
            for tensor in tensors:
                if tensor.protocol_range is None:
                    continue
                self._observe(
                    role=INFERENCE_CONDITIONING,
                    read_kind="conditioning",
                    tensor=tensor,
                    model_id=model_id,
                    node=arg,
                )
                if tensor.lineage_id is not None:
                    self.conditioning_values.add(
                        (tensor.lineage_id, tensor.root, model_id)
                    )
        return _PredictionValue(model_id)

    def _observe(
        self,
        *,
        role: str,
        read_kind: str,
        tensor: _TensorValue,
        model_id: str,
        node: ast.AST,
    ) -> None:
        if tensor.protocol_range is None:
            self._record_unresolved(
                node,
                f"{role} range is unresolved",
                role=role,
                read_kind=read_kind,
                root=tensor.root,
                model_id=model_id,
                protocol_range=tensor.protocol_range,
                subset=tensor.subset,
            )
            return
        self.observations.append(RoleObservation(
            role=role,
            read_kind=read_kind,
            root=tensor.root,
            protocol_range=tensor.protocol_range,
            subset=tensor.subset,
            model_id=model_id,
            line=getattr(node, "lineno", 0),
            source=self.source_label,
            expression=_source(node),
        ))

    def _emit_loss(self, loss: _LossValue, role: str, node: ast.AST) -> None:
        for tensor in loss.tensors:
            for model_id in loss.model_ids:
                self._observe(
                    role=role,
                    read_kind="target",
                    tensor=tensor,
                    model_id=model_id,
                    node=node,
                )

    def _metric_values(
        self,
        node: ast.AST,
        env: dict[str, _Value],
        evaluated: object = _UNEVALUATED,
    ) -> Iterable[_Value]:
        # Statements have already evaluated their outer expression.  Reuse
        # that value so semantic metric inspection cannot execute a trainer a
        # second time and duplicate its observations or call-site summary.
        value = self._eval(node, env) \
            if evaluated is _UNEVALUATED else evaluated
        if isinstance(value, _MetricExpressionValue):
            yield from value.tensors
            yield from value.predictions
            yield from value.unresolved_predictions
            return
        if isinstance(value, _TensorAlternativesValue):
            yield from value.tensors
            return
        if isinstance(value, (
                _TensorValue, _PredictionValue, _UnresolvedPredictionValue)):
            yield value
            return
        for child in ast.iter_child_nodes(node):
            yield from self._metric_values(child, env)

    @staticmethod
    def _looks_like_metric_expression(node: ast.AST) -> bool:
        calls = {
            _callee_name(item)
            for item in ast.walk(node)
            if isinstance(item, ast.Call)
        }
        call_tokens = set().union(*(_tokens(name) for name in calls)) \
            if calls else set()
        has_explicit_metric_call = bool(call_tokens & _METRIC_TOKENS)
        has_metric_reduction = bool(
            call_tokens & {"mean", "sum", "sqrt", "abs", "norm"}
        )
        has_comparison_arithmetic = any(
            isinstance(item, (ast.BinOp, ast.Compare))
            for item in ast.walk(node)
        )
        return has_explicit_metric_call \
            or has_metric_reduction and has_comparison_arithmetic

    def _record_metric(
        self,
        node: ast.AST,
        env: dict[str, _Value],
        evaluated: object = _UNEVALUATED,
    ) -> None:
        if isinstance(node, ast.Call) \
                and _tokens(_callee_name(node)) \
                & _NON_EVALUATION_CALL_TOKENS \
                and not self._looks_like_metric_expression(node):
            return
        values = list(self._metric_values(node, env, evaluated))
        predictions = [value for value in values
                       if isinstance(value, _PredictionValue)]
        unresolved_predictions = [
            value for value in values
            if isinstance(value, _UnresolvedPredictionValue)
        ]
        tensors = [
            value for value in values
            if isinstance(value, _TensorValue)
        ]
        explicit_metric = self._looks_like_metric_expression(node)
        if not explicit_metric \
                and not isinstance(evaluated, _MetricExpressionValue):
            return
        if not explicit_metric:
            model_ids = {
                value.model_id for value in predictions
            } | {
                value.model_id for value in unresolved_predictions
                if value.model_id is not None
            }
            tensors = [
                value for value in tensors
                if not any(
                    (
                        lineage_id, value.root, model_id,
                    ) in self.conditioning_values
                    for model_id in model_ids
                    for lineage_id in value.lineage
                )
            ]
        if not tensors or not predictions and not unresolved_predictions:
            # Baseline metrics can consume the same target without evaluating
            # a scored model.  Opaque display/logging calls can also place the
            # already-recorded conditioning history beside a prediction; that
            # history remains conditioning unless an explicit metric consumes
            # it.
            return
        model_ids = {value.model_id for value in predictions}
        unresolved_model_ids = {
            value.model_id for value in unresolved_predictions
            if value.model_id is not None
        }
        model_id = next(iter(model_ids)) if len(model_ids) == 1 else None
        unresolved_reason = None
        if unresolved_predictions:
            reasons = "; ".join(dict.fromkeys(
                value.reason for value in unresolved_predictions
            ))
            if not model_ids and len(unresolved_model_ids) == 1:
                model_id = next(iter(unresolved_model_ids))
            unresolved_reason = (
                f"reported metric prediction lineage is unresolved: {reasons}"
            )
        elif len(model_ids) != 1:
            unresolved_reason = "metric is not tied to one scored model"
        seen: set[tuple[str, ProtocolRange | None, str | None]] = set()
        for value in tensors:
            key = (value.root, value.protocol_range, value.subset)
            if key in seen:
                continue
            seen.add(key)
            if unresolved_reason is not None or model_id is None:
                self._record_unresolved(
                    node,
                    unresolved_reason or "scored model identity is unresolved",
                    role=REPORTED_EVALUATION,
                    read_kind="target",
                    root=value.root,
                    model_id=model_id,
                    protocol_range=value.protocol_range,
                    subset=value.subset,
                )
            else:
                self._observe(
                    role=REPORTED_EVALUATION,
                    read_kind="target",
                    tensor=value,
                    model_id=model_id,
                    node=node,
                )

    def _assign_target(
        self, target: ast.AST, value: _Value, env: dict[str, _Value],
    ) -> None:
        if isinstance(target, ast.Name):
            env[target.id] = value
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            if isinstance(value, _ShapeValue):
                for index, item in enumerate(target.elts):
                    assigned: _Value = UNKNOWN
                    if index == len(target.elts) - 1 \
                            and value.protocol_length is not None:
                        assigned = _NumberRange.exact(value.protocol_length)
                    self._assign_target(item, assigned, env)
                return
            if isinstance(value, _SequenceValue):
                for item, assigned in zip(target.elts, value.values):
                    self._assign_target(item, assigned, env)
                return
            for item in target.elts:
                self._assign_target(item, UNKNOWN, env)

    def _transition_target_outcomes(
        self,
        target: ast.AST,
        environments: Sequence[dict[str, _Value]],
        *,
        value: _Value = UNKNOWN,
        delete: bool = False,
    ) -> tuple[
        list[dict[str, _Value]],
        list[tuple[str, dict[str, _Value]]],
        list[_ImplicitRaiseState],
    ]:
        """Apply one Store/Delete target in Python's left-to-right order."""
        normal = [dict(environment) for environment in environments]
        raised: list[tuple[str, dict[str, _Value]]] = []
        implicit: list[_ImplicitRaiseState] = []

        if isinstance(target, ast.Name):
            transitioned: list[dict[str, _Value]] = []
            for environment in normal:
                state = dict(environment)
                if not delete:
                    state[target.id] = value
                    transitioned.append(state)
                    continue
                current = state.get(target.id, UNBOUND)
                if isinstance(current, _Unbound):
                    state[target.id] = UNBOUND
                    raised.append(("UnboundLocalError", state))
                    continue
                if isinstance(current, _PossiblyUnboundValue):
                    error_state = dict(state)
                    error_state[target.id] = UNBOUND
                    raised.append(("UnboundLocalError", error_state))
                state[target.id] = UNBOUND
                transitioned.append(state)
            return (
                self._dedupe_environments(transitioned),
                self._dedupe_raise_environments(raised),
                implicit,
            )

        if isinstance(target, ast.Starred):
            return self._transition_target_outcomes(
                target.value, normal, value=value, delete=delete
            )

        if isinstance(target, (ast.Tuple, ast.List)):
            starred = [
                index for index, item in enumerate(target.elts)
                if isinstance(item, ast.Starred)
            ]
            unpack_status: bool | None = None
            if not delete and isinstance(value, _SequenceValue):
                unpack_status = _unpack_target_accepts_length(
                    target, len(value.values)
                )
            elif not delete and isinstance(value, _RangeIterableValue):
                unpack_status = _unpack_target_accepts_length(
                    target, value.iterable_length
                )
            elif not delete and isinstance(value, _NumberRange):
                raised.extend(
                    ("TypeError", dict(environment))
                    for environment in normal
                )
                return [], self._dedupe_raise_environments(raised), implicit
            if not delete and unpack_status is None:
                implicit.extend(
                    _ImplicitRaiseState(dict(environment))
                    for environment in normal
                )
            if unpack_status is False:
                raised.extend(
                    ("ValueError", dict(environment))
                    for environment in normal
                )
                return [], self._dedupe_raise_environments(raised), implicit
            for index, item in enumerate(target.elts):
                assigned: _Value = UNKNOWN
                if isinstance(value, _ShapeValue) \
                        and index == len(target.elts) - 1 \
                        and value.protocol_length is not None:
                    assigned = _NumberRange.exact(value.protocol_length)
                elif isinstance(value, _SequenceValue) \
                        and index < len(value.values):
                    if not starred or index < starred[0]:
                        assigned = value.values[index]  # type: ignore[assignment]
                    elif index == starred[0]:
                        tail_count = len(target.elts) - index - 1
                        stop = len(value.values) - tail_count
                        assigned = _SequenceValue(tuple(
                            value.values[index:stop]
                        ))
                    else:
                        source_index = (
                            len(value.values) - len(target.elts) + index
                        )
                        assigned = value.values[source_index]  # type: ignore[assignment]
                elif isinstance(value, _RangeIterableValue):
                    if not starred or index < starred[0]:
                        assigned = value.item(index)
                    elif index == starred[0]:
                        assigned = UNKNOWN
                    else:
                        source_index = (
                            value.iterable_length - len(target.elts) + index
                        )
                        assigned = value.item(source_index)
                item_normal, item_raised, item_implicit = (
                    self._transition_target_outcomes(
                        item,
                        normal,
                        value=assigned,
                        delete=delete,
                    )
                )
                normal = item_normal
                raised.extend(item_raised)
                implicit.extend(item_implicit)
                if not normal:
                    break
            return (
                self._dedupe_environments(normal),
                self._dedupe_raise_environments(raised),
                implicit,
            )

        for expression in _target_evaluation_expressions(target):
            next_normal: list[dict[str, _Value]] = []
            for environment in normal:
                expression_normal, expression_raised = (
                    self._unbound_expression_outcomes(
                        expression, environment
                    )
                )
                next_normal.extend(expression_normal)
                raised.extend(
                    ("UnboundLocalError", state)
                    for state in expression_raised
                )
            normal = self._dedupe_environments(next_normal)
            if not normal:
                break
        implicit.extend(
            _ImplicitRaiseState(dict(environment))
            for environment in normal
        )
        return (
            normal,
            self._dedupe_raise_environments(raised),
            implicit,
        )

    def _run_literal_for(
        self,
        statement: ast.For,
        values: Sequence[_Value],
        env: dict[str, _Value],
    ) -> _FlowOutcome:
        """Execute a known ordered iterable without inventing later paths."""
        entry = dict(env)
        active = [dict(entry)]
        broken: list[dict[str, _Value]] = []
        terminal: list[_FlowOutcome] = []

        def retain_terminal(flow: _FlowOutcome) -> None:
            if flow.returned or flow.raised \
                    or flow.implicit_raise_environments:
                terminal.append(_FlowOutcome(
                    falls_through=False,
                    returned=flow.returned,
                    raised=flow.raised,
                    raise_types=flow.raise_types,
                    return_completions=flow.return_completions,
                    implicit_raise_environments=(
                        flow.implicit_raise_environments
                    ),
                    raise_environments=flow.raise_environments,
                ))

        for value in values:
            next_active: list[dict[str, _Value]] = []
            for state in active:
                target_normal, target_raises, target_implicit = (
                    self._transition_target_outcomes(
                        statement.target, [state], value=value
                    )
                )
                if target_raises or target_implicit:
                    terminal.append(_FlowOutcome(
                        falls_through=False,
                        raised=bool(target_raises),
                        raise_types=frozenset(
                            name for name, _ in target_raises
                        ),
                        implicit_raise_environments=tuple(
                            target_implicit
                        ),
                        raise_environments=tuple(target_raises),
                    ))
                for target_env in target_normal:
                    body_env = dict(target_env)
                    body_flow = self._run_block(
                        statement.body, body_env
                    )
                    retain_terminal(body_flow)
                    broken.extend(body_flow.break_environments)
                    if body_flow.falls_through:
                        next_active.extend(
                            body_flow.fallthrough_environments
                            or (body_env,)
                        )
                    next_active.extend(body_flow.continue_environments)
            active = self._dedupe_environments(next_active)
            if not active:
                break

        else_flows: list[_FlowOutcome] = []
        for state in active:
            else_env = dict(state)
            else_flow = self._run_block(statement.orelse, else_env)
            else_flows.append(else_flow)
            retain_terminal(else_flow)

        live = list(broken)
        for else_flow in else_flows:
            if else_flow.falls_through:
                live.extend(
                    else_flow.fallthrough_environments
                )
        live = self._dedupe_environments(live)
        terminal_flow = _union_flow(
            *terminal, falls_through=False
        ) if terminal else _FlowOutcome(False)
        env.clear()
        env.update(self._join_environments(
            live or active or broken or [entry]
        ))
        return _FlowOutcome(
            falls_through=bool(live),
            fallthrough_environments=tuple(live),
            returned=terminal_flow.returned,
            raised=terminal_flow.raised,
            raise_types=terminal_flow.raise_types,
            return_completions=terminal_flow.return_completions,
            implicit_raise_environments=(
                terminal_flow.implicit_raise_environments
            ),
            raise_environments=terminal_flow.raise_environments,
        )

    @staticmethod
    def _join_environments(
        environments: Sequence[dict[str, _Value]],
    ) -> dict[str, _Value]:
        if not environments:
            return {}
        joined = dict(environments[0])
        for environment in environments[1:]:
            for name in set(joined) | set(environment):
                joined[name] = _merge_branch_values(
                    joined.get(name, UNBOUND),
                    environment.get(name, UNBOUND),
                )
        return joined

    def _assignment(
        self, target: ast.AST, value_node: ast.AST,
        env: dict[str, _Value],
    ) -> None:
        value = self._eval(value_node, env)
        if isinstance(target, ast.Name):
            if _is_model_builder_call(value_node):
                # The binding names independently constructed model
                # instances.  Keeping that identity prevents training one
                # builder result from being joined to evaluation of another.
                value = self._model_from_constants(target.id)
            # Reduction facts describe the value after opaque loaders or
            # constructors.  Retain such a fact when this narrow interpreter
            # cannot evaluate the producing expression itself.
            if value is UNKNOWN:
                seeded = self._initial_value(target.id)
                if seeded is not UNKNOWN:
                    value = seeded
            if self.observe_reported_metrics:
                self._record_metric(value_node, env, value)
        self._assign_target(target, value, env)

    def _loss_values(
        self, node: ast.AST, env: dict[str, _Value],
    ) -> Iterable[_LossValue]:
        value = self._eval(node, env)
        if isinstance(value, _LossValue):
            yield value
            return
        for child in ast.iter_child_nodes(node):
            yield from self._loss_values(child, env)

    @staticmethod
    def _dedupe_environments(
        environments: Sequence[dict[str, _Value]],
    ) -> list[dict[str, _Value]]:
        unique: list[dict[str, _Value]] = []
        for environment in environments:
            if environment not in unique:
                unique.append(environment)
        return unique

    @staticmethod
    def _dedupe_raise_environments(
        states: Sequence[tuple[str, dict[str, _Value]]],
    ) -> list[tuple[str, dict[str, _Value]]]:
        unique: list[tuple[str, dict[str, _Value]]] = []
        for state in states:
            if state not in unique:
                unique.append(state)
        return unique

    def _unbound_expression_outcomes(
        self,
        node: ast.AST | None,
        env: Mapping[str, _Value],
        comprehension_names: frozenset[str] = frozenset(),
    ) -> tuple[list[dict[str, _Value]], list[dict[str, _Value]]]:
        """Split successful expression reads from local-unbound failures."""
        environment = dict(env)
        if node is None:
            return [environment], []
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in comprehension_names \
                    or node.id not in self.local_names:
                return [environment], []
            value = environment.get(node.id, UNBOUND)
            if isinstance(value, _Unbound):
                environment[node.id] = UNBOUND
                return [], [environment]
            if isinstance(value, _PossiblyUnboundValue):
                normal = dict(environment)
                normal[node.id] = value.value  # type: ignore[assignment]
                raised = dict(environment)
                raised[node.id] = UNBOUND
                return [normal], [raised]
            return [environment], []
        if isinstance(node, ast.NamedExpr):
            normal, raised = self._unbound_expression_outcomes(
                node.value, environment, comprehension_names
            )
            transitioned: list[dict[str, _Value]] = []
            for state in normal:
                value = self._eval(node.value, state)
                self._assign_target(node.target, value, state)
                transitioned.append(state)
            return self._dedupe_environments(transitioned), raised

        def sequence(
            nodes: Sequence[ast.AST],
            starts: Sequence[dict[str, _Value]],
            names: frozenset[str] = comprehension_names,
        ) -> tuple[list[dict[str, _Value]], list[dict[str, _Value]]]:
            normal = list(starts)
            raised: list[dict[str, _Value]] = []
            for child in nodes:
                next_normal: list[dict[str, _Value]] = []
                for state in normal:
                    child_normal, child_raised = (
                        self._unbound_expression_outcomes(
                            child, state, names
                        )
                    )
                    next_normal.extend(child_normal)
                    raised.extend(child_raised)
                normal = self._dedupe_environments(next_normal)
                if not normal:
                    break
            return normal, self._dedupe_environments(raised)

        def condition(
            expression: ast.AST, state: dict[str, _Value],
            names: frozenset[str] = comprehension_names,
        ) -> bool | None:
            if any(
                isinstance(item, ast.Name)
                and isinstance(item.ctx, ast.Load)
                and item.id in names
                for item in ast.walk(expression)
            ):
                return None
            return self._condition(expression, dict(state))

        if isinstance(node, ast.Lambda):
            defaults = tuple(node.args.defaults) + tuple(
                item for item in node.args.kw_defaults if item is not None
            )
            return sequence(defaults, [environment])
        if isinstance(node, ast.IfExp):
            test_normal, raised = self._unbound_expression_outcomes(
                node.test, environment, comprehension_names
            )
            normal: list[dict[str, _Value]] = []
            for state in test_normal:
                verdict = condition(node.test, state)
                branches = (
                    (node.body,) if verdict is True
                    else (node.orelse,) if verdict is False
                    else (node.body, node.orelse)
                )
                for branch in branches:
                    branch_normal, branch_raised = (
                        self._unbound_expression_outcomes(
                            branch, state, comprehension_names
                        )
                    )
                    normal.extend(branch_normal)
                    raised.extend(branch_raised)
            return (
                self._dedupe_environments(normal),
                self._dedupe_environments(raised),
            )
        if isinstance(node, ast.BoolOp):
            def evaluate_from(
                index: int, state: dict[str, _Value],
            ) -> tuple[list[dict[str, _Value]], list[dict[str, _Value]]]:
                value_normal, value_raised = (
                    self._unbound_expression_outcomes(
                        node.values[index], state, comprehension_names
                    )
                )
                if index == len(node.values) - 1:
                    return value_normal, value_raised
                normal: list[dict[str, _Value]] = []
                raised = list(value_raised)
                for value_state in value_normal:
                    verdict = condition(node.values[index], value_state)
                    stops = (
                        isinstance(node.op, ast.And) and verdict is False
                        or isinstance(node.op, ast.Or) and verdict is True
                    )
                    continues = (
                        isinstance(node.op, ast.And) and verdict is True
                        or isinstance(node.op, ast.Or) and verdict is False
                    )
                    if stops or not continues:
                        normal.append(value_state)
                    if continues or not stops:
                        rest_normal, rest_raised = evaluate_from(
                            index + 1, value_state
                        )
                        normal.extend(rest_normal)
                        raised.extend(rest_raised)
                return (
                    self._dedupe_environments(normal),
                    self._dedupe_environments(raised),
                )

            return evaluate_from(0, environment) \
                if node.values else ([environment], [])
        if isinstance(node, ast.Compare):
            active, raised = self._unbound_expression_outcomes(
                node.left, environment, comprehension_names
            )
            completed: list[dict[str, _Value]] = []
            previous = node.left
            for operator, comparator in zip(
                    node.ops, node.comparators):
                comparator_normal: list[dict[str, _Value]] = []
                for state in active:
                    item_normal, item_raised = (
                        self._unbound_expression_outcomes(
                            comparator, state, comprehension_names
                        )
                    )
                    comparator_normal.extend(item_normal)
                    raised.extend(item_raised)
                active = []
                pair = ast.Compare(
                    left=previous,
                    ops=[operator],
                    comparators=[comparator],
                )
                for state in comparator_normal:
                    verdict = condition(pair, state)
                    if verdict is False or verdict is None:
                        completed.append(state)
                    if verdict is True or verdict is None:
                        active.append(state)
                if not active:
                    break
                previous = comparator
            return (
                self._dedupe_environments(completed + active),
                self._dedupe_environments(raised),
            )
        if isinstance(node, ast.GeneratorExp):
            return self._unbound_expression_outcomes(
                node.generators[0].iter, environment, comprehension_names
            ) if node.generators else ([environment], [])
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
            normal = [environment]
            raised: list[dict[str, _Value]] = []
            empty: list[dict[str, _Value]] = []
            bound = set(comprehension_names)
            target_names = {
                item.id
                for generator in node.generators
                for item in ast.walk(generator.target)
                if isinstance(item, ast.Name)
                and isinstance(item.ctx, ast.Store)
            }

            def restore_outer_bindings(
                states: Sequence[dict[str, _Value]],
            ) -> list[dict[str, _Value]]:
                restored: list[dict[str, _Value]] = []
                for state in states:
                    result = dict(state)
                    for name in target_names:
                        if name in environment:
                            result[name] = environment[name]
                        else:
                            result.pop(name, None)
                    restored.append(result)
                return self._dedupe_environments(restored)

            if len(node.generators) == 1:
                generator = node.generators[0]
                iterator_normal, iterator_raised = sequence(
                    (generator.iter,), normal, frozenset(bound)
                )
                domains = [
                    (state, self._eval(generator.iter, state))
                    for state in iterator_normal
                ]
                if all(
                    isinstance(domain, _SequenceValue)
                    for _, domain in domains
                ):
                    completed: list[dict[str, _Value]] = []
                    ordered_raised = list(iterator_raised)
                    ordered_typed: list[
                        tuple[str, dict[str, _Value]]
                    ] = []
                    result_nodes: tuple[ast.AST, ...] = (
                        (node.key, node.value)
                        if isinstance(node, ast.DictComp)
                        else (node.elt,)
                    )
                    target_bound = frozenset(
                        item.id for item in ast.walk(generator.target)
                        if isinstance(item, ast.Name)
                        and isinstance(item.ctx, ast.Store)
                    )
                    for state, domain in domains:
                        active = [state]
                        assert isinstance(domain, _SequenceValue)
                        for candidate in domain.values:
                            next_active: list[dict[str, _Value]] = []
                            for active_state in active:
                                (
                                    target_normal,
                                    target_raises,
                                    _target_implicit,
                                ) = self._transition_target_outcomes(
                                    generator.target,
                                    [active_state],
                                    value=candidate,  # type: ignore[arg-type]
                                )
                                ordered_typed.extend(target_raises)
                                for target_state in target_normal:
                                    element_states = [target_state]
                                    skipped: list[
                                        dict[str, _Value]
                                    ] = []
                                    for filter_node in generator.ifs:
                                        filter_normal, filter_raised = (
                                            sequence(
                                                (filter_node,),
                                                element_states,
                                                target_bound,
                                            )
                                        )
                                        ordered_raised.extend(filter_raised)
                                        element_states = []
                                        for filter_state in filter_normal:
                                            verdict = self._condition(
                                                filter_node,
                                                dict(filter_state),
                                            )
                                            if verdict is not True:
                                                skipped.append(filter_state)
                                            if verdict is not False:
                                                element_states.append(
                                                    filter_state
                                                )
                                        if not element_states:
                                            break
                                    result_normal, result_raised = sequence(
                                        result_nodes,
                                        element_states,
                                        target_bound,
                                    ) if element_states else ([], [])
                                    ordered_raised.extend(result_raised)
                                    next_active.extend(skipped)
                                    next_active.extend(result_normal)
                            active = self._dedupe_environments(next_active)
                            if not active:
                                break
                        completed.extend(active)
                    restored_typed = [
                        (name, restored)
                        for name, state in ordered_typed
                        for restored in restore_outer_bindings((state,))
                    ]
                    self._pending_expression_raises.extend(
                        restored_typed
                    )
                    return (
                        restore_outer_bindings(completed),
                        restore_outer_bindings(ordered_raised),
                    )

            for generator in node.generators:
                normal, item_raised = sequence(
                    (generator.iter,), normal, frozenset(bound)
                )
                raised.extend(item_raised)
                iterator_nonempty = _literal_iterable_nonempty(
                    generator.iter
                )
                if iterator_nonempty is not True:
                    empty.extend(normal)
                if iterator_nonempty is False:
                    normal = []
                    break
                target_normal: list[dict[str, _Value]] = []
                target_raised: list[
                    tuple[str, dict[str, _Value]]
                ] = []
                for state in normal:
                    domain = self._eval(generator.iter, state)
                    candidates: Sequence[_Value]
                    if isinstance(domain, _SequenceValue):
                        candidates = domain.values  # type: ignore[assignment]
                    else:
                        candidates = (domain,)
                    for candidate in candidates:
                        item_normal, item_raised, _ = (
                            self._transition_target_outcomes(
                                generator.target,
                                [state],
                                value=candidate,
                            )
                        )
                        target_normal.extend(item_normal)
                        target_raised.extend(item_raised)
                target_normal = self._dedupe_environments(target_normal)
                target_raised = self._dedupe_raise_environments(
                    target_raised
                )
                normal = target_normal
                raised.extend(
                    state for name, state in target_raised
                    if name == "UnboundLocalError"
                )
                if not normal:
                    break
                bound.update(
                    item.id for item in ast.walk(generator.target)
                    if isinstance(item, ast.Name)
                    and isinstance(item.ctx, ast.Store)
                )
                for filter_node in generator.ifs:
                    filter_normal, filter_raised = sequence(
                        (filter_node,), normal, frozenset(bound)
                    )
                    raised.extend(filter_raised)
                    normal = []
                    for state in filter_normal:
                        verdict = condition(
                            filter_node, state, frozenset(bound)
                        )
                        if verdict is not True:
                            empty.append(state)
                        if verdict is not False:
                            normal.append(state)
                    if not normal:
                        break
                if not normal:
                    break
            results: tuple[ast.AST, ...] = (
                (node.key, node.value)
                if isinstance(node, ast.DictComp) else (node.elt,)
            )
            result_normal, result_raised = sequence(
                results, normal, frozenset(bound)
            ) if normal else ([], [])
            raised.extend(result_raised)
            return (
                restore_outer_bindings(empty + result_normal),
                restore_outer_bindings(raised),
            )

        normal, raised = sequence(
            tuple(ast.iter_child_nodes(node)), [environment]
        )
        for state in normal:
            bound_names = {
                name for name, value in state.items()
                if not isinstance(
                    value, (_Unbound, _PossiblyUnboundValue)
                )
            } | set(comprehension_names)
            if _expression_operation_may_raise(node, bound_names):
                self._pending_expression_implicit.append(
                    _ImplicitRaiseState(dict(state))
                )
        return normal, raised

    def _statement_evaluated_expressions(
        self, statement: ast.stmt,
    ) -> tuple[ast.AST, ...]:
        expressions: tuple[ast.AST, ...] = ()
        if isinstance(statement, ast.Assign):
            expressions = (statement.value,)
        elif isinstance(statement, ast.AnnAssign) \
                and statement.value is not None:
            expressions = (statement.value,)
        elif isinstance(statement, ast.AnnAssign):
            expressions = _target_evaluation_expressions(
                statement.target
            )
        elif isinstance(statement, ast.AugAssign):
            target: ast.AST = statement.target
            if isinstance(target, ast.Name):
                target = ast.Name(id=target.id, ctx=ast.Load())
            expressions = (target, statement.value)
        elif isinstance(statement, ast.Expr):
            expressions = (statement.value,)
        elif isinstance(statement, ast.Return) \
                and statement.value is not None:
            expressions = (statement.value,)
        elif isinstance(statement, ast.Raise):
            expressions = tuple(
                item for item in (statement.exc, statement.cause)
                if item is not None
            )
        elif isinstance(statement, (ast.If, ast.While, ast.Assert)):
            expressions = (statement.test,)
        elif isinstance(statement, (ast.For, ast.AsyncFor)):
            expressions = (statement.iter,)
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            # Context expressions and optional targets interleave item by
            # item; the With arm preserves that binding order.
            expressions = ()
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            expressions = _definition_evaluated_expressions(
                statement,
                annotations_deferred=self.future_annotations,
            )
        elif isinstance(statement, ast.ClassDef):
            expressions = _definition_evaluated_expressions(statement)
        elif isinstance(statement, ast.Match):
            expressions = (statement.subject,)
        elif isinstance(statement, ast.Delete):
            expressions = ()
        return expressions

    def _statement_unbound_outcomes(
        self, statement: ast.stmt, env: Mapping[str, _Value],
    ) -> tuple[
        list[dict[str, _Value]],
        list[dict[str, _Value]],
        list[tuple[str, dict[str, _Value]]],
        list[_ImplicitRaiseState],
    ]:
        expressions = self._statement_evaluated_expressions(statement)

        self._pending_expression_raises = []
        self._pending_expression_implicit = []
        normal = [dict(env)]
        raised: list[dict[str, _Value]] = []
        for expression in expressions:
            next_normal: list[dict[str, _Value]] = []
            for state in normal:
                item_normal, item_raised = (
                    self._unbound_expression_outcomes(expression, state)
                )
                next_normal.extend(item_normal)
                raised.extend(item_raised)
            normal = self._dedupe_environments(next_normal)
            if not normal:
                break
        return (
            normal,
            self._dedupe_environments(raised),
            self._dedupe_raise_environments(
                self._pending_expression_raises
            ),
            list(self._pending_expression_implicit),
        )

    def _unbound_paths(
        self,
        node: ast.AST | None,
        env: Mapping[str, _Value],
        comprehension_names: frozenset[str] = frozenset(),
    ) -> tuple[bool, bool]:
        """Return ``(normal_possible, unbound_error_possible)``."""
        if node is None:
            return True, False
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in comprehension_names \
                    or node.id not in self.local_names:
                return True, False
            value = env.get(node.id, UNBOUND)
            if isinstance(value, _Unbound):
                return False, True
            if isinstance(value, _PossiblyUnboundValue):
                return True, True
            return True, False
        if isinstance(node, ast.Lambda):
            return True, False
        if isinstance(node, ast.IfExp):
            test_normal, test_unbound = self._unbound_paths(
                node.test, env, comprehension_names
            )
            if not test_normal:
                return False, test_unbound
            condition = self._condition(node.test, dict(env))
            branches = (
                (node.body,) if condition is True
                else (node.orelse,) if condition is False
                else (node.body, node.orelse)
            )
            branch_states = [
                self._unbound_paths(item, env, comprehension_names)
                for item in branches
            ]
            return (
                any(normal for normal, _ in branch_states),
                test_unbound
                or any(unbound for _, unbound in branch_states),
            )
        if isinstance(node, ast.BoolOp):
            def from_index(index: int) -> tuple[bool, bool]:
                current_normal, current_unbound = self._unbound_paths(
                    node.values[index], env, comprehension_names
                )
                if not current_normal or index == len(node.values) - 1:
                    return current_normal, current_unbound
                condition = self._condition(
                    node.values[index], dict(env)
                )
                short_circuits = (
                    isinstance(node.op, ast.And) and condition is False
                    or isinstance(node.op, ast.Or) and condition is True
                )
                if short_circuits:
                    return True, current_unbound
                rest_normal, rest_unbound = from_index(index + 1)
                continues = (
                    isinstance(node.op, ast.And) and condition is True
                    or isinstance(node.op, ast.Or) and condition is False
                )
                if continues:
                    return (
                        rest_normal,
                        current_unbound or rest_unbound,
                    )
                return (
                    True,
                    current_unbound or rest_unbound,
                )

            return from_index(0) if node.values else (True, False)
        if isinstance(node, ast.Compare):
            left_normal, unbound = self._unbound_paths(
                node.left, env, comprehension_names
            )
            if not left_normal:
                return False, unbound
            reaches_next = True
            short_completion = False
            previous = node.left
            for operator, comparator in zip(
                    node.ops, node.comparators):
                if not reaches_next:
                    break
                right_normal, right_unbound = self._unbound_paths(
                    comparator, env, comprehension_names
                )
                unbound = unbound or right_unbound
                if not right_normal:
                    reaches_next = False
                    break
                pair = ast.Compare(
                    left=previous,
                    ops=[operator],
                    comparators=[comparator],
                )
                condition = self._condition(pair, dict(env))
                if condition is False:
                    short_completion = True
                    reaches_next = False
                    break
                if condition is None:
                    short_completion = True
                previous = comparator
            return short_completion or reaches_next, unbound
        if isinstance(node, ast.GeneratorExp):
            if not node.generators:
                return True, False
            return self._unbound_paths(
                node.generators[0].iter, env, comprehension_names
            )
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
            reaches_element = True
            empty_completion = False
            unbound = False
            bound = set(comprehension_names)

            def consume(item: ast.AST) -> bool:
                nonlocal reaches_element, unbound
                item_normal, item_unbound = self._unbound_paths(
                    item, env, frozenset(bound)
                )
                unbound = unbound or reaches_element and item_unbound
                reaches_element = reaches_element and item_normal
                return item_normal

            def nonempty(iterator: ast.AST) -> bool | None:
                if isinstance(iterator, (ast.List, ast.Tuple, ast.Set)):
                    return bool(iterator.elts)
                if isinstance(iterator, ast.Call) \
                        and isinstance(iterator.func, ast.Name) \
                        and iterator.func.id == "range":
                    try:
                        values = [
                            ast.literal_eval(item)
                            for item in iterator.args
                        ]
                        if all(isinstance(item, int) for item in values):
                            return bool(range(*values))
                    except (TypeError, ValueError):
                        return None
                return None

            for generator in node.generators:
                if not reaches_element:
                    break
                consume(generator.iter)
                if not reaches_element:
                    break
                iterator_nonempty = nonempty(generator.iter)
                if iterator_nonempty is False:
                    empty_completion = True
                    reaches_element = False
                    break
                if iterator_nonempty is None:
                    empty_completion = True
                bound.update(
                    item.id for item in ast.walk(generator.target)
                    if isinstance(item, ast.Name)
                    and isinstance(item.ctx, ast.Store)
                )
                for condition in generator.ifs:
                    if not reaches_element:
                        break
                    consume(condition)
                    if not reaches_element:
                        break
                    condition_value = self._condition(
                        condition, dict(env)
                    )
                    if condition_value is False:
                        empty_completion = True
                        reaches_element = False
                        break
                    if condition_value is None:
                        empty_completion = True
            result_normal = False
            result_unbound = False
            if reaches_element:
                results = (
                    (node.key, node.value)
                    if isinstance(node, ast.DictComp)
                    else (node.elt,)
                )
                result_normal = True
                for result in results:
                    item_normal, item_unbound = self._unbound_paths(
                        result, env, frozenset(bound)
                    )
                    result_unbound = (
                        result_unbound or result_normal and item_unbound
                    )
                    result_normal = result_normal and item_normal
            unbound = unbound or reaches_element and result_unbound
            return empty_completion or result_normal, unbound

        normal = True
        unbound = False
        for child in ast.iter_child_nodes(node):
            child_normal, child_unbound = self._unbound_paths(
                child, env, comprehension_names
            )
            unbound = unbound or normal and child_unbound
            normal = normal and child_normal
        return normal, unbound

    def _condition(
        self, node: ast.AST, env: dict[str, _Value],
    ) -> bool | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            value = self._condition(node.operand, env)
            return None if value is None else not value
        scalar = _number(self._eval(node, env))
        if scalar is not None:
            if scalar.lower == scalar.upper == 0:
                return False
            if scalar.upper < 0 or scalar.lower > 0:
                return True
        if not isinstance(node, ast.Compare) or not node.ops \
                or len(node.ops) != len(node.comparators):
            return None
        previous = node.left
        unresolved = False
        for op, comparator in zip(node.ops, node.comparators):
            left = _number(self._eval(previous, env))
            right = _number(self._eval(comparator, env))
            result: bool | None = None
            if left is not None and right is not None:
                if isinstance(op, ast.Lt):
                    if left.upper < right.lower:
                        result = True
                    elif left.lower >= right.upper:
                        result = False
                elif isinstance(op, ast.LtE):
                    if left.upper <= right.lower:
                        result = True
                    elif left.lower > right.upper:
                        result = False
                elif isinstance(op, ast.Gt):
                    if left.lower > right.upper:
                        result = True
                    elif left.upper <= right.lower:
                        result = False
                elif isinstance(op, ast.GtE):
                    if left.lower >= right.upper:
                        result = True
                    elif left.upper < right.lower:
                        result = False
                elif isinstance(op, ast.Eq):
                    if left.lower == left.upper == right.lower == right.upper:
                        result = True
                    elif left.upper < right.lower or right.upper < left.lower:
                        result = False
            if result is False:
                return False
            if result is None:
                unresolved = True
            previous = comparator
        return None if unresolved else True

    def _while_domain(
        self, node: ast.While, env: dict[str, _Value],
    ) -> tuple[str, _NumberRange] | None:
        increments = [stmt for stmt in node.body
                      if isinstance(stmt, ast.AugAssign)
                      and isinstance(stmt.target, ast.Name)
                      and isinstance(stmt.op, ast.Add)]
        for increment in reversed(increments):
            name = increment.target.id
            start = _number(env.get(name, UNKNOWN))
            step = _number(self._eval(increment.value, env))
            start_int = start.exact_int() if start is not None else None
            step_int = step.exact_int() if step is not None else None
            if start_int is None or step_int is None or step_int <= 0:
                continue
            values: list[int] = []
            current = start_int
            for _ in range(100000):
                trial = dict(env)
                trial[name] = _NumberRange.exact(current)
                condition = self._condition(node.test, trial)
                if condition is not True:
                    break
                values.append(current)
                current += step_int
            if values:
                return name, _NumberRange(
                    Fraction(values[0]), Fraction(values[-1]),
                    step_int != 1,
                )
        return None

    def _run_block(
        self, statements: Sequence[ast.stmt], env: dict[str, _Value],
    ) -> _FlowOutcome:
        """Execute a block while retaining Python completion kinds."""
        if len(statements) > 1:
            first_environment = dict(env)
            first_flow = self._run_block(
                statements[:1], first_environment
            )
            first_live = list(first_flow.fallthrough_environments)
            if first_flow.falls_through and not first_live:
                first_live.append(first_environment)
            terminal_prefix = _FlowOutcome(
                falls_through=False,
                returned=first_flow.returned,
                broke=first_flow.broke,
                continued=first_flow.continued,
                raised=first_flow.raised,
                raise_types=first_flow.raise_types,
                return_completions=first_flow.return_completions,
                implicit_raise_environments=(
                    first_flow.implicit_raise_environments
                ),
                break_environments=first_flow.break_environments,
                continue_environments=first_flow.continue_environments,
                raise_environments=first_flow.raise_environments,
            )
            continuation_flows: list[_FlowOutcome] = []
            for environment in self._dedupe_environments(first_live):
                continuation_environment = dict(environment)
                continuation_flows.append(self._run_block(
                    statements[1:], continuation_environment
                ))
            combined = _union_flow(
                terminal_prefix, *continuation_flows,
            )
            live = self._dedupe_environments(
                list(combined.fallthrough_environments)
            )
            if live:
                env.clear()
                env.update(self._join_environments(live))
            return _FlowOutcome(
                falls_through=bool(live),
                fallthrough_environments=tuple(live),
                returned=combined.returned,
                broke=combined.broke,
                continued=combined.continued,
                raised=combined.raised,
                raise_types=combined.raise_types,
                return_completions=combined.return_completions,
                implicit_raise_environments=(
                    combined.implicit_raise_environments
                ),
                break_environments=combined.break_environments,
                continue_environments=combined.continue_environments,
                raise_environments=combined.raise_environments,
            )

        returned = False
        broke = False
        continued = False
        raised = False
        raise_types: set[str | None] = set()
        return_completions: list[
            tuple[_Value | None, dict[str, _Value]]
        ] = []
        implicit_raise_environments: list[_ImplicitRaiseState] = []
        break_environments: list[dict[str, _Value]] = []
        continue_environments: list[dict[str, _Value]] = []
        raise_environments: list[
            tuple[str | None, dict[str, _Value]]
        ] = []
        fallthrough_environments: list[dict[str, _Value]] = []

        def absorb(outcome: _FlowOutcome) -> _FlowOutcome | None:
            nonlocal returned, broke, continued, raised
            returned = returned or outcome.returned
            broke = broke or outcome.broke
            continued = continued or outcome.continued
            raised = raised or outcome.raised
            raise_types.update(outcome.raise_types)
            fallthrough_environments.extend(
                outcome.fallthrough_environments
            )
            return_completions.extend(outcome.return_completions)
            implicit_raise_environments.extend(
                outcome.implicit_raise_environments
            )
            break_environments.extend(outcome.break_environments)
            continue_environments.extend(outcome.continue_environments)
            raise_environments.extend(outcome.raise_environments)
            if outcome.falls_through:
                return None
            return _FlowOutcome(
                falls_through=False,
                returned=returned,
                broke=broke,
                continued=continued,
                raised=raised,
                raise_types=frozenset(raise_types),
                return_completions=tuple(return_completions),
                implicit_raise_environments=tuple(
                    implicit_raise_environments
                ),
                break_environments=tuple(break_environments),
                continue_environments=tuple(continue_environments),
                raise_environments=tuple(raise_environments),
            )

        for stmt in statements:
            (
                normal_environments,
                unbound_environments,
                expression_raises,
                expression_implicit,
            ) = (
                self._statement_unbound_outcomes(stmt, env)
            )
            explicit_expression_raises = (
                [("UnboundLocalError", item)
                 for item in unbound_environments]
                + expression_raises
            )
            if explicit_expression_raises or expression_implicit:
                unbound_flow = _FlowOutcome(
                    falls_through=bool(normal_environments),
                    raised=bool(explicit_expression_raises),
                    raise_types=frozenset(
                        name for name, _ in explicit_expression_raises
                    ),
                    raise_environments=tuple(
                        explicit_expression_raises
                    ),
                    implicit_raise_environments=tuple(
                        expression_implicit
                    ),
                )
                terminal = absorb(unbound_flow)
                if terminal is not None:
                    return terminal
            if normal_environments:
                normal_environment = self._join_environments(
                    normal_environments
                )
                env.clear()
                env.update(normal_environment)
            bound_names = {
                name for name, value in env.items()
                if not isinstance(
                    value, (_Unbound, _PossiblyUnboundValue)
                )
            }
            if _statement_may_raise_implicitly(
                    stmt,
                    bound_names,
                    annotations_deferred=self.future_annotations,
            ) \
                    and not isinstance(stmt, (
                        ast.Delete, ast.With, ast.AsyncWith,
                    )):
                implicit_raise_environments.append(
                    _ImplicitRaiseState(
                        dict(env),
                        ((UnboundLocalError,)
                         if unbound_environments else ()),
                    )
                )
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)) \
                    and not (
                        isinstance(stmt, ast.AnnAssign)
                        and stmt.value is None
                    ):
                value_node = stmt.value
                targets = list(stmt.targets) \
                    if isinstance(stmt, ast.Assign) else [stmt.target]
                value = self._eval(value_node, env)
                first_name = next(
                    (
                        item.id
                        for target in targets
                        for item in ast.walk(target)
                        if isinstance(item, ast.Name)
                        and isinstance(item.ctx, ast.Store)
                    ),
                    f"model_at_line_{stmt.lineno}",
                )
                if _is_model_builder_call(value_node):
                    value = self._model_from_constants(first_name)
                elif value is UNKNOWN:
                    seeded = self._initial_value(first_name)
                    if seeded is not UNKNOWN:
                        value = seeded
                if self.observe_reported_metrics and (
                        len(targets) > 1
                        or isinstance(targets[0], ast.Name)):
                    self._record_metric(value_node, env, value)

                target_normal = [dict(env)]
                target_raises: list[
                    tuple[str, dict[str, _Value]]
                ] = []
                target_implicit: list[_ImplicitRaiseState] = []
                for target in targets:
                    target_normal, item_raises, item_implicit = (
                        self._transition_target_outcomes(
                            target, target_normal, value=value
                        )
                    )
                    target_raises.extend(item_raises)
                    target_implicit.extend(item_implicit)
                    if not target_normal:
                        break
                if target_normal:
                    env.clear()
                    env.update(self._join_environments(target_normal))
                target_flow = _FlowOutcome(
                    falls_through=bool(target_normal),
                    fallthrough_environments=tuple(target_normal),
                    raised=bool(target_raises),
                    raise_types=frozenset(
                        name for name, _ in target_raises
                    ),
                    implicit_raise_environments=tuple(target_implicit),
                    raise_environments=tuple(
                        target_raises
                    ),
                )
                terminal = absorb(target_flow)
                if terminal is not None:
                    return terminal
            elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                right = self._eval(stmt.value, env)
                left = env.get(stmt.target.id, UNKNOWN)
                loss = _merge_losses(left, right)
                if loss is not None:
                    env[stmt.target.id] = loss
                    continue
                left_num, right_num = _number(left), _number(right)
                if left_num is not None and right_num is not None \
                        and isinstance(stmt.op, ast.Add):
                    env[stmt.target.id] = _add_numbers(left_num, right_num)
                else:
                    env[stmt.target.id] = UNKNOWN
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if isinstance(call.func, ast.Attribute) and call.func.attr == "backward":
                    loss = self._eval(call.func.value, env)
                    if isinstance(loss, _LossValue):
                        self._emit_loss(loss, FITTING, call)
                else:
                    value = self._eval(call, env)
                    if self.observe_reported_metrics:
                        self._record_metric(call, env, value)
            elif isinstance(stmt, ast.Return):
                value: _Value | None = None
                if stmt.value is not None:
                    value = self._eval(stmt.value, env)
                    self.return_values.append(value)
                    if self.observe_returned_metrics:
                        self._record_metric(stmt.value, env, value)
                return _FlowOutcome(
                    falls_through=False,
                    returned=True,
                    broke=broke,
                    continued=continued,
                    raised=raised,
                    raise_types=frozenset(raise_types),
                    return_completions=tuple(
                        return_completions + [(value, dict(env))]
                    ),
                    implicit_raise_environments=tuple(
                        implicit_raise_environments
                    ),
                    break_environments=tuple(break_environments),
                    continue_environments=tuple(continue_environments),
                    raise_environments=tuple(raise_environments),
                )
            elif isinstance(stmt, ast.Raise):
                exception_name = _exception_type_name(stmt.exc)
                return _FlowOutcome(
                    falls_through=False,
                    returned=returned,
                    broke=broke,
                    continued=continued,
                    raised=True,
                    raise_types=frozenset(
                        raise_types | {exception_name}
                    ),
                    return_completions=tuple(return_completions),
                    implicit_raise_environments=tuple(
                        implicit_raise_environments
                    ),
                    break_environments=tuple(break_environments),
                    continue_environments=tuple(continue_environments),
                    raise_environments=tuple(
                        raise_environments
                        + [(exception_name, dict(env))]
                    ),
                )
            elif isinstance(stmt, ast.Break):
                return _FlowOutcome(
                    falls_through=False,
                    returned=returned,
                    broke=True,
                    continued=continued,
                    raised=raised,
                    raise_types=frozenset(raise_types),
                    return_completions=tuple(return_completions),
                    implicit_raise_environments=tuple(
                        implicit_raise_environments
                    ),
                    break_environments=tuple(
                        break_environments + [dict(env)]
                    ),
                    continue_environments=tuple(continue_environments),
                    raise_environments=tuple(raise_environments),
                )
            elif isinstance(stmt, ast.Continue):
                return _FlowOutcome(
                    falls_through=False,
                    returned=returned,
                    broke=broke,
                    continued=True,
                    raised=raised,
                    raise_types=frozenset(raise_types),
                    return_completions=tuple(return_completions),
                    implicit_raise_environments=tuple(
                        implicit_raise_environments
                    ),
                    break_environments=tuple(break_environments),
                    continue_environments=tuple(
                        continue_environments + [dict(env)]
                    ),
                    raise_environments=tuple(raise_environments),
                )
            elif isinstance(stmt, ast.Delete):
                delete_normal = [dict(env)]
                delete_raises: list[
                    tuple[str, dict[str, _Value]]
                ] = []
                delete_implicit: list[_ImplicitRaiseState] = []
                for target in stmt.targets:
                    delete_normal, item_raises, item_implicit = (
                        self._transition_target_outcomes(
                            target, delete_normal, delete=True
                        )
                    )
                    delete_raises.extend(item_raises)
                    delete_implicit.extend(item_implicit)
                    if not delete_normal:
                        break
                if delete_normal:
                    env.clear()
                    env.update(self._join_environments(delete_normal))
                delete_flow = _FlowOutcome(
                    falls_through=bool(delete_normal),
                    fallthrough_environments=tuple(delete_normal),
                    raised=bool(delete_raises),
                    raise_types=frozenset(
                        name for name, _ in delete_raises
                    ),
                    implicit_raise_environments=tuple(delete_implicit),
                    raise_environments=tuple(
                        delete_raises
                    ),
                )
                terminal = absorb(delete_flow)
                if terminal is not None:
                    return terminal
            elif isinstance(stmt, ast.Assert):
                condition = self._condition(stmt.test, env)
                assertion_raise = _FlowOutcome(
                    falls_through=condition is not False,
                    raised=condition is not True,
                    raise_types=(
                        frozenset({"AssertionError"})
                        if condition is not True else frozenset()
                    ),
                    raise_environments=(
                        (("AssertionError", dict(env)),)
                        if condition is not True else ()
                    ),
                )
                terminal = absorb(assertion_raise)
                if terminal is not None:
                    return terminal
            elif isinstance(stmt, (ast.For, ast.AsyncFor)):
                domain = self._eval(stmt.iter, env)
                if isinstance(stmt, ast.For) \
                        and isinstance(domain, _SequenceValue):
                    literal_flow = self._run_literal_for(
                        stmt,
                        domain.values,  # type: ignore[arg-type]
                        env,
                    )
                    terminal = absorb(literal_flow)
                    if terminal is not None:
                        return terminal
                    continue
                if isinstance(domain, _SequenceValue) and not domain.values:
                    terminal = absorb(self._run_block(stmt.orelse, env))
                    if terminal is not None:
                        return terminal
                    continue
                guaranteed_iteration = isinstance(domain, _NumberRange) \
                    or isinstance(domain, _SequenceValue) \
                    and bool(domain.values)
                if isinstance(stmt.target, ast.Name) \
                        and isinstance(domain, _NumberRange):
                    domain = _NumberRange(
                        domain.lower,
                        domain.upper,
                        domain.support_envelope
                        or _loop_body_may_use_partial_domain(stmt.body),
                        f"{self.scope}:{stmt.lineno}:{stmt.target.id}",
                    )
                multi_valued_domain = (
                    isinstance(domain, _NumberRange)
                    and (
                        domain.lower != domain.upper
                        or domain.support_envelope
                    )
                    or isinstance(domain, _SequenceValue)
                    and len(domain.values) > 1
                )
                entry = dict(env)
                iteration_values: Sequence[_Value]
                if isinstance(domain, _SequenceValue):
                    iteration_values = domain.values  # type: ignore[assignment]
                elif isinstance(domain, _RangeIterableValue) \
                        and isinstance(stmt.target, (ast.Tuple, ast.List)):
                    iteration_values = (_NumberRange(
                        domain.lower,
                        domain.upper,
                        domain.support_envelope,
                        domain.symbolic_id,
                    ),)
                else:
                    iteration_values = (domain,)
                target_normal: list[dict[str, _Value]] = []
                target_raises: list[
                    tuple[str, dict[str, _Value]]
                ] = []
                target_implicit: list[_ImplicitRaiseState] = []
                for iteration_value in iteration_values:
                    item_normal, item_raises, item_implicit = (
                        self._transition_target_outcomes(
                            stmt.target,
                            [dict(env)],
                            value=iteration_value,
                        )
                    )
                    target_normal.extend(item_normal)
                    target_raises.extend(item_raises)
                    target_implicit.extend(item_implicit)
                target_normal = self._dedupe_environments(target_normal)
                target_raises = self._dedupe_raise_environments(
                    target_raises
                )

                body_flows: list[_FlowOutcome] = []
                body_envs: list[dict[str, _Value]] = []
                for target_env in target_normal:
                    body_env = dict(target_env)
                    body_envs.append(body_env)
                    body_flows.append(self._run_block(
                        stmt.body, body_env
                    ))
                body_flow = _union_flow(*body_flows) \
                    if body_flows else _FlowOutcome(False)
                body_live = list(body_flow.fallthrough_environments)
                if body_flow.falls_through and not body_live:
                    body_live.extend(body_envs)
                escaped_body_envs = tuple(
                    _loop_escape_environment(
                        entry, item, multi_valued=multi_valued_domain,
                    )
                    for item in body_live
                )
                escaped_continue_envs = tuple(
                    _loop_escape_environment(
                        entry, item, multi_valued=multi_valued_domain,
                    )
                    for item in body_flow.continue_environments
                )
                escaped_break_envs = tuple(
                    _loop_escape_environment(
                        entry, item, multi_valued=multi_valued_domain,
                    )
                    for item in body_flow.break_environments
                )
                normal_envs: list[dict[str, _Value]] = []
                if not guaranteed_iteration:
                    normal_envs.append(entry)
                if body_flow.falls_through:
                    normal_envs.extend(escaped_body_envs)
                normal_envs.extend(escaped_continue_envs)

                else_flow = _FlowOutcome(False)
                else_flows: list[_FlowOutcome] = []
                if normal_envs:
                    for normal_env in self._dedupe_environments(normal_envs):
                        else_env = dict(normal_env)
                        else_flows.append(self._run_block(
                            stmt.orelse, else_env
                        ))
                    else_flow = _union_flow(*else_flows)

                live_envs: list[dict[str, _Value]] = []
                live_envs.extend(escaped_break_envs)
                live_envs.extend(else_flow.fallthrough_environments)
                live_envs.extend(else_flow.break_environments)
                live_envs.extend(else_flow.continue_environments)
                loop_flow = _FlowOutcome(
                    falls_through=bool(live_envs),
                    fallthrough_environments=tuple(live_envs),
                    returned=body_flow.returned or else_flow.returned,
                    raised=(
                        bool(target_raises)
                        or body_flow.raised
                        or else_flow.raised
                    ),
                    raise_types=(
                        body_flow.raise_types | else_flow.raise_types
                        | {name for name, _ in target_raises}
                    ),
                    raise_environments=(
                        tuple(target_raises)
                        + body_flow.raise_environments
                        + else_flow.raise_environments
                    ),
                    return_completions=(
                        body_flow.return_completions
                        + else_flow.return_completions
                    ),
                    implicit_raise_environments=(
                        tuple(target_implicit)
                        + tuple(
                            _ImplicitRaiseState(
                                _loop_escape_environment(
                                    entry, item.environment,
                                    multi_valued=multi_valued_domain,
                                ),
                                item.excluded_classes,
                            )
                            for item
                            in body_flow.implicit_raise_environments
                        )
                        + else_flow.implicit_raise_environments
                    ),
                )
                env.clear()
                env.update(self._join_environments(
                    live_envs or list(escaped_body_envs)
                    or body_envs or [entry]
                ))
                terminal = absorb(loop_flow)
                if terminal is not None:
                    return terminal
            elif isinstance(stmt, ast.While):
                initial_condition = self._condition(stmt.test, env)
                if initial_condition is False:
                    terminal = absorb(self._run_block(stmt.orelse, env))
                    if terminal is not None:
                        return terminal
                    continue
                domain = self._while_domain(stmt, env)
                entry = dict(env)
                if domain is None:
                    self._record_unresolved(stmt.test, "while induction domain is unresolved")
                    # Keep walking the body with its induction variables
                    # unknown.  Otherwise a loop whose bound is unsupported
                    # silently drops the fitting or selection target read that
                    # enforcement needs in order to fail closed.
                    trial = dict(env)
                    for inner in stmt.body:
                        if isinstance(inner, ast.AugAssign) \
                                and isinstance(inner.target, ast.Name):
                            trial[inner.target.id] = UNKNOWN
                    body_env = trial
                    guaranteed_iteration = initial_condition is True
                    finite_domain = False
                else:
                    name, values = domain
                    body_env = dict(env)
                    body_env[name] = _NumberRange(
                        values.lower,
                        values.upper,
                        values.support_envelope
                        or _loop_body_may_use_partial_domain(stmt.body),
                        f"{self.scope}:{stmt.lineno}:{name}",
                    )
                    guaranteed_iteration = True
                    finite_domain = True

                body_flow = self._run_block(stmt.body, body_env)
                loop_value = (
                    body_env.get(name)
                    if domain is not None else None
                )
                multi_valued_domain = isinstance(
                    loop_value, _NumberRange
                ) and (
                    loop_value.lower != loop_value.upper
                    or loop_value.support_envelope
                )
                body_live = list(body_flow.fallthrough_environments)
                if body_flow.falls_through and not body_live:
                    body_live.append(body_env)
                escaped_body_envs = tuple(
                    _loop_escape_environment(
                        entry, item, multi_valued=multi_valued_domain,
                    )
                    for item in body_live
                )
                escaped_continue_envs = tuple(
                    _loop_escape_environment(
                        entry, item, multi_valued=multi_valued_domain,
                    )
                    for item in body_flow.continue_environments
                )
                escaped_break_envs = tuple(
                    _loop_escape_environment(
                        entry, item, multi_valued=multi_valued_domain,
                    )
                    for item in body_flow.break_environments
                )
                normal_envs = [] if guaranteed_iteration else [entry]
                if finite_domain or initial_condition is not True:
                    if body_flow.falls_through:
                        normal_envs.extend(escaped_body_envs)
                    normal_envs.extend(escaped_continue_envs)
                else_flow = _FlowOutcome(False)
                else_flows = []
                if normal_envs:
                    for normal_env in self._dedupe_environments(normal_envs):
                        else_env = dict(normal_env)
                        else_flows.append(self._run_block(
                            stmt.orelse, else_env
                        ))
                    else_flow = _union_flow(*else_flows)
                live_envs = list(escaped_break_envs)
                live_envs.extend(else_flow.fallthrough_environments)
                live_envs.extend(else_flow.break_environments)
                live_envs.extend(else_flow.continue_environments)
                loop_flow = _FlowOutcome(
                    falls_through=bool(live_envs),
                    fallthrough_environments=tuple(live_envs),
                    returned=body_flow.returned or else_flow.returned,
                    raised=body_flow.raised or else_flow.raised,
                    raise_types=(
                        body_flow.raise_types | else_flow.raise_types
                    ),
                    raise_environments=(
                        body_flow.raise_environments
                        + else_flow.raise_environments
                    ),
                    return_completions=(
                        body_flow.return_completions
                        + else_flow.return_completions
                    ),
                    implicit_raise_environments=(
                        tuple(
                            _ImplicitRaiseState(
                                _loop_escape_environment(
                                    entry, item.environment,
                                    multi_valued=multi_valued_domain,
                                ),
                                item.excluded_classes,
                            )
                            for item
                            in body_flow.implicit_raise_environments
                        )
                        + else_flow.implicit_raise_environments
                    ),
                )
                env.clear()
                env.update(self._join_environments(
                    live_envs or list(escaped_body_envs) or [body_env]
                ))
                terminal = absorb(loop_flow)
                if terminal is not None:
                    return terminal
            elif isinstance(stmt, ast.If):
                if _captures_model_state(stmt.body + stmt.orelse):
                    for loss in self._loss_values(stmt.test, env):
                        self._emit_loss(loss, MODEL_SELECTION, stmt.test)
                condition = self._condition(stmt.test, env)
                if condition is True:
                    terminal = absorb(self._run_block(stmt.body, env))
                    if terminal is not None:
                        return terminal
                elif condition is False:
                    terminal = absorb(self._run_block(stmt.orelse, env))
                    if terminal is not None:
                        return terminal
                else:
                    left = dict(env)
                    right = dict(env)
                    left_flow = self._run_block(stmt.body, left)
                    right_flow = self._run_block(stmt.orelse, right)
                    live_environments = [
                        branch_env
                        for branch_env, branch_flow in (
                            (left, left_flow), (right, right_flow),
                        )
                        if branch_flow.falls_through
                    ]
                    env.clear()
                    env.update(self._join_environments(
                        live_environments or [left, right]
                    ))
                    branch_flow = _union_flow(
                        left_flow,
                        right_flow,
                        falls_through=bool(live_environments),
                    )
                    terminal = absorb(branch_flow)
                    if terminal is not None:
                        return terminal
            elif isinstance(stmt, ast.Match):
                subject = self._eval(stmt.subject, env)
                case_paths: list[
                    tuple[dict[str, _Value], _FlowOutcome]
                ] = []
                unmatched_possible = True
                for case in stmt.cases:
                    if not unmatched_possible:
                        break
                    pattern_status = _match_pattern_status(
                        case.pattern, subject
                    )
                    if pattern_status is False:
                        continue
                    case_env = dict(env)
                    _bind_match_pattern(case.pattern, subject, case_env)
                    guard = (
                        True if case.guard is None
                        else self._condition(case.guard, case_env)
                    )
                    if guard is not False:
                        case_paths.append((
                            case_env,
                            self._run_block(case.body, case_env),
                        ))
                    if pattern_status is True and guard is True:
                        unmatched_possible = False
                if unmatched_possible:
                    unmatched_env = dict(env)
                    case_paths.append((
                        unmatched_env,
                        _FlowOutcome(
                            fallthrough_environments=(unmatched_env,),
                        ),
                    ))
                live_environments = [
                    case_env for case_env, case_flow in case_paths
                    if case_flow.falls_through
                ]
                env.clear()
                env.update(self._join_environments(
                    live_environments
                    or [case_env for case_env, _ in case_paths]
                ))
                match_flow = _union_flow(
                    *(case_flow for _, case_flow in case_paths),
                    falls_through=bool(live_environments),
                )
                terminal = absorb(match_flow)
                if terminal is not None:
                    return terminal
            elif isinstance(stmt, ast.Try):
                try_return_start = len(self.return_values)
                entry = dict(env)
                body_env = dict(entry)
                protected_flow = self._run_block(stmt.body, body_env)
                body_flow = protected_flow
                else_flow = _FlowOutcome(False)
                if protected_flow.falls_through and stmt.orelse:
                    protected_terminal = _FlowOutcome(
                        falls_through=False,
                        returned=protected_flow.returned,
                        broke=protected_flow.broke,
                        continued=protected_flow.continued,
                        raised=protected_flow.raised,
                        raise_types=protected_flow.raise_types,
                        return_completions=(
                            protected_flow.return_completions
                        ),
                        implicit_raise_environments=(
                            protected_flow.implicit_raise_environments
                        ),
                        break_environments=(
                            protected_flow.break_environments
                        ),
                        continue_environments=(
                            protected_flow.continue_environments
                        ),
                        raise_environments=protected_flow.raise_environments,
                    )
                    else_flows: list[_FlowOutcome] = []
                    for protected_env in (
                            protected_flow.fallthrough_environments):
                        else_env = dict(protected_env)
                        else_flows.append(self._run_block(
                            stmt.orelse, else_env
                        ))
                    else_flow = _union_flow(*else_flows)
                    body_flow = _union_flow(
                        protected_terminal, else_flow,
                    )

                unmatched_raises = list(
                    protected_flow.raise_environments
                )
                path_envs = [body_env]
                path_flows: list[_FlowOutcome] = []
                implicit_coverage: list[type[BaseException]] = []
                for handler in stmt.handlers:
                    handler_entries: list[dict[str, _Value]] = []
                    still_unmatched: list[
                        tuple[str | None, dict[str, _Value]]
                    ] = []
                    for exception_name, exception_env in unmatched_raises:
                        if _handler_matches(handler, exception_name):
                            handler_entries.append(exception_env)
                        else:
                            still_unmatched.append((
                                exception_name, exception_env,
                            ))
                    unmatched_raises = still_unmatched
                    for implicit_state in (
                            protected_flow.implicit_raise_environments):
                        if not _implicit_handler_fully_covered(
                                handler,
                                implicit_state.excluded_classes
                                + tuple(implicit_coverage)):
                            handler_entries.append(
                                implicit_state.environment
                            )
                    handler_classes = _handler_builtin_exception_classes(
                        handler
                    )
                    if handler_classes is None:
                        implicit_coverage = [BaseException]
                    else:
                        implicit_coverage.extend(handler_classes)
                    for handler_entry in handler_entries:
                        handler_env = dict(handler_entry)
                        if handler.name is not None:
                            # Python shadows and then deletes the exception
                            # target.  UNKNOWN preserves neither an older
                            # temporal alias nor invented exception lineage.
                            handler_env[handler.name] = UNKNOWN
                        handler_flow = self._run_block(
                            handler.body, handler_env
                        )
                        if handler.name is not None:
                            handler_env[handler.name] = UNBOUND
                            handler_flow = _flow_with_unbound_name(
                                handler_flow, handler.name
                            )
                        path_envs.append(handler_env)
                        path_flows.append(handler_flow)

                remaining_raises = (
                    unmatched_raises
                    + list(else_flow.raise_environments)
                )
                catches_all_implicit = _handlers_catch_all_implicit(
                    stmt.handlers
                )
                remaining_implicit_raises = list(
                    else_flow.implicit_raise_environments
                )
                if not catches_all_implicit:
                    for implicit_state in (
                            protected_flow.implicit_raise_environments):
                        exclusions = tuple(dict.fromkeys(
                            implicit_state.excluded_classes
                            + tuple(implicit_coverage)
                        ))
                        remaining_implicit_raises.append(
                            _ImplicitRaiseState(
                                implicit_state.environment, exclusions,
                            )
                        )
                body_flow = _FlowOutcome(
                    falls_through=body_flow.falls_through,
                    fallthrough_environments=(
                        body_flow.fallthrough_environments
                    ),
                    returned=body_flow.returned,
                    broke=body_flow.broke,
                    continued=body_flow.continued,
                    raised=bool(remaining_raises),
                    raise_types=frozenset(
                        name for name, _ in remaining_raises
                    ),
                    return_completions=body_flow.return_completions,
                    implicit_raise_environments=tuple(
                        remaining_implicit_raises
                    ),
                    break_environments=body_flow.break_environments,
                    continue_environments=body_flow.continue_environments,
                    raise_environments=tuple(remaining_raises),
                )
                path_flows.insert(0, body_flow)

                prior_flow = _union_flow(
                    *path_flows,
                    falls_through=any(
                        item.falls_through for item in path_flows
                    ),
                )
                live_environments = [
                    live_env
                    for path_env, path_flow in zip(path_envs, path_flows)
                    for live_env in (
                        path_flow.fallthrough_environments
                        or ((path_env,) if path_flow.falls_through else ())
                    )
                ]
                if stmt.finalbody:
                    completions: list[
                        tuple[str, object, dict[str, _Value]]
                    ] = []
                    for path_env, path_flow in zip(path_envs, path_flows):
                        for live_env in (
                                path_flow.fallthrough_environments
                                or ((path_env,)
                                    if path_flow.falls_through else ())):
                            completions.append((
                                "fallthrough", None, dict(live_env),
                            ))
                        completions.extend(
                            ("return", value, dict(return_env))
                            for value, return_env
                            in path_flow.return_completions
                        )
                        completions.extend(
                            ("break", None, dict(break_env))
                            for break_env in path_flow.break_environments
                        )
                        completions.extend(
                            ("continue", None, dict(continue_env))
                            for continue_env
                            in path_flow.continue_environments
                        )
                        completions.extend(
                            ("raise", exception_name, dict(raise_env))
                            for exception_name, raise_env
                            in path_flow.raise_environments
                        )
                        completions.extend(
                            (
                                "implicit_raise", implicit_state,
                                dict(implicit_state.environment),
                            )
                            for implicit_state
                            in path_flow.implicit_raise_environments
                        )

                    # Original returns survive only on paths where ``finally``
                    # falls through.  Rebuild the aggregate after executing
                    # the final suite independently for every completion.
                    del self.return_values[try_return_start:]
                    final_outcomes: list[_FlowOutcome] = []
                    final_live_environments: list[dict[str, _Value]] = []
                    for kind, completion_value, completion_env in completions:
                        final_env = dict(completion_env)
                        final_flow = self._run_block(
                            stmt.finalbody, final_env
                        )
                        if final_flow.returned or final_flow.broke \
                                or final_flow.continued or final_flow.raised \
                                or final_flow.implicit_raise_environments:
                            final_outcomes.append(_FlowOutcome(
                                falls_through=False,
                                returned=final_flow.returned,
                                broke=final_flow.broke,
                                continued=final_flow.continued,
                                raised=final_flow.raised,
                                raise_types=final_flow.raise_types,
                                return_completions=(
                                    final_flow.return_completions
                                ),
                                implicit_raise_environments=(
                                    final_flow.implicit_raise_environments
                                ),
                                break_environments=(
                                    final_flow.break_environments
                                ),
                                continue_environments=(
                                    final_flow.continue_environments
                                ),
                                raise_environments=(
                                    final_flow.raise_environments
                                ),
                            ))
                        if not final_flow.falls_through:
                            continue
                        surviving_envs = (
                            final_flow.fallthrough_environments
                            or (final_env,)
                        )
                        for surviving_env in surviving_envs:
                            if kind == "fallthrough":
                                final_live_environments.append(surviving_env)
                                final_outcomes.append(_FlowOutcome(
                                    fallthrough_environments=(
                                        surviving_env,
                                    ),
                                ))
                            elif kind == "return":
                                return_value = completion_value \
                                    if not isinstance(
                                        completion_value, str
                                    ) else None
                                final_outcomes.append(_FlowOutcome(
                                    falls_through=False,
                                    returned=True,
                                    return_completions=((
                                        return_value, surviving_env,
                                    ),),
                                ))
                                if return_value is not None:
                                    self.return_values.append(return_value)
                            elif kind == "break":
                                final_outcomes.append(_FlowOutcome(
                                    falls_through=False,
                                    broke=True,
                                    break_environments=(surviving_env,),
                                ))
                            elif kind == "continue":
                                final_outcomes.append(_FlowOutcome(
                                    falls_through=False,
                                    continued=True,
                                    continue_environments=(surviving_env,),
                                ))
                            elif kind == "raise":
                                exception_name = completion_value \
                                    if isinstance(
                                        completion_value, str
                                    ) else None
                                final_outcomes.append(_FlowOutcome(
                                    falls_through=False,
                                    raised=True,
                                    raise_types=frozenset({exception_name}),
                                    raise_environments=((
                                        exception_name, surviving_env,
                                    ),),
                                ))
                            else:
                                implicit_state = completion_value \
                                    if isinstance(
                                        completion_value,
                                        _ImplicitRaiseState,
                                    ) else _ImplicitRaiseState(surviving_env)
                                final_outcomes.append(_FlowOutcome(
                                    falls_through=False,
                                    raised=True,
                                    raise_types=frozenset({None}),
                                    implicit_raise_environments=(
                                        _ImplicitRaiseState(
                                            surviving_env,
                                            implicit_state.excluded_classes,
                                        ),
                                    ),
                                ))
                    prior_flow = _union_flow(
                        *final_outcomes,
                        falls_through=bool(final_live_environments),
                    )
                    live_environments = final_live_environments
                env.clear()
                env.update(self._join_environments(
                    live_environments or path_envs
                ))
                terminal = absorb(prior_flow)
                if terminal is not None:
                    return terminal
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                with_entry = dict(env)
                with_environments = [dict(with_entry)]
                with_raises: list[
                    tuple[str, dict[str, _Value]]
                ] = []
                with_implicit: list[_ImplicitRaiseState] = []
                for item in stmt.items:
                    context_normal: list[dict[str, _Value]] = []
                    for with_env in with_environments:
                        item_normal, item_raised = (
                            self._unbound_expression_outcomes(
                                item.context_expr, with_env
                            )
                        )
                        context_normal.extend(item_normal)
                        with_raises.extend(
                            ("UnboundLocalError", state)
                            for state in item_raised
                        )
                    context_normal = self._dedupe_environments(
                        context_normal
                    )
                    with_implicit.extend(
                        _ImplicitRaiseState(dict(context_env))
                        for context_env in context_normal
                    )
                    if item.optional_vars is None:
                        with_environments = context_normal
                        continue

                    target_normal, target_raised, target_implicit = (
                        self._transition_target_outcomes(
                            item.optional_vars,
                            context_normal,
                            value=UNKNOWN,
                        )
                    )
                    with_raises.extend(target_raised)
                    with_implicit.extend(target_implicit)
                    with_environments = target_normal
                    if not with_environments:
                        break

                body_flows: list[_FlowOutcome] = []
                for with_env in with_environments:
                    body_flows.append(self._run_block(
                        stmt.body, with_env
                    ))
                body_flow = _union_flow(*body_flows) \
                    if body_flows else _FlowOutcome(False)
                live_environments = list(
                    body_flow.fallthrough_environments
                )
                with_flow = _FlowOutcome(
                    falls_through=body_flow.falls_through,
                    fallthrough_environments=tuple(live_environments),
                    returned=body_flow.returned,
                    broke=body_flow.broke,
                    continued=body_flow.continued,
                    raised=bool(with_raises) or body_flow.raised,
                    raise_types=(
                        body_flow.raise_types
                        | {name for name, _ in with_raises}
                    ),
                    return_completions=body_flow.return_completions,
                    implicit_raise_environments=(
                        tuple(with_implicit)
                        + body_flow.implicit_raise_environments
                    ),
                    break_environments=body_flow.break_environments,
                    continue_environments=(
                        body_flow.continue_environments
                    ),
                    raise_environments=(
                        tuple(with_raises)
                        + body_flow.raise_environments
                    ),
                )
                env.clear()
                env.update(self._join_environments(
                    live_environments or with_environments or [with_entry]
                ))
                terminal = absorb(with_flow)
                if terminal is not None:
                    return terminal
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                env[stmt.name] = UNKNOWN
            elif isinstance(stmt, ast.ClassDef):
                class_entry = dict(env)
                class_env = dict(class_entry)
                class_flow = self._run_block(stmt.body, class_env)
                class_live = list(class_flow.fallthrough_environments)
                if class_flow.falls_through and not class_live:
                    class_live.append(class_env)
                nonlocal_names = _scope_nonlocal_names(stmt.body)

                def project_class_state(
                    class_state: Mapping[str, _Value],
                ) -> dict[str, _Value]:
                    outer_state = dict(class_entry)
                    for name in nonlocal_names:
                        if name in class_state:
                            outer_state[name] = class_state[name]
                    return outer_state

                outer_live: list[dict[str, _Value]] = []
                for class_state in class_live:
                    outer_state = project_class_state(class_state)
                    attrs = tuple(sorted(
                        (name, value)
                        for name, value in class_state.items()
                        if name not in nonlocal_names
                        and (
                            name not in class_entry
                            or class_entry[name] != value
                        )
                    ))
                    outer_state[stmt.name] = _NamespaceValue(attrs)
                    outer_live.append(outer_state)
                projected_raises = tuple(
                    (exception_name, project_class_state(class_state))
                    for exception_name, class_state
                    in class_flow.raise_environments
                )
                projected_implicit = tuple(
                    _ImplicitRaiseState(
                        project_class_state(state.environment),
                        state.excluded_classes,
                    )
                    for state in class_flow.implicit_raise_environments
                )
                class_statement_flow = _FlowOutcome(
                    falls_through=bool(outer_live),
                    fallthrough_environments=tuple(outer_live),
                    raised=class_flow.raised,
                    raise_types=class_flow.raise_types,
                    implicit_raise_environments=projected_implicit,
                    raise_environments=projected_raises,
                )
                if outer_live:
                    env.clear()
                    env.update(self._join_environments(outer_live))
                terminal = absorb(class_statement_flow)
                if terminal is not None:
                    return terminal
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    env[alias.asname or alias.name.split(".", 1)[0]] = UNKNOWN
            elif isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    if alias.name != "*":
                        env[alias.asname or alias.name] = UNKNOWN
        return _FlowOutcome(
            falls_through=True,
            fallthrough_environments=tuple(
                self._dedupe_environments(
                    fallthrough_environments or [dict(env)]
                )
            ),
            returned=returned,
            broke=broke,
            continued=continued,
            raised=raised,
            raise_types=frozenset(raise_types),
            return_completions=tuple(return_completions),
            implicit_raise_environments=tuple(
                implicit_raise_environments
            ),
            break_environments=tuple(break_environments),
            continue_environments=tuple(continue_environments),
            raise_environments=tuple(raise_environments),
        )


def _dedupe_observations(
    observations: Iterable[RoleObservation],
) -> tuple[RoleObservation, ...]:
    """Collapse repeated propagation sinks to one semantic role read."""
    unique: dict[tuple[object, ...], RoleObservation] = {}
    for observation in observations:
        key = (
            observation.role,
            observation.read_kind,
            observation.root,
            observation.protocol_range,
            observation.subset,
            observation.model_id,
            observation.source,
        )
        previous = unique.get(key)
        if previous is None or observation.line < previous.line:
            unique[key] = observation
    return tuple(sorted(unique.values(), key=lambda item: (
        item.model_id, item.root, item.role, item.protocol_range,
        item.source, item.line, item.expression,
    )))


def _relations(observations: Iterable[RoleObservation]) -> tuple[RangeRelation, ...]:
    unique = _dedupe_observations(observations)
    unique = sorted(unique, key=lambda item: (
        item.model_id, item.root, item.role, item.protocol_range,
        item.source, item.line, item.expression,
    ))
    relations: list[RangeRelation] = []
    for index, first in enumerate(unique):
        for second in unique[index + 1:]:
            if first.model_id != second.model_id or first.root != second.root \
                    or first.role == second.role:
                continue
            overlap = first.protocol_range.intersection(second.protocol_range)
            ordered = sorted((first, second), key=lambda item: item.role)
            relations.append(RangeRelation(
                model_id=first.model_id,
                root=first.root,
                first_role=ordered[0].role,
                first_read_kind=ordered[0].read_kind,
                second_role=ordered[1].role,
                second_read_kind=ordered[1].read_kind,
                first_range=ordered[0].protocol_range,
                second_range=ordered[1].protocol_range,
                status="overlap" if overlap is not None else "disjoint",
                overlap=overlap,
                certainty=(
                    "support"
                    if first.subset is not None or second.subset is not None
                    else "exact"
                ),
            ))
    return tuple(sorted(set(relations), key=lambda item: (
        item.model_id, item.root, item.first_role, item.second_role,
        item.first_read_kind, item.second_read_kind,
        item.first_range, item.second_range,
    )))


def _final_report(analyzer: _Analyzer) -> SplitLineageReport:
    observations = _dedupe_observations(analyzer.observations)
    return SplitLineageReport(
        observations=observations,
        relations=_relations(observations),
        calls=tuple(analyzer.calls),
        unresolved=tuple(analyzer.unresolved),
    )


def analyze_split_lineage(
    notebook: Union[str, ast.Module],
    trainers: Mapping[str, Union[str, ast.Module]],
    *,
    constants: Mapping[str, object] | None = None,
    entry_function: str | None = None,
    observe_reported_metrics: bool = True,
    observe_returned_metrics: bool | None = None,
    project_child_observations: bool = False,
    source_label: str = "notebook",
) -> SplitLineageReport:
    """Analyze a notebook/module plus one-level trainer bodies.

    ``constants`` supplies concrete reduction facts.  A tensor protocol-axis
    extent is written as ``"demand_matrix.protocol_length": 1033``.  Model
    facts use dotted attributes such as ``"model.context_length": 10``.
    Ordinary scalar formals use their own names.  When ``entry_function`` is
    omitted, source-order analysis starts at module scope.
    """
    tree = _parse(notebook)
    trainer_trees = {name: _parse(source) for name, source in trainers.items()}
    analyzer = _Analyzer(
        trainers=trainer_trees,
        constants=constants or {},
        observe_reported_metrics=observe_reported_metrics,
        observe_returned_metrics=(
            observe_reported_metrics
            if observe_returned_metrics is None
            else observe_returned_metrics
        ),
        project_child_observations=project_child_observations,
    )
    analyzer.future_annotations = _future_annotations_enabled(tree)
    analyzer.source_label = source_label
    entry = _function(tree, entry_function) if entry_function else None
    if entry is not None:
        analyzer.scope = entry.name
        analyzer.local_names = _function_local_names(entry)
        env = analyzer._entry_env(entry)
        model = next((value for value in env.values()
                      if isinstance(value, _ModelValue)), None)
        if model is not None:
            analyzer.active_model = model.identity
        analyzer._run_block(entry.body, env)
    else:
        env = {
            key: analyzer._initial_value(key)
            for key in (constants or {})
            if "." not in key
        }
        analyzer._run_block(tree.body, env)
    return _final_report(analyzer)


def summarize_trainer(
    source: Union[str, ast.Module],
    *,
    function_name: str = "train_model",
    constants: Mapping[str, object] | None = None,
    observe_reported_metrics: bool = False,
    observe_returned_metrics: bool | None = None,
) -> SplitLineageReport:
    """Analyze one trainer relative to concrete formal and model facts."""
    tree = _parse(source)
    func = _function(tree, function_name)
    if func is None:
        raise ValueError(f"function `{function_name}` was not found")
    analyzer = _Analyzer(
        trainers={function_name: tree},
        constants=constants or {},
        observe_reported_metrics=observe_reported_metrics,
        observe_returned_metrics=(
            observe_reported_metrics
            if observe_returned_metrics is None
            else observe_returned_metrics
        ),
    )
    analyzer.future_annotations = _future_annotations_enabled(tree)
    analyzer.scope = function_name
    analyzer.source_label = function_name
    analyzer.local_names = _function_local_names(func)
    env = analyzer._entry_env(func)
    model = next((value for value in env.values()
                  if isinstance(value, _ModelValue)), None)
    if model is not None:
        analyzer.active_model = model.identity
    analyzer._run_block(func.body, env)
    return _final_report(analyzer)
