"""Enforce temporal split lineage at the generated-code producer seams.

``eval_split_lineage`` remains the source-ordered report generator.  This
module owns the policy layered over that report:

* fitting targets, model-selection targets, and reported-evaluation targets
  must be pairwise disjoint for the same root and scored model;
* inference-conditioning history remains a separate read kind and is handled
  by the existing notebook range arm, rather than being mistaken for a target;
* an exact overlap is a concrete defect;
* overlap with a masked support envelope is unresolved disjointness, never an
  exact element-set claim; and
* an unresolved read fails closed only when it can participate in one of the
  forbidden same-root, same-model relationships above.

The current enforcement scope is the recorded time-series forecasting family.
The validation facts below are a deterministic analysis fixture.  They do not
declare a paper protocol or a realized partition plan, both of which are owned
by later typed protocol work.
"""

from __future__ import annotations

import ast
import builtins
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Union

from scripts.eval_split_lineage import (
    FITTING,
    INFERENCE_CONDITIONING,
    MODEL_SELECTION,
    REPORTED_EVALUATION,
    ProtocolRange,
    RangeRelation,
    RoleObservation,
    SplitLineageReport,
    UnresolvedLineage,
    _definition_evaluated_expressions,
    _future_annotations_enabled,
    _guaranteed_namedexpr_bindings,
    _known_eager_comprehension_failure,
    _literal_iterable_nonempty,
    _literal_unpack_outcome,
    _scope_nonlocal_names,
    _target_evaluation_expressions,
    _unpack_target_accepts_length,
    analyze_split_lineage,
    summarize_trainer,
)


_TEMPORAL_PARADIGMS = frozenset({
    "time_series_forecasting",
    "graph_time_series_forecasting",
    "TE-TSF/time_series_forecasting",
})
_TRAINER_TOKENS = frozenset({"train", "fit", "retrain", "optimize"})
_EVALUATION_FUNCTION_TOKENS = frozenset({
    "evaluate", "evaluation", "score", "scoring", "metric", "metrics",
    "eval", "test",
})
_FORECAST_FUNCTION_TOKENS = frozenset({
    "forecast", "predict", "infer", "inference",
})
_METRIC_CONSUMER_TOKENS = frozenset({
    "metric", "metrics", "score", "scoring", "evaluate", "evaluation",
    "eval", "rmse", "mse", "mae", "mape", "wmape", "accuracy",
    "error", "errors",
})
_NON_EVALUATION_CONSUMER_TOKENS = frozenset({
    "append", "display", "extend", "fill", "hist", "imshow", "log",
    "plot", "print", "render", "scatter", "show", "visualize", "write",
})
_SCALAR_ANNOTATION_TOKENS = frozenset({
    "bool", "boolean", "float", "floating", "int", "integer", "str",
    "string",
})
_PREDICTION_POSTPROCESS_TOKENS = frozenset({
    "distribution", "output", "pack", "package", "result", "sample",
    "sampling",
})
_MODEL_TOKENS = frozenset({"model", "net", "network", "estimator"})
_TEMPORAL_TENSOR_TOKENS = frozenset({
    "demand", "target", "targets", "truth", "actual", "actuals", "series",
    "history", "label", "labels", "observed", "y",
})
_TIME_AXIS_SYMBOLS = frozenset({
    "T", "T_total", "T_max", "T_steps", "n_steps", "num_steps",
    "n_time_steps", "num_time_steps", "time_steps", "seq_len",
    "seq_length", "sequence_length", "n_timesteps",
})
_ROLE_LABELS = {
    FITTING: "fitting target",
    MODEL_SELECTION: "model-selection target",
    INFERENCE_CONDITIONING: "inference-conditioning history",
    REPORTED_EVALUATION: "reported-evaluation target",
}
_FORBIDDEN_RELATIONSHIPS = frozenset({
    frozenset(((FITTING, "target"), (MODEL_SELECTION, "target"))),
    frozenset(((FITTING, "target"), (REPORTED_EVALUATION, "target"))),
    frozenset(((MODEL_SELECTION, "target"),
               (REPORTED_EVALUATION, "target"))),
})
_UNBOUND_LOCAL_DEFINITION = "__r2c_unbound_local__"


def _tokens(name: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", name.lower()) if token}


def _exception_type_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Call):
        return _exception_type_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _handler_matches(
    handler: ast.ExceptHandler, exception_name: str | None,
) -> bool:
    if handler.type is None or exception_name is None:
        return True
    nodes = handler.type.elts \
        if isinstance(handler.type, ast.Tuple) else [handler.type]
    names = {
        name for item in nodes
        if (name := _exception_type_name(item)) is not None
    }
    if not names:
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
            return True
    return False


def _handler_builtin_exception_classes(
    handler: ast.ExceptHandler,
) -> tuple[type[BaseException], ...] | None:
    if handler.type is None:
        return None
    nodes = handler.type.elts \
        if isinstance(handler.type, ast.Tuple) else [handler.type]
    classes: list[type[BaseException]] = []
    for node in nodes:
        name = _exception_type_name(node)
        candidate = getattr(builtins, name, None) if name else None
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


@dataclass(frozen=True)
class _ImplicitExceptionState:
    excluded_classes: tuple[type[BaseException], ...] = ()


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


def _statement_may_raise_implicitly(
    statement: ast.stmt,
    bound_names: set[str] | None = None,
    *,
    annotations_deferred: bool = False,
) -> bool:
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


def _range_text(value: ProtocolRange | None) -> str:
    return "unresolved" if value is None else f"[{value.start},{value.stop})"


def _forbidden(
    first_role: str,
    first_kind: str,
    second_role: str,
    second_kind: str,
) -> bool:
    return frozenset(((first_role, first_kind),
                      (second_role, second_kind))) in _FORBIDDEN_RELATIONSHIPS


@dataclass(frozen=True)
class _Endpoint:
    role: str
    read_kind: str
    root: str
    model_id: str | None
    protocol_range: ProtocolRange | None
    subset: str | None
    source: str
    line: int
    reason: str | None = None


@dataclass(frozen=True)
class SplitLineageFinding:
    """One forbidden overlap or unresolved forbidden relationship."""

    kind: str
    model_id: str | None
    root: str
    first: _Endpoint
    second: _Endpoint | None
    overlap: ProtocolRange | None
    certainty: str

    @property
    def line(self) -> int:
        lines = [item.line for item in (self.first, self.second)
                 if item is not None and item.line]
        return min(lines, default=0)

    def message(
        self,
        file_label: str,
        source_labels: Mapping[str, str] | None = None,
    ) -> str:
        labels = source_labels or {}
        endpoints = [self.first]
        if self.second is not None:
            endpoints.append(self.second)
        primary = next(
            (item for item in endpoints
             if labels.get(item.source, item.source) == file_label),
            endpoints[0],
        )
        primary_source = labels.get(primary.source, primary.source)
        first_source = labels.get(self.first.source, self.first.source)
        first_label = _ROLE_LABELS[self.first.role]
        first_range = _range_text(self.first.protocol_range)
        model_text = (
            f"scored model `{self.model_id}`"
            if self.model_id is not None
            else "an unresolved scored-model binding"
        )
        prefix = (
            f"{primary_source}:{primary.line}: evaluation-split integrity is "
            f"not established for root `{self.root}` and {model_text}: "
            f"{first_label} range {first_range} at "
            f"{first_source}:{self.first.line}"
        )
        if self.second is not None:
            second_source = labels.get(self.second.source, self.second.source)
            second_label = _ROLE_LABELS[self.second.role]
            second_range = _range_text(self.second.protocol_range)
            prefix += (
                f" and {second_label} range {second_range} at "
                f"{second_source}:{self.second.line}"
            )
        if self.kind == "exact_overlap":
            state = f" overlap {_range_text(self.overlap)} exactly."
        elif self.kind == "support_overlap":
            subsets = {
                item.subset for item in endpoints if item.subset is not None
            }
            if "boolean_mask" in subsets:
                envelope = "masked support envelope"
                certainty_note = (
                    "This does not claim every envelope element survived the "
                    "mask"
                )
            elif "sparse_index_envelope" in subsets:
                envelope = "sparse-index support envelope"
                certainty_note = (
                    "This does not claim every index inside the envelope was "
                    "consumed"
                )
            else:
                envelope = "support envelope"
                certainty_note = (
                    "This does not claim every element inside the envelope "
                    "was consumed"
                )
            state = (
                f" have possible overlap {_range_text(self.overlap)} because "
                f"at least one range is a {envelope}. {certainty_note}, and "
                "it does not prove the consumed rows are disjoint."
            )
        else:
            reasons = "; ".join(dict.fromkeys(
                reason for reason in (
                    self.first.reason,
                    self.second.reason if self.second is not None else None,
                ) if reason
            ))
            state = (
                " is unresolved at a relevant temporal target consumer, so "
                "its required disjointness from the other target roles cannot "
                f"be proved ({reasons or 'range lineage unresolved'})."
            )
        owned_sources = {
            labels.get(item.source, item.source) for item in endpoints
        }
        if owned_sources == {"method/training.py"}:
            owner_note = (
                " Fix owner: `r2c-architecture-coder` because both temporal "
                "consumers are in method/training.py. If caller facts made "
                "this relationship decidable later, route the finding across "
                "producers rather than editing the notebook."
            )
        elif owned_sources == {"method/method.py"}:
            owner_note = (
                " Fix owner: `r2c-method-coder` because both temporal "
                "consumers are in method/method.py."
            )
        else:
            owner_note = ""
        return (
            f"{prefix}{state} Required protocol-axis overlap for these roles: "
            "empty. Change the split-boundary arithmetic, slice bounds, or "
            "caller-to-trainer binding until the consumed half-open ranges "
            f"are provably disjoint.{owner_note} "
            f"({'temporal_target_range_unresolved' if self.kind == 'unresolved' else 'temporal_target_role_overlap'})."
        )


@dataclass(frozen=True)
class _NotebookSplitLineageAnalysis:
    """One notebook analysis shared by enforcement and receipt projections."""

    report: SplitLineageReport
    findings: tuple[SplitLineageFinding, ...]
    emitted_findings: tuple[SplitLineageFinding, ...]
    source_labels: Mapping[str, str]


def _observation_endpoint(observation: RoleObservation) -> _Endpoint:
    return _Endpoint(
        role=observation.role,
        read_kind=observation.read_kind,
        root=observation.root,
        model_id=observation.model_id,
        protocol_range=observation.protocol_range,
        subset=observation.subset,
        source=observation.source,
        line=observation.line,
    )


def _unresolved_endpoint(item: UnresolvedLineage) -> _Endpoint | None:
    if item.role not in _ROLE_LABELS or not item.read_kind \
            or not item.root:
        return None
    return _Endpoint(
        role=item.role,
        read_kind=item.read_kind,
        root=item.root,
        model_id=item.model_id,
        protocol_range=item.protocol_range,
        subset=item.subset,
        source=item.source,
        line=item.line,
        reason=item.reason,
    )


def find_split_lineage_findings(
    report: SplitLineageReport,
) -> tuple[SplitLineageFinding, ...]:
    """Apply the temporal consumer policy to a lineage report."""
    observations = tuple(_observation_endpoint(item)
                         for item in report.observations)
    findings: list[SplitLineageFinding] = []

    for relation in report.relations:
        if relation.status != "overlap":
            continue
        first_candidates = [
            item for item in observations
            if item.model_id == relation.model_id
            and item.root == relation.root
            and item.role == relation.first_role
            and item.read_kind == relation.first_read_kind
            and item.protocol_range == relation.first_range
        ]
        second_candidates = [
            item for item in observations
            if item.model_id == relation.model_id
            and item.root == relation.root
            and item.role == relation.second_role
            and item.read_kind == relation.second_read_kind
            and item.protocol_range == relation.second_range
        ]
        pairs = [
            (first, second)
            for first in first_candidates
            for second in second_candidates
            if _forbidden(first.role, first.read_kind,
                          second.role, second.read_kind)
        ]
        if not pairs:
            continue
        first, second = min(pairs, key=lambda pair: (
            pair[0].line, pair[1].line, pair[0].source, pair[1].source,
        ))
        findings.append(SplitLineageFinding(
            kind=("support_overlap" if relation.certainty == "support"
                  else "exact_overlap"),
            model_id=relation.model_id,
            root=relation.root,
            first=first,
            second=second,
            overlap=relation.overlap,
            certainty=relation.certainty,
        ))

    unresolved = tuple(
        endpoint
        for endpoint in (_unresolved_endpoint(item) for item in report.unresolved)
        if endpoint is not None
    )
    concrete = observations
    emitted_pairs: set[tuple[int, int]] = set()
    for index, item in enumerate(unresolved):
        if item.read_kind != "target" or item.role not in {
                FITTING, MODEL_SELECTION, REPORTED_EVALUATION}:
            continue
        candidates = tuple(concrete) + tuple(unresolved)
        for candidate in candidates:
            if candidate is item:
                continue
            if candidate.root != item.root or not _forbidden(
                    item.role, item.read_kind,
                    candidate.role, candidate.read_kind):
                continue
            if item.model_id is not None and candidate.model_id is not None \
                    and item.model_id != candidate.model_id:
                continue
            if item.protocol_range is not None \
                    and candidate.protocol_range is not None \
                    and item.protocol_range.intersection(
                        candidate.protocol_range
                    ) is None:
                continue
            if candidate in unresolved:
                other_index = unresolved.index(candidate)
                pair = tuple(sorted((index, other_index)))
                if pair in emitted_pairs:
                    continue
                emitted_pairs.add(pair)
            model_id = (
                item.model_id
                if item.model_id is not None
                and item.model_id == candidate.model_id
                else None
            )
            findings.append(SplitLineageFinding(
                kind="unresolved",
                model_id=model_id,
                root=item.root,
                first=item,
                second=candidate,
                overlap=None,
                certainty="unresolved",
            ))

    return tuple(sorted(findings, key=lambda item: (
        item.line, item.model_id or "", item.root, item.first.role,
        item.second.role if item.second is not None else "", item.kind,
    )))


def _serialized_range(
    value: ProtocolRange | None,
) -> dict[str, int] | None:
    if value is None:
        return None
    return {"start": value.start, "stop": value.stop}


def _serialized_endpoint(endpoint: _Endpoint) -> dict[str, object]:
    return {
        "role": endpoint.role,
        "read_kind": endpoint.read_kind,
        "root": endpoint.root,
        "model_id": endpoint.model_id,
        "protocol_range": _serialized_range(endpoint.protocol_range),
        "subset": endpoint.subset,
        "source": endpoint.source,
        "line": endpoint.line,
        "reason": endpoint.reason,
    }


def _serialized_finding(
    finding: SplitLineageFinding,
) -> dict[str, object]:
    endpoints = [finding.first]
    if finding.second is not None:
        endpoints.append(finding.second)
    return {
        "kind": finding.kind,
        "certainty": finding.certainty,
        "model_id": finding.model_id,
        "root": finding.root,
        "overlap": _serialized_range(finding.overlap),
        "endpoints": [_serialized_endpoint(item) for item in endpoints],
    }


def _serialized_unresolved(
    item: UnresolvedLineage,
) -> dict[str, object]:
    return {
        "role": item.role,
        "read_kind": item.read_kind,
        "root": item.root,
        "model_id": item.model_id,
        "protocol_range": _serialized_range(item.protocol_range),
        "subset": item.subset,
        "source": item.source,
        "line": item.line,
        "expression": item.expression,
        "reason": item.reason,
    }


def _serialized_relation(relation: RangeRelation) -> dict[str, object]:
    return {
        "model_id": relation.model_id,
        "root": relation.root,
        "first": {
            "role": relation.first_role,
            "read_kind": relation.first_read_kind,
            "protocol_range": _serialized_range(relation.first_range),
        },
        "second": {
            "role": relation.second_role,
            "read_kind": relation.second_read_kind,
            "protocol_range": _serialized_range(relation.second_range),
        },
        "status": relation.status,
        "overlap": _serialized_range(relation.overlap),
        "certainty": relation.certainty,
    }


def _serialized_target_range(
    observation: RoleObservation,
) -> dict[str, object]:
    """Project one role-target observation without widening its range."""
    return {
        "root": observation.root,
        "model_id": observation.model_id,
        "protocol_range": _serialized_range(observation.protocol_range),
        "subset": observation.subset,
        "certainty": (
            "exact" if observation.subset is None else "support"
        ),
    }


def _is_relevant_target_unresolved(item: UnresolvedLineage) -> bool:
    return item.read_kind == "target" and item.role in {
        FITTING, MODEL_SELECTION, REPORTED_EVALUATION,
    }


def _is_fitting_to_reported_evaluation(
    relation: RangeRelation,
) -> bool:
    return frozenset((
        (relation.first_role, relation.first_read_kind),
        (relation.second_role, relation.second_read_kind),
    )) == frozenset((
        (FITTING, "target"),
        (REPORTED_EVALUATION, "target"),
    ))


def _reported_range(relation: RangeRelation) -> ProtocolRange:
    if relation.first_role == REPORTED_EVALUATION:
        return relation.first_range
    return relation.second_range


def _derive_split_validity_receipt(
    report: SplitLineageReport,
    policy_findings: Sequence[SplitLineageFinding],
    *,
    applicable: bool = True,
) -> dict[str, object]:
    """Project one trusted analysis into a JSON-serializable receipt."""
    policy_findings = tuple(policy_findings)
    overlap_findings = tuple(
        item for item in policy_findings
        if item.kind in {"exact_overlap", "support_overlap"}
    )
    unresolved_findings = tuple(
        item for item in policy_findings if item.kind == "unresolved"
    )
    relevant_unresolved = tuple(
        item for item in report.unresolved
        if _is_relevant_target_unresolved(item)
    )
    fitting_evaluation_relations = tuple(
        item for item in report.relations
        if _is_fitting_to_reported_evaluation(item)
    )
    disjoint_proofs = tuple(
        item for item in fitting_evaluation_relations
        if item.status == "disjoint"
    )
    reported_targets = tuple(
        item for item in report.observations
        if item.role == REPORTED_EVALUATION and item.read_kind == "target"
    )
    fitting_target_ranges = tuple(
        item for item in report.observations
        if item.role == FITTING and item.read_kind == "target"
    )
    selection_target_ranges = tuple(
        item for item in report.observations
        if item.role == MODEL_SELECTION and item.read_kind == "target"
    )
    every_reported_target_proved = bool(reported_targets) and all(
        any(
            proof.model_id == target.model_id
            and proof.root == target.root
            and _reported_range(proof) == target.protocol_range
            for proof in disjoint_proofs
        )
        for target in reported_targets
    )

    if not applicable:
        status = "not_applicable"
        reasons = ["non_temporal_paradigm"]
    elif overlap_findings:
        status = "invalid"
        reasons = []
        if any(item.kind == "exact_overlap" for item in overlap_findings):
            reasons.append("exact_temporal_target_overlap")
        if any(item.kind == "support_overlap" for item in overlap_findings):
            reasons.append("support_temporal_target_overlap")
    else:
        reasons = []
        if unresolved_findings or relevant_unresolved:
            reasons.append("relevant_temporal_target_lineage_unresolved")
        if not every_reported_target_proved:
            reasons.append(
                "fitting_to_reported_evaluation_disjointness_unproved"
            )
        if reasons:
            status = "unresolved"
        else:
            status = "valid"
            reasons = [
                "fitting_to_reported_evaluation_disjointness_proved"
            ]

    return {
        "schema_version": "1.0.0",
        "status": status,
        "validator": "eval_split_lineage",
        "reasons": reasons,
        "evidence": {
            "findings": [
                _serialized_finding(item) for item in policy_findings
            ],
            "relevant_unresolved": [
                _serialized_unresolved(item) for item in relevant_unresolved
            ],
            "fitting_to_reported_evaluation_relations": [
                _serialized_relation(item)
                for item in fitting_evaluation_relations
            ],
            "fitting_target_ranges": [
                _serialized_target_range(item)
                for item in fitting_target_ranges
            ],
            "selection_target_ranges": [
                _serialized_target_range(item)
                for item in selection_target_ranges
            ],
        },
    }


def derive_split_validity_receipt(
    report: SplitLineageReport,
    *,
    applicable: bool = True,
) -> dict[str, object]:
    """Derive a split receipt without accepting caller-supplied findings.

    ``valid`` requires positive, concrete proof that every reported-evaluation
    target is disjoint from a fitting target for the same root and scored
    model.  It is never inferred merely from an empty enforcement-error list.
    """
    return _derive_split_validity_receipt(
        report,
        find_split_lineage_findings(report),
        applicable=applicable,
    )


def _paradigm_id(spec: Mapping[str, object]) -> str:
    comparison = spec.get("comparison")
    if not isinstance(comparison, Mapping):
        return ""
    classification = comparison.get("classification")
    if not isinstance(classification, Mapping):
        return ""
    value = classification.get("id")
    return value if isinstance(value, str) else ""


def temporal_lineage_enabled(spec: Mapping[str, object]) -> bool:
    return _paradigm_id(spec) in _TEMPORAL_PARADIGMS


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _positive_number(value: object) -> Union[int, float, None]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if value > 0 else None


def _parameter_values(
    spec: Mapping[str, object],
    build_plan: Mapping[str, object] | None,
    run_dir: Path,
) -> dict[str, Union[int, float]]:
    values: dict[str, Union[int, float]] = {}
    if isinstance(build_plan, Mapping):
        derivation = build_plan.get("params_derivation")
        if isinstance(derivation, Mapping):
            for name, declaration in derivation.items():
                if not isinstance(name, str) or not isinstance(
                        declaration, Mapping):
                    continue
                value = _positive_number(declaration.get("demo_value"))
                if value is not None:
                    values[name] = value

    raw_params = _read_json(run_dir / ".pipeline" / "params.json")
    params = raw_params.get("params", raw_params)
    if isinstance(params, Mapping):
        for name, declaration in params.items():
            if not isinstance(name, str):
                continue
            raw = (declaration.get("value")
                   if isinstance(declaration, Mapping) else declaration)
            value = _positive_number(raw)
            if value is not None:
                values[name] = value
    return values


def _protocol_extents(run_dir: Path) -> tuple[int | None, int | None]:
    provenance = _read_json(
        run_dir / "method" / "example_data" / "PROVENANCE.json"
    )
    live: list[int] = []
    nominal: list[int] = []
    for entry in provenance.get("files") or []:
        if not isinstance(entry, Mapping):
            continue
        axis = entry.get("time_axis")
        if not isinstance(axis, Mapping):
            continue
        if isinstance(axis.get("live_steps"), int) \
                and not isinstance(axis.get("live_steps"), bool) \
                and axis["live_steps"] > 0:
            live.append(axis["live_steps"])
        if isinstance(axis.get("steps_kept"), int) \
                and not isinstance(axis.get("steps_kept"), bool) \
                and axis["steps_kept"] > 0:
            nominal.append(axis["steps_kept"])
    if live:
        return max(live), max(nominal) if nominal else max(live)
    if nominal:
        value = max(nominal)
        return value, value
    return None, None


def _arch_contract(run_dir: Path) -> dict:
    return _read_json(run_dir / ".pipeline" / "arch_contract.json")


def _shape_has_time_axis(shape: object) -> bool:
    if not isinstance(shape, str):
        return False
    return bool(set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", shape))
                & _TIME_AXIS_SYMBOLS)


def _typed_descriptor_has_time_axis(descriptor: object) -> bool:
    """Return whether one supported v2 array carries the canonical time axis.

    Semantic identity is the authority within this contract adapter. Display
    symbols, descriptor-map names, opaque descriptions, and unrelated derived
    dimensions cannot opt in merely through a familiar glyph or word. The
    independent source-argument token fallback below remains unchanged.
    """
    if not isinstance(descriptor, Mapping) \
            or descriptor.get("kind") not in {"tensor", "ndarray"}:
        return False
    dimensions = descriptor.get("dimensions")
    if not isinstance(dimensions, list):
        return False
    return any(
        isinstance(dimension, Mapping)
        and dimension.get("dimension") == "time_axis_steps"
        for dimension in dimensions
    )


def _training_temporal_input_names(contract: Mapping[str, object]) -> set[str]:
    """Read temporal training inputs without normalizing contract versions."""
    training_loop = contract.get("training_loop")
    if not isinstance(training_loop, Mapping):
        return set()
    if contract.get("schema_version") == "2.0.0":
        typed_inputs = training_loop.get("input")
        if not isinstance(typed_inputs, Mapping):
            return set()
        return {
            name for name, descriptor in typed_inputs.items()
            if isinstance(name, str)
            and _typed_descriptor_has_time_axis(descriptor)
        }
    input_shapes = training_loop.get("input_shapes")
    if not isinstance(input_shapes, Mapping):
        return set()
    return {
        name for name, shape in input_shapes.items()
        if isinstance(name, str) and _shape_has_time_axis(shape)
    }


def _top_level_functions(source: Union[str, ast.Module]) -> dict[str, ast.FunctionDef]:
    tree = source if isinstance(source, ast.Module) else ast.parse(source)
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _function_body_nodes(function: ast.FunctionDef) -> tuple[ast.AST, ...]:
    """Walk one function body without entering nested lexical scopes."""
    nodes: list[ast.AST] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, (
                ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                ast.Lambda)):
            return
        nodes.append(node)
        for child in ast.iter_child_nodes(node):
            visit(child)

    for statement in function.body:
        visit(statement)
    return tuple(nodes)


def _callable_formal_aliases(
    function: ast.FunctionDef,
    nodes: Sequence[ast.AST],
) -> set[str]:
    aliases = {
        arg.arg
        for arg in (
            list(function.args.posonlyargs)
            + list(function.args.args)
            + list(function.args.kwonlyargs)
        )
    }
    changed = True
    while changed:
        changed = False
        for node in nodes:
            if isinstance(node, ast.Assign):
                value = node.value
                targets = node.targets
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                value = node.value
                targets = [node.target]
            else:
                continue
            is_alias = isinstance(value, ast.Name) and value.id in aliases
            is_transparent_model_call = (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and isinstance(value.func.value, ast.Name)
                and value.func.value.id in aliases
                and value.func.attr in {
                    "to", "cpu", "cuda", "train", "eval",
                }
            )
            is_trainer_model_call = (
                isinstance(value, ast.Call)
                and _tokens(
                    getattr(value.func, "id", "")
                    or getattr(value.func, "attr", "")
                ) & _TRAINER_TOKENS
                and any(
                    isinstance(argument, ast.Name)
                    and argument.id in aliases
                    for argument in (
                        list(value.args)
                        + [item.value for item in value.keywords]
                    )
                )
            )
            if not is_alias \
                    and not is_transparent_model_call \
                    and not is_trainer_model_call:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in aliases:
                    aliases.add(target.id)
                    changed = True
    return aliases


def _has_temporal_consumer(name: str, function: ast.FunctionDef) -> bool:
    tokens = _tokens(name)
    if tokens & (_TRAINER_TOKENS | _EVALUATION_FUNCTION_TOKENS):
        return True
    nodes = _function_body_nodes(function)
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) \
                and node.func.attr == "backward":
            return True
        call_tokens = _tokens(
            getattr(node.func, "id", "")
            or getattr(node.func, "attr", "")
        )
        is_mode_switch = (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"eval", "train"}
            and not node.args
            and not node.keywords
        )
        if call_tokens & (
            _METRIC_CONSUMER_TOKENS
            | {"loss", "criterion", "objective"}
        ) and not is_mode_switch:
            return True
    callable_formals = _callable_formal_aliases(function, nodes)
    return any(
        isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id in callable_formals
            or isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in callable_formals
            and node.func.attr in {"forward", "loss", "predict"}
        )
        for node in nodes
    )


def _function_has_metric_sink(
    function: ast.FunctionDef,
    *,
    annotations_deferred: bool = False,
) -> bool:
    nodes = _function_body_nodes(function)
    callable_aliases = _callable_formal_aliases(function, nodes)
    positional = list(function.args.posonlyargs) + list(function.args.args)
    parameters = positional + list(function.args.kwonlyargs)
    formal_names = {item.arg for item in parameters}
    local_names = formal_names | {
        node.id for node in nodes
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, (ast.Store, ast.Del))
    } | {
        node.name for node in nodes
        if isinstance(node, ast.ExceptHandler)
        and node.name is not None
    }
    for node in nodes:
        if isinstance(node, ast.Import):
            local_names.update(
                alias.asname or alias.name.split(".", 1)[0]
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            local_names.update(
                alias.asname or alias.name
                for alias in node.names if alias.name != "*"
            )
    callable_formals = {
        name for name in formal_names if _tokens(name) & _MODEL_TOKENS
    }
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in formal_names:
            callable_formals.add(node.func.id)
        elif isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) \
                and node.func.value.id in formal_names:
            callable_formals.add(node.func.value.id)

    def scalar_annotation(argument: ast.arg) -> bool:
        if argument.annotation is None:
            return False
        return bool(
            _tokens(ast.unparse(argument.annotation))
            & _SCALAR_ANNOTATION_TOKENS
        )

    scalar_formals = {
        item.arg for item in parameters if scalar_annotation(item)
    }
    positional_defaults = list(function.args.defaults)
    default_start = len(positional) - len(positional_defaults)
    for index, default in enumerate(positional_defaults, start=default_start):
        if isinstance(default, ast.Constant) and isinstance(
                default.value, (bool, int, float, str)):
            scalar_formals.add(positional[index].arg)
    for argument, default in zip(
            function.args.kwonlyargs, function.args.kw_defaults):
        if isinstance(default, ast.Constant) and isinstance(
                default.value, (bool, int, float, str)):
            scalar_formals.add(argument.arg)
    data_formals = formal_names - scalar_formals - callable_formals

    DefinitionEnv = dict[str, tuple[ast.AST, ...]]
    class_execution_depth = 0
    class_expressions: list[tuple[ast.AST, DefinitionEnv]] = []

    def is_unbound_definition(values: tuple[ast.AST, ...]) -> bool:
        return any(
            isinstance(value, ast.Name)
            and value.id == _UNBOUND_LOCAL_DEFINITION
            for value in values
        )

    def expression_reads_unbound_local(
        expression: ast.AST, env: DefinitionEnv,
    ) -> bool:
        def name_is_unbound(name: str) -> bool:
            if name in env:
                return is_unbound_definition(env[name])
            return name not in formal_names

        return any(
            isinstance(item, ast.Name)
            and isinstance(item.ctx, ast.Load)
            and item.id in local_names
            and name_is_unbound(item.id)
            for item in reachable_expression_nodes(expression)
        )

    def mark_unbound(name: str, env: DefinitionEnv) -> None:
        env[name] = (
            ast.Name(id=_UNBOUND_LOCAL_DEFINITION, ctx=ast.Load()),
        )

    def materialize_name(
        node: ast.AST | _ImplicitExceptionState | None,
        name: str,
        env: DefinitionEnv,
    ) -> ast.AST | _ImplicitExceptionState | None:
        if not isinstance(node, ast.AST) or name not in env:
            return node
        definitions = env[name]
        if not definitions:
            return node
        replacement = definitions[0] if len(definitions) == 1 \
            else ast.Tuple(
                elts=[copy.deepcopy(item) for item in definitions],
                ctx=ast.Load(),
            )

        class Transformer(ast.NodeTransformer):
            def visit_Name(self, item: ast.Name) -> ast.AST:
                if isinstance(item.ctx, ast.Load) and item.id == name:
                    return copy.deepcopy(replacement)
                return item

        return Transformer().visit(copy.deepcopy(node))

    def bind_target(
        target: ast.AST, value: ast.AST, env: DefinitionEnv,
    ) -> None:
        if isinstance(target, ast.Name):
            env[target.id] = (value,)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                bind_target(item, value, env)

    def apply_named_expressions(
        expression: ast.AST | None, env: DefinitionEnv,
    ) -> None:
        if expression is None:
            return
        if isinstance(expression, ast.NamedExpr):
            apply_named_expressions(expression.value, env)
            bind_target(expression.target, expression.value, env)
            return
        for child in reachable_expression_children(expression):
            apply_named_expressions(child, env)

    def bind_pattern(
        pattern: ast.pattern, value: ast.AST, env: DefinitionEnv,
    ) -> None:
        if isinstance(pattern, ast.MatchAs):
            if pattern.name is not None:
                env[pattern.name] = (value,)
            if pattern.pattern is not None:
                bind_pattern(pattern.pattern, value, env)
        elif isinstance(pattern, ast.MatchStar) and pattern.name is not None:
            env[pattern.name] = (value,)
        elif isinstance(pattern, (ast.MatchSequence, ast.MatchOr)):
            for item in pattern.patterns:
                bind_pattern(item, value, env)
        elif isinstance(pattern, ast.MatchMapping):
            for item in pattern.patterns:
                bind_pattern(item, value, env)
            if pattern.rest is not None:
                env[pattern.rest] = (value,)
        elif isinstance(pattern, ast.MatchClass):
            for item in pattern.patterns + pattern.kwd_patterns:
                bind_pattern(item, value, env)

    def static_condition(
        node: ast.AST, env: DefinitionEnv | None = None,
    ) -> bool | None:
        if isinstance(node, ast.Name) and env is not None \
                and node.id in env and len(env[node.id]) == 1:
            value = env[node.id][0]
            if not (
                isinstance(value, ast.Name) and value.id == node.id
            ):
                return static_condition(value, env)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (bool, int, float, str)):
                return bool(node.value)
            return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            value = static_condition(node.operand, env)
            return None if value is None else not value
        if isinstance(node, ast.Compare) and node.ops \
                and len(node.ops) == len(node.comparators):
            previous = node.left
            unresolved = False
            for comparison, comparator in zip(
                    node.ops, node.comparators):
                try:
                    left = ast.literal_eval(previous)
                    right = ast.literal_eval(comparator)
                    if isinstance(comparison, ast.Lt):
                        result = left < right
                    elif isinstance(comparison, ast.LtE):
                        result = left <= right
                    elif isinstance(comparison, ast.Gt):
                        result = left > right
                    elif isinstance(comparison, ast.GtE):
                        result = left >= right
                    elif isinstance(comparison, ast.Eq):
                        result = left == right
                    elif isinstance(comparison, ast.NotEq):
                        result = left != right
                    else:
                        result = None
                except (TypeError, ValueError):
                    result = None
                if result is False:
                    return False
                if result is None:
                    unresolved = True
                previous = comparator
            return None if unresolved else True
        return None

    def reachable_expression_children(node: ast.AST) -> tuple[ast.AST, ...]:
        """Return expression children that Python can evaluate."""
        if isinstance(node, ast.IfExp):
            condition = static_condition(node.test)
            branches = (
                (node.body,) if condition is True
                else (node.orelse,) if condition is False
                else (node.body, node.orelse)
            )
            return (node.test,) + branches
        if isinstance(node, ast.BoolOp):
            values: list[ast.AST] = []
            for value in node.values:
                values.append(value)
                condition = static_condition(value)
                if isinstance(node.op, ast.And) and condition is False \
                        or isinstance(node.op, ast.Or) and condition is True:
                    break
            return tuple(values)
        if isinstance(node, ast.Compare):
            values = [node.left]
            previous = node.left
            for comparison, comparator in zip(
                    node.ops, node.comparators):
                values.append(comparator)
                pair = ast.Compare(
                    left=previous,
                    ops=[comparison],
                    comparators=[comparator],
                )
                if static_condition(pair) is False:
                    break
                previous = comparator
            return tuple(values)
        if isinstance(node, ast.Lambda):
            return tuple(node.args.defaults) + tuple(
                value for value in node.args.kw_defaults
                if value is not None
            )
        if isinstance(node, ast.GeneratorExp):
            return (node.generators[0].iter,) if node.generators else ()
        return tuple(ast.iter_child_nodes(node))

    def reachable_expression_nodes(node: ast.AST) -> Iterable[ast.AST]:
        yield node
        for child in reachable_expression_children(node):
            yield from reachable_expression_nodes(child)

    StaticState = tuple[
        str, DefinitionEnv, ast.AST | _ImplicitExceptionState | None,
    ]

    def dedupe_states(states: Sequence[StaticState]) -> list[StaticState]:
        unique: dict[tuple[object, ...], StaticState] = {}
        for kind, env, value in states:
            env_key = tuple(sorted(
                (name, tuple(ast.dump(item, include_attributes=False)
                             for item in values))
                for name, values in env.items()
            ))
            if isinstance(value, ast.AST):
                value_key: object = ast.dump(
                    value, include_attributes=False
                )
            elif isinstance(value, _ImplicitExceptionState):
                value_key = tuple(
                    item.__name__ for item in value.excluded_classes
                )
            else:
                value_key = None
            unique.setdefault((kind, env_key, value_key), (kind, env, value))
        return list(unique.values())

    def transition_target(
        target: ast.AST,
        value: ast.AST,
        env: DefinitionEnv,
        *,
        delete: bool = False,
    ) -> list[StaticState]:
        live = dict(env)
        if isinstance(target, ast.Name):
            if delete:
                values = live.get(target.id)
                mark_unbound(target.id, live)
                if values is None or is_unbound_definition(values):
                    return [(
                        "raise", live,
                        ast.Name(id="UnboundLocalError", ctx=ast.Load()),
                    )]
            else:
                live[target.id] = (value,)
            return [("fallthrough", live, None)]
        if isinstance(target, ast.Starred):
            return transition_target(
                target.value, value, live, delete=delete
            )
        if isinstance(target, (ast.Tuple, ast.List)):
            literal_values = (
                tuple(value.elts)
                if isinstance(value, (ast.Tuple, ast.List)) else None
            )
            range_values: range | None = None
            if isinstance(value, ast.Call) \
                    and isinstance(value.func, ast.Name) \
                    and value.func.id == "range" \
                    and not value.keywords:
                try:
                    arguments = [
                        ast.literal_eval(item) for item in value.args
                    ]
                    if all(isinstance(item, int) for item in arguments):
                        range_values = range(*arguments)
                except (TypeError, ValueError):
                    pass
            value_length = (
                len(literal_values)
                if literal_values is not None
                else len(range_values)
                if range_values is not None
                else None
            )
            unpack_status = (
                _unpack_target_accepts_length(target, value_length)
                if not delete and value_length is not None else None
            )
            scalar_non_iterable = (
                not delete
                and isinstance(value, ast.Constant)
                and isinstance(value.value, (bool, int, float, complex))
            )
            if scalar_non_iterable:
                return [(
                    "raise",
                    live,
                    ast.Name(id="TypeError", ctx=ast.Load()),
                )]
            states: list[StaticState] = []
            if delete or unpack_status is not False:
                states.append(("fallthrough", live, None))
            if not delete and unpack_status is None:
                states.append((
                    "implicit_raise", dict(live),
                    _ImplicitExceptionState(),
                ))
            elif not delete and unpack_status is False:
                states.append((
                    "raise",
                    dict(live),
                    ast.Name(id="ValueError", ctx=ast.Load()),
                ))
            starred = next(
                (
                    index for index, item in enumerate(target.elts)
                    if isinstance(item, ast.Starred)
                ),
                None,
            )
            for index, item in enumerate(target.elts):
                assigned = value
                if literal_values is not None and unpack_status is True:
                    if starred is None or index < starred:
                        assigned = literal_values[index]
                    elif index == starred:
                        tail_count = len(target.elts) - index - 1
                        stop = len(literal_values) - tail_count
                        assigned = ast.List(
                            elts=list(literal_values[index:stop]),
                            ctx=ast.Load(),
                        )
                    else:
                        source_index = (
                            len(literal_values) - len(target.elts) + index
                        )
                        assigned = literal_values[source_index]
                elif range_values is not None and unpack_status is True:
                    if starred is None or index < starred:
                        assigned = ast.Constant(value=range_values[index])
                    elif index == starred:
                        assigned = ast.List(elts=[], ctx=ast.Load())
                    else:
                        source_index = (
                            len(range_values) - len(target.elts) + index
                        )
                        assigned = ast.Constant(
                            value=range_values[source_index]
                        )
                elif not delete:
                    assigned = ast.Subscript(
                        value=copy.deepcopy(value),
                        slice=ast.Constant(value=index),
                        ctx=ast.Load(),
                    )
                advanced: list[StaticState] = []
                for kind, state_env, state_value in states:
                    if kind != "fallthrough":
                        advanced.append((kind, state_env, state_value))
                        continue
                    advanced.extend(transition_target(
                        item, assigned, state_env, delete=delete
                    ))
                states = dedupe_states(advanced)
            return states

        expressions = _target_evaluation_expressions(target)
        if any(
            expression_reads_unbound_local(expression, live)
            for expression in expressions
        ):
            return [(
                "raise", live,
                ast.Name(id="UnboundLocalError", ctx=ast.Load()),
            )]
        return [
            ("fallthrough", live, None),
            ("implicit_raise", dict(live), _ImplicitExceptionState()),
        ]

    def static_loop_nonempty(statement: ast.stmt) -> bool | None:
        if isinstance(statement, (ast.For, ast.AsyncFor)):
            iterator = statement.iter
            if isinstance(iterator, (ast.List, ast.Tuple, ast.Set)):
                return bool(iterator.elts)
            if isinstance(iterator, ast.Call) \
                    and getattr(iterator.func, "id", "") == "range":
                try:
                    values = [ast.literal_eval(item) for item in iterator.args]
                    if not all(isinstance(item, int) for item in values):
                        return None
                    return bool(range(*values))
                except (TypeError, ValueError):
                    return None
            return None
        if isinstance(statement, ast.While):
            return static_condition(statement.test)
        return None

    def execute_block(
        statements: Sequence[ast.stmt], env: DefinitionEnv,
    ) -> list[StaticState]:
        nonlocal class_execution_depth
        states: list[StaticState] = [("fallthrough", dict(env), None)]
        for statement in statements:
            next_states: list[StaticState] = []
            for kind, state_env, value in states:
                if kind != "fallthrough":
                    next_states.append((kind, state_env, value))
                    continue
                live = dict(state_env)
                simple_name_delete = isinstance(
                    statement, ast.Delete
                ) and all(
                    isinstance(target, ast.Name)
                    for target in statement.targets
                )
                expression: ast.AST | None = None
                if isinstance(statement, ast.Assign):
                    expression = statement.value
                elif isinstance(statement, ast.AnnAssign):
                    expression = statement.value
                elif isinstance(statement, ast.Expr):
                    expression = statement.value
                elif isinstance(statement, ast.Return):
                    expression = statement.value
                known_expression_failure = (
                    _known_eager_comprehension_failure(expression)
                )
                if class_execution_depth and expression is not None:
                    class_expressions.append((expression, dict(live)))
                if _statement_may_raise_implicitly(
                        statement, {
                            name for name, values in live.items()
                            if not is_unbound_definition(values)
                        } | {
                            name for name in formal_names
                            if name not in live
                            or not is_unbound_definition(live[name])
                        },
                        annotations_deferred=annotations_deferred,
                ) and not simple_name_delete \
                        and known_expression_failure is None:
                    next_states.append((
                        "implicit_raise", dict(live),
                        _ImplicitExceptionState(),
                    ))
                if known_expression_failure is not None:
                    next_states.append((
                        "raise",
                        live,
                        ast.Name(
                            id=known_expression_failure,
                            ctx=ast.Load(),
                        ),
                    ))
                    continue
                evaluated_expressions: tuple[ast.AST, ...] = ()
                if expression is not None:
                    evaluated_expressions = (expression,)
                elif isinstance(statement, (ast.If, ast.While)):
                    evaluated_expressions = (statement.test,)
                elif isinstance(statement, (ast.For, ast.AsyncFor)):
                    evaluated_expressions = (statement.iter,)
                elif isinstance(statement, (ast.FunctionDef,
                                              ast.AsyncFunctionDef,
                                              ast.ClassDef)):
                    evaluated_expressions = (
                        _definition_evaluated_expressions(
                            statement,
                            annotations_deferred=annotations_deferred,
                        )
                    )
                for evaluated_expression in evaluated_expressions:
                    apply_named_expressions(evaluated_expression, live)
                if isinstance(statement, ast.Assign):
                    assignment_states: list[StaticState] = [
                        ("fallthrough", live, None)
                    ]
                    for target in statement.targets:
                        advanced: list[StaticState] = []
                        for (
                            assignment_kind,
                            assignment_env,
                            assignment_value,
                        ) in assignment_states:
                            if assignment_kind != "fallthrough":
                                advanced.append((
                                    assignment_kind,
                                    assignment_env,
                                    assignment_value,
                                ))
                                continue
                            advanced.extend(transition_target(
                                target, statement.value, assignment_env
                            ))
                        assignment_states = dedupe_states(advanced)
                    next_states.extend(assignment_states)
                elif isinstance(statement, ast.AnnAssign) \
                        and statement.value is not None:
                    next_states.extend(transition_target(
                        statement.target, statement.value, live
                    ))
                elif isinstance(statement, ast.AugAssign):
                    bind_target(statement.target, statement.value, live)
                    next_states.append(("fallthrough", live, None))
                elif isinstance(statement, ast.Return):
                    next_states.append(("return", live, statement.value))
                elif isinstance(statement, ast.Raise):
                    next_states.append(("raise", live, statement.exc))
                elif isinstance(statement, ast.Break):
                    next_states.append(("break", live, None))
                elif isinstance(statement, ast.Continue):
                    next_states.append(("continue", live, None))
                elif isinstance(statement, ast.Delete):
                    delete_states: list[StaticState] = [
                        ("fallthrough", live, None)
                    ]
                    for target in statement.targets:
                        advanced = []
                        for delete_kind, delete_env, delete_value \
                                in delete_states:
                            if delete_kind != "fallthrough":
                                advanced.append((
                                    delete_kind, delete_env, delete_value,
                                ))
                                continue
                            advanced.extend(transition_target(
                                target,
                                ast.Constant(value=None),
                                delete_env,
                                delete=True,
                            ))
                        delete_states = dedupe_states(advanced)
                    next_states.extend(delete_states)
                elif isinstance(statement, ast.Assert):
                    condition = static_condition(statement.test)
                    if condition is not True:
                        next_states.append((
                            "raise", live,
                            ast.Name(id="AssertionError", ctx=ast.Load()),
                        ))
                    if condition is not False:
                        next_states.append(("fallthrough", live, None))
                elif isinstance(statement, ast.If):
                    condition = static_condition(statement.test, live)
                    branches = (
                        (statement.body,) if condition is True
                        else (statement.orelse,) if condition is False
                        else (statement.body, statement.orelse)
                    )
                    for branch in branches:
                        next_states.extend(execute_block(branch, live))
                elif isinstance(statement, ast.Match):
                    irrefutable = False
                    for case in statement.cases:
                        case_env = dict(live)
                        bind_pattern(
                            case.pattern, statement.subject, case_env
                        )
                        if case.guard is not None \
                                and static_condition(
                                    case.guard, case_env
                                ) is False:
                            continue
                        next_states.extend(execute_block(
                            case.body, case_env
                        ))
                        if isinstance(case.pattern, ast.MatchAs) \
                                and case.pattern.pattern is None \
                                and case.guard is None:
                            irrefutable = True
                            break
                    if not irrefutable:
                        next_states.append(("fallthrough", live, None))
                elif isinstance(statement, (ast.With, ast.AsyncWith)):
                    with_states: list[StaticState] = [
                        ("fallthrough", live, None)
                    ]
                    for item in statement.items:
                        if item.optional_vars is None:
                            continue
                        advanced: list[StaticState] = []
                        for with_kind, with_env, with_value in with_states:
                            if with_kind != "fallthrough":
                                advanced.append((
                                    with_kind, with_env, with_value,
                                ))
                                continue
                            advanced.extend(transition_target(
                                item.optional_vars,
                                item.context_expr,
                                with_env,
                            ))
                        with_states = dedupe_states(advanced)
                    for with_kind, with_env, with_value in with_states:
                        if with_kind == "fallthrough":
                            next_states.extend(execute_block(
                                statement.body, with_env
                            ))
                        else:
                            next_states.append((
                                with_kind, with_env, with_value,
                            ))
                elif isinstance(statement, ast.For) and isinstance(
                        statement.iter, (ast.List, ast.Tuple)):
                    active: list[DefinitionEnv] = [dict(live)]
                    loop_states: list[StaticState] = []
                    broken: list[DefinitionEnv] = []
                    for candidate in statement.iter.elts:
                        next_active: list[DefinitionEnv] = []
                        for iteration_env in active:
                            for target_kind, target_env, target_value \
                                    in transition_target(
                                        statement.target,
                                        candidate,
                                        iteration_env,
                                    ):
                                if target_kind != "fallthrough":
                                    loop_states.append((
                                        target_kind,
                                        target_env,
                                        target_value,
                                    ))
                                    continue
                                for body_kind, body_env, body_value \
                                        in execute_block(
                                            statement.body, target_env
                                        ):
                                    if body_kind in {
                                            "return", "raise",
                                            "implicit_raise",
                                    }:
                                        loop_states.append((
                                            body_kind,
                                            body_env,
                                            body_value,
                                        ))
                                    elif body_kind == "break":
                                        broken.append(body_env)
                                    else:
                                        next_active.append(body_env)
                        active = [
                            state_env for kind, state_env, _
                            in dedupe_states([
                                ("fallthrough", item, None)
                                for item in next_active
                            ])
                            if kind == "fallthrough"
                        ]
                        if not active:
                            break
                    loop_states.extend(
                        ("fallthrough", item, None)
                        for item in broken
                    )
                    for exhausted_env in active:
                        loop_states.extend(execute_block(
                            statement.orelse, exhausted_env
                        ))
                    next_states.extend(dedupe_states(loop_states))
                elif isinstance(statement, (
                        ast.For, ast.AsyncFor, ast.While)):
                    nonempty = static_loop_nonempty(statement)
                    normal_envs: list[DefinitionEnv] = []
                    loop_states: list[StaticState] = []
                    if nonempty is not True:
                        normal_envs.append(dict(live))
                    if nonempty is not False:
                        body_env = dict(live)
                        target_states: list[StaticState] = [
                            ("fallthrough", body_env, None)
                        ]
                        if isinstance(statement, (ast.For, ast.AsyncFor)):
                            candidate = statement.iter
                            if isinstance(
                                statement.target, (ast.Tuple, ast.List)
                            ) and isinstance(statement.iter, ast.Call) \
                                    and isinstance(
                                        statement.iter.func, ast.Name
                                    ) \
                                    and statement.iter.func.id == "range":
                                candidate = ast.Constant(value=0)
                            candidates = (candidate,)
                            target_states = dedupe_states([
                                target_state
                                for candidate in candidates
                                for target_state in transition_target(
                                    statement.target,
                                    candidate,
                                    body_env,
                                )
                            ])
                        body_states: list[StaticState] = []
                        for target_kind, target_env, target_value \
                                in target_states:
                            if target_kind != "fallthrough":
                                loop_states.append((
                                    target_kind, target_env, target_value,
                                ))
                                continue
                            body_states.extend(execute_block(
                                statement.body, target_env
                            ))
                        for body_kind, body_state_env, body_value in body_states:
                            if body_kind in {
                                    "return", "raise", "implicit_raise"}:
                                loop_states.append((
                                    body_kind, body_state_env, body_value,
                                ))
                            elif body_kind == "break":
                                loop_states.append((
                                    "fallthrough", body_state_env, None,
                                ))
                            elif not (
                                isinstance(statement, ast.While)
                                and static_condition(
                                    statement.test, body_state_env
                                ) is True
                            ):
                                normal_envs.append(body_state_env)
                    for normal_env in normal_envs:
                        loop_states.extend(execute_block(
                            statement.orelse, normal_env
                        ))
                    next_states.extend(loop_states)
                elif isinstance(statement, (
                        ast.FunctionDef, ast.AsyncFunctionDef)):
                    if any(
                        expression_reads_unbound_local(expression, live)
                        for expression
                        in _definition_evaluated_expressions(
                            statement,
                            annotations_deferred=annotations_deferred,
                        )
                    ):
                        next_states.append((
                            "raise",
                            live,
                            ast.Name(
                                id="UnboundLocalError", ctx=ast.Load()
                            ),
                        ))
                        continue
                    live[statement.name] = (ast.Constant(value=None),)
                    next_states.append(("fallthrough", live, None))
                elif isinstance(statement, ast.ClassDef):
                    if any(
                        expression_reads_unbound_local(expression, live)
                        for expression
                        in _definition_evaluated_expressions(
                            statement,
                            annotations_deferred=annotations_deferred,
                        )
                    ):
                        next_states.append((
                            "raise",
                            live,
                            ast.Name(
                                id="UnboundLocalError", ctx=ast.Load()
                            ),
                        ))
                        continue
                    class_execution_depth += 1
                    try:
                        class_states = execute_block(
                            statement.body, dict(live)
                        )
                    finally:
                        class_execution_depth -= 1
                    nonlocal_names = _scope_nonlocal_names(statement.body)

                    def project_class_env(
                        class_env: DefinitionEnv,
                    ) -> DefinitionEnv:
                        outer_env = dict(live)
                        for name in nonlocal_names:
                            if name in class_env:
                                outer_env[name] = class_env[name]
                        return outer_env

                    for class_kind, class_env, class_value \
                            in class_states:
                        if class_kind != "fallthrough":
                            next_states.append((
                                class_kind,
                                project_class_env(class_env),
                                class_value,
                            ))
                            continue
                        outer_env = project_class_env(class_env)
                        class_definitions = tuple(
                            definition
                            for name, definitions in class_env.items()
                            if name not in nonlocal_names
                            and (
                                name not in live
                                or live[name] != definitions
                            )
                            for definition in definitions
                        )
                        outer_env[statement.name] = class_definitions or (
                            ast.Constant(value=None),
                        )
                        next_states.append((
                            "fallthrough", outer_env, None,
                        ))
                elif isinstance(statement, ast.Import):
                    for alias in statement.names:
                        live[
                            alias.asname or alias.name.split(".", 1)[0]
                        ] = (ast.Constant(value=None),)
                    next_states.append(("fallthrough", live, None))
                elif isinstance(statement, ast.ImportFrom):
                    for alias in statement.names:
                        if alias.name != "*":
                            live[alias.asname or alias.name] = (
                                ast.Constant(value=None),
                            )
                    next_states.append(("fallthrough", live, None))
                elif isinstance(statement, ast.Try):
                    body_states: list[StaticState] = [
                        ("fallthrough", dict(live), None)
                    ]
                    for inner in statement.body:
                        advanced: list[StaticState] = []
                        for body_kind, body_env, body_value in body_states:
                            if body_kind != "fallthrough":
                                advanced.append((
                                    body_kind, body_env, body_value,
                                ))
                                continue
                            outputs = execute_block((inner,), body_env)
                            advanced.extend(outputs)
                        body_states = dedupe_states(advanced)
                    prior_states: list[StaticState] = []
                    raised_states: list[StaticState] = []
                    implicit_raised_states: list[StaticState] = []
                    for body_kind, body_env, body_value in body_states:
                        if body_kind == "fallthrough":
                            prior_states.extend(execute_block(
                                statement.orelse, body_env
                            ))
                        elif body_kind == "raise":
                            raised_states.append((
                                body_kind, body_env, body_value,
                            ))
                        elif body_kind == "implicit_raise":
                            implicit_raised_states.append((
                                body_kind, body_env, body_value,
                            ))
                        else:
                            prior_states.append((
                                body_kind, body_env, body_value,
                            ))
                    if statement.handlers:
                        implicit_coverage: list[type[BaseException]] = []
                        for handler in statement.handlers:
                            unmatched: list[StaticState] = []
                            handler_entries: list[DefinitionEnv] = []
                            for raised_state in raised_states:
                                exception_name = _exception_type_name(
                                    raised_state[2]
                                )
                                if _handler_matches(handler, exception_name):
                                    handler_entries.append(raised_state[1])
                                else:
                                    unmatched.append(raised_state)
                            raised_states = unmatched
                            for implicit_state in implicit_raised_states:
                                coverage = (
                                    implicit_state[2].excluded_classes
                                    if isinstance(
                                        implicit_state[2],
                                        _ImplicitExceptionState,
                                    ) else ()
                                )
                                if not _implicit_handler_fully_covered(
                                        handler,
                                        coverage
                                        + tuple(implicit_coverage)):
                                    handler_entries.append(
                                        implicit_state[1]
                                    )
                            handler_classes = (
                                _handler_builtin_exception_classes(handler)
                            )
                            if handler_classes is None:
                                implicit_coverage = [BaseException]
                            else:
                                implicit_coverage.extend(handler_classes)
                            for handler_entry in handler_entries:
                                bound_entry = dict(handler_entry)
                                if handler.name is not None:
                                    # Keep the shadowed exception target
                                    # explicitly non-temporal.  That also
                                    # models Python deleting any older binding
                                    # when the handler exits.
                                    bound_entry[handler.name] = (
                                        ast.Constant(value=None),
                                    )
                                handler_states = execute_block(
                                    handler.body, bound_entry
                                )
                                if handler.name is not None:
                                    cleaned_states: list[StaticState] = []
                                    for (
                                        handler_kind,
                                        handler_env,
                                        handler_value,
                                    ) in handler_states:
                                        cleaned_env = dict(handler_env)
                                        cleaned_value = materialize_name(
                                            handler_value,
                                            handler.name,
                                            handler_env,
                                        )
                                        mark_unbound(
                                            handler.name, cleaned_env
                                        )
                                        cleaned_states.append((
                                            handler_kind,
                                            cleaned_env,
                                            cleaned_value,
                                        ))
                                    handler_states = cleaned_states
                                prior_states.extend(handler_states)
                        if not _handlers_catch_all_implicit(
                                statement.handlers):
                            for implicit_state in implicit_raised_states:
                                coverage = (
                                    implicit_state[2].excluded_classes
                                    if isinstance(
                                        implicit_state[2],
                                        _ImplicitExceptionState,
                                    ) else ()
                                )
                                prior_states.append((
                                    "implicit_raise",
                                    implicit_state[1],
                                    _ImplicitExceptionState(tuple(
                                        dict.fromkeys(
                                            coverage
                                            + tuple(implicit_coverage)
                                        )
                                    )),
                                ))
                    else:
                        prior_states.extend(implicit_raised_states)
                    prior_states.extend(raised_states)
                    if not statement.finalbody:
                        next_states.extend(prior_states)
                    else:
                        for prior_kind, prior_env, prior_value in prior_states:
                            final_states = execute_block(
                                statement.finalbody, prior_env
                            )
                            for final_kind, final_env, final_value in final_states:
                                if final_kind == "fallthrough":
                                    next_states.append((
                                        prior_kind, final_env, prior_value,
                                    ))
                                else:
                                    next_states.append((
                                        final_kind, final_env, final_value,
                                    ))
                else:
                    next_states.append(("fallthrough", live, None))
            states = dedupe_states(next_states)
        return states

    def expanded_expressions(
        node: ast.AST, env: DefinitionEnv,
    ) -> list[ast.AST]:
        expressions = [node]
        seen: set[str] = set()
        pending = [
            item.id for item in reachable_expression_nodes(node)
            if isinstance(item, ast.Name) and item.id in env
        ]
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            for expression in env[name]:
                expressions.append(expression)
                pending.extend(
                    item.id for item in reachable_expression_nodes(expression)
                    if isinstance(item, ast.Name)
                    and item.id in env
                    and item.id not in seen
                )
        return expressions

    def is_model_call(node: ast.AST) -> bool:
        return isinstance(node, ast.Call) and (
            isinstance(node.func, ast.Name)
            and node.func.id in callable_aliases
            or isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in callable_aliases
            and node.func.attr in {"forward", "predict", "__call__"}
        )

    def is_prediction_postprocess_call(node: ast.Call) -> bool:
        name = (
            getattr(node.func, "id", "")
            or getattr(node.func, "attr", "")
        )
        return bool(name[:1].isupper()) \
            or bool(_tokens(name) & _PREDICTION_POSTPROCESS_TOKENS)

    def depends_on_prediction(
        node: ast.AST,
        env: DefinitionEnv,
        seen: frozenset[str] = frozenset(),
    ) -> bool:
        if is_model_call(node):
            return True
        if isinstance(node, ast.Name) and node.id in env:
            if node.id in seen:
                return False
            return any(
                depends_on_prediction(
                    value, env, seen | {node.id},
                )
                for value in env[node.id]
            )
        return any(
            depends_on_prediction(child, env, seen)
            for child in reachable_expression_children(node)
        )

    def data_sources(
        node: ast.AST,
        env: DefinitionEnv,
        seen: frozenset[str] = frozenset(),
    ) -> set[str]:
        if isinstance(node, ast.Name):
            if node.id in env:
                if node.id in seen:
                    return set()
                return set().union(*(
                    data_sources(value, env, seen | {node.id})
                    for value in env[node.id]
                ))
            return {node.id} if node.id in data_formals else set()
        sources: set[str] = set()
        for child in reachable_expression_children(node):
            sources.update(data_sources(child, env, seen))
        return sources

    def independent_data_operand(node: ast.AST, env: DefinitionEnv) -> bool:
        return bool(data_sources(node, env)) \
            and not depends_on_prediction(node, env)

    def mixed_prediction_data_operation(
        node: ast.AST, env: DefinitionEnv,
    ) -> bool:
        operands: list[ast.AST]
        if isinstance(node, ast.BinOp):
            operands = [node.left, node.right]
        elif isinstance(node, ast.Compare):
            operands = [node.left] + list(node.comparators)
        else:
            return False
        return any(depends_on_prediction(item, env) for item in operands) \
            and any(independent_data_operand(item, env) for item in operands)

    returned = [
        (value, state_env)
        for kind, state_env, value in execute_block(function.body, {})
        if kind == "return" and value is not None
    ]
    for returned_node, return_env in returned + class_expressions:
        expressions = expanded_expressions(returned_node, return_env)
        calls = [
            item
            for expression in expressions
            for item in reachable_expression_nodes(expression)
            if isinstance(item, ast.Call)
            and not (
                isinstance(item.func, ast.Attribute)
                and item.func.attr in {"eval", "train"}
                and not item.args
                and not item.keywords
            )
        ]
        call_tokens = {
            token
            for call in calls
            for token in _tokens(
                getattr(call.func, "id", "")
                or getattr(call.func, "attr", "")
            )
        }
        if call_tokens & _METRIC_CONSUMER_TOKENS:
            return True
        if any(
            not is_model_call(call)
            and not is_prediction_postprocess_call(call)
            and not _tokens(
                getattr(call.func, "id", "")
                or getattr(call.func, "attr", "")
            ) & _NON_EVALUATION_CONSUMER_TOKENS
            and any(
                depends_on_prediction(argument, return_env)
                for argument in (
                    list(call.args)
                    + [item.value for item in call.keywords]
                )
            )
            and any(
                independent_data_operand(argument, return_env)
                for argument in (
                    list(call.args)
                    + [item.value for item in call.keywords]
                )
            )
            for call in calls
        ):
            return True
        if call_tokens & {"mean", "sum", "sqrt", "abs", "norm"} \
                and any(
                    mixed_prediction_data_operation(item, return_env)
                    for expression in expressions
                    for item in reachable_expression_nodes(expression)
                ):
            return True
    return False


def _is_forecast_only(
    name: str,
    function: ast.FunctionDef,
    *,
    annotations_deferred: bool = False,
) -> bool:
    tokens = _tokens(name)
    if not tokens & _FORECAST_FUNCTION_TOKENS \
            or tokens & _TRAINER_TOKENS \
            or tokens & _EVALUATION_FUNCTION_TOKENS:
        return False
    has_fitting_sink = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "backward"
        for node in _function_body_nodes(function)
    )
    return not has_fitting_sink and not _function_has_metric_sink(
        function,
        annotations_deferred=annotations_deferred,
    )


def _trainer_names(
    source: Union[str, ast.Module], preferred: Iterable[str] = (),
) -> tuple[str, ...]:
    tree = source if isinstance(source, ast.Module) else ast.parse(source)
    functions = _top_level_functions(tree)
    annotations_deferred = _future_annotations_enabled(tree)
    names = {
        name for name in preferred
        if isinstance(name, str)
        and name in functions
        and not _is_forecast_only(
            name,
            functions[name],
            annotations_deferred=annotations_deferred,
        )
    }
    names.update(
        name for name in functions
        if _has_temporal_consumer(name, functions[name])
        and not _tokens(name) & {"build", "create", "make", "construct"}
        and not _is_forecast_only(
            name,
            functions[name],
            annotations_deferred=annotations_deferred,
        )
    )
    return tuple(sorted(names))


def _validation_constants(
    spec: Mapping[str, object],
    build_plan: Mapping[str, object] | None,
    run_dir: Path,
    sources: Sequence[Union[str, ast.Module]],
) -> dict[str, object]:
    extent, nominal_extent = _protocol_extents(run_dir)
    values = _parameter_values(spec, build_plan, run_dir)
    context = values.get("context_length", values.get("P"))
    horizon = values.get("forecast_horizon", values.get("K"))
    constants: dict[str, object] = dict(values)
    if extent is not None:
        constants.update({symbol: extent for symbol in _TIME_AXIS_SYMBOLS})
        # Realized bundle facts used by the generated forecasting notebook's
        # dead-tail trim.  These names describe implementation data flow only.
        constants.update({
            "last_nonzero_col": extent,
            "last_valid_idx": extent - 1,
            "usable_end": extent,
            "T_usable": extent,
            "trailing_zeros": max((nominal_extent or extent) - extent, 0),
        })
    if context is not None:
        constants.update({"P": context, "context_length": context})
    if horizon is not None:
        constants.update({"K": horizon, "forecast_horizon": horizon})

    contract = _arch_contract(run_dir)
    tensor_names = _training_temporal_input_names(contract)
    model_names = {"model"}
    for source in sources:
        for function in _top_level_functions(source).values():
            args = (list(function.args.posonlyargs) + list(function.args.args)
                    + list(function.args.kwonlyargs))
            for arg in args:
                tokens = _tokens(arg.arg)
                if tokens & _MODEL_TOKENS:
                    model_names.add(arg.arg)
                if tokens & _TEMPORAL_TENSOR_TOKENS \
                        and not tokens & _MODEL_TOKENS:
                    tensor_names.add(arg.arg)

    if extent is not None:
        for name in tensor_names:
            constants[f"{name}.protocol_length"] = extent
    for name in model_names:
        if context is not None:
            constants[f"{name}.context_length"] = context
        if horizon is not None:
            constants[f"{name}.forecast_horizon"] = horizon
    return constants


def _messages(
    report: SplitLineageReport,
    file_label: str,
    *,
    include_unresolved: bool = True,
    source_labels: Mapping[str, str] | None = None,
    findings: Sequence[SplitLineageFinding] | None = None,
) -> list[str]:
    return [
        finding.message(file_label, source_labels)
        for finding in (
            tuple(findings)
            if findings is not None else find_split_lineage_findings(report)
        )
        if include_unresolved or finding.kind != "unresolved"
    ]


def _shift_range(
    value: ProtocolRange | None, offset: int,
) -> ProtocolRange | None:
    if value is None:
        return None
    return ProtocolRange(value.start - offset, value.stop - offset)


def _local_finding_shape(
    finding: SplitLineageFinding,
    *,
    root: str | None = None,
    model_id: str | None = None,
    offset: int = 0,
) -> tuple:
    endpoints = tuple(sorted(
        (
            item.role,
            item.read_kind,
            _shift_range(item.protocol_range, offset),
            item.subset,
        )
        for item in (finding.first, finding.second)
        if item is not None
    ))
    return (
        root or finding.root,
        model_id or finding.model_id,
        finding.kind,
        endpoints,
        _shift_range(finding.overlap, offset),
        finding.certainty,
    )


def _caller_local_finding_shapes(
    finding: SplitLineageFinding,
    report: SplitLineageReport,
) -> set[tuple]:
    """Candidate trainer-formal identities for a caller-bound finding."""
    shapes = {_local_finding_shape(finding)}
    if finding.second is None \
            or finding.first.source != finding.second.source:
        return shapes
    for call in report.calls:
        if call.callee != finding.first.source:
            continue
        model_formals = {
            binding.formal
            for binding in call.model_bindings
            if binding.actual == finding.model_id
        }
        if not model_formals and call.model_id == finding.model_id:
            model_formals.add(call.model_formal or finding.model_id)
        if not model_formals:
            continue
        for binding in call.bindings:
            if binding.root == finding.root:
                offset = (
                    binding.protocol_range.start
                    if binding.protocol_range is not None else 0
                )
                for model_formal in model_formals:
                    shapes.add(_local_finding_shape(
                        finding,
                        root=binding.formal,
                        model_id=model_formal,
                        offset=offset,
                    ))
    return shapes


def _producer_split_lineage_errors(
    spec: Mapping[str, object],
    build_plan: Mapping[str, object] | None,
    run_dir: Path,
    *,
    filename: str,
    preferred: Iterable[str] = (),
) -> list[str]:
    if not temporal_lineage_enabled(spec):
        return []
    path = run_dir / "method" / filename
    try:
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
    except (OSError, SyntaxError, ValueError):
        return []
    external_trainers: dict[str, str] = {}
    analysis_sources = [source]
    if filename == "method.py":
        training_path = run_dir / "method" / "training.py"
        try:
            training_source = training_path.read_text(encoding="utf-8")
            ast.parse(training_source)
        except (OSError, SyntaxError, ValueError):
            training_source = ""
        if training_source:
            analysis_sources.append(training_source)
            external_trainers.update({
                name: training_source
                for name in _trainer_names(training_source)
            })
    constants = _validation_constants(
        spec, build_plan, run_dir, sources=tuple(analysis_sources)
    )
    errors: list[str] = []
    preferred_names = {
        name for name in preferred
        if isinstance(name, str)
    }
    for name in _trainer_names(source, preferred):
        if filename == "method.py":
            report = analyze_split_lineage(
                source,
                external_trainers,
                constants=constants,
                entry_function=name,
                # Method helpers may compute training-progress metrics.  A
                # metric becomes reported evaluation only when it reaches the
                # declared public component's return boundary.
                observe_reported_metrics=False,
                observe_returned_metrics=name in preferred_names,
                project_child_observations=True,
                source_label=name,
            )
        else:
            report = summarize_trainer(
                source,
                function_name=name,
                constants=constants,
            )
        errors.extend(_messages(
            report,
            f"method/{filename}",
            include_unresolved=filename == "method.py",
            source_labels={name: f"method/{filename}"},
        ))
    return errors


def architecture_split_lineage_errors(
    spec: Mapping[str, object],
    build_plan: Mapping[str, object] | None,
    run_dir: Path,
) -> list[str]:
    """Validate trainer-local target relationships at the architecture seam."""
    contract = _arch_contract(run_dir)
    training_loop = contract.get("training_loop")
    preferred = []
    if isinstance(training_loop, Mapping) \
            and isinstance(training_loop.get("function_name"), str):
        preferred.append(training_loop["function_name"])
    # Stage 2b lacks caller-bound values and params.json.  Emit concrete
    # trainer-local relations here, then let the Stage 3a whole-program seam
    # fail closed on any still-relevant unresolved target read.
    return _producer_split_lineage_errors(
        spec,
        build_plan,
        run_dir,
        filename="training.py",
        preferred=preferred,
    )


def method_split_lineage_errors(
    spec: Mapping[str, object],
    build_plan: Mapping[str, object] | None,
    run_dir: Path,
) -> list[str]:
    """Validate method-coder-owned temporal target relationships."""
    pluggable = spec.get("comparison")
    component = (pluggable.get("pluggable_component")
                 if isinstance(pluggable, Mapping) else {})
    preferred = []
    if isinstance(component, Mapping) \
            and isinstance(component.get("name"), str):
        preferred.append(component["name"])
    return _producer_split_lineage_errors(
        spec,
        build_plan,
        run_dir,
        filename="method.py",
        preferred=preferred,
    )


def _notebook_split_lineage_analysis(
    spec: Mapping[str, object],
    build_plan: Mapping[str, object] | None,
    run_dir: Path,
    tree: ast.Module,
) -> _NotebookSplitLineageAnalysis:
    """Analyze and policy-classify the stage 3a caller seam once."""
    sources: list[str] = []
    trainers: dict[str, str] = {}
    trainer_sources: dict[str, str] = {}
    contract = _arch_contract(run_dir)
    training_loop = contract.get("training_loop")
    preferred = []
    if isinstance(training_loop, Mapping) \
            and isinstance(training_loop.get("function_name"), str):
        preferred.append(training_loop["function_name"])
    comparison = spec.get("comparison")
    component = (comparison.get("pluggable_component")
                 if isinstance(comparison, Mapping) else {})
    method_preferred = []
    if isinstance(component, Mapping) \
            and isinstance(component.get("name"), str):
        method_preferred.append(component["name"])
    for filename in ("training.py", "method.py"):
        path = run_dir / "method" / filename
        try:
            source = path.read_text(encoding="utf-8")
            ast.parse(source)
        except (OSError, SyntaxError, ValueError):
            continue
        sources.append(source)
        names = _trainer_names(
            source,
            preferred if filename == "training.py" else method_preferred,
        )
        trainers.update({name: source for name in names})
        trainer_sources.update({name: f"method/{filename}" for name in names})
    constants = _validation_constants(
        spec, build_plan, run_dir, sources=tuple(sources)
    )
    report = analyze_split_lineage(tree, trainers, constants=constants)
    findings = tuple(find_split_lineage_findings(report))

    # A trainer-local relationship already emitted at its producing seam must
    # not be duplicated as a notebook-owned defect.  Retain relationships
    # that only become decidable after caller binding, including archived
    # caller-supplied split boundaries.
    local_shapes: dict[str, set[tuple]] = {}
    for name, source in trainers.items():
        local_report = summarize_trainer(
            source,
            function_name=name,
            constants=constants,
            observe_reported_metrics=bool(
                _tokens(name) & _EVALUATION_FUNCTION_TOKENS
            ),
        )
        local_shapes[name] = {
            _local_finding_shape(finding)
            for finding in find_split_lineage_findings(local_report)
            if finding.kind != "unresolved"
        }
    emitted_findings = tuple(
        finding for finding in findings
        if not (
            finding.second is not None
            and finding.first.source == finding.second.source
            and finding.first.source in trainer_sources
            and bool(
                _caller_local_finding_shapes(finding, report)
                & local_shapes.get(finding.first.source, set())
            )
        )
    )
    source_labels = {"notebook": "notebook.ipynb", **trainer_sources}
    return _NotebookSplitLineageAnalysis(
        report=report,
        findings=findings,
        emitted_findings=emitted_findings,
        source_labels=source_labels,
    )


def notebook_split_lineage_errors(
    spec: Mapping[str, object],
    build_plan: Mapping[str, object] | None,
    run_dir: Path,
    tree: ast.Module,
) -> list[str]:
    """Validate notebook-to-trainer ranges at the stage 3a caller seam."""
    if not temporal_lineage_enabled(spec):
        return []
    analysis = _notebook_split_lineage_analysis(
        spec, build_plan, run_dir, tree
    )
    return _messages(
        analysis.report,
        "notebook.ipynb",
        source_labels=analysis.source_labels,
        findings=analysis.emitted_findings,
    )


def notebook_split_validity_receipt(
    spec: Mapping[str, object],
    build_plan: Mapping[str, object] | None,
    run_dir: Path,
    tree: ast.Module,
) -> dict[str, object]:
    """Return the trusted split-validity receipt for one notebook surface."""
    if not temporal_lineage_enabled(spec):
        return derive_split_validity_receipt(
            SplitLineageReport((), (), (), ()),
            applicable=False,
        )
    analysis = _notebook_split_lineage_analysis(
        spec, build_plan, run_dir, tree
    )
    return _derive_split_validity_receipt(
        analysis.report,
        analysis.findings,
    )
