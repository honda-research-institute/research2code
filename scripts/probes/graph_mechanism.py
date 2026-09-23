"""Dependency-light executor for homogeneous graph-mechanism probes.

The executor deliberately does not import a method package, resolve taxonomy,
or interpret a generated callable signature.  Its caller supplies one frozen,
strict-JSON plan plus adapters for the exact callables named by that plan.  The
adapters have two small normalized interfaces::

    construct(
        *, feature_input, threshold, cap, seed
    ) -> {"graph": ..., "parameter_values": {<params name>: ...}}

    execute(
        *, graph, neighbor_signal, entity_ids, seed
    ) -> numeric output with the declared entity axis

This keeps delivery-time and portable execution on the same numerical path
without making this module discover generated code or vendor Pydantic plan
resolution.

The normalized plan consumed here has this version-one shape.  Disposition
plans retain every top-level key with empty or null payloads::

    {
      "schema_version": "1.0",
      "status": "ready" | "not_applicable" | "unprobeable" | "blocked",
      "reason": "...",
      "representation": "sparse_edge_index" | "dense_adjacency",
      "alignment": {"status": "pass", "reason": "..."},
      "alignment_runtime_receipt": {
        "receipt_version": "1.0",
        "probe_ref": "graph_mechanism.alignment_prerequisite",
        "status": "pass",
        "reason": "stage_2d_runtime_alignment_verified",
        "verified": true,
        "validator": "validate_arch_contract_runtime.py",
        "element_id": "...",
        "fixture_scope": "...",
        "representation": "sparse_edge_index" | "dense_adjacency",
        "entity_ids": ["..."],
        "authority_digests": {"method_spec.json": "sha256:<64 hex>"}
      },
      "construction": {
        "callable": {"module": "method.model", "qualname": "build_graph"},
        "feature_input_root": "static_features",
        "feature_parameter": "static_features",
        "output_selector": {...},
        "threshold": {
          "metric": "cosine_similarity",
          "comparison": "greater_than_or_equal",
          "parameter": {
            "params_name": "similarity_threshold",
            "callable_parameter": "similarity_threshold"
          },
          "value": 0.95
        },
        "cap": {
          "kind": "none"
        } | {
          "kind": "per_source_top_similarity",
          "parameter": {
            "params_name": "max_neighbors",
            "callable_parameter": "max_neighbors"
          },
          "value": 10
        },
        "self_loop_policy": "required" | "forbidden",
        "direction_policy": "directed" | "undirected_bidirectional"
      },
      "parameter_authority": {
        "status": "pass",
        "reason": "...",
        "bindings": {
          "similarity_threshold": {
            "carrier_value": 0.95,
            "params_value": 0.95,
            "paper_value": null,
            "source": "paper",
            "suppressed": false,
            "duplicates": []
          }
        }
      },
      "execution": {
        "callable": {"module": "method.model", "qualname": "encode"},
        "neighbor_signal_root": "node_signals",
        "neighbor_signal_parameter": "node_signals",
        "entity_axis": 0,
        "output_entity_axis": 0,
        "permutation_applicability": "required" | "not_applicable" |
                                      "unresolved",
        "tolerance": 1e-8,
        "seed": 1729
      },
      "ablation": {
        "kind": "empty_graph" | "identity_graph" | "permuted_graph",
        "element_id": "...",
        "discriminating_probe_ref": "graph_mechanism.neighbor_sensitivity"
      } | {
        "kind": "removed_message_passing" | "non_graph_decoder",
        "element_id": "...",
        "callable": {"module": "method.model", "qualname": "null_encode"},
        "graph_parameter": "edge_index",
        "neighbor_signal_parameter": "neighbor_history",
        "output_root": "outputs.graph_embeddings",
        "discriminating_probe_ref": "graph_mechanism.neighbor_sensitivity"
      },
      "groundings": {
        <probe ref>: {
          "element_ids": ["..."],
          "bound_callables": ["method.model:encode"]
        }
      },
      "fixture": {
        ...,
        "comparison_witness": {
          "metric": "cosine_similarity",
          "comparison": "greater_than_or_equal",
          "threshold_value": 1.0,
          "entity_axis": 0,
          "feature_input": [[1.0, 0.0], [1.0, 0.0], ...],
          "expected_graph": [[...], [...]]
        },
        "parameter_interventions": {
          "similarity_threshold": {
            "nominal_value": 0.95,
            "alternate_value": 0.75,
            "value": 0.75,
            "callable_parameter": "similarity_threshold",
            "expected_relation": "graph_must_differ",
            "expected_graph": [[...], [...]]
          }
        }
      }
    }

The plan's JSON fixture carries exact arrays and interventions: ``entity_ids``,
the construction and neighbor-signal roots, ``expected_graph``, the exact-1.0
comparison witness, topology and neighbor interventions, ``permutation``, and
independent alternate graph oracles.  The separately injected fixture must be
that frozen fixture (or an exact, tensorized copy).  This leaves callables
injectable without allowing generated code to choose its own evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


PROBE_REFS = {
    "HG-2": "graph_mechanism.parameter_agreement",
    "HG-3": "graph_mechanism.construction_semantics",
    "HG-4": "graph_mechanism.topology_sensitivity",
    "HG-5": "graph_mechanism.neighbor_sensitivity",
    "HG-6": "graph_mechanism.permutation_equivalence",
    "HG-7": "graph_mechanism.contribution_ablation",
}
PROBE_IDS = tuple(PROBE_REFS)
RESULT_STATUSES = ("pass", "fail", "unprobeable", "not_applicable")

_INPUT_ABLATIONS = frozenset({
    "empty_graph", "identity_graph", "permuted_graph",
})
_CALLABLE_ABLATIONS = frozenset({
    "removed_message_passing", "non_graph_decoder",
})


@dataclass(frozen=True)
class GraphProbeResult:
    """One adapter-friendly graph assessment with strict JSON evidence."""

    probe_id: str
    probe_ref: str
    status: str
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    element_ids: list[str] = field(default_factory=list)
    bound_callables: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        expected_ref = PROBE_REFS.get(self.probe_id)
        if expected_ref is None:
            raise ValueError(f"unknown graph probe id {self.probe_id!r}")
        if self.probe_ref != expected_ref:
            raise ValueError(
                f"{self.probe_id} requires probe_ref {expected_ref!r}"
            )
        if self.status not in RESULT_STATUSES:
            raise ValueError(f"unknown graph probe status {self.status!r}")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("graph probe reason must be nonblank")
        json.dumps(self.evidence, allow_nan=False)
        for label, values in (
            ("element_ids", self.element_ids),
            ("bound_callables", self.bound_callables),
        ):
            if any(not isinstance(value, str) or not value.strip()
                   for value in values):
                raise ValueError(f"{label} must contain nonblank strings")
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must be unique")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Assessment:
    status: str
    code: str
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _GraphView:
    representation: str
    node_count: int
    edges: frozenset[tuple[int, int]]
    duplicate_edges: int


def _strict_json_problem(value: object) -> str | None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        return f"frozen graph plan is not strict JSON: {exc}"
    return None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    result: list[str] = []
    for raw in value:
        if not isinstance(raw, str) or not raw.strip():
            continue
        text = raw.strip()
        if text not in result:
            result.append(text)
    return result


def _grounding(
    plan: Mapping[str, Any], probe_id: str,
) -> tuple[list[str], list[str]]:
    raw_groundings = plan.get("groundings")
    if not isinstance(raw_groundings, Mapping):
        return [], []
    raw = raw_groundings.get(PROBE_REFS[probe_id])
    if not isinstance(raw, Mapping):
        raw = raw_groundings.get(probe_id)
    if not isinstance(raw, Mapping):
        raw = next(
            (
                candidate
                for candidate in raw_groundings.values()
                if isinstance(candidate, Mapping)
                and candidate.get("probe_ref") == PROBE_REFS[probe_id]
            ),
            None,
        )
    if not isinstance(raw, Mapping):
        raw = {}
    bound_callables = _string_list(raw.get("bound_callables"))
    if not bound_callables:
        identities: list[object] = []
        if probe_id in ("HG-2", "HG-3"):
            construction = plan.get("construction")
            if isinstance(construction, Mapping):
                identities.append(construction.get("callable"))
        elif probe_id in ("HG-4", "HG-5", "HG-6"):
            execution = plan.get("execution")
            if isinstance(execution, Mapping):
                identities.append(execution.get("callable"))
        elif probe_id == "HG-7":
            ablation = plan.get("ablation")
            if isinstance(ablation, Mapping):
                execution = plan.get("execution")
                if isinstance(execution, Mapping):
                    identities.append(execution.get("callable"))
                identities.append(ablation.get("callable"))
        for identity in identities:
            key = _callable_key(identity)
            if key is not None and key not in bound_callables:
                bound_callables.append(key)
    return (
        _string_list(raw.get("element_ids")),
        bound_callables,
    )


def _result(
    plan: Mapping[str, Any],
    probe_id: str,
    status: str,
    reason: str,
    **evidence: Any,
) -> GraphProbeResult:
    element_ids, bound_callables = _grounding(plan, probe_id)
    traced_evidence = {"trace": graph_plan_trace(plan), **evidence}
    return GraphProbeResult(
        probe_id=probe_id,
        probe_ref=PROBE_REFS[probe_id],
        status=status,
        reason=reason,
        evidence=traced_evidence,
        element_ids=element_ids,
        bound_callables=bound_callables,
    )


def _disposition_results(
    plan: Mapping[str, Any], status: str, reason: str,
) -> list[GraphProbeResult]:
    evidence: dict[str, Any] = {}
    liveness = plan.get("callable_liveness_assessment")
    if isinstance(liveness, Mapping):
        try:
            evidence["callable_liveness_assessment"] = json.loads(
                json.dumps(liveness, allow_nan=False)
            )
        except (TypeError, ValueError):
            pass
    return [
        _result(plan, probe_id, status, reason, **evidence)
        for probe_id in PROBE_IDS
    ]


def _callable_key(identity: object) -> str | None:
    if isinstance(identity, str) and identity.strip():
        return identity.strip()
    if not isinstance(identity, Mapping):
        return None
    module = identity.get("module")
    qualname = identity.get("qualname")
    if not isinstance(module, str) or not module.strip() \
            or not isinstance(qualname, str) or not qualname.strip():
        return None
    return f"{module.strip()}:{qualname.strip()}"


def _resolve_callable(
    callables: Mapping[str, Callable[..., object]], identity: object,
) -> tuple[Callable[..., object] | None, str | None, str | None]:
    key = _callable_key(identity)
    if key is None:
        return None, None, "frozen plan has no exact callable identity"
    candidate = callables.get(key)
    if not callable(candidate):
        return None, key, f"injected callable adapter {key!r} is missing"
    return candidate, key, None


def _clone(value: object) -> object:
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    copier = getattr(value, "copy", None)
    if callable(copier):
        return copier()
    return copy.deepcopy(value)


def _fixture_json_value(value: object) -> object:
    """Normalize list/NumPy/Torch containers without changing their values."""

    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy = getattr(candidate, "numpy", None)
    if callable(numpy):
        candidate = numpy()
    if isinstance(candidate, np.ndarray):
        return _fixture_json_value(candidate.tolist())
    if isinstance(candidate, np.generic):
        return _fixture_json_value(candidate.item())
    if isinstance(candidate, Mapping):
        return {
            str(key): _fixture_json_value(item)
            for key, item in candidate.items()
        }
    if isinstance(candidate, Sequence) and not isinstance(
        candidate, (str, bytes)
    ):
        return [_fixture_json_value(item) for item in candidate]
    return candidate


def _fixture_digest(value: object) -> tuple[str | None, str | None]:
    try:
        payload = json.dumps(
            _fixture_json_value(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        return None, f"graph fixture is not strict JSON after tensorization: {exc}"
    return f"sha256:{hashlib.sha256(payload).hexdigest()}", None


def fixture_digest(value: object) -> str:
    """Canonical strict-JSON identity for a frozen or tensorized fixture."""
    digest, problem = _fixture_digest(value)
    if problem is not None or digest is None:
        raise ValueError(problem or "graph fixture digest is unavailable")
    return digest


def execution_plan_digest(plan: Mapping[str, Any]) -> str:
    """Canonical strict-JSON identity for the exact frozen execution plan."""
    payload = json.dumps(
        plan,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _fixture_array(value: object) -> np.ndarray | None:
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy = getattr(candidate, "numpy", None)
    if callable(numpy):
        candidate = numpy()
    return candidate if isinstance(candidate, np.ndarray) else None


def _fixtures_equivalent(frozen: object, injected: object) -> bool:
    """Compare the complete fixture, allowing declared tensor materialization."""

    frozen_array = _fixture_array(frozen)
    injected_array = _fixture_array(injected)
    if frozen_array is not None or injected_array is not None:
        try:
            left = np.asarray(frozen if frozen_array is None else frozen_array)
            right = np.asarray(
                injected if injected_array is None else injected_array
            )
        except (TypeError, ValueError):
            return False
        if left.shape != right.shape:
            return False
        if left.dtype.kind in "fc" or right.dtype.kind in "fc":
            return np.array_equal(
                left.astype(np.float32), right.astype(np.float32)
            )
        return np.array_equal(left.astype(np.int64), right.astype(np.int64))
    if isinstance(frozen, Mapping) or isinstance(injected, Mapping):
        if not isinstance(frozen, Mapping) or not isinstance(injected, Mapping):
            return False
        if set(frozen) != set(injected):
            return False
        return all(
            _fixtures_equivalent(frozen[key], injected[key])
            for key in frozen
        )
    sequence_types = (Sequence,)
    frozen_sequence = isinstance(frozen, sequence_types) and not isinstance(
        frozen, (str, bytes)
    )
    injected_sequence = isinstance(injected, sequence_types) and not isinstance(
        injected, (str, bytes)
    )
    if frozen_sequence or injected_sequence:
        if not frozen_sequence or not injected_sequence:
            return False
        if len(frozen) != len(injected):
            return False
        return all(
            _fixtures_equivalent(left, right)
            for left, right in zip(frozen, injected)
        )
    return type(frozen) is type(injected) and frozen == injected


def graph_plan_trace(plan: Mapping[str, Any]) -> dict[str, Any]:
    construction = plan.get("construction")
    execution = plan.get("execution")
    fixture = plan.get("fixture")
    fixture_digest, _ = _fixture_digest(fixture)
    return {
        "execution_plan_digest": execution_plan_digest(plan),
        "fixture_scope": (
            fixture.get("scope") if isinstance(fixture, Mapping) else None
        ),
        "fixture_digest": fixture_digest,
        "representation": plan.get("representation"),
        "construction_feature_root": (
            construction.get("feature_input_root")
            if isinstance(construction, Mapping) else None
        ),
        "neighbor_signal_root": (
            execution.get("neighbor_signal_root")
            if isinstance(execution, Mapping) else None
        ),
        "output_root": (
            execution.get("output_root")
            if isinstance(execution, Mapping) else None
        ),
        "seed": execution.get("seed") if isinstance(execution, Mapping) else None,
        "tolerance": (
            execution.get("tolerance")
            if isinstance(execution, Mapping) else None
        ),
    }


def _numeric_array(
    value: object, *, label: str,
) -> tuple[np.ndarray | None, str | None]:
    candidate = value
    if isinstance(candidate, np.ndarray) and type(candidate) is not np.ndarray:
        return (
            None,
            f"{label} must use an exact base NumPy array; ndarray subclass "
            "dispatch is outside the frozen-fixture proof",
        )
    try:
        import torch  # noqa: PLC0415
    except ImportError:  # pragma: no cover - torch is an optional dependency
        torch = None
    if (
        torch is not None
        and isinstance(candidate, torch.Tensor)
        and type(candidate) is not torch.Tensor
    ):
        return (
            None,
            f"{label} must use an exact base Torch tensor; Tensor subclass "
            "dispatch is outside the frozen-fixture proof",
        )
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy = getattr(candidate, "numpy", None)
    if callable(numpy):
        candidate = numpy()
    try:
        array = np.asarray(candidate, dtype=float)
    except (TypeError, ValueError) as exc:
        return None, f"{label} is not numeric: {exc}"
    if array.ndim == 0:
        return None, f"{label} must carry an entity axis"
    if type(array) is not np.ndarray:
        return None, f"{label} did not normalize to an exact base NumPy array"
    if not np.isfinite(array).all():
        return None, f"{label} contains a non-finite value"
    return array, None


def _fixture_entity_ids(fixture: Mapping[str, Any]) -> object:
    entity_ids = fixture.get("entity_ids")
    if entity_ids is not None:
        return entity_ids
    source = fixture.get("source")
    if not isinstance(source, Mapping):
        alignment_source = fixture.get("alignment_source")
        source = alignment_source.get("source") \
            if isinstance(alignment_source, Mapping) else None
    return source.get("stable_entity_ids") if isinstance(source, Mapping) else None


def _entity_count(fixture: Mapping[str, Any]) -> tuple[int | None, str | None]:
    entity_ids = _fixture_entity_ids(fixture)
    if not isinstance(entity_ids, Sequence) or isinstance(entity_ids, (str, bytes)):
        return None, "fixture entity_ids must be a sequence"
    if len(entity_ids) < 3:
        return None, "graph fixture requires at least three stable entities"
    try:
        identities = [json.dumps(value, sort_keys=True, allow_nan=False)
                      for value in entity_ids]
    except (TypeError, ValueError) as exc:
        return None, f"fixture entity_ids are not strict JSON: {exc}"
    if len(identities) != len(set(identities)):
        return None, "fixture entity_ids must be unique"
    return len(entity_ids), None


def _alignment_receipt_problem(
    plan: Mapping[str, Any], fixture: Mapping[str, Any],
) -> str | None:
    """Verify the runner-issued HG-1 receipt before any graph callable runs."""

    alignment = plan.get("alignment")
    plan_fixture = plan.get("fixture")
    receipt = plan.get("alignment_runtime_receipt")
    if not isinstance(alignment, Mapping) or alignment.get("status") != "pass":
        return "frozen relational alignment was not runtime-promoted to pass"
    if not isinstance(plan_fixture, Mapping):
        return "frozen plan has no alignment fixture authority"
    if not isinstance(receipt, Mapping):
        return "verified graph alignment runtime receipt is missing"
    expected_keys = {
        "receipt_version",
        "probe_ref",
        "status",
        "reason",
        "verified",
        "validator",
        "element_id",
        "fixture_scope",
        "representation",
        "entity_ids",
        "authority_digests",
    }
    if set(receipt) != expected_keys:
        return "graph alignment runtime receipt has an unexpected shape"
    expected_values = {
        "receipt_version": "1.0",
        "probe_ref": "graph_mechanism.alignment_prerequisite",
        "status": "pass",
        "reason": "stage_2d_runtime_alignment_verified",
        "verified": True,
        "validator": "validate_arch_contract_runtime.py",
        "element_id": alignment.get("element_id"),
        "fixture_scope": plan_fixture.get("scope"),
        "representation": plan.get("representation"),
    }
    for key, expected in expected_values.items():
        if receipt.get(key) != expected:
            return f"graph alignment runtime receipt {key} does not match the plan"
    entity_ids = _fixture_entity_ids(fixture)
    try:
        receipt_ids = json.dumps(
            receipt.get("entity_ids"), sort_keys=True, allow_nan=False
        )
        fixture_ids = json.dumps(entity_ids, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        return f"graph alignment runtime receipt entity ids are invalid: {exc}"
    if receipt_ids != fixture_ids:
        return "graph alignment runtime receipt entity ids do not match the fixture"
    authority_digests = receipt.get("authority_digests")
    if not isinstance(authority_digests, Mapping) or not authority_digests:
        return "graph alignment runtime receipt authority_digests are missing"
    for authority, digest in authority_digests.items():
        if not isinstance(authority, str) or not authority.strip():
            return "graph alignment runtime receipt has an invalid authority name"
        if not isinstance(digest, str) or len(digest) != 71 \
                or not digest.startswith("sha256:") \
                or digest != digest.lower() \
                or any(
                    character not in "0123456789abcdef"
                    for character in digest.removeprefix("sha256:")
                ):
            return "graph alignment runtime receipt has an invalid authority digest"
    return None


def _axis_index(axis: object, rank: int, *, label: str) -> tuple[int | None, str | None]:
    if isinstance(axis, bool) or not isinstance(axis, int):
        return None, f"{label} must be an integer"
    normalized = axis + rank if axis < 0 else axis
    if not 0 <= normalized < rank:
        return None, f"{label} is outside rank {rank}"
    return normalized, None


def _graph_view(
    graph: object,
    *,
    representation: str,
    node_count: int,
) -> tuple[_GraphView | None, str | None]:
    raw = graph
    detach = getattr(raw, "detach", None)
    if callable(detach):
        raw = detach()
    cpu = getattr(raw, "cpu", None)
    if callable(cpu):
        raw = cpu()
    numpy = getattr(raw, "numpy", None)
    if callable(numpy):
        raw = numpy()
    try:
        array = np.asarray(raw)
    except (TypeError, ValueError) as exc:
        return None, f"graph is not array-like: {exc}"

    if representation == "sparse_edge_index":
        if array.ndim != 2 or array.shape[0] != 2:
            return None, "sparse graph must have shape (2, E)"
        if array.dtype.kind == "b":
            return None, "sparse graph endpoints cannot be boolean"
        json_empty = array.size == 0 and isinstance(raw, (list, tuple))
        if array.dtype.kind not in "iu" and not json_empty:
            return None, "sparse graph endpoints must use an integer dtype"
        endpoints = array.astype(np.int64)
        if endpoints.size and (
            int(endpoints.min()) < 0 or int(endpoints.max()) >= node_count
        ):
            return None, "sparse graph endpoint is outside the entity axis"
        ordered = [
            (int(source), int(destination))
            for source, destination in zip(endpoints[0], endpoints[1])
        ]
        edges = frozenset(ordered)
        return _GraphView(
            representation=representation,
            node_count=node_count,
            edges=edges,
            duplicate_edges=len(ordered) - len(edges),
        ), None

    if representation == "dense_adjacency":
        if array.ndim != 2 or tuple(array.shape) != (node_count, node_count):
            return None, (
                "dense graph must be square on the canonical entity axis"
            )
        try:
            numeric = array.astype(float)
        except (TypeError, ValueError) as exc:
            return None, f"dense graph is not numeric: {exc}"
        if not np.isfinite(numeric).all():
            return None, "dense graph contains a non-finite value"
        sources, destinations = np.nonzero(numeric)
        edges = frozenset(
            (int(source), int(destination))
            for source, destination in zip(sources, destinations)
        )
        return _GraphView(
            representation=representation,
            node_count=node_count,
            edges=edges,
            duplicate_edges=0,
        ), None

    return None, f"unsupported graph representation {representation!r}"


def _graph_from_edges(
    original: object,
    *,
    view: _GraphView,
    edges: set[tuple[int, int]],
) -> object:
    if view.representation == "sparse_edge_index":
        ordered = sorted(edges)
        if not ordered:
            return np.empty((2, 0), dtype=np.int64)
        return np.asarray(ordered, dtype=np.int64).T
    array = np.asarray(_clone(original))
    result = np.zeros_like(array)
    for source, destination in edges:
        # Retained topology keeps its exact dense edge payload. Input
        # ablations may synthesize an identity edge that was absent; one is
        # the representation-neutral presence value for that case.
        original_value = array[source, destination]
        result[source, destination] = (
            original_value if original_value != 0 else 1
        )
    return result


def _numeric_value(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _values_equal(left: object, right: object) -> bool:
    if _numeric_value(left) and _numeric_value(right):
        return float(left) == float(right)
    return type(left) is type(right) and left == right


def _parameter_specs(
    construction: Mapping[str, Any],
) -> tuple[list[tuple[str, str, object]] | None, str | None]:
    threshold = construction.get("threshold")
    if not isinstance(threshold, Mapping):
        return None, "construction threshold binding is missing"
    threshold_parameter = threshold.get("parameter")
    if not isinstance(threshold_parameter, Mapping):
        return None, "construction threshold has no exact parameter binding"
    threshold_name = threshold_parameter.get("params_name")
    threshold_value = threshold.get("value")
    if not isinstance(threshold_name, str) or not threshold_name.strip():
        return None, "construction threshold params_name is missing"
    if not _numeric_value(threshold_value):
        return None, "construction threshold value must be finite numeric"
    result = [("threshold", threshold_name.strip(), threshold_value)]

    cap = construction.get("cap")
    if not isinstance(cap, Mapping):
        return None, "construction cap declaration is missing"
    kind = cap.get("kind")
    if kind == "none":
        return result, None
    if kind != "per_source_top_similarity":
        return None, f"unsupported construction cap kind {kind!r}"
    cap_parameter = cap.get("parameter")
    if not isinstance(cap_parameter, Mapping):
        return None, "construction cap has no exact parameter binding"
    cap_name = cap_parameter.get("params_name")
    cap_value = cap.get("value")
    if not isinstance(cap_name, str) or not cap_name.strip():
        return None, "construction cap params_name is missing"
    if isinstance(cap_value, bool) or not isinstance(cap_value, int) \
            or cap_value <= 0:
        return None, "construction cap value must be a positive integer"
    result.append(("cap", cap_name.strip(), cap_value))
    return result, None


def _call_with_isolated_rng(
    adapter: Callable[..., object],
    *,
    seed: int,
    kwargs: Mapping[str, object],
) -> object:
    """Seed supported global RNGs for one call and restore ambient state."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_generator = None
    torch_state = None
    try:
        try:
            import torch  # noqa: PLC0415

            candidate = getattr(torch, "default_generator", None)
            if candidate is not None and str(candidate.device) == "cpu":
                torch_generator = candidate
                torch_state = candidate.get_state()
        except Exception:  # noqa: BLE001 - torch is an optional dependency
            torch_generator = None
            torch_state = None

        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        if torch_generator is not None:
            torch_generator.manual_seed(seed)
        return adapter(**dict(kwargs))
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch_generator is not None and torch_state is not None:
            torch_generator.set_state(torch_state)


def _independent_clone(value: object) -> object:
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def _constructor_capture_snapshot(
    capture: Mapping[str, Any],
    *,
    representation: str,
    node_count: int,
) -> tuple[tuple[object, str] | None, str | None]:
    view, problem = _graph_view(
        capture.get("graph"),
        representation=representation,
        node_count=node_count,
    )
    if problem:
        return None, problem
    assert view is not None
    raw = capture.get("graph")
    detach = getattr(raw, "detach", None)
    if callable(detach):
        raw = detach()
    cpu = getattr(raw, "cpu", None)
    if callable(cpu):
        raw = cpu()
    as_numpy = getattr(raw, "numpy", None)
    if callable(as_numpy):
        raw = as_numpy()
    array = np.asarray(raw)
    if representation == "sparse_edge_index":
        endpoints = array.astype(np.int64)
        graph_snapshot: object = tuple(sorted(
            (int(source), int(target))
            for source, target in zip(endpoints[0], endpoints[1])
        ))
    else:
        graph_snapshot = array.astype(float).copy()
    try:
        parameter_snapshot = json.dumps(
            _fixture_json_value(capture.get("parameter_values")),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        return None, f"construction parameter_values are not strict JSON: {exc}"
    return (graph_snapshot, parameter_snapshot), None


def _constructor_capture_difference(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    representation: str,
    node_count: int,
) -> str | None:
    first_snapshot, first_problem = _constructor_capture_snapshot(
        first,
        representation=representation,
        node_count=node_count,
    )
    second_snapshot, second_problem = _constructor_capture_snapshot(
        second,
        representation=representation,
        node_count=node_count,
    )
    if first_problem or second_problem:
        return first_problem or second_problem
    assert first_snapshot is not None and second_snapshot is not None
    first_graph, first_parameters = first_snapshot
    second_graph, second_parameters = second_snapshot
    if representation == "dense_adjacency":
        graph_equal = np.array_equal(first_graph, second_graph)
    else:
        graph_equal = first_graph == second_graph
    if not graph_equal:
        return "same-seed graph construction returned a different graph"
    if first_parameters != second_parameters:
        return "same-seed graph construction returned different parameter_values"
    return None


def _call_constructor(
    plan: Mapping[str, Any],
    callables: Mapping[str, Callable[..., object]],
    fixture: Mapping[str, Any],
    *,
    node_count: int,
    parameter_overrides: Mapping[str, object] | None = None,
) -> tuple[Mapping[str, Any] | None, str | None, str | None]:
    construction = plan.get("construction")
    execution = plan.get("execution")
    if not isinstance(construction, Mapping) or not isinstance(execution, Mapping):
        return None, None, "frozen plan lacks construction or execution"
    adapter, callable_key, problem = _resolve_callable(
        callables, construction.get("callable")
    )
    if problem:
        return None, callable_key, problem
    assert adapter is not None
    root = construction.get("feature_input_root")
    if not isinstance(root, str) or not root.strip() or root not in fixture:
        return None, callable_key, "fixture lacks the exact construction input root"
    threshold = construction.get("threshold")
    cap = construction.get("cap")
    threshold_parameter = threshold.get("parameter") \
        if isinstance(threshold, Mapping) else None
    threshold_value = threshold.get("value") \
        if isinstance(threshold, Mapping) else None
    cap_parameter = cap.get("parameter") \
        if isinstance(cap, Mapping) and cap.get("kind") != "none" else None
    cap_value = cap.get("value") \
        if isinstance(cap, Mapping) and cap.get("kind") != "none" else None
    overrides = parameter_overrides or {}
    threshold_name = threshold_parameter.get("params_name") \
        if isinstance(threshold_parameter, Mapping) else None
    if isinstance(threshold_name, str) and threshold_name in overrides:
        threshold_value = overrides[threshold_name]
    cap_name = cap_parameter.get("params_name") \
        if isinstance(cap_parameter, Mapping) else None
    if isinstance(cap_name, str) and cap_name in overrides:
        cap_value = overrides[cap_name]
    seed = execution.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        return None, callable_key, "execution seed must be an integer"
    kwargs = {
        "feature_input": _clone(fixture[root]),
        "threshold": threshold_value,
        "cap": cap_value,
        "seed": seed,
    }
    captures: list[Mapping[str, Any]] = []
    for label in ("construction", "construction replay"):
        try:
            captured = _call_with_isolated_rng(
                adapter,
                seed=seed,
                kwargs={
                    **kwargs,
                    "feature_input": _clone(fixture[root]),
                },
            )
        except Exception as exc:  # noqa: BLE001 - generated code is arbitrary
            return None, callable_key, (
                f"{label} adapter raised {type(exc).__name__}: {exc}"
            )
        if not isinstance(captured, Mapping):
            return None, callable_key, (
                f"{label} adapter must return a graph and parameter_values"
            )
        if "graph" not in captured or not isinstance(
            captured.get("parameter_values"), Mapping
        ):
            return None, callable_key, (
                f"{label} capture lacks graph or parameter_values"
            )
        captures.append(captured)
    difference = _constructor_capture_difference(
        captures[0],
        captures[1],
        representation=str(plan.get("representation")),
        node_count=node_count,
    )
    if difference:
        return None, callable_key, difference
    return {
        "graph": _independent_clone(captures[0]["graph"]),
        "parameter_values": _independent_clone(
            captures[0]["parameter_values"]
        ),
    }, callable_key, None


def _assess_parameter_consumption(
    plan: Mapping[str, Any],
    callables: Mapping[str, Callable[..., object]],
    fixture: Mapping[str, Any],
    *,
    node_count: int,
    nominal_capture: Mapping[str, Any],
) -> _Assessment:
    """Prove that each graph parameter changes construction, not just a call.

    A constructor can accept a keyword and even echo it in telemetry while
    continuing to use an internal nominal literal.  The fixture therefore
    supplies one independently derived, topology-changing expected graph for
    each bound threshold/cap.  We invoke one variation at a time on the same
    features and seed and compare the returned topology with that oracle.
    """

    construction = plan.get("construction")
    representation = plan.get("representation")
    if not isinstance(construction, Mapping) or not isinstance(
        representation, str
    ):
        return _Assessment(
            "unprobeable",
            "parameter_consumption_plan_missing",
            "frozen graph construction plan is missing",
        )
    specs, specs_problem = _parameter_specs(construction)
    if specs_problem:
        return _Assessment(
            "unprobeable", "parameter_plan_unsupported", specs_problem,
        )
    assert specs is not None
    plan_fixture = plan.get("fixture")
    plan_interventions = plan_fixture.get("parameter_interventions") \
        if isinstance(plan_fixture, Mapping) else None
    if not isinstance(plan_interventions, Mapping):
        return _Assessment(
            "unprobeable",
            "parameter_intervention_plan_missing",
            "frozen plan has no graph-parameter intervention declarations",
        )
    if "expected_graph" not in plan_fixture or "expected_graph" not in fixture:
        return _Assessment(
            "unprobeable",
            "expected_graph_missing",
            "fixture has no independent nominal graph",
        )
    nominal, nominal_problem = _graph_view(
        plan_fixture["expected_graph"],
        representation=representation,
        node_count=node_count,
    )
    if nominal_problem:
        return _Assessment(
            "unprobeable", "expected_graph_invalid", nominal_problem,
        )
    assert nominal is not None
    injected_nominal, injected_problem = _graph_view(
        fixture["expected_graph"],
        representation=representation,
        node_count=node_count,
    )
    if injected_problem:
        return _Assessment(
            "unprobeable", "injected_expected_graph_invalid", injected_problem,
        )
    assert injected_nominal is not None
    if injected_nominal.edges != nominal.edges:
        return _Assessment(
            "unprobeable",
            "injected_fixture_drift",
            "injected nominal graph differs from the frozen fixture authority",
        )

    rows: list[dict[str, Any]] = []
    for role, name, nominal_value in specs:
        intervention = plan_interventions.get(name)
        if not isinstance(intervention, Mapping):
            return _Assessment(
                "unprobeable",
                "parameter_intervention_missing",
                f"frozen plan has no independent intervention for {name}",
                {"params_name": name},
            )
        planned_nominal = intervention.get("nominal_value")
        alternate_value = intervention.get("alternate_value")
        variant_value = intervention.get("value")
        if not _values_equal(alternate_value, variant_value):
            return _Assessment(
                "unprobeable",
                "parameter_intervention_alias_disagrees",
                f"frozen intervention aliases disagree for {name}",
                {"params_name": name},
            )
        if not _values_equal(planned_nominal, nominal_value):
            return _Assessment(
                "unprobeable",
                "parameter_intervention_nominal_disagrees",
                f"frozen intervention nominal value disagrees for {name}",
                {"params_name": name},
            )
        parameter_declaration = construction.get(role)
        parameter_binding = parameter_declaration.get("parameter") \
            if isinstance(parameter_declaration, Mapping) else None
        callable_parameter = parameter_binding.get("callable_parameter") \
            if isinstance(parameter_binding, Mapping) else None
        if intervention.get("callable_parameter") != callable_parameter:
            return _Assessment(
                "unprobeable",
                "parameter_intervention_callable_disagrees",
                f"frozen intervention callable binding disagrees for {name}",
                {"params_name": name},
            )
        if intervention.get("expected_relation") != "graph_must_differ":
            return _Assessment(
                "unprobeable",
                "parameter_intervention_relation_unsupported",
                f"frozen intervention relation is unsupported for {name}",
                {"params_name": name},
            )
        if role == "threshold":
            valid_variant = _numeric_value(variant_value)
        else:
            valid_variant = (
                not isinstance(variant_value, bool)
                and isinstance(variant_value, int)
                and variant_value > 0
            )
        if not valid_variant or _values_equal(variant_value, nominal_value):
            return _Assessment(
                "unprobeable",
                "parameter_intervention_value_invalid",
                f"fixture intervention for {name} must use a valid non-nominal value",
                {"params_name": name},
            )
        if "expected_graph" not in intervention:
            return _Assessment(
                "unprobeable",
                "parameter_intervention_oracle_missing",
                f"frozen fixture intervention for {name} has no independent "
                "expected graph",
                {"params_name": name},
            )
        expected, expected_problem = _graph_view(
            intervention["expected_graph"],
            representation=representation,
            node_count=node_count,
        )
        if expected_problem:
            return _Assessment(
                "unprobeable",
                "parameter_intervention_oracle_invalid",
                expected_problem,
                {"params_name": name},
            )
        assert expected is not None
        if expected.duplicate_edges:
            return _Assessment(
                "unprobeable",
                "parameter_intervention_oracle_invalid",
                f"independent expected graph for {name} contains duplicate edges",
                {"params_name": name},
            )
        if expected.edges == nominal.edges:
            return _Assessment(
                "unprobeable",
                "parameter_intervention_nondiscriminating",
                f"fixture intervention for {name} does not change expected topology",
                {"params_name": name},
            )

        capture, callable_key, capture_problem = _call_constructor(
            plan,
            callables,
            fixture,
            node_count=node_count,
            parameter_overrides={name: variant_value},
        )
        if capture_problem:
            return _Assessment(
                "unprobeable",
                "parameter_intervention_execution_failed",
                capture_problem,
                {"params_name": name, "construction_callable": callable_key},
            )
        assert capture is not None
        runtime_values = capture.get("parameter_values")
        assert isinstance(runtime_values, Mapping)
        runtime_present = name in runtime_values
        runtime_value = runtime_values.get(name)
        actual, actual_problem = _graph_view(
            capture.get("graph"),
            representation=representation,
            node_count=node_count,
        )
        row = {
            "role": role,
            "params_name": name,
            "callable_parameter": callable_parameter,
            "nominal_value": nominal_value,
            "intervention_value": variant_value,
            "runtime_value": runtime_value if runtime_present else None,
            "expected_edge_count": len(expected.edges),
            "constructed_edge_count": len(actual.edges) if actual else None,
            "expected_nominal_edge_symmetric_difference": len(
                expected.edges.symmetric_difference(nominal.edges)
            ),
            "observed_nominal_edge_symmetric_difference": (
                len(actual.edges.symmetric_difference(nominal.edges))
                if actual is not None else None
            ),
        }
        rows.append(row)
        problems: list[str] = []
        if not runtime_present:
            problems.append("runtime construction capture is missing")
        elif not _values_equal(runtime_value, variant_value):
            problems.append("runtime construction capture ignores the intervention")
        if actual_problem:
            problems.append(actual_problem)
        elif actual is not None:
            if actual.duplicate_edges:
                problems.append("constructed graph contains duplicate edges")
            if actual.edges != expected.edges:
                missing = sorted(expected.edges - actual.edges)
                unexpected = sorted(actual.edges - expected.edges)
                problems.append(
                    f"intervened topology differs: missing={missing!r}, "
                    f"unexpected={unexpected!r}"
                )
        if problems:
            row["problems"] = problems
            return _Assessment(
                "fail",
                "graph_parameter_not_consumed",
                "graph construction does not track an independently expected "
                "parameter intervention",
                {"parameter_interventions": rows},
            )

    nominal_replay, callable_key, replay_problem = _call_constructor(
        plan,
        callables,
        fixture,
        node_count=node_count,
    )
    if replay_problem:
        return _Assessment(
            "unprobeable",
            "construction_nominal_replay_failed",
            replay_problem,
            {"construction_callable": callable_key},
        )
    assert nominal_replay is not None
    replay_difference = _constructor_capture_difference(
        nominal_capture,
        nominal_replay,
        representation=str(representation),
        node_count=node_count,
    )
    if replay_difference:
        return _Assessment(
            "unprobeable",
            "construction_nominal_replay_disagrees",
            replay_difference,
            {"construction_callable": callable_key},
        )

    return _Assessment(
        "pass",
        "graph_parameter_consumption_confirmed",
        "each bound graph parameter changes construction as independently expected",
        {"parameter_interventions": rows},
    )


def _assess_parameter_agreement(
    plan: Mapping[str, Any], capture: Mapping[str, Any],
) -> _Assessment:
    construction = plan.get("construction")
    authority = plan.get("parameter_authority")
    if not isinstance(construction, Mapping) or not isinstance(authority, Mapping):
        return _Assessment(
            "unprobeable", "parameter_authority_missing",
            "frozen plan lacks construction parameter authority",
        )
    disposition = _parameter_authority_disposition(plan)
    if disposition is not None:
        return disposition
    bindings = authority.get("bindings")
    if not isinstance(bindings, Mapping):
        return _Assessment(
            "unprobeable", "parameter_bindings_missing",
            "frozen parameter authority has no exact bindings",
        )
    specs, problem = _parameter_specs(construction)
    if problem:
        return _Assessment(
            "unprobeable", "parameter_plan_unsupported", problem,
        )
    assert specs is not None
    runtime_values = capture.get("parameter_values")
    assert isinstance(runtime_values, Mapping)
    comparisons: list[dict[str, Any]] = []
    problems: list[str] = []
    for role, name, planned_value in specs:
        binding = bindings.get(name)
        if not isinstance(binding, Mapping):
            problems.append(f"{name}: exact authority binding is missing")
            continue
        carrier_value = binding.get("carrier_value")
        params_value = binding.get("params_value")
        paper_value = binding.get("paper_value")
        source = binding.get("source")
        suppressed = binding.get("suppressed")
        duplicates = binding.get("duplicates")
        runtime_present = name in runtime_values
        runtime_value = runtime_values.get(name)
        row = {
            "role": role,
            "params_name": name,
            "carrier_value": carrier_value,
            "params_value": params_value,
            "paper_value": paper_value,
            "planned_value": planned_value,
            "runtime_value": runtime_value if runtime_present else None,
            "source": source,
            "suppressed": suppressed,
            "duplicates": duplicates,
        }
        comparisons.append(row)
        if suppressed is not False:
            problems.append(f"{name}: carrier is suppressed or unresolved")
        if not isinstance(duplicates, list) or duplicates:
            problems.append(f"{name}: duplicate authority is present")
        if not isinstance(source, str) or not source.strip():
            problems.append(f"{name}: runtime source is missing")
        if paper_value is None:
            if not _values_equal(params_value, carrier_value):
                problems.append(
                    f"{name}: live params value disagrees with carrier truth"
                )
        elif not _values_equal(paper_value, carrier_value):
            problems.append(
                f"{name}: reconciled paper_value disagrees with carrier truth"
            )
        if not _values_equal(planned_value, params_value):
            problems.append(
                f"{name}: construction plan does not use live params authority"
            )
        if not runtime_present:
            problems.append(f"{name}: runtime construction capture is missing")
        elif not _values_equal(runtime_value, params_value):
            problems.append(
                f"{name}: runtime construction capture disagrees with params"
            )
    evidence = {"parameters": comparisons}
    if problems:
        evidence["problems"] = problems
        return _Assessment(
            "fail", "graph_parameter_authority_disagreement",
            "graph parameter values do not agree across carrier, params, "
            "construction plan, and runtime capture",
            evidence,
        )
    return _Assessment(
        "pass", "graph_parameter_authority_agrees",
        "graph parameters agree across carrier, params, construction plan, "
        "and runtime capture",
        evidence,
    )


def _parameter_authority_disposition(
    plan: Mapping[str, Any],
) -> _Assessment | None:
    """Return a closed HG-2 disposition, or None for runtime assessment."""

    construction = plan.get("construction")
    authority = plan.get("parameter_authority")
    if not isinstance(construction, Mapping) or not isinstance(authority, Mapping):
        return _Assessment(
            "unprobeable",
            "parameter_authority_missing",
            "frozen plan lacks construction parameter authority",
        )
    authority_status = authority.get("status")
    if authority_status in ("pass", "ready"):
        return None
    bindings = authority.get("bindings")
    structurally_bound = isinstance(bindings, Mapping) and bool(bindings)
    if authority_status in ("fail", "blocked") and structurally_bound:
        return _Assessment(
            "fail",
            "parameter_authority_disagreement",
            str(authority.get("reason") or "parameter authority disagrees"),
            {
                "authority_status": str(authority_status),
                "authority_reason": authority.get("reason"),
            },
        )
    return _Assessment(
        "unprobeable",
        "parameter_authority_unresolved",
        str(authority.get("reason") or "parameter authority is unresolved"),
        {"authority_status": str(authority_status)},
    )


def _comparison_witness_assessment(
    plan: Mapping[str, Any],
    callables: Mapping[str, Callable[..., object]],
    fixture: Mapping[str, Any],
    *,
    node_count: int,
) -> _Assessment:
    """Check ``>=`` versus ``>`` at an exact float32-stable boundary.

    Ready homogeneous version-one plans carry two distinct entities with
    identical unit one-hot vectors and invoke the real constructor at exactly
    1.0, where cosine similarity is exactly 1.0 in both supported float32
    backends.
    """

    witness = fixture.get("comparison_witness")
    if witness is None:
        return _Assessment(
            "unprobeable",
            "comparison_witness_missing",
            "frozen graph fixture has no exact comparison witness",
        )
    if not isinstance(witness, Mapping):
        return _Assessment(
            "unprobeable",
            "comparison_witness_invalid",
            "frozen comparison witness must be a mapping",
        )
    construction = plan.get("construction")
    representation = plan.get("representation")
    if not isinstance(construction, Mapping) or not isinstance(
        representation, str
    ):
        return _Assessment(
            "unprobeable",
            "comparison_witness_plan_missing",
            "frozen construction plan is missing for the comparison witness",
        )
    threshold = construction.get("threshold")
    threshold_parameter = threshold.get("parameter") \
        if isinstance(threshold, Mapping) else None
    metric = threshold.get("metric") if isinstance(threshold, Mapping) else None
    comparison = threshold.get("comparison") \
        if isinstance(threshold, Mapping) else None
    threshold_name = threshold_parameter.get("params_name") \
        if isinstance(threshold_parameter, Mapping) else None
    if witness.get("metric") != "cosine_similarity" \
            or metric != "cosine_similarity" \
            or comparison not in {"greater_than", "greater_than_or_equal"} \
            or witness.get("comparison") != comparison \
            or not isinstance(threshold_name, str):
        return _Assessment(
            "unprobeable",
            "comparison_witness_declaration_disagrees",
            "comparison witness does not match the frozen cosine declaration",
        )
    boundary = witness.get("threshold_value")
    if not _numeric_value(boundary) or float(boundary) != 1.0:
        return _Assessment(
            "unprobeable",
            "comparison_witness_boundary_invalid",
            "comparison witness must use the exact threshold 1.0",
        )

    features, feature_problem = _numeric_array(
        witness.get("feature_input"), label="comparison witness features"
    )
    if feature_problem:
        return _Assessment(
            "unprobeable",
            "comparison_witness_features_invalid",
            feature_problem,
        )
    assert features is not None
    entity_axis, axis_problem = _axis_index(
        witness.get("entity_axis"),
        features.ndim,
        label="comparison_witness.entity_axis",
    )
    if axis_problem:
        return _Assessment(
            "unprobeable",
            "comparison_witness_features_invalid",
            axis_problem,
        )
    assert entity_axis is not None
    canonical = np.moveaxis(features, entity_axis, 0)
    if canonical.shape != (node_count, node_count):
        return _Assessment(
            "unprobeable",
            "comparison_witness_features_invalid",
            "comparison witness must be a square one-hot entity fixture",
        )
    expected_features = np.zeros((node_count, node_count), dtype=float)
    expected_features[0, 0] = 1.0
    expected_features[1, 0] = 1.0
    for index in range(2, node_count):
        expected_features[index, index - 1] = 1.0
    if not np.array_equal(canonical, expected_features):
        return _Assessment(
            "unprobeable",
            "comparison_witness_features_invalid",
            "comparison witness must contain one identical unit-vector pair "
            "and orthogonal unit controls",
        )

    expected, expected_problem = _graph_view(
        witness.get("expected_graph"),
        representation=representation,
        node_count=node_count,
    )
    if expected_problem:
        return _Assessment(
            "unprobeable",
            "comparison_witness_oracle_invalid",
            expected_problem,
        )
    assert expected is not None
    oracle_edges: set[tuple[int, int]] = set()
    if comparison == "greater_than_or_equal":
        oracle_edges.update(((0, 1), (1, 0)))
    if construction.get("self_loop_policy") == "required":
        oracle_edges.update((index, index) for index in range(node_count))
    if expected.duplicate_edges or expected.edges != oracle_edges:
        return _Assessment(
            "unprobeable",
            "comparison_witness_oracle_invalid",
            "comparison witness graph does not match its exact boundary oracle",
        )

    root = construction.get("feature_input_root")
    if not isinstance(root, str) or not root.strip():
        return _Assessment(
            "unprobeable",
            "comparison_witness_plan_missing",
            "frozen construction feature root is missing",
        )
    witness_fixture = dict(fixture)
    witness_fixture[root] = witness["feature_input"]
    capture, callable_key, capture_problem = _call_constructor(
        plan,
        callables,
        witness_fixture,
        node_count=node_count,
        parameter_overrides={threshold_name: 1.0},
    )
    if capture_problem:
        return _Assessment(
            "unprobeable",
            "comparison_witness_execution_failed",
            capture_problem,
            {"construction_callable": callable_key},
        )
    assert capture is not None
    actual, actual_problem = _graph_view(
        capture.get("graph"),
        representation=representation,
        node_count=node_count,
    )
    runtime_values = capture.get("parameter_values")
    runtime_present = isinstance(runtime_values, Mapping) \
        and threshold_name in runtime_values
    runtime_value = runtime_values.get(threshold_name) \
        if isinstance(runtime_values, Mapping) else None
    row = {
        "metric": "cosine_similarity",
        "comparison": comparison,
        "threshold_value": 1.0,
        "identical_entity_positions": [0, 1],
        "runtime_threshold_value": runtime_value if runtime_present else None,
        "expected_edge_count": len(expected.edges),
        "constructed_edge_count": len(actual.edges) if actual else None,
        "observed_edge_symmetric_difference": (
            len(actual.edges.symmetric_difference(expected.edges))
            if actual is not None else None
        ),
    }
    problems: list[str] = []
    if not runtime_present or not _values_equal(runtime_value, 1.0):
        problems.append("runtime construction capture ignores threshold 1.0")
    if actual_problem:
        problems.append(actual_problem)
    elif actual is not None:
        if actual.duplicate_edges:
            problems.append("constructed comparison graph contains duplicate edges")
        if actual.edges != expected.edges:
            missing = sorted(expected.edges - actual.edges)
            unexpected = sorted(actual.edges - expected.edges)
            problems.append(
                f"comparison topology differs: missing={missing!r}, "
                f"unexpected={unexpected!r}"
            )
    if problems:
        row["problems"] = problems
        return _Assessment(
            "fail",
            "graph_comparison_semantics_disagree",
            "graph construction does not implement the declared exact "
            "cosine comparison",
            {"comparison_witness": row},
        )
    return _Assessment(
        "pass",
        "graph_comparison_semantics_agree",
        "graph construction implements the declared exact cosine comparison",
        {"comparison_witness": row},
    )


def _assess_construction(
    plan: Mapping[str, Any],
    callables: Mapping[str, Callable[..., object]],
    capture: Mapping[str, Any],
    fixture: Mapping[str, Any],
    *,
    node_count: int,
) -> _Assessment:
    representation = plan.get("representation")
    construction = plan.get("construction")
    if not isinstance(representation, str) or not isinstance(construction, Mapping):
        return _Assessment(
            "unprobeable", "construction_plan_missing",
            "frozen graph representation or construction plan is missing",
        )
    if "expected_graph" not in fixture:
        return _Assessment(
            "unprobeable", "expected_graph_missing",
            "fixture has no independent expected graph",
        )
    expected, expected_problem = _graph_view(
        fixture["expected_graph"],
        representation=representation,
        node_count=node_count,
    )
    if expected_problem:
        return _Assessment(
            "unprobeable", "expected_graph_invalid", expected_problem,
        )
    actual, actual_problem = _graph_view(
        capture.get("graph"),
        representation=representation,
        node_count=node_count,
    )
    if actual_problem:
        return _Assessment(
            "fail", "constructed_graph_invalid", actual_problem,
        )
    assert expected is not None and actual is not None
    problems: list[str] = []
    if actual.duplicate_edges:
        problems.append(
            f"constructed sparse graph has {actual.duplicate_edges} duplicate edges"
        )
    if actual.edges != expected.edges:
        missing = sorted(expected.edges - actual.edges)
        unexpected = sorted(actual.edges - expected.edges)
        problems.append(
            f"constructed topology differs: missing={missing!r}, "
            f"unexpected={unexpected!r}"
        )

    self_loop_policy = construction.get("self_loop_policy")
    loops = {(node, node) for node in range(node_count)}
    if self_loop_policy == "required":
        missing_loops = sorted(loops - actual.edges)
        if missing_loops:
            problems.append(f"required self loops are missing: {missing_loops!r}")
    elif self_loop_policy == "forbidden":
        present_loops = sorted(loops & actual.edges)
        if present_loops:
            problems.append(f"forbidden self loops are present: {present_loops!r}")
    else:
        return _Assessment(
            "unprobeable", "self_loop_policy_unsupported",
            f"unsupported self-loop policy {self_loop_policy!r}",
        )

    direction_policy = construction.get("direction_policy")
    if direction_policy == "undirected_bidirectional":
        one_way = sorted(
            (source, destination)
            for source, destination in actual.edges
            if source != destination and (destination, source) not in actual.edges
        )
        if one_way:
            problems.append(f"undirected graph has one-way edges: {one_way!r}")
    elif direction_policy != "directed":
        return _Assessment(
            "unprobeable", "direction_policy_unsupported",
            f"unsupported graph direction policy {direction_policy!r}",
        )

    cap = construction.get("cap")
    if not isinstance(cap, Mapping):
        return _Assessment(
            "unprobeable", "cap_policy_missing",
            "construction cap policy is missing",
        )
    if cap.get("kind") == "per_source_top_similarity":
        limit = cap.get("value")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            return _Assessment(
                "unprobeable", "cap_value_invalid",
                "per-source graph cap must be a positive integer",
            )
        counts = {
            source: sum(
                1 for edge_source, destination in actual.edges
                if edge_source == source and destination != source
            )
            for source in range(node_count)
        }
        exceeded = {source: count for source, count in counts.items()
                    if count > limit}
        if exceeded:
            problems.append(
                f"per-source neighborhood cap {limit} is exceeded: {exceeded!r}"
            )
    elif cap.get("kind") != "none":
        return _Assessment(
            "unprobeable", "cap_policy_unsupported",
            f"unsupported construction cap kind {cap.get('kind')!r}",
        )

    evidence = {
        "representation": representation,
        "node_count": node_count,
        "expected_edge_count": len(expected.edges),
        "constructed_edge_count": len(actual.edges),
        "observed_edge_symmetric_difference": len(
            actual.edges.symmetric_difference(expected.edges)
        ),
        "self_loop_policy": self_loop_policy,
        "direction_policy": direction_policy,
        "cap_kind": cap.get("kind"),
    }
    if problems:
        evidence["problems"] = problems
        return _Assessment(
            "fail", "graph_construction_semantics_disagree",
            "constructed graph does not follow the frozen topology semantics",
            evidence,
        )
    comparison_assessment = _comparison_witness_assessment(
        plan,
        callables,
        fixture,
        node_count=node_count,
    )
    combined_evidence = {
        **evidence,
        **comparison_assessment.evidence,
    }
    if comparison_assessment.status != "pass":
        return _Assessment(
            comparison_assessment.status,
            comparison_assessment.code,
            comparison_assessment.reason,
            combined_evidence,
        )
    evidence = combined_evidence
    return _Assessment(
        "pass", "graph_construction_semantics_agree",
        "constructed graph matches the independent fixture and declared "
        "topology semantics",
        evidence,
    )


def _execution_context(
    plan: Mapping[str, Any],
    callables: Mapping[str, Callable[..., object]],
    fixture: Mapping[str, Any],
    *,
    node_count: int,
) -> tuple[
    Callable[..., object] | None,
    np.ndarray | None,
    list[object] | None,
    int | None,
    int | None,
    float | None,
    int | None,
    str | None,
]:
    execution = plan.get("execution")
    if not isinstance(execution, Mapping):
        return (None, None, None, None, None, None, None,
                "frozen execution plan is missing")
    adapter, _, problem = _resolve_callable(callables, execution.get("callable"))
    if problem:
        return None, None, None, None, None, None, None, problem
    root = execution.get("neighbor_signal_root")
    if not isinstance(root, str) or not root.strip() or root not in fixture:
        return (None, None, None, None, None, None, None,
                "fixture lacks the exact neighbor-signal root")
    signals, signal_problem = _numeric_array(
        fixture[root], label="neighbor signal fixture"
    )
    if signal_problem:
        return None, None, None, None, None, None, None, signal_problem
    assert signals is not None
    entity_axis, axis_problem = _axis_index(
        execution.get("entity_axis"), signals.ndim, label="entity_axis"
    )
    if axis_problem:
        return None, None, None, None, None, None, None, axis_problem
    assert entity_axis is not None
    if signals.shape[entity_axis] != node_count:
        return (None, None, None, None, None, None, None,
                "neighbor signal entity axis disagrees with stable entity ids")
    output_axis = execution.get("output_entity_axis")
    if isinstance(output_axis, bool) or not isinstance(output_axis, int):
        return (None, None, None, None, None, None, None,
                "output_entity_axis must be an integer")
    tolerance = execution.get("tolerance")
    if not _numeric_value(tolerance) or float(tolerance) < 0.0:
        return (None, None, None, None, None, None, None,
                "execution tolerance must be finite and nonnegative")
    seed = execution.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        return (None, None, None, None, None, None, None,
                "execution seed must be an integer")
    entity_ids = fixture.get("entity_ids")
    assert isinstance(entity_ids, Sequence) and not isinstance(
        entity_ids, (str, bytes)
    )
    return (
        adapter,
        signals,
        list(entity_ids),
        entity_axis,
        output_axis,
        float(tolerance),
        seed,
        None,
    )


def _execute(
    adapter: Callable[..., object],
    *,
    graph: object,
    signals: object,
    entity_ids: list[object],
    seed: int,
    output_axis: int,
    node_count: int,
) -> tuple[np.ndarray | None, str | None]:
    try:
        raw = _call_with_isolated_rng(
            adapter,
            seed=seed,
            kwargs={
                "graph": _clone(graph),
                "neighbor_signal": _clone(signals),
                "entity_ids": copy.deepcopy(entity_ids),
                "seed": seed,
            },
        )
    except Exception as exc:  # noqa: BLE001 - injected generated code is arbitrary
        return None, f"execution adapter raised {type(exc).__name__}: {exc}"
    output, problem = _numeric_array(raw, label="graph mechanism output")
    if problem:
        return None, problem
    assert output is not None
    axis, axis_problem = _axis_index(
        output_axis, output.ndim, label="output_entity_axis"
    )
    if axis_problem:
        return None, axis_problem
    assert axis is not None
    if output.shape[axis] != node_count:
        return None, "graph output entity axis disagrees with stable entity ids"
    return output, None


_EXECUTION_STATE_ISOLATION_FAILURE = "execution adapter state isolation failed"


def _outputs_identical(first: np.ndarray, second: np.ndarray) -> bool:
    """Exact equality for replay authority, including the declared shape."""

    return first.shape == second.shape and np.array_equal(first, second)


def _bracketed_intervention(
    adapter: Callable[..., object],
    *,
    nominal_graph: object,
    nominal_signals: object,
    nominal_entity_ids: list[object],
    baseline_authority: np.ndarray,
    intervention_graph: object,
    intervention_signals: object,
    intervention_entity_ids: list[object],
    seed: int,
    output_axis: int,
    node_count: int,
    label: str,
) -> tuple[np.ndarray | None, str | None]:
    """Replay one intervention between exact nominal-state sentinels.

    Seeding global RNGs cannot reset mutable closure, module, or callable-object
    state.  A stateful adapter can therefore return two equal nominal outputs
    and make later calls look intervention-sensitive solely by call order.  Each
    comparison gets its own nominal-before/intervention/intervention-replay/
    nominal-after bracket.  Both arms must be byte-for-byte numerically stable,
    and both nominal sentinels must equal the original baseline authority.
    """

    calls = (
        (
            "nominal before",
            nominal_graph,
            nominal_signals,
            nominal_entity_ids,
        ),
        (
            "intervention",
            intervention_graph,
            intervention_signals,
            intervention_entity_ids,
        ),
        (
            "intervention replay",
            intervention_graph,
            intervention_signals,
            intervention_entity_ids,
        ),
        (
            "nominal after",
            nominal_graph,
            nominal_signals,
            nominal_entity_ids,
        ),
    )
    outputs: list[np.ndarray] = []
    for phase, graph, signals, entity_ids in calls:
        output, problem = _execute(
            adapter,
            graph=graph,
            signals=signals,
            entity_ids=entity_ids,
            seed=seed,
            output_axis=output_axis,
            node_count=node_count,
        )
        if problem:
            return None, f"{label} {phase} failed: {problem}"
        assert output is not None
        outputs.append(output)

    nominal_before, changed, changed_replay, nominal_after = outputs
    if (
        not _outputs_identical(baseline_authority, nominal_before)
        or not _outputs_identical(baseline_authority, nominal_after)
    ):
        return None, (
            f"{_EXECUTION_STATE_ISOLATION_FAILURE}: exact nominal output "
            f"drifted around the {label} intervention"
        )
    if not _outputs_identical(changed, changed_replay):
        return None, (
            f"{_EXECUTION_STATE_ISOLATION_FAILURE}: the {label} intervention "
            "did not replay exactly"
        )
    return changed, None


def _execution_problem_code(problem: str, fallback: str) -> str:
    if problem.startswith(_EXECUTION_STATE_ISOLATION_FAILURE):
        return "execution_state_isolation_failed"
    return fallback


def _effect_threshold(
    baseline: np.ndarray, changed: np.ndarray, tolerance: float,
) -> float:
    return tolerance * _effect_scale(baseline, changed)


def _effect_scale(baseline: np.ndarray, changed: np.ndarray) -> float:
    return max(
        1.0,
        float(np.max(np.abs(baseline))) if baseline.size else 0.0,
        float(np.max(np.abs(changed))) if changed.size else 0.0,
    )


def _effect_threshold_evidence(
    baseline: np.ndarray,
    changed: np.ndarray,
    tolerance: float,
) -> dict[str, float]:
    baseline_max_abs = (
        float(np.max(np.abs(baseline))) if baseline.size else 0.0
    )
    changed_max_abs = (
        float(np.max(np.abs(changed))) if changed.size else 0.0
    )
    effect_scale = max(1.0, baseline_max_abs, changed_max_abs)
    return {
        "baseline_max_abs": baseline_max_abs,
        "changed_max_abs": changed_max_abs,
        "effect_scale": effect_scale,
        "effect_threshold": tolerance * effect_scale,
    }


def _entity_slice(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    normalized = axis + array.ndim if axis < 0 else axis
    return np.take(array, index, axis=normalized)


def _topology_effect(
    plan: Mapping[str, Any],
    fixture: Mapping[str, Any],
    adapter: Callable[..., object],
    *,
    graph: object,
    signals: np.ndarray,
    entity_ids: list[object],
    baseline: np.ndarray,
    output_axis: int,
    node_count: int,
    tolerance: float,
    seed: int,
    require_edge: bool,
) -> _Assessment:
    representation = str(plan.get("representation"))
    construction = plan.get("construction")
    direction = construction.get("direction_policy") \
        if isinstance(construction, Mapping) else None
    intervention = fixture.get("topology_intervention")
    if not isinstance(intervention, Mapping):
        return _Assessment(
            "unprobeable", "topology_intervention_missing",
            "fixture has no exact topology intervention",
        )
    raw_edges = intervention.get("remove_edges")
    raw_targets = intervention.get("targets")
    if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, (str, bytes)) \
            or not raw_edges:
        return _Assessment(
            "unprobeable", "topology_edges_missing",
            "topology intervention must remove at least one exact edge",
        )
    if not isinstance(raw_targets, Sequence) or isinstance(raw_targets, (str, bytes)) \
            or not raw_targets:
        return _Assessment(
            "unprobeable", "topology_targets_missing",
            "topology intervention must name affected entity positions",
        )
    view, problem = _graph_view(
        graph, representation=representation, node_count=node_count
    )
    if problem:
        return _Assessment("unprobeable", "topology_graph_invalid", problem)
    assert view is not None
    removed: set[tuple[int, int]] = set()
    for raw_edge in raw_edges:
        if not isinstance(raw_edge, Sequence) or isinstance(raw_edge, (str, bytes)) \
                or len(raw_edge) != 2:
            return _Assessment(
                "unprobeable", "topology_edge_invalid",
                "topology intervention edge must contain source and destination",
            )
        source, destination = raw_edge
        if isinstance(source, bool) or not isinstance(source, int) \
                or isinstance(destination, bool) or not isinstance(destination, int) \
                or not 0 <= source < node_count \
                or not 0 <= destination < node_count:
            return _Assessment(
                "unprobeable", "topology_edge_out_of_bounds",
                "topology intervention edge is outside the entity axis",
            )
        removed.add((source, destination))
        if direction == "undirected_bidirectional":
            removed.add((destination, source))
    present = removed & set(view.edges)
    if require_edge and present != removed:
        return _Assessment(
            "unprobeable", "topology_edge_absent",
            "declared topology intervention edge is absent from the graph",
            {"requested_edges": [list(edge) for edge in sorted(removed)]},
        )
    altered_edges = set(view.edges) - removed
    altered = _graph_from_edges(graph, view=view, edges=altered_edges)
    changed, execution_problem = _bracketed_intervention(
        adapter,
        nominal_graph=graph,
        nominal_signals=signals,
        nominal_entity_ids=entity_ids,
        baseline_authority=baseline,
        intervention_graph=altered,
        intervention_signals=signals,
        intervention_entity_ids=entity_ids,
        seed=seed,
        output_axis=output_axis,
        node_count=node_count,
        label="topology",
    )
    if execution_problem:
        return _Assessment(
            "unprobeable",
            _execution_problem_code(
                execution_problem, "topology_execution_failed"
            ),
            execution_problem,
        )
    assert changed is not None
    if baseline.shape != changed.shape:
        return _Assessment(
            "fail",
            "topology_output_shape_changed",
            "topology intervention changes the declared output shape",
            {
                "baseline_shape": list(baseline.shape),
                "changed_shape": list(changed.shape),
            },
        )
    targets: list[int] = []
    for target in raw_targets:
        if isinstance(target, bool) or not isinstance(target, int) \
                or not 0 <= target < node_count:
            return _Assessment(
                "unprobeable", "topology_target_out_of_bounds",
                "topology target is outside the entity axis",
            )
        targets.append(target)
    deltas = [
        float(np.max(np.abs(
            _entity_slice(baseline, output_axis, target)
            - _entity_slice(changed, output_axis, target)
        )))
        for target in targets
    ]
    maximum = max(deltas)
    threshold_evidence = _effect_threshold_evidence(
        baseline, changed, tolerance
    )
    threshold = threshold_evidence["effect_threshold"]
    evidence = {
        "seed": seed,
        "tolerance": tolerance,
        **threshold_evidence,
        "removed_edges": [list(edge) for edge in sorted(removed)],
        "target_entities": targets,
        "target_max_abs_deltas": deltas,
        "maximum_target_delta": maximum,
    }
    if maximum <= threshold:
        return _Assessment(
            "fail", "topology_response_absent",
            "declared output does not respond to the exact topology intervention",
            evidence,
        )
    return _Assessment(
        "pass", "topology_response_present",
        "declared output responds to the exact topology intervention",
        evidence,
    )


def _neighbor_effect(
    plan: Mapping[str, Any],
    fixture: Mapping[str, Any],
    adapter: Callable[..., object],
    *,
    graph: object,
    signals: np.ndarray,
    entity_ids: list[object],
    baseline: np.ndarray,
    entity_axis: int,
    output_axis: int,
    node_count: int,
    tolerance: float,
    seed: int,
    require_edge: bool,
) -> _Assessment:
    intervention = fixture.get("neighbor_intervention")
    if not isinstance(intervention, Mapping):
        return _Assessment(
            "unprobeable", "neighbor_intervention_missing",
            "fixture has no exact neighbor-only intervention",
        )
    source = intervention.get("source")
    target = intervention.get("target")
    delta = intervention.get("delta")
    controls = intervention.get("control_targets")
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in (source, target)) \
            or not 0 <= int(source) < node_count \
            or not 0 <= int(target) < node_count \
            or source == target:
        return _Assessment(
            "unprobeable", "neighbor_pair_invalid",
            "neighbor intervention requires distinct in-bounds source and target",
        )
    assert isinstance(source, int) and isinstance(target, int)
    if not _numeric_value(delta) or float(delta) == 0.0:
        return _Assessment(
            "unprobeable", "neighbor_delta_invalid",
            "neighbor intervention delta must be finite and nonzero",
        )
    if not isinstance(controls, Sequence) or isinstance(controls, (str, bytes)) \
            or not controls:
        return _Assessment(
            "unprobeable", "neighbor_controls_missing",
            "neighbor intervention requires at least one unaffected control target",
        )
    control_targets: list[int] = []
    for control in controls:
        if isinstance(control, bool) or not isinstance(control, int) \
                or not 0 <= control < node_count \
                or control in (source, target):
            return _Assessment(
                "unprobeable", "neighbor_control_invalid",
                "neighbor control target must be distinct and in bounds",
            )
        control_targets.append(control)

    representation = str(plan.get("representation"))
    view, graph_problem = _graph_view(
        graph, representation=representation, node_count=node_count
    )
    if graph_problem:
        return _Assessment(
            "unprobeable", "neighbor_graph_invalid", graph_problem,
        )
    assert view is not None
    if require_edge and (source, target) not in view.edges:
        return _Assessment(
            "unprobeable", "neighbor_edge_absent",
            "declared neighbor source does not connect to the target",
            {"source": source, "target": target},
        )

    reachable = {source}
    frontier = [source]
    while frontier:
        current = frontier.pop()
        for edge_source, destination in view.edges:
            if edge_source == current and destination not in reachable:
                reachable.add(destination)
                frontier.append(destination)
    reachable_controls = sorted(set(control_targets).intersection(reachable))
    if reachable_controls:
        return _Assessment(
            "unprobeable",
            "neighbor_controls_reachable",
            "neighbor controls must be unreachable from the perturbed source",
            {
                "source_entity": source,
                "reachable_control_targets": reachable_controls,
                "reachable_entities": sorted(reachable),
            },
        )

    perturbed = signals.copy()
    source_slice = [slice(None)] * perturbed.ndim
    source_slice[entity_axis] = source
    before = perturbed[tuple(source_slice)].copy()
    perturbed[tuple(source_slice)] = before + float(delta)
    if not np.isfinite(perturbed).all() or np.array_equal(
        before, perturbed[tuple(source_slice)]
    ):
        return _Assessment(
            "unprobeable", "neighbor_intervention_unrepresentable",
            "neighbor-only signal intervention could not be represented",
        )

    changed, execution_problem = _bracketed_intervention(
        adapter,
        nominal_graph=graph,
        nominal_signals=signals,
        nominal_entity_ids=entity_ids,
        baseline_authority=baseline,
        intervention_graph=graph,
        intervention_signals=perturbed,
        intervention_entity_ids=entity_ids,
        seed=seed,
        output_axis=output_axis,
        node_count=node_count,
        label="neighbor-signal",
    )
    if execution_problem:
        return _Assessment(
            "unprobeable",
            _execution_problem_code(
                execution_problem, "neighbor_execution_failed"
            ),
            execution_problem,
        )
    assert changed is not None
    if baseline.shape != changed.shape:
        return _Assessment(
            "fail",
            "neighbor_output_shape_changed",
            "neighbor-only intervention changes the declared output shape",
            {
                "baseline_shape": list(baseline.shape),
                "changed_shape": list(changed.shape),
            },
        )
    target_delta = float(np.max(np.abs(
        _entity_slice(baseline, output_axis, target)
        - _entity_slice(changed, output_axis, target)
    )))
    threshold_evidence = _effect_threshold_evidence(
        baseline, changed, tolerance
    )
    threshold = threshold_evidence["effect_threshold"]
    control_deltas = {
        str(control): float(np.max(np.abs(
            _entity_slice(baseline, output_axis, control)
            - _entity_slice(changed, output_axis, control)
        )))
        for control in control_targets
    }
    evidence = {
        "seed": seed,
        "tolerance": tolerance,
        **threshold_evidence,
        "source_entity": source,
        "target_entity": target,
        "signal_delta": float(delta),
        "target_max_abs_delta": target_delta,
        "control_max_abs_deltas": control_deltas,
        "control_deltas_are_descriptive": True,
    }
    leaking_controls = {
        control: value
        for control, value in control_deltas.items()
        if value > threshold
    }
    if leaking_controls:
        evidence["leaking_control_deltas"] = leaking_controls
        return _Assessment(
            "fail",
            "neighbor_response_leaks_to_unreachable_controls",
            "neighbor-only intervention changes a graph-unreachable control",
            evidence,
        )
    if target_delta <= threshold:
        return _Assessment(
            "fail", "neighbor_response_absent",
            "connected neighbor signal does not affect the declared target output",
            evidence,
        )
    return _Assessment(
        "pass", "neighbor_response_present",
        "connected neighbor signal affects the declared target output",
        evidence,
    )


def _permuted_graph(
    graph: object,
    *,
    view: _GraphView,
    permutation: list[int],
) -> object:
    if view.representation == "dense_adjacency":
        array = np.asarray(_clone(graph))
        order = np.asarray(permutation, dtype=np.int64)
        return np.take(np.take(array, order, axis=0), order, axis=1)
    inverse = np.empty(len(permutation), dtype=np.int64)
    inverse[np.asarray(permutation, dtype=np.int64)] = np.arange(len(permutation))
    edges = {
        (int(inverse[source]), int(inverse[destination]))
        for source, destination in view.edges
    }
    return _graph_from_edges(graph, view=view, edges=edges)


def _permutation_assessment(
    plan: Mapping[str, Any],
    fixture: Mapping[str, Any],
    adapter: Callable[..., object],
    *,
    graph: object,
    signals: np.ndarray,
    entity_ids: list[object],
    baseline: np.ndarray,
    entity_axis: int,
    output_axis: int,
    node_count: int,
    tolerance: float,
    seed: int,
) -> _Assessment:
    execution = plan.get("execution")
    applicability = execution.get("permutation_applicability") \
        if isinstance(execution, Mapping) else None
    if applicability == "not_applicable":
        return _Assessment(
            "not_applicable", "permutation_not_declared",
            "the graph contract does not declare entity relabeling equivalence",
        )
    if applicability != "required":
        return _Assessment(
            "unprobeable", "permutation_applicability_unresolved",
            "entity relabeling applicability is unresolved",
        )
    raw = fixture.get("permutation")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return _Assessment(
            "unprobeable", "permutation_missing",
            "fixture has no coherent entity permutation",
        )
    permutation = list(raw)
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in permutation) \
            or sorted(permutation) != list(range(node_count)) \
            or permutation == list(range(node_count)):
        return _Assessment(
            "unprobeable", "permutation_invalid",
            "fixture permutation must be a non-identity bijection",
        )
    representation = str(plan.get("representation"))
    view, graph_problem = _graph_view(
        graph, representation=representation, node_count=node_count
    )
    if graph_problem:
        return _Assessment(
            "unprobeable", "permutation_graph_invalid", graph_problem,
        )
    assert view is not None
    permuted_graph = _permuted_graph(
        graph, view=view, permutation=permutation
    )
    permuted_signals = np.take(signals, permutation, axis=entity_axis)
    permuted_ids = [entity_ids[index] for index in permutation]
    permuted_output, execution_problem = _bracketed_intervention(
        adapter,
        nominal_graph=graph,
        nominal_signals=signals,
        nominal_entity_ids=entity_ids,
        baseline_authority=baseline,
        intervention_graph=permuted_graph,
        intervention_signals=permuted_signals,
        intervention_entity_ids=permuted_ids,
        seed=seed,
        output_axis=output_axis,
        node_count=node_count,
        label="permutation",
    )
    if execution_problem:
        return _Assessment(
            "unprobeable",
            _execution_problem_code(
                execution_problem, "permutation_execution_failed"
            ),
            execution_problem,
        )
    assert permuted_output is not None
    normalized_output_axis = output_axis + permuted_output.ndim \
        if output_axis < 0 else output_axis
    inverse = np.empty(node_count, dtype=np.int64)
    inverse[np.asarray(permutation, dtype=np.int64)] = np.arange(node_count)
    restored = np.take(permuted_output, inverse, axis=normalized_output_axis)
    if baseline.shape != restored.shape:
        return _Assessment(
            "fail",
            "permutation_output_shape_changed",
            "coherent entity relabeling changes the canonical output shape",
            {
                "seed": seed,
                "permutation_new_to_old": permutation,
                "baseline_shape": list(baseline.shape),
                "restored_shape": list(restored.shape),
            },
        )
    maximum = float(np.max(np.abs(baseline - restored)))
    threshold_evidence = _effect_threshold_evidence(
        baseline, restored, tolerance
    )
    threshold = threshold_evidence["effect_threshold"]
    evidence = {
        "seed": seed,
        "tolerance": tolerance,
        **threshold_evidence,
        "permutation_new_to_old": permutation,
        "restored_max_abs_delta": maximum,
    }
    if maximum > threshold:
        return _Assessment(
            "fail", "permutation_equivalence_violated",
            "coherent entity relabeling changes canonicalized graph behavior",
            evidence,
        )
    return _Assessment(
        "pass", "permutation_equivalence_preserved",
        "coherent entity relabeling preserves canonicalized graph behavior",
        evidence,
    )


def _ablation_graph(
    kind: str,
    graph: object,
    *,
    view: _GraphView,
    fixture: Mapping[str, Any],
) -> tuple[object | None, str | None]:
    if kind == "empty_graph":
        return _graph_from_edges(graph, view=view, edges=set()), None
    if kind == "identity_graph":
        loops = {(node, node) for node in range(view.node_count)}
        return _graph_from_edges(graph, view=view, edges=loops), None
    if kind == "permuted_graph":
        raw = fixture.get("permutation")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            return None, "permuted-graph ablation requires a permutation"
        permutation = list(raw)
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in permutation) \
                or sorted(permutation) != list(range(view.node_count)) \
                or permutation == list(range(view.node_count)):
            return None, "permuted-graph ablation requires a non-identity bijection"
        return _permuted_graph(
            graph, view=view, permutation=permutation
        ), None
    return None, f"unsupported input ablation {kind!r}"


def _contribution_assessment(
    plan: Mapping[str, Any],
    callables: Mapping[str, Callable[..., object]],
    fixture: Mapping[str, Any],
    real_assessments: Mapping[str, _Assessment],
    real_adapter: Callable[..., object],
    *,
    graph: object,
    signals: np.ndarray,
    entity_ids: list[object],
    entity_axis: int,
    output_axis: int,
    node_count: int,
    tolerance: float,
    seed: int,
) -> _Assessment:
    ablation = plan.get("ablation")
    if not isinstance(ablation, Mapping):
        return _Assessment(
            "unprobeable", "contribution_ablation_missing",
            "frozen plan has no paper-justified contribution ablation",
        )
    kind = ablation.get("kind")
    if not isinstance(kind, str):
        return _Assessment(
            "unprobeable", "contribution_ablation_kind_missing",
            "contribution ablation has no discriminated kind",
        )
    criterion = ablation.get("discriminating_probe_ref")
    if criterion not in (PROBE_REFS["HG-4"], PROBE_REFS["HG-5"]):
        return _Assessment(
            "unprobeable", "contribution_criterion_unsupported",
            "contribution ablation names an unsupported discriminating probe",
            {"discriminating_probe_ref": str(criterion)},
        )
    real = real_assessments.get(str(criterion))
    if real is None:
        return _Assessment(
            "unprobeable", "real_discriminating_result_missing",
            "real mechanism has no result for the declared discriminating probe",
        )
    if real.status == "unprobeable":
        return _Assessment(
            "unprobeable", "real_discriminating_result_unprobeable",
            "real mechanism could not execute the contribution discriminator",
            {
                "discriminating_probe_ref": criterion,
                "real_status": real.status,
                "real_reason": real.reason,
            },
        )
    if real.status != "pass":
        return _Assessment(
            "fail", "real_mechanism_not_discriminating",
            "real graph mechanism does not pass the declared discriminating check",
            {
                "discriminating_probe_ref": criterion,
                "real_status": real.status,
                "real_reason": real.reason,
            },
        )

    null_adapter = real_adapter
    null_graph = graph
    if kind in _INPUT_ABLATIONS:
        representation = str(plan.get("representation"))
        view, graph_problem = _graph_view(
            graph, representation=representation, node_count=node_count
        )
        if graph_problem:
            return _Assessment(
                "unprobeable", "ablation_graph_invalid", graph_problem,
            )
        assert view is not None
        null_graph, graph_problem = _ablation_graph(
            kind, graph, view=view, fixture=fixture
        )
        if graph_problem:
            return _Assessment(
                "unprobeable", "ablation_graph_unresolved", graph_problem,
            )
    elif kind in _CALLABLE_ABLATIONS:
        null_adapter, _, callable_problem = _resolve_callable(
            callables, ablation.get("callable")
        )
        if callable_problem:
            return _Assessment(
                "unprobeable", "ablation_callable_missing", callable_problem,
            )
    else:
        return _Assessment(
            "unprobeable", "contribution_ablation_unsupported",
            f"unsupported contribution ablation {kind!r}",
        )
    assert null_adapter is not None and null_graph is not None
    null_baseline, execution_problem = _execute(
        null_adapter,
        graph=null_graph,
        signals=signals,
        entity_ids=entity_ids,
        seed=seed,
        output_axis=output_axis,
        node_count=node_count,
    )
    if execution_problem:
        return _Assessment(
            "unprobeable", "ablation_baseline_failed", execution_problem,
        )
    assert null_baseline is not None
    null_replay, execution_problem = _execute(
        null_adapter,
        graph=null_graph,
        signals=signals,
        entity_ids=entity_ids,
        seed=seed,
        output_axis=output_axis,
        node_count=node_count,
    )
    if execution_problem:
        return _Assessment(
            "unprobeable", "ablation_baseline_failed", execution_problem,
        )
    assert null_replay is not None
    if not _outputs_identical(null_baseline, null_replay):
        return _Assessment(
            "unprobeable",
            "execution_state_isolation_failed",
            f"{_EXECUTION_STATE_ISOLATION_FAILURE}: the declared null "
            "baseline did not replay exactly",
        )
    if criterion == PROBE_REFS["HG-4"]:
        null = _topology_effect(
            plan,
            fixture,
            null_adapter,
            graph=null_graph,
            signals=signals,
            entity_ids=entity_ids,
            baseline=null_baseline,
            output_axis=output_axis,
            node_count=node_count,
            tolerance=tolerance,
            seed=seed,
            require_edge=kind in _CALLABLE_ABLATIONS,
        )
    else:
        null = _neighbor_effect(
            plan,
            fixture,
            null_adapter,
            graph=null_graph,
            signals=signals,
            entity_ids=entity_ids,
            baseline=null_baseline,
            entity_axis=entity_axis,
            output_axis=output_axis,
            node_count=node_count,
            tolerance=tolerance,
            seed=seed,
            require_edge=kind in _CALLABLE_ABLATIONS,
        )
    evidence = {
        "seed": seed,
        "tolerance": tolerance,
        "ablation_kind": kind,
        "discriminating_probe_ref": criterion,
        "real_status": real.status,
        "real_evidence": real.evidence,
        "null_status": null.status,
        "null_reason_code": null.code,
        "null_reason": null.reason,
        "null_evidence": null.evidence,
    }
    if null.status == "unprobeable":
        if null.code == "execution_state_isolation_failed":
            return _Assessment(
                "unprobeable",
                "execution_state_isolation_failed",
                null.reason,
                evidence,
            )
        return _Assessment(
            "unprobeable", "null_discriminating_result_unprobeable",
            "declared graph null could not execute the same discriminating check",
            evidence,
        )
    expected_absence_code = (
        "topology_response_absent"
        if criterion == PROBE_REFS["HG-4"]
        else "neighbor_response_absent"
    )
    if null.status == "fail" and null.code != expected_absence_code:
        return _Assessment(
            "unprobeable",
            "null_discriminating_failure_not_absence",
            "declared graph null failed for a reason other than absence of the "
            "claimed graph effect",
            evidence,
        )
    if null.status != "fail":
        return _Assessment(
            "fail", "contribution_ablation_nondiscriminating",
            "real and null graph arms both pass the same discriminating check",
            evidence,
        )
    return _Assessment(
        "pass", "contribution_ablation_discriminates",
        "real graph arm passes and the paper-justified null fails the same "
        "discriminating check",
        evidence,
    )


def _as_result(
    plan: Mapping[str, Any], probe_id: str, assessment: _Assessment,
) -> GraphProbeResult:
    return _result(
        plan,
        probe_id,
        assessment.status,
        assessment.reason,
        reason_code=assessment.code,
        **assessment.evidence,
    )


def run_graph_mechanism_probes(
    plan: Mapping[str, Any] | None,
    callables: Mapping[str, Callable[..., object]],
    fixture: Mapping[str, Any],
) -> list[GraphProbeResult]:
    """Execute HG-2 through HG-7 in prerequisite order.

    Every call returns exactly six ordered results.  A graph-free disposition
    returns six ``not_applicable`` rows without touching injected callables.
    Unsupported plans return named ``unprobeable`` rows.  A structurally bound
    parameter-authority disagreement is an HG-2 failure even when the planner
    marks the overall plan blocked.  Parameter and construction failures block
    every downstream behavioral result from becoming a pass.
    """

    empty_plan: Mapping[str, Any] = {}
    if not isinstance(plan, Mapping):
        return _disposition_results(
            empty_plan, "unprobeable", "frozen graph plan is missing"
        )
    problem = _strict_json_problem(plan)
    if problem:
        return _disposition_results(plan, "unprobeable", problem)
    status = plan.get("status")
    reason = str(plan.get("reason") or "graph plan disposition has no reason")
    if status == "not_applicable":
        return _disposition_results(plan, "not_applicable", reason)
    if status == "unprobeable":
        return _disposition_results(plan, "unprobeable", reason)
    if status not in ("ready", "blocked"):
        return _disposition_results(
            plan,
            "unprobeable",
            f"unknown frozen graph plan status {status!r}",
        )
    blocked_parameter_disposition: _Assessment | None = None
    if status == "blocked":
        blocked_parameter_disposition = _parameter_authority_disposition(plan)
        if blocked_parameter_disposition is None \
                or blocked_parameter_disposition.status != "fail":
            return _disposition_results(plan, "unprobeable", reason)
    if not isinstance(callables, Mapping) or not isinstance(fixture, Mapping):
        return _disposition_results(
            plan, "unprobeable", "callable adapters or graph fixture are missing"
        )
    frozen_fixture = plan.get("fixture")
    if not isinstance(frozen_fixture, Mapping):
        return _disposition_results(
            plan, "unprobeable", "frozen plan has no graph fixture authority"
        )
    frozen_digest, frozen_problem = _fixture_digest(frozen_fixture)
    injected_digest, injected_problem = _fixture_digest(fixture)
    fixture_digest_problem = frozen_problem or injected_problem
    if fixture_digest_problem:
        return _disposition_results(
            plan, "unprobeable", str(fixture_digest_problem)
        )
    if not _fixtures_equivalent(frozen_fixture, fixture):
        return _disposition_results(
            plan,
            "unprobeable",
            "injected graph fixture differs from the frozen fixture authority",
        )
    alignment = plan.get("alignment")
    if not isinstance(alignment, Mapping) or alignment.get("status") != "pass":
        alignment_reason = alignment.get("reason") \
            if isinstance(alignment, Mapping) else None
        return _disposition_results(
            plan,
            "unprobeable",
            str(alignment_reason or "relational alignment prerequisite did not pass"),
        )
    node_count, fixture_problem = _entity_count(fixture)
    if fixture_problem:
        return _disposition_results(plan, "unprobeable", fixture_problem)
    assert node_count is not None
    receipt_problem = _alignment_receipt_problem(plan, fixture)
    if receipt_problem:
        return _disposition_results(plan, "unprobeable", receipt_problem)

    parameter_disposition = _parameter_authority_disposition(plan)
    if parameter_disposition is not None:
        hg2 = _as_result(plan, "HG-2", parameter_disposition)
        return [
            hg2,
            *[
                _result(
                    plan,
                    probe_id,
                    "unprobeable",
                    "parameter-authority prerequisite did not pass",
                    prerequisite="HG-2",
                )
                for probe_id in PROBE_IDS[1:]
            ],
        ]

    capture, construction_callable, capture_problem = _call_constructor(
        plan, callables, fixture, node_count=node_count
    )
    if capture_problem:
        hg2 = _result(
            plan,
            "HG-2",
            "unprobeable",
            capture_problem,
            reason_code="construction_capture_unavailable",
            construction_callable=construction_callable,
        )
        blocked = [hg2]
        blocked.extend(
            _result(
                plan,
                probe_id,
                "unprobeable",
                "parameter and construction prerequisites did not pass",
                prerequisite="HG-2",
            )
            for probe_id in PROBE_IDS[1:]
        )
        return blocked
    assert capture is not None

    parameter = _assess_parameter_agreement(plan, capture)
    if parameter.status == "pass":
        consumption = _assess_parameter_consumption(
            plan,
            callables,
            fixture,
            node_count=node_count,
            nominal_capture=capture,
        )
        if consumption.status == "pass":
            parameter = _Assessment(
                "pass",
                parameter.code,
                parameter.reason,
                {**parameter.evidence, **consumption.evidence},
            )
        else:
            parameter = consumption
    hg2 = _as_result(plan, "HG-2", parameter)
    if parameter.status != "pass":
        blocked = [hg2]
        blocked.extend(
            _result(
                plan,
                probe_id,
                "unprobeable",
                "parameter-authority prerequisite did not pass",
                prerequisite="HG-2",
            )
            for probe_id in PROBE_IDS[1:]
        )
        return blocked

    construction = _assess_construction(
        plan, callables, capture, fixture, node_count=node_count
    )
    hg3 = _as_result(plan, "HG-3", construction)
    if construction.status != "pass":
        blocked = [hg2, hg3]
        blocked.extend(
            _result(
                plan,
                probe_id,
                "unprobeable",
                "graph-construction prerequisite did not pass",
                prerequisite="HG-3",
            )
            for probe_id in PROBE_IDS[2:]
        )
        return blocked

    (
        execution_adapter,
        signals,
        entity_ids,
        entity_axis,
        output_axis,
        tolerance,
        seed,
        execution_problem,
    ) = _execution_context(
        plan, callables, fixture, node_count=node_count
    )
    if execution_problem:
        return [
            hg2,
            hg3,
            *[
                _result(
                    plan,
                    probe_id,
                    "unprobeable",
                    execution_problem,
                    prerequisite="execution_context",
                )
                for probe_id in PROBE_IDS[2:]
            ],
        ]
    assert execution_adapter is not None
    assert signals is not None
    assert entity_ids is not None
    assert entity_axis is not None
    assert output_axis is not None
    assert tolerance is not None
    assert seed is not None
    graph = capture["graph"]
    baseline, baseline_problem = _execute(
        execution_adapter,
        graph=graph,
        signals=signals,
        entity_ids=entity_ids,
        seed=seed,
        output_axis=output_axis,
        node_count=node_count,
    )
    replay, replay_problem = _execute(
        execution_adapter,
        graph=graph,
        signals=signals,
        entity_ids=entity_ids,
        seed=seed,
        output_axis=output_axis,
        node_count=node_count,
    )
    if baseline_problem or replay_problem:
        problem = baseline_problem or replay_problem
        return [
            hg2,
            hg3,
            *[
                _result(
                    plan,
                    probe_id,
                    "unprobeable",
                    str(problem),
                    prerequisite="same_seed_baseline",
                )
                for probe_id in PROBE_IDS[2:]
            ],
        ]
    assert baseline is not None and replay is not None
    repeat_threshold = _effect_threshold(baseline, replay, tolerance)
    repeat_delta = (
        float(np.max(np.abs(baseline - replay)))
        if baseline.shape == replay.shape
        else None
    )
    if baseline.shape != replay.shape or not np.array_equal(baseline, replay):
        return [
            hg2,
            hg3,
            *[
                _result(
                    plan,
                    probe_id,
                    "unprobeable",
                    "same-seed graph execution is not repeatable",
                    repeat_max_abs_delta=repeat_delta,
                    repeat_effect_threshold=repeat_threshold,
                )
                for probe_id in PROBE_IDS[2:]
            ],
        ]

    topology = _topology_effect(
        plan,
        fixture,
        execution_adapter,
        graph=graph,
        signals=signals,
        entity_ids=entity_ids,
        baseline=baseline,
        output_axis=output_axis,
        node_count=node_count,
        tolerance=tolerance,
        seed=seed,
        require_edge=True,
    )
    if topology.code == "execution_state_isolation_failed":
        return [
            hg2,
            hg3,
            *[
                _result(
                    plan,
                    probe_id,
                    "unprobeable",
                    topology.reason,
                    reason_code="execution_state_isolation_failed",
                    prerequisite="execution_state_isolation",
                )
                for probe_id in PROBE_IDS[2:]
            ],
        ]
    neighbor = _neighbor_effect(
        plan,
        fixture,
        execution_adapter,
        graph=graph,
        signals=signals,
        entity_ids=entity_ids,
        baseline=baseline,
        entity_axis=entity_axis,
        output_axis=output_axis,
        node_count=node_count,
        tolerance=tolerance,
        seed=seed,
        require_edge=True,
    )
    if neighbor.code == "execution_state_isolation_failed":
        return [
            hg2,
            hg3,
            *[
                _result(
                    plan,
                    probe_id,
                    "unprobeable",
                    neighbor.reason,
                    reason_code="execution_state_isolation_failed",
                    prerequisite="execution_state_isolation",
                )
                for probe_id in PROBE_IDS[2:]
            ],
        ]
    permutation = _permutation_assessment(
        plan,
        fixture,
        execution_adapter,
        graph=graph,
        signals=signals,
        entity_ids=entity_ids,
        baseline=baseline,
        entity_axis=entity_axis,
        output_axis=output_axis,
        node_count=node_count,
        tolerance=tolerance,
        seed=seed,
    )
    if permutation.code == "execution_state_isolation_failed":
        return [
            hg2,
            hg3,
            *[
                _result(
                    plan,
                    probe_id,
                    "unprobeable",
                    permutation.reason,
                    reason_code="execution_state_isolation_failed",
                    prerequisite="execution_state_isolation",
                )
                for probe_id in PROBE_IDS[2:]
            ],
        ]
    real_assessments = {
        PROBE_REFS["HG-4"]: topology,
        PROBE_REFS["HG-5"]: neighbor,
    }
    contribution = _contribution_assessment(
        plan,
        callables,
        fixture,
        real_assessments,
        execution_adapter,
        graph=graph,
        signals=signals,
        entity_ids=entity_ids,
        entity_axis=entity_axis,
        output_axis=output_axis,
        node_count=node_count,
        tolerance=tolerance,
        seed=seed,
    )
    ablation = plan.get("ablation")
    if contribution.code == "execution_state_isolation_failed" \
            and isinstance(ablation, Mapping) \
            and ablation.get("kind") in _INPUT_ABLATIONS:
        return [
            hg2,
            hg3,
            *[
                _result(
                    plan,
                    probe_id,
                    "unprobeable",
                    contribution.reason,
                    reason_code="execution_state_isolation_failed",
                    prerequisite="execution_state_isolation",
                )
                for probe_id in PROBE_IDS[2:]
            ],
        ]
    return [
        hg2,
        hg3,
        _as_result(plan, "HG-4", topology),
        _as_result(plan, "HG-5", neighbor),
        _as_result(plan, "HG-6", permutation),
        _as_result(plan, "HG-7", contribution),
    ]


__all__ = [
    "GraphProbeResult",
    "PROBE_IDS",
    "PROBE_REFS",
    "execution_plan_digest",
    "fixture_digest",
    "graph_plan_trace",
    "run_graph_mechanism_probes",
]
