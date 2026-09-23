"""Freeze one exact homogeneous-graph mechanism execution plan.

The planner consumes plain method-spec, params, and schema-2 architecture
contract mappings.  R2C-084 remains the sole relational identity authority:
this module reuses the forecasting runtime normalizer and copies only its
resolved alignment facts.  R2C-083 remains the parameter authority: exact
carrier identities and value equivalence reuse the derivation helpers rather
than introducing a second alias or numeric matching rule.

Version one supports one homogeneous graph represented as a sparse edge index
or dense adjacency.  Graph-free methods are not applicable.  Explicitly
unsupported relational forms remain unprobeable rather than being coerced
into the homogeneous grammar.
"""

from __future__ import annotations

import json
import keyword
import math
from collections.abc import Mapping
from typing import Any

from scripts.arch_contract_runtime_plan import (
    RuntimePlanError,
    normalize_schema2_forecasting_plan,
)
from scripts.build_plan import load_build_plan
from scripts.derive_params import _glossary_param_name, _values_equivalent
from schemas.graph_wiring import mechanism_marker_element_ids


PLAN_SCHEMA_VERSION = "1.0"
HARNESS_SEED = 1729
HARNESS_TOLERANCE = 1.0e-8

ALIGNMENT_PROBE_REF = "graph_mechanism.alignment_prerequisite"
CANONICAL_PROBE_REFS = {
    "parameter_agreement": "graph_mechanism.parameter_agreement",
    "construction": "graph_mechanism.construction_semantics",
    "topology": "graph_mechanism.topology_sensitivity",
    "neighbor_signal": "graph_mechanism.neighbor_sensitivity",
    "permutation": "graph_mechanism.permutation_equivalence",
    "contribution_ablation": "graph_mechanism.contribution_ablation",
}


class GraphMechanismRuntimePlanError(ValueError):
    """The three authorities cannot produce an exact frozen graph plan."""


def _mapping(value: Any, *, root: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GraphMechanismRuntimePlanError(f"{root} must be a mapping")
    return value


def _exact_text(value: Any, *, root: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise GraphMechanismRuntimePlanError(
            f"{root} must be non-blank and already stripped"
        )
    return value


def _python_parameter(value: Any, *, root: str) -> str:
    name = _exact_text(value, root=root)
    if not name.isidentifier() or keyword.iskeyword(name):
        raise GraphMechanismRuntimePlanError(
            f"{root} must be one exact Python parameter identifier"
        )
    return name


def _json_copy(value: Any, *, root: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise GraphMechanismRuntimePlanError(
            f"{root} must contain JSON-only finite values: {exc}"
        ) from exc


def _base_plan(
    *,
    status: str,
    reason: str | None,
    groundings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": status,
        "reason": reason,
        "representation": None,
        "construction": None,
        "parameter_authority": {
            "status": "not_applicable" if status == "not_applicable" else status,
            "reason": reason,
            "bindings": {},
        },
        "execution": None,
        "ablation": None,
        "groundings": _json_copy(
            dict(groundings or {}), root="graph mechanism groundings"
        ),
        "alignment": None,
        "fixture": None,
    }


def _methodology(
    method_spec: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, list[Mapping[str, Any]]]:
    raw = method_spec.get("methodology_replication_contract")
    if raw is None:
        return None, []
    contract = _mapping(raw, root="methodology_replication_contract")
    elements = contract.get("elements")
    if not isinstance(elements, list):
        raise GraphMechanismRuntimePlanError(
            "methodology_replication_contract.elements must be a list"
        )
    normalized: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for index, raw_element in enumerate(elements):
        element = _mapping(
            raw_element,
            root=f"methodology_replication_contract.elements[{index}]",
        )
        element_id = _exact_text(
            element.get("element_id"),
            root=f"methodology_replication_contract.elements[{index}].element_id",
        )
        if element_id in seen:
            raise GraphMechanismRuntimePlanError(
                "methodology_replication_contract contains duplicate element "
                f"id {element_id!r}"
            )
        seen.add(element_id)
        normalized.append(element)
    return contract, normalized


def _relational_disposition(
    elements: list[Mapping[str, Any]],
) -> tuple[list[str], list[dict[str, str]]]:
    homogeneous: list[str] = []
    unsupported: list[dict[str, str]] = []
    for element in elements:
        raw_marker = element.get("relational_structure")
        if raw_marker is None:
            continue
        marker = _mapping(
            raw_marker,
            root=(
                "methodology element "
                f"{element.get('element_id')!r}.relational_structure"
            ),
        )
        kind = marker.get("kind")
        if kind == "homogeneous_graph":
            homogeneous.append(str(element["element_id"]))
        elif kind == "unsupported":
            unsupported_kind = _exact_text(
                marker.get("unsupported_kind"),
                root=(
                    "methodology element "
                    f"{element.get('element_id')!r}.unsupported_kind"
                ),
            )
            unsupported.append(
                {
                    "element_id": str(element["element_id"]),
                    "unsupported_kind": unsupported_kind,
                }
            )
        else:
            raise GraphMechanismRuntimePlanError(
                f"unsupported relational_structure.kind {kind!r}"
            )
    return homogeneous, unsupported


def _element_index(
    elements: list[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {str(element["element_id"]): element for element in elements}


def _grounding(
    *,
    probe_ref: str,
    element: Mapping[str, Any],
    bound_callables: list[str],
) -> dict[str, Any]:
    # Graph-ref ownership is derived from the mechanism block itself
    # (R2C-092, schemas/graph_wiring.py): the element this grounding names
    # owns the ref by construction, so the stored JSON's hand-transcribed
    # verification_probe_refs carry no authority here.
    paper_ids = element.get("paper_element_ids") or []
    if not isinstance(paper_ids, list) or any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in paper_ids
    ):
        raise GraphMechanismRuntimePlanError(
            f"methodology element {element.get('element_id')!r} has invalid "
            "paper_element_ids"
        )
    return {
        "probe_ref": probe_ref,
        "element_ids": [str(element["element_id"])],
        "paper_element_ids": list(paper_ids),
        "bound_callables": list(bound_callables),
    }


def _normalize_groundings(
    mechanism: Mapping[str, Any],
    elements: list[Mapping[str, Any]],
) -> dict[str, Any]:
    by_id = _element_index(elements)
    construction = _mapping(
        mechanism.get("construction"),
        root="homogeneous_graph_mechanism.construction",
    )
    message_passing = _mapping(
        mechanism.get("message_passing"),
        root="homogeneous_graph_mechanism.message_passing",
    )
    raw_ablation = mechanism.get("contribution_ablation")
    ablation = (
        None
        if raw_ablation is None
        else _mapping(
            raw_ablation,
            root="homogeneous_graph_mechanism.contribution_ablation",
        )
    )
    refs = _mapping(
        mechanism.get("probe_refs"),
        root="homogeneous_graph_mechanism.probe_refs",
    )

    expected = dict(CANONICAL_PROBE_REFS)
    applicability = mechanism.get("permutation_applicability")
    if applicability == "not_applicable":
        expected["permutation"] = None
    elif applicability != "equivariant":
        raise GraphMechanismRuntimePlanError(
            "homogeneous_graph_mechanism.permutation_applicability must be "
            "'equivariant' or 'not_applicable'"
        )
    if ablation is None:
        expected["contribution_ablation"] = None
    if dict(refs) != expected:
        raise GraphMechanismRuntimePlanError(
            "homogeneous_graph_mechanism.probe_refs must equal the canonical "
            f"graph refs; got {dict(refs)!r}"
        )

    construction_callable = _callable(
        construction.get("callable"),
        root="homogeneous_graph_mechanism.construction.callable",
    )
    message_callable = _callable(
        message_passing.get("callable"),
        root="homogeneous_graph_mechanism.message_passing.callable",
    )
    construction_key = (
        f"{construction_callable['module']}:{construction_callable['qualname']}"
    )
    message_key = f"{message_callable['module']}:{message_callable['qualname']}"
    ablation_callables = [message_key]
    raw_ablation_callable = (
        ablation.get("callable") if ablation is not None else None
    )
    if raw_ablation_callable is not None:
        ablation_callable = _callable(
            raw_ablation_callable,
            root="homogeneous_graph_mechanism.contribution_ablation.callable",
        )
        ablation_callables.append(
            f"{ablation_callable['module']}:{ablation_callable['qualname']}"
        )

    owner_ids = {
        "alignment_prerequisite": _exact_text(
            mechanism.get("alignment_element_id"),
            root="homogeneous_graph_mechanism.alignment_element_id",
        ),
        "parameter_agreement": _exact_text(
            construction.get("element_id"),
            root="homogeneous_graph_mechanism.construction.element_id",
        ),
        "construction_semantics": _exact_text(
            construction.get("element_id"),
            root="homogeneous_graph_mechanism.construction.element_id",
        ),
        "topology_sensitivity": _exact_text(
            message_passing.get("element_id"),
            root="homogeneous_graph_mechanism.message_passing.element_id",
        ),
        "neighbor_sensitivity": _exact_text(
            message_passing.get("element_id"),
            root="homogeneous_graph_mechanism.message_passing.element_id",
        ),
    }
    if ablation is not None:
        owner_ids["contribution_ablation"] = _exact_text(
            ablation.get("element_id"),
            root=(
                "homogeneous_graph_mechanism.contribution_ablation.element_id"
            ),
        )
    missing = sorted(set(owner_ids.values()) - set(by_id))
    if missing:
        raise GraphMechanismRuntimePlanError(
            f"homogeneous graph mechanism references unknown elements {missing!r}"
        )

    out = {
        ALIGNMENT_PROBE_REF: _grounding(
            probe_ref=ALIGNMENT_PROBE_REF,
            element=by_id[owner_ids["alignment_prerequisite"]],
            bound_callables=[],
        ),
        CANONICAL_PROBE_REFS["parameter_agreement"]: _grounding(
            probe_ref=CANONICAL_PROBE_REFS["parameter_agreement"],
            element=by_id[owner_ids["parameter_agreement"]],
            bound_callables=[construction_key],
        ),
        CANONICAL_PROBE_REFS["construction"]: _grounding(
            probe_ref=CANONICAL_PROBE_REFS["construction"],
            element=by_id[owner_ids["construction_semantics"]],
            bound_callables=[construction_key],
        ),
        CANONICAL_PROBE_REFS["topology"]: _grounding(
            probe_ref=CANONICAL_PROBE_REFS["topology"],
            element=by_id[owner_ids["topology_sensitivity"]],
            bound_callables=[message_key],
        ),
        CANONICAL_PROBE_REFS["neighbor_signal"]: _grounding(
            probe_ref=CANONICAL_PROBE_REFS["neighbor_signal"],
            element=by_id[owner_ids["neighbor_sensitivity"]],
            bound_callables=[message_key],
        ),
    }
    if ablation is not None:
        out[CANONICAL_PROBE_REFS["contribution_ablation"]] = _grounding(
            probe_ref=CANONICAL_PROBE_REFS["contribution_ablation"],
            element=by_id[owner_ids["contribution_ablation"]],
            bound_callables=ablation_callables,
        )
    if refs.get("permutation") is None:
        pass
    else:
        out[CANONICAL_PROBE_REFS["permutation"]] = _grounding(
            probe_ref=CANONICAL_PROBE_REFS["permutation"],
            element=by_id[owner_ids["topology_sensitivity"]],
            bound_callables=[message_key],
        )
    return out


def _callable(raw: Any, *, root: str) -> dict[str, str]:
    value = _mapping(raw, root=root)
    if set(value) != {"module", "qualname"}:
        raise GraphMechanismRuntimePlanError(
            f"{root} must contain exactly module and qualname"
        )
    module = _exact_text(value.get("module"), root=f"{root}.module")
    qualname = _exact_text(value.get("qualname"), root=f"{root}.qualname")
    if module not in {"method.model", "method.training", "method.method"}:
        raise GraphMechanismRuntimePlanError(
            f"{root}.module is outside the closed generated-package owners"
        )
    if (
        not qualname.isidentifier()
        or keyword.iskeyword(qualname)
        or qualname.startswith("_")
    ):
        raise GraphMechanismRuntimePlanError(
            f"{root}.qualname must name one public top-level helper"
        )
    return {"module": module, "qualname": qualname}


def _carrier_records(method_spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    critical = method_spec.get("critical_requirements") or {}
    critical = _mapping(critical, root="critical_requirements")
    records: list[dict[str, Any]] = []

    lane = critical.get("scale_dependent_hyperparameters") or []
    if not isinstance(lane, list):
        raise GraphMechanismRuntimePlanError(
            "critical_requirements.scale_dependent_hyperparameters must be a list"
        )
    for index, raw_entry in enumerate(lane):
        if not isinstance(raw_entry, Mapping):
            continue
        value = raw_entry.get("paper_value")
        if value is None or isinstance(value, bool):
            continue
        name = str(raw_entry.get("name") or "").strip()
        if name:
            records.append(
                {
                    "root": (
                        "critical_requirements."
                        f"scale_dependent_hyperparameters[{index}]"
                    ),
                    "identities": [name],
                    "paper_value": value,
                }
            )

    glossary = critical.get("param_glossary") or []
    if not isinstance(glossary, list):
        raise GraphMechanismRuntimePlanError(
            "critical_requirements.param_glossary must be a list"
        )
    for index, raw_entry in enumerate(glossary):
        if not isinstance(raw_entry, Mapping):
            continue
        entry = dict(raw_entry)
        value = entry.get("paper_value")
        if value is None or isinstance(value, bool):
            continue
        if _glossary_param_name(entry) is None:
            continue
        identities: list[str] = []
        for raw_name in (entry.get("name"), *(entry.get("aliases") or [])):
            name = str(raw_name or "").strip()
            if name and name not in identities:
                identities.append(name)
        records.append(
            {
                "root": f"critical_requirements.param_glossary[{index}]",
                "identities": identities,
                "paper_value": value,
            }
        )
    return records


def _params_entries(params: Mapping[str, Any]) -> Mapping[str, Any]:
    entries = params.get("params")
    return _mapping(entries, root="params.params")


def _parameter_binding(
    *,
    params_name: str,
    records: list[dict[str, Any]],
    entries: Mapping[str, Any],
) -> tuple[dict[str, Any], str | None]:
    matched = [record for record in records if params_name in record["identities"]]
    duplicates = [record["root"] for record in matched] if len(matched) > 1 else []
    raw_entry = entries.get(params_name)
    entry = raw_entry if isinstance(raw_entry, Mapping) else {}
    binding = {
        "carrier_value": matched[0]["paper_value"] if matched else None,
        "params_value": entry.get("value") if entry else None,
        "paper_value": entry.get("paper_value") if entry else None,
        "source": entry.get("source") if entry else None,
        "suppressed": False if raw_entry is not None else None,
        "duplicates": duplicates,
    }
    if not matched:
        return binding, f"parameter_carrier_missing:{params_name}"
    if duplicates:
        return binding, f"parameter_carrier_duplicated:{params_name}"
    if raw_entry is None:
        return binding, f"params_entry_missing_or_suppressed:{params_name}"
    if not isinstance(raw_entry, Mapping):
        return binding, f"params_entry_not_object:{params_name}"
    if not isinstance(binding["source"], str) or not binding["source"]:
        return binding, f"params_source_missing:{params_name}"

    carrier_value = binding["carrier_value"]
    paper_value = binding["paper_value"]
    params_value = binding["params_value"]
    if paper_value is not None:
        if not _values_equivalent(paper_value, carrier_value):
            return binding, f"parameter_paper_value_disagrees:{params_name}"
    elif not _values_equivalent(params_value, carrier_value):
        return binding, f"parameter_value_disagrees:{params_name}"
    return binding, None


def _parameter_name(raw: Any, *, root: str) -> tuple[str, str]:
    binding = _mapping(raw, root=root)
    if set(binding) != {"params_name", "callable_parameter"}:
        raise GraphMechanismRuntimePlanError(
            f"{root} must contain exactly params_name and callable_parameter"
        )
    return (
        _exact_text(binding.get("params_name"), root=f"{root}.params_name"),
        _python_parameter(
            binding.get("callable_parameter"),
            root=f"{root}.callable_parameter",
        ),
    )


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise GraphMechanismRuntimePlanError(
            "graph construction fixture contains a zero feature vector"
        )
    return numerator / (left_norm * right_norm)


def _unit_complement(coordinate: float) -> float:
    """Return the nearby component whose computed two-vector norm is one."""

    complement = math.sqrt(max(0.0, 1.0 - coordinate * coordinate))
    candidates = [complement]
    lower = complement
    upper = complement
    for _ in range(16):
        lower = math.nextafter(lower, -math.inf)
        upper = math.nextafter(upper, math.inf)
        candidates.extend((lower, upper))
    for candidate in candidates:
        if math.sqrt(coordinate * coordinate + candidate * candidate) == 1.0:
            return candidate
    raise GraphMechanismRuntimePlanError(
        "could not construct an exact cosine-threshold fixture witness"
    )


def _construction_features(
    count: int, threshold: float, comparison: str,
) -> list[list[float]]:
    """Create the ordinary disconnected construction fixture.

    Comparison semantics use a separate backend-stable witness.  Keeping the
    nominal fixture away from its arbitrary paper cutoff avoids asking a
    float32 implementation to preserve a float64 cosine equality while this
    fixture continues to discriminate metric, tie-breaking, and ordinary
    topology-changing parameter consumption.
    """

    if count < 5:
        raise GraphMechanismRuntimePlanError(
            "graph mechanism fixture requires at least five entities"
        )
    if not -1.0 <= threshold <= 1.0:
        raise GraphMechanismRuntimePlanError(
            "cosine-similarity threshold must lie in [-1, 1]"
        )
    if threshold == -1.0 and comparison == "greater_than_or_equal":
        raise GraphMechanismRuntimePlanError(
            "a cosine cutoff of -1 cannot provide a disconnected control "
            "component for the graph-mechanism fixture"
        )
    width = count
    features = [[0.0 for _ in range(width)] for _ in range(count)]

    # Nodes 0 and 1 have exactly equal cosine similarity to node 2, but
    # deliberately different norms.  A top-one cosine cap must therefore pick
    # canonical target index 0 from source 2, whereas raw dot-product ranking
    # picks target 1.  Power-of-two scales keep the cosine tie exact.
    base = math.sqrt((abs(threshold) + 1.0) / 2.0)
    tied_cosine = base + (1.0 - base) * 0.5
    tied_complement = _unit_complement(tied_cosine)
    low_scale = 0.5
    if threshold > 0.0:
        while low_scale * tied_cosine >= threshold:
            low_scale *= 0.5
    high_scale = 1.0 / low_scale
    features[0][0] = low_scale * tied_cosine
    features[0][1] = low_scale * tied_complement
    features[1][0] = high_scale * tied_cosine
    features[1][1] = -high_scale * tied_complement
    features[2][0] = 1.0

    # Nodes 3 and 4 are an exact, disconnected duplicate pair antipodal to the
    # primary component.  The -1 cross-component cosine keeps the control
    # disconnected for every supported cutoff above -1, including zero and
    # negative thresholds, while their mutual similarity remains exactly 1.
    # Exact comparison semantics are tested separately at the backend-stable
    # threshold 1.0.
    features[3][0] = -1.0
    features[4][0] = -1.0

    # Extra identities join the antipodal control component.  Keeping them on
    # the same exact ray cannot reconnect that component to the primary one at
    # any cutoff above -1 and cannot disturb the calibrated primary tie.
    for index in range(5, count):
        features[index][0] = -1.0

    tied_left = _cosine(features[2], features[0])
    tied_right = _cosine(features[2], features[1])
    duplicate_similarity = _cosine(features[3], features[4])
    if tied_left != tied_right or duplicate_similarity != 1.0:
        raise GraphMechanismRuntimePlanError(
            "graph construction fixture lost its exact tie or duplicate pair"
        )
    return features


def _comparison_features(count: int) -> list[list[float]]:
    """Create an exact float32-safe cosine-comparison boundary witness."""

    features = [[0.0 for _ in range(count)] for _ in range(count)]
    features[0][0] = 1.0
    features[1][0] = 1.0
    for index in range(2, count):
        features[index][index - 1] = 1.0
    return features


def _graph_edges(
    features: list[list[float]],
    *,
    threshold: float,
    comparison: str,
    cap: int | None,
    self_loop_policy: str,
    direction_policy: str,
) -> set[tuple[int, int]]:
    count = len(features)
    similarities = [
        [_cosine(features[source], features[target]) for target in range(count)]
        for source in range(count)
    ]

    def admitted(value: float) -> bool:
        if comparison == "greater_than":
            return value > threshold
        return value >= threshold

    selected: set[tuple[int, int]] = set()
    for source in range(count):
        candidates = [
            (similarities[source][target], target)
            for target in range(count)
            if target != source and admitted(similarities[source][target])
        ]
        candidates.sort(key=lambda item: (-item[0], item[1]))
        if cap is not None:
            candidates = candidates[:cap]
        selected.update((source, target) for _, target in candidates)

    if direction_policy == "undirected_bidirectional":
        selected = {
            (source, target)
            for source, target in selected
            if (target, source) in selected
        }
    if self_loop_policy == "required":
        selected.update((index, index) for index in range(count))
    else:
        selected = {
            edge for edge in selected if edge[0] != edge[1]
        }
    return selected


def _graph_payload(
    edges: set[tuple[int, int]],
    *,
    count: int,
    representation: str,
) -> list[Any]:
    ordered = sorted(edges)
    if representation == "sparse_edge_index":
        return [
            [source for source, _ in ordered],
            [target for _, target in ordered],
        ]
    dense: list[list[float]] = [
        [0.0 for _ in range(count)] for _ in range(count)
    ]
    for source, target in ordered:
        dense[source][target] = 1.0
    return dense


def _root_fixture(values: list[list[float]], axis: int) -> list[Any]:
    if axis == 0:
        return values
    if axis == 1:
        return [list(column) for column in zip(*values)]
    raise GraphMechanismRuntimePlanError(
        "homogeneous graph fixture supports coindexed entity axes 0 and 1"
    )


def _alternate_threshold(
    features: list[list[float]],
    *,
    nominal: float,
    comparison: str,
    cap: int | None,
    self_loop_policy: str,
    direction_policy: str,
    nominal_edges: set[tuple[int, int]],
) -> tuple[float, set[tuple[int, int]]]:
    similarities = sorted({
        round(_cosine(features[left], features[right]), 12)
        for left in range(len(features))
        for right in range(len(features))
        if left != right
    })
    candidates = [
        -1.0,
        -0.75,
        -0.5,
        0.0,
        0.5,
        0.75,
        0.9,
        0.99,
        1.0,
        *similarities,
    ]
    for candidate in candidates:
        if float(candidate) == float(nominal):
            continue
        edges = _graph_edges(
            features,
            threshold=float(candidate),
            comparison=comparison,
            cap=cap,
            self_loop_policy=self_loop_policy,
            direction_policy=direction_policy,
        )
        if edges != nominal_edges:
            return float(candidate), edges
    raise GraphMechanismRuntimePlanError(
        "no topology-changing cosine-threshold intervention exists for the "
        "normalized fixture"
    )


def _alternate_cap(
    features: list[list[float]],
    *,
    nominal: int,
    threshold: float,
    comparison: str,
    self_loop_policy: str,
    direction_policy: str,
    nominal_edges: set[tuple[int, int]],
) -> tuple[int, set[tuple[int, int]]]:
    for candidate in range(1, len(features) + 1):
        if candidate == nominal:
            continue
        edges = _graph_edges(
            features,
            threshold=threshold,
            comparison=comparison,
            cap=candidate,
            self_loop_policy=self_loop_policy,
            direction_policy=direction_policy,
        )
        if edges != nominal_edges:
            return candidate, edges
    raise GraphMechanismRuntimePlanError(
        "no topology-changing neighborhood-cap intervention exists for the "
        "normalized fixture"
    )


def _mechanism_fixture(
    *,
    alignment_fixture: Mapping[str, Any],
    representation: str,
    construction: Mapping[str, Any],
    execution: Mapping[str, Any],
    feature_entity_axis: int,
) -> dict[str, Any]:
    source = _mapping(
        alignment_fixture.get("source"),
        root="normalized alignment fixture source",
    )
    entity_ids = source.get("stable_entity_ids")
    if not isinstance(entity_ids, list):
        raise GraphMechanismRuntimePlanError(
            "normalized alignment fixture has no stable entity ids"
        )
    count = len(entity_ids)
    threshold = _mapping(
        construction.get("threshold"), root="normalized construction threshold"
    )
    threshold_value = threshold.get("value")
    if not _is_finite_number(threshold_value):
        raise GraphMechanismRuntimePlanError(
            "normalized threshold has no finite runtime value"
        )
    threshold_value = float(threshold_value)
    comparison = str(threshold.get("comparison"))
    cap = _mapping(construction.get("cap"), root="normalized construction cap")
    cap_value = cap.get("value") if cap.get("kind") != "none" else None
    if cap_value is not None and (
        isinstance(cap_value, bool)
        or not isinstance(cap_value, int)
        or cap_value <= 0
    ):
        raise GraphMechanismRuntimePlanError(
            "normalized cap has no positive integral runtime value"
        )
    self_loop_policy = str(construction.get("self_loop_policy"))
    direction_policy = str(construction.get("direction_policy"))
    features = _construction_features(count, threshold_value, comparison)
    comparison_features = _comparison_features(count)
    nominal_edges = _graph_edges(
        features,
        threshold=threshold_value,
        comparison=comparison,
        cap=cap_value,
        self_loop_policy=self_loop_policy,
        direction_policy=direction_policy,
    )
    comparison_edges = _graph_edges(
        comparison_features,
        threshold=1.0,
        comparison=comparison,
        cap=cap_value,
        self_loop_policy=self_loop_policy,
        direction_policy=direction_policy,
    )
    non_loop_edges = sorted(
        edge for edge in nominal_edges if edge[0] != edge[1]
    )
    if not non_loop_edges:
        raise GraphMechanismRuntimePlanError(
            "normalized construction fixture has no neighbor edge"
        )
    source_entity, target_entity = non_loop_edges[0]
    adjacency: dict[int, set[int]] = {index: set() for index in range(count)}
    for edge_source, edge_target in nominal_edges:
        if edge_source == edge_target:
            continue
        adjacency[edge_source].add(edge_target)
    reachable = {source_entity}
    frontier = [source_entity]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current] - reachable:
            reachable.add(neighbor)
            frontier.append(neighbor)
    controls = [index for index in range(count) if index not in reachable]
    if not controls:
        raise GraphMechanismRuntimePlanError(
            "normalized graph fixture has no neighbor control entity"
        )

    threshold_parameter = _mapping(
        threshold.get("parameter"), root="normalized threshold parameter"
    )
    threshold_name = str(threshold_parameter["params_name"])
    threshold_alternate, threshold_edges = _alternate_threshold(
        features,
        nominal=threshold_value,
        comparison=comparison,
        cap=cap_value,
        self_loop_policy=self_loop_policy,
        direction_policy=direction_policy,
        nominal_edges=nominal_edges,
    )
    parameter_interventions: dict[str, Any] = {
        threshold_name: {
            "nominal_value": threshold_value,
            "alternate_value": threshold_alternate,
            "value": threshold_alternate,
            "callable_parameter": threshold_parameter["callable_parameter"],
            "expected_relation": "graph_must_differ",
            "expected_graph": _graph_payload(
                threshold_edges, count=count, representation=representation
            ),
        }
    }
    if cap.get("kind") == "per_source_top_similarity":
        cap_parameter = _mapping(
            cap.get("parameter"), root="normalized cap parameter"
        )
        cap_name = str(cap_parameter["params_name"])
        alternate_cap, cap_edges = _alternate_cap(
            features,
            nominal=int(cap_value),
            threshold=threshold_value,
            comparison=comparison,
            self_loop_policy=self_loop_policy,
            direction_policy=direction_policy,
            nominal_edges=nominal_edges,
        )
        parameter_interventions[cap_name] = {
            "nominal_value": cap_value,
            "alternate_value": alternate_cap,
            "value": alternate_cap,
            "callable_parameter": cap_parameter["callable_parameter"],
            "expected_relation": "graph_must_differ",
            "expected_graph": _graph_payload(
                cap_edges, count=count, representation=representation
            ),
        }

    feature_root = str(construction["feature_input_root"])
    neighbor_root = str(execution["neighbor_signal_root"])
    return {
        "scope": "r2c088_asymmetric_homogeneous_graph_fixture",
        "entity_ids": list(entity_ids),
        feature_root: _root_fixture(
            features, feature_entity_axis
        ),
        neighbor_root: _root_fixture(
            [[float(index + 1), float((index + 1) ** 2)] for index in range(count)],
            int(execution["entity_axis"]),
        ),
        "comparison_witness": {
            "metric": "cosine_similarity",
            "comparison": comparison,
            "threshold_value": 1.0,
            "entity_axis": feature_entity_axis,
            "feature_input": _root_fixture(
                comparison_features, feature_entity_axis
            ),
            "expected_graph": _graph_payload(
                comparison_edges,
                count=count,
                representation=representation,
            ),
        },
        "expected_graph": _graph_payload(
            nominal_edges, count=count, representation=representation
        ),
        "parameter_interventions": parameter_interventions,
        "topology_intervention": {
            "remove_edges": [[source_entity, target_entity]],
            "targets": [target_entity],
        },
        "neighbor_intervention": {
            "source": source_entity,
            "target": target_entity,
            "delta": 7.0,
            "control_targets": controls[:2],
        },
        "permutation": list(range(2, count)) + [0, 1],
        "alignment_source": _json_copy(
            dict(alignment_fixture), root="normalized alignment fixture"
        ),
    }


def _normalize_construction(
    raw: Mapping[str, Any],
    *,
    records: list[dict[str, Any]],
    entries: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    threshold = _mapping(
        raw.get("threshold"),
        root="homogeneous_graph_mechanism.construction.threshold",
    )
    if threshold.get("metric") != "cosine_similarity" or threshold.get(
        "comparison"
    ) not in {"greater_than", "greater_than_or_equal"}:
        raise GraphMechanismRuntimePlanError(
            "homogeneous graph threshold must declare cosine_similarity and "
            "an exact supported comparison"
        )
    threshold_name, threshold_parameter = _parameter_name(
        threshold.get("parameter"),
        root="homogeneous_graph_mechanism.construction.threshold.parameter",
    )
    threshold_binding, reason = _parameter_binding(
        params_name=threshold_name,
        records=records,
        entries=entries,
    )

    cap = _mapping(
        raw.get("cap"),
        root="homogeneous_graph_mechanism.construction.cap",
    )
    cap_binding: dict[str, Any] | None = None
    cap_name: str | None = None
    cap_parameter: str | None = None
    if cap.get("kind") == "none":
        if set(cap) != {"kind"}:
            raise GraphMechanismRuntimePlanError(
                "cap kind='none' cannot carry parameter metadata"
            )
    elif cap.get("kind") == "per_source_top_similarity":
        if set(cap) != {"kind", "parameter"}:
            raise GraphMechanismRuntimePlanError(
                "per-source top-similarity cap accepts only its exact "
                "parameter binding"
            )
        cap_name, cap_parameter = _parameter_name(
            cap.get("parameter"),
            root="homogeneous_graph_mechanism.construction.cap.parameter",
        )
        if cap_name == threshold_name:
            raise GraphMechanismRuntimePlanError(
                "threshold and cap must bind different params entries"
            )
        cap_binding, cap_reason = _parameter_binding(
            params_name=cap_name,
            records=records,
            entries=entries,
        )
        reason = reason or cap_reason
    else:
        raise GraphMechanismRuntimePlanError(
            f"unsupported graph cap semantics {cap.get('kind')!r}"
        )

    authority_bindings = {threshold_name: threshold_binding}
    if cap_name is not None and cap_binding is not None:
        authority_bindings[cap_name] = cap_binding
    authority = {
        "status": "pass" if reason is None else "fail",
        "reason": reason,
        "bindings": authority_bindings,
    }

    threshold_value = threshold_binding["params_value"]
    if reason is None and not _is_finite_number(threshold_value):
        reason = f"threshold_value_not_finite_numeric:{threshold_name}"
    normalized_threshold = {
        "metric": "cosine_similarity",
        "comparison": str(threshold["comparison"]),
        "parameter": {
            "params_name": threshold_name,
            "callable_parameter": threshold_parameter,
        },
        "value": threshold_value,
    }
    if cap_name is None:
        normalized_cap: dict[str, Any] = {"kind": "none"}
    else:
        cap_value = cap_binding["params_value"] if cap_binding else None
        if reason is None and (
            isinstance(cap_value, bool)
            or not isinstance(cap_value, int)
            or cap_value <= 0
        ):
            reason = f"cap_value_not_positive_integer:{cap_name}"
        normalized_cap = {
            "kind": "per_source_top_similarity",
            "parameter": {
                "params_name": cap_name,
                "callable_parameter": cap_parameter,
            },
            "value": cap_value,
        }
    feature_parameter = _python_parameter(
        raw.get("feature_parameter"),
        root="homogeneous_graph_mechanism.construction.feature_parameter",
    )
    callable_parameters = [feature_parameter, threshold_parameter]
    if cap_parameter is not None:
        callable_parameters.append(cap_parameter)
    if len(callable_parameters) != len(set(callable_parameters)):
        raise GraphMechanismRuntimePlanError(
            "construction feature, threshold, and cap roles must bind "
            "pairwise-distinct callable parameters"
        )
    if reason is not None:
        authority["status"] = "fail"
        authority["reason"] = reason

    self_loop = raw.get("self_loop_policy")
    direction = raw.get("direction_policy")
    if self_loop not in {"required", "forbidden"}:
        raise GraphMechanismRuntimePlanError(
            "construction.self_loop_policy must be 'required' or 'forbidden'"
        )
    if direction not in {"directed", "undirected_bidirectional"}:
        raise GraphMechanismRuntimePlanError(
            "construction.direction_policy must be 'directed' or "
            "'undirected_bidirectional'"
        )
    output_selector = _mapping(
        raw.get("output_selector"),
        root="homogeneous_graph_mechanism.construction.output_selector",
    )
    selector_kind = output_selector.get("kind")
    if selector_kind == "return_value" and set(output_selector) != {"kind"}:
        raise GraphMechanismRuntimePlanError(
            "return_value output selector cannot carry extra fields"
        )
    if selector_kind == "tuple_item" and (
        set(output_selector) != {"kind", "index"}
        or isinstance(output_selector.get("index"), bool)
        or not isinstance(output_selector.get("index"), int)
        or int(output_selector["index"]) < 0
    ):
        raise GraphMechanismRuntimePlanError(
            "tuple_item output selector requires one non-negative integer index"
        )
    if selector_kind == "mapping_item" and (
        set(output_selector) != {"kind", "key"}
        or not isinstance(output_selector.get("key"), str)
        or not output_selector["key"]
        or output_selector["key"] != output_selector["key"].strip()
    ):
        raise GraphMechanismRuntimePlanError(
            "mapping_item output selector requires one exact key"
        )
    if selector_kind not in {"return_value", "tuple_item", "mapping_item"}:
        raise GraphMechanismRuntimePlanError(
            f"unsupported graph output selector {selector_kind!r}"
        )

    construction = {
        "callable": _callable(
            raw.get("callable"),
            root="homogeneous_graph_mechanism.construction.callable",
        ),
        "feature_input_root": _exact_text(
            raw.get("feature_input_root"),
            root="homogeneous_graph_mechanism.construction.feature_input_root",
        ),
        "feature_parameter": feature_parameter,
        "output_selector": _json_copy(
            dict(output_selector), root="graph construction output selector"
        ),
        "threshold": normalized_threshold,
        "cap": normalized_cap,
        "self_loop_policy": str(self_loop),
        "direction_policy": str(direction),
    }
    return construction, authority, reason


def _normalize_alignment(
    method_spec: Mapping[str, Any],
    arch_contract: Mapping[str, Any],
    *,
    homogeneous_ids: list[str],
    alignment_element_id: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    try:
        build_plan = load_build_plan(dict(method_spec))
    except (TypeError, ValueError, FileNotFoundError) as exc:
        return None, None, f"build_plan_unavailable:{exc}"
    if build_plan is None:
        return None, None, "build_plan_unavailable"
    try:
        normalized = normalize_schema2_forecasting_plan(
            dict(arch_contract),
            build_plan,
            method_spec=dict(method_spec),
        )
    except (RuntimePlanError, TypeError, ValueError) as exc:
        return None, None, f"relational_alignment_unresolved:{exc}"
    relational = normalized.get("relational")
    if not isinstance(relational, Mapping):
        return None, None, "relational_alignment_missing"
    representation = relational.get("representation")
    if representation not in {"sparse_edge_index", "dense_adjacency"}:
        return None, None, f"unsupported_graph_representation:{representation}"
    relational_ids = relational.get("methodology_element_ids")
    if not isinstance(relational_ids, list) or set(relational_ids) != set(
        homogeneous_ids
    ):
        return (
            None,
            None,
            "relational_methodology_grounding_disagrees",
        )
    if alignment_element_id not in relational_ids:
        return None, None, "alignment_element_not_relationally_grounded"

    alignment = {
        "status": "ready",
        "reason": None,
        "element_id": alignment_element_id,
        "methodology_element_ids": list(relational_ids),
        "entity_axis": relational.get("entity_axis"),
        "representation": representation,
        "entity_dimension": relational.get("entity_dimension"),
        "stable_entity_id_root": relational.get("stable_entity_id_root"),
        "graph_root": relational.get("graph_root"),
        "coindexed_roots": _json_copy(
            relational.get("coindexed_roots") or {},
            root="normalized relational coindexed roots",
        ),
        "source_endpoint_index_space": relational.get(
            "source_endpoint_index_space"
        ),
        "prepared_endpoint_index_space": relational.get(
            "prepared_endpoint_index_space"
        ),
        "edge_orientation": relational.get("edge_orientation"),
        "output_order": relational.get("output_order"),
        "phase_batch_modes": _json_copy(
            relational.get("phase_batch_modes") or {},
            root="normalized relational phase modes",
        ),
        "degree_root": relational.get("degree_root"),
        "degree_kind": relational.get("degree_kind"),
        "degree_semantics": relational.get("degree_semantics"),
        "preparation_callable": _json_copy(
            relational.get("preparation_callable") or {},
            root="normalized relational preparation callable",
        ),
        "execution": _json_copy(
            relational.get("execution") or {},
            root="normalized relational execution",
        ),
    }
    fixture = {
        "scope": "r2c084_asymmetric_relational_fixture",
        "source": _json_copy(
            relational.get("source_fixture") or {},
            root="normalized relational source fixture",
        ),
        "fitting_identity": _json_copy(
            relational.get("fitting_identity") or {},
            root="normalized relational fitting identity",
        ),
        "inference_identity": _json_copy(
            relational.get("inference_identity") or {},
            root="normalized relational inference identity",
        ),
    }
    return alignment, fixture, None


def normalize_graph_mechanism_runtime_plan(
    method_spec: Mapping[str, Any],
    params: Mapping[str, Any],
    arch_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a JSON-only frozen homogeneous-graph mechanism plan."""

    method_spec = _mapping(method_spec, root="method_spec")
    params = _mapping(params, root="params")
    arch_contract = _mapping(arch_contract, root="arch_contract")
    methodology, elements = _methodology(method_spec)
    homogeneous_ids, unsupported = _relational_disposition(elements)

    mechanism = (
        methodology.get("homogeneous_graph_mechanism")
        if methodology is not None
        else None
    )
    # R2C-092: markers on the block's own elements are derived wiring the
    # stored JSON may omit; the disposition unions declared markers with the
    # block-bound ids so the plan's applicability gate and alignment
    # grounding read the same set every other consumer derives.
    if isinstance(mechanism, Mapping):
        for element_id in mechanism_marker_element_ids(mechanism):
            if element_id in {str(e.get("element_id")) for e in elements} \
                    and element_id not in homogeneous_ids:
                homogeneous_ids.append(element_id)
    if unsupported:
        plan = _base_plan(
            status="unprobeable",
            reason="unsupported_relational_structure",
            groundings={"unsupported_relational_elements": unsupported},
        )
        return _json_copy(plan, root="graph mechanism runtime plan")
    if not homogeneous_ids:
        if mechanism is not None:
            raise GraphMechanismRuntimePlanError(
                "homogeneous_graph_mechanism exists without a paper-grounded "
                "homogeneous_graph element"
            )
        if arch_contract.get("relational_indexing") is not None:
            plan = _base_plan(
                status="blocked",
                reason="graph_free_method_has_relational_contract",
            )
        else:
            plan = _base_plan(
                status="not_applicable",
                reason="graph_free_method",
            )
        return _json_copy(plan, root="graph mechanism runtime plan")
    if mechanism is None:
        plan = _base_plan(
            status="unprobeable",
            reason="homogeneous_graph_mechanism_contract_missing",
            groundings={"homogeneous_graph_element_ids": homogeneous_ids},
        )
        return _json_copy(plan, root="graph mechanism runtime plan")
    mechanism = _mapping(
        mechanism,
        root="methodology_replication_contract.homogeneous_graph_mechanism",
    )
    if mechanism.get("schema_version") != "1.0":
        raise GraphMechanismRuntimePlanError(
            "homogeneous_graph_mechanism.schema_version must equal '1.0'"
        )

    groundings = _normalize_groundings(mechanism, elements)
    alignment_element_id = str(mechanism["alignment_element_id"])
    alignment, fixture, alignment_reason = _normalize_alignment(
        method_spec,
        arch_contract,
        homogeneous_ids=homogeneous_ids,
        alignment_element_id=alignment_element_id,
    )
    if alignment_reason is not None:
        plan = _base_plan(
            status="blocked",
            reason=alignment_reason,
            groundings=groundings,
        )
        plan["alignment"] = {
            "status": "blocked",
            "reason": alignment_reason,
            "element_id": alignment_element_id,
        }
        return _json_copy(plan, root="graph mechanism runtime plan")
    assert alignment is not None
    assert fixture is not None
    preparation = alignment.get("preparation_callable") or {}
    if isinstance(preparation, Mapping):
        prep_module = preparation.get("module")
        prep_name = preparation.get("name")
        if isinstance(prep_module, str) and isinstance(prep_name, str):
            groundings[ALIGNMENT_PROBE_REF]["bound_callables"] = [
                f"{prep_module}:{prep_name}"
            ]

    records = _carrier_records(method_spec)
    entries = _params_entries(params)
    raw_construction = _mapping(
        mechanism.get("construction"),
        root="homogeneous_graph_mechanism.construction",
    )
    construction, authority, parameter_reason = _normalize_construction(
        raw_construction,
        records=records,
        entries=entries,
    )

    coindexed_roots = alignment["coindexed_roots"]
    feature_root = construction["feature_input_root"]
    if feature_root not in coindexed_roots:
        raise GraphMechanismRuntimePlanError(
            f"construction feature root {feature_root!r} is not an exact "
            "R2C-084 coindexed root"
        )
    raw_message = _mapping(
        mechanism.get("message_passing"),
        root="homogeneous_graph_mechanism.message_passing",
    )
    neighbor_root = _exact_text(
        raw_message.get("neighbor_signal_root"),
        root="homogeneous_graph_mechanism.message_passing.neighbor_signal_root",
    )
    if neighbor_root not in coindexed_roots:
        raise GraphMechanismRuntimePlanError(
            f"neighbor signal root {neighbor_root!r} is not an exact R2C-084 "
            "coindexed root"
        )
    preparation_callable = _mapping(
        alignment.get("preparation_callable"),
        root="normalized relational preparation_callable",
    )
    tensor_backend = preparation_callable.get("tensor_backend")
    if tensor_backend not in {"numpy", "torch"}:
        raise GraphMechanismRuntimePlanError(
            "normalized relational preparation_callable.tensor_backend must "
            "be exactly 'numpy' or 'torch'"
        )
    output_root = _exact_text(
        raw_message.get("output_root"),
        root="homogeneous_graph_mechanism.message_passing.output_root",
    )
    if output_root not in coindexed_roots:
        raise GraphMechanismRuntimePlanError(
            f"message-passing output root {output_root!r} is not coindexed"
        )
    relational_execution = _mapping(
        alignment.get("execution"), root="normalized relational execution"
    )
    inference_inputs = _mapping(
        relational_execution.get("inference_coindexed"),
        root="normalized relational inference coindexed inputs",
    )
    if output_root in inference_inputs:
        raise GraphMechanismRuntimePlanError(
            f"message-passing output root {output_root!r} is an inference "
            "input root, not an output authority"
        )

    graph_parameter = _python_parameter(
        raw_message.get("graph_parameter"),
        root="homogeneous_graph_mechanism.message_passing.graph_parameter",
    )
    neighbor_parameter = _python_parameter(
        raw_message.get("neighbor_signal_parameter"),
        root=(
            "homogeneous_graph_mechanism.message_passing."
            "neighbor_signal_parameter"
        ),
    )
    if graph_parameter == neighbor_parameter:
        raise GraphMechanismRuntimePlanError(
            "message-passing graph and neighbor-signal roles must bind "
            "distinct callable parameters"
        )

    execution = {
        "callable": _callable(
            raw_message.get("callable"),
            root="homogeneous_graph_mechanism.message_passing.callable",
        ),
        "graph_parameter": graph_parameter,
        "neighbor_signal_root": neighbor_root,
        "neighbor_signal_parameter": neighbor_parameter,
        "entity_axis": coindexed_roots[neighbor_root],
        "output_root": output_root,
        "output_entity_axis": coindexed_roots[output_root],
        "tensor_backend": tensor_backend,
        "permutation_applicability": (
            "required"
            if mechanism.get("permutation_applicability") == "equivariant"
            else "not_applicable"
        ),
        "tolerance": HARNESS_TOLERANCE,
        "seed": HARNESS_SEED,
    }
    raw_ablation = mechanism.get("contribution_ablation")
    if raw_ablation is None:
        ablation = None
    else:
        ablation_mapping = _mapping(
            raw_ablation,
            root="homogeneous_graph_mechanism.contribution_ablation",
        )
        criterion = ablation_mapping.get("discriminating_probe_ref")
        if criterion not in {
            CANONICAL_PROBE_REFS["topology"],
            CANONICAL_PROBE_REFS["neighbor_signal"],
        }:
            raise GraphMechanismRuntimePlanError(
                "graph contribution ablation must name the exact topology or "
                "neighbor sensitivity discriminator"
            )
        ablation_kind = ablation_mapping.get("kind")
        element_id = _exact_text(
            ablation_mapping.get("element_id"),
            root="homogeneous_graph_mechanism.contribution_ablation.element_id",
        )
        if ablation_kind in {"empty_graph", "identity_graph", "permuted_graph"}:
            if set(ablation_mapping) != {
                "kind", "element_id", "discriminating_probe_ref"
            }:
                raise GraphMechanismRuntimePlanError(
                    "graph-input contribution ablation has unexpected fields"
                )
            ablation = {
                "kind": str(ablation_kind),
                "element_id": element_id,
                "discriminating_probe_ref": str(criterion),
            }
        elif ablation_kind in {
            "removed_message_passing", "non_graph_decoder"
        }:
            if set(ablation_mapping) != {
                "kind",
                "element_id",
                "callable",
                "graph_parameter",
                "neighbor_signal_parameter",
                "output_root",
                "discriminating_probe_ref",
            }:
                raise GraphMechanismRuntimePlanError(
                    "callable contribution ablation must declare its exact "
                    "graph, neighbor-signal, and output bindings"
                )
            null_graph_parameter = _python_parameter(
                ablation_mapping.get("graph_parameter"),
                root=(
                    "homogeneous_graph_mechanism.contribution_ablation."
                    "graph_parameter"
                ),
            )
            null_neighbor_parameter = _python_parameter(
                ablation_mapping.get("neighbor_signal_parameter"),
                root=(
                    "homogeneous_graph_mechanism.contribution_ablation."
                    "neighbor_signal_parameter"
                ),
            )
            if null_graph_parameter == null_neighbor_parameter:
                raise GraphMechanismRuntimePlanError(
                    "callable contribution-ablation graph and neighbor-signal "
                    "roles must bind distinct parameters"
                )
            null_output_root = _exact_text(
                ablation_mapping.get("output_root"),
                root=(
                    "homogeneous_graph_mechanism.contribution_ablation."
                    "output_root"
                ),
            )
            if null_output_root != output_root:
                raise GraphMechanismRuntimePlanError(
                    "callable contribution ablation must return the same "
                    "coindexed output root as message passing"
                )
            ablation = {
                "kind": str(ablation_kind),
                "element_id": element_id,
                "callable": _callable(
                    ablation_mapping.get("callable"),
                    root=(
                        "homogeneous_graph_mechanism.contribution_ablation."
                        "callable"
                    ),
                ),
                "graph_parameter": null_graph_parameter,
                "neighbor_signal_parameter": null_neighbor_parameter,
                "output_root": null_output_root,
                "discriminating_probe_ref": str(criterion),
            }
        else:
            raise GraphMechanismRuntimePlanError(
                f"unsupported graph contribution ablation {ablation_kind!r}"
            )

    supported_axes = {
        "construction_feature": coindexed_roots[feature_root],
        "neighbor_signal": execution["entity_axis"],
        "output": execution["output_entity_axis"],
    }
    unsupported_axes = {
        name: axis
        for name, axis in supported_axes.items()
        if isinstance(axis, bool) or not isinstance(axis, int) or axis not in {0, 1}
    }
    if unsupported_axes:
        reason = "unsupported_graph_entity_axis"
        plan = _base_plan(
            status="unprobeable", reason=reason, groundings=groundings
        )
        plan.update({
            "representation": alignment["representation"],
            "construction": construction,
            "parameter_authority": authority,
            "execution": execution,
            "ablation": ablation,
            "alignment": alignment,
            "fixture": fixture,
            "unsupported_axes": unsupported_axes,
        })
        return _json_copy(plan, root="graph mechanism runtime plan")

    if authority["status"] == "pass":
        fixture = _mechanism_fixture(
            alignment_fixture=fixture,
            representation=str(alignment["representation"]),
            construction=construction,
            execution=execution,
            feature_entity_axis=int(coindexed_roots[feature_root]),
        )
    else:
        alignment_fixture = fixture
        alignment_source = _mapping(
            alignment_fixture.get("source"),
            root="normalized alignment fixture source",
        )
        alignment_entity_ids = alignment_source.get("stable_entity_ids")
        if not isinstance(alignment_entity_ids, list):
            raise GraphMechanismRuntimePlanError(
                "normalized alignment fixture has no stable entity ids"
            )
        fixture = {
            "scope": "r2c088_parameter_authority_unresolved",
            # HG-1 remains independently assessable even when HG-2 parameter
            # authority fails.  Keep the exact R2C-084 identity carrier while
            # withholding all graph-construction interventions.
            "entity_ids": list(alignment_entity_ids),
            "alignment_source": alignment_fixture,
            "parameter_interventions": {},
        }

    plan = _base_plan(
        status="ready",
        reason=(
            "homogeneous_graph_runtime_plan_ready"
            if parameter_reason is None
            else parameter_reason
        ),
        groundings=groundings,
    )
    plan.update(
        {
            "representation": alignment["representation"],
            "construction": construction,
            "parameter_authority": authority,
            "execution": execution,
            "ablation": ablation,
            "alignment": alignment,
            "fixture": fixture,
        }
    )
    return _json_copy(plan, root="graph mechanism runtime plan")
