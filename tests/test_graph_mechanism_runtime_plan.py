"""Frozen authority and fixture plans for homogeneous graph probes."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from scripts.graph_mechanism_runtime_plan import (
    ALIGNMENT_PROBE_REF,
    CANONICAL_PROBE_REFS,
    GraphMechanismRuntimePlanError,
    normalize_graph_mechanism_runtime_plan,
)
from scripts.probes.graph_mechanism import run_graph_mechanism_probes
from run_probes import _graph_constructor_adapter
from tests.test_tsf_runtime_plan import (
    _graph_free_contract,
    _relational_contract,
)


def _element(
    element_id: str,
    refs: list[str],
    *,
    relational: dict | None,
    role: str = "core_methodology",
) -> dict:
    return {
        "element_id": element_id,
        "role": role,
        "paper_element_ids": [f"paper-{element_id}"],
        "verification_probe_refs": refs,
        "relational_structure": copy.deepcopy(relational),
    }


def _mechanism(*, with_cap: bool = False) -> dict:
    cap = {"kind": "none"}
    if with_cap:
        cap = {
            "kind": "per_source_top_similarity",
            "parameter": {
                "params_name": "max_neighbors",
                "callable_parameter": "max_neighbors",
            },
        }
    return {
        "schema_version": "1.0",
        "alignment_element_id": "graph-alignment",
        "construction": {
            "element_id": "graph-construction",
            "callable": {
                "module": "method.model",
                "qualname": "build_article_graph",
            },
            "feature_input_root": "batch.static_features",
            "feature_parameter": "static_features",
            "output_selector": {"kind": "return_value"},
            "threshold": {
                "metric": "cosine_similarity",
                "comparison": "greater_than_or_equal",
                "parameter": {
                    "params_name": "similarity_threshold",
                    "callable_parameter": "similarity_threshold",
                },
            },
            "cap": cap,
            "self_loop_policy": "required",
            "direction_policy": "undirected_bidirectional",
        },
        "message_passing": {
            "element_id": "graph-message-passing",
            "callable": {
                "module": "method.model",
                "qualname": "encode_graph",
            },
            "graph_parameter": "edge_index",
            "neighbor_signal_root": "batch.demand",
            "neighbor_signal_parameter": "demand_history",
            "output_root": "outputs.forecasts",
        },
        "permutation_applicability": "equivariant",
        "contribution_ablation": {
            "kind": "empty_graph",
            "element_id": "graph-ablation-control",
            "discriminating_probe_ref": CANONICAL_PROBE_REFS[
                "neighbor_signal"
            ],
        },
        "probe_refs": {
            "parameter_agreement": CANONICAL_PROBE_REFS[
                "parameter_agreement"
            ],
            "construction": CANONICAL_PROBE_REFS["construction"],
            "topology": CANONICAL_PROBE_REFS["topology"],
            "neighbor_signal": CANONICAL_PROBE_REFS["neighbor_signal"],
            "permutation": CANONICAL_PROBE_REFS["permutation"],
            "contribution_ablation": CANONICAL_PROBE_REFS[
                "contribution_ablation"
            ],
        },
    }


def _method_spec(*, with_cap: bool = False) -> dict:
    marker = {"kind": "homogeneous_graph"}
    return {
        "schema_version": "1.14.0",
        "comparison": {
            "classification": {"id": "time_series_forecasting"},
        },
        "critical_requirements": {
            "param_glossary": [
                {
                    "name": "Similarity cutoff",
                    "aliases": ["similarity_threshold"],
                    "paper_value": 0.5,
                },
                *(
                    [
                        {
                            "name": "Neighborhood cap",
                            "aliases": ["max_neighbors"],
                            "paper_value": 3,
                        }
                    ]
                    if with_cap
                    else []
                ),
            ],
            "scale_dependent_hyperparameters": [],
        },
        "methodology_replication_contract": {
            "schema_version": "1.0",
            "elements": [
                _element(
                    "graph-alignment",
                    [ALIGNMENT_PROBE_REF],
                    relational=marker,
                ),
                _element(
                    "graph-construction",
                    [
                        CANONICAL_PROBE_REFS["parameter_agreement"],
                        CANONICAL_PROBE_REFS["construction"],
                    ],
                    relational=marker,
                ),
                _element(
                    "graph-message-passing",
                    [
                        CANONICAL_PROBE_REFS["topology"],
                        CANONICAL_PROBE_REFS["neighbor_signal"],
                        CANONICAL_PROBE_REFS["permutation"],
                    ],
                    relational=marker,
                ),
                _element(
                    "graph-ablation-control",
                    [CANONICAL_PROBE_REFS["contribution_ablation"]],
                    relational=None,
                    role="evaluation_control",
                ),
            ],
            "homogeneous_graph_mechanism": _mechanism(with_cap=with_cap),
        },
    }


def _params(*, with_cap: bool = False) -> dict:
    entries = {
        "similarity_threshold": {
            "value": 0.5,
            "source": "paper",
            "paper_section": "Section 3",
        }
    }
    if with_cap:
        entries["max_neighbors"] = {
            "value": 3,
            "source": "paper",
            "paper_section": "Section 3",
        }
    return {"schema_version": "1.3.0", "params": entries}


def _arch_contract(
    *,
    representation: str = "sparse_edge_index",
    tensor_backend: str = "numpy",
) -> dict:
    contract = _relational_contract(representation=representation)
    contract["relational_indexing"]["methodology_element_ids"] = [
        "graph-alignment",
        "graph-construction",
        "graph-message-passing",
    ]
    contract["relational_indexing"]["preparation_callable"][
        "tensor_backend"
    ] = tensor_backend
    if tensor_backend == "torch":
        def convert_array_kinds(value):
            if isinstance(value, dict):
                if value.get("kind") == "ndarray":
                    value["kind"] = "tensor"
                for child in value.values():
                    convert_array_kinds(child)
            elif isinstance(value, list):
                for child in value:
                    convert_array_kinds(child)

        convert_array_kinds(contract)
    return contract


def _ready_inputs(
    *,
    representation: str = "sparse_edge_index",
    with_cap: bool = False,
    tensor_backend: str = "numpy",
) -> tuple[dict, dict, dict]:
    """Small shared planner fixture for planner-to-executor integration."""

    return (
        _method_spec(with_cap=with_cap),
        _params(with_cap=with_cap),
        _arch_contract(
            representation=representation,
            tensor_backend=tensor_backend,
        ),
    )


def _cap_one_ready_inputs(
    *, representation: str = "sparse_edge_index",
) -> tuple[dict, dict, dict]:
    method_spec, params, arch_contract = _ready_inputs(
        representation=representation,
        with_cap=True,
    )
    method_spec["critical_requirements"]["param_glossary"][1][
        "paper_value"
    ] = 1
    params["params"]["max_neighbors"]["value"] = 1
    return method_spec, params, arch_contract


def _independent_constructed_graph(
    feature_input,
    *,
    threshold: float,
    cap: int | None,
    representation: str,
    metric: str = "cosine_similarity",
    comparison: str = "greater_than_or_equal",
    tie_break: str = "lowest_target_index",
):
    features = np.asarray(feature_input, dtype=float)
    scores = features @ features.T
    if metric == "cosine_similarity":
        norms = np.linalg.norm(features, axis=1)
        scores = scores / np.outer(norms, norms)
    edges: set[tuple[int, int]] = set()
    for source in range(len(features)):
        candidates = []
        for target in range(len(features)):
            if source == target:
                continue
            value = float(scores[source, target])
            admitted = (
                value > threshold
                if comparison == "greater_than"
                else value >= threshold
            )
            if admitted:
                candidates.append((value, target))
        candidates.sort(
            key=lambda item: (
                -item[0],
                item[1]
                if tie_break == "lowest_target_index"
                else -item[1],
            )
        )
        if cap is not None:
            candidates = candidates[:cap]
        edges.update((source, target) for _, target in candidates)
    edges = {
        (source, target)
        for source, target in edges
        if (target, source) in edges
    }
    edges.update((node, node) for node in range(len(features)))
    ordered = sorted(edges)
    if representation == "sparse_edge_index":
        return np.asarray(
            [
                [source for source, _ in ordered],
                [target for _, target in ordered],
            ],
            dtype=np.int64,
        )
    graph = np.zeros((len(features), len(features)), dtype=float)
    for source, target in ordered:
        graph[source, target] = 1.0
    return graph


def _edge_set(graph, *, representation: str) -> set[tuple[int, int]]:
    array = np.asarray(graph)
    if representation == "sparse_edge_index":
        return {
            (int(source), int(target))
            for source, target in zip(array[0], array[1])
        }
    return {
        (int(source), int(target))
        for source, target in zip(*np.nonzero(array))
    }


def _real_float32_constructed_graph(
    feature_input,
    *,
    threshold: float,
    representation: str,
    comparison: str,
    tensor_backend: str,
):
    if tensor_backend == "torch":
        torch = pytest.importorskip("torch")
        assert type(feature_input) is torch.Tensor
        assert feature_input.dtype == torch.float32
        norms = torch.linalg.vector_norm(feature_input, dim=1)
        scores = feature_input @ feature_input.T / torch.outer(norms, norms)

        def score(source, target):
            return float(scores[source, target].item())
    else:
        assert type(feature_input) is np.ndarray
        assert feature_input.dtype == np.float32
        norms = np.linalg.norm(feature_input, axis=1)
        scores = feature_input @ feature_input.T / np.outer(norms, norms)

        def score(source, target):
            return float(scores[source, target])

    edges: set[tuple[int, int]] = set()
    for source in range(len(feature_input)):
        for target in range(len(feature_input)):
            if source == target:
                continue
            similarity = score(source, target)
            admitted = (
                similarity > threshold
                if comparison == "greater_than"
                else similarity >= threshold
            )
            if admitted:
                edges.add((source, target))
    edges = {
        (source, target)
        for source, target in edges
        if (target, source) in edges
    }
    edges.update((index, index) for index in range(len(feature_input)))
    ordered = sorted(edges)
    if tensor_backend == "torch":
        torch = pytest.importorskip("torch")
        if representation == "sparse_edge_index":
            return torch.tensor(
                [
                    [source for source, _ in ordered],
                    [target for _, target in ordered],
                ],
                dtype=torch.long,
            )
        graph = torch.zeros(
            (len(feature_input), len(feature_input)), dtype=torch.float32
        )
    else:
        if representation == "sparse_edge_index":
            return np.asarray(
                [
                    [source for source, _ in ordered],
                    [target for _, target in ordered],
                ],
                dtype=np.int64,
            )
        graph = np.zeros(
            (len(feature_input), len(feature_input)), dtype=np.float32
        )
    for source, target in ordered:
        graph[source, target] = 1.0
    return graph


def _attach_passing_alignment_receipt(plan: dict) -> None:
    plan["alignment"] = {
        **plan["alignment"],
        "status": "pass",
        "reason": "matching Stage 2.d runtime-validation receipt",
    }
    fixture = plan["fixture"]
    plan["alignment_runtime_receipt"] = {
        "receipt_version": "1.0",
        "probe_ref": ALIGNMENT_PROBE_REF,
        "status": "pass",
        "reason": "stage_2d_runtime_alignment_verified",
        "verified": True,
        "validator": "validate_arch_contract_runtime.py",
        "element_id": plan["alignment"]["element_id"],
        "fixture_scope": fixture["scope"],
        "representation": plan["representation"],
        "entity_ids": fixture["entity_ids"],
        "authority_digests": {
            "method_spec.json": f"sha256:{'a' * 64}",
            "arch_contract.json": f"sha256:{'b' * 64}",
        },
    }


@pytest.mark.parametrize(
    "representation", ["sparse_edge_index", "dense_adjacency"]
)
@pytest.mark.parametrize("tensor_backend", ["numpy", "torch"])
def test_ready_plan_reuses_relational_authority_and_is_strict_json(
    representation, tensor_backend,
):
    if tensor_backend == "torch":
        pytest.importorskip("torch")
    plan = normalize_graph_mechanism_runtime_plan(
        *_ready_inputs(
            representation=representation,
            tensor_backend=tensor_backend,
        )
    )

    assert plan["status"] == "ready"
    assert plan["reason"] == "homogeneous_graph_runtime_plan_ready"
    assert plan["representation"] == representation
    assert plan["alignment"]["status"] == "ready"
    assert plan["alignment"]["representation"] == representation
    assert plan["alignment"]["entity_axis"] == "series"
    assert plan["alignment"]["source_endpoint_index_space"] == (
        "source_entity_axis_positions"
    )
    assert plan["alignment"]["prepared_endpoint_index_space"] == (
        "local_batch_positions"
    )
    assert plan["alignment"]["execution"]["fitting_mode"] == (
        "induced_subgraph"
    )
    assert plan["execution"]["entity_axis"] == 0
    assert plan["execution"]["output_entity_axis"] == 0
    assert plan["execution"]["tensor_backend"] == tensor_backend
    assert plan["execution"]["permutation_applicability"] == "required"
    assert plan["execution"]["seed"] == 1729
    assert plan["execution"]["tolerance"] == 1.0e-8
    assert json.loads(json.dumps(plan, allow_nan=False)) == plan


def test_adjacent_unknown_graph_family_is_not_coerced_into_forecasting_plan():
    adjacent_inputs = copy.deepcopy(_ready_inputs())
    adjacent_inputs[0]["comparison"]["classification"]["id"] = (
        "graph_node_classification"
    )

    adjacent = normalize_graph_mechanism_runtime_plan(*adjacent_inputs)

    assert adjacent["status"] == "blocked"
    assert adjacent["reason"].startswith("build_plan_unavailable")
    assert adjacent["fixture"] is None


def test_missing_raw_backend_uses_r2c084_schema_default_exactly():
    pytest.importorskip("torch")
    method_spec, params, arch_contract = _ready_inputs(
        tensor_backend="torch"
    )
    arch_contract["relational_indexing"]["preparation_callable"].pop(
        "tensor_backend"
    )

    plan = normalize_graph_mechanism_runtime_plan(
        method_spec, params, arch_contract
    )

    assert plan["status"] == "ready"
    assert plan["alignment"]["preparation_callable"][
        "tensor_backend"
    ] == "torch"
    assert plan["execution"]["tensor_backend"] == "torch"


def test_invalid_r2c084_tensor_backend_fails_closed():
    method_spec, params, arch_contract = _ready_inputs()
    arch_contract["relational_indexing"]["preparation_callable"][
        "tensor_backend"
    ] = "jax"

    plan = normalize_graph_mechanism_runtime_plan(
        method_spec, params, arch_contract
    )

    assert plan["status"] == "blocked"
    assert "tensor_backend" in plan["reason"]


def test_plan_carries_exact_parameter_authority_and_runtime_intervention():
    plan = normalize_graph_mechanism_runtime_plan(*_ready_inputs())

    assert plan["parameter_authority"] == {
        "status": "pass",
        "reason": None,
        "bindings": {
            "similarity_threshold": {
                "carrier_value": 0.5,
                "params_value": 0.5,
                "paper_value": None,
                "source": "paper",
                "suppressed": False,
                "duplicates": [],
            }
        },
    }
    assert plan["construction"]["threshold"] == {
        "metric": "cosine_similarity",
        "comparison": "greater_than_or_equal",
        "parameter": {
            "params_name": "similarity_threshold",
            "callable_parameter": "similarity_threshold",
        },
        "value": 0.5,
    }
    fixture = plan["fixture"]
    intervention = fixture["parameter_interventions"][
        "similarity_threshold"
    ]
    assert intervention["nominal_value"] == 0.5
    assert intervention["alternate_value"] == intervention["value"]
    assert intervention["expected_relation"] == "graph_must_differ"
    assert intervention["expected_graph"] != fixture["expected_graph"]
    assert fixture["entity_ids"] == [101, 205, 309, 412, 518]
    assert "batch.static_features" in fixture
    assert "batch.demand" in fixture
    assert fixture["comparison_witness"] == {
        "metric": "cosine_similarity",
        "comparison": "greater_than_or_equal",
        "threshold_value": 1.0,
        "entity_axis": 0,
        "feature_input": [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ],
        "expected_graph": [
            [0, 0, 1, 1, 2, 3, 4],
            [0, 1, 0, 1, 2, 3, 4],
        ],
    }
    assert fixture["topology_intervention"]["remove_edges"]
    assert fixture["neighbor_intervention"]["control_targets"]
    assert sorted(fixture["permutation"]) == list(range(5))


def test_cap_binding_has_its_own_independent_runtime_intervention():
    plan = normalize_graph_mechanism_runtime_plan(
        *_ready_inputs(with_cap=True)
    )

    assert plan["construction"]["cap"] == {
        "kind": "per_source_top_similarity",
        "parameter": {
            "params_name": "max_neighbors",
            "callable_parameter": "max_neighbors",
        },
        "value": 3,
    }
    interventions = plan["fixture"]["parameter_interventions"]
    assert set(interventions) == {"similarity_threshold", "max_neighbors"}
    for params_name, intervention in interventions.items():
        assert intervention["alternate_value"] != intervention["nominal_value"]
        assert intervention["expected_graph"] != plan["fixture"][
            "expected_graph"
        ], params_name


def test_undirected_top_one_cap_uses_mutual_selection_before_self_loops():
    method_spec, params, arch_contract = _cap_one_ready_inputs()

    plan = normalize_graph_mechanism_runtime_plan(
        method_spec, params, arch_contract
    )

    sources, targets = plan["fixture"]["expected_graph"]
    edges = set(zip(sources, targets))
    independent_mutual_oracle = {
        (0, 0), (0, 2),
        (1, 1),
        (2, 0), (2, 2),
        (3, 3), (3, 4),
        (4, 3), (4, 4),
    }
    union_symmetrized_known_bad = independent_mutual_oracle | {
        (0, 1), (1, 0),
    }

    assert edges == independent_mutual_oracle
    assert edges != union_symmetrized_known_bad
    assert all(
        sum(source == node and target != node for source, target in edges) <= 1
        for node in range(5)
    )


@pytest.mark.parametrize(
    "representation", ["sparse_edge_index", "dense_adjacency"]
)
def test_frozen_construction_fixture_discriminates_metric_comparison_and_tie(
    representation,
):
    plan = normalize_graph_mechanism_runtime_plan(
        *_cap_one_ready_inputs(representation=representation)
    )
    fixture = plan["fixture"]
    features = np.asarray(fixture["batch.static_features"], dtype=float)
    threshold = plan["construction"]["threshold"]["value"]
    cap = plan["construction"]["cap"]["value"]

    similarities = features @ features.T / np.outer(
        np.linalg.norm(features, axis=1),
        np.linalg.norm(features, axis=1),
    )
    assert similarities[2, 0] == similarities[2, 1]
    assert similarities[2, 0] > threshold
    assert float(features[2] @ features[1]) > float(
        features[2] @ features[0]
    )
    assert similarities[3, 4] == 1.0
    assert similarities[3, 4] > threshold

    expected = _edge_set(
        fixture["expected_graph"], representation=representation
    )
    correct = _edge_set(
        _independent_constructed_graph(
            features,
            threshold=threshold,
            cap=cap,
            representation=representation,
        ),
        representation=representation,
    )
    assert correct == expected
    assert (2, 0) in expected and (2, 1) not in expected

    known_bad_graphs = {
        "dot_product": _independent_constructed_graph(
            features,
            threshold=threshold,
            cap=cap,
            representation=representation,
            metric="dot_product",
        ),
        "highest_target_tie_break": _independent_constructed_graph(
            features,
            threshold=threshold,
            cap=cap,
            representation=representation,
            tie_break="highest_target_index",
        ),
    }
    for name, graph in known_bad_graphs.items():
        assert _edge_set(graph, representation=representation) != expected, name

    comparison_witness = fixture["comparison_witness"]
    comparison_features = np.asarray(
        comparison_witness["feature_input"], dtype=np.float32
    )
    comparison_scores = comparison_features @ comparison_features.T
    assert comparison_scores[0, 1] == np.float32(1.0)
    comparison_expected = _edge_set(
        comparison_witness["expected_graph"], representation=representation
    )
    strict_comparison = _edge_set(
        _independent_constructed_graph(
            comparison_features,
            threshold=1.0,
            cap=cap,
            representation=representation,
            comparison="greater_than",
        ),
        representation=representation,
    )
    assert (0, 1) in comparison_expected
    assert strict_comparison != comparison_expected


@pytest.mark.parametrize(
    ("threshold", "comparison"),
    [
        (-0.5, "greater_than"),
        (-0.5, "greater_than_or_equal"),
        (0.0, "greater_than_or_equal"),
    ],
)
@pytest.mark.parametrize(
    "representation", ["sparse_edge_index", "dense_adjacency"]
)
def test_nonpositive_cosine_cutoffs_keep_a_disconnected_control_component(
    threshold, comparison, representation
):
    method_spec, params, arch_contract = _ready_inputs(
        representation=representation
    )
    method_spec["critical_requirements"]["param_glossary"][0][
        "paper_value"
    ] = threshold
    mechanism = method_spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]
    mechanism["construction"]["threshold"]["comparison"] = comparison
    params["params"]["similarity_threshold"]["value"] = threshold

    plan = normalize_graph_mechanism_runtime_plan(
        method_spec, params, arch_contract
    )

    assert plan["status"] == "ready"
    assert plan["fixture"]["topology_intervention"]["remove_edges"]
    assert plan["fixture"]["neighbor_intervention"]["control_targets"]
    features = np.asarray(
        plan["fixture"]["batch.static_features"], dtype=float
    )
    assert np.all(features[3:, 0] == -1.0)
    assert np.all(features[3:, 1:] == 0.0)


@pytest.mark.parametrize(
    ("metric", "comparison", "tie_break"),
    [
        ("dot_product", "greater_than_or_equal", "lowest_target_index"),
        ("cosine_similarity", "greater_than", "lowest_target_index"),
        (
            "cosine_similarity",
            "greater_than_or_equal",
            "highest_target_index",
        ),
    ],
    ids=["dot-product", "strict-comparison", "reverse-tie-break"],
)
@pytest.mark.parametrize(
    "representation", ["sparse_edge_index", "dense_adjacency"]
)
def test_full_executor_rejects_known_bad_construction_semantics(
    representation, metric, comparison, tie_break,
):
    plan = normalize_graph_mechanism_runtime_plan(
        *_cap_one_ready_inputs(representation=representation)
    )
    _attach_passing_alignment_receipt(plan)
    fixture = plan["fixture"]
    threshold_name = plan["construction"]["threshold"]["parameter"][
        "params_name"
    ]
    cap_name = plan["construction"]["cap"]["parameter"]["params_name"]

    def construct(*, feature_input, threshold, cap, seed):
        del seed
        return {
            "graph": _independent_constructed_graph(
                feature_input,
                threshold=float(threshold),
                cap=int(cap),
                representation=representation,
                metric=metric,
                comparison=comparison,
                tie_break=tie_break,
            ),
            "parameter_values": {
                threshold_name: threshold,
                cap_name: cap,
            },
        }

    def forbidden_execution(**kwargs):  # pragma: no cover - failure assertion
        raise AssertionError(f"execution ran after bad construction: {kwargs}")

    construction_identity = plan["construction"]["callable"]
    execution_identity = plan["execution"]["callable"]
    callables = {
        (
            f"{construction_identity['module']}:"
            f"{construction_identity['qualname']}"
        ): construct,
        (
            f"{execution_identity['module']}:"
            f"{execution_identity['qualname']}"
        ): forbidden_execution,
    }

    results = {
        result.probe_id: result
        for result in run_graph_mechanism_probes(plan, callables, fixture)
    }

    assert "fail" in {results["HG-2"].status, results["HG-3"].status}
    assert all(
        results[probe_id].status == "unprobeable"
        for probe_id in ("HG-4", "HG-5", "HG-6", "HG-7")
    )


@pytest.mark.parametrize("tensor_backend", ["numpy", "torch"])
@pytest.mark.parametrize(
    "representation", ["sparse_edge_index", "dense_adjacency"]
)
@pytest.mark.parametrize(
    ("declared_comparison", "implemented_comparison", "expected_status"),
    [
        (
            "greater_than_or_equal",
            "greater_than_or_equal",
            "pass",
        ),
        ("greater_than_or_equal", "greater_than", "fail"),
        ("greater_than", "greater_than", "pass"),
        ("greater_than", "greater_than_or_equal", "fail"),
    ],
    ids=["gte-good", "gte-as-gt", "gt-good", "gt-as-gte"],
)
def test_exact_comparison_witness_survives_real_float32_adapter(
    tensor_backend,
    representation,
    declared_comparison,
    implemented_comparison,
    expected_status,
):
    if tensor_backend == "torch":
        pytest.importorskip("torch")
    method_spec, params, arch_contract = _ready_inputs(
        representation=representation,
        tensor_backend=tensor_backend,
    )
    method_spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]["construction"]["threshold"]["comparison"] = declared_comparison
    plan = normalize_graph_mechanism_runtime_plan(
        method_spec, params, arch_contract
    )
    _attach_passing_alignment_receipt(plan)

    def build_article_graph(static_features, similarity_threshold):
        # Keep nominal and ordinary parameter-intervention semantics correct
        # so this test isolates the dedicated equality-boundary witness.  The
        # real adapter still carries float32 arrays through every invocation.
        observed_threshold = float(similarity_threshold)
        runtime_comparison = (
            implemented_comparison
            if observed_threshold == 1.0
            else declared_comparison
        )
        return _real_float32_constructed_graph(
            static_features,
            threshold=observed_threshold,
            representation=representation,
            comparison=runtime_comparison,
            tensor_backend=tensor_backend,
        )

    adapter = _graph_constructor_adapter(
        build_article_graph,
        plan["construction"],
        tensor_backend=tensor_backend,
    )
    assert adapter is not None
    identity = plan["construction"]["callable"]
    callable_key = f"{identity['module']}:{identity['qualname']}"
    results = {
        result.probe_id: result
        for result in run_graph_mechanism_probes(
            plan, {callable_key: adapter}, plan["fixture"]
        )
    }

    assert results["HG-2"].status == "pass"
    assert results["HG-3"].status == expected_status
    witness = results["HG-3"].evidence["comparison_witness"]
    assert witness["threshold_value"] == 1.0
    assert witness["identical_entity_positions"] == [0, 1]
    if expected_status == "pass":
        assert witness["observed_edge_symmetric_difference"] == 0
    else:
        assert results["HG-3"].evidence["reason_code"] == (
            "graph_comparison_semantics_disagree"
        )
        assert witness["observed_edge_symmetric_difference"] == 2


def test_groundings_use_exact_refs_and_exact_bound_callables():
    plan = normalize_graph_mechanism_runtime_plan(*_ready_inputs())
    groundings = plan["groundings"]

    assert set(groundings) == {
        ALIGNMENT_PROBE_REF,
        *CANONICAL_PROBE_REFS.values(),
    }
    assert groundings[ALIGNMENT_PROBE_REF]["bound_callables"] == [
        "method.training:_prepare_graph_batch"
    ]
    for ref in (
        CANONICAL_PROBE_REFS["parameter_agreement"],
        CANONICAL_PROBE_REFS["construction"],
    ):
        assert groundings[ref]["bound_callables"] == [
            "method.model:build_article_graph"
        ]
    for ref in (
        CANONICAL_PROBE_REFS["topology"],
        CANONICAL_PROBE_REFS["neighbor_signal"],
        CANONICAL_PROBE_REFS["permutation"],
    ):
        assert groundings[ref]["bound_callables"] == [
            "method.model:encode_graph"
        ]


def test_absent_contribution_control_preserves_liveness_plan_without_hg7():
    method_spec, params, arch_contract = _ready_inputs()
    contract = method_spec["methodology_replication_contract"]
    mechanism = contract["homogeneous_graph_mechanism"]
    mechanism["contribution_ablation"] = None
    mechanism["probe_refs"]["contribution_ablation"] = None
    contract["elements"] = [
        element
        for element in contract["elements"]
        if element["element_id"] != "graph-ablation-control"
    ]

    plan = normalize_graph_mechanism_runtime_plan(
        method_spec, params, arch_contract
    )

    assert plan["status"] == "ready"
    assert plan["ablation"] is None
    assert CANONICAL_PROBE_REFS["contribution_ablation"] not in plan[
        "groundings"
    ]
    assert CANONICAL_PROBE_REFS["topology"] in plan["groundings"]
    assert CANONICAL_PROBE_REFS["neighbor_signal"] in plan["groundings"]


def test_contribution_discriminator_round_trips_without_planner_invention():
    method_spec, params, arch_contract = _ready_inputs()
    mechanism = method_spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]
    mechanism["contribution_ablation"][
        "discriminating_probe_ref"
    ] = CANONICAL_PROBE_REFS["topology"]

    plan = normalize_graph_mechanism_runtime_plan(
        method_spec, params, arch_contract
    )

    assert plan["ablation"]["discriminating_probe_ref"] == (
        CANONICAL_PROBE_REFS["topology"]
    )


def test_message_helper_output_root_owns_its_axis_not_final_inference_root():
    method_spec, params, arch_contract = _ready_inputs()
    mechanism = method_spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]
    mechanism["message_passing"]["output_root"] = "outputs.graph_embeddings"
    arch_contract["relational_indexing"]["coindexed_roots"][
        "outputs.graph_embeddings"
    ] = 1

    plan = normalize_graph_mechanism_runtime_plan(
        method_spec, params, arch_contract
    )

    assert plan["execution"]["output_root"] == "outputs.graph_embeddings"
    assert plan["execution"]["output_entity_axis"] == 1
    assert arch_contract["relational_indexing"]["execution"]["inference"][
        "output_coindexed_root"
    ] == "outputs.forecasts"


def test_non_coindexed_message_helper_output_root_fails_closed():
    method_spec, params, arch_contract = _ready_inputs()
    method_spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]["message_passing"]["output_root"] = "outputs.nearby"

    with pytest.raises(
        GraphMechanismRuntimePlanError,
        match="message-passing output root.*not coindexed",
    ):
        normalize_graph_mechanism_runtime_plan(
            method_spec, params, arch_contract
        )


def test_message_helper_output_root_cannot_borrow_an_inference_input_root():
    method_spec, params, arch_contract = _ready_inputs()
    method_spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]["message_passing"]["output_root"] = "batch.demand"

    with pytest.raises(
        GraphMechanismRuntimePlanError,
        match="inference input root",
    ):
        normalize_graph_mechanism_runtime_plan(
            method_spec, params, arch_contract
        )


@pytest.mark.parametrize("qualname", ["_hidden", "for"])
def test_raw_plan_rejects_unnameable_graph_callable(qualname):
    method_spec, params, arch_contract = _ready_inputs()
    method_spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]["construction"]["callable"]["qualname"] = qualname

    with pytest.raises(
        GraphMechanismRuntimePlanError,
        match="public top-level helper",
    ):
        normalize_graph_mechanism_runtime_plan(
            method_spec, params, arch_contract
        )


def test_raw_plan_rejects_constructor_callable_parameter_collisions():
    method_spec, params, arch_contract = _ready_inputs()
    construction = method_spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]["construction"]
    construction["feature_parameter"] = construction["threshold"]["parameter"][
        "callable_parameter"
    ]

    with pytest.raises(
        GraphMechanismRuntimePlanError,
        match="pairwise-distinct callable parameters",
    ):
        normalize_graph_mechanism_runtime_plan(
            method_spec, params, arch_contract
        )


def test_callable_ablation_preserves_its_own_exact_bindings():
    method_spec, params, arch_contract = _ready_inputs()
    mechanism = method_spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]
    mechanism["contribution_ablation"] = {
        "kind": "non_graph_decoder",
        "element_id": "graph-ablation-control",
        "callable": {
            "module": "method.model",
            "qualname": "decode_without_messages",
        },
        "graph_parameter": "null_graph",
        "neighbor_signal_parameter": "null_signals",
        "output_root": "outputs.forecasts",
        "discriminating_probe_ref": CANONICAL_PROBE_REFS["neighbor_signal"],
    }

    plan = normalize_graph_mechanism_runtime_plan(
        method_spec, params, arch_contract
    )

    assert plan["ablation"]["graph_parameter"] == "null_graph"
    assert plan["ablation"]["neighbor_signal_parameter"] == "null_signals"
    assert plan["ablation"]["output_root"] == plan["execution"]["output_root"]


def test_entity_axis_outside_v1_is_an_honest_unprobeable_disposition():
    method_spec, params, arch_contract = _ready_inputs()
    mechanism = method_spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]
    mechanism["message_passing"]["output_root"] = "outputs.graph_embeddings"
    arch_contract["relational_indexing"]["coindexed_roots"][
        "outputs.graph_embeddings"
    ] = 2

    plan = normalize_graph_mechanism_runtime_plan(
        method_spec, params, arch_contract
    )

    assert plan["status"] == "unprobeable"
    assert plan["reason"] == "unsupported_graph_entity_axis"
    assert plan["unsupported_axes"] == {"output": 2}


def test_stronger_runtime_value_preserves_carrier_paper_truth():
    method_spec, params, arch_contract = _ready_inputs()
    params["params"]["similarity_threshold"].update(
        {
            "value": 0.7,
            "source": "system_default",
            "paper_value": 0.5,
            "reasoning": "Use a sparser graph at smoke scale.",
        }
    )

    plan = normalize_graph_mechanism_runtime_plan(
        method_spec, params, arch_contract
    )

    binding = plan["parameter_authority"]["bindings"][
        "similarity_threshold"
    ]
    assert plan["parameter_authority"]["status"] == "pass"
    assert binding["carrier_value"] == 0.5
    assert binding["paper_value"] == 0.5
    assert binding["params_value"] == 0.7
    assert plan["construction"]["threshold"]["value"] == 0.7


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing", "params_entry_missing_or_suppressed:similarity_threshold"),
        ("fuzzy", "parameter_carrier_missing:similarity_threshold"),
        ("disagreed", "parameter_value_disagrees:similarity_threshold"),
        ("duplicate", "parameter_carrier_duplicated:similarity_threshold"),
    ],
)
def test_parameter_authority_failure_stays_executable_for_hg2(mutation, reason):
    method_spec, params, arch_contract = _ready_inputs()
    if mutation == "missing":
        del params["params"]["similarity_threshold"]
    elif mutation == "fuzzy":
        method_spec["critical_requirements"]["param_glossary"][0][
            "aliases"
        ] = ["similarity_threshold_nearby"]
    elif mutation == "disagreed":
        params["params"]["similarity_threshold"]["value"] = 0.75
    else:
        method_spec["critical_requirements"][
            "scale_dependent_hyperparameters"
        ].append(
            {"name": "similarity_threshold", "paper_value": 0.5}
        )

    plan = normalize_graph_mechanism_runtime_plan(
        method_spec, params, arch_contract
    )

    assert plan["status"] == "ready"
    assert plan["reason"] == reason
    assert plan["parameter_authority"]["status"] == "fail"
    assert plan["parameter_authority"]["reason"] == reason
    assert plan["construction"]["threshold"]["parameter"][
        "params_name"
    ] == "similarity_threshold"
    # Parameter failure blocks construction but must not erase R2C-084's
    # independent identity carrier used by the HG-1 runtime receipt.
    assert plan["fixture"]["entity_ids"] == [101, 205, 309, 412, 518]


def test_graph_free_and_unsupported_relational_dispositions_are_distinct():
    graph_free = normalize_graph_mechanism_runtime_plan(
        {"comparison": {"classification": {"id": "time_series_forecasting"}}},
        {"params": {}},
        _graph_free_contract(),
    )
    assert graph_free["status"] == "not_applicable"
    assert graph_free["reason"] == "graph_free_method"

    unsupported_spec = {
        "methodology_replication_contract": {
            "elements": [
                _element(
                    "sampled-neighbor-graph",
                    [],
                    relational={
                        "kind": "unsupported",
                        "unsupported_kind": "sampled_neighbor_graph",
                    },
                )
            ]
        }
    }
    unsupported = normalize_graph_mechanism_runtime_plan(
        unsupported_spec, {"params": {}}, _graph_free_contract()
    )
    assert unsupported["status"] == "unprobeable"
    assert unsupported["reason"] == "unsupported_relational_structure"
    assert unsupported["groundings"]["unsupported_relational_elements"] == [
        {
            "element_id": "sampled-neighbor-graph",
            "unsupported_kind": "sampled_neighbor_graph",
        }
    ]


def test_archived_homogeneous_contract_without_mechanism_is_unprobeable():
    method_spec, params, arch_contract = _ready_inputs()
    del method_spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]

    plan = normalize_graph_mechanism_runtime_plan(
        method_spec, params, arch_contract
    )

    assert plan["status"] == "unprobeable"
    assert plan["reason"] == "homogeneous_graph_mechanism_contract_missing"


def test_alignment_must_resolve_before_any_mechanism_plan():
    method_spec, params, _ = _ready_inputs()

    plan = normalize_graph_mechanism_runtime_plan(
        method_spec, params, _graph_free_contract()
    )

    assert plan["status"] == "blocked"
    assert plan["reason"] == "relational_alignment_missing"
    assert plan["alignment"] == {
        "status": "blocked",
        "reason": "relational_alignment_missing",
        "element_id": "graph-alignment",
    }
    assert plan["construction"] is None


def test_source_contract_cannot_silently_change_probe_ref_or_semantics():
    method_spec, params, arch_contract = _ready_inputs()
    method_spec["methodology_replication_contract"][
        "homogeneous_graph_mechanism"
    ]["probe_refs"]["topology"] = "graph_mechanism.topology_nearby"

    with pytest.raises(
        GraphMechanismRuntimePlanError,
        match="must equal the canonical graph refs",
    ):
        normalize_graph_mechanism_runtime_plan(
            method_spec, params, arch_contract
        )


@pytest.mark.parametrize(
    "representation", ["sparse_edge_index", "dense_adjacency"]
)
def test_frozen_plan_flows_directly_into_executor_after_runtime_alignment_receipt(
    representation,
):
    plan = normalize_graph_mechanism_runtime_plan(
        *_ready_inputs(representation=representation)
    )
    plan["alignment"] = {
        **plan["alignment"],
        "status": "pass",
        "reason": "matching Stage 2.d runtime-validation receipt",
    }
    fixture = plan["fixture"]
    plan["alignment_runtime_receipt"] = {
        "receipt_version": "1.0",
        "probe_ref": ALIGNMENT_PROBE_REF,
        "status": "pass",
        "reason": "stage_2d_runtime_alignment_verified",
        "verified": True,
        "validator": "validate_arch_contract_runtime.py",
        "element_id": plan["alignment"]["element_id"],
        "fixture_scope": fixture["scope"],
        "representation": representation,
        "entity_ids": fixture["entity_ids"],
        "authority_digests": {
            "method_spec.json": f"sha256:{'a' * 64}",
            "arch_contract.json": f"sha256:{'b' * 64}",
        },
    }
    threshold = plan["construction"]["threshold"]
    cap = plan["construction"]["cap"]
    threshold_name = threshold["parameter"]["params_name"]

    def construct(*, feature_input, threshold, cap, seed):
        del seed
        graph = _independent_constructed_graph(
            feature_input,
            threshold=float(threshold),
            cap=int(cap) if cap is not None else None,
            representation=representation,
            comparison=plan["construction"]["threshold"]["comparison"],
        )
        runtime_values = {threshold_name: threshold}
        if plan["construction"]["cap"]["kind"] != "none":
            cap_name = plan["construction"]["cap"]["parameter"][
                "params_name"
            ]
            runtime_values[cap_name] = cap
        return {
            "graph": copy.deepcopy(graph),
            "parameter_values": runtime_values,
        }

    def execute(*, graph, neighbor_signal, entity_ids, seed):
        del entity_ids, seed
        signals = np.asarray(neighbor_signal, dtype=float)
        output = signals.copy()
        graph_array = np.asarray(graph)
        if plan["representation"] == "sparse_edge_index":
            edges = zip(graph_array[0], graph_array[1])
        else:
            edges = zip(*np.nonzero(graph_array))
        for source, target in edges:
            if int(source) != int(target):
                output[int(target)] += signals[int(source)]
        return output

    construction_identity = plan["construction"]["callable"]
    execution_identity = plan["execution"]["callable"]
    callables = {
        (
            f"{construction_identity['module']}:"
            f"{construction_identity['qualname']}"
        ): construct,
        (
            f"{execution_identity['module']}:"
            f"{execution_identity['qualname']}"
        ): execute,
    }

    results = run_graph_mechanism_probes(plan, callables, fixture)

    assert [result.probe_id for result in results] == [
        "HG-2",
        "HG-3",
        "HG-4",
        "HG-5",
        "HG-6",
        "HG-7",
    ]
    assert [result.status for result in results] == ["pass"] * 6


def test_plan_derives_markers_and_refs_the_stored_json_omits():
    # R2C-092: markers on the block's own elements and per-element graph
    # refs are derived wiring; a stored spec whose producer never
    # transcribed them still freezes the same ready plan.
    method_spec, params, arch_contract = _ready_inputs()
    for element in method_spec["methodology_replication_contract"]["elements"]:
        element["relational_structure"] = None
        element["verification_probe_refs"] = []

    plan = normalize_graph_mechanism_runtime_plan(
        method_spec, params, arch_contract
    )

    assert plan["status"] == "ready"
    groundings = plan["groundings"]
    assert groundings[ALIGNMENT_PROBE_REF]["element_ids"] == [
        "graph-alignment"
    ]
    assert groundings[CANONICAL_PROBE_REFS["construction"]][
        "element_ids"
    ] == ["graph-construction"]
    assert set(plan["alignment"]["methodology_element_ids"]) == {
        "graph-alignment", "graph-construction", "graph-message-passing",
    }
