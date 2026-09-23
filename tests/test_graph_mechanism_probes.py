"""Behavioral coverage for the frozen homogeneous-graph probe executor."""

from __future__ import annotations

import copy
import json
import random

import numpy as np
import pytest

from probes.graph_mechanism import (
    PROBE_REFS,
    _numeric_array,
    execution_plan_digest,
    run_graph_mechanism_probes,
)


def test_executor_rejects_ndarray_subclass_dispatch():
    class DispatchArray(np.ndarray):
        pass

    value = np.asarray([[1.0], [2.0]]).view(DispatchArray)

    array, problem = _numeric_array(value, label="graph mechanism output")

    assert array is None
    assert problem is not None
    assert "ndarray subclass dispatch" in problem


def test_executor_rejects_tensor_subclass_dispatch():
    torch = pytest.importorskip("torch")

    class DispatchTensor(torch.Tensor):
        pass

    value = torch.ones((2, 1)).as_subclass(DispatchTensor)

    array, problem = _numeric_array(value, label="graph mechanism output")

    assert array is None
    assert problem is not None
    assert "Tensor subclass dispatch" in problem


_DENSE_GRAPH = np.asarray(
    [
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=float,
)
_ALTERNATE_DENSE_GRAPH = np.asarray(
    [
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ],
    dtype=float,
)


def _sparse_graph(dense: np.ndarray = _DENSE_GRAPH) -> np.ndarray:
    sources, destinations = np.nonzero(dense)
    return np.asarray([sources, destinations], dtype=np.int64)


def _graph_edges(graph: object) -> list[tuple[int, int]]:
    array = np.asarray(graph)
    if array.shape[0] == 2 and array.shape != _DENSE_GRAPH.shape:
        return [
            (int(source), int(destination))
            for source, destination in zip(array[0], array[1])
        ]
    sources, destinations = np.nonzero(array)
    return [
        (int(source), int(destination))
        for source, destination in zip(sources, destinations)
    ]


def _message_passing(*, graph, neighbor_signal, entity_ids, seed):
    del seed
    assert len(entity_ids) == len(neighbor_signal)
    signals = np.asarray(neighbor_signal, dtype=float)
    output = signals.copy()
    for source, destination in _graph_edges(graph):
        output[destination] += signals[source]
    return output


def _ignores_graph(*, graph, neighbor_signal, entity_ids, seed):
    del graph, entity_ids, seed
    return np.asarray(neighbor_signal, dtype=float).copy()


def _topology_only(*, graph, neighbor_signal, entity_ids, seed):
    del neighbor_signal, entity_ids, seed
    output = np.zeros((4, 1), dtype=float)
    for _, destination in _graph_edges(graph):
        output[destination, 0] += 1.0
    return output


def _position_leaking_message_passing(*, graph, neighbor_signal, entity_ids, seed):
    output = _message_passing(
        graph=graph,
        neighbor_signal=neighbor_signal,
        entity_ids=entity_ids,
        seed=seed,
    )
    return output + np.arange(len(entity_ids), dtype=float)[:, None]


def _two_hop_message_passing(*, graph, neighbor_signal, entity_ids, seed):
    first = _message_passing(
        graph=graph,
        neighbor_signal=neighbor_signal,
        entity_ids=entity_ids,
        seed=seed,
    )
    return _message_passing(
        graph=graph,
        neighbor_signal=first,
        entity_ids=entity_ids,
        seed=seed,
    )


def _global_graph_aggregator(*, graph, neighbor_signal, entity_ids, seed):
    del seed
    signals = np.asarray(neighbor_signal, dtype=float)
    graph_scale = float(len(_graph_edges(graph)))
    aggregate = graph_scale * float(np.sum(signals))
    return np.full_like(signals, aggregate)


def _weighted_message_passing(*, graph, neighbor_signal, entity_ids, seed):
    del entity_ids, seed
    adjacency = np.asarray(graph, dtype=float)
    signals = np.asarray(neighbor_signal, dtype=float)
    output = signals.copy()
    sources, destinations = np.nonzero(adjacency)
    for source, destination in zip(sources, destinations):
        output[destination] += adjacency[source, destination] * signals[source]
    return output


def _shape_changing_null(*, graph, neighbor_signal, entity_ids, seed):
    del graph, entity_ids, seed
    signals = np.asarray(neighbor_signal, dtype=float)
    if float(signals[0, 0]) != 1.0:
        return np.zeros((signals.shape[0] - 1, signals.shape[1]), dtype=float)
    return signals.copy()


_DEFAULT_ABLATION = object()


def _plan(
    *,
    representation: str = "dense_adjacency",
    permutation_applicability: str = "required",
    ablation: dict | None | object = _DEFAULT_ABLATION,
) -> dict:
    frozen_fixture = json.loads(json.dumps(
        _fixture(representation=representation),
        default=lambda value: value.tolist(),
        allow_nan=False,
    ))
    chosen_ablation = (
        {
            "kind": "empty_graph",
            "element_id": "element:graph-ablation",
            "discriminating_probe_ref": PROBE_REFS["HG-5"],
        }
        if ablation is _DEFAULT_ABLATION else ablation
    )
    groundings = {
        probe_ref: {
            "element_ids": [f"element:{probe_id.lower()}"],
            "bound_callables": ["fixture.graph:execute"],
        }
        for probe_id, probe_ref in PROBE_REFS.items()
    }
    plan = {
        "schema_version": "1.0",
        "status": "ready",
        "reason": "paper declares a homogeneous graph mechanism",
        "representation": representation,
        "alignment": {
            "status": "pass",
            "reason": "stable entity order was established",
            "element_id": "element:graph-alignment",
        },
        "construction": {
            "callable": {
                "module": "fixture.graph",
                "qualname": "construct",
            },
            "feature_input_root": "static_features",
            "feature_parameter": "features",
            "output_selector": {"kind": "direct"},
            "threshold": {
                "metric": "cosine_similarity",
                "comparison": "greater_than_or_equal",
                "parameter": {
                    "params_name": "similarity_threshold",
                    "callable_parameter": "threshold",
                },
                "value": 0.5,
            },
            "cap": {"kind": "none"},
            "self_loop_policy": "forbidden",
            "direction_policy": "undirected_bidirectional",
        },
        "parameter_authority": {
            "status": "pass",
            "reason": "one live parameter authority",
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
        },
        "execution": {
            "callable": {
                "module": "fixture.graph",
                "qualname": "execute",
            },
            "neighbor_signal_root": "node_signals",
            "neighbor_signal_parameter": "node_signals",
            "entity_axis": 0,
            "output_root": "graph_outputs",
            "output_entity_axis": 0,
            "permutation_applicability": permutation_applicability,
            "tolerance": 1e-8,
            "seed": 1729,
        },
        "ablation": chosen_ablation,
        "groundings": groundings,
        "fixture": frozen_fixture,
    }
    if chosen_ablation is None:
        plan["groundings"].pop(PROBE_REFS["HG-7"])
    plan["alignment_runtime_receipt"] = {
        "receipt_version": "1.0",
        "probe_ref": "graph_mechanism.alignment_prerequisite",
        "status": "pass",
        "reason": "stage_2d_runtime_alignment_verified",
        "verified": True,
        "validator": "validate_arch_contract_runtime.py",
        "element_id": "element:graph-alignment",
        "fixture_scope": "asymmetric-four-node-star",
        "representation": representation,
        "entity_ids": ["entity-a", "entity-b", "entity-c", "entity-d"],
        "authority_digests": {
            "method_spec.json": f"sha256:{'a' * 64}",
            "arch_contract.json": f"sha256:{'b' * 64}",
        },
    }
    return plan


def _fixture(*, representation: str = "dense_adjacency") -> dict:
    graph = (
        _DENSE_GRAPH.copy()
        if representation == "dense_adjacency"
        else _sparse_graph()
    )
    alternate_graph = (
        _ALTERNATE_DENSE_GRAPH.copy()
        if representation == "dense_adjacency"
        else _sparse_graph(_ALTERNATE_DENSE_GRAPH)
    )
    return {
        "scope": "asymmetric-four-node-star",
        "entity_ids": ["entity-a", "entity-b", "entity-c", "entity-d"],
        "static_features": np.asarray(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]]
        ),
        "node_signals": np.asarray([[1.0], [2.0], [4.0], [8.0]]),
        "comparison_witness": {
            "metric": "cosine_similarity",
            "comparison": "greater_than_or_equal",
            "threshold_value": 1.0,
            "entity_axis": 0,
            "feature_input": np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ),
            "expected_graph": alternate_graph.copy(),
        },
        "expected_graph": graph.copy(),
        "parameter_interventions": {
            "similarity_threshold": {
                "nominal_value": 0.5,
                "alternate_value": 0.75,
                "value": 0.75,
                "callable_parameter": "threshold",
                "expected_relation": "graph_must_differ",
                "expected_graph": alternate_graph,
            }
        },
        "topology_intervention": {
            "remove_edges": [[0, 1]],
            "targets": [1],
        },
        "neighbor_intervention": {
            "source": 0,
            "target": 1,
            "delta": 5.0,
            "control_targets": [2, 3],
        },
        # permutation[new position] = old position
        "permutation": [2, 0, 3, 1],
    }


def _constructor(
    graph: object,
    *,
    alternate_graph: object,
    captured_threshold: float | None = None,
    hardcoded: bool = False,
):
    def construct(*, feature_input, threshold, cap, seed):
        del cap, seed
        assert np.asarray(feature_input).shape[0] == 4
        selected = graph if hardcoded or float(threshold) == 0.5 \
            else alternate_graph
        return {
            "graph": copy.deepcopy(selected),
            "parameter_values": {
                "similarity_threshold": (
                    threshold if captured_threshold is None else captured_threshold
                ),
            },
        }

    return construct


def _callables(
    fixture: dict,
    *,
    execute=_message_passing,
    captured_threshold: float | None = None,
    null_execute=None,
) -> dict:
    result = {
        "fixture.graph:construct": _constructor(
            fixture["expected_graph"],
            alternate_graph=fixture["parameter_interventions"][
                "similarity_threshold"
            ]["expected_graph"],
            captured_threshold=captured_threshold,
        ),
        "fixture.graph:execute": execute,
    }
    if null_execute is not None:
        result["fixture.graph:null_execute"] = null_execute
    return result


def _by_id(results):
    return {result.probe_id: result for result in results}


def test_graph_free_plan_is_not_applicable_without_invoking_adapters():
    def forbidden(**kwargs):  # pragma: no cover - failure path assertion
        raise AssertionError(f"adapter was invoked: {kwargs}")

    results = run_graph_mechanism_probes(
        {
            "schema_version": "1.0",
            "status": "not_applicable",
            "reason": "method is graph-free",
            "groundings": {},
        },
        {"fixture.graph:execute": forbidden},
        {},
    )

    assert [result.probe_id for result in results] == list(PROBE_REFS)
    assert {result.status for result in results} == {"not_applicable"}
    assert {result.reason for result in results} == {"method is graph-free"}


@pytest.mark.parametrize("plan_status", ["unprobeable", "blocked"])
def test_unsupported_plan_is_unprobeable_without_invoking_adapters(plan_status):
    def forbidden(**kwargs):  # pragma: no cover - failure path assertion
        raise AssertionError(f"adapter was invoked: {kwargs}")

    results = run_graph_mechanism_probes(
        {
            "schema_version": "1.0",
            "status": plan_status,
            "reason": "no stable graph callable binding",
            "groundings": {},
        },
        {"fixture.graph:execute": forbidden},
        {},
    )

    assert {result.status for result in results} == {"unprobeable"}
    assert {result.reason for result in results} == {
        "no stable graph callable binding"
    }


@pytest.mark.parametrize("representation", ["dense_adjacency", "sparse_edge_index"])
def test_known_good_graph_passes_every_behavioral_rung(representation):
    plan = _plan(representation=representation)
    fixture = _fixture(representation=representation)

    results = run_graph_mechanism_probes(plan, _callables(fixture), fixture)

    assert [result.probe_id for result in results] == [
        "HG-2", "HG-3", "HG-4", "HG-5", "HG-6", "HG-7"
    ]
    assert [result.status for result in results] == ["pass"] * 6
    assert [result.probe_ref for result in results] == list(PROBE_REFS.values())
    assert all(result.element_ids for result in results)
    assert all(result.bound_callables for result in results)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda fixture: fixture.update(scope="drifted-scope"),
        lambda fixture: fixture["static_features"][0].__setitem__(0, 9.0),
        lambda fixture: fixture["node_signals"][0].__setitem__(0, 9.0),
        lambda fixture: fixture["parameter_interventions"][
            "similarity_threshold"
        ].update(value=0.8),
        lambda fixture: fixture["topology_intervention"].update(targets=[2]),
        lambda fixture: fixture["neighbor_intervention"].update(delta=6.0),
        lambda fixture: fixture.update(permutation=[1, 0, 2, 3]),
    ],
    ids=[
        "scope",
        "features",
        "signals",
        "parameter-intervention",
        "topology-intervention",
        "neighbor-intervention",
        "permutation",
    ],
)
def test_any_injected_fixture_drift_is_rejected_before_adapters(mutation):
    plan = _plan()
    fixture = _fixture()
    mutation(fixture)

    def forbidden(**kwargs):  # pragma: no cover - failure path assertion
        raise AssertionError(f"adapter was invoked: {kwargs}")

    results = run_graph_mechanism_probes(
        plan,
        {
            "fixture.graph:construct": forbidden,
            "fixture.graph:execute": forbidden,
        },
        fixture,
    )

    assert {result.status for result in results} == {"unprobeable"}
    assert all("differs from the frozen fixture" in result.reason for result in results)


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    [
        (
            lambda fixture: fixture.pop("comparison_witness"),
            "comparison_witness_missing",
        ),
        (
            lambda fixture: fixture["comparison_witness"].update(
                threshold_value=0.999
            ),
            "comparison_witness_boundary_invalid",
        ),
    ],
    ids=["removed", "tampered-boundary"],
)
def test_ready_plan_cannot_certify_without_exact_comparison_witness(
    mutation, reason_code,
):
    plan = _plan()
    fixture = _fixture()
    mutation(plan["fixture"])
    mutation(fixture)

    results = _by_id(run_graph_mechanism_probes(
        plan, _callables(fixture), fixture
    ))

    assert results["HG-2"].status == "pass"
    assert results["HG-3"].status == "unprobeable"
    assert results["HG-3"].evidence["reason_code"] == reason_code
    assert all(
        results[probe_id].status == "unprobeable"
        for probe_id in ("HG-4", "HG-5", "HG-6", "HG-7")
    )


def test_exact_torch_tensorized_fixture_copy_preserves_frozen_authority():
    torch = pytest.importorskip("torch")
    plan = _plan(representation="sparse_edge_index")
    fixture = _fixture(representation="sparse_edge_index")
    fixture["static_features"] = torch.tensor(
        fixture["static_features"], dtype=torch.float32
    )
    fixture["node_signals"] = torch.tensor(
        fixture["node_signals"], dtype=torch.float32
    )
    fixture["expected_graph"] = torch.tensor(
        fixture["expected_graph"], dtype=torch.long
    )
    fixture["parameter_interventions"]["similarity_threshold"][
        "expected_graph"
    ] = torch.tensor(
        fixture["parameter_interventions"]["similarity_threshold"][
            "expected_graph"
        ],
        dtype=torch.long,
    )

    results = run_graph_mechanism_probes(
        plan, _callables(fixture), fixture
    )

    assert [result.status for result in results] == ["pass"] * 6


def test_every_applicable_result_carries_frozen_trace_and_observed_effect():
    plan = _plan()
    fixture = _fixture()

    raw_results = run_graph_mechanism_probes(
        plan, _callables(fixture), fixture
    )
    json.dumps([result.to_dict() for result in raw_results], allow_nan=False)
    results = _by_id(raw_results)
    expected_plan_digest = execution_plan_digest(plan)

    for result in results.values():
        trace = result.evidence["trace"]
        assert trace["execution_plan_digest"] == expected_plan_digest
        assert trace["fixture_scope"] == "asymmetric-four-node-star"
        assert trace["fixture_digest"].startswith("sha256:")
        assert trace["construction_feature_root"] == "static_features"
        assert trace["neighbor_signal_root"] == "node_signals"
        assert trace["output_root"] == "graph_outputs"
        assert trace["seed"] == 1729
        assert trace["tolerance"] == pytest.approx(1e-8)
    assert results["HG-2"].evidence["parameter_interventions"][0][
        "observed_nominal_edge_symmetric_difference"
    ] > 0
    assert results["HG-2"].evidence["parameters"][0]["suppressed"] is False
    assert results["HG-2"].evidence["parameters"][0]["duplicates"] == []
    assert results["HG-2"].evidence["parameter_interventions"][0][
        "callable_parameter"
    ] == "threshold"
    assert results["HG-3"].evidence[
        "observed_edge_symmetric_difference"
    ] == 0
    assert results["HG-4"].evidence["maximum_target_delta"] > 0
    assert results["HG-5"].evidence["target_max_abs_delta"] > 0
    assert results["HG-6"].evidence["restored_max_abs_delta"] == pytest.approx(0)
    for probe_id in ("HG-4", "HG-5", "HG-6"):
        evidence = results[probe_id].evidence
        assert evidence["effect_scale"] == max(
            1.0,
            evidence["baseline_max_abs"],
            evidence["changed_max_abs"],
        )
        assert evidence["effect_threshold"] == (
            evidence["tolerance"] * evidence["effect_scale"]
        )
    assert results["HG-7"].evidence["real_evidence"][
        "target_max_abs_delta"
    ] > 0
    for arm in ("real_evidence", "null_evidence"):
        evidence = results["HG-7"].evidence[arm]
        assert evidence["effect_threshold"] == (
            evidence["tolerance"] * evidence["effect_scale"]
        )


def test_execution_plan_digest_changes_with_plan_authority():
    plan = _plan()
    changed = copy.deepcopy(plan)
    changed["execution"]["seed"] += 1

    assert execution_plan_digest(changed) != execution_plan_digest(plan)


def test_missing_paper_justified_ablation_keeps_liveness_but_not_contribution():
    plan = _plan(ablation=None)
    fixture = _fixture()

    results = _by_id(run_graph_mechanism_probes(
        plan, _callables(fixture), fixture
    ))

    assert all(
        results[probe_id].status == "pass"
        for probe_id in ("HG-2", "HG-3", "HG-4", "HG-5", "HG-6")
    )
    assert results["HG-7"].status == "unprobeable"
    assert results["HG-7"].evidence["reason_code"] == (
        "contribution_ablation_missing"
    )
    assert results["HG-7"].element_ids == []
    assert results["HG-7"].bound_callables == []


def test_runtime_parameter_capture_disagreement_fails_before_construction():
    plan = _plan()
    fixture = _fixture()

    results = _by_id(run_graph_mechanism_probes(
        plan,
        _callables(fixture, captured_threshold=0.75),
        fixture,
    ))

    assert results["HG-2"].status == "fail"
    assert results["HG-2"].evidence["reason_code"] == (
        "graph_parameter_authority_disagreement"
    )
    assert {results[probe_id].status for probe_id in list(PROBE_REFS)[1:]} == {
        "unprobeable"
    }


def test_parameter_intervention_rejects_hardcoded_nominal_construction():
    plan = _plan()
    fixture = _fixture()
    alternate = fixture["parameter_interventions"]["similarity_threshold"][
        "expected_graph"
    ]
    good = _by_id(run_graph_mechanism_probes(
        plan,
        _callables(fixture),
        fixture,
    ))
    bad_callables = _callables(fixture)
    bad_callables["fixture.graph:construct"] = _constructor(
        fixture["expected_graph"],
        alternate_graph=alternate,
        hardcoded=True,
    )
    bad = _by_id(run_graph_mechanism_probes(plan, bad_callables, fixture))

    assert good["HG-2"].status == "pass"
    assert good["HG-3"].status == "pass"
    assert bad["HG-2"].status == "fail"
    assert bad["HG-2"].evidence["reason_code"] == "graph_parameter_not_consumed"
    assert bad["HG-3"].status == "unprobeable"


def test_stateful_call_order_constructor_cannot_certify_without_replay():
    plan = _plan()
    fixture = _fixture()
    calls = 0

    def call_order_construct(*, feature_input, threshold, cap, seed):
        nonlocal calls
        del feature_input, cap, seed
        calls += 1
        graph = (
            fixture["expected_graph"]
            if calls == 1
            else fixture["parameter_interventions"][
                "similarity_threshold"
            ]["expected_graph"]
        )
        return {
            "graph": graph.copy(),
            "parameter_values": {"similarity_threshold": threshold},
        }

    callables = _callables(fixture)
    callables["fixture.graph:construct"] = call_order_construct

    results = _by_id(run_graph_mechanism_probes(plan, callables, fixture))

    assert "pass" not in {
        results["HG-2"].status,
        results["HG-3"].status,
    }
    assert all(
        results[probe_id].status == "unprobeable"
        for probe_id in ("HG-4", "HG-5", "HG-6", "HG-7")
    )


def test_seeded_constructor_replays_but_ambient_randomness_cannot_certify():
    plan = _plan()
    fixture = _fixture()

    def seeded_construct(*, feature_input, threshold, cap, seed):
        del feature_input, cap
        nominal = threshold == 0.5
        choose_correct = int(np.random.default_rng(seed).integers(0, 2)) == 1
        if choose_correct == nominal:
            graph = fixture["expected_graph"]
        else:
            graph = fixture["parameter_interventions"][
                "similarity_threshold"
            ]["expected_graph"]
        return {
            "graph": graph.copy(),
            "parameter_values": {"similarity_threshold": threshold},
        }

    seeded_callables = _callables(fixture)
    seeded_callables["fixture.graph:construct"] = seeded_construct
    seeded = _by_id(run_graph_mechanism_probes(
        plan, seeded_callables, fixture
    ))
    assert [result.status for result in seeded.values()] == ["pass"] * 6

    ambient = np.random.default_rng(1)

    def ambient_construct(*, feature_input, threshold, cap, seed):
        del feature_input, cap, seed
        graph = (
            fixture["expected_graph"]
            if int(ambient.integers(0, 2)) == 0
            else fixture["parameter_interventions"][
                "similarity_threshold"
            ]["expected_graph"]
        )
        return {
            "graph": graph.copy(),
            "parameter_values": {"similarity_threshold": threshold},
        }

    ambient_callables = _callables(fixture)
    ambient_callables["fixture.graph:construct"] = ambient_construct
    uncontrolled = _by_id(run_graph_mechanism_probes(
        plan, ambient_callables, fixture
    ))

    assert "pass" not in {
        uncontrolled["HG-2"].status,
        uncontrolled["HG-3"].status,
    }
    assert all(
        uncontrolled[probe_id].status == "unprobeable"
        for probe_id in ("HG-4", "HG-5", "HG-6", "HG-7")
    )


def test_nominal_constructor_is_replayed_after_parameter_interventions():
    plan = _plan()
    fixture = _fixture()
    calls = 0

    def shifts_after_variant(*, feature_input, threshold, cap, seed):
        nonlocal calls
        del feature_input, cap, seed
        calls += 1
        graph = (
            fixture["expected_graph"]
            if calls <= 2
            else fixture["parameter_interventions"][
                "similarity_threshold"
            ]["expected_graph"]
        )
        return {
            "graph": graph.copy(),
            "parameter_values": {"similarity_threshold": threshold},
        }

    callables = _callables(fixture)
    callables["fixture.graph:construct"] = shifts_after_variant
    results = _by_id(run_graph_mechanism_probes(plan, callables, fixture))

    assert results["HG-2"].status == "unprobeable"
    assert results["HG-2"].evidence["reason_code"] == (
        "construction_nominal_replay_disagrees"
    )
    assert all(
        results[probe_id].status == "unprobeable"
        for probe_id in ("HG-3", "HG-4", "HG-5", "HG-6", "HG-7")
    )


def test_constructor_replay_compares_complete_parameter_values():
    plan = _plan()
    fixture = _fixture()
    calls = 0

    def changing_telemetry(*, feature_input, threshold, cap, seed):
        nonlocal calls
        del feature_input, cap, seed
        calls += 1
        graph = (
            fixture["expected_graph"]
            if threshold == 0.5
            else fixture["parameter_interventions"][
                "similarity_threshold"
            ]["expected_graph"]
        )
        return {
            "graph": graph.copy(),
            "parameter_values": {
                "similarity_threshold": threshold,
                "call_order": calls,
            },
        }

    callables = _callables(fixture)
    callables["fixture.graph:construct"] = changing_telemetry
    results = _by_id(run_graph_mechanism_probes(plan, callables, fixture))

    assert results["HG-2"].status == "unprobeable"
    assert "different parameter_values" in results["HG-2"].reason


def test_unseeded_default_rng_constructor_cannot_certify():
    plan = _plan()
    fixture = _fixture()

    def unseeded_construct(*, feature_input, threshold, cap, seed):
        del feature_input, cap, seed
        graph = (
            fixture["expected_graph"].copy()
            if threshold == 0.5
            else fixture["parameter_interventions"][
                "similarity_threshold"
            ]["expected_graph"].copy()
        )
        sources, targets = np.nonzero(graph)
        if len(sources):
            weight = float(np.random.default_rng().uniform(0.25, 0.75))
            graph[int(sources[0]), int(targets[0])] = weight
        return {
            "graph": graph,
            "parameter_values": {"similarity_threshold": threshold},
        }

    callables = _callables(fixture)
    callables["fixture.graph:construct"] = unseeded_construct
    results = _by_id(run_graph_mechanism_probes(plan, callables, fixture))

    assert results["HG-2"].status == "unprobeable"
    assert "different graph" in results["HG-2"].reason


def test_graph_calls_isolate_seeded_global_rngs_and_restore_ambient_state():
    plan = _plan()
    fixture = _fixture()
    try:
        import torch
    except ImportError:  # pragma: no cover - optional dependency environment
        torch = None

    random.seed(901)
    np.random.seed(902)
    if torch is not None:
        torch.default_generator.manual_seed(903)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = (
        torch.default_generator.get_state().clone()
        if torch is not None else None
    )

    def rng_values():
        values = [random.random(), float(np.random.random())]
        if torch is not None:
            values.append(float(torch.rand(()).item()))
        return values

    def global_rng_construct(*, feature_input, threshold, cap, seed):
        del feature_input, cap, seed
        graph = (
            fixture["expected_graph"]
            if threshold == 0.5
            else fixture["parameter_interventions"][
                "similarity_threshold"
            ]["expected_graph"]
        )
        return {
            "graph": graph.copy(),
            "parameter_values": {
                "similarity_threshold": threshold,
                "rng_draws": rng_values(),
            },
        }

    def global_rng_execute(*, graph, neighbor_signal, entity_ids, seed):
        output = _message_passing(
            graph=graph,
            neighbor_signal=neighbor_signal,
            entity_ids=entity_ids,
            seed=seed,
        )
        return output + sum(rng_values())

    callables = _callables(fixture, execute=global_rng_execute)
    callables["fixture.graph:construct"] = global_rng_construct
    results = run_graph_mechanism_probes(plan, callables, fixture)

    assert [result.status for result in results] == ["pass"] * 6
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    if torch is not None:
        assert torch.equal(torch.default_generator.get_state(), torch_before)


def test_sparse_integral_values_in_float_endpoint_container_do_not_certify():
    plan = _plan(representation="sparse_edge_index")
    fixture = _fixture(representation="sparse_edge_index")

    def float_endpoint_construct(*, feature_input, threshold, cap, seed):
        del feature_input, cap, seed
        graph = (
            fixture["expected_graph"]
            if threshold == 0.5
            else fixture["parameter_interventions"][
                "similarity_threshold"
            ]["expected_graph"]
        )
        return {
            "graph": np.asarray(graph, dtype=np.float32),
            "parameter_values": {"similarity_threshold": threshold},
        }

    callables = _callables(fixture)
    callables["fixture.graph:construct"] = float_endpoint_construct

    results = _by_id(run_graph_mechanism_probes(plan, callables, fixture))

    assert "pass" not in {
        results["HG-2"].status,
        results["HG-3"].status,
    }
    assert all(
        results[probe_id].status == "unprobeable"
        for probe_id in ("HG-4", "HG-5", "HG-6", "HG-7")
    )


def test_each_bound_threshold_and_cap_requires_runtime_consumption_proof():
    plan = _plan()
    fixture = _fixture()
    plan["construction"]["cap"] = {
        "kind": "per_source_top_similarity",
        "parameter": {
            "params_name": "max_neighbors",
            "callable_parameter": "cap",
        },
        "value": 3,
    }
    plan["parameter_authority"]["bindings"]["max_neighbors"] = {
        "carrier_value": 3,
        "params_value": 3,
        "paper_value": None,
        "source": "paper",
        "suppressed": False,
        "duplicates": [],
    }
    plan["fixture"]["parameter_interventions"]["max_neighbors"] = {
        "nominal_value": 3,
        "alternate_value": 1,
        "value": 1,
        "callable_parameter": "cap",
        "expected_relation": "graph_must_differ",
        "expected_graph": _ALTERNATE_DENSE_GRAPH.tolist(),
    }
    fixture["parameter_interventions"]["max_neighbors"] = copy.deepcopy(
        plan["fixture"]["parameter_interventions"]["max_neighbors"]
    )

    def construct(*, feature_input, threshold, cap, seed):
        del feature_input, seed
        graph = (
            fixture["expected_graph"]
            if threshold == 0.5 and cap == 3
            else _ALTERNATE_DENSE_GRAPH
        )
        return {
            "graph": graph.copy(),
            "parameter_values": {
                "similarity_threshold": threshold,
                "max_neighbors": cap,
            },
        }

    callables = _callables(fixture)
    callables["fixture.graph:construct"] = construct
    good = _by_id(run_graph_mechanism_probes(plan, callables, fixture))

    def ignores_cap(*, feature_input, threshold, cap, seed):
        del feature_input, seed
        graph = (
            _ALTERNATE_DENSE_GRAPH
            if threshold != 0.5
            else fixture["expected_graph"]
        )
        return {
            "graph": graph.copy(),
            "parameter_values": {
                "similarity_threshold": threshold,
                "max_neighbors": cap,
            },
        }

    callables["fixture.graph:construct"] = ignores_cap
    bad = _by_id(run_graph_mechanism_probes(plan, callables, fixture))

    assert good["HG-2"].status == "pass"
    assert bad["HG-2"].status == "fail"
    assert bad["HG-2"].evidence["parameter_interventions"][-1][
        "params_name"
    ] == "max_neighbors"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda plan: plan.pop("alignment_runtime_receipt"),
        lambda plan: plan["alignment_runtime_receipt"].update({"verified": False}),
        lambda plan: plan["alignment_runtime_receipt"].update(
            {"authority_digests": {"method_spec.json": "a" * 64}}
        ),
        lambda plan: plan["alignment_runtime_receipt"].update(
            {"entity_ids": ["wrong-a", "entity-b", "entity-c", "entity-d"]}
        ),
    ],
)
def test_unverified_alignment_receipt_blocks_every_graph_probe(mutate):
    plan = _plan()
    fixture = _fixture()
    mutate(plan)
    calls = []

    def forbidden(**kwargs):  # pragma: no cover - failure path assertion
        calls.append(kwargs)
        raise AssertionError("graph adapter ran before HG-1 receipt verification")

    callables = {
        "fixture.graph:construct": forbidden,
        "fixture.graph:execute": forbidden,
    }
    results = run_graph_mechanism_probes(plan, callables, fixture)

    assert calls == []
    assert {result.status for result in results} == {"unprobeable"}


def test_structural_parameter_disagreement_is_hg2_fail_after_receipt_gate():
    plan = _plan()
    fixture = _fixture()
    plan["status"] = "blocked"
    plan["reason"] = "parameter_value_disagrees:similarity_threshold"
    plan["parameter_authority"]["status"] = "blocked"
    plan["parameter_authority"]["reason"] = plan["reason"]
    plan["parameter_authority"]["bindings"]["similarity_threshold"][
        "params_value"
    ] = 0.7

    results = _by_id(run_graph_mechanism_probes(plan, {}, fixture))

    assert results["HG-2"].status == "fail"
    assert results["HG-2"].evidence["reason_code"] == (
        "parameter_authority_disagreement"
    )
    assert all(
        results[probe_id].status == "unprobeable"
        for probe_id in ("HG-3", "HG-4", "HG-5", "HG-6", "HG-7")
    )


def test_reconciled_paper_carrier_can_be_overridden_by_stronger_runtime_params():
    plan = _plan()
    fixture = _fixture()
    binding = plan["parameter_authority"]["bindings"]["similarity_threshold"]
    binding.update({"carrier_value": 0.5, "paper_value": 0.5, "params_value": 0.7})
    plan["construction"]["threshold"]["value"] = 0.7
    plan["fixture"]["parameter_interventions"]["similarity_threshold"][
        "nominal_value"
    ] = 0.7
    fixture["parameter_interventions"]["similarity_threshold"][
        "nominal_value"
    ] = 0.7

    def construct(*, feature_input, threshold, cap, seed):
        del feature_input, cap, seed
        graph = (
            fixture["expected_graph"]
            if threshold == pytest.approx(0.7)
            else fixture["parameter_interventions"]["similarity_threshold"][
                "expected_graph"
            ]
        )
        return {
            "graph": graph.copy(),
            "parameter_values": {"similarity_threshold": threshold},
        }

    callables = _callables(fixture)
    callables["fixture.graph:construct"] = construct
    result = _by_id(run_graph_mechanism_probes(plan, callables, fixture))["HG-2"]

    assert result.status == "pass"


def test_wrong_constructed_topology_fails_and_blocks_behavioral_checks():
    plan = _plan()
    fixture = _fixture()
    wrong_graph = fixture["expected_graph"].copy()
    wrong_graph[0, 0] = 1
    callables = _callables(fixture)
    callables["fixture.graph:construct"] = _constructor(
        wrong_graph,
        alternate_graph=fixture["parameter_interventions"][
            "similarity_threshold"
        ]["expected_graph"],
    )

    results = _by_id(run_graph_mechanism_probes(plan, callables, fixture))

    assert results["HG-2"].status == "pass"
    assert results["HG-3"].status == "fail"
    assert "forbidden self loops" in str(results["HG-3"].evidence["problems"])
    assert {results[probe_id].status for probe_id in ("HG-4", "HG-5", "HG-6", "HG-7")} == {
        "unprobeable"
    }


def test_union_symmetrized_top_one_graph_fails_the_mutual_cap_oracle():
    plan = _plan()
    fixture = _fixture()
    mutual = np.eye(4, dtype=float)
    mutual[1, 2] = mutual[2, 1] = 1.0
    union_symmetrized = mutual.copy()
    union_symmetrized[0, 1] = union_symmetrized[1, 0] = 1.0
    alternate = np.eye(4, dtype=float)
    fixture["expected_graph"] = mutual
    fixture["parameter_interventions"]["similarity_threshold"][
        "expected_graph"
    ] = alternate
    fixture["parameter_interventions"]["max_neighbors"] = {
        "nominal_value": 1,
        "alternate_value": 2,
        "value": 2,
        "callable_parameter": "cap",
        "expected_relation": "graph_must_differ",
        "expected_graph": alternate,
    }
    plan["construction"]["self_loop_policy"] = "required"
    plan["construction"]["cap"] = {
        "kind": "per_source_top_similarity",
        "parameter": {
            "params_name": "max_neighbors",
            "callable_parameter": "cap",
        },
        "value": 1,
    }
    plan["parameter_authority"]["bindings"]["max_neighbors"] = {
        "carrier_value": 1,
        "params_value": 1,
        "paper_value": 1,
        "source": "paper",
        "suppressed": False,
        "duplicates": [],
    }
    plan["fixture"] = json.loads(json.dumps(
        fixture,
        default=lambda value: value.tolist(),
        allow_nan=False,
    ))

    def construct(*, feature_input, threshold, cap, seed):
        del feature_input, seed
        graph = (
            union_symmetrized
            if threshold == 0.5 and cap == 1
            else alternate
        )
        return {
            "graph": graph.copy(),
            "parameter_values": {
                "similarity_threshold": threshold,
                "max_neighbors": cap,
            },
        }

    callables = _callables(fixture)
    callables["fixture.graph:construct"] = construct
    results = _by_id(run_graph_mechanism_probes(
        plan, callables, fixture
    ))

    assert results["HG-2"].status == "pass"
    assert results["HG-3"].status == "fail"
    assert results["HG-3"].evidence["observed_edge_symmetric_difference"] == 2
    assert all(
        results[probe_id].status == "unprobeable"
        for probe_id in ("HG-4", "HG-5", "HG-6", "HG-7")
    )


def test_graph_ignoring_execution_fails_topology_neighbor_and_contribution():
    plan = _plan()
    fixture = _fixture()

    results = _by_id(run_graph_mechanism_probes(
        plan,
        _callables(fixture, execute=_ignores_graph),
        fixture,
    ))

    assert results["HG-4"].status == "fail"
    assert results["HG-5"].status == "fail"
    assert results["HG-6"].status == "pass"
    assert results["HG-7"].status == "fail"
    assert results["HG-7"].evidence["reason_code"] == (
        "real_mechanism_not_discriminating"
    )


def test_seeded_stochastic_execution_replays_but_ambient_state_cannot_certify():
    plan = _plan()
    fixture = _fixture()

    def seeded_message(*, graph, neighbor_signal, entity_ids, seed):
        output = _message_passing(
            graph=graph,
            neighbor_signal=neighbor_signal,
            entity_ids=entity_ids,
            seed=seed,
        )
        return output + np.random.default_rng(seed).normal()

    seeded = _by_id(run_graph_mechanism_probes(
        plan,
        _callables(fixture, execute=seeded_message),
        fixture,
    ))
    assert [result.status for result in seeded.values()] == ["pass"] * 6

    ambient = np.random.default_rng(31337)

    def ambient_message(*, graph, neighbor_signal, entity_ids, seed):
        del seed
        output = _message_passing(
            graph=graph,
            neighbor_signal=neighbor_signal,
            entity_ids=entity_ids,
            seed=0,
        )
        return output + ambient.normal(size=output.shape)

    uncontrolled = _by_id(run_graph_mechanism_probes(
        plan,
        _callables(fixture, execute=ambient_message),
        fixture,
    ))

    assert uncontrolled["HG-2"].status == "pass"
    assert uncontrolled["HG-3"].status == "pass"
    assert all(
        uncontrolled[probe_id].status == "unprobeable"
        for probe_id in ("HG-4", "HG-5", "HG-6", "HG-7")
    )
    assert {
        uncontrolled[probe_id].reason
        for probe_id in ("HG-4", "HG-5", "HG-6", "HG-7")
    } == {"same-seed graph execution is not repeatable"}


def test_sub_tolerance_unseeded_default_rng_execution_cannot_certify():
    plan = _plan()
    fixture = _fixture()

    def unseeded_message(*, graph, neighbor_signal, entity_ids, seed):
        output = _message_passing(
            graph=graph,
            neighbor_signal=neighbor_signal,
            entity_ids=entity_ids,
            seed=seed,
        )
        return output + np.random.default_rng().uniform(0.0, 1.0e-12)

    results = _by_id(run_graph_mechanism_probes(
        plan,
        _callables(fixture, execute=unseeded_message),
        fixture,
    ))

    assert results["HG-2"].status == "pass"
    assert results["HG-3"].status == "pass"
    assert all(
        results[probe_id].status == "unprobeable"
        for probe_id in ("HG-4", "HG-5", "HG-6", "HG-7")
    )
    assert results["HG-4"].reason == (
        "same-seed graph execution is not repeatable"
    )


def test_call_order_state_cannot_fake_the_complete_graph_behavior_ladder():
    plan = _plan()
    fixture = _fixture()
    graph = np.asarray(fixture["expected_graph"], dtype=float)
    signals = np.asarray(fixture["node_signals"], dtype=float)

    altered = graph.copy()
    altered[0, 1] = 0.0
    altered[1, 0] = 0.0
    perturbed = signals.copy()
    perturbed[0] += 5.0
    permutation = np.asarray(fixture["permutation"], dtype=np.int64)
    permuted_graph = graph[np.ix_(permutation, permutation)]
    permuted_signals = signals[permutation]

    # These are the exact outputs a truthful adapter would produce for the old
    # seven-call schedule: nominal, nominal replay, topology, neighbor,
    # permutation, null baseline, and null neighbor.  The injected adapter
    # ignores every argument and emits them solely by call order, which used to
    # make HG-2 through HG-7 all pass.
    canned = [
        _message_passing(
            graph=graph, neighbor_signal=signals,
            entity_ids=fixture["entity_ids"], seed=1729,
        ),
        _message_passing(
            graph=graph, neighbor_signal=signals,
            entity_ids=fixture["entity_ids"], seed=1729,
        ),
        _message_passing(
            graph=altered, neighbor_signal=signals,
            entity_ids=fixture["entity_ids"], seed=1729,
        ),
        _message_passing(
            graph=graph, neighbor_signal=perturbed,
            entity_ids=fixture["entity_ids"], seed=1729,
        ),
        _message_passing(
            graph=permuted_graph, neighbor_signal=permuted_signals,
            entity_ids=[fixture["entity_ids"][index] for index in permutation],
            seed=1729,
        ),
        signals.copy(),
        perturbed.copy(),
    ]
    calls = 0

    def call_order_execute(*, graph, neighbor_signal, entity_ids, seed):
        nonlocal calls
        del graph, neighbor_signal, entity_ids, seed
        calls += 1
        return canned[min(calls - 1, len(canned) - 1)].copy()

    results = _by_id(run_graph_mechanism_probes(
        plan,
        _callables(fixture, execute=call_order_execute),
        fixture,
    ))

    assert results["HG-2"].status == "pass"
    assert results["HG-3"].status == "pass"
    assert all(
        results[probe_id].status == "unprobeable"
        for probe_id in ("HG-4", "HG-5", "HG-6", "HG-7")
    )
    assert {
        results[probe_id].evidence["reason_code"]
        for probe_id in ("HG-4", "HG-5", "HG-6", "HG-7")
    } == {"execution_state_isolation_failed"}


def test_execution_with_reset_per_call_scratch_state_remains_probeable():
    plan = _plan()
    fixture = _fixture()

    class ResettingMessagePassing:
        def __init__(self):
            self.scratch = None
            self.calls = 0

        def __call__(self, *, graph, neighbor_signal, entity_ids, seed):
            assert self.scratch is None
            self.scratch = (
                np.asarray(graph).copy(),
                np.asarray(neighbor_signal).copy(),
            )
            self.calls += 1
            try:
                return _message_passing(
                    graph=graph,
                    neighbor_signal=neighbor_signal,
                    entity_ids=entity_ids,
                    seed=seed,
                )
            finally:
                self.scratch = None

    execute = ResettingMessagePassing()
    results = run_graph_mechanism_probes(
        plan,
        _callables(fixture, execute=execute),
        fixture,
    )

    assert [result.status for result in results] == ["pass"] * 6
    assert execute.calls > 7
    assert execute.scratch is None


def test_topology_only_execution_does_not_fake_neighbor_signal_flow():
    plan = _plan()
    fixture = _fixture()

    results = _by_id(run_graph_mechanism_probes(
        plan,
        _callables(fixture, execute=_topology_only),
        fixture,
    ))

    assert results["HG-4"].status == "pass"
    assert results["HG-5"].status == "fail"


def test_topology_discriminator_round_trips_into_contribution_evidence():
    plan = _plan(ablation={
        "kind": "empty_graph",
        "element_id": "element:graph-ablation",
        "discriminating_probe_ref": PROBE_REFS["HG-4"],
    })
    fixture = _fixture()

    result = _by_id(run_graph_mechanism_probes(
        plan, _callables(fixture), fixture
    ))["HG-7"]

    assert result.status == "pass"
    assert result.evidence["discriminating_probe_ref"] == PROBE_REFS["HG-4"]
    assert result.evidence["real_evidence"]["maximum_target_delta"] > 0
    assert result.evidence["null_reason_code"] == "topology_response_absent"


def test_topology_only_real_cannot_certify_neighbor_discriminator():
    plan = _plan()
    fixture = _fixture()

    result = _by_id(run_graph_mechanism_probes(
        plan,
        _callables(fixture, execute=_topology_only),
        fixture,
    ))["HG-7"]

    assert result.status == "fail"
    assert result.evidence["reason_code"] == "real_mechanism_not_discriminating"


def test_unreachable_control_deltas_remain_descriptive_when_unchanged():
    plan = _plan()
    fixture = _fixture()

    result = _by_id(run_graph_mechanism_probes(
        plan,
        _callables(fixture, execute=_two_hop_message_passing),
        fixture,
    ))["HG-5"]

    assert result.status == "pass"
    assert all(
        delta <= result.evidence["effect_threshold"]
        for delta in result.evidence["control_max_abs_deltas"].values()
    )
    assert result.evidence["control_deltas_are_descriptive"] is True


@pytest.mark.parametrize(
    ("neighbor_weight", "expected_status"),
    [(0.001, "fail"), (0.002, "pass")],
)
def test_neighbor_effect_respects_the_frozen_tolerance_boundary(
    neighbor_weight, expected_status,
):
    plan = _plan()
    plan["execution"]["tolerance"] = 1e-3
    fixture = _fixture()

    def scaled_message(*, graph, neighbor_signal, entity_ids, seed):
        del entity_ids, seed
        signals = np.asarray(neighbor_signal, dtype=float)
        output = signals.copy()
        for source, destination in _graph_edges(graph):
            output[destination] += neighbor_weight * signals[source]
        return output

    result = _by_id(run_graph_mechanism_probes(
        plan,
        _callables(fixture, execute=scaled_message),
        fixture,
    ))["HG-5"]

    assert result.status == expected_status
    if expected_status == "pass":
        assert result.evidence["target_max_abs_delta"] > result.evidence[
            "effect_threshold"
        ]
    else:
        assert result.evidence["target_max_abs_delta"] <= result.evidence[
            "effect_threshold"
        ]


def test_graph_dependent_global_aggregator_fails_unreachable_controls():
    plan = _plan()
    fixture = _fixture()

    results = _by_id(run_graph_mechanism_probes(
        plan,
        _callables(fixture, execute=_global_graph_aggregator),
        fixture,
    ))

    assert results["HG-4"].status == "pass"
    assert results["HG-5"].status == "fail"
    assert results["HG-5"].evidence["reason_code"] == (
        "neighbor_response_leaks_to_unreachable_controls"
    )


def test_complete_graph_without_unreachable_control_is_unprobeable():
    plan = _plan()
    fixture = _fixture()
    complete = np.ones((4, 4), dtype=float) - np.eye(4, dtype=float)
    plan["fixture"]["expected_graph"] = complete.tolist()
    fixture["expected_graph"] = complete
    callables = _callables(fixture)

    result = _by_id(run_graph_mechanism_probes(
        plan, callables, fixture,
    ))["HG-5"]

    assert result.status == "unprobeable"
    assert result.evidence["reason_code"] == "neighbor_controls_reachable"


def test_position_leak_breaks_coherent_permutation_only():
    plan = _plan()
    fixture = _fixture()

    results = _by_id(run_graph_mechanism_probes(
        plan,
        _callables(fixture, execute=_position_leaking_message_passing),
        fixture,
    ))

    assert results["HG-4"].status == "pass"
    assert results["HG-5"].status == "pass"
    assert results["HG-6"].status == "fail"


def test_axis_one_neighbor_and_output_roots_execute_without_axis_guessing():
    plan = _plan()
    fixture = _fixture()
    fixture["node_signals"] = fixture["node_signals"].T
    plan["fixture"]["node_signals"] = fixture["node_signals"].tolist()
    plan["execution"]["entity_axis"] = 1
    plan["execution"]["output_entity_axis"] = 1

    def axis_one_message(*, graph, neighbor_signal, entity_ids, seed):
        output = _message_passing(
            graph=graph,
            neighbor_signal=np.asarray(neighbor_signal).T,
            entity_ids=entity_ids,
            seed=seed,
        )
        return output.T

    results = _by_id(run_graph_mechanism_probes(
        plan,
        _callables(fixture, execute=axis_one_message),
        fixture,
    ))

    assert all(
        results[probe_id].status == "pass"
        for probe_id in ("HG-2", "HG-3", "HG-4", "HG-5", "HG-6", "HG-7")
    )


def test_wrong_declared_output_axis_fails_closed():
    plan = _plan()
    fixture = _fixture()
    fixture["node_signals"] = fixture["node_signals"].T
    plan["fixture"]["node_signals"] = fixture["node_signals"].tolist()
    plan["execution"]["entity_axis"] = 1
    plan["execution"]["output_entity_axis"] = 0

    def axis_one_message(*, graph, neighbor_signal, entity_ids, seed):
        output = _message_passing(
            graph=graph,
            neighbor_signal=np.asarray(neighbor_signal).T,
            entity_ids=entity_ids,
            seed=seed,
        )
        return output.T

    results = _by_id(run_graph_mechanism_probes(
        plan,
        _callables(fixture, execute=axis_one_message),
        fixture,
    ))

    assert all(
        results[probe_id].status == "unprobeable"
        for probe_id in ("HG-4", "HG-5", "HG-6", "HG-7")
    )


def test_weighted_dense_graph_preserves_values_under_equivariant_permutation():
    plan = _plan()
    fixture = _fixture()
    weighted = np.asarray(
        [
            [0.0, 2.5, 0.0, 0.0],
            [0.75, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
            [0.0, 0.0, 1.25, 0.0],
        ]
    )
    plan["fixture"]["expected_graph"] = weighted.tolist()
    fixture["expected_graph"] = weighted

    results = _by_id(run_graph_mechanism_probes(
        plan,
        _callables(fixture, execute=_weighted_message_passing),
        fixture,
    ))

    assert results["HG-4"].status == "pass"
    assert results["HG-5"].status == "pass"
    assert results["HG-6"].status == "pass"


@pytest.mark.parametrize(
    ("null_execute", "expected"),
    [(_message_passing, "fail"), (_ignores_graph, "pass")],
)
def test_callable_ablation_must_fail_the_same_discriminating_check(
    null_execute, expected
):
    plan = _plan(ablation={
        "kind": "removed_message_passing",
        "element_id": "element:message-passing-ablation",
        "callable": {
            "module": "fixture.graph",
            "qualname": "null_execute",
        },
        "graph_parameter": "graph",
        "neighbor_signal_parameter": "neighbor_signal",
        "output_root": "graph_outputs",
        "discriminating_probe_ref": PROBE_REFS["HG-5"],
    })
    fixture = _fixture()

    results = _by_id(run_graph_mechanism_probes(
        plan,
        _callables(fixture, null_execute=null_execute),
        fixture,
    ))

    assert results["HG-5"].status == "pass"
    assert results["HG-7"].status == expected
    assert results["HG-7"].evidence["real_status"] == "pass"


def test_callable_ablation_shape_failure_is_not_contribution_evidence():
    plan = _plan(ablation={
        "kind": "removed_message_passing",
        "element_id": "element:message-passing-ablation",
        "callable": {
            "module": "fixture.graph",
            "qualname": "null_execute",
        },
        "graph_parameter": "graph",
        "neighbor_signal_parameter": "neighbor_signal",
        "output_root": "graph_outputs",
        "discriminating_probe_ref": PROBE_REFS["HG-5"],
    })
    fixture = _fixture()

    result = _by_id(run_graph_mechanism_probes(
        plan,
        _callables(fixture, null_execute=_shape_changing_null),
        fixture,
    ))["HG-7"]

    assert result.status == "unprobeable"
    assert result.evidence["reason_code"] == (
        "null_discriminating_result_unprobeable"
    )


def test_call_order_state_cannot_fake_a_callable_null_discriminator():
    plan = _plan(ablation={
        "kind": "removed_message_passing",
        "element_id": "element:message-passing-ablation",
        "callable": {
            "module": "fixture.graph",
            "qualname": "null_execute",
        },
        "graph_parameter": "graph",
        "neighbor_signal_parameter": "neighbor_signal",
        "output_root": "graph_outputs",
        "discriminating_probe_ref": PROBE_REFS["HG-5"],
    })
    fixture = _fixture()
    nominal = np.asarray(fixture["node_signals"], dtype=float)
    calls = 0

    def call_order_null(*, graph, neighbor_signal, entity_ids, seed):
        nonlocal calls
        del graph, neighbor_signal, entity_ids, seed
        calls += 1
        if calls <= 2:
            return nominal.copy()
        return nominal + float(calls)

    result = _by_id(run_graph_mechanism_probes(
        plan,
        _callables(fixture, null_execute=call_order_null),
        fixture,
    ))["HG-7"]

    assert result.status == "unprobeable"
    assert result.evidence["reason_code"] == (
        "execution_state_isolation_failed"
    )
    assert "declared null" in result.reason or "neighbor-signal" in result.reason


def test_callable_ablation_unrelated_failure_is_not_contribution_evidence():
    plan = _plan(ablation={
        "kind": "removed_message_passing",
        "element_id": "element:message-passing-ablation",
        "callable": {
            "module": "fixture.graph",
            "qualname": "null_execute",
        },
        "graph_parameter": "graph",
        "neighbor_signal_parameter": "neighbor_signal",
        "output_root": "graph_outputs",
        "discriminating_probe_ref": PROBE_REFS["HG-5"],
    })
    fixture = _fixture()

    result = _by_id(run_graph_mechanism_probes(
        plan,
        _callables(fixture, null_execute=_global_graph_aggregator),
        fixture,
    ))["HG-7"]

    assert result.status == "unprobeable"
    assert result.evidence["reason_code"] == (
        "null_discriminating_failure_not_absence"
    )
    assert result.evidence["null_reason_code"] == (
        "neighbor_response_leaks_to_unreachable_controls"
    )


def test_permutation_can_be_explicitly_not_applicable():
    plan = _plan(permutation_applicability="not_applicable")
    fixture = _fixture()

    results = _by_id(run_graph_mechanism_probes(
        plan,
        _callables(fixture),
        fixture,
    ))

    assert results["HG-6"].status == "not_applicable"
    assert all(
        results[probe_id].status == "pass"
        for probe_id in ("HG-2", "HG-3", "HG-4", "HG-5", "HG-7")
    )


def test_missing_or_raising_adapter_is_unprobeable():
    plan = _plan()
    fixture = _fixture()

    missing = _by_id(run_graph_mechanism_probes(plan, {}, fixture))
    assert missing["HG-2"].status == "unprobeable"
    assert "missing" in missing["HG-2"].reason

    def raises(**kwargs):
        raise RuntimeError(f"fixture rejected {sorted(kwargs)}")

    callables = _callables(fixture)
    callables["fixture.graph:execute"] = raises
    raised = _by_id(run_graph_mechanism_probes(plan, callables, fixture))
    assert raised["HG-2"].status == "pass"
    assert raised["HG-3"].status == "pass"
    assert {raised[probe_id].status for probe_id in ("HG-4", "HG-5", "HG-6", "HG-7")} == {
        "unprobeable"
    }
    assert "RuntimeError" in raised["HG-4"].reason
