"""Consumer coverage for exact-ref homogeneous-graph battery dispatch."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from build_probe_harness import (
    _frozen_graph_mechanism_execution_plan,
    _graph_callable_liveness_receipt,
    _stage_2d_alignment_receipt,
)
from graph_callable_liveness import (
    stage_2d_authority_digests,
    stage_3c_authority_digests,
)
from run_probes import (
    _GRAPH_PROBE_ROWS,
    _graph_backend_array,
    _graph_callable_adapters,
    _graph_constructor_adapter,
    _graph_execution_adapter,
    _graph_stage_2d_authority_digests,
    run_battery,
)


GRAPH_REFS = tuple(ref for _, ref in _GRAPH_PROBE_ROWS)


def _declared_context(*refs: str) -> dict[str, dict[str, str]]:
    return {
        ref: {
            "id": f"graph-check-{index}",
            "check": f"execute exact graph obligation {ref}",
            "why": "a silent graph-mechanism defect invalidates the demo",
        }
        for index, ref in enumerate(refs, start=1)
    }


def _write_run(root: Path) -> Path:
    run = root / "run"
    pipeline = run / ".pipeline"
    method = run / "method"
    pipeline.mkdir(parents=True)
    method.mkdir()
    (method / "__init__.py").write_text(
        "from .method import forecast\n"
        "from .model import (\n"
        "    GraphFixture, construct_graph_exact, execute_graph_exact,\n"
        ")\n"
        "from .training import train_graph\n",
        encoding="utf-8",
    )
    (method / "model.py").write_text(
        """import numpy as np


def prepare_relational(values):
    return values


def construct_graph_exact(node_features_exact, cosine_cutoff_exact):
    values = np.asarray(node_features_exact)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = values / norms
    similarity = normalized @ normalized.T
    adjacency = (similarity >= cosine_cutoff_exact).astype(np.float64)
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return {"chosen_graph": adjacency}


def execute_graph_exact(adjacency_exact, signals_exact):
    graph = np.asarray(adjacency_exact)
    signals = np.asarray(signals_exact, dtype=float)
    output = signals.copy()
    sources, destinations = np.nonzero(graph)
    for source, destination in zip(sources, destinations):
        output[destination] += signals[source]
    return output


class GraphFixture:
    def forward(self, adjacency_exact, signals_exact):
        outputs = execute_graph_exact(
            adjacency_exact=adjacency_exact,
            signals_exact=signals_exact,
        )
        return outputs
""",
        encoding="utf-8",
    )
    (method / "training.py").write_text(
        "def train_graph(*, model, graph, features):\n"
        "    return model\n",
        encoding="utf-8",
    )
    (method / "method.py").write_text(
        "def forecast(*, model, graph, signals):\n"
        "    return model.forward(\n"
        "        adjacency_exact=graph, signals_exact=signals,\n"
        "    )\n",
        encoding="utf-8",
    )
    method_spec = {
        "comparison": {
            "classification": {"id": "graph_fixture"},
            "pluggable_component": {"name": "execute_graph_exact"},
        },
        "methodology_replication_contract": {
            "homogeneous_graph_mechanism": {
                    "construction": {
                        "callable": {
                            "module": "method.model",
                            "qualname": "construct_graph_exact",
                        },
                        "feature_input_root": "fixture.node_features",
                        "feature_parameter": "node_features_exact",
                        "output_selector": {
                            "kind": "mapping_item",
                            "key": "chosen_graph",
                        },
                        "threshold": {
                            "metric": "cosine_similarity",
                            "comparison": "greater_than_or_equal",
                            "parameter": {
                                "params_name": "similarity_threshold",
                                "callable_parameter": "cosine_cutoff_exact",
                            },
                            "value": 0.9,
                        },
                        "cap": {"kind": "none"},
                        "self_loop_policy": "forbidden",
                        "direction_policy": "undirected_bidirectional",
                    },
                    "message_passing": {
                        "callable": {
                            "module": "method.model",
                            "qualname": "execute_graph_exact",
                        },
                        "graph_parameter": "adjacency_exact",
                        "neighbor_signal_root": "fixture.neighbor_signals",
                        "neighbor_signal_parameter": "signals_exact",
                        "output_root": "fixture.outputs",
                    },
            }
        },
    }
    (pipeline / "method_spec.json").write_text(
        json.dumps(method_spec),
        encoding="utf-8",
    )
    (pipeline / "params.json").write_text(
        json.dumps({"schema_version": "1.0.0", "params": {}}),
        encoding="utf-8",
    )
    model_descriptor = {"kind": "opaque", "type_description": "model"}
    arch_contract = {
        "architecture": {
            "model": {
                "class_name": "GraphFixture",
                "forward": {
                    "input": {"adjacency_exact": {}, "signals_exact": {}},
                    "output": {},
                },
            }
        },
        "training_loop": {
            "function_name": "train_graph",
            "input": {
                "model": model_descriptor,
                "graph": {"kind": "array", "role": "graph"},
                "features": {"kind": "array", "role": "features"},
            },
        },
        "pluggable_component": {
            "name": "forecast",
            "input": {
                "model": model_descriptor,
                "graph": {"kind": "array", "role": "graph"},
                "signals": {"kind": "array", "role": "signals"},
            },
        },
        "relational_indexing": {
            "execution": {
                "fitting": {
                    "model_input_root": "training_loop.input.model",
                    "graph_input_root": "training_loop.input.graph",
                    "coindexed_input_roots": {
                        "fixture.node_features": "training_loop.input.features",
                    },
                },
                "inference": {
                    "architecture_block": "model",
                    "graph_input_root": (
                        "architecture.model.forward.input.adjacency_exact"
                    ),
                    "coindexed_input_roots": {
                        "fixture.neighbor_signals": (
                            "architecture.model.forward.input.signals_exact"
                        ),
                    },
                    "output_coindexed_root": "fixture.outputs",
                },
            }
        },
    }
    (pipeline / "arch_contract.json").write_text(
        json.dumps(arch_contract), encoding="utf-8"
    )
    notebook_source = """\
from method import GraphFixture, construct_graph_exact, forecast, train_graph
params = {}
cfg = unpack(params)
graph = construct_graph_exact(
    node_features_exact=features,
    cosine_cutoff_exact=cfg["similarity_threshold"],
)["chosen_graph"]
fitting_graph = graph.copy()
model = GraphFixture()
trained = train_graph(
    model=model, graph=fitting_graph, features=features,
)
outputs = forecast(model=trained, graph=graph, signals=neighbor_signals)
"""
    (run / "notebook.ipynb").write_text(
        json.dumps({
            "cells": [{"cell_type": "code", "source": notebook_source}],
        }),
        encoding="utf-8",
    )
    (pipeline / "notebook_draft.py").write_text(
        notebook_source, encoding="utf-8"
    )
    (run / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    (pipeline / "stage_2d.complete").write_text("", encoding="utf-8")
    (pipeline / "stage_2d.upstream_digest.json").write_text(
        json.dumps(_graph_stage_2d_authority_digests(run), sort_keys=True),
        encoding="utf-8",
    )
    (pipeline / "stage_3c.complete").write_text("", encoding="utf-8")
    (pipeline / "stage_3c.upstream_digest.json").write_text(
        json.dumps(stage_3c_authority_digests(run), sort_keys=True),
        encoding="utf-8",
    )
    return run


def _ready_plan(run: Path) -> dict:
    refs_by_id = dict(_GRAPH_PROBE_ROWS)
    graph = [
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    groundings = {
        refs_by_id["HG-1"]: {
            "probe_ref": refs_by_id["HG-1"],
            "element_ids": ["graph-alignment"],
            "bound_callables": ["method.model:prepare_relational"],
        },
        refs_by_id["HG-2"]: {
            "probe_ref": refs_by_id["HG-2"],
            "element_ids": ["graph-construction"],
            "bound_callables": ["method.model:construct_graph_exact"],
        },
        refs_by_id["HG-3"]: {
            "probe_ref": refs_by_id["HG-3"],
            "element_ids": ["graph-construction"],
            "bound_callables": ["method.model:construct_graph_exact"],
        },
        refs_by_id["HG-4"]: {
            "probe_ref": refs_by_id["HG-4"],
            "element_ids": ["message-passing"],
            "bound_callables": ["method.model:execute_graph_exact"],
        },
        refs_by_id["HG-5"]: {
            "probe_ref": refs_by_id["HG-5"],
            "element_ids": ["message-passing"],
            "bound_callables": ["method.model:execute_graph_exact"],
        },
        refs_by_id["HG-6"]: {
            "probe_ref": refs_by_id["HG-6"],
            "element_ids": ["message-passing"],
            "bound_callables": ["method.model:execute_graph_exact"],
        },
        refs_by_id["HG-7"]: {
            "probe_ref": refs_by_id["HG-7"],
            "element_ids": ["graph-ablation"],
            "bound_callables": ["method.model:execute_graph_exact"],
        },
    }
    plan = {
        "schema_version": "1.0",
        "status": "ready",
        "reason": "homogeneous_graph_runtime_plan_ready",
        "representation": "dense_adjacency",
        "alignment": {
            "status": "ready",
            "reason": None,
            "element_id": "graph-alignment",
        },
        "construction": {
            "callable": {
                "module": "method.model",
                "qualname": "construct_graph_exact",
            },
            "feature_input_root": "fixture.node_features",
            "feature_parameter": "node_features_exact",
            "output_selector": {
                "kind": "mapping_item",
                "key": "chosen_graph",
            },
            "threshold": {
                "metric": "cosine_similarity",
                "comparison": "greater_than_or_equal",
                "parameter": {
                    "params_name": "similarity_threshold",
                    "callable_parameter": "cosine_cutoff_exact",
                },
                "value": 0.9,
            },
            "cap": {"kind": "none"},
            "self_loop_policy": "forbidden",
            "direction_policy": "undirected_bidirectional",
        },
        "parameter_authority": {
            "status": "pass",
            "reason": None,
            "bindings": {
                "similarity_threshold": {
                    "carrier_value": 0.9,
                    "params_value": 0.9,
                    "paper_value": None,
                    "source": "paper",
                    "suppressed": False,
                    "duplicates": [],
                }
            },
        },
        "execution": {
            "callable": {
                "module": "method.model",
                "qualname": "execute_graph_exact",
            },
            "graph_parameter": "adjacency_exact",
            "neighbor_signal_root": "fixture.neighbor_signals",
            "neighbor_signal_parameter": "signals_exact",
            "entity_axis": 0,
            "output_root": "fixture.outputs",
            "output_entity_axis": 0,
            "tensor_backend": "numpy",
            "permutation_applicability": "required",
            "tolerance": 1e-8,
            "seed": 1729,
        },
        "ablation": {
            "kind": "empty_graph",
            "element_id": "graph-ablation",
            "discriminating_probe_ref": refs_by_id["HG-5"],
        },
        "groundings": groundings,
        "fixture": {
            "scope": "r2c088-consumer-fixture",
            "entity_ids": ["entity-a", "entity-b", "entity-c", "entity-d"],
            "fixture.node_features": [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
                [0.2, 0.8],
            ],
            "fixture.neighbor_signals": [
                [1.0], [2.0], [4.0], [8.0],
            ],
            "comparison_witness": {
                "metric": "cosine_similarity",
                "comparison": "greater_than_or_equal",
                "threshold_value": 1.0,
                "entity_axis": 0,
                "feature_input": [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                "expected_graph": [
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
            },
            "expected_graph": graph,
            "parameter_interventions": {
                "similarity_threshold": {
                    "nominal_value": 0.9,
                    "alternate_value": 0.99,
                    "value": 0.99,
                    "callable_parameter": "cosine_cutoff_exact",
                    "expected_relation": "graph_must_differ",
                    "expected_graph": [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
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
            "permutation": [2, 0, 3, 1],
            "alignment_source": {"scope": "r2c084-source"},
        },
    }
    plan["alignment_runtime_receipt"] = _stage_2d_alignment_receipt(run, plan)
    assert plan["alignment_runtime_receipt"]["status"] == "pass"
    spec = json.loads(
        (run / ".pipeline" / "method_spec.json").read_text(encoding="utf-8")
    )
    contract = json.loads(
        (run / ".pipeline" / "arch_contract.json").read_text(encoding="utf-8")
    )
    plan["callable_liveness_receipt"] = _graph_callable_liveness_receipt(
        run, plan, spec, contract
    )
    assert plan["callable_liveness_receipt"]["status"] == "pass", plan[
        "callable_liveness_receipt"
    ]
    return plan


def _frozen(plan: dict, *refs: str) -> dict:
    selected = refs or GRAPH_REFS
    return {
        "schema_version": "1.0",
        "battery_version": "graph-consumer-test",
        "paradigm": "graph_fixture",
        "declared_context": _declared_context(*selected),
        "graph_mechanism_execution_plan": plan,
    }


def _graph_rows(report):
    return [row for row in report.verdicts if row.probe_id.startswith("HG-")]


def _run_frozen(run: Path, plan: dict, tmp_path: Path, *refs: str):
    return run_battery(
        run,
        ledger_path=tmp_path / "results" / "claims_ledger.json",
        frozen_gating=_frozen(plan, *refs),
    )


def test_known_good_frozen_graph_plan_emits_seven_ordered_exact_rows(tmp_path):
    run = _write_run(tmp_path)
    plan = _ready_plan(run)

    rows = _graph_rows(_run_frozen(run, plan, tmp_path))

    assert [row.probe_id for row in rows] == [row[0] for row in _GRAPH_PROBE_ROWS]
    assert [row.probe_ref for row in rows] == list(GRAPH_REFS)
    assert [row.verdict for row in rows] == ["pass"] * 7
    assert [row.pack_check_id for row in rows] == [
        f"graph-check-{index}" for index in range(1, 8)
    ]
    for row in rows:
        assert row.element_ids
        assert row.bound_callables
        assert isinstance(json.loads(row.evidence), dict)
    traces = [json.loads(row.evidence)["trace"] for row in rows]
    assert len({trace["execution_plan_digest"] for trace in traces}) == 1
    assert len({trace["fixture_digest"] for trace in traces}) == 1
    assert rows[0].element_ids == ["graph-alignment"]
    assert rows[1].bound_callables == [
        "method.model:construct_graph_exact"
    ]
    assert rows[3].bound_callables == ["method.model:execute_graph_exact"]


def test_graph_free_plan_emits_all_seven_not_applicable_rows(tmp_path):
    run = _write_run(tmp_path)
    plan = {
        "schema_version": "1.0",
        "status": "not_applicable",
        "reason": "graph_free_method",
        "groundings": {},
    }

    rows = _graph_rows(_run_frozen(run, plan, tmp_path))

    assert [row.probe_id for row in rows] == [row[0] for row in _GRAPH_PROBE_ROWS]
    assert {row.verdict for row in rows} == {"not_applicable"}
    assert {row.reason for row in rows} == {"graph_free_method"}


def test_unsupported_plan_emits_all_seven_unprobeable_rows(
    tmp_path, monkeypatch,
):
    import run_probes

    run = _write_run(tmp_path)
    plan = {
        "schema_version": "1.0",
        "status": "unprobeable",
        "reason": "unsupported_relational_structure",
        "groundings": {},
    }
    monkeypatch.setattr(
        run_probes,
        "_graph_callable_adapters",
        lambda _: (_ for _ in ()).throw(AssertionError("must not import")),
    )

    rows = _graph_rows(_run_frozen(run, plan, tmp_path))

    assert len(rows) == 7
    assert {row.verdict for row in rows} == {"unprobeable"}
    assert {row.reason for row in rows} == {"unsupported_relational_structure"}


def test_forged_callable_receipt_is_unprobeable_and_blocks_later_rows(tmp_path):
    run = _write_run(tmp_path)
    plan = _ready_plan(run)
    missing = "missing_constructor_exactly"
    plan["construction"]["callable"]["qualname"] = missing
    plan["groundings"][GRAPH_REFS[1]]["bound_callables"] = [
        f"method.model:{missing}"
    ]
    plan["groundings"][GRAPH_REFS[2]]["bound_callables"] = [
        f"method.model:{missing}"
    ]
    plan["callable_liveness_receipt"]["construction"]["callable"] = (
        copy.deepcopy(plan["construction"]["callable"])
    )

    rows = _graph_rows(_run_frozen(run, plan, tmp_path))

    assert rows[0].verdict == "pass"
    assert [row.verdict for row in rows[1:]] == ["unprobeable"] * 6
    assert rows[1].message == "graph_callable_liveness_receipt_proof_mismatch"
    evidence = json.loads(rows[1].evidence)
    assert evidence["callable_liveness_assessment"] == {
        "verified": False,
        "reason": "graph_callable_liveness_receipt_proof_mismatch",
        "evidence": {"changed_proof_fields": ["construction.callable"]},
    }
    assert rows[1].bound_callables == [f"method.model:{missing}"]


def test_stale_alignment_receipt_blocks_every_later_graph_row(tmp_path):
    run = _write_run(tmp_path)
    plan = _ready_plan(run)
    with (run / "method" / "model.py").open("a", encoding="utf-8") as stream:
        stream.write("\n# post-receipt drift\n")

    rows = _graph_rows(_run_frozen(run, plan, tmp_path))

    assert rows[0].verdict == "unprobeable"
    assert rows[0].reason == "alignment_runtime_receipt_stale"
    assert [row.verdict for row in rows[1:]] == ["unprobeable"] * 6
    assert all("HG-1" in row.message for row in rows[1:])


def test_missing_stage_2d_sentinel_invalidates_frozen_alignment_receipt(tmp_path):
    run = _write_run(tmp_path)
    plan = _ready_plan(run)
    (run / ".pipeline" / "stage_2d.complete").unlink()

    rows = _graph_rows(_run_frozen(run, plan, tmp_path))

    assert rows[0].verdict == "unprobeable"
    assert rows[0].reason == "alignment_runtime_stage_receipt_missing"
    evidence = json.loads(rows[0].evidence)
    assert evidence["missing_stage"] == "stage_2d"
    assert [row.verdict for row in rows[1:]] == ["unprobeable"] * 6
    assert all("HG-1" in row.message for row in rows[1:])


def test_dispatch_filters_by_exact_ref_and_preserves_grounding(tmp_path):
    run = _write_run(tmp_path)
    plan = _ready_plan(run)
    topology_ref = dict(_GRAPH_PROBE_ROWS)["HG-4"]

    rows = _graph_rows(
        _run_frozen(run, plan, tmp_path, topology_ref)
    )

    assert len(rows) == 1
    assert rows[0].probe_id == "HG-4"
    assert rows[0].probe_ref == topology_ref
    assert rows[0].pack_check_id == "graph-check-1"
    assert rows[0].element_ids == ["message-passing"]
    assert rows[0].bound_callables == ["method.model:execute_graph_exact"]


def test_live_plan_freezer_is_lazy_and_mints_stage_2d_receipt(
    tmp_path, monkeypatch,
):
    from scripts import graph_mechanism_runtime_plan as planner

    run = _write_run(tmp_path)
    expected = _ready_plan(run)
    expected.pop("alignment_runtime_receipt")
    calls = []

    def fake_normalizer(method_spec, params, arch_contract):
        calls.append((method_spec, params, arch_contract))
        return copy.deepcopy(expected)

    monkeypatch.setattr(
        planner, "normalize_graph_mechanism_runtime_plan", fake_normalizer
    )
    assert _frozen_graph_mechanism_execution_plan(run, {}) is None
    assert calls == []

    frozen = _frozen_graph_mechanism_execution_plan(
        run, _declared_context(GRAPH_REFS[0])
    )

    assert len(calls) == 1
    assert frozen["alignment"]["status"] == "pass"
    assert frozen["alignment_runtime_receipt"]["status"] == "pass"
    assert frozen["alignment_runtime_receipt"]["verified"] is True


def test_numpy_adapters_materialize_cpu_floating_inputs_exactly():
    observed = {}
    construction = {
        "feature_parameter": "features_exact",
        "output_selector": {"kind": "mapping_item", "key": "graph"},
        "threshold": {
            "parameter": {
                "params_name": "threshold",
                "callable_parameter": "threshold_exact",
            }
        },
        "cap": {"kind": "none"},
    }

    def construct(features_exact, threshold_exact):
        observed["features"] = features_exact
        observed["threshold"] = threshold_exact
        return {"graph": [[0.0, 1.0], [1.0, 0.0]]}

    constructor = _graph_constructor_adapter(
        construct, construction, tensor_backend="numpy"
    )
    assert constructor is not None
    constructor(
        feature_input=[[1, 0], [0, 1]], threshold=0.5, cap=None, seed=7
    )

    execution = {
        "graph_parameter": "graph_exact",
        "neighbor_signal_parameter": "signals_exact",
    }

    def execute(graph_exact, signals_exact):
        observed["graph"] = graph_exact
        observed["signals"] = signals_exact
        return signals_exact

    executor = _graph_execution_adapter(
        execute,
        execution,
        representation="dense_adjacency",
        tensor_backend="numpy",
    )
    assert executor is not None
    executor(
        graph=[[0, 1], [1, 0]],
        neighbor_signal=[[1], [2]],
        entity_ids=[101, 205],
        seed=7,
    )

    assert isinstance(observed["features"], np.ndarray)
    assert observed["features"].dtype == np.float32
    assert observed["threshold"] == 0.5
    assert isinstance(observed["graph"], np.ndarray)
    assert observed["graph"].dtype == np.float32
    assert observed["signals"].dtype == np.float32


def test_numpy_adapter_rejects_ndarray_subclass_dispatch():
    class DispatchArray(np.ndarray):
        pass

    value = np.asarray([[1.0, 0.0]]).view(DispatchArray)

    with pytest.raises(ValueError, match="ndarray subclass dispatch"):
        _graph_backend_array(
            value,
            tensor_backend="numpy",
            integral=False,
        )


def test_torch_adapters_materialize_cpu_float_and_sparse_long_inputs():
    torch = pytest.importorskip("torch")
    observed = {}
    construction = {
        "feature_parameter": "features_exact",
        "output_selector": {"kind": "return_value"},
        "threshold": {
            "parameter": {
                "params_name": "threshold",
                "callable_parameter": "threshold_exact",
            }
        },
        "cap": {"kind": "none"},
    }

    def construct(features_exact, threshold_exact):
        observed["features"] = features_exact
        observed["threshold"] = threshold_exact
        return torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    constructor = _graph_constructor_adapter(
        construct, construction, tensor_backend="torch"
    )
    assert constructor is not None
    constructor(
        feature_input=[[1, 0], [0, 1]], threshold=0.5, cap=None, seed=7
    )

    execution = {
        "graph_parameter": "graph_exact",
        "neighbor_signal_parameter": "signals_exact",
    }

    def execute(graph_exact, signals_exact):
        observed["graph"] = graph_exact
        observed["signals"] = signals_exact
        return signals_exact

    executor = _graph_execution_adapter(
        execute,
        execution,
        representation="sparse_edge_index",
        tensor_backend="torch",
    )
    assert executor is not None
    executor(
        graph=[[0, 1], [1, 0]],
        neighbor_signal=[[1], [2]],
        entity_ids=[101, 205],
        seed=7,
    )

    assert isinstance(observed["features"], torch.Tensor)
    assert observed["features"].dtype == torch.float32
    assert observed["features"].device.type == "cpu"
    assert observed["threshold"] == 0.5
    assert observed["graph"].dtype == torch.long
    assert observed["graph"].device.type == "cpu"
    assert observed["signals"].dtype == torch.float32
    assert observed["signals"].device.type == "cpu"


def test_torch_adapter_rejects_tensor_subclass_dispatch():
    torch = pytest.importorskip("torch")

    class DispatchTensor(torch.Tensor):
        pass

    value = torch.ones((1, 2)).as_subclass(DispatchTensor)

    with pytest.raises(ValueError, match="Tensor subclass dispatch"):
        _graph_backend_array(
            value,
            tensor_backend="torch",
            integral=False,
        )


def test_callable_ablation_adapter_uses_its_own_exact_keyword_bindings(
    monkeypatch,
):
    import run_probes

    observed = {}
    construction_identity = {
        "module": "method.model",
        "qualname": "construct_graph_exact",
    }
    execution_identity = {
        "module": "method.model",
        "qualname": "execute_graph_exact",
    }
    ablation_identity = {
        "module": "method.model",
        "qualname": "decode_without_messages",
    }
    plan = {
        "representation": "dense_adjacency",
        "construction": {
            "callable": construction_identity,
            "feature_parameter": "features_exact",
            "output_selector": {"kind": "return_value"},
            "threshold": {
                "parameter": {
                    "params_name": "threshold",
                    "callable_parameter": "threshold_exact",
                }
            },
            "cap": {"kind": "none"},
        },
        "execution": {
            "callable": execution_identity,
            "graph_parameter": "real_graph",
            "neighbor_signal_parameter": "real_signals",
            "tensor_backend": "numpy",
        },
        "ablation": {
            "kind": "non_graph_decoder",
            "callable": ablation_identity,
            "graph_parameter": "null_graph",
            "neighbor_signal_parameter": "null_signals",
            "output_root": "fixture.outputs",
        },
    }

    def construct_graph_exact(features_exact, threshold_exact):
        del features_exact, threshold_exact
        return [[0.0, 1.0], [1.0, 0.0]]

    def execute_graph_exact(real_graph, real_signals):
        del real_graph
        return real_signals

    def decode_without_messages(null_graph, null_signals):
        observed["graph"] = null_graph
        observed["signals"] = null_signals
        return null_signals

    functions = {
        "construct_graph_exact": construct_graph_exact,
        "execute_graph_exact": execute_graph_exact,
        "decode_without_messages": decode_without_messages,
    }
    monkeypatch.setattr(
        run_probes,
        "_exact_generated_callable",
        lambda identity: functions.get(identity.get("qualname")),
    )

    adapters = _graph_callable_adapters(plan)
    null_adapter = adapters["method.model:decode_without_messages"]
    returned = null_adapter(
        graph=[[0, 1], [1, 0]],
        neighbor_signal=[[1], [2]],
        entity_ids=[101, 205],
        seed=7,
    )

    assert isinstance(observed["graph"], np.ndarray)
    assert observed["graph"].dtype == np.float32
    assert observed["signals"].dtype == np.float32
    assert np.array_equal(returned, observed["signals"])


@pytest.mark.parametrize("backend", [None, "jax"])
def test_callable_adapters_fail_closed_without_supported_backend(
    tmp_path, backend,
):
    run = _write_run(tmp_path)
    plan = _ready_plan(run)
    if backend is None:
        plan["execution"].pop("tensor_backend")
    else:
        plan["execution"]["tensor_backend"] = backend

    assert _graph_callable_adapters(plan) == {}
