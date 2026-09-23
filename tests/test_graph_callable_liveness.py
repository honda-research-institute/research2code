"""Focused deterministic live-use proofs for homogeneous-graph callables."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.graph_callable_liveness import (
    NUMERIC_OPERAND_DISPATCH_PRECONDITION,
    RECEIPT_VERSION,
    GraphCallableCoverageError,
    GraphCallableProducerError,
    assess_liveness_receipt,
    graph_callable_identities,
    prove_constructor_notebook_flow,
    prove_message_live_path,
    stage_2d_authority_digests,
    stage_3c_authority_digests,
)


def _method_spec() -> dict:
    return {
        "methodology_replication_contract": {
            "homogeneous_graph_mechanism": {
                "construction": {
                    "element_id": "graph-construction",
                    "callable": {
                        "module": "method.model",
                        "qualname": "construct_graph_exact",
                    },
                    "feature_input_root": "batch.features",
                    "feature_parameter": "features_exact",
                    "output_selector": {"kind": "tuple_item", "index": 0},
                    "threshold": {
                        "metric": "cosine_similarity",
                        "comparison": "greater_than_or_equal",
                        "parameter": {
                            "params_name": "similarity_threshold",
                            "callable_parameter": "similarity_threshold_exact",
                        },
                    },
                    "cap": {"kind": "none"},
                    "self_loop_policy": "required",
                    "direction_policy": "undirected_bidirectional",
                },
                "message_passing": {
                    "element_id": "graph-message-passing",
                    "callable": {
                        "module": "method.model",
                        "qualname": "aggregate_neighbors_exact",
                    },
                    "graph_parameter": "graph_exact",
                    "neighbor_signal_root": "batch.signals",
                    "neighbor_signal_parameter": "signals_exact",
                    "output_root": "outputs.forecast",
                },
            }
        }
    }


def _arch_contract() -> dict:
    return {
        "architecture": {
            "model": {
                "class_name": "TinyGraphModel",
                "forward": {
                    "input": {"graph": {}, "signals": {}},
                    "output": {},
                },
            }
        },
        "training_loop": {
            "function_name": "train_model",
            "input": {
                "model": {"kind": "opaque", "type_description": "model"},
                "source_graph": {"kind": "array", "role": "graph"},
                "features": {"kind": "array", "role": "features"},
            },
        },
        "pluggable_component": {
            "name": "predict",
            "input": {
                "model": {"kind": "opaque", "type_description": "model"},
                "graph_payload": {"kind": "array", "role": "graph"},
                "signals": {"kind": "array", "role": "signals"},
            },
        },
        "relational_indexing": {
            "execution": {
                "fitting": {
                    "model_input_root": "training_loop.input.model",
                    "graph_input_root": "training_loop.input.source_graph",
                    "coindexed_input_roots": {
                        "batch.features": "training_loop.input.features",
                    },
                },
                "inference": {
                    "architecture_block": "model",
                    "graph_input_root": "architecture.model.forward.input.graph",
                    "coindexed_input_roots": {
                        "batch.signals": (
                            "architecture.model.forward.input.signals"
                        ),
                    },
                    "output_coindexed_root": "outputs.forecast",
                },
            }
        },
    }


_MODEL_SOURCE = """\
import numpy as np


def construct_graph_exact(*, features_exact, similarity_threshold_exact):
    return np.array([[0, 1], [1, 0]]), {"source": "exact"}


def aggregate_neighbors_exact(*, graph_exact, signals_exact):
    return [graph_exact, signals_exact]


class TinyGraphModel:
    def forward(self, graph, signals):
        forecast = aggregate_neighbors_exact(
            graph_exact=graph,
            signals_exact=signals,
        )
        return {"forecast": forecast}
"""

_TRAINING_SOURCE = """\
def train_model(*, model, source_graph, features):
    return model
"""

_METHOD_SOURCE = """\
def predict(*, model, graph_payload, signals):
    return model.forward(graph=graph_payload, signals=signals)
"""

_INIT_SOURCE = """\
from .model import TinyGraphModel
from .model import aggregate_neighbors_exact, construct_graph_exact
from .training import train_model
from .method import predict

__all__ = [
    "aggregate_neighbors_exact",
    "construct_graph_exact",
    "predict",
    "TinyGraphModel",
    "train_model",
]
"""

_NOTEBOOK_CELLS = [
    """\
from method import TinyGraphModel
from method import construct_graph_exact, predict, train_model
params = {}
cfg = unpack(params)
edge_index, graph_metadata = construct_graph_exact(
    features_exact=features,
    similarity_threshold_exact=cfg["similarity_threshold"],
)
fitting_graph = edge_index.copy()
model = TinyGraphModel()
trained = train_model(model=model, source_graph=fitting_graph, features=features)
forecast = predict(model=trained, graph_payload=edge_index, signals=signals)
"""
]


def _write_package(
    tmp_path: Path,
    *,
    model_source: str = _MODEL_SOURCE,
    training_source: str = _TRAINING_SOURCE,
    method_source: str = _METHOD_SOURCE,
    init_source: str = _INIT_SOURCE,
) -> Path:
    run_dir = tmp_path / "run"
    method_dir = run_dir / "method"
    pipeline_dir = run_dir / ".pipeline"
    method_dir.mkdir(parents=True)
    pipeline_dir.mkdir()
    (method_dir / "model.py").write_text(model_source, encoding="utf-8")
    (method_dir / "training.py").write_text(training_source, encoding="utf-8")
    (method_dir / "method.py").write_text(method_source, encoding="utf-8")
    (method_dir / "__init__.py").write_text(init_source, encoding="utf-8")
    return run_dir


def test_exact_constructor_output_reaches_fitting_and_inference(tmp_path):
    run_dir = _write_package(tmp_path)

    proof = prove_constructor_notebook_flow(
        _NOTEBOOK_CELLS,
        _method_spec(),
        _arch_contract(),
        run_dir / "method",
    )

    assert proof["status"] == "pass"
    assert proof["callable"] == {
        "module": "method.model",
        "qualname": "construct_graph_exact",
    }
    assert proof["selected_binding"] == "edge_index"
    assert proof["feature_input_root"] == "batch.features"
    assert proof["feature_parameter"] == "features_exact"
    assert proof["feature_binding"] == "features"
    assert proof["parameter_bindings"] == {
        "similarity_threshold": {
            "callable_parameter": "similarity_threshold_exact",
            "notebook_cfg_key": "similarity_threshold",
        }
    }
    assert proof["fitting_copy_binding"] == "fitting_graph"
    assert proof["fitting_copy_method"] == "copy"
    assert proof["fitting_graph_array_kind"] == "numpy"
    assert proof["numeric_operand_dispatch_precondition"] == (
        NUMERIC_OPERAND_DISPATCH_PRECONDITION
    )
    assert proof["fitting_graph_parameter"] == "source_graph"
    assert proof["fitting_model_parameter"] == "model"
    assert proof["architecture_class"] == "TinyGraphModel"
    assert proof["architecture_model_binding"] == "model"
    assert proof["fitting_model_identity_preserved"] is True
    assert proof["fitting_model_result_binding"] == "trained"
    assert proof["pluggable_model_parameter"] == "model"
    assert proof["pluggable_graph_parameter"] == "graph_payload"
    assert proof["pluggable_neighbor_signal_parameter"] == "signals"
    assert proof["inference_graph_parameter"] == "graph"
    assert proof["inference_neighbor_signal_parameter"] == "signals"
    assert proof["neighbor_signal_root"] == "batch.signals"
    assert proof["neighbor_signal_binding"] == "signals"
    assert proof["inference_output_binding"] == "forecast"
    assert proof["pluggable_result_consumed"] is True
    assert proof["phases"] == ["fitting", "inference"]


def test_inference_keyword_evaluation_cannot_replace_graph(tmp_path):
    run_dir = _write_package(tmp_path)
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "forecast = predict(model=trained, graph_payload=edge_index, "
            "signals=signals)",
            "forecast = predict(signals=(edge_index := signals), "
            "graph_payload=edge_index, model=trained)",
        )
    ]

    with pytest.raises(GraphCallableCoverageError):
        prove_constructor_notebook_flow(
            cells,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )


def test_inference_must_use_declared_neighbor_signal_binding(tmp_path):
    run_dir = _write_package(tmp_path)
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "graph_payload=edge_index, signals=signals",
            "graph_payload=edge_index, signals=unrelated_signals",
        )
    ]

    with pytest.raises(GraphCallableProducerError) as raised:
        prove_constructor_notebook_flow(
            cells,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code == "graph_constructor_inference_path_missing"


def test_pluggable_keyword_evaluation_cannot_replace_graph(tmp_path):
    source = """\
def predict(*, model, graph_payload, signals):
    return model.forward(
        signals=(graph_payload := signals),
        graph=graph_payload,
    )
"""
    run_dir = _write_package(tmp_path, method_source=source)

    with pytest.raises(GraphCallableCoverageError):
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )


def test_message_helper_keyword_evaluation_cannot_replace_graph(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "def aggregate_neighbors_exact(*, graph_exact, signals_exact):",
        "def aggregate_neighbors_exact(*, graph_exact, signals_exact, "
        "poison=None):",
    ).replace(
        "        forecast = aggregate_neighbors_exact(\n"
        "            graph_exact=graph,\n"
        "            signals_exact=signals,\n"
        "        )",
        "        forecast = aggregate_neighbors_exact(\n"
        "            poison=(graph := signals),\n"
        "            graph_exact=graph,\n"
        "            signals_exact=signals,\n"
        "        )",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError):
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )


@pytest.mark.parametrize("dispatch_attribute", ["dispatch_owner", "training"])
def test_fitting_model_data_alias_cannot_mutate_forward(
    tmp_path, dispatch_attribute
):
    model_source = _MODEL_SOURCE.replace(
        "class TinyGraphModel:\n",
        "class TinyGraphModel:\n"
        "    def __init__(self):\n"
        f"        self.{dispatch_attribute} = TinyGraphModel\n\n"
        "    def wrong(self, graph, signals):\n"
        "        return {\"forecast\": \"substituted\"}\n\n",
    )
    training_source = """\
def train_model(*, model, source_graph, features):
    owner = model.%s
    owner.forward = owner.wrong
    return model
""" % dispatch_attribute
    run_dir = _write_package(
        tmp_path,
        model_source=model_source,
        training_source=training_source,
    )

    with pytest.raises(GraphCallableCoverageError):
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )
    with pytest.raises(GraphCallableCoverageError):
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )


def test_notebook_cannot_mutate_numeric_constructor_namespace(tmp_path):
    run_dir = _write_package(tmp_path)
    cells = ["import numpy as np\nnp.array = list\n" + _NOTEBOOK_CELLS[0]]

    with pytest.raises(GraphCallableCoverageError):
        prove_constructor_notebook_flow(
            cells,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )


def test_duplicate_constructor_route_is_a_producer_disagreement(tmp_path):
    run_dir = _write_package(tmp_path)
    with (run_dir / "method" / "method.py").open("a", encoding="utf-8") as stream:
        stream.write(
            "\n\ndef construct_graph_exact(*, features_exact, similarity_threshold_exact):\n"
            "    return [[0, 0], [0, 0]], {}\n"
        )
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "from method import construct_graph_exact, predict, train_model",
            "from method.method import construct_graph_exact\n"
            "from method import predict, train_model",
        )
    ]

    with pytest.raises(GraphCallableProducerError) as raised:
        prove_constructor_notebook_flow(
            cells,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code == "graph_constructor_route_disagreement"


def test_indirect_constructor_route_is_honest_coverage(tmp_path):
    run_dir = _write_package(tmp_path)
    cells = [
        """\
from method import predict, train_model
edge_index = prepare_graph_through_package(features)
trained = train_model(model=model, source_graph=edge_index, features=features)
forecast = predict(model=trained, graph_payload=edge_index, signals=signals)
"""
    ]

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code == "graph_constructor_exact_import_missing"


def test_ambiguous_root_reexport_cannot_hide_constructor_duplicate(tmp_path):
    run_dir = _write_package(tmp_path)
    with (run_dir / "method" / "method.py").open("a", encoding="utf-8") as stream:
        stream.write(
            "\n\ndef construct_graph_exact(*, features_exact, similarity_threshold_exact):\n"
            "    return [[0, 0], [0, 0]], {}\n"
        )
    (run_dir / "method" / "__init__.py").write_text(
        _INIT_SOURCE
        + "\nfrom .method import construct_graph_exact\n",
        encoding="utf-8",
    )

    with pytest.raises(GraphCallableProducerError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code == "graph_constructor_route_disagreement"


def test_root_reexport_alias_cannot_impersonate_declared_constructor(tmp_path):
    run_dir = _write_package(tmp_path)
    model_path = run_dir / "method" / "model.py"
    model_path.write_text(
        model_path.read_text(encoding="utf-8").replace(
            "def construct_graph_exact(", "def wrong_constructor("
        ),
        encoding="utf-8",
    )
    (run_dir / "method" / "__init__.py").write_text(
        _INIT_SOURCE.replace(
            "from .model import aggregate_neighbors_exact, construct_graph_exact",
            "from .model import aggregate_neighbors_exact, "
            "wrong_constructor as construct_graph_exact",
        ),
        encoding="utf-8",
    )

    with pytest.raises(GraphCallableProducerError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code == "graph_constructor_route_disagreement"


def test_constructor_import_cannot_be_shadowed_before_live_call(tmp_path):
    run_dir = _write_package(tmp_path)
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "edge_index, graph_metadata = construct_graph_exact(",
            "construct_graph_exact = wrong_constructor\n"
            "edge_index, graph_metadata = construct_graph_exact(",
        )
    ]

    with pytest.raises(GraphCallableProducerError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_binding_shadowed"


def test_constructor_import_cannot_be_shadowed_inside_module_control(tmp_path):
    run_dir = _write_package(tmp_path)
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "edge_index, graph_metadata = construct_graph_exact(",
            "if True:\n"
            "    construct_graph_exact = wrong_constructor\n"
            "edge_index, graph_metadata = construct_graph_exact(",
        )
    ]

    with pytest.raises(GraphCallableProducerError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_binding_shadowed"


def test_unrelated_same_leaf_call_does_not_disagree_with_exact_route(tmp_path):
    run_dir = _write_package(tmp_path)
    cells = [_NOTEBOOK_CELLS[0] + "\nother.construct_graph_exact(features)\n"]

    proof = prove_constructor_notebook_flow(
        cells, _method_spec(), _arch_contract(), run_dir / "method"
    )

    assert proof["status"] == "pass"


@pytest.mark.parametrize(
    ("old", "new", "code"),
    [
        (
            'similarity_threshold_exact=cfg["similarity_threshold"]',
            "similarity_threshold_exact=0.5",
            "graph_constructor_parameter_binding_mismatch",
        ),
        (
            'similarity_threshold_exact=cfg["similarity_threshold"]',
            'similarity_threshold_exact=cfg["another_threshold"]',
            "graph_constructor_parameter_binding_mismatch",
        ),
        (
            "features_exact=features",
            "features_exact=unrelated_features",
            "graph_constructor_feature_binding_mismatch",
        ),
    ],
)
def test_constructor_live_call_requires_exact_feature_and_parameter_authority(
    tmp_path, old, new, code
):
    run_dir = _write_package(tmp_path)
    cells = [_NOTEBOOK_CELLS[0].replace(old, new)]

    with pytest.raises(GraphCallableProducerError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == code


def test_constructor_parameter_alias_is_honest_coverage(tmp_path):
    run_dir = _write_package(tmp_path)
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "edge_index, graph_metadata = construct_graph_exact(",
            'threshold = cfg["similarity_threshold"]\n'
            "edge_index, graph_metadata = construct_graph_exact(",
        ).replace(
            'similarity_threshold_exact=cfg["similarity_threshold"]',
            "similarity_threshold_exact=threshold",
        )
    ]

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == (
        "graph_constructor_parameter_binding_grammar_unsupported"
    )


def test_rendered_cfg_cannot_be_rebound_before_constructor(tmp_path):
    run_dir = _write_package(tmp_path)
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "edge_index, graph_metadata = construct_graph_exact(",
            "cfg = {'similarity_threshold': 0.5}\n"
            "edge_index, graph_metadata = construct_graph_exact(",
        )
    ]

    with pytest.raises(GraphCallableProducerError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_cfg_authority_rebound"


def test_module_qualified_import_survives_sibling_package_import(tmp_path):
    run_dir = _write_package(tmp_path)
    cells = [
        _NOTEBOOK_CELLS[0]
        .replace("from method import TinyGraphModel\n", "")
        .replace(
            "from method import construct_graph_exact, predict, train_model",
                "import method.model\nfrom method import predict, train_model\n"
                "import json",
        )
        .replace(
            "construct_graph_exact(", "method.model.construct_graph_exact("
        )
        .replace("TinyGraphModel()", "method.model.TinyGraphModel()")
    ]

    proof = prove_constructor_notebook_flow(
        cells, _method_spec(), _arch_contract(), run_dir / "method"
    )

    assert proof["status"] == "pass"


def test_reassigned_constructor_output_cannot_reach_fitting_or_inference(tmp_path):
    run_dir = _write_package(tmp_path)
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "trained = train_model", "edge_index = unrelated_graph\ntrained = train_model"
        )
    ]

    with pytest.raises(GraphCallableProducerError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_inference_path_missing"


@pytest.mark.parametrize(
    "mutation",
    [
        "edge_index[:] = 0",
        "edge_index[0] = unrelated_edges",
        "mutate_graph(edge_index)",
    ],
)
def test_constructor_output_mutation_or_escape_is_honest_coverage(
    tmp_path, mutation
):
    run_dir = _write_package(tmp_path)
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "trained = train_model", f"{mutation}\ntrained = train_model"
        )
    ]

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code in {
        "graph_constructor_fitting_original_mutation_unsupported",
        "graph_constructor_inference_mutation_unsupported",
        "graph_constructor_inference_alias_transform_unsupported",
    }


def test_match_capture_cannot_replace_constructor_graph_before_inference(
    tmp_path,
):
    run_dir = _write_package(tmp_path)
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "forecast = predict(",
            "match unrelated_graph:\n"
            "    case edge_index:\n"
            "        pass\n"
            "forecast = predict(",
        )
    ]

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == (
        "graph_constructor_inference_control_flow_unsupported"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        'globals()["predict"] = wrong_predict',
        'globals()["edge_index"] = unrelated_graph',
        'locals()["edge_index"] = unrelated_graph',
        'exec("edge_index = unrelated_graph")',
        'g = globals\ng()["edge_index"] = unrelated_graph',
        'from builtins import globals as g\ng()["edge_index"] = unrelated_graph',
        'import builtins\ngetattr(builtins, "globals")()["edge_index"] = unrelated_graph',
    ],
)
def test_dynamic_notebook_namespace_cannot_replace_live_bindings(
    tmp_path, mutation
):
    run_dir = _write_package(tmp_path)
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "fitting_graph = edge_index.copy()",
            mutation + "\nfitting_graph = edge_index.copy()",
        )
    ]

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_dynamic_namespace_unsupported"


@pytest.mark.parametrize(
    "unsupported_line",
    [
        "%run mutate_graph.py",
        "!python mutate_graph.py",
    ],
)
def test_notebook_magic_or_shell_escape_is_not_silently_discarded(
    tmp_path, unsupported_line
):
    run_dir = _write_package(tmp_path)
    cells = [unsupported_line + "\n" + _NOTEBOOK_CELLS[0]]

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_notebook_grammar_unsupported"


def test_notebook_get_ipython_escape_is_dynamic_namespace(tmp_path):
    run_dir = _write_package(tmp_path)
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "fitting_graph = edge_index.copy()",
            "get_ipython().run_line_magic('run', 'mutate_graph.py')\n"
            "fitting_graph = edge_index.copy()",
        )
    ]

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_dynamic_namespace_unsupported"


@pytest.mark.parametrize(
    ("marker", "mutation", "code"),
    [
        (
            "edge_index, graph_metadata = construct_graph_exact(",
            "construct_graph_exact.__code__ = wrong_constructor.__code__",
            "graph_constructor_binding_shadowed",
        ),
        (
            "trained = train_model(",
            "train_model.__code__ = wrong_train.__code__",
            "graph_fitting_binding_shadowed",
        ),
        (
            "forecast = predict(",
            "TinyGraphModel.forward = wrong_forward",
            "graph_architecture_model_binding_shadowed",
        ),
        (
            "forecast = predict(",
            "predict.__code__ = wrong_predict.__code__",
            "graph_inference_binding_shadowed",
        ),
    ],
)
def test_notebook_cannot_mutate_exact_callable_objects_before_use(
    tmp_path, marker, mutation, code
):
    run_dir = _write_package(tmp_path)
    cells = [_NOTEBOOK_CELLS[0].replace(marker, mutation + "\n" + marker)]

    with pytest.raises(
        (GraphCallableProducerError, GraphCallableCoverageError)
    ) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code in {
        code,
        "graph_constructor_notebook_route_escape_unsupported",
    }


def test_constructor_output_must_feed_both_exact_graph_parameters(tmp_path):
    run_dir = _write_package(tmp_path)
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "source_graph=fitting_graph", "source_graph=unrelated_graph"
        )
    ]

    with pytest.raises(GraphCallableProducerError) as raised:
        prove_constructor_notebook_flow(
            cells,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code == "graph_constructor_fitting_path_missing"


def test_constructor_nested_control_flow_is_honest_coverage(tmp_path):
    run_dir = _write_package(tmp_path)
    constructor_call = (
        "edge_index, graph_metadata = construct_graph_exact(\n"
        "    features_exact=features,\n"
        '    similarity_threshold_exact=cfg["similarity_threshold"],\n'
        ")"
    )
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            constructor_call,
            "if enabled:\n"
            + "\n".join(f"    {line}" for line in constructor_call.splitlines()),
        )
    ]

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code == "graph_constructor_control_flow_unsupported"


def test_exact_message_helper_result_reaches_forward_return(tmp_path):
    run_dir = _write_package(tmp_path)

    proof = prove_message_live_path(
        _method_spec(), _arch_contract(), run_dir / "method"
    )

    assert proof["status"] == "pass"
    assert proof["callable"] == {
        "module": "method.model",
        "qualname": "aggregate_neighbors_exact",
    }
    assert proof["helper_result_consumed"] is True
    assert proof["graph_parameter"] == "graph"
    assert proof["neighbor_signal_parameter"] == "signals"
    assert proof["output_root"] == "outputs.forecast"
    assert proof["output_binding"] == "forecast"
    assert proof["phases"] == ["fitting", "inference"]


def test_message_helper_call_cannot_be_discarded(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        'return {"forecast": forecast}', 'return {"forecast": signals}'
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableProducerError) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_message_result_discarded"


@pytest.mark.parametrize(
    "returned",
    [
        '{"forecast": signals, "aux": forecast}',
        '{"forecast": None, "aux": {"nested": forecast}}',
    ],
)
def test_message_helper_must_reach_the_exact_declared_output_leaf(
    tmp_path, returned
):
    model_source = _MODEL_SOURCE.replace(
        'return {"forecast": forecast}', f"return {returned}"
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises((GraphCallableProducerError, GraphCallableCoverageError)) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code in {
        "graph_message_output_root_mismatch",
        "graph_message_result_flow_unsupported",
    }


def test_unconditional_early_forward_return_cannot_hide_dead_helper(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "        forecast = aggregate_neighbors_exact(",
        '        return {"forecast": signals}\n'
        "        forecast = aggregate_neighbors_exact(",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_live_return_flow_unsupported"


def test_missing_message_helper_call_is_a_producer_disagreement(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "        forecast = aggregate_neighbors_exact(\n"
        "            graph_exact=graph,\n"
        "            signals_exact=signals,\n"
        "        )\n",
        "        forecast = signals\n",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableProducerError) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_message_live_call_missing"


def test_nested_message_flow_is_honest_coverage(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "        forecast = aggregate_neighbors_exact(\n"
        "            graph_exact=graph,\n"
        "            signals_exact=signals,\n"
        "        )\n",
        "        if graph is not None:\n"
        "            forecast = aggregate_neighbors_exact(\n"
        "                graph_exact=graph,\n"
        "                signals_exact=signals,\n"
        "            )\n",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code in {
        "graph_message_control_flow_unsupported",
        "graph_live_statement_unsupported",
    }


def test_positional_message_binding_is_honest_coverage(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "        forecast = aggregate_neighbors_exact(\n"
        "            graph_exact=graph,\n"
        "            signals_exact=signals,\n"
        "        )\n",
        "        forecast = aggregate_neighbors_exact(graph, signals)\n",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_message_helper_call_grammar_unsupported"


@pytest.mark.parametrize(
    "returned",
    [
        '{"forecast": forecast * 0}',
        '{"forecast": torch.zeros_like(forecast)}',
        '{"forecast": decoder(forecast)}',
    ],
)
def test_message_result_transform_cannot_certify_live_use(tmp_path, returned):
    model_source = _MODEL_SOURCE.replace(
        'return {"forecast": forecast}', f"return {returned}"
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_message_result_flow_unsupported"


def test_transparent_message_alias_and_container_remain_supported(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        'return {"forecast": forecast}',
        'forecast_alias = forecast\n        return {"forecast": forecast_alias}',
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    proof = prove_message_live_path(
        _method_spec(), _arch_contract(), run_dir / "method"
    )

    assert proof["status"] == "pass"
    assert proof["output_binding"] == "forecast"


def test_message_output_root_cannot_name_an_inference_input(tmp_path):
    run_dir = _write_package(tmp_path)
    spec = _method_spec()
    spec["methodology_replication_contract"]["homogeneous_graph_mechanism"][
        "message_passing"
    ]["output_root"] = "batch.signals"

    with pytest.raises(GraphCallableProducerError) as raised:
        prove_message_live_path(spec, _arch_contract(), run_dir / "method")

    assert raised.value.code == "graph_message_output_root_is_input"


def test_message_output_root_must_match_direct_helper_binding(tmp_path):
    run_dir = _write_package(tmp_path)
    spec = _method_spec()
    spec["methodology_replication_contract"]["homogeneous_graph_mechanism"][
        "message_passing"
    ]["output_root"] = "outputs.graph_embeddings"

    with pytest.raises(GraphCallableProducerError) as raised:
        prove_message_live_path(spec, _arch_contract(), run_dir / "method")

    assert raised.value.code == "graph_message_output_root_mismatch"


def test_distinct_intermediate_output_root_is_recorded_without_final_root_guess(
    tmp_path,
):
    model_source = _MODEL_SOURCE.replace(
        "forecast = aggregate_neighbors_exact(",
        "graph_embeddings = aggregate_neighbors_exact(",
    ).replace(
        'return {"forecast": forecast}',
        'return {"forecast": graph_embeddings}',
    )
    run_dir = _write_package(tmp_path, model_source=model_source)
    spec = _method_spec()
    spec["methodology_replication_contract"]["homogeneous_graph_mechanism"][
        "message_passing"
    ]["output_root"] = "outputs.graph_embeddings"

    proof = prove_message_live_path(spec, _arch_contract(), run_dir / "method")

    assert proof["output_root"] == "outputs.graph_embeddings"
    assert proof["output_binding"] == "graph_embeddings"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("graph_exact=graph", "graph_exact=graph * 0"),
        ("signals_exact=signals", "signals_exact=signals * 0"),
        (
            "forecast = aggregate_neighbors_exact(",
            "mutated_graph = graph * 0\n"
            "        forecast = aggregate_neighbors_exact(",
        ),
    ],
)
def test_message_inputs_require_transparent_exact_bindings(tmp_path, old, new):
    model_source = _MODEL_SOURCE.replace(old, new)
    if "mutated_graph" in model_source:
        model_source = model_source.replace(
            "graph_exact=graph", "graph_exact=mutated_graph"
        )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code in {
        "graph_message_input_flow_unsupported",
        "graph_message_input_escape_unsupported",
    }


@pytest.mark.parametrize(
    "mutation",
    ["graph[:] = 0", "mutate_graph(graph)", "graph.zero_()"],
)
def test_message_input_mutation_or_escape_is_honest_coverage(tmp_path, mutation):
    model_source = _MODEL_SOURCE.replace(
        "        forecast = aggregate_neighbors_exact(",
        f"        {mutation}\n        forecast = aggregate_neighbors_exact(",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code in {
        "graph_message_input_mutation_unsupported",
        "graph_message_input_escape_unsupported",
    }


def test_message_helper_result_container_cannot_be_mutated(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "        return {\"forecast\": forecast}",
        "        result = {\"forecast\": forecast}\n"
        "        result[\"forecast\"] = signals\n"
        "        return result",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_message_input_mutation_unsupported"


def test_message_helper_cannot_be_shadowed_in_forward(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "        forecast = aggregate_neighbors_exact(",
        "        aggregate_neighbors_exact = wrong_callable\n"
        "        forecast = aggregate_neighbors_exact(",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableProducerError) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_message_binding_shadowed"


def test_external_message_helper_import_cannot_be_shadowed_in_model(tmp_path):
    helper_source = """\
def aggregate_neighbors_exact(*, graph_exact, signals_exact):
    return [graph_exact, signals_exact]


"""
    model_source = (
        "from .training import aggregate_neighbors_exact\n\n"
        + _MODEL_SOURCE.replace(
            "def aggregate_neighbors_exact(*, graph_exact, signals_exact):\n"
            "    return [graph_exact, signals_exact]\n\n\n",
            "",
        )
        + "\n\ndef aggregate_neighbors_exact(*, graph_exact, signals_exact):\n"
        "    return signals_exact\n"
    )
    run_dir = _write_package(
        tmp_path,
        model_source=model_source,
        training_source=helper_source + _TRAINING_SOURCE,
    )
    spec = _method_spec()
    spec["methodology_replication_contract"]["homogeneous_graph_mechanism"][
        "message_passing"
    ]["callable"]["module"] = "method.training"

    with pytest.raises(GraphCallableProducerError) as raised:
        prove_message_live_path(spec, _arch_contract(), run_dir / "method")

    assert raised.value.code == "graph_message_binding_shadowed"


def test_external_message_helper_cannot_be_shadowed_under_module_control(
    tmp_path,
):
    helper_source = """\
def aggregate_neighbors_exact(*, graph_exact, signals_exact):
    return [graph_exact, signals_exact]


"""
    model_source = (
        "from .training import aggregate_neighbors_exact\n\n"
        + _MODEL_SOURCE.replace(
            "def aggregate_neighbors_exact(*, graph_exact, signals_exact):\n"
            "    return [graph_exact, signals_exact]\n\n\n",
            "",
        )
        + "\n\nif True:\n    aggregate_neighbors_exact = wrong_callable\n"
    )
    run_dir = _write_package(
        tmp_path,
        model_source=model_source,
        training_source=helper_source + _TRAINING_SOURCE,
    )
    spec = _method_spec()
    spec["methodology_replication_contract"]["homogeneous_graph_mechanism"][
        "message_passing"
    ]["callable"]["module"] = "method.training"

    with pytest.raises(
        (GraphCallableProducerError, GraphCallableCoverageError)
    ) as raised:
        prove_message_live_path(spec, _arch_contract(), run_dir / "method")

    assert raised.value.code in {
        "graph_message_binding_shadowed",
        "graph_live_module_init_unsupported",
    }


def test_architecture_class_final_binding_must_match_analyzed_class(tmp_path):
    model_source = (
        _MODEL_SOURCE
        + "\n\nclass WrongModel:\n"
        "    def forward(self, graph, signals):\n"
        "        return {\"forecast\": signals}\n\n"
        "TinyGraphModel = WrongModel\n"
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableProducerError) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_architecture_class_binding_shadowed"


@pytest.mark.parametrize("replacement", ["assignment", "decorator"])
def test_pluggable_final_binding_must_match_analyzed_bridge(
    tmp_path, replacement
):
    if replacement == "assignment":
        method_source = (
            _METHOD_SOURCE
            + "\n\ndef wrong_predict(**kwargs):\n"
            "    return kwargs.get('signals')\n\n"
            "predict = wrong_predict\n"
        )
        error_type = GraphCallableProducerError
        code = "graph_pluggable_binding_shadowed"
    else:
        method_source = (
            "def replace_predict(function):\n"
            "    return function\n\n\n"
            + _METHOD_SOURCE.replace(
                "def predict", "@replace_predict\ndef predict"
            )
        )
        error_type = GraphCallableCoverageError
        code = "graph_live_callable_header_unsupported"
    run_dir = _write_package(tmp_path, method_source=method_source)

    with pytest.raises(error_type) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code == code


@pytest.mark.parametrize(
    ("mutation", "error_type", "code"),
    [
        (
            "predict.__code__ = wrong_predict.__code__",
            GraphCallableProducerError,
            "graph_pluggable_binding_shadowed",
        ),
        (
            "setattr(predict, '__code__', wrong_predict.__code__)",
            GraphCallableCoverageError,
            "graph_pluggable_escape_unsupported",
        ),
    ],
)
def test_pluggable_callable_object_cannot_be_mutated(
    tmp_path, mutation, error_type, code
):
    method_source = (
        _METHOD_SOURCE
        + "\n\ndef wrong_predict(*, model, graph_payload, signals):\n"
        "    return signals\n\n"
        + mutation
        + "\n"
    )
    run_dir = _write_package(tmp_path, method_source=method_source)

    with pytest.raises(error_type) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code == code


@pytest.mark.parametrize(
    "escape",
    [
        "mutate(cfg)",
        "cfg_alias = cfg\nmutate(cfg_alias)",
    ],
)
def test_cfg_authority_cannot_escape_before_graph_construction(tmp_path, escape):
    run_dir = _write_package(tmp_path)
    marker = "edge_index, graph_metadata = construct_graph_exact("
    cells = [_NOTEBOOK_CELLS[0].replace(marker, f"{escape}\n{marker}")]

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == (
        "graph_constructor_cfg_authority_escape_unsupported"
    )


def test_match_capture_cannot_replace_cfg_authority(tmp_path):
    run_dir = _write_package(tmp_path)
    marker = "edge_index, graph_metadata = construct_graph_exact("
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            marker,
            "match unrelated_cfg:\n"
            "    case cfg:\n"
            "        pass\n"
            + marker,
        )
    ]

    with pytest.raises(GraphCallableProducerError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_cfg_authority_rebound"


def test_fitting_requires_the_exact_declared_architecture_instance(tmp_path):
    run_dir = _write_package(tmp_path)
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "train_model(model=model,",
            "train_model(model=unrelated_model,",
        )
    ]

    with pytest.raises(GraphCallableProducerError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_fitting_path_missing"


@pytest.mark.parametrize(
    "training_source",
    [
        """\
class WrongModel:
    def forward(self, graph, signals):
        return {"forecast": signals}


def train_model(*, model, source_graph, features):
    return WrongModel()
""",
        """\
def wrong_forward(graph, signals):
    return {"forecast": signals}


def train_model(*, model, source_graph, features):
    model.forward = wrong_forward
    return model
""",
    ],
)
def test_fitting_source_must_preserve_declared_model_identity(
    tmp_path, training_source
):
    run_dir = _write_package(tmp_path, training_source=training_source)

    with pytest.raises(GraphCallableProducerError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code == "graph_fitting_model_identity_not_preserved"


def test_fitting_preserves_identity_through_device_moves_and_attribute_reads(
    tmp_path,
):
    training_source = """\
def train_model(*, model, source_graph, features):
    if use_cuda:
        model = model.cuda()
    model = model.to(device)
    context_length = model.context_length
    forecast_horizon = model.forecast_horizon
    model.train()
    predictions = model.forward(features)
    return model
"""
    model_source = _MODEL_SOURCE.replace(
        "class TinyGraphModel:\n",
        "class TinyGraphModel:\n"
        "    def __init__(self):\n"
        "        self.context_length = 2\n"
        "        self.forecast_horizon = 1\n\n",
    )
    run_dir = _write_package(
        tmp_path, model_source=model_source, training_source=training_source
    )

    proof = prove_constructor_notebook_flow(
        _NOTEBOOK_CELLS,
        _method_spec(),
        _arch_contract(),
        run_dir / "method",
    )

    assert proof["fitting_model_identity_preserved"] is True


def test_fitting_rejects_overridden_identity_returning_model_method(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "    def forward(self, graph, signals):",
        "    def to(self, device):\n"
        "        return WrongModel()\n\n"
        "    def forward(self, graph, signals):",
    )
    training_source = """\
def train_model(*, model, source_graph, features):
    model = model.to(device)
    return model
"""
    run_dir = _write_package(
        tmp_path,
        model_source="class WrongModel:\n    pass\n\n\n" + model_source,
        training_source=training_source,
    )

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code == "graph_fitting_model_method_override_unsupported"


@pytest.mark.parametrize(
    "training_body",
    [
        "    set_mode = model.train\n    set_mode()\n    return model",
        "    callback = model.poison\n    callback()\n    return model",
        "    model.poison()\n    return model",
    ],
)
def test_fitting_cannot_dispatch_through_unproved_model_method(
    tmp_path, training_body
):
    model_source = _MODEL_SOURCE.replace(
        "    def forward(self, graph, signals):",
        "    def train(self):\n"
        "        TinyGraphModel.forward = wrong_forward\n"
        "        return self\n\n"
        "    def poison(self):\n"
        "        TinyGraphModel.forward = wrong_forward\n\n"
        "    def forward(self, graph, signals):",
    )
    model_source = (
        "def wrong_forward(self, graph, signals):\n"
        "    return {\"forecast\": signals}\n\n\n"
        + model_source
    )
    training_source = (
        "def train_model(*, model, source_graph, features):\n"
        + training_body
        + "\n"
    )
    run_dir = _write_package(
        tmp_path,
        model_source=model_source,
        training_source=training_source,
    )

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code in {
        "graph_fitting_model_identity_grammar_unsupported",
        "graph_fitting_model_method_override_unsupported",
    }


def test_fitting_cannot_dispatch_through_load_state_dict_extension(tmp_path):
    model_source = (
        "from torch import nn\n\n"
        + _MODEL_SOURCE.replace(
            "class TinyGraphModel:\n",
            "class TinyGraphModel(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n\n"
            "    def wrong(self, graph, signals):\n"
            "        return {\"forecast\": signals}\n\n"
            "    def _load_from_state_dict(self, *args, **kwargs):\n"
            "        TinyGraphModel.forward = TinyGraphModel.wrong\n\n",
        )
    )
    training_source = """\
def train_model(*, model, source_graph, features):
    model.load_state_dict({})
    return model
"""
    run_dir = _write_package(
        tmp_path,
        model_source=model_source,
        training_source=training_source,
    )

    with pytest.raises(GraphCallableCoverageError):
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )
    with pytest.raises(GraphCallableCoverageError):
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )


def test_inherited_train_cannot_reach_poisoned_producer_child_module(tmp_path):
    model_source = (
        "from torch import nn\n\n"
        + _MODEL_SOURCE.replace(
            "class TinyGraphModel:\n",
            "class PoisonChild(nn.Module):\n"
            "    def train(self, mode=True):\n"
            "        TinyGraphModel.forward = TinyGraphModel.wrong\n"
            "        return self\n\n\n"
            "class TinyGraphModel(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.child = PoisonChild()\n\n"
            "    def wrong(self, graph, signals):\n"
            "        return {\"forecast\": signals}\n\n",
        )
    )
    training_source = """\
def train_model(*, model, source_graph, features):
    model.train()
    return model
"""
    run_dir = _write_package(
        tmp_path,
        model_source=model_source,
        training_source=training_source,
    )

    with pytest.raises(GraphCallableCoverageError) as constructor_error:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )
    with pytest.raises(GraphCallableCoverageError) as message_error:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert constructor_error.value.code == (
        "graph_fitting_model_method_override_unsupported"
    )
    assert message_error.value.code == "graph_message_dynamic_dispatch_unsupported"


def test_inherited_train_cannot_reach_parent_through_child_state(tmp_path):
    model_source = (
        "from torch import nn\n\n"
        "def wrong_forward(self, graph, signals):\n"
        "    return {\"forecast\": signals}\n\n\n"
        "class PoisonChild(nn.Module):\n"
        "    def train(self, mode=True):\n"
        "        self.owner.forward = wrong_forward\n"
        "        return super().train(mode)\n\n\n"
        + _MODEL_SOURCE.replace(
            "class TinyGraphModel:\n",
            "class TinyGraphModel(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.child = PoisonChild()\n"
            "        self.child.owner = self\n\n",
        )
    )
    training_source = """\
def train_model(*, model, source_graph, features):
    model.train()
    return model
"""
    run_dir = _write_package(
        tmp_path,
        model_source=model_source,
        training_source=training_source,
    )

    with pytest.raises(GraphCallableCoverageError) as constructor_error:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )
    with pytest.raises(GraphCallableCoverageError) as message_error:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert constructor_error.value.code == (
        "graph_fitting_model_method_override_unsupported"
    )
    assert message_error.value.code == "graph_message_dynamic_dispatch_unsupported"


def test_inherited_train_allows_benign_producer_child_module(tmp_path):
    model_source = (
        "from torch import nn\n\n"
        + _MODEL_SOURCE.replace(
            "class TinyGraphModel:\n",
            "class BenignChild(nn.Module):\n"
            "    def train(self, mode=True):\n"
            "        return super().train(mode)\n\n"
            "    def forward(self, signals):\n"
            "        return signals\n\n\n"
            "class TinyGraphModel(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.child = BenignChild()\n\n",
        )
    )
    training_source = """\
def train_model(*, model, source_graph, features):
    model.train()
    return model
"""
    run_dir = _write_package(
        tmp_path,
        model_source=model_source,
        training_source=training_source,
    )

    constructor_proof = prove_constructor_notebook_flow(
        _NOTEBOOK_CELLS,
        _method_spec(),
        _arch_contract(),
        run_dir / "method",
    )
    message_proof = prove_message_live_path(
        _method_spec(), _arch_contract(), run_dir / "method"
    )

    assert constructor_proof["status"] == "pass"
    assert message_proof["status"] == "pass"


def test_fitting_cannot_recover_model_class_by_generic_introspection(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "    def forward(self, graph, signals):",
        "    def wrong(self, graph, signals):\n"
        "        return {\"forecast\": signals}\n\n"
        "    def forward(self, graph, signals):",
    )
    training_source = """\
def train_model(*, model, source_graph, features):
    classes = object.__subclasses__()
    for cls in classes:
        if cls.__name__ == "TinyGraphModel":
            cls.forward = cls.wrong
    return model
"""
    run_dir = _write_package(
        tmp_path,
        model_source=model_source,
        training_source=training_source,
    )

    with pytest.raises(GraphCallableCoverageError):
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )


@pytest.mark.parametrize(
    "training_source",
    [
        """\
import operator


def train_model(*, model, source_graph, features):
    owner = operator.attrgetter("__class__")(model)
    owner.forward = owner.wrong
    return model
""",
        """\
from operator import methodcaller as recover


def train_model(*, model, source_graph, features):
    classes = recover("__subclasses__")(object)
    for owner in classes:
        if owner.__name__ == "TinyGraphModel":
            owner.forward = owner.wrong
    return model
""",
        """\
import inspect as reflection


def train_model(*, model, source_graph, features):
    owner = reflection.getmro(model.__class__)[0]
    owner.forward = owner.wrong
    return model
""",
        """\
from gc import get_objects as recover


def train_model(*, model, source_graph, features):
    for owner in recover():
        if getattr(owner, "__name__", None) == "TinyGraphModel":
            owner.forward = owner.wrong
    return model
""",
    ],
)
def test_fitting_reflection_helpers_cannot_recover_architecture_class(
    tmp_path, training_source
):
    model_source = _MODEL_SOURCE.replace(
        "    def forward(self, graph, signals):",
        "    def wrong(self, graph, signals):\n"
        "        return {\"forecast\": signals}\n\n"
        "    def forward(self, graph, signals):",
    )
    run_dir = _write_package(
        tmp_path,
        model_source=model_source,
        training_source=training_source,
    )

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code == "graph_fitting_model_dynamic_namespace_unsupported"


def test_fitting_local_may_share_unimported_pluggable_spelling(tmp_path):
    arch_contract = _arch_contract()
    arch_contract["pluggable_component"]["name"] = "forecast"
    method_source = _METHOD_SOURCE.replace("predict", "forecast")
    init_source = _INIT_SOURCE.replace("predict", "forecast")
    notebook_cells = [_NOTEBOOK_CELLS[0].replace("predict", "forecast")]
    training_source = """\
def train_model(*, model, source_graph, features):
    forecast = model.forward(graph=source_graph, signals=features)
    return model
"""
    run_dir = _write_package(
        tmp_path,
        training_source=training_source,
        method_source=method_source,
        init_source=init_source,
    )

    proof = prove_constructor_notebook_flow(
        notebook_cells,
        _method_spec(),
        arch_contract,
        run_dir / "method",
    )

    assert proof["fitting_model_identity_preserved"] is True


@pytest.mark.parametrize(
    "protected_import",
    [
        "import method\nmethod.predict.__code__ = wrong_predict.__code__",
        "from method import predict\npredict.__code__ = wrong_predict.__code__",
        (
            "from method import method as mm\n"
            "mm.predict.__code__ = wrong_predict.__code__"
        ),
        (
            "def train_model(*, model, source_graph, features):\n"
            "    import method\n"
            "    method.predict.__code__ = wrong_predict.__code__\n"
            "    return model"
        ),
    ],
)
def test_fitting_cannot_mutate_package_root_graph_callable(
    tmp_path, protected_import
):
    wrong_predict = (
        "def wrong_predict(*, model, graph_payload, signals):\n"
        "    return signals\n\n\n"
    )
    if protected_import.startswith("def train_model"):
        training_source = wrong_predict + protected_import + "\n"
    else:
        training_source = (
            wrong_predict
            + protected_import
            + "\n\n\ndef train_model(*, model, source_graph, features):\n"
            "    return model\n"
        )
    run_dir = _write_package(tmp_path, training_source=training_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code in {
        "graph_fitting_protected_callable_mutation_unsupported",
        "graph_fitting_model_identity_grammar_unsupported",
        "graph_live_module_init_unsupported",
    }


def test_fitting_reflective_module_lookup_is_outside_closed_grammar(tmp_path):
    training_source = """\
import importlib


def train_model(*, model, source_graph, features):
    module = importlib.import_module("method.method")
    module.predict = wrong_predict
    return model
"""
    run_dir = _write_package(tmp_path, training_source=training_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code == "graph_fitting_model_dynamic_namespace_unsupported"


def test_fitting_cannot_alias_inherited_model_callback_dispatch(tmp_path):
    model_source = (
        "from torch import nn\n\n\n"
        + _MODEL_SOURCE.replace(
            "class TinyGraphModel:", "class TinyGraphModel(nn.Module):"
        )
    )
    training_source = """\
def poison(module):
    module.forward = wrong_forward


def train_model(*, model, source_graph, features):
    apply_to_modules = model.apply
    apply_to_modules(poison)
    return model
"""
    run_dir = _write_package(
        tmp_path, model_source=model_source, training_source=training_source
    )

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code == "graph_fitting_model_identity_grammar_unsupported"


@pytest.mark.parametrize(
    "mutation",
    [
        "model.forward = wrong_forward",
        "setattr(model, 'forward', wrong_forward)",
    ],
)
def test_notebook_cannot_mutate_pre_fitting_model_alias_before_inference(
    tmp_path, mutation
):
    run_dir = _write_package(tmp_path)
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "forecast = predict(", mutation + "\nforecast = predict("
        )
    ]

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code in {
        "graph_architecture_model_mutation_unsupported",
        "graph_constructor_dynamic_namespace_unsupported",
    }


def test_inference_requires_the_exact_fitting_result_model(tmp_path):
    run_dir = _write_package(tmp_path)
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "predict(model=trained,",
            "predict(model=unrelated_model,",
        )
    ]

    with pytest.raises(GraphCallableProducerError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_inference_path_missing"


def test_pluggable_must_return_the_architecture_call_result(tmp_path):
    method_source = _METHOD_SOURCE.replace(
        "    return model.forward(graph=graph_payload, signals=signals)",
        "    model.forward(graph=graph_payload, signals=signals)\n    return signals",
    )
    run_dir = _write_package(tmp_path, method_source=method_source)

    with pytest.raises(GraphCallableProducerError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code == "graph_constructor_pluggable_result_discarded"


@pytest.mark.parametrize("owner", ["constructor", "pluggable"])
def test_executed_graph_callable_cannot_mutate_architecture_dispatch(
    tmp_path, owner
):
    wrong_forward = (
        "def wrong_forward(self, graph, signals):\n"
        "    return {\"forecast\": signals}\n\n\n"
    )
    model_source = wrong_forward + _MODEL_SOURCE
    method_source = _METHOD_SOURCE
    if owner == "constructor":
        model_source = model_source.replace(
            "    return np.array([[0, 1], [1, 0]]),",
            "    TinyGraphModel.forward = wrong_forward\n"
            "    return np.array([[0, 1], [1, 0]]),",
        )
        expected = "graph_constructor_source_cross_callable_effect_unsupported"
    else:
        method_source = (
            "from .model import TinyGraphModel, wrong_forward\n\n\n"
            + method_source.replace(
                "    return model.forward(",
                "    TinyGraphModel.forward = wrong_forward\n"
                "    return model.forward(",
            )
        )
        expected = "graph_pluggable_cross_callable_effect_unsupported"
    run_dir = _write_package(
        tmp_path, model_source=model_source, method_source=method_source
    )

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code == expected


def test_notebook_local_callable_cannot_mutate_architecture_dispatch(tmp_path):
    run_dir = _write_package(tmp_path)
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "forecast = predict(",
            "def poison():\n"
            "    TinyGraphModel.forward = wrong_forward\n"
            "poison()\n"
            "forecast = predict(",
        )
    ]

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == (
        "graph_constructor_notebook_local_callable_unsupported"
    )


def test_unconditional_early_pluggable_return_cannot_hide_dead_model_call(
    tmp_path,
):
    method_source = _METHOD_SOURCE.replace(
        "    return model.forward(",
        "    return signals\n    return model.forward(",
    )
    run_dir = _write_package(tmp_path, method_source=method_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code == "graph_live_return_flow_unsupported"


@pytest.mark.parametrize(
    "returned",
    [
        '{"forecast": signals, "aux": output}',
        "[signals, output]",
    ],
)
def test_pluggable_model_result_must_reach_exact_output_leaf(tmp_path, returned):
    method_source = _METHOD_SOURCE.replace(
        "    return model.forward(graph=graph_payload, signals=signals)",
        "    output = model.forward(graph=graph_payload, signals=signals)\n"
        f"    return {returned}",
    )
    run_dir = _write_package(tmp_path, method_source=method_source)

    with pytest.raises((GraphCallableProducerError, GraphCallableCoverageError)):
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )


@pytest.mark.parametrize(
    "suffix",
    [
        "\nforecast = signals",
        '\nforecast["forecast"] = signals',
        "\nforecast.clear()",
        "\nout = forecast\nout.clear()",
    ],
)
def test_notebook_inference_output_sink_cannot_be_replaced_or_escape(
    tmp_path, suffix
):
    run_dir = _write_package(tmp_path)
    cells = [_NOTEBOOK_CELLS[0] + suffix + "\n"]

    with pytest.raises((GraphCallableProducerError, GraphCallableCoverageError)):
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )


def test_notebook_inference_call_must_bind_declared_output_sink(tmp_path):
    run_dir = _write_package(tmp_path)
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "forecast = predict(", "ignored = predict("
        )
        + "forecast = signals\n"
    ]

    with pytest.raises(GraphCallableProducerError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_inference_output_sink_mismatch"


def test_custom_architecture_call_cannot_bypass_analyzed_forward(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "class TinyGraphModel:\n",
        "class TinyGraphModel:\n"
        "    def __call__(self, graph, signals):\n"
        "        return {\"forecast\": signals}\n\n",
    )
    method_source = _METHOD_SOURCE.replace("model.forward(", "model(")
    run_dir = _write_package(
        tmp_path, model_source=model_source, method_source=method_source
    )

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code in {
        "graph_pluggable_dynamic_dispatch_unsupported",
        "graph_fitting_model_method_override_unsupported",
    }


def test_class_control_call_cannot_bypass_analyzed_forward(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "class TinyGraphModel:\n",
        "class TinyGraphModel:\n"
        "    if True:\n"
        "        def __call__(self, graph, signals):\n"
        "            return {\"forecast\": signals}\n\n",
    )
    method_source = _METHOD_SOURCE.replace("model.forward(", "model(")
    run_dir = _write_package(
        tmp_path, model_source=model_source, method_source=method_source
    )

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code in {
        "graph_pluggable_dynamic_dispatch_unsupported",
        "graph_fitting_model_method_override_unsupported",
        "graph_live_module_init_unsupported",
    }


def test_fitting_uses_numpy_copy_while_inference_keeps_original(tmp_path):
    run_dir = _write_package(tmp_path)

    proof = prove_constructor_notebook_flow(
        _NOTEBOOK_CELLS,
        _method_spec(),
        _arch_contract(),
        run_dir / "method",
    )

    assert proof["fitting_copy_binding"] == "fitting_graph"
    assert proof["fitting_copy_method"] == "copy"
    assert proof["fitting_graph_array_kind"] == "numpy"
    assert proof["selected_binding"] == "edge_index"


def test_fitting_uses_torch_clone_for_sparse_tensor_graph(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "import numpy as np",
        "import numpy as np\nimport torch",
    ).replace(
        "return np.array([[0, 1], [1, 0]]), {\"source\": \"exact\"}",
        "mask = torch.as_tensor(features_exact) >= similarity_threshold_exact\n"
        "    edge_index = torch.nonzero(mask).t().contiguous()\n"
        "    return edge_index, {\"source\": \"exact\"}",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)
    cells = [_NOTEBOOK_CELLS[0].replace("edge_index.copy()", "edge_index.clone()")]

    proof = prove_constructor_notebook_flow(
        cells, _method_spec(), _arch_contract(), run_dir / "method"
    )

    assert proof["fitting_copy_method"] == "clone"
    assert proof["fitting_graph_array_kind"] == "torch"


def test_notebook_cannot_mutate_torch_tensor_clone_dispatch(tmp_path):
    pytest.importorskip("torch")
    model_source = _MODEL_SOURCE.replace(
        "import numpy as np",
        "import numpy as np\nimport torch",
    ).replace(
        "return np.array([[0, 1], [1, 0]]), {\"source\": \"exact\"}",
        "mask = torch.as_tensor(features_exact) >= similarity_threshold_exact\n"
        "    edge_index = torch.nonzero(mask).t().contiguous()\n"
        "    return edge_index, {\"source\": \"exact\"}",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)
    cells = [
        "from torch import Tensor\n"
        "Tensor.clone = Tensor.detach\n"
        + _NOTEBOOK_CELLS[0].replace(
            "edge_index.copy()", "edge_index.clone()"
        )
    ]

    with pytest.raises(GraphCallableCoverageError):
        prove_constructor_notebook_flow(
            cells,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "from torch import *\nTensor.clone = Tensor.detach",
        "import torch\ntype(torch.tensor([])).clone = object.__repr__",
        "import torch\nT = torch.tensor([]).__class__\nT.clone = T.detach",
    ],
)
def test_notebook_cannot_recover_mutable_torch_dispatch_indirectly(
    tmp_path, mutation
):
    pytest.importorskip("torch")
    model_source = _MODEL_SOURCE.replace(
        "import numpy as np",
        "import numpy as np\nimport torch",
    ).replace(
        "return np.array([[0, 1], [1, 0]]), {\"source\": \"exact\"}",
        "mask = torch.as_tensor(features_exact) >= similarity_threshold_exact\n"
        "    edge_index = torch.nonzero(mask).t().contiguous()\n"
        "    return edge_index, {\"source\": \"exact\"}",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)
    cells = [
        mutation
        + "\n"
        + _NOTEBOOK_CELLS[0].replace(
            "edge_index.copy()", "edge_index.clone()"
        )
    ]

    with pytest.raises(GraphCallableCoverageError):
        prove_constructor_notebook_flow(
            cells,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )


def test_fitting_supports_dense_numpy_adjacency_copy(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "return np.array([[0, 1], [1, 0]]), {\"source\": \"exact\"}",
        "similarity = np.asarray(features_exact)\n"
        "    adjacency = (similarity >= similarity_threshold_exact).astype(np.float32)\n"
        "    return adjacency, {\"source\": \"exact\"}",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    proof = prove_constructor_notebook_flow(
        _NOTEBOOK_CELLS,
        _method_spec(),
        _arch_contract(),
        run_dir / "method",
    )

    assert proof["fitting_graph_array_kind"] == "numpy"
    assert proof["fitting_copy_method"] == "copy"


@pytest.mark.parametrize("constructor_variant", ["custom_graph", "rebound_numpy"])
def test_fitting_copy_requires_exact_unshadowed_numeric_array_origin(
    tmp_path, constructor_variant
):
    if constructor_variant == "custom_graph":
        model_source = _MODEL_SOURCE.replace(
            "import numpy as np\n\n\n",
            "import numpy as np\n\n\n"
            "class Graph:\n"
            "    def copy(self):\n"
            "        return self\n\n\n",
        ).replace(
            "return np.array([[0, 1], [1, 0]]), {\"source\": \"exact\"}",
            "return Graph(), {\"source\": \"exact\"}",
        )
    else:
        model_source = _MODEL_SOURCE.replace(
            "def construct_graph_exact",
            "class FakeNP:\n"
            "    @staticmethod\n"
            "    def array(value):\n"
            "        return value\n\n\n"
            "np = FakeNP()\n\n\n"
            "def construct_graph_exact",
        )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code in {
        "graph_constructor_output_type_unsupported",
        "graph_live_callable_header_unsupported",
    }


def test_fitting_cannot_receive_the_mutable_inference_graph_directly(tmp_path):
    run_dir = _write_package(tmp_path)
    cells = [
        _NOTEBOOK_CELLS[0]
        .replace("fitting_graph = edge_index.copy()\n", "")
        .replace("source_graph=fitting_graph", "source_graph=edge_index")
    ]

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_fitting_copy_unsupported"


def test_fitting_snapshot_receiver_must_still_be_the_selected_graph(tmp_path):
    run_dir = _write_package(tmp_path)
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "fitting_graph = edge_index.copy()",
            "selected_graph = edge_index\n"
            "edge_index = unrelated_graph\n"
            "fitting_graph = edge_index.copy()\n"
            "edge_index = selected_graph",
        )
    ]

    with pytest.raises(GraphCallableProducerError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_fitting_copy_source_missing"


def test_fitting_call_rejects_undeclared_graph_escape_keyword(tmp_path):
    run_dir = _write_package(tmp_path)
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "source_graph=fitting_graph, features=features)",
            "source_graph=fitting_graph, features=features, "
            "poison_graph=edge_index)",
        )
    ]

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_fitting_grammar_unsupported"


@pytest.mark.parametrize(
    ("suffix", "error_type", "code"),
    [
        (
            "\nTinyGraphModel.forward = wrong_forward\n",
            GraphCallableProducerError,
            "graph_architecture_class_binding_shadowed",
        ),
        (
            "\nsetattr(TinyGraphModel, 'forward', wrong_forward)\n",
            GraphCallableCoverageError,
            "graph_architecture_class_escape_unsupported",
        ),
    ],
)
def test_module_monkeypatch_cannot_replace_analyzed_architecture(
    tmp_path, suffix, error_type, code
):
    run_dir = _write_package(tmp_path, model_source=_MODEL_SOURCE + suffix)

    with pytest.raises(error_type) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == code


def test_function_local_import_cannot_replace_exact_message_helper(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "        forecast = aggregate_neighbors_exact(",
        "        from wrong_module import aggregate_neighbors_exact\n"
        "        forecast = aggregate_neighbors_exact(",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(
        (GraphCallableProducerError, GraphCallableCoverageError)
    ) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code in {
        "graph_message_binding_shadowed",
        "graph_live_statement_unsupported",
    }


@pytest.mark.parametrize(
    "suffix",
    [
        "\nwrong_helper = lambda **kwargs: None\n"
        "match wrong_helper:\n"
        "    case aggregate_neighbors_exact:\n"
        "        pass\n",
        "\nwrong_helper = lambda **kwargs: None\n"
        "if (aggregate_neighbors_exact := wrong_helper):\n"
        "    pass\n",
    ],
)
def test_module_control_binding_cannot_replace_exact_message_helper(
    tmp_path, suffix
):
    run_dir = _write_package(tmp_path, model_source=_MODEL_SOURCE + suffix)

    with pytest.raises(
        (GraphCallableProducerError, GraphCallableCoverageError)
    ) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code in {
        "graph_message_callable_binding_shadowed",
        "graph_live_module_init_unsupported",
    }


def test_root_reexport_cannot_be_shadowed_under_module_control(tmp_path):
    run_dir = _write_package(
        tmp_path,
        init_source=(
            _INIT_SOURCE
            + "\nif True:\n    construct_graph_exact = wrong_constructor\n"
        ),
    )

    with pytest.raises(
        (GraphCallableProducerError, GraphCallableCoverageError)
    ) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code in {
        "graph_constructor_route_disagreement",
        "graph_package_init_execution_unsupported",
    }


@pytest.mark.parametrize(
    "mutation",
    [
        'globals()["construct_graph_exact"] = wrong_constructor',
        'setattr(sys.modules[__name__], "construct_graph_exact", wrong_constructor)',
        'globals()["predict"] = wrong_predict',
        'g = globals\ng()["construct_graph_exact"] = wrong_constructor',
        'from builtins import globals as g\ng()["construct_graph_exact"] = wrong_constructor',
    ],
)
def test_root_reexport_cannot_be_replaced_dynamically(tmp_path, mutation):
    run_dir = _write_package(
        tmp_path,
        init_source=_INIT_SOURCE + "\n" + mutation + "\n",
    )

    with pytest.raises(
        (GraphCallableProducerError, GraphCallableCoverageError)
    ) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code in {
        "graph_constructor_route_disagreement",
        "graph_package_init_execution_unsupported",
    }


@pytest.mark.parametrize("shadow_kind", ["instance", "class"])
def test_traversed_self_method_cannot_be_shadowed(tmp_path, shadow_kind):
    route = """\
    def route(self, graph, signals):
        forecast = aggregate_neighbors_exact(
            graph_exact=graph,
            signals_exact=signals,
        )
        return forecast
"""
    if shadow_kind == "instance":
        shadow = """\
    def __init__(self):
        self.route = wrong_route

"""
    else:
        shadow = "    route = wrong_route\n\n"
    model_source = _MODEL_SOURCE.replace(
        "        forecast = aggregate_neighbors_exact(\n"
        "            graph_exact=graph,\n"
        "            signals_exact=signals,\n"
        "        )",
        "        forecast = self.route(graph, signals)",
        1,
    ).replace(
        "class TinyGraphModel:\n",
        "def wrong_route(*args, **kwargs):\n"
        "    return None\n\n\n"
        "class TinyGraphModel:\n"
        + (shadow if shadow_kind == "instance" else "")
        + route
        + (shadow if shadow_kind == "class" else ""),
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code in {
        "graph_message_method_binding_shadowed",
        "graph_message_dynamic_dispatch_unsupported",
    }


@pytest.mark.parametrize(
    "control_source",
    [
        "        if True:\n            graph = unrelated_graph\n",
        "        del graph\n",
    ],
)
def test_control_flow_cannot_sever_message_input_provenance(
    tmp_path, control_source
):
    model_source = _MODEL_SOURCE.replace(
        "        forecast = aggregate_neighbors_exact(",
        control_source + "        forecast = aggregate_neighbors_exact(",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code in {
        "graph_message_provenance_control_flow_unsupported",
        "graph_live_statement_unsupported",
    }


def test_match_capture_cannot_sever_message_input_provenance(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "        forecast = aggregate_neighbors_exact(",
        "        match wrong_graph:\n"
        "            case graph:\n"
        "                pass\n"
        "        forecast = aggregate_neighbors_exact(",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code in {
        "graph_message_provenance_control_flow_unsupported",
        "graph_live_statement_unsupported",
    }


def test_match_capture_cannot_sever_pluggable_graph_provenance(tmp_path):
    method_source = _METHOD_SOURCE.replace(
        "    return model.forward(",
        "    match wrong_graph:\n"
        "        case graph_payload:\n"
        "            pass\n"
        "    return model.forward(",
    )
    run_dir = _write_package(tmp_path, method_source=method_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code == (
        "graph_constructor_pluggable_graph_control_flow_unsupported"
    )


def test_control_flow_cannot_discard_the_message_helper_result(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "        return {\"forecast\": forecast}",
        "        if True:\n"
        "            forecast = signals\n"
        "        return {\"forecast\": forecast}",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code in {
        "graph_message_provenance_control_flow_unsupported",
        "graph_live_statement_unsupported",
    }


@pytest.mark.parametrize(
    "dynamic_source",
    [
        "    def __getattribute__(self, name):\n"
        "        if name == 'route':\n"
        "            return wrong_route\n"
        "        return object.__getattribute__(self, name)\n\n",
        "    @property\n",
    ],
)
def test_dynamic_self_method_dispatch_is_honest_coverage(
    tmp_path, dynamic_source
):
    route_prefix = dynamic_source if dynamic_source.startswith("    def") else ""
    decorator = dynamic_source if dynamic_source.startswith("    @") else ""
    route = (
        decorator
        + "    def route(self, graph, signals):\n"
        "        forecast = aggregate_neighbors_exact(\n"
        "            graph_exact=graph,\n"
        "            signals_exact=signals,\n"
        "        )\n"
        "        return forecast\n\n"
    )
    model_source = _MODEL_SOURCE.replace(
        "        forecast = aggregate_neighbors_exact(\n"
        "            graph_exact=graph,\n"
        "            signals_exact=signals,\n"
        "        )",
        "        forecast = self.route(graph, signals)",
        1,
    ).replace(
        "class TinyGraphModel:\n",
        "def wrong_route(*args, **kwargs):\n"
        "    return None\n\n\n"
        "class TinyGraphModel:\n"
        + route_prefix
        + route,
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_message_dynamic_dispatch_unsupported"


@pytest.mark.parametrize(
    "shadow",
    [
        "    def __init__(self):\n        self.forward = wrong_route\n\n",
        "@replace_forward\n",
    ],
)
def test_forward_descriptor_cannot_replace_the_analyzed_body(tmp_path, shadow):
    prefix = ""
    if shadow.startswith("@"):
        prefix = (
            "def replace_forward(function):\n"
            "    return wrong_route\n\n\n"
        )
        model_source = _MODEL_SOURCE.replace(
            "class TinyGraphModel:\n    def forward",
            prefix + "class TinyGraphModel:\n    @replace_forward\n    def forward",
        )
    else:
        model_source = _MODEL_SOURCE.replace(
            "class TinyGraphModel:\n",
            "def wrong_route(*args, **kwargs):\n"
            "    return None\n\n\n"
            "class TinyGraphModel:\n" + shadow,
        )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code in {
        "graph_message_method_binding_shadowed",
        "graph_message_dynamic_dispatch_unsupported",
    }


@pytest.mark.parametrize(
    "init_body",
    [
        "        TinyGraphModel.forward = wrong_forward",
        "        self.register_forward_hook(replace_output)",
        "        self._forward_hooks = {1: replace_output}",
    ],
)
def test_model_initialization_cannot_replace_analyzed_forward(
    tmp_path, init_body
):
    prefix = (
        "def wrong_forward(self, graph, signals):\n"
        "    return {\"forecast\": signals}\n\n\n"
        "def replace_output(*args):\n"
        "    return {\"forecast\": args[-1]}\n\n\n"
    )
    model_source = prefix + _MODEL_SOURCE.replace(
        "class TinyGraphModel:\n",
        "class TinyGraphModel:\n"
        "    def __init__(self):\n"
        + init_body
        + "\n\n",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code in {
        "graph_message_method_binding_shadowed",
        "graph_message_dynamic_dispatch_unsupported",
    }


def test_instance_cannot_shadow_trusted_fitting_model_method(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "class TinyGraphModel:\n",
        "class TinyGraphModel:\n"
        "    def __init__(self):\n"
        "        self.to = lambda device: WrongModel()\n\n",
    )
    model_source = "class WrongModel:\n    pass\n\n\n" + model_source
    training_source = """\
def train_model(*, model, source_graph, features):
    model = model.to(device)
    return model
"""
    run_dir = _write_package(
        tmp_path, model_source=model_source, training_source=training_source
    )

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code == "graph_fitting_model_method_override_unsupported"


def test_fake_nn_module_base_cannot_authorize_implicit_model_dispatch(tmp_path):
    model_source = (
        "class FakeModule:\n"
        "    def __call__(self, **kwargs):\n"
        "        return {\"forecast\": kwargs[\"signals\"]}\n\n\n"
        "class FakeNN:\n"
        "    Module = FakeModule\n\n\n"
        "nn = FakeNN()\n\n\n"
        + _MODEL_SOURCE.replace(
            "class TinyGraphModel:", "class TinyGraphModel(nn.Module):"
        )
    )
    method_source = _METHOD_SOURCE.replace("model.forward(", "model(")
    run_dir = _write_package(
        tmp_path, model_source=model_source, method_source=method_source
    )

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS,
            _method_spec(),
            _arch_contract(),
            run_dir / "method",
        )

    assert raised.value.code in {
        "graph_pluggable_model_call_dispatch_unsupported",
        "graph_live_module_init_unsupported",
    }


def test_class_control_cannot_replace_analyzed_forward(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "class TinyGraphModel:\n",
        "def wrong_route(*args, **kwargs):\n"
        "    return None\n\n\n"
        "class TinyGraphModel:\n",
    ).replace(
        "        return {\"forecast\": forecast}",
        "        return {\"forecast\": forecast}\n\n"
        "    if True:\n"
        "        forward = wrong_route",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code in {
        "graph_message_method_binding_shadowed",
        "graph_message_dynamic_dispatch_unsupported",
    }


def test_forward_cannot_mutate_its_declaring_class_dispatch(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "        forecast = aggregate_neighbors_exact(",
        "        TinyGraphModel.forward = wrong_forward\n"
        "        forecast = aggregate_neighbors_exact(",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_message_method_binding_shadowed"


@pytest.mark.parametrize("module", ["method.data", "method.graph", "method"])
def test_callable_owners_are_the_exact_supported_modules(module):
    spec = _method_spec()
    spec["methodology_replication_contract"]["homogeneous_graph_mechanism"][
        "construction"
    ]["callable"]["module"] = module

    with pytest.raises(GraphCallableProducerError) as raised:
        graph_callable_identities(spec)

    assert raised.value.code == "graph_callable_module_not_allowed"


@pytest.mark.parametrize(
    "constructor_body",
    [
        """\
    norms = np.linalg.norm(features_exact, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = features_exact / norms
    similarity = normalized @ normalized.T
    adjacency = (similarity >= similarity_threshold_exact).astype(np.float64)
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency, {"source": "exact"}
""",
        """\
    norms = torch.norm(features_exact, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=1e-8)
    normalized = features_exact / norms
    similarity = normalized @ normalized.t()
    adjacency = (similarity >= similarity_threshold_exact).float()
    torch.diag(torch.diagonal(adjacency)).fill_(1.0)
    return adjacency, {"source": "exact"}
""",
        """\
    features = torch.as_tensor(features_exact)
    mask = features >= similarity_threshold_exact
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()
    return edge_index, {"source": "exact"}
""",
        """\
    features = torch.as_tensor(features_exact)
    mask = features >= similarity_threshold_exact
    source, target = torch.where(mask)
    edge_index = torch.stack([source, target], dim=0)
    return edge_index, {"source": "exact"}
""",
    ],
)
def test_archived_dense_and_receiver_sparse_array_provenance_is_supported(
    tmp_path, constructor_body
):
    imports = "from __future__ import annotations\n\nimport numpy as np\nimport torch"
    model_source = _MODEL_SOURCE.replace("import numpy as np", imports).replace(
        "def construct_graph_exact(*, features_exact, similarity_threshold_exact):\n"
        "    return np.array([[0, 1], [1, 0]]), {\"source\": \"exact\"}",
        "def construct_graph_exact(\n"
        "    *, features_exact: torch.Tensor, similarity_threshold_exact: float\n"
        "):\n"
        + constructor_body.rstrip(),
    )
    if constructor_body.lstrip().startswith("norms = np"):
        model_source = model_source.replace(
            "features_exact: torch.Tensor", "features_exact: np.ndarray"
        )
        cells = _NOTEBOOK_CELLS
        expected_kind = "numpy"
        expected_copy = "copy"
    else:
        cells = [_NOTEBOOK_CELLS[0].replace("edge_index.copy()", "edge_index.clone()")]
        expected_kind = "torch"
        expected_copy = "clone"
    run_dir = _write_package(tmp_path, model_source=model_source)

    proof = prove_constructor_notebook_flow(
        cells, _method_spec(), _arch_contract(), run_dir / "method"
    )

    assert proof["fitting_graph_array_kind"] == expected_kind
    assert proof["fitting_copy_method"] == expected_copy


@pytest.mark.parametrize(
    ("selected_expression", "copy_method"),
    [
        ("np.asarray(features_exact)", "copy"),
        ("np.array(features_exact, copy=True, like=features_exact)", "copy"),
        (
            "np.asarray(features_exact).astype("
            "np.float64, 'K', 'unsafe', True, False)",
            "copy",
        ),
        (
            "np.maximum(np.asarray(features_exact), 0, "
            "np.asarray(features_exact))",
            "copy",
        ),
        ("np.ma.array(features_exact)", "copy"),
        ("torch.as_tensor(features_exact)", "clone"),
        ("+torch.as_tensor(features_exact)", "clone"),
        (
            "torch.norm(torch.as_tensor(features_exact), 'fro', None, False, "
            "torch.as_tensor(features_exact))",
            "clone",
        ),
    ],
)
def test_selected_graph_must_not_alias_constructor_features(
    tmp_path, selected_expression, copy_method
):
    model_source = _MODEL_SOURCE.replace(
        "import numpy as np", "import numpy as np\nimport torch"
    ).replace("np.array([[0, 1], [1, 0]])", selected_expression)
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "edge_index.copy()", f"edge_index.{copy_method}()"
        )
    ]
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code in {
        "graph_constructor_output_not_fresh",
        "graph_constructor_output_type_unsupported",
    }


@pytest.mark.parametrize(
    ("selected_expression", "copy_method"),
    [
        ("np.asarray(features_exact).copy()", "copy"),
        ("torch.as_tensor(features_exact).clone()", "clone"),
    ],
)
def test_explicit_copy_or_clone_isolates_constructor_features(
    tmp_path, selected_expression, copy_method
):
    model_source = _MODEL_SOURCE.replace(
        "import numpy as np", "import numpy as np\nimport torch"
    ).replace("np.array([[0, 1], [1, 0]])", selected_expression)
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "edge_index.copy()", f"edge_index.{copy_method}()"
        )
    ]
    run_dir = _write_package(tmp_path, model_source=model_source)

    proof = prove_constructor_notebook_flow(
        cells, _method_spec(), _arch_contract(), run_dir / "method"
    )

    assert proof["fitting_graph_array_kind"] == (
        "numpy" if copy_method == "copy" else "torch"
    )


@pytest.mark.parametrize(
    "constructor_return",
    [
        "return adjacency, adjacency",
        'return adjacency, {"graph": adjacency}',
        'return adjacency.T, {"base": adjacency}',
    ],
)
def test_selected_graph_cannot_alias_unselected_constructor_output(
    tmp_path, constructor_return
):
    model_source = _MODEL_SOURCE.replace(
        'return np.array([[0, 1], [1, 0]]), {"source": "exact"}',
        "adjacency = np.array([[0, 1], [1, 0]])\n"
        f"    {constructor_return}",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_output_alias_unsupported"


def test_independent_metadata_copy_does_not_alias_selected_graph(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        'return np.array([[0, 1], [1, 0]]), {"source": "exact"}',
        "adjacency = np.array([[0, 1], [1, 0]])\n"
        '    return adjacency, {"graph": adjacency.copy()}',
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    proof = prove_constructor_notebook_flow(
        _NOTEBOOK_CELLS, _method_spec(), _arch_contract(), run_dir / "method"
    )

    assert proof["status"] == "pass"


def test_torch_unary_plus_cannot_hide_duplicate_selected_storage(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "import numpy as np", "import numpy as np\nimport torch"
    ).replace(
        'return np.array([[0, 1], [1, 0]]), {"source": "exact"}',
        "edge_index = torch.tensor([[0, 1], [1, 0]])\n"
        "    return edge_index, +edge_index",
    )
    cells = [_NOTEBOOK_CELLS[0].replace("edge_index.copy()", "edge_index.clone()")]
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_output_alias_unsupported"


@pytest.mark.parametrize(
    "escape",
    [
        "global hidden_graph\n    hidden_graph = adjacency",
        "holder.graph = adjacency",
        "cache[0] = adjacency",
        "features_exact.cache += [adjacency]",
        "features_exact[0] += adjacency",
        "stash(adjacency)",
    ],
)
def test_selected_graph_cannot_escape_constructor_storage(tmp_path, escape):
    model_source = _MODEL_SOURCE.replace(
        'return np.array([[0, 1], [1, 0]]), {"source": "exact"}',
        "adjacency = np.array([[0, 1], [1, 0]])\n"
        f"    {escape}\n"
        '    return adjacency, {"source": "exact"}',
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_output_escape_unsupported"


def test_unknown_array_combiner_cannot_impersonate_a_dense_constructor(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "def construct_graph_exact",
        "class StickyGraph:\n"
        "    def copy(self):\n"
        "        return self\n\n"
        "    def __radd__(self, other):\n"
        "        return self\n\n\n"
        "def construct_graph_exact",
    ).replace(
        "return np.array([[0, 1], [1, 0]]), {\"source\": \"exact\"}",
        "array = np.array([[0, 1], [1, 0]])\n"
        "    return array + StickyGraph(), {\"source\": \"exact\"}",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_output_type_unsupported"


@pytest.mark.parametrize(
    "selected_expression",
    [
        "np.array([StickyGraph()], dtype=object)[0]",
        "np.array([[0, 1], [1, 0]]).shape",
        "StickyGraph() or np.array([[0, 1], [1, 0]])",
    ],
)
def test_ambiguous_subscript_attribute_or_boolean_result_is_not_an_array_proof(
    tmp_path, selected_expression
):
    model_source = _MODEL_SOURCE.replace(
        "def construct_graph_exact",
        "class StickyGraph:\n"
        "    def copy(self):\n"
        "        return self\n\n\n"
        "def construct_graph_exact",
    ).replace(
        "np.array([[0, 1], [1, 0]])", selected_expression
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_output_type_unsupported"


@pytest.mark.parametrize(
    ("selected_expression", "copy_method"),
    [
        (
            "np.nonzero(features_exact >= similarity_threshold_exact)",
            "copy",
        ),
        (
            "np.where(features_exact >= similarity_threshold_exact)",
            "copy",
        ),
        (
            "torch.where(features_exact >= similarity_threshold_exact)",
            "clone",
        ),
        (
            "torch.nonzero("
            "features_exact >= similarity_threshold_exact, as_tuple=True)",
            "clone",
        ),
        (
            "torch.as_tensor(features_exact).max(dim=0)",
            "clone",
        ),
    ],
)
def test_tuple_returning_numeric_operations_are_not_array_proofs(
    tmp_path, selected_expression, copy_method
):
    model_source = _MODEL_SOURCE.replace(
        "import numpy as np", "import numpy as np\nimport torch"
    ).replace("np.array([[0, 1], [1, 0]])", selected_expression)
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "edge_index.copy()", f"edge_index.{copy_method}()"
        )
    ]
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_output_type_unsupported"


@pytest.mark.parametrize(
    "namespace_attack",
    [
        "alias = np\nalias.array = lambda value: StickyGraph()",
        "np.array = lambda value: StickyGraph()\nimport numpy as np",
        "import numpy as other\nother.array = lambda value: StickyGraph()",
        'np.__dict__["array"] = lambda value: StickyGraph()',
        'np.__dict__.update(array=lambda value: StickyGraph())',
        (
            "from numpy import __dict__ as numeric_namespace\n"
            'numeric_namespace["array"] = lambda value: StickyGraph()'
        ),
    ],
)
def test_numeric_namespace_aliases_and_reimports_cannot_impersonate_array_origin(
    tmp_path, namespace_attack
):
    model_source = _MODEL_SOURCE.replace(
        "def construct_graph_exact",
        "class StickyGraph:\n"
        "    def copy(self):\n"
        "        return self\n\n\n"
        + namespace_attack
        + "\n\n\ndef construct_graph_exact",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code in {
        "graph_constructor_output_type_unsupported",
        "graph_live_module_init_unsupported",
    }


def test_numeric_families_cannot_share_one_import_binding(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "import numpy as np",
        "import torch as xp\nimport numpy as xp",
    ).replace(
        "np.array([[0, 1], [1, 0]])", "xp.zeros((2, 2))"
    )
    cells = [_NOTEBOOK_CELLS[0].replace("edge_index.copy()", "edge_index.clone()")]
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_output_type_unsupported"


def test_control_binder_cannot_shadow_numeric_namespace(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "def construct_graph_exact",
        "class FakeNP:\n"
        "    def array(self, value):\n"
        "        return StickyGraph()\n\n\n"
        "class StickyGraph:\n"
        "    def copy(self):\n"
        "        return self\n\n\n"
        "def construct_graph_exact",
    ).replace(
        "return np.array([[0, 1], [1, 0]]), {\"source\": \"exact\"}",
        "for np in [FakeNP()]:\n"
        "        pass\n"
        "    return np.array([[0, 1], [1, 0]]), {\"source\": \"exact\"}",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_output_type_unsupported"


def test_constructor_control_flow_cannot_replace_the_proven_array(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "def construct_graph_exact",
        "class StickyGraph:\n"
        "    def copy(self):\n"
        "        return self\n\n\n"
        "def construct_graph_exact",
    ).replace(
        "return np.array([[0, 1], [1, 0]]), {\"source\": \"exact\"}",
        "edge_index = np.array([[0, 1], [1, 0]])\n"
        "    if features_exact is not None:\n"
        "        edge_index = StickyGraph()\n"
        "    return edge_index, {\"source\": \"exact\"}",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_output_type_unsupported"


@pytest.mark.parametrize(
    "mapping_return",
    [
        '{"graph": np.array([[0, 1], [1, 0]]), "graph": StickyGraph()}',
        '{"graph": np.array([[0, 1], [1, 0]]), **{"graph": StickyGraph()}}',
        '{"graph": np.array([[0, 1], [1, 0]]), key(): StickyGraph()}',
    ],
)
def test_constructor_mapping_selector_uses_exact_runtime_key_semantics(
    tmp_path, mapping_return
):
    model_source = _MODEL_SOURCE.replace(
        "def construct_graph_exact",
        "class StickyGraph:\n"
        "    def copy(self):\n"
        "        return self\n\n\n"
        "def key():\n"
        "    return \"graph\"\n\n\n"
        "def construct_graph_exact",
    ).replace(
        "return np.array([[0, 1], [1, 0]]), {\"source\": \"exact\"}",
        "return " + mapping_return,
    )
    spec = _method_spec()
    spec["methodology_replication_contract"]["homogeneous_graph_mechanism"][
        "construction"
    ]["output_selector"] = {"kind": "mapping_item", "key": "graph"}
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "edge_index, graph_metadata = construct_graph_exact(",
            "edge_index = construct_graph_exact(",
        ).replace(
            '    similarity_threshold_exact=cfg["similarity_threshold"],\n)',
            '    similarity_threshold_exact=cfg["similarity_threshold"],\n)["graph"]',
        )
    ]
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells, spec, _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_output_type_unsupported"


@pytest.mark.parametrize("nested_kind", ["def", "lambda"])
def test_fitting_nested_callable_cannot_mutate_model_dispatch(tmp_path, nested_kind):
    model_source = (
        "def wrong_forward(self, graph, signals):\n"
        "    return {\"forecast\": signals}\n\n\n"
        + _MODEL_SOURCE
    )
    if nested_kind == "def":
        nested = (
            "    def poison():\n"
            "        model.forward = wrong_forward\n"
            "    poison()\n"
        )
    else:
        nested = (
            "    poison = lambda: setattr(model, 'forward', wrong_forward)\n"
            "    poison()\n"
        )
    training_source = (
        "from .model import wrong_forward\n\n\n"
        "def train_model(*, model, source_graph, features):\n"
        + nested
        + "    return model\n"
    )
    run_dir = _write_package(
        tmp_path, model_source=model_source, training_source=training_source
    )

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_live_nested_callable_unsupported"


def test_module_init_local_call_cannot_mutate_numeric_namespace(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "def construct_graph_exact",
        "class StickyGraph:\n"
        "    def copy(self):\n"
        "        return self\n\n\n"
        "def poison():\n"
        "    np.array = lambda value: StickyGraph()\n\n\n"
        "poison()\n\n\n"
        "def construct_graph_exact",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code in {
        "graph_constructor_output_type_unsupported",
        "graph_live_nested_callable_unsupported",
    }


@pytest.mark.parametrize(
    "class_body",
    [
        "@poison\nclass Other:\n    pass",
        "class Other:\n    def method(self, value=poison()):\n        return value",
        "class Other:\n    descriptor = poison()",
    ],
)
def test_module_class_creation_cannot_hide_import_time_dispatch_mutation(
    tmp_path, class_body
):
    suffix = (
        "\n\ndef wrong_forward(self, graph, signals):\n"
        "    return {\"forecast\": signals}\n\n\n"
        "def poison():\n"
        "    TinyGraphModel.forward = wrong_forward\n"
        "    return object\n\n\n"
        + class_body
        + "\n"
    )
    run_dir = _write_package(tmp_path, model_source=_MODEL_SOURCE + suffix)

    with pytest.raises(GraphCallableCoverageError):
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )


@pytest.mark.parametrize("parameter_kind", ["vararg", "kwarg"])
def test_executable_variadic_annotations_are_outside_closed_headers(
    tmp_path, parameter_kind
):
    poison = (
        "def poison():\n"
        "    np.array = lambda value: value\n"
        "    return object\n\n\n"
    )
    if parameter_kind == "vararg":
        signature = (
            "def construct_graph_exact(\n"
            "    *args: poison(), features_exact, similarity_threshold_exact\n"
            "):"
        )
    else:
        signature = (
            "def construct_graph_exact(\n"
            "    *, features_exact, similarity_threshold_exact, **kwargs: poison()\n"
            "):"
        )
    model_source = poison + _MODEL_SOURCE.replace(
        "def construct_graph_exact(*, features_exact, similarity_threshold_exact):",
        signature,
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_live_callable_header_unsupported"


@pytest.mark.parametrize(
    "returned",
    [
        '{"forecast": forecast, "forecast": signals}',
        '{"forecast": forecast, **{"forecast": signals}}',
        '{"forecast": forecast, output_key(): signals}',
        '{"forecast": {"nested": forecast}}',
    ],
)
def test_message_output_mapping_cannot_shadow_or_contain_the_exact_leaf(
    tmp_path, returned
):
    model_source = _MODEL_SOURCE.replace(
        "def aggregate_neighbors_exact",
        "def output_key():\n"
        "    return \"forecast\"\n\n\n"
        "def aggregate_neighbors_exact",
    ).replace('return {"forecast": forecast}', f"return {returned}")
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises((GraphCallableProducerError, GraphCallableCoverageError)):
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )


def test_pluggable_output_mapping_cannot_shadow_the_exact_leaf(tmp_path):
    method_source = _METHOD_SOURCE.replace(
        "    return model.forward(graph=graph_payload, signals=signals)",
        "    output = model.forward(graph=graph_payload, signals=signals)\n"
        "    return {\"forecast\": output, \"forecast\": signals}",
    )
    run_dir = _write_package(tmp_path, method_source=method_source)

    with pytest.raises((GraphCallableProducerError, GraphCallableCoverageError)):
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS, _method_spec(), _arch_contract(), run_dir / "method"
        )


@pytest.mark.parametrize(
    "escape",
    [
        "(alias,) = (forecast,)\nalias.clear()",
        "box.value = forecast\nbox.value.clear()",
        "container = [forecast]\ncontainer[0].clear()",
    ],
)
def test_notebook_output_sink_cannot_escape_through_container_or_attribute(
    tmp_path, escape
):
    run_dir = _write_package(tmp_path)
    cells = [_NOTEBOOK_CELLS[0] + "\n" + escape + "\n"]

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == (
        "graph_constructor_inference_output_sink_escape_unsupported"
    )


@pytest.mark.parametrize(
    "hook_source",
    [
        "module_hooks.register_module_forward_hook(replace_output)",
        "hook = self.register_forward_hook\n        hook(replace_output)",
    ],
)
def test_forward_hook_registration_cannot_replace_analyzed_output(
    tmp_path, hook_source
):
    prefix = (
        "from torch.nn.modules import module as module_hooks\n\n\n"
        "def replace_output(*args):\n"
        "    return {\"forecast\": args[-1]}\n\n\n"
    )
    model_source = prefix + _MODEL_SOURCE.replace(
        "class TinyGraphModel:\n",
        "class TinyGraphModel:\n"
        "    def __init__(self):\n"
        "        "
        + hook_source
        + "\n\n",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_message_dynamic_dispatch_unsupported"


def test_forward_local_helper_cannot_install_an_output_hook(tmp_path):
    prefix = (
        "from torch.nn.modules import module as module_hooks\n\n\n"
        "def replace_output(*args):\n"
        "    return {\"forecast\": args[-1]}\n\n\n"
        "def install_hook():\n"
        "    module_hooks.register_module_forward_hook(replace_output)\n\n\n"
    )
    model_source = prefix + _MODEL_SOURCE.replace(
        "        forecast = aggregate_neighbors_exact(",
        "        install_hook()\n"
        "        forecast = aggregate_neighbors_exact(",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_message_dynamic_dispatch_unsupported"


def test_custom_setattr_cannot_replace_analyzed_forward_during_init(tmp_path):
    prefix = (
        "def wrong_forward(self, graph, signals):\n"
        "    return {\"forecast\": signals}\n\n\n"
    )
    model_source = prefix + _MODEL_SOURCE.replace(
        "class TinyGraphModel:\n",
        "class TinyGraphModel:\n"
        "    def __setattr__(self, name, value):\n"
        "        object.__setattr__(self, name, value)\n"
        "        if name == \"armed\":\n"
        "            object.__setattr__(self, \"forward\", wrong_forward.__get__(self))\n\n"
        "    def __init__(self):\n"
        "        self.armed = True\n\n",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_message_dynamic_dispatch_unsupported"


def test_inherited_setattr_cannot_replace_analyzed_forward_during_init(tmp_path):
    prefix = (
        "def wrong_forward(*, graph, signals):\n"
        "    return {\"forecast\": signals}\n\n\n"
    )
    model_source = prefix + _MODEL_SOURCE.replace(
        "class TinyGraphModel:\n",
        "class TinyGraphModel:\n"
        "    def __init__(self):\n"
        "        super().__setattr__(\"forward\", wrong_forward)\n\n",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_message_dynamic_dispatch_unsupported"


def test_benign_super_init_keeps_analyzed_forward_dispatch(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "class TinyGraphModel:\n",
        "class TinyGraphModel:\n"
        "    def __init__(self):\n"
        "        super().__init__()\n\n",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    proof = prove_message_live_path(
        _method_spec(), _arch_contract(), run_dir / "method"
    )

    assert proof["status"] == "pass"


def test_nested_notebook_method_sibling_import_is_rejected(tmp_path):
    run_dir = _write_package(tmp_path)
    (run_dir / "method" / "evil.py").write_text("VALUE = 1\n", encoding="utf-8")
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "params = {}",
            "if True:\n    import method.evil\nparams = {}",
        )
    ]

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_notebook_local_import_unsupported"


def test_run_local_absolute_import_cannot_hide_source_side_effects(tmp_path):
    run_dir = _write_package(
        tmp_path, model_source="import evil\n" + _MODEL_SOURCE
    )
    (run_dir / "evil.py").write_text(
        "from method.model import TinyGraphModel\n", encoding="utf-8"
    )

    with pytest.raises(GraphCallableCoverageError):
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS, _method_spec(), _arch_contract(), run_dir / "method"
        )


def test_namespace_package_import_is_producer_controlled(tmp_path):
    run_dir = _write_package(
        tmp_path, model_source="import evil.side\n" + _MODEL_SOURCE
    )
    namespace_dir = run_dir / "evil"
    namespace_dir.mkdir()
    (namespace_dir / "side.py").write_text(
        "from method.model import TinyGraphModel\n", encoding="utf-8"
    )

    with pytest.raises(GraphCallableCoverageError):
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS, _method_spec(), _arch_contract(), run_dir / "method"
        )
    with pytest.raises(GraphCallableCoverageError):
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )


def test_package_init_cannot_execute_a_core_module_mutator(tmp_path):
    init_source = _INIT_SOURCE + """\

from . import model as core


def poison_package():
    core.TinyGraphModel.forward = wrong_forward


poison_package()
"""
    run_dir = _write_package(tmp_path, init_source=init_source)

    for proof in (prove_constructor_notebook_flow, prove_message_live_path):
        with pytest.raises(GraphCallableCoverageError) as raised:
            if proof is prove_constructor_notebook_flow:
                proof(
                    _NOTEBOOK_CELLS,
                    _method_spec(),
                    _arch_contract(),
                    run_dir / "method",
                )
            else:
                proof(_method_spec(), _arch_contract(), run_dir / "method")
        assert raised.value.code == "graph_package_init_execution_unsupported"


def test_notebook_cannot_call_unproved_core_module_member(tmp_path):
    model_source = _MODEL_SOURCE + """\


def poison():
    TinyGraphModel.forward = wrong_forward
"""
    run_dir = _write_package(tmp_path, model_source=model_source)
    cells = [
        _NOTEBOOK_CELLS[0]
        .replace("from method import TinyGraphModel\n", "")
        .replace(
            "from method import construct_graph_exact, predict, train_model",
            "import method.model as mm\nfrom method import predict, train_model",
        )
        .replace("construct_graph_exact(", "mm.construct_graph_exact(")
        .replace("TinyGraphModel()", "mm.TinyGraphModel()")
        .replace("forecast = predict(", "mm.poison()\nforecast = predict(")
    ]

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_notebook_route_escape_unsupported"


@pytest.mark.parametrize(
    "training_source",
    [
        """\
def train_model(*, model, source_graph, features):
    import torch.nn as local_nn
    local_nn.Module.__call__ = wrong_call
    return model
""",
        """\
def train_model(*, model, source_graph, features):
    from torch.nn import modules as dispatched
    dispatched.module.Module.__call__ = wrong_call
    return model
""",
        """\
def train_model(*, model, source_graph, features):
    from torch.nn.modules import module as local_hooks
    local_hooks._global_forward_hooks["bad"] = replace
    return model
""",
        """\
from torch.nn.modules.module import _global_forward_hooks as hooks


def train_model(*, model, source_graph, features):
    hooks["bad"] = replace
    return model
""",
        """\
def train_model(*, model, source_graph, features):
    from torch.nn.modules.module import register_module_forward_hook as install
    install(replace)
    return model
""",
        """\
import torch.nn as framework


def mutate(namespace):
    namespace.Module.__call__ = wrong_call


def train_model(*, model, source_graph, features):
    mutate(framework)
    return model
""",
    ],
)
def test_fitting_cannot_mutate_framework_dispatch_through_import_aliases(
    tmp_path, training_source
):
    run_dir = _write_package(tmp_path, training_source=training_source)

    with pytest.raises(GraphCallableCoverageError):
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS, _method_spec(), _arch_contract(), run_dir / "method"
        )


def test_nested_notebook_framework_dispatch_mutation_is_rejected(tmp_path):
    run_dir = _write_package(tmp_path)
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "params = {}",
            "if True:\n"
            "    import torch.nn as local_nn\n"
            "    local_nn.Module.__call__ = wrong_call\n"
            "params = {}",
        )
    ]

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_dynamic_namespace_unsupported"


@pytest.mark.parametrize(
    "escape",
    [
        "from IPython import get_ipython as shell\n"
        "shell().run_line_magic('run', 'mutate.py')",
        "import IPython\n"
        "IPython.get_ipython().run_line_magic('run', 'mutate.py')",
    ],
)
def test_notebook_ipython_alias_escape_is_dynamic_namespace(tmp_path, escape):
    run_dir = _write_package(tmp_path)
    cells = [_NOTEBOOK_CELLS[0].replace("params = {}", escape + "\nparams = {}")]

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_dynamic_namespace_unsupported"


def test_notebook_runpy_cannot_execute_a_run_local_mutator(tmp_path):
    run_dir = _write_package(tmp_path)
    (run_dir / "evil.py").write_text(
        "from method.model import TinyGraphModel\n", encoding="utf-8"
    )
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "params = {}",
            "import runpy\nrunpy.run_path('evil.py')\nparams = {}",
        )
    ]

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_dynamic_namespace_unsupported"


def test_exact_torch_module_implicit_dispatch_remains_supported(tmp_path):
    model_source = "from torch import nn\n" + _MODEL_SOURCE.replace(
        "class TinyGraphModel:", "class TinyGraphModel(nn.Module):"
    )
    method_source = _METHOD_SOURCE.replace(
        "model.forward(graph=graph_payload, signals=signals)",
        "model(graph=graph_payload, signals=signals)",
    )
    run_dir = _write_package(
        tmp_path, model_source=model_source, method_source=method_source
    )

    proof = prove_constructor_notebook_flow(
        _NOTEBOOK_CELLS, _method_spec(), _arch_contract(), run_dir / "method"
    )

    assert proof["status"] == "pass"


def test_framework_subclass_introspection_cannot_replace_forward(tmp_path):
    model_source = "from torch import nn\n" + _MODEL_SOURCE.replace(
        "class TinyGraphModel:", "class TinyGraphModel(nn.Module):"
    )
    method_source = _METHOD_SOURCE.replace(
        "model.forward(graph=graph_payload, signals=signals)",
        "model(graph=graph_payload, signals=signals)",
    )
    training_source = """\
from torch.nn import Module


def train_model(*, model, source_graph, features):
    Module.__subclasses__()[-1].forward = wrong_forward
    return model
"""
    run_dir = _write_package(
        tmp_path,
        model_source=model_source,
        method_source=method_source,
        training_source=training_source,
    )

    with pytest.raises(GraphCallableCoverageError):
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS, _method_spec(), _arch_contract(), run_dir / "method"
        )


def test_unconditional_raise_cannot_precede_proved_message_path(tmp_path):
    model_source = _MODEL_SOURCE.replace(
        "        forecast = aggregate_neighbors_exact(",
        "        raise RuntimeError('unreachable message path')\n"
        "        forecast = aggregate_neighbors_exact(",
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    with pytest.raises(GraphCallableCoverageError) as constructor_error:
        prove_constructor_notebook_flow(
            _NOTEBOOK_CELLS, _method_spec(), _arch_contract(), run_dir / "method"
        )
    with pytest.raises(GraphCallableCoverageError) as message_error:
        prove_message_live_path(
            _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert constructor_error.value.code == "graph_live_statement_unsupported"
    assert message_error.value.code == "graph_live_statement_unsupported"


def test_ordinary_torch_functional_helper_use_remains_supported(tmp_path):
    model_source = (
        "import torch\nimport torch.nn.functional as functional\n"
        + _MODEL_SOURCE.replace(
            "    return [graph_exact, signals_exact]",
            "    graph_tensor = torch.as_tensor(graph_exact)\n"
            "    signal_tensor = functional.relu(torch.as_tensor(signals_exact))\n"
            "    return graph_tensor + signal_tensor",
        )
    )
    run_dir = _write_package(tmp_path, model_source=model_source)

    proof = prove_message_live_path(
        _method_spec(), _arch_contract(), run_dir / "method"
    )

    assert proof["status"] == "pass"


def test_declared_additional_fitting_input_remains_supported(tmp_path):
    contract = _arch_contract()
    contract["training_loop"]["input"]["epochs"] = {
        "kind": "scalar",
        "role": "epoch_count",
    }
    training_source = """\
def train_model(*, model, source_graph, features, epochs):
    return model
"""
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "source_graph=fitting_graph, features=features)",
            "source_graph=fitting_graph, features=features, epochs=one_epoch)",
        )
    ]
    run_dir = _write_package(tmp_path, training_source=training_source)

    proof = prove_constructor_notebook_flow(
        cells, _method_spec(), contract, run_dir / "method"
    )

    assert proof["status"] == "pass"


@pytest.mark.parametrize("escaped_role", ["model", "pluggable", "graph"])
def test_declared_fitting_input_cannot_receive_protected_extra_role(
    tmp_path, escaped_role
):
    contract = _arch_contract()
    extra_name = {
        "model": "other_model",
        "pluggable": "other_predict",
        "graph": "poison_graph",
    }[escaped_role]
    extra_value = {
        "model": "model",
        "pluggable": "predict",
        "graph": "edge_index",
    }[escaped_role]
    contract["training_loop"]["input"][extra_name] = {
        "kind": "opaque",
        "role": escaped_role,
    }
    training_source = (
        "def train_model(*, model, source_graph, features, "
        + extra_name
        + "):\n    return model\n"
    )
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "source_graph=fitting_graph, features=features)",
            "source_graph=fitting_graph, features=features, "
            f"{extra_name}={extra_value})",
        )
    ]
    run_dir = _write_package(tmp_path, training_source=training_source)

    with pytest.raises((GraphCallableProducerError, GraphCallableCoverageError)):
        prove_constructor_notebook_flow(
            cells, _method_spec(), contract, run_dir / "method"
        )


def test_starred_constructor_unpack_cannot_impersonate_selected_index(tmp_path):
    run_dir = _write_package(tmp_path)
    cells = [
        _NOTEBOOK_CELLS[0].replace(
            "edge_index, graph_metadata = construct_graph_exact(",
            "*graph_metadata, edge_index = construct_graph_exact(",
        )
    ]

    with pytest.raises(GraphCallableCoverageError) as raised:
        prove_constructor_notebook_flow(
            cells, _method_spec(), _arch_contract(), run_dir / "method"
        )

    assert raised.value.code == "graph_constructor_output_selector_unsupported"


def _write_receipt_authority(run_dir: Path) -> tuple[dict, dict]:
    pipeline_dir = run_dir / ".pipeline"
    spec = _method_spec()
    contract = _arch_contract()
    (pipeline_dir / "method_spec.json").write_text(
        json.dumps(spec), encoding="utf-8"
    )
    (pipeline_dir / "arch_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    (pipeline_dir / "params.json").write_text(
        json.dumps(
            {
                "schema_version": "1.3.0",
                "params": {
                    "similarity_threshold": {
                        "value": 0.5,
                        "source": "paper",
                        "paper_section": "Section 3",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (pipeline_dir / "notebook_draft.py").write_text(
        _NOTEBOOK_CELLS[0], encoding="utf-8"
    )
    (run_dir / "notebook.ipynb").write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "code", "source": _NOTEBOOK_CELLS[0]}
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "requirements.txt").write_text("", encoding="utf-8")
    return spec, contract


def _live_plan(run_dir: Path) -> dict:
    spec, contract = _write_receipt_authority(run_dir)
    _write_stage_receipts(run_dir)
    construction = prove_constructor_notebook_flow(
        _NOTEBOOK_CELLS, spec, contract, run_dir / "method"
    )
    message = prove_message_live_path(spec, contract, run_dir / "method")
    return {
        "status": "ready",
        "construction": {
            "callable": construction["callable"],
            "feature_input_root": "batch.features",
            "feature_parameter": "features_exact",
            "output_selector": {"kind": "tuple_item", "index": 0},
            "threshold": {
                "parameter": {
                    "params_name": "similarity_threshold",
                    "callable_parameter": "similarity_threshold_exact",
                }
            },
            "cap": {"kind": "none"},
        },
        "execution": {
            "callable": message["callable"],
            "graph_parameter": "graph_exact",
            "neighbor_signal_root": "batch.signals",
            "neighbor_signal_parameter": "signals_exact",
            "output_root": "outputs.forecast",
        },
        "callable_liveness_receipt": {
            "receipt_version": RECEIPT_VERSION,
            "status": "pass",
            "verified": True,
            "reason": "graph_callable_liveness_verified",
            "construction": construction,
            "message_passing": message,
            "authority_digests": {
                "stage_2d": stage_2d_authority_digests(run_dir),
                "stage_3c": stage_3c_authority_digests(run_dir),
            },
        },
    }


def _write_stage_receipts(run_dir: Path) -> None:
    pipeline_dir = run_dir / ".pipeline"
    (pipeline_dir / "stage_2d.complete").write_text("", encoding="utf-8")
    (pipeline_dir / "stage_2d.upstream_digest.json").write_text(
        json.dumps(stage_2d_authority_digests(run_dir), sort_keys=True),
        encoding="utf-8",
    )
    (pipeline_dir / "stage_3c.complete").write_text("", encoding="utf-8")
    (pipeline_dir / "stage_3c.upstream_digest.json").write_text(
        json.dumps(stage_3c_authority_digests(run_dir), sort_keys=True),
        encoding="utf-8",
    )


def _minimal_ready_plan() -> dict:
    return {
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
            "feature_input_root": "batch.features",
            "feature_parameter": "features_exact",
            "output_selector": {"kind": "tuple_item", "index": 0},
            "threshold": {
                "parameter": {
                    "params_name": "similarity_threshold",
                    "callable_parameter": "similarity_threshold_exact",
                }
            },
            "cap": {"kind": "none"},
        },
        "execution": {
            "callable": {
                "module": "method.model",
                "qualname": "aggregate_neighbors_exact",
            },
            "graph_parameter": "graph_exact",
            "neighbor_signal_root": "batch.signals",
            "neighbor_signal_parameter": "signals_exact",
            "output_root": "outputs.forecast",
        },
        "groundings": {
            "graph_mechanism.alignment_prerequisite": {
                "probe_ref": "graph_mechanism.alignment_prerequisite",
                "element_ids": ["graph-alignment"],
                "bound_callables": ["method.training:_prepare_graph_batch"],
            }
        },
        "fixture": {
            "scope": "graph-callable-liveness-fixture",
            "entity_ids": ["node-a", "node-b"],
        },
    }


def test_receipt_rechecks_exact_authority_bytes(tmp_path):
    run_dir = _write_package(tmp_path)
    plan = _live_plan(run_dir)

    verified, reason, evidence = assess_liveness_receipt(run_dir, plan)

    assert verified is True
    assert reason == "graph_callable_liveness_verified"
    assert evidence["construction_callable"] == plan["construction"]["callable"]
    assert evidence["message_callable"] == plan["execution"]["callable"]


def test_receipt_fails_closed_after_method_drift(tmp_path):
    run_dir = _write_package(tmp_path)
    plan = _live_plan(run_dir)
    with (run_dir / "method" / "model.py").open("a", encoding="utf-8") as stream:
        stream.write("\n# post-receipt drift\n")

    verified, reason, evidence = assess_liveness_receipt(run_dir, plan)

    assert verified is False
    assert reason == "graph_callable_liveness_receipt_stale"
    assert evidence["changed_authority_paths"] == [
        "stage_2d:method/model.py",
        "stage_3c:method/model.py",
    ]


def test_receipt_fails_closed_after_nested_method_dependency_drift(tmp_path):
    run_dir = _write_package(tmp_path)
    support_dir = run_dir / "method" / "support"
    support_dir.mkdir()
    implementation = support_dir / "impl.py"
    implementation.write_text("VALUE = 1\n", encoding="utf-8")
    plan = _live_plan(run_dir)
    implementation.write_text("VALUE = 2\n", encoding="utf-8")

    verified, reason, evidence = assess_liveness_receipt(run_dir, plan)

    assert verified is False
    assert reason == "graph_callable_liveness_receipt_stale"
    assert evidence["changed_authority_paths"] == [
        "stage_2d:method/support/impl.py",
        "stage_3c:method/support/impl.py",
    ]


def test_receipt_fails_closed_after_params_drift(tmp_path):
    run_dir = _write_package(tmp_path)
    plan = _live_plan(run_dir)
    (run_dir / ".pipeline" / "params.json").write_text(
        '{"changed": true}', encoding="utf-8"
    )

    verified, reason, evidence = assess_liveness_receipt(run_dir, plan)

    assert verified is False
    assert reason == "graph_callable_liveness_receipt_stale"
    assert evidence["changed_authority_paths"] == [
        "stage_3c:.pipeline/params.json"
    ]


def test_receipt_fails_closed_when_stage_evidence_is_removed(tmp_path):
    run_dir = _write_package(tmp_path)
    plan = _live_plan(run_dir)
    (run_dir / ".pipeline" / "stage_3c.complete").unlink()

    verified, reason, evidence = assess_liveness_receipt(run_dir, plan)

    assert verified is False
    assert reason == "graph_callable_liveness_stage_receipt_missing"
    assert evidence == {"missing_stages": ["stage_3c"]}


def test_receipt_identity_disagreement_fails_closed(tmp_path):
    run_dir = _write_package(tmp_path)
    plan = _live_plan(run_dir)
    mismatched = copy.deepcopy(plan)
    mismatched["execution"]["callable"] = {
        "module": "method.model",
        "qualname": "another_helper",
    }

    verified, reason, evidence = assess_liveness_receipt(run_dir, mismatched)

    assert verified is False
    assert reason == "graph_callable_liveness_identity_mismatch"
    assert evidence == {}


def test_shallow_pass_receipt_cannot_claim_a_proof(tmp_path):
    run_dir = _write_package(tmp_path)
    plan = _live_plan(run_dir)
    plan["callable_liveness_receipt"]["message_passing"] = {
        "status": "pass",
        "callable": plan["execution"]["callable"],
        "phases": ["fitting", "inference"],
        "helper_result_consumed": True,
    }

    verified, reason, evidence = assess_liveness_receipt(run_dir, plan)

    assert verified is False
    assert reason == "graph_callable_liveness_proof_shape_mismatch"
    assert evidence == {}


def test_receipt_cannot_widen_numeric_operand_dispatch_scope(tmp_path):
    run_dir = _write_package(tmp_path)
    plan = _live_plan(run_dir)
    plan["callable_liveness_receipt"]["construction"][
        "numeric_operand_dispatch_precondition"
    ] = "all_array_subclasses"

    verified, reason, evidence = assess_liveness_receipt(run_dir, plan)

    assert verified is False
    assert reason == "graph_callable_liveness_proof_shape_mismatch"
    assert evidence == {}


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("construction", "architecture_class"),
        ("construction", "pluggable_callable"),
        ("message_passing", "architecture_class"),
        ("message_passing", "forward_callable"),
        ("message_passing", "output_binding"),
    ],
)
def test_receipt_recomputes_every_stored_proof_field(
    tmp_path, section, field
):
    run_dir = _write_package(tmp_path)
    plan = _live_plan(run_dir)
    plan["callable_liveness_receipt"][section][field] = "forged"

    verified, reason, evidence = assess_liveness_receipt(run_dir, plan)

    assert verified is False
    assert reason == "graph_callable_liveness_receipt_proof_mismatch"
    assert evidence == {"changed_proof_fields": [f"{section}.{field}"]}


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("construction", "inference_graph_parameter"),
        ("message_passing", "graph_parameter"),
    ],
)
def test_receipt_rejects_forged_cross_bound_proof_fields_before_recomputation(
    tmp_path, section, field
):
    run_dir = _write_package(tmp_path)
    plan = _live_plan(run_dir)
    plan["callable_liveness_receipt"][section][field] = "forged"

    verified, reason, evidence = assess_liveness_receipt(run_dir, plan)

    assert verified is False
    assert reason == "graph_callable_liveness_proof_shape_mismatch"
    assert evidence == {}


def test_freezer_mints_live_receipt_only_from_current_stage_authority(
    tmp_path, monkeypatch
):
    from scripts import graph_mechanism_runtime_plan as planner
    from scripts.build_probe_harness import (
        _frozen_graph_mechanism_execution_plan,
    )

    run_dir = _write_package(tmp_path)
    _write_receipt_authority(run_dir)
    _write_stage_receipts(run_dir)
    monkeypatch.setattr(
        planner,
        "normalize_graph_mechanism_runtime_plan",
        lambda *_: _minimal_ready_plan(),
    )

    frozen = _frozen_graph_mechanism_execution_plan(
        run_dir,
        {"graph_mechanism.alignment_prerequisite": {}},
    )

    assert frozen is not None
    assert frozen["alignment_runtime_receipt"]["verified"] is True
    assert frozen["callable_liveness_receipt"]["status"] == "pass"
    assert frozen["callable_liveness_receipt"]["verified"] is True
    assert frozen["callable_liveness_receipt"]["authority_digests"] == {
        "stage_2d": stage_2d_authority_digests(run_dir),
        "stage_3c": stage_3c_authority_digests(run_dir),
    }


def test_freezer_leaves_graph_free_plan_shape_unchanged(tmp_path, monkeypatch):
    from scripts import graph_mechanism_runtime_plan as planner
    from scripts.build_probe_harness import (
        _frozen_graph_mechanism_execution_plan,
    )

    run_dir = _write_package(tmp_path)
    _write_receipt_authority(run_dir)
    graph_free = {
        "schema_version": "1.0",
        "status": "not_applicable",
        "reason": "graph_free_method",
        "groundings": {},
    }
    monkeypatch.setattr(
        planner,
        "normalize_graph_mechanism_runtime_plan",
        lambda *_: copy.deepcopy(graph_free),
    )

    frozen = _frozen_graph_mechanism_execution_plan(
        run_dir,
        {"graph_mechanism.alignment_prerequisite": {}},
    )

    assert frozen is not None
    assert "callable_liveness_receipt" not in frozen
    assert {
        key: value
        for key, value in frozen.items()
        if key != "alignment_runtime_receipt"
    } == graph_free


def test_runner_blocks_behavioral_rows_without_liveness_receipt(
    tmp_path, monkeypatch
):
    import probes.graph_mechanism
    import scripts.run_probes as runner
    from scripts.build_probe_harness import _stage_2d_alignment_receipt

    run_dir = _write_package(tmp_path)
    _write_receipt_authority(run_dir)
    _write_stage_receipts(run_dir)
    plan = _minimal_ready_plan()
    plan["alignment_runtime_receipt"] = _stage_2d_alignment_receipt(
        run_dir, plan
    )
    plan["alignment"]["status"] = "pass"
    observed = {}

    def fake_probes(execution_plan, adapters, fixture):
        observed["plan"] = execution_plan
        observed["adapters"] = adapters
        observed["fixture"] = fixture
        return []

    monkeypatch.setattr(
        probes.graph_mechanism, "run_graph_mechanism_probes", fake_probes
    )
    monkeypatch.setattr(
        runner,
        "_graph_callable_adapters",
        lambda _: (_ for _ in ()).throw(
            AssertionError("unverified liveness must block imports")
        ),
    )

    rows = runner._run_graph_mechanism_consumers(run_dir, plan)

    assert len(rows) == 1
    assert rows[0].probe_id == "HG-1"
    assert rows[0].verdict == "pass"
    assert observed["plan"]["status"] == "unprobeable"
    assert observed["plan"]["reason"] == (
        "graph_callable_liveness_receipt_missing"
    )
    assert observed["adapters"] == {}


def test_notebook_validator_halts_only_supported_constructor_disagreement(
    tmp_path
):
    from scripts.validate_notebook_output import (
        _graph_constructor_notebook_flow_errors,
    )

    run_dir = _write_package(tmp_path)
    spec, _ = _write_receipt_authority(run_dir)
    disagreeing = [
        _NOTEBOOK_CELLS[0].replace(
            "source_graph=fitting_graph", "source_graph=unrelated_graph"
        )
    ]
    unsupported = [
        _NOTEBOOK_CELLS[0].replace(
            "edge_index, graph_metadata = construct_graph_exact(features)",
            "if enabled:\n"
            "    edge_index, graph_metadata = construct_graph_exact(features)",
        )
    ]

    errors = _graph_constructor_notebook_flow_errors(
        disagreeing, spec, run_dir
    )

    assert len(errors) == 1
    assert "graph_constructor_fitting_path_missing" in errors[0]
    assert _graph_constructor_notebook_flow_errors(
        unsupported, spec, run_dir
    ) == []


def test_runtime_validator_halts_only_supported_message_disagreement(tmp_path):
    from scripts.validate_arch_contract_runtime import (
        _graph_message_live_path_errors,
    )

    discarded = _MODEL_SOURCE.replace(
        'return {"forecast": forecast}', 'return {"forecast": signals}'
    )
    run_dir = _write_package(tmp_path, model_source=discarded)
    spec, contract = _write_receipt_authority(run_dir)

    errors = _graph_message_live_path_errors(
        spec, contract, run_dir, typed_contract=False
    )

    assert len(errors) == 1
    assert "graph_message_result_discarded" in errors[0]
