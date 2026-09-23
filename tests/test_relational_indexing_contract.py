"""R2C-084 — relational index spaces survive homogeneous graph batching.

The shaping cases are the pdfgnn generations, but the fixtures below are a
generic five-node graph package: current-style uncoordinated permutation,
archived ``_10`` global endpoints against local rows, archived ``_11``-style
induced remapping, and archived ``_7``-style dense adjacency.  The asymmetric
graph and unique root values make orientation, identity, and degree mistakes
observable without constructing a model.
"""

from __future__ import annotations

import copy
import json
import textwrap

import pytest
from pydantic import ValidationError

from schemas.arch_contract import ArchContract, RelationalIndexing
from schemas.method_spec import MethodologyContractElement
from scripts.arch_contract_semantics import parse_semantic_issue
from scripts.validate_arch_contract import _check_relational_indexing_contract
from scripts.validate_arch_contract_runtime import validate as runtime_validate

pytestmark = pytest.mark.probe_runtime


def _element(**updates):
    raw = {
        "element_id": "graph-message-passing",
        "role": "core_methodology",
        "replication_status": "must_replicate",
        "paper_section": "Section 3",
        "paper_evidence": "The paper defines message passing on one item graph.",
        "technical_concept": "Homogeneous item graph",
        "required_behavior": "Use aligned item neighborhoods.",
        "demo_scale_implementation": "Use the same graph on the demo items.",
        "acceptable_approximations": [],
        "forbidden_substitutions": ["Do not discard the graph."],
        "required_controls": ["Keep item identity fixed."],
        "fairness_checks": ["Use the same item rows."],
        "feasibility_rationale": "The graph is available at demo scale.",
        "verification_expectations": ["Graph batches preserve item ids."],
        "blockers": [],
        "paper_element_ids": ["alg-graph-message-passing"],
        "relational_structure": {"kind": "homogeneous_graph"},
    }
    raw.update(updates)
    return raw


def _relational(
    *,
    representation: str = "sparse_edge_index",
    backend: str = "torch",
    degree_semantics: str = "source_graph",
    output_order: str = "prepared_entity_order",
):
    return {
        "schema_version": "1.0",
        "methodology_element_ids": ["graph-message-passing"],
        "entity_axis": "item",
        "stable_entity_id_root": "batch.entity_ids",
        "graph_root": "batch.edge_index" if representation == "sparse_edge_index" else "batch.adjacency",
        "representation": representation,
        "source_endpoint_index_space": "source_entity_axis_positions",
        "prepared_endpoint_index_space": "local_batch_positions",
        "edge_orientation": "source_to_destination",
        "coindexed_roots": {
            "batch.entity_ids": 0,
            "batch.demand": 0,
            "batch.targets": 0,
            "batch.static_features": 0,
            "batch.time_varying_features": 0,
            "batch.time_major_covariates": 1,
            "batch.in_degree": 0,
            "batch.mask": 0,
            "batch.embeddings": 0,
            "outputs.forecasts": 0,
            "outputs.actuals": 0,
            "outputs.baselines": 0,
            "metrics.per_entity_rows": 0,
        },
        "phase_batch_modes": {
            "fitting": "induced_subgraph",
            "model_selection": "canonical_full_graph",
            "inference": "canonical_full_graph",
            "reported_evaluation": "canonical_full_graph",
        },
        "degree_root": "batch.in_degree",
        "degree_kind": "in_degree",
        "degree_semantics": degree_semantics,
        "output_order": output_order,
        "preparation_callable": {
            "module": "method.training",
            "name": "_prepare_graph_batch",
            "tensor_backend": backend,
        },
    }


def _arch_contract(relational: dict | None):
    raw = {
        "schema_version": "1.0.0",
        "paradigm_id": "graph_node_classification_test",
        "data_loader": {"load_data_returns": {"x": "(N, 2)"}},
        "architecture": {
            "model": {
                "class_name": "TinyGraphModel",
                "forward": {
                    "input": {"x": "(B, 2)"},
                    "output_type": "tensor",
                    "output_shape": "(B, 2)",
                },
            }
        },
        "pluggable_component": {
            "name": "predict_nodes",
            "input_shapes": {"x": "(B, 2)"},
            "output_shape": "(B, 2)",
        },
        "training_loop": {
            "function_name": "train_model",
            "input_shapes": {"x": "(N, 2)"},
        },
    }
    if relational is not None:
        raw["relational_indexing"] = relational
    return raw


def _method_spec(marker: dict | None):
    element = _element()
    if marker is None:
        element.pop("relational_structure")
    else:
        element["relational_structure"] = marker
    return {"methodology_replication_contract": {"elements": [element]}}


def test_homogeneous_graph_marker_requires_paper_map_grounding():
    parsed = MethodologyContractElement.model_validate(_element())
    assert parsed.relational_structure.kind.value == "homogeneous_graph"

    bad = _element(paper_element_ids=[])
    with pytest.raises(ValidationError, match="graph activation must be grounded"):
        MethodologyContractElement.model_validate(bad)


def test_unsupported_relational_marker_keeps_concrete_paper_truth():
    element = _element(relational_structure={
        "kind": "unsupported", "unsupported_kind": "heterogeneous_graph"
    })
    parsed = MethodologyContractElement.model_validate(element)
    assert parsed.relational_structure.unsupported_kind == "heterogeneous_graph"

    element["relational_structure"] = {"kind": "unsupported"}
    with pytest.raises(ValidationError, match="requires a non-empty unsupported_kind"):
        MethodologyContractElement.model_validate(element)


def test_relational_schema_requires_explicit_degree_and_identity_roots():
    parsed = RelationalIndexing.model_validate(_relational())
    assert parsed.coindexed_roots[parsed.stable_entity_id_root] == 0
    assert parsed.coindexed_roots[parsed.degree_root] == 0

    bad = _relational()
    bad["degree_semantics"] = None
    with pytest.raises(ValidationError, match="must be supplied together"):
        RelationalIndexing.model_validate(bad)

    bad = _relational()
    del bad["coindexed_roots"]["batch.entity_ids"]
    with pytest.raises(ValidationError, match="stable_entity_id_root must appear"):
        RelationalIndexing.model_validate(bad)


@pytest.mark.parametrize(
    ("field_name", "blank_value"),
    [("entity_axis", "   "), ("graph_root", "\t")],
)
def test_relational_schema_rejects_whitespace_only_semantic_names(
    field_name, blank_value
):
    bad = _relational()
    bad[field_name] = blank_value
    with pytest.raises(
        ValidationError,
        match=rf"relational_indexing\.{field_name} cannot be blank or whitespace-only",
    ):
        RelationalIndexing.model_validate(bad)


def test_relational_schema_rejects_boolean_coindexed_axis():
    bad = _relational()
    bad["coindexed_roots"]["batch.demand"] = True
    with pytest.raises(ValidationError):
        RelationalIndexing.model_validate(bad)


def test_homogeneous_activation_requires_exact_conditional_arch_metadata():
    spec = _method_spec({"kind": "homogeneous_graph"})
    missing = ArchContract.model_validate(_arch_contract(None))
    errors = _check_relational_indexing_contract(spec, missing)
    assert len(errors) == 1
    assert "graph-message-passing" in errors[0]
    assert "roles, roots, endpoint spaces" in errors[0]

    block = _relational()
    block["methodology_element_ids"] = ["different-element"]
    mismatched = ArchContract.model_validate(_arch_contract(block))
    errors = _check_relational_indexing_contract(spec, mismatched)
    assert any("missing graph element ids" in error for error in errors)
    assert any("non-graph element ids" in error for error in errors)

    matched = ArchContract.model_validate(_arch_contract(_relational()))
    assert _check_relational_indexing_contract(spec, matched) == []


def test_graph_free_control_is_unchanged_and_cannot_invent_graph_truth():
    spec = _method_spec(None)
    graph_free = ArchContract.model_validate(_arch_contract(None))
    assert _check_relational_indexing_contract(spec, graph_free) == []

    invented = ArchContract.model_validate(_arch_contract(_relational()))
    errors = _check_relational_indexing_contract(spec, invented)
    assert len(errors) == 1
    assert "no methodology_replication_contract element" in errors[0]


def test_unsupported_form_is_a_pipeline_owned_terminal_issue_not_a_producer_retry():
    spec = _method_spec({
        "kind": "unsupported", "unsupported_kind": "dynamic_heterogeneous_graph"
    })
    graph_free_contract = ArchContract.model_validate(_arch_contract(None))
    errors = _check_relational_indexing_contract(spec, graph_free_contract)
    assert len(errors) == 1
    issue = parse_semantic_issue(errors[0])
    assert issue is not None
    assert issue.code == "unsupported_validator_feature"
    assert issue.owner == "pipeline"
    assert issue.roots == [
        "methodology_replication_contract.elements[0].relational_structure",
        "scripts/validate_arch_contract.py",
    ]
    assert issue.values == {
        "element_id": "graph-message-passing",
        "unsupported_kind": "dynamic_heterogeneous_graph",
        "supported_boundary": "homogeneous_graph",
    }


_MODEL = textwrap.dedent(
    """
    import torch

    class TinyGraphModel:
        def __call__(self, x):
            return x
    """
)


_TORCH_PREPARATION = textwrap.dedent(
    """
    import torch

    _BAD_MODE = "__BAD_MODE__"
    _DEGREE_SEMANTICS = "__DEGREE_SEMANTICS__"
    _DEGREE_KIND = "in_degree"
    _DEGREE_ROOT = "batch.in_degree"
    _OUTPUT_ORDER = "__OUTPUT_ORDER__"

    def _axis_for_source(value, source_count):
        axes = [index for index, size in enumerate(value.shape) if size == source_count]
        if len(axes) != 1:
            raise ValueError("fixture root has no unique source entity axis")
        return axes[0]

    def _prepare_graph_batch(*, source_entity_ids, batch_entity_ids, coindexed, graph, degrees):
        positions = []
        for entity_id in batch_entity_ids:
            matches = torch.nonzero(source_entity_ids == entity_id, as_tuple=False).flatten()
            if len(matches) != 1:
                raise ValueError("batch id must resolve exactly once")
            positions.append(int(matches.item()))
        if _OUTPUT_ORDER == "canonical_source_order":
            positions = sorted(positions)
        local_to_source = torch.tensor(positions, dtype=torch.long, device=source_entity_ids.device)
        source_to_local = torch.full(
            (len(source_entity_ids),), -1, dtype=torch.long, device=source_entity_ids.device
        )
        source_to_local[local_to_source] = torch.arange(
            len(positions), dtype=torch.long, device=source_entity_ids.device
        )

        prepared = {}
        for root, value in coindexed.items():
            axis = _axis_for_source(value, len(source_entity_ids))
            prepared[root] = torch.index_select(value, axis, local_to_source)

        sparse = graph.ndim == 2 and graph.shape[0] == 2 and graph.shape[1] != len(source_entity_ids)
        if sparse:
            src_local = source_to_local[graph[0]]
            dst_local = source_to_local[graph[1]]
            keep = (src_local >= 0) & (dst_local >= 0)
            prepared_graph = torch.stack((src_local[keep], dst_local[keep]))
        else:
            prepared_graph = torch.index_select(
                torch.index_select(graph, 0, local_to_source), 1, local_to_source
            )

        if _DEGREE_SEMANTICS == "source_graph":
            prepared_degrees = torch.index_select(degrees, 0, local_to_source)
        elif sparse:
            endpoint = prepared_graph[1] if _DEGREE_KIND == "in_degree" else prepared_graph[0]
            prepared_degrees = torch.bincount(
                endpoint, minlength=len(positions)
            ).to(dtype=torch.float32)
        else:
            axis = 0 if _DEGREE_KIND == "in_degree" else 1
            prepared_degrees = torch.count_nonzero(prepared_graph, dim=axis).to(dtype=torch.float32)
        prepared[_DEGREE_ROOT] = prepared_degrees

        is_permuted_full = len(positions) == len(source_entity_ids) and positions != list(range(len(source_entity_ids)))
        is_subset = len(positions) < len(source_entity_ids)
        if _BAD_MODE == "unreindexed_full" and is_permuted_full:
            prepared_graph = graph
            prepared_degrees = degrees
            prepared[_DEGREE_ROOT] = degrees
        if _BAD_MODE == "global_graph_on_subset" and is_subset:
            prepared_graph = graph
        if _BAD_MODE == "float_sparse_edges" and sparse:
            prepared_graph = prepared_graph.to(dtype=torch.float32)
        if _BAD_MODE == "reordered_sparse_columns" and sparse:
            prepared_graph = prepared_graph.flip(dims=(1,))
        if _BAD_MODE == "deduplicated_sparse_edges" and sparse:
            prepared_graph = torch.unique(prepared_graph, dim=1)
        if _BAD_MODE == "degree_left_unmapped" and is_subset:
            prepared_degrees = degrees[:len(positions)]
            prepared[_DEGREE_ROOT] = prepared_degrees
        if _BAD_MODE == "axis_one_left_unmapped" and is_subset:
            prepared["batch.time_major_covariates"] = coindexed["batch.time_major_covariates"]

        return {
            "local_to_source": local_to_source,
            "source_to_local": source_to_local,
            "coindexed": prepared,
            "graph": prepared_graph,
            "degrees": prepared_degrees,
            "output_entity_ids": torch.index_select(source_entity_ids, 0, local_to_source),
        }

    def train_model(*args, **kwargs):
        return _prepare_graph_batch(*args, **kwargs)
    """
)


_NUMPY_PREPARATION = textwrap.dedent(
    """
    import numpy as np

    _DEGREE_SEMANTICS = "__DEGREE_SEMANTICS__"
    _DEGREE_ROOT = "batch.in_degree"
    _OUTPUT_ORDER = "__OUTPUT_ORDER__"

    def _prepare_graph_batch(*, source_entity_ids, batch_entity_ids, coindexed, graph, degrees):
        positions = np.asarray([
            int(np.flatnonzero(source_entity_ids == entity_id)[0])
            for entity_id in batch_entity_ids
        ], dtype=np.int64)
        if _OUTPUT_ORDER == "canonical_source_order":
            positions = np.sort(positions)
        source_to_local = np.full((len(source_entity_ids),), -1, dtype=np.int64)
        source_to_local[positions] = np.arange(len(positions), dtype=np.int64)
        prepared = {}
        for root, value in coindexed.items():
            axes = [index for index, size in enumerate(value.shape) if size == len(source_entity_ids)]
            prepared[root] = np.take(value, positions, axis=axes[0])
        prepared_graph = graph[np.ix_(positions, positions)]
        if _DEGREE_SEMANTICS == "source_graph":
            prepared_degrees = degrees[positions]
        else:
            prepared_degrees = np.count_nonzero(prepared_graph, axis=0).astype(np.float32)
        prepared[_DEGREE_ROOT] = prepared_degrees
        return {
            "local_to_source": positions,
            "source_to_local": source_to_local,
            "coindexed": prepared,
            "graph": prepared_graph,
            "degrees": prepared_degrees,
            "output_entity_ids": source_entity_ids[positions],
        }

    def train_model(*args, **kwargs):
        return _prepare_graph_batch(*args, **kwargs)
    """
)


def _training_source(
    *, backend: str, degree_semantics: str, bad_mode: str = "",
    output_order: str = "prepared_entity_order",
):
    template = _TORCH_PREPARATION if backend == "torch" else _NUMPY_PREPARATION
    return (
        template.replace("__DEGREE_SEMANTICS__", degree_semantics)
        .replace("__BAD_MODE__", bad_mode)
        .replace("__OUTPUT_ORDER__", output_order)
    )


def _write_runtime_package(tmp_path, *, relational: dict | None, training_source: str):
    run_dir = tmp_path / "run"
    method = run_dir / "method"
    pipeline = run_dir / ".pipeline"
    method.mkdir(parents=True)
    pipeline.mkdir(parents=True)
    (method / "model.py").write_text(_MODEL, encoding="utf-8")
    (method / "training.py").write_text(training_source, encoding="utf-8")
    (method / "data.py").write_text("# graph fixture has no loader call\n", encoding="utf-8")
    (method / "__init__.py").write_text(
        "from .model import TinyGraphModel\n", encoding="utf-8"
    )
    (pipeline / "arch_contract.json").write_text(
        json.dumps(_arch_contract(relational)), encoding="utf-8"
    )
    return run_dir


def _relational_errors(errors):
    return [error for error in errors if "relational_indexing" in error]


@pytest.mark.parametrize("degree_semantics", ["source_graph", "induced_graph"])
def test_archived_11_sparse_induced_remapping_passes_all_static_cases(
    tmp_path, degree_semantics
):
    relational = _relational(degree_semantics=degree_semantics)
    run_dir = _write_runtime_package(
        tmp_path,
        relational=relational,
        training_source=_training_source(
            backend="torch", degree_semantics=degree_semantics
        ),
    )
    assert _relational_errors(runtime_validate({}, run_dir)) == []


def test_sparse_edge_column_reordering_preserves_the_directed_graph(tmp_path):
    run_dir = _write_runtime_package(
        tmp_path,
        relational=_relational(),
        training_source=_training_source(
            backend="torch",
            degree_semantics="source_graph",
            bad_mode="reordered_sparse_columns",
        ),
    )
    assert _relational_errors(runtime_validate({}, run_dir)) == []


def test_sparse_edge_multiplicity_is_not_laundered_by_set_equality(tmp_path):
    run_dir = _write_runtime_package(
        tmp_path,
        relational=_relational(),
        training_source=_training_source(
            backend="torch",
            degree_semantics="source_graph",
            bad_mode="deduplicated_sparse_edges",
        ),
    )
    errors = _relational_errors(runtime_validate({}, run_dir))
    assert any(
        "wrong directed edge multiset" in error
        and "multiplicity is preserved" in error
        for error in errors
    ), errors


def test_archived_7_dense_adjacency_and_source_degree_semantics_pass(tmp_path):
    relational = _relational(
        representation="dense_adjacency", backend="numpy",
        degree_semantics="source_graph",
    )
    run_dir = _write_runtime_package(
        tmp_path,
        relational=relational,
        training_source=_training_source(
            backend="numpy", degree_semantics="source_graph"
        ),
    )
    assert _relational_errors(runtime_validate({}, run_dir)) == []


def test_canonical_source_output_order_restores_permuted_and_subset_rows(tmp_path):
    relational = _relational(output_order="canonical_source_order")
    run_dir = _write_runtime_package(
        tmp_path,
        relational=relational,
        training_source=_training_source(
            backend="torch", degree_semantics="source_graph",
            output_order="canonical_source_order",
        ),
    )
    assert _relational_errors(runtime_validate({}, run_dir)) == []


def test_current_unreindexed_full_permutation_fails_orientation_and_identity(tmp_path):
    relational = _relational()
    run_dir = _write_runtime_package(
        tmp_path,
        relational=relational,
        training_source=_training_source(
            backend="torch", degree_semantics="source_graph",
            bad_mode="unreindexed_full",
        ),
    )
    errors = _relational_errors(runtime_validate({}, run_dir))
    assert any("coherent_permutation" in error and "source_to_destination" in error for error in errors)
    assert any("batch.in_degree" in error and "source_graph" in error for error in errors)


def test_archived_10_global_endpoints_cannot_reach_a_local_subset(tmp_path):
    relational = _relational()
    run_dir = _write_runtime_package(
        tmp_path,
        relational=relational,
        training_source=_training_source(
            backend="torch", degree_semantics="source_graph",
            bad_mode="global_graph_on_subset",
        ),
    )
    errors = _relational_errors(runtime_validate({}, run_dir))
    assert any("induced_subset_[4,2,0]" in error for error in errors)
    assert any("local_batch_positions [0, 3)" in error and "max=4" in error for error in errors)


def test_sparse_endpoint_dtype_and_degree_mapping_fail_closed(tmp_path):
    for bad_mode, expected in (
        ("float_sparse_edges", "must have integer dtype"),
        ("degree_left_unmapped", "degree root 'batch.in_degree'"),
    ):
        run_dir = _write_runtime_package(
            tmp_path / bad_mode,
            relational=_relational(),
            training_source=_training_source(
                backend="torch", degree_semantics="source_graph", bad_mode=bad_mode
            ),
        )
        errors = _relational_errors(runtime_validate({}, run_dir))
        assert any(expected in error for error in errors), errors


def test_axis_one_coindexed_root_cannot_be_left_in_source_order(tmp_path):
    run_dir = _write_runtime_package(
        tmp_path,
        relational=_relational(),
        training_source=_training_source(
            backend="torch", degree_semantics="source_graph",
            bad_mode="axis_one_left_unmapped",
        ),
    )
    errors = _relational_errors(runtime_validate({}, run_dir))
    assert any(
        "coindexed root 'batch.time_major_covariates'" in error
        and "induced_subset_[4,2,0]" in error
        for error in errors
    ), errors


def test_declared_callable_must_reach_fitting_and_no_fallback_identity_is_invented(tmp_path):
    dead = _training_source(backend="torch", degree_semantics="source_graph").replace(
        "return _prepare_graph_batch(*args, **kwargs)", "return None"
    )
    run_dir = _write_runtime_package(
        tmp_path / "dead", relational=_relational(), training_source=dead
    )
    errors = _relational_errors(runtime_validate({}, run_dir))
    assert any(
        "non-statically-dead top-level call graph" in error
        and "cannot certify fitting" in error
        for error in errors
    )

    missing_map = _training_source(
        backend="torch", degree_semantics="source_graph"
    ).replace(
        '"source_to_local": source_to_local,',
        '"not_the_mapping": source_to_local,',
    )
    run_dir = _write_runtime_package(
        tmp_path / "missing", relational=_relational(), training_source=missing_map
    )
    errors = _relational_errors(runtime_validate({}, run_dir))
    assert any("no fallback identity or row-order inference is allowed" in error for error in errors)


@pytest.mark.parametrize(
    "replacement",
    [
        textwrap.dedent(
            """
            def train_model(*args, **kwargs):
                if False:
                    return _prepare_graph_batch(*args, **kwargs)
                return None
            """
        ).strip(),
        textwrap.dedent(
            """
            def train_model(*args, **kwargs):
                def decoy():
                    return _prepare_graph_batch(*args, **kwargs)
                return None
            """
        ).strip(),
    ],
    ids=["literal_dead_branch", "uncalled_nested_function"],
)
def test_dead_or_nested_calls_cannot_launder_fitting_reachability(
    tmp_path, replacement
):
    good_entry = (
        "def train_model(*args, **kwargs):\n"
        "    return _prepare_graph_batch(*args, **kwargs)"
    )
    training = _training_source(
        backend="torch", degree_semantics="source_graph"
    ).replace(good_entry, replacement)
    assert good_entry not in training
    run_dir = _write_runtime_package(
        tmp_path, relational=_relational(), training_source=training
    )
    errors = _relational_errors(runtime_validate({}, run_dir))
    assert any(
        "non-statically-dead top-level call graph" in error
        and "cannot certify fitting" in error
        for error in errors
    ), errors


def test_runtime_graph_free_forecasting_control_does_not_require_a_callable(tmp_path):
    training = "def train_model(*args, **kwargs):\n    return None\n"
    run_dir = _write_runtime_package(
        tmp_path, relational=None, training_source=training
    )
    assert _relational_errors(runtime_validate({}, run_dir)) == []


def test_all_three_producers_receive_the_relational_contract():
    from scripts.dispatch_templates import STAGE_TASK_SUMMARIES

    analyzer = (
        open(".opencode/agents/r2c-method-analyzer.md", encoding="utf-8").read()
        + STAGE_TASK_SUMMARIES["stage_1_analyzer"]
    )
    architecture = (
        open(".opencode/agents/r2c-architecture-coder.md", encoding="utf-8").read()
        + STAGE_TASK_SUMMARIES["stage_2b_architecture"]
    )
    method = (
        open(".opencode/agents/r2c-method-coder.md", encoding="utf-8").read()
        + STAGE_TASK_SUMMARIES["stage_2c_method"]
    )

    assert "relational_structure" in analyzer
    assert "unsupported_kind" in analyzer
    for needle in (
        "relational_indexing",
        "source_to_destination",
        "local_to_source",
        "source_to_local",
        "output_entity_ids",
        "induced_subgraph",
        "relational_indexing.execution",
        "model_input_root",
    ):
        assert needle in architecture
    assert "relational_indexing" in method
    assert "do not independently" in method


# ---------------------------------------------------------------------------
# Schema-2 real fitting and inference calls
# ---------------------------------------------------------------------------


def _typed_dimension(value: int) -> dict:
    return {"expression": {"kind": "literal", "value": value}}


def _typed_array(
    backend: str,
    dtype: str,
    *dimensions: str,
    constraint: dict | None = None,
) -> dict:
    descriptor = {
        "kind": "tensor" if backend == "torch" else "ndarray",
        "dtype": dtype,
        "dimensions": [{"dimension": dimension} for dimension in dimensions],
    }
    if backend == "torch":
        descriptor["device"] = "cpu"
    if constraint is not None:
        descriptor["constraint"] = constraint
    return descriptor


def _typed_opaque(type_description: str) -> dict:
    return {
        "kind": "opaque",
        "type_description": type_description,
        "reason": (
            "The exact runtime object is supplied by the declared relational "
            "execution crosswalk."
        ),
    }


def _typed_relational_contract(
    *,
    representation: str,
    backend: str,
    degree_semantics: str,
    fitting_mode: str,
    output_order: str = "prepared_entity_order",
) -> dict:
    source_count = 5
    fitting_count = 3 if fitting_mode == "induced_subgraph" else source_count
    fitting_entity_dimension = (
        "fitting_entity_count"
        if fitting_mode == "induced_subgraph"
        else "source_entity_count"
    )
    source_ids = _typed_array(backend, "int64", "source_entity_count")
    batch_ids = _typed_array(backend, "int64", fitting_entity_dimension)
    source_features = _typed_array(
        backend, "float32", "source_entity_count", "feature_width"
    )
    source_degrees = _typed_array(backend, "float32", "source_entity_count")
    if representation == "sparse_edge_index":
        graph = _typed_array(
            backend,
            "int64",
            "endpoint_count",
            "edge_count",
            constraint={
                "kind": "index",
                "indexed_dimension": "source_entity_count",
            },
        )
    else:
        graph = _typed_array(
            backend,
            "float32",
            "source_entity_count",
            "source_entity_count",
        )

    relational = _relational(
        representation=representation,
        backend=backend,
        degree_semantics=degree_semantics,
        output_order=output_order,
    )
    relational["phase_batch_modes"]["fitting"] = fitting_mode
    relational["execution"] = {
        "entity_dimension": "source_entity_count",
        "fitting": {
            "model_input_root": "training_loop.input.model",
            "source_entity_ids_input_root": (
                "training_loop.input.source_entity_ids"
            ),
            "batch_entity_ids_input_root": (
                "training_loop.input.batch_entity_ids"
            ),
            "graph_input_root": "training_loop.input.graph",
            "degree_input_root": "training_loop.input.degrees",
            "coindexed_input_roots": {
                "batch.demand": "training_loop.input.features",
            },
            "fitting_entry_is_one_epoch": True,
        },
        "inference": {
            "architecture_block": "model",
            "graph_input_root": "architecture.model.forward.input.graph",
            "degree_input_root": "architecture.model.forward.input.degrees",
            "coindexed_input_roots": {
                "batch.demand": "architecture.model.forward.input.features",
            },
            "output_coindexed_root": "outputs.forecasts",
        },
    }

    return {
        "schema_version": "2.0.0",
        "paradigm_id": "typed_relational_test",
        "dimensions": {
            "source_entity_count": _typed_dimension(source_count),
            "fitting_entity_count": _typed_dimension(fitting_count),
            "endpoint_count": _typed_dimension(2),
            "edge_count": _typed_dimension(7),
            "feature_width": _typed_dimension(2),
        },
        "data_loader": {"load_data_returns": {}},
        "architecture": {
            "model": {
                "class_name": "TinyTypedGraphModel",
                "constructor_args": {"constructor_token": {"literal": 37}},
                "forward": {
                    "input": {
                        "features": copy.deepcopy(source_features),
                        "graph": copy.deepcopy(graph),
                        "degrees": copy.deepcopy(source_degrees),
                    },
                    "output": copy.deepcopy(source_features),
                },
                "additional_methods": {},
            }
        },
        "pluggable_component": {
            "name": "predict_nodes",
            "input": {"features": copy.deepcopy(source_features)},
            "output": copy.deepcopy(source_features),
        },
        "training_loop": {
            "function_name": "train_model",
            "input": {
                "model": _typed_opaque("constructed TinyTypedGraphModel"),
                "source_entity_ids": copy.deepcopy(source_ids),
                "batch_entity_ids": copy.deepcopy(batch_ids),
                "features": copy.deepcopy(source_features),
                "graph": copy.deepcopy(graph),
                "degrees": copy.deepcopy(source_degrees),
            },
            "output": None,
        },
        "optimizer_state": None,
        "family_components": {},
        "relational_indexing": relational,
    }


_TYPED_REAL_CALL_MODEL = textwrap.dedent(
    r"""
    from pathlib import Path

    class TinyTypedGraphModel:
        def __init__(self, constructor_token):
            if constructor_token != 37:
                raise AssertionError(
                    "constructor declaration was not consumed exactly"
                )
            self.constructor_token = constructor_token
            Path("constructor_called.marker").write_text(
                str(constructor_token), encoding="utf-8"
            )

        def forward(self, features, graph, degrees):
            if self.constructor_token != 37:
                raise AssertionError("wrong constructed model instance")
            with Path("inference_called.marker").open(
                "a", encoding="utf-8"
            ) as marker:
                marker.write("call\n")
            return features

        def __call__(self, features, graph, degrees):
            return self.forward(features, graph, degrees)
    """
)


_TYPED_REAL_CALL_ENTRY = textwrap.dedent(
    """
    from pathlib import Path
    from .model import TinyTypedGraphModel

    _FITTING_BAD_MODE = "__FITTING_BAD_MODE__"

    def build_model(constructor_token):
        return TinyTypedGraphModel(constructor_token)

    def predict_nodes(features):
        return features

    def train_model(
        model,
        source_entity_ids,
        batch_entity_ids,
        features,
        graph,
        degrees,
    ):
        Path("training_called.marker").write_text("called", encoding="utf-8")
        prepared = _prepare_graph_batch(
            source_entity_ids=source_entity_ids,
            batch_entity_ids=batch_entity_ids,
            coindexed={
                "batch.entity_ids": source_entity_ids,
                "batch.demand": features,
                "batch.in_degree": degrees,
            },
            graph=graph,
            degrees=degrees,
        )
        model_graph = prepared["graph"]
        model_degrees = prepared["degrees"]
        if _FITTING_BAD_MODE == "raw_global_graph_and_degrees":
            model_graph = graph
            model_degrees = degrees
        return model(
            features=prepared["coindexed"]["batch.demand"],
            graph=model_graph,
            degrees=model_degrees,
        )
    """
)


_TYPED_GRAPH_ONLY_REAL_CALL_MODEL = textwrap.dedent(
    r"""
    from pathlib import Path

    class TinyTypedGraphModel:
        def __init__(self, constructor_token):
            if constructor_token != 37:
                raise AssertionError(
                    "constructor declaration was not consumed exactly"
                )
            Path("constructor_called.marker").write_text(
                str(constructor_token), encoding="utf-8"
            )

        def forward(self, graph, degrees):
            node_count = int(degrees.shape[0])
            if graph.numel() and int(graph.max().item()) >= node_count:
                raise AssertionError("graph endpoint escaped the local domain")
            with Path("inference_called.marker").open(
                "a", encoding="utf-8"
            ) as marker:
                marker.write(str(node_count) + "\n")
            return degrees

        def __call__(self, graph, degrees):
            return self.forward(graph, degrees)
    """
)


_TYPED_GRAPH_ONLY_REAL_CALL_ENTRY = textwrap.dedent(
    """
    from pathlib import Path
    from .model import TinyTypedGraphModel

    def build_model(constructor_token):
        return TinyTypedGraphModel(constructor_token)

    def predict_nodes(degrees):
        return degrees

    def train_model(
        model,
        source_entity_ids,
        batch_entity_ids,
        graph,
        degrees,
    ):
        Path("training_called.marker").write_text(
            str(len(batch_entity_ids)), encoding="utf-8"
        )
        prepared = _prepare_graph_batch(
            source_entity_ids=source_entity_ids,
            batch_entity_ids=batch_entity_ids,
            coindexed={
                "batch.entity_ids": source_entity_ids,
                "batch.in_degree": degrees,
            },
            graph=graph,
            degrees=degrees,
        )
        return model(
            graph=prepared["graph"],
            degrees=prepared["degrees"],
        )
    """
)


def _typed_real_call_training_source(
    *, backend: str, degree_semantics: str, output_order: str,
    fitting_bad_mode: str = "",
    fitting_entry_source: str = _TYPED_REAL_CALL_ENTRY,
) -> str:
    source = _training_source(
        backend=backend,
        degree_semantics=degree_semantics,
        output_order=output_order,
    )
    legacy_entry = (
        "def train_model(*args, **kwargs):\n"
        "    return _prepare_graph_batch(*args, **kwargs)\n"
    )
    assert legacy_entry in source
    return source.replace(
        legacy_entry,
        fitting_entry_source.replace(
            "__FITTING_BAD_MODE__", fitting_bad_mode
        ),
    )


def _write_typed_relational_package(
    tmp_path,
    *,
    contract: dict,
    backend: str,
    degree_semantics: str,
    output_order: str = "prepared_entity_order",
    fitting_bad_mode: str = "",
    model_source: str = _TYPED_REAL_CALL_MODEL,
    fitting_entry_source: str = _TYPED_REAL_CALL_ENTRY,
):
    run_dir = tmp_path / "run"
    method = run_dir / "method"
    pipeline = run_dir / ".pipeline"
    method.mkdir(parents=True)
    pipeline.mkdir(parents=True)
    (method / "model.py").write_text(
        model_source, encoding="utf-8"
    )
    (method / "training.py").write_text(
        _typed_real_call_training_source(
            backend=backend,
            degree_semantics=degree_semantics,
            output_order=output_order,
            fitting_bad_mode=fitting_bad_mode,
            fitting_entry_source=fitting_entry_source,
        ),
        encoding="utf-8",
    )
    (method / "data.py").write_text("# no loader needed\n", encoding="utf-8")
    (method / "__init__.py").write_text(
        "from .model import TinyTypedGraphModel\n"
        "from .training import build_model, predict_nodes, train_model\n",
        encoding="utf-8",
    )
    (pipeline / "arch_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    return run_dir


def _parsed_typed_issues(errors: list[str]):
    issues = [parse_semantic_issue(error) for error in errors]
    assert issues and all(issue is not None for issue in issues), errors
    return issues


@pytest.mark.parametrize(
    (
        "representation",
        "backend",
        "degree_semantics",
        "output_order",
    ),
    [
        (
            "sparse_edge_index",
            "torch",
            "induced_graph",
            "prepared_entity_order",
        ),
        (
            "dense_adjacency",
            "numpy",
            "source_graph",
            "canonical_source_order",
        ),
    ],
    ids=["archived_11_sparse_induced", "archived_7_dense_source_degree"],
)
def test_v2_real_fitting_call_accepts_known_good_graph_conventions(
    tmp_path,
    representation,
    backend,
    degree_semantics,
    output_order,
):
    contract = _typed_relational_contract(
        representation=representation,
        backend=backend,
        degree_semantics=degree_semantics,
        fitting_mode="induced_subgraph",
        output_order=output_order,
    )
    run_dir = _write_typed_relational_package(
        tmp_path,
        contract=contract,
        backend=backend,
        degree_semantics=degree_semantics,
        output_order=output_order,
    )

    assert runtime_validate({}, run_dir) == []
    assert (run_dir / "constructor_called.marker").read_text(
        encoding="utf-8"
    ) == "37"
    assert (run_dir / "training_called.marker").read_text(
        encoding="utf-8"
    ) == "called"
    assert (run_dir / "inference_called.marker").read_text(
        encoding="utf-8"
    ).splitlines() == ["call", "call"]


def test_v2_graph_only_real_calls_use_output_identity_for_local_inference(
    tmp_path,
):
    contract = _typed_relational_contract(
        representation="sparse_edge_index",
        backend="torch",
        degree_semantics="induced_graph",
        fitting_mode="induced_subgraph",
    )
    contract["dimensions"]["inference_entity_count"] = _typed_dimension(2)
    contract["dimensions"]["inference_edge_count"] = _typed_dimension(1)
    local_graph = _typed_array(
        "torch",
        "int64",
        "endpoint_count",
        "inference_edge_count",
        constraint={
            "kind": "index",
            "indexed_dimension": "inference_entity_count",
        },
    )
    local_degrees = _typed_array(
        "torch", "float32", "inference_entity_count"
    )
    execution = contract["relational_indexing"]["execution"]
    execution["fitting"]["coindexed_input_roots"] = {}
    execution["inference"]["coindexed_input_roots"] = {}
    contract["relational_indexing"]["phase_batch_modes"]["inference"] = (
        "induced_subgraph"
    )
    contract["architecture"]["model"]["forward"] = {
        "input": {
            "graph": copy.deepcopy(local_graph),
            "degrees": copy.deepcopy(local_degrees),
        },
        "output": copy.deepcopy(local_degrees),
    }
    contract["pluggable_component"] = {
        "name": "predict_nodes",
        "input": {"degrees": copy.deepcopy(local_degrees)},
        "output": copy.deepcopy(local_degrees),
    }
    training_inputs = contract["training_loop"]["input"]
    contract["training_loop"]["input"] = {
        name: descriptor
        for name, descriptor in training_inputs.items()
        if name != "features"
    }

    assert execution["fitting"]["coindexed_input_roots"] == {}
    assert execution["inference"]["coindexed_input_roots"] == {}
    assert set(contract["training_loop"]["input"]) == {
        "model",
        "source_entity_ids",
        "batch_entity_ids",
        "graph",
        "degrees",
    }
    assert set(contract["architecture"]["model"]["forward"]["input"]) == {
        "graph",
        "degrees",
    }
    run_dir = _write_typed_relational_package(
        tmp_path,
        contract=contract,
        backend="torch",
        degree_semantics="induced_graph",
        model_source=_TYPED_GRAPH_ONLY_REAL_CALL_MODEL,
        fitting_entry_source=_TYPED_GRAPH_ONLY_REAL_CALL_ENTRY,
    )

    assert runtime_validate({}, run_dir) == []
    assert (run_dir / "constructor_called.marker").read_text(
        encoding="utf-8"
    ) == "37"
    assert (run_dir / "training_called.marker").read_text(
        encoding="utf-8"
    ) == "3"
    assert (run_dir / "inference_called.marker").read_text(
        encoding="utf-8"
    ).splitlines() == ["3", "2"]


def test_v2_real_fitting_call_catches_current_unreindexed_permutation(tmp_path):
    contract = _typed_relational_contract(
        representation="sparse_edge_index",
        backend="torch",
        degree_semantics="source_graph",
        fitting_mode="coherent_full_graph_permutation",
    )
    run_dir = _write_typed_relational_package(
        tmp_path,
        contract=contract,
        backend="torch",
        degree_semantics="source_graph",
        fitting_bad_mode="raw_global_graph_and_degrees",
    )

    issues = _parsed_typed_issues(runtime_validate({}, run_dir))
    mismatch = next(
        issue
        for issue in issues
        if "did not consume the prepared graph" in issue.message
    )
    assert mismatch.code == "contract_code_disagreement"
    assert mismatch.owner == "producer"
    assert mismatch.roots == ["relational_indexing", "method/training.py"]
    assert "source and local entity domains disagree" in mismatch.message
    assert (run_dir / "training_called.marker").is_file()


def test_v2_real_fitting_call_catches_archived_10_global_endpoints(tmp_path):
    contract = _typed_relational_contract(
        representation="sparse_edge_index",
        backend="torch",
        degree_semantics="source_graph",
        fitting_mode="induced_subgraph",
    )
    run_dir = _write_typed_relational_package(
        tmp_path,
        contract=contract,
        backend="torch",
        degree_semantics="source_graph",
        fitting_bad_mode="raw_global_graph_and_degrees",
    )

    issues = _parsed_typed_issues(runtime_validate({}, run_dir))
    escaped = next(
        issue
        for issue in issues
        if "endpoints outside local_batch_positions" in issue.message
    )
    assert escaped.code == "contract_code_disagreement"
    assert escaped.owner == "producer"
    assert escaped.roots == ["relational_indexing", "method/training.py"]
    assert "local_batch_positions [0, 3)" in escaped.message
    assert "max=4" in escaped.message
    assert (run_dir / "training_called.marker").is_file()


def test_v2_opaque_relational_execution_input_is_pipeline_owned(tmp_path):
    contract = _typed_relational_contract(
        representation="sparse_edge_index",
        backend="torch",
        degree_semantics="source_graph",
        fitting_mode="induced_subgraph",
    )
    contract["training_loop"]["input"]["graph"] = _typed_opaque(
        "structured graph batch"
    )
    run_dir = _write_typed_relational_package(
        tmp_path,
        contract=contract,
        backend="torch",
        degree_semantics="source_graph",
    )

    issues = _parsed_typed_issues(runtime_validate({}, run_dir))
    unsupported = next(
        issue
        for issue in issues
        if issue.code == "unsupported_validator_feature"
    )
    assert unsupported.owner == "pipeline"
    assert "cannot construct that relational grammar" in unsupported.message
    assert "opaque callable input" in unsupported.message
    assert "training_loop.input.graph" in unsupported.roots
    assert "scripts/validate_arch_contract_runtime.py" in unsupported.roots
    assert not (run_dir / "training_called.marker").exists()


def test_v2_real_fitting_call_overrides_declared_epoch_scalar_to_one(tmp_path):
    contract = _typed_relational_contract(
        representation="sparse_edge_index",
        backend="torch",
        degree_semantics="source_graph",
        fitting_mode="induced_subgraph",
    )
    fitting = contract["relational_indexing"]["execution"]["fitting"]
    del fitting["fitting_entry_is_one_epoch"]
    fitting["one_epoch_input_root"] = "training_loop.input.max_epochs"
    contract["training_loop"]["input"]["max_epochs"] = {
        "kind": "scalar",
        "dtype": "int64",
        "source": {"literal": 50},
    }
    run_dir = _write_typed_relational_package(
        tmp_path,
        contract=contract,
        backend="torch",
        degree_semantics="source_graph",
    )
    training_path = run_dir / "method" / "training.py"
    training_source = training_path.read_text(encoding="utf-8")
    fitting_signature = "    degrees,\n):\n"
    assert fitting_signature in training_source
    training_path.write_text(
        training_source.replace(
            fitting_signature,
            "    degrees,\n"
            "    max_epochs=50,\n"
            "):\n"
            "    Path(\"one_epoch_observed.marker\").write_text(\n"
            "        str(int(max_epochs)), encoding=\"utf-8\"\n"
            "    )\n"
            "    if int(max_epochs) != 1:\n"
            "        raise AssertionError(\"expected exact one epoch\")\n",
            1,
        ),
        encoding="utf-8",
    )

    assert runtime_validate({}, run_dir) == []
    assert (run_dir / "one_epoch_observed.marker").read_text(
        encoding="utf-8"
    ) == "1"
    assert (run_dir / "training_called.marker").read_text(
        encoding="utf-8"
    ) == "called"


def test_v2_blocked_supported_one_epoch_fitting_is_producer_owned(
    tmp_path,
    monkeypatch,
):
    contract = _typed_relational_contract(
        representation="sparse_edge_index",
        backend="torch",
        degree_semantics="source_graph",
        fitting_mode="induced_subgraph",
    )
    fitting = contract["relational_indexing"]["execution"]["fitting"]
    del fitting["fitting_entry_is_one_epoch"]
    fitting["one_epoch_input_root"] = "training_loop.input.max_epochs"
    contract["training_loop"]["input"]["max_epochs"] = {
        "kind": "scalar",
        "dtype": "int64",
        "source": {"literal": 50},
    }
    run_dir = _write_typed_relational_package(
        tmp_path,
        contract=contract,
        backend="torch",
        degree_semantics="source_graph",
    )
    training_path = run_dir / "method" / "training.py"
    training_source = training_path.read_text(encoding="utf-8")
    fitting_signature = "    degrees,\n):\n"
    training_marker = (
        '    Path("training_called.marker").write_text('
        '"called", encoding="utf-8")\n'
    )
    assert fitting_signature in training_source
    assert training_marker in training_source
    training_source = training_source.replace(
        fitting_signature,
        "    degrees,\n    max_epochs=50,\n):\n",
        1,
    ).replace(
        training_marker,
        training_marker
        + '    Path("timeout_epoch.marker").write_text(\n'
        + '        str(int(max_epochs)), encoding="utf-8"\n'
        + "    )\n"
        + "    if int(max_epochs) != 1:\n"
        + '        raise AssertionError("expected exact one epoch")\n'
        + '    __import__("time").sleep(30)\n'
        + '    Path("training_finished.marker").write_text(\n'
        + '        "finished", encoding="utf-8"\n'
        + "    )\n",
        1,
    )
    training_path.write_text(training_source, encoding="utf-8")
    monkeypatch.setenv("R2C_DRY_RUN_TIMEOUT_S", "5")

    issues = _parsed_typed_issues(runtime_validate({}, run_dir))
    assert all(issue.owner == "producer" for issue in issues)
    timed_out = next(
        (
            issue
            for issue in issues
            if "declared one-epoch fitting call did not complete"
            in issue.message
        ),
        None,
    )
    assert timed_out is not None, issues
    assert timed_out.code == "contract_code_disagreement"
    assert "bounded execution contract" in timed_out.message
    assert (run_dir / "timeout_epoch.marker").read_text(
        encoding="utf-8"
    ) == "1"
    assert (run_dir / "training_called.marker").read_text(
        encoding="utf-8"
    ) == "called"
    assert not (run_dir / "training_finished.marker").exists()


def test_v2_omitted_direct_fitting_input_is_producer_owned_before_call(
    tmp_path,
):
    contract = _typed_relational_contract(
        representation="sparse_edge_index",
        backend="torch",
        degree_semantics="source_graph",
        fitting_mode="induced_subgraph",
    )
    contract["training_loop"]["input"]["targets"] = _typed_array(
        "torch", "float32", "source_entity_count", "feature_width"
    )
    run_dir = _write_typed_relational_package(
        tmp_path,
        contract=contract,
        backend="torch",
        degree_semantics="source_graph",
    )

    issues = _parsed_typed_issues(runtime_validate({}, run_dir))
    assert all(issue.owner == "producer" for issue in issues)
    incomplete = next(
        issue
        for issue in issues
        if issue.code == "incomplete_generated_contract"
        and "training_loop.input.targets" in issue.message
    )
    assert "must close the typed fitting input surface" in incomplete.message
    assert incomplete.roots == [
        "relational_indexing.execution.fitting.coindexed_input_roots",
        "training_loop.input",
    ]
    assert not (run_dir / "training_called.marker").exists()


def test_v2_collapsed_opaque_fitting_roles_are_pipeline_owned_only(tmp_path):
    contract = _typed_relational_contract(
        representation="sparse_edge_index",
        backend="torch",
        degree_semantics="source_graph",
        fitting_mode="induced_subgraph",
    )
    fitting = contract["relational_indexing"]["execution"]["fitting"]
    structured_root = "training_loop.input.batch"
    fitting["source_entity_ids_input_root"] = structured_root
    fitting["batch_entity_ids_input_root"] = structured_root
    fitting["graph_input_root"] = structured_root
    fitting["degree_input_root"] = structured_root
    fitting["coindexed_input_roots"] = {"batch.demand": structured_root}
    contract["training_loop"]["input"] = {
        "model": _typed_opaque("constructed TinyTypedGraphModel"),
        "batch": _typed_opaque("structured relational training batch"),
    }
    run_dir = _write_typed_relational_package(
        tmp_path,
        contract=contract,
        backend="torch",
        degree_semantics="source_graph",
    )

    issues = _parsed_typed_issues(runtime_validate({}, run_dir))
    assert len(issues) == 1
    unsupported = issues[0]
    assert unsupported.code == "unsupported_validator_feature"
    assert unsupported.owner == "pipeline"
    assert "one opaque structured input collapses multiple relational roles" in (
        unsupported.message
    )
    assert structured_root in unsupported.roots
    assert "relational_indexing.execution" in unsupported.roots
    assert "scripts/validate_arch_contract_runtime.py" in unsupported.roots
    assert not (run_dir / "training_called.marker").exists()


def test_v2_multi_axis_relational_output_is_pipeline_owned_before_call(
    tmp_path,
):
    contract = _typed_relational_contract(
        representation="sparse_edge_index",
        backend="torch",
        degree_semantics="source_graph",
        fitting_mode="induced_subgraph",
    )
    contract["architecture"]["model"]["forward"]["output"] = _typed_array(
        "torch",
        "float32",
        "source_entity_count",
        "source_entity_count",
    )
    run_dir = _write_typed_relational_package(
        tmp_path,
        contract=contract,
        backend="torch",
        degree_semantics="source_graph",
    )

    issues = _parsed_typed_issues(runtime_validate({}, run_dir))
    assert len(issues) == 1
    unsupported = issues[0]
    assert unsupported.code == "unsupported_validator_feature"
    assert unsupported.owner == "pipeline"
    assert "entity dimension on multiple axes" in unsupported.message
    assert unsupported.roots == [
        "architecture.model.forward.output",
        "relational_indexing.execution.inference.output_coindexed_root",
        "scripts/validate_arch_contract_runtime.py",
    ]
    assert not (run_dir / "training_called.marker").exists()


def test_v2_equal_cardinality_dimensions_cannot_launder_inference_identity(
    tmp_path,
):
    contract = _typed_relational_contract(
        representation="sparse_edge_index",
        backend="torch",
        degree_semantics="source_graph",
        fitting_mode="induced_subgraph",
    )
    for dimension in (
        "output_entity_count",
        "graph_entity_count",
        "degree_entity_count",
    ):
        contract["dimensions"][dimension] = _typed_dimension(5)
    contract["architecture"]["model"]["forward"] = {
        "input": {
            "graph": _typed_array(
                "torch",
                "int64",
                "endpoint_count",
                "edge_count",
                constraint={
                    "kind": "index",
                    "indexed_dimension": "graph_entity_count",
                },
            ),
            "degrees": _typed_array(
                "torch", "float32", "degree_entity_count"
            ),
        },
        "output": _typed_array(
            "torch", "float32", "output_entity_count", "feature_width"
        ),
    }
    contract["relational_indexing"]["execution"]["inference"][
        "coindexed_input_roots"
    ] = {}
    run_dir = _write_typed_relational_package(
        tmp_path,
        contract=contract,
        backend="torch",
        degree_semantics="source_graph",
    )

    issues = _parsed_typed_issues(runtime_validate({}, run_dir))
    assert all(issue.owner == "producer" for issue in issues)
    assert all(issue.code == "incomplete_generated_contract" for issue in issues)
    graph_identity = next(
        issue
        for issue in issues
        if "must index the exact semantic entity dimension" in issue.message
    )
    assert "'output_entity_count'" in graph_identity.message
    assert "'graph_entity_count'" in graph_identity.message
    degree_identity = next(
        issue
        for issue in issues
        if "inference degree input must use exactly" in issue.message
    )
    assert "'output_entity_count'" in degree_identity.message
    assert "['degree_entity_count']" in degree_identity.message
    assert not (run_dir / "training_called.marker").exists()


_TYPED_GRAPH_FREE_MODEL = textwrap.dedent(
    r"""
    from pathlib import Path

    class TinyTypedModel:
        def __init__(self, constructor_token):
            if constructor_token != 37:
                raise AssertionError("wrong constructor value")

        def forward(self, features):
            return features

        def __call__(self, features):
            return self.forward(features)
    """
)


_TYPED_GRAPH_FREE_TRAINING = textwrap.dedent(
    """
    from pathlib import Path
    from .model import TinyTypedModel

    def build_model(constructor_token):
        return TinyTypedModel(constructor_token)

    def predict_nodes(features):
        return features

    def train_model(features):
        Path("training_called.marker").write_text("called", encoding="utf-8")
    """
)


def _typed_graph_free_contract() -> dict:
    features = _typed_array(
        "torch", "float32", "source_entity_count", "feature_width"
    )
    return {
        "schema_version": "2.0.0",
        "paradigm_id": "typed_graph_free_control",
        "dimensions": {
            "source_entity_count": _typed_dimension(5),
            "feature_width": _typed_dimension(2),
        },
        "data_loader": {"load_data_returns": {}},
        "architecture": {
            "model": {
                "class_name": "TinyTypedModel",
                "constructor_args": {"constructor_token": {"literal": 37}},
                "forward": {
                    "input": {"features": copy.deepcopy(features)},
                    "output": copy.deepcopy(features),
                },
                "additional_methods": {},
            }
        },
        "pluggable_component": {
            "name": "predict_nodes",
            "input": {"features": copy.deepcopy(features)},
            "output": copy.deepcopy(features),
        },
        "training_loop": {
            "function_name": "train_model",
            "input": {"features": copy.deepcopy(features)},
            "output": None,
        },
        "optimizer_state": None,
        "family_components": {},
    }


def test_v2_graph_free_typed_training_still_runs(tmp_path):
    run_dir = tmp_path / "run"
    method = run_dir / "method"
    pipeline = run_dir / ".pipeline"
    method.mkdir(parents=True)
    pipeline.mkdir(parents=True)
    (method / "model.py").write_text(
        _TYPED_GRAPH_FREE_MODEL, encoding="utf-8"
    )
    (method / "training.py").write_text(
        _TYPED_GRAPH_FREE_TRAINING, encoding="utf-8"
    )
    (method / "data.py").write_text("# no loader needed\n", encoding="utf-8")
    (method / "__init__.py").write_text(
        "from .model import TinyTypedModel\n"
        "from .training import build_model, predict_nodes, train_model\n",
        encoding="utf-8",
    )
    (pipeline / "arch_contract.json").write_text(
        json.dumps(_typed_graph_free_contract()), encoding="utf-8"
    )

    assert runtime_validate({}, run_dir) == []
    assert (run_dir / "training_called.marker").read_text(
        encoding="utf-8"
    ) == "called"


def test_block_bound_elements_activate_the_arch_contract_without_raw_markers():
    # R2C-092: the graph-mechanism block's alignment, construction, and
    # message-passing ids carry the homogeneous marker by derivation, so a
    # stored spec whose producer omitted the markers still requires (and
    # matches) the same relational_indexing contract.
    element = _element()
    element.pop("relational_structure")
    spec = {
        "methodology_replication_contract": {
            "elements": [element],
            "homogeneous_graph_mechanism": {
                "schema_version": "1.0",
                "alignment_element_id": "graph-message-passing",
                "construction": {"element_id": "graph-message-passing"},
                "message_passing": {"element_id": "graph-message-passing"},
            },
        },
    }

    missing = ArchContract.model_validate(_arch_contract(None))
    errors = _check_relational_indexing_contract(spec, missing)
    assert len(errors) == 1
    assert "graph-message-passing" in errors[0]

    matched = ArchContract.model_validate(_arch_contract(_relational()))
    assert _check_relational_indexing_contract(spec, matched) == []
