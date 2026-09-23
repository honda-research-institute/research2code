"""R2C-077 piece 3 reductions for source-ordered split lineage.

These tests pin analysis facts only.  They deliberately do not define an
enforcement verdict, a finding message, or validator integration.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import scripts.eval_split_lineage as lineage
from scripts.eval_split_lineage import (
    FITTING,
    INFERENCE_CONDITIONING,
    MODEL_SELECTION,
    REPORTED_EVALUATION,
    ProtocolRange,
    analyze_split_lineage,
    summarize_trainer,
)


_FIXTURES = (
    Path("tests/fixtures/evidence/pdfgnn-split-lineage-0807")
)
_CURRENT_CONSTANTS = {
    "demand_matrix.protocol_length": 1033,
    "model.context_length": 10,
    "model.forecast_horizon": 4,
}
_FLEET_SCALAR_FACTS = {
    "context_length": 10,
    "forecast_horizon": 4,
    "prediction_length": 4,
    "horizon": 4,
    "val_split_start": 96,
    "train_split_start": 80,
    "train_end": 80,
    "val_split": 0.1,
    "time_stride": 1,
    "stride": 1,
    "max_epochs": 2,
    "num_epochs": 2,
    "n_epochs": 2,
    "epochs": 2,
    "batch_size": 8,
    "patience": 2,
    "seed": 0,
    "num_steps": 2,
    "learning_rate": 0.001,
    "train_until_accuracy": 0.9,
    "max_neighbors": 4,
    "num_batches_per_epoch": 2,
}
_FLEET_TENSOR_TOKENS = {
    "demand", "target", "targets", "history", "series", "label", "labels",
    "data", "train", "training", "val", "validation", "test", "y",
}
_FLEET_MODEL_TOKENS = {"model", "net", "network", "estimator"}


def _source(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


def _tokens(name: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", name.lower()) if token}


def _archived_report(*, horizon: int):
    total_steps = 100
    split_start = total_steps - horizon
    return summarize_trainer(
        _source("archived_trainer.py"),
        constants={
            "demand_matrix.protocol_length": total_steps,
            "context_length": 10,
            "forecast_horizon": horizon,
            "val_split_start": split_start,
            "max_epochs": 2,
        },
    )


def _current_report(*, disjoint: bool = False):
    suffix = "_disjoint" if disjoint else ""
    return analyze_split_lineage(
        _source(f"current_notebook{suffix}.py"),
        {"train_model": _source(f"current_trainer{suffix}.py")},
        constants=_CURRENT_CONSTANTS,
        entry_function="run_notebook_fragment",
    )


def _observation_by_role(report):
    by_role = {observation.role: observation
               for observation in report.observations}
    assert len(by_role) == len(report.observations), report.observations
    return by_role


def _relation(report, first_role: str, second_role: str):
    roles = {first_role, second_role}
    matches = [
        relation
        for relation in report.relations
        if {relation.first_role, relation.second_role} == roles
    ]
    assert len(matches) == 1, (roles, report.relations)
    return matches[0]


def _semantic_observations(report):
    """Observation identity with incidental source spelling removed."""
    return {
        (
            observation.role,
            observation.read_kind,
            observation.root,
            observation.protocol_range,
            observation.subset,
            observation.model_id,
            observation.source,
        )
        for observation in report.observations
    }


def test_archived_randperm_alias_min_reduction_has_exact_k_minus_one_overlap():
    report = _archived_report(horizon=4)

    assert report.unresolved == ()
    observations = _observation_by_role(report)
    assert set(observations) == {FITTING, MODEL_SELECTION}
    assert observations[FITTING].protocol_range == ProtocolRange(10, 99)
    assert observations[MODEL_SELECTION].protocol_range == ProtocolRange(96, 100)
    assert {observation.root for observation in report.observations} == {
        "demand_matrix"
    }
    assert {observation.model_id for observation in report.observations} == {
        "model"
    }

    relation = _relation(report, FITTING, MODEL_SELECTION)
    assert relation.status == "overlap"
    assert relation.overlap == ProtocolRange(96, 99)
    assert relation.overlap.stop - relation.overlap.start == 4 - 1
    assert relation.certainty == "exact"


def test_archived_target_local_renames_do_not_change_semantic_lineage():
    original_source = _source("archived_trainer.py")
    renamed_source = (
        original_source
        .replace("best_val_loss", "selection_floor")
        .replace("val_demand_target", "second_values")
        .replace("demand_target", "first_values")
        .replace("val_loss", "checkpoint_score")
    )
    assert "demand_target" not in renamed_source
    assert "val_demand_target" not in renamed_source

    constants = {
        "demand_matrix.protocol_length": 100,
        "context_length": 10,
        "forecast_horizon": 4,
        "val_split_start": 96,
        "max_epochs": 2,
    }
    original = summarize_trainer(original_source, constants=constants)
    renamed = summarize_trainer(renamed_source, constants=constants)

    assert renamed.unresolved == original.unresolved == ()
    assert _semantic_observations(renamed) == _semantic_observations(original)
    assert renamed.relations == original.relations


def test_current_selection_role_survives_neutral_loss_and_target_renames():
    original_source = _source("current_trainer.py")
    renamed_source = (
        original_source
        .replace("batch_target", "first_values")
        .replace("val_target", "second_values")
        .replace("val_loss", "checkpoint_score")
        .replace("best_score", "selection_floor")
    )
    original = analyze_split_lineage(
        _source("current_notebook.py"),
        {"train_model": original_source},
        constants=_CURRENT_CONSTANTS,
        entry_function="run_notebook_fragment",
    )
    renamed = analyze_split_lineage(
        _source("current_notebook.py"),
        {"train_model": renamed_source},
        constants=_CURRENT_CONSTANTS,
        entry_function="run_notebook_fragment",
    )

    assert renamed.unresolved == original.unresolved == ()
    assert _semantic_observations(renamed) == _semantic_observations(original)
    assert renamed.relations == original.relations


def test_archived_horizon_one_control_is_disjoint():
    report = _archived_report(horizon=1)

    assert report.unresolved == ()
    observations = _observation_by_role(report)
    assert observations[FITTING].protocol_range == ProtocolRange(10, 99)
    assert observations[MODEL_SELECTION].protocol_range == ProtocolRange(99, 100)
    relation = _relation(report, FITTING, MODEL_SELECTION)
    assert relation.status == "disjoint"
    assert relation.overlap is None
    assert relation.certainty == "exact"


def test_current_call_site_preserves_four_exact_ranges_mask_and_model_identity():
    report = _current_report()

    assert report.unresolved == ()
    observations = _observation_by_role(report)
    assert set(observations) == {
        FITTING,
        MODEL_SELECTION,
        INFERENCE_CONDITIONING,
        REPORTED_EVALUATION,
    }
    assert observations[FITTING].protocol_range == ProtocolRange(10, 930)
    assert observations[MODEL_SELECTION].protocol_range == ProtocolRange(10, 1033)
    assert observations[INFERENCE_CONDITIONING].protocol_range == ProtocolRange(
        816, 826
    )
    reported = observations[REPORTED_EVALUATION]
    assert reported.protocol_range == ProtocolRange(826, 830)
    assert reported.subset == "boolean_mask"
    assert {observation.root for observation in report.observations} == {
        "demand_matrix"
    }
    assert {observation.model_id for observation in report.observations} == {
        "model"
    }

    assert len(report.calls) == 1
    call = report.calls[0]
    assert call.callee == "train_model"
    assert call.model_id == "model"
    assert [(binding.formal, binding.root, binding.protocol_range)
            for binding in call.bindings] == [
        ("demand_matrix", "demand_matrix", ProtocolRange(0, 1033))
    ]


def test_current_relations_record_overlap_and_disjointness_without_a_verdict():
    report = _current_report()

    exact = "exact"
    support = "support"
    expected = {
        frozenset((FITTING, INFERENCE_CONDITIONING)):
            ("overlap", ProtocolRange(816, 826), exact),
        frozenset((FITTING, MODEL_SELECTION)):
            ("overlap", ProtocolRange(10, 930), exact),
        frozenset((FITTING, REPORTED_EVALUATION)):
            ("overlap", ProtocolRange(826, 830), support),
        frozenset((INFERENCE_CONDITIONING, MODEL_SELECTION)):
            ("overlap", ProtocolRange(816, 826), exact),
        frozenset((INFERENCE_CONDITIONING, REPORTED_EVALUATION)):
            ("disjoint", None, support),
        frozenset((MODEL_SELECTION, REPORTED_EVALUATION)):
            ("overlap", ProtocolRange(826, 830), support),
    }
    actual = {
        frozenset((relation.first_role, relation.second_role)):
            (relation.status, relation.overlap, relation.certainty)
        for relation in report.relations
    }
    assert actual == expected


def test_current_disjoint_control_binds_the_sliced_formal_and_stays_disjoint():
    report = _current_report(disjoint=True)

    assert report.unresolved == ()
    observations = _observation_by_role(report)
    assert observations[FITTING].protocol_range == ProtocolRange(10, 744)
    assert observations[MODEL_SELECTION].protocol_range == ProtocolRange(744, 826)
    assert observations[INFERENCE_CONDITIONING].protocol_range == ProtocolRange(
        816, 826
    )
    assert observations[REPORTED_EVALUATION].protocol_range == ProtocolRange(
        826, 830
    )
    assert observations[REPORTED_EVALUATION].subset == "boolean_mask"

    assert len(report.calls) == 1
    binding = report.calls[0].bindings
    assert [(item.formal, item.root, item.protocol_range) for item in binding] == [
        ("demand_matrix", "demand_matrix", ProtocolRange(0, 826))
    ]

    for first, second in (
        (FITTING, MODEL_SELECTION),
        (FITTING, REPORTED_EVALUATION),
        (MODEL_SELECTION, REPORTED_EVALUATION),
        (INFERENCE_CONDITIONING, REPORTED_EVALUATION),
    ):
        relation = _relation(report, first, second)
        assert relation.status == "disjoint"
        assert relation.overlap is None
        assert relation.certainty == (
            "support" if REPORTED_EVALUATION in {first, second} else "exact"
        )


def test_relations_do_not_join_training_reads_to_a_different_scored_model():
    notebook = """
def run_notebook_fragment(
    model_a, model_b, demand_matrix, train_model, forecast
):
    total_steps = demand_matrix.shape[1]
    train_end = int(total_steps * 0.8)
    trained_model = train_model(
        model_a,
        demand_matrix=demand_matrix,
        val_split=0.1,
        time_stride=1,
    )
    history = demand_matrix[
        :, train_end - model_b.context_length:train_end
    ]
    actual = demand_matrix[
        :, train_end:train_end + model_b.forecast_horizon
    ]
    result = forecast(model_b, history=history)
    finite_mask = isfinite(actual)
    actual_finite = actual[finite_mask]
    mean_finite = result.mean[finite_mask]
    rmse = sqrt(mean((actual_finite - mean_finite) ** 2))
    return trained_model, rmse
"""
    report = analyze_split_lineage(
        notebook,
        {"train_model": _source("current_trainer.py")},
        constants={
            "demand_matrix.protocol_length": 1033,
            "model_a.context_length": 10,
            "model_a.forecast_horizon": 4,
            "model_b.context_length": 10,
            "model_b.forecast_horizon": 4,
        },
        entry_function="run_notebook_fragment",
    )

    assert report.unresolved == ()
    model_roles = {
        observation.model_id: set()
        for observation in report.observations
    }
    for observation in report.observations:
        model_roles[observation.model_id].add(observation.role)
    assert model_roles == {
        "model_a": {FITTING, MODEL_SELECTION},
        "model_b": {INFERENCE_CONDITIONING, REPORTED_EVALUATION},
    }
    assert {
        (relation.model_id, relation.first_role, relation.second_role)
        for relation in report.relations
    } == {
        ("model_a", FITTING, MODEL_SELECTION),
        ("model_b", INFERENCE_CONDITIONING, REPORTED_EVALUATION),
    }


def test_unknown_range_keeps_role_root_and_model_identity():
    source = """
def train_model(model, series):
    selected = series[:, :opaque_bound()]
    loss = model.loss(selected)
    loss.backward()
    return model
"""
    report = summarize_trainer(
        source,
        constants={"series.protocol_length": 100},
    )

    assert report.observations == ()
    assert report.relations == ()
    assert len(report.unresolved) == 1
    unresolved = report.unresolved[0]
    assert unresolved.role == FITTING
    assert unresolved.root == "series"
    assert unresolved.model_id == "model"
    assert "range is unresolved" in unresolved.reason


@pytest.mark.manual_only
def test_delivered_fleet_has_no_concrete_relation_noise_outside_pdfgnn():
    """Stress the report-only summary before choosing an enforcement scope.

    The test deliberately gives target-like formals a common fake 100-step
    envelope, even for non-temporal families.  That makes accidental joins
    easier to produce.  Unsupported rows stay visible in the failure payload;
    they are not treated as clean.

    Corpus breadth is the point: the live fleet (r2c_runs/, workstation
    only) plus the committed example_runs/ floor. An empty corpus skips
    instead of passing vacuously (W3 verdicts §5).
    """
    scanned: list[str] = []
    errors: list[str] = []
    relations: list[str] = []
    unresolved: list[str] = []

    fleet_files = [
        path
        for root in (Path("r2c_runs"), Path("example_runs"))
        for path in sorted(root.glob("*/method/*.py"))
    ]
    for path in fleet_files:
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError, ValueError):
            continue
        for function in tree.body:
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _tokens(function.name) & {"train", "fit", "retrain"}:
                continue

            args = (
                list(function.args.posonlyargs)
                + list(function.args.args)
                + list(function.args.kwonlyargs)
            )
            constants = dict(_FLEET_SCALAR_FACTS)
            for arg in args:
                tokens = _tokens(arg.arg)
                if tokens & _FLEET_TENSOR_TOKENS \
                        and not tokens & _FLEET_MODEL_TOKENS:
                    constants[f"{arg.arg}.protocol_length"] = 100
                if tokens & _FLEET_MODEL_TOKENS:
                    constants[f"{arg.arg}.context_length"] = 10
                    constants[f"{arg.arg}.forecast_horizon"] = 4

            label = f"{path.parent.parent.name}/{path.name}:{function.name}"
            try:
                report = summarize_trainer(
                    source,
                    function_name=function.name,
                    constants=constants,
                )
            except Exception as error:  # pragma: no cover - diagnostic guard
                errors.append(f"{label}: {error!r}")
                continue
            scanned.append(label)
            for relation in report.relations:
                canonical_slug = re.sub(r"_\d+$", "", path.parent.parent.name)
                relations.append(
                    f"{canonical_slug}|{label}|{relation.first_role}|"
                    f"{relation.second_role}|{relation.status}|{relation.overlap}"
                )
            for item in report.unresolved:
                unresolved.append(
                    f"{label}|{item.role}|{item.root}|{item.reason}"
                )

    if not scanned:
        pytest.skip("live fleet absent")
    assert errors == []
    outside_pdfgnn = [
        row for row in relations
        if not row.startswith(
            "probabilistic-demand-forecasting-with-graph-neural-networks|"
        )
    ]
    assert outside_pdfgnn == [], {
        "scanned": scanned,
        "relations": relations,
        "unresolved": unresolved,
    }

    current_label = (
        "probabilistic-demand-forecasting-with-graph-neural-networks/"
        "training.py:train_model"
    )
    archived_label = (
        "probabilistic-demand-forecasting-with-graph-neural-networks_11/"
        "training.py:train_model"
    )
    # Guard on the scanned file, not the run dir: a halted explanation-only
    # roll leaves the dir in place without method/training.py (pdfgnn
    # Attempt 1, 2026-08-10), and --fresh archives move deliveries to
    # suffixed dirs.
    if Path(
        "r2c_runs/probabilistic-demand-forecasting-with-graph-neural-networks"
        "/method/training.py"
    ).is_file():
        assert current_label in scanned
    if Path(
        "r2c_runs/probabilistic-demand-forecasting-with-graph-neural-networks"
        "_11/method/training.py"
    ).is_file():
        assert archived_label in scanned


def test_delivered_fleet_noise_floor_has_durable_reductions():
    source = _source("fleet_noise_trainers.py")
    progress = summarize_trainer(
        source,
        function_name="train_accuracy_progress",
        constants={"y_train.protocol_length": 100},
    )
    assert progress.unresolved == ()
    assert [(item.role, item.root) for item in progress.observations] == [
        (FITTING, "y_train")
    ]
    assert progress.relations == ()

    disjoint = summarize_trainer(
        source,
        function_name="train_with_disjoint_selection",
        constants={
            "y_train.protocol_length": 80,
            "y_validation.protocol_length": 20,
        },
    )
    assert disjoint.unresolved == ()
    assert {(item.role, item.root) for item in disjoint.observations} == {
        (FITTING, "y_train"),
        (MODEL_SELECTION, "y_validation"),
    }
    assert disjoint.relations == ()


def test_lineage_api_remains_report_only():
    report = _current_report()

    assert set(report.__dataclass_fields__) == {
        "observations", "relations", "calls", "unresolved"
    }
    assert not hasattr(report, "findings")
    assert not hasattr(lineage.RangeRelation, "message")
    assert not hasattr(lineage, "SplitFinding")
