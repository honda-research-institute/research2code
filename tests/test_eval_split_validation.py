"""R2C-077 enforcement over the source-ordered split-lineage report."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from schemas.arch_contract_v2 import ArchContractV2
from scripts.eval_split_lineage import (
    FITTING,
    INFERENCE_CONDITIONING,
    MODEL_SELECTION,
    REPORTED_EVALUATION,
    ProtocolRange,
    RoleObservation,
    SplitLineageReport,
    UnresolvedLineage,
    analyze_split_lineage,
    summarize_trainer,
)
from scripts.eval_split_validation import (
    _training_temporal_input_names,
    _trainer_names,
    architecture_split_lineage_errors,
    derive_split_validity_receipt,
    find_split_lineage_findings,
    method_split_lineage_errors,
    notebook_split_lineage_errors,
    notebook_split_validity_receipt,
    temporal_lineage_enabled,
)


_FIXTURES = Path("tests/fixtures/evidence/pdfgnn-split-lineage-0807")
_SPEC = {
    "comparison": {"classification": {"id": "time_series_forecasting"}}
}
_BUILD_PLAN = {
    "params_derivation": {
        "context_length": {"demo_value": 10},
        "forecast_horizon": {"demo_value": 4},
    }
}


def _source(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


def _v2_adapter_contract(input_name: str = "matrix") -> dict:
    opaque = {
        "kind": "opaque",
        "type_description": "fixture value",
        "reason": "This surface is outside the split-lineage adapter test.",
    }
    return {
        "schema_version": "2.0.0",
        "paradigm_id": "time_series_forecasting",
        "dimensions": {
            identity: {
                "expression": {"kind": "literal", "value": value}
            }
            for identity, value in (
                ("entity_count", 30),
                # The contract's nominal extent differs deliberately from
                # PROVENANCE live_steps=1033. R2C-077 uses semantic identity
                # to select the root, but realized live extent remains owned
                # by the existing provenance adapter.
                ("time_axis_steps", 1092),
                ("feature_width", 3),
            )
        },
        "data_loader": {"load_data_returns": {}},
        "architecture": {
            "model": {
                "class_name": "Net",
                "constructor_args": {},
                "forward": {"input": {}, "output": opaque},
            }
        },
        "pluggable_component": {
            "name": "forecast",
            "input": {},
            "output": opaque,
        },
        "training_loop": {
            "function_name": "train_model",
            "input": {
                input_name: {
                    "kind": "ndarray",
                    "dtype": "float32",
                    "dimensions": [
                        {"dimension": "entity_count"},
                        {"dimension": "time_axis_steps"},
                    ],
                }
            },
        },
    }


def _archived_report(source: str | None = None, *, horizon: int = 4):
    return summarize_trainer(
        source or _source("archived_trainer.py"),
        constants={
            "demand_matrix.protocol_length": 100,
            "context_length": 10,
            "forecast_horizon": horizon,
            "val_split_start": 100 - horizon,
            "max_epochs": 2,
        },
    )


def _current_report(*, disjoint: bool = False):
    suffix = "_disjoint" if disjoint else ""
    return analyze_split_lineage(
        _source(f"current_notebook{suffix}.py"),
        {"train_model": _source(f"current_trainer{suffix}.py")},
        constants={
            "demand_matrix.protocol_length": 1033,
            "model.context_length": 10,
            "model.forecast_horizon": 4,
        },
        entry_function="run_notebook_fragment",
    )


def _forecast_report(helper: str, arguments: str = "") -> SplitLineageReport:
    return analyze_split_lineage(
        f"""
series = opaque_loader()
model = build_model()
fitting = series[:, :90]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(model, history{arguments})
render(result)
""",
        {"forecast": helper},
        constants={"series.protocol_length": 100},
    )


def _seed_adapter_run(
    run_dir: Path,
    training_source: str,
    *,
    arch_contract: dict | None = None,
) -> None:
    method = run_dir / "method"
    pipeline = run_dir / ".pipeline"
    data = method / "example_data"
    data.mkdir(parents=True)
    pipeline.mkdir(parents=True)
    (method / "training.py").write_text(training_source, encoding="utf-8")
    (method / "method.py").write_text("", encoding="utf-8")
    contract = arch_contract or {
        "training_loop": {
            "function_name": "train_model",
            "input_shapes": {
                "demand_matrix": "(N, T)",
                "time_varying_matrix": "(N, L, T)",
            },
        }
    }
    (pipeline / "arch_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    (pipeline / "params.json").write_text(json.dumps({
        "schema_version": "1.1.0",
        "params": {
            "context_length": {"value": 10},
            "forecast_horizon": {"value": 4},
        },
    }), encoding="utf-8")
    (data / "PROVENANCE.json").write_text(json.dumps({
        "files": [{
            "file": "sales.csv",
            "time_axis": {"steps_kept": 1092, "live_steps": 1033},
        }]
    }), encoding="utf-8")


def test_archived_exact_overlap_verdict_survives_neutral_renames():
    source = _source("archived_trainer.py")
    renamed = (
        source.replace("best_val_loss", "selection_floor")
        .replace("val_demand_target", "second_values")
        .replace("demand_target", "first_values")
        .replace("val_loss", "checkpoint_score")
    )

    original_findings = find_split_lineage_findings(_archived_report(source))
    renamed_findings = find_split_lineage_findings(_archived_report(renamed))

    assert len(original_findings) == len(renamed_findings) == 1
    finding = original_findings[0]
    assert finding.kind == "exact_overlap"
    assert finding.certainty == "exact"
    assert finding.overlap == ProtocolRange(96, 99)
    assert {finding.first.role, finding.second.role} == {
        FITTING, MODEL_SELECTION,
    }
    assert renamed_findings[0] == finding
    message = finding.message("method/training.py")
    assert "fitting target range [10,99)" in message
    assert "model-selection target range [96,100)" in message
    assert "overlap [96,99) exactly" in message
    assert "root `demand_matrix`" in message
    assert "scored model `model`" in message
    assert "Required protocol-axis overlap for these roles: empty" in message
    assert "split-boundary arithmetic" in message
    assert "temporal_target_role_overlap" in message
    assert "demand_target" not in message
    assert "val_demand_target" not in message


def test_horizon_one_and_separately_disjoint_controls_pass():
    assert find_split_lineage_findings(_archived_report(horizon=1)) == ()
    assert find_split_lineage_findings(_current_report(disjoint=True)) == ()


def test_current_verdict_has_three_target_findings_and_keeps_support_honest():
    findings = find_split_lineage_findings(_current_report())

    assert len(findings) == 3
    by_roles = {
        frozenset((item.first.role, item.second.role)): item
        for item in findings
    }
    assert set(by_roles) == {
        frozenset((FITTING, MODEL_SELECTION)),
        frozenset((FITTING, REPORTED_EVALUATION)),
        frozenset((MODEL_SELECTION, REPORTED_EVALUATION)),
    }
    assert by_roles[frozenset((FITTING, MODEL_SELECTION))].kind \
        == "exact_overlap"
    for roles in (
        frozenset((FITTING, REPORTED_EVALUATION)),
        frozenset((MODEL_SELECTION, REPORTED_EVALUATION)),
    ):
        finding = by_roles[roles]
        assert finding.kind == "support_overlap"
        assert finding.certainty == "support"
        assert finding.overlap == ProtocolRange(826, 830)
        message = finding.message("notebook.ipynb")
        assert "masked support envelope" in message
        assert "does not claim every envelope element" in message

    # Conditioning history deliberately remains a separate read kind.  The
    # existing notebook-range arm owns conditioning-versus-evaluation.
    assert all(INFERENCE_CONDITIONING not in roles for roles in by_roles)


def test_split_validity_receipt_rejects_current_and_accepts_disjoint_control(
    tmp_path,
):
    bad_run = tmp_path / "bad"
    _seed_adapter_run(bad_run, _source("current_trainer.py"))
    bad_receipt = notebook_split_validity_receipt(
        _SPEC,
        _BUILD_PLAN,
        bad_run,
        ast.parse(_source("current_notebook_generated.py")),
    )

    assert bad_receipt["schema_version"] == "1.0.0"
    assert bad_receipt["validator"] == "eval_split_lineage"
    assert bad_receipt["status"] == "invalid"
    assert bad_receipt["reasons"] == [
        "exact_temporal_target_overlap",
        "support_temporal_target_overlap",
    ]
    assert {
        item["kind"] for item in bad_receipt["evidence"]["findings"]
    } == {"exact_overlap", "support_overlap"}

    good_run = tmp_path / "good"
    _seed_adapter_run(
        good_run, _source("current_trainer_disjoint.py")
    )
    good_receipt = notebook_split_validity_receipt(
        _SPEC,
        _BUILD_PLAN,
        good_run,
        ast.parse(_source("current_notebook_generated_disjoint.py")),
    )

    assert good_receipt["status"] == "valid"
    assert good_receipt["reasons"] == [
        "fitting_to_reported_evaluation_disjointness_proved"
    ]
    relations = good_receipt["evidence"][
        "fitting_to_reported_evaluation_relations"
    ]
    assert relations
    assert {item["status"] for item in relations} == {"disjoint"}
    assert {item["certainty"] for item in relations} == {"support"}
    assert good_receipt["evidence"]["fitting_target_ranges"] == [{
        "root": "demand_matrix",
        "model_id": "model",
        "protocol_range": {"start": 10, "stop": 744},
        "subset": None,
        "certainty": "exact",
    }]
    json.dumps(good_receipt)


def test_split_receipt_projects_fitting_ranges_without_union_or_inference():
    observations = (
        RoleObservation(
            role=FITTING,
            read_kind="target",
            root="series",
            protocol_range=ProtocolRange(0, 8),
            subset=None,
            model_id="model_a",
            line=7,
            source="train_model_a",
            expression="series[:, :8]",
        ),
        RoleObservation(
            role=FITTING,
            read_kind="target",
            root="series",
            protocol_range=ProtocolRange(6, 12),
            subset="boolean_mask",
            model_id="model_b",
            line=19,
            source="train_model_b",
            expression="series[:, mask]",
        ),
        RoleObservation(
            role=MODEL_SELECTION,
            read_kind="target",
            root="series",
            protocol_range=ProtocolRange(12, 16),
            subset=None,
            model_id="model_b",
            line=23,
            source="train_model_b",
            expression="series[:, 12:16]",
        ),
        RoleObservation(
            role=MODEL_SELECTION,
            read_kind="target",
            root="series",
            protocol_range=ProtocolRange(14, 18),
            subset="boolean_mask",
            model_id="model_c",
            line=31,
            source="train_model_c",
            expression="series[:, selection_mask]",
        ),
    )

    receipt = derive_split_validity_receipt(SplitLineageReport(
        observations=observations,
        relations=(),
        calls=(),
        unresolved=(),
    ))

    assert receipt["evidence"]["fitting_target_ranges"] == [
        {
            "root": "series",
            "model_id": "model_a",
            "protocol_range": {"start": 0, "stop": 8},
            "subset": None,
            "certainty": "exact",
        },
        {
            "root": "series",
            "model_id": "model_b",
            "protocol_range": {"start": 6, "stop": 12},
            "subset": "boolean_mask",
            "certainty": "support",
        },
    ]
    assert receipt["evidence"]["selection_target_ranges"] == [
        {
            "root": "series",
            "model_id": "model_b",
            "protocol_range": {"start": 12, "stop": 16},
            "subset": None,
            "certainty": "exact",
        },
        {
            "root": "series",
            "model_id": "model_c",
            "protocol_range": {"start": 14, "stop": 18},
            "subset": "boolean_mask",
            "certainty": "support",
        },
    ]


def test_split_validity_receipt_rejects_archived_trainer_leak(tmp_path):
    run_dir = tmp_path / "archived"
    _seed_adapter_run(run_dir, _source("archived_trainer.py"))

    receipt = notebook_split_validity_receipt(
        _SPEC,
        _BUILD_PLAN,
        run_dir,
        ast.parse(_source("archived_notebook_generated.py")),
    )

    assert receipt["status"] == "invalid"
    assert "exact_temporal_target_overlap" in receipt["reasons"]
    assert any(
        item["overlap"] == {"start": 1029, "stop": 1032}
        for item in receipt["evidence"]["findings"]
    )


def test_split_validity_receipt_fails_closed_on_unresolved_or_absent_proof():
    fitting = RoleObservation(
        role=FITTING,
        read_kind="target",
        root="series",
        protocol_range=ProtocolRange(0, 80),
        subset=None,
        model_id="model",
        line=3,
        source="train_model",
        expression="series[:, :80]",
    )
    unresolved_evaluation = UnresolvedLineage(
        source="notebook",
        line=12,
        expression="series[:, opaque_start():]",
        reason="slice lower bound is unresolved",
        role=REPORTED_EVALUATION,
        read_kind="target",
        root="series",
        model_id="model",
    )
    unresolved_report = SplitLineageReport(
        observations=(fitting,),
        relations=(),
        calls=(),
        unresolved=(unresolved_evaluation,),
    )

    unresolved_receipt = derive_split_validity_receipt(unresolved_report)
    assert unresolved_receipt["status"] == "unresolved"
    assert unresolved_receipt["reasons"] == [
        "relevant_temporal_target_lineage_unresolved",
        "fitting_to_reported_evaluation_disjointness_unproved",
    ]
    assert unresolved_receipt["evidence"]["relevant_unresolved"][0][
        "reason"
    ] == "slice lower bound is unresolved"

    absent_proof = derive_split_validity_receipt(SplitLineageReport(
        observations=(fitting,),
        relations=(),
        calls=(),
        unresolved=(),
    ))
    assert absent_proof["status"] == "unresolved"
    assert absent_proof["reasons"] == [
        "fitting_to_reported_evaluation_disjointness_unproved"
    ]


def test_range_relations_and_unresolved_rows_carry_read_kind():
    report = _current_report()
    relation = next(
        item for item in report.relations
        if {item.first_role, item.second_role} == {
            INFERENCE_CONDITIONING, REPORTED_EVALUATION,
        }
    )
    assert {relation.first_read_kind, relation.second_read_kind} == {
        "conditioning", "target",
    }

    unresolved = summarize_trainer(
        """
def train_model(model, series):
    selected = series[:, :opaque_bound()]
    loss = model.loss(selected)
    loss.backward()
    return model
""",
        constants={"series.protocol_length": 100},
    )
    assert unresolved.unresolved[0].read_kind == "target"
    # A lone unresolved target has no forbidden relationship to evaluate.
    assert find_split_lineage_findings(unresolved) == ()

    fitting = unresolved.unresolved[0]
    selection = RoleObservation(
        role=MODEL_SELECTION,
        read_kind="target",
        root="series",
        protocol_range=ProtocolRange(80, 100),
        subset=None,
        model_id="model",
        line=9,
        source="train_model",
        expression="series[:, 80:100]",
    )
    paired = SplitLineageReport(
        observations=(selection,),
        relations=(),
        calls=(),
        unresolved=(fitting,),
    )
    findings = find_split_lineage_findings(paired)
    assert len(findings) == 1
    assert findings[0].kind == "unresolved"
    message = findings[0].message("method/training.py")
    assert "fitting target range unresolved" in message
    assert "model-selection target range [80,100)" in message
    assert "temporal_target_range_unresolved" in message

    for changed in (
        RoleObservation(
            **{**selection.__dict__, "root": "other_series"}
        ),
        RoleObservation(
            **{**selection.__dict__, "model_id": "other_model"}
        ),
    ):
        control = SplitLineageReport(
            observations=(changed,), relations=(), calls=(),
            unresolved=(fitting,),
        )
        assert find_split_lineage_findings(control) == ()


def test_roleless_and_conditioning_unresolved_rows_do_not_fail_target_policy():
    report = SplitLineageReport(
        observations=(),
        relations=(),
        calls=(),
        unresolved=(
            UnresolvedLineage(
                source="notebook", line=3, expression="opaque()",
                reason="generic parser gap",
            ),
            UnresolvedLineage(
                source="notebook", line=7, expression="history",
                reason="conditioning range unresolved",
                role=INFERENCE_CONDITIONING,
                read_kind="conditioning",
                root="series",
                model_id="model",
            ),
        ),
    )
    assert find_split_lineage_findings(report) == ()


def test_unresolved_scored_model_fails_only_with_same_root_target_counterpart():
    unresolved_eval = UnresolvedLineage(
        source="notebook",
        line=12,
        expression="score",
        reason="prediction call unsupported",
        role=REPORTED_EVALUATION,
        read_kind="target",
        root="series",
        protocol_range=ProtocolRange(90, 100),
    )
    fitting = RoleObservation(
        role=FITTING,
        read_kind="target",
        root="series",
        protocol_range=ProtocolRange(0, 95),
        subset=None,
        model_id="model_a",
        line=4,
        source="train_model",
        expression="series[:, :95]",
    )
    report = SplitLineageReport(
        observations=(fitting,), relations=(), calls=(),
        unresolved=(unresolved_eval,),
    )

    finding = find_split_lineage_findings(report)[0]
    assert finding.kind == "unresolved"
    assert finding.model_id is None
    message = finding.message("notebook.ipynb")
    assert "unresolved scored-model binding" in message
    assert "reported-evaluation target range [90,100)" in message

    disjoint = SplitLineageReport(
        observations=(RoleObservation(
            **{
                **fitting.__dict__,
                "protocol_range": ProtocolRange(0, 80),
            }
        ),),
        relations=(),
        calls=(),
        unresolved=(unresolved_eval,),
    )
    assert find_split_lineage_findings(disjoint) == ()


def test_builder_bindings_preserve_scored_model_isolation():
    notebook = """
model_a = build_model(context_length=10, forecast_horizon=4)
model_b = build_model(context_length=10, forecast_horizon=4)
trained_model = train_model(
    model_a,
    demand_matrix=demand_matrix,
    val_split=0.1,
    time_stride=1,
)
train_end = 826
history = demand_matrix[:, train_end - 10:train_end]
actual = demand_matrix[:, train_end:train_end + 4]
prediction = forecast(model_b, history=history)
score = sqrt(mean((actual - prediction.mean) ** 2))
"""
    report = analyze_split_lineage(
        notebook,
        {"train_model": _source("current_trainer.py")},
        constants={
            "demand_matrix.protocol_length": 1033,
            "model.context_length": 10,
            "model.forecast_horizon": 4,
        },
    )

    roles_by_model = {
        model_id: {item.role for item in report.observations
                   if item.model_id == model_id}
        for model_id in {item.model_id for item in report.observations}
    }
    assert roles_by_model == {
        "model_a": {FITTING, MODEL_SELECTION},
        "model_b": {INFERENCE_CONDITIONING, REPORTED_EVALUATION},
    }
    findings = find_split_lineage_findings(report)
    assert len(findings) == 1
    assert {findings[0].first.role, findings[0].second.role} == {
        FITTING, MODEL_SELECTION,
    }


def test_one_builder_with_two_aliases_keeps_one_scored_model_identity():
    notebook = """
model_a = model_b = build_model(context_length=10, forecast_horizon=4)
trained_model = train_model(
    model_a,
    demand_matrix=demand_matrix,
    val_split=0.1,
    time_stride=1,
)
actual = demand_matrix[:, 826:830]
prediction = forecast(model_b, history=demand_matrix[:, 816:826])
score = sqrt(mean((actual - prediction.mean) ** 2))
"""
    report = analyze_split_lineage(
        notebook,
        {"train_model": _source("current_trainer.py")},
        constants={
            "demand_matrix.protocol_length": 1033,
            "model.context_length": 10,
            "model.forecast_horizon": 4,
        },
    )

    assert {item.model_id for item in report.observations} == {"model_a"}
    assert len(find_split_lineage_findings(report)) == 3


def test_method_style_forecast_keeps_receiver_model_identity():
    source = """
candidate = build_model(context_length=10, forecast_horizon=4)
trained_model = train_model(
    candidate,
    demand_matrix=demand_matrix,
    val_split=0.1,
    time_stride=1,
)
actual = demand_matrix[:, 826:830]
prediction = trained_model.predict(demand_matrix[:, 816:826])
score = sqrt(mean((actual - prediction.mean) ** 2))
"""
    report = analyze_split_lineage(
        source,
        {"train_model": _source("current_trainer.py")},
        constants={
            "demand_matrix.protocol_length": 1033,
            "model.context_length": 10,
            "model.forecast_horizon": 4,
        },
    )

    assert {item.model_id for item in report.observations} == {"candidate"}
    assert len(find_split_lineage_findings(report)) == 3


def test_architecture_adapter_catches_resolved_trainer_local_overlap(tmp_path):
    run_dir = tmp_path / "run"
    _seed_adapter_run(run_dir, _source("current_trainer.py"))

    errors = architecture_split_lineage_errors(
        _SPEC, _BUILD_PLAN, run_dir
    )

    assert len(errors) == 1
    assert "fitting target range [10,930)" in errors[0]
    assert "model-selection target range [10,1033)" in errors[0]
    assert "temporal_target_role_overlap" in errors[0]


def test_every_declared_tsf_family_identity_activates_lineage_enforcement(
    tmp_path,
):
    """Canonical, legacy alias, and taxonomy-id forms share one policy."""
    run_dir = tmp_path / "run"
    _seed_adapter_run(run_dir, _source("current_trainer.py"))

    for paradigm_id in (
        "time_series_forecasting",
        "graph_time_series_forecasting",
        "TE-TSF/time_series_forecasting",
    ):
        spec = {"comparison": {"classification": {"id": paradigm_id}}}
        assert temporal_lineage_enabled(spec), paradigm_id
        errors = architecture_split_lineage_errors(
            spec, _BUILD_PLAN, run_dir
        )
        assert len(errors) == 1, paradigm_id
        assert "fitting target range [10,930)" in errors[0]
        assert "model-selection target range [10,1033)" in errors[0]


def test_architecture_adapter_does_not_invent_missing_caller_split_facts(
    tmp_path,
):
    run_dir = tmp_path / "run"
    _seed_adapter_run(run_dir, _source("archived_trainer.py"))

    errors = architecture_split_lineage_errors(
        _SPEC, _BUILD_PLAN, run_dir
    )

    # val_split_start belongs to the later caller.  Exact verdict coverage is
    # above; Stage 2b does not fabricate its value to manufacture a range.
    assert errors == []


def test_notebook_adapter_consumes_realized_params_and_live_bundle_extent(
    tmp_path,
):
    run_dir = tmp_path / "run"
    _seed_adapter_run(run_dir, _source("current_trainer.py"))
    tree = ast.parse(_source("current_notebook_generated.py"))

    errors = notebook_split_lineage_errors(
        _SPEC, _BUILD_PLAN, run_dir, tree
    )

    # The trainer-local fitting/selection pair is already owned by Stage 2b,
    # so Stage 3a reports only the two caller/evaluation relationships.
    assert len(errors) == 2
    assert any("fitting target range [10,930)" in error for error in errors)
    assert any("reported-evaluation target range [826,830)" in error
               and "masked support envelope" in error for error in errors)
    assert all("root `demand_matrix`" in error for error in errors)
    assert not any(
        "fitting target range [10,930)" in error
        and "model-selection target range [10,1033)" in error
        for error in errors
    )


def test_forecast_only_helper_is_not_discovered_as_a_trainer(tmp_path):
    run_dir = tmp_path / "run"
    _seed_adapter_run(run_dir, _source("current_trainer.py"))
    method_source = """
def forecast(model, history, **kwargs):
    def diagnostic(prediction, target):
        return mean_squared_error(prediction, target)

    model.eval()
    prediction = model(history)
    debug_score = mean_squared_error(prediction, history)
    model.train()
    return prediction

def train_and_forecast(model, target):
    loss = model.loss(target)
    loss.backward()
    return model
"""
    (run_dir / "method" / "method.py").write_text(
        method_source, encoding="utf-8"
    )

    assert _trainer_names(method_source) == ("train_and_forecast",)
    errors = notebook_split_lineage_errors(
        _SPEC,
        _BUILD_PLAN,
        run_dir,
        ast.parse(_source("current_notebook_generated.py")),
    )
    assert len(errors) == 2
    assert all("reported-evaluation target" in error for error in errors)


def test_current_metric_and_caller_renames_do_not_clear_notebook_findings(
    tmp_path,
):
    run_dir = tmp_path / "run"
    _seed_adapter_run(run_dir, _source("current_trainer.py"))
    source = _source("current_notebook_generated.py")
    renamed = (
        source.replace("demand_matrix", "observed_values")
        .replace("observed_values=observed_values", "demand_matrix=observed_values")
        .replace("rmse =", "score =")
    )

    original = notebook_split_lineage_errors(
        _SPEC, _BUILD_PLAN, run_dir, ast.parse(source)
    )
    renamed_errors = notebook_split_lineage_errors(
        _SPEC, _BUILD_PLAN, run_dir, ast.parse(renamed)
    )

    assert len(original) == len(renamed_errors) == 2
    assert all("reported-evaluation target" in error
               for error in renamed_errors)

    metric_call = source.replace(
        "rmse = sqrt(mean((actual_finite - mean_finite) ** 2))",
        "score = mean_squared_error(actual_finite, mean_finite)",
    )
    assert len(notebook_split_lineage_errors(
        _SPEC, _BUILD_PLAN, run_dir, ast.parse(metric_call)
    )) == 2


def test_neutral_temporal_root_rename_does_not_clear_reported_evaluation(
    tmp_path,
):
    run_dir = tmp_path / "run"
    trainer = _source("current_trainer.py").replace(
        "demand_matrix", "matrix"
    )
    notebook = _source("current_notebook_generated.py").replace(
        "demand_matrix", "matrix"
    )
    _seed_adapter_run(run_dir, trainer)
    contract_path = run_dir / ".pipeline" / "arch_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["training_loop"]["input_shapes"] = {"matrix": "(N, T)"}
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    errors = notebook_split_lineage_errors(
        _SPEC, _BUILD_PLAN, run_dir, ast.parse(notebook)
    )

    assert len(errors) == 2
    assert all("reported-evaluation target" in error for error in errors)
    assert all("root `matrix`" in error for error in errors)


def test_training_temporal_inputs_preserve_versions_and_semantic_identity():
    legacy = {
        "schema_version": "1.1.0",
        "training_loop": {
            "input_shapes": {
                "legacy_values": "(N, T)",
                "legacy_features": "(N, M)",
            }
        },
    }
    typed = _v2_adapter_contract("neutral_values")
    typed["training_loop"]["input"].update({
        "tensor_values": {
            "kind": "tensor",
            "dtype": "float32",
            "dimensions": [{"dimension": "time_axis_steps"}],
        },
        "glyph_only": {
            "kind": "ndarray",
            "dtype": "float64",
            "dimensions": [
                {"dimension": "feature_width", "display_symbol": "T"}
            ],
        },
        "time_series_by_description": {
            "kind": "opaque",
            "type_description": "time-series table",
            "reason": "No typed array contract.",
        },
    })

    ArchContractV2.model_validate(typed)
    assert _training_temporal_input_names(legacy) == {"legacy_values"}
    assert _training_temporal_input_names(typed) == {
        "neutral_values",
        "tensor_values",
    }


def test_v2_training_time_axis_keeps_neutral_root_in_split_enforcement(
    tmp_path,
):
    run_dir = tmp_path / "run"
    trainer = _source("current_trainer.py").replace(
        "demand_matrix", "matrix"
    )
    notebook = _source("current_notebook_generated.py").replace(
        "demand_matrix", "matrix"
    )
    typed_contract = _v2_adapter_contract()
    ArchContractV2.model_validate(typed_contract)
    _seed_adapter_run(
        run_dir, trainer, arch_contract=typed_contract
    )

    errors = notebook_split_lineage_errors(
        _SPEC, _BUILD_PLAN, run_dir, ast.parse(notebook)
    )

    assert len(errors) == 2
    assert all("reported-evaluation target" in error for error in errors)
    assert all("root `matrix`" in error for error in errors)
    assert all("masked support envelope" in error for error in errors)


def test_v2_training_time_axis_preserves_exact_architecture_certainty(
    tmp_path,
):
    run_dir = tmp_path / "run"
    trainer = _source("current_trainer.py").replace(
        "demand_matrix", "matrix"
    )
    typed_contract = _v2_adapter_contract()
    _seed_adapter_run(
        run_dir, trainer, arch_contract=typed_contract
    )

    errors = architecture_split_lineage_errors(
        _SPEC, _BUILD_PLAN, run_dir
    )

    assert len(errors) == 1
    assert "fitting target range [10,930)" in errors[0]
    assert "model-selection target range [10,1033)" in errors[0]
    assert "overlap [10,930) exactly" in errors[0]


def test_v2_training_time_axis_keeps_disjoint_public_seams_clean(tmp_path):
    run_dir = tmp_path / "run"
    trainer = _source("current_trainer_disjoint.py").replace(
        "demand_matrix", "matrix"
    )
    notebook = _source("current_notebook_disjoint.py").replace(
        "demand_matrix", "matrix"
    )
    _seed_adapter_run(
        run_dir, trainer, arch_contract=_v2_adapter_contract()
    )

    assert architecture_split_lineage_errors(
        _SPEC, _BUILD_PLAN, run_dir
    ) == []
    assert notebook_split_lineage_errors(
        _SPEC, _BUILD_PLAN, run_dir, ast.parse(notebook)
    ) == []


def test_helper_call_and_decomposed_metric_are_semantic_evaluation_sinks(
    tmp_path,
):
    run_dir = tmp_path / "run"
    _seed_adapter_run(run_dir, _source("current_trainer.py"))
    source = _source("current_notebook_generated.py")
    variants = (
        source.replace(
            "rmse = sqrt(mean((actual_finite - mean_finite) ** 2))",
            "score = evaluate_model(mean_finite, actual_finite)",
        ),
        source.replace(
            "rmse = sqrt(mean((actual_finite - mean_finite) ** 2))",
            "score = compare(mean_finite, actual_finite)",
        ),
        source.replace(
            "rmse = sqrt(mean((actual_finite - mean_finite) ** 2))",
            "residuals = actual_finite - mean_finite\n"
            "score = sqrt(mean(residuals ** 2))",
        ),
        source.replace(
            "rmse = sqrt(mean((actual_finite - mean_finite) ** 2))",
            "print(mean_squared_error(mean_finite, actual_finite))",
        ),
        source.replace(
            "rmse = sqrt(mean((actual_finite - mean_finite) ** 2))",
            "score = evaluate_model(trained_model, actual_finite)",
        ),
    )

    for variant in variants:
        errors = notebook_split_lineage_errors(
            _SPEC, _BUILD_PLAN, run_dir, ast.parse(variant)
        )
        assert len(errors) == 2
        assert all("reported-evaluation target" in error for error in errors)


def test_conditioning_history_visualization_is_not_reported_evaluation():
    prefix = """
series = opaque_loader()
model = build_model()
fitting = series[:, :90]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
prediction = forecast(model, history=history)
"""
    consumers = (
        "plt.plot(history, prediction)",
        "render_pair(history, prediction)",
        "artifact = render_pair(history, prediction)",
        "logger.info(history, prediction)",
        "render_pair(history[:, :5], prediction)",
        "artifact = render_pair(history[:, 5:], prediction)",
        "logger.info(history[isfinite(history)], prediction)",
    )

    for consumer in consumers:
        report = analyze_split_lineage(
            f"{prefix}\n{consumer}\n",
            {},
            constants={"series.protocol_length": 100},
        )
        assert {item.role for item in report.observations} == {
            FITTING, INFERENCE_CONDITIONING,
        }
        assert find_split_lineage_findings(report) == ()


def test_neutral_forecast_argument_is_conditioning_without_name_tokens():
    report = analyze_split_lineage(
        """
matrix = opaque_loader()
model = build_model()
fitting = matrix[:, :90]
loss = model.loss(fitting)
loss.backward()
prefix = matrix[:, 80:90]
prediction = forecast(model, prefix)
render_pair(prefix, prediction)
""",
        {},
        constants={"matrix.protocol_length": 100},
    )

    assert {item.role for item in report.observations} == {
        FITTING, INFERENCE_CONDITIONING,
    }
    assert find_split_lineage_findings(report) == ()


def test_distinct_same_range_target_is_not_laundered_as_conditioning():
    prefix = """
series = opaque_loader()
model = build_model()
fitting = series[:, :90]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
prediction = forecast(model, history)
actual = series[:, 80:90]
"""
    for metric in (
        "score = compare(prediction, actual)",
        "score = mean_squared_error(prediction, history)",
    ):
        report = analyze_split_lineage(
            f"{prefix}\n{metric}\n",
            {},
            constants={"series.protocol_length": 100},
        )
        finding = find_split_lineage_findings(report)[0]
        assert finding.kind == "exact_overlap"
        assert finding.overlap == ProtocolRange(80, 90)


def test_conditioning_lineage_is_scoped_to_each_helper_call():
    helper = """
def process(model, tensor, mode):
    value = tensor[:, 80:90]
    if mode == 0:
        prediction = forecast(model, value)
        render_pair(value, prediction)
        return model
    prediction = model(value)
    return compare(prediction, value)
"""
    report = analyze_split_lineage(
        """
series = opaque_loader()
model = build_model()
conditioned = process(model, series, mode=0)
fitting = series[:, :90]
loss = model.loss(fitting)
loss.backward()
score = process(model, series, mode=1)
""",
        {"process": helper},
        constants={"series.protocol_length": 100},
    )

    finding = find_split_lineage_findings(report)[0]
    assert finding.kind == "exact_overlap"
    assert finding.overlap == ProtocolRange(80, 90)


def test_unsupported_prediction_call_fails_closed_at_relevant_metric(
    tmp_path,
):
    run_dir = tmp_path / "run"
    _seed_adapter_run(run_dir, _source("current_trainer.py"))
    source = _source("current_notebook_generated.py").replace(
        "forecast(trained_model", "rollout(trained_model"
    )

    errors = notebook_split_lineage_errors(
        _SPEC, _BUILD_PLAN, run_dir, ast.parse(source)
    )

    assert len(errors) == 2
    assert all("temporal_target_range_unresolved" in error
               for error in errors)
    assert all("unsupported model-consuming call `rollout`" in error
               for error in errors)


def test_archived_generated_adapter_restores_wrapper_and_live_end_lineage(
    tmp_path,
):
    run_dir = tmp_path / "run"
    _seed_adapter_run(run_dir, _source("archived_trainer.py"))

    errors = notebook_split_lineage_errors(
        _SPEC,
        _BUILD_PLAN,
        run_dir,
        ast.parse(_source("archived_notebook_generated.py")),
    )

    assert len(errors) == 3
    assert any(
        "fitting target range [10,1032)" in error
        and "model-selection target range [1029,1033)" in error
        and "overlap [1029,1032) exactly" in error
        and "Fix owner: `r2c-architecture-coder`" in error
        and "route the finding across producers" in error
        for error in errors
    )
    assert sum("reported-evaluation target range [1029,1033)" in error
               for error in errors) == 2
    assert all("root `demand_matrix`" in error for error in errors)
    assert not any("range unresolved" in error for error in errors)


def test_notebook_deduplication_is_scoped_to_the_producing_trainer(tmp_path):
    run_dir = tmp_path / "run"
    training_source = """
def train_local(model, series):
    fitting = series[:, :90]
    loss = model.loss(fitting)
    loss.backward()
    selected = series[:, 80:100]
    selected_loss = model.loss(selected)
    if selected_loss < 1:
        best_state = model.state_dict()
    return model

def train_bound(model, series, split):
    fitting = series[:, :split]
    loss = model.loss(fitting)
    loss.backward()
    selected = series[:, split - 10:split + 10]
    selected_loss = model.loss(selected)
    if selected_loss < 1:
        best_state = model.state_dict()
    return model
"""
    _seed_adapter_run(run_dir, training_source)
    notebook = ast.parse("""
series = opaque_loader()
first = build_model()
second = build_model()
trained_first = train_local(first, series)
trained_second = train_bound(second, series, split=90)
""")

    architecture_errors = architecture_split_lineage_errors(
        _SPEC, _BUILD_PLAN, run_dir
    )
    notebook_errors = notebook_split_lineage_errors(
        _SPEC, _BUILD_PLAN, run_dir, notebook
    )

    assert len(architecture_errors) == 1
    assert len(notebook_errors) == 1
    assert "fitting target range [0,90)" in notebook_errors[0]
    assert "model-selection target range [80,100)" in notebook_errors[0]


def test_notebook_deduplication_preserves_a_second_root_in_one_trainer(
    tmp_path,
):
    run_dir = tmp_path / "run"
    training_source = """
def train_both(model, demand_matrix, time_varying_matrix, split):
    demand_fit = demand_matrix[:, :90]
    demand_loss = model.loss(demand_fit)
    demand_loss.backward()
    demand_selected = demand_matrix[:, 80:100]
    demand_selected_loss = model.loss(demand_selected)
    if demand_selected_loss < 1:
        demand_state = model.state_dict()

    temporal_fit = time_varying_matrix[:, :split]
    temporal_loss = model.loss(temporal_fit)
    temporal_loss.backward()
    temporal_selected = time_varying_matrix[:, split - 10:split + 10]
    temporal_selected_loss = model.loss(temporal_selected)
    if temporal_selected_loss < 1:
        temporal_state = model.state_dict()
    return model
"""
    _seed_adapter_run(run_dir, training_source)
    contract_path = run_dir / ".pipeline" / "arch_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["training_loop"]["function_name"] = "train_both"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    notebook = ast.parse("""
demand_matrix = opaque_loader()
time_varying_matrix = opaque_loader()
model = build_model()
trained = train_both(
    model, demand_matrix, time_varying_matrix, split=90
)
""")

    architecture_errors = architecture_split_lineage_errors(
        _SPEC, _BUILD_PLAN, run_dir
    )
    notebook_errors = notebook_split_lineage_errors(
        _SPEC, _BUILD_PLAN, run_dir, notebook
    )

    assert len(architecture_errors) == 1
    assert "root `demand_matrix`" in architecture_errors[0]
    assert len(notebook_errors) == 1
    assert "root `time_varying_matrix`" in notebook_errors[0]


def test_notebook_deduplication_normalizes_a_sliced_formal_binding(tmp_path):
    run_dir = tmp_path / "run"
    training_source = """
def train_local(model, series):
    fitting = series[:, :90]
    loss = model.loss(fitting)
    loss.backward()
    selected = series[:, 80:100]
    selected_loss = model.loss(selected)
    if selected_loss < 1:
        best_state = model.state_dict()
    return model
"""
    _seed_adapter_run(run_dir, training_source)
    notebook = ast.parse("""
demand_matrix = opaque_loader()
window = demand_matrix[:, 100:200]
model = build_model()
trained = train_local(model, window)
""")

    assert len(architecture_split_lineage_errors(
        _SPEC, _BUILD_PLAN, run_dir
    )) == 1
    assert notebook_split_lineage_errors(
        _SPEC, _BUILD_PLAN, run_dir, notebook
    ) == []


def test_model_receiver_identity_separates_roles_and_dedupes_nonfirst_model(
    tmp_path,
):
    different_receivers = summarize_trainer(
        """
def train_both(model_a, model_b, series):
    fitting = series[:, :90]
    fitting_loss = model_a.loss(fitting)
    fitting_loss.backward()
    selected = series[:, 80:100]
    selected_loss = model_b.loss(selected)
    if selected_loss < 1:
        best_state = model_b.state_dict()
    return model_a
""",
        function_name="train_both",
        constants={"series.protocol_length": 100},
    )
    by_role = {item.role: item.model_id
               for item in different_receivers.observations}
    assert by_role == {
        FITTING: "model_a",
        MODEL_SELECTION: "model_b",
    }
    assert find_split_lineage_findings(different_receivers) == ()

    functional_receivers = summarize_trainer(
        """
def train_both(model_a, model_b, series):
    fitting = series[:, :90]
    fitting_loss = criterion(model_a(fitting), fitting)
    fitting_loss.backward()
    selected = series[:, 80:100]
    selected_loss = criterion(model_b(selected), selected)
    if selected_loss < 1:
        best_state = model_b.state_dict()
    return model_a
""",
        function_name="train_both",
        constants={"series.protocol_length": 100},
    )
    assert {item.role: item.model_id
            for item in functional_receivers.observations} == {
        FITTING: "model_a",
        MODEL_SELECTION: "model_b",
    }
    assert find_split_lineage_findings(functional_receivers) == ()

    functional_same_model = summarize_trainer(
        """
def train_both(model_a, model_b, series):
    fitting = series[:, :90]
    prediction = model_b(fitting)
    fitting_loss = criterion(prediction, fitting)
    fitting_loss.backward()
    selected = series[:, 80:100]
    selected_loss = model_b.loss(selected)
    if selected_loss < 1:
        best_state = model_b.state_dict()
    return model_b
""",
        function_name="train_both",
        constants={"series.protocol_length": 100},
    )
    same_model_finding = find_split_lineage_findings(
        functional_same_model
    )[0]
    assert same_model_finding.model_id == "model_b"

    multi_model_loss = summarize_trainer(
        """
def train_both(model_a, model_b, series):
    fitting = series[:, :90]
    prediction_a = model_a(fitting)
    prediction_b = model_b(fitting)
    fitting_loss = criterion(prediction_a + prediction_b, fitting)
    fitting_loss.backward()
    selected = series[:, 80:100]
    selected_loss = model_b.loss(selected)
    if selected_loss < 1:
        best_state = model_b.state_dict()
    return model_b
""",
        function_name="train_both",
        constants={"series.protocol_length": 100},
    )
    multi_model_finding = find_split_lineage_findings(
        multi_model_loss
    )[0]
    assert multi_model_finding.model_id == "model_b"

    run_dir = tmp_path / "run"
    training_source = """
def train_second(model_a, model_b, series):
    fitting = series[:, :90]
    fitting_loss = model_b.loss(fitting)
    fitting_loss.backward()
    selected = series[:, 80:100]
    selected_loss = model_b.loss(selected)
    if selected_loss < 1:
        best_state = model_b.state_dict()
    return model_b
"""
    _seed_adapter_run(run_dir, training_source)
    contract_path = run_dir / ".pipeline" / "arch_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["training_loop"]["function_name"] = "train_second"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    notebook = ast.parse("""
series = opaque_loader()
first = build_model()
second = build_model()
trained = train_second(first, second, series)
""")

    assert len(architecture_split_lineage_errors(
        _SPEC, _BUILD_PLAN, run_dir
    )) == 1
    assert notebook_split_lineage_errors(
        _SPEC, _BUILD_PLAN, run_dir, notebook
    ) == []


def test_method_adapter_catches_method_coder_owned_overlap(tmp_path):
    run_dir = tmp_path / "run"
    _seed_adapter_run(run_dir, _source("current_trainer.py"))
    (run_dir / "method" / "method.py").write_text(
        _source("current_trainer.py"), encoding="utf-8"
    )

    errors = method_split_lineage_errors(_SPEC, _BUILD_PLAN, run_dir)

    assert len(errors) == 1
    assert errors[0].startswith("method/method.py:")
    assert "fitting target range [10,930)" in errors[0]


def test_method_adapter_separates_reported_evaluation_from_progress_metrics(
    tmp_path,
):
    run_dir = tmp_path / "run"
    _seed_adapter_run(run_dir, _source("current_trainer.py"))
    method_source = """
def train_and_eval(model, series):
    fitting_target = series[:, :90]
    fitting_loss = model.loss(fitting_target)
    fitting_loss.backward()
    actual = series[:, 80:100]
    prediction = model(actual)
    score = mean_squared_error(prediction, actual)
    return score

def train_accuracy_progress(model, y_train):
    progress_target = y_train[:, :40]
    loss = model.loss(progress_target)
    loss.backward()
    train_rmse = mean_squared_error(model(progress_target), progress_target)
    return loss
"""
    (run_dir / "method" / "method.py").write_text(
        method_source, encoding="utf-8"
    )
    spec = {
        "comparison": {
            "classification": {"id": "time_series_forecasting"},
            "pluggable_component": {"name": "train_and_eval"},
        }
    }

    errors = method_split_lineage_errors(spec, _BUILD_PLAN, run_dir)

    assert len(errors) == 1
    assert "fitting target range [0,90)" in errors[0]
    assert "reported-evaluation target range [80,100)" in errors[0]
    assert "train_accuracy_progress" not in errors[0]


def test_interprocedural_training_progress_metric_is_not_reported(
    tmp_path,
):
    run_dir = tmp_path / "run"
    training_source = """
def train_model(model, target):
    loss = model.loss(target)
    loss.backward()
    return model

def train_accuracy_progress(model, target):
    prediction = model(target)
    score = mean_squared_error(prediction, target)
    return score
"""
    _seed_adapter_run(run_dir, training_source)
    (run_dir / "method" / "method.py").write_text(
        """
def run_epoch(model, series):
    target = series[:, :90]
    trained = train_model(model, target)
    progress = train_accuracy_progress(trained, target)
    return trained
""",
        encoding="utf-8",
    )
    spec = {
        "comparison": {
            "classification": {"id": "time_series_forecasting"},
            "pluggable_component": {"name": "run_epoch"},
        }
    }

    assert method_split_lineage_errors(
        spec, _BUILD_PLAN, run_dir
    ) == []


def test_neutral_preferred_method_return_is_reported_evaluation(tmp_path):
    run_dir = tmp_path / "run"
    _seed_adapter_run(run_dir, """
def train_model(model, target):
    loss = model.loss(target)
    loss.backward()
    return model
""")
    (run_dir / "method" / "method.py").write_text(
        """
def run_epoch(model, series):
    target = series[:, 80:100]
    trained = train_model(model, target)
    prediction = trained(target)
    residual = target - prediction
    return sqrt(mean(residual ** 2))
""",
        encoding="utf-8",
    )
    spec = {
        "comparison": {
            "classification": {"id": "time_series_forecasting"},
            "pluggable_component": {"name": "run_epoch"},
        }
    }

    errors = method_split_lineage_errors(spec, _BUILD_PLAN, run_dir)

    assert len(errors) == 1
    assert "fitting target range [80,100)" in errors[0]
    assert "reported-evaluation target range [80,100)" in errors[0]


def test_metric_bearing_forecast_helper_is_a_temporal_consumer(tmp_path):
    run_dir = tmp_path / "run"
    _seed_adapter_run(run_dir, """
def train_model(model, target):
    loss = model.loss(target)
    loss.backward()
    return model
""")
    method_source = """
def compare(prediction, actual):
    residual = actual - prediction
    score = sqrt(mean(residual ** 2))
    return score

def forecast(model, series, split):
    fitting = series[:, :split]
    trained = train_model(model, fitting)
    actual = series[:, split - 10:split + 10]
    prediction = trained(actual)
    output = prediction.detach()
    return compare(output, actual)
"""
    (run_dir / "method" / "method.py").write_text(
        method_source, encoding="utf-8"
    )
    spec = {
        "comparison": {
            "classification": {"id": "time_series_forecasting"},
            "pluggable_component": {"name": "forecast"},
        }
    }

    assert _trainer_names(method_source, ("forecast",)) == ("forecast",)
    assigned_metric_source = """
def forecast(model, history, actual):
    prediction = model(history)
    residual = actual - prediction
    score: float = sqrt(mean(residual ** 2))
    return score
"""
    assert _trainer_names(
        assigned_metric_source, ("forecast",)
    ) == ("forecast",)
    for declaration in (
        "score = compare(prediction, actual)",
        "score: float = compare(prediction, actual)",
    ):
        neutral_assigned_source = f"""
def forecast(model, history, actual):
    prediction = model(history)
    {declaration}
    return score
"""
        assert _trainer_names(
            neutral_assigned_source, ("forecast",)
        ) == ("forecast",)
    for branches in (
        (
            "result = compare(prediction, actual)",
            "result = prediction",
        ),
        (
            "result = prediction",
            "result = compare(prediction, actual)",
        ),
    ):
        branched_source = f"""
def forecast(model, history, actual, metric_mode):
    prediction = model(history)
    if metric_mode:
        {branches[0]}
    else:
        {branches[1]}
    return result
"""
        assert _trainer_names(
            branched_source, ("forecast",)
        ) == ("forecast",)
    method_errors = method_split_lineage_errors(
        spec, _BUILD_PLAN, run_dir
    )
    notebook_errors = notebook_split_lineage_errors(
        spec,
        _BUILD_PLAN,
        run_dir,
        ast.parse("""
series = opaque_loader()
model = build_model()
score = forecast(model, series, split=90)
"""),
    )

    assert method_errors
    assert any("range unresolved" in error for error in method_errors)
    assert len(notebook_errors) == 1
    assert "fitting target range [0,90)" in notebook_errors[0]
    assert "reported-evaluation target range [80,100)" in notebook_errors[0]


def test_metric_forecast_branch_preserves_concrete_and_unresolved_paths():
    helper = """
def forecast(model, history, metric_mode):
    prediction = model(history)
    if metric_mode == 1:
        result = compare(prediction, history)
    else:
        result = prediction
    return result
"""

    def report(flag: str):
        return analyze_split_lineage(
            f"""
series = opaque_loader()
model = build_model()
fitting = series[:, :90]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(model, history, {flag})
render(history, result)
""",
            {"forecast": helper},
            constants={"series.protocol_length": 100},
        )

    prediction_path = report("0")
    assert {
        (item.role, item.protocol_range)
        for item in prediction_path.observations
    } == {
        (FITTING, ProtocolRange(0, 90)),
        (INFERENCE_CONDITIONING, ProtocolRange(80, 90)),
    }
    assert find_split_lineage_findings(prediction_path) == ()

    metric_path = report("1")
    assert any(
        item.role == REPORTED_EVALUATION
        and item.protocol_range == ProtocolRange(80, 90)
        for item in metric_path.observations
    )
    assert len(find_split_lineage_findings(metric_path)) == 1

    unresolved_path = report("opaque_flag()")
    assert any(
        item.role == REPORTED_EVALUATION
        and "conditional return" in item.reason
        for item in unresolved_path.unresolved
    )
    unresolved_findings = find_split_lineage_findings(unresolved_path)
    assert len(unresolved_findings) == 1
    assert unresolved_findings[0].certainty == "unresolved"


def test_forecast_metric_discovery_uses_reaching_data_operands():
    overwritten_metric = """
def forecast(model, history):
    prediction = model(history)
    debug = compare(prediction, history)
    result = debug
    result = prediction
    return result
"""
    returned_metric = """
def forecast(model, history):
    prediction = model(history)
    result = prediction
    result = compare(prediction, history)
    return result
"""
    renamed_target_metric = """
def forecast(engine, context, reference):
    estimate = engine(context)
    packaged = estimate.detach()
    return compare(packaged, reference)
"""
    probabilistic_constructor = """
def forecast(model, history, seed: int, num_samples: int = 100):
    result = model(history)
    mu = result.mean
    sigma = result.scale
    df = result.df
    distribution = StudentT(df, mu, sigma)
    samples = distribution.rsample((num_samples,))
    return ForecastResult(
        mean=mu,
        variance=sigma ** 2,
        samples=samples,
        distribution_params={"mu": mu, "sigma": sigma, "df": df},
    )
"""
    sampled_constructor = """
def forecast(model, history, seed: int, num_samples: int = 100):
    rng = default_rng(seed)
    output = model(history)
    mu = output["mu"]
    scale = output["scale"]
    df = output["df"]
    variance = (scale ** 2) * maximum(df, 2.001)
    samples = sample_from_student_t(mu, scale, df, num_samples, rng)
    return ForecastResult(mu, variance, samples)
"""
    untyped_packaging = """
def forecast(model, history, config):
    prediction = model(history)
    return ForecastResult(prediction, config)
"""
    untyped_sampling = """
def forecast(model, history, num_samples):
    prediction = model(history)
    samples = sample_prediction(prediction, num_samples)
    return samples
"""

    assert _trainer_names(overwritten_metric, ("forecast",)) == ()
    assert _trainer_names(returned_metric, ("forecast",)) == ("forecast",)
    assert _trainer_names(
        renamed_target_metric, ("forecast",)
    ) == ("forecast",)
    assert _trainer_names(
        probabilistic_constructor, ("forecast",)
    ) == ()
    assert _trainer_names(sampled_constructor, ("forecast",)) == ()
    assert _trainer_names(untyped_packaging, ("forecast",)) == ()
    assert _trainer_names(untyped_sampling, ("forecast",)) == ()


def test_try_paths_retain_relevant_temporal_consumers():
    trainer = summarize_trainer(
        """
def train_model(model, series):
    try:
        fitting = series[:, :90]
        loss = model.loss(fitting)
        loss.backward()
    except RuntimeError:
        pass
    selected = series[:, 80:100]
    selected_loss = model.loss(selected)
    if selected_loss < 1:
        best_state = model.state_dict()
    return model
""",
        constants={"series.protocol_length": 100},
    )
    trainer_findings = find_split_lineage_findings(trainer)
    assert len(trainer_findings) == 1
    assert trainer_findings[0].overlap == ProtocolRange(80, 90)

    helper = """
def forecast(model, history):
    prediction = model(history)
    try:
        return compare(prediction, history)
    except RuntimeError:
        return prediction
"""
    report = analyze_split_lineage(
        """
series = opaque_loader()
model = build_model()
fitting = series[:, :90]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(model, history)
render(history, result)
""",
        {"forecast": helper},
        constants={"series.protocol_length": 100},
    )
    findings = find_split_lineage_findings(report)
    assert len(findings) == 1
    assert findings[0].certainty == "unresolved"


def test_exact_loop_extent_controls_return_reachability():
    helper = """
def forecast(model, history, iterations):
    prediction = model(history)
    for _ in range(iterations):
        return compare(prediction, history)
    return prediction
"""

    def report(iterations: str):
        return analyze_split_lineage(
            f"""
series = opaque_loader()
model = build_model()
fitting = series[:, :90]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(model, history, {iterations})
render(history, result)
""",
            {"forecast": helper},
            constants={"series.protocol_length": 100},
        )

    empty = report("0")
    assert find_split_lineage_findings(empty) == ()
    assert any(
        item.role == INFERENCE_CONDITIONING
        and item.protocol_range == ProtocolRange(80, 90)
        for item in empty.observations
    )

    nonempty = report("1")
    findings = find_split_lineage_findings(nonempty)
    assert len(findings) == 1
    assert findings[0].certainty == "exact"

    unresolved = report("opaque_iterations()")
    findings = find_split_lineage_findings(unresolved)
    assert len(findings) == 1
    assert findings[0].certainty == "unresolved"


def test_loop_completion_skips_unreachable_metrics_and_loop_else():
    break_else = """
def forecast(model, history):
    prediction = model(history)
    for _ in range(1):
        break
    else:
        return compare(prediction, history)
    return prediction
"""
    break_tail = """
def forecast(model, history):
    prediction = model(history)
    for _ in range(1):
        break
        return compare(prediction, history)
    return prediction
"""
    continue_tail = """
def forecast(model, history):
    prediction = model(history)
    for _ in range(1):
        continue
        return compare(prediction, history)
    return prediction
"""
    normal_else = """
def forecast(model, history):
    prediction = model(history)
    for _ in range(1):
        pass
    else:
        return compare(prediction, history)
    return prediction
"""

    for source in (break_else, break_tail, continue_tail):
        assert _trainer_names(source, ("forecast",)) == ()
        report = analyze_split_lineage(
            """
series = opaque_loader()
model = build_model()
fitting = series[:, :90]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(model, history)
render(history, result)
""",
            {"forecast": source},
            constants={"series.protocol_length": 100},
        )
        assert find_split_lineage_findings(report) == ()
    assert _trainer_names(
        normal_else, ("forecast",)
    ) == ("forecast",)


def test_loop_completion_environments_join_as_support_not_fallthrough():
    break_source = """
def forecast(model, history):
    prediction = model(history)
    for i in range(2):
        if i == 0:
            target = history[:, :5]
            break
        else:
            target = history[:, 5:]
    return compare(prediction, target)
"""
    continue_source = """
def forecast(model, history, flag):
    prediction = model(history)
    for _ in range(1):
        if flag:
            target = history[:, :5]
            continue
        target = history[:, 5:]
    else:
        return compare(prediction, target)
    return prediction
"""

    def report(source: str, arguments: str):
        return analyze_split_lineage(
            f"""
series = opaque_loader()
model = build_model()
fitting = series[:, :83]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(model, history{arguments})
render(history, result)
""",
            {"forecast": source},
            constants={"series.protocol_length": 100},
        )

    for source, arguments in (
            (break_source, ""),
            (continue_source, ", opaque_flag()")):
        findings = find_split_lineage_findings(report(source, arguments))
        assert findings
        assert all(item.certainty != "exact" for item in findings)


def test_try_prefix_raise_and_finally_override_completion():
    exceptional_alias = """
def forecast(model, history):
    prediction = model(history)
    try:
        target = history
        risky()
    except RuntimeError:
        return compare(prediction, target)
    return prediction
"""
    metric_finally = """
def forecast(model, history):
    prediction = model(history)
    try:
        return prediction
    finally:
        return compare(prediction, history)
"""
    prediction_finally = """
def forecast(model, history):
    prediction = model(history)
    try:
        return compare(prediction, history)
    finally:
        return prediction
"""
    raised_tail = """
def forecast(model, history):
    prediction = model(history)
    raise RuntimeError()
    return compare(prediction, history)
"""
    bare_return_tail = """
def forecast(model, history):
    prediction = model(history)
    return
    return compare(prediction, history)
"""
    raised_finally = """
def forecast(model, history):
    prediction = model(history)
    try:
        return compare(prediction, history)
    finally:
        raise RuntimeError()
"""

    assert _trainer_names(
        exceptional_alias, ("forecast",)
    ) == ("forecast",)
    assert _trainer_names(
        metric_finally, ("forecast",)
    ) == ("forecast",)
    for source in (
            prediction_finally, raised_tail, bare_return_tail,
            raised_finally):
        assert _trainer_names(source, ("forecast",)) == ()

    def report(source: str):
        return analyze_split_lineage(
            """
series = opaque_loader()
model = build_model()
fitting = series[:, :90]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(model, history)
render(history, result)
""",
            {"forecast": source},
            constants={"series.protocol_length": 100},
        )

    alias_findings = find_split_lineage_findings(
        report(exceptional_alias)
    )
    assert alias_findings
    assert all(item.certainty == "unresolved" for item in alias_findings)

    metric_findings = find_split_lineage_findings(report(metric_finally))
    assert len(metric_findings) == 1
    assert metric_findings[0].certainty == "exact"

    for source in (
            prediction_finally, raised_tail, bare_return_tail,
            raised_finally):
        assert find_split_lineage_findings(report(source)) == ()


def test_try_handler_matching_and_prefix_overwrite_are_conservative():
    unmatched_tail = """
def forecast(model, history):
    prediction = model(history)
    try:
        raise ValueError()
    except RuntimeError:
        pass
    return compare(prediction, history)
"""
    unmatched_handler = """
def forecast(model, history):
    prediction = model(history)
    try:
        raise ValueError()
    except RuntimeError:
        return compare(prediction, history)
    return prediction
"""
    matched_handler = """
def forecast(model, history):
    prediction = model(history)
    try:
        raise ValueError()
    except ValueError:
        return compare(prediction, history)
    return prediction
"""
    overwritten_prefix = """
def forecast(model, history, config: int):
    prediction = model(history)
    try:
        target = history
        risky()
        target = config
    except RuntimeError:
        return compare(prediction, target)
    return prediction
"""
    builtin_parent_handler = """
def forecast(model, history):
    prediction = model(history)
    try:
        raise KeyError()
    except LookupError:
        pass
    return compare(prediction, history)
"""
    earlier_implicit_exception = """
def forecast(model, history):
    prediction = model(history)
    try:
        target = history
        risky()
        raise ValueError()
    except RuntimeError:
        return compare(prediction, target)
"""
    implicit_subscript_exception = """
def forecast(model, history, config):
    prediction = model(history)
    try:
        target = history
        value = config["missing"]
        raise ValueError()
    except KeyError:
        return compare(prediction, target)
    return prediction
"""
    exception_target_shadow = """
def forecast(model, history):
    prediction = model(history)
    target = history
    try:
        raise RuntimeError()
    except RuntimeError as target:
        return compare(prediction, target)
    return prediction
"""
    unreachable_handler = """
def forecast(model, history):
    prediction = model(history)
    try:
        pass
    except RuntimeError:
        return compare(prediction, history)
    return prediction
"""
    else_raise_bypasses_handler = """
def forecast(model, history):
    prediction = model(history)
    try:
        pass
    except ValueError:
        return compare(prediction, history)
    else:
        raise ValueError()
    return compare(prediction, history)
"""

    for source in (unmatched_tail, unmatched_handler):
        assert _trainer_names(source, ("forecast",)) == ()
    assert _trainer_names(
        matched_handler, ("forecast",)
    ) == ("forecast",)
    assert _trainer_names(
        overwritten_prefix, ("forecast",)
    ) == ("forecast",)
    assert _trainer_names(
        builtin_parent_handler, ("forecast",)
    ) == ("forecast",)
    assert _trainer_names(
        earlier_implicit_exception, ("forecast",)
    ) == ("forecast",)
    assert _trainer_names(
        implicit_subscript_exception, ("forecast",)
    ) == ("forecast",)
    for source in (
            exception_target_shadow, unreachable_handler,
            else_raise_bypasses_handler):
        assert _trainer_names(source, ("forecast",)) == ()

    def report(source: str, arguments: str = ""):
        return analyze_split_lineage(
            f"""
series = opaque_loader()
model = build_model()
fitting = series[:, :90]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(model, history{arguments})
render(history, result)
""",
            {"forecast": source},
            constants={"series.protocol_length": 100},
        )

    for source in (unmatched_tail, unmatched_handler):
        assert find_split_lineage_findings(report(source)) == ()
    matched = find_split_lineage_findings(report(matched_handler))
    assert len(matched) == 1
    assert matched[0].certainty == "exact"
    overwritten = find_split_lineage_findings(
        report(overwritten_prefix, ", 7")
    )
    assert overwritten
    assert all(item.certainty == "unresolved" for item in overwritten)
    assert find_split_lineage_findings(report(builtin_parent_handler))
    assert find_split_lineage_findings(report(earlier_implicit_exception))
    assert find_split_lineage_findings(
        report(implicit_subscript_exception, ", {}")
    )
    for source in (
            exception_target_shadow, unreachable_handler,
            else_raise_bypasses_handler):
        assert find_split_lineage_findings(report(source)) == ()


def test_early_loop_exit_downgrades_iteration_domain_to_support():
    early_exit = _loop_report("""    for i in range(0, 10):
        target = series[:, i]
        break
    loss = model.loss(target)
    loss.backward()
""")
    finding = find_split_lineage_findings(early_exit)[0]
    assert finding.overlap == ProtocolRange(5, 6)
    assert finding.certainty == "support"

    full_iteration = _loop_report("""    for i in range(0, 10):
        target = series[:, i]
        loss = model.loss(target)
        loss.backward()
""")
    finding = find_split_lineage_findings(full_iteration)[0]
    assert finding.overlap == ProtocolRange(5, 6)
    assert finding.certainty == "exact"

    post_loop = _loop_report("""    for i in range(0, 10):
        target = series[:, i]
    loss = model.loss(target)
    loss.backward()
""")
    finding = find_split_lineage_findings(post_loop)[0]
    assert finding.overlap == ProtocolRange(5, 6)
    assert finding.certainty == "support"


def test_while_early_exit_downgrades_iteration_domain_to_support():
    report = summarize_trainer(
        """
def train_model(model, series):
    i = 0
    while i < 10:
        target = series[:, i]
        break
        i += 1
    loss = model.loss(target)
    loss.backward()
    selected = series[:, 5:6]
    selected_loss = model.loss(selected)
    if selected_loss < 999:
        best_state = model.state_dict()
    return model
""",
        constants={"series.protocol_length": 20},
    )
    finding = find_split_lineage_findings(report)[0]
    assert finding.overlap == ProtocolRange(5, 6)
    assert finding.certainty == "support"


def test_finally_executes_against_each_return_path_environment():
    source = """
def forecast(model, history, flag):
    prediction = model(history)
    try:
        if flag:
            target = history[:, :5]
            return prediction
        target = history[:, 5:]
    finally:
        return compare(prediction, target)
"""
    assert _trainer_names(source, ("forecast",)) == ("forecast",)
    report = analyze_split_lineage(
        """
series = opaque_loader()
model = build_model()
fitting = series[:, :83]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(model, history, opaque_flag())
render(history, result)
""",
        {"forecast": source},
        constants={"series.protocol_length": 100},
    )
    findings = find_split_lineage_findings(report)
    assert findings
    assert all(item.certainty == "support" for item in findings)


def test_nested_non_call_implicit_exception_prefixes_reach_handlers():
    bodies = (
        """        if flag:
            target = history
            value = config["missing"]
""",
        """        with context:
            target = history
            value = config["missing"]
""",
        """        for _ in range(1):
            target = history
            value = config["missing"]
""",
        """        try:
            target = history
            value = config["missing"]
        finally:
            pass
""",
    )

    for body in bodies:
        source = f"""
def forecast(model, history, config, flag, context):
    prediction = model(history)
    try:
{body}    except KeyError:
        return compare(prediction, target)
    return prediction
"""
        assert _trainer_names(source, ("forecast",)) == ("forecast",)
        report = analyze_split_lineage(
            """
series = opaque_loader()
model = build_model()
fitting = series[:, :90]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(model, history, {}, 1, opaque_context())
render(history, result)
""",
            {"forecast": source},
            constants={"series.protocol_length": 100},
        )
        assert find_split_lineage_findings(report)


def test_implicit_exception_safety_uses_live_bound_names():
    safe_return = """
def forecast(model, history):
    prediction = model(history)
    try:
        return prediction
    except RuntimeError:
        return compare(prediction, history)
"""
    missing_name = """
def forecast(model, history):
    prediction = model(history)
    try:
        copied = missing_name
    except NameError:
        return compare(prediction, history)
    return prediction
"""
    assert _trainer_names(safe_return, ("forecast",)) == ()
    assert _trainer_names(missing_name, ("forecast",)) == ("forecast",)

    def report(source: str):
        return analyze_split_lineage(
            """
series = opaque_loader()
model = build_model()
fitting = series[:, :90]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(model, history)
render(history, result)
""",
            {"forecast": source},
            constants={"series.protocol_length": 100},
        )

    assert find_split_lineage_findings(report(safe_return)) == ()
    assert find_split_lineage_findings(report(missing_name))


def test_unbound_local_paths_preserve_only_evaluated_expression_branches():
    exact_metric_sources = (
        """
def forecast(model, history, flag):
    prediction = model(history)
    if flag:
        target = history
    return compare(prediction, target)
""",
        """
def forecast(model, history, flag):
    prediction = model(history)
    try:
        if flag:
            target = history
    finally:
        return compare(prediction, target)
""",
        """
def forecast(model, history, flag):
    prediction = model(history)
    if False:
        target = history
    return compare(prediction, history) if True else target
""",
        """
def forecast(model, history, flag):
    prediction = model(history)
    if False:
        target = history
    return compare(prediction, history) or target
""",
        """
def forecast(model, history, flag):
    prediction = model(history)
    if False:
        target = history
    return compare(prediction, history) + len([
        target for _ in []
    ])
""",
        """
def forecast(model, history, flag):
    prediction = model(history)
    if False:
        target = history
    return compare(prediction, history) + len([
        target for _ in range(1) if False
    ])
""",
        """
def forecast(model, history, flag):
    prediction = model(history)
    if False:
        target = history
    return (
        compare(prediction, history)
        if 0 > 1 > target
        else compare(prediction, history)
    )
""",
        """
def forecast(model, history, flag):
    prediction = model(history)
    if False:
        target = history
    return (
        prediction
        if 0 > 1 > target
        else compare(prediction, history)
    )
""",
    )
    clean_sources = (
        """
def forecast(model, history, flag):
    prediction = model(history)
    if False:
        target = history
    try:
        return target if False else prediction
    except UnboundLocalError:
        return compare(prediction, history)
""",
        """
def forecast(model, history, flag):
    prediction = model(history)
    if False:
        target = history
    try:
        del target
    except UnboundLocalError:
        return prediction
    return compare(prediction, history)
""",
        """
def forecast(model, history, flag):
    prediction = model(history)
    if False:
        target = history
    try:
        return (
            compare(prediction, history)
            if 1 > 0 > target
            else prediction
        )
    except UnboundLocalError:
        return prediction
""",
        """
def forecast(model, history, flag):
    prediction = model(history)
    if False:
        target = history
    try:
        return (target for _ in range(1))
    except UnboundLocalError:
        return compare(prediction, history)
""",
        """
def forecast(model, history, flag):
    prediction = model(history)
    if False:
        target = history
    try:
        return [target for target in range(1)]
    except UnboundLocalError:
        return compare(prediction, history)
""",
        """
def forecast(model, history, flag):
    prediction = model(history)
    if False:
        target = history
    return (
        compare(prediction, history)
        if 0 > 1 > target
        else prediction
    )
""",
    )
    raising_sources = (
        """
def forecast(model, history, flag):
    prediction = model(history)
    try:
        try:
            raise RuntimeError()
        except RuntimeError as target:
            pass
        return target
    except UnboundLocalError:
        return compare(prediction, history)
""",
        """
def forecast(model, history, flag):
    prediction = model(history)
    target = history
    del target
    try:
        return target
    except UnboundLocalError:
        return compare(prediction, history)
""",
        """
def forecast(model, history, flag):
    prediction = model(history)
    if False:
        target = history
    try:
        return (_ for _ in target)
    except (UnboundLocalError, NameError):
        return compare(prediction, history)
""",
        """
def forecast(model, history, flag):
    prediction = model(history)
    if False:
        target = history
    try:
        return [target for _ in range(1)]
    except (UnboundLocalError, NameError):
        return compare(prediction, history)
""",
        """
def forecast(model, history, flag):
    prediction = model(history)
    try:
        raise RuntimeError()
    except RuntimeError as target:
        target = history
        return compare(prediction, target)
""",
    )

    def findings(source: str):
        report = analyze_split_lineage(
            """
series = opaque_loader()
model = build_model()
fitting = series[:, :90]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(model, history, opaque_flag())
render(history, result)
""",
            {"forecast": source},
            constants={"series.protocol_length": 100},
        )
        return find_split_lineage_findings(report)

    for source in exact_metric_sources + raising_sources:
        assert _trainer_names(source, ("forecast",)) == ("forecast",)
        result = findings(source)
        assert len(result) == 1
        assert result[0].certainty == "exact"
    for index, source in enumerate(clean_sources):
        expected_names = ("forecast",) if index == 2 else ()
        assert _trainer_names(source, ("forecast",)) == expected_names
        assert findings(source) == ()

    possible_assignment = """
def forecast(model, history, flag):
    prediction = model(history)
    first = history[:, :5]
    second = history[:, 5:]
    if flag:
        target = first
    try:
        copied = target
    except UnboundLocalError:
        return compare(prediction, second)
    return compare(prediction, copied)
"""
    assert _trainer_names(
        possible_assignment, ("forecast",)
    ) == ("forecast",)
    result = findings(possible_assignment)
    assert result
    assert all(item.certainty == "support" for item in result)


def test_unbound_handlers_do_not_recover_the_value_that_failed_to_load():
    statements = (
        "del target",
        "copied = target",
    )
    for statement in statements:
        helper = f"""
def forecast(model, history, report_metric, bind_target):
    prediction = model(history)
    if report_metric:
        return compare(prediction, history[:, 5:])
    if bind_target:
        target = history[:, :5]
    try:
        {statement}
    except UnboundLocalError:
        return compare(prediction, target)
    raise RuntimeError()
"""
        report = analyze_split_lineage(
            """
series = opaque_loader()
model = build_model()
fitting = series[:, :83]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(
    model, history, opaque_flag(), opaque_flag()
)
render(result)
""",
            {"forecast": helper},
            constants={"series.protocol_length": 100},
        )

        reported_ranges = {
            item.protocol_range
            for item in report.observations
            if item.role == REPORTED_EVALUATION
        }
        assert reported_ranges == {ProtocolRange(85, 90)}, statement
        assert find_split_lineage_findings(report) == (), statement


def test_target_operations_keep_bound_implicit_exception_path_conservative():
    statements = (
        "target[0] = 0",
        "target.attr = 0",
        "del target[0]",
        "del target.attr",
    )
    for statement in statements:
        helper = f"""
def forecast(model, history, bind_target):
    prediction = model(history)
    if bind_target:
        target = history[:, :5]
    try:
        {statement}
    except UnboundLocalError:
        return compare(prediction, target)
    raise RuntimeError()
"""
        report = analyze_split_lineage(
            """
series = opaque_loader()
model = build_model()
fitting = series[:, :83]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(model, history, opaque_flag())
render(result)
""",
            {"forecast": helper},
            constants={"series.protocol_length": 100},
        )

        findings = find_split_lineage_findings(report)
        assert findings, statement
        assert all(item.certainty == "exact" for item in findings), statement


def test_definitely_unbound_target_operation_cannot_reach_handler_metric():
    statements = (
        "target[0] = 0",
        "target.attr = 0",
        "del target[0]",
        "del target.attr",
    )
    for statement in statements:
        helper = f"""
def forecast(model, history):
    prediction = model(history)
    if False:
        target = history
    try:
        {statement}
    except UnboundLocalError:
        return compare(prediction, target)
    return prediction
"""
        report = analyze_split_lineage(
            """
series = opaque_loader()
model = build_model()
fitting = series[:, :90]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(model, history)
render(result)
""",
            {"forecast": helper},
            constants={"series.protocol_length": 100},
        )

        assert not any(
            item.role == REPORTED_EVALUATION
            for item in report.observations
        ), statement
        assert find_split_lineage_findings(report) == (), statement


def test_loop_targets_evaluate_bases_only_on_iteration():
    for target in ("target.attr", "target[0]"):
        nonempty = f"""
def forecast(model, history, bind_target):
    prediction = model(history)
    if bind_target:
        target = history
    try:
        for {target} in range(1):
            pass
    except UnboundLocalError:
        return compare(prediction, history)
    raise RuntimeError()
"""
        empty = nonempty.replace("range(1)", "range(0)").replace(
            "raise RuntimeError()", "return prediction"
        )
        bound_nonempty = nonempty.replace(
            "    if bind_target:\n        target = history\n",
            "    target = history\n",
        )

        def findings(source: str):
            report = analyze_split_lineage(
                """
series = opaque_loader()
model = build_model()
fitting = series[:, :90]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(model, history, opaque_flag())
render(result)
""",
                {"forecast": source},
                constants={"series.protocol_length": 100},
            )
            return find_split_lineage_findings(report)

        assert _trainer_names(nonempty, ("forecast",)) == ("forecast",)
        assert findings(nonempty)
        assert _trainer_names(
            bound_nonempty, ("forecast",)
        ) == ("forecast",)
        assert findings(bound_nonempty)
        assert _trainer_names(empty, ("forecast",)) == ()
        assert findings(empty) == ()


def test_eager_comprehension_targets_evaluate_bases_only_when_nonempty():
    def helper(expression: str, tail: str) -> str:
        return f"""
def forecast(model, history, bind_target):
    prediction = model(history)
    if bind_target:
        target = history
    try:
        result = {expression}
    except UnboundLocalError:
        return compare(prediction, history)
    {tail}
"""

    def findings(source: str):
        report = analyze_split_lineage(
            """
series = opaque_loader()
model = build_model()
fitting = series[:, :90]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(model, history, opaque_flag())
render(result)
""",
            {"forecast": source},
            constants={"series.protocol_length": 100},
        )
        return find_split_lineage_findings(report)

    forms = (
        ("[", "0", "]"),
        ("{", "0", "}"),
        ("{", "0: 0", "}"),
    )
    for target in ("target.attr", "target[0]"):
        for opening, value, closing in forms:
            nonempty_expression = (
                f"{opening}{value} for {target} in range(1){closing}"
            )
            empty_expression = (
                f"{opening}{value} for {target} in range(0){closing}"
            )
            dead_filter_expression = (
                f"{opening}{value} for _ in range(1) if False "
                f"for {target} in range(1){closing}"
            )
            dead_later_filter_expression = (
                f"{opening}{value} for _ in range(1) "
                f"if False if target{closing}"
            )
            nonempty = helper(nonempty_expression, "raise RuntimeError()")
            bound_nonempty = nonempty.replace(
                "    if bind_target:\n        target = history\n",
                "    target = history\n",
            )
            empty = helper(empty_expression, "return prediction")
            dead_filter = helper(
                dead_filter_expression, "return prediction"
            )
            dead_later_filter = helper(
                dead_later_filter_expression, "return prediction"
            )

            assert _trainer_names(
                nonempty, ("forecast",)
            ) == ("forecast",)
            assert findings(nonempty)
            assert _trainer_names(
                bound_nonempty, ("forecast",)
            ) == ("forecast",)
            assert findings(bound_nonempty)
            assert _trainer_names(empty, ("forecast",)) == ()
            assert findings(empty) == ()
            assert _trainer_names(dead_filter, ("forecast",)) == ()
            assert findings(dead_filter) == ()
            assert _trainer_names(
                dead_later_filter, ("forecast",)
            ) == ()
            assert findings(dead_later_filter) == ()


def test_ordered_store_with_and_delete_targets_preserve_runtime_bindings():
    metric_sources = (
        """
def forecast(model, history):
    prediction = model(history)
    if False:
        holder = history
    holder, holder.attr = opaque_pair()
    return compare(prediction, history)
""",
        """
def forecast(model, history):
    prediction = model(history)
    if False:
        holder = history
    holder = holder.attr = opaque_value()
    return compare(prediction, history)
""",
        """
def forecast(model, history):
    prediction = model(history)
    if False:
        holder = history
    for holder, holder.attr in [
        (opaque_value(), opaque_value()),
        (opaque_value(), opaque_value()),
    ]:
        return compare(prediction, history)
    raise RuntimeError()
""",
        """
def forecast(model, history, context):
    prediction = model(history)
    with context as nested, nested as leaf:
        return compare(prediction, history)
""",
        """
def forecast(model, history, context):
    prediction = model(history)
    if False:
        holder = history
    with context as (holder, holder.attr):
        return compare(prediction, history)
""",
    )
    arguments = ("", "", "", ", opaque_context()", ", opaque_context()")
    for source, args in zip(metric_sources, arguments):
        assert _trainer_names(source, ("forecast",)) == ("forecast",)
        findings = find_split_lineage_findings(
            _forecast_report(source, args)
        )
        assert findings, source
        assert all(item.certainty == "exact" for item in findings), source

    ordered_delete = """
def forecast(model, history):
    prediction = model(history)
    target = opaque_value()
    try:
        del target, target.attr
    except UnboundLocalError:
        return prediction
    return compare(prediction, history)
"""
    assert _trainer_names(ordered_delete, ("forecast",)) == ()
    assert find_split_lineage_findings(
        _forecast_report(ordered_delete)
    ) == ()


def test_destructuring_distinguishes_exact_mismatch_and_unknown_arity():
    unknown_sources = (
        """
def forecast(model, history, values):
    prediction = model(history)
    try:
        first, second = values
    except ValueError:
        return compare(prediction, history)
    raise RuntimeError()
""",
        """
def forecast(model, history, values):
    prediction = model(history)
    try:
        for first, second in values:
            pass
    except ValueError:
        return compare(prediction, history)
    raise RuntimeError()
""",
    )
    for source in unknown_sources:
        assert _trainer_names(source, ("forecast",)) == ("forecast",)
        findings = find_split_lineage_findings(
            _forecast_report(source, ", opaque_values()")
        )
        assert findings, source

    exact_controls = (
        "first, second = (1, 2)",
        "first, second = [1, 2]",
    )
    for statement in exact_controls:
        source = f"""
def forecast(model, history):
    prediction = model(history)
    try:
        {statement}
    except ValueError:
        return compare(prediction, history)
    return prediction
"""
        assert _trainer_names(source, ("forecast",)) == ()
        assert find_split_lineage_findings(_forecast_report(source)) == ()

    exact_loop = """
def forecast(model, history):
    prediction = model(history)
    try:
        for first, second in [(1, 2), (3, 4), (5, 6)]:
            pass
    except ValueError:
        return compare(prediction, history)
    return prediction
"""
    assert _trainer_names(exact_loop, ("forecast",)) == ()
    assert find_split_lineage_findings(_forecast_report(exact_loop)) == ()

    mismatch_sources = (
        """
def forecast(model, history):
    prediction = model(history)
    try:
        first, second = (1,)
    except ValueError:
        return prediction
    return compare(prediction, history)
""",
        """
def forecast(model, history):
    prediction = model(history)
    try:
        for first, second in [(1,)]:
            pass
    except ValueError:
        return prediction
    return compare(prediction, history)
""",
    )
    for source in mismatch_sources:
        assert _trainer_names(source, ("forecast",)) == ()
        assert find_split_lineage_findings(_forecast_report(source)) == ()


def test_literal_loops_execute_elements_and_completions_in_source_order():
    unreachable_metric = (
        """
def forecast(model, history):
    prediction = model(history)
    try:
        for first, second in [(1,), (2, 3)]:
            return compare(prediction, history)
    except ValueError:
        return prediction
    return prediction
""",
        """
def forecast(model, history):
    prediction = model(history)
    try:
        for first, second in [(1, 2), (3,)]:
            break
    except ValueError:
        return compare(prediction, history)
    return prediction
""",
        """
def forecast(model, history):
    prediction = model(history)
    try:
        for first, second in [(1, 2), (3,)]:
            return prediction
    except ValueError:
        return compare(prediction, history)
    return prediction
""",
        """
def forecast(model, history):
    prediction = model(history)
    try:
        for first, second in [(1, 2), (3,)]:
            pass
    except ValueError:
        return prediction
    else:
        return compare(prediction, history)
""",
    )
    for source in unreachable_metric:
        assert _trainer_names(source, ("forecast",)) == ()
        assert find_split_lineage_findings(_forecast_report(source)) == ()

    loop_carried_clean = """
def forecast(model, history):
    prediction = model(history)
    target = history
    for flag in [True, False]:
        if flag:
            target = opaque_value()
    return compare(prediction, target)
"""
    assert _trainer_names(loop_carried_clean, ("forecast",)) == ()
    assert find_split_lineage_findings(
        _forecast_report(loop_carried_clean)
    ) == ()

    loop_carried_metric = """
def forecast(model, history):
    prediction = model(history)
    if False:
        target = history
    for flag in [False, True]:
        if flag:
            target = history
    return compare(prediction, target)
"""
    assert _trainer_names(
        loop_carried_metric, ("forecast",)
    ) == ("forecast",)
    findings = find_split_lineage_findings(
        _forecast_report(loop_carried_metric)
    )
    assert findings
    assert all(item.certainty == "exact" for item in findings)


def test_known_unpack_failures_keep_valueerror_and_typeerror_distinct():
    clean_sources = (
        """
def forecast(model, history):
    prediction = model(history)
    try:
        first, second = (1,)
    except TypeError:
        return compare(prediction, history)
    except ValueError:
        return prediction
    return prediction
""",
        """
def forecast(model, history):
    prediction = model(history)
    try:
        first, second = 1
    except ValueError:
        return compare(prediction, history)
    except TypeError:
        return prediction
    return prediction
""",
        """
def forecast(model, history):
    prediction = model(history)
    try:
        first, second = range(2)
    except (TypeError, ValueError):
        return compare(prediction, history)
    return prediction
""",
        """
def forecast(model, history):
    prediction = model(history)
    try:
        first, second = range(1)
    except ValueError:
        return prediction
    return compare(prediction, history)
""",
        """
def forecast(model, history):
    prediction = model(history)
    try:
        for first, second in [1]:
            return compare(prediction, history)
    except TypeError:
        return prediction
    return prediction
""",
        """
def forecast(model, history):
    prediction = model(history)
    try:
        for first, second in range(1):
            return compare(prediction, history)
    except TypeError:
        return prediction
    return prediction
""",
    )
    for source in clean_sources:
        assert _trainer_names(source, ("forecast",)) == ()
        assert find_split_lineage_findings(_forecast_report(source)) == ()


def test_eager_comprehensions_share_ordered_typed_unpack_contract():
    expressions = (
        "[0 for first, second in {domain}]",
        "{0 for first, second in DOMAIN}",
        "{0: 0 for first, second in DOMAIN}",
    )
    controls = (
        (
            "[(1, 2), (3, 4)]",
            "except (TypeError, ValueError):\n"
            "        return compare(prediction, history)\n"
            "    return prediction",
        ),
        (
            "[(1,), (2, 3)]",
            "except ValueError:\n"
            "        return prediction\n"
            "    return compare(prediction, history)",
        ),
        (
            "[1]",
            "except ValueError:\n"
            "        return compare(prediction, history)\n"
            "    except TypeError:\n"
            "        return prediction\n"
            "    return prediction",
        ),
    )
    for template in expressions:
        for domain, tail in controls:
            expression = template.replace("{domain}", domain).replace(
                "DOMAIN", domain
            )
            source = f"""
def forecast(model, history):
    prediction = model(history)
    try:
        values = {expression}
    {tail}
"""
            assert _trainer_names(source, ("forecast",)) == (), source
            assert find_split_lineage_findings(
                _forecast_report(source)
            ) == (), source


def test_function_and_class_definitions_rebind_after_definition_inputs():
    rebinding_controls = (
        """
def forecast(model, history):
    prediction = model(history)
    target = history
    def target():
        return None
    return compare(prediction, target)
""",
        """
def forecast(model, history):
    prediction = model(history)
    target = history
    class target:
        pass
    return compare(prediction, target)
""",
    )
    for source in rebinding_controls:
        assert _trainer_names(source, ("forecast",)) == ()
        assert find_split_lineage_findings(_forecast_report(source)) == ()

    evaluated_before_binding = (
        """
def forecast(model, history):
    prediction = model(history)
    if False:
        target = history
    try:
        def target(value=target):
            return value
    except UnboundLocalError:
        return compare(prediction, history)
    raise RuntimeError()
""",
        """
def forecast(model, history):
    prediction = model(history)
    if False:
        target = history
    try:
        class target(target):
            pass
    except UnboundLocalError:
        return compare(prediction, history)
    raise RuntimeError()
""",
    )
    for source in evaluated_before_binding:
        assert _trainer_names(source, ("forecast",)) == ("forecast",)
        findings = find_split_lineage_findings(_forecast_report(source))
        assert findings, source
        assert all(item.certainty == "exact" for item in findings), source


def test_safe_definitions_deferred_annotations_and_rebinding_statements():
    safe_definitions = (
        "def helper():\n            pass",
        "async def helper():\n            pass",
        "class helper:\n            pass",
        "def helper(value=lambda: target):\n            pass",
    )
    for definition in safe_definitions:
        source = f"""
def forecast(model, history):
    prediction = model(history)
    if False:
        target = history
    try:
        {definition}
    except UnboundLocalError:
        return compare(prediction, history)
    return prediction
"""
        assert _trainer_names(source, ("forecast",)) == (), source
        assert find_split_lineage_findings(_forecast_report(source)) == ()

    deferred = """
from __future__ import annotations

def forecast(model, history):
    prediction = model(history)
    if False:
        marker = history
    try:
        def helper(value: marker):
            return value
    except UnboundLocalError:
        return compare(prediction, history)
    return prediction
"""
    assert _trainer_names(deferred, ("forecast",)) == ()
    assert find_split_lineage_findings(_forecast_report(deferred)) == ()

    walrus_default = """
def forecast(model, history):
    prediction = model(history)
    if False:
        marker = history
    try:
        def helper(first=(marker := 0), second=marker):
            return first, second
    except UnboundLocalError:
        return compare(prediction, history)
    return prediction
"""
    assert _trainer_names(walrus_default, ("forecast",)) == ()
    assert find_split_lineage_findings(
        _forecast_report(walrus_default)
    ) == ()

    rebindings = (
        "(target := opaque_value())",
        "if (target := opaque_value()):\n        pass",
        "import math as target",
        "from math import sin as target",
    )
    for statement in rebindings:
        source = f"""
def forecast(model, history):
    prediction = model(history)
    target = history
    {statement}
    return compare(prediction, target)
"""
        assert _trainer_names(source, ("forecast",)) == (), source
        assert find_split_lineage_findings(_forecast_report(source)) == ()


def test_valueless_complex_annotations_evaluate_runtime_target_reads():
    for target in ("target.attr", "target[0]"):
        source = f"""
def forecast(model, history):
    prediction = model(history)
    if False:
        target = history
    try:
        {target}: int
    except UnboundLocalError:
        return compare(prediction, history)
    return prediction
"""
        assert _trainer_names(source, ("forecast",)) == ("forecast",)
        findings = find_split_lineage_findings(_forecast_report(source))
        assert findings, target
        assert all(item.certainty == "exact" for item in findings), target


def test_class_body_nonlocal_rebinding_updates_the_enclosing_value():
    nonlocal_rebinding = """
def forecast(model, history):
    prediction = model(history)
    target = history
    class Evaluation:
        nonlocal target
        target = 0
    return compare(prediction, target)
"""
    assert _trainer_names(nonlocal_rebinding, ("forecast",)) == ()
    assert find_split_lineage_findings(
        _forecast_report(nonlocal_rebinding)
    ) == ()


def test_class_body_failures_restore_outer_locals_for_handlers():
    source = """
def forecast(model, history):
    prediction = model(history)
    target = history
    try:
        class Evaluation:
            target = 0
            raise ValueError()
    except ValueError:
        return compare(prediction, target)
    return prediction
"""

    assert _trainer_names(source, ("forecast",)) == ("forecast",)
    findings = find_split_lineage_findings(_forecast_report(source))
    assert findings
    assert all(item.certainty == "exact" for item in findings)


def test_class_namespace_preserves_returned_metric_lineage():
    source = """
def forecast(model, history):
    prediction = model(history)
    class Evaluation:
        score = compare(prediction, history)
    return Evaluation.score
"""

    assert _trainer_names(source, ("forecast",)) == ("forecast",)
    findings = find_split_lineage_findings(_forecast_report(source))
    assert findings
    assert all(item.certainty == "exact" for item in findings)


def test_expression_prefix_exceptions_survive_later_unbound_reads():
    expressions = (
        "opaque() + marker",
        "consume(opaque(), marker)",
        "opaque()[marker]",
    )
    for expression in expressions:
        source = f"""
def forecast(model, history):
    prediction = model(history)
    if False:
        marker = history
    try:
        result = {expression}
    except ValueError:
        return compare(prediction, history)
    except UnboundLocalError:
        return prediction
    return prediction
"""
        assert _trainer_names(source, ("forecast",)) == ("forecast",)
        findings = find_split_lineage_findings(_forecast_report(source))
        assert findings, expression
        assert all(
            item.certainty in {"support", "unresolved"}
            for item in findings
        ), expression


def test_unknown_call_keeps_bound_callee_exception_path_conservative():
    helper = """
def forecast(model, history, report_metric, bind_target):
    prediction = model(history)
    if report_metric:
        return compare(prediction, history[:, 5:])
    if bind_target:
        target = history[:, :5]
    try:
        consume(target)
    except UnboundLocalError:
        return compare(prediction, target)
    raise RuntimeError()
"""
    report = analyze_split_lineage(
        """
series = opaque_loader()
model = build_model()
fitting = series[:, :83]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(
    model, history, opaque_flag(), opaque_flag()
)
render(result)
""",
        {"forecast": helper},
        constants={"series.protocol_length": 100},
    )

    findings = find_split_lineage_findings(report)
    assert findings
    assert all(item.certainty == "support" for item in findings)


def test_mutually_exclusive_bindings_cannot_fabricate_normal_call_path():
    helper = """
def forecast(model, history, flag):
    prediction = model(history)
    if flag:
        first = history[:, :5]
    else:
        second = history[:, 5:]
    try:
        consume(first, second)
    except UnboundLocalError:
        raise RuntimeError()
    return compare(prediction, history)
"""
    report = analyze_split_lineage(
        """
series = opaque_loader()
model = build_model()
fitting = series[:, :90]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(model, history, opaque_flag())
render(result)
""",
        {"forecast": helper},
        constants={"series.protocol_length": 100},
    )

    assert not any(
        item.role == REPORTED_EVALUATION
        for item in report.observations
    )
    assert find_split_lineage_findings(report) == ()


def test_builtin_handler_order_consumes_unknown_implicit_paths():
    def nested(first: str, second: str) -> str:
        return f"""
def forecast(model, history):
    prediction = model(history)
    try:
        try:
            risky()
        except {first}:
            return prediction
    except {second}:
        return compare(prediction, history)
    return prediction
"""

    covered = (
        nested("BaseException", "RuntimeError"),
        nested("(Exception, BaseException)", "RuntimeError"),
        nested("Exception", "RuntimeError"),
    )
    uncovered = nested("RuntimeError", "Exception")
    for source in covered:
        assert _trainer_names(source, ("forecast",)) == ()
    assert _trainer_names(uncovered, ("forecast",)) == ("forecast",)

    def report(source: str):
        return analyze_split_lineage(
            """
series = opaque_loader()
model = build_model()
fitting = series[:, :90]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(model, history)
render(history, result)
""",
            {"forecast": source},
            constants={"series.protocol_length": 100},
        )

    for source in covered:
        assert find_split_lineage_findings(report(source)) == ()
    findings = find_split_lineage_findings(report(uncovered))
    assert findings
    assert all(item.certainty == "unresolved" for item in findings)


def test_metric_return_branches_and_non_metric_returns_keep_certainty_honest():
    both_metric = """
def forecast(model, history):
    prediction = model(history)
    target = history
    try:
        target = history[:, :5]
        raise ValueError()
    except ValueError:
        return compare(prediction, target)
"""
    bare_return = """
def forecast(model, history, flag):
    prediction = model(history)
    if flag:
        return
    return compare(prediction, history)
"""
    implicit_return = """
def forecast(model, history, flag):
    prediction = model(history)
    if flag:
        return compare(prediction, history)
"""

    def report(source: str, arguments: str):
        return analyze_split_lineage(
            f"""
series = opaque_loader()
model = build_model()
fitting = series[:, :83]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(model, history{arguments})
render(history, result)
""",
            {"forecast": source},
            constants={"series.protocol_length": 100},
        )

    both_findings = find_split_lineage_findings(report(both_metric, ""))
    assert both_findings
    assert all(item.certainty == "support" for item in both_findings)
    for source in (bare_return, implicit_return):
        findings = find_split_lineage_findings(
            report(source, ", opaque_flag()")
        )
        assert findings
        assert all(item.certainty == "unresolved" for item in findings)


def test_match_ifexp_and_boolop_metric_returns_are_reachable():
    sources = (
        """
def forecast(model, history, mode):
    prediction = model(history)
    match mode:
        case 1:
            return compare(prediction, history)
        case _:
            return prediction
""",
        """
def forecast(model, history, flag):
    prediction = model(history)
    return compare(prediction, history) if flag else prediction
""",
        """
def forecast(model, history, flag):
    prediction = model(history)
    return flag and compare(prediction, history)
""",
        """
def forecast(model, history, flag):
    prediction = model(history)
    return flag or compare(prediction, history)
""",
    )
    arguments = (", 1", ", 1", ", 1", ", 0")
    for source, args in zip(sources, arguments):
        assert _trainer_names(source, ("forecast",)) == ("forecast",)
        report = analyze_split_lineage(
            f"""
series = opaque_loader()
model = build_model()
fitting = series[:, :90]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(model, history{args})
render(history, result)
""",
            {"forecast": source},
            constants={"series.protocol_length": 100},
        )
        findings = find_split_lineage_findings(report)
        assert len(findings) == 1
        assert findings[0].certainty == "exact"


def test_assert_false_makes_following_metric_unreachable():
    source = """
def forecast(model, history):
    prediction = model(history)
    assert False
    return compare(prediction, history)
"""
    assert _trainer_names(source, ("forecast",)) == ()
    report = analyze_split_lineage(
        """
series = opaque_loader()
model = build_model()
fitting = series[:, :90]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(model, history)
render(history, result)
""",
        {"forecast": source},
        constants={"series.protocol_length": 100},
    )
    assert find_split_lineage_findings(report) == ()


def test_temporal_and_scalar_branch_join_fails_closed_at_metric():
    direct = """
def forecast(model, history, config, flag):
    prediction = model(history)
    if flag:
        target = history
    else:
        target = config
    return compare(prediction, target)
"""
    in_finally = """
def forecast(model, history, config, flag):
    prediction = model(history)
    try:
        if flag:
            target = history
        else:
            target = config
    finally:
        return compare(prediction, target)
"""

    for source in (direct, in_finally):
        report = analyze_split_lineage(
            """
series = opaque_loader()
model = build_model()
fitting = series[:, :90]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(model, history, 7, opaque_flag())
render(history, result)
""",
            {"forecast": source},
            constants={"series.protocol_length": 100},
        )
        findings = find_split_lineage_findings(report)
        assert findings
        assert all(item.certainty == "unresolved" for item in findings)

    different_roots = """
def forecast(model, history, reference, flag):
    prediction = model(history)
    if flag:
        target = history
    else:
        target = reference
    return compare(prediction, target)
"""
    report = analyze_split_lineage(
        """
series = opaque_loader()
other = opaque_loader()
model = build_model()
fitting = series[:, :90]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
reference = other[:, :10]
result = forecast(model, history, reference, opaque_flag())
render(history, result)
""",
        {"forecast": different_roots},
        constants={
            "series.protocol_length": 100,
            "other.protocol_length": 100,
        },
    )
    findings = find_split_lineage_findings(report)
    assert findings
    assert all(item.certainty == "support" for item in findings)


def test_literal_true_while_has_guaranteed_entry_and_no_normal_else():
    metric_return = """
def forecast(model, history):
    prediction = model(history)
    while True:
        return compare(prediction, history)
    return prediction
"""
    break_else = """
def forecast(model, history):
    prediction = model(history)
    while 1:
        break
    else:
        return compare(prediction, history)
    return prediction
"""
    assert _trainer_names(
        metric_return, ("forecast",)
    ) == ("forecast",)
    assert _trainer_names(break_else, ("forecast",)) == ()

    def report(source: str):
        return analyze_split_lineage(
            """
series = opaque_loader()
model = build_model()
fitting = series[:, :90]
loss = model.loss(fitting)
loss.backward()
history = series[:, 80:90]
result = forecast(model, history)
render(history, result)
""",
            {"forecast": source},
            constants={"series.protocol_length": 100},
        )

    metric_findings = find_split_lineage_findings(report(metric_return))
    assert len(metric_findings) == 1
    assert metric_findings[0].certainty == "exact"
    assert find_split_lineage_findings(report(break_else)) == ()


def test_method_adapter_follows_explicit_evaluator_consumer_across_modules(
    tmp_path,
):
    run_dir = tmp_path / "run"
    training_source = """
def train_model(model, target):
    loss = model.loss(target)
    loss.backward()
    return model

def compare(model, target):
    scorer = model.to(device)
    prediction = scorer.forward(target)
    residual = target - prediction
    return sqrt(mean(residual ** 2))
"""
    _seed_adapter_run(run_dir, training_source)
    (run_dir / "method" / "method.py").write_text(
        """
def train_and_forecast(model, series):
    target = series[:, 80:100]
    trained = train_model(model, target)
    return compare(trained, target)
""",
        encoding="utf-8",
    )
    spec = {
        "comparison": {
            "classification": {"id": "time_series_forecasting"},
            "pluggable_component": {"name": "train_and_forecast"},
        }
    }

    errors = method_split_lineage_errors(spec, _BUILD_PLAN, run_dir)

    assert len(errors) == 1
    assert errors[0].startswith("method/method.py:")
    assert "fitting target range [80,100)" in errors[0]
    assert "reported-evaluation target range [80,100)" in errors[0]
    assert "Fix owner: `r2c-method-coder`" in errors[0]

    (run_dir / "method" / "method.py").write_text(
        """
def train_and_forecast(model, series):
    fitting = series[:, :opaque_bound()]
    trained = train_model(model, fitting)
    actual = series[:, 80:100]
    return compare(trained, actual)
""",
        encoding="utf-8",
    )
    unresolved_errors = method_split_lineage_errors(
        spec, _BUILD_PLAN, run_dir
    )
    assert len(unresolved_errors) == 1
    assert "fitting target range unresolved" in unresolved_errors[0]
    assert "reported-evaluation target range [80,100)" \
        in unresolved_errors[0]
    assert "temporal_target_range_unresolved" in unresolved_errors[0]


def test_neutral_method_pluggable_is_instantiated_at_notebook_seam(tmp_path):
    run_dir = tmp_path / "run"
    _seed_adapter_run(run_dir, _source("current_trainer.py"))
    (run_dir / "method" / "method.py").write_text(
        """
def run_epoch(model, series, train_end):
    fitting_target = series[:, :train_end]
    loss = model.loss(fitting_target)
    loss.backward()
    return model
""",
        encoding="utf-8",
    )
    spec = {
        "comparison": {
            "classification": {"id": "time_series_forecasting"},
            "pluggable_component": {"name": "run_epoch"},
        }
    }
    notebook = ast.parse("""
series = opaque_loader()
model = build_model(context_length=10, forecast_horizon=4)
trained = run_epoch(model, series, train_end=900)
actual = series[:, 850:854]
prediction = forecast(trained, history=series[:, 840:850])
score = mean_squared_error(prediction, actual)
""")

    errors = notebook_split_lineage_errors(
        spec, _BUILD_PLAN, run_dir, notebook
    )

    assert len(errors) == 1
    assert "method/method.py" in errors[0]
    assert "fitting target range [0,900)" in errors[0]
    assert "reported-evaluation target range [850,854)" in errors[0]


def _loop_report(loop_body: str):
    return summarize_trainer(
        f"""
def train_model(model, series):
{loop_body}
    selected = series[:, 5:6]
    selected_loss = model.loss(selected)
    if selected_loss < 999:
        best_state = model.state_dict()
    return model
""",
        constants={"series.protocol_length": 20},
    )


def test_sparse_loop_and_affine_domains_remain_support_envelopes():
    bodies = (
        """    for t in range(0, 11, 10):
        target = series[:, t:t + 1]
        loss = model.loss(target)
        loss.backward()
""",
        """    t = 0
    while t < 11:
        target = series[:, t:t + 1]
        loss = model.loss(target)
        loss.backward()
        t += 10
""",
        """    for i in range(0, 6):
        t = i * 2
        target = series[:, t:t + 1]
        loss = model.loss(target)
        loss.backward()
""",
        """    for i in range(0, 6):
        t = i + i
        target = series[:, t:t + 1]
        loss = model.loss(target)
        loss.backward()
""",
    )

    for body in bodies:
        finding = find_split_lineage_findings(_loop_report(body))[0]
        assert finding.kind == "support_overlap"
        assert finding.overlap == ProtocolRange(5, 6)
        assert "sparse-index support envelope" in finding.message(
            "method/training.py"
        )

    cancellation = _loop_report("""    for i in range(0, 6):
        t = i - i
        target = series[:, t:t + 1]
        loss = model.loss(target)
        loss.backward()
""")
    assert find_split_lineage_findings(cancellation) == ()

    for alias in ("i + 0", "i - 0", "i * 1", "1 * i"):
        aliased_cancellation = _loop_report(f"""    for i in range(0, 6):
        j = {alias}
        t = i - j
        target = series[:, t:t + 1]
        loss = model.loss(target)
        loss.backward()
""")
        assert find_split_lineage_findings(aliased_cancellation) == ()


def test_independent_equal_loop_domains_do_not_collapse_to_one_symbol():
    report = summarize_trainer(
        """
def train_model(model, series):
    for i in range(0, 11, 2):
        for j in range(0, 11, 2):
            offset = i - j
            target = series[:, 20 + offset:21 + offset]
            loss = model.loss(target)
            loss.backward()
    selected = series[:, 16:17]
    selected_loss = model.loss(selected)
    if selected_loss < 1:
        best_state = model.state_dict()
    return model
""",
        constants={"series.protocol_length": 50},
    )

    finding = find_split_lineage_findings(report)[0]
    assert finding.kind == "support_overlap"
    assert finding.overlap == ProtocolRange(16, 17)


def test_negative_slice_bounds_are_relative_to_the_base_tensor():
    report = summarize_trainer(
        """
def train_model(model, series):
    history = series[:, 80:90]
    target = history[:, -5:]
    loss = model.loss(target)
    loss.backward()
    return model
""",
        constants={"series.protocol_length": 100},
    )

    fitting = next(item for item in report.observations
                   if item.role == FITTING)
    assert fitting.protocol_range == ProtocolRange(85, 90)

    gather_report = summarize_trainer(
        """
def train_model(model, series):
    history = series[:, 80:90]
    target = history[:, -1]
    loss = model.loss(target)
    loss.backward()
    selected = series[:, 89:90]
    selected_loss = model.loss(selected)
    if selected_loss < 1:
        best_state = model.state_dict()
    return model
""",
        constants={"series.protocol_length": 100},
    )
    finding = find_split_lineage_findings(gather_report)[0]
    assert finding.kind == "exact_overlap"
    assert finding.overlap == ProtocolRange(89, 90)


def test_protocol_axis_gathers_preserve_sparse_exact_and_unresolved_certainty():
    def report(index_expression: str):
        return _loop_report(f"""    indices = {index_expression}
    target = series[:, indices]
    loss = model.loss(target)
    loss.backward()
""")

    sparse = find_split_lineage_findings(report("arange(0, 10, 2)"))[0]
    assert sparse.kind == "support_overlap"
    assert sparse.first.protocol_range == ProtocolRange(0, 9) \
        or sparse.second.protocol_range == ProtocolRange(0, 9)

    dense = find_split_lineage_findings(report("arange(0, 10)"))[0]
    assert dense.kind == "exact_overlap"

    assert find_split_lineage_findings(report("arange(0, 4, 2)")) == ()

    unresolved = find_split_lineage_findings(report("opaque_indices()"))[0]
    assert unresolved.kind == "unresolved"
    assert unresolved.first.read_kind == "target"
    assert unresolved.first.root == "series"
    assert unresolved.first.model_id == "model"


def test_sparse_row_gather_does_not_weaken_exact_time_slice():
    report = _loop_report("""    rows = range(0, 10, 2)
    target = series[rows, :5]
    loss = model.loss(target)
    loss.backward()
""")

    fitting = next(item for item in report.observations
                   if item.role == FITTING)
    assert fitting.protocol_range == ProtocolRange(0, 5)
    assert fitting.subset is None
    assert find_split_lineage_findings(report) == ()


def test_non_temporal_family_is_outside_unresolved_enforcement_scope(tmp_path):
    run_dir = tmp_path / "run"
    _seed_adapter_run(run_dir, _source("current_trainer.py"))
    spec = {"comparison": {"classification": {"id": "active_learning"}}}

    assert architecture_split_lineage_errors(spec, _BUILD_PLAN, run_dir) == []
    assert notebook_split_lineage_errors(
        spec,
        _BUILD_PLAN,
        run_dir,
        ast.parse(_source("current_notebook_generated.py")),
    ) == []
    receipt = notebook_split_validity_receipt(
        spec,
        _BUILD_PLAN,
        run_dir,
        ast.parse(_source("current_notebook_generated.py")),
    )
    assert receipt["status"] == "not_applicable"
    assert receipt["reasons"] == ["non_temporal_paradigm"]
    assert receipt["evidence"] == {
        "findings": [],
        "relevant_unresolved": [],
        "fitting_to_reported_evaluation_relations": [],
        "fitting_target_ranges": [],
        "selection_target_ranges": [],
    }


def test_real_architecture_validator_entrypoint_emits_range_finding(tmp_path):
    from scripts.validate_architecture_coder_output import validate

    run_dir = tmp_path / "run"
    _seed_adapter_run(run_dir, _source("current_trainer.py"))
    (run_dir / "method" / "model.py").write_text(
        "class ForecastModel:\n"
        "    def forward(self, x):\n"
        "        return x\n",
        encoding="utf-8",
    )

    errors = validate(_SPEC, run_dir, Path.cwd())

    assert any(
        "fitting target range [10,930)" in error
        and "model-selection target range [10,1033)" in error
        for error in errors
    )


def test_real_method_validator_entrypoint_emits_range_finding(tmp_path):
    from scripts.validate_method_coder_output import validate

    run_dir = tmp_path / "run"
    _seed_adapter_run(run_dir, _source("current_trainer.py"))
    (run_dir / "method" / "method.py").write_text(
        _source("current_trainer.py"), encoding="utf-8"
    )
    spec = {
        "comparison": {
            "classification": {"id": "time_series_forecasting"},
            "pluggable_component": {"name": "train_model"},
        }
    }

    errors, _warnings = validate(spec, run_dir, Path.cwd())

    assert any(
        "fitting target range [10,930)" in error
        and "model-selection target range [10,1033)" in error
        for error in errors
    )


def test_real_notebook_validator_entrypoint_emits_only_interprocedural_findings(
    tmp_path,
):
    import nbformat

    from scripts.validate_notebook_output import validate

    run_dir = tmp_path / "run"
    _seed_adapter_run(run_dir, _source("current_trainer.py"))
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell("# Temporal demo"),
        nbformat.v4.new_code_cell("%pip install -r requirements.txt"),
        nbformat.v4.new_code_cell("%matplotlib inline"),
        nbformat.v4.new_code_cell(
            "params = " + repr({
                "context_length": {"value": 10},
                "forecast_horizon": {"value": 4},
            })
        ),
        nbformat.v4.new_code_cell(
            _source("current_notebook_generated.py")
        ),
    ]
    nbformat.write(nb, run_dir / "notebook.ipynb")

    errors = validate(_SPEC, run_dir, Path.cwd())
    lineage_errors = [
        error for error in errors if "temporal_target_role_overlap" in error
    ]

    assert len(lineage_errors) == 2
    assert all("reported-evaluation target range [826,830)" in error
               for error in lineage_errors)


def test_real_notebook_validator_entrypoint_accepts_disjoint_ranges(tmp_path):
    import nbformat

    from scripts.validate_notebook_output import validate

    run_dir = tmp_path / "run"
    _seed_adapter_run(run_dir, _source("current_trainer_disjoint.py"))
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell("# Temporal demo"),
        nbformat.v4.new_code_cell("%pip install -r requirements.txt"),
        nbformat.v4.new_code_cell("%matplotlib inline"),
        nbformat.v4.new_code_cell(
            "params = " + repr({
                "context_length": {"value": 10},
                "forecast_horizon": {"value": 4},
            })
        ),
        nbformat.v4.new_code_cell(
            _source("current_notebook_generated_disjoint.py")
        ),
    ]
    nbformat.write(nb, run_dir / "notebook.ipynb")

    errors = validate(_SPEC, run_dir, Path.cwd())

    assert not any(
        "temporal_target_role_overlap" in error
        or "temporal_target_range_unresolved" in error
        for error in errors
    )
