"""Stage 5 findings-routing regression tests."""

from __future__ import annotations

import json

import run_layout
from tests.helpers.state import make_state


def _rescale_finding() -> dict:
    return {
        "id": "F001",
        "severity": "critical",
        "description": "R_0 must be rescaled for normalized pixels.",
        "proposed_resolution": {
            "kind": "rescale_param",
            "param": "R_0",
            "new_value": 7.843,
            "factor_derivation": "2000.0 * (1/255) = 7.843",
            "expert_reasoning": "Normalized pixels shrink distances by 255x.",
            "alternative": "Use raw pixels and keep R_0=2000.0.",
        },
    }


def test_stage5_revalidation_rerenders_after_parameter_deriver_change(
    fake_subprocess, run_dir,
):
    """params.json feeds render_notebook.py, so Stage 5 parameter fixes must
    re-render before validate_notebook_output.py and smoke execution."""
    import run_pipeline

    state = make_state(run_dir)

    fake_subprocess.expect_stage2_script(ok=True)  # validate_params_output.py
    fake_subprocess.expect_script(returncode=0)  # render_notebook.py
    fake_subprocess.expect_stage2_script(ok=True)  # validate_notebook_output.py
    fake_subprocess.expect_script(returncode=0)  # smoke_run_notebook.py

    result = run_pipeline._run_relevant_revalidators(
        state, "stage_5", {"parameter-deriver"},
    )

    assert result is None
    assert fake_subprocess.stage2_calls[0]["args"] == [
        "scripts/validate_params_output.py",
    ]
    assert fake_subprocess.script_calls[0]["args"][0] == "scripts/render_notebook.py"
    assert fake_subprocess.stage2_calls[1]["args"] == [
        "scripts/validate_notebook_output.py",
    ]
    assert fake_subprocess.script_calls[1]["args"][0] == "scripts/smoke_run_notebook.py"


def _write_post_smoke_artifacts(run_dir, *, identity: str) -> dict[str, str]:
    pipeline = run_dir / ".pipeline"
    payloads = {
        "demo_verdict.json": json.dumps({
            "executed_evaluation": {
                "model_id": f"{identity}-model",
                "checkpoint_id": f"{identity}-checkpoint",
            },
        }),
        "target_scaling_state.json": json.dumps({
            "state_digest": f"{identity}-scaling-state",
        }),
        "training_history.json": json.dumps({
            "model_id": f"{identity}-model",
            "checkpoint_id": f"{identity}-checkpoint",
            "record_digest": f"{identity}-history",
        }),
    }
    for name, payload in payloads.items():
        (pipeline / name).write_text(payload, encoding="utf-8")
    return payloads


def test_stage5_fix_rebinds_all_post_smoke_artifacts_to_new_execution(
    fake_subprocess,
    run_dir,
    monkeypatch,
):
    """A producer fix cannot deliver the pre-fix model/checkpoint history."""
    import run_pipeline

    state = make_state(run_dir)
    _write_post_smoke_artifacts(run_dir, identity="old")

    fake_subprocess.expect_stage2_script(ok=True)  # validate_params_output.py
    fake_subprocess.expect_script(returncode=0)  # render_notebook.py
    fake_subprocess.expect_stage2_script(ok=True)  # validate_notebook_output.py
    fake_subprocess.expect_script(returncode=0)  # smoke_run_notebook.py

    calls = []

    def record_new_execution(candidate_state, *, stage_id="stage_3c"):
        assert candidate_state is state
        assert stage_id == "stage_5"
        # Invalidation precedes rendering/re-execution and recomputation.
        assert all(
            not (run_dir / ".pipeline" / name).exists()
            for name in run_pipeline._POST_SMOKE_DERIVED_ARTIFACTS
        )
        calls.append(stage_id)
        _write_post_smoke_artifacts(run_dir, identity="new")

    monkeypatch.setattr(
        run_pipeline, "_record_demo_verdict", record_new_execution
    )

    result = run_pipeline._run_relevant_revalidators(
        state, "stage_5", {"parameter-deriver"},
    )

    assert result is None
    assert calls == ["stage_5"]
    for name in run_pipeline._POST_SMOKE_DERIVED_ARTIFACTS:
        payload = (run_dir / ".pipeline" / name).read_text(encoding="utf-8")
        assert "new-" in payload
        assert "old-model" not in payload
        assert "old-checkpoint" not in payload


def test_stage5_no_fix_preserves_post_smoke_artifacts(run_dir, monkeypatch):
    import run_pipeline

    state = make_state(run_dir)
    prior = _write_post_smoke_artifacts(run_dir, identity="prior")

    def unexpected_recompute(*_args, **_kwargs):
        raise AssertionError("no-fix Stage 5 path must not recompute evidence")

    monkeypatch.setattr(
        run_pipeline, "_record_demo_verdict", unexpected_recompute
    )

    result = run_pipeline._run_relevant_revalidators(state, "stage_5", set())

    assert result is None
    for name, payload in prior.items():
        assert (run_dir / ".pipeline" / name).read_text(
            encoding="utf-8"
        ) == payload


def test_stage5_failed_resmoke_cannot_leave_pre_fix_evidence(
    fake_subprocess,
    run_dir,
    monkeypatch,
):
    import run_pipeline

    state = make_state(run_dir)
    _write_post_smoke_artifacts(run_dir, identity="old")

    fake_subprocess.expect_stage2_script(ok=True)  # validate_params_output.py
    fake_subprocess.expect_script(returncode=0)  # render_notebook.py
    fake_subprocess.expect_stage2_script(ok=True)  # validate_notebook_output.py
    fake_subprocess.expect_script(
        returncode=1, stderr="RuntimeError: fixed notebook regressed"
    )

    def unexpected_recompute(*_args, **_kwargs):
        raise AssertionError("failed smoke must not persist new evidence")

    monkeypatch.setattr(
        run_pipeline, "_record_demo_verdict", unexpected_recompute
    )

    result = run_pipeline._run_relevant_revalidators(
        state, "stage_5", {"parameter-deriver"},
    )

    assert result is not None
    assert result.status == "halted"
    assert all(
        not (run_dir / ".pipeline" / name).exists()
        for name in run_pipeline._POST_SMOKE_DERIVED_ARTIFACTS
    )


def test_rescale_param_resolution_is_idempotent_on_resume(run_dir):
    """If Stage 5 halts after applying a deterministic param resolution, resume
    must not append a duplicate assumptions.md entry or duplicate param note."""
    import run_pipeline

    state = make_state(run_dir)
    state.paths.method_spec.write_text(json.dumps({}), encoding="utf-8")
    state.paths.pipeline_dir.joinpath("params.json").write_text(
        json.dumps({
            "params": {
                "R_0": {
                    "value": 2000.0,
                    "source": "paper",
                    "paper_section": "Section 4",
                    "note": "Paper reports the raw-pixel threshold.",
                },
            },
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    ok1, msg1 = run_pipeline._apply_rescale_param_resolution(
        state, _rescale_finding(),
    )
    params_after_first = json.loads(state.paths.pipeline_dir.joinpath("params.json").read_text(
        encoding="utf-8",
    ))
    assumptions_after_first = state.paths.run_dir.joinpath(run_layout.ASSUMPTIONS_MD).read_text(
        encoding="utf-8",
    )

    ok2, msg2 = run_pipeline._apply_rescale_param_resolution(
        state, _rescale_finding(),
    )
    params_after_second = json.loads(state.paths.pipeline_dir.joinpath("params.json").read_text(
        encoding="utf-8",
    ))
    assumptions_after_second = state.paths.run_dir.joinpath(run_layout.ASSUMPTIONS_MD).read_text(
        encoding="utf-8",
    )

    assert ok1 is True
    assert msg1 == "A001"
    assert ok2 is True
    assert msg2 == "already_resolved:R_0"
    assert params_after_second == params_after_first
    assert assumptions_after_second == assumptions_after_first
    entry = params_after_second["params"]["R_0"]
    assert entry["source"] == "system_default"
    assert entry["paper_value"] == 2000.0
    assert entry["value"] == 7.843
    assert "note" not in entry
    assert entry["reasoning"].count("AUTO-RESOLVED") == 1
    assert "See assumptions.md entry A001" in entry["reasoning"]


def test_rescale_param_resolution_reverts_schema_invalid_result(run_dir):
    """The rescale applier must validate params.json and revert if provenance
    normalization cannot produce a schema-valid system_default entry."""
    import run_pipeline

    state = make_state(run_dir)
    params_path = state.paths.pipeline_dir / "params.json"
    params_path.write_text(
        json.dumps({
            "params": {
                "R_0": {
                    "value": None,
                    "source": "paper",
                    "paper_section": "Section 4",
                },
            },
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    before = params_path.read_text(encoding="utf-8")

    ok, msg = run_pipeline._apply_rescale_param_resolution(
        state, _rescale_finding(),
    )

    assert ok is False
    assert "schema-invalid params.json" in msg
    assert params_path.read_text(encoding="utf-8") == before
    assert not state.paths.run_dir.joinpath(run_layout.ASSUMPTIONS_MD).exists()


def test_rescale_param_resolution_normalizes_legacy_partial_without_duplicate_assumption(run_dir):
    """If an older Stage 5 run already changed the value but left paper
    provenance behind, resume should fix provenance without adding A002."""
    import run_pipeline

    state = make_state(run_dir)
    assumptions_path = run_layout.run_path(state.paths.run_dir, run_layout.ASSUMPTIONS_MD)
    assumptions_path.write_text(
        "# Pipeline assumptions\n\n"
        "## A001 - R_0 rescaled from 2000.0 to 7.843\n\n"
        "**Detected.** prior run\n\n",
        encoding="utf-8",
    )
    state.paths.pipeline_dir.joinpath("params.json").write_text(
        json.dumps({
            "params": {
                "R_0": {
                    "value": 7.843,
                    "source": "paper",
                    "paper_section": "Section 4",
                    "note": "AUTO-RESOLVED (A001): rescaled from 2000.0 to 7.843.",
                },
            },
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    ok, msg = run_pipeline._apply_rescale_param_resolution(
        state, _rescale_finding(),
    )
    params_after = json.loads(state.paths.pipeline_dir.joinpath("params.json").read_text(
        encoding="utf-8",
    ))

    assert ok is True
    assert msg == "already_resolved:R_0"
    entry = params_after["params"]["R_0"]
    assert entry["source"] == "system_default"
    assert entry["paper_value"] == 2000.0
    assert entry["value"] == 7.843
    assert "A002" not in assumptions_path.read_text(encoding="utf-8")


def test_relabel_param_source_resolution_logs_numeric_add_fields(run_dir):
    """Stage 2.x relabel auto-resolutions may add numeric paper_value metadata.

    The assumption logger must summarize those fields without assuming every
    `add_fields` value is a string.
    """
    import run_pipeline

    state = make_state(run_dir)
    params_path = state.paths.pipeline_dir / "params.json"
    params_path.write_text(
        json.dumps({
            "params": {
                "core_set_size": {
                    "value": 100,
                    "source": "spec_default",
                    "reasoning": "Value from the signature default.",
                },
            },
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    finding = {
        "id": "F002",
        "severity": "important",
        "description": "The paper reports N_M=5000, but the demo uses 100.",
        "proposed_resolution": {
            "kind": "relabel_param_source",
            "param": "core_set_size",
            "new_source": "system_default",
            "add_fields": {
                "paper_value": 5000,
                "paper_section": "Section 7.9, Table 6",
                "reasoning": "Use 100 at smoke scale; use 5000 at paper scale.",
            },
            "expert_reasoning": "The paper value is preserved separately from the demo value.",
            "alternative": "Use N_M=5000 for a paper-scale run.",
        },
    }

    ok, msg = run_pipeline._apply_relabel_param_source_resolution(state, finding)

    params_after = json.loads(params_path.read_text(encoding="utf-8"))
    assumptions = (state.paths.run_dir / run_layout.ASSUMPTIONS_MD).read_text(encoding="utf-8")
    entry = params_after["params"]["core_set_size"]
    assert ok is True
    assert msg == "A001"
    assert entry["source"] == "system_default"
    assert entry["value"] == 100
    assert entry["paper_value"] == 5000
    assert "paper_value=5000" in assumptions
    assert "paper_section='Section 7.9, Table 6'" in assumptions


def test_relabel_to_system_default_without_paper_value_maps_to_system_inferred(run_dir):
    """bev-distill 2026-07-02 F002: the reviewer's relabel chose
    source=system_default while removing paper_value — but system_default
    MEANS "the paper states a value we deviate from", and the finding's whole
    point was that the paper states none. The applier must map to
    system_inferred (the no-paper-value source kind) instead of
    revert-and-halt."""
    import run_pipeline

    state = make_state(run_dir)
    params_path = state.paths.pipeline_dir / "params.json"
    params_path.write_text(
        json.dumps({
            "params": {
                "temperature": {
                    "value": 1.0,
                    "source": "paper",
                    "paper_value": 1.0,
                    "note": "Paper uses T=1.0 for logit softening.",
                },
            },
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    finding = {
        "id": "F002",
        "severity": "important",
        "check_id": "paper_source_values_match_paper",
        "description": "The paper replaced logit distillation with InfoNCE "
                       "and never states tau; paper_value=1.0 is fabricated.",
        "proposed_resolution": {
            "kind": "relabel_param_source",
            "param": "temperature",
            "new_source": "system_default",
            "remove_fields": ["paper_value", "note"],
            "add_fields": {
                "reasoning": "Paper does not specify tau for InfoNCE; the "
                             "1.0 came from the pluggable signature default.",
            },
            "expert_reasoning": "The paper never states a temperature value.",
        },
    }

    ok, msg = run_pipeline._apply_relabel_param_source_resolution(state, finding)

    params_after = json.loads(params_path.read_text(encoding="utf-8"))
    assumptions = (state.paths.run_dir / run_layout.ASSUMPTIONS_MD).read_text(encoding="utf-8")
    entry = params_after["params"]["temperature"]
    assert ok is True
    assert msg == "A001"
    assert entry["source"] == "system_inferred"
    assert "paper_value" not in entry
    assert "reasoning" in entry
    assert "mapped it to 'system_inferred'" in assumptions


def test_relabel_to_system_default_with_paper_value_is_not_remapped(run_dir):
    """The remap fires ONLY when the resolution leaves no paper_value —
    a legitimate system_default relabel (paper states a value, demo uses a
    smaller one) stays system_default."""
    import run_pipeline

    state = make_state(run_dir)
    params_path = state.paths.pipeline_dir / "params.json"
    params_path.write_text(
        json.dumps({
            "params": {
                "mc_samples": {
                    "value": 100,
                    "source": "paper",
                    "paper_section": "Section 5",
                },
            },
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    finding = {
        "id": "F001",
        "severity": "important",
        "check_id": "paper_source_values_match_paper",
        "description": "Paper uses 2000 MC samples; the demo runs 100.",
        "proposed_resolution": {
            "kind": "relabel_param_source",
            "param": "mc_samples",
            "new_source": "system_default",
            "remove_fields": ["paper_section"],
            "add_fields": {
                "paper_value": 2000,
                "reasoning": "Demo-scale reduction from the paper's 2000.",
            },
            "expert_reasoning": "Paper states 2000; 100 is the smoke value.",
        },
    }

    ok, msg = run_pipeline._apply_relabel_param_source_resolution(state, finding)

    entry = json.loads(params_path.read_text(encoding="utf-8"))["params"]["mc_samples"]
    assert ok is True
    assert entry["source"] == "system_default"
    assert entry["paper_value"] == 2000


def test_relabel_without_expert_reasoning_falls_back_to_entry_reasoning(run_dir):
    """R2C-044 rider (SRL 2026-07-29 delivery): a resolution artifact missing
    `expert_reasoning` must not render an empty Reasoning field in
    assumptions.md — it falls back to the reasoning text the relabel itself
    put on the entry."""
    import run_pipeline

    state = make_state(run_dir)
    params_path = state.paths.pipeline_dir / "params.json"
    params_path.write_text(
        json.dumps({
            "params": {
                "temperature": {
                    "value": 1.0,
                    "source": "paper",
                    "paper_value": 1.0,
                    "note": "Paper uses T=1.0.",
                },
            },
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    finding = {
        "id": "F002",
        "severity": "important",
        "check_id": "paper_source_values_match_paper",
        "description": "paper_value=1.0 is not stated by the paper.",
        "proposed_resolution": {
            "kind": "relabel_param_source",
            "param": "temperature",
            "new_source": "system_inferred",
            "remove_fields": ["paper_value", "note"],
            "add_fields": {
                "reasoning": "Signature default; the paper never names tau.",
            },
        },
    }
    ok, msg = run_pipeline._apply_relabel_param_source_resolution(state, finding)

    assumptions = (state.paths.run_dir / run_layout.ASSUMPTIONS_MD).read_text(encoding="utf-8")
    assert ok is True
    assert "**Reasoning.** Signature default; the paper never names tau." in assumptions
    assert "**Reasoning.**\n" not in assumptions


def test_relabel_without_any_reasoning_renders_deterministic_sentence(run_dir):
    """R2C-044 rider, second fallback tier: a resolution carrying neither
    `expert_reasoning` nor an add_fields reasoning text renders the
    deterministic fallback sentence, never an empty Reasoning field. Relabel
    to source=paper, the one source whose schema requires no entry-level
    reasoning, so the resolution legitimately carries none anywhere."""
    import run_pipeline

    state = make_state(run_dir)
    params_path = state.paths.pipeline_dir / "params.json"
    params_path.write_text(
        json.dumps({
            "params": {
                "temperature": {
                    "value": 1.0,
                    "source": "system_inferred",
                    "reasoning": "Signature default carried from the pluggable.",
                },
            },
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    finding = {
        "id": "F003",
        "severity": "important",
        "check_id": "paper_source_values_match_paper",
        "description": "The paper states T=1.0 in Section 3.",
        "proposed_resolution": {
            "kind": "relabel_param_source",
            "param": "temperature",
            "new_source": "paper",
            "add_fields": {"paper_section": "Section 3"},
        },
    }

    ok, msg = run_pipeline._apply_relabel_param_source_resolution(state, finding)

    assumptions = (state.paths.run_dir / run_layout.ASSUMPTIONS_MD).read_text(encoding="utf-8")
    assert ok is True
    assert "carried no expert reasoning" in assumptions
    assert "**Reasoning.**\n" not in assumptions


def _unstructured_blocking_finding() -> dict:
    """The GBALD 2026-07-02 re-roll shape: relabel-worthy provenance finding
    whose reviewer wrote only prose (proposed_resolution=null)."""
    return {
        "id": "F001",
        "severity": "important",
        "check_id": "paper_source_values_match_paper",
        "target_agent": "parameter-deriver",
        "issue_type": "provenance_reasoning_inaccurate",
        "file": ".pipeline/params.json",
        "location": "params.initial_labeled",
        "description": "initial_labeled claims source=paper but GBALD "
                       "initializes via core-set construction.",
        "proposed_fix": "Relabel to source: system_inferred with reasoning.",
        "proposed_resolution": None,
        "resolution_status": "pending",
    }


def _reask_params_json(state) -> None:
    state.paths.pipeline_dir.joinpath("params.json").write_text(
        json.dumps({
            "params": {
                "initial_labeled": {
                    "value": 1000,
                    "source": "paper",
                    "paper_section": "Section 7.4",
                    "used_in_notebook": False,
                    "unused_reason": "GBALD bootstraps via core-set construction.",
                },
            },
        }, indent=2) + "\n",
        encoding="utf-8",
    )


def test_resolution_reask_adopts_structured_resolution_and_applies(run_dir, monkeypatch):
    """One reviewer re-ask turns an unstructured blocking finding into an
    applied relabel: the run continues instead of halting."""
    import run_pipeline

    state = make_state(run_dir)
    _reask_params_json(state)
    reask_path = state.paths.pipeline_dir / "stage_review_stage_2x_params_resolution.json"

    def fake_reask_dispatch(_state, reviewer_stage_id, findings):
        assert reviewer_stage_id == "stage_2x_params"
        assert [f["id"] for f in findings] == ["F001"]
        reask_path.write_text(json.dumps({
            "schema_version": "1.0.0",
            "stage_id": "stage_2x_params",
            "review_status": "issues_found",
            "summary": "resolution re-ask",
            "findings": [{
                **findings[0],
                "proposed_resolution": {
                    "kind": "relabel_param_source",
                    "param": "initial_labeled",
                    "new_source": "system_inferred",
                    "remove_fields": ["paper_section"],
                    "add_fields": {
                        "reasoning": "Derived from the paper's minimum "
                                     "core-set size, not a stated bootstrap.",
                    },
                    "expert_reasoning": "The paper has no random-bootstrap "
                                        "initial_labeled protocol.",
                },
            }],
            "checks_summary": [],
            "caveats": [],
        }, indent=2), encoding="utf-8")

    monkeypatch.setattr(
        run_pipeline, "_dispatch_stage_reviewer_resolution_reask",
        fake_reask_dispatch,
    )

    remaining, applied, failed = run_pipeline._reask_unstructured_blocking_resolutions(
        state, "stage_2x", "stage_2x_params", [_unstructured_blocking_finding()],
    )

    assert applied == ["F001"]
    assert failed == []
    assert remaining == []
    entry = json.loads(
        state.paths.pipeline_dir.joinpath("params.json").read_text(encoding="utf-8")
    )["params"]["initial_labeled"]
    assert entry["source"] == "system_inferred"
    assert "reasoning" in entry


def test_resolution_reask_id_fallback_adopts_renamed_answer(run_dir, monkeypatch):
    """ICRA21_HICA 2026-07-08 (12b layer 3): the re-ask reviewer produced a
    textbook structured relabel but filed it under its original-review
    numbering (F001) instead of the dispatched synthesized id (DEF-...).
    The merge must adopt it via the unique (check_id, location) fallback
    instead of halting a run that is holding its own fix."""
    import run_pipeline

    state = make_state(run_dir)
    _reask_params_json(state)
    reask_path = state.paths.pipeline_dir / "stage_review_stage_2x_params_resolution.json"
    dispatched = {**_unstructured_blocking_finding(), "id": "DEF-initial_labeled"}

    def fake_reask_dispatch(_state, reviewer_stage_id, findings):
        assert [f["id"] for f in findings] == ["DEF-initial_labeled"]
        renamed = {
            **findings[0],
            "id": "F001",  # the reviewer's renumbering — the layer-3 defect
            "proposed_resolution": {
                "kind": "relabel_param_source",
                "param": "initial_labeled",
                "new_source": "system_inferred",
                "remove_fields": ["paper_section"],
                "add_fields": {
                    "reasoning": "Derived from the paper's minimum core-set "
                                 "size, not a stated bootstrap.",
                },
                "expert_reasoning": "No random-bootstrap protocol in paper.",
            },
        }
        reask_path.write_text(json.dumps({
            "schema_version": "1.0.0",
            "stage_id": "stage_2x_params",
            "review_status": "issues_found",
            "summary": "resolution re-ask",
            "findings": [renamed],
            "checks_summary": [],
            "caveats": ["This resolution addresses the DEF-initial_labeled finding."],
        }, indent=2), encoding="utf-8")

    monkeypatch.setattr(
        run_pipeline, "_dispatch_stage_reviewer_resolution_reask",
        fake_reask_dispatch,
    )

    remaining, applied, failed = run_pipeline._reask_unstructured_blocking_resolutions(
        state, "stage_2x", "stage_2x_params", [dispatched],
    )

    assert applied == ["DEF-initial_labeled"]
    assert failed == []
    assert remaining == []
    entry = json.loads(
        state.paths.pipeline_dir.joinpath("params.json").read_text(encoding="utf-8")
    )["params"]["initial_labeled"]
    assert entry["source"] == "system_inferred"


def test_resolution_reask_id_fallback_requires_two_way_uniqueness(run_dir, monkeypatch):
    """Guard rail on the layer-3 fallback: when two unmatched candidates
    share the same (check_id, location), an id-renamed answer is ambiguous
    and must NOT be adopted — the halt proceeds exactly as before."""
    import run_pipeline

    state = make_state(run_dir)
    _reask_params_json(state)
    reask_path = state.paths.pipeline_dir / "stage_review_stage_2x_params_resolution.json"
    cand_a = {**_unstructured_blocking_finding(), "id": "DEF-a"}
    cand_b = {**_unstructured_blocking_finding(), "id": "DEF-b"}

    def fake_reask_dispatch(_state, reviewer_stage_id, findings):
        reask_path.write_text(json.dumps({
            "schema_version": "1.0.0",
            "stage_id": "stage_2x_params",
            "review_status": "issues_found",
            "summary": "resolution re-ask",
            "findings": [{
                **findings[0],
                "id": "F001",
                "proposed_resolution": {
                    "kind": "relabel_param_source",
                    "param": "initial_labeled",
                    "new_source": "system_inferred",
                    "remove_fields": [],
                    "add_fields": {"reasoning": "x"},
                    "expert_reasoning": "y",
                },
            }],
            "checks_summary": [],
            "caveats": [],
        }, indent=2), encoding="utf-8")

    monkeypatch.setattr(
        run_pipeline, "_dispatch_stage_reviewer_resolution_reask",
        fake_reask_dispatch,
    )

    remaining, applied, failed = run_pipeline._reask_unstructured_blocking_resolutions(
        state, "stage_2x", "stage_2x_params", [cand_a, cand_b],
    )

    assert applied == []
    assert failed == []
    assert remaining == [cand_a, cand_b]


def test_resolution_reask_respects_needs_user(run_dir, monkeypatch):
    """A re-ask that answers needs_user keeps the finding blocking — the halt
    becomes a deliberate reviewer decision, not an omission."""
    import run_pipeline

    state = make_state(run_dir)
    _reask_params_json(state)
    reask_path = state.paths.pipeline_dir / "stage_review_stage_2x_params_resolution.json"

    def fake_reask_dispatch(_state, _reviewer_stage_id, findings):
        reask_path.write_text(json.dumps({
            "schema_version": "1.0.0",
            "stage_id": "stage_2x_params",
            "review_status": "issues_found",
            "summary": "resolution re-ask",
            "findings": [{**findings[0], "resolution_status": "needs_user"}],
            "checks_summary": [],
            "caveats": [],
        }, indent=2), encoding="utf-8")

    monkeypatch.setattr(
        run_pipeline, "_dispatch_stage_reviewer_resolution_reask",
        fake_reask_dispatch,
    )

    remaining, applied, failed = run_pipeline._reask_unstructured_blocking_resolutions(
        state, "stage_2x", "stage_2x_params", [_unstructured_blocking_finding()],
    )

    assert applied == []
    assert failed == []
    assert len(remaining) == 1
    assert remaining[0]["resolution_status"] == "needs_user"


def test_resolution_reask_dispatch_failure_falls_through_to_halt(run_dir, monkeypatch):
    """The re-ask is best-effort: a dispatch failure returns the findings
    unchanged so the caller halts exactly as before."""
    import run_pipeline

    state = make_state(run_dir)
    _reask_params_json(state)

    def fake_reask_dispatch(_state, _reviewer_stage_id, _findings):
        raise run_pipeline.OpencodeClientError("dispatch POST timed out")

    monkeypatch.setattr(
        run_pipeline, "_dispatch_stage_reviewer_resolution_reask",
        fake_reask_dispatch,
    )

    findings = [_unstructured_blocking_finding()]
    remaining, applied, failed = run_pipeline._reask_unstructured_blocking_resolutions(
        state, "stage_2x", "stage_2x_params", findings,
    )

    assert (remaining, applied, failed) == (findings, [], [])


def test_resolution_reask_skips_when_all_blocking_findings_are_structured(run_dir, monkeypatch):
    """No candidates → no dispatch. Findings that already carry a structured
    resolution (applied or failed earlier) and explicit needs_user findings
    never trigger the re-ask."""
    import run_pipeline

    state = make_state(run_dir)

    def fail_if_dispatched(*_args, **_kwargs):
        raise AssertionError("re-ask dispatched with no candidates")

    monkeypatch.setattr(
        run_pipeline, "_dispatch_stage_reviewer_resolution_reask",
        fail_if_dispatched,
    )

    findings = [
        {**_unstructured_blocking_finding(), "id": "F001",
         "resolution_status": "needs_user"},
        {**_unstructured_blocking_finding(), "id": "F002",
         "proposed_resolution": {"kind": "relabel_param_source"}},
        {**_unstructured_blocking_finding(), "id": "F003",
         "severity": "nice-to-have"},
    ]
    remaining, applied, failed = run_pipeline._reask_unstructured_blocking_resolutions(
        state, "stage_2x", "stage_2x_params", findings,
    )

    assert (remaining, applied, failed) == (findings, [], [])


def test_stage5_logs_completed_stage_review_nice_findings_and_caveats(run_dir):
    """Final paper-fidelity routing stays authoritative, but completed
    stage-review nice-to-have findings and caveats remain visible."""
    import run_pipeline

    state = make_state(run_dir)
    state.paths.pipeline_dir.joinpath("review_report.json").write_text(
        json.dumps({"findings": []}, indent=2) + "\n",
        encoding="utf-8",
    )
    state.paths.pipeline_dir.joinpath("stage_review_stage_2x_params.json").write_text(
        json.dumps({
            "stage_id": "stage_2x_params",
            "findings": [
                {
                    "id": "SR-NICE-001",
                    "severity": "nice-to-have",
                    "target_agent": "human",
                    "file": ".pipeline/params.json",
                    "description": "mc_samples uses the approved demo value.",
                    "proposed_fix": "Document the paper-scale value beside the demo value.",
                },
                {
                    "id": "SR-IMPORTANT-001",
                    "severity": "important",
                    "description": "Should not be included by this helper.",
                },
            ],
            "caveats": ["eta is exposed for audit but not used by the demo."],
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    result = run_pipeline.run_stage_5(state)

    deferred = state.paths.run_dir.joinpath(run_layout.DEFERRED_FINDINGS_MD).read_text(
        encoding="utf-8",
    )
    assert result.status == "completed"
    assert "Completed stage-review nice-to-have findings" in deferred
    assert "mc_samples uses the approved demo value" in deferred
    assert "Completed stage-review caveats" in deferred
    assert "eta is exposed for audit" in deferred
    assert "Should not be included by this helper" not in deferred


def test_stage5_method_touching_fix_refinalizes_package(
    fake_subprocess, run_dir,
):
    """Item 26 (ms3d overnight 07-08): a stage-5 auto-routed fix that adds a
    public symbol to method/ must re-run the idempotent init finalizer + the
    import validator, or the delivered package raises ImportError on its own
    export (__all__ was finalized back at 2d)."""
    import run_pipeline

    state = make_state(run_dir)

    fake_subprocess.expect_stage2_script(ok=True)  # validate_method_coder_output.py
    fake_subprocess.expect_script(returncode=0)  # finalize_package_init.py
    fake_subprocess.expect_stage2_script(ok=True)  # validate_package_imports.py
    fake_subprocess.expect_script(returncode=0)  # render_notebook.py
    fake_subprocess.expect_stage2_script(ok=True)  # validate_notebook_output.py
    fake_subprocess.expect_script(returncode=0)  # smoke_run_notebook.py

    result = run_pipeline._run_relevant_revalidators(
        state, "stage_5", {"method-coder"},
    )

    assert result is None
    assert fake_subprocess.stage2_calls[0]["args"] == [
        "scripts/validate_method_coder_output.py",
    ]
    assert fake_subprocess.script_calls[0]["args"][0] == \
        "scripts/finalize_package_init.py"
    assert fake_subprocess.stage2_calls[1]["args"] == [
        "scripts/validate_package_imports.py",
    ]
    assert fake_subprocess.script_calls[1]["args"][0] == \
        "scripts/render_notebook.py"


def test_stage5_import_regression_after_fix_halts_honestly(
    fake_subprocess, run_dir,
):
    """When the re-finalized package still fails the import validator, the
    stage halts as an invalid producer output rather than shipping a package
    that cannot import its own exports."""
    import run_pipeline

    state = make_state(run_dir)

    fake_subprocess.expect_stage2_script(ok=True)  # validate_method_coder_output.py
    fake_subprocess.expect_script(returncode=0)  # finalize_package_init.py
    fake_subprocess.expect_stage2_script(
        ok=False, err="ImportError: cannot import name 'multi_round_self_train'",
    )

    result = run_pipeline._run_relevant_revalidators(
        state, "stage_5", {"method-coder"},
    )

    assert result is not None and result.status == "halted"
    assert "validate_package_imports.py failed after Stage 5 fix-mode" in \
        result.halt_artifact["reason"]
    assert result.halt_artifact["halt_class"] == "producer_output_invalid"


def test_relabel_to_paper_drops_stale_reasoning(run_dir):
    """fedavg C 2026-07-21: a relabel to source=paper carried paper_section
    and note in add_fields but never touched `reasoning`, leaving the OLD
    source's justification ("value from the signature default ...") on an
    entry now labelled paper-stated. On a source change to `paper` (the one
    source whose schema does not require reasoning), unreplaced reasoning
    is dropped."""
    import run_pipeline

    state = make_state(run_dir)
    params_path = state.paths.pipeline_dir / "params.json"
    params_path.write_text(
        json.dumps({
            "params": {
                "C": {
                    "value": 0.1,
                    "source": "spec_default",
                    "reasoning": "Value from the pluggable_component "
                                 "signature default (C=0.1).",
                },
            },
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    finding = {
        "id": "F003",
        "severity": "important",
        "check_id": "FL-params-hyperparameter-provenance",
        "description": "C=0.1 is paper-stated but labelled spec_default.",
        "proposed_resolution": {
            "kind": "relabel_param_source",
            "param": "C",
            "new_source": "paper",
            "add_fields": {
                "paper_section": "Section 3 (Increasing parallelism)",
                "note": "Paper states 'we fix C=0.1'.",
            },
            "expert_reasoning": "The paper explicitly fixes C=0.1.",
        },
    }

    ok, _ = run_pipeline._apply_relabel_param_source_resolution(state, finding)

    entry = json.loads(params_path.read_text(encoding="utf-8"))["params"]["C"]
    assert ok is True
    assert entry["source"] == "paper"
    assert entry["paper_section"] == "Section 3 (Increasing parallelism)"
    assert "reasoning" not in entry


def test_relabel_to_paper_keeps_reasoning_the_reviewer_replaced(run_dir):
    """When add_fields carries a fresh `reasoning`, it is the reviewer's
    replacement text and must survive the stale-reasoning guard."""
    import run_pipeline

    state = make_state(run_dir)
    params_path = state.paths.pipeline_dir / "params.json"
    params_path.write_text(
        json.dumps({
            "params": {
                "C": {
                    "value": 0.1,
                    "source": "spec_default",
                    "reasoning": "Old signature-default justification.",
                },
            },
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    finding = {
        "id": "F003",
        "severity": "important",
        "proposed_resolution": {
            "kind": "relabel_param_source",
            "param": "C",
            "new_source": "paper",
            "add_fields": {
                "note": "Paper states 'we fix C=0.1'.",
                "reasoning": "Replacement text from the reviewer.",
            },
            "expert_reasoning": "x",
        },
    }

    ok, _ = run_pipeline._apply_relabel_param_source_resolution(state, finding)

    entry = json.loads(params_path.read_text(encoding="utf-8"))["params"]["C"]
    assert ok is True
    assert entry["reasoning"] == "Replacement text from the reviewer."
