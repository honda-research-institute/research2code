"""B-07 step 3: run_pipeline's --self-test body, migrated to pytest.

The 170 asserts are carried verbatim, split into per-surface test
functions, per the maintainer's 2026-08-18 decision to retire the flag. The rev-2
migration conditions applied here:

- Every `globals()["..."] = fake` self-patching site became
  `monkeypatch.setattr(run_pipeline, ...)` — a verbatim copy would have
  installed the fakes in THIS module's namespace and let the code under
  test call its real globals (the red team's repro, including a live
  POST into a production opencode server).
- The asserts silently gated on gitignored r2c_runs/ artifacts
  (`if <path>.exists():` — invisible permanent skips) are explicit
  manual_only tests that pytest.skip when the live fleet is absent.
- An autouse network guard fails any test in this module that reaches
  the real HTTP transport.

Underscore names are referenced module-qualified (`run_pipeline._x`) on
purpose: it is the explicit import surface and the monkeypatch target in
one spelling.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest

import opencode_client
import run_layout
import run_pipeline
from dispatch_templates import (
    STAGE_TASK_SUMMARIES,
    WRITEABLE_PATHS,
    build_halt_artifact,
)
from route_findings import critical_findings
from run_pipeline import (
    OutOfScopeWritesError,
    PipelinePaths,
    PipelineState,
    build_paths_block,
    degrade,
    finalize_assumptions_md,
    finalize_run_report,
    run_fix_loop,
    stage_heading,
    stage_label,
    taxonomy_node_ref_from_spec,
)

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _no_real_transport(monkeypatch):
    """Network guard: nothing in this module may reach the real HTTP
    transport. The original self-test's fire-and-forget todo test spawned a
    thread that would have POSTed into any live opencode server on the
    default port had a patch site missed."""
    def _blocked(*a, **k):
        raise AssertionError(
            "real opencode_client._http reached from the driver self-test "
            "module — a patch site is missing")
    monkeypatch.setattr(opencode_client, "_http", _blocked)


def _paths_for(root: Path, slug: str = "selftest") -> PipelinePaths:
    return PipelinePaths.from_setup_result({
        "repo_root": str(root),
        "input_path": str(root / "paper.md"),
        "input_kind": "markdown",
        "slug": slug,
        "run_dir": str(root),
        "pipeline_dir": str(root / ".pipeline"),
        "paper_md_path": str(root / ".pipeline" / "paper.md"),
    })


def test_pipeline_paths_from_setup_result_roundtrip():
    # Synthetic setup_result (not a real r2c_runs/ fixture): r2c_runs/ is
    # gitignored and routinely wiped (R2C 2026-05-27 — a wipe broke this
    # assert). Exercises from_setup_result's path derivation only.
    _bal = REPO / "r2c_runs" / "bayesian-active-learning"
    paths = PipelinePaths.from_setup_result({
        "repo_root": str(REPO),
        "input_path": str(REPO / "input_papers" / "bayesian-active-learning.md"),
        "input_kind": "markdown",
        "slug": "bayesian-active-learning",
        "run_dir": str(_bal),
        "pipeline_dir": str(_bal / ".pipeline"),
        "paper_md_path": str(_bal / ".pipeline" / "paper.md"),
    })
    assert paths.slug == "bayesian-active-learning"
    assert paths.paper_md.name == "paper.md"
    assert paths.method_spec.name == "method_spec.json"
    assert paths.paper_map.name == "paper_map.json"
    assert paths.feasibility_halt.name == "feasibility_gate.json.halt"
    assert paths.paper_map_halt.name == "paper_map.json.halt"
    assert paths.method_spec_halt.name == "method_spec.json.halt"


def test_build_paths_block_placeholder_fallback():
    synth_paths = PipelinePaths.from_setup_result({
        "repo_root": "/tmp/r2c-driver-selftest-nonexistent",
        "input_path": "/tmp/r2c-driver-selftest-nonexistent/paper.md",
        "input_kind": "markdown",
        "slug": "selftest",
        "run_dir": "/tmp/r2c-driver-selftest-nonexistent/run",
        "pipeline_dir": "/tmp/r2c-driver-selftest-nonexistent/run/.pipeline",
        "paper_md_path": "/tmp/r2c-driver-selftest-nonexistent/run/.pipeline/paper.md",
    })
    block = build_paths_block(synth_paths)
    assert block.taxonomy_source == "(analyzer determines from paper)"
    rendered = block.to_block()
    assert "Spec:" in rendered and "method_spec.json" in rendered


@pytest.mark.manual_only
def test_taxonomy_node_ref_on_live_spec():
    spec_in_repo = (REPO / "r2c_runs" / "bayesian-active-learning"
                    / ".pipeline" / "method_spec.json")
    if not spec_in_repo.exists():
        pytest.skip("live fleet absent")
    paths_live = PipelinePaths.from_setup_result({
        "repo_root": str(REPO),
        "input_path": str(spec_in_repo.parent / "paper.md"),
        "input_kind": "markdown",
        "slug": "bayesian-active-learning",
        "run_dir": str(spec_in_repo.parent.parent),
        "pipeline_dir": str(spec_in_repo.parent),
        "paper_md_path": str(spec_in_repo.parent / "paper.md"),
    })
    if run_pipeline._classification_from_spec(paths_live):
        node_ref = taxonomy_node_ref_from_spec(paths_live)
        if node_ref:
            assert "docs/ssot/taxonomies.yaml#" in node_ref, \
                f"unexpected taxonomy node ref: {node_ref!r}"
            assert build_paths_block(paths_live).taxonomy_source == node_ref
        else:
            assert build_paths_block(paths_live).taxonomy_source == \
                "(analyzer determines from paper)"


@pytest.mark.manual_only
def test_critical_finding_routing_on_passing_review_fixture():
    review_path = (REPO / "r2c_runs" / "bayesian-active-learning"
                   / ".pipeline" / "stage_review_stage_1_analyzer.json")
    if not review_path.exists():
        pytest.skip("live fleet absent")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    findings = review.get("findings", []) or []
    assert critical_findings(findings) == [], "expected no critical findings"


@pytest.mark.manual_only
def test_critical_finding_routing_on_halt_fixture():
    # All historical bev-distill 2.x findings are 'important', so
    # critical_findings returns [] — "important" alone must not trip a
    # halt-on-critical path.
    halt_fixture = (REPO / "r2c_runs" / "bev-distill" / ".pipeline"
                    / "stage_2x_params.halt")
    if not halt_fixture.exists():
        pytest.skip("live fleet absent")
    artifact = json.loads(halt_fixture.read_text(encoding="utf-8"))
    findings = artifact.get("findings", []) or []
    assert len(findings) == 3
    assert critical_findings(findings) == []


def test_build_halt_artifact_shape():
    artifact = build_halt_artifact(
        stage="stage_1",
        reason="test",
        retry_count=1,
        findings=[{"id": "F001", "severity": "critical"}],
    )
    json.dumps(artifact)
    assert artifact["status"] == "halted"
    assert artifact["stage"] == "stage_1"
    assert artifact["retry_count"] == 1


def test_stderr_to_finding_shape():
    finding = run_pipeline._stderr_to_finding(
        "  - elements.1.source_text: Field required [missing]",
        "validate_paper_map.py",
    )
    assert finding["severity"] == "critical"
    assert "validate_paper_map.py" in finding["description"]
    assert "elements.1.source_text" in finding["description"]
    assert finding["proposed_fix"]  # non-empty


def test_file_ownership_enforcement(tmp_path):
    # Snapshot + diff + driver-managed exemptions (the bev-distill class:
    # an agent edits files outside its scope).
    (tmp_path / "method").mkdir()
    (tmp_path / ".pipeline").mkdir()
    model_py = tmp_path / "method" / "model.py"
    data_py = tmp_path / "method" / "data.py"
    model_py.write_text("# initial\n")
    data_py.write_text("# initial\n")
    snap = run_pipeline._snapshot_run_dir(tmp_path)
    assert sorted(snap) == ["method/data.py", "method/model.py"], snap

    # Untouched files → no violations.
    v = run_pipeline._detect_out_of_scope_writes(tmp_path, snap, ["method/model.py"])
    assert v == [], f"untouched files must not violate: {v}"

    # Touching an out-of-scope file → flagged.
    time.sleep(0.05)
    data_py.write_text("# touched\n")
    v = run_pipeline._detect_out_of_scope_writes(tmp_path, snap, ["method/model.py"])
    assert v == ["method/data.py"], v

    # Glob patterns in the allowlist match correctly.
    snap2 = run_pipeline._snapshot_run_dir(tmp_path)
    (tmp_path / ".pipeline" / "stage_review_stage_2b_architecture.json").write_text("{}")
    v = run_pipeline._detect_out_of_scope_writes(
        tmp_path, snap2, [".pipeline/stage_review_*.json"])
    assert v == [], f"glob should match stage_review_*.json: {v}"

    # Driver-managed paths are exempt even when not in the allowlist —
    # including the judge decision scratch (the 2026-06-18 stage_2d halt:
    # the judge's decision part was mis-attributed to the arch-coder).
    snap3 = run_pipeline._snapshot_run_dir(tmp_path)
    (tmp_path / ".pipeline" / "stage_2c.halt").write_text("{}")
    (tmp_path / ".pipeline" / "driver_state.json").write_text("{}")
    (tmp_path / ".pipeline" / "progress.json").write_text("{}")
    (tmp_path / ".pipeline" / "judge_decision_parts").mkdir(exist_ok=True)
    (tmp_path / ".pipeline" / "judge_decision_parts"
     / "stage_2d__iter_0__validate_arch_contract.py.json").write_text("{}")
    v = run_pipeline._detect_out_of_scope_writes(tmp_path, snap3, [])
    assert v == [], f"driver-managed files must be exempt: {v}"
    assert run_pipeline._is_driver_managed(".pipeline/stage_2c.halt") is True
    assert run_pipeline._is_driver_managed(".pipeline/driver_state.json") is True
    assert run_pipeline._is_driver_managed(".pipeline/progress.json") is True
    assert run_pipeline._is_driver_managed(".pipeline/stage_2d.complete") is True
    assert run_pipeline._is_driver_managed(
        ".pipeline/judge_decision_parts/stage_2d__iter_0__validate_arch_contract.py.json"
    ) is True
    assert run_pipeline._is_driver_managed("method/model.py") is False

    # New files outside the allowlist are flagged (bev-distill case:
    # notebook-generator created/modified method/model.py).
    snap4 = run_pipeline._snapshot_run_dir(tmp_path)
    (tmp_path / "method" / "sneaky.py").write_text("# new\n")
    v = run_pipeline._detect_out_of_scope_writes(
        tmp_path, snap4, [".pipeline/notebook_draft.py"])
    assert v == ["method/sneaky.py"], v


def test_writeable_paths_covers_every_scoped_agent():
    expected = {
        "r2c-decomposer", "r2c-method-analyzer",
        "r2c-architecture-coder", "r2c-method-coder",
        "r2c-notebook-generator",
        "r2c-stage-reviewer", "r2c-paper-fidelity-reviewer",
    }
    assert expected.issubset(WRITEABLE_PATHS.keys()), (
        f"agents missing from WRITEABLE_PATHS: "
        f"{expected - WRITEABLE_PATHS.keys()}")


def test_content_snapshot_revert_and_method_py_valid(tmp_path):
    (tmp_path / "method").mkdir()
    (tmp_path / ".pipeline").mkdir()
    pre_existing = tmp_path / "method" / "training.py"
    pre_existing.write_text("# original training.py content\n")
    snap_content = run_pipeline._snapshot_run_dir_content(tmp_path)
    assert "method/training.py" in snap_content
    assert snap_content["method/training.py"] == b"# original training.py content\n"

    # Simulate the diagnostician drift: modify pre-existing + create new.
    pre_existing.write_text("# CORRUPTED by drifting agent\n")
    new_file = tmp_path / ".pipeline" / "notebook_draft.py"
    new_file.write_text("# created by drifting agent\n")

    restored, deleted = run_pipeline._revert_out_of_scope_writes(
        tmp_path,
        violations=["method/training.py", ".pipeline/notebook_draft.py"],
        snapshot_content=snap_content,
    )
    assert restored == ["method/training.py"]
    assert pre_existing.read_text() == "# original training.py content\n", \
        "revert must restore pre-dispatch content"
    assert deleted == [".pipeline/notebook_draft.py"]
    assert not new_file.exists(), "new file created by drifting agent must be removed"

    # B-004 (pdwa 2026-05-27): _method_py_valid is the method-coder's
    # recovery check.
    mp_paths = _paths_for(tmp_path, slug="selftest-b004")
    mp_state = PipelineState(session_id="ses_test", port=0, paths=mp_paths)
    method_py = tmp_path / "method" / "method.py"
    method_py.unlink(missing_ok=True)
    assert run_pipeline._method_py_valid(mp_state) is False, "no method.py → invalid"
    mp_paths.method_spec.write_text(
        json.dumps({"comparison": {"pluggable_component": {"name": "plan"}}}),
        encoding="utf-8",
    )
    method_py.write_text("def plan(start, goal):\n    return None\n", encoding="utf-8")
    assert run_pipeline._method_py_valid(mp_state) is True, \
        "parses + defines pluggable `plan` → valid"
    method_py.write_text("def plan(:\n", encoding="utf-8")
    assert run_pipeline._method_py_valid(mp_state) is False, "syntax error → invalid"
    method_py.write_text("def helper():\n    return 1\n", encoding="utf-8")
    assert run_pipeline._method_py_valid(mp_state) is False, \
        "missing the spec's pluggable function → invalid"


def test_finalize_assumptions_md(tmp_path):
    (tmp_path / ".pipeline").mkdir()
    astate = PipelineState(session_id="ses_t", port=0,
                           paths=_paths_for(tmp_path, slug="selftest-assum"))
    # Append chronologically: auto (A001), needs-user (A002), auto (A003).
    run_pipeline._append_assumption(
        astate, aid="A001", title="rescaled R_0 for normalized pixels",
        detected="d", action="a", reasoning="r", alternative=None, override="o")
    run_pipeline._append_needs_user(
        astate, {"id": "F012", "description": "eta scale ambiguity",
                 "file": "method/method.py", "location": "L42"})
    run_pipeline._append_assumption(
        astate, aid="A003", title="relabelled batch_size source",
        detected="d", action="a", reasoning="r", alternative=None, override="o")
    finalize_assumptions_md(astate)
    out = (tmp_path / run_layout.ASSUMPTIONS_MD).read_text(encoding="utf-8")
    assert "## Summary" in out, "TOC section must be present"
    assert "🔴" in out and "🟢" in out, "severity icons must appear"
    assert "jump to the 🔴 entries first" in out, "how-to-use intro must be present"
    # NEEDS-USER (A002) must sort before the auto-applied entries in the body.
    body = out.split("## Summary", 1)[1].split("---", 1)[1]
    assert body.index("## A002") < body.index("## A001"), "NEEDS-USER must come first in body"
    assert body.index("## A002") < body.index("## A003")
    # Idempotent: a second pass reproduces identical content.
    finalize_assumptions_md(astate)
    assert (tmp_path / run_layout.ASSUMPTIONS_MD).read_text(encoding="utf-8") == out, \
        "finalize must be idempotent"
    # ID allocation still works after finalize (next is A004, not a re-use).
    assert run_pipeline._next_assumption_id(astate) == "A004"


def test_degrade_and_finalize_run_report(tmp_path):
    (tmp_path / ".pipeline").mkdir()
    dstate = PipelineState(session_id="s", port=0,
                           paths=_paths_for(tmp_path, slug="selftest-degrade"))
    res = degrade(dstate.paths, "stage_2d",
                  reason="validate_arch_contract.py failed after cap",
                  what_failed="arch_contract validation could not be satisfied",
                  where="arch_contract.json",
                  what_to_do="review the contract / report to engineering")
    assert res.status == "degraded", res.status
    assert (tmp_path / "stage_2d.complete").exists() is False  # no clean sentinel
    ki = (tmp_path / run_layout.KNOWN_ISSUES_MD).read_text(encoding="utf-8")
    assert "stage_2d" in ki and "arch_contract validation could not be satisfied" in ki
    assert "What to do" in ki and "never edits your code" in ki  # intro present
    # A second degrade appends (does not overwrite).
    degrade(dstate.paths, "stage_3c", reason="smoke gate failed after cap",
            what_failed="notebook does not run end-to-end",
            where="notebook.ipynb cell 7",
            what_to_do="fix cell 7 or hand to your AI assistant")
    ki2 = (tmp_path / run_layout.KNOWN_ISSUES_MD).read_text(encoding="utf-8")
    assert len(re.findall(r"(?m)^## ", ki2)) == 2, "two issues logged"
    # finalize prepends a banner naming the issue count.
    run_layout.run_path(tmp_path, run_layout.PACKAGE_README).write_text(
        "# pdwa\n\noriginal readme\n", encoding="utf-8")
    finalize_run_report(dstate)
    readme = (tmp_path / run_layout.PACKAGE_README).read_text(encoding="utf-8")
    assert readme.startswith("> **Known issues:**"), "banner must lead the README"
    assert "2 known issues" in readme
    assert "REPORT.md" in readme
    assert "original readme" in readme, "must preserve existing README content"
    # Idempotent: a second finalize does not double-banner.
    finalize_run_report(dstate)
    assert (tmp_path / run_layout.PACKAGE_README).read_text(
        encoding="utf-8").count("This package has") == 1
    # Resolved on resume: KNOWN_ISSUES.md pruned away → banner stripped.
    (tmp_path / run_layout.KNOWN_ISSUES_MD).unlink()
    finalize_run_report(dstate)
    readme_resolved = (tmp_path / run_layout.PACKAGE_README).read_text(encoding="utf-8")
    assert "Known issues" not in readme_resolved, \
        "stale banner must be stripped once issues are resolved"
    assert "KNOWN_ISSUES.md" not in readme_resolved, "no dangling reference may survive"
    assert "original readme" in readme_resolved, "real README content is preserved"


def test_snapshot_skips_files_over_size_cap(tmp_path):
    small = tmp_path / "small.txt"
    small.write_bytes(b"x" * 100)
    big = tmp_path / "big.bin"
    big.write_bytes(b"y" * (run_pipeline._CONTENT_SNAPSHOT_MAX_BYTES + 1))
    snap = run_pipeline._snapshot_run_dir_content(tmp_path)
    assert "small.txt" in snap
    assert "big.bin" not in snap, "files over the size cap must be skipped"


def test_out_of_scope_writes_error_context():
    err = OutOfScopeWritesError(
        "r2c-notebook-generator",
        violations=["method/model.py"],
        allowed=[".pipeline/notebook_draft.py"],
    )
    assert err.agent == "r2c-notebook-generator"
    assert err.violations == ["method/model.py"]
    assert err.allowed == [".pipeline/notebook_draft.py"]
    assert "outside its allowlist" in str(err)
    assert "method/model.py" in str(err)


def test_stage_display_names():
    assert stage_label("stage_2b") == "Stage 2.b - Architecture & Training"
    assert stage_heading("stage_2b") == (
        "Stage 2.b - Architecture & Training (`stage_2b`)")
    assert stage_label("stage_not_real") == "stage_not_real"


def test_halt_notices():
    hp = Path("/tmp/run/.pipeline/stage_1.halt")
    raw_reason = "halt-judge wrote stage_1a but prompt sent stage_1; no decision matches"
    # Researcher-actionable: user_message leads; technical reason collapsible.
    notice = run_pipeline._render_halt_notice(
        "stage_1", raw_reason, hp, run_pipeline._HALT_MSG_NOT_FEASIBLE)
    assert "🛑" in notice and "stage_1" in notice
    assert "Stage 1 - Paper Decomposition & Method Analysis" in notice
    assert run_pipeline._HALT_MSG_NOT_FEASIBLE in notice, \
        "user_message must be rendered prominently"
    assert "internal pipeline error" not in notice, \
        "actionable halt must NOT use the stock message"
    assert "<details>" in notice and raw_reason in notice, \
        "raw reason goes behind the collapsible"
    assert notice.index(run_pipeline._HALT_MSG_NOT_FEASIBLE) < notice.index("<details>"), \
        "message before detail"
    # Engineering-actionable (no user_message): stock notice.
    eng = run_pipeline._render_halt_notice("stage_2d", raw_reason, hp, None)
    assert "internal pipeline error" in eng and "report" in eng.lower()
    assert raw_reason in eng and "<details>" in eng, \
        "raw reason still available to engineering"
    # build_halt_artifact carries user_message only when provided.
    assert build_halt_artifact(stage="stage_1", reason="x",
                               user_message="hi")["user_message"] == "hi"
    assert "user_message" not in build_halt_artifact(stage="stage_1", reason="x")
    # Judge-decided DEGRADE entries quote the assessment inline.
    jmsg = run_pipeline._judge_halt_user_message(
        "spec contradicts paper at Eq 3;\n  producer cannot fix")
    assert "Eq 3" in jmsg and "re-run" in jmsg
    assert "\n" not in jmsg, "rationale must be flattened to one line"
    # Judge-decided HALTS lead with the catalog story (item 12 rule 3).
    jrationale = "most likely due to another 900s dispatch timeout"
    jnotice = run_pipeline._render_halt_notice(
        "stage_2c", f"halt-judge decided to halt: {jrationale}", hp, None,
        halt_class="judge_halt")
    assert "internal pipeline error" not in jnotice, \
        "judge halt renders its class story, not stock"
    assert "automated reviewer" in jnotice, "judge_halt story leads"
    assert jnotice.index(jrationale) > jnotice.index("<details>"), \
        "rationale stays behind the collapsible"


def test_extract_failing_cell_source():
    # Markdown cells don't count in nbclient's code-cell index.
    fake_nb = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Title\n"]},
            {"cell_type": "code", "source": ["x = 1\n", "y = 2\n"]},
            {"cell_type": "markdown", "source": ["## §3\n"]},
            {"cell_type": "code", "source": ["import torch\n"]},
            {"cell_type": "code", "source": ["boxes = torch.randn(5, 9)\n"]},
        ]
    }
    assert run_pipeline._extract_failing_cell_source(fake_nb, 0) == "x = 1\ny = 2\n"
    assert run_pipeline._extract_failing_cell_source(fake_nb, 1) == "import torch\n"
    assert run_pipeline._extract_failing_cell_source(fake_nb, 2) == "boxes = torch.randn(5, 9)\n"
    assert run_pipeline._extract_failing_cell_source(fake_nb, 99) == "", \
        "out-of-range index returns empty"


def test_smoke_target_agent_to_producer_short():
    f = run_pipeline._smoke_target_agent_to_producer_short
    assert f("r2c-method-coder") == "method-coder"
    assert f("r2c-architecture-coder") == "architecture-coder"
    assert f("r2c-notebook-generator") == "notebook-generator"
    with pytest.raises(ValueError):
        f("r2c-method-analyzer")


_VALID_TRACE = [
    "method/method.py:175 — error site (IndexError on boxes[:, 8])",
    "method/method.py: boxes parameter ← caller in compute_quality_scores",
    "notebook §3 cell 22: teacher_boxes = torch.randn(B, 5, 9) — derivation site",
]


def test_read_smoke_diagnosis(tmp_path):
    diag_file = tmp_path / "smoke_diagnosis.json"

    class _DiagPaths:
        pipeline_dir = tmp_path

    class _DiagState:
        paths = _DiagPaths()

    read = run_pipeline._read_smoke_diagnosis

    # Missing file → error
    diag, err = read(_DiagState())
    assert diag is None and err and "did not write" in err

    # Malformed JSON → error
    diag_file.write_text("{not json")
    diag, err = read(_DiagState())
    assert diag is None and err and "not valid JSON" in err

    # Valid schema, target_file in target_agent's writeable_paths → success
    diag_file.write_text(json.dumps({
        "schema_version": "1.0.0",
        "target_agent": "r2c-method-coder",
        "target_file": "method/method.py",
        "root_cause": "compute_3d_box_iou indexes column 8 but boxes have 8 columns.",
        "proposed_fix": "Adjust the slicing on line 162.",
        "reasoning": "Walked the value back through the cell and method.py.",
        "paper_fidelity_check": "Fix preserves the 9-column box format the paper specifies.",
        "value_origin_trace": _VALID_TRACE,
        "bug_shape": "uncatalogued",
    }))
    diag, err = read(_DiagState())
    assert err is None, err
    assert diag is not None and diag["target_agent"] == "r2c-method-coder"
    assert diag["target_file"] == "method/method.py"
    assert "paper_fidelity_check" in diag, \
        "Phase 1 schema growth: paper_fidelity_check must round-trip"
    assert diag["value_origin_trace"] == _VALID_TRACE, \
        "value_origin_trace must round-trip"

    # target_file outside target_agent's allowlist → error
    diag_file.write_text(json.dumps({
        "schema_version": "1.0.0",
        "target_agent": "r2c-method-coder",
        "target_file": "method/model.py",  # method-coder doesn't own this
        "root_cause": "x", "proposed_fix": "y", "reasoning": "z",
        "paper_fidelity_check": "w",
        "value_origin_trace": _VALID_TRACE,
        "bug_shape": "uncatalogued",
    }))
    diag, err = read(_DiagState())
    assert diag is None and err and "not in target_agent" in err

    # Schema violation (missing required fields) → error
    diag_file.write_text(json.dumps({
        "schema_version": "1.0.0",
        "target_agent": "r2c-method-coder",
        "target_file": "method/method.py",
        "root_cause": "x",
    }))
    diag, err = read(_DiagState())
    assert diag is None and err and "schema validation" in err

    # Missing paper_fidelity_check specifically → named in the error.
    diag_file.write_text(json.dumps({
        "schema_version": "1.0.0",
        "target_agent": "r2c-method-coder",
        "target_file": "method/method.py",
        "root_cause": "x", "proposed_fix": "y", "reasoning": "z",
        "value_origin_trace": _VALID_TRACE,
    }))
    diag, err = read(_DiagState())
    assert diag is None and err and "schema validation" in err
    assert "paper_fidelity_check" in (err or ""), \
        "error message must name the missing field"

    # value_origin_trace too short (min_length=3 forces a real walk).
    diag_file.write_text(json.dumps({
        "schema_version": "1.0.0",
        "target_agent": "r2c-method-coder",
        "target_file": "method/method.py",
        "root_cause": "x", "proposed_fix": "y", "reasoning": "z",
        "paper_fidelity_check": "w",
        "value_origin_trace": ["only the failure site"],
        "bug_shape": "uncatalogued",
    }))
    diag, err = read(_DiagState())
    assert diag is None and err and "schema validation" in err
    assert "value_origin_trace" in (err or ""), \
        "error message must name the under-length field"


def test_validator_retry_loops_then_succeeds(tmp_path):
    (tmp_path / ".pipeline").mkdir()

    class _FakePaths:
        run_dir = tmp_path
        pipeline_dir = tmp_path / ".pipeline"

    class _FakeState:
        paths = _FakePaths()
        session_id = "ses_test"
        port = 0
        agent_models: dict = {}

    calls = {"validator": 0, "fix": 0}

    def fake_validator(_s):
        calls["validator"] += 1
        return (calls["validator"] >= 2, "schema error: missing foo")

    def fake_fix(s, findings):
        calls["fix"] += 1
        assert len(findings) == 1
        assert findings[0]["severity"] == "critical"
        (s.paths.pipeline_dir / "fix-marker.txt").write_text(
            "fix attempted", encoding="utf-8")
        return None

    result = run_pipeline._validator_retry(
        _FakeState(),
        stage_id="stage_1",
        validator_fn=fake_validator,
        fix_dispatch_fn=fake_fix,
        validator_label="fake_validator",
    )
    assert result is None, "validator_retry should return None on eventual success"
    assert calls["validator"] == 2, "validator should run twice (fail, then pass)"
    assert calls["fix"] == 1, "fix-mode should dispatch once"


def test_stage2_dispatches_reference_real_summaries():
    for key in ("stage_2b_architecture", "stage_2c_method"):
        assert key in STAGE_TASK_SUMMARIES, f"missing summary for {key}"


def test_skip_if_done_branches(tmp_path):
    class _SkipPaths:
        pipeline_dir = tmp_path

    f1 = tmp_path / "out1.json"
    f2 = tmp_path / "out2.json"
    halt = tmp_path / "stage_test.halt"
    sentinel = tmp_path / "stage_test.complete"

    # branch 1: outputs missing → run (returns None)
    result = run_pipeline._skip_if_done(_SkipPaths(), "stage_test", [f1, f2])
    assert result is None, "missing outputs should trigger run"

    # branch 2: outputs present BUT sentinel missing → run, not skip
    f1.write_text("{}")
    f2.write_text("{}")
    result = run_pipeline._skip_if_done(_SkipPaths(), "stage_test", [f1, f2])
    assert result is None, (
        "outputs alone should NOT trigger skip without complete-sentinel; "
        "this is the bev-distill 2026-05-19 regression: stage 2.d's "
        "partial outputs from a halted prior run let validators be skipped")

    # branch 3: outputs + sentinel present, no halt → skip
    sentinel.write_text("")
    result = run_pipeline._skip_if_done(_SkipPaths(), "stage_test", [f1, f2])
    assert result is not None, "outputs + sentinel + no halt should skip"
    assert result.status == "skipped"
    assert result.stage_id == "stage_test"
    assert result.paths_written == [f1, f2]

    # branch 4: prior halt artifact present → run, and stale halt cleared.
    halt.write_text("{}")
    result = run_pipeline._skip_if_done(_SkipPaths(), "stage_test", [f1, f2])
    assert result is None, "prior halt should force a re-run even with sentinel"
    assert not halt.exists(), "_skip_if_done should have cleared the stale halt"

    # sentinel write/clear helpers
    sentinel.unlink()
    run_pipeline._write_stage_complete_sentinel(_SkipPaths(), "stage_test")
    assert sentinel.exists(), "_write_stage_complete_sentinel should create the sentinel"
    run_pipeline._clear_stage_complete_sentinel(_SkipPaths(), "stage_test")
    assert not sentinel.exists(), "_clear_stage_complete_sentinel should remove the sentinel"
    # idempotent on missing
    run_pipeline._clear_stage_complete_sentinel(_SkipPaths(), "stage_test")


def test_run_fix_loop_reviewer_driven(tmp_path, monkeypatch):
    # validator passes, reviewer reports one critical on iteration 0,
    # producer fix-mode dispatches, reviewer clears on iteration 1.
    class _LoopPaths:
        pipeline_dir = tmp_path
        run_dir = tmp_path.parent
        method_spec = tmp_path / "method_spec.json"

    class _LoopState:
        paths = _LoopPaths()
        session_id = "ses_loop"
        port = 0
        agent_models: dict = {}

    loop_calls = {"validator": 0, "reviewer": 0, "fix": 0}
    review_path = tmp_path / "stage_review_stage_2b_architecture.json"

    def fake_v(state, *, stage_id, args, timeout=120):
        loop_calls["validator"] += 1
        return True, ""  # always passes; loop is reviewer-driven

    def fake_r(state, stage_id):
        loop_calls["reviewer"] += 1
        # No target_agent: exercises the documented fallback that routes
        # such findings to the stage's own producer (fake_fix2).
        findings = [{
            "id": "F001", "severity": "critical", "issue_type": "other",
            "file": "method/model.py", "location": "lines 1-2",
            "description": "x",
        }] if loop_calls["reviewer"] == 1 else []
        review_path.write_text(json.dumps({
            "schema_version": "1.0.0",
            "stage_id": "stage_2b_architecture",
            "review_status": "issues_found" if findings else "passed",
            "summary": "fabricated review",
            "findings": findings,
            "checks_summary": [], "caveats": [],
        }) + "\n")
        return None

    def fake_fix2(_s, findings):
        loop_calls["fix"] += 1
        assert len(findings) >= 1
        return None

    # Rev-2 condition: monkeypatch on the run_pipeline module, never a
    # verbatim globals() assignment (which would patch THIS module).
    monkeypatch.setattr(run_pipeline, "_run_stage2_script", fake_v)
    monkeypatch.setattr(run_pipeline, "_dispatch_stage_reviewer", fake_r)
    result = run_fix_loop(
        state=_LoopState(),
        stage_id="stage_2b",
        fix_dispatch_fn=fake_fix2,
        validator_args=["scripts/fake_validator.py"],
        validator_label="fake",
        reviewer_stage_id="stage_2b_architecture",
    )
    assert result is None, "run_fix_loop should succeed when reviewer clears post-fix"
    assert loop_calls["validator"] == 2, f"validator should run twice; got {loop_calls['validator']}"
    assert loop_calls["reviewer"] == 2, f"reviewer should run twice; got {loop_calls['reviewer']}"
    assert loop_calls["fix"] == 1, f"fix-mode should dispatch once; got {loop_calls['fix']}"


def test_run_fix_loop_validator_fn_shape(tmp_path, monkeypatch):
    class _VPaths:
        pipeline_dir = tmp_path
        run_dir = tmp_path.parent
        method_spec = tmp_path / "method_spec.json"

    class _VState:
        paths = _VPaths()
        session_id = "ses_vf"
        port = 0
        agent_models: dict = {}

    review_file = tmp_path / "stage_review_stage_fake_review.json"
    monkeypatch.setattr(
        run_pipeline, "_dispatch_stage_reviewer",
        lambda s, sid: review_file.write_text(json.dumps({
            "schema_version": "1.0.0",
            "stage_id": "stage_fake_review",
            "review_status": "passed",
            "summary": "fabricated review",
            "findings": [], "checks_summary": [], "caveats": [],
        }) + "\n") or None)
    result = run_fix_loop(
        state=_VState(),
        stage_id="stage_fake",
        fix_dispatch_fn=lambda *a: None,
        validator_fn=lambda s: (True, ""),
        validator_label="fake_callable_validator",
        reviewer_stage_id="stage_fake_review",
    )
    assert result is None, "validator_fn happy path should succeed"


def test_run_fix_loop_requires_a_validator(tmp_path):
    class _VPaths:
        pipeline_dir = tmp_path
        run_dir = tmp_path.parent
        method_spec = tmp_path / "method_spec.json"

    class _VState:
        paths = _VPaths()
        session_id = "ses_vf"
        port = 0
        agent_models: dict = {}

    with pytest.raises(ValueError):
        run_fix_loop(
            state=_VState(),
            stage_id="stage_fake",
            fix_dispatch_fn=lambda *a: None,
            validator_label="x",
            reviewer_stage_id="x",
        )


def test_stage5_fix_dispatcher_routing():
    f = run_pipeline._stage5_fix_dispatcher
    for short in ("method-coder", "architecture-coder", "notebook-generator",
                  "decomposer", "method-analyzer"):
        assert callable(f(short)), f"{short} should route"
    for full in ("r2c-method-coder", "r2c-architecture-coder",
                 "r2c-notebook-generator", "r2c-decomposer",
                 "r2c-method-analyzer"):
        assert callable(f(full)), f"{full} should route"
    for unrouteable in ("human", "user", "paper-fidelity-reviewer",
                        "r2c-paper-fidelity-reviewer", "", "unknown"):
        assert f(unrouteable) is None, f"{unrouteable} should be unrouteable"


def test_ask_user_log_only(tmp_path):
    class _LogPaths:
        pipeline_dir = tmp_path
        run_dir = tmp_path

    class _LogState:
        paths = _LogPaths()
        session_id = "ses_log"
        port = 0
        agent_models: dict = {}

    sample_findings = [
        {"id": "F001", "severity": "important", "target_agent": "notebook-generator",
         "description": "narrative drift", "proposed_fix": "rewrite markdown",
         "file": ".pipeline/notebook_draft.py", "location": "§4.2"},
        {"id": "F002", "severity": "important", "target_agent": "human",
         "description": "analyzer ambiguity", "proposed_fix": ""},
    ]
    run_pipeline._ask_user_log_only(_LogState(), sample_findings, "Important findings")
    log_path = tmp_path / run_layout.DEFERRED_FINDINGS_MD
    assert log_path.exists(), "log file should be created"
    content = log_path.read_text(encoding="utf-8")
    assert content.startswith("# Findings deferred to manual review"), \
        "header on first write"
    assert "## Important findings (2)" in content
    assert "### F001 (important, target_agent=notebook-generator)" in content
    assert "narrative drift" in content
    assert "rewrite markdown" in content

    # Second call: append, no second header
    run_pipeline._ask_user_log_only(_LogState(), [sample_findings[0]],
                                    "Nice-to-have findings")
    content2 = log_path.read_text(encoding="utf-8")
    assert content2.count("# Findings deferred to manual review") == 1, \
        "header written exactly once across multiple calls"
    assert "## Nice-to-have findings (1)" in content2

    # Empty findings list: no-op (file unchanged)
    before = log_path.read_text(encoding="utf-8")
    run_pipeline._ask_user_log_only(_LogState(), [], "Important findings")
    assert log_path.read_text(encoding="utf-8") == before, \
        "empty findings should be a no-op"


def test_parse_smoke_fail_cell():
    f = run_pipeline._parse_smoke_fail_cell
    assert f("notebook execution failed at cell 13\n--- failing cell source ---\n") == 13, \
        "CellExecutionError pattern"
    assert f("notebook execution TIMED OUT at cell 7 after 1200s\n") == 7, \
        "CellTimeoutError pattern"
    assert f("kernel channel broke...\nLast code cell sent to the kernel: cell 5.\n") == 5, \
        "kernel-died (last cell sent) pattern"
    assert f("nbclient was waiting on cell 9, which is markdown\n") == 9, \
        "kernel-died (waiting on) pattern"
    assert f("notebook executed cleanly in 12s") is None, "no failure pattern → None"


def test_smoke_producer_to_fix_fn():
    for producer in ("notebook-generator", "method-coder", "architecture-coder"):
        assert callable(run_pipeline._smoke_producer_to_fix_fn(producer)), \
            f"{producer} should route to a callable"
    with pytest.raises(ValueError):
        run_pipeline._smoke_producer_to_fix_fn("paper-fidelity-reviewer")


class _TodoState:
    session_id = "ses_todo_test"
    port = 0
    workspace = ""
    agent_models: dict = {}

    def __init__(self):
        self.todos: list[dict] = []
        self.post_breaker = run_pipeline.SessionPostBreaker()


def test_pipeline_todos_state_machine(monkeypatch):
    posted: list[list[dict]] = []
    monkeypatch.setattr(
        run_pipeline, "_post_todos",
        lambda s: posted.append(
            [{k: v for k, v in t.items() if k != "stage_id"} for t in s.todos]))

    ts = _TodoState()
    run_pipeline._init_pipeline_todos(ts)
    assert len(ts.todos) == len(run_pipeline.PIPELINE_TODOS)
    assert ts.todos[0]["status"] == "in_progress"
    assert ts.todos[0]["stage_id"] == "stage_0"
    assert all(t["status"] == "pending" for t in ts.todos[1:])

    # happy path: stage_0 completed → stage_1 in_progress
    run_pipeline._advance_todos(ts, "stage_0", "completed")
    assert ts.todos[0]["status"] == "completed"
    assert ts.todos[1]["status"] == "in_progress"
    assert ts.todos[1]["stage_id"] == "stage_1"
    assert posted[-1][0]["status"] == "completed"

    # skipped: stage_1 skipped → marked completed, stage_1x in_progress
    run_pipeline._advance_todos(ts, "stage_1", "skipped")
    assert ts.todos[1]["status"] == "completed"
    assert ts.todos[2]["status"] == "in_progress"
    assert ts.todos[2]["stage_id"] == "stage_1x"
    run_pipeline._advance_todos(ts, "stage_1x", "completed")
    assert ts.todos[3]["stage_id"] == "stage_2a"

    # halted: stays in_progress with rewritten activeForm; next stays pending
    run_pipeline._advance_todos(ts, "stage_2a", "halted")
    assert ts.todos[3]["status"] == "in_progress", "halted stage stays in_progress"
    assert "Halted" in ts.todos[3]["activeForm"], "activeForm should surface the halt"
    assert ts.todos[4]["status"] == "pending", "next stage stays pending on halt"

    # stage_id is driver-internal and must be stripped from POST payloads;
    # TodoWrite's schema needs priority alongside {content, activeForm, status}.
    for payload in posted:
        for entry in payload:
            assert "stage_id" not in entry, "stage_id leaked into POST payload"
            assert set(entry.keys()) == {"content", "activeForm", "status", "priority"}
            assert entry["priority"] in {"high", "medium", "low"}


def test_post_todos_is_fire_and_forget(monkeypatch):
    # Even if the underlying dispatch takes long, _post_todos returns
    # ~immediately (thread-based fix for the 60s-per-transition stall).
    dispatched_count = [0]

    def _slow_dispatch(**kwargs):
        time.sleep(0.4)
        dispatched_count[0] += 1
        return None

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", _slow_dispatch)
    ff_state = _TodoState()
    run_pipeline._init_pipeline_todos(ff_state)
    t0 = time.time()
    run_pipeline._post_todos(ff_state)
    elapsed = time.time() - t0
    assert elapsed < 0.1, (
        f"_post_todos must return ~immediately (fire-and-forget); "
        f"took {elapsed:.2f}s")
    # Give the background thread time to complete its mock dispatch BEFORE
    # monkeypatch.undo() — the thread must finish against the fake.
    time.sleep(0.6)
    assert dispatched_count[0] == 1, \
        f"background thread should have dispatched once; got {dispatched_count[0]}"
