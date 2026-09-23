"""R2C-068 — an important stage-review finding reaches the researcher.

The failure class is `finding_tier_dropped`: a defect the system detected,
recorded, and then failed to either route or disclose, because its severity
fell between two handlers. The stage fix loops route critical findings only;
the delivery-time collector kept nice-to-have and minor only; important fell
through the gap.

The shaping known-bad is the 2026-08-06 pdfgnn loop-2 stage 2c review
(`tests/fixtures/evidence/pdfgnn-loop2-0806/`), whose two important-tier
algorithm findings reached no researcher-facing surface. Both must appear on
the delivered deferred-findings surface, and the stage 4 fidelity reviewer must
be handed them before it certifies the same elements.
"""

from __future__ import annotations

import json
from pathlib import Path

import run_pipeline
from delivery_label import derive_delivery_label
from run_layout import DEFERRED_FINDINGS_MD
from tests.helpers.state import make_state

_LOOP2 = (
    Path(__file__).parent
    / "fixtures/evidence/pdfgnn-loop2-0806/stage_review_stage_2c_method.json"
)
# The second concrete case, from a different paper (GBALD, active learning) and
# a different producer (method-coder): its stage 2c review filed one important
# finding about the Eq. 13 argmax direction that also reached no surface. The
# Eq. 13 SUBJECT reached assumptions.md only because a sibling caveat asked for
# it in prose; the finding itself, with its file:line and proposed fix, was
# dropped exactly like the pdfgnn pair.
_GBALD = (
    Path(__file__).parent
    / "fixtures/evidence/gbald-verified-first/stage_review_stage_2c_method.json"
)


def _install(state, source: Path, name: str = "stage_review_stage_2c_method.json"):
    (state.paths.pipeline_dir / name).write_text(
        source.read_text(encoding="utf-8"), encoding="utf-8")


def _write_review(state, name: str, review: dict) -> None:
    (state.paths.pipeline_dir / name).write_text(
        json.dumps(review), encoding="utf-8")


def _deferred(state) -> str:
    path = state.paths.run_dir / DEFERRED_FINDINGS_MD
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _collect(state) -> list[dict]:
    important, _nice, _caveats = run_pipeline._stage_review_deferred_findings(state)
    return important


# --- Test 1: the known-bad, collected --------------------------------------


def test_loop2_dropped_findings_are_collected(tmp_path):
    state = make_state(tmp_path / "run")
    _install(state, _LOOP2)

    important = _collect(state)

    assert len(important) == 2
    by_id = {f["id"]: f for f in important}
    assert set(by_id) == {"F001", "F002"}
    # Each names its owning producer and the stage review that raised it, so a
    # researcher knows who wrote the code and where the objection came from.
    for finding in important:
        assert finding["severity"] == "important"
        assert finding["target_agent"] == "architecture-coder"
        assert finding["raised_at"] == "stage_2c_method"
        assert finding["resolution_status"] == "pending"
    # The two real defects, identified by what they are about.
    assert "df.mean()" in by_id["F002"]["description"]
    assert by_id["F002"]["file"] == "method/training.py"
    assert "gnn-neighborhood-aggregation" in by_id["F001"]["check_id"]
    assert by_id["F001"]["file"] == "method/model.py"


# --- Test 2: the delivered surface -----------------------------------------


def test_loop2_findings_reach_the_delivered_surface(tmp_path):
    state = make_state(tmp_path / "run")
    _install(state, _LOOP2)
    important = _collect(state)

    run_pipeline._ask_user_log_only(
        state, important, "Open important findings from earlier stage reviews")

    body = _deferred(state)
    # Both entries, at the important tier, with their producer named.
    assert body.count("(important, target_agent=architecture-coder)") == 2
    assert "### F001 (important" in body
    assert "### F002 (important" in body
    # The surface says where the objection came from and that nothing fixed it,
    # so the entry cannot read as an already-handled note.
    assert body.count("**Raised by:** the stage_2c_method stage review") == 2
    assert "still open (pending)" in body
    assert "critical findings only" in body
    # The reviewer's own file AND its symbol/line range both survive.
    assert "method/training.py — student_t_nll, line 99" in body
    assert "GraphDeepAR.forward, lines 293-311" in body
    # But a location that already spells out the file is not doubled up.
    run_pipeline._ask_user_log_only(state, [{
        "id": "F003", "severity": "important", "target_agent": "method-coder",
        "file": "method/method.py", "location": "method/method.py:141",
        "description": "the ranking direction is inverted.",
    }], "Open important findings from earlier stage reviews")
    doubled = _deferred(state)
    assert "**Location:** method/method.py:141" in doubled
    assert "method/method.py — method/method.py:141" not in doubled
    # The substance a researcher acts on.
    assert "df.mean()" in body
    assert "per-element df" in body


# --- Test 3: the front door names the tier ---------------------------------


def test_report_md_names_the_important_tier(tmp_path):
    from render_run_report import (  # noqa: PLC0415
        _count_deferred_important, render_run_report,
    )

    state = make_state(tmp_path / "run")
    _install(state, _LOOP2)
    run_pipeline._ask_user_log_only(
        state, _collect(state), "Open important findings from earlier stage reviews")

    body = _deferred(state)
    assert _count_deferred_important(body) == 2

    report = render_run_report(state.paths.run_dir)
    # "2 deferred findings" alone reads as housekeeping. The front door has to
    # say that two of them are unresolved important-tier defects.
    assert "2 deferred findings" in report
    assert "2 at the important tier, unresolved" in report


# --- Test 4: a resolved finding is not disclosed as open -------------------


def test_resolved_important_finding_is_not_disclosed(tmp_path):
    state = make_state(tmp_path / "run")
    _write_review(state, "stage_review_stage_2c_method.json", {
        "stage_id": "stage_2c_method",
        "review_status": "issues_found",
        "findings": [
            {
                "id": "F001", "severity": "important",
                "target_agent": "method-coder",
                "description": "the loss omits the entropy term.",
                "resolution_status": "applied",
            },
            {
                "id": "F002", "severity": "important",
                "target_agent": "method-coder",
                "description": "the ranking direction is inverted.",
                "resolution_status": "failed",
            },
        ],
    })

    important = _collect(state)

    # Applied means handled. Failed means the driver tried and did not
    # converge, which is exactly the case the researcher must hear about.
    assert [f["id"] for f in important] == ["F002"]
    assert important[0]["resolution_status"] == "failed"


# --- Test 5: the nice tier and the caveat tier are untouched ---------------


def test_nice_tier_and_caveats_unchanged_by_the_widening(tmp_path):
    state = make_state(tmp_path / "run")
    _install(state, _GBALD)

    important, nice, caveats = run_pipeline._stage_review_deferred_findings(state)

    # The two lower tiers keep collecting exactly what they collected before.
    assert [f["id"] for f in nice] == ["F002"]
    assert nice[0]["severity"] == "nice-to-have"
    assert any("Eq. 13" in c["description"] for c in caveats)
    # And the important tier picks up the finding this review dropped.
    assert [f["id"] for f in important] == ["F001"]


def test_second_case_is_a_different_paper_and_producer(tmp_path):
    """The failure class is paper-agnostic. Any run whose stage reviewer files
    an important finding loses it, so the fix must not be shaped around the
    forecasting family."""
    state = make_state(tmp_path / "run")
    _install(state, _GBALD)

    important = _collect(state)

    assert len(important) == 1
    assert important[0]["target_agent"] == "method-coder"  # pdfgnn's was arch
    assert important[0]["check_id"] == "equation_matches_paper:equation-single-rank"
    assert "Eq. 13" in important[0]["description"]


def test_a_review_without_important_findings_yields_nothing(tmp_path):
    state = make_state(tmp_path / "run")
    _write_review(state, "stage_review_stage_1_analyzer.json", {
        "stage_id": "stage_1_analyzer", "review_status": "passed",
        "findings": [
            {"id": "F001", "severity": "nice-to-have",
             "description": "the docstring could name the equation."},
        ],
        "caveats": ["the paper's notation is inconsistent in Section 4."],
    })

    important, nice, caveats = run_pipeline._stage_review_deferred_findings(state)

    assert important == []
    assert len(nice) == 1 and len(caveats) == 1


# --- Test 5b: a stage that blocks on the tier itself is left alone ---------


def test_a_stage_that_blocks_on_important_is_not_double_reported(tmp_path):
    """run_stage_2x gates on should_halt_stage, so its important findings are
    resolved, routed to assumptions.md, or the run halts before delivery. The
    resolutions merge in memory, so the on-disk file still reads `pending` —
    collecting it would put a false "nobody acted on this" line on a delivered
    surface. example_runs/pdwa is the case: two stage-2x findings read pending
    on disk while assumptions.md carries A004 (the relabel it applied) and
    A003 (the needs-user entry it raised).
    """
    state = make_state(tmp_path / "run")
    handled = {
        "id": "F001", "severity": "important",
        "target_agent": "parameter-deriver",
        "location": ".pipeline/params.json | c_safe",
        "check_id": "paper_source_values_match_paper",
        "description": "c_safe=0.25 is labelled source: paper but never stated.",
        "resolution_status": "pending",
    }
    _write_review(state, "stage_review_stage_2x_params.json", {
        "stage_id": "stage_2x_params", "review_status": "issues_found",
        "findings": [handled],
    })
    # The re-ask sidecar echoes the same stage_id, so it is covered too.
    _write_review(state, "stage_review_stage_2x_params_resolution.json", {
        "stage_id": "stage_2x_params", "review_status": "issues_found",
        "findings": [{**handled, "id": "DEF-c_safe"}],
    })

    assert _collect(state) == []


def test_the_blocking_stage_assumption_still_holds_in_the_driver(tmp_path):
    """The exclusion above is only sound while run_stage_2x really does gate on
    the important tier. If that gate ever moves, the exclusion silently starts
    dropping findings again, so the assumption is checked here rather than
    trusted."""
    import inspect  # noqa: PLC0415

    source = inspect.getsource(run_pipeline.run_stage_2x)
    assert "should_halt_stage(findings)" in source
    assert run_pipeline._IMPORTANT_TIER_BLOCKING_STAGES == {"stage_2x_params"}


# --- Test 6: a critical finding stays out of the disclosure tier -----------


def test_critical_findings_are_not_swept_into_disclosure(tmp_path):
    state = make_state(tmp_path / "run")
    _write_review(state, "stage_review_stage_2b_architecture.json", {
        "stage_id": "stage_2b_architecture",
        "review_status": "issues_found",
        "findings": [
            {
                "id": "F001", "severity": "critical",
                "target_agent": "architecture-coder",
                "description": "forward() never consumes the graph input.",
                "resolution_status": "pending",
            },
        ],
    })

    # Critical findings route to the stage fix loop. A critical finding that
    # survives to delivery is a degrade/halt, not a disclosure line, so
    # collecting it here would double-report it.
    assert _collect(state) == []


# --- Test 7: one entry per finding across the resolution re-ask overlay ----


def test_reask_overlay_does_not_duplicate_the_entry(tmp_path):
    state = make_state(tmp_path / "run")
    finding = {
        "id": "F001", "severity": "important",
        "target_agent": "architecture-coder",
        "file": "method/training.py",
        "location": "student_t_nll, line 99",
        "check_id": "methodology_contract:student-t-distribution-loss",
        "description": "The normalization constant averages df across the batch.",
        "resolution_status": "pending",
    }
    _write_review(state, "stage_review_stage_2c_method.json", {
        "stage_id": "stage_2c_method", "review_status": "issues_found",
        "findings": [finding],
    })
    # The resolution re-ask writes a SECOND file beside the original, and it
    # matches the same stage_review_*.json glob.
    _write_review(state, "stage_review_stage_2c_method_resolution.json", {
        "stage_id": "stage_2c_method", "review_status": "issues_found",
        "findings": [{**finding, "id": "F001-reask"}],
    })

    important = _collect(state)

    assert len(important) == 1


# --- Test 8: severity spelling aliases ------------------------------------


def test_severity_aliases_are_collected(tmp_path):
    state = make_state(tmp_path / "run")
    _write_review(state, "stage_review_stage_3a_notebook.json", {
        "stage_id": "stage_3a_notebook", "review_status": "issues_found",
        "findings": [
            {"id": "F001", "severity": "Important",
             "description": "capitalized spelling"},
            {"id": "F002", "severity": "major",
             "description": "the major synonym"},
        ],
    })

    important = _collect(state)

    assert [f["id"] for f in important] == ["F001", "F002"]
    # Normalized to the schema token regardless of how it arrived.
    assert {f["severity"] for f in important} == {"important"}
    # No target_agent named by the reviewer falls back to human review.
    assert {f["target_agent"] for f in important} == {"human"}


# --- Test 9: the stage 4 reviewer is handed the open findings --------------


def test_stage4_dispatch_carries_the_open_findings(tmp_path):
    state = make_state(tmp_path / "run")
    _install(state, _LOOP2)

    note = run_pipeline._open_important_findings_note(state)

    assert "Open findings from earlier stage reviews (2)" in note
    # Named by element, so the reviewer cannot certify the same element blind.
    assert "methodology_contract:gnn-neighborhood-aggregation" in note
    assert "methodology_contract:student-t-distribution-loss" in note
    assert "stage_2c_method" in note
    # Context, not a verdict: the reviewer is told to re-check and decide.
    assert "unverified" in note
    assert "decide for" in note
    # And it is told the one thing it must not do, which is what the loop-2
    # roll did.
    assert "without addressing the objection already on record" in note


def test_stage4_dispatch_note_is_empty_without_open_findings(tmp_path):
    state = make_state(tmp_path / "run")
    _write_review(state, "stage_review_stage_2c_method.json", {
        "stage_id": "stage_2c_method", "review_status": "passed",
        "findings": [
            {"id": "F001", "severity": "nice-to-have",
             "description": "a shorter docstring would read better."},
        ],
    })

    # No important-tier findings means no note at all, so a clean run's
    # reviewer prompt is byte-identical to before this change.
    assert run_pipeline._open_important_findings_note(state) == ""


def test_stage4_note_survives_an_unreadable_review(tmp_path):
    state = make_state(tmp_path / "run")
    (state.paths.pipeline_dir / "stage_review_stage_2c_method.json").write_text(
        "{not json", encoding="utf-8")

    # Review context is best-effort: a malformed review file must never take
    # down the stage 4 dispatch.
    assert run_pipeline._open_important_findings_note(state) == ""
    assert _collect(state) == []


# --- Test 10: disclosure only, never a label move -------------------------


def test_disclosure_never_moves_the_delivery_label(tmp_path):
    state = make_state(tmp_path / "run")
    _install(state, _LOOP2)
    probe = {"verdicts": [{"tier": "contribution", "status": "pass",
                           "element_ids": ["e1"]}]}
    review = {"findings": []}

    before = derive_delivery_label(probe, review)
    run_pipeline._ask_user_log_only(
        state, _collect(state), "Open important findings from earlier stage reviews")
    after = derive_delivery_label(probe, review)

    assert json.dumps(before, sort_keys=True) == json.dumps(after, sort_keys=True)
