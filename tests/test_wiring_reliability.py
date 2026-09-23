"""Wiring batch, reliability slice (2026-06-10): the judge no-write retry,
direction-aware provenance severity, and the judge-blind events log.

Evidence base: all four of 2026-06-10's halts. Three were safe-direction
provenance under-claims that halted runs for zero correctness risk; the
fourth was a halt-judge dispatch that completed without writing its decision
while the underlying fix loop was working correctly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

import run_layout
import run_pipeline
from opencode_client import (
    DispatchResponseTimeout,
    DispatchTimeout,
    OpencodeClientError,
    ServerUnreachable,
    SessionNotFound,
)
from run_events import EVENT_TYPES  # noqa: E402
from tests.helpers.state import make_state  # noqa: E402


def test_judge_decision_recorded_event_is_registered():
    # Reproduced live three times as `[events][append_failed]`.
    assert "judge_decision_recorded" in EVENT_TYPES


def test_pipeline_validation_issue_event_is_registered():
    assert "pipeline_validation_issue" in EVENT_TYPES


# ---------------------------------------------------------------------------
# Judge no-write retry
# ---------------------------------------------------------------------------


def _sentinel_outcome():
    return run_pipeline.JudgeOutcome(
        action="halt", classification="test", rationale="sentinel",
        target_agent=None, finding=None, confidence="high")


def test_judge_no_decision_redispatches_once(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    dispatches = {"n": 0}
    decodes = {"n": 0}

    monkeypatch.setattr(
        run_pipeline, "_dispatch_halt_judge",
        lambda *a, **k: dispatches.__setitem__("n", dispatches["n"] + 1))

    def _decode(*_a, **_k):
        decodes["n"] += 1
        if decodes["n"] == 1:
            raise ValueError("halt-judge dispatched but did not write x.json")
        return _sentinel_outcome()

    monkeypatch.setattr(run_pipeline, "_decode_judge_decision", _decode)

    outcome = run_pipeline._invoke_judge(
        state, stage_id="stage_3a", validator_label="v",
        stderr_tail="boom", iteration=0)
    assert outcome.rationale == "sentinel"
    assert dispatches["n"] == 2
    assert decodes["n"] == 2


def test_judge_no_decision_twice_propagates(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    dispatches = {"n": 0}

    monkeypatch.setattr(
        run_pipeline, "_dispatch_halt_judge",
        lambda *a, **k: dispatches.__setitem__("n", dispatches["n"] + 1))
    monkeypatch.setattr(
        run_pipeline, "_decode_judge_decision",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("still nothing")))

    with pytest.raises(ValueError, match="still nothing"):
        run_pipeline._invoke_judge(
            state, stage_id="stage_3a", validator_label="v",
            stderr_tail="boom", iteration=0)
    # Exactly one retry — never a loop.
    assert dispatches["n"] == 2


def test_findings_judge_path_shares_the_retry(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    dispatches = {"n": 0}
    decodes = {"n": 0}

    monkeypatch.setattr(
        run_pipeline, "_dispatch_halt_judge_for_findings",
        lambda *a, **k: dispatches.__setitem__("n", dispatches["n"] + 1))

    def _decode(*_a, **_k):
        decodes["n"] += 1
        if decodes["n"] == 1:
            raise ValueError("no decision")
        return _sentinel_outcome()

    monkeypatch.setattr(run_pipeline, "_decode_judge_decision", _decode)

    outcome = run_pipeline._invoke_judge_for_findings(
        state, stage_id="stage_2c", reviewer_stage_id="stage_2c_method",
        findings=[{"id": "F001"}], iteration=1)
    assert outcome.rationale == "sentinel"
    assert dispatches["n"] == 2


# ---------------------------------------------------------------------------
# Direction-aware provenance severity (stage 2x)
# ---------------------------------------------------------------------------


def _provenance_finding(fid, location, severity="important"):
    return {
        "id": fid,
        "check_id": "paper_source_values_match_paper",
        "severity": severity,
        "location": location,
        "description": f"{location} provenance mismatch",
        "proposed_resolution": None,
        "resolution_status": "pending",
    }


def _write_params(state, params):
    (state.paths.pipeline_dir / "params.json").write_text(
        json.dumps({"schema_version": "1.0.0", "params": params}),
        encoding="utf-8")


def test_underclaims_defer_overclaims_halt(tmp_path):
    state = make_state(tmp_path / "run")
    _write_params(state, {
        # Under-claim shape: labeled conservative, reviewer says paper states it.
        "train_until_accuracy": {"value": 0.99, "source": "system_inferred",
                                 "reasoning": "convention"},
        # Over-claim shape: labeled paper, reviewer says the paper does not
        # support it — the fabrication class, must still halt.
        "mc_samples": {"value": 20, "source": "paper",
                       "paper_section": "Section 7"},
    })
    findings = [
        _provenance_finding("F001", "train_until_accuracy"),
        _provenance_finding("F002", "mc_samples"),
        {"id": "F003", "check_id": "some_other_check",
         "severity": "important", "location": "train_until_accuracy",
         "description": "unrelated"},
    ]
    keep, deferred = run_pipeline._defer_safe_provenance_findings(
        state, "stage_2x", findings)
    assert deferred == ["F001"]
    assert [f["id"] for f in keep] == ["F002", "F003"]
    # The deferral is researcher-visible with an override path.
    assumptions = (state.paths.run_dir / run_layout.ASSUMPTIONS_MD).read_text()
    assert "train_until_accuracy" in assumptions
    assert "To override" in assumptions
    assert "mc_samples" not in assumptions


def test_unknown_location_or_missing_params_never_defers(tmp_path):
    state = make_state(tmp_path / "run")
    findings = [_provenance_finding("F001", "ghost_param")]
    # No params.json at all → conservative pass-through.
    keep, deferred = run_pipeline._defer_safe_provenance_findings(
        state, "stage_2x", findings)
    assert deferred == []
    assert keep == findings
    # params.json present but the location is unknown → also pass-through.
    _write_params(state, {"other": {"value": 1, "source": "system_inferred"}})
    keep, deferred = run_pipeline._defer_safe_provenance_findings(
        state, "stage_2x", findings)
    assert deferred == []
    assert keep == findings


def _demo_scale_finding(fid, location, severity="important"):
    return {
        "id": fid,
        "check_id": "system_default_reasoning_explains_deviation",
        "severity": severity,
        "location": location,
        "description": f"{location} budget-to-pool ratio exceeds the demo_scale limit",
        "proposed_fix": "Increase pool_size or document the trade-off.",
        "proposed_resolution": None,
        "resolution_status": "pending",
    }


def test_disclosed_demo_scale_default_defers_paper_value_halts(tmp_path):
    state = make_state(tmp_path / "run")
    _write_params(state, {
        # Disclosed non-paper default: routes to assumptions.md, never halts.
        "pool_size": {"value": 800, "source": "system_default",
                      "reasoning": "Subsample to 800; ratio ~0.75 exceeds the "
                                   "field guide's 0.05 demo_scale limit."},
        # Same check on a paper-sourced value still halts: it claims paper
        # provenance, so there is nothing to disclose-and-continue.
        "batch_size": {"value": 100, "source": "paper",
                       "paper_section": "Section 4"},
        # Non-paper default but with NO disclosing reasoning → pass through.
        "max_epochs": {"value": 8, "source": "system_inferred"},
    })
    findings = [
        _demo_scale_finding("F001", "pool_size"),
        _demo_scale_finding("F002", "batch_size"),
        _demo_scale_finding("F003", "max_epochs"),
        {"id": "F004", "check_id": "some_other_check", "severity": "important",
         "location": "pool_size", "description": "unrelated"},
    ]
    keep, deferred = run_pipeline._defer_disclosed_demo_scale_findings(
        state, "stage_2x", findings)
    assert deferred == ["F001"]
    assert [f["id"] for f in keep] == ["F002", "F003", "F004"]
    # The deferral is researcher-visible with an override path.
    assumptions = (state.paths.run_dir / run_layout.ASSUMPTIONS_MD).read_text()
    assert "pool_size" in assumptions
    assert "To override" in assumptions
    assert "batch_size" not in assumptions


def test_typed_protocol_contradictions_never_defer_to_assumptions(tmp_path):
    state = make_state(tmp_path / "run")
    _write_params(state, {
        "forecast_horizon": {
            "value": 4,
            "source": "system_inferred",
            "reasoning": "System-owned one-call demo horizon.",
            "protocol_role": "forecast_call_horizon",
            "paper_value_status": "paper_unspecified",
        },
        "context_length": {
            "value": 8,
            "source": "system_default",
            "paper_value": 10,
            "reasoning": "Demo context differs from the paper-stated value.",
            "protocol_role": "context_length",
            "paper_value_status": "paper_stated",
        },
    })

    provenance = _provenance_finding("F001", "forecast_horizon")
    stated_provenance = _provenance_finding("F002", "context_length")
    keep, deferred = run_pipeline._defer_safe_provenance_findings(
        state, "stage_2x", [provenance, stated_provenance]
    )
    assert keep == [provenance, stated_provenance]
    assert deferred == []

    demo_scale = _demo_scale_finding("F003", "forecast_horizon")
    stated_demo_scale = _demo_scale_finding("F004", "context_length")
    keep, deferred = run_pipeline._defer_disclosed_demo_scale_findings(
        state, "stage_2x", [demo_scale, stated_demo_scale]
    )
    assert keep == [demo_scale, stated_demo_scale]
    assert deferred == []
    assert not (state.paths.run_dir / run_layout.ASSUMPTIONS_MD).exists()


# ---------------------------------------------------------------------------
# Lint gate in the stage-3a validator chain (US-10 wiring)
# ---------------------------------------------------------------------------


def test_lint_gate_fails_on_undefined_name(tmp_path):
    state = make_state(tmp_path / "run")
    method = state.paths.run_dir / "method"
    method.mkdir()
    (method / "method.py").write_text(
        "def select_batch(model, x, batch_size, seed):\n"
        "    return ranked_indices[:batch_size]  # NameError class\n",
        encoding="utf-8")
    ok, err = run_pipeline._run_lint_gate(state, stage_id="stage_3a")
    assert not ok
    assert "ranked_indices" in err


def test_lint_gate_passes_clean_package(tmp_path):
    state = make_state(tmp_path / "run")
    method = state.paths.run_dir / "method"
    method.mkdir()
    (method / "method.py").write_text(
        "def select_batch(model, x, batch_size, seed):\n"
        "    return list(range(batch_size))\n",
        encoding="utf-8")
    ok, err = run_pipeline._run_lint_gate(state, stage_id="stage_3a")
    assert ok, err


# ---------------------------------------------------------------------------
# Progress-aware validator retry (the GBALD stage-1 halt's lesson)
# ---------------------------------------------------------------------------


def test_validator_retry_halts_immediately_on_fixation(tmp_path):
    state = make_state(tmp_path / "run")
    calls = {"validate": 0, "fix": 0}

    def validator(_s):
        calls["validate"] += 1
        return False, "  - same error: mc_samples conflict  "

    def fix(s, _finding):
        calls["fix"] += 1
        marker = s.paths.pipeline_dir / "fix-marker.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("fix attempted", encoding="utf-8")

    result = run_pipeline._validator_retry(
        state, stage_id="stage_1", validator_fn=validator,
        fix_dispatch_fn=fix, validator_label="fake_validator")
    assert result is not None and result.status == "halted"
    assert "UNCHANGED" in (result.notes or "") or calls["validate"] == 2
    # One fix dispatch, then the repeated signature stops the loop — the
    # raised cap never burns iterations on a non-moving error.
    assert calls["validate"] == 2
    assert calls["fix"] == 1


def test_validator_retry_rides_cascading_progress_to_the_cap(tmp_path):
    state = make_state(tmp_path / "run")
    calls = {"validate": 0, "fix": 0}

    def validator(_s):
        calls["validate"] += 1
        return False, f"error #{calls['validate']} — different each time"

    def fix(s, _finding):
        calls["fix"] += 1
        marker = s.paths.pipeline_dir / f"fix-marker-{calls['fix']}.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("fix attempted", encoding="utf-8")

    result = run_pipeline._validator_retry(
        state, stage_id="stage_1", validator_fn=validator,
        fix_dispatch_fn=fix, validator_label="fake_validator", cap=2)
    assert result is not None and result.status == "halted"
    # cap=2 → three validator runs (iterations 0..2), two fix dispatches.
    assert calls["validate"] == 3
    assert calls["fix"] == 2


def test_validator_retry_no_write_halt_names_the_truncation_class(tmp_path):
    """Overnight 2026-07-08 (ACC2021_MPC_CBF): the fix dispatch and its
    no-write retry both completed without writing, and the halt reason said
    'fixation' — misattributing transport-class noise to producer
    stubbornness. When nothing was ever written, the reason must name the
    output-cap truncation class instead, and the retry finding must carry
    the act-first sentinel that hoists the preamble to the prompt top."""
    state = make_state(tmp_path / "run")
    calls = {"validate": 0, "fix": 0}
    seen_findings: list[dict] = []

    def validator(_s):
        calls["validate"] += 1
        return False, "  - same error: eq-cbf not verbatim  "

    def fix(_s, findings):
        calls["fix"] += 1
        seen_findings.extend(findings)
        # Writes NOTHING — the cap-truncation shape.

    result = run_pipeline._validator_retry(
        state, stage_id="stage_1", validator_fn=validator,
        fix_dispatch_fn=fix, validator_label="fake_validator")
    assert result is not None and result.status == "halted"
    notes = result.notes or ""
    assert "output-cap truncation" in notes, notes
    assert "fixation" not in notes.split("not ")[0] or "not fixation" in notes, notes
    # Two fix dispatches: the original and the one act-first retry.
    assert calls["fix"] == 2
    assert seen_findings[-1].get("_act_first_retry") is True


def test_validator_retry_written_fix_unchanged_error_still_says_fixation(tmp_path):
    """Companion: when a fix WAS written and the error signature still did
    not move, the fixation wording stands — the truthful-reason branch must
    not water down the real fixation class."""
    state = make_state(tmp_path / "run")
    calls = {"fix": 0}

    def validator(_s):
        return False, "  - same error: mc_samples conflict  "

    def fix(s, _findings):
        calls["fix"] += 1
        marker = s.paths.pipeline_dir / "fix-marker.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("fix attempted", encoding="utf-8")

    result = run_pipeline._validator_retry(
        state, stage_id="stage_1", validator_fn=validator,
        fix_dispatch_fn=fix, validator_label="fake_validator")
    assert result is not None and result.status == "halted"
    assert "fixation" in (result.notes or "")
    assert "output-cap truncation" not in (result.notes or "")


def test_build_fix_mode_prompt_act_first_sentinel_leads_the_prompt():
    """The sentinel must hoist the act-first preamble to the very TOP of
    the fix prompt (position is the proven differentiator for the cap-burn
    class) and must not leak into the rendered findings."""
    from scripts.dispatch_templates import DispatchPaths, build_fix_mode_prompt

    paths = DispatchPaths(
        spec="/r/.pipeline/method_spec.json", paper="/r/.pipeline/paper.md",
        paper_map="/r/.pipeline/paper_map.json", run_dir="/r",
        taxonomy_source="/repo/taxonomy",
    )
    finding = {
        "id": "VAL001", "severity": "critical",
        "description": "quote not verbatim", "proposed_fix": "requote",
        "_act_first_retry": True,
    }
    prompt = build_fix_mode_prompt(
        target_agent="r2c-decomposer", findings=[finding], paths=paths,
    )
    assert prompt.startswith("**RETRY — ACT FIRST.**")
    assert "_act_first_retry" not in prompt

    plain = build_fix_mode_prompt(
        target_agent="r2c-decomposer",
        findings=[{k: v for k, v in finding.items() if k != "_act_first_retry"}],
        paths=paths,
    )
    assert not plain.startswith("**RETRY — ACT FIRST.**")


def test_producer_required_symbols_derive_from_the_paradigm_manifest(tmp_path):
    """iDb-RRT 2026-07-08 regression: the halt summarizer hardcoded the
    active-learning shape (training.py::build_model) and told the researcher
    a fabricated missing symbol on a motion-planning paper. The symbols now
    derive from the paradigm's own package manifest, and a genuinely missing
    manifest function is still reported."""
    import json as _json

    run_dir = tmp_path / "run"
    pipeline_dir = run_dir / ".pipeline"
    method_dir = run_dir / "method"
    pipeline_dir.mkdir(parents=True)
    method_dir.mkdir(parents=True)
    (pipeline_dir / "method_spec.json").write_text(_json.dumps({
        "comparison": {
            "classification": {"id": "motion_planning"},
            "pluggable_component": {"name": "plan"},
        },
        "critical_requirements": {},
    }))

    class _Paths:
        pass

    paths = _Paths()
    paths.run_dir = run_dir
    paths.pipeline_dir = pipeline_dir

    derived = run_pipeline._producer_required_symbols_from_manifest(
        "stage_2b", paths)
    assert derived.get("method/training.py") == ["precompute_motion_primitives"]
    assert all(
        "build_model" not in names for names in derived.values()
    ), f"AL vocabulary leaked into a motion-planning derivation: {derived}"

    # File present but the manifest function missing → reported by name.
    (method_dir / "training.py").write_text("def other():\n    return 1\n")
    missing = run_pipeline._producer_missing_required_symbols("stage_2b", paths)
    assert "method/training.py::precompute_motion_primitives" in missing

    # Function present → nothing missing.
    (method_dir / "training.py").write_text(
        "def precompute_motion_primitives(dynamics, *, n_primitives=100, seed=0):\n"
        "    return []\n")
    assert run_pipeline._producer_missing_required_symbols("stage_2b", paths) == []


def test_validator_retry_fix_then_pass_returns_none(tmp_path):
    state = make_state(tmp_path / "run")
    calls = {"validate": 0}

    def validator(_s):
        calls["validate"] += 1
        return (calls["validate"] >= 2), "first failure"

    result = run_pipeline._validator_retry(
        state, stage_id="stage_1", validator_fn=validator,
        fix_dispatch_fn=lambda _s, _f: None,
        validator_label="fake_validator")
    assert result is None
    assert calls["validate"] == 2


# ---------------------------------------------------------------------------
# Backend-unreachable circuit breaker
#
# Evidence base: the 2026-06-18 Stage 3.a halt cluster. The Code endpoint
# backend was taken down mid-run; the notebook-generator dispatch failed three
# different ways across resumes (truncated turn, gateway 400, 1800s timeout).
# Each halt then spent 6-51 minutes because the halt/finalize path kept trying
# best-effort TUI notice posts against the same dead backend, each blocking for
# its full timeout. The breaker makes a backend-caused halt fail fast.
# ---------------------------------------------------------------------------


def test_is_backend_outage_classifies_transport_failures_as_outage():
    assert run_pipeline._is_backend_outage(ServerUnreachable("connection refused"))
    assert run_pipeline._is_backend_outage(DispatchTimeout("timed out"))
    # The blocking-read-timeout subclass — exactly run #3's signature.
    assert run_pipeline._is_backend_outage(
        DispatchResponseTimeout("dispatch POST timed out after 1800s"))
    # Assistant-error payloads with gateway/abort signatures (runs #2 and #4).
    assert run_pipeline._is_backend_outage(OpencodeClientError(
        'dispatch produced an assistant error: {"name": "MessageAbort"}'))
    # The gateway HTML 400 — content-type text/html + an HTML body, exactly
    # what the proxy returned when the Code endpoint was down.
    assert run_pipeline._is_backend_outage(OpencodeClientError(
        'dispatch produced an assistant error: {"name": "APIError", '
        '"statusCode": 400, "content-type": "text/html", '
        '"responseBody": "<html><body><h1>400 Bad request</h1>"}'))


def test_is_backend_outage_leaves_alive_backend_errors_alone():
    # A healthy backend that rejected a tool call mid-summary — transient.
    assert not run_pipeline._is_backend_outage(OpencodeClientError(
        "Tool call not allowed while generating summary"))
    # Session bookkeeping, not a transport outage.
    assert not run_pipeline._is_backend_outage(SessionNotFound("no session"))
    # A contract violation the (healthy) backend's output triggered.
    assert not run_pipeline._is_backend_outage(
        run_pipeline.OutOfScopeWritesError(
            "r2c-method-coder", ["notebook.ipynb"], ["method/*.py"]))
    # A model *semantic* 400 — JSON content type, live API. The discriminating
    # case: a JSON 400 is the backend rejecting bad input, not the gateway down.
    assert not run_pipeline._is_backend_outage(OpencodeClientError(
        'dispatch produced an assistant error: {"statusCode": 400, '
        '"content-type": "application/json", "message": "invalid role"}'))
    # A generic assistant error with no outage signature.
    assert not run_pipeline._is_backend_outage(OpencodeClientError(
        "schema validation failed: missing field"))
    assert not run_pipeline._is_backend_outage(ValueError("nope"))


def test_dispatch_agent_trips_breaker_then_fails_fast(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    calls = {"n": 0}

    def _boom(*_a, **_k):
        calls["n"] += 1
        raise ServerUnreachable("dispatch POST connection refused")

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", _boom)

    # First dispatch reaches the network, fails, and trips the breaker.
    with pytest.raises(ServerUnreachable):
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-notebook-generator", prompt="x", timeout_s=1800)
    assert state.backend_unreachable is True
    assert calls["n"] == 1

    # Second dispatch fails fast WITHOUT touching the network — this is what
    # turns a 51-minute halt-path hang into seconds.
    with pytest.raises(ServerUnreachable):
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-halt-judge", prompt="y", timeout_s=900)
    assert calls["n"] == 1  # dispatch_and_wait was not called a second time


def test_dispatch_timeout_with_generating_session_leaves_breaker_closed(
    tmp_path, monkeypatch,
):
    """The ADAM 2026-07-13 false outage: a dispatch POST outlasts its
    timeout because the turn is still generating (a cap-burning runaway
    step), NOT because the backend is down — the agent had completed a
    Write 75s into the window. With per-dispatch sessions, assistant parts
    in the fresh work session prove the backend is alive, so the breaker
    stays closed and only this dispatch fails."""
    from opencode_client import DispatchResponseTimeout  # noqa: PLC0415

    state = make_state(tmp_path / "run")
    monkeypatch.setattr(
        run_pipeline, "create_session", lambda **_k: "fresh-work-ses")

    def _slow_turn(*_a, **_k):
        raise DispatchResponseTimeout("dispatch POST timed out after 240s")

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", _slow_turn)
    fetched = []

    def _messages(session_id, *, port):
        fetched.append(session_id)
        return [
            {"info": {"role": "assistant"},
             "parts": [{"type": "tool", "tool": "write"},
                       {"type": "step-finish"}]},
        ]

    monkeypatch.setattr(
        run_pipeline.opencode_client, "fetch_session_messages", _messages)

    with pytest.raises(DispatchResponseTimeout):
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-method-analyzer", prompt="x",
            timeout_s=240)
    assert state.backend_unreachable is False
    assert fetched == ["fresh-work-ses"]
    events = [json.loads(line) for line in
              (state.paths.pipeline_dir / "run_events.jsonl")
              .read_text().splitlines()]
    assert any(e["event_type"] == "dispatch_timeout_backend_alive"
               for e in events)


def test_dispatch_timeout_with_empty_session_still_trips_breaker(
    tmp_path, monkeypatch,
):
    """The conservative half: a timed-out dispatch whose fresh session has
    no assistant output gives no proof of life, so the breaker trips
    exactly as before."""
    from opencode_client import DispatchResponseTimeout  # noqa: PLC0415

    state = make_state(tmp_path / "run")
    monkeypatch.setattr(
        run_pipeline, "create_session", lambda **_k: "fresh-work-ses")
    fetched = []
    monkeypatch.setattr(
        run_pipeline, "dispatch_and_wait",
        lambda **_k: (_ for _ in ()).throw(
            DispatchResponseTimeout("dispatch POST timed out after 240s")))
    monkeypatch.setattr(
        run_pipeline.opencode_client, "fetch_session_messages",
        lambda session_id, port: (fetched.append(session_id), [])[1])

    with pytest.raises(DispatchResponseTimeout):
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-method-analyzer", prompt="x",
            timeout_s=240)
    assert state.backend_unreachable is True
    assert fetched == ["fresh-work-ses"]


def test_dispatch_agent_leaves_breaker_closed_on_alive_backend_error(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")

    def _content_err(*_a, **_k):
        raise OpencodeClientError("schema validation failed: missing field")

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", _content_err)
    with pytest.raises(OpencodeClientError):
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-method-coder", prompt="x", timeout_s=900)
    assert state.backend_unreachable is False


def test_breaker_probation_probe_closes_on_success(tmp_path, monkeypatch):
    """Past the probation cooldown the next dispatch goes through as a live
    probe, and a completed probe closes the breaker (the 2026-07-06 fix: the
    breaker had no reset path, so one transient hang silenced recovery for
    the rest of the run against a backend that demonstrably flaps back
    within a minute)."""
    from opencode_client import DispatchResult  # noqa: PLC0415

    state = make_state(tmp_path / "run")
    calls = {"n": 0}

    def _flap(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ServerUnreachable("dispatch POST connection refused")
        return DispatchResult(
            session_id="fake-session", user_message_id="u1",
            assistant_message_id="a1", completed=True, elapsed_s=0.1)

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", _flap)
    with pytest.raises(ServerUnreachable):
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-notebook-generator", prompt="x",
            timeout_s=1800)
    assert state.backend_unreachable is True
    assert state.backend_unreachable_at is not None

    # Within the cooldown: fail fast, no network touch (unchanged behavior).
    with pytest.raises(ServerUnreachable):
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-halt-judge", prompt="y", timeout_s=900)
    assert calls["n"] == 1

    # Past the cooldown: the dispatch is a live probe and success closes
    # the breaker.
    state.backend_unreachable_at -= run_pipeline._BREAKER_PROBATION_S + 1
    result = run_pipeline.dispatch_agent(
        state=state, agent="r2c-stage-reviewer", prompt="z", timeout_s=900)
    assert calls["n"] == 2
    assert result.completed is True
    assert state.backend_unreachable is False
    assert state.backend_unreachable_at is None


def test_breaker_probe_failure_restarts_cooldown(tmp_path, monkeypatch):
    """A probe that fails with an outage signature re-opens the breaker with
    a fresh cooldown, so subsequent dispatches fail fast again instead of
    each blocking for a full timeout against a backend that is still down."""
    state = make_state(tmp_path / "run")
    calls = {"n": 0}

    def _still_down(*_a, **_k):
        calls["n"] += 1
        raise ServerUnreachable("dispatch POST connection refused")

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", _still_down)
    with pytest.raises(ServerUnreachable):
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-notebook-generator", prompt="x",
            timeout_s=1800)
    first_trip_at = state.backend_unreachable_at

    state.backend_unreachable_at -= run_pipeline._BREAKER_PROBATION_S + 1
    with pytest.raises(ServerUnreachable):
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-stage-reviewer", prompt="y", timeout_s=900)
    assert calls["n"] == 2  # the probe reached the network
    assert state.backend_unreachable is True
    assert state.backend_unreachable_at > first_trip_at - run_pipeline._BREAKER_PROBATION_S

    # Fresh cooldown: the next dispatch fails fast without dispatching.
    with pytest.raises(ServerUnreachable):
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-halt-judge", prompt="z", timeout_s=900)
    assert calls["n"] == 2


def test_artifact_landed_recovery_closes_breaker(tmp_path, monkeypatch):
    """The artifact-landed soft-recovery must close the breaker it just
    tripped (the 2026-07-06 fix: the recovery soft-completed the stage and
    the still-open breaker then failed the very next dispatch fast,
    defeating the recovery one dispatch later)."""
    state = make_state(tmp_path / "run")
    canonical = state.paths.run_dir / ".pipeline" / "out.json"

    def _write_then_hang(*_a, **_k):
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_text("{}")
        raise DispatchTimeout("dispatch POST timed out after 900s")

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", _write_then_hang)
    result = run_pipeline._dispatch_with_scope_check(
        state=state, agent="r2c-smoke-diagnostician", prompt="p",
        timeout_s=900,
        recovery_check_fn=lambda s: canonical.is_file(),
        writeable_paths_override=[".pipeline/out.json"])

    assert result.error["recovered_from"] == "DispatchTimeout"
    assert state.backend_unreachable is False
    assert state.backend_unreachable_at is None
    events = [json.loads(line) for line in
              (state.paths.pipeline_dir / "run_events.jsonl")
              .read_text().splitlines()]
    recovered = [e for e in events
                 if e["event_type"] == "agent_dispatch_recovered"]
    assert len(recovered) == 1
    assert recovered[0]["details"]["breaker_closed"] is True


def test_halt_notice_post_skips_when_backend_unreachable(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    calls = {"n": 0}
    monkeypatch.setattr(run_pipeline, "dispatch_and_wait",
                        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))
    halt_path = state.paths.pipeline_dir / "stage_3a.halt"

    # Backend healthy → the must-deliver halt notice is posted.
    run_pipeline._post_halt_to_session(state, "stage_3a", "boom", halt_path)
    assert calls["n"] == 1
    # Backend down → skipped entirely (the halt artifact on disk is the record).
    state.backend_unreachable = True
    run_pipeline._post_halt_to_session(state, "stage_3a", "boom", halt_path)
    assert calls["n"] == 1


def test_end_of_run_notice_post_skips_when_backend_unreachable(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    calls = {"n": 0}
    monkeypatch.setattr(run_pipeline, "dispatch_and_wait",
                        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))

    run_pipeline._post_end_of_run_notice(
        state, run_status="halted", manifest_status="blocked")
    assert calls["n"] == 1
    state.backend_unreachable = True
    run_pipeline._post_end_of_run_notice(
        state, run_status="halted", manifest_status="blocked")
    assert calls["n"] == 1


def test_prefixed_location_form_defers_like_bare(tmp_path):
    """GBALD 2026-07-03 round 2: the reviewer wrote `params.dropout_rate`
    (prefixed) while the lookup used the bare key, so a safe-direction
    under-claim missed the defer and cost a re-ask dispatch. Both forms
    must resolve to the same params.json entry, in both defer paths."""
    state = make_state(tmp_path / "run")
    _write_params(state, {
        "dropout_rate": {"value": 0.5, "source": "system_inferred",
                         "reasoning": "convention"},
        "budget": {"value": 50, "source": "system_default",
                   "reasoning": "demo-scale trade-off disclosed here"},
    })

    keep, deferred = run_pipeline._defer_safe_provenance_findings(
        state, "stage_2x", [_provenance_finding("F001", "params.dropout_rate")])
    assert deferred == ["F001"]
    assert keep == []

    keep, deferred = run_pipeline._defer_disclosed_demo_scale_findings(
        state, "stage_2x", [_demo_scale_finding("F002", "params.budget")])
    assert deferred == ["F002"]
    assert keep == []


def test_diagnostician_budget_is_producer_class():
    """Casualty #10 (the maintainer's 2026-07-07 mock run): an anchor-less
    trainability diagnosis is a producer-grade turn, not a light
    reviewer read. The legitimate turn ran ~26.5 minutes; the old 900s
    reviewer-class budget halted the run while the correct diagnosis
    was still being produced (it landed, valid, 11 minutes after the
    driver gave up). The budget must stay in the producer class."""
    assert (run_pipeline.DIAGNOSTICIAN_TIMEOUT_S
            >= run_pipeline.METHOD_CODER_TIMEOUT_S)


# ---------------------------------------------------------------------------
# Dispatch-recovery ladder rung 3 (queue item 8): the one bounded transport
# re-dispatch per run, behind the probation cooldown and the session-liveness
# precheck (maintainer-approved interim for the T1b alive-but-over-budget hazard).
# The conftest autouse fixture disables rung 3 suite-wide (real 120s waits +
# a network liveness probe); these tests restore the real function and stub
# the sleep + liveness probe.
# ---------------------------------------------------------------------------

# Captured at import time, BEFORE the autouse fixture patches it away.
_REAL_RUNG3 = run_pipeline._rung3_transport_redispatch


def _enable_rung3(
    monkeypatch,
    fingerprints=("fp-stable", "fp-stable"),
    transport_trace=None,
):
    from opencode_client import DispatchResult  # noqa: PLC0415
    trace = transport_trace if transport_trace is not None else {}
    trace["created"] = []
    trace["liveness"] = []
    monkeypatch.setattr(run_pipeline, "_rung3_transport_redispatch",
                        _REAL_RUNG3)
    monkeypatch.setattr(run_pipeline, "_rung3_sleep", lambda s: None)
    fp = iter(fingerprints)

    def _fingerprint(session_id, *, port):
        trace["liveness"].append(session_id)
        return next(fp)

    def _create_session(**_kwargs):
        session_id = f"rung3-work-session-{len(trace['created']) + 1}"
        trace["created"].append(session_id)
        return session_id

    monkeypatch.setattr(run_pipeline, "session_activity_fingerprint", _fingerprint)
    monkeypatch.setattr(run_pipeline, "create_session", _create_session)
    monkeypatch.setattr(
        run_pipeline.opencode_client,
        "fetch_session_messages",
        lambda _session_id, *, port: [],
    )
    return DispatchResult


def _event_types(state):
    path = state.paths.pipeline_dir / "run_events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line)["event_type"]
            for line in path.read_text().splitlines() if line.strip()]


def test_rung3_recovers_hung_dispatch_and_keeps_failure_on_books(
        tmp_path, monkeypatch):
    """Design test 1 (T1 recovered) + test 6 (signal integrity): the run
    continues, events carry failed -> retry -> recovered in order, and the
    recovered hang still counts exactly one dispatch failure."""
    transport = {}
    DispatchResult = _enable_rung3(
        monkeypatch, transport_trace=transport)
    state = make_state(tmp_path / "run")
    calls = {"n": 0}
    dispatched_sessions = []

    def _flap(*_a, **kwargs):
        calls["n"] += 1
        dispatched_sessions.append(kwargs["session_id"])
        if calls["n"] == 1:
            raise DispatchTimeout("dispatch POST timed out after 900s")
        return DispatchResult(session_id=kwargs["session_id"], user_message_id="u",
                              assistant_message_id="a", completed=True,
                              elapsed_s=0.2)

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", _flap)
    result = run_pipeline.dispatch_agent(
        state=state, agent="r2c-method-coder", prompt="x", timeout_s=900)

    assert result.completed is True and calls["n"] == 2
    types = _event_types(state)
    assert types.count("agent_dispatch_failed") == 1
    assert (types.index("agent_dispatch_failed")
            < types.index("dispatch_retry_after_transport")
            < types.index("agent_dispatch_recovered"))
    assert state.transport_retry_used is True
    assert state.backend_unreachable is False  # the probe success closed it
    assert transport["created"] == [
        "rung3-work-session-1", "rung3-work-session-2"]
    assert dispatched_sessions == transport["created"]
    assert transport["liveness"] == [
        "rung3-work-session-1", "rung3-work-session-1"]


def test_rung3_is_one_per_run(tmp_path, monkeypatch):
    """Design test 2: a second transport failure later in the same run halts
    exactly as today — no second retry event."""
    DispatchResult = _enable_rung3(
        monkeypatch, fingerprints=("a", "a", "b", "b"))
    state = make_state(tmp_path / "run")
    calls = {"n": 0}

    def _flap(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise DispatchTimeout("dispatch POST timed out after 900s")
        if calls["n"] == 2:
            return DispatchResult(session_id="s", user_message_id="u",
                                  assistant_message_id="a", completed=True,
                                  elapsed_s=0.1)
        raise DispatchTimeout("dispatch POST timed out after 900s")

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", _flap)
    run_pipeline.dispatch_agent(
        state=state, agent="r2c-method-coder", prompt="x", timeout_s=900)
    with pytest.raises(DispatchTimeout):
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-stage-reviewer", prompt="y", timeout_s=900)
    assert _event_types(state).count("dispatch_retry_after_transport") == 1
    assert calls["n"] == 3  # no fourth network touch


def test_rung3_refuses_a_live_session_and_halts_as_today(
        tmp_path, monkeypatch):
    """The liveness interim: a still-generating work session (moving
    fingerprint) refuses the re-dispatch — two writers on one run dir is
    worse than one casualty. The attempt is consumed, the refusal is a
    first-class event, and the original failure propagates."""
    _enable_rung3(monkeypatch, fingerprints=("fp-1", "fp-2"))
    state = make_state(tmp_path / "run")
    calls = {"n": 0}

    def _boom(*_a, **_k):
        calls["n"] += 1
        raise DispatchTimeout("dispatch POST timed out after 900s")

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", _boom)
    with pytest.raises(DispatchTimeout):
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-method-coder", prompt="x", timeout_s=900)
    assert calls["n"] == 1  # the re-dispatch never went out
    types = _event_types(state)
    assert "dispatch_retry_skipped_live_session" in types
    assert "agent_dispatch_recovered" not in types
    assert state.transport_retry_used is True


def test_rung3_refuses_when_liveness_is_unreadable(tmp_path, monkeypatch):
    """Server can't answer the liveness probe -> conservative refusal."""
    _enable_rung3(monkeypatch)

    def _raise(sid, *, port):
        raise OpencodeClientError("GET /session returned 500")
    monkeypatch.setattr(run_pipeline, "session_activity_fingerprint", _raise)
    state = make_state(tmp_path / "run")
    monkeypatch.setattr(
        run_pipeline, "dispatch_and_wait",
        lambda **k: (_ for _ in ()).throw(
            DispatchTimeout("dispatch POST timed out after 900s")))
    with pytest.raises(DispatchTimeout):
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-method-coder", prompt="x", timeout_s=900)
    assert "dispatch_retry_skipped_live_session" in _event_types(state)


def test_rung3_proceeds_when_session_is_gone(tmp_path, monkeypatch):
    """A 404 session cannot be writing: fingerprint None counts as
    quiescent and the re-dispatch proceeds."""
    DispatchResult = _enable_rung3(monkeypatch)
    monkeypatch.setattr(run_pipeline, "session_activity_fingerprint",
                        lambda sid, *, port: None)
    state = make_state(tmp_path / "run")
    calls = {"n": 0}

    def _flap(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ServerUnreachable("connection refused")
        return DispatchResult(session_id="s", user_message_id="u",
                              assistant_message_id="a", completed=True,
                              elapsed_s=0.1)

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", _flap)
    result = run_pipeline.dispatch_agent(
        state=state, agent="r2c-method-coder", prompt="x", timeout_s=900)
    assert result.completed is True and calls["n"] == 2


def test_terminal_breaker_after_two_failed_probes(tmp_path, monkeypatch):
    """Design test 3: two consecutive failed probation probes convert the
    breaker to terminal — every subsequent dispatch fails fast with no
    network touch and the error names the terminal breaker."""
    state = make_state(tmp_path / "run")
    state.transport_retry_used = True  # isolate probe accounting from rung 3
    calls = {"n": 0}

    def _boom(*_a, **_k):
        calls["n"] += 1
        raise ServerUnreachable("connection refused")

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", _boom)
    with pytest.raises(ServerUnreachable):
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-method-coder", prompt="x", timeout_s=900)
    assert state.failed_probes == 0  # first failure was not a probe

    state.backend_unreachable_at -= run_pipeline._BREAKER_PROBATION_S + 1
    with pytest.raises(ServerUnreachable):
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-method-coder", prompt="x", timeout_s=900)
    assert state.failed_probes == 1 and not state.backend_terminal

    state.backend_unreachable_at -= run_pipeline._BREAKER_PROBATION_S + 1
    with pytest.raises(ServerUnreachable):
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-method-coder", prompt="x", timeout_s=900)
    assert state.backend_terminal is True

    n_before = calls["n"]
    with pytest.raises(ServerUnreachable, match="TERMINAL"):
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-halt-judge", prompt="y", timeout_s=900)
    assert calls["n"] == n_before  # fail fast, zero network


def test_probe_success_resets_the_failed_probe_count(tmp_path, monkeypatch):
    """One failed probe followed by a successful one closes the breaker and
    clears the count — terminal needs two CONSECUTIVE failures."""
    from opencode_client import DispatchResult  # noqa: PLC0415
    state = make_state(tmp_path / "run")
    state.transport_retry_used = True
    calls = {"n": 0}

    def _flap(*_a, **_k):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise ServerUnreachable("connection refused")
        return DispatchResult(session_id="s", user_message_id="u",
                              assistant_message_id="a", completed=True,
                              elapsed_s=0.1)

    monkeypatch.setattr(run_pipeline, "dispatch_and_wait", _flap)
    with pytest.raises(ServerUnreachable):
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-method-coder", prompt="x", timeout_s=900)
    state.backend_unreachable_at -= run_pipeline._BREAKER_PROBATION_S + 1
    with pytest.raises(ServerUnreachable):
        run_pipeline.dispatch_agent(
            state=state, agent="r2c-method-coder", prompt="x", timeout_s=900)
    assert state.failed_probes == 1
    state.backend_unreachable_at -= run_pipeline._BREAKER_PROBATION_S + 1
    run_pipeline.dispatch_agent(
        state=state, agent="r2c-method-coder", prompt="x", timeout_s=900)
    assert state.failed_probes == 0 and not state.backend_terminal
    assert state.backend_unreachable is False


def test_completed_turn_never_ladders(tmp_path, monkeypatch):
    """Design test 4 (T3 exclusion, structural): a turn that completes —
    including completed-no-write — raises nothing, so rung 3 can never see
    it; stage-level machinery owns that class."""
    DispatchResult = _enable_rung3(monkeypatch)
    state = make_state(tmp_path / "run")
    monkeypatch.setattr(
        run_pipeline, "dispatch_and_wait",
        lambda **k: DispatchResult(session_id="s", user_message_id="u",
                                   assistant_message_id="a", completed=True,
                                   elapsed_s=0.1))
    run_pipeline.dispatch_agent(
        state=state, agent="r2c-method-coder", prompt="x", timeout_s=900)
    types = _event_types(state)
    assert "dispatch_retry_after_transport" not in types
    assert "dispatch_retry_skipped_live_session" not in types


def test_halt_notice_path_never_ladders(tmp_path, monkeypatch):
    """Design test 5: notice posts stay best-effort skip-on-open-breaker;
    they call dispatch_and_wait directly, never dispatch_agent, so no
    retry event can appear."""
    _enable_rung3(monkeypatch)
    state = make_state(tmp_path / "run")
    state.backend_unreachable = True
    state.backend_unreachable_at = __import__("time").monotonic()
    posted = []
    monkeypatch.setattr(run_pipeline, "dispatch_and_wait",
                        lambda **k: posted.append(k))
    run_pipeline._post_halt_to_session(
        state, "stage_3c", "reason", tmp_path / "halt.json")
    assert posted == []  # skipped on the open breaker
    types = _event_types(state)
    assert "dispatch_retry_after_transport" not in types


def test_rung3_event_types_are_registered():
    assert "dispatch_retry_after_transport" in EVENT_TYPES
    assert "dispatch_retry_skipped_live_session" in EVENT_TYPES
    import fleet_state  # noqa: PLC0415
    assert fleet_state.EVENT_CATEGORIES[
        "dispatch_retry_skipped_live_session"] == "problem"
    assert "agent_dispatch_recovered" in fleet_state.EVENT_CATEGORIES


def test_session_activity_fingerprint_shapes(monkeypatch):
    """The liveness probe's reduction: 404 -> None (a gone session cannot
    write); a message list reduces to a stable comparable string."""
    import opencode_client

    monkeypatch.setattr(opencode_client, "_http",
                        lambda path, *, port, **k: (404, "not found"))
    assert opencode_client.session_activity_fingerprint("s1", port=1) is None

    body = json.dumps([
        {"info": {"id": "m1", "time": {"completed": 111}}, "parts": [{}]},
        {"info": {"id": "m2", "time": {"completed": 222}},
         "parts": [{}, {}]},
    ])
    monkeypatch.setattr(opencode_client, "_http",
                        lambda path, *, port, **k: (200, body))
    fp = opencode_client.session_activity_fingerprint("s1", port=1)
    assert fp == "2:m2:222:2"

    monkeypatch.setattr(opencode_client, "_http",
                        lambda path, *, port, **k: (500, "boom"))
    with pytest.raises(OpencodeClientError):
        opencode_client.session_activity_fingerprint("s1", port=1)
