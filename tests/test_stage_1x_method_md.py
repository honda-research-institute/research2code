"""Stage 1.x — METHOD.md explanation layer wiring (slice 2.1).

Best-effort contract under test: the stage NEVER halts. Dispatch
failures, validator rejections, and missing parts all degrade to a
METHOD.md with explicit pending markers, and the resume path
re-dispatches exactly the missing/rejected parts (landed good parts are
never re-dispatched). The deterministic generator runs as a real
subprocess; only the agent dispatch is faked.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import run_pipeline
from opencode_client import OpencodeClientError
from run_pipeline import run_stage_1x
from tests.helpers.state import make_state

PART_PATH_RE = re.compile(r"Write EXACTLY ONE file at: (\S+)")

GOOD_ENTRY = {
    "what": "Computes the score the selection step ranks candidates by, "
            "using the model's current predictions.",
    "why_novel": "Standard machinery per the paper's own positioning, "
                 "included because the embedding builds on it.",
    "intuition": "Uncertain points produce large values, so ranking by it "
                 "surfaces the labels worth buying.",
}


def _eq(i):
    return {"id": f"eq-{i}", "type": "equation", "code_role": "implement",
            "name": f"Eq {i}", "section": "S3",
            "source_text": f"paper statement {i}",
            "description": f"decomposition note {i}"}


def _prepare(tmp_path, n_eqs=5):
    state = make_state(tmp_path / "run")
    (state.paths.paper_map).write_text(json.dumps(
        {"title": "T", "elements": [_eq(i) for i in range(1, n_eqs + 1)]}))
    return state


def _ids_from_prompt(prompt):
    return re.findall(r"^- (eq-\d+)$", prompt, re.MULTILINE)


def _fake_dispatch(entry_factory=lambda eid: dict(GOOD_ENTRY)):
    calls = []

    def fake(*, state, agent, prompt, timeout_s, **kwargs):
        calls.append(agent)
        path = Path(PART_PATH_RE.search(prompt).group(1))
        ids = _ids_from_prompt(prompt)
        path.write_text(json.dumps({
            "schema_version": "1.0.0",
            "explanations": {i: entry_factory(i) for i in ids}}))
    return fake, calls


def test_happy_path_completes_with_full_explanations(tmp_path, monkeypatch):
    state = _prepare(tmp_path)  # 5 equations -> 5 single-equation parts
    fake, calls = _fake_dispatch()
    monkeypatch.setattr(run_pipeline, "_dispatch_with_scope_check", fake)
    result = run_stage_1x(state)
    assert result.status == "completed", result.notes
    assert calls.count("r2c-method-explainer") == 5
    md = (state.paths.run_dir / "METHOD.md").read_text()
    assert "explanation pending" not in md
    assert "surfaces the labels worth buying" in md
    sidecar = json.loads((state.paths.pipeline_dir /
                          "method_explanations.json").read_text())
    assert len(sidecar["explanations"]) == 5


def test_resume_skips_landed_parts(tmp_path, monkeypatch):
    state = _prepare(tmp_path)
    # Parts 1-4 landed by a prior roll (one equation per part).
    for i in range(1, 5):
        part = (state.paths.pipeline_dir /
                f"method_explanations.part-{i}.json")
        part.write_text(json.dumps({
            "schema_version": "1.0.0",
            "explanations": {f"eq-{i}": dict(GOOD_ENTRY)}}))
    fake, calls = _fake_dispatch()
    monkeypatch.setattr(run_pipeline, "_dispatch_with_scope_check", fake)
    result = run_stage_1x(state)
    assert result.status == "completed"
    assert calls.count("r2c-method-explainer") == 1  # only part 5 dispatched


def test_resume_reconciles_parts_from_an_older_chunk_layout(tmp_path,
                                                            monkeypatch):
    # The A8-interlude mixed-layout bug: parts landed under the old
    # four-equation-per-part layout, then the chunk size changed to 1.
    # Old part-1 (eq 1-4) and part-2 (eq-5) sit where the new plan means
    # eq-1 and eq-2 alone; without reconciliation the freshly dispatched
    # parts collide with them at merge (duplicate element id) and the
    # stage degrades every explanation to pending.
    state = _prepare(tmp_path)
    old_layout = {1: ["eq-1", "eq-2", "eq-3", "eq-4"], 2: ["eq-5"]}
    for n, ids in old_layout.items():
        part = (state.paths.pipeline_dir /
                f"method_explanations.part-{n}.json")
        part.write_text(json.dumps({
            "schema_version": "1.0.0",
            "explanations": {i: dict(GOOD_ENTRY) for i in ids}}))
    fake, calls = _fake_dispatch()
    monkeypatch.setattr(run_pipeline, "_dispatch_with_scope_check", fake)
    result = run_stage_1x(state)
    assert result.status == "completed", result.notes
    assert "part merge failed" not in result.notes
    sidecar = json.loads((state.paths.pipeline_dir /
                          "method_explanations.json").read_text())
    assert len(sidecar["explanations"]) == 5
    # The stale parts disagreed with the plan, so all 5 re-dispatched.
    assert calls.count("r2c-method-explainer") == 5


def test_dispatch_failure_degrades_with_pending_markers(tmp_path,
                                                        monkeypatch):
    state = _prepare(tmp_path)

    def failing(**kwargs):
        raise OpencodeClientError("session busy")
    monkeypatch.setattr(run_pipeline, "_dispatch_with_scope_check", failing)
    result = run_stage_1x(state)
    assert result.status == "degraded"
    assert "dispatch failed" in result.notes
    md = (state.paths.run_dir / "METHOD.md").read_text()
    assert "explanation pending" in md
    assert "paper statement 1" in md  # structure + quotes still delivered


def test_validator_rejection_drops_entries_and_requeues_their_part(
        tmp_path, monkeypatch):
    state = _prepare(tmp_path)

    def entry_factory(eid):
        if eid == "eq-2":
            return {**GOOD_ENTRY, "intuition": "Too short."}
        return dict(GOOD_ENTRY)
    fake, calls = _fake_dispatch(entry_factory)
    monkeypatch.setattr(run_pipeline, "_dispatch_with_scope_check", fake)
    result = run_stage_1x(state)
    assert result.status == "degraded"
    assert "eq-2" in result.notes
    sidecar = json.loads((state.paths.pipeline_dir /
                          "method_explanations.json").read_text())
    assert "eq-2" not in sidecar["explanations"]
    assert "eq-5" in sidecar["explanations"]  # other parts' entries kept
    # The offending PART (2: eq-2 alone under single-equation parts) is
    # deleted so resume re-dispatches exactly it; clean parts stay.
    assert not (state.paths.pipeline_dir /
                "method_explanations.part-2.json").is_file()
    assert (state.paths.pipeline_dir /
            "method_explanations.part-1.json").is_file()
    assert (state.paths.pipeline_dir /
            "method_explanations.part-5.json").is_file()
    md = (state.paths.run_dir / "METHOD.md").read_text()
    assert "explanation pending" in md


def test_validator_rejection_retries_part_before_degrading(
        tmp_path, monkeypatch):
    state = _prepare(tmp_path)
    attempts = {"eq-2": 0}

    def entry_factory(eid):
        if eid == "eq-2":
            attempts[eid] += 1
            if attempts[eid] == 1:
                return {**GOOD_ENTRY, "intuition": "Too short."}
        return dict(GOOD_ENTRY)

    fake, calls = _fake_dispatch(entry_factory)
    monkeypatch.setattr(run_pipeline, "_dispatch_with_scope_check", fake)
    result = run_stage_1x(state)
    assert result.status == "completed", result.notes
    # five single-equation parts, then the part-2 retry
    assert calls.count("r2c-method-explainer") == 6
    assert (state.paths.pipeline_dir /
            "method_explanations.part-2.json").is_file()
    sidecar = json.loads((state.paths.pipeline_dir /
                          "method_explanations.json").read_text())
    assert "eq-2" in sidecar["explanations"]
    md = (state.paths.run_dir / "METHOD.md").read_text()
    assert "explanation pending" not in md


def test_skip_if_done(tmp_path, monkeypatch):
    state = _prepare(tmp_path)
    (state.paths.run_dir / "METHOD.md").write_text("existing")
    (state.paths.pipeline_dir / "stage_1x.complete").write_text("")
    called = []
    monkeypatch.setattr(run_pipeline, "_dispatch_with_scope_check",
                        lambda **kw: called.append(1))
    result = run_stage_1x(state)
    assert result.status == "skipped"
    assert not called


# --- item 3 part 2b: the live math-sanity dispatch in the stage-1x loop ------
#
# The extractor is a REUSED Think-tier dispatch (r2c-method-analyzer) that
# writes structured claims to a per-entry scratch file; the deterministic
# checker refutes them. Both the explainer and the extractor route through
# `_dispatch_with_scope_check`, so the fakes below branch on `agent`.

NO_WRITE = object()  # extractor sentinel: simulate a no-write Think turn


def _refutable_entry():
    """An entry whose intuition states a monotone claim the checker refutes."""
    return {
        "what": "Computes the ranking score from the model's current "
                "predictions over the unlabeled pool of candidate points.",
        "why_novel": "The paper frames this as the load-bearing step and "
                     "derives the score's monotonic behavior explicitly.",
        "intuition": "The score rises monotonically as the input grows MONO_BAD.",
    }


# A quote that is a byte-substring of _refutable_entry()['intuition'].
_REFUTABLE_QUOTE = "The score rises monotonically as the input grows"
# A quote that is a byte-substring of GOOD_ENTRY['intuition'].
_GOOD_QUOTE = "surfaces the labels worth buying"


def _refuting_monotone_claims():
    # lhs = -x is DECREASING, so the monotone_increasing assertion is refuted.
    return {"claims": [{
        "claim_type": "monotonicity", "quote": _REFUTABLE_QUOTE, "formulas": [],
        "formal": {"lhs": "-x", "relation": "monotone_increasing",
                   "variables": [{"name": "x", "domain": [0, 1]}]}}]}


def _dual_fake(explainer_entry=lambda eid: dict(GOOD_ENTRY),
               claims_factory=lambda eid, prompt: {"claims": []}):
    """Fake `_dispatch_with_scope_check` handling BOTH dispatched agents.

    `explainer_entry(eid)` -> the sidecar entry the explainer writes.
    `claims_factory(eid, prompt)` -> the extractor's claim object (or NO_WRITE
    to simulate a completed-but-no-write turn; raise to simulate a dispatch
    failure)."""
    calls = {"explainer": [], "analyzer": []}

    def fake(*, state, agent, prompt, timeout_s, **kwargs):
        path = Path(PART_PATH_RE.search(prompt).group(1))
        if agent == "r2c-method-explainer":
            calls["explainer"].append(path.name)
            ids = _ids_from_prompt(prompt)
            path.write_text(json.dumps({
                "schema_version": "1.0.0",
                "explanations": {i: explainer_entry(i) for i in ids}}))
        else:  # r2c-method-analyzer, reused as the claim extractor
            eid = path.stem
            calls["analyzer"].append(eid)
            obj = claims_factory(eid, prompt)
            if obj is NO_WRITE:
                return
            path.write_text(json.dumps(obj))
    return fake, calls


def _events(state):
    p = state.paths.pipeline_dir / "run_events.jsonl"
    return p.read_text(encoding="utf-8") if p.is_file() else ""


def test_math_refuted_requeues_then_clean_run_keeps_label(tmp_path, monkeypatch):
    """Design test 7 (label invariance): a refuted claim that RETRIES clean
    lands the SAME completed/no-pending outcome as a run with no findings —
    only cap-exhausted pending markers degrade."""
    state = _prepare(tmp_path)
    attempts = {"eq-2": 0}

    def explainer_entry(eid):
        if eid == "eq-2":
            attempts[eid] += 1
            if attempts[eid] == 1:
                return _refutable_entry()  # attempt 0: refutable prose
        return dict(GOOD_ENTRY)             # retry: clean prose

    def claims_factory(eid, prompt):
        # The extractor only emits the refuting claim while the refutable prose
        # is in front of it; the clean retry entry yields nothing to refute.
        if "MONO_BAD" in prompt:
            return _refuting_monotone_claims()
        return {"claims": []}

    fake, calls = _dual_fake(explainer_entry, claims_factory)
    monkeypatch.setattr(run_pipeline, "_dispatch_with_scope_check", fake)
    result = run_stage_1x(state)

    assert result.status == "completed", result.notes
    md = (state.paths.run_dir / "METHOD.md").read_text()
    assert "explanation pending" not in md
    sidecar = json.loads((state.paths.pipeline_dir /
                          "method_explanations.json").read_text())
    assert "eq-2" in sidecar["explanations"]  # retried clean, not dropped
    # A refutation event fired for the attempt-0 counterexample.
    assert "math_claim_refuted" in _events(state)
    # eq-2 was re-extracted (prose changed); clean entries were cache-reused.
    assert calls["analyzer"].count("eq-2") == 2


def test_retry_dispatch_prompt_carries_the_counterexample(tmp_path,
                                                          monkeypatch):
    """The requeue retry is informed: the re-dispatch prompt for a refuted
    entry's part carries the refutation message (counterexample included),
    and a plain validator rejection rides the same channel. The initial
    dispatch prompts stay pristine."""
    state = _prepare(tmp_path)
    explainer_prompts: list[str] = []
    attempts = {"eq-2": 0}

    def explainer_entry(eid):
        if eid == "eq-2":
            attempts[eid] += 1
            if attempts[eid] == 1:
                return _refutable_entry()
        if eid == "eq-3" and attempts["eq-2"] == 1:
            return {**GOOD_ENTRY, "intuition": "Too short."}
        return dict(GOOD_ENTRY)

    def claims_factory(eid, prompt):
        return _refuting_monotone_claims() if "MONO_BAD" in prompt \
            else {"claims": []}

    inner, _ = _dual_fake(explainer_entry, claims_factory)

    def capturing(*, state, agent, prompt, timeout_s, **kwargs):
        if agent == "r2c-method-explainer":
            explainer_prompts.append(prompt)
        return inner(state=state, agent=agent, prompt=prompt,
                     timeout_s=timeout_s, **kwargs)

    monkeypatch.setattr(run_pipeline, "_dispatch_with_scope_check", capturing)
    result = run_stage_1x(state)

    assert result.status == "completed", result.notes
    # Five single-equation initial parts; under per-equation parts the
    # refuted entry (eq-2) and the validator-rejected entry (eq-3) live in
    # SEPARATE parts, so each offending part gets its own informed retry.
    initial, retries = explainer_prompts[:5], explainer_prompts[5:]
    assert len(retries) == 2
    for p in initial:
        assert "previous attempt rejected" not in p
    math_retry = next(p for p in retries
                      if "refuted by a numeric counterexample" in p)
    assert "previous attempt rejected" in math_retry
    assert _REFUTABLE_QUOTE in math_retry
    plain_retry = next(p for p in retries if "eq-3.intuition" in p)
    assert "previous attempt rejected" in plain_retry


def test_math_refuted_persists_to_pending_when_retry_still_refuted(
        tmp_path, monkeypatch):
    """A claim still refuted after the retry cap renders the existing pending
    marker (degrade), exactly like a validator rejection that never clears."""
    state = _prepare(tmp_path)

    def explainer_entry(eid):
        return _refutable_entry() if eid == "eq-2" else dict(GOOD_ENTRY)

    def claims_factory(eid, prompt):
        return _refuting_monotone_claims() if "MONO_BAD" in prompt \
            else {"claims": []}

    fake, _ = _dual_fake(explainer_entry, claims_factory)
    monkeypatch.setattr(run_pipeline, "_dispatch_with_scope_check", fake)
    result = run_stage_1x(state)

    assert result.status == "degraded"
    assert "eq-2" in result.notes
    sidecar = json.loads((state.paths.pipeline_dir /
                          "method_explanations.json").read_text())
    assert "eq-2" not in sidecar["explanations"]  # popped -> pending
    assert "eq-5" in sidecar["explanations"]       # untouched entries stay
    md = (state.paths.run_dir / "METHOD.md").read_text()
    assert "explanation pending" in md


def test_not_checkable_disclosed_never_degrades(tmp_path, monkeypatch):
    """not_checkable claims are counted for REPORT.md disclosure and never
    block: the run completes and the sidecar carries the count."""
    state = _prepare(tmp_path)

    def claims_factory(eid, prompt):
        if eid == "eq-1":
            # A free variable with no declared domain => not_checkable.
            return {"claims": [{
                "claim_type": "monotonicity", "quote": _GOOD_QUOTE,
                "formulas": [], "formal": {
                    "lhs": "x", "relation": "monotone_increasing",
                    "variables": []}}]}
        return {"claims": []}

    fake, _ = _dual_fake(claims_factory=claims_factory)
    monkeypatch.setattr(run_pipeline, "_dispatch_with_scope_check", fake)
    result = run_stage_1x(state)

    assert result.status == "completed", result.notes
    assert "explanation pending" not in \
        (state.paths.run_dir / "METHOD.md").read_text()
    sidecar = json.loads((state.paths.pipeline_dir /
                          "method_explanations.json").read_text())
    assert sidecar["math_sanity"]["not_checkable"] >= 1


def test_dropped_claim_emits_event_and_does_not_reject(tmp_path, monkeypatch):
    """A claim whose quote is not a byte-substring of the entry is dropped by
    the anti-hallucination floor: a math_claim_dropped event fires and the
    entry is NOT rejected."""
    state = _prepare(tmp_path)

    def claims_factory(eid, prompt):
        if eid == "eq-3":
            return {"claims": [{
                "claim_type": "monotonicity",
                "quote": "this sentence is nowhere in the entry prose",
                "formulas": [], "formal": {
                    "lhs": "-x", "relation": "monotone_increasing",
                    "variables": [{"name": "x", "domain": [0, 1]}]}}]}
        return {"claims": []}

    fake, _ = _dual_fake(claims_factory=claims_factory)
    monkeypatch.setattr(run_pipeline, "_dispatch_with_scope_check", fake)
    result = run_stage_1x(state)

    assert result.status == "completed", result.notes
    sidecar = json.loads((state.paths.pipeline_dir /
                          "method_explanations.json").read_text())
    assert "eq-3" in sidecar["explanations"]  # a dropped claim never rejects
    assert "math_claim_dropped" in _events(state)


def test_extractor_no_write_is_best_effort(tmp_path, monkeypatch):
    """A completed-but-no-write extractor turn degrades to 'no claims' — the
    stage never halts and the entry ships."""
    state = _prepare(tmp_path)
    fake, _ = _dual_fake(claims_factory=lambda eid, prompt: NO_WRITE)
    monkeypatch.setattr(run_pipeline, "_dispatch_with_scope_check", fake)
    result = run_stage_1x(state)
    assert result.status == "completed", result.notes
    assert "explanation pending" not in \
        (state.paths.run_dir / "METHOD.md").read_text()


def test_extractor_dispatch_failure_is_best_effort(tmp_path, monkeypatch):
    """An extractor dispatch that raises degrades to 'no claims'; the explainer
    parts still land and the stage completes."""
    state = _prepare(tmp_path)

    def raising(eid, prompt):
        raise OpencodeClientError("session busy")

    fake, _ = _dual_fake(claims_factory=raising)
    monkeypatch.setattr(run_pipeline, "_dispatch_with_scope_check", fake)
    result = run_stage_1x(state)
    assert result.status == "completed", result.notes
    assert "explanation pending" not in \
        (state.paths.run_dir / "METHOD.md").read_text()


def test_report_issue_table_surfaces_not_checkable_count(tmp_path):
    """The REPORT.md issues table discloses the not-machine-checkable count
    from the sidecar without treating it as a defect."""
    import render_run_report

    run_dir = tmp_path / "run"
    (run_dir / ".pipeline").mkdir(parents=True)
    (run_dir / "METHOD.md").write_text("# Method\n", encoding="utf-8")
    (run_dir / ".pipeline" / "method_explanations.json").write_text(
        json.dumps({"explanations": {"eq-1": dict(GOOD_ENTRY)},
                    "math_sanity": {"refuted": 0, "consistent": 1,
                                    "not_checkable": 2, "dropped": 0}}),
        encoding="utf-8")

    summary = render_run_report._issue_summary(run_dir)
    assert "2 mechanism claims were not machine-checkable" in summary
    assert "disclosure only" in summary


# §6b (failure-path spec): halts that fire BEFORE stage 1.x must still ship
# METHOD.md when its inputs exist — the 2026-07-02 sweep's ICRA21_HICA
# feasibility block and SRL gap-path halt both delivered no METHOD.md
# although the decomposition they needed was on disk.

def _halt_result(stage_id="stage_1"):
    return run_pipeline.StageResult(
        status="halted", stage_id=stage_id, notes="feasibility gate blocked")


def test_halt_generates_method_md_when_inputs_exist(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    state.paths.paper_map.write_text(
        json.dumps({"schema_version": "1.0.0", "elements": []}),
        encoding="utf-8")
    calls = []

    def fake_stage_1x(s):
        calls.append(True)
        (s.paths.run_dir / "METHOD.md").write_text("# Method\n",
                                                   encoding="utf-8")
        return run_pipeline.StageResult(
            status="completed", stage_id="stage_1x", notes="test")

    monkeypatch.setattr(run_pipeline, "run_stage_1x", fake_stage_1x)

    run_pipeline._ensure_method_md_on_halt(state, _halt_result())

    assert calls == [True]
    assert (state.paths.run_dir / "METHOD.md").is_file()


def test_halt_skips_method_md_when_already_present(tmp_path, monkeypatch):
    state = make_state(tmp_path / "run")
    (state.paths.run_dir / "METHOD.md").write_text("# Existing\n",
                                                   encoding="utf-8")
    monkeypatch.setattr(
        run_pipeline, "run_stage_1x",
        lambda s: (_ for _ in ()).throw(AssertionError("must not re-run")))

    run_pipeline._ensure_method_md_on_halt(state, _halt_result("stage_2x"))

    assert (state.paths.run_dir / "METHOD.md").read_text(
        encoding="utf-8") == "# Existing\n"


def test_halt_skips_method_md_when_inputs_missing(tmp_path, monkeypatch):
    # A halt that precedes the decomposition (e.g., decomposer dispatch
    # failure) has nothing to explain from — package without METHOD.md.
    state = make_state(tmp_path / "run")
    monkeypatch.setattr(
        run_pipeline, "run_stage_1x",
        lambda s: (_ for _ in ()).throw(AssertionError("must not run")))

    run_pipeline._ensure_method_md_on_halt(state, _halt_result())

    assert not (state.paths.run_dir / "METHOD.md").exists()


def test_halt_method_md_failure_never_masks_packaging(tmp_path, monkeypatch):
    # run_stage_1x never halts by design, but if it ever RAISES on the halt
    # path, the packaging must proceed (the helper swallows and logs).
    state = make_state(tmp_path / "run")
    state.paths.paper_map.write_text(
        json.dumps({"schema_version": "1.0.0", "elements": []}),
        encoding="utf-8")
    monkeypatch.setattr(
        run_pipeline, "run_stage_1x",
        lambda s: (_ for _ in ()).throw(RuntimeError("boom")))

    run_pipeline._ensure_method_md_on_halt(state, _halt_result())  # no raise

    assert not (state.paths.run_dir / "METHOD.md").exists()
