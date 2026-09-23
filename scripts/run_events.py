#!/usr/bin/env python3
"""Append-only event log and run-lock helpers for the R2C driver.

This module is intentionally independent from `run_pipeline.py` so tests can
exercise event append/replay and lock behavior without importing the full
driver or contacting opencode.
"""

from __future__ import annotations

import json
import os
import socket
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"
EVENT_LOG_NAME = "run_events.jsonl"
EVENT_LOCK_NAME = "run_events.jsonl.lock"
RUN_LOCK_NAME = "_lock"
RUN_LOCK_OWNER_NAME = "owner.json"

EVENT_TYPES = {
    "run_started",
    "run_finished",
    "run_lock_acquired",
    "run_lock_released",
    "stage_started",
    "stage_completed",
    "stage_skipped",
    "stage_halted",
    "stage_degraded",
    "agent_dispatch_started",
    "agent_dispatch_completed",
    "agent_dispatch_failed",
    "validation_passed",
    "validation_failed",
    # A trusted architecture validator reported a coverage/setup limitation
    # owned by the pipeline. This is separate from validation_failed so an
    # audit can prove that no producer retry was spent on pipeline grammar.
    "pipeline_validation_issue",
    "smoke_passed",
    "smoke_failed",
    "contract_halted",
    # The judge layer was blind in the events log until 2026-06-10: the
    # driver emits this on every appended judge decision, and the missing
    # registry entry produced `[events][append_failed]` on three separate
    # live runs (M-LOAD-1 matrix, both GBALD halts).
    "judge_decision_recorded",
    # B-13 change 2: a manifest-hashed artifact changed between the
    # final_manifest.json write and the delivered-baseline commit, so the
    # as-delivered git baseline was NOT committed. Structured record of a
    # late unattributable overwrite from any source.
    "delivered_baseline_blocked",
    # A producer dispatch that "completed" without writing a single file
    # (the truncated/no-write turn class, B-002): the dispatch layer logs
    # the detection and retries once. First seen live on the M-LOAD-1
    # matrix (analyzer twice, arch-coder once, halt-judge once).
    "truncated_turn_detected",
    # The truncated-turn sub-shape with a deterministic signature (item 23
    # / item 8 datum): the dispatch's final step burned output tokens at
    # the per-step cap with no tool call — alive and on-task, but the
    # write was never issued. Transport-class, a plain resume re-rolls it;
    # emitted ALONGSIDE truncated_turn_detected (which carries turn_shape
    # for all three shapes: cap_burn, short_empty, unmeasured).
    "cap_burn_turn_detected",
    # A successful dispatch that cap-burned one or more steps on the way
    # and then landed its write in a later step (ms3d + iDb-RRT
    # 2026-07-13: write calls whose JSON truncated at the per-step cap,
    # retried in-turn). Measurement only — the item 8 taxonomy's
    # self-recovered shape; the dispatch outcome is untouched.
    "cap_burn_recovered",
    # A dispatch timed out but its fresh work session holds assistant
    # output produced inside the window — the backend is generating, not
    # down, so the circuit breaker is left closed (ADAM 2026-07-13: a
    # false outage killed stage 2b of a healthy run).
    "dispatch_timeout_backend_alive",
    # 2026-08-24: an abandoned still-generating turn is aborted server-side
    # before the driver walks away (it otherwise keeps writing — the
    # test-generator wander re-created quarantined files post-halt).
    "abandoned_turn_aborted",
    # Cap-burn corrective resume (measurement note fix 2, approved
    # 2026-07-17): a dispatch died at the per-step output cap without
    # landing a write (turn shape cap_burn), and the driver resumed the
    # SAME work session exactly once with a short write-now nudge — the
    # stage-owned re-roll of an identical prompt reproduced identical
    # burns (SRL 2026-07-15, three consecutive method-coder burns).
    # `_attempted` fires before the resume; then exactly one of
    # `_succeeded` (an allowlisted write landed; the normal allowlist
    # flow re-checks it) or `_exhausted` (second burn / still no write /
    # resume failure — stage-owned recovery proceeds unchanged).
    "cap_burn_corrective_resume_attempted",
    "cap_burn_corrective_resume_succeeded",
    "cap_burn_corrective_resume_exhausted",
    # The proof-of-life extension ceiling fired (failure class:
    # unbounded_timeout_extension_on_generating_backend, SRL matrix row
    # 2026-07-16: a paper-fidelity-reviewer dispatch declared 1800s and
    # ran 14019s because a still-generating backend kept the blocking
    # POST fed). The dispatch wall-clock crossed the hard ceiling
    # (default 2x the declared timeout) and the wait was abandoned; the
    # error then follows the existing backend-alive timeout path into
    # stage-owned recovery.
    "dispatch_extension_ceiling_hit",
    # Stage 1's passage locator (item 23) attached candidate paper
    # passages to a quote-floor fix finding. Driver-log-only until
    # 2026-07-14, which made the feature invisible to event-stream audits.
    "quote_candidates_attached",
    "equation_quotes_reanchored",
    # The method-spec quote floors (param-glossary meaning quotes,
    # scenario-assumption evidence quotes) get the paper map's
    # deterministic repair chain (0728b bayesian-active-learning halt:
    # three render-equivalent meaning quotes — spaces inside math
    # delimiters — burned all 3 stage-1 retries). Re-anchors are
    # disclosure-light like the equation surface; adoptions on this
    # surface are loud (assumptions.md + quote_adoptions.json row with
    # surface="method_spec").
    "spec_quotes_reanchored",
    "spec_quotes_adopted",
    # Accept-the-candidate adoption (R2C-030, approved 2026-07-28): an
    # equation quote failing the verbatim floor had exactly one
    # near-perfect candidate under the adoption rule, and the paper's
    # own bytes were adopted under a logged assumption BEFORE any
    # fix-loop retry (both recorded cases proved the retry channel
    # cannot repair a defect the producer cannot perceive).
    "equation_quotes_adopted",
    # Stage 2.x provenance fix loop (R2C-033, approved 2026-07-28): a
    # provenance-probe failure was routed mechanically back to the method
    # analyzer, the bounded retry survived every guardrail (scoped-edit
    # diff, spec re-validation, deterministic re-derivation, probe
    # re-run), and a parameter's provenance changed on the way (e.g.
    # paper -> derived/assumption). One event per repaired parameter,
    # beside its assumptions.md entry — a provenance downgrade is never
    # silent.
    "params_provenance_repaired",
    # The fix loop's laundering guard fired: a provenance-repair retry
    # edited surfaces OUTSIDE the probe-named parameter(s) (on the spec
    # or on the re-derived params output) and was rejected outright, with
    # the prior bytes restored. Driver-log-only until 2026-08-03; the maintainer's
    # call promoted it to an event so event-stream audits can see a
    # producer attempting to satisfy the probes by editing what the
    # findings never named.
    "params_fix_scope_rejected",
    # Stage 3a degraded past static notebook-validation failure(s); the
    # failing bullets are persisted as the accepted baseline so later
    # revalidations (3c entry, 3c post-fix, stage-5 post-fix) halt only on
    # NEW failures (the ICRA 2026-07-13 re-litigated-degrade halt).
    "notebook_validation_baseline_recorded",
    # A notebook revalidation failed ONLY on baseline-accepted failures and
    # the run continued. Reads alongside the validation_failed event it
    # follows — without it, failed-validation-then-no-halt looks like a
    # driver bug in the event stream.
    "accepted_validation_failures_only",
    # Structural write-first (maintainer-approved 2026-07-05): the driver authors
    # a fallback smoke diagnosis after two zero-write diagnostician turns.
    "smoke_diagnosis_driver_fallback",
    # Item 29 (researcher run, 07-08): an existing run dir from an older R2C version
    # was moved aside at stage 0 (no current progress.json) so the run could
    # proceed fresh instead of failing on stale sentinels. Never a delete.
    "run_dir_archived",
    # Item 2 (07-06 audit): a nice-to-have reviewer caveat/finding that asked
    # to be documented was promoted from deferred_findings.md (which no
    # researcher reads) to an assumptions.md entry at stage-5 routing.
    "reviewer_disclosure_routed",
    # Gap-report salvage (SRL re-roll #2, 2026-07-05): a typo'd key in ONE
    # supporting-evidence entry failed the whole report's strict validation,
    # the recovery declined silently, and a correct 0.85-confidence gap
    # decision collapsed into a generic mismatch halt. The driver now drops
    # malformed evidence entries (decision untouched), preserves the raw
    # file as .rejected, and records the salvage here.
    "gap_report_salvaged",
    # Gap path: a provisional pack was authored and installed for a
    # taxonomy-gap paper (the driver emitted this unregistered from day
    # one — found 2026-07-06 when the fedavg run's install event silently
    # failed to append; the registry sweep test now pins the whole class).
    "provisional_pack_installed",
    # Gap path: a new-top-level gap claim proposed an id the committed
    # taxonomy already owns (pdfgnn Attempt 2, 2026-08-10 — the analyzer
    # missed the committed match and the pack author burned three
    # dispatches on a proposal the validator had to reject). The driver
    # skips authoring and retries stage 1 once; a repeat halts.
    "gap_claim_shadows_committed_family",
    # Smoke pre-check: the trainability gate (UB-6's rule imported at the
    # earlier enforcement point) failed. Same day-one unregistered-emit
    # class as above.
    "smoke_trainability_failed",
    # Stage 2.d dependency routing (R2C-050): a pip-unresolvable requirement
    # traced back to a producer's own import, and that producer was
    # dispatched a fix finding instead of the run dying under the
    # internal-bug story (the pdfgnn 2026-08-03 dgl halt).
    "uninstallable_dependency_routed",
    # Stage 2.a demo-data acquisition (R2C-052): a gap-path paper's own cited
    # public dataset was fetched anonymously, subsampled to demo scale, and
    # bundled into method/example_data/ with provenance. Emitted only on
    # success; failures write .pipeline/dataset_acquisition.json and log,
    # never halting the run.
    "demo_dataset_acquired",
    # The same fetch, refused (R2C-065): the bundle's realized time axis
    # cannot support the paper's own protocol arithmetic, so shipping it
    # would force the demo onto data the method is unable to run on. The
    # bundled files are deleted and the run continues without a bundle.
    "demo_dataset_refused",
    # The bundle has a time axis, but its exact source cadence cannot yet be
    # joined to the typed protocol cadence (R2C-065). The bundle is preserved;
    # this records why no feasible/infeasible claim was made.
    "demo_dataset_axis_unresolved",
    # Public bundle publication is staged and hash-bound (R2C-052). A resumed
    # Stage 2.a either accepts a complete publication, removes an unchanged
    # partial, or refuses changed output without consuming a producer retry.
    "demo_dataset_public_recovered",
    "demo_dataset_public_recovery_refused",
    # Stage 2.a offline arm (R2C-052): public acquisition could not supply a
    # usable bundle, and a supported family-owned contract either generated a
    # deterministic typed fixture or refused the family/protocol shape. Both
    # outcomes are non-fatal and remain distinct from paper-cited public data.
    "demo_dataset_fallback_generated",
    "demo_dataset_fallback_refused",
    # Demo-success verdict (design approved 2026-07-16): after a clean smoke
    # execution the deterministic pass records whether the headline demo
    # demonstrably worked. `_recorded` covers succeeded/undetermined;
    # `_failed` is the demo_failure_invisible_to_smoke class landing (a
    # delivery demoter, never a halt); `demo_kit_coverage_gap` is the
    # committed-family-without-markers finding we own.
    "demo_verdict_recorded",
    "demo_verdict_failed",
    "demo_kit_coverage_gap",
    # Fix-loop data-signal check (same design): a Stage 3.c fix dispatch
    # changed the demo-data surface and the deterministic winnability check
    # ran. `_failed` means the new data can never pass the beats-chance gate
    # (the Rethinking 2026-07-14 impossible-gate burn) and the next fix
    # dispatch is told so plainly.
    "fix_data_signal_checked",
    "fix_data_signal_failed",
    # 2026-07-06 fedavg (both attempts): the analyzer proposed ids in the
    # canonical taxonomy vocabulary (uppercase family codes) while the gap
    # schema and proposal pipeline speak lowercase legacy ids. The driver
    # translates the documented equivalence deterministically (family-code
    # parent -> new top level; judgment unchanged), preserves the raw file
    # as .rejected, and records the translation here.
    "gap_report_vocabulary_translated",
    # Partial-delivery producer #1 (maintainer-approved 2026-07-05): the
    # feasibility gate records not-replicable SUPPORTING elements as
    # stubbed elements with spec-time work orders instead of proceeding
    # silently past them.
    "gate_stubs_recorded",
    "artifact_candidate_recorded",
    "artifact_promotion_failed",
    "artifact_promoted",
    # Stage 1 Think-class producer wrote a halt signal, reconsidered in the
    # same dispatch, and then completed the canonical artifact.  The canonical
    # is sent through its normal validator and the superseded sidecar is
    # removed; the event preserves that precedence decision for replay.
    "artifact_halt_sidecar_superseded",
    "dispatch_reconciled",
    # Slice 1.5: the probe battery runs at delivery time and its verdicts
    # (with unresolved critical/important fidelity findings) set the
    # two-tier label recorded in final_manifest.json.
    "delivery_label_derived",
    # A scoped dispatch that timed out (DispatchTimeout/ServerUnreachable) but
    # had already landed its canonical artifact, soft-recovered after reverting
    # any out-of-scope side effects. Emitted by _recover_scoped_dispatch_after_error
    # since that path was added, but never registered — so it silently failed to
    # persist (`[events][append_failed]`). Registered 2026-06-15.
    "agent_dispatch_recovered",
    # Graceful out-of-scope-write recovery, branch A (2026-06-15): an opted-in
    # agent wrote outside its scope AND its output was invalid even after the
    # strays were reverted (a wander). The driver re-dispatches the same agent
    # once with a corrective preamble; `_attempted` fires before the retry (and
    # is what the whole-run per-agent cap counts), then exactly one of
    # `_succeeded` (continue) or `_exhausted` (cap hit / still invalid / transport
    # error → halt). See the out of scope recovery design note (internal, not shipped).
    "out_of_scope_corrective_redispatch_attempted",
    "out_of_scope_corrective_redispatch_succeeded",
    "out_of_scope_corrective_redispatch_exhausted",
    # Repo-scope dispatch write boundary (2026-07-22): a dispatch wrote
    # OUTSIDE the run dir entirely (git-porcelain + top-level listing diff,
    # see repo_scope_boundary.py). New strays are quarantined into
    # `<run_dir>/.pipeline/quarantine/<dispatch-label>/`; modified/deleted
    # tracked files are listed but never restored. Always followed by an
    # OutOfScopeWritesError halt.
    "repo_scope_violation_detected",
    # Stage 3.a re-derives requirements.txt to also cover the rendered
    # notebook's third-party imports (it is built at Stage 2.d from the method
    # package alone, before the notebook exists). Emitted only when a notebook-
    # only dependency was actually added. Root-caused 2026-06-15 (the F001
    # sklearn-missing-from-requirements draft demotion).
    "requirements_reconciled",
    # Explanation math-sanity pass (item 3 part 2b): stage 1x refuted a
    # METHOD.md mechanism claim with a numeric counterexample (rejects the
    # entry into the retry loop), or dropped an extracted claim at the
    # byte-substring anti-hallucination floor. Categorized in fleet_state.
    "math_claim_refuted",
    "math_claim_dropped",
    # Fix-loop third option (item 7): at a terminal give-up point the
    # deterministic G1-G5 gates either fire a one-per-run stub conversion
    # (the failing component ships as a raising stub, delivery is PARTIAL),
    # decline (behavior stays today's degrade), or the conversion rolls
    # back because the stubbed notebook still failed the smoke re-gate.
    "fix_loop_stub_conversion",
    "fix_loop_stub_gates_unmet",
    "fix_loop_stub_regate_failed",
    # Dispatch-recovery ladder rung 3 (item 8): the one bounded transport
    # re-dispatch per run. The retry event lands BEFORE the second attempt
    # (both dispatch ids, cooldown waited, liveness evidence); the
    # skipped-live-session event records the T1b interim refusing to race a
    # still-generating work session. agent_dispatch_failed is ALWAYS
    # appended for the underlying failure — recovery never erases it.
    "dispatch_retry_after_transport",
    "dispatch_retry_skipped_live_session",
    # Insight shadow hook (Block 7 continuation, 2026-07-20): the one
    # namespaced record the best-effort generic-understanding shadow pass
    # appends at run end — outcome (passed/partial/invalid/absent/timeout/
    # exception), semantic-review sub-status, and materialization stats.
    # Shadow outcomes are observable ONLY through this event and the four
    # namespaced .pipeline artifacts; they never change delivery or state.
    "insight_shadow_recorded",
}

STAGE_TERMINAL_EVENTS = {
    "stage_completed": "completed",
    "stage_skipped": "skipped",
    "stage_halted": "halted",
    "stage_degraded": "degraded",
}


class EventLogError(RuntimeError):
    """Raised when an event log is malformed or cannot be written."""


class RunDirectoryLockError(RuntimeError):
    """Raised when another driver owns the run-directory lock."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def pipeline_dir_for(run_dir_or_pipeline_dir: Path) -> Path:
    path = Path(run_dir_or_pipeline_dir)
    return path if path.name == ".pipeline" else path / ".pipeline"


def event_log_path(pipeline_dir: Path) -> Path:
    return pipeline_dir / EVENT_LOG_NAME


def event_lock_path(pipeline_dir: Path) -> Path:
    return pipeline_dir / EVENT_LOCK_NAME


def _pid_alive(pid: int) -> bool:
    """Whether a local process id is running.

    Signal 0 is the standard liveness probe: it performs the permission and
    existence checks without delivering anything. A PermissionError means the
    process exists and belongs to another user, which is still alive.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        # Anything unrecognised fails toward "alive", so an unclear answer
        # refuses the reclaim rather than letting two drivers into one run.
        return True
    return True


def run_lock_path(pipeline_dir: Path) -> Path:
    return pipeline_dir / RUN_LOCK_NAME


def run_lock_owner_path(pipeline_dir: Path) -> Path:
    return run_lock_path(pipeline_dir) / RUN_LOCK_OWNER_NAME


def read_run_lock_owner(pipeline_dir: Path) -> dict | None:
    """The run lock's owner record, or None when absent or unreadable."""
    try:
        data = json.loads(
            run_lock_owner_path(pipeline_dir).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def run_lock_owner_is_provably_gone(pipeline_dir: Path) -> bool:
    """Whether the run lock's owning process is conclusively dead (R2C-076).

    The single staleness rule. Says yes ONLY when:

    - the owner record is readable and names a positive integer pid, and
    - the owner's host is this host, since a pid on another machine says
      nothing about liveness here, and
    - that pid is not running.

    Anything else says no and keeps the honest failure. A missing or
    unparseable owner record is NOT stale: a lock directory with no owner is
    more likely a write that has not landed yet than an abandoned run, and
    racing it would let two drivers into one run dir.

    Read-only by construction, because it has two consumers that must agree.
    The lock's own acquire path pairs it with the takeover, and the --fresh
    launch path asks it whether archiving the run dir is safe. When the rule
    lived only inside acquire, --fresh kept its own bare existence check and
    the two disagreed: a dead lock made --fresh skip its archive and then let
    the reclaim through, so a clean roll silently started in the previous
    roll's directory on top of its stale sentinels.
    """
    owner = read_run_lock_owner(pipeline_dir)
    if owner is None:
        return False
    pid = owner.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    if str(owner.get("host") or "") != socket.gethostname():
        return False
    return not _pid_alive(pid)


def run_lock_blocks_launch(pipeline_dir: Path) -> bool:
    """Whether a run lock present here will refuse a new launch.

    False both when there is no lock and when the lock is reclaimable, so a
    caller can ask this before deciding to touch the run directory.
    """
    if not run_lock_path(pipeline_dir).exists():
        return False
    return not run_lock_owner_is_provably_gone(pipeline_dir)


def describe_run_lock_owner(pipeline_dir: Path) -> str:
    owner = read_run_lock_owner(pipeline_dir)
    if owner is None:
        return "owner record unreadable"
    return (f"held by pid {owner.get('pid')} on "
            f"{owner.get('host')} since {owner.get('created_at')}")


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class _AppendLock(AbstractContextManager):
    """Small lock for sequence-number-safe event appends.

    Uses a lock directory rather than fcntl so tests can assert contention in
    the same process and the implementation stays portable enough for local
    developer machines.
    """

    def __init__(self, pipeline_dir: Path):
        self.pipeline_dir = pipeline_dir
        self.path = event_lock_path(pipeline_dir)

    def __enter__(self):
        self.pipeline_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.path.mkdir()
        except FileExistsError as exc:
            raise EventLogError(f"event log is locked at {self.path}") from exc
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.path.rmdir()
        except FileNotFoundError:
            pass
        return False


def build_event(
    *,
    sequence: int,
    event_type: str,
    run_id: str,
    stage_id: str | None = None,
    stage_label: str | None = None,
    stage_category: str | None = None,
    status: str | None = None,
    summary: str | None = None,
    artifacts: list[str] | None = None,
    details: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    if event_type not in EVENT_TYPES:
        raise EventLogError(f"unknown event_type {event_type!r}")
    if sequence < 1:
        raise EventLogError("event sequence must be >= 1")
    event: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sequence": sequence,
        "event_type": event_type,
        "run_id": run_id,
        "timestamp": timestamp or utc_now_iso(),
    }
    if stage_id is not None:
        event["stage_id"] = stage_id
    if stage_label is not None:
        event["stage_label"] = stage_label
    if stage_category is not None:
        event["stage_category"] = stage_category
    if status is not None:
        event["status"] = status
    if summary is not None:
        event["summary"] = summary
    if artifacts:
        event["artifacts"] = list(artifacts)
    if details:
        event["details"] = details
    return event


def load_events(pipeline_dir: Path) -> list[dict[str, Any]]:
    path = event_log_path(pipeline_dir)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                raise EventLogError(f"{path} line {line_no} is blank")
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EventLogError(f"{path} line {line_no} is invalid JSON: {exc}") from exc
            if not isinstance(event, dict):
                raise EventLogError(f"{path} line {line_no} must be a JSON object")
            if event.get("schema_version") != SCHEMA_VERSION:
                raise EventLogError(
                    f"{path} line {line_no} has schema_version "
                    f"{event.get('schema_version')!r}; expected {SCHEMA_VERSION!r}"
                )
            if event.get("sequence") != line_no:
                raise EventLogError(
                    f"{path} line {line_no} has sequence "
                    f"{event.get('sequence')!r}; expected {line_no}"
                )
            event_type = event.get("event_type")
            if event_type not in EVENT_TYPES:
                raise EventLogError(f"{path} line {line_no} has unknown event_type {event_type!r}")
            events.append(event)
    return events


def append_event(
    pipeline_dir: Path,
    *,
    event_type: str,
    run_id: str,
    stage_id: str | None = None,
    stage_label: str | None = None,
    stage_category: str | None = None,
    status: str | None = None,
    summary: str | None = None,
    artifacts: list[str] | None = None,
    details: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    with _AppendLock(pipeline_dir):
        events = load_events(pipeline_dir)
        event = build_event(
            sequence=len(events) + 1,
            event_type=event_type,
            run_id=run_id,
            stage_id=stage_id,
            stage_label=stage_label,
            stage_category=stage_category,
            status=status,
            summary=summary,
            artifacts=artifacts,
            details=details,
            timestamp=timestamp,
        )
        with event_log_path(pipeline_dir).open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(event))
            handle.write("\n")
    # History mirror (write-through, mirror-only): after the local append
    # succeeds, tee the same canonical line into r2c_runs/_history/. The
    # mirror can never block or fail an append — run_history contains its
    # own failures, and this guard is the belt over those braces.
    try:
        from run_history import mirror_event_line

        mirror_event_line(pipeline_dir, canonical_json(event))
    except Exception:
        pass
    return event


def current_invocation_slice(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Events belonging to the most recent driver invocation.

    Every driver invocation appends a `run_started` event unconditionally, so
    the current invocation's segment is everything from the LAST `run_started`
    onward (inclusive). Researcher-facing surfaces that disclose transient,
    per-attempt facts (e.g. REPORT.md's recovered-transport count) must scope
    to this slice: the append-only log spans every leg of a resumed run, and
    counting the whole file attributes an earlier dead attempt's recoveries to
    the current delivery (SRL 2026-07-15: a 07-13 rung-3 recovery rode into
    the 07-15 delivery's REPORT).

    A log with no `run_started` boundary is not something the driver produces,
    but a truncated or legacy log must degrade conservatively: return the full
    list unchanged (whole-log behavior) rather than an empty slice that could
    hide real events. Callers needing per-invocation identity should read the
    `run_started` event's provenance details; this helper only segments.
    """
    for i in range(len(events) - 1, -1, -1):
        if isinstance(events[i], dict) and events[i].get("event_type") == "run_started":
            return events[i:]
    return list(events)


def replay_run_dir(run_dir_or_pipeline_dir: Path) -> dict[str, Any]:
    pipeline_dir = pipeline_dir_for(Path(run_dir_or_pipeline_dir))
    events = load_events(pipeline_dir)
    projection: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": events[0]["run_id"] if events else None,
        "last_sequence": 0,
        "stages": {},
        "validations": [],
        "dispatches": [],
        "smoke": [],
        "artifacts": [],
    }
    for event in events:
        projection["last_sequence"] = event["sequence"]
        event_type = event["event_type"]
        stage_id = event.get("stage_id")
        if isinstance(stage_id, str):
            row = projection["stages"].setdefault(
                stage_id,
                {
                    "stage_id": stage_id,
                    "stage_label": event.get("stage_label", stage_id),
                    "stage_category": event.get("stage_category"),
                    "status": "pending",
                    "started_at": None,
                    "completed_at": None,
                    "last_event_sequence": None,
                },
            )
            row["stage_label"] = event.get("stage_label", row["stage_label"])
            row["stage_category"] = event.get("stage_category", row.get("stage_category"))
            row["last_event_sequence"] = event["sequence"]
            if event_type == "stage_started":
                row["status"] = "running"
                row["started_at"] = event["timestamp"]
                row["completed_at"] = None
            elif event_type in STAGE_TERMINAL_EVENTS:
                row["status"] = STAGE_TERMINAL_EVENTS[event_type]
                row["completed_at"] = event["timestamp"]
        if event_type in {"validation_passed", "validation_failed", "contract_halted"}:
            projection["validations"].append(event)
        elif event_type.startswith("agent_dispatch_") or event_type == "dispatch_reconciled":
            projection["dispatches"].append(event)
        elif event_type.startswith("smoke_"):
            projection["smoke"].append(event)
        elif event_type.startswith("artifact_"):
            projection["artifacts"].append(event)
    projection["stages"] = list(projection["stages"].values())
    return projection


class RunDirectoryLock(AbstractContextManager):
    """Atomic run-directory lock rooted at `<run>/.pipeline/_lock`."""

    def __init__(self, pipeline_dir: Path, *, run_id: str):
        self.pipeline_dir = pipeline_dir
        self.run_id = run_id
        self.path = run_lock_path(pipeline_dir)
        self.owner_path = run_lock_owner_path(pipeline_dir)
        self.acquired = False

    def acquire(self) -> "RunDirectoryLock":
        self.pipeline_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.path.mkdir()
        except FileExistsError as exc:
            if not self._reclaim_if_stale():
                raise RunDirectoryLockError(
                    f"run directory is already locked at {self.path} "
                    f"({self._owner_description()})"
                ) from exc
        owner = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "created_at": utc_now_iso(),
        }
        try:
            self.owner_path.write_text(json.dumps(owner, indent=2) + "\n", encoding="utf-8")
            self.acquired = True
        except Exception:
            try:
                self.owner_path.unlink()
            except OSError:
                pass
            try:
                self.path.rmdir()
            except OSError:
                pass
            raise
        return self

    def _owner_description(self) -> str:
        return describe_run_lock_owner(self.pipeline_dir)

    def _reclaim_if_stale(self) -> bool:
        """Take over a lock whose owning process is provably gone (R2C-076).

        A run that dies without releasing (killed mid-roll, crashed, machine
        rebooted) used to block its paper forever: the lock recorded the owning
        pid and host and never read them back, so the only cure was deleting
        the directory by hand. That is a footgun on exactly the path where
        someone has just had to stop a run.

        The staleness rule itself lives in `run_lock_owner_is_provably_gone`,
        because the --fresh launch path has to reach the same verdict. This
        method is the takeover half: read-only decision, then the mutation.
        """
        if not run_lock_owner_is_provably_gone(self.pipeline_dir):
            return False
        try:
            self.owner_path.unlink()
        except OSError:
            return False
        # The directory itself stays; we are taking it over rather than
        # removing and recreating it, which would open a window for a second
        # driver to mkdir into.
        return True

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            self.owner_path.unlink()
        except FileNotFoundError:
            pass
        try:
            self.path.rmdir()
        except FileNotFoundError:
            pass
        self.acquired = False

    def __enter__(self) -> "RunDirectoryLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


