"""R2C pipeline driver — Python replacement for the legacy LLM orchestrator
(the archived v2 orchestrator spec (internal, not shipped); migration rationale at
the orchestrator redesign plan (internal, not shipped)).

The driver dispatches producer / reviewer agents via opencode's HTTP API and
runs validators as subprocesses. It has no Edit tool in scope: it never
modifies files under `<RUN_DIR>/` itself, which makes the orchestrator's
"band-aid spiral" failure mode structurally impossible.

Phase 3 — implements Stage 0 (setup + paper parsing) and Stage 1 (analyzer +
paper_map + feasibility gate + stage-reviewer). Stages 2-5 land in subsequent
phases.

Usage:
  python3 scripts/run_pipeline.py --paper <slug-or-path> [--port 4096]
                                  [--dir <workspace>] [--stop-after stage_N]

Self-tests live in tests/test_driver_selftest.py (B-07).
"""

from __future__ import annotations

import argparse
import ast
import csv
import fnmatch
import hashlib
import json
import math
import html
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, NamedTuple

# Fail clearly on an under-version interpreter before the first-party imports
# below pull in 3.10+ features (see pyproject.toml `requires-python`). A
# 3.11-only `datetime.UTC` import in one of them once crashed a 3.10 box with a
# cryptic ImportError; this guard needs only `sys` and runs before any 3.10+
# construct, so it surfaces a clear message even on older pythons.
if sys.version_info < (3, 10):
    raise SystemExit(
        f"R2C requires Python 3.10+ but is running {sys.version.split()[0]} "
        f"({sys.executable}). Activate the project venv or use a newer python3."
    )

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pdf_parse_quality import replace_payloads_with_placeholders  # noqa: E402
from schemas.judge_decision import JudgeDecision  # noqa: E402
from schemas.stage_review_report import StageReviewReport  # noqa: E402
from scripts.arch_contract_semantics import (  # noqa: E402
    SEMANTIC_ISSUE_PREFIX,
    SemanticIssue,
    parse_semantic_issue,
)
from schemas.paradigm_gap import (  # noqa: E402
    EvidenceQuote,
    GapDecision,
    ParadigmGapReport,
)
from finalize_package_init import reconcile_requirements_for_notebook  # noqa: E402
from dispatch_templates import (  # noqa: E402
    CAP_BURN_CORRECTIVE_RESUME_NUDGE,
    SHORT_EMPTY_CORRECTIVE_RESUME_NUDGE,
    DispatchPaths,
    PACK_INSTALLED_CLASSIFICATION_BLOCK,
    FEASIBILITY_SURROGATE_REASK_TEMPLATE,
    INCREMENTAL_REREVIEW_TEMPLATE,
    JUDGE_ELEMENT_TEST_TEMPLATE,
    JUDGE_REVIEWER_TASK_TEMPLATE,
    JUDGE_TASK_TEMPLATE,
    REVIEWER_CLOSING,
    STAGE_REVIEW_RESOLUTION_REASK_TEMPLATE,
    STAGE_REVIEW_TASK_TEMPLATE,
    STAGE_REVIEWER_RETRY_TASK_TEMPLATE,
    STAGE_1_OUTPUT_MODE_WRITEABLE_PATHS,
    STAGE_TASK_SUMMARIES,
    WRITEABLE_PATHS,
    build_corrective_redispatch_preamble,
    build_dispatch_prompt,
    build_fix_mode_prompt,
    build_halt_artifact,
    build_smoke_diagnosis_prompt,
    build_write_first_retry_preamble,
    stage_1_schema_embeds,
)
from final_manifest import sha256_file, sha256_text, write_final_manifest  # noqa: E402
import halt_catalog  # noqa: E402
import render_claims_report  # noqa: E402
import dependency_readiness  # noqa: E402
import repo_scope_boundary  # noqa: E402
import run_history  # noqa: E402
import run_layout  # noqa: E402
from trusted import trusted_path, trusted_text  # noqa: E402
from dispatch_reconciliation import reconcile_candidate_dispatches  # noqa: E402
from taxonomy import load_taxonomy, registered_paradigm_ids, serves  # noqa: E402
from author_field_guide_proposal import (  # noqa: E402
    AUTHOR_AGENT as PACK_AUTHOR_AGENT,
    AuthoringError,
    author_proposal,
)
from bundle_axis_floor import (  # noqa: E402
    load_bundle_manifest,
    protocol_axis_param_shrinks,
)
from stage1_artifact_parts import (  # noqa: E402
    AssemblyResult,
    assemble_method_spec_parts,
    assemble_paper_map_parts,
    parts_newer_than_output,
)
import opencode_client  # noqa: E402
from opencode_client import (  # noqa: E402
    DEFAULT_PORT,
    DispatchResult,
    DispatchTimeout,
    OpencodeClientError,
    SessionNotFound,
    ServerUnreachable,
    abort_session,
    create_session,
    dispatch_and_wait,
    discover_session,
    health_check,
    list_agents,
    session_activity_fingerprint,
)
from route_findings import (  # noqa: E402
    cell_index_to_section,
    critical_findings,
    filter_by_severity,
    group_by_target_agent,
    important_findings,
    nice_to_have_findings,
    should_halt_stage,
    smoke_cell_to_producer,
    smoke_traceback_deepest_owned_frame,
    smoke_traceback_to_producer,
)
from run_events import (  # noqa: E402
    RunDirectoryLock,
    RunDirectoryLockError,
    append_event,
    describe_run_lock_owner,
    run_lock_blocks_launch,
)
from progress_state import (  # noqa: E402
    PROGRESS_FILE_NAME,
    is_run_dir_layout_compatible,
    utc_now_iso,
    write_progress,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_CMD = sys.executable  # use the same interpreter that ran the driver

# Stage-1 structural validator retry cap: progress-aware. Unchanged errors
# halt immediately as fixation, while changed validator signatures can use
# the same bounded budget as Stage 2/3 producer loops.
STAGE_1_VALIDATOR_RETRY_CAP = 3

# Stage-1 semantic reviewer retry cap: analyzer is heavyweight, and reviewer
# findings are less mechanical than schema errors, so keep one fix-mode retry.
STAGE_1_REVIEWER_RETRY_CAP = 1

# Stage-2/3 retry cap: standard for LLM producer + validator + stage-reviewer
# loops, AND for the smoke-gate fix loop in Stage 3.c. Empirically, when a
# fix loop hasn't converged by iteration 3 it almost never converges by 5 —
# the extra iterations just burn time before the inevitable halt. (cap=5 was
# tried briefly after the bev-distill 2026-05-14 cascade run, but the more
# durable fix to that case was per-target_agent dispatch routing in tier-3
# plus the L2 smoke-diagnostician at stage 3.c.)
STAGE_2_RETRY_CAP = 3

# Generous per-dispatch timeout: the decomposer and analyzer both routinely
# take 5-10 minutes. The reviewer is lighter. The coders sit between.
# Every value below is a BASE budget: dispatch_agent multiplies it by
# R2C_TIMEOUT_SCALE (default 3.0, see opencode_client.scaled_timeout_s)
# because these were calibrated on a self-hosted Qwen 27B.
# Recalibrated 2026-08-21 for Qwen 3.8 (serving switch 2026-08-18): every
# turn now pays a heavy reasoning tax, so measured turn durations grew
# 2-3x (first 3.8 roll: a healthy 36-part decomposer write ran 1361s of
# the old 1800s budget; the lightest stage review ran 411s of 900s). The
# heavy-producer class gets 1.5x, the 900s review class 2x — same
# false-trip protection the math extractor needed at 240s->600s in July.
DECOMPOSER_TIMEOUT_S = 2700
# 4500 (was 2700): the message POST is silent until the dispatch finishes,
# so this is a hard wall, and a healthy Qwen 3.8 stage-2x params fix
# dispatch runs 6-8 turns at ~6min each (bayesian 2026-09-01: killed at
# 2700s mid-progress, turns still landing, halt-classed transport_failure).
ANALYZER_TIMEOUT_S = 4500
ARCH_CODER_TIMEOUT_S = 2700
METHOD_CODER_TIMEOUT_S = 2700
# Per-element test generation (R2C-024): fixture-scale modules over an
# already-assembled package — lighter than method coding.
TEST_GENERATOR_TIMEOUT_S = 1800
NOTEBOOK_GEN_TIMEOUT_S = 2700
PAPER_FIDELITY_REVIEWER_TIMEOUT_S = 2700
REVIEWER_TIMEOUT_S = 1800
# Smoke-gate diagnostician (Think-class). The traceback-anchored case is
# light (~3 min live), but an anchor-less diagnosis (trainability-class:
# notebook ran clean, model silently doesn't learn) has no stack frame to
# start from and must read the notebook, params, and package before
# reasoning — a LEGITIMATE turn of that class ran ~26.5 min on 2026-07-07
# (the maintainer's mock run, casualty #10: the 900s budget halted a run whose
# diagnosis completed correctly 11 minutes after the driver gave up).
# Producer-class budget since then.
DIAGNOSTICIAN_TIMEOUT_S = 2700
# Halt-judge (Think-class). Reads validator + failing artifact +
# taxonomy/build-plan context and emits a routing decision; same budget
# as the reviewer class (2026-08-21 Qwen 3.8 recalibration).
JUDGE_TIMEOUT_S = 1800

# Stage 5 outer-loop cap per the design plan: 3 iterations of
# (review → route critical → fix-mode → re-validate → smoke → re-review).
STAGE_5_OUTER_CAP = 3
STAGE_4_RETRY_CAP = 1  # cap=1 fix-mode retry on validate_review_report.py failure

# Smoke-gate budgets. `--timeout` is the per-cell limit; the subprocess
# `timeout=` is the wall-clock ceiling for the whole notebook. A *smoke* run is
# meant to be fast — a healthy run finishes well under a minute — so these are
# deliberately tight: their job is to fail a runaway/mis-scaled cell quickly,
# NOT to accommodate a full-scale run. The per-cell budget still leaves generous
# headroom for a first-run dataset download plus a CPU train-from-scratch round.
# (A flat 1200s per-cell previously let a mis-scaled acquisition loop burn ~20
# min before the gate gave up; retried up to the cap, that was ~an hour of
# wasted wall-clock. The deriver has no cost model, so the smoke gate is where a
# scale blowup actually surfaces — make it surface fast.)
SMOKE_CELL_TIMEOUT_S = 180
SMOKE_WALLCLOCK_TIMEOUT_S = 600

# ---------------------------------------------------------------------------
# Defensive startup check — refuse to run inside an opencode bash tool
# ---------------------------------------------------------------------------


def _detect_session_deadlock_risk() -> tuple[bool, str]:
    """Detect whether this run_pipeline.py invocation will deadlock against
    the driver's own producer-agent POSTs back into the opencode session
    that launched it.

    Three signals must all hold for a positive detection:
      1. No controlling TTY on stdin/stdout/stderr (the bash tool inherits
         pipes, not a TTY).
      2. An `opencode` process exists in the ancestor chain.
      3. stdout is neither a regular file nor `/dev/null` (i.e., a pipe).

    Signal #3 distinguishes the deadlock case from the legitimate
    `/r2c-run` slash-command path, which redirects stdout to
    `/tmp/r2c-run-$$.log` — a regular file. The foreground bash tool's
    stdout is a pipe inherited from opencode.

    Returns (is_risk, diagnostic) — best-effort; probe failures fall
    through with is_risk=False (we'd rather miss a real deadlock case
    than block a legitimate run on a false positive). The instructional
    fix lives in README.md ("How it works").
    """
    import stat as _stat  # local: only needed here

    # Escape hatch for CI/unusual environments
    if os.environ.get("R2C_SKIP_DEADLOCK_CHECK"):
        return False, "R2C_SKIP_DEADLOCK_CHECK set"

    # Signal 1: any TTY → interactive, not a bash-tool subprocess
    try:
        if any(os.isatty(fd) for fd in (0, 1, 2)):
            return False, "tty present on stdin/stdout/stderr"
    except OSError:
        return False, "tty probe failed"

    # Windows: `ps` probe not implemented
    if sys.platform == "win32":
        return False, "windows: probe not implemented"

    # Signal 2: walk ancestor chain looking for an opencode process
    found_opencode_pid: int | None = None
    pid = os.getppid()
    seen: set[int] = set()
    for _ in range(20):  # bounded
        if pid in seen or pid <= 1:
            break
        seen.add(pid)
        try:
            proc = subprocess.run(
                ["ps", "-p", str(pid), "-o", "comm=,ppid="],
                capture_output=True, text=True, timeout=2,
            )
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return False, "ps probe failed"
        if proc.returncode != 0:
            break
        line = proc.stdout.strip()
        if not line:
            break
        # comm may contain spaces; ppid is always the last token
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            break
        comm, ppid_str = parts
        if "opencode" in comm.lower():
            found_opencode_pid = pid
            break
        try:
            pid = int(ppid_str)
        except ValueError:
            break
    if found_opencode_pid is None:
        return False, "no opencode in ancestor chain"

    # Signal 3: stdout is a pipe (not a regular file, not /dev/null)
    try:
        st_out = os.fstat(1)
    except OSError:
        return False, "stdout stat failed"
    if _stat.S_ISREG(st_out.st_mode):
        return False, "stdout is a regular file (nohup-redirected — safe)"
    try:
        devnull_st = os.stat(os.devnull)
        if _stat.S_ISCHR(st_out.st_mode) and st_out.st_rdev == devnull_st.st_rdev:
            return False, "stdout is /dev/null (safe)"
    except OSError:
        pass

    return True, (
        f"opencode ancestor (pid={found_opencode_pid}); no TTY; "
        f"stdout is non-regular non-/dev/null fd"
    )


def _check_for_session_deadlock_or_exit() -> None:
    """Front-of-`main` guard: if a session-deadlock-prone launch is
    detected, print a clear pointer to `/r2c-run` and exit with code 2.
    Bypassed by R2C_SKIP_DEADLOCK_CHECK=1. See README.md, "How it works"."""
    is_risk, diag = _detect_session_deadlock_risk()
    if not is_risk:
        return
    sys.stderr.write(
        "\n"
        "ERROR: scripts/run_pipeline.py appears to be running synchronously\n"
        "inside an opencode bash tool. This would deadlock against the driver's\n"
        "own producer-agent dispatches back into the same session.\n"
        "\n"
        f"  Detection: {diag}\n"
        "\n"
        "Fix: invoke `/r2c-run <slug>` from the opencode TUI. The command's\n"
        "r2c_run plugin tool spawns the driver detached so the session stays\n"
        "free for its dispatches.\n"
        "\n"
        "If you're certain this is a false positive (e.g., unusual CI env where\n"
        "a parent process happens to be named `opencode`), set\n"
        "R2C_SKIP_DEADLOCK_CHECK=1 to bypass.\n"
        "\n"
        "See README.md, \"How it works\", for the full mechanism.\n"
    )
    sys.exit(2)


StageStatus = Literal["completed", "skipped", "halted", "degraded"]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class PipelinePaths:
    """Run-specific paths resolved from setup_pipeline_dirs.py output.

    The single source of truth for downstream stages; never recomputed."""

    repo_root: Path
    run_dir: Path
    pipeline_dir: Path
    paper_md: Path
    input_path: Path
    input_kind: str  # "markdown" | "pdf"
    slug: str
    method_spec: Path
    paper_map: Path
    paper_map_halt: Path  # `<paper_map>.halt` sidecar written by decomposer/analyzer
    method_spec_halt: Path  # `<method_spec>.halt` sidecar written by analyzer (paradigm mismatch)
    paradigm_gap_report: Path  # structured analyzer gap report for unmatched paradigms
    paradigm_gap_report_md: Path
    provisional_packs_dir: Path
    provisional_pack_manifest: Path
    feasibility_gate: Path
    feasibility_halt: Path  # `<gate>.halt` sidecar written by check_feasibility.py

    @classmethod
    def from_setup_result(cls, result: dict) -> "PipelinePaths":
        pipeline_dir = Path(result["pipeline_dir"])
        gate = pipeline_dir / "feasibility_gate.json"
        spec = pipeline_dir / "method_spec.json"
        pmap = pipeline_dir / "paper_map.json"
        return cls(
            repo_root=Path(result["repo_root"]),
            run_dir=Path(result["run_dir"]),
            pipeline_dir=pipeline_dir,
            paper_md=Path(result["paper_md_path"]),
            input_path=Path(result["input_path"]),
            input_kind=result["input_kind"],
            slug=result["slug"],
            method_spec=spec,
            paper_map=pmap,
            paper_map_halt=pmap.with_suffix(pmap.suffix + ".halt"),
            method_spec_halt=spec.with_suffix(spec.suffix + ".halt"),
            paradigm_gap_report=pipeline_dir / "paradigm_gap_report.json",
            paradigm_gap_report_md=pipeline_dir / "paradigm_gap_report.md",
            provisional_packs_dir=pipeline_dir / "provisional_packs",
            provisional_pack_manifest=pipeline_dir / "provisional_pack.json",
            feasibility_gate=gate,
            feasibility_halt=gate.with_suffix(gate.suffix + ".halt"),
        )


@dataclass
class StageResult:
    status: StageStatus
    stage_id: str
    notes: str = ""
    paths_written: list[Path] = field(default_factory=list)
    halt_artifact: dict | None = None


# Static todo-list scaffold for the TUI side panel. One entry per major stage;
# substages (Stage 1.a/1.b, fix-loop iterations) are reflected in activeForm
# text rather than as separate entries. Internal stage IDs stay stable for
# sentinels, halt artifacts, reviewer contracts, and --stop-after; the content
# field is the human-facing display name.
PIPELINE_TODOS: list[tuple[str, str, str]] = [
    ("stage_0", "Stage 0 - Setup & Paper Ingestion", "Setting up + parsing paper"),
    ("stage_1", "Stage 1 - Paper Decomposition & Method Analysis", "Analyzing paper"),
    ("stage_1x", "Stage 1.x - Method Explanation (METHOD.md)", "Explaining the method"),
    ("stage_2a", "Stage 2.a - Package Scaffold", "Scaffolding package"),
    ("stage_2b", "Stage 2.b - Architecture & Training", "Generating architecture"),
    ("stage_2c", "Stage 2.c - Method Implementation", "Generating method"),
    ("stage_2d", "Stage 2.d - Package Finalization", "Finalizing package"),
    ("stage_2x", "Stage 2.x - Parameter Derivation", "Deriving parameters"),
    ("stage_3a", "Stage 3.a - Notebook Authoring", "Generating notebook"),
    ("stage_3b", "Stage 3.b - Notebook Rendering", "Rendering notebook"),
    ("stage_3c", "Stage 3.c - Smoke Execution", "Running smoke gate"),
    ("stage_4", "Stage 4 - Paper-Fidelity Review", "Reviewing paper fidelity"),
    ("stage_5", "Stage 5 - Findings Routing", "Routing findings"),
]


STAGE_DISPLAY_NAMES: dict[str, str] = {
    stage_id: content for stage_id, content, _ in PIPELINE_TODOS
}

STAGE_CATEGORIES: dict[str, str] = {
    "stage_0": "setup_analysis",
    "stage_1": "setup_analysis",
    "stage_1x": "setup_analysis",
    "stage_2a": "package_generation",
    "stage_2b": "package_generation",
    "stage_2c": "package_generation",
    "stage_2d": "package_generation",
    "stage_2x": "package_generation",
    "stage_3a": "notebook_smoke",
    "stage_3b": "notebook_smoke",
    "stage_3c": "notebook_smoke",
    "stage_4": "review_routing",
    "stage_5": "review_routing",
}


def stage_label(stage_id: str) -> str:
    """Return the human-facing display name for a stable internal stage ID."""
    return STAGE_DISPLAY_NAMES.get(stage_id, stage_id)


def stage_heading(stage_id: str) -> str:
    """Human-facing stage heading that preserves the internal ID for debugging."""
    label = stage_label(stage_id)
    if label == stage_id:
        return f"`{stage_id}`"
    return f"{label} (`{stage_id}`)"


def stage_category(stage_id: str) -> str:
    return STAGE_CATEGORIES.get(stage_id, "unknown")


def _progress_stage_catalog() -> list[tuple[str, str]]:
    return [(stage_id, content) for stage_id, content, _active_form in PIPELINE_TODOS]


def _append_run_event(
    paths: PipelinePaths,
    event_type: str,
    *,
    stage_id: str | None = None,
    status: str | None = None,
    summary: str | None = None,
    artifacts: list[str] | None = None,
    details: dict | None = None,
) -> dict | None:
    """Append an observability event, but keep events mirror-only.

    The run lock is authoritative for concurrency. Sentinels and
    `driver_state.json` remain authoritative for runtime state. If event
    append fails, log the issue and keep the pipeline moving.
    """
    try:
        event = append_event(
            paths.pipeline_dir,
            event_type=event_type,
            run_id=paths.slug,
            stage_id=stage_id,
            stage_label=stage_label(stage_id) if stage_id else None,
            stage_category=stage_category(stage_id) if stage_id else None,
            status=status,
            summary=summary,
            artifacts=artifacts,
            details=details,
        )
        try:
            write_progress(
                paths.pipeline_dir,
                run_id=paths.slug,
                stage_catalog=_progress_stage_catalog(),
            )
        except Exception as progress_exc:  # noqa: BLE001 - observability only
            log("progress", "write_failed", f"{event_type}: {progress_exc}")
        return event
    except Exception as e:
        log("events", "append_failed", f"{event_type}: {e}")
        return None


def _append_stage_result_event(paths: PipelinePaths, result: StageResult) -> None:
    event_type_by_status = {
        "completed": "stage_completed",
        "skipped": "stage_skipped",
        "halted": "stage_halted",
        "degraded": "stage_degraded",
    }
    event_type = event_type_by_status.get(result.status)
    if event_type is None:
        return
    _append_run_event(
        paths,
        event_type,
        stage_id=result.stage_id,
        status=result.status,
        summary=result.notes,
        artifacts=[str(p) for p in result.paths_written],
        details={"halt_artifact": result.halt_artifact} if result.halt_artifact else None,
    )


_GUARD_TRACEBACK_FRAMES = 30
_GUARD_STATE_SNAPSHOT_CAP = 16_384  # bytes of driver_state.json kept in context


def _last_resort_stage_guard(
    state: "PipelineState", stage_id: str, exc: BaseException
) -> StageResult:
    """Convert an exception that escaped a stage function into a normal
    halted StageResult (stage-exception-guard-design.md, decisions baked in
    2026-07-14). Observability only: the same crash, now with a front door
    — halt artifact, catalog story, explanation-only packaging through the
    EXISTING halt machinery. Never retries, never resumes, never edits.

    The guard must not be able to crash the run harder: if halt packaging
    itself raises, fall back to a minimal literal JSON write with no
    dependencies on run state, plus one log line."""
    paths = state.paths
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tb_tail = "".join(tb_lines[-_GUARD_TRACEBACK_FRAMES:])
    context: dict = {
        "exception_type": type(exc).__name__,
        "exception_message": str(exc)[:2000],
        "traceback_tail": tb_tail,
    }
    # Decision 4: snapshot driver_state.json into the halt context for
    # postmortems, size-capped.
    try:
        state_text = (paths.pipeline_dir / "driver_state.json").read_text(
            encoding="utf-8")
        context["driver_state_snapshot"] = state_text[:_GUARD_STATE_SNAPSHOT_CAP]
        if len(state_text) > _GUARD_STATE_SNAPSHOT_CAP:
            context["driver_state_snapshot_truncated"] = True
    except Exception:
        pass
    # Double-halt safety: the guard only fires when no StageResult was
    # returned, so a stage that wrote its own halt artifact and THEN raised
    # (should not happen) gets last-writer-wins, said out loud.
    if (paths.pipeline_dir / f"{stage_id}.halt").is_file():
        context["overwrote_partial_halt"] = True
    log("driver", "stage_exception_guard",
        f"{stage_id}: unhandled {type(exc).__name__} escaped the stage "
        f"function; packaging what exists and halting")
    try:
        return halt(
            paths, stage_id,
            reason=(f"unhandled pipeline exception in {stage_id}: "
                    f"{type(exc).__name__}: {str(exc)[:500]}"),
            halt_class="driver_exception",
            context=context,
            state=state,
        )
    except Exception as pkg_exc:  # noqa: BLE001 - the floor, not a seam
        minimal = {
            "schema_version": "1.0",
            "status": "halted",
            "stage": stage_id,
            "halt_class": "driver_exception",
            "reason": (f"unhandled pipeline exception in {stage_id}: "
                       f"{type(exc).__name__}: {str(exc)[:500]}"),
            "guard_packaging_error": f"{type(pkg_exc).__name__}: {pkg_exc}",
            "created_at": utc_now_iso(),
        }
        try:
            (paths.pipeline_dir / f"{stage_id}.halt").write_text(
                json.dumps(minimal, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
        log("driver", "stage_exception_guard_fallback",
            f"{stage_id}: halt packaging itself raised "
            f"({type(pkg_exc).__name__}: {pkg_exc}); wrote minimal halt")
        return StageResult(
            status="halted",
            stage_id=stage_id,
            notes=f"unhandled pipeline exception ({type(exc).__name__})",
            halt_artifact=minimal,
        )


def _finalize_run_history(
    paths: PipelinePaths,
    run_status: str,
    state: "PipelineState | None" = None,
) -> None:
    """Terminal ledger hook: snapshot + row into r2c_runs/_history/.

    Mirror-only and best-effort, same contract as `_append_run_event`. Model
    ids come from the primed agent map when the run got that far: the Think
    tier from the analyzer, the Code tier from the architecture coder (the
    agent-model split's representative agents)."""
    try:
        agent_models = getattr(state, "agent_models", None) or {}

        def _model_id(agent_name: str) -> str | None:
            model = agent_models.get(agent_name) or {}
            provider, model_id = model.get("providerID"), model.get("modelID")
            return f"{provider}/{model_id}" if provider and model_id else None

        run_history.finalize_invocation(
            paths.run_dir,
            run_status=run_status,
            think_model=_model_id("r2c-method-analyzer"),
            code_model=_model_id("r2c-architecture-coder"),
            # Schema compatibility: this field predates the retirement of the
            # selectable shared-session transport. New driver invocations are
            # now structurally per-dispatch, so provenance is always true.
            per_dispatch_sessions=True,
        )
    except Exception as e:  # noqa: BLE001 - observability only
        log("history", "finalize_failed", f"{type(e).__name__}: {e}")


def _reconcile_candidate_dispatches(paths: PipelinePaths, *, context: str) -> None:
    """Best-effort Step 5 reconciliation at safe driver boundaries.

    Candidate reconciliation is additive. It only inspects registered candidate
    attempts under `.pipeline/candidates/`; if none exist, it is a no-op. Any
    unexpected reconciler error is logged but does not replace the existing
    sentinel/canonical-artifact recovery behavior.
    """

    try:
        results = reconcile_candidate_dispatches(
            paths.pipeline_dir,
            run_id=paths.slug,
        )
    except Exception as exc:  # noqa: BLE001
        log("reconcile", "failed", f"{context}: {exc}")
        return
    for result in results:
        log(
            "reconcile",
            result.status,
            (
                f"{context}: {result.stage_id}/{result.artifact_id} "
                f"attempt {result.attempt}: {result.reason}"
            ),
        )


def _append_validation_event(
    paths: PipelinePaths,
    *,
    stage_id: str,
    validator: str,
    ok: bool,
    stderr_tail: str = "",
    artifacts: list[str] | None = None,
    details: dict | None = None,
) -> None:
    event_details = {
        "validator": validator,
        "stderr_tail": _stderr_excerpt(stderr_tail, 1000) if stderr_tail else "",
    }
    if details:
        event_details.update(details)
    _append_run_event(
        paths,
        "validation_passed" if ok else "validation_failed",
        stage_id=stage_id,
        status="passed" if ok else "failed",
        summary=f"{validator} {'passed' if ok else 'failed'}",
        artifacts=artifacts,
        # Excerpt, not a tail slice: the caller usually passes an already-
        # excerpted string, and re-slicing its tail clipped into the elision
        # marker itself (iDb 2026-07-14 run events).
        details=event_details,
    )


@dataclass
class PipelineState:
    """In-memory state for one run; serialized after each stage."""

    session_id: str
    port: int
    paths: PipelinePaths
    agent_models: dict[str, dict | None] = field(default_factory=dict)
    stage_results: list[StageResult] = field(default_factory=list)
    # Set only when this invocation successfully writes a delivery manifest.
    # The main-loop finally block consumes it after terminal events, history,
    # and physical lock cleanup so the nested baseline captures terminal truth.
    delivery_baseline_pending: bool = False
    # Workspace directory (the dir the TUI session lives in). Pinned onto each
    # fresh work session so the agent's file tools and agent/config resolution
    # match the parent. The user-facing notices still post to `session_id`.
    workspace: str = ""
    # In-memory todo list mirroring what the TUI side panel renders. Each
    # entry has {stage_id, content, activeForm, status}; status is one of
    # pending | in_progress | completed. `stage_id` is driver-internal —
    # stripped before POSTing to the build agent's TodoWrite.
    todos: list[dict] = field(default_factory=list)
    # Backend-unreachable circuit breaker. Set whenever a dispatch fails with
    # a transport/backend-outage signature (server unreachable, dispatch
    # timeout, gateway/proxy rejection, dropped message). While open and
    # within the probation cooldown (`_BREAKER_PROBATION_S`), every agent
    # dispatch fails fast instead of blocking for its full timeout, and the
    # best-effort TUI notice posters skip entirely — so a run that halts
    # BECAUSE the backend is down doesn't then spend minutes trying to reach
    # that same dead backend to announce the halt. Past the cooldown the next
    # dispatch goes through as a live probe: the observed backend failure
    # mode is a flap, not an outage (2026-07-05: the SRL row dispatched
    # cleanly one minute after detr's hang), so a breaker with no reset path
    # silently converted every later recovery into a guaranteed fast-fail.
    # A resume is a fresh state and starts clear.
    backend_unreachable: bool = False
    # Monotonic timestamp of the most recent breaker trip; None when closed.
    backend_unreachable_at: float | None = None
    # Rung 3 (dispatch-recovery ladder): the ONE bounded transport
    # re-dispatch this run may spend, consumed even when the attempt fails.
    transport_retry_used: bool = False
    # Consecutive failed probation probes. Two convert the breaker to
    # TERMINAL: no more probes, no rung-3 attempts, every remaining dispatch
    # fails fast — a genuinely dead backend must not buy N more 900-second
    # blocks. Reset when any probe succeeds (breaker closes).
    failed_probes: int = 0
    backend_terminal: bool = False
    # Circuit breaker for best-effort posts into the researcher's session
    # (R2C-061). Cosmetic todo posts stop entirely once it opens; the
    # must-deliver notices borrow only its rediscovery arm.
    post_breaker: SessionPostBreaker = field(
        default_factory=lambda: SessionPostBreaker())

    def record(self, result: StageResult) -> None:
        if result.status in {"completed", "skipped"}:
            _remove_known_issues_for_stage(self.paths, stage_id=result.stage_id)
        elif result.status == "degraded":
            # Block 8 (detr 2026-07-15): a resumed stage that ends degraded
            # supersedes a prior invocation's terminal-halt entry for the
            # same stage — the halt claimed the run stopped here and any
            # code artifacts are unreliable, which the current degraded
            # delivery contradicts. Remove ONLY that halt-shaped entry;
            # the current degrade entry (appended by degrade() during the
            # stage, under a different heading) and any prior degrade
            # entries stay — persistent quality limits are pruned solely
            # by the completed/skipped branch above. Legacy entries whose
            # heading differs are conservatively left in place.
            _remove_known_issue(
                self.paths,
                stage_id=result.stage_id,
                what_failed=_EXPLANATION_ONLY_WHAT_FAILED,
            )
        self.stage_results.append(result)
        snapshot = {
            "session_id": self.session_id,
            "port": self.port,
            "slug": self.paths.slug,
            "stages": [
                {
                    "stage_id": r.stage_id,
                    "stage_label": stage_label(r.stage_id),
                    "status": r.status,
                    "notes": r.notes,
                    "paths_written": [str(p) for p in r.paths_written],
                    "halt_artifact": r.halt_artifact,
                }
                for r in self.stage_results
            ],
        }
        state_path = self.paths.pipeline_dir / "driver_state.json"
        state_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Logging + helpers
# ---------------------------------------------------------------------------


def log(stage_id: str, status: str, msg: str) -> None:
    print(f"[{stage_id}][{status}] {msg}", flush=True)


def _stage_complete_sentinel_path(paths: PipelinePaths, stage_id: str) -> Path:
    """Path to the per-stage "fully complete" sentinel.

    Written ONLY when a stage returns StageResult(status="completed") — i.e.,
    every step inside the stage (producer dispatch + ALL validators +
    reviewer when applicable) passed. The skip-check at the top of each
    stage requires this sentinel, NOT just the output files; outputs alone
    are insufficient because a partial-completion halt (e.g., producer
    wrote model.py but the validator then failed) leaves outputs on disk
    that look "done" but represent unvalidated work.

    Cleared on halt so the next resume re-runs the stage. Driver-managed
    (not an agent artifact); exempt from out-of-scope-write enforcement
    via `_is_driver_managed`."""
    return paths.pipeline_dir / f"{stage_id}.complete"


def _write_stage_complete_sentinel(paths: PipelinePaths, stage_id: str) -> None:
    """Mark this stage's full flow (producer + validators + reviewer) as
    passed. Called by the driver after a stage returns
    `StageResult(status="completed", ...)`. Idempotent: re-writes on each
    completion, mtime = latest pass."""
    p = _stage_complete_sentinel_path(paths, stage_id)
    p.write_text("", encoding="utf-8")


def _clear_stage_complete_sentinel(paths: PipelinePaths, stage_id: str) -> None:
    """Remove the per-stage sentinel. Called when a stage halts so the
    next resume re-runs the stage. No-op if absent."""
    p = _stage_complete_sentinel_path(paths, stage_id)
    if p.exists():
        p.unlink()


def _skip_if_done(
    paths: PipelinePaths,
    stage_id: str,
    outputs: list[Path],
) -> StageResult | None:
    """If the stage's complete-sentinel is present AND all `outputs` exist
    AND no prior halt artifact remains, return a skipped StageResult.
    Otherwise clear any stale halt artifact and return None to signal the
    stage should run.

    This makes incremental re-runs cheap: a fix to a single producer or
    script (after a halt) only re-runs that stage and the ones downstream
    — earlier stages that fully completed are skipped.

    The sentinel is what makes this safe. The earlier output-existence-only
    check let a partial-completion bug through (bev-distill, 2026-05-19):
    stage 2.d halted at the arch_contract validator AFTER its producer-side
    init_finalizer wrote __init__.py + requirements.txt; on resume the
    skip-check saw those outputs and skipped 2.d entirely, so the validator
    that would have surfaced the upstream architecture bug never re-ran.
    The sentinel is written only when the WHOLE stage (including all
    validators) passes."""
    halt_path = paths.pipeline_dir / f"{stage_id}.halt"
    sentinel = _stage_complete_sentinel_path(paths, stage_id)
    all_present = all(p.exists() for p in outputs)
    if sentinel.exists() and all_present and not halt_path.exists():
        log(stage_id, "skipped", f"{len(outputs)} expected output(s) present + sentinel")
        return StageResult(
            status="skipped",
            stage_id=stage_id,
            notes=f"{stage_id} outputs + complete-sentinel present; no prior halt",
            paths_written=outputs,
        )
    if halt_path.exists():
        log(stage_id, "clear_halt", f"removing stale {halt_path.name} from prior run")
        halt_path.unlink()
    if all_present and not sentinel.exists():
        # Outputs present but sentinel missing — partial-completion from a
        # halted prior run. Re-run the stage so validators get exercised.
        log(stage_id, "re_run_no_sentinel",
            f"outputs present but no {sentinel.name}; re-running stage to "
            f"re-exercise validators (prior run may have halted mid-stage)")
    return None


def _stage_3c_upstream_files(paths: PipelinePaths) -> list[Path]:
    """Files whose modification would invalidate a prior smoke-gate result.

    The smoke gate executes the rendered notebook end-to-end. Its outcome
    depends on every code path the notebook touches:

      - `method/**/*.py` — package source (including nested local dependencies)
      - `.pipeline/notebook_draft.py` — notebook source
      - `notebook.ipynb` — rendered notebook (stage 3.b output)
      - `.pipeline/params.json` — runtime parameters wired into the notebook
      - `requirements.txt` — pinned dependencies (affects import behavior)

    If any of these is newer than `stage_3c.complete`, the prior smoke
    result is stale and the gate must re-execute."""
    rd = paths.run_dir
    files: list[Path] = []
    method_dir = rd / "method"
    if method_dir.is_dir():
        files.extend(sorted(method_dir.rglob("*.py")))
    candidates = [
        paths.pipeline_dir / "notebook_draft.py",
        rd / "notebook.ipynb",
        paths.pipeline_dir / "params.json",
        rd / "requirements.txt",
    ]
    files.extend(p for p in candidates if p.is_file())
    return files


_STAGE_3C_RENDER_INPUTS = {
    ".pipeline/notebook_draft.py",
    ".pipeline/params.json",
}


def _stage_3c_needs_prerender(paths: PipelinePaths) -> bool:
    """Return True when a prior completed smoke result is stale because the
    rendered notebook's inputs changed.

    Method package changes only need a fresh smoke execution. Notebook draft or
    params changes must regenerate `notebook.ipynb` first, otherwise Stage 3.c
    can execute and certify an old rendered notebook.
    """
    sentinel = _stage_complete_sentinel_path(paths, "stage_3c")
    if not sentinel.exists():
        return False
    digest_path = _stage_3c_upstream_digest_path(paths)
    if not digest_path.is_file():
        return False
    try:
        stored = json.loads(digest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return bool((paths.pipeline_dir / "notebook_draft.py").is_file())
    if not isinstance(stored, dict):
        return bool((paths.pipeline_dir / "notebook_draft.py").is_file())
    current = _compute_stage_3c_upstream_digest(paths)
    changed = {
        rel for rel, current_hash in current.items()
        if stored.get(rel) != current_hash
    }
    deleted = {rel for rel in stored if rel not in current}
    if "notebook.ipynb" in deleted:
        return True
    return bool((changed | deleted) & _STAGE_3C_RENDER_INPUTS)


# ---------------------------------------------------------------------------
# Stage 2.d smart-skip — same content-digest pattern as stage 3.c. Stage 2.d
# generates method/__init__.py + requirements.txt by reading the producers'
# method/*.py source files. The R2C 2026-05-22 bev-distill cascade exposed
# the gap: when stage 2.c regenerated method.py with a public-API rename
# (compute_instance_distillation_loss → compute_sparse_instance_distillation_loss),
# stage 2.d's standard `_skip_if_done` saw its sentinel + outputs and skipped
# — leaving the now-stale __init__.py on disk. Stage 3.c then failed with
# an ImportError. Same fix shape: content-hash check against a stored digest.
# ---------------------------------------------------------------------------


def _stage_2d_upstream_files(paths: PipelinePaths) -> list[Path]:
    """Files whose modification would invalidate a prior stage_2d result.

    Stage 2.d's finalize/package validators depend on every method Python source
    the public API for __init__.py and the dependency set for requirements.txt.
    Stage 2.d's validators also consume arch_contract.json and method_spec.json.
    Any of these changing means stage 2.d's outputs are stale."""
    rd = paths.run_dir
    files: list[Path] = []
    method_dir = rd / "method"
    if method_dir.is_dir():
        # Skip __init__.py itself (it's stage 2.d's OUTPUT, not its upstream)
        files.extend(
            sorted(
                p
                for p in method_dir.rglob("*.py")
                if p != method_dir / "__init__.py"
            )
        )
    candidates = [
        paths.pipeline_dir / "arch_contract.json",
        paths.pipeline_dir / "method_spec.json",
    ]
    files.extend(p for p in candidates if p.is_file())
    return files


# --- shared upstream-digest machinery (B-06 digest-twin merge) --------------
# One parameterized implementation for the stage-2d and stage-3c smart skips.
# The two stages' upstream FILE SETS stay separate on purpose: 2d excludes
# its own __init__.py output and reads the contract pair; 3c includes 2d's
# outputs plus requirements.txt (pulling requirements.txt into 2d's set would
# make 2d re-run on every resume after the reconcile rewrites it). The
# missing-digest POLICIES are opposite by design: 3c bootstraps (trust the
# sentinel, snapshot, skip — the prior run really did pass the smoke gate),
# 2d re-runs (a sentinel without a digest cannot prove upstream is unchanged;
# the bev-distill 2026-05-22 cascade). Log event names (`bootstrap_digest`,
# `no_digest_re_run`, `digest_malformed`, `upstream_changed`, `skipped`) ship
# in delivered run artifacts and are preserved verbatim.
#
# The compute path is also called CROSS-PROCESS at delivery time:
# build_probe_harness.py passes a SimpleNamespace with only `run_dir` and
# `pipeline_dir`, and its except clause does not catch AttributeError — so
# the merged helpers touch only those two attributes, take one positional
# paths argument, and carry NO caching of any kind (a paths-keyed cache
# would snapshot pre-stage content on the completion-time write path).


class _UpstreamDigestStage(NamedTuple):
    stage_id: str
    upstream_files: Callable  # (paths) -> list[Path]
    rerun_label: str          # "smoke gate" / "stage 2.d" — verbatim in logs
    still_valid_label: str    # tail of the `skipped` log line, verbatim
    trust_sentinel_on_missing_digest: bool
    required_outputs: tuple[str, ...] = ()   # run-dir-relative; 2d only
    check_halt_artifact: bool = False        # 2d only


_STAGE_2D_DIGEST = _UpstreamDigestStage(
    stage_id="stage_2d",
    upstream_files=_stage_2d_upstream_files,
    rerun_label="stage 2.d",
    still_valid_label="prior stage 2.d outputs still valid",
    trust_sentinel_on_missing_digest=False,
    required_outputs=("method/__init__.py", "requirements.txt"),
    check_halt_artifact=True,
)

_STAGE_3C_DIGEST = _UpstreamDigestStage(
    stage_id="stage_3c",
    upstream_files=_stage_3c_upstream_files,
    rerun_label="smoke gate",
    still_valid_label="prior smoke result still valid",
    trust_sentinel_on_missing_digest=True,
)


def _upstream_digest_path(paths: PipelinePaths, spec: _UpstreamDigestStage) -> Path:
    """Path to the stored-content digest written when the stage completes.
    Paired with `<stage>.complete`: the sentinel marks completion, the digest
    records file content at that moment so the next run detects ACTUAL
    changes (not just mtime touches from idempotent re-runs)."""
    return paths.pipeline_dir / f"{spec.stage_id}.upstream_digest.json"


def _compute_upstream_digest(paths, spec: _UpstreamDigestStage) -> dict[str, str]:
    """SHA256 of every upstream file, keyed by run-dir-relative path. Used at
    sentinel-write time (snapshot) and skip-check time (compare)."""
    digest: dict[str, str] = {}
    for f in spec.upstream_files(paths):
        rel = str(f.relative_to(paths.run_dir))
        digest[rel] = "sha256:" + hashlib.sha256(f.read_bytes()).hexdigest()
    return digest


def _write_upstream_digest(paths: PipelinePaths, spec: _UpstreamDigestStage) -> None:
    """Snapshot the upstream-file digest. Called when the stage returns
    `StageResult(status="completed")`. Pair with `<stage>.complete`."""
    _upstream_digest_path(paths, spec).write_text(
        json.dumps(_compute_upstream_digest(paths, spec),
                   indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _skip_stage_if_no_upstream_changes(
    paths: PipelinePaths, spec: _UpstreamDigestStage,
) -> StageResult | None:
    """The shared smart skip-check. Content-hash check (authoritative): if
    `<stage>.complete` is present AND the stored digest still matches current
    file contents → skip. Mtime alone is wrong (R2C 2026-05-22 case):
    finalize_package_init.py is idempotent — it re-writes method/__init__.py
    and requirements.txt with identical content on every resume, bumping
    mtimes past the sentinel without changing content.

    Returns a skipped StageResult on cache-hit, None on miss (caller
    proceeds to run the stage). Caller still handles halt-artifact cleanup
    independently. Missing-digest policy comes from the spec — see the
    block comment above for why the two stages are deliberate opposites."""
    sentinel = _stage_complete_sentinel_path(paths, spec.stage_id)
    if not sentinel.exists():
        return None
    outputs = [paths.run_dir / rel for rel in spec.required_outputs]
    if outputs and not all(p.exists() for p in outputs):
        # Same output-presence guarantee as _skip_if_done (2d only).
        return None
    if spec.check_halt_artifact and (
        paths.pipeline_dir / f"{spec.stage_id}.halt"
    ).exists():
        return None
    digest_path = _upstream_digest_path(paths, spec)
    current = _compute_upstream_digest(paths, spec)
    if not digest_path.is_file():
        if not spec.trust_sentinel_on_missing_digest:
            # No digest = cannot verify the sentinel was written when current
            # upstream state existed (an upstream stage may have re-generated
            # its output since — the bev-distill cascade). Re-run; the
            # completion path writes the digest for next time.
            log(spec.stage_id, "no_digest_re_run",
                f"sentinel present but {digest_path.name} missing — cannot "
                f"verify upstream is unchanged since last completion; "
                f"re-running {spec.rerun_label} (the completion path will "
                f"write the digest)")
            return None
        # Bootstrap: sentinel was written by a prior driver version that
        # didn't snapshot content. Trust the sentinel (the prior run DID
        # complete the smoke gate) and snapshot current upstream as the new
        # ground truth.
        log(spec.stage_id, "bootstrap_digest",
            f"sentinel present but {digest_path.name} missing — first run "
            f"with digest support; snapshotting {len(current)} upstream "
            f"file(s) and skipping {spec.rerun_label}")
        digest_path.write_text(
            json.dumps(current, indent=2, sort_keys=True), encoding="utf-8",
        )
        return StageResult(
            status="skipped",
            stage_id=spec.stage_id,
            notes=f"{spec.stage_id}.complete present; bootstrapped upstream digest",
        )
    try:
        stored = json.loads(digest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        stored = None
    if not isinstance(stored, dict):
        log(spec.stage_id, "digest_malformed",
            f"{digest_path.name} is malformed; treating as cache-miss "
            f"and re-running {spec.rerun_label}")
        return None
    changed = [rel for rel, h in current.items() if stored.get(rel) != h]
    deleted = [rel for rel in stored if rel not in current]
    if changed or deleted:
        log(spec.stage_id, "upstream_changed",
            f"sentinel exists but content changed for "
            f"{len(changed) + len(deleted)} file(s); re-running "
            f"{spec.rerun_label}. Changed: {changed}; Deleted: {deleted}")
        return None
    log(spec.stage_id, "skipped",
        f"sentinel + content digest match — no upstream content change; "
        f"{spec.still_valid_label}")
    return StageResult(
        status="skipped",
        stage_id=spec.stage_id,
        notes=f"{spec.stage_id}.complete + upstream content digest matches",
        paths_written=outputs,
    )


# Surviving per-stage names. The write/path/compute names are external
# surface (tests, delivery-time build_probe_harness, the main loop); the
# skip names keep each stage's call site self-describing.


def _stage_2d_upstream_digest_path(paths: PipelinePaths) -> Path:
    return _upstream_digest_path(paths, _STAGE_2D_DIGEST)


def _stage_3c_upstream_digest_path(paths: PipelinePaths) -> Path:
    return _upstream_digest_path(paths, _STAGE_3C_DIGEST)


def _compute_stage_2d_upstream_digest(paths) -> dict[str, str]:
    return _compute_upstream_digest(paths, _STAGE_2D_DIGEST)


def _compute_stage_3c_upstream_digest(paths) -> dict[str, str]:
    return _compute_upstream_digest(paths, _STAGE_3C_DIGEST)


def write_stage_2d_upstream_digest(paths: PipelinePaths) -> None:
    _write_upstream_digest(paths, _STAGE_2D_DIGEST)


def write_stage_3c_upstream_digest(paths: PipelinePaths) -> None:
    _write_upstream_digest(paths, _STAGE_3C_DIGEST)


def _skip_stage_2d_if_no_upstream_changes(paths: PipelinePaths) -> StageResult | None:
    return _skip_stage_if_no_upstream_changes(paths, _STAGE_2D_DIGEST)


def _skip_stage_3c_if_no_upstream_changes(paths: PipelinePaths) -> StageResult | None:
    return _skip_stage_if_no_upstream_changes(paths, _STAGE_3C_DIGEST)


def write_halt(paths: PipelinePaths, stage_id: str, artifact: dict) -> Path:
    halt_path = paths.pipeline_dir / f"{stage_id}.halt"
    halt_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    log(stage_id, "halted", f"wrote {halt_path}")
    return halt_path


def _init_pipeline_todos(state: "PipelineState") -> None:
    """Seed PipelineState.todos from PIPELINE_TODOS. Marks stage_0 in_progress;
    everything else starts pending. Idempotent — repeats just reset state."""
    state.todos = [
        {
            "stage_id": stage_id,
            "content": content,
            "activeForm": active_form,
            "status": "in_progress" if i == 0 else "pending",
            "priority": "high" if i == 0 else "medium",
        }
        for i, (stage_id, content, active_form) in enumerate(PIPELINE_TODOS)
    ]


def _todo_priority(status: str) -> str:
    if status == "completed":
        return "low"
    if status == "in_progress":
        return "high"
    return "medium"


NO_PROGRESS_TODOS_ENV = "R2C_NO_PROGRESS_TODOS"


class SessionPostBreaker:
    """Stop re-posting into a session that rejects every message (R2C-061).

    Across three 2026-08 rolls every progress-todo post targeted the same
    stale parent session and every one returned a server 500, retried once
    with a 30s pause, and never learned: about eleven failure pairs in one
    log. Discovery could not self-correct either, because the broken
    session's freshness is renewed by the very child dispatches that work.

    The state machine, in the order a run meets it: post, and on a second
    consecutive failure rediscover the session once and try that. If the
    rediscovered session fails too, the mechanism is done for the run
    (`zombie_notification_target`). One success resets everything, since a
    session that answers once is alive.

    Cosmetic posts (todos) go quiet. Must-deliver posts (halt and end-of-run
    notices) share the REDISCOVERY arm and never the disable arm: a halt
    notice that silently dies leaves someone watching a session that never
    learns the run ended."""

    __slots__ = ("failures", "rediscovered", "disabled")

    def __init__(self) -> None:
        self.failures = 0
        self.rediscovered = False
        self.disabled = False

    def record_success(self) -> None:
        self.failures = 0
        self.rediscovered = False

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= 2 and self.rediscovered:
            self.disabled = True

    def wants_rediscovery(self) -> bool:
        """A second consecutive failure earns exactly one rediscovery."""
        return self.failures >= 2 and not self.rediscovered

    def note_rediscovery(self) -> None:
        self.rediscovered = True


def _rediscover_session(state: "PipelineState", *, label: str) -> bool:
    """Re-resolve the workspace's session after repeated post failures.

    Returns True when a DIFFERENT session id was found and adopted. Silent
    about failures by design: this runs on best-effort notification paths."""
    if not state.workspace:
        return False
    try:
        found = discover_session(directory=state.workspace, port=state.port)
    except (OpencodeClientError, OutOfScopeWritesError, SessionNotFound):
        return False
    if not found or found == state.session_id:
        return False
    log(label, "session_rediscovered",
        f"posts kept failing against {state.session_id}; switching to {found}")
    state.session_id = found
    return True


def _post_todos(state: "PipelineState") -> None:
    """POST a TodoWrite update to the active session via the build agent.

    Fire-and-forget via a daemon thread. The driver's main loop never blocks
    on this — slow or unresponsive sessions (the port-4096 GBALD case during
    Phase 6 validation) used to block the entire pipeline for 60s per stage
    transition while the build agent's queue cleared. The thread completes
    on its own (success or timeout) and logs the result; the driver moves on.

    Snapshot the todos payload BEFORE spawning the thread — state.todos is
    mutated by subsequent _advance_todos calls and we want each POST to send
    its own moment-in-time state, not whatever the list ends up being when
    the thread actually runs.

    Skipped entirely when the run has no interactive consumer
    (`R2C_NO_PROGRESS_TODOS`, set by a headless launcher) or once the post
    breaker has opened."""
    if os.environ.get(NO_PROGRESS_TODOS_ENV):
        return
    if state.post_breaker.disabled:
        return
    payload = []
    for t in state.todos:
        status = t["status"]
        payload.append({
            "content": t["content"],
            "activeForm": t["activeForm"],
            "status": status,
            "priority": t.get("priority") or _todo_priority(status),
        })
    prompt = (
        "R2C DRIVER PROGRESS UPDATE — call the TodoWrite tool with EXACTLY "
        "the todo list below, then respond with a single brief acknowledgement "
        "('todos updated.') and stop. Do not analyze, do not propose work, do "
        "not continue the pipeline; the Python driver controls progression.\n\n"
        f"```json\n{json.dumps(payload, indent=2)}\n```"
    )
    threading.Thread(
        target=_post_todos_worker, args=(state, prompt),
        daemon=True, name="r2c-todo-post",
    ).start()


def _post_todos_worker(state: "PipelineState", prompt: str) -> None:
    """Deliver one todo update, with ONE retry after a pause.

    The M-LOAD-1 matrix showed the dominant failure shape: the session is
    busy with a long agent turn, message processing blocks, and the post
    times out at 180s. The pause gives the turn a chance to clear; a second
    failure is logged and dropped (the todo panel self-heals on the next
    stage transition — only halt notices are must-deliver).

    The breaker (R2C-061) is what keeps a permanently dead target from
    costing every later stage the same failure pair: two consecutive
    failures buy one rediscovery, and a failure after that disables the
    mechanism for the run."""
    breaker = state.post_breaker
    attempt = 0
    while attempt < 3 and not breaker.disabled:
        attempt += 1
        try:
            dispatch_and_wait(
                session_id=state.session_id, agent="build", prompt=prompt,
                model=state.agent_models.get("build"), port=state.port,
                # Generous timeout — we're not blocking the driver, so a
                # busy build agent can take its time without consequences.
                timeout_s=180,
            )
            breaker.record_success()
            if attempt > 1:
                log("todos", "post_recovered",
                    f"todo update delivered on attempt {attempt}")
            return
        except (OpencodeClientError, OutOfScopeWritesError) as e:
            breaker.record_failure()
            if breaker.disabled:
                log("todos", "post_disabled",
                    f"todo posts kept failing after a session rediscovery "
                    f"({e}); disabling progress todos for this run")
                return
            if breaker.wants_rediscovery() and _rediscover_session(
                    state, label="todos"):
                breaker.note_rediscovery()
                continue  # straight at the new session, no pause
            if attempt == 1:
                log("todos", "post_retry",
                    f"todo post failed ({e}); retrying once in 30s")
                time.sleep(30)
            else:
                log("todos", "post_failed_async",
                    f"could not post todo update after retry: {e}")
                return


def _advance_todos(
    state: "PipelineState", just_finished_stage_id: str, status: str
) -> None:
    """Mark the just-finished stage with the appropriate terminal state and
    advance the next pending stage to in_progress (unless we halted).

    Status mapping → TodoWrite states (pending|in_progress|completed):
      - 'completed' → completed (and next pending → in_progress)
      - 'skipped'   → completed (renders as ✓; same as completed for the panel)
      - 'halted'    → keep in_progress, rewrite activeForm to "⚠ Halted —
                      see halt artifact"; do NOT advance to the next stage.

    The halt-status POST that fires alongside provides the textual reason; the
    todo entry's "stuck in_progress" rendering is the visual cue that the
    pipeline stopped here."""
    advanced = False
    for entry in state.todos:
        if entry["stage_id"] == just_finished_stage_id:
            if status == "halted":
                entry["activeForm"] = "⚠ Halted — see halt artifact"
                # status stays in_progress so the panel shows the stuck point
            else:
                if status == "degraded":
                    # Stage finished but a quality gate couldn't be satisfied;
                    # it's logged to KNOWN_ISSUES.md and the run continues.
                    entry["activeForm"] = "⚠ Completed with known issues"
                entry["status"] = "completed"
            entry["priority"] = _todo_priority(entry["status"])
        elif (
            status != "halted"
            and entry["status"] == "pending"
            and not advanced
        ):
            entry["status"] = "in_progress"
            entry["priority"] = _todo_priority(entry["status"])
            advanced = True
    _post_todos(state)


# Researcher-actionable halt messages (hri-readiness Task 1). These render as
# the prominent TUI notice; the technical `reason` goes behind a collapsible.
# Every other halt site is engineering-actionable and renders the stock
# "internal error, report to engineering" notice (user_message left None).
_HALT_MSG_NOT_DECOMPOSABLE = (
    "R2C couldn't break this paper down into an implementable method — it may "
    "not describe a single reproducible algorithm (e.g., a survey or position "
    "paper, or one that defers the method to external code). See the halt "
    "artifact's `decomposer_halt` for specifics, and consider whether this "
    "paper is a fit for R2C."
)
_HALT_MSG_PARADIGM_MISMATCH = (
    "R2C understood the document, but couldn't safely route its contribution "
    "to a registered build-and-check recipe. This is a coverage or routing "
    "decision, not a document-parsing failure. See the halt artifact's "
    "`paradigm_gap_report` and `.pipeline/paradigm_gap_report.md` for the "
    "evidence and next-action status."
)
_HALT_MSG_NOT_FEASIBLE = (
    "R2C judged this paper not reproducible at demo scale — usually because it "
    "needs a dataset, pretrained weights, or infrastructure that can't be "
    "synthesized. See the halt artifact's `feasibility_halt` for the specific "
    "blocker; you may need a different paper or to supply the missing resource."
)
_HALT_MSG_BAD_INPUT = (
    "R2C couldn't read or set up this paper. Check that the paper name/path you "
    "passed exists under `input_papers/` and (for PDFs) that the file isn't "
    "corrupt or a scanned-image-only document. Fix the input and re-run."
)
_HALT_MSG_PDF_PARSER_MISSING = (
    "This paper is a PDF and the local PDF parser (Marker) is not installed on "
    "this machine. Install it once (`pip install marker-pdf` on Python 3.10-3.13 "
    "and `brew install llama.cpp`, see README and .env.example) and re-run, or "
    "drop the paper's markdown export into `input_papers/` instead, which needs "
    "no parser."
)

# Degrade "what to do" for the quality/reporting stages (4 + 5). The notebook +
# method package already exist before these stages run, so a failure here never
# blocks delivery — it just means the quality review couldn't be completed.
_REVIEW_DEGRADE_TODO = (
    "Your notebook and method package were produced and are unaffected — only the "
    "automated paper-fidelity review (the quality check that compares the generated "
    "code to the paper's equations) couldn't be completed. Review the implementation "
    "against the paper yourself, or ask the run chat to help: `/r2c-chat <this "
    "run's folder name>` in the opencode window can compare the code to the paper "
    "and flag divergences, grounded in this run's own records."
)

# Degrade "what to do" for the smoke gate (3.c). The notebook was generated and
# rendered; it just raises at a cell. Ship it + the failing-cell pointer.
_SMOKE_DEGRADE_TODO = (
    "The notebook was generated and runs up to the failing cell, but that cell raises "
    "an error. Open the notebook and fix the cell against method/, or ask the run "
    "chat to fix it: `/r2c-chat <this run's folder name>` in the opencode window "
    "can make the fix, re-run the checks, and track the change in the run's own "
    "history. The traceback is in the technical detail below."
)

# Timeout variant (item 20, bev-distill overnight 07-08): the cell never
# raised, so no traceback language and no bug hunt — the honest fix is
# cheaper work.
_SMOKE_TIMEOUT_DEGRADE_TODO = (
    "The notebook was generated and runs up to the slow cell, but that cell ran "
    "longer than the per-cell time budget of the automated check. It did not "
    "error — there is no traceback — so the fix is making the cell cheaper "
    "(fewer epochs, a smaller demo slice, or splitting the cell), not hunting "
    "for a bug. You can also simply run the notebook yourself without the "
    "budget: the cell may finish fine given more time. The run chat can make "
    "the work-reduction edit for you: `/r2c-chat <this run's folder name>` in "
    "the opencode window."
)


def _judge_halt_user_message(rationale: str) -> str:
    """Researcher-facing wrapper for judge-decided DEGRADE entries
    (KNOWN_ISSUES what_to_do), where quoting the assessment inline is still
    the right call — the run continues and the researcher reads the entry
    with working artifacts in hand.

    HALT sites no longer use this (item 12 rule 3): the detr 2026-07-04
    fixture put a wrong-mechanism judge guess in the halt headline and asked
    the researcher to adjudicate paper-vs-internal, a call they cannot make.
    Judge halts now lead with the catalog's judge_halt story and keep the
    rationale behind the technical collapsible, labeled as an internal
    note. (This supersedes finding R-004's surface-it-prominently rule for
    halts.)"""
    flat = " ".join(rationale.split())
    return (
        "An automated reviewer stopped the run because it couldn't resolve an "
        f"issue on its own. Its assessment: \"{flat}\" — if that points to "
        "something in the paper or your inputs you can adjust, do so and re-run; "
        "if it reads like an internal R2C issue, share it with engineering."
    )


def _render_halt_notice(
    stage_id: str, reason: str, halt_path: Path, user_message: str | None,
    halt_class: str | None = None, evidence: dict | None = None,
    context: dict | str | None = None,
) -> str:
    """Build the researcher-facing halt notice (Markdown). Pure + testable.

    Four modes, in precedence order:
      - terminal scope decision (`unsupported_or_unclear` in the structured
        gap context): render the gap report's recorded next action (or explicit
        missing-action guidance) and do not add a generic resume instruction.
      - researcher-actionable (`user_message` set): lead with the hand-written
        plain-language message + what-to-do; tuck the technical
        `reason`/artifact path into a `<details>` collapsible. Hand-set
        messages win over the catalog — the feasibility and bad-input
        messages already read well.
      - catalog-classified (`halt_class` set, no user_message): lead with the
        halt catalog's plain-language story (what happened / why we stopped /
        what to do next), mechanism claims bounded by `evidence`
        (scripts/halt_catalog.py, queue item 12); same collapsible.
      - unclassified (`user_message` and `halt_class` both None): the stock
        "internal error, report to engineering" notice; same collapsible.
        Old halt paths render exactly as before.

    Either way the top line is human-readable — never the raw driver-debug
    `reason` that researchers couldn't act on before."""
    heading = stage_heading(stage_id)
    # Front-door rule (failure-path spec §5): every terminal surface names
    # r2c_runs/<slug>/REPORT.md first; everything else links from there.
    front_door = halt_path.parent.parent / "REPORT.md"
    front_door_line = (
        f"Start at `{front_door}` — the stopped-run summary is at the top, "
        f"and every other artifact is linked from there."
    )
    terminal_action = halt_catalog.terminal_gap_action(halt_class, context)
    story = None
    if terminal_action or not user_message:
        story = halt_catalog.render_halt_story(
            halt_class or ("paradigm_mismatch" if terminal_action else None),
            stage_id=stage_id,
            evidence=evidence,
        )
    if terminal_action and story is not None:
        headline = (
            f"🛑 **Pipeline halted at {heading}**\n\n"
            f"{front_door_line}\n\n"
            f"**What happened.** {story.what_happened}\n\n"
            f"**Why we stopped instead of guessing.** {story.why_stopped}\n\n"
            f"**What to do next.** {terminal_action}"
        )
    elif user_message:
        headline = (
            f"🛑 **Pipeline halted at {heading}**\n\n"
            f"{front_door_line}\n\n"
            f"{user_message}"
        )
    elif story is not None:
        headline = (
            f"🛑 **Pipeline halted at {heading}**\n\n"
            f"{front_door_line}\n\n"
            f"**What happened.** {story.what_happened}\n\n"
            f"**Why we stopped instead of guessing.** {story.why_stopped}\n\n"
            f"**What to do next.** {story.what_next}"
        )
    else:
        headline = (
            f"🛑 **Pipeline halted at {heading}** — internal pipeline error\n\n"
            f"{front_door_line}\n\n"
            "This is an internal R2C error, not something you did. Please report "
            "it to engineering with the halt artifact below attached; the run can "
            "be re-tried after a fix."
        )
    if terminal_action:
        recovery_line = (
            "This is a terminal scope decision for the current input. Do not "
            "resume the unchanged run; follow the recorded action above."
        )
    else:
        recovery_line = (
            "To resume after addressing the issue, re-run "
            "`python3 scripts/run_pipeline.py` from a backgrounded shell."
        )
    return (
        f"{headline}\n\n"
        "<details>\n"
        "<summary>Technical detail (for engineering)</summary>\n\n"
        f"**Reason:** {reason}\n\n"
        f"**Halt artifact:** `{halt_path}`\n\n"
        f"{recovery_line}\n\n"
        "</details>"
    )


# Pause before the halt-post retry: the dominant failure is a session busy
# with a long agent turn, and the pause lets the turn clear. Module constant
# so the test suite can zero it (tests/conftest.py autouse fixture) without
# touching the stdlib time module.
HALT_POST_RETRY_DELAY_S = 15


def _post_halt_to_session(
    state: "PipelineState", stage_id: str, reason: str, halt_path: Path,
    user_message: str | None = None,
    halt_class: str | None = None, evidence: dict | None = None,
    context: dict | str | None = None,
) -> None:
    """POST a halt notice into the active session so the TUI surfaces it.

    Without this, halts are invisible from the TUI: the driver exits silently
    after the last producer's success message, the researcher sees the TUI go
    idle, and a natural "keep going" message goes to whichever agent the TUI
    has selected (typically `build`), bypassing every validator.

    Best-effort: a failed POST is logged but doesn't mask the original halt
    (the halt artifact is already on disk). Dispatches to the `build` agent
    with an explicit instruction to render the notice verbatim and stop."""
    notice = _render_halt_notice(
        stage_id, reason, halt_path, user_message,
        halt_class=halt_class, evidence=evidence, context=context,
    )
    prompt = (
        "R2C DRIVER HALTED.\n\n"
        "**STRICT RULES FOR YOUR RESPONSE:**\n"
        "1. Your response must contain ZERO tool calls. Use NO tools: NO "
        "Bash, NO Write, NO Edit, NO Read, NO Skill, NO TodoWrite. The "
        "Python driver has already exited; nothing you do here affects the "
        "pipeline.\n"
        "2. Render the notice text below EXACTLY as written, verbatim, as "
        "a single Markdown response. Nothing before it. Nothing after it.\n"
        "3. Do NOT analyze the halt. Do NOT propose fixes. Do NOT delete "
        "the halt artifact. Do NOT delete any files. Do NOT modify pipeline "
        "state in any way. The halt artifact at the path below is the "
        "researcher's evidence record — touching it loses information.\n"
        "4. If you find yourself wanting to be helpful by cleaning up files "
        "or recommending a re-run, STOP — that's the wrong instinct here. "
        "Your only job is to surface the halt notice to the researcher. "
        "They decide what to do next.\n\n"
        "**Notice to render (verbatim):**\n\n"
        f"{notice}"
    )
    if state.backend_unreachable:
        log(stage_id, "halt_post_skipped",
            "backend marked unreachable; skipping halt-notice post — the halt "
            "artifact on disk is the authoritative record")
        return
    # Synchronous (unlike _post_todos) with ONE retry after a pause — halts
    # are rare and this TUI message is the researcher's primary signal that
    # the run stopped (the halt artifact on disk is the fallback record).
    # The dominant failure shape is a session busy with a long agent turn
    # (M-LOAD-1's 180s post timeouts); the pause lets the turn clear. Worst
    # case this delays the (already-exiting) driver a few minutes, which is
    # the right trade for notice delivery.
    # Must-deliver, so the third attempt goes to a rediscovered session
    # rather than a third try at a target that has refused twice (R2C-061).
    for attempt in (1, 2, 3):
        try:
            dispatch_and_wait(
                session_id=state.session_id,
                agent="build",
                prompt=prompt,
                model=state.agent_models.get("build"),
                port=state.port,
                timeout_s=180,
            )
            if attempt > 1:
                log(stage_id, "halt_post_recovered",
                    f"halt notice delivered on attempt {attempt}")
            return
        except (OpencodeClientError, OutOfScopeWritesError) as e:
            if attempt == 1:
                log(stage_id, "halt_post_retry",
                    f"halt post failed ({e}); retrying once in "
                    f"{HALT_POST_RETRY_DELAY_S}s")
                time.sleep(HALT_POST_RETRY_DELAY_S)
            elif attempt == 2 and _rediscover_session(
                    state, label=stage_id):
                continue
            else:
                log(stage_id, "halt_post_failed",
                    f"could not post halt to session after retry: {e}")
                return


# Scrub grammar owned by render_claims_report (one definition for every
# researcher-facing scrub path). A drifting local copy here is how
# stage-scoped ids (AL-S1-1 style) leaked bare into notices when only the
# run-report copy was widened (2026-07-22 design review).
_NOTICE_CODE_RE = render_claims_report.CODE_RE
_NOTICE_FINDING_RE = render_claims_report.FINDING_RE
FINAL_NOTICE_TIMEOUT_S = 30
FINAL_NOTICE_RETRY_DELAY_S = 5


def _scrub_notice_codes(text: object) -> str:
    # Quoted-noun gloss form, never bare in-place replacement — the
    # glosses are verb phrases and substituting them into attributive
    # positions broke grammar (pdwa 2026-07-05 REPORT; same fix as
    # render_run_report._scrub_codes). Codes still never appear bare.
    raw = " ".join(str(text or "").split())
    try:
        from probes.catalog import PROBE_GLOSS as probe_gloss  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        probe_gloss = {}

    def repl(match: re.Match[str]) -> str:
        gloss = probe_gloss.get(match.group(0))
        if gloss is None:
            return "this verification check"
        return f'the "{gloss}" check'

    return _NOTICE_FINDING_RE.sub("the linked finding", _NOTICE_CODE_RE.sub(repl, raw))


def _manifest_notice_items(paths: "PipelinePaths", manifest_status: str) -> list[str]:
    items: list[str] = []
    if manifest_status not in {"passed", "degraded", "blocked"}:
        items.append(f"Manifest note: {_scrub_notice_codes(manifest_status)}")

    manifest_path = paths.run_dir / run_layout.FINAL_MANIFEST_JSON
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = None
    if isinstance(manifest, dict):
        for diag in manifest.get("diagnostics") or []:
            message = _scrub_notice_codes(diag)
            if message:
                items.append(f"Manifest note: {message}")
    elif manifest_status not in {"passed", "degraded"}:
        items.append("Manifest note: final_manifest.json was not readable.")
    if not manifest_path.is_file() and manifest_status not in {"passed", "degraded"}:
        items.append("Manifest note: final_manifest.json was not written.")
    return items


def _delivery_notice_items(delivery: dict | None) -> list[str]:
    items: list[str] = []
    if not isinstance(delivery, dict):
        return items
    for reason in delivery.get("reasons") or []:
        if isinstance(reason, dict):
            message = _scrub_notice_codes(reason.get("message", ""))
            if message:
                items.append(f"Finding: {message}")
    for disclosure in delivery.get("disclosures") or []:
        if isinstance(disclosure, dict):
            message = _scrub_notice_codes(disclosure.get("message", ""))
            if message:
                items.append(f"Disclosure: {message}")
    return items


def _notice_findings(
    paths: "PipelinePaths",
    delivery: dict | None,
    *,
    run_status: str,
    manifest_status: str,
) -> list[str]:
    manifest_items = _manifest_notice_items(paths, manifest_status)
    delivery_items = _delivery_notice_items(delivery)
    if run_status == "failed" or manifest_status == "blocked":
        items = [*manifest_items, *delivery_items]
    else:
        items = [*delivery_items, *manifest_items]

    if not items:
        return ["No delivery-blocking or delivery-demoting findings were recorded."]
    return items[:3]


def _render_end_of_run_notice(
    paths: "PipelinePaths",
    *,
    run_status: str,
    manifest_status: str,
    delivery: dict | None = None,
) -> str:
    delivery_label = "not recorded"
    if isinstance(delivery, dict) and delivery.get("label"):
        delivery_label = str(delivery["label"])
    # The notice names the label in the same plain terms as the README
    # banner (design note §4) — never a bare enum value. The uncertified
    # floor clause derives from what actually ran (2026-07-04).
    from delivery_label import (contribution_gap_clause,  # noqa: PLC0415
                                universal_floor_clause)
    label_gloss = {
        "verified": "verified",
        "draft": "draft (findings need your attention before relying on "
                 "the output)",
        "explanation_only": "explanation only (code was not produced or is "
                            "not reliable)",
        "uncertified_new_territory": (
            f"uncertified — new territory "
            f"({universal_floor_clause(delivery).rstrip('.')}; "
            f"{contribution_gap_clause(delivery).rstrip('.')})"),
    }.get(delivery_label, f"`{delivery_label}`")
    manifest_path = paths.run_dir / run_layout.FINAL_MANIFEST_JSON
    if manifest_path.is_file():
        manifest_line = f"Machine-readable manifest: `{run_layout.FINAL_MANIFEST_JSON}`."
    else:
        manifest_line = "Machine-readable manifest: not written; see the manifest status above."
    # Front-door rule (failure-path spec §5): the notice's first line names
    # the one path to open, repo-relative so it can be pasted as-is.
    try:
        front_door = paths.run_dir.relative_to(paths.repo_root) / "REPORT.md"
    except ValueError:
        front_door = paths.run_dir / "REPORT.md"
    # §3.5 criterion 1: a partial delivery says PARTIAL in the notice's
    # FIRST line, with the completeness numbers, never further down.
    partial_clause = ""
    if isinstance(delivery, dict) and delivery.get("partial"):
        from partial_delivery import (completeness_counts,  # noqa: PLC0415
                                      completeness_statement)
        counts = completeness_counts(
            paths.pipeline_dir, delivery.get("stubbed_elements") or [])
        partial_clause = (f" — PARTIAL delivery: "
                          f"{completeness_statement(counts)}")
    lines = [
        f"**R2C run finished: {run_status}{partial_clause}**",
        "",
        f"Start at `{front_door}` — every other artifact is linked from there.",
        "",
        f"Manifest status: `{manifest_status}`.",
        f"Delivery label: {label_gloss}.",
        "",
        manifest_line,
        "",
        "Top findings:",
    ]
    lines.extend(
        f"- {item}" for item in _notice_findings(
            paths,
            delivery,
            run_status=run_status,
            manifest_status=manifest_status,
        )
    )
    return "\n".join(lines)


def _post_end_of_run_notice(
    state: "PipelineState",
    *,
    run_status: str,
    manifest_status: str,
    delivery: dict | None = None,
) -> None:
    """POST the final run status into the active session.

    This is the successful-run counterpart to the halt notice: the last TUI
    signal should name the manifest status and the top things to inspect,
    not leave the user at a generic todo acknowledgement.
    """
    try:
        notice = _render_end_of_run_notice(
            state.paths,
            run_status=run_status,
            manifest_status=manifest_status,
            delivery=delivery,
        )
        prompt = (
            "R2C DRIVER FINISHED.\n\n"
            "**STRICT RULES FOR YOUR RESPONSE:**\n"
            "1. Your response must contain ZERO tool calls. Use NO tools: NO "
            "Bash, NO Write, NO Edit, NO Read, NO Skill, NO TodoWrite.\n"
            "2. Render the notice text below EXACTLY as written, verbatim, as "
            "a single Markdown response. Nothing before it. Nothing after it.\n"
            "3. Do NOT continue the pipeline. The Python driver has already "
            "finished and wrote the researcher-facing files.\n\n"
            "**Notice to render (verbatim):**\n\n"
            f"{notice}"
        )
    except Exception as e:  # noqa: BLE001 - final notice is best-effort
        log("final_notice", "render_failed", f"{type(e).__name__}: {e}")
        return
    if state.backend_unreachable:
        log("final_notice", "post_skipped",
            "backend marked unreachable; skipping end-of-run notice post")
        return
    # Must-deliver, so no disable arm: only the rediscovery one (R2C-061).
    # An end-of-run notice that silently dies leaves someone watching a
    # session that never learns the run finished.
    for attempt in (1, 2, 3):
        try:
            dispatch_and_wait(
                session_id=state.session_id,
                agent="build",
                prompt=prompt,
                model=state.agent_models.get("build"),
                port=state.port,
                timeout_s=FINAL_NOTICE_TIMEOUT_S,
            )
            if attempt > 1:
                log("final_notice", "post_recovered",
                    f"end-of-run notice delivered on attempt {attempt}")
            return
        except Exception as e:  # noqa: BLE001 - final notice is best-effort
            if attempt == 1:
                log("final_notice", "post_retry",
                    f"end-of-run notice failed ({type(e).__name__}: {e}); "
                    f"retrying once in {FINAL_NOTICE_RETRY_DELAY_S}s")
                time.sleep(FINAL_NOTICE_RETRY_DELAY_S)
            elif attempt == 2 and _rediscover_session(
                    state, label="final_notice"):
                continue
            else:
                log("final_notice", "post_failed",
                    f"could not post end-of-run notice after retry: "
                    f"{type(e).__name__}: {e}")
                return


def halt(
    paths: PipelinePaths,
    stage_id: str,
    *,
    reason: str,
    notes: str = "",
    retry_count: int = 0,
    context: dict | str | None = None,
    findings: list[dict] | None = None,
    user_message: str | None = None,
    halt_class: str | None = None,
    evidence: dict | None = None,
    state: "PipelineState | None" = None,
) -> StageResult:
    """Build halt artifact, write it to `<PIPELINE_DIR>/<stage_id>.halt`,
    optionally surface it to the TUI session, return a halted StageResult.

    When `state` is provided (i.e., Stage 1+ where the session is known),
    a halt-status message is POSTed into the session so the researcher sees
    it in the TUI instead of having to check the log file. Stage 0 halts
    fire before session discovery and call halt() without state.

    `user_message` (hri-readiness Task 1): set it on researcher-actionable
    halts (paper not reproducible / not a paradigm fit / bad input) for a
    plain-language TUI notice.

    `halt_class` + `evidence` (halt-reason catalog, queue item 12): declare
    the closed catalog class this halt belongs to and the structured
    dispatch evidence the mechanism resolver may speak from (see
    scripts/halt_catalog.py). When set without a `user_message`, the TUI
    notice and the REPORT.md halt block render the catalog's plain-language
    story instead of the stock 'internal error, report to engineering'."""
    artifact = build_halt_artifact(
        stage=stage_id,
        reason=reason,
        retry_count=retry_count,
        context=context,
        findings=findings,
        user_message=user_message,
        halt_class=halt_class,
        evidence=evidence,
    )
    halt_path = write_halt(paths, stage_id, artifact)
    result = StageResult(
        status="halted",
        stage_id=stage_id,
        notes=notes or reason,
        halt_artifact=artifact,
    )
    if state is not None:
        finalize_explanation_only_package(state, result)
        _post_halt_to_session(
            state, stage_id, reason, halt_path, user_message,
            halt_class=halt_class, evidence=evidence,
            context=artifact.get("context"),
        )
    return result


def degrade(
    paths: PipelinePaths,
    stage_id: str,
    *,
    reason: str,
    what_failed: str,
    what_to_do: str,
    where: str | None = None,
    notes: str = "",
    state: "PipelineState | None" = None,
) -> StageResult:
    """Type-(D) failure: a usable artifact exists but a quality gate could not
    be satisfied within the recovery cap. Log a structured entry to
    `<RUN_DIR>/KNOWN_ISSUES.md` and CONTINUE (return a `degraded` StageResult)
    instead of halting the whole run.

    Critical invariant — degrade NEVER edits the artifact. The driver has no
    Edit tool by design (v1's band-aid-spiral failure mode is structurally
    impossible). degrade only *surfaces* the unresolved issue honestly, leaving
    the artifact as the producer left it. recover-via-redispatch → if exhausted,
    surface + continue. Never silently patch to pass.

    `what_failed` / `where` / `what_to_do` are researcher-facing (what broke,
    where: file/cell/line, and what to try — including handing the notebook +
    KNOWN_ISSUES.md to an AI assistant). `reason` is the technical detail.
    See the halt-to-degrade redesign note (internal, not shipped)."""
    _append_known_issue(
        paths, stage_id=stage_id, what_failed=what_failed,
        where=where, what_to_do=what_to_do, reason=reason,
    )
    log(stage_id, "degraded",
        f"{what_failed} — logged to details/KNOWN_ISSUES.md; continuing with best-effort output")
    return StageResult(
        status="degraded",
        stage_id=stage_id,
        notes=notes or f"degraded: {what_failed}"[:200],
    )


# Producer stages routed via run_fix_loop (2.b architecture / 2.c method /
# 3.a notebook). On cap-exhaustion their artifact EXISTS but couldn't pass its
# validator/reviewer — type-(D), so degrade rather than deny the whole package.
_PRODUCER_ARTIFACT = {
    "stage_1": "the analyzer artifacts (paper_map.json + method_spec.json)",
    "stage_2b": "the architecture code (method/model.py + method/training.py)",
    "stage_2c": "the method code (method/method.py)",
    "stage_3a": "the notebook (notebook.ipynb)",
}


# Required producer-owned artifacts per run_fix_loop stage. These are the
# minimum outputs downstream stages can safely consume; if any are absent, the
# stage is type-(T) even when some sibling files exist, and must halt for a
# clean re-run instead of degrading over an incomplete contract.
_PRODUCER_REQUIRED_ARTIFACTS: dict[str, list[str]] = {
    "stage_2b": [
        "method/model.py",
        "method/training.py",
        ".pipeline/arch_contract.json",
    ],
    "stage_2c": ["method/method.py"],
    "stage_3a": [".pipeline/notebook_draft.py"],
}


# Which build-plan manifest role produces each fix-loop stage's files. The
# required-symbols check derives its names from the manifest entries this
# role owns — the same source the stage validators use.
_PRODUCER_MANIFEST_ROLE: dict[str, str] = {
    "stage_2b": "architecture_coder",
    "stage_2c": "method_coder",
}


def _producer_required_symbols_from_manifest(
    stage_id: str, paths: "PipelinePaths",
) -> dict[str, list[str]]:
    """Required top-level symbol names per producer file, derived from the
    spec's build-plan package manifest.

    Replaces a hardcoded active-learning table
    (stage_2b → training.py::build_model) that fabricated halt reasons on
    every other paradigm: iDb-RRT 2026-07-08 halted with "producer is
    missing required contract item(s): method/training.py::build_model" on
    a motion-planning paper whose manifest wants
    precompute_motion_primitives — which the producer had correctly
    written. The 2.b validator was cured of this exact staleness in May by
    driving off the manifest; this gives the halt summarizer the same
    medicine.

    Placeholder symbols (angle-bracket names like `<StudentClass>`) are
    role-shaped rather than name-checkable and are skipped. Best-effort:
    any spec/build-plan load failure returns {} so the file-presence check
    still stands and a degrade decision is never blocked on a summarizer
    error."""
    role = _PRODUCER_MANIFEST_ROLE.get(stage_id)
    if role is None:
        return {}
    spec_path = paths.pipeline_dir / "method_spec.json"
    try:
        from scripts.build_plan import load_build_plan  # noqa: PLC0415
        from scripts.taxonomy import run_overlay_dir  # noqa: PLC0415

        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        build_plan = load_build_plan(
            spec, REPO_ROOT,
            provisional_packs_dir=run_overlay_dir(paths.run_dir),
        )
    except Exception:
        return {}
    if not build_plan:
        return {}
    out: dict[str, list[str]] = {}
    manifest = build_plan.get("package_manifest") or {}
    for entry in manifest.get("files", []) or []:
        if entry.get("produced_by") != role:
            continue
        rel = entry.get("path") or ""
        names = [
            s["name"] for s in (entry.get("public_symbols") or [])
            if isinstance(s.get("name"), str)
            and "<" not in s["name"]
            and s["name"].isidentifier()
        ]
        if rel and names:
            out[rel] = names
    return out


def _producer_missing_required_artifacts(
    stage_id: str, paths: "PipelinePaths",
) -> list[str]:
    """Return run-dir-relative required producer files missing for a stage."""
    rels = _PRODUCER_REQUIRED_ARTIFACTS.get(stage_id)
    if not rels:
        return []
    return [rel for rel in rels if not (paths.run_dir / rel).is_file()]


def _producer_missing_required_symbols(
    stage_id: str, paths: "PipelinePaths",
) -> list[str]:
    """Return required top-level producer symbols missing from existing files."""
    rel_to_names = _producer_required_symbols_from_manifest(stage_id, paths)
    missing: list[str] = []
    for rel, names in rel_to_names.items():
        path = paths.run_dir / rel
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as e:
            missing.extend(f"{rel}::{name} (unparseable: {e.msg})" for name in names)
            continue
        found = {
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        missing.extend(f"{rel}::{name}" for name in names if name not in found)
    return missing


def _producer_missing_required_contract(
    stage_id: str, paths: "PipelinePaths",
) -> list[str]:
    """Return missing files/symbols that make a producer stage non-consumable.

    A degrade only makes sense when a usable artifact exists (degrade()'s own
    contract). When a producer misses a required output (a genuine no-write or
    partial-write stage-completion failure, type-(T)), degrading falsely logs a
    'best-effort artifact' and lets a downstream stage hard-halt on the missing
    input. Stages with no declared required artifact default to no missing
    artifacts, preserving legacy degrade behavior."""
    return [
        *_producer_missing_required_artifacts(stage_id, paths),
        *_producer_missing_required_symbols(stage_id, paths),
    ]


def _producer_required_contract_present(
    stage_id: str, paths: "PipelinePaths",
) -> bool:
    """True iff all required producer-owned files and entrypoints exist."""
    return not _producer_missing_required_contract(stage_id, paths)


def _producer_missing_required_reason(
    stage_id: str, paths: "PipelinePaths",
) -> str:
    missing = _producer_missing_required_contract(stage_id, paths)
    return f"producer is missing required contract item(s): {', '.join(missing)}"


def _degrade_producer(
    paths: PipelinePaths, stage_id: str, *, reason: str,
    state: "PipelineState | None" = None,
) -> StageResult:
    """degrade() wrapper for the run_fix_loop producer stages (2b/2c/3a): the
    artifact was generated but couldn't pass its quality checks within the cap.
    Ship best-effort + log; later stages run on it."""
    artifact = _PRODUCER_ARTIFACT.get(stage_id, f"the {stage_id} output")
    return degrade(
        paths, stage_id, reason=reason,
        what_failed=f"{artifact} could not pass its quality checks automatically",
        what_to_do=(
            "It was generated but a validator/reviewer flagged issues R2C couldn't "
            "auto-fix within its retry budget; later stages ran on the best-effort "
            "output. Review it against the paper, or ask the run chat to finish it: "
            "`/r2c-chat <this run's folder name>` in the opencode window."
        ),
        state=state,
    )


def _degrade_reviewer_unusable(
    paths: PipelinePaths, stage_id: str, *, reason: str,
    state: "PipelineState | None" = None,
) -> StageResult:
    """degrade() for the case where the producer artifact PASSED its validator
    but the stage-reviewer's output was missing or unusable (e.g. the reviewer
    omitted its `stage_id` contract echo, or wrote no file) — so the artifact
    is shipped UNREVIEWED.

    Distinct from `_degrade_producer` purely in its researcher-facing message:
    blaming the producer's code ("could not pass its quality checks") is FALSE
    here — the code passed its validator; only the semantic review could not be
    completed. Emitting the accurate cause is the DRV-1 fix (the stage_2b
    'passed-review-but-degraded' puzzle, 2026-06-24). This changes the message
    only, never the pass/degrade decision."""
    artifact = _PRODUCER_ARTIFACT.get(stage_id, f"the {stage_id} output")
    return degrade(
        paths, stage_id, reason=reason,
        what_failed=(
            f"{artifact} passed its validator, but the stage-reviewer's output "
            f"was unusable, so it was shipped WITHOUT a completed semantic review"
        ),
        what_to_do=(
            "The artifact itself passed its automated validator; only the semantic "
            "reviewer pass could not be completed (the reviewer produced no usable "
            "review JSON). Review the artifact against the paper yourself, or ask "
            "the run chat to review it: `/r2c-chat <this run's folder name>` in "
            "the opencode window."
        ),
        state=state,
    )


def run_script(
    stage_id: str, args: list[str], *, timeout: int = 300
) -> subprocess.CompletedProcess:
    """Run a pipeline script via subprocess from the repo root. Stdout/stderr
    are passed through so the researcher sees them in the TUI.

    A timeout returns a failed CompletedProcess (returncode 124, the timeout
    naming itself in stderr) instead of raising: every call site already
    routes a nonzero returncode into its fix-loop or halt machinery, whereas
    an escaped TimeoutExpired unwinds past all packaging and leaves the run
    with no manifest at all (the Rethinking-Grouping 2026-07-13 crash: a
    validator dry-run stalled on a network fetch)."""
    cmd = [PYTHON_CMD, *(
        str(trusted_path(a)) if os.path.isabs(a) and os.path.exists(a)
        else trusted_text(a)
        for a in args
    )]
    log(stage_id, "subprocess", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), capture_output=True, text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        def _as_text(v: bytes | str | None) -> str:
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="replace")
            return v or ""
        script = next((a for a in args if not a.startswith("-")), "script")
        msg = f"{script} timed out after {timeout}s"
        log(stage_id, "subprocess_timeout", msg)
        proc = subprocess.CompletedProcess(
            args=cmd, returncode=124,
            stdout=_as_text(exc.stdout),
            stderr=_as_text(exc.stderr) + f"\nerror: {msg}\n",
        )
    if proc.stdout:
        sys.stdout.write(proc.stdout)
        sys.stdout.flush()
    if proc.stderr:
        sys.stderr.write(proc.stderr)
        sys.stderr.flush()
    return proc


_BACKEND_OUTAGE_SIGNATURES = (
    "messageabort",         # backend dropped the message mid-turn
    "text/html",            # gateway/proxy HTML error page to a JSON client
    "<html",                # same, response-body form
    "bad gateway",          # 502 proxy-level outage
    "service unavailable",  # 503
    "gateway timeout",      # 504
    "connection refused", "connection reset", "connection closed",
    "connection aborted", "connection error",
)


def _is_backend_outage(exc: BaseException) -> bool:
    """True when an exception means the model backend is unreachable or down,
    as opposed to a content/contract error a healthy backend produced.

    Trips the run's backend-unreachable circuit breaker. Conservative by
    design: matches the unambiguous transport failures (server unreachable,
    dispatch timeout — including the blocking-read-timeout subclass) and
    assistant-error payloads whose signature is a gateway/proxy outage or a
    dropped message. The discriminating tell for a gateway rejection vs. a
    model *semantic* 400 is the content type: a proxy returns an HTML error
    page (`text/html`), a live model API returns JSON. Does NOT trip on
    SessionNotFound, out-of-scope writes, or a JSON semantic 400 — all of
    which mean the backend is alive."""
    if isinstance(exc, (ServerUnreachable, DispatchTimeout)):
        return True
    if isinstance(exc, OpencodeClientError):
        msg = str(exc).lower()
        return any(sig in msg for sig in _BACKEND_OUTAGE_SIGNATURES)
    return False


# Half-open probation window for the backend-unreachable breaker. Within this
# many seconds of the trip, dispatches fail fast (the designed win: the
# halt-judge dispatch that follows a hang within seconds turns a 51-minute
# halt-path hang into seconds). Past it, the next dispatch is allowed through
# as a live probe — success closes the breaker, another outage failure
# restarts the cooldown. 120s sits between the two observed timescales: the
# post-trip dispatches worth fast-failing arrive within seconds, and the
# backend demonstrably answers again within about a minute of a hang ending
# (the flap pattern in infra-dispatch-hang-report.md).
_BREAKER_PROBATION_S = 120


def _open_breaker(state: PipelineState, agent: str,
                  was_probe: bool = False) -> None:
    """Trip (or re-trip) the backend-unreachable breaker, restarting the
    probation cooldown. Logs loudly only on the closed→open transition.

    `was_probe`: this failure was a dispatch the probation window let
    through as a live probe (or a rung-3 re-dispatch, which is a probe by
    construction). Two consecutive failed probes convert the breaker to
    TERMINAL — the run stops paying full timeouts for a backend that has
    now failed a first dispatch and two spaced probes."""
    was_open = state.backend_unreachable
    state.backend_unreachable = True
    state.backend_unreachable_at = time.monotonic()
    if was_probe:
        state.failed_probes += 1
        if state.failed_probes >= 2 and not state.backend_terminal:
            state.backend_terminal = True
            log(agent, "backend_unreachable",
                f"{state.failed_probes} consecutive probation probes failed; "
                "the breaker is now TERMINAL — every remaining dispatch "
                "fails fast and the run halts at its current stage")
            return
    if was_open:
        log(agent, "backend_unreachable",
            "probe dispatch failed with a backend-outage signature; breaker "
            "stays open and the probation cooldown restarts")
    else:
        log(agent, "backend_unreachable",
            "dispatch failed with a backend-outage signature; opening the "
            "circuit breaker — further dispatches fail fast for "
            f"{_BREAKER_PROBATION_S}s (then one probes the backend live) and "
            "best-effort TUI notices are skipped for the rest of this run")


def _close_breaker(state: PipelineState, agent: str, why: str) -> None:
    if not state.backend_unreachable:
        return
    state.backend_unreachable = False
    state.backend_unreachable_at = None
    state.failed_probes = 0
    log(agent, "breaker_closed", why)


_RUNG3_LIVENESS_SETTLE_S = 5.0

# Injectable so tests exercise rung 3 without real 120s cooldown waits
# (mirrors the pip-preflight pattern: default-on machinery, suite-off by
# an autouse fixture, dedicated tests re-enable).
_rung3_sleep = time.sleep


def _work_session_quiescent(
    state: PipelineState, session_id: str | None
) -> tuple[bool, str]:
    """The rung-3 liveness precheck (maintainer-approved interim, 2026-07-10, for
    the T1b alive-but-over-budget hazard): a timed-out turn can still be
    generating server-side and WRITING into the run dir (casualty #10 landed
    a correct artifact 11 minutes after its timeout) — a re-dispatch over
    that is two writers on one tree. Quiescent = the session's activity
    fingerprint is stable across a settle pause, or the session is gone.
    Unreadable (server can't answer) = NOT quiescent, conservative. The
    real abort-before-redispatch machinery stays with the taxonomy item."""
    if not session_id:
        return True, "no work session was created (the failure preceded it)"
    try:
        a = session_activity_fingerprint(session_id, port=state.port)
        if a is None:
            return True, "work session no longer exists"
        _rung3_sleep(_RUNG3_LIVENESS_SETTLE_S)
        b = session_activity_fingerprint(session_id, port=state.port)
    except (OpencodeClientError, ServerUnreachable) as e:
        return False, (f"liveness check could not read the session "
                       f"({type(e).__name__}: {e}); refusing conservatively")
    if b is None or a == b:
        return True, (f"activity fingerprint stable over "
                      f"{_RUNG3_LIVENESS_SETTLE_S:.0f}s")
    return False, (f"session is still generating (fingerprint moved "
                   f"{a!r} -> {b!r}) — a re-dispatch would race a live "
                   f"writer; halting as today")


def _rung3_transport_redispatch(
    state: PipelineState,
    *,
    agent: str,
    prompt: str,
    model,
    timeout_s: int,
    first_dispatch_id: str,
    first_error: BaseException,
    work_session_id: str | None,
) -> DispatchResult | None:
    """Rung 3 of the dispatch-recovery ladder: ONE bounded same-agent
    same-prompt re-dispatch per run when a transport failure would halt the
    stage. Waits out the probation cooldown (the re-dispatch doubles as the
    breaker's live probe), runs the session-liveness precheck, and
    re-dispatches into a FRESH session. Returns the successful
    DispatchResult, or None — the caller re-raises the original failure and
    the stage halts exactly as today. The attempt is consumed either way."""
    if state.backend_terminal or state.transport_retry_used:
        return None
    state.transport_retry_used = True
    open_for = time.monotonic() - (state.backend_unreachable_at or 0.0)
    wait_s = max(0.0, _BREAKER_PROBATION_S - open_for)
    log(agent, "dispatch_retry_after_transport",
        f"transport failure would halt the stage; waiting out {wait_s:.0f}s "
        f"probation cooldown, then one re-dispatch (rung 3)")
    if wait_s > 0:
        _rung3_sleep(wait_s)
    quiescent, evidence = _work_session_quiescent(state, work_session_id)
    if not quiescent:
        _append_run_event(
            state.paths,
            "dispatch_retry_skipped_live_session",
            status="failed",
            summary=f"{agent} rung-3 re-dispatch refused: prior work "
                    f"session not quiescent",
            details={"agent": agent, "dispatch_id": first_dispatch_id,
                     "work_session_id": work_session_id,
                     "evidence": evidence},
        )
        log(agent, "dispatch_retry_skipped_live_session", evidence)
        return None
    retry_dispatch_id = f"{agent}-{uuid.uuid4().hex[:12]}"
    _append_run_event(
        state.paths,
        "dispatch_retry_after_transport",
        summary=f"{agent} re-dispatch after transport failure (rung 3)",
        details={
            "agent": agent,
            "first_dispatch_id": first_dispatch_id,
            "retry_dispatch_id": retry_dispatch_id,
            "first_error": str(first_error)[:300],
            "cooldown_waited_s": round(wait_s, 1),
            "liveness_evidence": evidence,
        },
    )
    started = time.time()
    try:
        retry_session_id = create_session(
            title=f"r2c {agent} retry ({state.paths.slug})",
            parent_id=state.session_id,
            directory=state.workspace or None,
            port=state.port,
        )
        result = dispatch_and_wait(
            session_id=retry_session_id,
            agent=agent,
            prompt=prompt,
            model=model,
            port=state.port,
            timeout_s=timeout_s,
        )
    except Exception as e:
        _append_run_event(
            state.paths,
            "agent_dispatch_failed",
            status="failed",
            summary=f"{agent} rung-3 re-dispatch failed: {type(e).__name__}",
            details={"agent": agent, "dispatch_id": retry_dispatch_id,
                     "error": str(e), "elapsed_s": time.time() - started,
                     "rung": 3},
        )
        if _is_backend_outage(e):
            # The re-dispatch ran past the cooldown, so it IS a probe; a
            # second consecutive probe failure makes the breaker terminal.
            _open_breaker(state, agent, was_probe=True)
        return None
    _append_run_event(
        state.paths,
        "agent_dispatch_recovered",
        status="completed",
        summary=f"{agent} dispatch recovered by rung-3 transport re-dispatch",
        details={
            "agent": agent,
            "rung": 3,
            "first_dispatch_id": first_dispatch_id,
            "retry_dispatch_id": retry_dispatch_id,
            "elapsed_s": result.elapsed_s,
        },
    )
    log(agent, "dispatch_retry_after_transport",
        f"rung-3 re-dispatch succeeded in {result.elapsed_s:.1f}s; the "
        f"original failure stays on the casualty books")
    return result


def _timed_out_backend_is_alive(
    state: PipelineState, work_session_id: str | None
) -> tuple[bool, str]:
    """After a dispatch TIMEOUT, decide whether the backend is actually
    down or just still generating (ADAM 2026-07-13: a math-extractor turn
    completed its Write at 75s, then burned the per-step output cap on a
    follow-up step; the POST outlasted its 240s timeout, the driver
    declared a backend outage, and the circuit breaker killed stage 2b of
    a healthy run — the next paper dispatched normally 61s later).

    Discriminator: the work session was created FRESH for this dispatch, so
    any assistant part in it was produced during the timed-out window — proof
    the backend accepted the work and generated. Missing, unexpectedly aliased,
    unreadable, and empty sessions all report not-alive, and the breaker
    behaves conservatively."""
    if not work_session_id:
        return False, "no fresh work session to inspect"
    if work_session_id == state.session_id:
        return False, "fresh work session unexpectedly aliases the parent session"
    try:
        messages = opencode_client.fetch_session_messages(
            work_session_id, port=state.port)
    except OpencodeClientError as e:
        return False, f"work session unreadable ({type(e).__name__})"
    parts = sum(
        len(m.get("parts") or []) for m in messages
        if isinstance(m, dict)
        and (m.get("info") or {}).get("role") == "assistant")
    if parts:
        return True, (
            f"the fresh work session holds {parts} assistant part(s) "
            f"produced inside the timed-out window — the backend is "
            f"generating (likely a runaway or slow turn), not down")
    return False, "work session readable but the model produced nothing"


def dispatch_agent(
    *, state: PipelineState, agent: str, prompt: str, timeout_s: int
) -> DispatchResult:
    """Dispatch a named agent in the active session with its declared model.

    The model is resolved server-side via `GET /agent` (frontmatter + extends
    merged) and cached on PipelineState. The HTTP API would otherwise fall
    back to the session default; we pass `model` explicitly so each producer
    runs on the model its definition specifies (Think for analysis, Code for
    generation).

    Circuit breaker: once a prior dispatch this run tripped
    `state.backend_unreachable`, fail fast here instead of blocking for the
    full timeout against a backend we already know is down — but only within
    the probation cooldown. Past it this dispatch proceeds as a live probe
    (the backend flaps rather than dies; see `_BREAKER_PROBATION_S`)."""
    if state.backend_terminal:
        log(agent, "dispatch_skipped_backend_down",
            "breaker is TERMINAL (two consecutive failed probes); failing "
            "fast without dispatching")
        raise ServerUnreachable(
            f"{agent} dispatch skipped: circuit breaker TERMINAL after two "
            f"consecutive failed probation probes this run"
        )
    was_probe = False
    if state.backend_unreachable:
        open_for = time.monotonic() - (state.backend_unreachable_at or 0.0)
        if open_for < _BREAKER_PROBATION_S:
            log(agent, "dispatch_skipped_backend_down",
                f"backend marked unreachable {open_for:.0f}s ago (< "
                f"{_BREAKER_PROBATION_S}s probation); failing fast without "
                "dispatching")
            raise ServerUnreachable(
                f"{agent} dispatch skipped: backend marked unreachable earlier "
                f"this run (circuit breaker open)"
            )
        was_probe = True
        log(agent, "breaker_probation",
            f"breaker has been open {open_for:.0f}s (>= "
            f"{_BREAKER_PROBATION_S}s); allowing this dispatch as a live "
            "probe — success closes the breaker")
    model = state.agent_models.get(agent)
    dispatch_id = f"{agent}-{uuid.uuid4().hex[:12]}"
    timeout_s = opencode_client.scaled_timeout_s(timeout_s)
    log(agent, "dispatching", f"prompt={len(prompt)}chars, timeout={timeout_s:.0f}s")
    _append_run_event(
        state.paths,
        "agent_dispatch_started",
        summary=f"{agent} dispatch started",
        details={
            "agent": agent,
            "dispatch_id": dispatch_id,
            "prompt_chars": len(prompt),
            "timeout_s": timeout_s,
            # Retained as a compatibility field for event consumers. The
            # transport is unconditional for every new dispatch.
            "per_dispatch_session": True,
        },
    )
    started = time.time()
    work_session_id = None
    try:
        # Work isolation is structural: every agent turn runs in a FRESH child
        # of the user-facing TUI session. Creation stays inside the failure
        # boundary so POST /session errors are logged and fail loud; there is
        # deliberately no fallback to the parent session.
        work_session_id = create_session(
            title=f"r2c {agent} ({state.paths.slug})",
            parent_id=state.session_id,
            directory=state.workspace or None,
            port=state.port,
        )
        log(agent, "work_session_created",
            f"dispatching into fresh session {work_session_id} "
            f"nested under {state.session_id}")
        result = dispatch_and_wait(
            session_id=work_session_id,
            agent=agent,
            prompt=prompt,
            model=model,
            port=state.port,
            timeout_s=timeout_s,
        )
    except Exception as e:
        _append_run_event(
            state.paths,
            "agent_dispatch_failed",
            status="failed",
            summary=f"{agent} dispatch failed: {type(e).__name__}",
            details={
                "agent": agent,
                "dispatch_id": dispatch_id,
                "error": str(e),
                "elapsed_s": time.time() - started,
            },
        )
        if isinstance(e, opencode_client.DispatchExtensionCeiling):
            # Loud by design: one runaway turn silently burned ~4 wall-hours
            # on the SRL matrix row (declared 1800s, completed 14019s)
            # because a still-generating backend extends the socket-level
            # timeout without bound. From here the error follows the
            # ordinary timeout handling below — the proof-of-life check
            # keeps the breaker closed (the backend IS generating) and the
            # raise lands in stage-owned recovery, abandoning (not killing)
            # the generating session: the fedavg 2026-07-17 path.
            log(agent, "dispatch_extension_ceiling_hit", str(e))
            _append_run_event(
                state.paths,
                "dispatch_extension_ceiling_hit",
                status="failed",
                summary=(f"{agent} crossed the dispatch extension ceiling; "
                         f"the wait was abandoned and stage-owned recovery "
                         f"proceeds"),
                details={
                    "agent": agent,
                    "dispatch_id": dispatch_id,
                    "timeout_s": timeout_s,
                    "elapsed_s": time.time() - started,
                    "work_session_id": work_session_id,
                    "failure_class":
                        "unbounded_timeout_extension_on_generating_backend",
                },
            )
        outage = _is_backend_outage(e)
        if outage and isinstance(e, DispatchTimeout):
            # A timeout is ambiguous: dead backend vs a turn still
            # generating past the window. Only the former is an outage.
            alive, evidence = _timed_out_backend_is_alive(
                state, work_session_id)
            if alive:
                outage = False
                log(agent, "dispatch_timeout_backend_alive", evidence)
                _append_run_event(
                    state.paths,
                    "dispatch_timeout_backend_alive",
                    summary=(f"{agent} timed out but the backend is "
                             f"generating; breaker left closed"),
                    details={"agent": agent, "dispatch_id": dispatch_id,
                             "work_session_id": work_session_id,
                             "evidence": evidence},
                )
                # An abandoned turn keeps generating AND WRITING server-side
                # (2026-08-24: the test-generator's stray docs/ writes were
                # quarantined, then the still-running turn re-created them
                # minutes after the halt). Abort it before walking away.
                aborted = abort_session(work_session_id, port=state.port)
                abort_word = ("ok" if aborted
                              else "FAILED (best-effort; the turn may keep "
                                   "writing)")
                log(agent, "abandoned_turn_abort",
                    f"abort of the still-generating session "
                    f"{work_session_id}: {abort_word}")
                _append_run_event(
                    state.paths,
                    "abandoned_turn_aborted",
                    status="completed" if aborted else "failed",
                    summary=(f"{agent} abandoned turn "
                             f"{'aborted' if aborted else 'abort FAILED'}"),
                    details={"agent": agent, "dispatch_id": dispatch_id,
                             "work_session_id": work_session_id,
                             "aborted": aborted},
                )
        if outage:
            _open_breaker(state, agent, was_probe=was_probe)
        # Rung 3 (dispatch-recovery ladder): a transport failure (T1/T2)
        # that would halt the stage gets ONE bounded same-agent same-prompt
        # re-dispatch per run. The failure above stays on the books — a
        # recovered hang is still a hang. Content failures never ladder
        # (outage is False), completed-no-write turns never reach this
        # except block (they return normally; stage machinery owns T3).
        recovered = None
        if outage:
            recovered = _rung3_transport_redispatch(
                state, agent=agent, prompt=prompt, model=model,
                timeout_s=timeout_s, first_dispatch_id=dispatch_id,
                first_error=e, work_session_id=work_session_id)
        if recovered is None:
            raise
        result = recovered
        work_session_id = recovered.session_id or work_session_id
    _close_breaker(
        state, agent,
        "probe dispatch completed; the backend answers again — closing the "
        "circuit breaker")
    log(agent, "completed", f"{result.elapsed_s:.1f}s")
    _append_run_event(
        state.paths,
        "agent_dispatch_completed",
        status="completed" if result.completed else "failed",
        summary=f"{agent} dispatch completed",
        details={
            "agent": agent,
            "dispatch_id": dispatch_id,
            "elapsed_s": result.elapsed_s,
            "completed": result.completed,
            "error": result.error,
            "session_id": work_session_id,
        },
    )
    return result


# ---------------------------------------------------------------------------
# File-ownership enforcement
#
# Producer dispatches declare their writeable_paths via WRITEABLE_PATHS (in
# dispatch_templates.py). After each dispatch we diff the run dir against a
# pre-dispatch mtime snapshot; any file modified or created outside the
# allowlist is an out-of-scope edit and the driver halts.
#
# This catches the class of failure where an agent reaches outside its lane
# to satisfy a downstream finding, breaking a cross-file invariant the
# earlier stage's validators had already established. See dispatch_templates'
# WRITEABLE_PATHS docstring for the bev-distill motivating case.
# ---------------------------------------------------------------------------


class OutOfScopeWritesError(Exception):
    """Raised when a producer dispatch modified or created files outside
    its `WRITEABLE_PATHS` allowlist. The dispatch may have completed
    normally on the LLM side — this is a contract violation, not a
    server/network failure, and is handled by halt() in the stage handler."""

    def __init__(self, agent: str, violations: list[str], allowed: list[str]):
        self.agent = agent
        self.violations = list(violations)
        self.allowed = list(allowed)
        super().__init__(
            f"{agent} modified or created files outside its allowlist. "
            f"Out-of-scope writes: {self.violations}. Allowed: {self.allowed}."
        )


def _dispatch_error_halt_class(e: Exception) -> str:
    """Halt-catalog class for a failed dispatch (queue item 12).

    Many halt sites catch `(OpencodeClientError, OutOfScopeWritesError)`
    in one arm with a shared "dispatch failed" reason; the catalog class
    differs by which exception actually fired, so it is resolved here at
    halt time rather than duplicated across every site."""
    if isinstance(e, OutOfScopeWritesError):
        return "out_of_scope_write"
    return "transport_failure"


def _judge_invocation_halt_class(e: Exception) -> str:
    """Halt-catalog class for a failed halt-judge invocation.

    The judge-invocation sites catch `(OpencodeClientError,
    OutOfScopeWritesError, ValueError)` in one arm: ValueError means the
    judge's turn ran but wrote no usable decision (a producer-output
    problem), the other two are dispatch problems."""
    if isinstance(e, ValueError):
        return "producer_output_invalid"
    return _dispatch_error_halt_class(e)


def _pack_authoring_halt_class(e: Exception) -> str:
    """Halt-catalog class for a failed pack-authoring loop (B-13 change 1).

    AuthoringError is a genuine pack rejection — "nothing on your side is
    wrong" is the right researcher message. A dispatch or write-integrity
    failure must NOT be reported as that benign coverage gap; those route
    through the same resolver every other scope-halt site uses."""
    if isinstance(e, AuthoringError):
        return "gap_pack_rejected"
    return _dispatch_error_halt_class(e)


def _is_driver_managed(rel_path: str) -> bool:
    """True iff `rel_path` (relative to run_dir, POSIX-form) is written by
    the driver itself rather than any LLM agent. Driver-managed files are
    exempt from the out-of-scope-write check."""
    if rel_path == ".pipeline/driver_state.json":
        return True
    if rel_path in {".pipeline/run_events.jsonl", ".pipeline/progress.json"}:
        return True
    if rel_path in {
        ".pipeline/token_usage.json",       # ledger token harvest (decision 2)
        ".pipeline/history_pointer.json",   # run-history invocation identity
        ".pipeline/data_flow_slice.json",   # diagnostician input hygiene
        ".pipeline/notebook_for_review.ipynb",  # reviewer input hygiene
    }:
        # Driver-written observability/hygiene files from the 2026-07-14
        # build set. token_usage.json is written INSIDE the post-dispatch
        # classification — between the scope snapshot and the violation
        # check — so without this exemption the drift detector attributed
        # the driver's own write to the agent and deleted it on every
        # dispatch (caught live 40 minutes into the 0715 matrix, ACC row;
        # worse, a dispatch with invalid output would have burned its
        # corrective redispatch or halted with false attribution).
        return True
    if rel_path.startswith(".pipeline/run_events.jsonl.lock"):
        return True
    if rel_path.startswith(".pipeline/_lock"):
        return True
    if rel_path.startswith(".pipeline/quarantine/"):
        # Repo-scope boundary evidence: strays the DRIVER moved into the
        # run dir after a dispatch wrote outside it (repo_scope_boundary.py).
        # Driver-written, never an agent output.
        return True
    if rel_path.startswith(".pipeline/judge_decision_parts/"):
        # Halt-judge decision scratch, written by the judge under driver
        # orchestration and appended to judge_decisions.json. It is the
        # judge's own allowlisted output, never an artifact of whatever
        # producer dispatch is under scope check. A fix-loop dispatch that
        # snapshots the run dir while a judge decision part is present must
        # not flag it as an out-of-scope write by the producer (the cause of
        # the 2026-06-18 stage_2d arch-contract-fix halt).
        return True
    if rel_path.endswith(".halt"):
        # Agents may legitimately write halt sidecars (decomposer writes
        # paper_map.json.halt for non-decomposable papers, analyzer writes
        # method_spec.json.halt for paradigm mismatches, etc.). Halts are
        # protocol traffic between stages, not artifact writes.
        return True
    if rel_path.startswith(".pipeline/") and rel_path.endswith(".complete"):
        # Per-stage complete-sentinels written by the driver after a stage
        # returns StageResult(status="completed"). See
        # _stage_complete_sentinel_path. Never produced by agents.
        return True
    return False


def _is_python_bytecode_artifact(rel_path: str) -> bool:
    """True iff `rel_path` is Python interpreter bytecode (auto-generated as
    a side effect of `import`, not an explicit agent Write/Edit call).

    These shouldn't count as scope violations. An agent that imports its own
    output to verify it parses (legitimate behavior) would otherwise be
    halted because Python silently creates `__pycache__/*.pyc` files as a
    side effect — the mtime check is shape-blind to *how* the file was
    written. Exempting bytecode keeps the scope check focused on deliberate
    file writes."""
    parts = rel_path.split("/")
    return "__pycache__" in parts or rel_path.endswith(".pyc")


def _is_runtime_data_cache(rel_path: str) -> bool:
    """True iff `rel_path` sits inside a `_cache/` directory anywhere under
    the run dir. The generated loaders cache under `<caller path>/_cache`
    resolved against the working directory, so an agent that runs its own
    output to verify it — legitimate, contract-encouraged behavior —
    populates a cache wherever the call was made from. Runtime data is
    interpreter-style side traffic like bytecode, never authored content.
    The underscore-directory convention is the package's own rule for "side
    traffic, not data" (the generated data.py skips underscore-prefixed
    directories when scanning for user data).

    Two live fixtures: BADGE 2026-07-05 (nine MNIST cache artifacts under
    the designed method/example_data/_cache/ hard-halted a healthy stage-2b
    dispatch) and detr-distill 2026-07-08 (the same genus one
    path-generalization away: a self-test wrote `_cache/` at the RUN ROOT
    because callers legally choose the base path — queue item 19)."""
    return "_cache" in rel_path.split("/")[:-1]


# ---------------------------------------------------------------------------
# Repo-scope dispatch write boundary (2026-07-22)
#
# The run-dir scan above only sees writes INSIDE the run dir. Two recorded
# escapes wrote outside it entirely: a hallucinated sibling of the runs root
# (`<runs-sibling>/<slug>/.pipeline/...`) and an audit-style report at the
# repo root. This layer diffs repo-level state (git porcelain + top-level
# listings of the repo root and the runs root) around every dispatch and
# quarantines new strays into the run dir before raising the same
# OutOfScopeWritesError the run-dir scan uses, so halt-class resolution and
# judge routing engage unchanged. Mechanics, exemptions, and honest
# limitations live in repo_scope_boundary.py.
# ---------------------------------------------------------------------------


def _repo_scope_check_enabled() -> bool:
    """Rollback lever for the repo-scope write boundary. Default ON;
    disable with R2C_REPO_SCOPE_CHECK=0 (the standard env-lever
    convention)."""
    return os.environ.get("R2C_REPO_SCOPE_CHECK", "").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _repo_scope_exempt_rel_path(rel: str) -> bool:
    """Interpreter side-traffic exemptions, shared with the run-dir scan so
    both boundaries agree on what a deliberate write is."""
    return _is_python_bytecode_artifact(rel) or _is_runtime_data_cache(rel)


def _capture_repo_scope_baseline(
    state: "PipelineState",
) -> repo_scope_boundary.RepoScopeSnapshot | None:
    """Pre-dispatch repo-scope snapshot, or None when the check is disabled
    or capture itself failed (the boundary must never kill a dispatch)."""
    if not _repo_scope_check_enabled():
        return None
    try:
        snap = repo_scope_boundary.capture(
            state.paths.repo_root, state.paths.run_dir.parent)
    except Exception as e:  # noqa: BLE001 — observability layer, never fatal
        log("repo-scope", "capture_failed",
            f"{type(e).__name__}: {e}; repo-scope check skipped for this "
            f"dispatch")
        return None
    if snap.git_warning:
        log("repo-scope", "git_degraded", snap.git_warning)
    return snap


def _enforce_repo_scope_boundary(
    state: "PipelineState",
    *,
    agent: str,
    baseline: repo_scope_boundary.RepoScopeSnapshot | None,
    writeable: list[str] | None,
) -> None:
    """Post-dispatch repo-scope diff. On violations: quarantine new strays
    into `<run_dir>/.pipeline/quarantine/<dispatch-label>/`, log a
    structured event, and raise OutOfScopeWritesError naming every
    violating path and what happened to it. Modified/deleted tracked files
    are listed, never restored (halt-and-investigate). No-op when the
    baseline is None (lever off or capture failed)."""
    if baseline is None:
        return
    try:
        after = repo_scope_boundary.capture(
            state.paths.repo_root, state.paths.run_dir.parent)
        found = repo_scope_boundary.detect_violations(
            baseline, after,
            repo_root=state.paths.repo_root,
            runs_root=state.paths.run_dir.parent,
            run_dir=state.paths.run_dir,
            path_exempt=_repo_scope_exempt_rel_path,
        )
    except Exception as e:  # noqa: BLE001 — observability layer, never fatal
        log("repo-scope", "check_failed",
            f"{type(e).__name__}: {e}; repo-scope check skipped for this "
            f"dispatch")
        return
    if not found.any_violation():
        return
    label = "%s-%s" % (
        re.sub(r"[^A-Za-z0-9._-]+", "-", agent),
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
    )
    quarantined, leftovers = repo_scope_boundary.quarantine_new_strays(
        found.new_stray_paths(),
        repo_root=state.paths.repo_root,
        run_dir=state.paths.run_dir,
        label=label,
    )
    described = repo_scope_boundary.describe_violations(
        found, quarantined, leftovers)
    log(agent, "repo_scope_violation",
        f"dispatch wrote outside the run dir: {'; '.join(described)}")
    _append_run_event(
        state.paths,
        "repo_scope_violation_detected",
        summary=(f"{agent} wrote outside the run directory; "
                 f"{len(quarantined)} new stray path(s) quarantined, "
                 f"{len(found.modified_tracked) + len(found.deleted_tracked)} "
                 f"tracked file(s) left in place for investigation"),
        details={
            "agent": agent,
            "dispatch_label": label,
            "violations": described,
            "new_untracked": found.new_untracked,
            "modified_tracked": found.modified_tracked,
            "deleted_tracked": found.deleted_tracked,
            "new_repo_top_entries": found.new_repo_top_entries,
            "new_runs_top_entries": found.new_runs_top_entries,
            "quarantined": quarantined,
            "quarantine_leftovers": leftovers,
            "git_degraded": found.git_degraded,
        },
    )
    raise OutOfScopeWritesError(
        agent, described,
        list(writeable) if writeable else
        ["<paths inside the run directory only>"],
    )


def _snapshot_run_dir(run_dir: Path) -> dict[str, float]:
    """Snapshot mtime of every file under `run_dir` (recursively, POSIX-form
    relative paths). Used to detect out-of-scope writes after a dispatch."""
    snapshot: dict[str, float] = {}
    for entry in run_dir.rglob("*"):
        if not entry.is_file():
            continue
        rel = entry.relative_to(run_dir).as_posix()
        snapshot[rel] = entry.stat().st_mtime
    return snapshot


# Files larger than this aren't content-snapshotted (paper.md, large notebooks).
# Agents shouldn't be touching them; if they do, we can't revert, but the mtime
# check still catches the violation and we halt instead of recovering.
_CONTENT_SNAPSHOT_MAX_BYTES = 5 * 1024 * 1024  # 5 MiB


def _snapshot_run_dir_content(run_dir: Path) -> dict[str, bytes]:
    """Snapshot file content (not just mtime) for every file under `run_dir`.

    Used by the driver-side recovery path: when an agent's dispatch writes
    out-of-scope files but its declared canonical output is still valid,
    the driver restores the violated files from this snapshot and continues
    instead of halting. Bounded by `_CONTENT_SNAPSHOT_MAX_BYTES` per file —
    huge inputs (paper.md, notebook checkpoints) are skipped, and writes to
    them halt without recovery (which is the right behavior; agents have no
    reason to touch them)."""
    snapshot: dict[str, bytes] = {}
    for entry in run_dir.rglob("*"):
        if not entry.is_file():
            continue
        try:
            size = entry.stat().st_size
        except OSError:
            continue
        if size > _CONTENT_SNAPSHOT_MAX_BYTES:
            continue
        rel = entry.relative_to(run_dir).as_posix()
        try:
            snapshot[rel] = entry.read_bytes()
        except OSError:
            continue
    return snapshot


def _revert_out_of_scope_writes(
    run_dir: Path, violations: list[str], snapshot_content: dict[str, bytes]
) -> tuple[list[str], list[str]]:
    """Revert each violation: restore content from snapshot if the file
    existed pre-dispatch; delete it if the agent created it fresh.

    Returns (restored, deleted) lists of relative paths. Files not in the
    content snapshot AND not currently new are silently skipped (the size
    cap was exceeded, so we can't recover them — caller should already
    have halted in that case)."""
    restored: list[str] = []
    deleted: list[str] = []
    for rel in violations:
        full = run_dir / rel
        if rel in snapshot_content:
            try:
                full.write_bytes(snapshot_content[rel])
                restored.append(rel)
            except OSError:
                pass
        else:
            # New file the agent created — wasn't there pre-dispatch.
            try:
                if full.is_file():
                    full.unlink()
                    deleted.append(rel)
            except OSError:
                pass
    return restored, deleted


def _detect_out_of_scope_writes(
    run_dir: Path,
    snapshot: dict[str, float],
    writeable: list[str],
) -> list[str]:
    """Return a sorted list of relative paths that were modified or created
    between `snapshot` and now and are NOT covered by `writeable` (which
    supports fnmatch globs). Driver-managed paths are exempt."""
    return sorted(
        rel
        for rel in _changed_paths_since_snapshot(run_dir, snapshot)
        if not any(fnmatch.fnmatchcase(rel, pat) for pat in writeable)
    )


def _changed_paths_since_snapshot(
    run_dir: Path,
    snapshot: dict[str, float],
    *,
    include_driver_managed: bool = False,
) -> list[str]:
    """Return files modified or created since `snapshot`.

    Used by both the scope check and dispatch-timeout recovery. Keeping this
    separate lets recovery require an actual allowed artifact write instead of
    accepting a stale pre-existing file that merely still validates.

    By default driver-managed paths (driver_state.json, progress.json,
    judge_decision_parts/, .complete sentinels, halt sidecars) are excluded so
    the out-of-scope-write scope check never blames a producer for a driver
    write. Pass `include_driver_managed=True` for the no-write/truncation check,
    which must count an agent's OWN output even when that output lives under a
    driver-managed prefix — the halt-judge's sole legitimate output is
    `.pipeline/judge_decision_parts/*.json`, so excluding it made every
    successful judge dispatch look like a no-write (the false positive behind
    ~5 of 9 truncation events in the 2026-06-24 run). Python bytecode is always
    excluded (a `.pyc`-only side effect is not a deliberate write)."""
    changed: list[str] = []
    for entry in run_dir.rglob("*"):
        if not entry.is_file():
            continue
        rel = entry.relative_to(run_dir).as_posix()
        if _is_python_bytecode_artifact(rel):
            continue
        if _is_runtime_data_cache(rel):
            continue
        if not include_driver_managed and _is_driver_managed(rel):
            continue
        prior = snapshot.get(rel)
        current = entry.stat().st_mtime
        if prior is not None and prior == current:
            continue  # untouched
        changed.append(rel)
    return sorted(changed)


def _recovery_check_passed(agent: str, state: PipelineState, recovery_check_fn) -> bool:
    try:
        result = recovery_check_fn(state)
    except Exception as e:
        log(agent, "drift_recovery_check_failed",
            f"recovery check raised {type(e).__name__}: {e}; treating as unrecoverable")
        return False
    # Some checks (e.g. _render_and_validate_notebook) return (ok, detail). A
    # non-empty tuple is always truthy, so a FAILED (False, "...") tuple would
    # bool-coerce to True and read as PASSED — routing a broken artifact down
    # the recover-and-continue branch. Normalize to the leading element so the
    # A/B discriminator is the real verdict, not the tuple's truthiness.
    if isinstance(result, tuple):
        return bool(result[0]) if result else False
    return bool(result)


def _unrestorable_violations(
    violations: list[str],
    snapshot: dict[str, float],
    snapshot_content: dict[str, bytes],
) -> list[str]:
    """Pre-existing out-of-scope files absent from the content snapshot cannot
    be safely reverted, usually because they exceeded the snapshot size cap."""
    return [rel for rel in violations if rel in snapshot and rel not in snapshot_content]


def _recover_scoped_dispatch_after_error(
    *,
    state: PipelineState,
    agent: str,
    error: Exception,
    started: float,
    snapshot: dict[str, float],
    snapshot_content: dict[str, bytes] | None,
    writeable: list[str],
    recovery_check_fn,
) -> DispatchResult:
    """Recover a timed-out scoped dispatch when its canonical artifact landed.

    A blocking opencode POST can time out after the agent has already written
    the requested file. We only soft-complete when all of these are true:
      - this is an agent with a recovery validator,
      - at least one allowlisted output changed during this dispatch,
      - any out-of-scope changes are restorable from the pre-dispatch snapshot,
      - the recovery validator passes after out-of-scope changes are reverted.
    """
    changed = _changed_paths_since_snapshot(state.paths.run_dir, snapshot)
    allowed_changed = [
        rel for rel in changed
        if any(fnmatch.fnmatchcase(rel, pat) for pat in writeable)
    ]
    violations = [
        rel for rel in changed
        if not any(fnmatch.fnmatchcase(rel, pat) for pat in writeable)
    ]

    if recovery_check_fn is None or snapshot_content is None:
        if violations:
            raise OutOfScopeWritesError(agent, violations, writeable) from error
        raise error

    if not allowed_changed:
        log(agent, "dispatch_error_recovery_skipped",
            f"{type(error).__name__}: no allowlisted output changed; propagating error")
        if violations:
            raise OutOfScopeWritesError(agent, violations, writeable) from error
        raise error

    if not _recovery_check_passed(agent, state, recovery_check_fn):
        if violations:
            raise OutOfScopeWritesError(agent, violations, writeable) from error
        raise error

    if violations:
        unrestorable = _unrestorable_violations(
            violations, snapshot, snapshot_content,
        )
        if unrestorable:
            log(agent, "dispatch_error_recovery_skipped",
                f"out-of-scope writes cannot be restored: {unrestorable}")
            raise OutOfScopeWritesError(agent, violations, writeable) from error
        restored, deleted = _revert_out_of_scope_writes(
            state.paths.run_dir, violations, snapshot_content,
        )
        if not _recovery_check_passed(agent, state, recovery_check_fn):
            raise OutOfScopeWritesError(agent, violations, writeable) from error
    else:
        restored, deleted = [], []

    log(agent, "dispatch_error_recovered",
        f"{type(error).__name__}: canonical output validated after transport "
        f"error; changed={allowed_changed}; reverted={violations}")
    # The failed dispatch tripped the breaker on its way here, but the landed
    # artifact proves the backend actually did the work (the response path
    # died, not the model). This recovery means the run CONTINUES — leaving
    # the breaker open would fail the very next dispatch fast and turn the
    # recovery into a one-stage stay of execution (the self-defeat verified
    # on 2026-07-05).
    breaker_closed = state.backend_unreachable
    _close_breaker(
        state, agent,
        "canonical artifact landed during the failed dispatch; the backend "
        "did the work, so the continuing run closes the breaker instead of "
        "fast-failing its next dispatch")
    _append_run_event(
        state.paths,
        "agent_dispatch_recovered",
        status="completed",
        summary=f"{agent} dispatch recovered after {type(error).__name__}",
        details={
            "agent": agent,
            "error": str(error),
            "elapsed_s": time.time() - started,
            "allowed_changed": allowed_changed,
            "out_of_scope_restored": restored,
            "out_of_scope_deleted": deleted,
            "breaker_closed": breaker_closed,
        },
    )
    return DispatchResult(
        session_id=state.session_id,
        user_message_id="",
        assistant_message_id="",
        completed=False,
        error={"recovered_from": type(error).__name__, "message": str(error)},
        elapsed_s=time.time() - started,
    )


# Whole-run cap on graceful out-of-scope-write corrective re-dispatches, PER
# AGENT. Branch A (below) re-dispatches a wandered agent once with a corrective
# preamble; this bounds how many times that can happen for the same agent across
# a run so a recurring wanderer still halts for human inspection. The count is
# read off the persistent run-event log rather than held on PipelineState,
# because the resume path rebuilds PipelineState fresh (it does not reload
# driver_state.json) — the run dir's event log is the only shared layer that
# genuinely survives a resume. "Third wander halts": attempts 1 and 2 each get a
# corrective re-dispatch; the 3rd sees prior>=2 and halts.
_MAX_CORRECTIVE_REDISPATCHES_PER_AGENT = 2


def _prior_corrective_attempts(paths: PipelinePaths, agent: str) -> int:
    """Count prior out-of-scope corrective re-dispatch ATTEMPTS for `agent` in
    this run, from the persistent event log. Resume-safe by construction."""
    events_path = paths.pipeline_dir / "run_events.jsonl"
    if not events_path.is_file():
        return 0
    n = 0
    try:
        for line in events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (ev.get("event_type") == "out_of_scope_corrective_redispatch_attempted"
                    and (ev.get("details") or {}).get("agent") == agent):
                n += 1
    except OSError:
        return 0
    return n


def _corrective_redispatch_after_wander(
    *,
    state: PipelineState,
    agent: str,
    prompt: str,
    timeout_s: int,
    recovery_check_fn,
    writeable: list[str],
    first_violations: list[str],
) -> DispatchResult:
    """Branch A: an opted-in agent wrote out of scope AND its required output is
    invalid even after the strays are reverted — a wander, not a benign extra
    write. Re-dispatch the SAME agent ONCE in a fresh work session with an
    agent-neutral corrective preamble, then accept ONLY a clean result (no new
    out-of-scope writes, completeness passes, AND at least one allowlisted path
    written). Otherwise revert and raise so the stage handler halts.

    Loud: a run event + an assumptions.md entry the researcher reads at end of
    run. Bounded: a per-agent whole-run cap (recurring wanders still halt).

    The strays must already be reverted by the caller before this runs (the
    caller reverts and re-checks first, so a stray that was merely *masking*
    valid output continues silently without spending a re-dispatch)."""
    paths = state.paths
    run_dir = paths.run_dir

    prior = _prior_corrective_attempts(paths, agent)
    if prior >= _MAX_CORRECTIVE_REDISPATCHES_PER_AGENT:
        log(agent, "out_of_scope_corrective_exhausted",
            f"whole-run corrective cap reached ({prior} prior attempt(s)); halting. "
            f"violations={first_violations}")
        _append_run_event(
            paths, "out_of_scope_corrective_redispatch_exhausted",
            summary=f"{agent} exceeded the whole-run corrective re-dispatch cap",
            details={"agent": agent, "prior_attempts": prior,
                     "violations": first_violations, "reason": "whole_run_cap"},
        )
        raise OutOfScopeWritesError(agent, first_violations, writeable)

    # Loud surface BEFORE the attempt: a resume sees it even if the process dies
    # mid-dispatch, and this event is what the whole-run cap counts.
    _append_run_event(
        paths, "out_of_scope_corrective_redispatch_attempted",
        summary=f"{agent} wrote out of scope and its output did not validate; "
                f"re-dispatching once with a corrective preamble",
        details={"agent": agent, "violations": first_violations,
                 "prior_attempts": prior},
    )
    _append_assumption(
        state,
        aid=_next_assumption_id(state),
        title=f"Out-of-scope write recovered — {agent}",
        detected=f"{agent} wrote file(s) outside its assigned scope "
                 f"({', '.join(first_violations) or 'see run events'}) and its "
                 f"required output did not validate even after those files were "
                 f"removed.",
        action="The orchestrator removed the stray file(s) and re-dispatched the "
               "same agent once with an explicit scope-correction note, rather "
               "than halting the run.",
        reasoning="A single out-of-scope wander is usually transient task "
                  "confusion, not a defect needing human judgment; a bounded "
                  "corrective re-dispatch recovers it "
                  "without losing the run. The downstream stage validator remains "
                  "the authority on whether the re-produced output is correct.",
        alternative="Halt on the first out-of-scope write and inspect the agent's "
                    "transcript before resuming.",
        override="If this agent wanders here again, the run halts on the next "
                 "occurrence (whole-run cap); inspect the named file(s) and the "
                 "agent transcript.",
    )

    # Fresh snapshots taken NOW, after any prior completeness-check side effects
    # are already on disk (e.g. a rendered notebook.ipynb), so the corrective
    # dispatch is measured only against its OWN writes. Critical fix 1: the
    # completeness check is not side-effect-free; baselining after it keeps a
    # check-rendered file from being mistaken for a fresh stray and deleted.
    snapshot2 = _snapshot_run_dir(run_dir)
    snapshot2_content = _snapshot_run_dir_content(run_dir)
    # The corrective re-dispatch happens AFTER the wrapper's repo-scope
    # enforcement point, so it gets its own bracket — a wandering agent is
    # exactly the dispatch most likely to write outside the run dir.
    repo_baseline2 = _capture_repo_scope_baseline(state)

    corrective_prompt = (
        build_corrective_redispatch_preamble(first_violations, writeable) + prompt
    )
    started2 = time.time()
    try:
        result2 = dispatch_agent(
            state=state, agent=agent, prompt=corrective_prompt, timeout_s=timeout_s,
        )
    except OpencodeClientError as e:
        # ANY opencode-layer error on the corrective attempt — a transport
        # timeout / unreachable server (DispatchTimeout, ServerUnreachable) OR a
        # plain non-2xx / session-level OpencodeClientError — means the result is
        # unreliable. Catch the BASE class (not only the transport subclasses) so
        # a plain OpencodeClientError can't escape with new strays left on disk:
        # revert them, log, and re-raise the original error so the stage handler
        # halts. Never silently re-run.
        _revert_if_restorable(run_dir, snapshot2, snapshot2_content, writeable)
        _append_run_event(
            paths, "out_of_scope_corrective_redispatch_exhausted",
            summary=f"{agent} corrective re-dispatch failed with {type(e).__name__}",
            details={"agent": agent, "error": str(e), "reason": "dispatch_error"},
        )
        # After the run-dir revert: the failed corrective attempt may still
        # have written outside the run dir. A violation raised here outranks
        # the transport error (both routes halt).
        _enforce_repo_scope_boundary(
            state, agent=agent, baseline=repo_baseline2, writeable=writeable)
        raise

    _enforce_repo_scope_boundary(
        state, agent=agent, baseline=repo_baseline2, writeable=writeable)
    # Critical fix 1: detect the corrective dispatch's violations BEFORE running
    # the completeness check (which may render side-effect files). Mirrors the
    # first-dispatch ordering — detect, then validate.
    new_violations = _detect_out_of_scope_writes(run_dir, snapshot2, writeable)
    allowed_changed = [
        rel for rel in _changed_paths_since_snapshot(run_dir, snapshot2)
        if any(fnmatch.fnmatchcase(rel, pat) for pat in writeable)
    ]
    check_ok = _recovery_check_passed(agent, state, recovery_check_fn)
    success = (not new_violations) and check_ok and bool(allowed_changed)

    if success:
        log(agent, "out_of_scope_corrective_redispatch_succeeded",
            f"corrective re-dispatch produced valid in-scope output; "
            f"changed={allowed_changed}; continuing")
        _append_run_event(
            paths, "out_of_scope_corrective_redispatch_succeeded",
            status="completed",
            summary=f"{agent} produced valid output after a scope-correction "
                    f"re-dispatch",
            details={"agent": agent, "allowed_changed": allowed_changed,
                     "first_violations": first_violations,
                     "elapsed_s": time.time() - started2},
        )
        return result2

    # Failure: revert any new strays and halt. The stage handler's existing
    # `except OutOfScopeWritesError: halt` takes it from here.
    _revert_if_restorable(run_dir, snapshot2, snapshot2_content, writeable)
    log(agent, "out_of_scope_corrective_exhausted",
        f"corrective re-dispatch still invalid (new_violations={new_violations}, "
        f"check_ok={check_ok}, allowed_changed={bool(allowed_changed)}); halting")
    _append_run_event(
        paths, "out_of_scope_corrective_redispatch_exhausted",
        summary=f"{agent} still produced invalid or out-of-scope output after "
                f"correction",
        details={"agent": agent, "new_violations": new_violations,
                 "check_ok": check_ok, "allowed_changed": bool(allowed_changed),
                 "reason": "still_invalid"},
    )
    raise OutOfScopeWritesError(agent, new_violations or first_violations, writeable)


def _revert_if_restorable(
    run_dir: Path,
    snapshot: dict[str, float],
    snapshot_content: dict[str, bytes] | None,
    writeable: list[str],
) -> None:
    """Revert any currently-detected out-of-scope writes that are restorable.
    Best-effort cleanup on a failed/aborted corrective attempt; over-cap strays
    are left in place (the caller is halting anyway)."""
    if snapshot_content is None:
        return
    violations = _detect_out_of_scope_writes(run_dir, snapshot, writeable)
    if not violations:
        return
    unrestorable = _unrestorable_violations(violations, snapshot, snapshot_content)
    restorable = [v for v in violations if v not in unrestorable]
    if restorable:
        _revert_out_of_scope_writes(run_dir, restorable, snapshot_content)


def _strays_match_other_agent_owned(violations: list[str]) -> list[str]:
    """Out-of-scope strays whose path matches some agent's declared
    WRITEABLE_PATHS glob. A stray is by definition already outside the
    dispatching agent's OWN allowlist, so a match here means it landed on
    ANOTHER agent's owned output path — a role-confusion wander serious enough
    to halt for human investigation rather than silently revert + corrective
    re-dispatch (the snapshot alone cannot tell a fresh creation from a
    re-creation of that other agent's legitimate output). Branch A only; branch
    B's documented B-004 recovery (a producer that ALSO writes a
    stage_review_*.json) is unaffected — it routes through the valid-output
    path, never here."""
    owned = [pat for pats in WRITEABLE_PATHS.values() for pat in pats]
    return [
        rel for rel in violations
        if any(fnmatch.fnmatchcase(rel, pat) for pat in owned)
    ]


# Agents that produce NO code/artifact output — their only legitimate write is
# a driver-managed scratch file (the judge's decision part). For these, an
# out-of-scope write onto another agent's owned path can never be load-bearing,
# so the artifact under review is restored from snapshot before halting (rather
# than left mutated for investigation as it is for producers). Audit 2026-06-25
# J1: the halt-judge edited method/method.py and the driver halted with the file
# left corrupted; reverting first preserves the reviewed artifact's integrity.
_READ_ONLY_DECISION_AGENTS = frozenset({"r2c-halt-judge"})


def _short_empty_resume_qualifies(
    *, tool_calls: int, final_step_output: int | None,
) -> bool:
    """Is this dead-under-the-cap turn worth resuming? (R2C-060)

    Yes when the session shows at least one completed tool call: the
    investigation happened and only the write is missing, so a resume
    recovers work a fresh re-roll would pay for twice.

    Yes for the single-token shape too, whatever it did: one token is an
    inference-side immediate stop, transient by nature, and an in-session
    retry costs one prompt when there is nothing to lose.

    No for a session with no completed work and a real output — nothing to
    preserve, and a fresh roll gets a clean context, which is the existing
    path."""
    if tool_calls > 0:
        return True
    return final_step_output is not None and final_step_output <= 1


def _cap_burn_corrective_resume(
    state: PipelineState,
    *,
    agent: str,
    first_result: DispatchResult,
    timeout_s: int,
    writeable: list[str],
    snapshot: dict[str, float],
    first_burns: list[dict],
    shape: str = "cap_burn",
    nudge: str = CAP_BURN_CORRECTIVE_RESUME_NUDGE,
) -> DispatchResult | None:
    """ONE corrective resume of a work session that died without landing a
    write (cap-burn measurement note, fix 2 — maintainer-approved 2026-07-17;
    extended to the short-empty shape 2026-08-05 per R2C-060).

    A dead turn's session still holds everything it paid for, and the
    stage-owned retry re-rolls the SAME prompt on the same model — which
    reproduced the same ramble three consecutive times on SRL (2026-07-15,
    method-coder, 20k tokens plus 5-10 minutes per repeat). So the cheapest
    rung on the recovery ladder is resuming THAT session with a short
    corrective nudge: issue the write now, in bounded chunks.

    Two shapes qualify, each with its own nudge wording:
      - `cap_burn`: the final step hit the per-step output cap with no tool
        call. The artifact was composed and never issued.
      - `short_empty`: the turn ended UNDER the cap without its write, in a
        session that shows at least one completed tool call (or emitted a
        single token, an inference-side immediate stop). The 2026-08-05
        pdfgnn diagnostician did four investigation steps and seven file
        reads, then spent 15.6k tokens of reasoning without ever writing;
        the driver fallback authored a generic diagnosis and that
        fallback SHIPPED as the researcher-facing known-issues entry
        (`abandoned_investigation`).

    Guards, each load-bearing:
      - fires only from the no-write branch of `_dispatch_with_scope_check`
        (never on self-recovered turns, never on unmeasured shapes);
      - exactly ONE resume attempt per dispatch; on a second burn, a resume
        that still writes nothing, or ANY failure, it returns None and the
        caller falls through to the existing stage-owned recovery unchanged,
        so retry budgets never stack;
      - a resume failure is contained here (no raise, no breaker changes):
        the original dispatch already "completed" on the transport level and
        the stage machinery owns this failure class.

    The run-event names keep their `cap_burn_` prefix, which named the first
    shape this mechanism covered; `details.shape` carries which one fired so
    the history stays comparable across the extension.

    Returns the resume's DispatchResult when an allowlisted write landed —
    the caller swaps it in and the normal allowlist/violation flow re-checks
    the outputs — or None to fall through."""
    session_id = first_result.session_id
    if not session_id or session_id == state.session_id:
        # No isolated work session to resume into (defensive: the no-write
        # branch only runs after a completed per-dispatch-session turn).
        return None
    log(agent, "cap_burn_corrective_resume",
        f"resuming dead work session {session_id} once with the "
        f"corrective write-now nudge ({shape}, timeout {timeout_s}s)")
    _append_run_event(
        state.paths,
        "cap_burn_corrective_resume_attempted",
        summary=(f"{agent} corrective resume after a dead turn "
                 f"({shape}; one attempt, same work session)"),
        details={"agent": agent, "session_id": session_id, "shape": shape,
                 "burned_steps": len(first_burns), "timeout_s": timeout_s},
    )
    started = time.time()
    try:
        resumed = dispatch_and_wait(
            session_id=session_id,
            agent=agent,
            prompt=nudge,
            model=state.agent_models.get(agent),
            port=state.port,
            timeout_s=timeout_s,
        )
    except Exception as e:
        reason = "resume_dispatch_failed"
        if isinstance(e, opencode_client.DispatchExtensionCeiling):
            # The runaway class struck DURING the resume: label it
            # distinctly and keep the ceiling loud here too, exactly as
            # dispatch_agent does for a first-try dispatch. Worst-case wall
            # for one burned dispatch is ~4x the declared timeout (2x
            # ceiling on the original turn + 2x on the resume) — bounded
            # by design.
            reason = "resume_extension_ceiling"
            log(agent, "dispatch_extension_ceiling_hit", str(e))
            _append_run_event(
                state.paths,
                "dispatch_extension_ceiling_hit",
                status="failed",
                summary=(f"{agent} corrective resume crossed the dispatch "
                         f"extension ceiling; the wait was abandoned and "
                         f"stage-owned recovery proceeds"),
                details={
                    "agent": agent,
                    "session_id": session_id,
                    "timeout_s": timeout_s,
                    "elapsed_s": time.time() - started,
                    "context": "cap_burn_corrective_resume",
                    "failure_class":
                        "unbounded_timeout_extension_on_generating_backend",
                },
            )
        log(agent, "cap_burn_corrective_resume",
            f"resume dispatch failed ({type(e).__name__}: {e}); falling "
            f"through to stage-owned recovery")
        _append_run_event(
            state.paths,
            "cap_burn_corrective_resume_exhausted",
            status="failed",
            summary=f"{agent} corrective resume failed; stage recovery owns it",
            details={"agent": agent, "session_id": session_id,
                     "reason": reason,
                     "error": str(e)[:300],
                     "elapsed_s": time.time() - started},
        )
        return None
    own_writes = [
        rel
        for rel in _changed_paths_since_snapshot(
            state.paths.run_dir, snapshot, include_driver_managed=True
        )
        if any(fnmatch.fnmatchcase(rel, pat) for pat in writeable)
    ]
    # A resume that wrote ONLY out-of-scope files leaves own_writes empty and
    # intentionally converts to the scope-violation path: it falls through as
    # exhausted and the caller's violation check halts on the strays —
    # ownership enforcement wins, and the budget extended by one bounded step.
    if own_writes:
        log(agent, "cap_burn_corrective_resume",
            f"resume landed allowlisted write(s) in {resumed.elapsed_s:.1f}s: "
            f"{own_writes}; re-checking through the normal allowlist flow")
        _append_run_event(
            state.paths,
            "cap_burn_corrective_resume_succeeded",
            status="completed",
            summary=f"{agent} corrective resume landed the write",
            details={"agent": agent, "session_id": session_id, "shape": shape,
                     "writes": own_writes, "elapsed_s": resumed.elapsed_s},
        )
        # Token note: the pre-resume harvest already counted this dispatch
        # once (record_dispatch_tokens increments dispatches_counted); the
        # resume turn's tokens are deliberately not re-harvested to keep
        # each dispatch counted once — a small, one-sided undercount.
        return resumed
    reason = "no_write_after_resume"
    try:
        messages = opencode_client.fetch_session_messages(
            session_id, port=state.port)
        if len(opencode_client.cap_burn_steps(messages)) > len(first_burns):
            reason = "second_cap_burn"
    except OpencodeClientError:
        pass
    log(agent, "cap_burn_corrective_resume",
        f"resume completed but wrote nothing ({reason}); falling through "
        f"to stage-owned recovery")
    _append_run_event(
        state.paths,
        "cap_burn_corrective_resume_exhausted",
        status="failed",
        summary=f"{agent} corrective resume wrote nothing ({reason})",
        details={"agent": agent, "session_id": session_id, "reason": reason,
                 "elapsed_s": time.time() - started},
    )
    return None


def _dispatch_with_scope_check(
    *, state: PipelineState, agent: str, prompt: str, timeout_s: int,
    recovery_check_fn=None, writeable_paths_override: list[str] | None = None,
    allow_corrective_redispatch: bool = False,
) -> DispatchResult:
    """`dispatch_agent` + post-dispatch out-of-scope-write enforcement,
    with optional drift recovery.

    If `agent` has an entry in `WRITEABLE_PATHS`, snapshot the run dir
    before dispatching; after the dispatch returns, diff and raise
    `OutOfScopeWritesError` if any file outside the allowlist was modified
    or created. Otherwise behave exactly like `dispatch_agent`.

    `recovery_check_fn` (optional): a callable taking `state` and returning
    bool. Used by Think-class agents that have shown after-write drift the
    file-ownership check catches (most notably the smoke-diagnostician).
    When the function is provided AND scope violations are detected, the
    driver calls it; if it returns True, the violations are reverted from
    a pre-dispatch content snapshot and the dispatch is treated as a soft
    success (the agent produced its canonical output AND some side
    effects; we keep the output, drop the side effects). If False, the
    OutOfScopeWritesError raises as normal. This works WITH the LLM's
    "complete the task" prior instead of fighting it — four layers of
    prompt-level hardening for the diagnostician didn't deter the drift,
    so we recover from it instead. (This is "branch B".)

    `allow_corrective_redispatch` (default False): opt a dispatch into
    "branch A" — graceful recovery from a WANDER. When violations are present,
    the output is invalid even after the strays are reverted (so it is NOT a
    benign extra write), and the strays are restorable, the driver re-dispatches
    the SAME agent ONCE in a fresh work session with an agent-neutral scope-
    correction preamble, accepts only a clean re-result, and otherwise halts.
    The recovery is loud (run event + assumptions.md entry) and whole-run-capped
    per agent (`_corrective_redispatch_after_wander`). Left False, a wander still
    hard-halts exactly as before. Opted in only for the agents whose retry
    surface was audited (stage-reviewer, method-coder, halt-judge) plus the
    paper-fidelity reviewer's stage-4 dispatch (added 2026-07-03 after the
    detr-distill sidecar-overwrite wander, the maintainer signed off — reversing the
    2026-06-25 exclusion for that one dispatch); everything else —
    Stage-1 canonical dispatches whose chunk-mode fallback is handled
    upstream, the diagnostician, the fidelity reviewer's fix/incremental
    dispatches — keeps the hard halt.

    A content snapshot is taken only when `recovery_check_fn` is provided
    (cost: ~50ms per dispatch and bounded memory).

    Independent of the allowlist scan, every dispatch (scoped or not) is
    bracketed by the repo-scope write boundary: a git-porcelain + top-level
    listing diff of the repo root and the runs root that quarantines new
    strays written OUTSIDE the run dir and raises the same
    OutOfScopeWritesError (see repo_scope_boundary.py; rollback lever
    R2C_REPO_SCOPE_CHECK=0)."""
    writeable = (
        list(writeable_paths_override)
        if writeable_paths_override is not None
        else WRITEABLE_PATHS.get(agent)
    )
    repo_baseline = _capture_repo_scope_baseline(state)
    if writeable is None:
        try:
            return dispatch_agent(
                state=state, agent=agent, prompt=prompt, timeout_s=timeout_s,
            )
        finally:
            # Runs on the error path too: a failed dispatch may still have
            # written the stray. A violation raised here outranks the
            # transport error (both routes halt; this one names the paths).
            _enforce_repo_scope_boundary(
                state, agent=agent, baseline=repo_baseline,
                writeable=writeable)
    snapshot = _snapshot_run_dir(state.paths.run_dir)
    snapshot_content: dict[str, bytes] | None = None
    if recovery_check_fn is not None:
        snapshot_content = _snapshot_run_dir_content(state.paths.run_dir)
    started = time.time()
    try:
        result = dispatch_agent(
            state=state, agent=agent, prompt=prompt, timeout_s=timeout_s,
        )
    except (DispatchTimeout, ServerUnreachable) as e:
        # Repo-scope boundary first: a timed-out dispatch may already have
        # written outside the run dir, and the artifact-recovery path below
        # only reasons about run-dir contents.
        _enforce_repo_scope_boundary(
            state, agent=agent, baseline=repo_baseline, writeable=writeable)
        return _recover_scoped_dispatch_after_error(
            state=state,
            agent=agent,
            error=e,
            started=started,
            snapshot=snapshot,
            snapshot_content=snapshot_content,
            writeable=writeable,
            recovery_check_fn=recovery_check_fn,
        )
    own_writes = [
        rel
        for rel in _changed_paths_since_snapshot(
            state.paths.run_dir, snapshot, include_driver_managed=True
        )
        if any(fnmatch.fnmatchcase(rel, pat) for pat in writeable)
    ]
    if not own_writes:
        # Truncated/no-write turn (B-002): the dispatch "completed" but the
        # producer wrote nothing matching its OWN writeable allowlist — not even
        # its canonical output. We test the agent's writeable globs INCLUDING
        # driver-managed prefixes (e.g. `.pipeline/judge_decision_parts/*.json`,
        # the halt-judge's sole legitimate output) so a SUCCESSFUL judge /
        # diagnostician dispatch is no longer mis-flagged as a no-write — the
        # false positive that polluted ~5 of 9 truncation events on 2026-06-24
        # and masked the genuine notebook-generator no-write in the same stream.
        # Seen live (analyzer, arch-coder, notebook-generator). Detection
        # stays the default: every stage already owns its recovery for this
        # class (decomposer/analyzer/reviewer missing-output retries, the
        # judge no-decision retry, the coder fix loops) — an unconditional
        # dispatch-layer retry would stack on those and double dispatch
        # counts on persistent failures. ONE exception, promoted 2026-07-17
        # with the cap-burn measurement note as the evidence this comment
        # used to ask for: the cap_burn sub-shape gets a single corrective
        # SAME-SESSION resume below (never a re-roll), because the
        # stage-owned re-roll of an identical prompt reproduced identical
        # burns (SRL 07-15, three consecutive method-coder burns).
        # Sub-shape split (item 23's last piece, the item 8 taxonomy
        # datum): cap-burn (final step at the per-step output cap, no
        # tool call — a deterministic, resumable transport-class death)
        # vs short-empty (ended early with a small output count) vs
        # unmeasured (session unreadable or the dying step never got its
        # step-finish). Best-effort by design: classification failure
        # never touches the dispatch outcome.
        turn_shape, final_step_output = "unmeasured", None
        burns: list[dict] = []
        tool_calls = 0
        try:
            session_messages = opencode_client.fetch_session_messages(
                result.session_id, port=state.port)
            # Ledger decision 2 (2026-07-14): per-run token totals, harvested
            # from the fresh work-session messages already fetched for
            # turn-shape classification. Each dispatch is counted once.
            run_history.record_dispatch_tokens(
                state.paths.pipeline_dir, agent=agent,
                messages=session_messages)
            burns = opencode_client.cap_burn_steps(session_messages)
            if burns:
                # Any cap-burned step in a turn that wrote nothing is the
                # cap-burn class — the 2026-07-13 audit found the burn one
                # message BEFORE a short stub step (ICRA) and inside
                # truncated write-call JSON (iDb-RRT), both mislabeled
                # short_empty by the final-step-only detector.
                turn_shape = "cap_burn"
                final_step_output = burns[-1]["output_tokens"]
            else:
                turn_shape = "short_empty"
                final_step_output = opencode_client.final_step_output_tokens(
                    session_messages)
            tool_calls = opencode_client.completed_tool_calls(session_messages)
        except OpencodeClientError:
            pass
        log(agent, "truncated_turn_detected",
            f"dispatch completed in {result.elapsed_s:.1f}s without "
            f"writing any files in its writeable paths "
            f"(turn shape: {turn_shape})")
        _append_run_event(
            state.paths,
            "truncated_turn_detected",
            summary=f"{agent} completed without writing any files",
            details={"agent": agent, "elapsed_s": result.elapsed_s,
                     "writeable_paths": list(writeable),
                     "turn_shape": turn_shape,
                     "completed_tool_calls": tool_calls,
                     "final_step_output_tokens": final_step_output},
        )
        if turn_shape == "short_empty" and _short_empty_resume_qualifies(
                tool_calls=tool_calls, final_step_output=final_step_output):
            # R2C-060: the investigation happened and only the write is
            # missing, so the session is worth more than a fresh re-roll.
            log(agent, "short_empty_resume_qualified",
                f"turn ended under the cap with {tool_calls} completed tool "
                f"call(s) and a {final_step_output}-token final step; "
                f"resuming the session rather than discarding its work")
            resumed = _cap_burn_corrective_resume(
                state, agent=agent, first_result=result,
                timeout_s=timeout_s, writeable=writeable,
                snapshot=snapshot, first_burns=burns,
                shape="short_empty",
                nudge=SHORT_EMPTY_CORRECTIVE_RESUME_NUDGE)
            if resumed is not None:
                result = resumed
        elif turn_shape == "cap_burn":
            truncated_call = any(b.get("tool_call_truncated") for b in burns)
            shape_note = (
                "a tool call outgrew the cap and its JSON arrived cut off"
                if truncated_call else "no tool call was ever issued")
            log(agent, "cap_burn_turn_detected",
                f"{len(burns)} step(s) burned the per-step output cap "
                f"({final_step_output} tokens) — {shape_note}; the write "
                f"never landed and a plain resume re-rolls this turn")
            _append_run_event(
                state.paths,
                "cap_burn_turn_detected",
                summary=(f"{agent} died at the per-step output cap "
                         f"({final_step_output} tokens, "
                         f"{'truncated tool call' if truncated_call else 'no tool call'})"),
                details={"agent": agent, "elapsed_s": result.elapsed_s,
                         "final_step_output_tokens": final_step_output,
                         "burned_steps": len(burns),
                         "tool_call_truncated": truncated_call,
                         "session_id": result.session_id},
            )
            # Cap-burn fix 2: one corrective resume of the SAME session.
            # On success the resumed result flows through the normal
            # allowlist/violation checks below exactly like a first-try
            # write; on any failure the original result falls through to
            # stage-owned recovery unchanged (budgets never stack).
            resumed = _cap_burn_corrective_resume(
                state, agent=agent, first_result=result,
                timeout_s=timeout_s, writeable=writeable,
                snapshot=snapshot, first_burns=burns)
            if resumed is not None:
                result = resumed
    else:
        # Measurement-only sweep for SELF-RECOVERED cap burns: the turn
        # wrote its output, but one or more steps burned the full cap on
        # the way (ms3d and iDb-RRT 2026-07-13: write calls whose JSON
        # truncated at the cap, retried successfully in a later step).
        # Fleet-state signal only — never touches the dispatch outcome.
        try:
            session_messages = opencode_client.fetch_session_messages(
                result.session_id, port=state.port)
            # Same token harvest as the no-write branch (ledger decision 2).
            run_history.record_dispatch_tokens(
                state.paths.pipeline_dir, agent=agent,
                messages=session_messages)
            recovered_burns = opencode_client.cap_burn_steps(session_messages)
        except OpencodeClientError:
            recovered_burns = []
        if recovered_burns:
            # Order-neutral wording: the burn can precede OR follow the
            # write that landed (ADAM 2026-07-14 burned AFTER its write;
            # iDb burned before). Only the outcome is asserted.
            log(agent, "cap_burn_recovered",
                f"{len(recovered_burns)} step(s) burned the per-step "
                f"output cap, but the dispatch landed its writes and "
                f"succeeded — {sum(b['output_tokens'] for b in recovered_burns)} "
                f"output tokens wasted")
            _append_run_event(
                state.paths,
                "cap_burn_recovered",
                summary=(f"{agent} self-recovered from "
                         f"{len(recovered_burns)} cap-burned step(s)"),
                details={"agent": agent, "elapsed_s": result.elapsed_s,
                         "burned_steps": len(recovered_burns),
                         "session_id": result.session_id},
            )
    # Repo-scope boundary before the run-dir violation handling: writes
    # OUTSIDE the run dir are never recoverable via branch A/B, so they
    # short-circuit straight to quarantine + halt.
    _enforce_repo_scope_boundary(
        state, agent=agent, baseline=repo_baseline, writeable=writeable)
    violations = _detect_out_of_scope_writes(
        state.paths.run_dir, snapshot, writeable,
    )
    if violations:
        if recovery_check_fn is not None and snapshot_content is not None:
            unrestorable = _unrestorable_violations(
                violations, snapshot, snapshot_content,
            )
            if not unrestorable:
                if _recovery_check_passed(agent, state, recovery_check_fn):
                    # Branch B: the required output is valid even WITH the strays
                    # present — a benign extra write. Drop the strays, re-confirm,
                    # continue silently. (No re-dispatch; this is the common,
                    # cheap case.)
                    restored, deleted = _revert_out_of_scope_writes(
                        state.paths.run_dir, violations, snapshot_content,
                    )
                    if _recovery_check_passed(agent, state, recovery_check_fn):
                        log(agent, "drift_recovered",
                            f"reverted {len(restored)} restored + {len(deleted)} "
                            f"deleted out-of-scope writes; canonical output is valid; "
                            f"continuing. violations={violations}")
                        return result
                    # Reverting the strays broke a previously-passing output: the
                    # stray was load-bearing. Ambiguous — fall through to halt.
                elif allow_corrective_redispatch:
                    # Branch A: the required output is INVALID even with the
                    # strays present — a wander.
                    owned_strays = _strays_match_other_agent_owned(violations)
                    if owned_strays and agent in _READ_ONLY_DECISION_AGENTS:
                        # J1: a read-only decision agent (halt-judge) wrote onto
                        # another agent's owned path. It has no load-bearing
                        # output, so the stray is pure corruption of the artifact
                        # under review — revert it (restore from snapshot) before
                        # halting, so the reviewed artifact is preserved, not left
                        # mutated. The decision was invalid, so we still halt
                        # (fall through to the raise) for investigation.
                        restored, deleted = _revert_out_of_scope_writes(
                            state.paths.run_dir, violations, snapshot_content,
                        )
                        log(agent, "out_of_scope_owned_file_reverted_halt",
                            f"read-only judge wrote another agent's owned path(s) "
                            f"({owned_strays}); reverted {len(restored)} restored "
                            f"+ {len(deleted)} deleted to preserve the artifact "
                            f"under review, then halting. violations={violations}")
                    elif owned_strays:
                        # A stray landed on ANOTHER agent's owned output path —
                        # too serious to auto-recover. Halt for investigation with
                        # the strays preserved on disk (fall through to the raise
                        # below); do NOT revert or re-dispatch.
                        log(agent, "out_of_scope_owned_file_halt",
                            f"out-of-scope stray(s) match another agent's owned "
                            f"paths ({owned_strays}); halting for investigation "
                            f"instead of corrective re-dispatch. violations={violations}")
                    else:
                        # First try a plain revert: a stray may merely have been
                        # MASKING valid output (e.g. shadowing an import), in which
                        # case removing it is enough and we continue silently
                        # without spending a re-dispatch.
                        restored, deleted = _revert_out_of_scope_writes(
                            state.paths.run_dir, violations, snapshot_content,
                        )
                        if _recovery_check_passed(agent, state, recovery_check_fn):
                            log(agent, "drift_recovered",
                                f"reverted {len(restored)} restored + {len(deleted)} "
                                f"deleted out-of-scope writes; output valid after revert "
                                f"(stray was masking it); continuing. violations={violations}")
                            return result
                        # Genuine wander: re-dispatch the same agent once (loud +
                        # whole-run-capped), or halt if exhausted. Strays already
                        # reverted above, so the corrective dispatch starts clean.
                        return _corrective_redispatch_after_wander(
                            state=state, agent=agent, prompt=prompt, timeout_s=timeout_s,
                            recovery_check_fn=recovery_check_fn, writeable=writeable,
                            first_violations=violations,
                        )
        raise OutOfScopeWritesError(agent, violations, writeable)
    return result


def _classification_from_spec(paths: PipelinePaths) -> dict | None:
    if not paths.method_spec.exists():
        return None
    try:
        spec = json.loads(paths.method_spec.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    classification = spec.get("comparison", {}).get("classification") or {}
    return classification if isinstance(classification, dict) else None


def taxonomy_node_ref_from_spec(paths: PipelinePaths) -> str | None:
    """Return an SSOT node reference for taxonomy-served specs, else None."""
    classification = _classification_from_spec(paths)
    if not classification:
        return None
    paradigm_id = classification.get("id")
    if not isinstance(paradigm_id, str) or not paradigm_id:
        return None
    try:
        tax = load_taxonomy(paths.repo_root)
    except FileNotFoundError:
        tax = load_taxonomy(REPO_ROOT)
    node = serves(paradigm_id, tax)
    if node is None:
        return None
    taxonomy_id = getattr(node, "taxonomy_id", getattr(node, "id", paradigm_id))
    return f"docs/ssot/taxonomies.yaml#{taxonomy_id}"

def provisional_pack_ref(paths: PipelinePaths) -> str | None:
    """Return the run-local provisional pack path, if one was installed."""
    manifest_path = paths.provisional_pack_manifest
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    pack_rel = manifest.get("pack_path") if isinstance(manifest, dict) else None
    if not isinstance(pack_rel, str) or not pack_rel:
        return None
    pack_path = paths.repo_root / pack_rel
    if not pack_path.is_file():
        return None
    return pack_rel


def build_paths_block(paths: PipelinePaths) -> DispatchPaths:
    """Standard paths block populated from PipelinePaths."""
    fg = (
        taxonomy_node_ref_from_spec(paths)
        or provisional_pack_ref(paths)
        or "(analyzer determines from paper)"
    )
    return DispatchPaths(
        spec=str(paths.method_spec),
        paper=str(paths.paper_md),
        paper_map=str(paths.paper_map),
        run_dir=str(paths.run_dir),
        taxonomy_source=fg,
    )


# ---------------------------------------------------------------------------
# Stage 0 — setup + paper parsing
# ---------------------------------------------------------------------------


def _parse_pdf_budget_s() -> float:
    """Stage-0 subprocess budget for parse_pdf.py: every attempt the parser
    will make plus slack, from the same env knobs it reads."""
    from parse_pdf import total_budget_s  # noqa: PLC0415
    return total_budget_s()


def _pdf_parse_cache_valid(paths: PipelinePaths, quality_report: Path) -> bool:
    if paths.input_kind != "pdf":
        return False
    if not paths.paper_md.is_file() or not quality_report.is_file():
        return False
    try:
        report = json.loads(quality_report.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(report, dict) or not report.get("passed"):
        return False
    if not paths.input_path.is_file():
        return False
    expected_path = str(paths.input_path.resolve())
    try:
        expected_hash = sha256_file(paths.input_path)
    except OSError:
        return False
    return (
        report.get("source_pdf_path") == expected_path
        and report.get("source_pdf_sha256") == expected_hash
    )


def _archive_incompatible_run_dir(
    run_dir: Path, *, today: str | None = None
) -> Path | None:
    """Move an existing but layout-incompatible run dir aside, never delete it.

    Item 29 (researcher run, 07-08): a researcher's first solo run failed because a `pdwa/` dir
    from a previous version of R2C sat under r2c_runs/, and he had to figure
    out the delete himself. Minting unique run-dir names per invocation is not
    the answer — same-dir resume is load-bearing (the documented halt-recovery
    path, every halted REPORT's printed next step). Instead, only reuse a dir
    the current driver understands; move an incompatible one aside and start
    fresh.

    Returns the archive path when a move happened, else None. Three cases
    return None and leave today's behavior exactly as it was: a fresh slug (no
    dir), and a current-layout dir (resume, running or halted or delivered).
    A dir that exists but is not layout-compatible is renamed to
    `<slug>_pre-upgrade-<date>` (a numeric suffix on collision) and the caller
    proceeds fresh. A live run is always layout-compatible (it wrote a current
    progress.json), so it is never archived — the run lock refuses the
    duplicate launch as before."""
    if not run_dir.exists():
        return None
    if is_run_dir_layout_compatible(run_dir):
        return None
    stamp = today or utc_now_iso()[:10]
    base = run_dir.parent / f"{run_dir.name}_pre-upgrade-{stamp}"
    archive = base
    suffix = 2
    while archive.exists():
        archive = run_dir.parent / f"{base.name}-{suffix}"
        suffix += 1
    shutil.move(str(run_dir), str(archive))
    log(
        "stage_0",
        "run_dir_archived",
        f"existing '{run_dir.name}' run dir predates the current layout "
        f"(no current {PROGRESS_FILE_NAME}); moved aside to '{archive.name}' "
        f"and starting fresh (nothing deleted)",
    )
    return archive


def _archive_for_fresh_roll(run_dir: Path) -> Path | None:
    """Deliberately archive an existing run dir to `<slug>_<n>` for a fresh roll.

    Item 29 phase 2: `/r2c-run <slug> --fresh` is the researcher's explicit
    "start over, keep the old delivery" request. Unlike
    `_archive_incompatible_run_dir` (which fires automatically ONLY on a
    layout-incompatible dir and uses a `_pre-upgrade-<date>` suffix), this
    archives regardless of layout compatibility and uses a bare numeric suffix —
    a `<slug>_<n>` folder exists ONLY because a researcher CHOSE a fresh roll, so
    the suffix carries that meaning. Never deletes. Returns the archive path, or
    None when there is no dir to archive.

    The caller guards against archiving a dir a live run is holding (a present
    `_lock`), so this never yanks the directory out from under a running roll."""
    if not run_dir.exists():
        return None
    n = 1
    archive = run_dir.parent / f"{run_dir.name}_{n}"
    while archive.exists():
        n += 1
        archive = run_dir.parent / f"{run_dir.name}_{n}"
    shutil.move(str(run_dir), str(archive))
    log(
        "stage_0",
        "run_dir_archived",
        f"--fresh: archived the current '{run_dir.name}' delivery to "
        f"'{archive.name}' and starting a clean roll (nothing deleted)",
    )
    return archive


# --- Item 30: dependency-source preflight + wheelhouse -----------------------
#
# On a corporate network that blackholes external PyPI (researcher run,
# 07-08, corporate WSL), the run died 30+ minutes in when it installed the
# paper's heavy deps at
# stage 2.d. Dependency readiness is decided only after the finalizer has
# written requirements.txt: an already-prepared environment proceeds without
# an index probe or package-manager invocation, while a missing environment
# gets fast, actionable guidance. The wheelhouse remains an optional offline
# source for users who prefer it to installing while temporarily on another
# network.

# Set by the stage-2.d source preflight, read by the install helper to decide
# --no-index. None means the source preflight did not run (do not force offline).
_PIP_INDEX_REACHABLE: bool | None = None


def _resolve_wheelhouse(repo_root: Path) -> Path | None:
    """The local wheel directory to feed pip installs, or None.

    `R2C_WHEELHOUSE` (an explicit path) wins; otherwise a conventional
    `r2c_wheelhouse/` at the repo root is used when it exists. A value that is
    set but does not point at a directory is ignored (treated as unset) rather
    than failing the run."""
    env = os.environ.get("R2C_WHEELHOUSE", "").strip()
    if env:
        candidate = Path(env).expanduser()
        if candidate.is_dir():
            return candidate
    default = repo_root / "r2c_wheelhouse"
    if default.is_dir():
        return default
    return None


def _pip_index_reachable(*, timeout: float = 8.0) -> bool:
    """Best-effort check that pip could reach a package index.

    A single short request to the effective index URL (`PIP_INDEX_URL` if set,
    else PyPI) through urllib's default opener, which honors the system proxy
    env vars the way pip does. Any HTTP response counts as reachable; a
    connection error, timeout, or TLS failure counts as unreachable. This does
    not read pip.conf, so a proxy configured only there yields a false negative
    — the halt message names the R2C_SKIP_PIP_PREFLIGHT override for that."""
    index_url = (
        os.environ.get("PIP_INDEX_URL", "").strip() or "https://pypi.org/simple/"
    )
    try:
        req = urllib.request.Request(index_url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        # A real HTTP status (e.g. 403/405 to a HEAD) still proves reachability.
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _pip_wheelhouse_install_args(repo_root: Path) -> list[str]:
    """pip flags that route installs through the wheelhouse when one exists.

    `--find-links <dir>` whenever a wheelhouse is resolved, plus `--no-index`
    when the preflight found the index unreachable (fully offline) so pip does
    not stall trying to contact it. Empty list when there is no wheelhouse."""
    wheelhouse = _resolve_wheelhouse(repo_root)
    if wheelhouse is None:
        return []
    args = ["--find-links", str(wheelhouse)]
    if _PIP_INDEX_REACHABLE is False:
        args.append("--no-index")
    return args


def _preflight_pip_for_install(paths: PipelinePaths, *, stage_id: str) -> str | None:
    """Decide whether an unsatisfied environment has a usable package source.

    Called only after requirements.txt exists and a read-only metadata check
    found missing or incompatible dependencies. Returns None to install, or a
    short halt reason. A locally satisfied environment never calls this helper,
    so returning to a restricted network after an off-network install performs
    no index probe. A configured wheelhouse remains a valid offline source."""
    global _PIP_INDEX_REACHABLE
    if _env_truthy("R2C_SKIP_PIP_PREFLIGHT"):
        log(stage_id, "pip_preflight", "skipped (R2C_SKIP_PIP_PREFLIGHT set)")
        return None
    wheelhouse = _resolve_wheelhouse(paths.repo_root)
    if wheelhouse is not None:
        _PIP_INDEX_REACHABLE = False
        log(stage_id, "pip_preflight",
            f"using the configured wheelhouse at {wheelhouse} without probing "
            "a package index (--find-links --no-index)")
        return None
    if _PIP_INDEX_REACHABLE is not True:
        _PIP_INDEX_REACHABLE = _pip_index_reachable()
    if _PIP_INDEX_REACHABLE:
        return None
    log(stage_id, "pip_preflight",
        "package index unreachable and no wheelhouse configured; halting before "
        "the dependency install")
    return "pip cannot reach a package index and no wheelhouse is configured"


def _dependency_halt_user_message(
    paths: PipelinePaths,
    check: dependency_readiness.LocalRequirementsCheck,
) -> str:
    """Researcher guidance for the switch-network/install/resume workflow."""

    requirements_path = paths.run_dir / run_layout.REQUIREMENTS_TXT
    try:
        display_path = requirements_path.relative_to(paths.repo_root)
    except ValueError:
        display_path = requirements_path
    install_command = (
        f"{shlex.quote(sys.executable)} -m pip install -r "
        f"{shlex.quote(str(display_path))}"
    )
    problems = check.problem_descriptions
    problem_lines = "\n".join(f"  - {problem}" for problem in problems[:12])
    if len(problems) > 12:
        problem_lines += f"\n  - ... and {len(problems) - 12} more"

    return (
        f"R2C wrote this paper's dependency list to `{display_path}`, but the "
        "current Python environment is not ready:\n"
        f"{problem_lines}\n"
        "This machine cannot reach a Python package index, so R2C stopped "
        "without invoking pip. To use a network-switch workflow:\n"
        "  1. Connect this same machine to a network where package installation "
        "works.\n"
        "  2. From the R2C repo root, install into the exact Python environment "
        f"used by this run:\n     `{install_command}`\n"
        "  3. Return to the restricted network and resume the same run. Stage "
        "2.d will check the installed versions locally and continue without "
        "probing the index or invoking pip once every requirement is satisfied.\n"
        "Alternatives: set `PIP_INDEX_URL` to a reachable internal mirror, or "
        "configure `R2C_WHEELHOUSE` for a fully offline install. If pip works "
        "through pip.conf but this reachability check cannot see that setup, "
        "resume with `R2C_SKIP_PIP_PREFLIGHT=1`."
    )


def run_stage_0(
    paper_input: str,
    *,
    workspace: str,
    fresh: bool = False,
    before_parse: Callable[[PipelinePaths], None] | None = None,
) -> tuple[PipelinePaths | None, StageResult]:
    """Setup + paper parsing. Pure-script: no LLM dispatch.

    Returns (paths, result). If setup fails before paths can be resolved,
    paths is None and the caller surfaces the halt directly."""
    stage_id = "stage_0"
    log(stage_id, "started", f"input={paper_input}")

    proc = run_script(
        stage_id,
        ["scripts/setup_pipeline_dirs.py", paper_input, "--resolve-only"],
        timeout=60,
    )
    if proc.returncode != 0:
        # No pipeline_dir yet; surface the halt to the workspace root.
        artifact = build_halt_artifact(
            stage=stage_id,
            reason=f"setup_pipeline_dirs.py --resolve-only exit {proc.returncode}",
            context={"stderr": proc.stderr[-2000:]},
            user_message=_HALT_MSG_BAD_INPUT,
        )
        halt_path = Path(workspace) / f"{stage_id}.halt"
        halt_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        log(stage_id, "halted", f"wrote {halt_path}")
        return None, StageResult(
            status="halted", stage_id=stage_id,
            notes="setup_pipeline_dirs.py resolve failed", halt_artifact=artifact,
        )

    try:
        setup_result = json.loads(proc.stdout.strip())
    except json.JSONDecodeError as e:
        raise SystemExit(
            f"Stage 0: setup_pipeline_dirs.py stdout not JSON: {e}\nstdout: {proc.stdout[:500]}"
        )
    paths = PipelinePaths.from_setup_result(setup_result)
    # Item 29: before anything writes into the run dir (the lock in
    # before_parse creates .pipeline/), decide resume vs fresh.
    #   --fresh (phase 2): the researcher explicitly asked to start over — archive
    #     the current delivery to <slug>_<n> and roll clean, WHATEVER its layout.
    #     But never yank a dir a live run is holding: when the lock would refuse
    #     this launch, skip the archive and let the lock acquisition below do the
    #     refusing (kill the running roll first, then --fresh).
    #
    #     This asks whether the lock BLOCKS rather than whether it EXISTS, and
    #     the difference matters (R2C-076 follow-up): a lock whose owner is
    #     provably dead gets reclaimed by the acquisition below, so treating it
    #     as live here skipped the archive and then let the roll through anyway,
    #     starting a supposedly clean roll in the old directory on top of its
    #     stale stage-complete sentinels. One staleness rule, both consumers.
    #   otherwise (phase 1): resume a dir this driver understands; an older-version
    #     dir is moved aside so its stale sentinels never corrupt this run.
    if fresh and run_lock_blocks_launch(paths.pipeline_dir):
        log(stage_id, "fresh_skipped",
            f"--fresh requested but '{paths.slug}' is locked by a running roll "
            f"({describe_run_lock_owner(paths.pipeline_dir)}); "
            f"not archiving — the lock will refuse this launch")
        archived_run_dir = None
    elif fresh:
        archived_run_dir = _archive_for_fresh_roll(paths.run_dir)
    else:
        archived_run_dir = _archive_incompatible_run_dir(paths.run_dir)
    fresh_roll = fresh and archived_run_dir is not None
    if before_parse is not None:
        before_parse(paths)
    if archived_run_dir is not None:
        # Now that the fresh dir + lock exist, surface the move to the
        # researcher timeline (the log line already went to the driver log).
        if fresh_roll:
            summary = (
                f"You asked for a fresh roll (--fresh), so R2C archived the "
                f"current '{paths.slug}' delivery to '{archived_run_dir.name}' "
                f"and started clean — nothing was deleted, your previous files "
                f"are in that folder."
            )
        else:
            summary = (
                f"Found an existing '{paths.slug}' run folder from an older "
                f"version of R2C. Moved it aside to '{archived_run_dir.name}' "
                f"and started a fresh run — nothing was deleted, your previous "
                f"files are in that folder."
            )
        _append_run_event(
            paths,
            "run_dir_archived",
            stage_id="stage_0",
            status="archived",
            summary=summary,
            details={
                "archived_to": archived_run_dir.name,
                "reason": "fresh_flag" if fresh_roll else "incompatible_layout",
            },
        )

    proc = run_script(
        stage_id, ["scripts/setup_pipeline_dirs.py", paper_input], timeout=60
    )
    if proc.returncode != 0:
        return paths, halt(
            paths,
            stage_id,
            reason=f"setup_pipeline_dirs.py exit {proc.returncode}",
            halt_class="bad_input_file",
            context={"stderr": proc.stderr[-2000:]},
            notes="setup_pipeline_dirs.py failed",
            user_message=_HALT_MSG_BAD_INPUT,
        )
    try:
        setup_result = json.loads(proc.stdout.strip())
    except json.JSONDecodeError as e:
        raise SystemExit(
            f"Stage 0: setup_pipeline_dirs.py stdout not JSON: {e}\nstdout: {proc.stdout[:500]}"
        )
    paths = PipelinePaths.from_setup_result(setup_result)

    paths_written: list[Path] = []
    quality_report = paths.pipeline_dir / "parse_quality.json"
    pdf_needs_parse = (
        paths.input_kind == "pdf"
        and not _pdf_parse_cache_valid(paths, quality_report)
    )
    if paths.paper_md.exists() and not pdf_needs_parse:
        log(stage_id, "paper_md", f"{paths.paper_md} present; skipping parse")
    elif paths.input_kind == "pdf":
        if paths.paper_md.exists():
            log(stage_id, "pdf_parse_cache_invalid",
                f"{paths.paper_md} cache missing or mismatched provenance; reparsing")
        log(stage_id, "parsing_pdf", f"{paths.input_path} -> {paths.paper_md}")
        proc = run_script(
            stage_id,
            [
                "scripts/parse_pdf.py",
                str(paths.input_path),
                "--output",
                str(paths.paper_md),
                "--quality-report",
                str(quality_report),
            ],
            timeout=_parse_pdf_budget_s(),
        )
        if proc.returncode != 0:
            from parse_pdf import PARSER_MISSING_EXIT  # noqa: PLC0415
            parser_missing = proc.returncode == PARSER_MISSING_EXIT
            return paths, halt(
                paths, stage_id,
                reason=f"parse_pdf.py exit {proc.returncode}",
                halt_class="bad_input_file",
                context={"stderr": proc.stderr[-2000:]},
                notes="PDF parser not installed" if parser_missing else "parse_pdf failed",
                user_message=(_HALT_MSG_PDF_PARSER_MISSING if parser_missing
                              else _HALT_MSG_BAD_INPUT),
            )
        paths_written.append(paths.paper_md)

    # Embedded base64 image payloads make every downstream agent read a
    # screen of image data instead of the paper (Interaction_Tax 08-27: 90%
    # of paper.md was payload, 830k-token analyzer input). One enforcement
    # point for every input kind and cache state: paper.md never carries a
    # data-URI payload past stage 0. Full text is archived beside it first,
    # so the strip stays reversible and auditable (same convention as the
    # input_papers/ base64 guard).
    if paths.paper_md.is_file():
        paper_text = paths.paper_md.read_text(encoding="utf-8")
        stripped_text, payload_count = replace_payloads_with_placeholders(
            paper_text
        )
        if payload_count:
            archive = paths.paper_md.with_name(paths.paper_md.stem + "_images.md")
            if not archive.exists():
                archive.write_text(paper_text, encoding="utf-8")
            paths.paper_md.write_text(stripped_text, encoding="utf-8")
            log(
                stage_id, "paper_md_base64_stripped",
                f"{payload_count} embedded image payload(s) replaced with "
                f"figure-N.png placeholders; full text archived at {archive}",
            )
            for written in (archive, paths.paper_md):
                if written not in paths_written:
                    paths_written.append(written)

    if paths.input_kind == "pdf":
        proc = run_script(
            stage_id,
            [
                "scripts/pdf_parse_quality.py",
                str(paths.paper_md),
                "--report",
                str(quality_report),
                "--source-pdf",
                str(paths.input_path),
            ],
            timeout=60,
        )
        if proc.returncode != 0:
            return paths, halt(
                paths, stage_id,
                reason=f"PDF parse quality gate failed for {paths.paper_md}",
                halt_class="bad_input_file",
                context={
                    "stdout": proc.stdout[-2000:],
                    "stderr": proc.stderr[-2000:],
                },
                notes="PDF parse quality gate failed",
                user_message=_HALT_MSG_BAD_INPUT,
            )
        paths_written.append(quality_report)

    return paths, StageResult(
        status="completed",
        stage_id=stage_id,
        notes=f"paper.md present at {paths.paper_md}",
        paths_written=paths_written,
    )


# ---------------------------------------------------------------------------
# Stage 1 — analyzer + paper_map + feasibility gate + stage-reviewer
# ---------------------------------------------------------------------------


def _paper_map_valid(state: PipelineState) -> bool:
    """Recovery check for decomposer drift: paper_map.json exists and is
    a JSON dict with `elements`."""
    if not state.paths.paper_map.is_file():
        return False
    try:
        data = json.loads(state.paths.paper_map.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return isinstance(data, dict) and isinstance(data.get("elements"), list)


def _method_spec_valid(state: PipelineState) -> bool:
    """Recovery check for analyzer drift: method_spec.json exists and is
    a JSON dict with `core_method`."""
    if not state.paths.method_spec.is_file():
        return False
    try:
        data = json.loads(state.paths.method_spec.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return isinstance(data, dict) and isinstance(data.get("core_method"), dict)


def _paper_map_parts_dir(paths: PipelinePaths) -> Path:
    return paths.pipeline_dir / "paper_map_parts"


def _method_spec_parts_dir(paths: PipelinePaths) -> Path:
    return paths.pipeline_dir / "method_spec_parts"


def _clear_stage_1_parts_dir(parts_dir: Path) -> None:
    """Clear stale chunk files before a fresh producer dispatch.

    Without this, a no-output dispatch could be masked by parts from a prior
    run. These are agent-owned generated scratch artifacts; the canonical
    downstream files remain paper_map.json and method_spec.json.
    """
    if parts_dir.is_dir():
        shutil.rmtree(parts_dir)


def _assembly_context(result: AssemblyResult | None) -> dict[str, object]:
    if result is None:
        return {"attempted": False}
    return {
        "attempted": True,
        "assembled": result.assembled,
        "message": result.message,
        "source_dir": str(result.source_dir),
        "output_path": str(result.output_path),
        "part_count": len(result.part_paths),
    }


def _json_parse_error(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"{path}: invalid JSON: {exc}"
    except OSError as exc:
        return f"{path}: failed to read JSON: {exc}"
    return None


def _stage_1_chunk_mode_violation(
    exc: OutOfScopeWritesError, *, parts_rel_prefix: str,
) -> bool:
    """True when a canonical-only Stage 1 dispatch wrote only chunk files.

    This is not accepted as success because the driver owns output-mode
    selection, but it is retryable: clear the attempted chunks and redispatch
    with an explicit chunk-only prompt.
    """
    return bool(exc.violations) and all(
        rel.startswith(parts_rel_prefix) for rel in exc.violations
    )


def _stage_1_canonical_retry_write(
    exc: OutOfScopeWritesError, *, canonical_rel_path: str,
) -> bool:
    """True when a chunk-only retry ignored mode and wrote its canonical.

    The prompt still tells the agent not to do this.  The canonical is allowed
    to reach the normal Stage 1 validator even when it is malformed; output
    precedence and artifact validity are separate decisions.
    """
    return bool(exc.violations) and all(
        rel == canonical_rel_path for rel in exc.violations
    )


@dataclass(frozen=True)
class _Stage1FileRevision:
    """Content and filesystem revision evidence for one Stage 1 output."""

    mtime_ns: int
    ctime_ns: int
    size: int
    sha256: str


@dataclass(frozen=True)
class _Stage1ArtifactHaltSnapshot:
    artifact: _Stage1FileRevision | None
    halt_sidecar: _Stage1FileRevision | None


def _stage_1_file_revision(path: Path) -> _Stage1FileRevision | None:
    """Return a stable-enough revision fingerprint, or None if unreadable.

    Stage 1 artifacts are small JSON files.  Including content alongside
    nanosecond timestamps avoids treating an unrelated, future-dated file as a
    current-dispatch write and catches rewrites whose size did not change.
    """
    try:
        stat = path.stat()
        if not path.is_file():
            return None
        digest = sha256_file(path)
    except OSError:
        return None
    return _Stage1FileRevision(
        mtime_ns=stat.st_mtime_ns,
        ctime_ns=stat.st_ctime_ns,
        size=stat.st_size,
        sha256=digest,
    )


def _snapshot_stage_1_artifact_halt(
    artifact: Path, halt_sidecar: Path,
) -> _Stage1ArtifactHaltSnapshot:
    """Capture both sides immediately before one Stage 1 dispatch."""
    return _Stage1ArtifactHaltSnapshot(
        artifact=_stage_1_file_revision(artifact),
        halt_sidecar=_stage_1_file_revision(halt_sidecar),
    )


def _stage_1_halt_sidecar_is_authoritative(
    state: PipelineState,
    *,
    artifact: Path,
    halt_sidecar: Path,
    before_dispatch: _Stage1ArtifactHaltSnapshot,
    agent: str,
    attempt: str,
) -> bool:
    """Apply the shared completed-artifact-versus-halt precedence contract.

    A sidecar remains authoritative unless the canonical artifact demonstrably
    changed during this dispatch.  If both files changed, the canonical wins
    only when its nanosecond mtime is strictly later: halt-then-reconsider is a
    completed artifact, while artifact-then-halt (and ambiguous ties) remains a
    halt.  Winning here says nothing about validity; the normal parser and
    validator/fix loop own that decision.
    """
    after_halt = _stage_1_file_revision(halt_sidecar)
    if after_halt is None:
        return False

    after_artifact = _stage_1_file_revision(artifact)
    artifact_changed = (
        after_artifact is not None
        and after_artifact != before_dispatch.artifact
    )
    halt_changed = after_halt != before_dispatch.halt_sidecar

    if not artifact_changed:
        return True
    if (
        halt_changed
        and after_artifact.mtime_ns <= after_halt.mtime_ns
    ):
        return True

    reason = (
        "artifact_changed_after_stale_sidecar"
        if not halt_changed
        else "artifact_written_after_current_halt_sidecar"
    )
    try:
        halt_sidecar.unlink()
    except OSError as exc:
        log(
            "stage_1",
            "halt_sidecar_supersede_failed",
            f"could not remove {halt_sidecar.name}: {exc}; keeping halt authoritative",
        )
        return True

    artifact_rel = artifact.relative_to(state.paths.run_dir).as_posix()
    halt_rel = halt_sidecar.relative_to(state.paths.run_dir).as_posix()
    log(
        "stage_1",
        "artifact_halt_sidecar_superseded",
        f"{agent} completed {artifact.name} after its halt signal; "
        "continuing to validation",
    )
    _append_run_event(
        state.paths,
        "artifact_halt_sidecar_superseded",
        stage_id="stage_1",
        status="completed",
        summary=(
            f"{agent} completed {artifact.name} after its halt signal; "
            "the completed artifact will continue to validation"
        ),
        artifacts=[artifact_rel],
        details={
            "agent": agent,
            "attempt": attempt,
            "artifact": artifact_rel,
            "halt_sidecar": halt_rel,
            "precedence_reason": reason,
            "artifact_changed_in_dispatch": artifact_changed,
            "halt_changed_in_dispatch": halt_changed,
            "artifact_mtime_ns": after_artifact.mtime_ns,
            "halt_sidecar_mtime_ns": after_halt.mtime_ns,
        },
    )
    return False


def _assemble_paper_map_if_fresh(state: PipelineState) -> AssemblyResult | None:
    """Assemble paper_map_parts when no canonical file exists or parts are newer."""
    parts_dir = _paper_map_parts_dir(state.paths)
    if not parts_dir.is_dir():
        return None
    if state.paths.paper_map.exists() and not parts_newer_than_output(
        parts_dir, state.paths.paper_map
    ):
        return None
    result = assemble_paper_map_parts(state.paths.pipeline_dir, state.paths.paper_map)
    log(
        "stage_1",
        "paper_map_parts_assembled" if result.assembled else "paper_map_parts_assembly_failed",
        result.message,
    )
    return result


def _assemble_method_spec_if_fresh(state: PipelineState) -> AssemblyResult | None:
    """Assemble method_spec_parts when no canonical file exists or parts are newer."""
    parts_dir = _method_spec_parts_dir(state.paths)
    if not parts_dir.is_dir():
        return None
    if state.paths.method_spec.exists() and not parts_newer_than_output(
        parts_dir, state.paths.method_spec
    ):
        return None
    result = assemble_method_spec_parts(state.paths.pipeline_dir, state.paths.method_spec)
    log(
        "stage_1",
        "method_spec_parts_assembled" if result.assembled else "method_spec_parts_assembly_failed",
        result.message,
    )
    return result


def _method_py_valid(state: PipelineState) -> bool:
    """Recovery check for method-coder drift: method/method.py exists, parses
    as Python, and defines the spec's pluggable-component function at top level
    (falls back to "has >=1 top-level def" if the spec can't be read).

    Used so a dispatch that produced a valid method.py PLUS an out-of-scope
    side effect — e.g. the method-coder inlining the stage-reviewer's job and
    writing `.pipeline/stage_review_stage_2c_method.json` (B-004, pdwa
    2026-05-27) — is treated as a soft success: the side effect is reverted
    from the pre-dispatch snapshot, the canonical method.py is kept, and the
    fix loop's real stage-reviewer dispatch writes the review legitimately."""
    import ast as _ast  # local: only needed here

    method_py = state.paths.run_dir / "method" / "method.py"
    if not method_py.is_file():
        return False
    try:
        tree = _ast.parse(method_py.read_text(encoding="utf-8"))
    except (SyntaxError, OSError, ValueError):
        return False
    top_level_defs = {
        node.name
        for node in tree.body
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef))
    }
    if not top_level_defs:
        return False
    # If the spec is readable, require the pluggable function specifically;
    # otherwise accept any top-level function (parses + non-empty surface).
    try:
        spec = json.loads(state.paths.method_spec.read_text(encoding="utf-8"))
        pluggable = (
            spec.get("comparison", {}).get("pluggable_component", {}).get("name")
        )
    except (json.JSONDecodeError, OSError):
        pluggable = None
    return pluggable in top_level_defs if pluggable else True


def _stage_review_valid(stage_id: str):
    """Recovery-check factory for stage-reviewer drift: returns a
    closure that validates `.pipeline/stage_review_<stage_id>.json`
    exists, parses, and has a `findings` field."""
    def _check(state: PipelineState) -> bool:
        path = state.paths.pipeline_dir / f"stage_review_{stage_id}.json"
        if not path.is_file():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        return isinstance(data, dict) and "findings" in data
    return _check


def _review_report_valid(state: PipelineState) -> bool:
    """Recovery check for paper-fidelity-reviewer drift: review_report.json
    exists and parses with a `findings` field."""
    path = state.paths.pipeline_dir / "review_report.json"
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return isinstance(data, dict) and "findings" in data


def _read_review_json_or_err(
    path: Path, *, kind: str = "review", expected_stage_id: str | None = None,
) -> tuple[dict | None, str | None]:
    """Safely read a reviewer's JSON output. Returns (review_dict, None) on
    success or (None, err_str) on failure. Handles four failure modes:

      1. File missing (caller should typically have already checked, but
         we surface a clear error rather than crashing).
      2. Malformed JSON (caller should halt with a "reviewer wrote
         malformed JSON" message — distinct from "reviewer wrote nothing").
      3. Valid JSON but not a dict, or missing the `findings` key (driver
         can't proceed; treat as schema-invalid).
      4. (When `expected_stage_id` is set) Valid JSON with the wrong
         `stage_id` — the Think-class agent didn't copy the contract field
         verbatim (R2C 2026-05-20 halt-judge stage_1a regression class).

    `kind` is a human-readable label embedded in error messages
    (e.g., "stage_2b_architecture review", "paper-fidelity review").

    Without this helper, the bare `json.loads(...)` calls in stage 1, 2.b,
    2.c, 2.x, 3.a, 4, and 5 would propagate as uncaught exceptions if a
    Think-class reviewer wrote a schema-invalid file — no halt artifact,
    just a crash. Same root cause as the smoke-diagnostician 2026-05-19
    halt; same fix pattern (validate at read time, surface a clean error).
    """
    if not path.is_file():
        return None, f"{kind}: file {path.name} not found"
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        return None, f"{kind}: failed to read {path.name}: {e}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, (
            f"{kind}: {path.name} is not valid JSON (reviewer wrote "
            f"a structurally-invalid file): {e}"
        )
    if not isinstance(data, dict):
        return None, (
            f"{kind}: {path.name} top-level value is "
            f"{type(data).__name__}, not a dict"
        )
    if "findings" not in data:
        return None, (
            f"{kind}: {path.name} missing required `findings` key "
            f"(top-level keys: {sorted(data.keys())})"
        )
    if expected_stage_id is not None:
        err = _verify_agent_contract_fields(
            data, expected={"stage_id": expected_stage_id},
            agent="r2c-stage-reviewer", artifact=path.name,
        )
        if err is not None:
            return None, err
        # `expected_stage_id` marks the read as a stage-reviewer artifact
        # (the stage_5 review_report.json reads never pass it), so this is
        # where the StageReviewReport contract applies. The model is
        # calibrated on the full recorded artifact history and rejects
        # nothing that flows today; a failure here is the same
        # unusable-envelope class as a missing `findings` key and takes
        # the caller's existing degrade/retry route, never a new halt.
        try:
            StageReviewReport.model_validate(data)
        except Exception as e:  # pydantic.ValidationError
            return None, (
                f"{kind}: {path.name} does not validate against "
                f"schemas/stage_review_report.py: {e}"
            )
    return data, None


def _verify_agent_contract_fields(
    written: dict, *, expected: dict[str, object], agent: str, artifact: str,
) -> str | None:
    """Check that the just-written agent output contains every entry in
    `expected` verbatim. Returns None on match, or a halt-ready error
    message naming the specific mismatch on first violation.

    Use this for Think-class agents whose prompts pass contract fields
    (stage_id, iteration, reviewer_stage_id, ...) and whose outputs must
    echo those values back. The classic failure mode (R2C 2026-05-20):
    the halt-judge's prompt passed `stage_id=stage_1` but the agent
    wrote `stage_1a` into its decision file, breaking the driver's
    (stage_id, iteration) lookup with a cryptic "no decision matches"
    error. This helper turns that into a clean
        "agent r2c-halt-judge violated its output contract: wrote
         stage_id='stage_1a' but prompt sent stage_id='stage_1'"
    halt so the failure mode is obvious instead of cryptic.

    `agent` and `artifact` are baked into the error string so the halt
    reason is self-contained — no need for the caller to wrap it. List
    semantics: scalar values are compared by `==`; nested structures are
    out of scope (no contract field is currently nested)."""
    for field, want in expected.items():
        got = written.get(field, _MISSING)
        if got is _MISSING:
            return (
                f"agent {agent} violated its output contract in "
                f"{artifact}: missing required field {field!r} "
                f"(prompt sent {field}={want!r}); top-level keys: "
                f"{sorted(written.keys())}"
            )
        if got != want:
            return (
                f"agent {agent} violated its output contract in "
                f"{artifact}: wrote {field}={got!r} but prompt sent "
                f"{field}={want!r}. Think-class agents must copy "
                f"contract fields verbatim from the prompt — see "
                f"CONTRACT_FIELDS_VERBATIM_CLAUSE in "
                f"scripts/dispatch_templates.py"
            )
    return None


# Sentinel for "key not present" — distinguishes "missing field" from
# "field present with value None" in _verify_agent_contract_fields.
_MISSING = object()


def _stage_1_writeable_paths(agent: str, output_mode: str) -> list[str]:
    try:
        return STAGE_1_OUTPUT_MODE_WRITEABLE_PATHS[agent][output_mode]
    except KeyError as exc:
        raise ValueError(
            f"unknown Stage 1 output mode {output_mode!r} for {agent}"
        ) from exc


def _dispatch_decomposer(
    state: PipelineState, *, output_mode: str = "canonical",
) -> DispatchResult:
    """Dispatch the decomposer to produce paper_map.json. The decomposer's
    output is the analyzer's input — must complete before Stage 1.b."""
    agent = "r2c-decomposer"
    writeable_paths = _stage_1_writeable_paths(agent, output_mode)
    task_key = (
        "stage_1_decomposer_retry"
        if output_mode == "chunk"
        else "stage_1_decomposer"
    )
    task_summary = STAGE_TASK_SUMMARIES[task_key]
    if output_mode == "chunk":
        task_summary = task_summary.format(
            previous_output_error=(
                "The canonical single-file attempt did not produce a usable "
                "paper_map.json."
            ),
        )
    prompt = build_dispatch_prompt(
        task_summary=task_summary,
        paths=build_paths_block(state.paths),
        writeable_paths=writeable_paths,
        think_anchor_output_path=str(state.paths.paper_map),
        extra_sections=stage_1_schema_embeds(agent, output_mode) or None,
    )
    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt=prompt, timeout_s=DECOMPOSER_TIMEOUT_S,
        recovery_check_fn=_paper_map_valid if output_mode == "canonical" else None,
        writeable_paths_override=writeable_paths,
    )


def _pack_installed_classification_block(paths: PipelinePaths) -> str | None:
    """The resolved-classification section for analyzer dispatches after a
    provisional pack install, or None when no pack is installed.

    Stated from the manifest the driver wrote at install time, so the
    analyzer retry receives the classification as driver fact instead of
    re-deciding it (R2C-031: the fresh-session retry re-halted on the
    reserved-target rule whenever the pack's target disagreed with the
    route_elsewhere signal's named home)."""
    manifest = _read_provisional_pack_manifest(paths)
    if manifest is None:
        return None
    target = manifest.get("target_paradigm_id")
    pack_rel = manifest.get("pack_path")
    if not (isinstance(target, str) and target
            and isinstance(pack_rel, str) and pack_rel):
        return None
    return PACK_INSTALLED_CLASSIFICATION_BLOCK.format(
        target_paradigm_id=target, pack_path=pack_rel,
    )


def _dispatch_analyzer(
    state: PipelineState, *, output_mode: str = "canonical",
) -> DispatchResult:
    """Dispatch the analyzer to produce method_spec.json. Requires
    paper_map.json to already exist (Stage 1.a precondition)."""
    agent = "r2c-method-analyzer"
    writeable_paths = _stage_1_writeable_paths(agent, output_mode)
    task_key = (
        "stage_1_analyzer_retry"
        if output_mode == "chunk"
        else "stage_1_analyzer"
    )
    task_summary = STAGE_TASK_SUMMARIES[task_key]
    if output_mode == "chunk":
        task_summary = task_summary.replace(
            "{previous_output_error}",
            "The canonical single-file attempt did not produce a usable "
            "method_spec.json.",
        )
    pack_block = _pack_installed_classification_block(state.paths)
    # Gap-schema embed only on true gap-report dispatches: canonical mode
    # (chunk parts are untyped) with NO installed pack (a pack-installed
    # dispatch is forbidden from writing a gap report).
    extra_sections = (
        [pack_block] if pack_block
        else stage_1_schema_embeds(agent, output_mode) or None
    )
    prompt = build_dispatch_prompt(
        task_summary=task_summary,
        paths=build_paths_block(state.paths),
        writeable_paths=writeable_paths,
        think_anchor_output_path=str(state.paths.method_spec),
        extra_sections=extra_sections,
    )
    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt=prompt, timeout_s=ANALYZER_TIMEOUT_S,
        recovery_check_fn=_method_spec_valid if output_mode == "canonical" else None,
        writeable_paths_override=writeable_paths,
    )


def _dispatch_analyzer_fix(
    state: PipelineState, findings: list[dict]
) -> DispatchResult:
    agent = "r2c-method-analyzer"
    prompt = build_fix_mode_prompt(
        target_agent=agent,
        findings=findings,
        paths=build_paths_block(state.paths),
        writeable_paths=WRITEABLE_PATHS[agent],
        think_anchor_output_path=str(state.paths.method_spec),
    )
    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt=prompt, timeout_s=ANALYZER_TIMEOUT_S,
        recovery_check_fn=_method_spec_valid,
    )


def _dispatch_analyzer_feasibility_reask(
    state: PipelineState, blockers: list[dict]
) -> DispatchResult:
    """One targeted analyzer dispatch: commit to each surrogate it proposed
    (status → can_approximate with an approximation contract) or confirm
    the blocker is genuinely unimplementable (keep cannot_implement)."""
    agent = "r2c-method-analyzer"
    prompt = build_dispatch_prompt(
        task_summary=FEASIBILITY_SURROGATE_REASK_TEMPLATE.format(
            n=len(blockers),
            blockers_json=json.dumps(blockers, indent=2),
        ),
        paths=build_paths_block(state.paths),
        writeable_paths=WRITEABLE_PATHS[agent],
        think_anchor_output_path=str(state.paths.method_spec),
    )
    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt=prompt, timeout_s=ANALYZER_TIMEOUT_S,
        recovery_check_fn=_method_spec_valid,
    )


def _reask_feasibility_cannot_implement(
    state: PipelineState, stage_id: str, halt_record: dict | None,
) -> tuple[bool, dict | None]:
    """ONE analyzer re-ask before a cannot_implement feasibility halt whose
    blockers carry the analyzer's own proposed surrogate.

    The 2026-07-03 bev-distill roll halted on two blockers marked
    `cannot_implement` while their own resolution text proposed the
    synthetic-BEV demo every prior roll of the same paper used — analyzer
    roll variance turned an approvable approximation into a dead run. The
    re-ask makes the analyzer commit: surrogate-is-faithful (blocker
    becomes can_approximate with an explicit approximation contract; the
    run proceeds with a logged assumption) or genuinely-unimplementable
    (the halt stands, now as a deliberate decision).

    Mirrors the stage-2x resolution re-ask: one dispatch, best-effort —
    any dispatch, validation, or gate failure restores the original spec
    where needed and falls through to the exact old halt. Returns
    (ok, halt_record): (True, None) when the gate now passes, otherwise
    (False, <freshest halt record>) for the caller's halt."""
    blockers = (halt_record or {}).get("cannot_implement_blockers") or []
    candidates = [
        b for b in blockers if isinstance(b, dict)
        and str(b.get("resolution_proposed_by_analyzer") or "").strip()
    ]
    if not blockers or len(candidates) != len(blockers):
        # Any blocker WITHOUT a proposed surrogate keeps the halt: the
        # re-ask can only ever clear the whole gate, and dispatching for a
        # subset burns a turn to reach the same halt.
        return False, halt_record
    spec_path = state.paths.method_spec
    try:
        original_spec = spec_path.read_text(encoding="utf-8")
    except OSError as e:
        log(stage_id, "feasibility_reask_failed",
            f"cannot snapshot method_spec.json ({e}); halting as before")
        return False, halt_record
    log(stage_id, "feasibility_reask",
        f"{len(candidates)} cannot_implement blocker(s) carry an "
        f"analyzer-proposed surrogate; one analyzer re-ask before halting")
    try:
        _dispatch_analyzer_feasibility_reask(state, candidates)
    except (OpencodeClientError, OutOfScopeWritesError) as e:
        log(stage_id, "feasibility_reask_failed",
            f"re-ask dispatch failed ({e}); falling through to halt")
        _restore_text(spec_path, original_spec)
        return False, halt_record
    ok_valid, err_tail = _run_fresh_spec_validator(state)
    if not ok_valid:
        log(stage_id, "feasibility_reask_failed",
            "re-asked spec fails strict validation; restoring the original "
            f"spec and halting as before: {err_tail[:300]}")
        _restore_text(spec_path, original_spec)
        return False, halt_record
    ok_gate, new_halt = _run_feasibility_gate(state)
    if not ok_gate:
        # The analyzer confirmed at least one blocker as genuinely
        # unimplementable — the halt stands as a deliberate decision, with
        # the fresh record (which carries the strengthened reasons).
        log(stage_id, "feasibility_reask_confirmed_halt",
            "analyzer confirmed cannot_implement on re-ask; halting")
        return False, new_halt or halt_record
    aid = _next_assumption_id(state)
    names = "; ".join(
        str(b.get("requirement", "?"))[:120] for b in candidates)
    _append_assumption(
        state, aid=aid,
        title="Feasibility blockers proceeded as approved approximations",
        detected=(
            f"The feasibility gate halted on {len(candidates)} requirement(s) "
            f"marked cannot_implement whose own resolution text proposed a "
            f"workable surrogate: {names}"),
        action=(
            "The method analyzer was re-asked once and committed to its "
            "proposed surrogate(s): the blockers are now can_approximate "
            "with an explicit approximation contract in method_spec.json, "
            "and the run proceeded."),
        reasoning=(
            "cannot_implement paired with a concrete proposed surrogate is "
            "internally inconsistent — the analyzer either stands behind "
            "the surrogate (approved approximation) or the run halts "
            "deliberately. It committed to the surrogate."),
        alternative=(
            "Treat the original requirement as hard (obtain the real "
            "dataset/checkpoint) and re-run without the approximation."),
        override=(
            "Edit critical_requirements.blockers in "
            ".pipeline/method_spec.json back to cannot_implement and "
            "re-run, or supply the missing resource."),
    )
    log(stage_id, "feasibility_reask_applied",
        f"analyzer committed to its surrogate(s); gate now passes; "
        f"logged as {aid}")
    return True, None


def _restore_text(path: Path, content: str) -> None:
    """Best-effort restore of a snapshotted artifact after a failed re-ask."""
    try:
        path.write_text(content, encoding="utf-8")
    except OSError:
        pass


def _dispatch_decomposer_fix(
    state: PipelineState, findings: list[dict]
) -> DispatchResult:
    agent = "r2c-decomposer"
    prompt = build_fix_mode_prompt(
        target_agent=agent,
        findings=findings,
        paths=build_paths_block(state.paths),
        writeable_paths=WRITEABLE_PATHS[agent],
        think_anchor_output_path=str(state.paths.paper_map),
    )
    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt=prompt, timeout_s=DECOMPOSER_TIMEOUT_S,
        recovery_check_fn=_paper_map_valid,
    )


def _dispatch_decomposer_retry(
    state: PipelineState, *, previous_output_error: str | None = None,
) -> DispatchResult:
    """Decomposer missing-output retry. Used when the initial dispatch
    didn't write paper_map.json (Think-class stop-early). Stronger prompt
    that tells the agent to skip further reading and immediately Write."""
    agent = "r2c-decomposer"
    writeable_paths = _stage_1_writeable_paths(agent, "chunk")
    prompt = build_dispatch_prompt(
        task_summary=STAGE_TASK_SUMMARIES["stage_1_decomposer_retry"].format(
            previous_output_error=(
                previous_output_error
                or "No usable paper_map.json or paper_map_parts were found."
            ),
        ),
        paths=build_paths_block(state.paths),
        writeable_paths=writeable_paths,
        think_anchor_output_path=str(state.paths.paper_map),
    )
    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt=prompt, timeout_s=DECOMPOSER_TIMEOUT_S,
        recovery_check_fn=None,
        writeable_paths_override=writeable_paths,
    )


def _dispatch_analyzer_retry(
    state: PipelineState, *, previous_output_error: str | None = None,
) -> DispatchResult:
    """Analyzer missing-output retry. Used when the initial dispatch
    didn't write method_spec.json (Think-class stop-early)."""
    agent = "r2c-method-analyzer"
    writeable_paths = _stage_1_writeable_paths(agent, "chunk")
    pack_block = _pack_installed_classification_block(state.paths)
    prompt = build_dispatch_prompt(
        task_summary=STAGE_TASK_SUMMARIES["stage_1_analyzer_retry"].replace(
            "{previous_output_error}",
            previous_output_error
            or "No usable method_spec.json or method_spec_parts were found.",
        ),
        paths=build_paths_block(state.paths),
        writeable_paths=writeable_paths,
        think_anchor_output_path=str(state.paths.method_spec),
        extra_sections=[pack_block] if pack_block else None,
    )
    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt=prompt, timeout_s=ANALYZER_TIMEOUT_S,
        recovery_check_fn=None,
        writeable_paths_override=writeable_paths,
    )


def _stderr_to_finding(stderr_tail: str, validator_label: str) -> dict:
    """Synthesize a fix-mode finding from a validator's stderr output. The
    proposed_fix is intentionally generic — the stderr itself names the
    specific schema violation(s) the producer must address."""
    return {
        "id": "VAL001",
        "severity": "critical",
        "description": (
            f"{validator_label} rejected the file you just produced. "
            f"Validator output:\n{stderr_tail.strip()}"
        ),
        "proposed_fix": (
            "Re-read the schema for the file you produced, address every "
            "violation listed above (typically a missing required field, a "
            "wrong type, or an unknown field), and rewrite the file. Then "
            "report back briefly."
        ),
    }


def _no_write_fix_retry_finding(
    finding: dict,
    *,
    validator_label: str,
    stderr_tail: str,
) -> dict:
    """Make a one-shot stronger finding after a fix dispatch wrote nothing.

    The dispatch layer logs no-write turns, but the validator loop needs to
    distinguish "the producer tried and the error persisted" from "the producer
    returned without touching its artifact." The latter deserves one direct
    retry before the anti-fixation guard fires.
    """
    base = dict(finding or {})
    base["id"] = str(base.get("id") or "VAL001")
    base["severity"] = str(base.get("severity") or "critical")
    previous_description = str(base.get("description") or "").strip()
    previous_fix = str(base.get("proposed_fix") or "").strip()
    base["description"] = (
        "Your previous fix-mode dispatch completed without writing any "
        "allowed output file, so the validator re-ran against the unchanged "
        f"artifact and still failed. Validator: {validator_label}.\n\n"
        f"Current validator output:\n{stderr_tail.strip()}\n\n"
        f"Original finding:\n{previous_description}"
    )
    base["proposed_fix"] = (
        "Do not analyze further and do not restate a plan. Your previous "
        "turn most likely exhausted its per-step output budget composing "
        "the full implementation BEFORE its first write call (three "
        "consecutive turns died exactly this way on 2026-07-04; the softer "
        "version of this instruction was ignored). Your FIRST tool call "
        "this turn must be a write.\n"
        "- If your owned artifact does NOT exist on disk yet: your ONLY "
        "job this turn is a skeleton — ONE write call, immediately, "
        "producing a minimal syntactically-complete file (module "
        "docstring, imports, the exact contract signature(s), stub bodies "
        "that raise NotImplementedError). Do NOT implement the algorithm "
        "this turn. An incomplete skeleton is the EXPECTED outcome here; "
        "the normal fix loop fills it in helper-by-helper on later turns, "
        "each with its own budget.\n"
        "- If the artifact already exists: apply ONLY the validator-named "
        "fixes, as targeted edit calls, starting immediately.\n"
        + (f"\nOriginal proposed fix (context only — the write-first rule "
           f"above overrides its scope):\n{previous_fix}" if previous_fix
           else "")
    )
    # Sentinel consumed by build_fix_mode_prompt: hoists the act-first
    # instruction to the TOP of the prompt as a preamble. Position is the
    # proven differentiator for the output-cap truncation class — the
    # 2026-07-08 ACC/SRL retries carried the text above inside the findings
    # block and died at the cap anyway, while the judge's top-of-prompt
    # write-first preamble recovered in seconds in the same runs.
    base["_act_first_retry"] = True
    return base


def _validator_failing_count(stderr: str) -> int | None:
    """Number of failing items a structured validator reported, or None
    when the stderr has no countable shape.

    Prefers an explicit "(N element(s))" / "(N item(s))" / "(N error(s))"
    header; falls back to counting the "  - ..." bullet lines the stage-1/4
    validators emit one-per-failure. Used by the monotone-progress bonus
    (item 27) — the header can be truncated out of a stderr tail while the
    bullets survive, and vice versa."""
    m = re.search(r"\((\d+) (?:element|item|error)\(s\)\)", stderr or "")
    if m:
        return int(m.group(1))
    bullets = len(re.findall(r"^\s*-\s", stderr or "", re.MULTILINE))
    return bullets or None


def _stderr_excerpt(stderr: str, cap: int = 2000) -> str:
    """Head-plus-tail excerpt of validator output, elision marked.

    A plain tail slice silently drops the "(N element(s))" count header and
    the FIRST failing items when a structured error list outgrows the cap.
    iDb-RRT 2026-07-13 stage 1: the judge saw 5 of 7 findings (the two it
    never saw went unfixed for three iterations) and the monotone-bonus
    counts read 4 where the validator said 7. Keeping both ends preserves
    the header and the early items alongside the freshest output."""
    s = stderr or ""
    if len(s) <= cap:
        return s
    marker = "\n... [middle of validator output elided] ...\n"
    keep = cap - len(marker)
    head = s[: keep // 2]
    tail = s[-(keep - len(head)):]
    return head + marker + tail


def _validator_retry(
    state: PipelineState,
    *,
    stage_id: str,
    validator_fn,
    fix_dispatch_fn,
    validator_label: str,
    cap: int = 3,
    use_judge: bool = False,
    finding_enricher=None,
) -> StageResult | None:
    """Run `validator_fn`; on failure, dispatch `fix_dispatch_fn` with the
    stderr-as-finding and re-validate, up to `cap` retries. Returns None on
    success, a halted StageResult if the cap is exhausted.

    Progress-aware since 2026-06-10 (the GBALD stage-1 halt's lesson): the
    cap is 3 like the other fix loops, but an UNCHANGED error signature
    after a fix dispatch halts immediately — fixation burns iterations
    without progress (the original reason the cap was 1), while a CHANGED
    signature is cascading progress and earns the next bounded iteration.
    GBALD's halt was exactly the cascading case: the fix cleared both
    original schema errors and exposed one different error, then cap=1
    killed a run that one more 45-second dispatch would very likely have
    cleared. The signature is whitespace-normalized validator stderr —
    stage-1/4 validators emit deterministic structured error lists, so
    identical errors produce identical text.

    `use_judge` (stage 1, stage 4) — when True, the failure branch invokes
    the r2c-halt-judge to classify the failure and pick the fixer (with a
    structured finding) instead of mechanically dispatching `fix_dispatch_fn`
    with the stderr. The judge may decide to halt instead of dispatching.
    Same pattern as `run_fix_loop`'s use_judge.

    Monotone-progress bonus (item 27, SRL overnight 07-08): when every
    iteration STRICTLY reduced the validator's failing count (SRL's
    equation-quote trajectory was 14 → 8 → 3 at the cap), one bonus
    iteration is granted before the cap-exhaustion halt — the smoke gate's
    judge_cap_recovery mirrored onto this loop. Judge-gated: the bonus
    iteration routes through the judge like any other (use_judge=True at
    every stage-1 call site), so the judge can still refuse it by halting.
    A flat or growing count earns nothing, and the anti-fixation halt on an
    unchanged signature fires before this is ever consulted.

    `finding_enricher` (item 23 part 3) — optional callable applied to the
    outgoing fix finding right before each dispatch, on both routing paths
    (mechanical stderr finding and judge-authored finding). Best-effort:
    an enricher failure is logged and the original finding dispatches
    unchanged. The stage-1 paper-map call site uses it to attach candidate
    paper passages to quote-floor findings."""
    paths = state.paths
    prev_signature: str | None = None
    last_fix_wrote_output = True
    no_write_retry_used = False
    bonus_granted = False
    failing_counts: list[int | None] = []
    last_dispatcher = None
    last_fix_finding: dict | None = None
    for iteration in range(cap + 2):  # +1 for the potential monotone bonus
        ok, stderr_tail = validator_fn(state)
        if ok:
            return None
        if last_fix_wrote_output or not failing_counts:
            failing_counts.append(_validator_failing_count(stderr_tail))
        # else: the previous dispatch wrote nothing, so this validation
        # re-observes the same artifact. Recording it would put a flat
        # pair into the trajectory and deny the monotone bonus for an
        # infrastructure failure, not model regress (iDb-RRT 2026-07-13:
        # one cap-burned no-write iteration turned a strictly shrinking
        # 7 → 3 → 2 burn-down into 4, 4, 3, 2 and cost the bonus at the
        # cap). The anti-fixation branch below already treats no-write
        # iterations as "the error never had a chance to move"; the
        # trajectory bookkeeping now agrees with it.
        signature = " ".join(stderr_tail.split())
        if prev_signature is not None and signature == prev_signature:
            if (
                not last_fix_wrote_output
                and not no_write_retry_used
                and last_dispatcher is not None
                and last_fix_finding is not None
            ):
                no_write_retry_used = True
                retry_finding = _no_write_fix_retry_finding(
                    last_fix_finding,
                    validator_label=validator_label,
                    stderr_tail=stderr_tail,
                )
                log(stage_id, "fix_no_write_retry",
                    f"iteration {iteration}: prior fix-mode dispatch wrote no "
                    f"allowed artifact; retrying {validator_label} fixer once "
                    "before anti-fixation halt")
                fix_snapshot = _snapshot_run_dir(paths.run_dir)
                try:
                    last_dispatcher(state, [retry_finding])
                except (OpencodeClientError, OutOfScopeWritesError) as e:
                    return halt(
                        paths, stage_id,
                        reason=f"fix-mode no-write retry failed during "
                               f"{validator_label}: {e}",
                        halt_class=_dispatch_error_halt_class(e),
                        retry_count=iteration,
                        state=state,
                    )
                changed = _changed_paths_since_snapshot(paths.run_dir, fix_snapshot)
                last_fix_wrote_output = bool(changed)
                last_fix_finding = retry_finding
                continue
            # Truthful halt reason: an unchanged signature means FIXATION
            # only when a fix was actually written. When the fix dispatch
            # and its act-first retry both wrote nothing (the output-cap
            # truncation class, overnight 2026-07-08), the error never had
            # a chance to move and a plain resume is the first lever — a
            # researcher told "fixation" would wrongly suspect a hard
            # target.
            if not last_fix_wrote_output:
                reason = (
                    f"{validator_label} still fails and the fix dispatch "
                    f"plus its act-first retry both completed without "
                    f"writing anything (output-cap truncation class, not "
                    f"fixation — the error was never acted on; iteration "
                    f"{iteration}). A plain resume re-rolls the fix "
                    f"dispatch and is the first lever."
                )
            else:
                reason = (
                    f"{validator_label} failed with an UNCHANGED error "
                    f"signature after a fix dispatch (fixation — the fix "
                    f"did not move the error; iteration {iteration})"
                )
            return halt(
                paths, stage_id,
                reason=reason,
                halt_class="fix_loop_exhausted",
                retry_count=iteration,
                context={"stderr": stderr_tail},
                state=state,
            )
        prev_signature = signature
        if iteration >= cap:
            trajectory = " → ".join(
                "?" if c is None else str(c) for c in failing_counts)
            counted = (
                len(failing_counts) >= 2
                and all(c is not None for c in failing_counts)
            )
            strictly_decreasing = counted and all(
                failing_counts[i + 1] < failing_counts[i]
                for i in range(len(failing_counts) - 1)
            )
            # Second qualifying shape (pdfgnn 2026-08-11 attempts 5-7): a
            # non-increasing trajectory that burned down to exactly ONE
            # remaining error (3 → 3 → 2 → 1, 4 → 3 → 1 → 1, 3 → 2 → 2 → 1)
            # is one plausible dispatch from converging, and a fresh re-roll
            # costs far more than that dispatch. A plateau at a higher count
            # (14 → 8 → 8) still earns nothing, and the anti-fixation halt
            # on an unchanged signature fires before this is consulted.
            plateaued_to_one = (
                counted
                and failing_counts[0] > 1
                and failing_counts[-1] == 1
                and all(
                    failing_counts[i + 1] <= failing_counts[i]
                    for i in range(len(failing_counts) - 1)
                )
            )
            monotone = strictly_decreasing or plateaued_to_one
            if use_judge and monotone and not bonus_granted:
                bonus_granted = True
                shape_note = (
                    "every iteration strictly reduced the failing count"
                    if strictly_decreasing
                    else "the failing count never grew and burned down to one"
                )
                log(stage_id, "validator_cap_bonus_granted",
                    f"iteration {iteration}: {shape_note} ({trajectory}); "
                    f"granting one judge-routed bonus iteration before the "
                    f"cap halt (item 27)")
                # Fall through to the judge-routed fix branch below — the
                # judge can still refuse the bonus by deciding halt.
            else:
                bonus_note = (
                    " plus one bonus iteration granted for monotone "
                    f"progress (failing-count trajectory: {trajectory})"
                    if bonus_granted else ""
                )
                # Wording is trajectory-driven, not categorical: the old
                # "error signature changed each iteration" claim shipped
                # verbatim into researcher-facing reports on a run where
                # the signature was byte-identical across a no-write
                # iteration (iDb-RRT 2026-07-13).
                return halt(
                    paths, stage_id,
                    reason=f"{validator_label} failed after {cap} fix-mode "
                           f"retries{bonus_note} (failing-count trajectory: "
                           f"{trajectory} — each landed fix moved the "
                           f"errors, but they did not converge within the "
                           f"cap)",
                    halt_class="fix_loop_exhausted",
                    retry_count=iteration,
                    context={"stderr": stderr_tail},
                    state=state,
                )
        # Judge-routed when use_judge=True; mechanical routing otherwise.
        if use_judge:
            try:
                outcome = _invoke_judge(
                    state, stage_id=stage_id,
                    validator_label=validator_label,
                    stderr_tail=stderr_tail, iteration=iteration,
                )
            except (OpencodeClientError, OutOfScopeWritesError, ValueError) as e:
                return halt(
                    paths, stage_id,
                    reason=f"halt-judge invocation failed at {validator_label}: {e}",
                    halt_class=_judge_invocation_halt_class(e),
                    retry_count=iteration,
                    context={"stderr": stderr_tail}, state=state,
                )
            if outcome.action == "halt":
                log(stage_id, "judge_halt",
                    f"iteration {iteration}: judge classified as "
                    f"{outcome.classification!r} ({outcome.confidence}); halting")
                # No user_message: the catalog's judge_halt story leads and
                # the rationale stays in the technical collapsible, labeled
                # as an internal note (item 12 rule 3 — the detr fixture's
                # rationale asserted a wrong mechanism as the headline).
                return halt(
                    paths, stage_id,
                    reason=f"halt-judge decided to halt: {outcome.rationale}",
                    halt_class="judge_halt",
                    retry_count=iteration,
                    context={
                        "stderr": stderr_tail,
                        "judge_classification": outcome.classification,
                        "judge_confidence": outcome.confidence,
                        "validator_label": validator_label,
                    },
                    state=state,
                )
            dispatcher = outcome.dispatcher
            fix_finding = outcome.finding
            log(stage_id, "judge_dispatch",
                f"iteration {iteration}: judge routed to "
                f"{outcome.target_agent} ({outcome.classification}, "
                f"{outcome.confidence})")
        else:
            dispatcher = fix_dispatch_fn
            fix_finding = _stderr_to_finding(stderr_tail, validator_label)
            log(stage_id, "validator_retry",
                f"iteration {iteration}: {validator_label} failed; "
                f"re-dispatching producer in fix-mode")
        if finding_enricher is not None:
            try:
                fix_finding = finding_enricher(fix_finding)
            except Exception as e:  # noqa: BLE001 — enrichment is advisory
                log(stage_id, "finding_enricher_failed",
                    f"iteration {iteration}: finding enricher raised "
                    f"({e}); dispatching the unenriched finding")
        fix_snapshot = _snapshot_run_dir(paths.run_dir)
        try:
            dispatcher(state, [fix_finding])
        except (OpencodeClientError, OutOfScopeWritesError) as e:
            return halt(
                paths, stage_id,
                reason=f"fix-mode dispatch failed during {validator_label} retry: {e}",
                halt_class=_dispatch_error_halt_class(e),
                retry_count=iteration,
                state=state,
            )
        changed = _changed_paths_since_snapshot(paths.run_dir, fix_snapshot)
        last_fix_wrote_output = bool(changed)
        last_dispatcher = dispatcher
        last_fix_finding = fix_finding
    return None  # unreachable; loop either returns success or halts at cap


def _dispatch_stage_1_reviewer(state: PipelineState) -> DispatchResult:
    agent = "r2c-stage-reviewer"
    review_path = state.paths.pipeline_dir / "stage_review_stage_1_analyzer.json"
    prompt = build_dispatch_prompt(
        task_summary=STAGE_REVIEW_TASK_TEMPLATE.format(stage_id="stage_1_analyzer"),
        paths=build_paths_block(state.paths),
        closing=REVIEWER_CLOSING,
        writeable_paths=WRITEABLE_PATHS[agent],
        think_anchor_output_path=str(review_path),
    )
    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt=prompt, timeout_s=REVIEWER_TIMEOUT_S,
        recovery_check_fn=_stage_review_valid("stage_1_analyzer"),
        allow_corrective_redispatch=True,
    )


def _dispatch_stage_reviewer_retry(state: PipelineState, stage_id: str) -> DispatchResult:
    """Missing-output retry for stage-reviewer when the prior dispatch returned
    without producing `<run_dir>/.pipeline/stage_review_<stage_id>.json`. Same
    Think-class stop-early recovery pattern as the decomposer + analyzer:
    one stronger-prompt retry that says 'you didn't Write; do it NOW' before
    halting. Generic across stage_ids so all callers (stage_1, stage_2x, and
    the run_fix_loop helper for stage_2b/2c/3a) share the same pattern."""
    agent = "r2c-stage-reviewer"
    review_path = state.paths.pipeline_dir / f"stage_review_{stage_id}.json"
    prompt = build_dispatch_prompt(
        task_summary=STAGE_REVIEWER_RETRY_TASK_TEMPLATE.format(stage_id=stage_id),
        paths=build_paths_block(state.paths),
        closing=REVIEWER_CLOSING,
        writeable_paths=WRITEABLE_PATHS[agent],
        think_anchor_output_path=str(review_path),
    )
    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt=prompt, timeout_s=REVIEWER_TIMEOUT_S,
        recovery_check_fn=_stage_review_valid(stage_id),
        allow_corrective_redispatch=True,
    )


_QUOTE_CANDIDATE_PASSAGE_CAP = 1500   # chars per element's passage
_QUOTE_CANDIDATES_BLOCK_CAP = 12000   # chars for the whole findings block


def _equation_quote_candidates_block(state: PipelineState) -> str:
    """Candidate-passages block for the CURRENTLY failing equation quotes,
    or "" when there is nothing useful to attach (item 23 part 3).

    Re-runs the equation quote floor in process against the live paper
    map, locates each failing quote's best paper passage, and renders the
    block the fix finding carries so the retry can copy the paper's own
    bytes instead of re-reading the paper (the output-cap burn shape).
    Candidates are advisory by wording: a wrong-but-real passage would
    pass the floor while being semantically wrong, so adoption stays the
    agent's call. Caps are named, never silent: elements that do not fit
    the block cap are listed as omitted."""
    from passage_locator import locate_passages  # noqa: PLC0415
    from validate_paper_map import check_equation_quotes  # noqa: PLC0415

    paper_md = state.paths.pipeline_dir / "paper.md"
    if not paper_md.is_file() or not state.paths.paper_map.is_file():
        return ""
    try:
        paper_map = json.loads(
            state.paths.paper_map.read_text(encoding="utf-8"))
        paper_text = paper_md.read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError):
        return ""
    failing_errors = check_equation_quotes(paper_map, paper_text)
    if not failing_errors:
        return ""
    failing_ids = set()
    for line in failing_errors:
        m = re.match(r"\s*-\s*(\S+):", line)
        if m:
            failing_ids.add(m.group(1))

    # Fair-share passage budget (SRL 2026-07-21 fix_loop_exhausted root
    # cause): with 15 failing equations at the flat 1500-char per-passage
    # cap, only ~7 candidates fit the block cap and the rest were dropped
    # as "block size cap" — the starved equations never saw the paper's
    # bytes and failed every retry. Split the block budget across ALL
    # currently-failing elements instead, so every element gets a
    # candidate (the equation-mode locator slices to the math block when
    # its share is tight). The floor keeps a share from degenerating
    # below a usable equation snippet.
    failing_elements = [e for e in paper_map.get("elements", [])
                        if e.get("id") in failing_ids]
    per_element_cap = min(
        _QUOTE_CANDIDATE_PASSAGE_CAP,
        max(400, _QUOTE_CANDIDATES_BLOCK_CAP
            // max(1, len(failing_elements))),
    )

    entries: list[str] = []
    attached: list[tuple[str, float]] = []
    omitted: list[str] = []
    total = 0
    for element in failing_elements:
        el_id = element.get("id")
        candidates = locate_passages(
            str(element.get("source_text") or ""), paper_text,
            element_type="equation",
            max_passage_chars=per_element_cap)
        if not candidates:
            omitted.append(f"{el_id} (no confident match)")
            continue
        cand = candidates[0]
        entry = (f"- `{el_id}` (match score {cand['score']}):\n"
                 f"```\n{cand['passage']}\n```")
        if total + len(entry) > _QUOTE_CANDIDATES_BLOCK_CAP:
            omitted.append(f"{el_id} (block size cap)")
            continue
        entries.append(entry)
        attached.append((str(el_id), float(cand["score"])))
        total += len(entry)
    if not entries and not omitted:
        return ""

    parts = []
    if entries:
        parts.append(
            "**Candidate paper passages (located for you — no paper "
            "re-read needed for these):**\n"
            "Each passage below is copied VERBATIM from paper.md and is "
            "the best match for that element's rejected quote. Use a "
            "passage ONLY if it is the one you meant to quote; if it is "
            "not, find the right passage in the paper instead. Copy "
            "byte-for-byte, and remember every backslash must be doubled "
            "in the JSON file (\\\\theta, not \\theta).\n\n"
            + "\n".join(entries))
    if omitted:
        parts.append(
            "No candidate is included for: " + ", ".join(omitted)
            + " — locate those in the paper yourself.")
    block = "\n\n".join(parts)
    log("stage_1", "quote_candidates_attached",
        f"attached {len(attached)} candidate passage(s) to the fix "
        f"finding ({total} chars): "
        + ", ".join(f"{i}={s}" for i, s in attached)
        + (f"; omitted: {', '.join(omitted)}" if omitted else ""))
    # Also a first-class run event: until 2026-07-14 this only reached the
    # driver log, so the event stream showed the passage locator (item 23)
    # as never firing on runs where it demonstrably converged the fix loop
    # (ACC and ms3d, 2026-07-13 audit).
    _append_run_event(
        state.paths,
        "quote_candidates_attached",
        summary=(f"stage 1 attached {len(attached)} candidate paper "
                 f"passage(s) to the quote-floor fix finding"),
        details={
            "attached": [{"element_id": i, "score": s} for i, s in attached],
            "omitted": list(omitted),
            "block_chars": total,
        },
    )
    return block


def _paper_map_quote_enricher(state: PipelineState):
    """finding_enricher for the stage-1 paper-map validator retry loop:
    appends the candidate-passages block to the outgoing fix finding.
    One seam covers both routing paths (mechanical stderr finding and
    judge-authored finding) plus the no-write act-first retry, which
    embeds the enriched description."""
    def _enrich(finding: dict) -> dict:
        block = _equation_quote_candidates_block(state)
        if not block:
            return finding
        enriched = dict(finding)
        enriched["description"] = (
            str(enriched.get("description") or "").rstrip()
            + "\n\n" + block)
        return enriched
    return _enrich


def _reanchor_failing_equation_quotes(state: PipelineState) -> int:
    """Deterministic render-equivalence re-anchoring for equation quotes
    that fail the verbatim floor (queue item 11, 2026-07-21).

    Root cause this closes (SRL fix_loop_exhausted): producers emit
    quotes that are render-EQUIVALENT to the paper but byte-different
    (combining vs spacing diacritics, math-delimiter substitution,
    markup loss, mid-token whitespace), and demonstrably cannot
    transcribe the difference even with the exact bytes in the fix
    prompt. For each failing equation whose quote folds to EXACTLY ONE
    span of paper.md, replace source_text with the paper's own bytes —
    the quote was a locator, the paper is the authority, meaning is
    never edited, and ambiguity stays on the honest fix-loop path. The
    validator re-runs afterwards as the terminal gate. Returns the
    number of re-anchored elements."""
    from quote_reanchor import reanchor_quote  # noqa: PLC0415
    from validate_paper_map import check_equation_quotes  # noqa: PLC0415

    paper_md = state.paths.pipeline_dir / "paper.md"
    if not paper_md.is_file() or not state.paths.paper_map.is_file():
        return 0
    try:
        paper_map = json.loads(
            state.paths.paper_map.read_text(encoding="utf-8"))
        paper_text = paper_md.read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError):
        return 0
    failing_errors = check_equation_quotes(paper_map, paper_text)
    if not failing_errors:
        return 0
    failing_ids = set()
    for line in failing_errors:
        m = re.match(r"\s*-\s*(\S+):", line)
        if m:
            failing_ids.add(m.group(1))

    reanchored: list[str] = []
    for element in paper_map.get("elements", []):
        el_id = element.get("id")
        if el_id not in failing_ids:
            continue
        anchored = reanchor_quote(
            str(element.get("source_text") or ""), paper_text)
        if anchored is not None:
            element["source_text"] = anchored
            reanchored.append(str(el_id))
    if not reanchored:
        return 0
    state.paths.paper_map.write_text(
        json.dumps(paper_map, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    log("stage_1", "equation_quotes_reanchored",
        f"re-anchored {len(reanchored)} render-equivalent equation "
        f"quote(s) to the paper's own bytes: {', '.join(reanchored)}")
    _append_run_event(
        state.paths,
        "equation_quotes_reanchored",
        summary=(f"stage 1 re-anchored {len(reanchored)} render-equivalent "
                 f"equation quote(s) to paper.md's own bytes"),
        details={"element_ids": reanchored},
    )
    return len(reanchored)


def _adopt_failing_equation_quotes(state: PipelineState) -> int:
    """Accept-the-candidate adoption for equation quotes still failing
    the verbatim floor after render-equivalent re-anchoring (R2C-030,
    design approved 2026-07-28).

    The fix loop has no deterministic path for a defect the producer
    cannot perceive: both recorded cases (SRL 2026-07-27,
    bayesian-active-learning 2026-07-28) returned the identical error
    when handed the exact bytes at candidate similarity 1.0. When the
    paper holds EXACTLY ONE near-perfect candidate inside the element's
    own anchored region and nothing close anywhere else, adopt the
    paper's own bytes under a logged assumption instead of dispatching
    a retry the producer cannot perform — the quote is a locator, the
    paper is the authority, and adopting the paper's own bytes is
    strictly more faithful than the producer's transcription. Ambiguity
    of any kind stays on the honest fix-loop path, and the validator
    re-runs afterwards as the terminal gate. Every adoption is loud: an
    assumptions.md entry, a run event, and a report-facing sidecar at
    .pipeline/quote_adoptions.json preserving the original quote.
    Returns the number of adopted elements."""
    from quote_adopt import find_adoption, resolve_section_regions  # noqa: PLC0415
    from validate_paper_map import check_equation_quotes  # noqa: PLC0415

    paper_md = state.paths.pipeline_dir / "paper.md"
    if not paper_md.is_file() or not state.paths.paper_map.is_file():
        return 0
    try:
        paper_map = json.loads(
            state.paths.paper_map.read_text(encoding="utf-8"))
        paper_text = paper_md.read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError):
        return 0
    failing_errors = check_equation_quotes(paper_map, paper_text)
    if not failing_errors:
        return 0
    failing_ids = set()
    for line in failing_errors:
        m = re.match(r"\s*-\s*(\S+):", line)
        if m:
            failing_ids.add(m.group(1))

    adopted: list[dict] = []
    for element in paper_map.get("elements", []):
        el_id = str(element.get("id"))
        if el_id not in failing_ids:
            continue
        original = str(element.get("source_text") or "")
        regions = resolve_section_regions(
            str(element.get("section") or ""), paper_text)
        adoption = find_adoption(original, paper_text,
                                 region=regions or None)
        if adoption is None:
            continue
        element["source_text"] = adoption["passage"]
        adopted.append({
            "element_id": el_id,
            "original_quote": original,
            "adopted_passage": adoption["passage"],
            "similarity": round(adoption["score"], 4),
            "runner_up": (round(adoption["runner_up"], 4)
                          if adoption["runner_up"] is not None else None),
            "region_resolved": adoption["region_resolved"],
        })
    if not adopted:
        return 0
    state.paths.paper_map.write_text(
        json.dumps(paper_map, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    for record in adopted:
        aid = _next_assumption_id(state)
        record["assumption_id"] = aid
        uniqueness = ("the best other passage anywhere in the paper "
                      f"scored {record['runner_up']}"
                      if record["runner_up"] is not None
                      else "no other passage came close anywhere in the paper")
        scope = ("inside the element's own anchored section"
                 if record["region_resolved"]
                 else "with uniqueness required over the whole paper "
                      "(the element's section label did not resolve to a "
                      "heading)")
        _append_assumption(
            state, aid=aid,
            title=f"Adopted the paper's own bytes for equation "
                  f"`{record['element_id']}`",
            detected=(f"The quote for `{record['element_id']}` failed the "
                      f"verbatim floor, and the paper holds exactly one "
                      f"near-perfect match (similarity "
                      f"{record['similarity']} on render-folded text, "
                      f"{scope}; {uniqueness})."),
            action=("source_text was replaced with the paper's verbatim "
                    "passage BEFORE any fix-loop retry. The producer's "
                    "original quote is preserved in "
                    "`.pipeline/quote_adoptions.json` for audit."),
            reasoning=("Producer retries demonstrably return identical "
                       "bytes for defects they cannot perceive (both "
                       "recorded cases, 2026-07-27/28). The quote is a "
                       "locator and the paper's bytes are the authority — "
                       "the same principle as render-equivalent "
                       "re-anchoring, extended to a unique near-match. "
                       "Adopting the paper's own bytes is strictly more "
                       "faithful than the producer's transcription."),
            alternative=("Keep the halt and have a human transcribe the "
                         "passage."),
            override=("Restore the original quote from "
                      "`.pipeline/quote_adoptions.json` into the paper "
                      "map and re-run stage 1."),
        )
    sidecar = state.paths.pipeline_dir / "quote_adoptions.json"
    existing: list[dict] = []
    if sidecar.is_file():
        try:
            loaded = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = loaded
        except (OSError, json.JSONDecodeError):
            existing = []
    sidecar.write_text(
        json.dumps(existing + adopted, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    ids = [r["element_id"] for r in adopted]
    log("stage_1", "equation_quotes_adopted",
        f"adopted the paper's own bytes for {len(adopted)} near-match "
        f"equation quote(s) under logged assumptions: {', '.join(ids)}")
    _append_run_event(
        state.paths,
        "equation_quotes_adopted",
        summary=(f"stage 1 adopted the paper's own bytes for {len(adopted)} "
                 f"equation quote(s) whose unique near-perfect candidate "
                 f"cleared the adoption rule"),
        details={"adoptions": [
            {"element_id": r["element_id"],
             "similarity": r["similarity"],
             "runner_up": r["runner_up"],
             "region_resolved": r["region_resolved"],
             "assumption_id": r["assumption_id"]} for r in adopted]},
    )
    return len(adopted)


def _run_paper_map_validator(state: PipelineState) -> tuple[bool, str]:
    _assemble_paper_map_if_fresh(state)
    proc = run_script(
        "stage_1",
        ["scripts/validate_paper_map.py", str(state.paths.paper_map)],
        timeout=60,
    )
    ok = proc.returncode == 0
    if not ok:
        repaired = _reanchor_failing_equation_quotes(state)
        # R2C-030: near-match adoption for what the render-equivalence
        # fold cannot absorb, BEFORE any fix-loop retry — the recorded
        # cases prove the retry channel cannot repair a defect the
        # producer cannot perceive.
        adopted = _adopt_failing_equation_quotes(state)
        if repaired or adopted:
            # Deterministic repairs landed; the re-run below is the
            # terminal gate (remaining failures continue to the enriched
            # fix loop).
            proc = run_script(
                "stage_1",
                ["scripts/validate_paper_map.py", str(state.paths.paper_map)],
                timeout=60,
            )
            ok = proc.returncode == 0
    err_tail = _stderr_excerpt(proc.stderr)
    _append_validation_event(
        state.paths,
        stage_id="stage_1",
        validator="validate_paper_map.py",
        ok=ok,
        stderr_tail=err_tail,
        artifacts=[str(state.paths.paper_map)],
    )
    return ok, err_tail


# ---------------------------------------------------------------------------
# Method-spec quote-floor repair: the same deterministic chain the paper
# map's equation floor has (render-equivalent re-anchor, then R2C-030
# near-match adoption), extended to the spec's verbatim floors —
# critical_requirements.param_glossary meaning quotes,
# scenario_assumptions evidence quotes, and comparison.evaluation_protocol
# evidence quotes. Evidence (0728b batch,
# bayesian-active-learning fresh roll): stage 1 exhausted all 3
# fix-loop retries on three meaning quotes whose only defect was spaces
# inserted inside math delimiters (`$ R_0 $` vs the paper's `$R_0$`) —
# render-equivalent, deterministic to repair, and the producer
# demonstrably could not byte-fix them across retries. The equation
# surface got this chain on 2026-07-21/28; the spec floors never did.
# ---------------------------------------------------------------------------

# Near-match ADOPTION on the spec's quote floors extends R2C-030's
# approved scope (equation quotes in the paper map) to a new surface.
# The rule, margins, and loud surfaces (assumptions.md entry, run
# event, quote_adoptions.json sidecar row) are identical; only the
# artifact differs. Flip to False to keep this surface re-anchor-only
# until the maintainer approves the scope extension.
SPEC_QUOTE_ADOPTION_ENABLED = True


def _spec_quote_floor_failures(spec: dict, paper_text: str) -> list[dict]:
    """The method-spec verbatim quote floors, re-checked in process.

    Replicates validate_method_spec.py's --paper-md floors faithfully
    (they live inline in its main() and are not importable): a quote
    passes when its whitespace-normalized form is a substring of
    whitespace-normalized paper.md. The subprocess validator re-run
    stays the terminal gate — this in-process check only selects which
    quotes the deterministic repair chain may touch.

    Returns one record per failing quote:
    {"key": the validator-style label, "container": the dict holding
    the quote (mutating it mutates the spec), "field": the quote field
    name, "location": the entry's own paper-location label}.

    scenario_assumptions is dict-shaped per the schema
    (dimension_id -> entry); a list shape is handled defensively so a
    malformed producer artifact degrades to a no-repair rather than a
    crash inside the repair path."""
    def _nws(t: object) -> str:
        return " ".join((str(t) if t is not None else "").split())

    haystack = _nws(paper_text)
    failures: list[dict] = []

    crit = spec.get("critical_requirements")
    glossary = crit.get("param_glossary") if isinstance(crit, dict) else None
    if isinstance(glossary, list):
        for entry in glossary:
            if not isinstance(entry, dict):
                continue
            quote = _nws(entry.get("meaning_quote"))
            if quote and quote not in haystack:
                failures.append({
                    "key": (f"param_glossary[{str(entry.get('name'))!r}]"
                            ".meaning_quote"),
                    "container": entry,
                    "field": "meaning_quote",
                    "location": str(entry.get("paper_section") or ""),
                })

    scenario = spec.get("scenario_assumptions")
    if isinstance(scenario, dict):
        items = list(scenario.items())
    elif isinstance(scenario, list):
        items = [
            (str(entry.get("dimension_id", i))
             if isinstance(entry, dict) else str(i), entry)
            for i, entry in enumerate(scenario)
        ]
    else:
        items = []
    for dim_id, entry in items:
        if not isinstance(entry, dict):
            continue
        quote = _nws(entry.get("evidence_quote"))
        if quote and quote not in haystack:
            failures.append({
                "key": f"scenario_assumptions[{dim_id!r}].evidence_quote",
                "container": entry,
                "field": "evidence_quote",
                "location": str(entry.get("paper_location") or ""),
            })

    comparison = spec.get("comparison")
    protocol = (
        comparison.get("evaluation_protocol")
        if isinstance(comparison, dict)
        else None
    )
    if isinstance(protocol, dict):
        records: list[tuple[str, object, str, str]] = [
            (
                "scheme",
                protocol.get("scheme"),
                "evidence_quote",
                "paper_section",
            ),
        ]
        quantities = protocol.get("quantities")
        if isinstance(quantities, list):
            records.extend(
                (
                    f"quantities[{index}]",
                    entry,
                    "evidence_quote",
                    "paper_section",
                )
                for index, entry in enumerate(quantities)
            )
            records.extend(
                (
                    f"quantities[{index}]",
                    entry,
                    "axis_evidence_quote",
                    "axis_paper_section",
                )
                for index, entry in enumerate(quantities)
            )
        for label, entry, quote_field, section_field in records:
            if not isinstance(entry, dict):
                continue
            quote = _nws(entry.get(quote_field))
            if quote and quote not in haystack:
                failures.append({
                    "key": (
                        f"comparison.evaluation_protocol.{label}"
                        f".{quote_field}"
                    ),
                    "container": entry,
                    "field": quote_field,
                    "location": str(entry.get(section_field) or ""),
                })
    return failures


def _repair_failing_spec_quotes(state: PipelineState) -> tuple[int, int]:
    """Deterministic verbatim-quote repair for the method-spec floors,
    mirroring the paper map's chain: render-equivalent re-anchor first
    (the quote is a locator, the paper's bytes are the authority), then
    R2C-030 near-match adoption for what the fold cannot absorb —
    BEFORE any fix-loop retry, because the recorded cases prove the
    retry channel cannot repair a defect the producer cannot perceive
    (bayesian-active-learning 0728b: three meaning quotes with spaces
    inside math delimiters burned all 3 retries and halted the run).

    Re-anchors are disclosure-light (a run event, like the equation
    surface); every adoption is loud: an assumptions.md entry
    preserving the original quote, a run event, and a row in the
    .pipeline/quote_adoptions.json sidecar with surface="method_spec".
    Ambiguity of any kind keeps the honest fix-loop path, and the
    caller re-runs the subprocess validator as the terminal gate.
    Returns (reanchored_count, adopted_count)."""
    from quote_adopt import find_adoption, resolve_section_regions  # noqa: PLC0415
    from quote_reanchor import reanchor_quote  # noqa: PLC0415

    paper_md = state.paths.pipeline_dir / "paper.md"
    if not paper_md.is_file() or not state.paths.method_spec.is_file():
        return 0, 0
    try:
        spec = json.loads(
            state.paths.method_spec.read_text(encoding="utf-8"))
        paper_text = paper_md.read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError):
        return 0, 0
    if not isinstance(spec, dict):
        return 0, 0
    failures = _spec_quote_floor_failures(spec, paper_text)
    if not failures:
        return 0, 0

    reanchored: list[str] = []
    adopted: list[dict] = []
    for failure in failures:
        container = failure["container"]
        field = failure["field"]
        original = str(container.get(field) or "")
        anchored = reanchor_quote(original, paper_text)
        if anchored is not None:
            container[field] = anchored
            reanchored.append(failure["key"])
            continue
        if not SPEC_QUOTE_ADOPTION_ENABLED:
            continue
        regions = resolve_section_regions(failure["location"], paper_text)
        adoption = find_adoption(original, paper_text,
                                 region=regions or None)
        if adoption is None:
            continue
        container[field] = adoption["passage"]
        adopted.append({
            "element_id": failure["key"],
            "surface": "method_spec",
            "original_quote": original,
            "adopted_passage": adoption["passage"],
            "similarity": round(adoption["score"], 4),
            "runner_up": (round(adoption["runner_up"], 4)
                          if adoption["runner_up"] is not None else None),
            "region_resolved": adoption["region_resolved"],
        })
    if not reanchored and not adopted:
        return 0, 0
    state.paths.method_spec.write_text(
        json.dumps(spec, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    if reanchored:
        log("stage_1", "spec_quotes_reanchored",
            f"re-anchored {len(reanchored)} render-equivalent method-spec "
            f"quote(s) to the paper's own bytes: {', '.join(reanchored)}")
        _append_run_event(
            state.paths,
            "spec_quotes_reanchored",
            summary=(f"stage 1 re-anchored {len(reanchored)} "
                     f"render-equivalent method-spec quote(s) to "
                     f"paper.md's own bytes"),
            details={"quote_keys": reanchored},
        )

    for record in adopted:
        aid = _next_assumption_id(state)
        record["assumption_id"] = aid
        uniqueness = ("the best other passage anywhere in the paper "
                      f"scored {record['runner_up']}"
                      if record["runner_up"] is not None
                      else "no other passage came close anywhere in the paper")
        scope = ("inside the entry's own paper location"
                 if record["region_resolved"]
                 else "with uniqueness required over the whole paper "
                      "(the entry's location label did not resolve to a "
                      "heading)")
        _append_assumption(
            state, aid=aid,
            title=f"Adopted the paper's own bytes for spec quote "
                  f"`{record['element_id']}`",
            detected=(f"The quote at `{record['element_id']}` failed the "
                      f"method-spec verbatim floor, and the paper holds "
                      f"exactly one near-perfect match (similarity "
                      f"{record['similarity']} on render-folded text, "
                      f"{scope}; {uniqueness})."),
            action=("The quote was replaced with the paper's verbatim "
                    "passage BEFORE any fix-loop retry. The producer's "
                    "original quote is preserved in "
                    "`.pipeline/quote_adoptions.json` for audit."),
            reasoning=("Producer retries demonstrably return identical "
                       "bytes for defects they cannot perceive (recorded "
                       "cases 2026-07-27/28). The quote is a locator and "
                       "the paper's bytes are the authority — the same "
                       "principle as render-equivalent re-anchoring, "
                       "extended to a unique near-match on the spec's "
                       "quote floors."),
            alternative=("Keep the halt and have a human transcribe the "
                         "passage."),
            override=("Restore the original quote from "
                      "`.pipeline/quote_adoptions.json` into "
                      ".pipeline/method_spec.json and re-run stage 1."),
        )
    if adopted:
        sidecar = state.paths.pipeline_dir / "quote_adoptions.json"
        existing: list[dict] = []
        if sidecar.is_file():
            try:
                loaded = json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    existing = loaded
            except (OSError, json.JSONDecodeError):
                existing = []
        sidecar.write_text(
            json.dumps(existing + adopted, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8")
        keys = [r["element_id"] for r in adopted]
        log("stage_1", "spec_quotes_adopted",
            f"adopted the paper's own bytes for {len(adopted)} near-match "
            f"method-spec quote(s) under logged assumptions: "
            f"{', '.join(keys)}")
        _append_run_event(
            state.paths,
            "spec_quotes_adopted",
            summary=(f"stage 1 adopted the paper's own bytes for "
                     f"{len(adopted)} method-spec quote(s) whose unique "
                     f"near-perfect candidate cleared the adoption rule"),
            details={"adoptions": [
                {"element_id": r["element_id"],
                 "surface": r["surface"],
                 "similarity": r["similarity"],
                 "runner_up": r["runner_up"],
                 "region_resolved": r["region_resolved"],
                 "assumption_id": r["assumption_id"]} for r in adopted]},
        )
    return len(reanchored), len(adopted)


def _run_spec_validator(
    state: PipelineState,
    *,
    require_current_schema: bool = False,
    require_methodology_contract: bool = False,
) -> tuple[bool, str]:
    _assemble_method_spec_if_fresh(state)
    cmd = [
        "scripts/validate_method_spec.py",
        str(state.paths.method_spec),
        "--strict",
    ]
    if require_current_schema:
        cmd.append("--require-current-schema")
    if require_methodology_contract:
        cmd.append("--require-methodology-contract")
    # Gap-path runs: the strict cross-check must see the run's own installed
    # provisional packs (the serves() overlay), or the run halts right after
    # the gap path did its job. Non-gap runs have no pack dir and are
    # byte-identical to before.
    if state.paths.provisional_packs_dir.is_dir():
        cmd += ["--provisional-packs-dir", str(state.paths.provisional_packs_dir)]
    paper_md = state.paths.pipeline_dir / "paper.md"
    if paper_md.is_file():
        # Verbatim evidence floors: param glossary plus scenario assumptions
        # (scenario-fidelity slice A, 2026-07-23).
        cmd += ["--paper-md", str(paper_md)]
    proc = run_script(
        "stage_1",
        cmd,
        timeout=60,
    )
    ok = proc.returncode == 0
    if not ok:
        # The spec's quote floors get the same deterministic repair
        # chain the paper map's equation floor has (render-equivalent
        # re-anchor, then R2C-030 near-match adoption) BEFORE any
        # fix-loop retry — the 0728b bayesian-active-learning roll
        # burned all 3 retries on meaning quotes whose only defect was
        # spaces inside math delimiters. Any repair re-runs the
        # validator below as the terminal gate; anything else flows to
        # the fix loop unchanged.
        reanchored, adopted = _repair_failing_spec_quotes(state)
        if reanchored or adopted:
            proc = run_script(
                "stage_1",
                cmd,
                timeout=60,
            )
            ok = proc.returncode == 0
    err_tail = _stderr_excerpt(proc.stderr)
    _append_validation_event(
        state.paths,
        stage_id="stage_1",
        validator="validate_method_spec.py --strict",
        ok=ok,
        stderr_tail=err_tail,
        artifacts=[str(state.paths.method_spec)],
    )
    return ok, err_tail


def _run_fresh_spec_validator(state: PipelineState) -> tuple[bool, str]:
    """Validate a fresh Stage 1 producer artifact against this checkout.

    Later repair/revalidation seams intentionally call ``_run_spec_validator``
    without this gate so resumed archived specs remain readable and scoped
    repairs are not forced to rewrite an unrelated schema surface.
    """
    return _run_spec_validator(
        state,
        require_current_schema=True,
        require_methodology_contract=True,
    )


def _run_feasibility_gate(state: PipelineState) -> tuple[bool, dict | None]:
    """Run check_feasibility.py. Returns (ok, halt_record). The script always
    exits 0; a `.halt` sidecar signals a block."""
    paths = state.paths
    if paths.feasibility_halt.exists():
        paths.feasibility_halt.unlink()  # clear stale halt before re-running
    proc = run_script(
        "stage_1",
        [
            "scripts/check_feasibility.py",
            str(paths.method_spec),
            "--output",
            str(paths.feasibility_gate),
        ],
        timeout=60,
    )
    if proc.returncode != 0:
        _append_validation_event(
            paths,
            stage_id="stage_1",
            validator="check_feasibility.py",
            ok=False,
            stderr_tail=proc.stderr[-2000:],
            artifacts=[str(paths.feasibility_gate)],
        )
        return False, {"script_exit": proc.returncode, "stderr": proc.stderr[-2000:]}
    if paths.feasibility_halt.exists():
        halt = json.loads(paths.feasibility_halt.read_text(encoding="utf-8"))
        is_contract_halt = bool(halt.get("not_replicable_core_methodology"))
        _append_run_event(
            paths,
            "contract_halted" if is_contract_halt else "validation_failed",
            stage_id="stage_1",
            status="halted" if is_contract_halt else "failed",
            summary="check_feasibility.py halted the run",
            artifacts=[str(paths.feasibility_halt)],
            details={"validator": "check_feasibility.py", "halt": halt},
        )
        return False, halt
    _append_validation_event(
        paths,
        stage_id="stage_1",
        validator="check_feasibility.py",
        ok=True,
        artifacts=[str(paths.feasibility_gate)],
    )
    return True, None


def _record_feasibility_gate_stubs(state: PipelineState, stage_id: str) -> None:
    """Partial-delivery producer #1 (maintainer-approved 2026-07-05, option A).

    A supporting element the contract marks `not_replicable` proceeds
    today, SILENTLY. Now it proceeds recorded: a stubbed-element record
    plus a spec-time work order (no stub module — these are obligations
    the package will not carry, so `stub_path` stays empty and the
    notebook rule requires only the notice, not a raising cell), one
    assumptions.md entry per element, and a run event. Core elements
    marked not_replicable keep halting at the gate (rule 1, unchanged).

    Runs once per stage-1 completion, AFTER every gate re-run and
    reviewer fix has settled, and reconciles: a corrected spec on resume
    withdraws a stale gate stub (gate-origin records are the ones with
    no stub_path). Never raises — a recording failure must not convert a
    passing gate into a halt; it logs and the delivery-time verifier
    still catches a record whose surfaces are missing.
    """
    from partial_delivery import (load_stubbed_elements,
                                  record_stubbed_element,
                                  remove_stubbed_element, render_work_order)
    try:
        spec = json.loads(state.paths.method_spec.read_text(encoding="utf-8"))
        elements = ((spec.get("methodology_replication_contract") or {})
                    .get("elements") or [])
        stubbed = {
            str(el.get("element_id")): el for el in elements
            if isinstance(el, dict)
            and el.get("role") == "supporting_mechanism"
            and el.get("replication_status") == "not_replicable"
            and el.get("element_id")
        }
        existing, load_error = load_stubbed_elements(state.paths.pipeline_dir)
        if load_error:
            log(stage_id, "gate_stub_recording_failed", load_error)
            return
        existing_gate_ids = {r["element_id"] for r in existing
                             if not r.get("stub_path")}
        # Withdraw gate records the corrected contract no longer supports.
        for eid in sorted(existing_gate_ids - set(stubbed)):
            remove_stubbed_element(state.paths.pipeline_dir, eid)
            aid = _next_assumption_id(state)
            _append_assumption(
                state, aid=aid,
                title=f"Stub record for `{eid}` withdrawn",
                detected=(f"A corrected method spec no longer marks "
                          f"supporting element `{eid}` as not_replicable."),
                action="The partial-delivery stub record was withdrawn; the "
                       "package is no longer partial on this element.",
                reasoning="Stub records mirror the CURRENT contract, never a "
                          "stale one.",
                alternative=None,
                override="No action needed.",
            )
            log(stage_id, "gate_stub_withdrawn",
                f"supporting element {eid} is no longer not_replicable — "
                f"stub record withdrawn")
        new_ids = [eid for eid in stubbed if eid not in existing_gate_ids]
        for eid in new_ids:
            el = stubbed[eid]
            work_order_rel = f"work_orders/{eid}.md"
            wo_path = state.paths.run_dir / work_order_rel
            wo_path.parent.mkdir(parents=True, exist_ok=True)
            rationale = (str(el.get("feasibility_rationale") or "")
                         or "no rationale recorded")
            why = (f"The methodology replication contract marked this "
                   f"supporting element not_replicable at analysis time: "
                   f"{rationale}")
            blockers = [str(b) for b in el.get("blockers") or []
                        if str(b).strip()]
            if blockers:
                why += ("\n\nRecorded blockers:\n"
                        + "\n".join(f"- {b}" for b in blockers))
            quotes = [q for q in [el.get("paper_evidence")] if q]
            wo_path.write_text(render_work_order(
                element_id=eid,
                role="supporting",
                interface=None,
                paper_anchor={"section": str(el.get("paper_section") or ""),
                              "quotes": quotes},
                why_not_built=why,
                verified_neighborhood=None,
            ), encoding="utf-8")
            record_stubbed_element(
                state.paths.pipeline_dir,
                element_id=eid, role="supporting",
                work_order=work_order_rel,
            )
            aid = _next_assumption_id(state)
            _append_assumption(
                state, aid=aid,
                title=f"PARTIAL delivery: supporting component `{eid}` "
                      f"ships as a stub",
                detected=(f"The methodology contract marks supporting "
                          f"element `{eid}` not_replicable: {rationale}"),
                action=(f"The run proceeds WITHOUT this component. It is "
                        f"recorded as a stubbed element, with a "
                        f"researcher-consumable work order at "
                        f"`{work_order_rel}`; every delivery surface leads "
                        f"with PARTIAL and the label can never read "
                        f"verified."),
                reasoning=("A partial but accurate package beats "
                           "explanation-only: the buildable fraction ships "
                           "with the gap explicit on every surface, never "
                           "silent (partial-delivery design §3.5). A CORE "
                           "element in this state still halts the run."),
                alternative=("Halt before generation and supply the missing "
                             "resource or capability first."),
                override=(f"Provide the missing capability, correct the "
                          f"element's replication_status in the spec, and "
                          f"re-run — the stub record withdraws itself."),
            )
            log(stage_id, "gate_stub_recorded",
                f"supporting element {eid} marked not_replicable — recorded "
                f"as a partial-delivery stub with work order {work_order_rel}")
        if new_ids:
            _append_run_event(
                state.paths,
                "gate_stubs_recorded",
                stage_id=stage_id,
                status="completed",
                summary=f"{len(new_ids)} not-replicable supporting "
                        f"element(s) recorded as partial-delivery stubs",
                details={"element_ids": new_ids},
            )
    except Exception as e:  # noqa: BLE001 — recording must not mask a passing gate
        log(stage_id, "gate_stub_recording_failed",
            f"{type(e).__name__}: {e}")


def _registered_paradigm_ids(repo_root: Path) -> list[str]:
    try:
        return registered_paradigm_ids(repo_root)
    except FileNotFoundError:
        return registered_paradigm_ids(REPO_ROOT)


def _markdown_for_gap_report(report: ParadigmGapReport) -> str:
    lines = [
        "# Paradigm Gap Report",
        "",
        f"- Decision: `{report.decision.value}`",
        f"- Confidence: {report.confidence:.2f}",
        f"- Paper slug: {report.paper_slug or '-'}",
        f"- Paper title: {report.paper_title or '-'}",
        f"- Matched existing paradigm: {report.matched_existing_paradigm or '-'}",
        f"- Proposed parent paradigm: {report.proposed_parent_paradigm or '-'}",
        f"- Proposed new paradigm id: {report.proposed_new_paradigm_id or '-'}",
        "",
        "## Paper Paradigm Summary",
        "",
        report.paper_paradigm_summary,
        "",
        "## Paper Evidence",
        "",
    ]
    for item in report.paper_evidence:
        lines.extend(
            [
                f"- `{item.paper_section}`: {item.quote_or_observation}",
                f"  - Relevance: {item.relevance}",
            ]
        )
    lines.extend(["", "## Candidate Matches", ""])
    if report.candidate_matches:
        for item in report.candidate_matches:
            lines.append(f"- `{item.paradigm_id}`: {item.decision} ({item.fit})")
    else:
        lines.append("- None accepted.")
    lines.extend(["", "## Rejected Matches", ""])
    if report.rejected_matches:
        for item in report.rejected_matches:
            lines.append(f"- `{item.paradigm_id}`: {item.decision} ({item.fit})")
    else:
        lines.append("- None recorded.")
    lines.extend(
        [
            "",
            "## Registered Paradigms",
            "",
            ", ".join(f"`{item}`" for item in report.registered_paradigms) or "-",
            "",
            "## Recommended Next Action",
            "",
            report.recommended_next_action,
            "",
        ]
    )
    return "\n".join(lines)


def _write_gap_report(paths: PipelinePaths, report: ParadigmGapReport) -> None:
    paths.paradigm_gap_report.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    paths.paradigm_gap_report_md.write_text(
        _markdown_for_gap_report(report),
        encoding="utf-8",
    )


def _clear_analyzer_halt_artifacts(paths: PipelinePaths) -> None:
    """Clear stale analyzer halt/gap artifacts before a fresh analyzer dispatch."""
    for path in (
        paths.method_spec_halt,
        paths.paradigm_gap_report,
        paths.paradigm_gap_report_md,
    ):
        if path.exists():
            path.unlink()


def _gap_report_from_halt(state: PipelineState, halt_record: dict) -> ParadigmGapReport:
    existing_candidate = None
    if isinstance(halt_record, dict):
        try:
            existing_candidate = ParadigmGapReport.model_validate(halt_record)
        except Exception:
            existing_candidate = None
    if existing_candidate is not None:
        return existing_candidate

    registered = halt_record.get("registered_paradigms") if isinstance(halt_record, dict) else None
    if not isinstance(registered, list) or not all(isinstance(item, str) for item in registered):
        registered = _registered_paradigm_ids(state.paths.repo_root)
    summary = ""
    evidence = ""
    if isinstance(halt_record, dict):
        summary = str(halt_record.get("paper_paradigm_summary") or "")
        evidence = str(halt_record.get("evidence") or "")
    if not summary:
        summary = (
            "The analyzer halted because the paper did not match any registered "
            "taxonomy node closely enough to safely produce a method_spec.json."
        )
    if not evidence:
        if isinstance(halt_record, dict):
            evidence = str(halt_record.get("reason") or "paradigm mismatch")
        else:
            evidence = str(halt_record or "paradigm mismatch")

    return ParadigmGapReport(
        decision=GapDecision.unsupported_or_unclear,
        confidence=0.5,
        paper_slug=state.paths.slug,
        paper_title=None,
        paper_paradigm_summary=summary,
        registered_paradigms=registered,
        paper_evidence=[
            EvidenceQuote(
                paper_section="analyzer_halt",
                quote_or_observation=evidence,
                relevance="Analyzer reported this as the reason it could not select a registered taxonomy node.",
            )
        ],
        recommended_next_action=(
            "Review this gap report. If the paper should be supported, run the "
            "taxonomy pack proposal workflow to draft a candidate pack, validate it, "
            "review it, promote it into docs/ssot/proposed_packs/, and rerun the paper."
        ),
    )


def _taxonomy_group_codes() -> set[str]:
    """Casefolded method-root and family codes from the SSOT (TE, TE-TS,
    OM-OPT, ...) — the canonical taxonomy vocabulary's grouping prefixes,
    which are NOT legacy paradigm ids."""
    tax = load_taxonomy()
    codes = {rid.casefold() for rid in tax.roots}
    for root in tax.roots.values():
        codes.update(fid.casefold() for fid in root.families)
    return codes


def _strip_trailing_gloss(value: str) -> str:
    """'TE-TS (Training Strategy family)' -> 'TE-TS'."""
    return re.sub(r"\s*\([^()]*\)\s*$", "", value).strip()


def _translate_gap_report_vocabulary(raw: object) -> tuple[dict, list[str]] | None:
    """Deterministic vocabulary translation for a gap report whose id
    fields use the canonical taxonomy vocabulary instead of legacy ids.

    Two live cases (fedavg 2026-07-06, both attempts): the analyzer
    proposed `proposed_parent_paradigm="TE-TS"` (attempt 2; attempt 1
    added a prose gloss) with `proposed_new_paradigm_id=
    "TE-TS/federated_learning"`. The analyzer's context speaks canonical
    taxonomy ids, the gap schema and the proposal pipeline speak
    lowercase legacy ids, and a prompt rule alone cannot be trusted to
    hold (and agent definitions load at server start, so a prompt fix
    is not even live until a restart). This is the prevention ladder's
    deterministic rung.

    The translation implements exactly the equivalence the analyzer
    prompt documents: a sub-paradigm proposal whose parent is a taxonomy
    GROUP CODE (method root or family, not a legacy id) IS a new
    top-level proposal in the legacy vocabulary, with family placement
    deferred to pack promotion. Scope is deliberately narrow — parent
    resolves (case-insensitively, after one trailing-gloss strip) to a
    known group code, the proposed id is `<same code>/<one segment>`,
    and the segment is already legal lowercase form. Anything else
    returns None and the caller falls through to the exact old halt.
    Returns (translated_raw, notes) on success."""
    if not isinstance(raw, dict):
        return None
    if raw.get("decision") != "new_subparadigm_needed":
        return None
    parent = raw.get("proposed_parent_paradigm")
    proposed = raw.get("proposed_new_paradigm_id")
    if not isinstance(parent, str) or not isinstance(proposed, str):
        return None
    parent_code = _strip_trailing_gloss(parent)
    proposed_clean = _strip_trailing_gloss(proposed)
    group_codes = _taxonomy_group_codes()
    if parent_code.casefold() not in group_codes:
        return None
    # The proposed id arrives in either of two live shapes for the same
    # semantic statement: prefixed by the parent's group code (attempts 1
    # and 2: "TE-TS/federated_learning") or bare (attempt 3:
    # "federated_learning"). Both mean "one new node under that group".
    head, sep, rest = proposed_clean.partition("/")
    if sep:
        if head.casefold() != parent_code.casefold():
            return None
    else:
        rest = proposed_clean
    if "/" in rest or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", rest):
        return None
    translated = dict(raw)
    translated["decision"] = "new_top_level_needed"
    translated["proposed_parent_paradigm"] = None
    translated["proposed_new_paradigm_id"] = rest
    action = str(raw.get("recommended_next_action") or "").rstrip()
    placement = (
        f"Taxonomy placement suggested by the analyzer: under the "
        f"'{parent_code}' group (recorded here because the legacy "
        f"vocabulary has no id for that group; placement is decided at "
        f"pack-promotion time)."
    )
    translated["recommended_next_action"] = (
        f"{action} {placement}".strip() if action else placement
    )
    notes = [
        f"proposed_parent_paradigm: {parent!r} -> None "
        f"(taxonomy group code, not a legacy id)",
        f"proposed_new_paradigm_id: {proposed!r} -> {rest!r}",
        "decision: new_subparadigm_needed -> new_top_level_needed "
        "(the documented vocabulary equivalence, not a judgment change)",
    ]
    return translated, notes


def _repair_evidence_field_alias(entry: object) -> "EvidenceQuote | None":
    """Rename a single drifted field name on one evidence entry.

    Applies ONLY to the unambiguous one-to-one shape: exactly one key the
    schema does not know, exactly one required field missing, and a string
    value to carry over. The content is the analyzer's own text, moved not
    invented, and anything less clear-cut falls through to the drop path.

    Why renaming beats dropping (pdfgnn 2026-07-27): a model that drifts on
    a field name drifts on EVERY entry, so the drop-and-keep-the-rest
    salvage below could not keep anything and refused, discarding a correct
    and complete gap decision (`new_top_level_needed` for
    time_series_forecasting, five real quotes) over one wrong key repeated
    five times. Dropping evidence we can plainly read is the wrong
    primitive; the params applier's almost-correct-resolution remaps are
    the same call made twice before."""
    if not isinstance(entry, dict):
        return None
    known = set(EvidenceQuote.model_fields)
    unknown = [k for k in entry if k not in known]
    missing = [k for k in known if k not in entry]
    if len(unknown) != 1 or len(missing) != 1:
        return None
    if not isinstance(entry[unknown[0]], str):
        return None
    candidate = {k: v for k, v in entry.items() if k != unknown[0]}
    candidate[missing[0]] = entry[unknown[0]]
    try:
        return EvidenceQuote.model_validate(candidate)
    except Exception:  # noqa: BLE001 — the entry was wrong beyond its key
        return None


def _salvage_gap_report(raw: object) -> tuple["ParadigmGapReport | None", int, int]:
    """One deterministic salvage shape for an analyzer gap report that fails
    strict validation: repair drifted evidence field names, drop the
    still-malformed SUPPORTING-evidence entries, and re-try.

    The decision, confidence, and summary are the load-bearing fields the
    recovery keys on; an evidence quote with a typo'd key (SRL re-roll #2,
    2026-07-05: `quote_or_obsorption`) must not silently discard a correct
    gap decision. Anything wrong outside `paper_evidence` still refuses —
    a bad decision value is not salvageable. Returns
    (report, n_dropped, n_repaired) with report None when salvage does not
    apply."""
    if not isinstance(raw, dict) or not isinstance(raw.get("paper_evidence"), list):
        return None, 0, 0
    kept, dropped, repaired = [], 0, 0
    for entry in raw["paper_evidence"]:
        try:
            kept.append(EvidenceQuote.model_validate(entry))
            continue
        except Exception:  # noqa: BLE001 — try the bounded key repair next
            pass
        fixed = _repair_evidence_field_alias(entry)
        if fixed is not None:
            kept.append(fixed)
            repaired += 1
        else:
            dropped += 1
    if (dropped == 0 and repaired == 0) or not kept:
        return None, dropped, repaired
    try:
        report = ParadigmGapReport.model_validate(
            {**raw, "paper_evidence": [e.model_dump() for e in kept]})
    except Exception:  # noqa: BLE001 — defect is not confined to evidence
        return None, dropped, repaired
    return report, dropped, repaired


def _ensure_paradigm_gap_report(
    state: PipelineState,
    halt_record: dict,
) -> dict:
    paths = state.paths
    if paths.paradigm_gap_report.is_file():
        try:
            raw = json.loads(paths.paradigm_gap_report.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "invalid",
                "path": str(paths.paradigm_gap_report),
                "error": str(exc),
            }
        try:
            report = ParadigmGapReport.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            translated_report = None
            translation = _translate_gap_report_vocabulary(raw)
            if translation is not None:
                translated_raw, notes = translation
                try:
                    translated_report = ParadigmGapReport.model_validate(
                        translated_raw)
                except Exception:  # noqa: BLE001 — defect beyond vocabulary
                    translated_report = None
            if translated_report is not None:
                rejected = paths.paradigm_gap_report.with_suffix(
                    ".json.rejected")
                rejected.write_text(json.dumps(raw, indent=2) + "\n",
                                    encoding="utf-8")
                _write_gap_report(paths, translated_report)
                log("stage_1", "gap_report_vocabulary_translated",
                    "gap report used canonical taxonomy ids; translated "
                    "deterministically to the legacy vocabulary (raw "
                    f"preserved at {rejected.name}): " + "; ".join(notes))
                _append_run_event(
                    paths,
                    "gap_report_vocabulary_translated",
                    stage_id="stage_1",
                    status="completed",
                    summary="gap report translated from taxonomy to legacy "
                            "id vocabulary; analyzer judgment unchanged",
                    details={"notes": notes, "rejected_path": str(rejected)},
                )
                report = translated_report
            else:
                salvaged, dropped, repaired = _salvage_gap_report(raw)
                if salvaged is None:
                    log("stage_1", "gap_report_invalid",
                        f"paradigm_gap_report.json failed schema validation "
                        f"and is not salvageable: {str(exc)[:300]}")
                    return {
                        "status": "invalid",
                        "path": str(paths.paradigm_gap_report),
                        "error": str(exc),
                    }
                # Preserve the raw artifact for audit, write the repaired
                # report as the canonical file (downstream pack authoring
                # reads it from disk), and say so loudly.
                rejected = paths.paradigm_gap_report.with_suffix(
                    ".json.rejected")
                rejected.write_text(json.dumps(raw, indent=2) + "\n",
                                    encoding="utf-8")
                _write_gap_report(paths, salvaged)
                actions = []
                if repaired:
                    actions.append(
                        f"renamed one drifted field name on {repaired} "
                        f"supporting-evidence entr"
                        f"{'y' if repaired == 1 else 'ies'}")
                if dropped:
                    actions.append(
                        f"dropped {dropped} malformed supporting-evidence "
                        f"entr{'y' if dropped == 1 else 'ies'}")
                action_text = " and ".join(actions)
                log("stage_1", "gap_report_salvaged",
                    f"{action_text} in paradigm_gap_report.json (raw "
                    f"preserved at {rejected.name}); decision "
                    f"'{salvaged.decision.value}' is intact")
                _append_run_event(
                    paths,
                    "gap_report_salvaged",
                    stage_id="stage_1",
                    status="completed",
                    summary=f"gap report salvaged: {action_text}, "
                            f"decision intact",
                    details={"dropped_entries": dropped,
                             "repaired_entries": repaired,
                             "decision": salvaged.decision.value,
                             "rejected_path": str(rejected)},
                )
                report = salvaged
        if not paths.paradigm_gap_report_md.is_file():
            paths.paradigm_gap_report_md.write_text(
                _markdown_for_gap_report(report),
                encoding="utf-8",
            )
        return report.model_dump(mode="json") | {
            "path": str(paths.paradigm_gap_report),
            "markdown_path": str(paths.paradigm_gap_report_md),
        }

    report = _gap_report_from_halt(state, halt_record)
    _write_gap_report(paths, report)
    return report.model_dump(mode="json") | {
        "path": str(paths.paradigm_gap_report),
        "markdown_path": str(paths.paradigm_gap_report_md),
        "synthesized_by_driver": True,
    }


def _safe_path_for_manifest(paths: PipelinePaths, path: Path) -> str:
    try:
        return path.resolve().relative_to(paths.repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _read_provisional_pack_manifest(paths: PipelinePaths) -> dict | None:
    if not paths.provisional_pack_manifest.is_file():
        return None
    try:
        raw = json.loads(paths.provisional_pack_manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return raw if isinstance(raw, dict) else None


def _make_pack_author_dispatch_fn(state: PipelineState):
    # _dispatch_with_scope_check, not bare dispatch_agent: the pack author
    # has no WRITEABLE_PATHS entry, so the wrapper adds exactly the repo-scope
    # write boundary (quarantine + structured OutOfScopeWritesError) and
    # nothing else. This was the last bare dispatch_agent caller in the
    # driver; the authoring loop's own six-root snapshot guard keeps its
    # stricter run-dir coverage on top.
    def _dispatch(prompt: str) -> DispatchResult:
        return _dispatch_with_scope_check(
            state=state,
            agent=PACK_AUTHOR_AGENT,
            prompt=prompt,
            timeout_s=ANALYZER_TIMEOUT_S,
        )

    return _dispatch


def _install_provisional_pack(
    paths: PipelinePaths,
    *,
    proposal_dir: Path,
    authoring_iterations: int,
    validation_errors: list[str],
    validation_warnings: list[str],
) -> dict:
    proposal = json.loads((proposal_dir / "proposal.json").read_text(encoding="utf-8"))
    install_dir = paths.provisional_packs_dir / proposal["proposal_id"]
    if install_dir.exists():
        shutil.rmtree(install_dir)
    install_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(proposal_dir, install_dir)

    pack_path = install_dir / "pack.yaml"
    manifest = {
        "schema_version": "1.0.0",
        "status": "installed",
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "proposal_id": proposal["proposal_id"],
        "decision": proposal["decision"],
        "target_paradigm_id": proposal["target_paradigm_id"],
        "target_taxonomy_id": proposal.get("target_taxonomy_id"),
        "extends": proposal.get("extends"),
        "pack_path": _safe_path_for_manifest(paths, pack_path),
        "installed_dir": _safe_path_for_manifest(paths, install_dir),
        "proposal_dir": _safe_path_for_manifest(paths, proposal_dir),
        "validation_report_path": _safe_path_for_manifest(
            paths, install_dir / "validation_report.json"
        ),
        "source_gap_report": _safe_path_for_manifest(paths, paths.paradigm_gap_report),
        "authoring_iterations": authoring_iterations,
        "validation_errors": validation_errors,
        "validation_warnings": validation_warnings,
        "promotion_required_for_reuse": True,
    }
    paths.provisional_pack_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _gap_claim_shadowed_committed_family(
    gap_report: dict, repo_root: Path,
) -> str | None:
    """Return the proposed-new-family id when a new_top_level_needed gap
    claim proposes an id the committed taxonomy already owns, else None.

    Failure class gap_claim_shadows_committed_family (pdfgnn Attempt 2,
    2026-08-10): a nondeterministic classification miss looks like a gap,
    and authoring a pack for it burns the author budget on a proposal the
    validator must reject as a duplicate.
    """
    if gap_report.get("decision") != GapDecision.new_top_level_needed.value:
        return None
    proposed_id = gap_report.get("proposed_new_paradigm_id")
    if not isinstance(proposed_id, str) or not proposed_id:
        return None
    try:
        tax = load_taxonomy(repo_root)
    except (OSError, ValueError):
        # No readable committed taxonomy (e.g. a harness repo root): the
        # shadow cannot be proven, so fail open — the proposal validator
        # still rejects a duplicate downstream.
        return None
    if tax.node_for_legacy(proposed_id) is None:
        return None
    return proposed_id


def _recover_with_provisional_pack(
    state: PipelineState,
    stage_id: str,
    *,
    halt_record: dict,
    gap_report: dict,
    reason: str,
) -> StageResult | None:
    """Author, validate, install, and retry with a run-local pack when Stage 1
    gets a structured paradigm gap. Returns None when the gap is not eligible."""

    paths = state.paths
    decision = gap_report.get("decision")
    if decision not in {
        GapDecision.new_subparadigm_needed.value,
        GapDecision.new_top_level_needed.value,
    }:
        # Never decline silently (SRL re-roll #2, 2026-07-05: an invalid
        # gap report fell through here with no trace and the run collapsed
        # to a generic mismatch halt).
        if gap_report.get("status") == "invalid":
            log(stage_id, "provisional_pack_declined",
                f"gap report failed schema validation, recovery cannot "
                f"read a decision: {str(gap_report.get('error'))[:300]}")
        else:
            log(stage_id, "provisional_pack_declined",
                f"gap decision {decision!r} is not pack-eligible "
                f"(needs new_subparadigm_needed or new_top_level_needed)")
        return None

    existing = _read_provisional_pack_manifest(paths)
    if existing is not None:
        log(stage_id, "provisional_pack_already_attempted",
            "analyzer halted after a provisional pack was already installed")
        return None

    # Failure class gap_claim_shadows_committed_family (pdfgnn Attempt 2,
    # 2026-08-10): the analyzer nondeterministically misses a committed
    # taxonomy match and proposes a "new" family whose id already exists.
    # Authoring a pack for it burns three author dispatches on a proposal
    # the validator must reject. The gap report names the proposed id, so
    # check it deterministically first: grant ONE fresh stage 1 retry (the
    # same paper classified into the committed node on adjacent rolls), and
    # halt honestly if the shadow claim repeats.
    proposed_id = _gap_claim_shadowed_committed_family(
        gap_report, paths.repo_root
    )
    if proposed_id is not None:
        marker = paths.pipeline_dir / "gap_committed_family_retry.json"
        if marker.exists():
            return halt(
                paths, stage_id,
                reason=(
                    f"the analyzer twice claimed a paradigm gap proposing "
                    f"{proposed_id!r}, which is already a committed taxonomy "
                    f"family; classification is failing to match the "
                    f"committed node, not finding a real gap"
                ),
                halt_class="gap_pack_rejected",
                context={
                    "analyzer_halt": halt_record,
                    "paradigm_gap_report": gap_report,
                    "failure_class": "gap_claim_shadows_committed_family",
                },
                state=state,
            )
        marker.write_text(
            json.dumps({
                "proposed_new_paradigm_id": proposed_id,
                "first_claim_recorded": True,
            }),
            encoding="utf-8",
        )
        log(stage_id, "gap_claim_shadows_committed_family",
            f"gap report proposes {proposed_id!r}, which already exists in "
            f"the committed taxonomy; skipping pack authoring and retrying "
            f"stage 1 once")
        _append_run_event(
            paths,
            "gap_claim_shadows_committed_family",
            status="completed",
            summary=(
                "Gap claim proposed an already-committed family; retrying "
                "stage 1 instead of authoring a pack"
            ),
            details={"proposed_new_paradigm_id": proposed_id},
        )
        _clear_analyzer_halt_artifacts(paths)
        _clear_stage_1_parts_dir(_method_spec_parts_dir(paths))
        return run_stage_1(state)

    log(stage_id, "provisional_pack_authoring",
        f"analyzer reported {decision}; dispatching {PACK_AUTHOR_AGENT}")
    try:
        result = author_proposal(
            run_dir=paths.run_dir,
            repo_root=paths.repo_root,
            max_iterations=3,
            dispatch_fn=_make_pack_author_dispatch_fn(state),
        )
    except (AuthoringError, OpencodeClientError, OutOfScopeWritesError) as exc:
        return halt(
            paths,
            stage_id,
            reason=f"pack proposal authoring failed after {reason}: {exc}",
            halt_class=_pack_authoring_halt_class(exc),
            context={
                "analyzer_halt": halt_record,
                "paradigm_gap_report": gap_report,
                "authoring_error": str(exc),
            },
            state=state,
        )

    if not result.valid:
        return halt(
            paths,
            stage_id,
            reason="pack proposal did not validate; cannot continue with provisional pack",
            halt_class="gap_pack_rejected",
            context={
                "analyzer_halt": halt_record,
                "paradigm_gap_report": gap_report,
                "proposal_dir": str(result.proposal_dir),
                "validation_report_path": str(result.validation_report_path),
                "authoring_log_path": str(result.authoring_log_path),
                "errors": result.errors,
                "warnings": result.warnings,
            },
            state=state,
        )

    manifest = _install_provisional_pack(
        paths,
        proposal_dir=result.proposal_dir,
        authoring_iterations=result.iterations,
        validation_errors=result.errors,
        validation_warnings=result.warnings,
    )
    log(stage_id, "provisional_pack_installed",
        f"installed {manifest['pack_path']}; retrying analyzer")
    _append_run_event(
        paths,
        "provisional_pack_installed",
        status="completed",
        summary="Installed run-local provisional pack after paradigm gap",
        details=manifest,
    )

    _clear_analyzer_halt_artifacts(paths)
    _clear_stage_1_parts_dir(_method_spec_parts_dir(paths))
    return run_stage_1(state)


def _analyzer_paradigm_halt_result(
    state: PipelineState,
    stage_id: str,
    *,
    halt_record: dict,
    reason: str,
) -> StageResult:
    gap_report = _ensure_paradigm_gap_report(state, halt_record)
    recovery = _recover_with_provisional_pack(
        state,
        stage_id,
        halt_record=halt_record,
        gap_report=gap_report,
        reason=reason,
    )
    if recovery is not None:
        return recovery

    context = {
        "analyzer_halt": halt_record,
        "paradigm_gap_report": gap_report,
    }
    provisional_manifest = _read_provisional_pack_manifest(state.paths)
    if provisional_manifest is not None:
        context["provisional_pack"] = provisional_manifest
    if gap_report.get("status") == "invalid":
        # The halt must name the real blocker: the gap report exists but
        # could not be read, which is a different fact than "no gap".
        reason = (f"{reason}; the analyzer's paradigm_gap_report.json "
                  f"failed schema validation and was not salvageable, so "
                  f"the provisional-pack recovery could not run")
    return halt(
        state.paths,
        stage_id,
        reason=reason,
        halt_class="paradigm_mismatch",
        context=context,
        user_message=_HALT_MSG_PARADIGM_MISMATCH,
        state=state,
    )


def run_stage_1(state: PipelineState) -> StageResult:
    """Stage 1: decomposer → paper_map → analyzer → method_spec → feasibility
    gate → stage-reviewer (cap 1 fix-mode retry).

    Split into 1.a (decomposer + validate_paper_map) and 1.b (analyzer +
    validate_method_spec + feasibility + reviewer). The split matches each
    agent's actual contract — the analyzer's prompt declares paper_map as
    input, not output — and ensures method_spec.core_method.key_elements
    can reference IDs that already exist in paper_map.json."""
    stage_id = "stage_1"
    paths = state.paths

    skipped = _skip_if_done(
        paths, stage_id,
        [paths.method_spec, paths.paper_map,
         paths.pipeline_dir / "stage_review_stage_1_analyzer.json"],
    )
    if skipped is not None:
        return skipped

    paths_written: list[Path] = []

    # --- Stage 1.a: decomposer → paper_map.json -----------------------------
    if not paths.paper_map.exists():
        # Clear any prior decomposer halt sidecar before re-dispatching.
        if paths.paper_map_halt.exists():
            paths.paper_map_halt.unlink()
        _clear_stage_1_parts_dir(_paper_map_parts_dir(paths))
        paper_map_retry_error: str | None = None
        paper_map_scope_error: OutOfScopeWritesError | None = None
        paper_map_attempt = _snapshot_stage_1_artifact_halt(
            paths.paper_map, paths.paper_map_halt,
        )
        try:
            _dispatch_decomposer(state)
        except OpencodeClientError as e:
            return halt(paths, stage_id,
                        reason=f"decomposer dispatch failed: {e}",
                        halt_class="transport_failure",
                        state=state)
        except OutOfScopeWritesError as e:
            if paths.paper_map_halt.exists():
                paper_map_scope_error = e
                log(stage_id, "decomposer_halt_with_scope_violation",
                    f"decomposer wrote halt sidecar plus out-of-scope files: {e}")
            elif _stage_1_chunk_mode_violation(
                e, parts_rel_prefix=".pipeline/paper_map_parts/",
            ):
                paper_map_retry_error = (
                    "canonical-only decomposer dispatch wrote chunk files "
                    f"instead of paper_map.json: {e.violations}"
                )
                log(stage_id, "decomposer_output_mode_retry",
                    paper_map_retry_error)
                _clear_stage_1_parts_dir(_paper_map_parts_dir(paths))
            else:
                return halt(paths, stage_id,
                            reason=f"decomposer dispatch failed: {e}",
                            halt_class="out_of_scope_write",
                            state=state)

        if _stage_1_halt_sidecar_is_authoritative(
            state,
            artifact=paths.paper_map,
            halt_sidecar=paths.paper_map_halt,
            before_dispatch=paper_map_attempt,
            agent="r2c-decomposer",
            attempt="initial",
        ):
            halt_record = json.loads(paths.paper_map_halt.read_text(encoding="utf-8"))
            return halt(paths, stage_id,
                        reason="decomposer halted (paper not decomposable)",
                        halt_class="bad_input_file",
                        context={"decomposer_halt": halt_record},
                        user_message=_HALT_MSG_NOT_DECOMPOSABLE, state=state)
        if paper_map_scope_error is not None:
            return halt(
                paths,
                stage_id,
                reason=f"decomposer dispatch failed: {paper_map_scope_error}",
                halt_class="out_of_scope_write",
                state=state,
            )
        paper_map_assembly = None
        if paper_map_retry_error is None:
            paper_map_parse_error = _json_parse_error(paths.paper_map)
            if paper_map_parse_error is not None:
                paper_map_retry_error = paper_map_parse_error
                log(stage_id, "decomposer_canonical_json_retry",
                    paper_map_retry_error)
                try:
                    paths.paper_map.unlink()
                except OSError:
                    pass
            else:
                paper_map_assembly = _assemble_paper_map_if_fresh(state)
        if not paths.paper_map.exists():
            # Missing-output retry — Think-class stop-early recovery. The agent
            # did the analytical work but never wrote either the canonical file
            # or chunk parts the driver could assemble. One retry with a
            # stronger prompt; if that ALSO fails, halt.
            log(stage_id, "decomposer_missing_output_retry",
                "initial decomposer dispatch produced no paper_map.json or "
                "assemblable paper_map_parts; retrying with stronger prompt")
            paper_map_retry_scope_error: OutOfScopeWritesError | None = None
            try:
                previous_output_error = paper_map_retry_error
                if previous_output_error is None and paper_map_assembly is not None:
                    previous_output_error = paper_map_assembly.message
                paper_map_retry_attempt = _snapshot_stage_1_artifact_halt(
                    paths.paper_map, paths.paper_map_halt,
                )
                _dispatch_decomposer_retry(
                    state,
                    previous_output_error=previous_output_error,
                )
            except OpencodeClientError as e:
                return halt(paths, stage_id,
                            reason=f"decomposer retry dispatch failed: {e}",
                            halt_class="transport_failure",
                            state=state)
            except OutOfScopeWritesError as e:
                if _stage_1_canonical_retry_write(
                    e,
                    canonical_rel_path=".pipeline/paper_map.json",
                ):
                    log(stage_id, "decomposer_chunk_retry_wrote_canonical",
                        "chunk-only retry wrote canonical paper_map.json; "
                        "continuing to normal validation")
                    paper_map_assembly = None
                elif paths.paper_map_halt.exists():
                    paper_map_retry_scope_error = e
                    log(
                        stage_id,
                        "decomposer_halt_with_scope_violation",
                        "decomposer retry wrote a halt sidecar plus "
                        f"out-of-scope files: {e}",
                    )
                else:
                    return halt(paths, stage_id,
                                reason=f"decomposer retry dispatch failed: {e}",
                                halt_class="out_of_scope_write",
                                state=state)
            if _stage_1_halt_sidecar_is_authoritative(
                state,
                artifact=paths.paper_map,
                halt_sidecar=paths.paper_map_halt,
                before_dispatch=paper_map_retry_attempt,
                agent="r2c-decomposer",
                attempt="retry",
            ):
                halt_record = json.loads(
                    paths.paper_map_halt.read_text(encoding="utf-8")
                )
                return halt(
                    paths,
                    stage_id,
                    reason="decomposer halted (paper not decomposable) on retry",
                    halt_class="bad_input_file",
                    context={"decomposer_halt": halt_record},
                    user_message=_HALT_MSG_NOT_DECOMPOSABLE,
                    state=state,
                )
            if paper_map_retry_scope_error is not None:
                return halt(
                    paths,
                    stage_id,
                    reason=(
                        "decomposer retry dispatch failed: "
                        f"{paper_map_retry_scope_error}"
                    ),
                    halt_class="out_of_scope_write",
                    state=state,
                )
            paper_map_assembly = _assemble_paper_map_if_fresh(state)
            if not paths.paper_map.exists() and paper_map_assembly is not None:
                failed_paper_map_assembly = paper_map_assembly
                log(stage_id, "decomposer_chunk_assembly_retry",
                    "chunk-mode decomposer output was not assemblable; "
                    "retrying once with exact assembly error")
                _clear_stage_1_parts_dir(_paper_map_parts_dir(paths))
                paper_map_chunk_retry_attempt = _snapshot_stage_1_artifact_halt(
                    paths.paper_map, paths.paper_map_halt,
                )
                paper_map_chunk_scope_error: OutOfScopeWritesError | None = None
                try:
                    _dispatch_decomposer_retry(
                        state,
                        previous_output_error=paper_map_assembly.message,
                    )
                except OpencodeClientError as e:
                    return halt(paths, stage_id,
                                reason=f"decomposer chunk retry dispatch failed: {e}",
                                halt_class="transport_failure",
                                state=state)
                except OutOfScopeWritesError as e:
                    if _stage_1_canonical_retry_write(
                        e,
                        canonical_rel_path=".pipeline/paper_map.json",
                    ):
                        log(stage_id, "decomposer_chunk_retry_wrote_canonical",
                            "chunk-only retry wrote canonical paper_map.json; "
                            "continuing to normal validation")
                        paper_map_assembly = None
                    elif paths.paper_map_halt.exists():
                        paper_map_chunk_scope_error = e
                        log(
                            stage_id,
                            "decomposer_halt_with_scope_violation",
                            "decomposer chunk retry wrote a halt sidecar plus "
                            f"out-of-scope files: {e}",
                        )
                    else:
                        return halt(paths, stage_id,
                                    reason=f"decomposer chunk retry dispatch failed: {e}",
                                    halt_class="out_of_scope_write",
                                    state=state)
                if _stage_1_halt_sidecar_is_authoritative(
                    state,
                    artifact=paths.paper_map,
                    halt_sidecar=paths.paper_map_halt,
                    before_dispatch=paper_map_chunk_retry_attempt,
                    agent="r2c-decomposer",
                    attempt="chunk_retry",
                ):
                    halt_record = json.loads(
                        paths.paper_map_halt.read_text(encoding="utf-8")
                    )
                    return halt(
                        paths,
                        stage_id,
                        reason=(
                            "decomposer halted (paper not decomposable) "
                            "on chunk retry"
                        ),
                        halt_class="bad_input_file",
                        context={"decomposer_halt": halt_record},
                        user_message=_HALT_MSG_NOT_DECOMPOSABLE,
                        state=state,
                    )
                if paper_map_chunk_scope_error is not None:
                    return halt(
                        paths,
                        stage_id,
                        reason=(
                            "decomposer chunk retry dispatch failed: "
                            f"{paper_map_chunk_scope_error}"
                        ),
                        halt_class="out_of_scope_write",
                        state=state,
                    )
                paper_map_assembly = _assemble_paper_map_if_fresh(state)
                if paper_map_assembly is None:
                    paper_map_assembly = failed_paper_map_assembly
            if not paths.paper_map.exists():
                return halt(paths, stage_id,
                            reason="decomposer did not produce paper_map.json "
                                   "or assemblable paper_map_parts after "
                                   "chunk-mode retry",
                            halt_class="producer_wrote_nothing",
                            context={
                                "paper_map_parts": _assembly_context(paper_map_assembly),
                            },
                            state=state)

        retry_halt = _validator_retry(
            state, stage_id=stage_id,
            validator_fn=_run_paper_map_validator,
            fix_dispatch_fn=_dispatch_decomposer_fix,
            validator_label="validate_paper_map.py",
            cap=STAGE_1_VALIDATOR_RETRY_CAP,
            use_judge=True,
            finding_enricher=_paper_map_quote_enricher(state),
        )
        if retry_halt is not None:
            return retry_halt
        paths_written.append(paths.paper_map)
    else:
        if paths.paper_map_halt.exists():
            halt_record = json.loads(
                paths.paper_map_halt.read_text(encoding="utf-8")
            )
            return halt(
                paths,
                stage_id,
                reason=(
                    "decomposer halt sidecar remains authoritative because "
                    "paper_map.json was not written by a current dispatch"
                ),
                halt_class="bad_input_file",
                context={"decomposer_halt": halt_record},
                user_message=_HALT_MSG_NOT_DECOMPOSABLE,
                state=state,
            )
        log(stage_id, "skipped_1a", "paper_map.json already present; skipping decomposer")

    # --- Stage 1.b: analyzer → method_spec.json -----------------------------
    _clear_analyzer_halt_artifacts(paths)
    _clear_stage_1_parts_dir(_method_spec_parts_dir(paths))
    method_spec_retry_error: str | None = None
    method_spec_scope_error: OutOfScopeWritesError | None = None
    method_spec_attempt = _snapshot_stage_1_artifact_halt(
        paths.method_spec, paths.method_spec_halt,
    )
    try:
        _dispatch_analyzer(state)
    except OpencodeClientError as e:
        return halt(paths, stage_id,
                    reason=f"analyzer dispatch failed: {e}",
                    halt_class="transport_failure",
                    state=state)
    except OutOfScopeWritesError as e:
        if paths.method_spec_halt.exists():
            method_spec_scope_error = e
            log(stage_id, "analyzer_halt_with_scope_violation",
                f"analyzer wrote halt sidecar plus out-of-scope files: {e}")
        elif _stage_1_chunk_mode_violation(
            e, parts_rel_prefix=".pipeline/method_spec_parts/",
        ):
            method_spec_retry_error = (
                "canonical-only analyzer dispatch wrote chunk files instead "
                f"of method_spec.json: {e.violations}"
            )
            log(stage_id, "analyzer_output_mode_retry",
                method_spec_retry_error)
            _clear_stage_1_parts_dir(_method_spec_parts_dir(paths))
        else:
            return halt(paths, stage_id,
                        reason=f"analyzer dispatch failed: {e}",
                        halt_class="out_of_scope_write",
                        state=state)

    if _stage_1_halt_sidecar_is_authoritative(
        state,
        artifact=paths.method_spec,
        halt_sidecar=paths.method_spec_halt,
        before_dispatch=method_spec_attempt,
        agent="r2c-method-analyzer",
        attempt="initial",
    ):
        halt_record = json.loads(paths.method_spec_halt.read_text(encoding="utf-8"))
        return _analyzer_paradigm_halt_result(
            state,
            stage_id,
            halt_record=halt_record,
            reason="analyzer halted (paradigm mismatch)",
        )
    if method_spec_scope_error is not None:
        return halt(
            paths,
            stage_id,
            reason=f"analyzer dispatch failed: {method_spec_scope_error}",
            halt_class="out_of_scope_write",
            state=state,
        )
    method_spec_assembly = None
    if method_spec_retry_error is None:
        method_spec_parse_error = _json_parse_error(paths.method_spec)
        if method_spec_parse_error is not None:
            method_spec_retry_error = method_spec_parse_error
            log(stage_id, "analyzer_canonical_json_retry",
                method_spec_retry_error)
            try:
                paths.method_spec.unlink()
            except OSError:
                pass
        else:
            method_spec_assembly = _assemble_method_spec_if_fresh(state)
    if not paths.method_spec.exists():
        # Missing-output retry — same shape as decomposer's. Think-class
        # stop-early recovery; one stronger-prompt retry, then halt.
        log(stage_id, "analyzer_missing_output_retry",
            "initial analyzer dispatch produced no method_spec.json or "
            "assemblable method_spec_parts; retrying with stronger prompt")
        method_spec_retry_scope_error: OutOfScopeWritesError | None = None
        try:
            previous_output_error = method_spec_retry_error
            if previous_output_error is None and method_spec_assembly is not None:
                previous_output_error = method_spec_assembly.message
            method_spec_retry_attempt = _snapshot_stage_1_artifact_halt(
                paths.method_spec, paths.method_spec_halt,
            )
            _dispatch_analyzer_retry(
                state,
                previous_output_error=previous_output_error,
            )
        except OpencodeClientError as e:
            return halt(paths, stage_id,
                        reason=f"analyzer retry dispatch failed: {e}",
                        halt_class="transport_failure",
                        state=state)
        except OutOfScopeWritesError as e:
            if _stage_1_canonical_retry_write(
                e,
                canonical_rel_path=".pipeline/method_spec.json",
            ):
                log(stage_id, "analyzer_chunk_retry_wrote_canonical",
                    "chunk-only retry wrote canonical method_spec.json; "
                    "continuing to normal validation")
                method_spec_assembly = None
            elif paths.method_spec_halt.exists():
                method_spec_retry_scope_error = e
                log(
                    stage_id,
                    "analyzer_halt_with_scope_violation",
                    "analyzer retry wrote a halt sidecar plus out-of-scope "
                    f"files: {e}",
                )
            else:
                return halt(paths, stage_id,
                            reason=f"analyzer retry dispatch failed: {e}",
                            halt_class="out_of_scope_write",
                            state=state)
        if _stage_1_halt_sidecar_is_authoritative(
            state,
            artifact=paths.method_spec,
            halt_sidecar=paths.method_spec_halt,
            before_dispatch=method_spec_retry_attempt,
            agent="r2c-method-analyzer",
            attempt="retry",
        ):
            halt_record = json.loads(paths.method_spec_halt.read_text(encoding="utf-8"))
            return _analyzer_paradigm_halt_result(
                state,
                stage_id,
                halt_record=halt_record,
                reason="analyzer halted (paradigm mismatch) on retry",
            )
        if method_spec_retry_scope_error is not None:
            return halt(
                paths,
                stage_id,
                reason=(
                    "analyzer retry dispatch failed: "
                    f"{method_spec_retry_scope_error}"
                ),
                halt_class="out_of_scope_write",
                state=state,
            )
        method_spec_assembly = _assemble_method_spec_if_fresh(state)
        if not paths.method_spec.exists() and method_spec_assembly is not None:
            failed_method_spec_assembly = method_spec_assembly
            log(stage_id, "analyzer_chunk_assembly_retry",
                "chunk-mode analyzer output was not assemblable; retrying "
                "once with exact assembly error")
            _clear_stage_1_parts_dir(_method_spec_parts_dir(paths))
            method_spec_chunk_retry_attempt = _snapshot_stage_1_artifact_halt(
                paths.method_spec, paths.method_spec_halt,
            )
            method_spec_chunk_scope_error: OutOfScopeWritesError | None = None
            try:
                _dispatch_analyzer_retry(
                    state,
                    previous_output_error=method_spec_assembly.message,
                )
            except OpencodeClientError as e:
                return halt(paths, stage_id,
                            reason=f"analyzer chunk retry dispatch failed: {e}",
                            halt_class="transport_failure",
                            state=state)
            except OutOfScopeWritesError as e:
                if _stage_1_canonical_retry_write(
                    e,
                    canonical_rel_path=".pipeline/method_spec.json",
                ):
                    log(stage_id, "analyzer_chunk_retry_wrote_canonical",
                        "chunk-only retry wrote canonical method_spec.json; "
                        "continuing to normal validation")
                    method_spec_assembly = None
                elif paths.method_spec_halt.exists():
                    method_spec_chunk_scope_error = e
                    log(
                        stage_id,
                        "analyzer_halt_with_scope_violation",
                        "analyzer chunk retry wrote a halt sidecar plus "
                        f"out-of-scope files: {e}",
                    )
                else:
                    return halt(paths, stage_id,
                                reason=f"analyzer chunk retry dispatch failed: {e}",
                                halt_class="out_of_scope_write",
                                state=state)
            if _stage_1_halt_sidecar_is_authoritative(
                state,
                artifact=paths.method_spec,
                halt_sidecar=paths.method_spec_halt,
                before_dispatch=method_spec_chunk_retry_attempt,
                agent="r2c-method-analyzer",
                attempt="chunk_retry",
            ):
                halt_record = json.loads(paths.method_spec_halt.read_text(encoding="utf-8"))
                return _analyzer_paradigm_halt_result(
                    state,
                    stage_id,
                    halt_record=halt_record,
                    reason="analyzer halted (paradigm mismatch) on chunk retry",
                )
            if method_spec_chunk_scope_error is not None:
                return halt(
                    paths,
                    stage_id,
                    reason=(
                        "analyzer chunk retry dispatch failed: "
                        f"{method_spec_chunk_scope_error}"
                    ),
                    halt_class="out_of_scope_write",
                    state=state,
                )
            method_spec_assembly = _assemble_method_spec_if_fresh(state)
            if method_spec_assembly is None:
                method_spec_assembly = failed_method_spec_assembly
        if not paths.method_spec.exists():
            return halt(paths, stage_id,
                        reason="analyzer did not produce method_spec.json "
                               "or assemblable method_spec_parts after "
                               "chunk-mode retry",
                        halt_class="producer_wrote_nothing",
                        context={
                            "method_spec_parts": _assembly_context(method_spec_assembly),
                        },
                        state=state)
    paths_written.append(paths.method_spec)

    # Structural validator — cap=1 fix-mode retry per design plan.
    retry_halt = _validator_retry(
        state, stage_id=stage_id,
        validator_fn=_run_fresh_spec_validator,
        fix_dispatch_fn=_dispatch_analyzer_fix,
        validator_label="validate_method_spec.py --strict",
        cap=STAGE_1_VALIDATOR_RETRY_CAP,
        use_judge=True,
    )
    if retry_halt is not None:
        return retry_halt

    ok, halt_record = _run_feasibility_gate(state)
    if not ok:
        # One analyzer re-ask when every cannot_implement blocker carries
        # the analyzer's own proposed surrogate (2026-07-03 bev-distill
        # roll-variance case); any failure falls through to the old halt.
        ok, halt_record = _reask_feasibility_cannot_implement(
            state, stage_id, halt_record)
    if not ok:
        return halt(paths, stage_id,
                    reason="feasibility gate blocked the run",
                    halt_class="not_feasible",
                    context={"feasibility_halt": halt_record},
                    user_message=_HALT_MSG_NOT_FEASIBLE, state=state)
    paths_written.append(paths.feasibility_gate)

    # Stage-reviewer loop — cap = 1 fix-mode retry on critical findings.
    review_path = paths.pipeline_dir / "stage_review_stage_1_analyzer.json"
    for iteration in range(STAGE_1_REVIEWER_RETRY_CAP + 1):
        try:
            _dispatch_stage_1_reviewer(state)
        except (OpencodeClientError, OutOfScopeWritesError) as e:
            return halt(paths, stage_id,
                        reason=f"stage-reviewer dispatch failed: {e}",
                        halt_class=_dispatch_error_halt_class(e),
                        retry_count=iteration, state=state)
        if not review_path.exists():
            # Missing-output retry — Think-class stop-early recovery
            # (mirrors decomposer + analyzer retry patterns).
            log(stage_id, "stage_reviewer_missing_output_retry",
                "initial stage-reviewer dispatch produced no "
                "stage_review_stage_1_analyzer.json; retrying with stronger prompt")
            try:
                _dispatch_stage_reviewer_retry(state, "stage_1_analyzer")
            except (OpencodeClientError, OutOfScopeWritesError) as e:
                return halt(paths, stage_id,
                            reason=f"stage-reviewer retry dispatch failed: {e}",
                            halt_class=_dispatch_error_halt_class(e),
                            retry_count=iteration, state=state)
            if not review_path.exists():
                return halt(paths, stage_id,
                            reason="stage-reviewer did not write its findings file "
                                   "(even after one missing-output retry)",
                            halt_class="producer_wrote_nothing",
                            retry_count=iteration, state=state)

        review, read_err = _read_review_json_or_err(
            review_path, kind="stage_1_analyzer review",
            expected_stage_id="stage_1_analyzer",
        )
        if read_err:
            return _degrade_reviewer_unusable(
                paths, stage_id, state=state,
                reason=f"stage-reviewer output unusable: {read_err}; "
                       f"shipping the analyzer artifacts unreviewed",
            )
        findings = review.get("findings", []) or []
        crit = critical_findings(findings)
        if not crit:
            paths_written.append(review_path)
            log(stage_id, "review_passed",
                f"iteration {iteration}: {len(findings)} findings, 0 critical")
            break

        if iteration == STAGE_1_REVIEWER_RETRY_CAP:
            return halt(paths, stage_id,
                        reason=f"{len(crit)} critical finding(s) persist after analyzer fix-mode retry",
                        halt_class="fix_loop_exhausted",
                        retry_count=iteration, findings=crit, state=state)

        log(stage_id, "fix_dispatch",
            f"iteration {iteration}: {len(crit)} critical finding(s); re-dispatching analyzer")
        try:
            _dispatch_analyzer_fix(state, crit)
        except (OpencodeClientError, OutOfScopeWritesError) as e:
            return halt(paths, stage_id,
                        reason=f"analyzer fix-mode dispatch failed: {e}",
                        halt_class=_dispatch_error_halt_class(e),
                        retry_count=iteration, state=state)
        # Re-run the structural + feasibility checks against the fixed spec.
        retry_halt = _validator_retry(
            state, stage_id=stage_id,
            validator_fn=_run_fresh_spec_validator,
            fix_dispatch_fn=_dispatch_analyzer_fix,
            validator_label="validate_method_spec.py --strict (post-reviewer-fix)",
            use_judge=True,
        )
        if retry_halt is not None:
            return retry_halt
        ok, halt_record = _run_feasibility_gate(state)
        if not ok:
            return halt(paths, stage_id,
                        reason="feasibility gate blocked after analyzer fix-mode",
                        halt_class="not_feasible",
                        retry_count=iteration,
                        context={"feasibility_halt": halt_record},
                        user_message=_HALT_MSG_NOT_FEASIBLE, state=state)

    # Partial-delivery producer #1: record (and reconcile) stubs for
    # not-replicable supporting elements, after every gate re-run and
    # reviewer fix has settled on the final spec.
    _record_feasibility_gate_stubs(state, stage_id)

    return StageResult(
        status="completed",
        stage_id=stage_id,
        notes="analyzer + paper_map + feasibility + review clean",
        paths_written=paths_written,
    )


# ---------------------------------------------------------------------------
# Stage 2 — package generation (2.a scaffolder, 2.b architecture-coder,
#                               2.c method-coder, 2.d init-finalizer,
#                               2.x parameter-deriver)
# ---------------------------------------------------------------------------


def _dispatch_stage_reviewer(state: PipelineState, stage_id: str) -> DispatchResult:
    """Generic stage-reviewer dispatch. Used by Stage 2.b/2.c/2.x. The reviewer
    writes `<PIPELINE_DIR>/stage_review_<stage_id>.json` per its agent contract."""
    agent = "r2c-stage-reviewer"
    review_path = state.paths.pipeline_dir / f"stage_review_{stage_id}.json"
    prompt = build_dispatch_prompt(
        task_summary=STAGE_REVIEW_TASK_TEMPLATE.format(stage_id=stage_id),
        paths=build_paths_block(state.paths),
        closing=REVIEWER_CLOSING,
        writeable_paths=WRITEABLE_PATHS[agent],
        think_anchor_output_path=str(review_path),
    )
    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt=prompt, timeout_s=REVIEWER_TIMEOUT_S,
        recovery_check_fn=_stage_review_valid(stage_id),
        allow_corrective_redispatch=True,
    )


def _read_paper_map_element_ids(pipeline_dir: Path) -> list[str]:
    """Sorted list of valid `# paper-element:` IDs from paper_map.json (the same
    set validate_architecture_coder_output.py checks: elements[].id). Returns []
    when the map is missing/unparseable so the closed-set block is just omitted."""
    path = pipeline_dir / "paper_map.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    elements = data.get("elements") or []
    ids = {e.get("id") for e in elements if isinstance(e, dict) and e.get("id")}
    return sorted(i for i in ids if isinstance(i, str))


def _read_essential_feature_strings(pipeline_dir: Path) -> list[str]:
    """Verbatim severity:essential feature strings from method_spec.json's
    critical_requirements.model.specific_features — exactly what
    validate_method_coder_output.py matches by string equality. Returns []
    when unavailable."""
    path = pipeline_dir / "method_spec.json"
    if not path.is_file():
        return []
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    cr = spec.get("critical_requirements") or {}
    feats = (cr.get("model") or {}).get("specific_features") or []
    return [
        f.get("feature", "")
        for f in feats
        if isinstance(f, dict) and f.get("severity") == "essential" and f.get("feature")
    ]


def _coder_closed_set_sections(paths: PipelinePaths) -> list[str]:
    """Materialize the stage-2.b/2.c closed-set contracts into the coder
    dispatch so the producer copies tokens VERBATIM rather than regenerating
    them from memory (DRV-D1 / DRV-D2):

      - the exact set of valid `# paper-element:` IDs (the arch-coder
        abbreviated `eq-ellipsoid-geodesic`->`eq-geodesic` and failed 2.b,
        2026-06-24); and
      - the verbatim `# essential:` strings the validator matches byte-for-byte
        (the method-coder paraphrased them and failed 2.c).

    Both sections are purely additive context; each is omitted when its source
    artifact is not yet readable."""
    sections: list[str] = []
    ids = _read_paper_map_element_ids(paths.pipeline_dir)
    if ids:
        sections.append(
            "**Valid `# paper-element:` IDs — CLOSED SET.** When you annotate code "
            "with `# paper-element: <id>`, the id MUST be copied character-for-"
            "character from this list. Never abbreviate, pluralize, or invent one "
            "(abbreviating `eq-ellipsoid-geodesic` to `eq-geodesic` is the canonical "
            "failure — it breaks the code->paper trace and fails validation):\n"
            + "\n".join(f"  - `{i}`" for i in ids)
        )
    essential = _read_essential_feature_strings(paths.pipeline_dir)
    if essential:
        sections.append(
            "**Essential-feature annotations — VERBATIM.** For each severity:"
            "essential feature below, emit a `# essential:` comment on the "
            "function/class that implements it, with the string copied EXACTLY "
            "(byte-for-byte — do not paraphrase, reorder, or drop the parenthetical). "
            "The validator and reviewer trace spec->code by exact string match:\n"
            + "\n".join(f"  - `# essential: {s}`" for s in essential)
        )
    bundle_section = _bundled_demo_data_section(paths)
    if bundle_section:
        sections.append(bundle_section)
    return sections


def _bundle_briefing_parts(paths: PipelinePaths) -> dict | None:
    """Shared BUILT body of the two provisioned-data briefings (R2C-062).

    Both consumers — the coder dispatch and the notebook-generator dispatch —
    render the same built pieces: the loader's mechanical return contract
    with a usage example, the dtype/missing-value policy, and the per-file
    blocks carrying the measured axis and column facts. Framing sentences
    are the only per-consumer text. The notebook section once derived its
    facts by string-filtering the coder section's rendered lines, which
    silently dropped every indented continuation line (the night3 roll paid
    three of its four smoke failures for that), so the sharing happens here,
    at the built structure, never on rendered text."""
    example_data = paths.run_dir / "method" / "example_data"
    provenance_path = example_data / "PROVENANCE.json"
    if not provenance_path.is_file():
        return None
    try:
        manifest = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    from demo_data_provenance import disclosure_for  # noqa: PLC0415

    disclosure = disclosure_for(manifest)
    raw_files = manifest.get("files") or []
    if isinstance(raw_files, dict):
        file_entries = []
        for name, facts in sorted(raw_files.items()):
            if not isinstance(name, str) or not name:
                continue
            entry = dict(facts) if isinstance(facts, dict) else {}
            entry["file"] = name
            file_entries.append(entry)
    elif isinstance(raw_files, list):
        file_entries = [entry for entry in raw_files if isinstance(entry, dict)]
    else:
        file_entries = []
    lines: list[str] = []
    for entry in file_entries:
        name = entry.get("file")
        if not name:
            continue
        line = f"  - `method/example_data/{name}`"
        if entry.get("truncated"):
            line += f" ({entry.get('rows_kept')} rows after subsampling)"
        axis = entry.get("time_axis")
        if isinstance(axis, dict) and axis.get("steps_kept"):
            line += (
                f"\n      TIME AXIS: column `{axis.get('column')}` carries "
                f"{axis.get('steps_kept')} distinct steps, "
                f"{axis.get('first_step')} through {axis.get('last_step')}"
                f"{_rows_per_step_note(axis)}. This is the measured granularity "
                f"of the data — if the paper's protocol works at a coarser "
                f"one, YOUR code must aggregate to it."
                + _live_extent_note(axis)
            )
        entity_note = _entity_set_note(entry)
        if entity_note:
            line += f"\n      {entity_note}"
        for column_line in _column_facts(example_data / name):
            line += f"\n      {column_line}"
        lines.append(line)
    if not lines:
        return None
    source_record = manifest.get("source")
    source = (
        source_record.get("cited_url")
        if isinstance(source_record, dict)
        else None
    )
    stems = sorted(
        Path(entry["file"]).stem for entry in file_entries
        if entry.get("file")
    )
    example_stem = stems[0] if stems else "table"
    contract = (
        "`load_data()` returns a dict keyed by file stem "
        f"({', '.join(f'`{s}`' for s in stems)}); each value is itself a "
        "dict mapping COLUMN NAME to a plain 1-D numpy array of that "
        "column's values. There are no DataFrames anywhere in this "
        "contract, so pandas idioms (`.loc`, boolean masks over a table, "
        "`.groupby`) do not apply:\n\n"
        "```python\n"
        "tables = load_data()\n"
        f"columns = tables[{example_stem!r}]          # dict: column name -> np.ndarray\n"
        "values = columns[next(iter(columns))]  # one column, shape (n_rows,)\n"
        "```\n\n"
        "A column whose every non-blank cell parses as a number arrives as a "
        "float array with `nan` in the blanks; any other column arrives as a "
        "STRING array where a blank is `''`. So a numeric column with gaps "
        "needs an explicit nan policy before it reaches a loss, and a "
        "categorical column needs encoding — `float(cell)` on a raw string "
        "column dies on the first blank."
    )
    return {
        "source": source,
        "disclosure": disclosure,
        "contract": contract,
        "file_blocks": "\n".join(lines),
    }


def _bundled_demo_data_section(paths: PipelinePaths) -> str | None:
    """Coder-dispatch section describing provisioned demo data (R2C-052).

    Coders must write their data handling against the provisioned files rather
    than inventing shapes, while preserving the manifest's public, synthetic,
    or unresolved origin. Omitted entirely when no bundle exists, like the
    other sections."""
    parts = _bundle_briefing_parts(paths)
    if parts is None:
        return None
    disclosure = parts["disclosure"]
    if disclosure.is_public:
        framing = (
            "**Bundled demo data — REAL, from the paper's own cited source.** "
            f"R2C fetched a demo-scale extract of the dataset this paper cites "
            f"({parts['source'] or 'a declared paper-cited public source'}) "
            "into `method/example_data/`. Write your data "
            "handling against these actual files and their actual columns — do "
            "NOT invent synthetic shapes, and do NOT assume a dataset the demo "
            "cannot obtain."
        )
    elif disclosure.is_synthetic:
        framing = (
            "**Bundled demo data — FAMILY-OWNED SYNTHETIC OFFLINE FALLBACK.** "
            "R2C generated this deterministic forecasting fixture locally and "
            "placed it in `method/example_data/`. It is not the paper's dataset; "
            "results on it demonstrate package mechanics only and are not "
            "paper-comparable or benchmark evidence. Write your data handling "
            "against these actual fixture files and columns. Do NOT invent a "
            "second synthetic stand-in or describe this bundle as real or "
            "from-paper data."
        )
    else:
        framing = (
            "**Bundled demo data — ORIGIN UNRESOLVED.** A provenance manifest "
            "and data files are present in `method/example_data/`, but their "
            "source tier is not recognized. Write against the measured files "
            "without claiming that they are real, synthetic, or from the paper."
        )
    return (
        framing + "\n\n" + parts["contract"] + "\n\n"
        "If you author `arch_contract.json`, declare "
        "`data_loader.load_data_returns` using exactly those table names — "
        "the stage 2.d dry-run checks the declared names against the dict "
        "`load_data()` actually returns. Whatever transformation the method "
        "needs from these raw columns to its inputs is part of the method's "
        "own code, and the per-column facts below are measured from the "
        "bundled files themselves, so missing-value counts and non-numeric "
        "dtypes must be handled rather than assumed away:\n"
        + parts["file_blocks"]
    )


def _entity_set_note(entry: dict) -> str:
    """The joinable entity set carried by each table (R2C-078).

    New manifests place the facts on every affected file.  The time-axis
    fallback keeps briefings for pre-R2C-078 bundles informative.
    """
    axis = entry.get("time_axis")
    legacy = axis if isinstance(axis, dict) else {}
    count = entry.get("entities_kept", legacy.get("entities_kept"))
    source_count = entry.get(
        "entities_in_source", legacy.get("entities_in_source"))
    column = entry.get("entity_column", legacy.get("entity_column"))
    if not count or not column:
        return ""
    source = (
        f", retained from {source_count} in this source table"
        if source_count and source_count != count else ""
    )
    return (
        f"ENTITY SET: column `{column}` carries {count} distinct value(s)"
        f"{source}. Every bundled table carrying `{column}` uses this same "
        "retained set, so join by the named key rather than row position. "
        "This is a demo-scale population: state it plainly wherever you "
        "report results, and never present demo numbers as the paper's."
    )


def _live_extent_note(axis: dict) -> str:
    """Where the usable data actually ends (R2C-069).

    A public forecasting dataset normally withholds its target over the final
    period, because that period is the competition's own prediction window.
    The 2026-08-06 pdfgnn bundle's last 59 of 1,092 daily steps carry no
    `sales` value; the demo correctly held out its last four weeks, that
    window sat entirely inside the withheld region, and RMSE, MAE and WMAPE
    all printed nan. Nothing in the briefing had said the tail was empty, and
    the per-column line said the opposite, because it samples the head.

    Per-column rather than one table-wide number, because the consumer knows
    which column it treats as the target and the materializer does not."""
    dead = axis.get("dead_tail_columns")
    if not isinstance(dead, list) or not dead:
        return ""
    entries = [d for d in dead if isinstance(d, dict) and d.get("column")]
    if not entries:
        return ""
    detail = "; ".join(
        f"`{d.get('column')}` ends at {d.get('last_live_step')} "
        f"({d.get('dead_tail_steps')} step(s) with no value after it)"
        for d in entries
    )
    earliest = min(
        (d for d in entries if d.get("last_live_step")),
        key=lambda d: d.get("live_steps") or 0, default=None,
    )
    boundary = earliest.get("last_live_step") if earliest else None
    rule = (
        f" Any window your code SCORES on must end at or before the last live "
        f"step of the column you are predicting (earliest boundary here: "
        f"{boundary})"
        if boundary else
        " Any window your code SCORES on must end at or before the last live "
        "step of the column you are predicting"
    )
    return (
        f" USABLE DATA ENDS EARLY: {detail}. This is normal for a public "
        f"forecasting dataset, whose final period is withheld because it is "
        f"the source's own prediction window.{rule} — holding out the empty "
        f"tail computes every metric against missing values and prints nan."
    )


def _rows_per_step_note(axis: dict) -> str:
    cap = axis.get("rows_per_step_cap")
    if not cap:
        return ""
    source_steps = axis.get("steps_in_source")
    tail = (f" out of {source_steps} in the source" if source_steps
            and source_steps != axis.get("steps_kept") else "")
    return (f", thinned to at most {cap} rows per step so the full span "
            f"survived{tail}")


# Per-column facts are measured, never assumed: a briefing that lists only
# column NAMES leaves the model's priors to fill in dtype and completeness,
# and the priors are pandas and clean data (briefing_prior_gap, R2C-062).
_COLUMN_FACT_ROWS = 5000
_COLUMN_FACT_LIMIT = 40


def _column_facts(table: Path) -> list[str]:
    """One line per column: dtype as it actually parses, plus how many of the
    sampled cells are empty. Bounded read — the point is the shape of the
    data, and 5,000 rows settle both questions for a provisioned table.

    The sample is the HEAD of the file, and the line says so (R2C-069). It
    used to read "0/5000 empty", which on the 2026-08-06 pdfgnn bundle
    asserted that `sales` had no gaps anywhere while the last 59 of its 1,092
    days carried no value at all — the tail lives 494,000 rows past the
    sample. Where the data ends is the axis note's job, measured over every
    kept row at bundling time; this line's job is dtype and gap DENSITY, and
    it must not be read as a completeness claim."""
    if table.suffix.lower() not in (".csv", ".tsv") or not table.is_file():
        return []
    delimiter = "\t" if table.suffix.lower() == ".tsv" else ","
    try:
        with table.open("r", encoding="utf-8", errors="replace") as fh:
            rows = csv.reader(fh, delimiter=delimiter)
            header = next(rows, [])
            if not header:
                return []
            sampled = [row for _, row in zip(range(_COLUMN_FACT_ROWS), rows)]
    except (OSError, csv.Error):
        return []
    if not sampled:
        return []
    facts: list[str] = []
    for index, name in enumerate(header[:_COLUMN_FACT_LIMIT]):
        cells = [row[index] if index < len(row) else "" for row in sampled]
        present = [c for c in cells if c.strip()]
        facts.append(
            f"column `{name}`: {_cell_dtype(present)}, "
            f"{len(cells) - len(present)} of the first {len(cells)} rows empty"
        )
    if len(header) > _COLUMN_FACT_LIMIT:
        facts.append(f"({len(header) - _COLUMN_FACT_LIMIT} further column(s) "
                     f"not summarized)")
    return facts


def _cell_dtype(values: list[str]) -> str:
    """How a column's non-empty cells actually parse. `text` is the answer
    that matters most: a bare float() cast over one of these is the failure
    this line exists to prevent."""
    if not values:
        return "all cells empty"
    ints = floats = 0
    for value in values:
        try:
            float(value)
        except ValueError:
            return f"text (e.g. {values[0][:24]!r})"
        ints += value.strip().lstrip("+-").isdigit()
        floats += 1
    if ints == floats:
        return "integer"
    return "float"


def _notebook_bundle_section(paths: PipelinePaths) -> str | None:
    """Notebook-generator counterpart of `_bundled_demo_data_section`
    (notebook_data_binding): the demo must RUN on the provisioned data, not
    merely sit beside it. The pdfgnn 2026-08-04 delivery imported load_data
    but never called it, so the stage 3a validator now fails that shape and
    this section tells the generator the rule up front. Omitted when no bundle
    exists. Renders the same built
    contract and per-file facts as the coder section, via
    `_bundle_briefing_parts` — the generator writes the cells that actually
    execute on this data, so it needs the dtypes, the axis, and the return
    contract at least as much as the coder does."""
    parts = _bundle_briefing_parts(paths)
    if parts is None:
        return None
    disclosure = parts["disclosure"]
    if disclosure.is_public:
        framing = (
            "**Bundled demo data — the demo MUST run on it.** "
            "`method/example_data/` carries a demo-scale extract of the paper's "
            "own cited dataset (provenance recorded beside it). The data section "
            "MUST call `load_data()` and derive the method's inputs from the "
            "returned named tables — do NOT generate synthetic stand-ins for "
            "quantities these tables provide. Synthetic values are acceptable "
            "only for quantities the bundle genuinely lacks, stated in prose."
        )
    elif disclosure.is_synthetic:
        framing = (
            "**Family-owned synthetic fallback — the demo MUST run on it and "
            "disclose it.** `method/example_data/` carries R2C's deterministic "
            "offline forecasting fixture. The data section MUST call "
            "`load_data()` and derive the method's inputs from the returned "
            "named tables; do NOT generate a second stand-in. Notebook markdown "
            "MUST say `family-owned synthetic fallback`, `not the paper's "
            "dataset`, and that its numbers are `not paper-comparable or "
            "benchmark evidence`."
        )
    else:
        framing = (
            "**Bundled demo data — the demo MUST run on it, but its origin is "
            "unresolved.** The data section MUST call `load_data()` and derive "
            "inputs from the returned named tables. Make no real, synthetic, "
            "or from-paper origin claim not established by PROVENANCE.json."
        )
    return (
        framing
        + " The stage 3a validator fails a notebook that never calls "
        "`load_data()` when this bundle exists.\n\n"
        + parts["contract"] + "\n\n"
        "The bundled tables, with per-column facts measured from the files "
        "themselves (missing counts and non-numeric dtypes are measured — "
        "handle them, never assume them away):\n"
        + parts["file_blocks"]
    )


def _arch_outputs_valid(state: PipelineState) -> bool:
    """Recovery check for architecture-coder drift (queue item 19,
    mirroring `_method_py_valid` for the method-coder): the three declared
    outputs exist, the two Python files parse, and the arch contract is
    valid JSON. Deliberately the same lightweight altitude as the
    method-coder's check — the full validator runs right after in the fix
    loop, where its failures get the designed repair path instead of a
    hard halt. Used so a dispatch that produced valid outputs PLUS an
    out-of-scope side effect (detr 2026-07-08: a contract-encouraged
    self-test wrote a run-root `_cache/` stray) is a soft success: the
    side effect is reverted, the canonical outputs are kept."""
    import ast as _ast  # local: only needed here

    method_dir = state.paths.run_dir / "method"
    for name in ("model.py", "training.py"):
        path = method_dir / name
        if not path.is_file():
            return False
        try:
            _ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, OSError, ValueError):
            return False
    contract = state.paths.pipeline_dir / "arch_contract.json"
    if not contract.is_file():
        return False
    try:
        json.loads(contract.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return True


def _dispatch_arch_coder(state: PipelineState) -> DispatchResult:
    agent = "r2c-architecture-coder"
    prompt = build_dispatch_prompt(
        task_summary=STAGE_TASK_SUMMARIES["stage_2b_architecture"],
        paths=build_paths_block(state.paths),
        writeable_paths=WRITEABLE_PATHS[agent],
        extra_sections=_coder_closed_set_sections(state.paths),
        # Cap-burn fix 1: this agent composes large source files, the
        # post-hygiene fatal-burn site (2 of 8 burns after 2026-07-14).
        writing_discipline=True,
    )
    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt=prompt, timeout_s=ARCH_CODER_TIMEOUT_S,
        recovery_check_fn=_arch_outputs_valid,
        allow_corrective_redispatch=True,
    )


def _dispatch_arch_coder_fix(
    state: PipelineState, findings: list[dict],
    *, smoke_context: dict | None = None,
) -> DispatchResult:
    agent = "r2c-architecture-coder"
    prompt = build_fix_mode_prompt(
        target_agent=agent,
        findings=findings,
        paths=build_paths_block(state.paths),
        writeable_paths=WRITEABLE_PATHS[agent],
        smoke_context=smoke_context,
        extra_sections=_coder_closed_set_sections(state.paths),
    )
    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt=prompt, timeout_s=ARCH_CODER_TIMEOUT_S,
        recovery_check_fn=_arch_outputs_valid,
        allow_corrective_redispatch=True,
    )


def _dispatch_method_coder(state: PipelineState) -> DispatchResult:
    agent = "r2c-method-coder"
    prompt = build_dispatch_prompt(
        task_summary=STAGE_TASK_SUMMARIES["stage_2c_method"],
        paths=build_paths_block(state.paths),
        writeable_paths=WRITEABLE_PATHS[agent],
        extra_sections=_coder_closed_set_sections(state.paths),
        # Cap-burn fix 1: the dominant post-hygiene fatal-burn site (4 of 8
        # burns after 2026-07-14, incl. the SRL 07-15 triple burn).
        writing_discipline=True,
    )
    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt=prompt, timeout_s=METHOD_CODER_TIMEOUT_S,
        recovery_check_fn=_method_py_valid,
        allow_corrective_redispatch=True,
    )


def _dispatch_method_coder_fix(
    state: PipelineState, findings: list[dict],
    *, smoke_context: dict | None = None,
) -> DispatchResult:
    agent = "r2c-method-coder"
    prompt = build_fix_mode_prompt(
        target_agent=agent,
        findings=findings,
        paths=build_paths_block(state.paths),
        writeable_paths=WRITEABLE_PATHS[agent],
        smoke_context=smoke_context,
        extra_sections=_coder_closed_set_sections(state.paths),
    )
    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt=prompt, timeout_s=METHOD_CODER_TIMEOUT_S,
        recovery_check_fn=_method_py_valid,
        allow_corrective_redispatch=True,
    )


# ---------------------------------------------------------------------------
# Halt-judge — covers stages 1, 2.b, 2.c, 2.d, 3.a, 3.c, 4
# ---------------------------------------------------------------------------
# Dispatch table for fixers the judge can route to. Keys must match the
# TargetAgent literal in schemas/judge_decision.py — extending the literal
# without extending this table (or vice versa) is a bug; the runtime guard
# in _invoke_judge surfaces the mismatch with a clear error.
#
# r2c-smoke-diagnostician needs runtime smoke context (stderr_tail,
# failing_cell_source) to do a repair-mode dispatch, so stage 3.c's
# `_invoke_judge` call passes a context-bound dispatcher via the
# `local_dispatchers` kwarg instead of relying on the global table entry.
# The global entry is intentionally absent — if the judge picks the
# diagnostician outside stage 3.c (where smoke context isn't bound),
# `_invoke_judge` downgrades the decision to halt.
# Entries for dispatchers defined LATER in this module (notebook-generator,
# paper-fidelity-reviewer) are added below their definitions; see
# `JUDGE_FIX_DISPATCHERS[...] = ...` near _dispatch_notebook_generator_fix
# and _dispatch_paper_fidelity_reviewer_fix.
JUDGE_FIX_DISPATCHERS: dict[str, callable] = {
    "r2c-architecture-coder": _dispatch_arch_coder_fix,
    "r2c-method-coder": _dispatch_method_coder_fix,
    "r2c-method-analyzer": _dispatch_analyzer_fix,
    "r2c-decomposer": _dispatch_decomposer_fix,
}


def _read_prior_judge_decisions(
    paths: PipelinePaths, stage_id: str
) -> list[dict]:
    """Read existing entries in `.pipeline/judge_decisions.json` for this
    stage. Returns [] if the file doesn't exist or is empty.

    The canonical history is driver-owned, but older halted runs may contain
    the pre-scratch protocol's observed failure shape: a single valid
    JudgeDecision object instead of a top-level array. Treat that as one
    legacy decision for prompt context only; `_append_judge_decision` will
    normalize it when the next valid scratch decision is recorded."""
    p = _judge_decisions_path(paths)
    if not p.is_file():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(raw, dict):
        try:
            JudgeDecision.model_validate(raw)
        except Exception:
            return []
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [d for d in raw if isinstance(d, dict) and d.get("stage_id") == stage_id]


def _judge_decisions_path(paths: PipelinePaths) -> Path:
    return paths.pipeline_dir / "judge_decisions.json"


def _safe_judge_path_part(value: str) -> str:
    """Convert a stage or validator label into a stable filename segment."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return (safe or "decision")[:96]


def _reviewer_judge_validator_label(reviewer_stage_id: str) -> str:
    return f"reviewer_findings:{reviewer_stage_id}"


def _judge_decision_scratch_path(
    paths: PipelinePaths, *, stage_id: str, iteration: int, validator_label: str,
) -> Path:
    """Per-dispatch halt-judge output path.

    The judge writes one decision object here. The driver validates it and
    appends it to the canonical `.pipeline/judge_decisions.json` history.
    Keeping this path deterministic makes halted runs inspectable and lets
    the driver delete stale scratch before a retry/resume."""
    stage_part = _safe_judge_path_part(stage_id)
    label_part = _safe_judge_path_part(validator_label)
    return (
        paths.pipeline_dir
        / "judge_decision_parts"
        / f"{stage_part}__iter_{iteration}__{label_part}.json"
    )


def _judge_decision_scratch_relpath(
    *, stage_id: str, iteration: int, validator_label: str,
) -> str:
    """Test/fixture-friendly relative form of `_judge_decision_scratch_path`."""
    stage_part = _safe_judge_path_part(stage_id)
    label_part = _safe_judge_path_part(validator_label)
    return (
        ".pipeline/judge_decision_parts/"
        f"{stage_part}__iter_{iteration}__{label_part}.json"
    )


def _clear_judge_decision_scratch(path: Path) -> None:
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_judge_decision_object(
    decision_path: Path, *, stage_id: str, iteration: int, validator_label: str,
) -> dict:
    """Read and contract-check the judge's single scratch decision object."""
    artifact = decision_path.name
    if not decision_path.is_file():
        raise ValueError(
            f"halt-judge dispatched but did not write {artifact} "
            f"({stage_id}, iteration={iteration}, validator_label={validator_label})"
        )
    try:
        raw = json.loads(decision_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"halt-judge wrote {artifact} but it is not valid JSON "
            f"({stage_id}, iteration={iteration}): {e}"
        )
    if not isinstance(raw, dict):
        raise ValueError(
            f"halt-judge's {artifact} is not a top-level object "
            f"({stage_id}, iteration={iteration}); got {type(raw).__name__}"
        )
    err = _verify_agent_contract_fields(
        raw,
        expected={
            "stage_id": stage_id,
            "iteration": iteration,
            "validator_label": validator_label,
        },
        agent="r2c-halt-judge",
        artifact=artifact,
    )
    if err is not None:
        raise ValueError(err)
    return raw


def _judge_scratch_decision_valid(
    decision_path: Path, *, stage_id: str, iteration: int, validator_label: str,
):
    """Recovery-check fn for halt-judge drift.

    If the judge writes the scoped scratch decision correctly but also edits
    some out-of-scope file, `_dispatch_with_scope_check` may revert the drift
    and continue."""
    def _check(_state: PipelineState) -> bool:
        try:
            decision_dict = _read_judge_decision_object(
                decision_path,
                stage_id=stage_id,
                iteration=iteration,
                validator_label=validator_label,
            )
            JudgeDecision.model_validate(decision_dict)
        except Exception:
            return False
        return True
    return _check


def _load_judge_history_for_append(paths: PipelinePaths) -> list[dict]:
    """Load canonical judge history before appending a driver-owned entry.

    Supports one narrow legacy normalization: a single valid JudgeDecision
    object at the canonical path. That shape was produced by the old
    agent-owned append protocol; converting it to a one-element list lets
    halted runs recover without paper-specific repair logic."""
    p = _judge_decisions_path(paths)
    if not p.is_file():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"canonical {p.name} is not valid JSON and cannot be appended: {e}"
        )
    if isinstance(raw, list):
        return [d for d in raw if isinstance(d, dict)]
    if isinstance(raw, dict):
        try:
            JudgeDecision.model_validate(raw)
        except Exception as e:
            raise ValueError(
                f"canonical {p.name} is not a top-level array and is not "
                f"a valid legacy single judge decision: {e}"
            )
        log("judge", "legacy_decision_history_normalized",
            f"normalizing legacy single-object {p.name} to an array before append")
        return [raw]
    raise ValueError(
        f"canonical {p.name} is not a top-level array; got {type(raw).__name__}"
    )


def _append_judge_decision(
    paths: PipelinePaths, decision_dict: dict, *, scratch_path: Path,
) -> None:
    """Append a validated decision to driver-owned canonical history."""
    history = _load_judge_history_for_append(paths)
    history.append(decision_dict)
    decisions_path = _judge_decisions_path(paths)
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    _append_run_event(
        paths,
        "judge_decision_recorded",
        status="completed",
        summary=(
            f"recorded halt-judge decision for "
            f"{decision_dict.get('stage_id')} iteration {decision_dict.get('iteration')}"
        ),
        details={
            "stage_id": decision_dict.get("stage_id"),
            "iteration": decision_dict.get("iteration"),
            "validator_label": decision_dict.get("validator_label"),
            "classification": decision_dict.get("classification"),
            "action": decision_dict.get("action"),
            "scratch_path": scratch_path.relative_to(paths.run_dir).as_posix(),
            "canonical_path": decisions_path.relative_to(paths.run_dir).as_posix(),
        },
    )


def _prior_decisions_block(prior_decisions: list[dict]) -> str:
    """Render the last-3 prior-decisions block every judge prompt carries."""
    if prior_decisions:
        return json.dumps(prior_decisions[-3:], indent=2)  # last 3 only
    return "(no prior decisions in this stage)"


def _dispatch_judge_with_task(
    state: PipelineState, *, task_summary: str, decision_path: Path,
    stage_id: str, iteration: int, validator_label: str,
    write_first_retry: bool = False,
) -> DispatchResult:
    """Shared halt-judge dispatch skeleton (B-06 trio merge): scratch clear,
    prompt assembly, write-first retry preamble, and the scope-checked
    dispatch with the scratch-decision recovery check. The three named
    entry points below differ ONLY in how they build `task_summary`; they
    stay real module attributes because tests string-patch two of them and
    the corrective-redispatch source audit names all three.

    `write_first_retry`: set by `_judge_dispatch_decode_with_retry` on the
    second dispatch after a no-decision first attempt. Prepends the shared
    write-first preamble — an identical re-dispatch dies identically when
    the first turn burned its output budget on pre-write reasoning (the
    detr 2026-07-04 diagnostician evidence, generalized)."""
    agent = "r2c-halt-judge"
    _clear_judge_decision_scratch(decision_path)
    prompt = build_dispatch_prompt(
        task_summary=task_summary,
        paths=build_paths_block(state.paths),
        closing=REVIEWER_CLOSING,
        writeable_paths=WRITEABLE_PATHS[agent],
        think_anchor_output_path=str(decision_path),
    )
    if write_first_retry:
        prompt = build_write_first_retry_preamble(
            artifact_name=str(decision_path), artifact_kind="decision",
        ) + prompt
    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt=prompt, timeout_s=JUDGE_TIMEOUT_S,
        recovery_check_fn=_judge_scratch_decision_valid(
            decision_path,
            stage_id=stage_id,
            iteration=iteration,
            validator_label=validator_label,
        ),
        allow_corrective_redispatch=True,
    )


def _dispatch_halt_judge(
    state: PipelineState, *, stage_id: str, validator_label: str,
    stderr_tail: str, iteration: int, prior_decisions: list[dict],
    decision_path: Path, write_first_retry: bool = False,
) -> DispatchResult:
    """Dispatch the r2c-halt-judge with the failure context. The agent reads
    the validator + failing artifact + taxonomy/build-plan context and writes
    one scoped decision object. The driver appends that object to canonical
    history."""
    task_summary = JUDGE_TASK_TEMPLATE.format(
        stage_id=stage_id,
        iteration=iteration,
        validator_label=validator_label,
        decision_output_path=str(decision_path),
        stderr_tail=stderr_tail[-2000:],
        prior_decisions_block=_prior_decisions_block(prior_decisions),
    )
    return _dispatch_judge_with_task(
        state, task_summary=task_summary, decision_path=decision_path,
        stage_id=stage_id, iteration=iteration,
        validator_label=validator_label, write_first_retry=write_first_retry,
    )


@dataclass(frozen=True)
class JudgeOutcome:
    """Driver-facing decoded judge decision. The judge may decide
    `dispatch_fix` (in which case `target_agent` + `finding` + `dispatcher`
    are set) or `halt` (in which case `rationale` carries the halt reason).

    `dispatcher` is resolved by `_invoke_judge` from local_dispatchers /
    JUDGE_FIX_DISPATCHERS so callers can just call `outcome.dispatcher(...)`
    without re-doing the lookup."""
    action: Literal["dispatch_fix", "halt"]
    classification: str
    rationale: str
    target_agent: str | None
    finding: dict | None
    confidence: str
    dispatcher: callable | None = None


def _decode_judge_decision(
    state: PipelineState, *, decision_path: Path, stage_id: str,
    iteration: int, validator_label: str,
    local_dispatchers: dict[str, callable] | None = None,
) -> JudgeOutcome:
    """Read the just-written judge decision and turn it into a typed
    `JudgeOutcome`. Shared by `_invoke_judge` (validator-failure path) and
    `_invoke_judge_for_findings` (reviewer-findings path) — both dispatch
    the same agent against a scoped decision file, then need
    identical validation + dispatcher lookup.

    Raises ValueError if the judge didn't write a valid decision (file
    missing, malformed JSON, wrong top-level shape, contract-field mismatch,
    or schema validation failure). Returns a typed outcome on success;
    downgrades to `halt` if the judge picked a target_agent we have no
    dispatcher for or omitted the finding."""
    decision_dict = _read_judge_decision_object(
        decision_path,
        stage_id=stage_id,
        iteration=iteration,
        validator_label=validator_label,
    )
    try:
        decision = JudgeDecision.model_validate(decision_dict)
    except Exception as e:
        raise ValueError(
            f"halt-judge wrote an invalid decision: {e}"
        )
    _append_judge_decision(
        state.paths, decision.model_dump(), scratch_path=decision_path,
    )
    if decision.action == "dispatch_fix":
        # Lookup order: local override → global table → halt.
        dispatchers = {**JUDGE_FIX_DISPATCHERS, **(local_dispatchers or {})}
        if decision.target_agent not in dispatchers:
            log(stage_id, "judge_unknown_target_agent",
                f"judge picked target_agent={decision.target_agent!r} which is "
                f"not in JUDGE_FIX_DISPATCHERS (or local_dispatchers); "
                f"downgrading to halt")
            return JudgeOutcome(
                action="halt", classification=decision.classification,
                rationale=(
                    f"judge picked unknown target_agent "
                    f"{decision.target_agent!r}; original rationale: "
                    f"{decision.rationale}"
                ),
                target_agent=None, finding=None,
                confidence=decision.confidence,
            )
        if decision.finding is None:
            log(stage_id, "judge_missing_finding",
                "judge picked dispatch_fix but provided no finding; "
                "downgrading to halt")
            return JudgeOutcome(
                action="halt", classification=decision.classification,
                rationale=(
                    f"judge picked dispatch_fix but provided no finding; "
                    f"original rationale: {decision.rationale}"
                ),
                target_agent=None, finding=None,
                confidence=decision.confidence,
            )
        return JudgeOutcome(
            action="dispatch_fix", classification=decision.classification,
            rationale=decision.rationale, target_agent=decision.target_agent,
            finding=decision.finding.model_dump(),
            confidence=decision.confidence,
            dispatcher=dispatchers[decision.target_agent],
        )
    # action == "halt"
    return JudgeOutcome(
        action="halt", classification=decision.classification,
        rationale=decision.rationale, target_agent=None, finding=None,
        confidence=decision.confidence,
    )


def _invoke_judge(
    state: PipelineState, *, stage_id: str, validator_label: str,
    stderr_tail: str, iteration: int,
    local_dispatchers: dict[str, callable] | None = None,
) -> JudgeOutcome:
    """Dispatch the halt-judge for a tier-1+2 validator failure, then
    decode the decision. Thin orchestration wrapper around
    `_dispatch_halt_judge` + `_decode_judge_decision`.

    Raises OpencodeClientError on dispatch failure. Raises ValueError if
    the judge didn't write a valid decision. Caller is responsible for
    turning either into a stage halt with a clean error message.

    `local_dispatchers` (optional): per-call override map that takes
    precedence over `JUDGE_FIX_DISPATCHERS` when resolving the picked
    target_agent. Used by stage 3.c for the smoke-context-bound
    diagnostician dispatcher."""
    prior = _read_prior_judge_decisions(state.paths, stage_id)
    decision_path = _judge_decision_scratch_path(
        state.paths,
        stage_id=stage_id,
        iteration=iteration,
        validator_label=validator_label,
    )

    def _dispatch(write_first_retry: bool = False):
        _dispatch_halt_judge(
            state, stage_id=stage_id, validator_label=validator_label,
            stderr_tail=stderr_tail, iteration=iteration,
            prior_decisions=prior, decision_path=decision_path,
            write_first_retry=write_first_retry,
        )

    return _judge_dispatch_decode_with_retry(
        state, _dispatch,
        decision_path=decision_path, stage_id=stage_id,
        iteration=iteration, validator_label=validator_label,
        local_dispatchers=local_dispatchers,
    )


def _judge_dispatch_decode_with_retry(
    state: PipelineState, dispatch_fn, *, decision_path: Path,
    stage_id: str, iteration: int, validator_label: str,
    local_dispatchers: dict[str, callable] | None = None,
) -> JudgeOutcome:
    """Dispatch the judge and decode its decision, re-dispatching ONCE when
    no usable decision was written.

    The no-write Think-turn class (B-002) lands on the judge layer too: the
    2026-06-10 BADGE run died at stage 3a when a judge dispatch completed
    after 155 s without writing its decision file — while the underlying fix
    loop was working correctly. No-write turns are sometimes stochastic, but
    when the first turn burned its output budget on pre-write reasoning an
    identical re-dispatch dies identically (detr 2026-07-04, both
    diagnostician turns), so the retry dispatch carries the write-first
    preamble (`write_first_retry=True`); a second failure propagates and the
    caller halts with the judge-layer reason as before."""
    dispatch_fn()
    try:
        return _decode_judge_decision(
            state,
            decision_path=decision_path,
            stage_id=stage_id,
            iteration=iteration,
            validator_label=validator_label,
            local_dispatchers=local_dispatchers,
        )
    except ValueError as e:
        log(stage_id, "judge_no_decision_retry",
            f"halt-judge produced no usable decision (attempt 1: {e}); "
            f"re-dispatching once with the write-first retry preamble")
    dispatch_fn(write_first_retry=True)
    return _decode_judge_decision(
        state,
        decision_path=decision_path,
        stage_id=stage_id,
        iteration=iteration,
        validator_label=validator_label,
        local_dispatchers=local_dispatchers,
    )


def _dispatch_halt_judge_for_findings(
    state: PipelineState, *, stage_id: str, reviewer_stage_id: str,
    findings: list[dict], iteration: int, prior_decisions: list[dict],
    decision_path: Path, write_first_retry: bool = False,
) -> DispatchResult:
    """Dispatch the r2c-halt-judge with persisting reviewer findings (the
    tier-3 cap-exhaustion path). Same agent and scratch-output protocol as
    the validator-failure path; only the prompt framing differs.
    `write_first_retry` behaves as in `_dispatch_judge_with_task`."""
    validator_label = _reviewer_judge_validator_label(reviewer_stage_id)
    task_summary = JUDGE_REVIEWER_TASK_TEMPLATE.format(
        stage_id=stage_id,
        reviewer_stage_id=reviewer_stage_id,
        iteration=iteration,
        validator_label=validator_label,
        decision_output_path=str(decision_path),
        n_findings=len(findings),
        findings_block=json.dumps(findings, indent=2),
        prior_decisions_block=_prior_decisions_block(prior_decisions),
    )
    return _dispatch_judge_with_task(
        state, task_summary=task_summary, decision_path=decision_path,
        stage_id=stage_id, iteration=iteration,
        validator_label=validator_label, write_first_retry=write_first_retry,
    )


def _invoke_judge_for_findings(
    state: PipelineState, *, stage_id: str, reviewer_stage_id: str,
    findings: list[dict], iteration: int,
    local_dispatchers: dict[str, callable] | None = None,
) -> JudgeOutcome:
    """Dispatch the halt-judge against persisting tier-3 reviewer findings
    (called from `run_fix_loop` at cap-exhaustion when the stage-reviewer
    still reports critical findings after `cap` producer retries). Thin
    orchestration wrapper around `_dispatch_halt_judge_for_findings` +
    `_decode_judge_decision` — same outcome contract as `_invoke_judge`."""
    prior = _read_prior_judge_decisions(state.paths, stage_id)
    validator_label = _reviewer_judge_validator_label(reviewer_stage_id)
    decision_path = _judge_decision_scratch_path(
        state.paths,
        stage_id=stage_id,
        iteration=iteration,
        validator_label=validator_label,
    )

    def _dispatch(write_first_retry: bool = False):
        _dispatch_halt_judge_for_findings(
            state, stage_id=stage_id, reviewer_stage_id=reviewer_stage_id,
            findings=findings, iteration=iteration, prior_decisions=prior,
            decision_path=decision_path,
            write_first_retry=write_first_retry,
        )

    return _judge_dispatch_decode_with_retry(
        state, _dispatch,
        decision_path=decision_path, stage_id=stage_id,
        iteration=iteration, validator_label=validator_label,
        local_dispatchers=local_dispatchers,
    )


_SEMANTIC_ISSUE_VALIDATORS = frozenset(
    {"validate_arch_contract.py", "validate_arch_contract_runtime.py"}
)
_VALIDATOR_FAILURE_HEADER_RE = re.compile(r"^FAIL: \d+ .+error\(s\):$")


@dataclass(frozen=True)
class Stage2ScriptOutcome:
    """Backward-compatible Stage-2 result with trusted ownership metadata."""

    ok: bool
    error_tail: str
    returncode: int
    semantic_issues: tuple[SemanticIssue, ...] = ()
    unstructured_lines: tuple[str, ...] = ()
    protocol_error: str | None = None
    raw_output: str = ""

    def __iter__(self):
        # Existing Stage-2 callers intentionally keep their two-value API.
        yield self.ok
        yield self.error_tail

    @property
    def pipeline_issues(self) -> tuple[SemanticIssue, ...]:
        return tuple(
            issue for issue in self.semantic_issues
            if issue.owner == "pipeline"
        )

    @property
    def producer_issues(self) -> tuple[SemanticIssue, ...]:
        return tuple(
            issue for issue in self.semantic_issues
            if issue.owner == "producer"
        )

    @property
    def pipeline_only(self) -> bool:
        return bool(self.pipeline_issues) and not (
            self.producer_issues
            or self.unstructured_lines
        )

    def producer_error_text(self) -> str:
        lines = [
            _render_semantic_issue(issue) for issue in self.producer_issues
        ]
        lines.extend(self.unstructured_lines)
        if self.protocol_error:
            lines.append(
                "Validator ownership protocol error: " + self.protocol_error
            )
        if self.pipeline_issues:
            return "\n".join(lines) or (
                "Validator reported a pipeline-owned issue with no separate "
                "producer failure text"
            )
        return "\n".join(lines) or self.error_tail


_SEMANTIC_ISSUE_LABELS = {
    "incomplete_generated_contract": "Incomplete generated contract",
    "unsupported_validator_feature": "Pipeline validator coverage gap",
    "contract_code_disagreement": "Contract and generated code disagree",
    "bundle_contract_disagreement": "Bundle and contract disagree",
}


def _render_semantic_issue(issue: SemanticIssue) -> str:
    label = _SEMANTIC_ISSUE_LABELS.get(issue.code, "Semantic validation issue")
    rendered = (
        f"{label} ({issue.code}): {issue.message}; "
        f"roots={issue.roots!r}"
    )
    if issue.values:
        rendered += "; values=" + json.dumps(issue.values, sort_keys=True)
    return rendered


def _semantic_issue_line_payload(line: str) -> str | None:
    candidate = line.strip()
    if candidate.startswith("- "):
        candidate = candidate[2:].lstrip()
    if candidate.startswith(SEMANTIC_ISSUE_PREFIX):
        return candidate
    return None


def _stage2_script_outcome(
    *,
    validator: str,
    returncode: int,
    stdout: str,
    stderr: str,
) -> Stage2ScriptOutcome:
    """Classify full trusted-validator output before excerpting it."""
    raw_output = (stderr or "").strip() or (stdout or "").strip()
    issues: list[SemanticIssue] = []
    malformed: list[str] = []
    unstructured: list[str] = []
    trusted = validator in _SEMANTIC_ISSUE_VALIDATORS

    if returncode != 0 or trusted:
        for line in raw_output.splitlines():
            stripped = line.strip()
            if not stripped or _VALIDATOR_FAILURE_HEADER_RE.match(stripped):
                continue
            payload = _semantic_issue_line_payload(line) if trusted else None
            if payload is not None:
                issue = parse_semantic_issue(payload)
                if issue is None:
                    malformed.append(stripped)
                else:
                    issues.append(issue)
                continue
            if returncode != 0:
                unstructured.append(stripped)

    protocol_parts: list[str] = []
    if malformed:
        protocol_parts.append(
            f"{len(malformed)} malformed semantic issue line(s)"
        )
    if returncode == 3:
        if not issues:
            protocol_parts.append("reserved exit 3 carried no semantic issues")
        if any(issue.owner != "pipeline" for issue in issues):
            protocol_parts.append("reserved exit 3 carried a non-pipeline issue")
        if unstructured:
            protocol_parts.append("reserved exit 3 carried unstructured failure text")
    if returncode == 2:
        protocol_parts.append("validator reported a setup error on reserved exit 2")
    if returncode == 0 and issues:
        protocol_parts.append("exit 0 carried semantic failure issues")
    protocol_error = "; ".join(protocol_parts) or None
    error_tail = _stderr_excerpt(raw_output)
    ok = returncode == 0 and not issues and protocol_error is None
    return Stage2ScriptOutcome(
        ok=ok,
        error_tail=error_tail,
        returncode=returncode,
        semantic_issues=tuple(issues),
        unstructured_lines=tuple(unstructured),
        protocol_error=protocol_error,
        raw_output=raw_output,
    )


def _run_stage2_script(
    state: PipelineState, *, stage_id: str, args: list[str], timeout: int = 120
) -> Stage2ScriptOutcome:
    """Run a Stage-2 script that takes `--spec <spec> --run-dir <run_dir>`.
    The outcome still unpacks as ``(ok, error_tail)`` for legacy callers, and
    also preserves the return code plus trusted architecture-issue metadata."""
    paths = state.paths
    proc = run_script(
        stage_id,
        [*args, "--spec", str(paths.method_spec), "--run-dir", str(paths.run_dir)],
        timeout=timeout,
    )
    validator = Path(args[0]).name if args else "stage_script"
    outcome = _stage2_script_outcome(
        validator=validator,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )
    _append_validation_event(
        paths,
        stage_id=stage_id,
        validator=validator,
        ok=outcome.ok,
        stderr_tail=outcome.error_tail,
        details={
            "returncode": outcome.returncode,
            "semantic_issues": [
                issue.model_dump(exclude_none=True)
                for issue in outcome.semantic_issues
            ],
            "semantic_protocol_error": outcome.protocol_error,
        },
    )
    return outcome


def _record_pipeline_validation_issues(
    state: PipelineState,
    *,
    validator: str,
    outcome: Stage2ScriptOutcome,
    iteration: int,
    terminal: bool,
) -> None:
    issue_count = len(outcome.pipeline_issues)
    if issue_count:
        summary = (
            f"{validator} reported {issue_count} pipeline-owned "
            "architecture validation issue(s)"
        )
    elif outcome.returncode == 2:
        summary = f"{validator} reported a pipeline validator setup failure"
    else:
        summary = f"{validator} reported an invalid ownership protocol"
    _append_run_event(
        state.paths,
        "pipeline_validation_issue",
        stage_id="stage_2d",
        status="degraded" if terminal else "running",
        summary=summary,
        details={
            "validator": validator,
            "iteration": iteration,
            "returncode": outcome.returncode,
            "terminal": terminal,
            "issues": [
                issue.model_dump(exclude_none=True)
                for issue in outcome.pipeline_issues
            ],
            "protocol_error": outcome.protocol_error,
        },
    )


def _degrade_pipeline_validation_issue(
    state: PipelineState,
    *,
    validator: str,
    outcome: Stage2ScriptOutcome,
    iteration: int,
) -> StageResult:
    protocol_detail = outcome.protocol_error
    if outcome.pipeline_only and outcome.returncode == 1:
        protocol_detail = (
            "pipeline-only semantic issues used exit 1 instead of reserved "
            "exit 3; ownership remained pipeline-derived from validated codes"
        )
    rendered = [
        _render_semantic_issue(issue) for issue in outcome.pipeline_issues
    ]
    if protocol_detail:
        rendered.append(f"Validator protocol error: {protocol_detail}")
    if outcome.returncode == 2 and outcome.error_tail:
        rendered.append(f"Validator setup detail: {outcome.error_tail}")
    if not rendered:
        rendered.append(
            "Validator protocol error: the reserved ownership channel could "
            "not be validated"
        )
    roots = sorted(
        {
            root
            for issue in outcome.pipeline_issues
            for root in issue.roots
        }
    )
    _record_pipeline_validation_issues(
        state,
        validator=validator,
        outcome=outcome,
        iteration=iteration,
        terminal=True,
    )
    return degrade(
        state.paths,
        "stage_2d",
        reason="\n".join(rendered),
        what_failed=(
            "R2C's architecture validator could not certify a declared "
            "contract surface"
        ),
        where=(
            f"{validator}; contract/evidence roots: "
            + (", ".join(roots) if roots else "validator ownership protocol")
        ),
        what_to_do=(
            "This is an R2C validator coverage or routing limitation, not a "
            "request to rewrite the generated method. Preserve the package "
            "and review the technical roots below; add validator support "
            "before relying on this unchecked surface."
        ),
        notes="pipeline-owned architecture validation gap",
        state=state,
    )


def run_fix_loop(
    *,
    state: PipelineState,
    stage_id: str,
    fix_dispatch_fn,
    validator_label: str,
    reviewer_stage_id: str,
    validator_args: list[str] | None = None,
    validator_fn=None,
    cap: int = STAGE_2_RETRY_CAP,
    use_judge: bool = False,
) -> StageResult | None:
    """Generalized validator + stage-reviewer fix loop for Stage 2.b/2.c/3.a.

    Either `validator_args` (a Stage-2-style `--spec/--run-dir` CLI) OR
    `validator_fn` (a callable `(state) -> (ok, err_tail)`) is required.
    The Callable form lets Stage 3.a chain render+validate into a single
    validator step (since render failures and validator failures both mean
    "the producer's draft is broken").

    The caller is responsible for the initial (non-fix-mode) producer dispatch;
    this loop handles the validator-and-review cycle that follows, with up to
    `cap` fix-mode re-dispatches.

    Each iteration:
      1. Run validator. If it fails → fix-mode dispatch (judge-routed when
         `use_judge=True`, otherwise stderr-as-finding to fix_dispatch_fn),
         then continue to next iteration (which re-validates).
      2. If validator passes → dispatch stage-reviewer.
      3. If reviewer reports critical findings → group them by
         `target_agent` and dispatch each group to the agent that owns its
         files (resolved through JUDGE_FIX_DISPATCHERS, with short-form
         names like 'architecture-coder' normalized to 'r2c-architecture-coder').
         Findings missing target_agent fall back to fix_dispatch_fn (the
         stage's own producer); findings whose target_agent has no
         registered dispatcher trigger an immediate halt. Then continue to
         next iteration (which re-validates THEN re-reviews — the three-tier
         check).
      4. If validator passes AND reviewer has no critical findings → success.

    Cap-exhaustion behavior (degrade, not halt — the artifact exists):
      - Validator still failing at iter=cap → DEGRADE (log + continue).
      - Reviewer still has critical findings at iter=cap → invoke the
        halt-judge on the persisting findings (`_invoke_judge_for_findings`).
        Judge either degrades with a rationale, or routes ONE more producer
        dispatch + final validate+review; still failing after that → DEGRADE.

    Returns None on success; a **degraded** StageResult when a quality gate
    can't be satisfied within the cap (the producer's artifact exists, so the
    issue is logged to KNOWN_ISSUES.md and the pipeline continues — halt→degrade
    redesign 2026-05-27); a **halted** StageResult only for irrecoverable
    dispatch/infra errors (opencode failures, unregistered target_agent).

    `use_judge` (v1: stages 2.b, 2.c, 2.d) — when True, the per-iter
    validator-failure branch invokes the r2c-halt-judge to classify the
    failure and pick the fixer (with structured finding) instead of
    mechanically dispatching `fix_dispatch_fn` with stderr-as-finding.
    The judge may decide to halt instead of dispatching, in which case
    the loop returns immediately with its rationale. Reviewer-findings
    (tier 3) ALWAYS route by `target_agent` regardless of `use_judge`,
    since reviewers explicitly carry that field per finding."""
    if validator_fn is None and validator_args is None:
        raise ValueError("run_fix_loop requires either validator_fn or validator_args")
    if validator_fn is None:
        # Default: run a Stage-2-style script with the standard CLI.
        def validator_fn(s, _args=validator_args, _stage_id=stage_id):
            return _run_stage2_script(s, stage_id=_stage_id, args=_args)
    paths = state.paths
    review_path = paths.pipeline_dir / f"stage_review_{reviewer_stage_id}.json"
    no_write_retry_used = False
    for iteration in range(cap + 1):
        # --- Validator (tier 1+2: existence + mechanical) ---
        ok, err_tail = validator_fn(state)
        if not ok:
            if iteration == cap:
                if not _producer_required_contract_present(stage_id, paths):
                    missing_reason = _producer_missing_required_reason(stage_id, paths)
                    # Type-(T): cap exhausted and the producer is still missing
                    # at least one required output. Halt for a clean re-run
                    # rather than degrade over an incomplete stage contract.
                    return halt(
                        paths, stage_id,
                        reason=f"{validator_label} failed after cap={cap} fix-mode "
                               f"retries and {missing_reason} "
                               f"(re-run required). stderr: {err_tail[-500:]}",
                        halt_class="fix_loop_exhausted",
                        retry_count=iteration, state=state,
                    )
                return _degrade_producer(
                    paths, stage_id, state=state,
                    reason=f"{validator_label} failed after cap={cap} fix-mode "
                           f"retries. stderr: {err_tail[-500:]}",
                )
            # Pick the fixer + finding for this iteration. Judge-routed when
            # use_judge=True; otherwise legacy mechanical routing.
            if use_judge:
                try:
                    outcome = _invoke_judge(
                        state, stage_id=stage_id,
                        validator_label=validator_label,
                        stderr_tail=err_tail, iteration=iteration,
                    )
                except (OpencodeClientError, OutOfScopeWritesError, ValueError) as e:
                    return halt(
                        paths, stage_id,
                        reason=f"halt-judge invocation failed at {validator_label}: {e}",
                        halt_class=_judge_invocation_halt_class(e),
                        retry_count=iteration,
                        context={"stderr": err_tail}, state=state,
                    )
                if outcome.action == "halt":
                    if not _producer_required_contract_present(stage_id, paths):
                        missing_reason = _producer_missing_required_reason(stage_id, paths)
                        # Type-(T): the judge said halt AND the producer is
                        # missing a required output. Degrading here would log a
                        # false 'best-effort artifact exists' issue and poison
                        # downstream stages with an incomplete input contract.
                        log(stage_id, "judge_halt_missing_required_artifact",
                            f"iteration {iteration}: judge classified "
                            f"{outcome.classification!r} ({outcome.confidence}); "
                            f"{missing_reason} — halting for a clean re-run "
                            f"instead of degrading over a missing artifact")
                        return halt(
                            paths, stage_id,
                            reason=f"halt-judge halted ({outcome.classification}/"
                                   f"{outcome.confidence}) at {validator_label} and "
                                   f"{missing_reason} (re-run required): "
                                   f"{outcome.rationale}",
                            halt_class="judge_halt",
                            # No user_message: the catalog story leads, the
                            # rationale stays behind the collapsible (item 12
                            # rule 3).
                            retry_count=iteration, state=state,
                            context={
                                "stderr": err_tail,
                                "judge_classification": outcome.classification,
                                "judge_confidence": outcome.confidence,
                                "validator_label": validator_label,
                            },
                        )
                    log(stage_id, "judge_degrade",
                        f"iteration {iteration}: judge classified as "
                        f"{outcome.classification!r} ({outcome.confidence}); "
                        f"degrading (artifact exists) and continuing")
                    return _degrade_producer(
                        paths, stage_id, state=state,
                        reason=f"halt-judge decided to halt ({outcome.classification}/"
                               f"{outcome.confidence}) at {validator_label}: "
                               f"{outcome.rationale}",
                    )
                dispatcher = outcome.dispatcher
                fix_finding = outcome.finding
                log(stage_id, "judge_dispatch",
                    f"iteration {iteration}: judge routed to "
                    f"{outcome.target_agent} ({outcome.classification}, "
                    f"{outcome.confidence})")
            else:
                dispatcher = fix_dispatch_fn
                fix_finding = _stderr_to_finding(err_tail, validator_label)
                log(stage_id, "validator_fix",
                    f"iteration {iteration}: {validator_label} failed; "
                    f"re-dispatching producer")
            fix_snapshot = _snapshot_run_dir(paths.run_dir)
            try:
                dispatcher(state, [fix_finding])
            except (OpencodeClientError, OutOfScopeWritesError) as e:
                return halt(
                    paths, stage_id,
                    reason=f"fix-mode dispatch failed during {validator_label} retry: {e}",
                    halt_class=_dispatch_error_halt_class(e),
                    retry_count=iteration, state=state,
                )
            changed = _changed_paths_since_snapshot(paths.run_dir, fix_snapshot)
            if not changed and not no_write_retry_used:
                no_write_retry_used = True
                retry_finding = _no_write_fix_retry_finding(
                    fix_finding,
                    validator_label=validator_label,
                    stderr_tail=err_tail,
                )
                log(stage_id, "fix_no_write_retry",
                    f"iteration {iteration}: fix-mode dispatch wrote no "
                    f"allowed artifact; retrying {validator_label} fixer once")
                retry_snapshot = _snapshot_run_dir(paths.run_dir)
                try:
                    dispatcher(state, [retry_finding])
                except (OpencodeClientError, OutOfScopeWritesError) as e:
                    return halt(
                        paths, stage_id,
                        reason=f"fix-mode no-write retry failed during "
                               f"{validator_label}: {e}",
                        halt_class=_dispatch_error_halt_class(e),
                        retry_count=iteration, state=state,
                    )
                retry_changed = _changed_paths_since_snapshot(
                    paths.run_dir, retry_snapshot
                )
                if not retry_changed:
                    log(stage_id, "fix_no_write_retry_empty",
                        f"iteration {iteration}: no-write retry also wrote no "
                        "allowed artifact; continuing to validator for normal "
                        "judge/cap handling")
            continue

        # --- Stage-reviewer (tier 3: semantic) ---
        try:
            _dispatch_stage_reviewer(state, reviewer_stage_id)
        except (OpencodeClientError, OutOfScopeWritesError) as e:
            return halt(paths, stage_id,
                        reason=f"stage-reviewer ({reviewer_stage_id}) dispatch failed: {e}",
                        halt_class=_dispatch_error_halt_class(e),
                        retry_count=iteration, state=state)
        if not review_path.exists():
            # Missing-output retry — Think-class stop-early recovery.
            log(stage_id, "stage_reviewer_missing_output_retry",
                f"iteration {iteration}: initial reviewer dispatch produced no "
                f"{review_path.name}; retrying with stronger prompt")
            try:
                _dispatch_stage_reviewer_retry(state, reviewer_stage_id)
            except (OpencodeClientError, OutOfScopeWritesError) as e:
                return halt(paths, stage_id,
                            reason=f"stage-reviewer ({reviewer_stage_id}) retry dispatch failed: {e}",
                            halt_class=_dispatch_error_halt_class(e),
                            retry_count=iteration, state=state)
            if not review_path.exists():
                return _degrade_reviewer_unusable(paths, stage_id, state=state,
                            reason=f"stage-reviewer did not write {review_path.name} "
                                   f"(even after one missing-output retry); shipping "
                                   f"the producer artifact unreviewed")
        review, read_err = _read_review_json_or_err(
            review_path, kind=f"{reviewer_stage_id} review",
            expected_stage_id=reviewer_stage_id,
        )
        if read_err:
            # Unusable-envelope retry (R2): the producer artifact already passed
            # its validator, so an unusable reviewer envelope (missing stage_id
            # echo, wrapper key, malformed JSON) is a reviewer-output fragility —
            # retry once with the stronger prompt before shipping unreviewed,
            # mirroring the missing-output retry above. A clean review was being
            # degraded purely on envelope shape (audit 2026-06-25 R2).
            log(stage_id, "stage_reviewer_unusable_output_retry",
                f"iteration {iteration}: {review_path.name} unusable ({read_err}); "
                f"retrying with stronger prompt before degrading")
            try:
                _dispatch_stage_reviewer_retry(state, reviewer_stage_id)
            except (OpencodeClientError, OutOfScopeWritesError) as e:
                return halt(paths, stage_id,
                            reason=f"stage-reviewer ({reviewer_stage_id}) unusable-output "
                                   f"retry dispatch failed: {e}",
                            halt_class=_dispatch_error_halt_class(e),
                            retry_count=iteration, state=state)
            review, read_err = _read_review_json_or_err(
                review_path, kind=f"{reviewer_stage_id} review",
                expected_stage_id=reviewer_stage_id,
            )
            if read_err:
                return _degrade_reviewer_unusable(paths, stage_id, state=state,
                            reason=f"stage-reviewer output unusable: {read_err}; "
                                   f"shipping the producer artifact unreviewed "
                                   f"(even after one unusable-output retry)")
        findings = review.get("findings", []) or []
        crit = critical_findings(findings)
        if not crit:
            log(stage_id, "review_passed",
                f"iteration {iteration}: {len(findings)} findings, 0 critical")
            return None  # success

        if iteration == cap:
            # Cap-exhausted: invoke halt-judge on the persisting findings
            # (parallel to the validator-failure judge path). The judge
            # classifies and either halts with a clear rationale or picks
            # ONE more producer dispatch + final re-validate.
            judge_halt = _tier3_cap_exhausted_judge(
                state=state, stage_id=stage_id, paths=paths,
                reviewer_stage_id=reviewer_stage_id, validator_fn=validator_fn,
                validator_label=validator_label, review_path=review_path,
                findings=crit, iteration=iteration,
            )
            if judge_halt is not None:
                return judge_halt
            return None  # judge-routed post-cap fix succeeded

        # Per-iter (iteration < cap) tier-3 routing: group by target_agent,
        # dispatch each group to its declared agent. Reviewers may flag
        # cross-stage findings (e.g., stage_3a notebook reviewer flagging
        # method/training.py issues that only r2c-architecture-coder can
        # fix) — honor each finding's target_agent instead of blindly
        # routing everything to the stage's own producer.
        grouped = group_by_target_agent(crit)
        log(stage_id, "reviewer_fix",
            f"iteration {iteration}: {len(crit)} critical finding(s) across "
            f"{len(grouped)} target_agent group(s)")
        for raw_agent_name, group_findings in grouped.items():
            dispatcher, dispatch_label = _resolve_finding_dispatcher(
                raw_agent_name, fix_dispatch_fn,
            )
            if dispatcher is None:
                return halt(
                    paths, stage_id,
                    reason=(
                        f"reviewer flagged {len(group_findings)} finding(s) "
                        f"with target_agent={raw_agent_name!r} but no "
                        f"dispatcher is registered for that agent"
                    ),
                    halt_class="internal_contract_violation",
                    retry_count=iteration, findings=group_findings, state=state,
                )
            log(stage_id, "reviewer_fix_dispatch",
                f"iteration {iteration}: dispatching {dispatch_label} with "
                f"{len(group_findings)} finding(s)")
            try:
                dispatcher(state, group_findings)
            except (OpencodeClientError, OutOfScopeWritesError) as e:
                return halt(
                    paths, stage_id,
                    reason=(
                        f"fix-mode dispatch failed during reviewer retry "
                        f"({dispatch_label}): {e}"
                    ),
                    halt_class=_dispatch_error_halt_class(e),
                    retry_count=iteration, state=state,
                )
    return None  # unreachable


def _resolve_finding_dispatcher(
    raw_target_agent: str, fix_dispatch_fn,
) -> tuple[callable | None, str]:
    """Resolve a finding's `target_agent` to its dispatcher.

    Stage-reviewer outputs come in two `target_agent` conventions:
      - Full form: 'r2c-architecture-coder' (per schemas/judge_decision.py)
      - Short form: 'architecture-coder' (per schemas/review_report.py)

    JUDGE_FIX_DISPATCHERS is keyed by full form. Normalize short forms by
    adding the 'r2c-' prefix, then look up. Returns:
      - (dispatcher, label) on success — `label` is a human-readable agent name
      - (fix_dispatch_fn, 'stage producer (missing target_agent)') for
        findings missing `target_agent` entirely (the 'unknown' bucket from
        `group_by_target_agent`) — preserves backward compat
      - (None, label) when target_agent is set but no dispatcher exists —
        caller should halt with this label in the reason
    """
    # Missing target_agent → fall back to stage producer (legacy behavior
    # for reviewers that haven't been updated to set the field).
    if not raw_target_agent or raw_target_agent == "unknown":
        return fix_dispatch_fn, "stage producer (missing target_agent)"
    full_name = raw_target_agent if raw_target_agent.startswith("r2c-") \
        else f"r2c-{raw_target_agent}"
    if full_name in JUDGE_FIX_DISPATCHERS:
        return JUDGE_FIX_DISPATCHERS[full_name], full_name
    return None, raw_target_agent


def _tier3_cap_exhausted_judge(
    *,
    state: PipelineState, stage_id: str, paths: PipelinePaths,
    reviewer_stage_id: str, validator_fn, validator_label: str,
    review_path: Path, findings: list[dict], iteration: int,
) -> StageResult | None:
    """Cap-exhaustion path for tier-3 reviewer findings. Invokes the
    halt-judge on the persisting critical findings; the judge classifies
    and either halts with a rationale or picks ONE more producer dispatch
    + we do a final validate+review. Returns a *degraded* StageResult when the
    gate still can't be satisfied (artifact exists → log + continue), a *halted*
    StageResult only for dispatch/infra errors, or None if the post-cap judge
    dispatch succeeded and the stage is now clean."""
    try:
        outcome = _invoke_judge_for_findings(
            state, stage_id=stage_id,
            reviewer_stage_id=reviewer_stage_id,
            findings=findings, iteration=iteration,
        )
    except (OpencodeClientError, OutOfScopeWritesError, ValueError) as e:
        return halt(
            paths, stage_id,
            reason=(
                f"halt-judge invocation failed at {reviewer_stage_id} "
                f"cap-exhaustion: {e}"
            ),
            halt_class=_judge_invocation_halt_class(e),
            retry_count=iteration, findings=findings, state=state,
        )
    if outcome.action == "halt":
        if not _producer_required_contract_present(stage_id, paths):
            missing_reason = _producer_missing_required_reason(stage_id, paths)
            log(stage_id, "judge_halt_missing_required_artifact",
                f"iteration {iteration}: cap-exhausted; judge classified as "
                f"{outcome.classification!r} ({outcome.confidence}); "
                f"{missing_reason} — halting for a clean re-run")
            return halt(
                paths, stage_id,
                reason=f"halt-judge halted ({outcome.classification}/"
                       f"{outcome.confidence}) on persisting {reviewer_stage_id} "
                       f"findings and {missing_reason} (re-run required): "
                       f"{outcome.rationale}",
                halt_class="judge_halt",
                retry_count=iteration, findings=findings, state=state,
            )
        log(stage_id, "judge_degrade",
            f"iteration {iteration}: cap-exhausted; judge classified as "
            f"{outcome.classification!r} ({outcome.confidence}); "
            f"degrading (artifact exists) and continuing")
        return _degrade_producer(
            paths, stage_id, state=state,
            reason=f"halt-judge decided to halt ({outcome.classification}/"
                   f"{outcome.confidence}) on persisting {reviewer_stage_id} "
                   f"findings: {outcome.rationale}",
        )
    # action == "dispatch_fix" — one post-cap producer dispatch + final
    # validate + review. If still failing, halt for real.
    log(stage_id, "judge_dispatch",
        f"iteration {iteration}: cap-exhausted; judge routed to "
        f"{outcome.target_agent} ({outcome.classification}, "
        f"{outcome.confidence}); attempting one final fix dispatch")
    try:
        outcome.dispatcher(state, [outcome.finding])
    except (OpencodeClientError, OutOfScopeWritesError) as e:
        return halt(
            paths, stage_id,
            reason=f"post-cap judge-dispatched fix failed: {e}",
            halt_class=_dispatch_error_halt_class(e),
            retry_count=iteration, findings=findings, state=state,
        )
    ok2, err_tail2 = validator_fn(state)
    if not ok2:
        if not _producer_required_contract_present(stage_id, paths):
            missing_reason = _producer_missing_required_reason(stage_id, paths)
            # Type-(T): producer is still missing at least one required output
            # after the post-cap judge dispatch. Halt for a clean re-run rather
            # than degrade over an incomplete stage contract.
            return halt(
                paths, stage_id,
                reason=f"{validator_label} still failed after post-cap judge dispatch "
                       f"(judge routed to {outcome.target_agent}) and {missing_reason} "
                       f"(re-run required). stderr: {err_tail2[-500:]}",
                halt_class="fix_loop_exhausted",
                retry_count=iteration, state=state,
            )
        return _degrade_producer(
            paths, stage_id, state=state,
            reason=f"{validator_label} still failed after post-cap judge dispatch "
                   f"(judge routed to {outcome.target_agent}). stderr: {err_tail2[-500:]}",
        )
    try:
        _dispatch_stage_reviewer(state, reviewer_stage_id)
    except (OpencodeClientError, OutOfScopeWritesError) as e:
        return halt(
            paths, stage_id,
            reason=f"post-cap stage-reviewer dispatch failed: {e}",
            halt_class=_dispatch_error_halt_class(e),
            retry_count=iteration + 1, state=state,
        )
    if not review_path.exists():
        return _degrade_reviewer_unusable(
            paths, stage_id, state=state,
            reason=f"post-cap stage-reviewer did not write {review_path.name} "
                   f"after judge-routed dispatch; shipping the artifact unreviewed",
        )
    review2, read_err2 = _read_review_json_or_err(
        review_path, kind=f"{reviewer_stage_id} review",
        expected_stage_id=reviewer_stage_id,
    )
    if read_err2:
        return _degrade_reviewer_unusable(
            paths, stage_id, state=state,
            reason=f"post-cap stage-reviewer output unusable: {read_err2}; "
                   f"shipping the artifact unreviewed",
        )
    crit2 = critical_findings(review2.get("findings", []) or [])
    if not crit2:
        log(stage_id, "review_passed_post_cap",
            f"iteration {iteration}+1 (post-cap): reviewer clean after "
            f"judge dispatch to {outcome.target_agent}")
        return None
    if not _producer_required_contract_present(stage_id, paths):
        missing_reason = _producer_missing_required_reason(stage_id, paths)
        return halt(
            paths, stage_id,
            reason=f"{len(crit2)} critical finding(s) from {reviewer_stage_id} STILL "
                   f"persist after post-cap judge dispatch (judge routed to "
                   f"{outcome.target_agent}) and {missing_reason} "
                   f"(re-run required)",
            halt_class="fix_loop_exhausted",
            retry_count=iteration, findings=crit2, state=state,
        )
    return _degrade_producer(
        paths, stage_id, state=state,
        reason=f"{len(crit2)} critical finding(s) from {reviewer_stage_id} STILL "
               f"persist after post-cap judge dispatch (judge routed to "
               f"{outcome.target_agent}, classification={outcome.classification!r}, "
               f"rationale: {outcome.rationale})",
    )


def _bundle_axis_refusal(
    state: PipelineState, *, stage_id: str, bundle: dict, example_data: Path,
) -> dict | None:
    """Refuse a bundle whose time axis cannot support the paper's protocol
    (R2C-065). Returns a refusal or unresolved record, or None after a
    conclusive pass / when no typed temporal floor applies.

    A bundle short on the time axis is worse than no bundle at all: the
    notebook-binding check then FORCES the demo onto data that is
    arithmetically unable to run the method, which is how the 2026-08-05
    roll spent its whole smoke budget on a shape crash three stages
    downstream of a data defect. Refusing returns the run to the no-bundle
    path it took before this seam existed (synthetic demo arrays, no binding
    check), which is a worse demo and an honest one.

    The bundled files are deleted only after a conclusive INFEASIBLE verdict.
    A missing or incompatible cadence is recorded as unresolved and the bundle
    stays intact: lack of conversion evidence is not evidence of shortfall."""
    from bundle_axis_floor import (  # noqa: PLC0415
        ProtocolAxisUnresolved, check_bundle_axis, protocol_axis_floor,
    )

    paths = state.paths
    try:
        spec = json.loads(paths.method_spec.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        requirement = protocol_axis_floor(spec)
    except ProtocolAxisUnresolved as exc:
        unresolved = {
            "status": "unresolved",
            "reason": str(exc),
        }
        log(stage_id, "demo_dataset_axis_unresolved", str(exc))
        _append_run_event(
            paths, "demo_dataset_axis_unresolved",
            status="completed",
            summary=("The bundle was preserved because exact protocol-axis "
                     "feasibility could not be resolved"),
            details=unresolved,
        )
        return unresolved
    if requirement is None:
        return None
    verdict = check_bundle_axis(bundle, requirement)
    if verdict is None:
        return None
    if verdict.feasible is None:
        unresolved = {
            "status": "unresolved",
            "reason": verdict.message(),
            "protocol_floor_steps": requirement.floor,
            "realized_source_steps": verdict.realized_steps,
            "axis_table": verdict.table,
            "axis_column": verdict.column,
            "source_unit": verdict.source_unit,
            "source_granularity": verdict.source_granularity,
            "protocol_unit": requirement.unit,
            "protocol_granularity": requirement.granularity,
        }
        log(stage_id, "demo_dataset_axis_unresolved", verdict.message())
        _append_run_event(
            paths, "demo_dataset_axis_unresolved",
            status="completed",
            summary=("The bundle was preserved because its source cadence "
                     "cannot yet be joined to the protocol axis"),
            details=unresolved,
        )
        return unresolved
    if verdict.feasible:
        log(stage_id, "demo_dataset_axis_feasible", verdict.message())
        return None

    for entry in bundle.get("files") or []:
        name = entry.get("file")
        if name:
            (example_data / name).unlink(missing_ok=True)
    (example_data / "PROVENANCE.json").unlink(missing_ok=True)
    log(stage_id, "demo_dataset_axis_infeasible",
        f"{verdict.message()}; the bundle is REFUSED and the run continues "
        f"on synthetic demo data")
    refusal = {
        "status": "infeasible",
        "reason": verdict.message(),
        "protocol_floor_steps": requirement.floor,
        "realized_steps": verdict.realized_steps,
        "axis_table": verdict.table,
        "axis_column": verdict.column,
    }
    _append_run_event(
        paths, "demo_dataset_refused",
        status="completed",
        summary=("The paper's cited dataset was fetched and refused: its "
                 "demo-scale time axis cannot support the paper's protocol"),
        details=refusal,
    )
    return refusal


def _read_paradigm_id(paths: "PipelinePaths") -> str | None:
    """The run's taxonomy node id from its own spec, or None."""
    try:
        spec = json.loads(paths.method_spec.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(spec, dict):
        return None
    node = ((spec.get("comparison") or {}).get("classification") or {}).get("id")
    return str(node) if node else None


_OFFLINE_FALLBACK_SEED = 0


def _offline_fallback_attempt_trail(
    acquisition_record: dict, *, trigger_reason: str,
) -> list[dict]:
    """Normalize the public attempt trail and append the fallback decision.

    The synthetic manifest must retain why the higher-authority public tier
    did not supply its numbers. The appended row is deliberately nullable for
    host/URL: an acquisition exception or a paper with no recognized source
    must not acquire an invented citation merely to satisfy provenance shape.
    """
    attempts = [
        {
            "host": item.get("host"),
            "cited_url": item.get("cited_url"),
            "status": item.get("status"),
            "reason": item.get("reason"),
            "http_code": item.get("http_code"),
        }
        for item in (acquisition_record.get("attempts") or [])
        if isinstance(item, dict)
    ]
    acquired = acquisition_record.get("acquired")
    acquired = acquired if isinstance(acquired, dict) else {}
    attempts.append({
        "host": acquired.get("host"),
        "cited_url": acquired.get("cited_url"),
        "status": "fallback_trigger",
        "reason": trigger_reason,
        "http_code": None,
    })
    return attempts


def _materialize_family_owned_offline_fallback(
    state: PipelineState,
    *,
    stage_id: str,
    acquisition_record: dict,
    trigger_reason: str,
) -> None:
    """Attempt the resolved family's typed third-tier data contract.

    This is a deterministic pipeline-owned seam, never a producer repair. A
    missing family contract, unsupported protocol grammar, or write failure is
    recorded as an honest refusal and consumes no model turn.
    """
    from scripts.build_plan import load_build_plan  # noqa: PLC0415
    from scripts.taxonomy import run_overlay_dir  # noqa: PLC0415
    from time_series_offline_fallback import (  # noqa: PLC0415
        generate_time_series_offline_fallback,
    )

    paths = state.paths
    fallback_record: dict
    try:
        spec = json.loads(paths.method_spec.read_text(encoding="utf-8"))
        build_plan = load_build_plan(
            spec,
            paths.repo_root,
            provisional_packs_dir=run_overlay_dir(paths.run_dir),
        )
        resolved_plan = build_plan if isinstance(build_plan, dict) else {}
        if not isinstance(resolved_plan.get("offline_demo_data"), dict):
            fallback_record = {
                "status": "refused",
                "code": "offline_fallback_family_unsupported",
                "reason": (
                    "the resolved family does not declare a closed "
                    "build_plan.offline_demo_data contract"
                ),
                "provenance": None,
                "written_files": [],
            }
        else:
            outcome = generate_time_series_offline_fallback(
                spec,
                resolved_plan,
                paths.run_dir / "method" / "example_data",
                seed=_OFFLINE_FALLBACK_SEED,
                acquisition_attempts=_offline_fallback_attempt_trail(
                    acquisition_record, trigger_reason=trigger_reason,
                ),
                staging_parent=paths.pipeline_dir / "_dataset_staging",
            )
            fallback_record = outcome.as_dict()
    except Exception as exc:  # noqa: BLE001 — provisioning remains non-fatal
        fallback_record = {
            "status": "refused",
            "code": "offline_fallback_pipeline_error",
            "reason": (
                "pipeline could not resolve or invoke the family-owned "
                f"offline fallback: {exc}"
            ),
            "provenance": None,
            "written_files": [],
        }

    acquisition_record["offline_fallback"] = fallback_record
    if fallback_record.get("status") == "generated":
        acquisition_record["status"] = "fallback_generated"
        provenance = fallback_record.get("provenance")
        provenance = provenance if isinstance(provenance, dict) else {}
        summary = (
            "Family-owned synthetic forecasting fixture generated after the "
            "paper-cited public tier could not supply a usable bundle"
        )
        log(stage_id, "demo_dataset_fallback_generated", summary)
        _append_run_event(
            paths,
            "demo_dataset_fallback_generated",
            status="completed",
            summary=summary,
            details={
                "tier": provenance.get("tier"),
                "schema_id": provenance.get("schema_id"),
                "bundle_digest": provenance.get("bundle_digest"),
                "written_files": fallback_record.get("written_files") or [],
                "trigger_reason": trigger_reason,
            },
        )
        return

    summary = (
        "Family-owned offline demo fallback refused; the run continues "
        "without provisioned demo data"
    )
    acquisition_record["status"] = "fallback_refused"
    log(
        stage_id,
        "demo_dataset_fallback_refused",
        f"{summary}: {fallback_record.get('code')}: "
        f"{fallback_record.get('reason')}",
    )
    _append_run_event(
        paths,
        "demo_dataset_fallback_refused",
        status="completed",
        summary=summary,
        details={
            "code": fallback_record.get("code"),
            "reason": fallback_record.get("reason"),
            "trigger_reason": trigger_reason,
            "producer_retry_consumed": False,
        },
    )


def _acquire_demo_dataset_if_needed(
    state: PipelineState, *, stage_id: str,
) -> dict[str, object] | None:
    """Fetch the paper's own cited public dataset into the scaffolded
    package's example_data/, demo-scale, with provenance (R2C-052).

    Fires for a run whose package cannot get its own data. Two ways to be that
    run:

    - the GAP PATH, where a new-territory paper inherits the file-only
      provisional loader, and until this seam existed nothing anywhere could
      obtain the data its own feasibility contract approved (the pdfgnn
      contract said "public Kaggle retail data" over a loader that raises on
      an empty directory);
    - a COMMITTED family that declares `scaffold_hints.demo_data_source:
      needs_acquisition`, because its templates read local files only.

    The second arm exists because the first one used to be the whole rule
    (R2C-074). "Gap-path runs only" was a proxy for "the committed families own
    their data stories", true while every committed family's template
    downloaded its own data. Promoting a family out of the gap path with the
    gap path's file-only loader still in its templates moved it across this
    gate and silently switched its data off: the forecasting family was
    committed on 2026-08-05, and its next run fetched nothing and delivered a
    demo with no data. A family's data story is now something it says rather
    than something inferred from which side of a promotion it sits on.

    Ordinary source and fallback outcomes are NEVER fatal, and this is NEVER a
    substitute for the loader's local-file tier. A failed, offline,
    credentialed, or typed-axis-infeasible public attempt is recorded and then
    offered to the resolved family's closed offline fallback contract.
    Unsupported families/protocols refuse without a producer turn. The one
    safety exception is a changed or malformed interrupted-publication marker:
    this function returns a typed pipeline issue so Stage 2.a halts before a
    partial file can become loader input. `R2C_OFFLINE` short-circuits inside
    the acquisition module itself."""
    from dataset_acquisition import (  # noqa: PLC0415
        PublicBundleRecoveryError,
        acquire_for_paper,
        materialize_demo_bundle,
        recover_interrupted_public_bundle,
        refusal_summary,
    )
    from scripts.taxonomy import (  # noqa: PLC0415
        DEMO_DATA_NEEDS_ACQUISITION,
        demo_data_source,
        run_overlay_dir,
    )
    from time_series_offline_fallback import (  # noqa: PLC0415
        OfflineFallbackValidationError,
        recover_interrupted_time_series_fallback,
    )

    paths = state.paths
    gap_run = run_overlay_dir(paths.run_dir) is not None
    declared = None
    if not gap_run:
        try:
            declared = demo_data_source(_read_paradigm_id(paths))
        except Exception:  # noqa: BLE001 — acquisition is never fatal
            declared = None
    if not gap_run and declared != DEMO_DATA_NEEDS_ACQUISITION:
        return
    if not gap_run:
        log(stage_id, "demo_dataset_acquisition_declared",
            "this family's templates read local files only "
            "(scaffold_hints.demo_data_source: needs_acquisition), so the run "
            "attempts to bundle the paper's own cited public dataset")
    example_data = paths.run_dir / "method" / "example_data"
    provenance_path = paths.pipeline_dir / "dataset_acquisition.json"
    try:
        public_recovery = recover_interrupted_public_bundle(example_data)
    except PublicBundleRecoveryError as exc:
        record = {
            "status": "public_bundle_recovery_refused",
            "error": str(exc),
            "attempts": [],
            "public_bundle_recovery": {
                "status": "refused",
                "code": exc.code,
                "reason": str(exc),
                "producer_retry_consumed": False,
            },
        }
        try:
            provenance_path.write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8",
            )
        except OSError:
            pass
        _append_run_event(
            paths,
            "demo_dataset_public_recovery_refused",
            status="completed",
            summary=(
                "Interrupted public bundle could not be recovered safely; "
                "partial files were preserved and no fallback was attempted"
            ),
            details=record["public_bundle_recovery"],
        )
        return {
            "tier": "paper_cited_public",
            "code": exc.code,
            "reason": str(exc),
            "producer_retry_consumed": False,
        }
    if public_recovery is not None:
        summary = (
            "removed hash-matched files from an interrupted public bundle "
            "before retrying provisioning"
            if public_recovery == "removed_partial"
            else "accepted a complete hash-matched public bundle and removed "
            "its stale transaction marker"
        )
        log(stage_id, "demo_dataset_public_recovered", summary)
        _append_run_event(
            paths,
            "demo_dataset_public_recovered",
            status="completed",
            summary=summary,
            details={"outcome": public_recovery},
        )
    try:
        recovered = recover_interrupted_time_series_fallback(example_data)
    except OfflineFallbackValidationError as exc:
        record = {
            "status": "fallback_recovery_refused",
            "attempts": [],
            "offline_fallback": {
                "status": "refused",
                "code": exc.code,
                "reason": str(exc),
                "provenance": None,
                "written_files": [],
            },
        }
        try:
            provenance_path.write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8",
            )
        except OSError:
            pass
        _append_run_event(
            paths,
            "demo_dataset_fallback_refused",
            status="completed",
            summary=(
                "Interrupted offline fallback could not be recovered safely; "
                "Stage 2.a will halt before treating partial files as local data"
            ),
            details={
                "code": exc.code,
                "reason": str(exc),
                "producer_retry_consumed": False,
            },
        )
        return {
            "tier": "family_owned_synthetic",
            "code": exc.code,
            "reason": str(exc),
            "producer_retry_consumed": False,
        }
    if recovered:
        log(
            stage_id,
            "demo_dataset_fallback_recovered",
            "removed hash-matched files from an interrupted family-owned "
            "fallback publication before retrying provisioning",
        )
    has_data = any(
        p.is_file()
        and not p.name.startswith(".")
        and p.name != "PROVENANCE.json"
        and p.suffix.lower() in (".csv", ".tsv", ".json", ".npz")
        for p in example_data.glob("*")
    ) if example_data.is_dir() else False
    if has_data:
        return
    try:
        paper_text = paths.paper_md.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return

    staging = paths.pipeline_dir / "_dataset_staging"
    record: dict = {}
    fallback_reason: str | None = None
    try:
        report = acquire_for_paper(paper_text, staging)
        record = report.as_provenance()
        if report.acquired is not None:
            bundle = materialize_demo_bundle(
                report.acquired, example_data, report=report)
            record["bundle"] = {
                "files": bundle["files"],
                "skipped_members": bundle["skipped_members"],
                "subsample_rule": bundle["subsample_rule"],
            }
            if not bundle["files"]:
                fallback_reason = (
                    "the fetched public artifact contained no supported "
                    "demo-data files"
                )
                record["status"] = "not_acquired"
                record["bundle"]["refused"] = {
                    "status": "no_supported_files",
                    "reason": fallback_reason,
                }
                (example_data / "PROVENANCE.json").unlink(missing_ok=True)
                _append_run_event(
                    paths,
                    "demo_dataset_refused",
                    status="completed",
                    summary=(
                        "The paper-cited public artifact was fetched but "
                        "contained no supported demo-data files"
                    ),
                    details=record["bundle"]["refused"],
                )
            else:
                axis_assessment = _bundle_axis_refusal(
                    state, stage_id=stage_id, bundle=bundle,
                    example_data=example_data)
                if (
                    axis_assessment is not None
                    and axis_assessment.get("status") == "infeasible"
                ):
                    record["bundle"]["refused"] = axis_assessment
                    record["status"] = "not_acquired"
                    fallback_reason = axis_assessment.get("reason") or (
                        "the public bundle's typed protocol axis is infeasible"
                    )
                else:
                    if axis_assessment is not None:
                        record["bundle"]["axis_feasibility"] = axis_assessment
                    log(stage_id, "demo_dataset_acquired",
                        f"bundled {len(bundle['files'])} demo-scale table(s) from "
                        f"{report.acquired.source.cited_url} into "
                        f"method/example_data/")
                    _append_run_event(
                        paths, "demo_dataset_acquired",
                        status="completed",
                        summary=("Demo-scale extract of the paper's cited dataset "
                                 "bundled into the package"),
                        details=record,
                    )
        else:
            fallback_reason = refusal_summary(report)
            log(stage_id, "demo_dataset_unavailable",
                f"no cited dataset acquired ({fallback_reason}); "
                f"the family-owned offline fallback will be attempted")
    except Exception as e:  # noqa: BLE001 — this seam must never kill a run
        fallback_reason = f"public dataset acquisition failed: {e}"
        # If fetching succeeded and materialization or axis validation failed,
        # retain that higher-tier source and probe trail. Synthetic provenance
        # must explain the real attempt rather than replacing it with an empty
        # story at the exception boundary.
        record["status"] = "error"
        record["error"] = str(e)
        record.setdefault("attempts", [])
        log(stage_id, "demo_dataset_acquisition_error",
            f"dataset acquisition failed non-fatally: {e}; the family-owned "
            f"offline fallback will be attempted")

    if fallback_reason is not None:
        _materialize_family_owned_offline_fallback(
            state,
            stage_id=stage_id,
            acquisition_record=record,
            trigger_reason=fallback_reason,
        )
    try:
        provenance_path.write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8")
    except OSError as e:
        log(stage_id, "demo_dataset_acquisition_record_error",
            f"could not write dataset acquisition record non-fatally: {e}")
    finally:
        try:
            if staging.is_dir() and not any(staging.iterdir()):
                staging.rmdir()
        except OSError:
            pass


_STAGE_2A_FALLBACK_OUTPUTS = (
    run_layout.PACKAGE_README,
    "method/data.py",
    "method/example_data/README.md",
)


def _stage_2a_skip_outputs(paths: PipelinePaths) -> list[Path]:
    """Return the deterministic scaffolder outputs a resume must retain.

    Stage 2.a used to pin three generic paths here.  That made its complete
    sentinel unsafe across manifest migrations: when a paradigm gained a new
    fixed helper, an older completed run could still skip scaffolding even
    though the helper did not exist.  Derive the list from the same build-plan
    ownership contract used by the scaffolder and its validator.  Keep the
    historical three-file list only as a best-effort fallback when the spec or
    build plan cannot be loaded; resume must not fail before the stage gets a
    chance to run and report the underlying contract error.

    Manifest paths are required to be normalized run-relative paths.  A
    malformed or escaping path falls back instead of letting a skip check
    inspect an arbitrary filesystem location.
    """
    fallback = [paths.run_dir / rel for rel in _STAGE_2A_FALLBACK_OUTPUTS]
    try:
        from scripts.build_plan import load_build_plan  # noqa: PLC0415
        from scripts.taxonomy import run_overlay_dir  # noqa: PLC0415

        spec = json.loads(paths.method_spec.read_text(encoding="utf-8"))
        build_plan = load_build_plan(
            spec,
            paths.repo_root,
            provisional_packs_dir=run_overlay_dir(paths.run_dir),
        )
        files = ((build_plan or {}).get("package_manifest") or {}).get("files")
        if not isinstance(files, list):
            return fallback

        run_root = paths.run_dir.resolve()
        outputs: list[Path] = []
        seen: set[str] = set()
        for entry in files:
            if not isinstance(entry, dict):
                return fallback
            if entry.get("produced_by") != "package_scaffolder":
                continue
            rel = entry.get("path")
            if not isinstance(rel, str) or not rel:
                return fallback
            rel_path = Path(rel)
            if (
                rel_path.is_absolute()
                or rel_path.as_posix() != rel
                or any(part in {"", ".", ".."} for part in rel_path.parts)
            ):
                return fallback
            try:
                target = (paths.run_dir / rel_path).resolve().relative_to(run_root)
            except (OSError, ValueError):
                return fallback
            normalized = target.as_posix()
            if normalized in {"", "."} or normalized in seen:
                continue
            seen.add(normalized)
            outputs.append(paths.run_dir / target)
        return outputs or fallback
    except Exception:  # noqa: BLE001 — best-effort resume migration seam
        return fallback


def run_stage_2a(state: PipelineState) -> StageResult:
    """2.a — package_scaffolder. Pure-script. Halt on validator failure (no
    LLM to re-dispatch; the scaffolder is deterministic)."""
    stage_id = "stage_2a"
    paths = state.paths
    skipped = _skip_if_done(
        paths, stage_id, _stage_2a_skip_outputs(paths))
    if skipped is not None:
        return skipped
    proc = run_script(
        stage_id,
        ["scripts/scaffold_package.py", "--spec", str(paths.method_spec),
         "--run-dir", str(paths.run_dir)],
        timeout=60,
    )
    if proc.returncode != 0:
        return halt(paths, stage_id,
                    reason=f"scaffold_package.py exit {proc.returncode}",
                    halt_class=("gap_pack_rejected" if provisional_pack_ref(paths)
                                else "internal_contract_violation"),
                    context={"stderr": (proc.stderr or proc.stdout)[-2000:]},
                    state=state)
    ok, err = _run_stage2_script(
        state, stage_id=stage_id,
        args=["scripts/validate_scaffolder_output.py"],
    )
    if not ok:
        return halt(paths, stage_id,
                    reason="validate_scaffolder_output.py failed",
                    halt_class=("gap_pack_rejected" if provisional_pack_ref(paths)
                                else "internal_contract_violation"),
                    context={"stderr": err,
                             "note": "scaffolder is deterministic; no LLM to re-dispatch"},
                    state=state)
    provisioning_issue = _acquire_demo_dataset_if_needed(
        state, stage_id=stage_id,
    )
    if provisioning_issue is not None:
        return halt(
            paths,
            stage_id,
            reason=(
                "unsafe interrupted demo-data publication requires "
                "pipeline-owned recovery before later stages"
            ),
            halt_class="internal_contract_violation",
            context={"demo_data_provisioning": provisioning_issue},
            state=state,
        )
    return StageResult(status="completed", stage_id=stage_id,
                       notes="scaffolder + validator clean")


def _record_build_plan_signature_collisions(
    state: "PipelineState", stage_id: str,
) -> None:
    """Log + record in assumptions.md any same-name-different-signature
    collision the build plan resolved (maintainer decision 2026-08-18: the
    paper-declared signature wins; the choice is flagged, never silent).
    build_plan is a pure library, so the structured `signature_collisions`
    it carries is written to the researcher-facing record here, at the
    stage that first consumes the plan. Best-effort: a read failure never
    blocks the stage."""
    try:
        from scripts.build_plan import load_build_plan  # noqa: PLC0415
        from scripts.taxonomy import run_overlay_dir  # noqa: PLC0415

        spec = json.loads(
            state.paths.method_spec.read_text(encoding="utf-8"))
        plan = load_build_plan(
            spec, state.paths.repo_root,
            provisional_packs_dir=run_overlay_dir(state.paths.run_dir))
    except Exception:  # noqa: BLE001 - observability only
        return
    collisions = (plan or {}).get("signature_collisions") or []
    if not collisions:
        return
    detail = "; ".join(
        f"{c['method']}: manifest `{c['manifest_signature']}` vs paper "
        f"`{c['spec_signature']}` ({c['resolution']})"
        for c in collisions)
    log(stage_id, "signature_collision",
        f"{len(collisions)} spec/manifest method-signature collision(s): "
        f"{detail}")
    _append_assumption(
        state,
        aid=_next_assumption_id(state),
        title="Model-method signature collision resolved to the paper",
        detected=(
            "method_spec.json restates a model method the package manifest "
            f"already declares, with a different signature: {detail}"),
        action=(
            "The paper-declared signature from method_spec.json replaced "
            "the manifest default in the build plan (multi-class manifests "
            "are flagged without replacement, since spec methods carry no "
            "class attribution)."),
        reasoning=(
            "A paper that declares a richer interface (an "
            "embedding-returning forward, an extra keyword) means the "
            "manifest default is the incomplete one; silently dropping the "
            "spec-declared interface was the bev-distill F001/F002 class."),
        alternative=(
            "Keep the manifest default and treat the spec's restatement as "
            "advisory. If the generated architecture genuinely cannot "
            "satisfy the paper's signature, that is a stage-2b finding to "
            "route, not a reason to un-declare it."),
        override=(
            "Edit critical_requirements.required_model_methods in "
            "method_spec.json (or the family manifest) so the two agree, "
            "then re-run stage 2b."),
    )


def run_stage_2b(state: PipelineState) -> StageResult:
    """2.b — architecture-coder produces model.py + training.py +
    arch_contract.json. LLM producer with cap=3 fix loop (validator +
    stage-reviewer). Skip-check includes the contract: a surgical wipe
    that deletes ONLY arch_contract.json would otherwise leave the stage
    skipping with the contract permanently missing, halting 2.d's contract
    validator on every re-run."""
    stage_id = "stage_2b"
    paths = state.paths
    skipped = _skip_if_done(paths, stage_id, [
        paths.run_dir / "method" / "model.py",
        paths.run_dir / "method" / "training.py",
        paths.pipeline_dir / "stage_review_stage_2b_architecture.json",
        paths.pipeline_dir / "arch_contract.json",
    ])
    if skipped is not None:
        return skipped
    _record_build_plan_signature_collisions(state, stage_id)
    try:
        _dispatch_arch_coder(state)
    except (OpencodeClientError, OutOfScopeWritesError) as e:
        return halt(paths, stage_id,
                    reason=f"architecture-coder dispatch failed: {e}",
                    halt_class=_dispatch_error_halt_class(e),
                    state=state)
    loop_halt = run_fix_loop(
        state=state, stage_id=stage_id,
        fix_dispatch_fn=_dispatch_arch_coder_fix,
        validator_args=["scripts/validate_architecture_coder_output.py"],
        validator_label="validate_architecture_coder_output.py",
        reviewer_stage_id="stage_2b_architecture",
        use_judge=True,
    )
    if loop_halt is not None:
        return loop_halt
    return StageResult(status="completed", stage_id=stage_id,
                       notes="architecture-coder + validator + review clean")


def run_stage_2c(state: PipelineState) -> StageResult:
    """2.c — method-coder produces method.py. LLM producer with cap=3 fix
    loop. The bayesian sub-paradigm adds mc-sampling-uses-train-mode and
    scale-dependent-hyperparameter checks at the stage-reviewer step — those
    catch silent-degeneration failure modes that smoke gates can't see."""
    stage_id = "stage_2c"
    paths = state.paths
    skipped = _skip_if_done(paths, stage_id, [
        paths.run_dir / "method" / "method.py",
        paths.pipeline_dir / "stage_review_stage_2c_method.json",
    ])
    if skipped is not None:
        return skipped
    try:
        _dispatch_method_coder(state)
    except (OpencodeClientError, OutOfScopeWritesError) as e:
        return halt(paths, stage_id,
                    reason=f"method-coder dispatch failed: {e}",
                    halt_class=_dispatch_error_halt_class(e),
                    state=state)
    loop_halt = run_fix_loop(
        state=state, stage_id=stage_id,
        fix_dispatch_fn=_dispatch_method_coder_fix,
        validator_args=["scripts/validate_method_coder_output.py"],
        validator_label="validate_method_coder_output.py",
        reviewer_stage_id="stage_2c_method",
        use_judge=True,
    )
    if loop_halt is not None:
        return loop_halt
    return StageResult(status="completed", stage_id=stage_id,
                       notes="method-coder + validator + review clean")


def _ensure_package_requirements(state: "PipelineState", *, stage_id: str) -> tuple[bool, str]:
    """Install the generated package's own declared dependencies into the
    driver's environment before the first `from method import *`.

    The heavy method deps (torch and friends) are BY DESIGN not workspace
    requirements — but until 2026-07-07 nothing installed them before the
    stage 2.d import validation either; the notebook's `%pip install -r
    requirements.txt` only runs at the stage 3.c smoke, three stages after
    the first import. Every environment so far happened to carry torch
    already, which masked the ordering — the fresh-clone dry run failed
    here on the README's own example paper. This targets the SAME
    environment the smoke kernel's install cell writes to, so it moves
    the install earlier rather than adding a second one; when the deps
    are already present it is a fast no-op."""
    req = state.paths.run_dir / run_layout.REQUIREMENTS_TXT
    if not req.is_file():
        return False, f"generated requirements file is missing: {req}"
    log(stage_id, "package_requirements",
        f"pip install -r {req} (the generated package is imported for the "
        f"first time in this stage; heavy deps may take a few minutes on "
        f"a fresh environment)")
    # Item 30: route installs through a local wheelhouse when one is configured
    # (and skip the index entirely when the preflight found it unreachable).
    wheelhouse_args = _pip_wheelhouse_install_args(state.paths.repo_root)
    if wheelhouse_args:
        log(stage_id, "wheelhouse",
            f"installing through wheelhouse: pip {' '.join(wheelhouse_args)}")
    try:
        proc = run_script(
            stage_id,
            ["-m", "pip", "install", "--disable-pip-version-check",
             *wheelhouse_args, "-r", str(req)],
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        return False, "pip install of the package requirements timed out after 900s"
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout)[-2000:]
    return True, ""


# Wire-level pip failure markers (urllib3/requests retry phrases plus the
# driver's own install-timeout note). Checked BEFORE the resolution markers
# because an unreachable index also ends in "No matching distribution found
# (from versions: none)" after the retries — the wire evidence must win.
_PIP_NETWORK_MARKERS = (
    "connection broken",
    "newconnectionerror",
    "connecttimeouterror",
    "readtimeouterror",
    "read timed out",
    "timed out",
    "proxyerror",
    "sslerror",
    "certificate verify failed",
    "temporary failure in name resolution",
    "name or service not known",
    "network is unreachable",
    "connection refused",
    "connection reset",
)

# The resolver rejected the requirement set itself: a name no index carries,
# or version constraints that cannot be satisfied together.
_PIP_RESOLUTION_MARKERS = (
    "no matching distribution found",
    "could not find a version that satisfies the requirement",
    "resolutionimpossible",
    "conflicting dependencies",
)


def _pip_resolution_failure(pip_output: object) -> bool:
    """True when a failed `pip install -r requirements.txt` failed because
    pip's RESOLVER rejected the requirement set, not because the wire did.

    The distinction decides the stage-2.d halt class: a resolution failure
    means the pipeline shipped an uninstallable requirements.txt — our bug,
    not infrastructure (DomIndOnto 2026-07-21 roll 2: the finalizer's
    import scan let the local package name into requirements.txt and pip
    could not resolve it, yet the halt read as a server-connection
    problem). Network markers win over resolution markers, and anything
    unrecognized keeps the historical transport read."""
    blob = str(pip_output or "").lower()
    if any(marker in blob for marker in _PIP_NETWORK_MARKERS):
        return False
    return any(marker in blob for marker in _PIP_RESOLUTION_MARKERS)


# Producer-agent fix dispatchers by the manifest's `produced_by` value.
# Deliberately only the two LLM coders: a scaffolder-owned file (data.py) is
# deterministic output, so an unresolvable import there is pipeline-owned and
# must keep the internal-bug halt.
_DEPENDENCY_FIX_DISPATCHERS: dict[str, str] = {
    "architecture_coder": "arch",
    "method_coder": "method",
}


def _trace_pip_resolution_failure(
    state: "PipelineState", pip_error: str,
) -> list[dict]:
    """Trace pip-unresolvable requirement names back to producer imports.

    R2C-050. Returns one trace dict per failing requirement that a
    ROUTABLE producer's file imports ({requirement, files, producers});
    empty when nothing traces, which keeps the finalizer-artifact case on
    the internal-bug halt."""
    from finalize_package_init import (  # noqa: PLC0415
        trace_requirement_to_producers,
        unresolvable_requirement_names,
    )
    from scripts.build_plan import load_build_plan  # noqa: PLC0415
    from scripts.taxonomy import run_overlay_dir  # noqa: PLC0415

    paths = state.paths
    try:
        spec = json.loads(paths.method_spec.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    build_plan = load_build_plan(
        spec, paths.repo_root,
        provisional_packs_dir=run_overlay_dir(paths.run_dir))
    manifest_files = ((build_plan or {}).get("package_manifest") or {}).get("files")

    traces: list[dict] = []
    for requirement in unresolvable_requirement_names(pip_error):
        trace = trace_requirement_to_producers(
            requirement, paths.run_dir / "method", manifest_files)
        if trace is None:
            continue
        if not any(p in _DEPENDENCY_FIX_DISPATCHERS for p in trace["producers"]):
            continue
        traces.append(trace)
    return traces


def _route_uninstallable_dependency_to_producer(
    state: "PipelineState", *, stage_id: str, pip_error: str, iteration: int,
) -> "StageResult | bool | None":
    """Dispatch the producer(s) whose imports made requirements.txt
    unresolvable, with a structured fix finding naming the honest options.

    Returns True when at least one producer was dispatched (caller then
    re-finalizes and retries the install), None when nothing traces to a
    routable producer (caller falls through to the internal-bug halt), or a
    StageResult halt when the dispatch layer itself failed."""
    paths = state.paths
    traces = _trace_pip_resolution_failure(state, pip_error)
    if not traces:
        return None

    stderr_tail = (pip_error or "")[-1200:]
    dispatched = False
    for trace in traces:
        requirement = trace["requirement"]
        files = ", ".join(trace["files"])
        finding = {
            "id": f"DEP{iteration:03d}",
            "severity": "critical",
            "description": (
                f"pip cannot install `{requirement}`: no distribution "
                f"exists for this platform and Python version. The "
                f"requirement entered requirements.txt from YOUR import(s) "
                f"in {files} — the manifest is reconciled from the "
                f"package's imports, so the dependency choice is yours to "
                f"fix. pip said:\n{stderr_tail}"
            ),
            "proposed_fix": (
                f"Drop the `{requirement}` import and implement the "
                f"operation directly against the already-required stack "
                f"(torch / numpy), or switch to a library pip can install "
                f"on this platform. Do not leave a guarded "
                f"try/except-ImportError fallback that silently disables "
                f"the mechanism when the library is absent — if the code "
                f"path matters, it must not depend on an uninstallable "
                f"import; if it does not matter, delete it."
            ),
        }
        for producer in trace["producers"]:
            kind = _DEPENDENCY_FIX_DISPATCHERS.get(producer)
            if kind is None:
                continue
            log(stage_id, "dependency_fix_routed",
                f"iteration {iteration}: `{requirement}` (from {files}) "
                f"routed to {producer}")
            _append_run_event(
                paths, "uninstallable_dependency_routed",
                status="running",
                summary=(f"Unresolvable dependency `{requirement}` routed "
                         f"to {producer} as a fix finding"),
                details={"requirement": requirement,
                         "files": trace["files"],
                         "producer": producer,
                         "iteration": iteration},
            )
            try:
                if kind == "arch":
                    _dispatch_arch_coder_fix(state, [finding])
                else:
                    _dispatch_method_coder_fix(state, [finding])
            except (OpencodeClientError, OutOfScopeWritesError) as e:
                return halt(
                    paths, stage_id,
                    reason=("fix-mode dispatch failed while routing an "
                            f"uninstallable dependency to {producer}: {e}"),
                    halt_class=_dispatch_error_halt_class(e),
                    retry_count=iteration,
                    context={"requirement": requirement,
                             "stderr": stderr_tail},
                    state=state,
                )
            dispatched = True
    return True if dispatched else None


def run_stage_2d(state: PipelineState) -> StageResult:
    """2.d — init_finalizer assembles __init__.py + requirements.txt from
    the producers' outputs, then validates imports and the architecture
    contract.

    Two phases:
      Phase 1: finalize + import validation. The finalizer's failures are
        producer contract violations attributed in its own stderr, so they
        route through the halt judge with a fix dispatch to the owning
        producer (cap=STAGE_2_RETRY_CAP); halt on exhaustion or a judge
        halt decision. The post-finalize checks (requirements manifest,
        dependency readiness, import validation) stay deterministic halts.
      Phase 2: architecture-contract validation (structural + runtime dry-
        run). Producer-owned failures retain the bounded judge/fix loop.
        Trusted pipeline-owned semantic issues and validator setup/protocol
        failures bypass the judge and producer retry budget and terminate
        under this stage's existing degrade semantics. Both validators re-run
        after producer fixes so a fix that breaks one while satisfying the
        other surfaces immediately."""
    stage_id = "stage_2d"
    paths = state.paths
    # Clear stale halt artifact from a prior run. Stage 2.d's smart-skip
    # is the sole authority for whether to skip — DO NOT fall through to
    # _skip_if_done since it would override smart-skip's cache-miss verdict
    # when the sentinel + outputs are both present (which is exactly the
    # bev-distill 2026-05-22 case the smart-skip exists to catch).
    halt_path = paths.pipeline_dir / f"{stage_id}.halt"
    if halt_path.exists():
        log(stage_id, "clear_halt", f"removing stale {halt_path.name} from prior run")
        halt_path.unlink()
    # Smart skip: content-digest check against upstream method/*.py + spec +
    # arch_contract. Catches the bev-distill 2026-05-22 cascade where stage 2.c
    # regenerated method.py but stage 2.d's standard sentinel-check skipped
    # → stale __init__.py → stage 3.c ImportError.
    skipped = _skip_stage_2d_if_no_upstream_changes(paths)
    if skipped is not None:
        return skipped
    # Phase 1 entry — the deterministic init-finalizer, judge-routed. Its
    # failures are producer contract violations it attributes in its own
    # stderr (an R1 class-count violation names the architecture-coder; a
    # promised bridge symbol nobody defined names the method-coder, or is a
    # Stage 1 mis-declaration the judge can classify upstream), so they get
    # the same judge-routed fix loop as Phase 2 instead of a first-strike
    # hard halt. Two producer-fixable shapes hit the old hard halt in one
    # day (DomIndOnto's class count and detr's train_from_scratch promise,
    # 2026-07-20) — the second was one public alias away from a full
    # delivery. The terminal on exhaustion stays a HALT, not a degrade:
    # without the finalizer there is no package front door to ship.
    finalize_args = ["scripts/finalize_package_init.py", "--spec",
                     str(paths.method_spec), "--run-dir", str(paths.run_dir)]
    for iteration in range(STAGE_2_RETRY_CAP + 1):
        proc = run_script(stage_id, finalize_args, timeout=60)
        if proc.returncode == 0:
            break
        stderr_tail = (proc.stderr or proc.stdout)[-2000:]
        if iteration == STAGE_2_RETRY_CAP:
            return halt(paths, stage_id,
                        reason=(f"finalize_package_init.py exit "
                                f"{proc.returncode} after "
                                f"cap={STAGE_2_RETRY_CAP} judge-routed fixes"),
                        halt_class="internal_contract_violation",
                        retry_count=iteration,
                        context={"stderr": stderr_tail},
                        state=state)
        try:
            outcome = _invoke_judge(
                state, stage_id=stage_id,
                validator_label="finalize_package_init.py",
                stderr_tail=stderr_tail, iteration=iteration,
            )
        except (OpencodeClientError, OutOfScopeWritesError, ValueError) as e:
            return halt(
                paths, stage_id,
                reason=("halt-judge invocation failed at "
                        f"finalize_package_init.py: {e}"),
                halt_class=_judge_invocation_halt_class(e),
                retry_count=iteration,
                context={"stderr": stderr_tail}, state=state,
            )
        if outcome.action == "halt":
            return halt(
                paths, stage_id,
                reason=(f"halt-judge decided to halt ({outcome.classification}/"
                        f"{outcome.confidence}) at finalize_package_init.py: "
                        f"{outcome.rationale}"),
                halt_class="internal_contract_violation",
                retry_count=iteration,
                context={"stderr": stderr_tail}, state=state,
            )
        log(stage_id, "judge_dispatch",
            f"iteration {iteration}: judge routed finalize contract "
            f"violation to {outcome.target_agent} ({outcome.classification}, "
            f"{outcome.confidence})")
        try:
            outcome.dispatcher(state, [outcome.finding])
        except (OpencodeClientError, OutOfScopeWritesError) as e:
            return halt(
                paths, stage_id,
                reason=("fix-mode dispatch failed during "
                        f"finalize_package_init.py retry: {e}"),
                halt_class=_dispatch_error_halt_class(e),
                retry_count=iteration, state=state,
            )

    # The manifest is a declared output of the deterministic finalizer and is
    # also the researcher's recovery artifact on a restricted network. Never
    # make dependency readiness or transport decisions before it exists.
    requirements_path = paths.run_dir / run_layout.REQUIREMENTS_TXT
    if not requirements_path.is_file():
        return halt(
            paths,
            stage_id,
            reason="finalize_package_init.py did not create requirements.txt",
            halt_class="internal_contract_violation",
            context={"requirements_path": str(requirements_path)},
            notes="init_finalizer output contract violation",
            state=state,
        )

    local_requirements = dependency_readiness.check_local_requirements(
        requirements_path
    )
    if local_requirements.checker_error:
        return halt(
            paths,
            stage_id,
            reason="could not inspect locally installed package requirements",
            halt_class="internal_contract_violation",
            context={
                "requirements_path": str(requirements_path),
                "checker_error": local_requirements.checker_error,
                "unsatisfied_requirements": list(local_requirements.unsatisfied),
                "invalid_requirements": list(local_requirements.invalid),
            },
            notes="local dependency readiness check failed closed",
            user_message=(
                "R2C created the generated package's `requirements.txt`, but "
                "could not inspect the current Python environment without a "
                "package manager. This is an R2C runtime/setup problem, not a "
                "paper or generated-code failure. See the halt details, repair "
                "the R2C environment, and resume the same run."
            ),
            state=state,
        )
    if local_requirements.invalid:
        return halt(
            paths,
            stage_id,
            reason="generated requirements.txt contains unsupported entries",
            halt_class="internal_contract_violation",
            context={
                "requirements_path": str(requirements_path),
                "invalid_requirements": list(local_requirements.invalid),
            },
            notes="init_finalizer requirements contract violation",
            state=state,
        )

    if local_requirements.satisfied:
        log(
            stage_id,
            "package_requirements",
            "all generated requirements are already installed; continuing "
            "without an index probe or pip invocation",
        )
    else:
        # Only an actually unsatisfied environment needs a package source.
        # This ordering is what lets a researcher install while temporarily on
        # another network, return to a restricted network, and resume locally.
        preflight_reason = _preflight_pip_for_install(paths, stage_id=stage_id)
        if preflight_reason is not None:
            return halt(
                paths, stage_id,
                reason=preflight_reason,
                halt_class="transport_failure",
                context={
                    "index_url": os.environ.get("PIP_INDEX_URL", "").strip()
                    or "https://pypi.org/simple/",
                    "requirements_path": str(requirements_path),
                    "python_executable": sys.executable,
                    "unsatisfied_requirements": list(
                        local_requirements.problem_descriptions
                    ),
                },
                notes="local requirements unsatisfied and no package source reachable",
                user_message=_dependency_halt_user_message(
                    paths, local_requirements
                ),
                state=state,
            )
        ok, err = _ensure_package_requirements(state, stage_id=stage_id)
        # R2C-050: a RESOLUTION failure splits by requirement origin.
        # requirements.txt is reconciled FROM the producers' own imports, so
        # an unresolvable name traced to a producer import is that producer's
        # library choice — dispatch it a fix finding through the same bounded
        # loop phase 1 uses, then re-finalize (requirements.txt regenerates
        # from the changed imports) and retry the install. A failure that
        # traces to NO producer import keeps the internal-bug story, which is
        # correct for finalizer artifacts (the DomIndOnto 2026-07-21 local
        # package name leak). Transport failures stay transport failures.
        dependency_fix_iterations = 0
        while (not ok and _pip_resolution_failure(err)
               and dependency_fix_iterations < STAGE_2_RETRY_CAP):
            routed = _route_uninstallable_dependency_to_producer(
                state, stage_id=stage_id, pip_error=err,
                iteration=dependency_fix_iterations,
            )
            if routed is None:
                break  # untraceable — fall through to the internal-bug halt
            if isinstance(routed, StageResult):
                return routed  # dispatch-layer failure, already a halt
            dependency_fix_iterations += 1
            proc = run_script(stage_id, finalize_args, timeout=60)
            if proc.returncode != 0:
                return halt(
                    paths, stage_id,
                    reason=("finalize_package_init.py failed while "
                            "regenerating requirements after a dependency "
                            f"fix (exit {proc.returncode})"),
                    halt_class="internal_contract_violation",
                    retry_count=dependency_fix_iterations,
                    context={"stderr": (proc.stderr or proc.stdout)[-2000:]},
                    state=state,
                )
            ok, err = _ensure_package_requirements(state, stage_id=stage_id)
        if not ok:
            resolution_bug = _pip_resolution_failure(err)
            traced = (
                _trace_pip_resolution_failure(state, err)
                if resolution_bug else None
            )
            if resolution_bug and traced:
                # Producer-owned and the fix loop could not clear it: the
                # halt names the producer's choice, not an internal bug, so
                # the researcher-facing story is actionable.
                return halt(
                    paths, stage_id,
                    reason=("pip cannot install a producer-selected "
                            "dependency (no distribution for this platform) "
                            "after the bounded fix loop"),
                    halt_class="producer_output_invalid",
                    retry_count=dependency_fix_iterations,
                    context={
                        "stderr": err,
                        "requirements_path": str(requirements_path),
                        "python_executable": sys.executable,
                        "pip_failure_kind": "resolution",
                        "traced_requirements": traced,
                    },
                    state=state,
                )
            return halt(paths, stage_id,
                        reason=("pip rejected the generated requirements "
                                "(dependency resolution failed)"
                                if resolution_bug
                                else "package dependency install failed"),
                        halt_class=("internal_contract_violation"
                                    if resolution_bug
                                    else "transport_failure"),
                        retry_count=dependency_fix_iterations,
                        context={
                            "stderr": err,
                            "requirements_path": str(requirements_path),
                            "python_executable": sys.executable,
                            "pip_failure_kind": (
                                "resolution" if resolution_bug
                                else "transport_or_unknown"),
                        },
                        user_message=(None if resolution_bug else (
                            "The generated package's dependencies could not be "
                            f"installed from `{requirements_path}`. Connect to a "
                            "network or package source that works for this Python "
                            f"environment (`{sys.executable}`), install that file, "
                            "then resume the same run.")),
                        state=state)
    ok, err = _run_stage2_script(
        state, stage_id=stage_id,
        args=["scripts/validate_package_imports.py"],
    )
    if not ok:
        return halt(paths, stage_id,
                    reason="validate_package_imports.py failed",
                    halt_class="producer_output_invalid",
                    context={"stderr": err,
                             "note": "init_finalizer is deterministic; fix the upstream producer contract violation it surfaced"},
                    state=state)

    # Phase 2 — architecture-contract checks with fix-mode retry. Two-tier:
    #   - validate_arch_contract.py: structural + paradigm-completeness
    #   - validate_arch_contract_runtime.py: dry-run the package against
    #     the contract's declared shapes
    # Both attribute to architecture-coder (the contract + model.py/training.py
    # are all its outputs). Re-run BOTH validators each iteration so a fix that
    # passes one but breaks the other surfaces immediately rather than masking.
    def _run_arch_contract_validators(
        s: PipelineState,
    ) -> tuple[Stage2ScriptOutcome, str]:
        outcome1 = _run_stage2_script(
            s, stage_id=stage_id, args=["scripts/validate_arch_contract.py"],
        )
        if not outcome1.ok:
            return outcome1, "validate_arch_contract.py"
        outcome2 = _run_stage2_script(
            s, stage_id=stage_id,
            args=["scripts/validate_arch_contract_runtime.py"],
            timeout=180,  # subprocess imports torch
        )
        if not outcome2.ok:
            return outcome2, "validate_arch_contract_runtime.py"
        return outcome2, ""

    for iteration in range(STAGE_2_RETRY_CAP + 1):
        validator_outcome, validator_label = _run_arch_contract_validators(state)
        if validator_outcome.ok:
            break
        if (
            validator_outcome.protocol_error
            and validator_outcome.returncode in {0, 2, 3}
        ) or validator_outcome.pipeline_only:
            return _degrade_pipeline_validation_issue(
                state,
                validator=validator_label,
                outcome=validator_outcome,
                iteration=iteration,
            )

        pipeline_detail = ""
        if validator_outcome.pipeline_issues:
            _record_pipeline_validation_issues(
                state,
                validator=validator_label,
                outcome=validator_outcome,
                iteration=iteration,
                terminal=False,
            )
            pipeline_detail = "\nPipeline-owned issue(s) observed separately: " + "; ".join(
                _render_semantic_issue(issue)
                for issue in validator_outcome.pipeline_issues
            )

        # Pipeline-owned issues never enter the judge prompt. Mixed batches
        # spend a retry only on their real producer-owned or unstructured arm;
        # a subsequent pipeline-only result degrades directly above.
        # Always use the classified rendering. In particular, malformed wire
        # lines are represented by a neutral protocol finding; passing the raw
        # excerpt would let a forged owner field influence the LLM judge even
        # though the parser correctly rejected it.
        err_tail = validator_outcome.producer_error_text()
        if iteration == STAGE_2_RETRY_CAP:
            note = (
                "architecture-coder's arch_contract.json is malformed or "
                "paradigm-incomplete"
                if validator_label == "validate_arch_contract.py"
                else "architecture-coder's code does not accept the shapes "
                "declared in arch_contract.json — internal inconsistency "
                "between model.py/training.py and the contract"
            )
            # Type-(D) degrade: the package was assembled and imports cleanly
            # (Phase 1 passed); this is a contract/shape gate, and notebook
            # generation does not read arch_contract.json. Log + continue rather
            # than deny the researcher the whole package over a contract check.
            return degrade(
                paths, stage_id,
                reason=f"{validator_label} failed after cap={STAGE_2_RETRY_CAP} "
                       f"fix-mode retries. {note}. stderr: {err_tail[-500:]}"
                       f"{pipeline_detail}",
                what_failed="architecture-contract validation could not be satisfied",
                where=f"{validator_label} on .pipeline/arch_contract.json",
                what_to_do=(
                    "The method package was assembled and imports cleanly — this is a "
                    "contract/shape consistency check, not a syntax error. If the notebook "
                    "runs end-to-end, the package is likely usable. Review arch_contract.json "
                    "against method/model.py + method/training.py, or hand them to an AI "
                    "assistant. (This can also be an R2C validator/schema gap — see technical detail.)"
                ),
                state=state,
            )
        # Route through the halt-judge: read validator + artifact +
        # taxonomy/build-plan context and decide fix-vs-halt with a structured finding instead of
        # mechanically dispatching the architecture-coder on stderr.
        try:
            outcome = _invoke_judge(
                state, stage_id=stage_id,
                validator_label=validator_label,
                stderr_tail=err_tail, iteration=iteration,
            )
        except (OpencodeClientError, OutOfScopeWritesError, ValueError) as e:
            return halt(
                paths, stage_id,
                reason=f"halt-judge invocation failed at {validator_label}: {e}",
                halt_class=_judge_invocation_halt_class(e),
                retry_count=iteration,
                context={"stderr": err_tail}, state=state,
            )
        if outcome.action == "halt":
            log(stage_id, "judge_degrade",
                f"iteration {iteration}: judge classified as "
                f"{outcome.classification!r} ({outcome.confidence}); "
                f"degrading (contract gate, package imports) and continuing")
            # Type-(D) degrade — see cap-exhaustion branch above.
            return degrade(
                paths, stage_id,
                reason=f"halt-judge decided to halt ({outcome.classification}/"
                       f"{outcome.confidence}) at {validator_label}: "
                       f"{outcome.rationale}{pipeline_detail}",
                what_failed="architecture-contract validation could not be satisfied",
                where=f"{validator_label} on .pipeline/arch_contract.json",
                what_to_do=_judge_halt_user_message(outcome.rationale),
                state=state,
            )
        dispatcher = outcome.dispatcher
        log(stage_id, "judge_dispatch",
            f"iteration {iteration}: judge routed to "
            f"{outcome.target_agent} ({outcome.classification}, "
            f"{outcome.confidence})")
        try:
            dispatcher(state, [outcome.finding])
        except (OpencodeClientError, OutOfScopeWritesError) as e:
            return halt(
                paths, stage_id,
                reason=f"fix-mode dispatch failed during {validator_label} retry: {e}",
                halt_class=_dispatch_error_halt_class(e),
                retry_count=iteration, state=state,
            )

    # Per-element test generation (R2C-024): while the stage is still
    # open, so a test that exposes a real code defect routes to the
    # method-coder as a normal finding. Optional surface — only infra
    # errors halt; everything else drops-and-discloses.
    element_halt = _run_element_test_step(state)
    if element_halt is not None:
        return element_halt

    return StageResult(status="completed", stage_id=stage_id,
                       notes="init_finalizer + import validation + arch contract + dry-run clean + element tests")


# ---------------------------------------------------------------------------
# Per-element generated tests (R2C-024) — the stage 2.d test-generator
# dispatch surface. The deterministic core (eligibility plan, vacuity
# floor, runner, mutation check, coverage README) lives in
# scripts/element_tests.py; the dispatchers here put the agent behind it.
# ---------------------------------------------------------------------------

def _element_tests_written(state: PipelineState) -> bool:
    """Recovery check for test-generator drift: at least one generated
    test module exists and parses as Python."""
    import ast as _ast  # noqa: PLC0415
    tests_dir = state.paths.run_dir / "method" / "tests"
    for module in sorted(tests_dir.glob("test_*.py")) if tests_dir.is_dir() else []:
        try:
            _ast.parse(module.read_text(encoding="utf-8"))
            return True
        except (OSError, SyntaxError):
            continue
    return False


def _element_plan_sections(plan: dict[str, dict]) -> list[str]:
    """The Eligible elements prompt section: everything the generator may
    write a test for, with the element's own statement as the source of
    assertions. One section string, driver-rendered, deterministic."""
    lines = ["## Eligible elements (one test module per element)", ""]
    for element_id in sorted(plan):
        e = plan[element_id]
        lines.append(f"### `{element_id}` — {e['name'] or element_id}")
        lines.append(f"- Paper section: {e['section'] or '(not stated)'}")
        lines.append(f"- Implementing function: `{e['qualname']}` in "
                     f"`{e['file']}` (line {e['function_line']})")
        if e.get("source_text"):
            lines.append(f"- Verbatim statement: {e['source_text']}")
        if e.get("pseudocode"):
            lines.append(f"- Pseudocode: {e['pseudocode']}")
        if e.get("dependencies"):
            lines.append(f"- Depends on elements: "
                         f"{', '.join(e['dependencies'])}")
        lines.append("")
    return ["\n".join(lines)]


def _element_test_plan(state: PipelineState) -> dict[str, dict]:
    """The run's eligibility plan, rebuilt deterministically (the fix
    dispatcher cannot thread it through the judge table's signature)."""
    from element_tests import eligible_elements  # noqa: PLC0415
    return eligible_elements(state.paths.run_dir)


def _dispatch_test_generator(state: PipelineState,
                             plan: dict[str, dict]) -> DispatchResult:
    agent = "r2c-test-generator"
    prompt = build_dispatch_prompt(
        task_summary=STAGE_TASK_SUMMARIES["stage_2d_tests"],
        paths=build_paths_block(state.paths),
        writeable_paths=WRITEABLE_PATHS[agent],
        extra_sections=_element_plan_sections(plan),
        writing_discipline=True,
    )
    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt=prompt,
        timeout_s=TEST_GENERATOR_TIMEOUT_S,
        recovery_check_fn=_element_tests_written,
        allow_corrective_redispatch=True,
    )


def _dispatch_test_generator_fix(
    state: PipelineState, findings: list[dict],
) -> DispatchResult:
    agent = "r2c-test-generator"
    prompt = build_fix_mode_prompt(
        target_agent=agent,
        findings=findings,
        paths=build_paths_block(state.paths),
        writeable_paths=WRITEABLE_PATHS[agent],
        extra_sections=_element_plan_sections(_element_test_plan(state)),
    )
    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt=prompt,
        timeout_s=TEST_GENERATOR_TIMEOUT_S,
        recovery_check_fn=_element_tests_written,
        allow_corrective_redispatch=True,
    )


# Late registration, same pattern as notebook-generator: the judge may
# route a failing generated test to its producer (the one-regeneration
# cap is enforced by the element-test loop and the judge prompt).
JUDGE_FIX_DISPATCHERS["r2c-test-generator"] = _dispatch_test_generator_fix


def _element_test_validator_label(module_name: str) -> str:
    return f"element_tests:{module_name}"


def _dispatch_halt_judge_for_element_test(
    state: PipelineState, *, stage_id: str, module_name: str,
    entry: dict, failure_tail: str, iteration: int,
    prior_decisions: list[dict], decision_path: Path,
    write_first_retry: bool = False,
) -> DispatchResult:
    """Dispatch the halt-judge on one failing generated element test —
    the two-producer ownership decision (code vs test). Same agent and
    scratch-output protocol as the other judge paths; the prompt carries
    the paper element's verbatim text as the arbiter."""
    element_block = "\n".join(
        f"{field}: {entry.get(field)}"
        for field in ("name", "section", "source_text", "pseudocode")
        if entry.get(field))
    validator_label = _element_test_validator_label(module_name)
    task_summary = JUDGE_ELEMENT_TEST_TEMPLATE.format(
        stage_id=stage_id,
        iteration=iteration,
        validator_label=validator_label,
        decision_output_path=str(decision_path),
        element_id=entry["id"],
        element_block=element_block or "(element carries no statement)",
        test_module_path=f"method/tests/{module_name}",
        qualname=entry["qualname"],
        target_file=entry["file"],
        failure_tail=failure_tail,
        prior_decisions_block=_prior_decisions_block(prior_decisions),
    )
    return _dispatch_judge_with_task(
        state, task_summary=task_summary, decision_path=decision_path,
        stage_id=stage_id, iteration=iteration,
        validator_label=validator_label, write_first_retry=write_first_retry,
    )


def _invoke_judge_for_element_test(
    state: PipelineState, *, stage_id: str, module_name: str,
    entry: dict, failure_tail: str, iteration: int,
) -> JudgeOutcome:
    """Judge one failing element test. Same outcome contract as
    `_invoke_judge`; the target is either r2c-method-coder (the test
    correctly caught a code defect) or r2c-test-generator (the test
    misreads its element; one regeneration only)."""
    prior = _read_prior_judge_decisions(state.paths, stage_id)
    validator_label = _element_test_validator_label(module_name)
    decision_path = _judge_decision_scratch_path(
        state.paths, stage_id=stage_id, iteration=iteration,
        validator_label=validator_label,
    )

    def _dispatch(write_first_retry: bool = False):
        _dispatch_halt_judge_for_element_test(
            state, stage_id=stage_id, module_name=module_name,
            entry=entry, failure_tail=failure_tail, iteration=iteration,
            prior_decisions=prior, decision_path=decision_path,
            write_first_retry=write_first_retry,
        )

    return _judge_dispatch_decode_with_retry(
        state, _dispatch,
        decision_path=decision_path, stage_id=stage_id,
        iteration=iteration, validator_label=validator_label,
    )


def _floor_finding(module_name: str, problems: list[str]) -> dict:
    return {
        "id": f"ETF-{module_name}",
        "severity": "critical",
        "target_agent": "test-generator",
        "issue_type": "test_defect",
        "file": f"method/tests/{module_name}",
        "description": (
            f"method/tests/{module_name} failed the deterministic vacuity "
            "floor: " + "; ".join(problems)),
        "proposed_fix": (
            "Rewrite this one module against its element's verbatim "
            "statement so it clears every floor rule listed above. Do not "
            "touch other modules."),
    }


def _run_element_test_step(state: PipelineState) -> StageResult | None:
    """Stage 2.d per-element test generation (R2C-024). Optional surface
    by construction: unresolved tests are dropped and disclosed in the
    coverage README, never a stage halt — only infrastructure errors
    (dispatch transport, out-of-scope writes, judge invocation failure)
    return a halt StageResult. Runs while the stage is open so a test
    that exposes a real code defect routes to the method-coder as a
    normal finding."""
    stage_id = "stage_2d"
    paths = state.paths
    from build_anchor_join_table import (  # noqa: PLC0415
        JoinTableError, JoinTableSetupError)
    from element_tests import (  # noqa: PLC0415
        README_NAME, TESTS_REL_DIR, eligible_elements, mutation_check,
        render_coverage_readme, run_test_module, validate_test_module,
        validate_tests_dir)

    tests_dir = paths.run_dir / TESTS_REL_DIR
    try:
        plan = eligible_elements(paths.run_dir)
    except (JoinTableError, JoinTableSetupError) as e:
        log(stage_id, "element_tests_skipped",
            f"eligibility plan unavailable: {e}")
        return None
    if not plan:
        tests_dir.mkdir(parents=True, exist_ok=True)
        (tests_dir / README_NAME).write_text(
            render_coverage_readme({}, {}), encoding="utf-8")
        log(stage_id, "element_tests",
            "no implemented element maps to a function; disclosed in "
            "method/tests/README.md")
        return None

    # Per-element dispatch (2026-08-24, the maintainer's call after the Qwen 3.8
    # pacing evidence: ~7 min/element means a whole-plan batch cannot fit
    # any sane dispatch budget). One dispatch per eligible element, so a
    # timeout costs one element instead of the batch, and the stage
    # resumes per element: an element already claimed by a floor-passing
    # module is never re-dispatched, while a module below the floor is
    # deleted and its element regenerated fresh.
    pre_floor = validate_tests_dir(paths.run_dir, plan)
    already_claimed: set[str] = set()
    for module_name, result in sorted(pre_floor.items()):
        if result["problems"]:
            (tests_dir / module_name).unlink(missing_ok=True)
            log(stage_id, "element_tests_resume_regen",
                f"{module_name}: existing module below the vacuity floor; "
                f"deleted, its element will be regenerated")
        elif result["element_id"] in plan:
            already_claimed.add(result["element_id"])
            log(stage_id, "element_tests_resume_skip",
                f"{result['element_id']}: floor-passing module "
                f"{module_name} already exists; not re-dispatched")

    to_dispatch = [eid for eid in sorted(plan)
                   if eid not in already_claimed]
    log(stage_id, "element_tests",
        f"generating tests for {len(plan)} eligible element(s): "
        f"{len(to_dispatch)} dispatched one element per dispatch, "
        f"{len(already_claimed)} already covered")
    dispatch_failures: dict[str, BaseException] = {}
    for element_id in to_dispatch:
        try:
            _dispatch_test_generator(state, {element_id: plan[element_id]})
        except (OpencodeClientError, OutOfScopeWritesError) as e:
            # Isolation over abortion: one element's transport failure is
            # disclosed (the coverage README renders it not shipped by
            # construction) while the rest of the surface still lands.
            dispatch_failures[element_id] = e
            log(stage_id, "element_test_dispatch_failed",
                f"{element_id}: dispatch failed ({e}); element dropped, "
                f"continuing with the remaining elements")
    if to_dispatch and len(dispatch_failures) == len(to_dispatch):
        # Every dispatch failed: that is a systemic transport problem,
        # not per-element bad luck — halt exactly as the batch shape did.
        first_exc = dispatch_failures[to_dispatch[0]]
        return halt(paths, stage_id,
                    reason=(f"test-generator dispatch failed for all "
                            f"{len(to_dispatch)} elements; first error: "
                            f"{first_exc}"),
                    halt_class=_dispatch_error_halt_class(first_exc),
                    state=state)

    drops: dict[str, str] = {}

    # Vacuity floor, with the approved single corrective re-dispatch.
    floor = validate_tests_dir(paths.run_dir, plan)
    bad = {m: r["problems"] for m, r in floor.items() if r["problems"]}
    if bad:
        findings = [_floor_finding(m, p) for m, p in sorted(bad.items())]
        log(stage_id, "element_tests_floor",
            f"{len(bad)} module(s) below the vacuity floor; one "
            "corrective dispatch")
        try:
            _dispatch_test_generator_fix(state, findings)
        except (OpencodeClientError, OutOfScopeWritesError) as e:
            return halt(paths, stage_id,
                        reason=("test-generator floor fix dispatch "
                                f"failed: {e}"),
                        halt_class=_dispatch_error_halt_class(e),
                        state=state)
        floor = validate_tests_dir(paths.run_dir, plan)
        for module_name, result in sorted(floor.items()):
            if result["problems"]:
                (tests_dir / module_name).unlink(missing_ok=True)
                drops[module_name] = ("vacuity floor after one corrective "
                                      "dispatch: "
                                      + "; ".join(result["problems"]))
        floor = {m: r for m, r in floor.items() if not r["problems"]}

    # One module per element: keep the first module claiming an element.
    claimed: dict[str, str] = {}
    for module_name in sorted(floor):
        element_id = floor[module_name]["element_id"]
        if element_id in claimed:
            (tests_dir / module_name).unlink(missing_ok=True)
            drops[module_name] = (f"duplicate module for {element_id} "
                                  f"(kept {claimed[element_id]})")
        else:
            claimed[element_id] = module_name

    # Run each surviving module; judge failures for ownership (code vs
    # test, the element's verbatim text as arbiter).
    statuses: dict[str, dict] = {}
    for element_id, module_name in sorted(claimed.items()):
        module = tests_dir / module_name
        entry = plan[element_id]
        test_regens = 0
        for iteration in range(STAGE_2_RETRY_CAP + 1):
            result = run_test_module(paths.run_dir, module)
            if result["status"] == "passed":
                break
            if result["status"] != "failed":
                drops[module_name] = (f"{result['status']} at fixture "
                                      f"scale: {result['detail'][:200]}")
                break
            if iteration == STAGE_2_RETRY_CAP:
                drops[module_name] = (
                    f"still failing after cap={STAGE_2_RETRY_CAP} "
                    "judge-routed repairs")
                break
            try:
                outcome = _invoke_judge_for_element_test(
                    state, stage_id=stage_id, module_name=module_name,
                    entry=entry, failure_tail=result["detail"],
                    iteration=iteration)
            except (OpencodeClientError, OutOfScopeWritesError,
                    ValueError) as e:
                return halt(paths, stage_id,
                            reason=("halt-judge invocation failed at "
                                    f"element test {module_name}: {e}"),
                            halt_class=_judge_invocation_halt_class(e),
                            state=state)
            if outcome.action == "halt":
                drops[module_name] = (
                    f"judge declined ownership ({outcome.classification}/"
                    f"{outcome.confidence}): {outcome.rationale[:300]}")
                break
            if outcome.target_agent == "r2c-test-generator":
                if test_regens >= 1:
                    drops[module_name] = ("test defect persisted after "
                                          "its one regeneration")
                    break
                test_regens += 1
            log(stage_id, "element_tests_judge",
                f"{module_name} iteration {iteration}: routed to "
                f"{outcome.target_agent} ({outcome.classification})")
            try:
                outcome.dispatcher(state, [outcome.finding])
            except (OpencodeClientError, OutOfScopeWritesError) as e:
                return halt(paths, stage_id,
                            reason=("fix dispatch failed during element "
                                    f"test repair ({module_name}): {e}"),
                            halt_class=_dispatch_error_halt_class(e),
                            state=state)
            if outcome.target_agent == "r2c-test-generator":
                # A regenerated module must clear the floor again; a
                # regen that ducked below it is dropped, never re-judged.
                _, problems = validate_test_module(module, plan)
                if problems:
                    drops[module_name] = ("regenerated module fell below "
                                          "the vacuity floor: "
                                          + "; ".join(problems))
                    break
        if module_name in drops:
            (tests_dir / module_name).unlink(missing_ok=True)
            continue
        mutant = mutation_check(paths.run_dir, entry, module)
        key = {"caught": "tested", "survived": "weak"}.get(
            mutant["status"], "no_mutant")
        statuses[element_id] = {"status": key, "module": module_name,
                                "detail": mutant["detail"]}

    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / README_NAME).write_text(
        render_coverage_readme(plan, statuses), encoding="utf-8")
    results = {
        "schema_version": "1.0.0",
        "eligible": sorted(plan),
        "statuses": statuses,
        "dropped": drops,
    }
    (paths.pipeline_dir / "element_tests.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    tested = sum(1 for s in statuses.values() if s["status"] == "tested")
    log(stage_id, "element_tests",
        f"{tested} tested / {len(statuses)} shipped / {len(plan)} "
        f"eligible; {len(drops)} dropped (disclosed in README)")
    return None


# 900, was 300 until 2026-08-21: the first Qwen 3.8 roll produced FOUR
# consecutive alive-and-generating explainer timeouts (12-28 assistant
# parts inside each 300s window) — the model was mid-composition every
# time, and each abandonment paid the reasoning tax again from zero.
EXPLAINER_DISPATCH_TIMEOUT_S = 900

# Explanation math-sanity pass (item 3 part 2b). The claim extractor is a
# Think-tier dispatch, REUSED — no new agent surface. r2c-method-analyzer is the
# reuse target: its charter is extraction-to-structured-JSON that never judges
# truth ("you only analyze and write JSON"), which is exactly the extractor's
# contract (surface the claims verbatim, let the deterministic checker refute).
# The stage-reviewer was the other candidate but its whole ethos is judging +
# filing findings, and its hard "write ONLY the review file or the pipeline
# halts" rule fights a scoped write to a new scratch path.
MATH_EXTRACTOR_AGENT = "r2c-method-analyzer"
# 600, was 240 until 2026-07-14: healthy extractor turns in the 07-13
# batch ran to 238.8s (2 seconds under the old cap), and two runs
# false-tripped it — ADAM's trip opened the circuit breaker and killed an
# otherwise healthy run at stage 2b.
# 1200, was 600 until 2026-08-21: Qwen 3.8 recalibration (2x the review
# class, same reasoning-tax evidence as the explainer above).
MATH_EXTRACTOR_TIMEOUT_S = 1200
MATH_CLAIMS_SUBDIR = "math_claims"


def _math_formulas_context(eid: str, paper_map: dict) -> str:
    """The paper's own formula text for the element this entry explains, so the
    extractor can bind claims to real expressions. Item 3b put the paper's
    verbatim LaTeX in `source_text`; the other fields add binding context."""
    for e in paper_map.get("elements", []):
        if e.get("id") == eid:
            parts = []
            for f in ("name", "source_text", "pseudocode", "description"):
                v = str(e.get(f, "")).strip()
                if v:
                    parts.append(f"{f}: {v}")
            return "\n".join(parts)
    return ""


def _dispatch_math_extractor(state: PipelineState, eid: str, entry: dict,
                             paper_map: dict) -> str | None:
    """Best-effort Think-tier claim extraction for ONE explanation entry.

    Reuses MATH_EXTRACTOR_AGENT (Think tier) with a fully scoped extraction
    prompt and a scratch output path outside its usual writeable set, so a stray
    write to method_spec.json is caught as an out-of-scope violation and this
    entry simply yields no claims. Every failure mode — dispatch error, a
    no-write turn, an unreadable scratch file — degrades to `None` ("no claims"),
    never a halt (stage 1x never halts by design). Returns the raw scratch text
    for `parse_extractor_response`, or None."""
    from explanation_math_sanity import build_extractor_prompt  # noqa: PLC0415
    scratch_dir = state.paths.pipeline_dir / MATH_CLAIMS_SUBDIR
    scratch_dir.mkdir(parents=True, exist_ok=True)
    out_path = scratch_dir / f"{eid}.json"
    out_path.unlink(missing_ok=True)
    writeable = [f".pipeline/{MATH_CLAIMS_SUBDIR}/*.json"]
    prompt = build_extractor_prompt(
        eid, entry, _math_formulas_context(eid, paper_map), str(out_path))
    try:
        _dispatch_with_scope_check(
            state=state, agent=MATH_EXTRACTOR_AGENT, prompt=prompt,
            timeout_s=MATH_EXTRACTOR_TIMEOUT_S,
            writeable_paths_override=writeable)
    except (OpencodeClientError, OutOfScopeWritesError) as e:
        log("stage_1x", "math_extractor_dispatch_failed",
            f"{eid}: claim-extractor dispatch failed ({e}); "
            f"no math-sanity claims for this entry")
        return None
    if not out_path.is_file():
        log("stage_1x", "math_extractor_no_write",
            f"{eid}: claim-extractor turn completed without writing its file; "
            f"no math-sanity claims for this entry")
        return None
    try:
        return out_path.read_text(encoding="utf-8")
    except OSError:
        return None


def _emit_math_sanity_events(state: PipelineState, math_findings: list[dict],
                             meta: dict) -> None:
    """Emit the registered timeline events for a math-sanity pass: one
    `math_claim_refuted` (problem) per refuted claim, one `math_claim_dropped`
    (decision) per entry whose extracted claims hit the anti-hallucination
    floor. not_checkable rides in entry metadata, not a per-claim event."""
    for f in math_findings:
        _append_run_event(
            state.paths, "math_claim_refuted", stage_id="stage_1x",
            status="refuted", summary=(f.get("message") or "")[:200],
            details={"element_id": f.get("element_id"),
                     "counterexample": f.get("counterexample")})
    for eid, counts in (meta.get("entries") or {}).items():
        dropped = counts.get("dropped") or 0
        if dropped:
            _append_run_event(
                state.paths, "math_claim_dropped", stage_id="stage_1x",
                summary=(f"{eid}: {dropped} extracted claim(s) dropped by the "
                         f"quote-anchoring floor (not a byte-substring of the "
                         f"entry, or an unknown formula label)"),
                details={"element_id": eid, "dropped": dropped})


def _run_math_sanity(state: PipelineState, sidecar: dict, paper_map: dict,
                     check_findings: list[dict],
                     cache: dict) -> tuple[list[dict], dict]:
    """Extract + refute mechanism claims for the entries check_explanations did
    NOT already reject (a rejected entry is being requeued anyway). Best-effort,
    never halts. Returns (findings, metadata): refuted claims become
    check_explanations-shaped rejections (so a refutation requeues the entry and
    the counterexample rides the retry prompt); consistent/not_checkable never
    block.

    `cache` maps eid -> (entry_prose, raw_extraction) and is reused across merge
    attempts so an UNCHANGED entry is never re-extracted — a retried entry gets
    fresh prose and is re-extracted, a clean entry is not."""
    explanations = sidecar.get("explanations")
    if not isinstance(explanations, dict):
        return [], {"entries": {}, "totals": {}}
    from explanation_math_sanity import math_sanity_findings  # noqa: PLC0415
    excluded = {f.get("element_id") for f in check_findings if f.get("element_id")}
    for f in check_findings:
        excluded.update(f.get("element_ids") or [])
    spec_labels = {e.get("id") for e in paper_map.get("elements", [])
                   if e.get("id")}
    extractions: dict[str, object] = {}
    for eid, entry in explanations.items():
        if eid in excluded or not isinstance(entry, dict):
            continue
        prose = "\n".join(str(entry.get(k, ""))
                          for k in ("what", "why_novel", "intuition"))
        cached = cache.get(eid)
        if cached is not None and cached[0] == prose:
            raw = cached[1]
        else:
            raw = _dispatch_math_extractor(state, eid, entry, paper_map)
            cache[eid] = (prose, raw)
        if raw is not None:
            extractions[eid] = raw
    if not extractions:
        return [], {"entries": {}, "totals": {}}
    math_findings, meta = math_sanity_findings(
        sidecar, extractions, spec_labels=spec_labels)
    _emit_math_sanity_events(state, math_findings, meta)
    return math_findings, meta


def run_stage_1x(state: PipelineState) -> StageResult:
    """1.x — the METHOD.md explanation layer (slice 2.1 wiring).

    Best-effort BY DESIGN, never a halt: this deliverable is the fallback
    researchers get when downstream stages fail, so it cannot itself block
    the pipeline. Every failure degrades to the deterministic generator's
    explicit PENDING markers, and a degraded result clears the sentinel so
    a resume re-attempts only what's missing: landed part files are never
    re-dispatched, and parts whose entries the validator rejected are
    deleted so the resume re-dispatches exactly those.

    Placed right after stage 1 (paper_map is its only required input) so
    the explanation exists before code generation starts; the parameter
    provenance table renders "not derived" until stage 2x and is refreshed
    at finalization.
    """
    stage_id = "stage_1x"
    paths = state.paths
    sidecar_path = paths.pipeline_dir / "method_explanations.json"
    method_md = paths.run_dir / "METHOD.md"
    skipped = _skip_if_done(paths, stage_id, [method_md])
    if skipped is not None:
        return skipped

    from explainer_dispatch import (augment_prompt_with_rejections,  # noqa: PLC0415
                                    build_dispatches, merge_sidecar_parts,
                                    reconcile_landed_parts)
    from validate_method_explanations import \
        check_explanations  # noqa: PLC0415

    issues: list[str] = []
    paper_map = json.loads(paths.paper_map.read_text(encoding="utf-8"))
    paper_file = paths.pipeline_dir / "paper.md"
    paper_text = paper_file.read_text(encoding="utf-8") \
        if paper_file.is_file() else ""
    dispatches = build_dispatches(
        paper_map, paper_text, paper_path=str(paper_file),
        output_dir=str(paths.pipeline_dir))

    explained_count = 0
    explanations_arg: list[str] = []
    math_cache: dict = {}          # eid -> (entry_prose, raw_extraction)
    math_totals: dict = {}         # last pass's refuted/consistent/not_checkable/dropped
    retry_notes: dict[str, list[str]] = {}   # eid -> rejection messages

    def _dispatch_missing_parts(label: str) -> list[str]:
        dispatch_issues: list[str] = []
        for d in dispatches:
            part = Path(d["output_path"])
            if part.is_file():
                continue
            notes = [n for eid in d["element_ids"]
                     for n in retry_notes.get(eid, [])]
            try:
                _dispatch_with_scope_check(
                    state=state, agent="r2c-method-explainer",
                    prompt=augment_prompt_with_rejections(d["prompt"], notes),
                    timeout_s=EXPLAINER_DISPATCH_TIMEOUT_S)
            except (OpencodeClientError, OutOfScopeWritesError) as e:
                dispatch_issues.append(
                    f"{label} part {d['part']}: dispatch failed: {e}")
                continue
            if not part.is_file():
                dispatch_issues.append(
                    f"{label} part {d['part']}: turn completed without "
                    f"writing its file")
        return dispatch_issues

    # A resume after a chunk-size change finds part files whose numbers
    # mean a different equation set; reconcile before treating any landed
    # file as done, or the stale parts collide with fresh ones at merge.
    stale_parts = reconcile_landed_parts(dispatches)
    if stale_parts:
        log(stage_id, "explainer_parts_reconciled",
            "deleted landed part(s) from an older dispatch layout: "
            + ", ".join(p.name for p in stale_parts))

    terminal_dispatch_issues = _dispatch_missing_parts("initial")

    for merge_attempt in range(2):
        existing = [Path(d["output_path"]) for d in dispatches
                    if Path(d["output_path"]).is_file()]
        missing_ids = sorted({
            eid for d in dispatches
            if not Path(d["output_path"]).is_file()
            for eid in d["element_ids"]})
        sidecar = None
        skipped_parts: list[Path] = []
        if existing:
            try:
                sidecar, skipped_parts = merge_sidecar_parts(existing)
            except ValueError as e:
                # Disjoint-chunk invariant violated (a build_dispatches logic
                # bug) — rare; degrade rather than ship a half-merged sidecar.
                # Ordinary unparseable/missing parts are retried below.
                issues.append(f"part merge failed: {e}")
                break

        if sidecar is None:
            if merge_attempt == 0 and missing_ids:
                terminal_dispatch_issues = _dispatch_missing_parts("retry")
                continue
            issues.extend(terminal_dispatch_issues)
            if missing_ids:
                issues.append(
                    "missing explanation parts after retry (render as "
                    f"pending): {', '.join(missing_ids)}")
            break

        findings = check_explanations(sidecar, paper_map, paper_text or None)
        # Explanation math-sanity pass (item 3): extract each not-yet-rejected
        # entry's mechanism claims (Think tier, best-effort) and refute them
        # numerically. Runs BEFORE `rejected` is computed so a refuted claim
        # requeues its entry (merge_attempt 0), the counterexample rides the
        # retry prompt (via retry_notes below), and cap exhaustion renders
        # the existing pending marker. not_checkable/consistent never block.
        math_findings, math_meta = _run_math_sanity(
            state, sidecar, paper_map, findings, math_cache)
        findings.extend(math_findings)
        math_totals = math_meta.get("totals") or {}
        # Entries to re-do = validator-rejected element-ids PLUS every
        # element-id whose part was unparseable/missing. One bad part never
        # wipes the rest; good explanations stay landed.
        rejected = sorted({f["element_id"] for f in findings
                           if f.get("element_id")})
        skipped_set = {str(p) for p in skipped_parts}
        unparsed_ids = sorted({
            eid for d in dispatches
            if str(d["output_path"]) in skipped_set
            for eid in d["element_ids"]})
        requeue = set(rejected) | set(unparsed_ids) | set(missing_ids)
        if requeue and merge_attempt == 0:
            # The retry must know WHY the entry was rejected: carry each
            # requeued entry's finding messages (counterexample included)
            # into the re-dispatch prompt instead of blind re-rolling.
            retry_notes.clear()
            for f in findings:
                message = f.get("message")
                if not message:
                    continue
                note = str(message)
                # Item 23 part 3: a fabricated-quote finding may carry the
                # paper's best-matching passage; riding it on the retry
                # note removes the paper re-read. Advisory by wording.
                candidate = f.get("candidate_passage")
                if candidate:
                    note += (
                        " Candidate paper passage (verbatim from paper.md; "
                        "use ONLY if this is the passage you meant, and "
                        f"quote it byte-for-byte): \"{candidate}\"")
                ids = [f["element_id"]] if f.get("element_id") \
                    else list(f.get("element_ids") or [])
                for eid in ids:
                    if eid in requeue:
                        retry_notes.setdefault(eid, []).append(note)
            for d in dispatches:
                if requeue & set(d["element_ids"]):
                    Path(d["output_path"]).unlink(missing_ok=True)
            log(stage_id, "explainer_part_retry",
                "retrying rejected/missing METHOD.md explanation part(s): "
                + ", ".join(sorted(requeue))[:300])
            terminal_dispatch_issues = _dispatch_missing_parts("retry")
            continue

        for eid in rejected:
            sidecar["explanations"].pop(eid, None)
        if rejected:
            issues.append(f"validator rejected entries after retry "
                          f"(render as pending): {', '.join(rejected)}")
        if unparsed_ids:
            issues.append(f"unparseable parts skipped after retry "
                          f"(render as pending): {', '.join(unparsed_ids)}")
        if missing_ids:
            issues.append(f"missing explanation parts after retry "
                          f"(render as pending): {', '.join(missing_ids)}")
        if requeue:
            for d in dispatches:
                if requeue & set(d["element_ids"]):
                    Path(d["output_path"]).unlink(missing_ok=True)
            issues.extend(terminal_dispatch_issues)
        explained_count = len(sidecar["explanations"])
        # Disclosure only (never blocks): the count of mechanism claims the
        # math-sanity pass could not machine-check rides in the sidecar so the
        # REPORT.md issues table can surface "N ... were not machine-checkable".
        if math_totals:
            sidecar["math_sanity"] = math_totals
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        explanations_arg = ["--explanations", str(sidecar_path)]
        break

    proc = run_script(
        stage_id,
        ["scripts/generate_method_md.py", "--run-dir", str(paths.run_dir),
         *explanations_arg],
        timeout=60,
    )
    if proc.returncode != 0:
        issues.append(f"generate_method_md.py exit {proc.returncode}: "
                      f"{(proc.stderr or proc.stdout)[-300:]}")

    if not method_md.is_file():
        return StageResult(
            status="degraded", stage_id=stage_id,
            notes="; ".join(issues) or "generator produced no METHOD.md")
    pending_count = len(re.findall(
        r"\(explanation pending\b",
        method_md.read_text(encoding="utf-8"),
        flags=re.IGNORECASE,
    ))
    if pending_count:
        issues.append(
            f"METHOD.md contains {pending_count} pending explanation marker(s)"
        )
    if issues:
        log(stage_id, "explanations_degraded", "; ".join(issues)[:400])
        return StageResult(status="degraded", stage_id=stage_id,
                           notes="; ".join(issues)[:800],
                           paths_written=[method_md])
    return StageResult(
        status="completed", stage_id=stage_id,
        notes=f"METHOD.md generated with {explained_count} explanations",
        paths_written=[method_md, sidecar_path])


def _params_overclaim_only(err_tail: str) -> bool:
    """True iff every validate_params_output.py error is a US-3 `source=paper`
    over-claim — the relabel-eligible class.

    Such over-claims (a value/quote labelled paper-sourced that isn't findable
    in the paper) are correctable by flipping `source` to `system_inferred` via
    the stage-reviewer + `relabel_param_source` auto-resolver, so they should
    NOT terminate the stage at the first deterministic gate. Every other error
    (schema, missing params, US-1 range, US-2 locator/convention, US-3b
    fabricated-claim-regardless-of-source) is not a simple relabel and still
    halts. `US-3:` matches only the source=paper arm — `US-3b:` is excluded by
    the trailing colon."""
    bullets = [ln.strip()[2:].strip()
               for ln in err_tail.splitlines() if ln.strip().startswith("- ")]
    if not bullets:
        return False
    return all(b.startswith("provenance probe US-3:") for b in bullets)


_US3_BULLET_PREFIX = "provenance probe US-3:"


def _params_overclaim_bullets(err_tail: str) -> list[tuple[str, str]]:
    """(param_name, verbatim_bullet) per US-3 source=paper over-claim bullet.

    Bullet shape (validate_params_output.py):
    `- provenance probe US-3: <param>: <param>=<value> is source=paper but
    no rendering ... appears in the paper text`. Order preserved, duplicate
    param names collapsed to the first bullet."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for ln in err_tail.splitlines():
        s = ln.strip()
        if not s.startswith("- "):
            continue
        bullet = s[2:].strip()
        if not bullet.startswith(_US3_BULLET_PREFIX):
            continue
        name = bullet[len(_US3_BULLET_PREFIX):].strip().split(":", 1)[0].strip()
        if name and name not in seen:
            seen.add(name)
            out.append((name, bullet))
    return out


def _post_resolve_halt_reason(
    err_tail: str, applied_locations: set[str],
    apply_failed_params: frozenset[str] | set[str] = frozenset(),
    needs_user_params: frozenset[str] | set[str] = frozenset(),
) -> str:
    """Truthful halt reason for a failed post-auto-resolve re-validation.

    "Internal inconsistency between applier and validator" is only true when
    a param a resolution CLAIMED to fix still fails. The pdwa 2026-07-07
    halt was the other case: the still-failing param was a deferred
    over-claim nobody ever proposed a resolution for (the reviewer flagged
    a different param), and the misleading reason misdirected the diagnosis.
    Pick by whether any still-failing param is among the applied locations;
    a param whose re-asked resolution was adopted but REVERTED on apply
    (`apply_failed_params`) gets its own truthful message pointing at the
    recorded failure.

    `needs_user_params` — params whose re-ask answer was an explicit
    `needs_user` (the reviewer deliberately declined to relabel). When
    EVERY still-failing param is one of these, the halt is a deliberate
    escalation to the researcher, not a producer defect; the caller
    reclassifies to the needs_user_input halt class and this reason names
    the decision being asked (queue item: reclassify deferred-overclaim
    re-ask halts on the next touch of this path)."""
    failing = [name for name, _ in _params_overclaim_bullets(err_tail)]
    if failing and all(name in needs_user_params for name in failing):
        return ("the stage reviewer examined the deferred provenance "
                f"over-claim(s) on {', '.join(failing)} (value(s) labeled "
                "as paper-stated without paper evidence) and explicitly "
                "answered needs_user: deciding whether each value is truly "
                "paper-stated, or should be relabeled as derived or a "
                "default, needs your judgment. The validator's evidence "
                "per param is in the technical detail; relabel or confirm "
                "the source, then resume")
    apply_failed = [name for name in failing if name in apply_failed_params]
    if apply_failed:
        return ("validate_params_output.py still fails on deferred US-3 "
                f"over-claim(s) ({', '.join(apply_failed)}) whose re-asked "
                "resolution failed to apply (edit reverted; assumptions.md "
                "records the apply failure)")
    applied_params = {loc.rsplit(".", 1)[-1]
                      for loc in applied_locations if loc}
    if failing and not any(name in applied_params for name in failing):
        return ("validate_params_output.py still fails on deferred US-3 "
                "over-claim(s) no resolution ever addressed "
                f"({', '.join(failing)}); the stage-reviewer flagged "
                "different param(s) and the targeted re-ask did not clear it")
    return ("validate_params_output.py failed after auto-resolve "
            "(internal inconsistency between applier and validator)")


def _synthesize_deferred_overclaim_findings(
    err_tail: str, applied_locations: set[str],
) -> list[dict]:
    """Reviewer-shaped findings for deferred US-3 over-claims that survived
    the auto-resolve pass with NO resolution addressing them (the pdwa
    2026-07-07 seam: the reviewer flagged a different param, so the deferred
    over-claim never became a finding and the one-shot re-ask never saw it).

    Only params NOT among the applied locations are synthesized — a param a
    resolution claimed to fix but which still fails is a genuine
    applier/validator inconsistency and must halt, not re-ask. The
    description carries the validator's own bullet verbatim so the reviewer
    resolves against the deterministic evidence, not a paraphrase."""
    applied_params = {loc.rsplit(".", 1)[-1]
                      for loc in applied_locations if loc}
    return [
        {
            "id": f"DEF-{name}",
            "check_id": "paper_source_values_match_paper",
            "severity": "important",
            "target_agent": "parameter-deriver",
            "issue_type": "provenance_reasoning_inaccurate",
            "file": ".pipeline/params.json",
            "location": f"params.{name}",
            "description": bullet,
        }
        for name, bullet in _params_overclaim_bullets(err_tail)
        if name not in applied_params
    ]


def _apply_budget_raise(
    state: "PipelineState", *, suggested_max_epochs: int, assessment: dict
) -> tuple[bool, str]:
    """Arm A auto-raise: set params.max_epochs to a value the sufficiency check
    verified fits a toy set, log an assumption. Returns (ok, aid_or_error).

    Only max_epochs is touched. batch_size is left alone because it is the
    acquisition batch (the paper invariant b >= b' rides on it), not a training
    knob."""
    params_path = state.paths.pipeline_dir / "params.json"
    if not params_path.exists():
        return False, "params.json missing"
    data = json.loads(params_path.read_text(encoding="utf-8"))
    params = data.get("params") or {}
    entry = params.get("max_epochs")
    if not isinstance(entry, dict):
        return False, "max_epochs not in params.json"

    old = entry.get("value")
    pre_edit_snapshot = json.dumps(data, indent=2)
    aid = _next_assumption_id(state)
    prev_reasoning = entry.get("reasoning", "")
    entry["value"] = int(suggested_max_epochs)
    entry["reasoning"] = (
        f"AUTO-RAISED ({aid}): max_epochs raised from {old} to {suggested_max_epochs}. "
        f"The training-budget sufficiency check found the derived budget could not fit a "
        f"separable toy set sized to the round-one labeled set (train acc "
        f"{assessment.get('live_acc')} vs chance+margin {assessment.get('target')}); "
        f"{suggested_max_epochs} epochs fit it with batch and learning rate unchanged. "
        f"See assumptions.md entry {aid}."
        + (f" Previous reasoning: {prev_reasoning}" if prev_reasoning else "")
    )
    data["params"] = params
    params_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    sys.path.insert(0, str(REPO_ROOT))
    try:
        from schemas.params import Params  # noqa: PLC0415
        Params.model_validate(data)
    except Exception as e:  # noqa: BLE001
        params_path.write_text(pre_edit_snapshot + "\n", encoding="utf-8")
        return False, f"raise produced schema-invalid params.json (reverted): {e}"

    _append_assumption(
        state, aid=aid,
        title=f"max_epochs raised from {old} to {suggested_max_epochs} for training sufficiency",
        detected=(
            f"The derived training budget (max_epochs={old}, batch={assessment.get('live_batch')}) "
            f"could not fit a separable toy set sized to the round-one labeled set "
            f"({assessment.get('n_labeled_fixture')} samples): train accuracy "
            f"{assessment.get('live_acc')} vs chance+margin {assessment.get('target')}. "
            f"Cross-stage incoherence — the budget was set without reference to the "
            f"architecture's convergence behavior (dropout {assessment.get('live_dropout')})."
        ),
        action=(
            f"Set params.json `params.max_epochs.value` to {suggested_max_epochs}, the smallest "
            f"value on the search grid that fits the toy set with batch and learning rate "
            f"unchanged. batch_size was left alone — it is the acquisition batch, not a training knob."
        ),
        reasoning=(
            "A high-dropout model on a few-hundred-example labeled set needs more epochs than a low "
            "smoke-scale cap allows, or training stays near chance and the acquisition signal "
            "collapses with it."
        ),
        alternative=(
            "Reduce the model's dropout, or have the training function mini-batch the labeled set so "
            "each epoch does more gradient steps, instead of raising max_epochs."
        ),
        override=(
            f"Edit `<run_dir>/.pipeline/params.json` to set `params.max_epochs.value` to your "
            f"preferred value, or revert to {old} if your training protocol differs."
        ),
    )
    return True, aid


def _apply_scale_rescale(
    state: "PipelineState", *, entry: dict
) -> tuple[bool, str]:
    """Deterministic scale-calibration rescale: set the param to the
    converted value, log an assumption (the A002 pattern from the 2026-07-02
    run, now applied at 2.x instead of by a smoke-fix-loop agent two failed
    executions later). Returns (ok, aid_or_error)."""
    name = entry["name"]
    new_value = entry["new_value"]
    params_path = state.paths.pipeline_dir / "params.json"
    if not params_path.exists():
        return False, "params.json missing"
    data = json.loads(params_path.read_text(encoding="utf-8"))
    params = data.get("params") or {}
    param_entry = params.get(name)
    if not isinstance(param_entry, dict):
        return False, f"{name} not in params.json"

    old = param_entry.get("value")
    pre_edit_snapshot = json.dumps(data, indent=2)
    aid = _next_assumption_id(state)
    param_entry["value"] = new_value
    param_entry["source"] = "system_default"
    param_entry.setdefault("paper_value", entry.get("paper_value"))
    param_entry["reasoning"] = (
        f"AUTO-RESOLVED ({aid}): the paper value is {entry.get('paper_value')}, "
        f"but this run uses {new_value}. Derivation: {entry.get('derivation')}. "
        f"See assumptions.md entry {aid}."
    )
    data["params"] = params
    params_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    sys.path.insert(0, str(REPO_ROOT))
    try:
        from schemas.params import Params  # noqa: PLC0415
        Params.model_validate(data)
    except Exception as e:  # noqa: BLE001
        params_path.write_text(pre_edit_snapshot + "\n", encoding="utf-8")
        return False, f"rescale produced schema-invalid params.json (reverted): {e}"

    _append_assumption(
        state, aid=aid,
        title=f"{name} rescaled from {old} to {new_value}",
        detected=(
            f"{name}={old} is declared scale-dependent "
            f"(calibration context: {entry.get('assumes_declared')!r}) but the "
            f"data the delivered load_data actually returns measures "
            f"{entry.get('measured')}-scaled. Shipping the unrescaled paper "
            f"value degenerates the scale-dependent term at runtime."
        ),
        action=(
            f"Set params.json `params.{name}.value` to {new_value} "
            f"(source system_default, paper_value preserved). "
            f"Derivation: {entry.get('derivation')}."
        ),
        reasoning=(
            f"A distance-like threshold scales linearly with the data's "
            f"per-feature scale, so the paper's calibration must shrink by "
            f"the same factor as the preprocessing shrinks distances. The "
            f"conversion is applied deterministically at parameter "
            f"derivation; the stage-5 scale-mismatch probe independently "
            f"re-checks the shipped value against the delivered data."
        ),
        alternative=(
            f"Change method/data.py preprocessing to match the paper's "
            f"assumed scale and keep {name}={entry.get('paper_value')} "
            f"paper-exact."
        ),
        override=(
            f"Edit `<run_dir>/.pipeline/params.json` to set "
            f"`params.{name}.value` to your preferred value (or change the "
            f"data preprocessing in `method/data.py`), then re-run."
        ),
    )
    return True, aid


def _log_per_dataset_binding(state: "PipelineState", stage_id: str) -> None:
    """Surface the deriver's per-dataset binding record.

    Bound params: one run event each (paper-faithful, nothing to disclose
    beyond the audit trail already in params.json). Fallbacks: one
    assumptions.md entry each — the paper stated per-dataset values, the
    demo's dataset could not be bound, and the primary-dataset scalar
    shipped. Never halts."""
    params_path = state.paths.pipeline_dir / "params.json"
    if not params_path.exists():
        return
    try:
        data = json.loads(params_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    record = data.get("per_dataset_binding")
    if not isinstance(record, dict):
        return

    for row in record.get("bound", []):
        log(stage_id, "per_dataset_value_bound",
            f"{row.get('param')} bound to {row.get('dataset')} paper value "
            f"{row.get('value')} ({row.get('evidence')})")

    for row in record.get("fallbacks", []):
        name = row.get("param", "?")
        aid = _next_assumption_id(state)
        _append_assumption(
            state, aid=aid,
            title=f"{name}: per-dataset paper values could not be bound to the demo",
            detected=(
                f"The paper states dataset-specific values for {name} "
                f"({row.get('dataset_map')}), but {row.get('reason')}."
            ),
            action=(
                f"Shipped the spec's primary-dataset scalar "
                f"({row.get('scalar_value')}) unchanged. params.json records "
                f"no bound_dataset for {name}."
            ),
            reasoning=(
                "Binding by anything other than the demo's own dataset would "
                "be a guess; the honest fallback is the analyzer's "
                "primary-benchmark scalar with this disclosure."
            ),
            alternative=(
                "If your demo targets one of the paper's datasets, set "
                f"params.json `params.{name}.value` to that dataset's map "
                "entry before re-running downstream stages."
            ),
            override=(
                f"Edit `<run_dir>/.pipeline/params.json` "
                f"`params.{name}.value` to the value for your dataset."
            ),
        )
        log(stage_id, "per_dataset_value_fallback",
            f"{name}: unbindable per-dataset map, scalar shipped "
            f"(assumption {aid})")


def _apply_scale_calibration(state: "PipelineState", stage_id: str) -> None:
    """Hardening item 1 (2026-07-02 GBALD run): deterministic scale
    calibration for declared scale-dependent hyperparameters. Never halts —
    defined conversions are applied + logged, undefined pairs log a
    needs-attention note, and the stage-5 scale probe stays the backstop.
    Runs before the stage reviewer so the review sees corrected params."""
    paths = state.paths
    out = paths.pipeline_dir / "scale_calibration.json"
    proc = run_script(
        stage_id,
        ["scripts/scale_calibration_2x.py", "--run-dir", str(paths.run_dir),
         "--out", str(out)],
        timeout=240,
    )
    if proc.returncode != 0:
        log(stage_id, "scale_calibration_skipped",
            f"scale calibration exited {proc.returncode}; continuing "
            f"(stderr: {(proc.stderr or proc.stdout or '')[-400:]})")
        return
    try:
        assessment = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log(stage_id, "scale_calibration_skipped",
            f"could not read scale calibration assessment ({e}); continuing")
        return
    if not assessment.get("applicable"):
        log(stage_id, "scale_calibration_na",
            f"scale calibration N/A: {assessment.get('reason', '')}")
        return

    for entry in assessment.get("entries", []):
        status = entry.get("status")
        name = entry.get("name", "?")
        if status == "rescale":
            ok, aid_or_err = _apply_scale_rescale(state, entry=entry)
            if ok:
                log(stage_id, "scale_rescale_applied",
                    f"{name} rescaled to {entry.get('new_value')} "
                    f"(assumption {aid_or_err}): {entry.get('derivation', '')[:140]}")
            else:
                log(stage_id, "scale_calibration_skipped",
                    f"{name} rescale apply failed: {aid_or_err}")
        elif status == "needs_attention":
            aid = _next_assumption_id(state)
            _append_assumption(
                state, aid=aid,
                title=f"{name}: scale calibration needs researcher attention",
                detected=(
                    f"{name} (paper value {entry.get('paper_value')}) is "
                    f"declared scale-dependent "
                    f"(assumes {entry.get('assumes_declared')!r}) but "
                    f"{entry.get('reason')}."
                ),
                action=(
                    "None — the paper value was shipped unchanged. The "
                    "stage-5 scale-mismatch probe re-checks it against the "
                    "delivered data and will flag a real mismatch."
                ),
                reasoning=(
                    "No deterministic conversion is defined for this scale "
                    "pair; inventing a factor would be a fabrication."
                ),
                alternative=(
                    "Provide the correct rescale by editing "
                    "`params.json`, or align the data preprocessing with "
                    "the paper's assumed scale."
                ),
                override=(
                    f"Edit `<run_dir>/.pipeline/params.json` "
                    f"`params.{name}.value` if the shipped value is wrong "
                    f"for your data scale."
                ),
            )
            log(stage_id, "scale_calibration_needs_attention",
                f"{name}: {entry.get('reason', '')[:160]} (assumption {aid})")
        elif status == "unprobeable":
            # R2C-091: explicit other/unit-norm (and unsupported observation
            # grammar) is evidence about coverage, not a researcher decision.
            # Preserve it in the assessment/log without manufacturing an
            # assumption or asking the user to invent a conversion.
            log(stage_id, "scale_calibration_unprobeable",
                f"{name}: {entry.get('reason', '')[:160]}")
        else:
            log(stage_id, "scale_calibration_ok",
                f"{name}: {status} — {entry.get('reason', '')[:120]}")


def _check_training_budget_sufficiency(
    state: "PipelineState", stage_id: str
) -> "StageResult | None":
    """Arm A of the numeric-coherence pass (queue 11d). Runs at 2.x because both
    params (derived just above) and the built package (finalized at 2.d) exist.

    A starved training budget produces a degenerate notebook with no crash, so
    catch it before notebook generation. On a starved-but-fixable budget,
    auto-raise max_epochs and log an assumption. When no budget on the grid can
    make the live config learn, the problem is the learning rate or the
    architecture, not the budget — log a needs-attention assumption and let the
    run continue (the downstream executed-notebook probe is the backstop), per
    the unattended auto-resolve-and-log policy."""
    paths = state.paths
    out = paths.pipeline_dir / "budget_assessment.json"
    proc = run_script(
        stage_id,
        ["scripts/budget_sufficiency.py", "--run-dir", str(paths.run_dir),
         "--out", str(out)],
        timeout=240,
    )
    if proc.returncode != 0:
        log(stage_id, "budget_check_skipped",
            f"budget sufficiency check exited {proc.returncode}; continuing "
            f"(stderr: {(proc.stderr or proc.stdout or '')[-400:]})")
        return None
    try:
        assessment = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log(stage_id, "budget_check_skipped",
            f"could not read budget assessment ({e}); continuing")
        return None

    if not assessment.get("applicable"):
        log(stage_id, "budget_check_na",
            f"training-budget sufficiency N/A: {assessment.get('reason', '')}")
        return None
    if assessment.get("sufficient_at_live"):
        log(stage_id, "budget_sufficient", assessment.get("reason", ""))
        return None

    suggested = assessment.get("suggested_max_epochs")
    if suggested is not None:
        ok, aid_or_err = _apply_budget_raise(
            state, suggested_max_epochs=int(suggested), assessment=assessment)
        if ok:
            log(stage_id, "budget_auto_raised",
                f"{aid_or_err}: max_epochs -> {suggested} "
                f"({assessment.get('reason', '')})")
        else:
            log(stage_id, "budget_raise_failed",
                f"could not apply budget raise ({aid_or_err}); continuing")
        return None

    # No budget on the grid fits — not a shortfall the deriver can patch.
    aid = _next_assumption_id(state)
    _append_assumption(
        state, aid=aid,
        title="training config cannot fit a toy set at any budget on the grid",
        detected=assessment.get("reason", ""),
        action=(
            "No change made. max_epochs was left as derived because raising it alone "
            "does not make the configuration learn."
        ),
        reasoning=(
            "When the live config cannot fit a separable toy set even at the epoch ceiling, "
            "the cause is the learning rate or the architecture (e.g. dropout too high for the "
            "labeled-set size), not the epoch budget. The executed-notebook sanity probe will "
            "also flag the degenerate curve downstream."
        ),
        alternative="Lower the model dropout, lower the learning rate, or mini-batch training.",
        override=(
            "Inspect `params.json` learning_rate and the model in `method/model.py`; adjust and "
            "re-run."
        ),
    )
    log(stage_id, "budget_unfittable_flagged",
        f"{aid}: {assessment.get('reason', '')}")
    return None


# ---------------------------------------------------------------------------
# Stage 2.x — bounded analyzer fix loop for provenance-probe failures
# (R2C-033, all five design points maintainer-approved 2026-07-28)
# ---------------------------------------------------------------------------


_PARAMS_PROVENANCE_RETRY_CAP = 2

_PROVENANCE_BULLET_RE = re.compile(
    r"^provenance probe (?P<probe>\S+): (?P<param>[^:]+): (?P<message>.+)$")

# Fields of a params.json entry that carry provenance (schemas/params.py).
# Used only for naming what changed in the loud-downgrade disclosure.
_PARAM_PROVENANCE_FIELDS = (
    "source", "paper_section", "paper_value", "note", "reasoning",
    "paper_says",
)


def _params_provenance_bullets(err_tail: str) -> list[dict] | None:
    """Parse validate_params_output.py stderr into provenance-probe bullets.

    Returns [{probe, param, bullet}, ...] when EVERY failure bullet is a
    provenance-probe finding (`- provenance probe <id>: <param>:
    <message>` — the shape `_provenance_errors` renders). Returns None when
    any bullet belongs to a different class (schema, missing params,
    runtime drift, acquisition invariant) or when nothing parses at all
    (validator crash) — those keep today's immediate-halt behavior; the
    analyzer cannot fix them through the spec's provenance surfaces."""
    bullets = [ln.strip()[2:].strip()
               for ln in err_tail.splitlines() if ln.strip().startswith("- ")]
    if not bullets:
        return None
    parsed: list[dict] = []
    for bullet in bullets:
        m = _PROVENANCE_BULLET_RE.match(bullet)
        if not m:
            return None
        parsed.append({
            "probe": m.group("probe"),
            "param": m.group("param").strip(),
            "bullet": bullet,
        })
    return parsed


def _glossary_linked_param_names(spec: dict, names: set[str]) -> set[str]:
    """Expand probe-named params with their glossary-linked names.

    The analyzer's param glossary links a paper symbol to the derived
    signature name via `aliases` (the `_glossary_anchor_names` seam in
    derive_params.py). A probe naming the derived name must permit edits
    to the glossary entry recorded under the paper's own symbol, and vice
    versa — the entry is one parameter under two names, never two
    parameters."""
    expanded = set(names)
    glossary = ((spec.get("critical_requirements") or {})
                .get("param_glossary") or [])
    for entry in glossary:
        if not isinstance(entry, dict):
            continue
        group = {str(entry.get("name"))} | {
            str(a) for a in (entry.get("aliases") or [])}
        if group & expanded:
            expanded |= group
    return expanded


def _spec_entry_param_names(entry: object) -> set[str]:
    if not isinstance(entry, dict):
        return set()
    return {str(entry.get("name"))} | {
        str(a) for a in (entry.get("aliases") or [])}


def _params_spec_scope_violations(
    old_spec: dict, new_spec: dict, allowed: set[str],
) -> list[str]:
    """Deterministic scoped-edit check for the provenance fix loop (design
    point 2 — the laundering guardrail).

    The analyzer's retry may change ONLY spec surfaces attributable to the
    probe-named parameters (`allowed`, already glossary-expanded). The
    surfaces derive_params.py reads for per-parameter provenance are
    diffed per parameter; every other spec change is rejected outright:

      - critical_requirements.param_glossary — entries keyed by
        name/aliases;
      - critical_requirements.scale_dependent_hyperparameters — entries
        keyed by name (the pdwa d_safe fabrication lived here);
      - critical_requirements.training / data_setup — structured
        per-parameter fields keyed by field name (shared fields like
        `paper_section` are conservatively rejected);
      - methodology_replication_contract — prose elements keyed by
        element_id; a changed element is allowed only when an allowed
        name appears as a whole word in its old or new JSON text (the
        contract feeds `_lookup_contract_paper_value`);
      - comparison.evaluation_protocol — for an explicitly bound allowed
        parameter, only value/status/evidence fields may change. Scientific
        role, parameter binding, unit, granularity, scheme, and list order are
        immutable in this provenance-only retry;
      - anything else (paper, core_method, comparison fields outside the
        typed protocol, pluggable signature, methodology_contract_pack, ...)
        — any change is a violation.

    Returns human-readable violation strings; empty means in scope."""

    def canon(x: object) -> str:
        return json.dumps(x, sort_keys=True, ensure_ascii=False)

    violations: list[str] = []
    old_cr = old_spec.get("critical_requirements") or {}
    new_cr = new_spec.get("critical_requirements") or {}

    # Per-parameter keyed LISTS (entries carry name [+ aliases]).
    for list_key in ("param_glossary", "scale_dependent_hyperparameters"):
        old_by = {str(e.get("name")): e
                  for e in (old_cr.get(list_key) or []) if isinstance(e, dict)}
        new_by = {str(e.get("name")): e
                  for e in (new_cr.get(list_key) or []) if isinstance(e, dict)}
        for name in sorted(set(old_by) | set(new_by)):
            o, n = old_by.get(name), new_by.get(name)
            if canon(o) == canon(n):
                continue
            touched = _spec_entry_param_names(o) | _spec_entry_param_names(n)
            if not (touched & allowed):
                violations.append(
                    f"critical_requirements.{list_key} entry {name!r} "
                    f"changed, but {name!r} is not named in the probe "
                    f"findings")

    # Per-parameter keyed DICTS (field name IS the parameter name).
    for dict_key in ("training", "data_setup"):
        old_d = old_cr.get(dict_key) or {}
        new_d = new_cr.get(dict_key) or {}
        if not isinstance(old_d, dict) or not isinstance(new_d, dict):
            if canon(old_d) != canon(new_d):
                violations.append(
                    f"critical_requirements.{dict_key} changed shape")
            continue
        for key in sorted(set(old_d) | set(new_d)):
            if canon(old_d.get(key)) == canon(new_d.get(key)):
                continue
            if key not in allowed:
                violations.append(
                    f"critical_requirements.{dict_key}.{key} changed, but "
                    f"{key!r} is not named in the probe findings")

    # Methodology contract prose — element-level, attributed by mention.
    old_els = ((old_spec.get("methodology_replication_contract") or {})
               .get("elements") or [])
    new_els = ((new_spec.get("methodology_replication_contract") or {})
               .get("elements") or [])
    old_el_by = {str(e.get("element_id", i)): e
                 for i, e in enumerate(old_els) if isinstance(e, dict)}
    new_el_by = {str(e.get("element_id", i)): e
                 for i, e in enumerate(new_els) if isinstance(e, dict)}
    for el_id in sorted(set(old_el_by) | set(new_el_by)):
        o, n = old_el_by.get(el_id), new_el_by.get(el_id)
        if canon(o) == canon(n):
            continue
        text = canon(o) + canon(n)
        if not any(re.search(rf"\b{re.escape(name)}\b", text)
                   for name in allowed):
            violations.append(
                f"methodology_replication_contract element {el_id!r} "
                f"changed but mentions no probe-named parameter")

    # Everything else, wholesale — including critical_requirements subkeys
    # not handled above and the contract's non-element fields.
    handled_cr = {"param_glossary", "scale_dependent_hyperparameters",
                  "training", "data_setup"}
    for key in sorted(set(old_cr) | set(new_cr)):
        if key in handled_cr:
            continue
        if canon(old_cr.get(key)) != canon(new_cr.get(key)):
            violations.append(
                f"critical_requirements.{key} changed — outside the "
                f"parameter-provenance surfaces the fix may touch")
    old_mrc = dict(old_spec.get("methodology_replication_contract") or {})
    new_mrc = dict(new_spec.get("methodology_replication_contract") or {})
    old_mrc.pop("elements", None)
    new_mrc.pop("elements", None)
    if canon(old_mrc) != canon(new_mrc):
        violations.append(
            "methodology_replication_contract changed outside its elements")

    # Role-typed protocol facts are the sole provenance authority for bound
    # temporal params. Permit the analyzer to correct the exact T+1 -> K=1
    # class here, while keeping the laundering guard narrow: this retry may
    # change only status/value/evidence on the already bound, probe-named
    # quantity. Role, binding, unit, granularity, scheme, and order require a
    # fresh Stage-1 analysis rather than a provenance repair.
    old_comparison = old_spec.get("comparison") or {}
    new_comparison = new_spec.get("comparison") or {}
    if not isinstance(old_comparison, dict) or not isinstance(new_comparison, dict):
        if canon(old_comparison) != canon(new_comparison):
            violations.append("spec field 'comparison' changed shape")
    else:
        old_protocol = old_comparison.get("evaluation_protocol")
        new_protocol = new_comparison.get("evaluation_protocol")
        if canon(old_protocol) != canon(new_protocol):
            if not isinstance(old_protocol, dict) or not isinstance(new_protocol, dict):
                violations.append(
                    "comparison.evaluation_protocol changed shape during a "
                    "parameter-provenance retry"
                )
            else:
                if canon(old_protocol.get("scheme")) != canon(
                    new_protocol.get("scheme")
                ):
                    violations.append(
                        "comparison.evaluation_protocol.scheme changed — the "
                        "probe names a parameter quantity, not the evaluation "
                        "scheme"
                    )
                old_quantities = old_protocol.get("quantities") or []
                new_quantities = new_protocol.get("quantities") or []
                if not isinstance(old_quantities, list) or not isinstance(
                    new_quantities, list
                ):
                    violations.append(
                        "comparison.evaluation_protocol.quantities changed shape"
                    )
                elif len(old_quantities) != len(new_quantities):
                    violations.append(
                        "comparison.evaluation_protocol.quantities changed "
                        "length during a parameter-provenance retry"
                    )
                else:
                    permitted = {
                        "value",
                        "paper_value_status",
                        "evidence_quote",
                        "paper_section",
                        "paper_element_ids",
                    }
                    identity = {
                        "role", "parameter_name", "paper_names",
                        "paper_symbols", "unit", "granularity",
                        "axis_evidence_quote", "axis_paper_section",
                        "axis_paper_element_ids",
                    }
                    for index, (old_quantity, new_quantity) in enumerate(
                        zip(old_quantities, new_quantities)
                    ):
                        if canon(old_quantity) == canon(new_quantity):
                            continue
                        if not isinstance(old_quantity, dict) or not isinstance(
                            new_quantity, dict
                        ):
                            violations.append(
                                "comparison.evaluation_protocol.quantities["
                                f"{index}] changed shape"
                            )
                            continue
                        changed = {
                            field
                            for field in set(old_quantity) | set(new_quantity)
                            if canon(old_quantity.get(field))
                            != canon(new_quantity.get(field))
                        }
                        immutable_changes = sorted(changed & identity)
                        if immutable_changes:
                            violations.append(
                                "comparison.evaluation_protocol.quantities["
                                f"{index}] changed immutable protocol identity "
                                f"field(s) {immutable_changes}; a provenance "
                                "retry may not change role, binding, paper "
                                "names/symbols, unit/granularity, or axis "
                                "evidence"
                            )
                        unexpected = sorted(changed - permitted - identity)
                        if unexpected:
                            violations.append(
                                "comparison.evaluation_protocol.quantities["
                                f"{index}] changed unsupported field(s) "
                                f"{unexpected}"
                            )
                        parameter_name = old_quantity.get("parameter_name")
                        if parameter_name not in allowed:
                            violations.append(
                                "comparison.evaluation_protocol.quantities["
                                f"{index}] changed for parameter_name="
                                f"{parameter_name!r}, which is not named in "
                                "the probe findings"
                            )
                old_protocol_other = dict(old_protocol)
                new_protocol_other = dict(new_protocol)
                old_protocol_other.pop("scheme", None)
                new_protocol_other.pop("scheme", None)
                old_protocol_other.pop("quantities", None)
                new_protocol_other.pop("quantities", None)
                if canon(old_protocol_other) != canon(new_protocol_other):
                    violations.append(
                        "comparison.evaluation_protocol changed outside scheme "
                        "and quantities"
                    )

        old_comparison_other = dict(old_comparison)
        new_comparison_other = dict(new_comparison)
        old_comparison_other.pop("evaluation_protocol", None)
        new_comparison_other.pop("evaluation_protocol", None)
        if canon(old_comparison_other) != canon(new_comparison_other):
            violations.append(
                "spec field 'comparison' changed outside "
                "evaluation_protocol"
            )

    for key in sorted(set(old_spec) | set(new_spec)):
        if key in ("critical_requirements",
                   "methodology_replication_contract", "comparison"):
            continue
        if canon(old_spec.get(key)) != canon(new_spec.get(key)):
            violations.append(
                f"spec field {key!r} changed — outside the "
                f"parameter-provenance surfaces the fix may touch")
    return violations


def _params_output_scope_violations(
    old_params: dict | None, new_params: dict | None, allowed: set[str],
) -> list[str]:
    """Output-side laundering guardrail: derive_params.py is deterministic,
    so after the re-derivation any params.json entry for a parameter NOT
    named by the probes that changed means the spec edit had effects the
    spec-side diff could not attribute (e.g. through contract prose). The
    retry is rejected."""

    def canon(x: object) -> str:
        return json.dumps(x, sort_keys=True, ensure_ascii=False)

    old_p = (old_params or {}).get("params") or {}
    new_p = (new_params or {}).get("params") or {}
    violations: list[str] = []
    for name in sorted(set(old_p) | set(new_p)):
        if name in allowed:
            continue
        if canon(old_p.get(name)) != canon(new_p.get(name)):
            violations.append(
                f"params.json entry {name!r} changed under the retry, but "
                f"{name!r} is not named in the probe findings")
    return violations


def _params_provenance_slices(
    spec: dict, params: dict | None, allowed: set[str],
) -> str:
    """Verbatim spec/params slices for the probe-named parameters, inlined
    into the stage-2x fix finding so the analyzer retry edits the right
    surfaces without re-reading the whole spec (agent dispatch optimization
    2026-09-01: fix dispatches re-read method_spec.json and the deriver
    source to locate these). Mirrors exactly the surfaces
    `_params_spec_scope_violations` allows the retry to touch."""

    def canon(x: object) -> str:
        return json.dumps(x, indent=2, ensure_ascii=False)

    parts: list[str] = []
    cr = spec.get("critical_requirements") or {}
    derived = (params or {}).get("params") or {}
    for name in sorted(allowed & set(derived)):
        parts.append(
            f"params.json entry `{name}` (derived output — you cannot edit "
            f"this file):\n{canon(derived[name])}")
    for list_key in ("param_glossary", "scale_dependent_hyperparameters"):
        for e in (cr.get(list_key) or []):
            if isinstance(e, dict) and (_spec_entry_param_names(e) & allowed):
                parts.append(
                    f"critical_requirements.{list_key} entry "
                    f"`{e.get('name')}`:\n{canon(e)}")
    for dict_key in ("training", "data_setup"):
        d = cr.get(dict_key) or {}
        if isinstance(d, dict):
            for key in sorted(set(d) & allowed):
                parts.append(
                    f"critical_requirements.{dict_key}.{key}:"
                    f"\n{canon(d[key])}")
    quantities = (((spec.get("comparison") or {})
                   .get("evaluation_protocol") or {}).get("quantities")
                  or [])
    for q in quantities:
        if isinstance(q, dict) and str(q.get("parameter_name")) in allowed:
            parts.append(
                f"comparison.evaluation_protocol quantity bound to "
                f"`{q.get('parameter_name')}`:\n{canon(q)}")
    elements = ((spec.get("methodology_replication_contract") or {})
                .get("elements") or [])
    for i, e in enumerate(elements):
        if not isinstance(e, dict):
            continue
        text = json.dumps(e, ensure_ascii=False)
        if any(re.search(rf"\b{re.escape(n)}\b", text) for n in allowed):
            parts.append(
                f"methodology_replication_contract element "
                f"`{e.get('element_id', i)}` (mentions a failing "
                f"parameter):\n{canon(e)}")
    if not parts:
        return ""
    body = "\n\n".join(parts)
    if len(body) > 12000:
        body = body[:12000] + (
            "\n... [slices truncated — read method_spec.json only for "
            "surfaces beyond this point]")
    return body


def _params_provenance_fix_finding(
    bullets: list[dict], iteration: int, *,
    spec: dict | None = None, params: dict | None = None,
    allowed: set[str] | None = None,
) -> dict:
    """Mechanical fix finding for the analyzer, built directly from the
    probe stderr (design point 1 — the quote-floor enricher pattern: the
    validator's own output IS the finding; no halt judge).

    When `spec`/`params`/`allowed` are given, the finding is self-sufficient:
    it inlines each failing probe's acceptance contract and the verbatim
    spec/params slices for the probe-named parameters, so the retry needs
    zero pipeline-source reads and no whole-spec hunt."""
    names = list(dict.fromkeys(b["param"] for b in bullets))
    listed = "\n".join(f"- {b['bullet']}" for b in bullets)
    description = (
        f"validate_params_output.py rejected params.json on {len(bullets)} "
        f"provenance-probe finding(s) checked against the paper text "
        f"(failing parameter(s): {', '.join(names)}):\n{listed}\n\n"
        "params.json is derived DETERMINISTICALLY from "
        ".pipeline/method_spec.json by the pipeline's deriver, so the "
        "dishonest provenance lives in the spec's fields for the failing "
        "parameter(s): its param_glossary entry, its "
        "scale_dependent_hyperparameters entry, a structured "
        "training/data_setup value, a methodology-contract claim, or its "
        "explicitly bound comparison.evaluation_protocol quantity. Fix "
        "the SPEC; you cannot and must not edit params.json."
    )
    from validate_params_provenance import (  # noqa: PLC0415
        PROBE_ACCEPTANCE_CONTRACTS,
    )
    contracts = [
        PROBE_ACCEPTANCE_CONTRACTS[p]
        for p in dict.fromkeys(b["probe"] for b in bullets)
        if p in PROBE_ACCEPTANCE_CONTRACTS
    ]
    if contracts:
        description += (
            "\n\nAcceptance contracts — the COMPLETE rule each failing "
            "probe re-checks (do not read pipeline source to re-derive "
            "these):\n" + "\n".join(f"- {c}" for c in contracts)
        )
    if spec is not None and allowed:
        slices = _params_provenance_slices(spec, params, allowed)
        if slices:
            description += (
                "\n\nCurrent spec/params slices for the failing "
                "parameter(s), copied verbatim so you do not need to hunt "
                "method_spec.json — these are the ONLY surfaces this retry "
                "may change:\n\n" + slices
            )
    proposed_fix = (
        "Re-read what the paper ACTUALLY states about each failing "
        "parameter and re-derive honest provenance in "
        ".pipeline/method_spec.json. Exactly three honest resolutions "
        "exist — pick per parameter:\n"
        "(1) derived statistic — the paper defines the value only by a "
        "formula: record the formula as the provenance and STOP claiming "
        "the computed number is paper-stated (no paper_value for a number "
        "the paper never prints; describe the formula and the status of "
        "its inputs instead).\n"
        "(2) demo assumption — an input the paper never fixes: present "
        "the value as an assumed demo default, stated as an assumption, "
        "never attributed to the paper.\n"
        "(3) corrected quote — the value IS in the paper but the quote or "
        "locator is wrong: replace them with the paper's verbatim words "
        "and the correct section.\n"
        "For a parameter bound by comparison.evaluation_protocol, correct "
        "that quantity's value, paper_value_status, and evidence directly. "
        "Preserve its role, parameter_name, paper_names, paper_symbols, "
        "unit, granularity, and axis evidence; those are outside this "
        "provenance-only retry.\n"
        "Edit ONLY the spec surfaces for the parameter(s) named above. "
        "The driver deterministically diffs the spec and the re-derived "
        "params.json after your fix: if any OTHER parameter changed, the "
        "whole retry is rejected. The provenance probes then re-run "
        "against the paper text as the terminal gate — they cannot be "
        "satisfied by rewording a claim the paper does not make."
    )
    return {
        "id": f"PROV{iteration:03d}",
        "severity": "critical",
        "description": description,
        "proposed_fix": proposed_fix,
    }


def _log_params_provenance_repairs(
    state: PipelineState, stage_id: str, *,
    baseline_params: dict | None, final_params: dict | None,
    bullets_by_param: dict[str, list[str]], iteration: int,
) -> list[str]:
    """Loud downgrades (design point 3): every provenance change that
    SURVIVED the fix loop gets one assumptions.md entry plus one
    `params_provenance_repaired` run event, naming the parameter, the old
    and new provenance, and the triggering probe finding(s). Returns the
    repaired parameter names."""

    def canon(x: object) -> str:
        return json.dumps(x, sort_keys=True, ensure_ascii=False)

    old_p = (baseline_params or {}).get("params") or {}
    new_p = (final_params or {}).get("params") or {}
    repaired: list[str] = []
    for name in sorted(bullets_by_param):
        old_e = old_p.get(name)
        new_e = new_p.get(name)
        if canon(old_e) == canon(new_e):
            continue
        old_e = old_e if isinstance(old_e, dict) else {}
        new_e = new_e if isinstance(new_e, dict) else {}
        old_source = old_e.get("source")
        new_source = new_e.get("source")
        changed_fields = sorted(
            f for f in set(old_e) | set(new_e)
            if canon(old_e.get(f)) != canon(new_e.get(f)))
        probe_findings = bullets_by_param.get(name, [])
        aid = _next_assumption_id(state)
        _append_assumption(
            state, aid=aid,
            title=f"Parameter provenance repaired — {name} "
                  f"(source: {old_source} -> {new_source})",
            detected=(
                f"The deterministic provenance probes rejected "
                f"params.json's labeling of `{name}`:\n"
                + "\n".join(f"- {b}" for b in probe_findings)),
            action=(
                f"The method analyzer was re-dispatched with the probe "
                f"findings and re-derived honest provenance in "
                f"method_spec.json (fix-loop iteration {iteration}); "
                f"derive_params.py then re-derived params.json. The "
                f"`{name}` entry changed from source={old_source!r} to "
                f"source={new_source!r} (fields changed: "
                f"{', '.join(changed_fields) or 'none'}). Old entry: "
                f"{canon(old_e)}. New entry: {canon(new_e)}."),
            reasoning=(
                "derive_params.py is deterministic, so honest provenance "
                "must come from the analyzer's spec. The probes re-ran "
                "against the paper text as the terminal gate and now "
                "pass, and a scoped-edit diff confirmed no parameter "
                "outside the probe findings changed."),
            alternative=(
                "Keep the original labeling and halt the run for a manual "
                "review of what the paper actually states."),
            override=(
                f"Inspect `{name}` in .pipeline/params.json and its spec "
                f"sources in .pipeline/method_spec.json; correct them and "
                f"re-run the pipeline from stage 2.x."),
        )
        _append_run_event(
            state.paths,
            "params_provenance_repaired",
            stage_id=stage_id,
            summary=(f"stage 2.x fix loop repaired `{name}` provenance "
                     f"(source: {old_source} -> {new_source}), logged as "
                     f"{aid}"),
            details={
                "param": name,
                "old_source": old_source,
                "new_source": new_source,
                "changed_fields": changed_fields,
                "probe_findings": probe_findings,
                "iteration": iteration,
                "assumption_id": aid,
            },
        )
        log(stage_id, "params_provenance_repaired",
            f"{name}: source {old_source} -> {new_source} "
            f"(assumption {aid}; probe findings: "
            f"{'; '.join(probe_findings)[:300]})")
        repaired.append(name)
    return repaired


def _read_pipeline_json(path: Path) -> dict | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _run_params_provenance_fix_loop(
    state: PipelineState, stage_id: str, first_err: str,
) -> StageResult | None:
    """Bounded producer fix loop for provenance-probe failures at stage
    2.x (R2C-033). Before it existed, the first probe finding cost the
    entire runnable delivery: stage 2.x halted with retry_count 0 and
    degraded to the explanation-only package (pdwa 2026-07-28, a fixable
    labeling error on d_safe).

    The provenance a probe rejects lives in the method spec — the
    analyzer's output; derive_params.py is deterministic — so each
    iteration routes a MECHANICAL fix finding built from the probe stderr
    back to the method analyzer (no halt judge for probe-class failures;
    non-probe validator failures never enter the loop and keep today's
    immediate halt). Guardrails, in order, per iteration:

      1. Scoped-edit enforcement (`_params_spec_scope_violations`): only
         spec surfaces attributable to the probe-named parameters may
         change; anything else rejects the retry (spec restored, the
         iteration counts as failed).
      2. The spec must stay valid (validate_method_spec.py --strict);
         invalid-after-retry restores the spec and counts as failed.
      3. derive_params.py re-runs, then the output-side diff
         (`_params_output_scope_violations`) rejects the retry if any
         UNNAMED parameter's derived entry changed.
      4. validate_params_output.py re-runs as the terminal gate — the
         probes compare against the paper text and cannot be satisfied
         by wording.

    Two iterations max (design point 4); exhaustion halts exactly as the
    pre-loop behavior did (producer_output_invalid; halt() finalizes the
    explanation-only package). Every provenance change that survives is
    loud: an assumptions.md entry plus a `params_provenance_repaired` run
    event per parameter (design point 3). The loop never edits values or
    labels itself — only the producer writes, only the probes accept.

    Returns None on convergence (the caller proceeds exactly as if the
    first validator pass had succeeded); a halted StageResult otherwise.
    """
    paths = state.paths
    if _params_provenance_bullets(first_err) is None:
        # Not purely probe-class (schema failure, missing params, runtime
        # drift, validator crash): today's behavior, verbatim.
        return halt(paths, stage_id,
                    reason="validate_params_output.py failed",
                    halt_class="producer_output_invalid",
                    context={"stderr": first_err}, state=state)

    params_path = paths.pipeline_dir / "params.json"
    baseline_params = _read_pipeline_json(params_path)
    bullets_by_param: dict[str, list[str]] = {}
    err = first_err
    for iteration in range(1, _PARAMS_PROVENANCE_RETRY_CAP + 1):
        bullets = _params_provenance_bullets(err)
        if bullets is None:
            # A retry moved the failure OUT of the probe class (e.g. into
            # a schema error) — that class keeps today's halt.
            return halt(paths, stage_id,
                        reason="validate_params_output.py failed",
                        halt_class="producer_output_invalid",
                        context={"stderr": err},
                        retry_count=iteration - 1, state=state)
        named = {b["param"] for b in bullets}
        for b in bullets:
            per_param = bullets_by_param.setdefault(b["param"], [])
            if b["bullet"] not in per_param:
                per_param.append(b["bullet"])
        try:
            spec_before_text = paths.method_spec.read_text(encoding="utf-8")
            spec_before = json.loads(spec_before_text)
        except (OSError, json.JSONDecodeError) as e:
            return halt(paths, stage_id,
                        reason=f"method_spec.json unreadable at the params "
                               f"provenance fix loop: {e}",
                        halt_class="internal_contract_violation",
                        retry_count=iteration - 1, state=state)
        try:
            params_before_text = params_path.read_text(encoding="utf-8")
        except OSError:
            params_before_text = None
        allowed = _glossary_linked_param_names(spec_before, named)
        finding = _params_provenance_fix_finding(
            bullets, iteration, spec=spec_before,
            params=_read_pipeline_json_text(params_before_text),
            allowed=allowed)
        log(stage_id, "params_provenance_fix",
            f"iteration {iteration}: validate_params_output.py failed on "
            f"provenance probe finding(s) for {', '.join(sorted(named))}; "
            f"re-dispatching the method analyzer with the probe findings")
        try:
            _dispatch_analyzer_fix(state, [finding])
        except (OpencodeClientError, OutOfScopeWritesError) as e:
            return halt(paths, stage_id,
                        reason=f"analyzer fix dispatch failed during the "
                               f"params provenance fix loop: {e}",
                        halt_class=_dispatch_error_halt_class(e),
                        retry_count=iteration - 1, state=state)

        # -- guardrail 1: scoped-edit enforcement (laundering guard) -----
        spec_after = _read_pipeline_json(paths.method_spec)
        if spec_after is None:
            _restore_text(paths.method_spec, spec_before_text)
            log(stage_id, "params_fix_rejected",
                f"iteration {iteration}: retry left method_spec.json "
                f"unreadable; spec restored, iteration counts as failed")
            continue
        violations = _params_spec_scope_violations(
            spec_before, spec_after, allowed)
        if violations:
            _restore_text(paths.method_spec, spec_before_text)
            log(stage_id, "params_fix_scope_rejected",
                f"iteration {iteration}: retry edited spec surfaces "
                f"outside the probe-named parameter(s) "
                f"({', '.join(sorted(named))}); spec restored, iteration "
                f"counts as failed: " + "; ".join(violations))
            _append_run_event(
                paths,
                "params_fix_scope_rejected",
                summary=(f"stage 2x rejected a provenance-repair retry "
                         f"that edited spec surfaces outside the "
                         f"probe-named parameter(s); the prior spec was "
                         f"restored (laundering guard, iteration "
                         f"{iteration})"),
                details={"surface": "method_spec",
                         "iteration": iteration,
                         "probe_named_parameters": sorted(named),
                         "violations": violations},
            )
            continue

        # -- guardrail 2: the spec must stay valid ------------------------
        ok_spec, spec_err = _run_spec_validator(state)
        if not ok_spec:
            _restore_text(paths.method_spec, spec_before_text)
            log(stage_id, "params_fix_spec_invalid",
                f"iteration {iteration}: retried spec fails "
                f"validate_method_spec.py --strict; spec restored, "
                f"iteration counts as failed: {spec_err[:300]}")
            continue

        # -- deterministic re-derivation ----------------------------------
        proc = run_script(
            stage_id,
            ["scripts/derive_params.py", "--spec", str(paths.method_spec),
             "--run-dir", str(paths.run_dir)],
            timeout=60,
        )
        if proc.returncode != 0:
            return halt(paths, stage_id,
                        reason=f"derive_params.py exit {proc.returncode} "
                               f"on a validated spec during the params "
                               f"provenance fix loop",
                        halt_class="internal_contract_violation",
                        context={"stderr":
                                 (proc.stderr or proc.stdout)[-2000:]},
                        retry_count=iteration, state=state)

        # -- guardrail 3: output-side laundering guard --------------------
        params_after = _read_pipeline_json(params_path)
        out_violations = _params_output_scope_violations(
            _read_pipeline_json_text(params_before_text), params_after,
            allowed)
        if out_violations:
            _restore_text(paths.method_spec, spec_before_text)
            if params_before_text is not None:
                _restore_text(params_path, params_before_text)
            log(stage_id, "params_fix_scope_rejected",
                f"iteration {iteration}: the re-derived params.json "
                f"changed parameter(s) outside the probe findings; spec "
                f"and params restored, iteration counts as failed: "
                + "; ".join(out_violations))
            _append_run_event(
                paths,
                "params_fix_scope_rejected",
                summary=(f"stage 2x rejected a provenance-repair retry "
                         f"whose re-derived params output changed "
                         f"parameter(s) outside the probe findings; the "
                         f"prior spec and params were restored "
                         f"(laundering guard, iteration {iteration})"),
                details={"surface": "params_output",
                         "iteration": iteration,
                         "probe_named_parameters": sorted(named),
                         "violations": out_violations},
            )
            continue

        # -- guardrail 4: the probes are the terminal gate ----------------
        ok, err = _run_stage2_script(
            state, stage_id=stage_id,
            args=["scripts/validate_params_output.py"],
        )
        if ok:
            repaired = _log_params_provenance_repairs(
                state, stage_id,
                baseline_params=baseline_params,
                final_params=params_after,
                bullets_by_param=bullets_by_param,
                iteration=iteration)
            log(stage_id, "params_provenance_fix_converged",
                f"iteration {iteration}: provenance probes pass after the "
                f"analyzer repair; "
                f"{len(repaired)} parameter(s) repaired "
                f"({', '.join(repaired) or 'no entry changed'})")
            return None
        # Honest fix attempt that did not converge: keep the retried spec
        # and params (they moved the errors), rebuild the finding from the
        # fresh stderr on the next iteration.

    return halt(paths, stage_id,
                reason=(f"validate_params_output.py failed after "
                        f"{_PARAMS_PROVENANCE_RETRY_CAP} analyzer "
                        f"provenance-repair retries (params provenance "
                        f"fix loop exhausted)"),
                halt_class="producer_output_invalid",
                context={"stderr": err},
                retry_count=_PARAMS_PROVENANCE_RETRY_CAP, state=state)


def _read_pipeline_json_text(text: str | None) -> dict | None:
    if text is None:
        return None
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def run_stage_2x(state: PipelineState) -> StageResult:
    """2.x — parameter-deriver. Pure-script for the producer + semantic
    review via stage-reviewer. Per the archived v2 orchestrator spec (internal, not shipped), since params.json is
    deterministic-produced, reviewer findings halt rather than retry —
    there's no LLM-driven fix path for a script. One exception (R2C-033):
    a provenance-probe failure at the deterministic gate routes a
    mechanical finding back to the METHOD ANALYZER via
    `_run_params_provenance_fix_loop` — the dishonest provenance lives in
    the spec, not the script — bounded, scope-diffed, probes re-run as
    the terminal gate."""
    stage_id = "stage_2x"
    paths = state.paths
    skipped = _skip_if_done(paths, stage_id, [
        paths.pipeline_dir / "params.json",
        paths.pipeline_dir / "stage_review_stage_2x_params.json",
    ])
    if skipped is not None:
        return skipped
    proc = run_script(
        stage_id,
        ["scripts/derive_params.py", "--spec", str(paths.method_spec),
         "--run-dir", str(paths.run_dir)],
        timeout=60,
    )
    if proc.returncode != 0:
        return halt(paths, stage_id,
                    reason=f"derive_params.py exit {proc.returncode}",
                    halt_class="internal_contract_violation",
                    context={"stderr": (proc.stderr or proc.stdout)[-2000:]},
                    state=state)
    ok, err = _run_stage2_script(
        state, stage_id=stage_id,
        args=["scripts/validate_params_output.py"],
    )
    # P2: a US-3 `source=paper` over-claim is relabel-eligible, not terminal.
    # Defer it to the stage-reviewer + relabel auto-resolver below instead of
    # halting here; the post-resolve re-validation is its terminal gate. Any
    # other validator error still halts immediately.
    deferred_overclaim = False
    if not ok:
        if _params_overclaim_only(err):
            deferred_overclaim = True
            log(stage_id, "params_overclaim_deferred",
                "validate_params_output.py reported only US-3 source=paper "
                "over-claim(s); deferring to stage-reviewer relabel "
                "auto-resolve instead of halting (re-validated after resolve)")
        else:
            # R2C-033 (approved 2026-07-28): probe-class provenance
            # failures get a bounded analyzer fix loop before the terminal
            # halt — the pdwa d_safe halt cost the runnable delivery on a
            # one-turn-fixable labeling error. Non-probe failures (schema,
            # missing params, crashes) keep today's immediate halt inside
            # the helper.
            loop_halt = _run_params_provenance_fix_loop(
                state, stage_id, err)
            if loop_halt is not None:
                return loop_halt

    # Per-dataset value binding disclosure (per-dataset-value-binding-design
    # .md): a clean bind is a run event; an unbindable map falls back to the
    # spec's primary-dataset scalar with one assumptions.md entry per param.
    _log_per_dataset_binding(state, stage_id)

    # Deterministic scale calibration (hardening item 1, 2026-07-02): declared
    # scale-dependent params get reconciled with the scale load_data actually
    # produces, BEFORE the reviewer (whose preservation bias certified the
    # unrescaled value on the 2026-07-02 run) and before budget sufficiency.
    _apply_scale_calibration(state, stage_id)

    # Arm A (queue 11d) — training-budget sufficiency. params (just derived) and
    # the built package (finalized at 2.d) both exist now, so this is the
    # earliest fail-fast point before notebook generation. Auto-raises
    # max_epochs and logs an assumption when the derived budget would starve
    # training; runs before the reviewer so it sees the corrected budget.
    budget_halt = _check_training_budget_sufficiency(state, stage_id)
    if budget_halt is not None:
        return budget_halt

    try:
        _dispatch_stage_reviewer(state, "stage_2x_params")
    except (OpencodeClientError, OutOfScopeWritesError) as e:
        return halt(paths, stage_id,
                    reason=f"stage_2x_params reviewer dispatch failed: {e}",
                    halt_class=_dispatch_error_halt_class(e),
                    state=state)
    review_path = paths.pipeline_dir / "stage_review_stage_2x_params.json"
    if not review_path.exists():
        # Missing-output retry — Think-class stop-early recovery.
        log(stage_id, "stage_reviewer_missing_output_retry",
            "initial stage_2x_params reviewer dispatch produced no "
            "stage_review_stage_2x_params.json; retrying with stronger prompt")
        try:
            _dispatch_stage_reviewer_retry(state, "stage_2x_params")
        except (OpencodeClientError, OutOfScopeWritesError) as e:
            return halt(paths, stage_id,
                        reason=f"stage_2x_params reviewer retry dispatch failed: {e}",
                        halt_class=_dispatch_error_halt_class(e),
                        state=state)
        if not review_path.exists():
            return halt(paths, stage_id,
                        reason="stage_2x_params reviewer did not write its findings file "
                               "(even after one missing-output retry)",
                        halt_class="producer_wrote_nothing",
                        state=state)
    review, read_err = _read_review_json_or_err(
        review_path, kind="stage_2x_params review",
        expected_stage_id="stage_2x_params",
    )
    if read_err:
        return halt(paths, stage_id,
                    reason=f"stage_2x_params reviewer output unusable: {read_err}",
                    halt_class="producer_output_invalid",
                    state=state)
    findings = review.get("findings", []) or []

    # Auto-resolve findings whose reviewer attached a structured
    # `proposed_resolution`. Today this covers `relabel_param_source` (for
    # `paper_source_values_match_paper` violations). Edits land in
    # params.json + are logged to <run_dir>/assumptions.md.
    review_findings = list(findings)
    findings, auto_applied_ids, auto_failed_ids = (
        _try_auto_resolve_stage_review_findings(state, stage_id, findings)
    )
    applied_locations = {
        str(f.get("location") or "") for f in review_findings
        if str(f.get("id")) in {str(i) for i in auto_applied_ids}
    }
    if auto_applied_ids or deferred_overclaim:
        # Re-run the deterministic validator as the terminal provenance gate.
        # Two reasons reach here: (a) an auto-resolve edit landed in params.json
        # (belt-and-suspenders schema re-check — the applier already did a
        # per-edit pydantic check and would have reverted on failure), or (b) a
        # US-3 over-claim was deferred from the first gate and must now be
        # confirmed cleared (relabelled). If it still fails, halt — no
        # over-claim ships.
        ok, err = _run_stage2_script(
            state, stage_id=stage_id,
            args=["scripts/validate_params_output.py"],
        )
        reask_apply_failed_params: set[str] = set()
        reask_needs_user_params: set[str] = set()
        if not ok and deferred_overclaim and _params_overclaim_only(err):
            # The pdwa 2026-07-07 seam: a deferred over-claim survives the
            # reviewer pass whenever the reviewer flagged DIFFERENT params —
            # nobody ever proposed a relabel for the deferred one, and the
            # ordinary re-ask below never sees it because it only walks
            # reviewer findings. Synthesize reviewer-shaped findings from the
            # validator's own bullets and give the reviewer ONE targeted
            # re-ask before halting. Params an applied resolution claimed to
            # fix are excluded — those are genuine inconsistencies and halt.
            synthesized = _synthesize_deferred_overclaim_findings(
                err, applied_locations)
            if synthesized:
                log(stage_id, "deferred_overclaim_reask",
                    f"{len(synthesized)} deferred US-3 over-claim(s) got no "
                    "resolution from the reviewer pass; synthesizing "
                    "finding(s) from the validator bullets for one targeted "
                    "re-ask: "
                    + ", ".join(f["id"] for f in synthesized))
                reask_remaining, reask_applied_ids, reask_failed_ids = (
                    _reask_unstructured_blocking_resolutions(
                        state, stage_id, "stage_2x_params", synthesized))
                reask_apply_failed_params = {
                    _finding_param_key(f.get("location"))
                    for f in synthesized
                    if str(f.get("id")) in {str(i) for i in reask_failed_ids}
                }
                # An explicit needs_user answer on the re-ask is a deliberate
                # reviewer decision, not a failed fix — remembered so the
                # halt below can carry the needs_user_input class (the
                # reserved catalog class, unused until now).
                synthesized_ids = {str(f.get("id")) for f in synthesized}
                reask_needs_user_params = {
                    _finding_param_key(f.get("location"))
                    for f in reask_remaining
                    if str(f.get("id")) in synthesized_ids
                    and f.get("resolution_status") == "needs_user"
                }
                if reask_applied_ids:
                    auto_applied_ids = [*auto_applied_ids,
                                        *reask_applied_ids]
                    applied_locations |= {
                        str(f.get("location") or "") for f in synthesized
                        if str(f.get("id"))
                        in {str(i) for i in reask_applied_ids}
                    }
                    ok, err = _run_stage2_script(
                        state, stage_id=stage_id,
                        args=["scripts/validate_params_output.py"],
                    )
        if not ok:
            still_failing = [
                name for name, _ in _params_overclaim_bullets(err)]
            reviewer_escalated = bool(still_failing) and all(
                name in reask_needs_user_params for name in still_failing)
            return halt(
                paths, stage_id,
                reason=_post_resolve_halt_reason(
                    err, applied_locations, reask_apply_failed_params,
                    needs_user_params=reask_needs_user_params),
                halt_class=("needs_user_input" if reviewer_escalated
                            else "producer_output_invalid"),
                context={"stderr": err, "auto_applied": auto_applied_ids},
                state=state)

    # Direction-aware severity: after auto-resolve has had its chance to fix
    # labels properly, remaining SAFE-direction (under-claim) provenance
    # findings route to assumptions.md instead of halting. Over-claims and
    # everything else still reach the halt check unchanged.
    findings, deferred_ids = _defer_safe_provenance_findings(
        state, stage_id, findings)

    # Disclosed demo-scale trade-offs on non-paper defaults (e.g. the AL
    # budget-to-pool ratio above the taxonomy smoke-economics floor) route to
    # assumptions.md instead of halting an unattended run; the value is a
    # disclosed system default, not a paper claim, so nothing dishonest ships.
    findings, demo_scale_deferred_ids = _defer_disclosed_demo_scale_findings(
        state, stage_id, findings)

    reask_applied_ids: list[str] = []
    if should_halt_stage(findings):
        # ONE reviewer re-ask for blocking findings without a structured
        # resolution (2026-07-02 GBALD re-roll case: relabel-shaped finding
        # with prose-only proposed_fix), then re-check before halting.
        findings, reask_applied_ids, reask_failed_ids = (
            _reask_unstructured_blocking_resolutions(
                state, stage_id, "stage_2x_params", findings))
        auto_applied_ids = [*auto_applied_ids, *reask_applied_ids]
        auto_failed_ids = [*auto_failed_ids, *reask_failed_ids]
        if reask_applied_ids:
            # Same terminal provenance gate as the first auto-resolve pass.
            ok, err = _run_stage2_script(
                state, stage_id=stage_id,
                args=["scripts/validate_params_output.py"],
            )
            if not ok:
                return halt(
                    paths, stage_id,
                    reason=("validate_params_output.py failed after "
                            "re-asked auto-resolve (internal inconsistency "
                            "between applier and validator)"),
                    halt_class="internal_contract_violation",
                    context={"stderr": err, "auto_applied": auto_applied_ids},
                    state=state)

    if should_halt_stage(findings):
        # Count only the halt-driving severities: nice-to-have findings ride
        # along in the artifact but must not inflate the reason (the
        # bev-distill 2026-07-01 halt said "2 important-or-above" for 1
        # important + 1 nice-to-have).
        n_blocking = len(critical_findings(findings)) + len(important_findings(findings))
        return halt(paths, stage_id,
                    reason=f"stage_2x_params review found {n_blocking} important-or-above finding(s) "
                           f"that could not be auto-resolved",
                    halt_class="judge_halt",
                    findings=findings,
                    context={"target_script": "scripts/derive_params.py",
                             "auto_applied": auto_applied_ids,
                             "auto_failed": auto_failed_ids,
                             "note": "deterministic producer — auto-resolve runs first "
                                     "for findings with a structured proposed_resolution "
                                     "(plus one reviewer re-ask for blocking findings "
                                     "without one); remaining findings need a manual "
                                     "derive_params.py fix"},
                    state=state)
    notes = "derive_params + validator + review clean"
    extras = []
    if auto_applied_ids:
        extras.append(f"auto-resolved {len(auto_applied_ids)} finding(s)")
    if deferred_ids:
        extras.append(f"{len(deferred_ids)} safe-direction provenance "
                      f"finding(s) routed to assumptions.md")
    if demo_scale_deferred_ids:
        extras.append(f"{len(demo_scale_deferred_ids)} disclosed demo-scale "
                      f"trade-off(s) routed to assumptions.md")
    if extras:
        notes = f"derive_params + validator + review ({'; '.join(extras)})"
    return StageResult(status="completed", stage_id=stage_id, notes=notes)


# ---------------------------------------------------------------------------
# Stage 3 — notebook generation (3.a notebook-generator + render + validate
#           + review; 3.b render checkpoint; 3.c smoke gate)
# ---------------------------------------------------------------------------


def _dispatch_notebook_generator(state: PipelineState) -> DispatchResult:
    agent = "r2c-notebook-generator"
    bundle_section = _notebook_bundle_section(state.paths)
    prompt = build_dispatch_prompt(
        task_summary=STAGE_TASK_SUMMARIES["stage_3a_notebook"],
        paths=build_paths_block(state.paths),
        writeable_paths=WRITEABLE_PATHS[agent],
        extra_sections=[bundle_section] if bundle_section else None,
        # Cap-burn fix 1: composes the largest single artifact of the run
        # (a genuine no-write casualty on the M-LOAD-1 matrix).
        writing_discipline=True,
    )
    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt=prompt, timeout_s=NOTEBOOK_GEN_TIMEOUT_S,
        recovery_check_fn=_render_and_validate_notebook,
    )


def _dispatch_notebook_generator_fix(
    state: PipelineState, findings: list[dict],
    *, smoke_context: dict | None = None,
) -> DispatchResult:
    agent = "r2c-notebook-generator"
    prompt = build_fix_mode_prompt(
        target_agent=agent,
        findings=findings,
        paths=build_paths_block(state.paths),
        writeable_paths=WRITEABLE_PATHS[agent],
        smoke_context=smoke_context,
    )
    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt=prompt, timeout_s=NOTEBOOK_GEN_TIMEOUT_S,
        recovery_check_fn=_render_and_validate_notebook,
    )


# Late registration — _dispatch_notebook_generator_fix is now defined.
JUDGE_FIX_DISPATCHERS["r2c-notebook-generator"] = _dispatch_notebook_generator_fix


def _render_notebook(state: PipelineState) -> tuple[bool, str]:
    """Run render_notebook.py. Returns (ok, err_tail). Idempotent — safe to
    call multiple times; output is fully derived from notebook_draft.py + the
    method/ package."""
    proc = run_script(
        "stage_3", ["scripts/render_notebook.py", "--run-dir", str(state.paths.run_dir)],
        timeout=60,
    )
    err_tail = _stderr_excerpt(proc.stderr.strip() or proc.stdout.strip())
    ok = proc.returncode == 0
    _append_validation_event(
        state.paths,
        stage_id="stage_3a",
        validator="render_notebook.py",
        ok=ok,
        stderr_tail=err_tail,
        artifacts=[str(state.paths.run_dir / "notebook.ipynb")],
    )
    return ok, err_tail


def _render_and_validate_notebook(state: PipelineState) -> tuple[bool, str]:
    """Stage 3.a's validator step: render the draft, then run the static
    notebook validator, then re-verify the method/ package still imports
    cleanly. Any of the three failing means the notebook-generator's draft
    (or a stray edit it made under method/) is broken — same producer to
    dispatch, same fix-mode shape. Returns the first failure's stderr tail.

    The package-import re-check is belt-and-suspenders alongside the
    out-of-scope-write enforcement: if the agent somehow lands an edit that
    sneaks past the file-ownership check (e.g., the agent's writeable_paths
    legitimately include a file the change shouldn't have touched, or a
    future agent type isn't yet in `WRITEABLE_PATHS`), this re-runs the
    `from method import *` smoke that Stage 2.d's validator runs, but at
    Stage 3.a's boundary."""
    ok, err = _render_notebook(state)
    if not ok:
        return False, f"render_notebook.py failed:\n{err}"
    ok, err = _run_stage2_script(
        state, stage_id="stage_3a",
        args=["scripts/validate_notebook_output.py"],
    )
    if not ok:
        return False, err
    ok, err = _run_stage2_script(
        state, stage_id="stage_3a",
        args=["scripts/validate_package_imports.py"],
    )
    if not ok:
        return False, err
    # Lint gate before any smoke execution (US-10, wired 2026-06-10): a
    # NameError or missing import in the draft costs seconds here vs a
    # 5–11 minute smoke dispatch + diagnostician at stage 3.c — two of the
    # june9 run's four smoke iterations were exactly this class. Failures
    # inherit the same judge-routed fix loop as the other chain links.
    return _run_lint_gate(state, stage_id="stage_3a")


def _run_lint_gate(state: PipelineState, *, stage_id: str) -> tuple[bool, str]:
    """Run lint_generated_code.py over method/*.py + notebook_draft.py.
    Returns (ok, error_tail) in the validator-chain shape; appends the same
    validation event the stage-2-style scripts do."""
    paths = state.paths
    proc = run_script(
        stage_id,
        ["scripts/lint_generated_code.py", "--run-dir", str(paths.run_dir)],
        timeout=120,
    )
    err_tail = _stderr_excerpt(proc.stderr.strip() or proc.stdout.strip())
    ok = proc.returncode == 0
    _append_validation_event(
        paths,
        stage_id=stage_id,
        validator="lint_generated_code.py",
        ok=ok,
        stderr_tail=err_tail,
    )
    return ok, err_tail


def _reconcile_requirements_with_notebook(
    state: PipelineState, stage_id: str = "stage_3a",
) -> None:
    """Best-effort: re-derive requirements.txt to also cover the rendered
    notebook's third-party imports. requirements.txt is frozen at Stage 2.d
    from the method package, but the Stage 3.a notebook can import libraries
    the package never used (e.g. scikit-learn for a plot), which would break a
    clean `pip install -r requirements.txt` + run (the F001 draft-demotion
    class, root-caused 2026-06-15). Idempotent; never halts — the paper-fidelity
    reviewer remains the backstop if this is skipped. Called from every path
    where a notebook exists and the run continues: stage 3.a completion/skip/
    degrade, and stage 3.c continue-exits (its fix loop can rewrite the
    notebook — and add imports — after 3.a's reconcile already ran)."""
    try:
        changed, added = reconcile_requirements_for_notebook(
            state.paths.run_dir, state.paths.method_spec,
        )
    except Exception as e:  # noqa: BLE001 — quality step, must never break the stage
        log(stage_id, "requirements_reconcile_failed",
            f"could not reconcile requirements.txt with notebook imports: "
            f"{type(e).__name__}: {e}")
        return
    if changed:
        log(stage_id, "requirements_reconciled",
            f"added notebook-only dependencies to requirements.txt: {added}")
        _append_run_event(
            state.paths, "requirements_reconciled",
            stage_id=stage_id,
            summary=f"requirements.txt updated to cover {len(added)} notebook import(s)",
            details={"added": added},
        )


_ACCEPTED_NOTEBOOK_VALIDATION_FILE = "notebook_validation_accepted.json"


def _notebook_validation_failure_bullets(stderr: str) -> list[str]:
    """Whitespace-normalized per-failure bullets from a
    validate_notebook_output.py failure (`FAIL: N validation error(s):`
    followed by one `  - <error>` line per failure)."""
    bullets: list[str] = []
    for line in (stderr or "").splitlines():
        normalized = " ".join(line.split())
        if normalized.startswith("- "):
            bullets.append(normalized[2:])
    return bullets


def _record_accepted_notebook_validation(state: PipelineState) -> None:
    """Persist the static notebook validator's current failing bullets as
    the ACCEPTED baseline when stage 3a degrades past them.

    The ICRA 2026-07-13 halt: the 3a judge examined a static validation
    failure (a private-name import), classified it as an upstream issue,
    and the run degraded and continued BY DESIGN (notebook quality is
    best-effort) — then the 3c post-fix revalidation re-ran the same
    validator and halted the run on the same already-accepted failure,
    right after an unrelated smoke fix had landed correctly. A fix should
    be judged on whether it made the validation state WORSE, not on
    whether it cleared findings the run had already accepted.

    Runs the validator once (plain run_script, not _run_stage2_script, so
    baseline bookkeeping does not append a second validation event) and
    writes the failing bullets to `.pipeline/notebook_validation_accepted
    .json`. A passing validator writes an empty baseline (the degrade was
    for reviewer-side reasons and every later failure stays halt-worthy)."""
    paths = state.paths
    proc = run_script(
        "stage_3a",
        ["scripts/validate_notebook_output.py",
         "--spec", str(paths.method_spec), "--run-dir", str(paths.run_dir)],
        timeout=120,
    )
    err_tail = _stderr_excerpt(proc.stderr.strip() or proc.stdout.strip())
    accepted = ([] if proc.returncode == 0
                else _notebook_validation_failure_bullets(err_tail))
    (paths.pipeline_dir / _ACCEPTED_NOTEBOOK_VALIDATION_FILE).write_text(
        json.dumps({
            "schema_version": "1.0",
            "validator": "validate_notebook_output.py",
            "recorded_at_stage": "stage_3a",
            "accepted_failures": accepted,
        }, indent=2) + "\n",
        encoding="utf-8")
    if accepted:
        log("stage_3a", "notebook_validation_baseline_recorded",
            f"stage_3a degraded past {len(accepted)} static validation "
            f"failure(s); later notebook revalidations halt only on NEW "
            f"failures")
        _append_run_event(
            paths,
            "notebook_validation_baseline_recorded",
            stage_id="stage_3a",
            summary=(f"stage_3a degraded past {len(accepted)} static "
                     f"validation failure(s); revalidations halt only on "
                     f"new failures"),
            details={"accepted_failures": accepted},
        )


def _notebook_validation_new_failures(
    state: PipelineState, stderr: str,
) -> list[str]:
    """The failing bullets in `stderr` that are NOT in the stage-3a
    accepted baseline — the ones a revalidation is allowed to halt on.

    No baseline file (stage 3a passed clean, or a pre-baseline run dir)
    means every failure is new: behavior identical to before the baseline
    existed. A failure with no parseable bullets is returned whole so it
    always halts. Both the baseline and the live stderr ride 2000-char
    tails, so truncation can only mis-tag an accepted failure as new —
    an unnecessary halt, never a silent pass of a genuinely new one."""
    current = _notebook_validation_failure_bullets(stderr)
    if not current:
        summary = " ".join((stderr or "").split())[:300]
        return [summary or "validator failed with no parseable failure list"]
    path = state.paths.pipeline_dir / _ACCEPTED_NOTEBOOK_VALIDATION_FILE
    if not path.is_file():
        return current
    try:
        accepted = set(json.loads(path.read_text(encoding="utf-8"))
                       .get("accepted_failures") or [])
    except (OSError, json.JSONDecodeError, AttributeError):
        return current
    return [bullet for bullet in current if bullet not in accepted]


def _log_accepted_validation_continue(
    state: PipelineState, stage_id: str, where: str,
) -> None:
    """Log + event for a revalidation that failed ONLY on baseline-accepted
    failures and therefore continues. The event matters for audits: the
    validator's own validation_failed event lands right before this, and
    without the explaining event a failed-validation-then-no-halt sequence
    reads as a driver bug."""
    message = (f"static validator failures are all within the stage-3a "
               f"accepted baseline; continuing {where}")
    log(stage_id, "accepted_validation_failures_only", message)
    _append_run_event(
        state.paths,
        "accepted_validation_failures_only",
        stage_id=stage_id,
        summary=message,
    )


def run_stage_3a(state: PipelineState) -> StageResult:
    """3.a — notebook-generator produces notebook_draft.py (jupytext-percent),
    then we render it and run the static validator. Stage-reviewer runs the
    semantic checks (most importantly `al_loop_evaluates_each_round`, which
    catches the placeholder-update failure mode that smoke gates can't see).
    Cap=3 fix-mode retries via run_fix_loop. On a clean pass (or skip), the
    notebook now exists, so we reconcile requirements.txt to cover its imports.

    On a DEGRADE, the static validator's failing state is persisted as the
    accepted baseline (see _record_accepted_notebook_validation) so later
    notebook revalidations judge fixes on whether they made things WORSE
    instead of re-litigating findings this stage already accepted."""
    stage_id = "stage_3a"
    paths = state.paths
    skipped = _skip_if_done(paths, stage_id, [
        paths.pipeline_dir / "notebook_draft.py",
        paths.run_dir / "notebook.ipynb",
        paths.pipeline_dir / "stage_review_stage_3a_notebook.json",
    ])
    if skipped is not None:
        _reconcile_requirements_with_notebook(state)
        return skipped
    if (paths.pipeline_dir / "notebook_draft.py").exists():
        log(stage_id, "existing_notebook_draft",
            "notebook_draft.py exists without a clean stage_3a sentinel; "
            "validating/reviewing it before redispatch")
    else:
        try:
            _dispatch_notebook_generator(state)
        except (OpencodeClientError, OutOfScopeWritesError) as e:
            return halt(paths, stage_id,
                        reason=f"notebook-generator dispatch failed: {e}",
                        halt_class=_dispatch_error_halt_class(e),
                        state=state)
    loop_halt = run_fix_loop(
        state=state, stage_id=stage_id,
        fix_dispatch_fn=_dispatch_notebook_generator_fix,
        validator_fn=_render_and_validate_notebook,
        validator_label="render_notebook.py + validate_notebook_output.py",
        reviewer_stage_id="stage_3a_notebook",
        use_judge=True,
    )
    if loop_halt is not None:
        if loop_halt.status == "degraded":
            # Cap-exhaustion degrade: the notebook exists and the run CONTINUES
            # to delivery, so its imports must be covered just like on the
            # clean-completion path (O2 reconcile gap, repo-map verdict 2026-07-04).
            _reconcile_requirements_with_notebook(state)
            _record_accepted_notebook_validation(state)
        return loop_halt
    _reconcile_requirements_with_notebook(state)
    # A clean pass supersedes any baseline from an earlier degraded attempt:
    # the validator is green, so every later failure is genuinely new.
    (paths.pipeline_dir / _ACCEPTED_NOTEBOOK_VALIDATION_FILE).unlink(
        missing_ok=True)
    return StageResult(status="completed", stage_id=stage_id,
                       notes="notebook-generator + render + validator + review clean")


def run_stage_3b(state: PipelineState) -> StageResult:
    """3.b — render notebook_draft.py to notebook.ipynb. Pure-script,
    idempotent. After 3.a's fix loop, notebook.ipynb is already on disk and
    this stage skips; the stage exists as a checkpoint for partial-run
    workflows (`--stop-after stage_3a` to inspect the draft, then resume)."""
    stage_id = "stage_3b"
    paths = state.paths
    skipped = _skip_if_done(paths, stage_id, [paths.run_dir / "notebook.ipynb"])
    if skipped is not None:
        return skipped
    ok, err = _render_notebook(state)
    if not ok:
        return halt(paths, stage_id,
                    reason="render_notebook.py failed",
                    halt_class="producer_output_invalid",
                    context={"stderr": err},
                    state=state)
    return StageResult(status="completed", stage_id=stage_id,
                       notes="render clean")


def _parse_smoke_fail_cell(stderr: str) -> int | None:
    """Extract the failing cell index from smoke_run_notebook.py's stderr.

    Matches the tail-anchored index repeat first — the header patterns sit
    at the TOP of the failure excerpt and scrolled out of the driver's
    stderr-tail window once the excerpt grew (the detr 2026-07-15 halt:
    "smoke gate failed but could not parse failing cell index") — then the
    four header patterns for older outputs:
      - "(failing cell index: N)"                   (tail-anchored, always last)
      - "notebook execution failed at cell N"      (CellExecutionError)
      - "TIMED OUT at cell N"                       (CellTimeoutError, kernel alive)
      - "Last code cell sent to the kernel: cell N" (kernel died — use last sent)
      - "nbclient was waiting on cell N"            (kernel died fallback)
    """
    for pat in (
        r"failing cell index: (\d+)",
        r"failed at cell (\d+)",
        r"TIMED OUT at cell (\d+)",
        r"Last code cell sent to the kernel:\s*cell\s*(\d+)",
        r"nbclient was waiting on cell (\d+)",
    ):
        m = re.search(pat, stderr)
        if m:
            return int(m.group(1))
    return None


_SMOKE_TIMEOUT_RE = re.compile(
    r"notebook execution TIMED OUT at cell (\d+) "
    r"after (\d+)s \(wall ([\d.]+)s\)"
)


def _smoke_timeout_facts(stderr: str) -> dict | None:
    """Parse the smoke runner's genuine-timeout report into structured
    facts, or None when the failure is not a timeout.

    Item 20 (bev-distill overnight 07-08): a timeout is a first-class
    smoke-failure kind — the cell never raised, so there is no traceback,
    and the fix is cheaper work, not a bug hunt. Every consumer that words
    a researcher surface or shapes a fix dispatch checks this FIRST so
    timeout failures stop masquerading as errors."""
    m = _SMOKE_TIMEOUT_RE.search(stderr or "")
    if not m:
        return None
    return {
        "cell": int(m.group(1)),
        "budget_s": int(m.group(2)),
        "wall_s": float(m.group(3)),
    }


def _smoke_degrade_fields(
    stderr_tail: str, fail_cell: int, section: int | None,
) -> tuple[str, str, str]:
    """(what_failed, where, what_to_do) for a smoke-gate degrade entry,
    worded by failure kind. The error wording promises a traceback; the
    timeout wording must not (there is none — the bev-distill KNOWN_ISSUES
    entry sent the researcher hunting for a traceback that did not exist
    and labeled the cell source \"stderr:\")."""
    where = f"notebook.ipynb — cell {fail_cell}" + (
        f" (section {section})" if section else "")
    timeout = _smoke_timeout_facts(stderr_tail)
    if timeout is None:
        return ("the notebook does not run end-to-end", where,
                _SMOKE_DEGRADE_TODO)
    where += (f" — ran ~{timeout['wall_s']:.0f}s against the "
              f"{timeout['budget_s']}s per-cell budget")
    return ("the notebook does not finish within the smoke time budget",
            where, _SMOKE_TIMEOUT_DEGRADE_TODO)


def _extract_exception_class(stderr: str) -> str | None:
    """Extract the Python exception class name (e.g., 'RuntimeError',
    'TypeError', 'IndexError') from a smoke-gate stderr blob.

    Used by the smoke-loop anti-fixation guard to differentiate two cases
    when iteration N+1 fails at the same cell as iteration N:

      - Same cell, **same** exception class → likely fixation; the prior
        target_file's fix didn't land. Add it to failed_targets so the
        next dispatch routes elsewhere.
      - Same cell, **different** exception class → cascading progress;
        the prior fix landed and exposed a new layer. Don't penalize the
        file; the diagnostician may legitimately need it again.

    The R2C 2026-05-21 bev-distill halt was a false-positive on the old
    guard, which treated 'same cell = fixation' without checking the
    exception. iter 0 caught a RuntimeError (uint8 dtype); iter 1 caught
    a TypeError (single dict vs list[dict]) at the SAME cell. Both bugs
    lived in notebook_draft.py, but they were genuinely different layers.

    Returns the last exception-class token found in the stderr — that's
    the actual raised exception (earlier tokens may be in chained-from
    traceback frames). Returns None if no exception line matches.
    """
    # ANSI escapes from nbclient's rich tracebacks; strip first.
    text = re.sub(r"\x1b\[[0-9;]*m", "", stderr)
    # Match the standard Python exception line: "<Name>Error: ..." or
    # "<Name>Exception: ..." at the start of a line. Last match wins.
    pattern = re.compile(r"^([A-Z]\w*(?:Error|Exception|Warning))\b", re.MULTILINE)
    matches = pattern.findall(text)
    if not matches:
        return None
    return matches[-1]


def _smoke_local_dispatchers(
    *,
    stderr_tail: str,
    failing_cell_source: str,
    fail_cell: int,
    section: int,
    prior_iterations: list[dict] | None,
    failed_targets: list[str] | None,
) -> dict[str, callable]:
    """Build the local_dispatchers map for `_invoke_judge` at stage 3.c.

    Two flavors of context-bound closure:

    1. **Smoke-routable producers** (architecture-coder, method-coder,
       notebook-generator) — wrapped to inject `smoke_context` with the
       deepest-owned-frame anchor. Matches what the in-loop smoke fix path
       does, so a judge-routed cap-recovery dispatch reaches the producer
       with the same anchor it would have gotten via the diagnostician.

    2. **Smoke-diagnostician** — wrapped to plumb the smoke context plus
       a `repair_findings` (schema-invalid case) or `retry_mode` (missing-
       file case) flag based on the on-disk diagnosis file's existence at
       dispatch time. Used by the diagnostician-output-validation path
       (see run_stage_3c above) AND available at cap-exhaustion in case
       the judge decides the diagnostician is itself the producer that
       needs re-emitting.
    """
    def _wrap_producer(fix_fn, agent_name: str):
        def _dispatcher(s: PipelineState, findings: list[dict]):
            producer_writeable = WRITEABLE_PATHS[agent_name]
            anchor = smoke_traceback_deepest_owned_frame(stderr_tail, producer_writeable)
            return fix_fn(s, findings, smoke_context={"traceback_hint": anchor or ""})
        return _dispatcher

    def _diagnostician_dispatcher(s: PipelineState, findings: list[dict]):
        file_exists = (
            s.paths.pipeline_dir / "smoke_diagnosis.json"
        ).is_file()
        return _dispatch_smoke_diagnostician(
            s,
            stderr_tail=stderr_tail,
            failing_cell_source=failing_cell_source,
            failing_cell_index=fail_cell,
            section=section,
            prior_iterations=prior_iterations if prior_iterations else None,
            failed_targets=failed_targets if failed_targets else None,
            repair_findings=findings if file_exists else None,
            retry_mode=not file_exists,
        )

    return {
        "r2c-architecture-coder": _wrap_producer(
            _dispatch_arch_coder_fix, "r2c-architecture-coder",
        ),
        "r2c-method-coder": _wrap_producer(
            _dispatch_method_coder_fix, "r2c-method-coder",
        ),
        "r2c-notebook-generator": _wrap_producer(
            _dispatch_notebook_generator_fix, "r2c-notebook-generator",
        ),
        "r2c-smoke-diagnostician": _diagnostician_dispatcher,
    }


def _smoke_producer_to_fix_fn(producer: str):
    """Map a smoke-cell-to-producer routing decision to the actual fix-mode
    dispatch function. The three producers that can be routed to from a
    smoke-gate failure (per route_findings.SECTION_TO_PRODUCER) all expose a
    fix-mode dispatcher."""
    table = {
        "notebook-generator": _dispatch_notebook_generator_fix,
        "method-coder": _dispatch_method_coder_fix,
        "architecture-coder": _dispatch_arch_coder_fix,
    }
    if producer not in table:
        raise ValueError(f"smoke-gate routed to unknown producer: {producer!r}")
    return table[producer]


def _smoke_producer_to_writeable_paths(producer: str) -> list[str]:
    """Map a smoke-routable producer short-name to its WRITEABLE_PATHS entry.

    Used by the smoke-gate fix path to compute the deepest-owned-frame
    traceback anchor — the L1 smoke-fix-mode guidance needs to tell the
    agent which specific frame in its allowlist is closest to the failure."""
    table = {
        "notebook-generator": WRITEABLE_PATHS["r2c-notebook-generator"],
        "method-coder": WRITEABLE_PATHS["r2c-method-coder"],
        "architecture-coder": WRITEABLE_PATHS["r2c-architecture-coder"],
    }
    if producer not in table:
        raise ValueError(f"smoke-gate routed to unknown producer: {producer!r}")
    return table[producer]


def _extract_failing_cell_source(notebook_json: dict, fail_cell_index: int) -> str:
    """Return the source of the cell at index `fail_cell_index` in the
    rendered notebook. Walks code cells only — the smoke gate's cell index
    counts code cells, matching nbclient's enumeration. If the index is
    out of range, returns an empty string (caller falls back to no cell
    source in the diagnostician prompt)."""
    code_cells = [c for c in notebook_json.get("cells", []) if c.get("cell_type") == "code"]
    if 0 <= fail_cell_index < len(code_cells):
        cell = code_cells[fail_cell_index]
        return "".join(cell.get("source", []))
    return ""


def _refresh_data_flow_context(state: PipelineState) -> Path | None:
    """Run the data-flow extractor over the package + notebook, writing
    `.pipeline/data_flow.json`. Called just-in-time before each
    diagnostician dispatch so the structured value-origin context is fresh
    against the latest notebook_draft.py + method/*.py.

    Returns the output path on success, None on failure (the diagnostician
    falls back to reading files manually). Non-fatal — extraction errors
    are logged but don't halt the pipeline."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts.extract_data_flow import extract  # noqa: PLC0415
    except ImportError as e:
        log("data_flow", "extract_import_failed", f"could not import extract_data_flow: {e}")
        return None
    try:
        result = extract(state.paths.run_dir)
    except Exception as e:
        log("data_flow", "extract_failed",
            f"extraction raised {type(e).__name__}: {e}; diagnostician will fall back to reading files")
        return None
    out_path = state.paths.pipeline_dir / "data_flow.json"
    try:
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except OSError as e:
        log("data_flow", "extract_write_failed", f"could not write {out_path}: {e}")
        return None
    n_symbols = len(result.get("symbols") or {})
    n_files = len(result.get("files_scanned") or [])
    log("data_flow", "extracted",
        f"{n_files} files, {n_symbols} symbols → {out_path.name}")
    return out_path


def _slice_data_flow_for_diagnosis(
    state: PipelineState,
    data_flow_path: Path,
    *,
    failing_cell_source: str,
    stderr_tail: str,
) -> Path | None:
    """Write `.pipeline/data_flow_slice.json`: the symbols named in the
    failing cell + traceback, plus one upstream hop through their
    assignment expressions (item: reviewer/diagnostician input hygiene —
    detr 2026-07-14 read the whole 477KB data_flow.json and burned its
    output caps). The full map stays on disk as the fallback reference.
    Best-effort: returns None on any failure or when no symbol matches."""
    try:
        payload = json.loads(Path(data_flow_path).read_text(encoding="utf-8"))
        symbols = payload.get("symbols") or {}
        idents = set(re.findall(
            r"[A-Za-z_][A-Za-z0-9_]*",
            f"{failing_cell_source}\n{stderr_tail}"))
        seeds = sorted(idents & set(symbols))
        if not seeds:
            return None
        selected = set(seeds)
        for name in seeds:
            for assignment in (symbols[name].get("assignments") or []):
                for ident in re.findall(
                        r"[A-Za-z_][A-Za-z0-9_]*",
                        str(assignment.get("expr") or "")):
                    if ident in symbols:
                        selected.add(ident)
        slice_payload = {
            "schema_version": payload.get("schema_version"),
            "sliced_from": Path(data_flow_path).name,
            "seed_symbols": seeds,
            "symbols": {name: symbols[name] for name in sorted(selected)},
        }
        out_path = state.paths.pipeline_dir / "data_flow_slice.json"
        out_path.write_text(
            json.dumps(slice_payload, indent=2) + "\n", encoding="utf-8")
        log("data_flow", "sliced",
            f"{len(seeds)} seed / {len(selected)} total symbols "
            f"→ {out_path.name}")
        return out_path
    except Exception as e:  # noqa: BLE001 - hygiene is best-effort
        log("data_flow", "slice_failed", f"{type(e).__name__}: {e}")
        return None


def _dispatch_smoke_diagnostician(
    state: PipelineState,
    *,
    stderr_tail: str,
    failing_cell_source: str,
    failing_cell_index: int,
    section: int,
    prior_iterations: list[dict] | None = None,
    failed_targets: list[str] | None = None,
    retry_mode: bool = False,
    forbidden_pick_retry: str | None = None,
    repair_findings: list[dict] | None = None,
) -> DispatchResult:
    """Dispatch the smoke-gate diagnostician (Think-class). Writes
    `.pipeline/smoke_diagnosis.json` with target_agent + root_cause +
    proposed_fix. The driver consumes that on return.

    `prior_iterations` (when non-empty) carries the structured iteration
    history of THIS smoke fix loop so the diagnostician at iteration N can
    distinguish regression from new-layer-exposed from misroute. Without
    this history each iteration re-derives context from session conversation,
    which is the contamination path that caused iter-2 of the bev-distill
    2026-05-14 run to follow the producer's (a)-(e) skeleton.

    Drift recovery is wired here (Option A from the 2026-05-14 gbald
    investigation). Four layers of prompt-level hardening did not deter
    this agent from continuing past `smoke_diagnosis.json` to Edit
    `method/*.py`. The recovery_check_fn validates that the canonical
    output is on disk and valid; if so, the driver reverts the out-of-
    scope writes from a pre-dispatch content snapshot and continues. The
    diagnosis (the actual point of the dispatch) is preserved; the side
    effects are not."""
    agent = "r2c-smoke-diagnostician"
    diagnosis_path = state.paths.pipeline_dir / "smoke_diagnosis.json"
    # (a) — refresh the data-flow extractor's output BEFORE building the
    # dispatch prompt so the diagnostician has the freshest structured
    # value-origin context. Cost ~50-100ms per dispatch; non-fatal on
    # failure (diagnostician falls back to reading files manually).
    data_flow_path = _refresh_data_flow_context(state)
    data_flow_slice_path = (
        _slice_data_flow_for_diagnosis(
            state, data_flow_path,
            failing_cell_source=failing_cell_source,
            stderr_tail=stderr_tail,
        )
        if data_flow_path is not None else None
    )
    # Schema repair is a pure merge: the prior JSON rides the prompt so the
    # repair turn's first (and only) action is the corrected Write — the
    # 07-13 ICRA repair turn spent itself re-reading inputs and died with
    # nothing written.
    prior_diagnosis_json = None
    if repair_findings and diagnosis_path.is_file():
        try:
            prior_diagnosis_json = diagnosis_path.read_text(
                encoding="utf-8")[:8000]
        except OSError:
            prior_diagnosis_json = None
    prompt = build_smoke_diagnosis_prompt(
        paths=build_paths_block(state.paths),
        stderr_tail=stderr_tail,
        failing_cell_source=failing_cell_source,
        failing_cell_index=failing_cell_index,
        section=section,
        diagnosis_output_path=str(diagnosis_path),
        data_flow_path=str(data_flow_path) if data_flow_path else None,
        data_flow_slice_path=(
            str(data_flow_slice_path) if data_flow_slice_path else None),
        prior_iterations=prior_iterations,
        failed_targets=failed_targets,
        retry_mode=retry_mode,
        forbidden_pick_retry=forbidden_pick_retry,
        repair_findings=repair_findings,
        prior_diagnosis_json=prior_diagnosis_json,
    )

    def _diagnosis_valid(s: PipelineState) -> bool:
        diag, err = _read_smoke_diagnosis(s)
        return diag is not None and err is None

    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt=prompt,
        timeout_s=DIAGNOSTICIAN_TIMEOUT_S,
        recovery_check_fn=_diagnosis_valid,
    )


def _close_truncated_json(text: str) -> str | None:
    """Append the closers a truncated JSON document is missing (an unclosed
    string, then the open brace/bracket stack). Returns None on a structural
    impossibility (a closer with no opener — mid-file corruption, not
    truncation)."""
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return None
            stack.pop()
    out = text
    if in_string:
        out += '"'
    out = out.rstrip()
    if out.endswith(":"):
        out += " null"
    if out.endswith(","):
        out = out[:-1]
    while stack:
        out += "}" if stack.pop() == "{" else "]"
    return out


def _salvage_truncated_json(raw: str) -> dict | None:
    """Parse a JSON object whose TAIL was truncated mid-write — the
    Think-class truncation shape: a valid prefix cut inside a string, after
    a value, or before the closing braces (detr-distill 2026-07-03: two
    smoke-diagnostician outputs in a row ended at EOF one `}` short, and
    each cost the run a halt).

    Strictly scoped to truncation: the parse error must point at (or
    within a few chars of) end-of-input, else this returns None and the
    caller reports the original error. Recovery closes the open
    string/bracket stack; if that alone doesn't parse, it trims back to
    the previous structural boundary (dropping only the partially-written
    tail token) and retries, bounded. Content before the cut is never
    altered, and the caller's schema validation still requires every
    mandatory field — a salvage that lost a required field fails closed
    exactly like the unparseable original."""
    text = raw.rstrip()
    if not text.startswith("{"):
        return None
    try:
        json.loads(text)
        return None  # valid JSON is not this function's business
    except json.JSONDecodeError as e:
        if e.pos < len(text) - 2:
            return None  # defect is mid-file, not a truncated tail
    for _ in range(40):
        candidate = _close_truncated_json(text)
        if candidate is not None:
            try:
                out = json.loads(candidate)
                if isinstance(out, dict):
                    return out
                return None
            except json.JSONDecodeError:
                pass
        cut = max(text.rfind(","), text.rfind("{"), text.rfind("["))
        if cut <= 0:
            return None
        text = text[:cut].rstrip()
    return None


def _diagnostician_dispatch_evidence(
    attempts: list[tuple[str, "DispatchResult"]],
) -> str:
    """Driver-recorded facts about the diagnostician dispatch attempts,
    appended to the judge's stderr block when the diagnosis is unusable.

    A judge shown only "did not write smoke_diagnosis.json" invents the
    mechanism: detr 15:47 and detr 01:23 both asserted a "900s dispatch
    timeout" in the halt rationale (quoted verbatim to the researcher) for
    dispatches the driver recorded as COMPLETED, under budget, with zero
    writes. Mechanism claims must be bounded by evidence the driver already
    holds (halt-reason-rewrite-design.md, rule 2) — so hand the judge those
    facts and forbid the unevidenced reading explicitly."""
    if not attempts:
        return ""
    lines = [
        "**Dispatch evidence (driver-recorded — bound any mechanism claim "
        "to these facts):**",
    ]
    any_completed_no_write = False
    for label, r in attempts:
        if r.completed:
            any_completed_no_write = True
            lines.append(
                f"- {label}: the dispatch COMPLETED in {r.elapsed_s:.1f}s — "
                f"it did NOT time out — and ended without a usable "
                f"diagnosis file."
            )
        else:
            err = (r.error or {}).get("name") or "dispatch error"
            lines.append(
                f"- {label}: the dispatch did not complete "
                f"({err} after {r.elapsed_s:.1f}s)."
            )
    if any_completed_no_write:
        lines.append(
            "A completed dispatch that wrote nothing is the long-analysis "
            "output-budget class (the turn's reasoning exhausted its "
            "per-step output limit before the Write landed), not a timeout "
            "and not a transport failure. Do NOT attribute it to a dispatch "
            "timeout; base your classification on the facts above."
        )
    return "\n".join(lines)


def _diagnostician_halt_evidence(
    attempts: list[tuple[str, "DispatchResult"]],
    *,
    diagnosis_file_exists: bool,
) -> dict | None:
    """Structured halt-catalog evidence from the diagnostician attempts —
    the same driver-recorded facts `_diagnostician_dispatch_evidence`
    narrates for the judge, in the shape `halt_catalog.resolve_mechanism`
    consumes for the researcher story.

    `diagnosis_file_exists` keeps the zero-writes claim honest: an unusable
    diagnosis can be an ABSENT file (the no-write class the mechanism
    resolver names) or a PRESENT-but-invalid one (no mechanism claim; the
    story then says what is known and stops)."""
    if not attempts:
        return None
    last = attempts[-1][1]
    evidence = {
        "completed": all(r.completed for _, r in attempts),
        "elapsed_s": last.elapsed_s,
        "attempts": len(attempts),
        "step_gloss": "the debugger's analysis step",
    }
    if not diagnosis_file_exists:
        evidence["writes_observed"] = 0
    return evidence


# The one file each smoke-routable producer most plausibly owns a smoke bug
# in, used ONLY by the driver-authored fallback diagnosis when the traceback
# yields no owned frame. architecture-coder's model.py is the conventional
# pick over training.py because smoke failures reaching arch code fire in
# forward paths far more often than in training helpers; the traceback
# anchor wins whenever it exists.
_SMOKE_PRODUCER_PRIMARY_FILE: dict[str, str] = {
    "method-coder": "method/method.py",
    "architecture-coder": "method/model.py",
    "notebook-generator": ".pipeline/notebook_draft.py",
}


def _write_driver_fallback_smoke_diagnosis(
    state: PipelineState, *,
    mechanical_producer: str,
    stderr_tail: str,
    fail_cell: int,
    section: int,
    exception_class: str | None,
    failed_targets: list[str],
    dispatch_evidence: str,
) -> dict | None:
    """Author the imperfect-but-honest fallback diagnosis when both
    diagnostician turns completed with zero writes (maintainer-approved
    2026-07-05, option 2 of the detr Think-tier brief). The detr evidence:
    prompt-level write-first cannot preempt a thinking channel that
    exhausts the per-step output budget before the first tool call, so the
    write must not depend on the model's turn surviving — structural
    write-first, the driver does the write.

    Built ONLY from driver-recorded facts: the mechanical routing
    (traceback or section table), the deepest owned traceback frame, and
    the failing cell. Every judgment field states explicitly that it is a
    driver-authored fallback, so the producer, the stage reviewer, and the
    researcher all read it as coarse routing, never as an analyzed
    diagnosis. Returns the diagnosis dict on success, or None when no
    honest fallback exists (the mechanical pick's file is already in
    failed_targets, or the producer is not smoke-routable) — the caller
    then keeps the judge path exactly as before."""
    producer = mechanical_producer
    target_agent = f"r2c-{producer}"
    if target_agent not in WRITEABLE_PATHS:
        return None
    anchor = smoke_traceback_deepest_owned_frame(
        stderr_tail, _smoke_producer_to_writeable_paths(producer))
    target_file: str | None = None
    if anchor:
        # anchor shape: "method/method.py:120 in _hungarian_matching"
        target_file = anchor.split(":", 1)[0]
    if not target_file:
        target_file = _SMOKE_PRODUCER_PRIMARY_FILE.get(producer)
    if not target_file or target_file in failed_targets:
        return None
    exc = exception_class or "an exception"
    site = anchor or f"notebook cell {fail_cell}"
    diagnosis = {
        "schema_version": "1.0.0",
        "target_agent": target_agent,
        "target_file": target_file,
        "bug_shape": "uncatalogued",
        "root_cause": (
            f"DRIVER-AUTHORED FALLBACK, not an analyzed diagnosis: notebook "
            f"cell {fail_cell} (§{section}) raised {exc}; the deepest "
            f"traceback frame in {target_agent}'s files is {site}. Both "
            f"diagnostician turns completed without writing a diagnosis "
            f"(the long-analysis output-budget class), so the actual root "
            f"cause is unanalyzed."
        ),
        "proposed_fix": (
            f"Read the failing cell and traceback in the finding, verify "
            f"the bug's actual location per the (i)-(iv) checklist starting "
            f"at {site}, and fix it in `{target_file}`. This routing is "
            f"mechanical (traceback/section based); if your analysis places "
            f"the bug outside your writeable paths, report that in your "
            f"status table instead of editing."
        ),
        "reasoning": (
            "Driver-authored fallback after two zero-write diagnostician "
            "turns. Routing basis: mechanical traceback/section analysis "
            "only; no data-flow trace was performed.\n\n" + dispatch_evidence
        ),
        "paper_fidelity_check": (
            "Not assessed (driver-authored fallback). The producer must "
            "keep its fix paper-faithful per its own contract and flag any "
            "smoke-scale deviation in its report; the stage reviewer checks "
            "fidelity downstream."
        ),
        "value_origin_trace": [
            f"{site} — error site per the traceback (driver-extracted)",
            f"notebook cell {fail_cell} (§{section}) raised {exc}",
            "upstream derivation NOT walked — driver-authored fallback "
            "after two zero-write diagnostician turns; treat the trace as "
            "absent",
        ],
    }
    # Schema-gate the driver's own artifact exactly like an agent's.
    sys.path.insert(0, str(REPO_ROOT))
    from schemas.smoke_diagnosis import SmokeDiagnosis  # noqa: PLC0415
    SmokeDiagnosis.model_validate(diagnosis)
    path = state.paths.pipeline_dir / "smoke_diagnosis.json"
    path.write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    return diagnosis


def _read_smoke_diagnosis(state: PipelineState) -> tuple[dict | None, str | None]:
    """Read + schema-validate `.pipeline/smoke_diagnosis.json`. Returns
    `(diagnosis_dict, None)` on success or `(None, error_message)` if the
    file is missing / malformed / target_file is outside the target_agent's
    allowlist (the orchestrator would halt downstream anyway on that).

    The driver halts on any error — the user inspects, fixes whatever's
    wrong with the diagnostician's output (prompt drift, schema breakage),
    and re-runs. Per the halt-and-investigate principle, no silent fallback
    to the mechanical routing. One deterministic exception: a tail-truncated
    file (valid prefix, missing closers) gets a salvage parse first — the
    full schema validation below still gates it, so nothing partial ships."""
    path = state.paths.pipeline_dir / "smoke_diagnosis.json"
    if not path.exists():
        return None, f"diagnostician did not write {path.name}"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return None, f"smoke_diagnosis.json is unreadable: {e}"
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raw = _salvage_truncated_json(text)
        if raw is None:
            return None, f"smoke_diagnosis.json is not valid JSON: {e}"
        log("stage_3c", "smoke_diagnosis_salvaged",
            f"smoke_diagnosis.json was tail-truncated ({e}); salvage parse "
            f"recovered a complete object — schema validation still gates it")
    # Pydantic validation — catches schema drift, missing fields, wrong types.
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from schemas.smoke_diagnosis import SmokeDiagnosis  # noqa: PLC0415
    except ImportError as e:
        return None, f"could not import smoke_diagnosis schema: {e}"
    try:
        diag = SmokeDiagnosis.model_validate(raw)
    except Exception as e:  # pydantic.ValidationError or similar
        return None, f"smoke_diagnosis.json failed schema validation: {e}"
    # target_file must be in target_agent's allowlist (so the next dispatch
    # doesn't immediately halt on out-of-scope writes).
    allowed = WRITEABLE_PATHS.get(diag.target_agent, [])
    import fnmatch as _fn  # noqa: PLC0415
    if not any(_fn.fnmatchcase(diag.target_file, p) for p in allowed):
        return None, (
            f"diagnostician's target_file {diag.target_file!r} is not in "
            f"target_agent {diag.target_agent!r}'s writeable_paths "
            f"({allowed}); the next fix dispatch would halt on out-of-scope writes"
        )
    return diag.model_dump(), None


def _smoke_target_agent_to_producer_short(target_agent: str) -> str:
    """Map a diagnosis target_agent (e.g., 'r2c-method-coder') to the
    short producer name used by `_smoke_producer_to_fix_fn` (e.g.,
    'method-coder'). The two naming conventions exist because smoke
    routing pre-dates WRITEABLE_PATHS and uses short names everywhere."""
    table = {
        "r2c-method-coder": "method-coder",
        "r2c-architecture-coder": "architecture-coder",
        "r2c-notebook-generator": "notebook-generator",
    }
    if target_agent not in table:
        raise ValueError(f"diagnosis target_agent not smoke-routable: {target_agent!r}")
    return table[target_agent]


def _run_smoke_gate(state: PipelineState) -> tuple[int, str, str]:
    """Run smoke_run_notebook.py. Returns (exit_code, stdout_tail, stderr_tail).
    Exit codes per the script: 0=clean, 1=cell error/timeout, 2=setup error."""
    paths = state.paths
    proc = run_script(
        "stage_3c",
        ["scripts/smoke_run_notebook.py", "--run-dir", str(paths.run_dir),
         "--timeout", str(SMOKE_CELL_TIMEOUT_S)],
        timeout=SMOKE_WALLCLOCK_TIMEOUT_S,
    )
    ok = proc.returncode == 0
    details = {
        "exit_code": proc.returncode,
        "stdout_tail": proc.stdout[-1000:],
        "stderr_tail": proc.stderr[-1000:],
    }
    if ok:
        # R2C-039 block 2: the digest of the notebook the smoke gate just
        # verified and saved. Delivery validation compares the delivered
        # notebook against the LAST smoke-passed digest, so a notebook
        # modified after its last verified execution cannot present itself
        # as verified.
        try:
            details["notebook_sha256"] = hashlib.sha256(
                (paths.run_dir / "notebook.ipynb").read_bytes()).hexdigest()
        except OSError:
            pass
    _append_run_event(
        paths,
        "smoke_passed" if ok else "smoke_failed",
        stage_id="stage_3c",
        status="passed" if ok else "failed",
        summary="smoke_run_notebook.py passed" if ok else "smoke_run_notebook.py failed",
        artifacts=[str(paths.run_dir / "notebook.ipynb")],
        details=details,
    )
    return proc.returncode, proc.stdout[-2000:], proc.stderr[-3000:]


class SmokeSanityFailure(NamedTuple):
    """One arm of the smoke-time sanity pre-check, failing.

    `kind` exists because the two arms are different failures and the caller
    logs what happened: before the divergence arm landed, the call site's log
    line and run event said "the metric series never beats chance"
    unconditionally, which would have described a NaN metric as
    non-learning."""

    cell: int
    stderr: str
    kind: str          # "nonfinite_metric" | "below_chance"
    summary: str       # one line for the log and the run event


def _trainability_smoke_check(state: PipelineState) -> SmokeSanityFailure | None:
    """UB-6's sanity rules enforced at smoke time. Two arms.

    GBALD 2026-07-03 round 2: a training-tensor reshape bug produced a demo
    that executed cleanly while the model never learned (accuracy peaked at
    0.087, below the 0.10 chance floor). Smoke can't see silent non-learning,
    so the defect surfaced as a stage-5 delivery demoter — after the fix
    loop that could have acted on it had already closed. This pre-check runs
    the same accuracy-vs-chance rule (same regexes, same margin — one rule,
    two enforcement points) against the just-executed notebook and, on
    failure, synthesizes a smoke-shaped failure routed at the cell that
    printed the series, so the standard diagnostician → producer fix loop
    gets its shot.

    The DIVERGENCE arm runs first and is paradigm-independent (R2C-071). The
    chance arm above needs a class count, so before this arm existed the whole
    pre-check returned None on its third line for any paper that is not
    classification — and the promise of "one rule, two enforcement points"
    held for exactly one of the rules. The 2026-08-06 pdfgnn roll executed
    cleanly with `RMSE: nan`, `MAE: nan`, `WMAPE: nan`, and the NaN surfaced
    for the first time as a stage-5 delivery demoter with four iterations of
    smoke budget unspent. A non-finite metric is not a classification
    question, so it is asked of every paradigm.

    Returns (metric_cell_index, synthetic_stderr) on either failure, else
    None. Every not-checkable path returns None — no printed accuracy series,
    no detectable class count, unreadable notebook — because the stage-5
    battery already discloses those honestly (warn/unprobeable) and a smoke
    gate must not fail on a missing signal."""
    nb_path = state.paths.run_dir / "notebook.ipynb"
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        from probes.universal import (  # noqa: PLC0415
            CHANCE_MARGIN, detect_n_classes, extract_metric_series_with_cell,
            first_nan_inf_with_cell, nonfinite_metric_message)
    except ImportError:
        return None

    nonfinite = first_nan_inf_with_cell(nb)
    if nonfinite is not None:
        key, token, cell = nonfinite
        return SmokeSanityFailure(
            cell=cell,
            stderr=(
                "NON-FINITE METRIC detected by the smoke-time sanity check.\n"
                "The notebook executed end-to-end without errors, but "
                + nonfinite_metric_message(key, token, state.paths.run_dir)
                + "\n"
                "A delivery whose only quantitative result is nan demonstrates "
                "nothing, so this is a smoke failure rather than a note.\n"
                "The failing-cell pointer is the cell that PRINTED the value; "
                "the cause may live upstream in the data preparation, the "
                "evaluation windowing, or the training loop.\n"
                "This is the UB-6 divergence rule enforced at smoke time so "
                "the fix loop can act on it; it would otherwise surface as a "
                "stage-5 delivery demoter after every loop has closed."
            ),
            kind="nonfinite_metric",
            summary=f"executed metric {key!r} printed {token!r}",
        )

    n_classes = detect_n_classes(nb)
    if not n_classes:
        return None
    series, metric_cell = extract_metric_series_with_cell(nb)
    if len(series) < 2 or metric_cell is None:
        return None
    chance = 1.0 / n_classes
    peak = max(series)
    if peak >= chance + CHANCE_MARGIN:
        return None
    synthetic_stderr = (
        f"SILENT NON-LEARNING detected by the smoke-time trainability "
        f"check.\n"
        f"The notebook executed end-to-end without errors, but the printed "
        f"accuracy series peaks at {peak:.3f} while chance level is "
        f"{chance:.3f} (1/{n_classes} classes) — the model never learns "
        f"above chance at any round, so the demo does not demonstrate the "
        f"method.\n"
        f"Executed series: {['%.3f' % s for s in series]}\n"
        f"The failing-cell pointer is the cell that PRINTED the series (the "
        f"evaluation loop); the root cause may equally live in the training "
        f"loop it evaluates (e.g. a data/label reshape or wiring defect that "
        f"silently trains on garbage) — the outputs alone cannot "
        f"distinguish these.\n"
        f"This is the UB-6 beats-chance rule enforced at smoke time so the "
        f"fix loop can act on it; it would otherwise surface as a stage-5 "
        f"delivery demoter."
    )
    return SmokeSanityFailure(
        cell=metric_cell, stderr=synthetic_stderr, kind="below_chance",
        summary=(f"the printed accuracy series peaks at {peak:.3f} against "
                 f"chance {chance:.3f}"),
    )


def _demo_data_digest(paths: PipelinePaths) -> dict[str, str]:
    """Content digest of the demo-data surface: method/data.py, every file
    under method/example_data/, AND the rendered notebook's setup-section
    sources. The notebook side is in the set because fix-mode producers can
    write nothing but the draft — the recorded Rethinking 2026-07-14 swap
    installed its label-free data through the notebook source, not the
    package. Only the SETUP-section slice is hashed (the same sources the
    data-signal check itself executes, resolved from the layout SSOT), so a
    cosmetic fix in a demo cell never triggers the check; when no rendered
    notebook is readable the whole draft is the coarser fallback. The fix
    loop compares this digest across a producer dispatch to detect a
    demo-data swap/synthesis deterministically, whoever wrote it and however
    it was phrased."""
    digest: dict[str, str] = {}
    data_py = paths.run_dir / "method" / "data.py"
    if data_py.is_file():
        digest["method/data.py"] = sha256_file(data_py)
    example_dir = paths.run_dir / "method" / "example_data"
    if example_dir.is_dir():
        for f in sorted(example_dir.rglob("*")):
            if f.is_file():
                rel = f.relative_to(paths.run_dir).as_posix()
                digest[rel] = sha256_file(f)
    blob: str | None = None
    try:
        from demo_verdict import setup_section_source_blob  # noqa: PLC0415
        blob = setup_section_source_blob(paths.run_dir)
    except Exception:  # noqa: BLE001 — the digest must never break the loop
        blob = None
    if blob is not None:
        digest["notebook:setup_sections"] = sha256_text(blob)
    else:
        draft = paths.pipeline_dir / "notebook_draft.py"
        if draft.is_file():
            digest[".pipeline/notebook_draft.py"] = sha256_file(draft)
    return digest


FIX_DATA_SIGNAL_TIMEOUT_S = 180


def _post_fix_data_signal_note(state: PipelineState, iteration: int) -> str | None:
    """Run the deterministic data-signal check after a fix dispatch changed
    the demo-data surface (demo-success design piece 2, the Rethinking
    2026-07-14 burn). Returns the plain-language note the NEXT fix dispatch
    must carry when the new data can never pass the beats-chance gate, else
    None (winnable data, an inapplicable family, or a check that could not
    run — advisory only, never blocking).

    Fix loop only by design (maintainer decision 2026-07-16): this is not wired into first
    notebook authoring."""
    stage_id = "stage_3c"
    paths = state.paths
    try:
        from fix_data_signal import declares_beats_chance  # noqa: PLC0415
        if not declares_beats_chance(paths.run_dir):
            return None
    except Exception as e:  # noqa: BLE001 — advisory check, never blocks
        log(stage_id, "fix_data_signal_error",
            f"data-signal applicability check failed: {type(e).__name__}: {e}")
        return None
    # Subprocess on purpose: the check imports the GENERATED package to call
    # its own load_data, and generated code must not run inside the driver
    # process (same isolation the probe battery gets at stage 5).
    proc = run_script(
        stage_id,
        ["scripts/fix_data_signal.py", "--run-dir", str(paths.run_dir)],
        timeout=FIX_DATA_SIGNAL_TIMEOUT_S,
    )
    payload: dict = {}
    for line in reversed((proc.stdout or "").strip().splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("verdict"):
            payload = candidate
            break
    verdict = str(payload.get("verdict") or "")
    message = str(payload.get("message") or "")
    if verdict == "fail":
        log(stage_id, "fix_data_signal_failed",
            f"iteration {iteration}: the swapped demo data cannot carry the "
            f"class signal the beats-chance gate needs ({message})")
        _append_run_event(
            paths, "fix_data_signal_failed", stage_id=stage_id,
            status="failed",
            summary="a fix-loop demo-data swap produced data that can never "
                    "pass the beats-chance gate",
            details={"iteration": iteration, "message": message,
                     "stats": payload.get("stats")},
        )
        return (
            f"DATA-SIGNAL CHECK FAILED for the demo data a previous fix "
            f"installed: {message}\n"
            f"The beats-chance gate this notebook must pass needs demo data "
            f"whose labels correlate with features BY CONSTRUCTION (e.g. "
            f"class-conditional separated synthetic classes). With the "
            f"current data, NO model change can ever pass that gate — fix "
            f"the data, not the model."
        )
    log(stage_id, "fix_data_signal_checked",
        f"iteration {iteration}: data-signal check verdict "
        f"{verdict or 'unreadable'}"
        + (f" ({message})" if message else ""))
    _append_run_event(
        paths, "fix_data_signal_checked", stage_id=stage_id,
        status=verdict or "unreadable",
        summary=f"fix-loop data-signal check: {verdict or 'unreadable'}",
        details={"iteration": iteration, "message": message},
    )
    return None


def _attempt_fix_loop_stub_conversion(
    state: PipelineState,
    stage_id: str,
    *,
    give_up_point: str,
    prior_iterations: list[dict],
    failing_cell_source: str,
    fail_cell,
    section,
    stderr_tail: str,
) -> StageResult | None:
    """Fix-loop third option (queue item 7): at a terminal give-up point,
    convert THE single failing component to a stub and continue toward a
    partial delivery — gated by the deterministic G1-G5 reachability gates
    (fix_loop_stub_gate.py), never by agent judgment.

    Returns None when the gates do not fire OR the conversion cannot land
    safely (the caller then degrades exactly as today); a halted StageResult
    on a G4 plausibility mismatch (a broken diagnosis chain needs a human,
    and stubbing would hide it); or the completed StageResult after the
    MANDATORY smoke re-gate passes on the stubbed notebook. A re-gate
    failure rolls the whole conversion back (component file, draft, record,
    work order) and returns None, so the researcher-facing outcome is
    byte-identical to today's degrade."""
    from fix_loop_stub_gate import (evaluate_stub_gates,
                                    replace_draft_cell_with_stub_marker)
    from partial_delivery import (load_stubbed_elements,
                                  remove_stubbed_element, render_work_order,
                                  write_stub)
    paths = state.paths

    def _stub_event(event_type: str, summary: str) -> None:
        # Loud on BOTH channels: driver log line + a first-class run event
        # (the timeline map pins every registered type to a category).
        log(stage_id, event_type, summary)
        _append_run_event(paths, event_type, stage_id=stage_id,
                          summary=summary[:500])
    try:
        spec = json.loads(paths.method_spec.read_text(encoding="utf-8"))
        contract_elements = ((spec.get("methodology_replication_contract")
                              or {}).get("elements") or [])
    except (OSError, json.JSONDecodeError):
        contract_elements = []
    existing, load_error = load_stubbed_elements(paths.pipeline_dir)
    if load_error:
        _stub_event("fix_loop_stub_gates_unmet",
            f"stub artifact unreadable ({load_error}); not firing")
        return None
    verdict = evaluate_stub_gates(
        give_up_point=give_up_point,
        prior_iterations=prior_iterations,
        current_frame=smoke_traceback_deepest_owned_frame(
            stderr_tail, ["method/*.py"]),
        cell_source=failing_cell_source,
        existing_stubs=existing,
        contract_elements=contract_elements,
    )
    if verdict.halt_on_g4:
        _stub_event("fix_loop_stub_gates_unmet", verdict.summary()[:500])
        return halt(
            paths, stage_id,
            reason=("fix-loop stub conversion refused by the diagnosis "
                    "plausibility screen (G4): the recorded diagnosis class "
                    "does not match the failing cell's source shape — the "
                    "diagnosis chain is broken and needs a human look "
                    "before anything ships"),
            halt_class="fix_loop_exhausted",
            context={"stub_gate_verdict": verdict.reasons,
                     "last_failing_cell": fail_cell,
                     "last_section": section},
            state=state,
        )
    if not verdict.fire:
        _stub_event("fix_loop_stub_gates_unmet", verdict.summary()[:500])
        return None

    eid = verdict.element_id
    role = verdict.role
    component_file = verdict.component_file
    draft_path = paths.run_dir / "notebook_draft.py"
    component_path = paths.run_dir / component_file
    if not draft_path.is_file() or not component_path.is_file():
        _stub_event("fix_loop_stub_gates_unmet",
            "notebook draft or component file missing on disk; not firing")
        return None
    draft_text = draft_path.read_text(encoding="utf-8")
    component_text = component_path.read_text(encoding="utf-8")
    new_draft = replace_draft_cell_with_stub_marker(
        draft_text, failing_cell_source, eid)
    if new_draft is None:
        _stub_event("fix_loop_stub_gates_unmet",
            "failing cell not uniquely locatable in notebook_draft.py; an "
            "ambiguous edit could stub the wrong cell — not firing")
        return None

    # Everything below mutates the run dir; keep a full rollback set so a
    # failed re-gate leaves the tree byte-identical to the degrade path.
    backup_dir = paths.pipeline_dir / "stub_conversion_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    (backup_dir / "component.py").write_text(component_text, encoding="utf-8")
    (backup_dir / "notebook_draft.py").write_text(draft_text, encoding="utf-8")

    n_iter = len(prior_iterations)
    targets = sorted({str((it.get("diagnosis") or {}).get("target_file") or "")
                      for it in prior_iterations} - {""})
    fix_history = (
        f"{n_iter} fix iterations plus the judge-blessed extra were consumed "
        f"({give_up_point}). Distinct targets attempted: "
        f"{', '.join(targets)}. Last recorded diagnosis: "
        f"{(prior_iterations[-1].get('diagnosis') or {}).get('root_cause', '?')}"
    )
    work_order_md = render_work_order(
        element_id=eid, role=role, interface=None, paper_anchor=None,
        why_not_built=(
            f"The smoke fix loop exhausted its cap against failures every "
            f"one of which traced into `{component_file}` (isolation gate "
            f"G3), after attempting {len(targets)} distinct fix targets "
            f"(breadth gate G2). The component ships as a raising stub so "
            f"the rest of the package stays usable. Element mapping: "
            f"{verdict.mapping_basis}."),
        verified_neighborhood=None,
        fix_history=fix_history,
    )
    write_stub(paths.run_dir, element_id=eid, role=role,
               stub_rel_path=component_file,
               work_order_markdown=work_order_md, signatures=None)
    draft_path.write_text(new_draft, encoding="utf-8")
    _stub_event("fix_loop_stub_conversion",
        f"gates G1-G5 hold; converting {component_file} to a stub "
        f"(element {eid}, role={role}, {verdict.mapping_basis}) and "
        f"re-running the smoke gate once")

    def _rollback(why: str) -> None:
        component_path.write_text(component_text, encoding="utf-8")
        draft_path.write_text(draft_text, encoding="utf-8")
        try:
            remove_stubbed_element(paths.pipeline_dir, eid)
        except ValueError as e:
            _stub_event("fix_loop_stub_regate_failed",
                f"rollback could not remove the stub record: {e}")
        (paths.run_dir / "work_orders" / f"{eid}.md").unlink(missing_ok=True)
        ok_r, err_r = _render_notebook(state)
        _stub_event("fix_loop_stub_regate_failed",
            f"{why}; conversion rolled back"
            + ("" if ok_r else f" (post-rollback re-render failed: {err_r[-200:]})"))

    ok, err = _render_notebook(state)
    if not ok:
        _rollback(f"post-stub render failed: {err[-300:]}")
        return None
    ok, err = _run_stage2_script(
        state, stage_id=stage_id,
        args=["scripts/validate_notebook_output.py"])
    if not ok:
        _rollback(f"post-stub notebook validation failed: {err[-300:]}")
        return None
    exit_code, _, regate_stderr = _run_smoke_gate(state)
    if exit_code != 0:
        _rollback(f"stubbed notebook still fails the smoke gate: "
                  f"{regate_stderr[-300:]}")
        return None

    aid = _next_assumption_id(state)
    _append_assumption(
        state, aid=aid,
        title=f"PARTIAL delivery: `{eid}` ships as a stub after fix-loop "
              f"exhaustion",
        detected=(f"The smoke fix loop gave up ({give_up_point}) with every "
                  f"failure isolated to `{component_file}`. Gate evidence: "
                  f"{verdict.summary()}"),
        action=(f"The component was converted to a self-identifying raising "
                f"stub, the notebook marks the spot with a PARTIAL notice + "
                f"raising cell, and the stubbed notebook passed the smoke "
                f"re-gate. Work order: `work_orders/{eid}.md`. The delivery "
                f"label can never read verified with a stub recorded."),
        reasoning=("A partial but accurate package beats explanation-only: "
                   "the buildable fraction ships with the gap explicit on "
                   "every surface (fix-loop-third-option design; one "
                   "conversion per run, deterministic gates only)."),
        alternative="Halt at cap exhaustion as before and fix by hand.",
        override=(f"Implement the component per the work order, delete the "
                  f"stub record for `{eid}` in "
                  f".pipeline/stubbed_elements.json, and resume."),
    )
    return StageResult(
        status="completed", stage_id=stage_id,
        notes=(f"smoke gate clean after fix-loop stub conversion: "
               f"{eid} ({role}) ships as a stub, delivery is PARTIAL "
               f"(work order work_orders/{eid}.md)"))


def run_stage_3c(state: PipelineState) -> StageResult:
    """3.c — smoke gate (see `_run_stage_3c_smoke_loop` for the loop itself).

    This wrapper is the single enforcement point for the post-smoke
    requirements reconcile: any 3.c exit where the run continues to delivery
    (completed or degraded — NOT skipped, the notebook is unchanged there,
    and NOT halted, the run stops) re-runs the reconcile, because the fix
    loop's producer dispatches can rewrite the notebook — and add imports —
    after stage 3.a's reconcile already ran (O2 gap, repo-map verdict
    2026-07-04).

    It is also the demo-success verdict point (design approved 2026-07-16):
    after a CLEAN smoke execution (completed only — a degraded exit never
    persisted executed outputs, so there is nothing honest to read) the
    deterministic post-smoke pass records whether the headline demonstration
    demonstrably worked. Never a halt, never a degrade — verdict `failed` is
    a first-class delivery demoter at stage 5. A crashed legacy pass leaves no
    artifact; a family declaring structured demo evidence records an unresolved
    schema-2 fallback and delivery also fails closed if that artifact is absent."""
    result = _run_stage_3c_smoke_loop(state)
    if result.status in ("completed", "degraded"):
        _reconcile_requirements_with_notebook(state, stage_id="stage_3c")
    if result.status == "completed":
        _record_demo_verdict(state)
    return result


_POST_SMOKE_DERIVED_ARTIFACTS = (
    "demo_verdict.json",
    "target_scaling_state.json",
    "training_history.json",
)


def _clear_post_smoke_derived_artifacts(paths: PipelinePaths) -> None:
    """Invalidate evidence derived from an executed notebook.

    Any producer change that will replace or re-execute the notebook makes all
    three artifacts one atomic lifecycle unit: none may continue describing
    the prior notebook while another is recomputed from the new one.
    """
    for name in _POST_SMOKE_DERIVED_ARTIFACTS:
        (paths.pipeline_dir / name).unlink(missing_ok=True)


def _record_demo_verdict(
    state: PipelineState,
    *,
    stage_id: str = "stage_3c",
) -> None:
    """Run the deterministic demo-success verdict pass and log it, loudly on
    a failed demo and on a kit-coverage gap, quietly otherwise. Structurally
    unable to change the stage outcome: every exception is caught and logged
    (the smoke gate's semantics are untouched by design)."""
    paths = state.paths
    structured_required = False
    structured_paradigm = ""
    try:
        from demo_verdict import (  # noqa: PLC0415
            record_demo_verdict,
            record_unresolved_structured_demo_verdict,
            structured_demo_requirement,
        )
        structured_paradigm, structured_required = (
            structured_demo_requirement(paths.run_dir)
        )
        payload = record_demo_verdict(paths.run_dir)
    except Exception as e:  # noqa: BLE001 — the verdict must never block 3.c
        log(stage_id, "demo_verdict_error",
            f"demo-success verdict pass could not run: {type(e).__name__}: {e}")
        if not structured_required:
            return
        try:
            payload = record_unresolved_structured_demo_verdict(
                paths.run_dir,
                paradigm=structured_paradigm,
                error=e,
            )
        except Exception as fallback_error:  # noqa: BLE001 - logged, delivery demotes absence
            log(
                stage_id,
                "demo_verdict_fallback_error",
                "structured demo-evidence fallback could not be recorded: "
                f"{type(fallback_error).__name__}: {fallback_error}",
            )
            return
    if payload is None:
        return
    verdict = payload.get("verdict")
    detail = payload.get("evidence_line") or payload.get("reason") or ""
    axes = payload.get("evidence_status")
    if isinstance(axes, dict):
        evaluation = axes.get("evaluation_validity")
        skill = axes.get("skill")
        evaluation_status = (
            evaluation.get("status") if isinstance(evaluation, dict)
            else "unresolved"
        )
        skill_status = (
            skill.get("status") if isinstance(skill, dict)
            else "undetermined"
        )
        log(
            stage_id,
            "demo_evidence",
            "structured demo evidence: execution=completed, "
            f"evaluation_validity={evaluation_status}, skill={skill_status}",
        )
        event_type = (
            "demo_verdict_failed"
            if skill_status == "not_demonstrated"
            else "demo_verdict_recorded"
        )
        _append_run_event(
            paths, event_type, stage_id=stage_id,
            status=("failed" if skill_status == "not_demonstrated"
                    else str(skill_status)),
            summary=(
                "structured demo evidence recorded: execution completed; "
                f"evaluation {evaluation_status}; skill {skill_status}"
            ),
            details={
                "decided_by": payload.get("decided_by"),
                "evaluation_validity": evaluation_status,
                "skill": skill_status,
                "evidence_cell": payload.get("evidence_cell"),
            },
        )
        finding = payload.get("kit_coverage_finding")
        if isinstance(finding, dict) and finding.get("message"):
            log(stage_id, "demo_kit_coverage_gap", str(finding["message"]))
            _append_run_event(
                paths, "demo_kit_coverage_gap", stage_id=stage_id,
                summary=str(finding["message"])[:500],
                details={"family": finding.get("family")},
            )
        return
    if verdict == "failed":
        log(stage_id, "demo_verdict_failed",
            f"notebook executed cleanly but its headline demo does not "
            f"succeed: {detail}")
        _append_run_event(
            paths, "demo_verdict_failed", stage_id=stage_id, status="failed",
            summary="the headline demo visibly fails in its own executed "
                    "output (demotes the delivery label, never halts)",
            details={"evidence_line": payload.get("evidence_line"),
                     "evidence_cell": payload.get("evidence_cell"),
                     "decided_by": payload.get("decided_by")},
        )
    else:
        log(stage_id, "demo_verdict",
            f"demo-success verdict: {verdict}" + (f" — {detail}" if detail else ""))
        _append_run_event(
            paths, "demo_verdict_recorded", stage_id=stage_id, status=verdict,
            summary=f"post-smoke demo-success verdict: {verdict}",
            details={"decided_by": payload.get("decided_by"),
                     "reason": payload.get("reason")},
        )
    finding = payload.get("kit_coverage_finding")
    if isinstance(finding, dict) and finding.get("message"):
        # Committed-family coverage gap we own (maintainer decision 2026-07-16): loud on the
        # event stream so the demand signal is auditable, disclosed at
        # delivery via the verdict artifact.
        log(stage_id, "demo_kit_coverage_gap", str(finding["message"]))
        _append_run_event(
            paths, "demo_kit_coverage_gap", stage_id=stage_id,
            summary=str(finding["message"])[:500],
            details={"family": finding.get("family")},
        )


def _run_stage_3c_smoke_loop(state: PipelineState) -> StageResult:
    """3.c — smoke gate. Executes notebook.ipynb end-to-end via nbclient and
    on cell-failure routes the fix to the responsible producer (notebook-
    generator, method-coder, or architecture-coder) based on which §-section
    the failing cell is in.

    Per the archived v2 orchestrator spec (internal, not shipped), this gate is the ONLY check that the notebook
    actually runs as a notebook (kernel-state, magic-command, cell-ordering
    bugs that static validators miss). The naive "always re-run" policy
    catches upstream regressions but burns minutes of wall time when nothing
    has actually changed — the resume-from-stage-4-halt case (R2C 2026-05-22)
    being the canonical waste. So skip-if-done DOES apply, but with an
    mtime check: skip only if `stage_3c.complete` is present AND no
    upstream file (`method/*.py`, `notebook_draft.py`, `notebook.ipynb`,
    `params.json`, `requirements.txt`) is newer than the sentinel. Any
    producer re-dispatch updates one of those mtimes, naturally invalidating
    the sentinel — so the regression-catching property is preserved.

    Cap=3 outer iterations. After each fix-mode dispatch, re-render + re-run
    the static validator + re-run smoke gate. On cap-hit, halt cleanly with
    smoke_gate.halt — do NOT band-aid the notebook source or the method/
    package; per the driver's no-Edit invariant, recovery is structurally
    bounded to producer re-dispatches."""
    stage_id = "stage_3c"
    paths = state.paths
    if (paths.pipeline_dir / f"{stage_id}.halt").exists():
        (paths.pipeline_dir / f"{stage_id}.halt").unlink()
    skipped = _skip_stage_3c_if_no_upstream_changes(paths)
    if skipped is not None:
        return skipped
    # Cache miss: the notebook is about to be (re)executed, so a prior leg's
    # demo verdict no longer describes anything on disk. Clear it exactly
    # like the stale halt artifact above — the artifact may only ever
    # describe the notebook THIS leg executed (the dead-attempt-leak class,
    # same family as commit 70a7a3206). A degraded exit then delivers with
    # NO verdict record rather than a stale one. The skip path above keeps
    # the prior verdict on purpose: a resume that never re-runs smoke still
    # delivers the prior leg's executed outputs, so its verdict stands.
    _clear_post_smoke_derived_artifacts(paths)
    if _stage_3c_needs_prerender(paths):
        ok, err = _render_notebook(state)
        if not ok:
            return halt(paths, stage_id,
                        reason="render_notebook.py failed before smoke re-run",
                        halt_class="producer_output_invalid",
                        context={"stderr": err}, state=state)
        ok, err = _run_stage2_script(
            state, stage_id=stage_id,
            args=["scripts/validate_notebook_output.py"],
        )
        if not ok:
            new_failures = _notebook_validation_new_failures(state, err)
            if new_failures:
                return halt(paths, stage_id,
                            reason=f"validate_notebook_output.py failed "
                                   f"before smoke re-run "
                                   f"({len(new_failures)} failure(s) not in "
                                   f"the stage-3a accepted baseline)",
                            halt_class="producer_output_invalid",
                            context={"stderr": err,
                                     "new_failures": new_failures},
                            state=state)
            _log_accepted_validation_continue(state, stage_id, "to smoke")

    # Cross-iteration memory for the diagnostician. Each completed iteration
    # appends a structured record; iter N+1's diagnostician dispatch receives
    # the full list. Without this, each iteration re-derives context from
    # session conversation, contaminating the diagnostician's procedural
    # identity (bev-distill 2026-05-14, iter 2: diagnostician started using
    # the producer's (a)-(e) skeleton and edited method.py instead of just
    # writing the diagnosis JSON).
    prior_iterations: list[dict] = []
    # (c) — Anti-repeat: target_files we've tried + smoke still failing at
    # the SAME cell. Diagnostician must not pick from this set (it tried, it
    # didn't converge). Driver halts if it picks one anyway. Bounds the
    # "diagnostician fixates on one wrong file" failure mode we observed
    # in the 2026-05-14 gbald run (5 iterations all targeting method/method.py
    # for a bug that lived in notebook_draft.py).
    failed_targets: list[str] = []
    # Data-signal note (demo-success design piece 2): set when a fix dispatch
    # swapped/synthesized demo data that can never pass the beats-chance gate
    # (label-free/random — the Rethinking 2026-07-14 burn). Carried into the
    # NEXT fix dispatch's prompt so the producer targets the data, and reset
    # by any later data change whose check passes.
    data_signal_note: str | None = None

    for iteration in range(STAGE_2_RETRY_CAP + 1):
        exit_code, stdout_tail, stderr_tail = _run_smoke_gate(state)
        fail_cell_override: int | None = None
        if exit_code == 0:
            trainability = _trainability_smoke_check(state)
            if trainability is None:
                log(stage_id, "smoke_passed", f"iteration {iteration}: notebook executed cleanly")
                return StageResult(status="completed", stage_id=stage_id,
                                   notes=f"smoke gate clean (iteration {iteration})")
            # The notebook ran clean but its own numbers say the demo does not
            # demonstrate the method: a non-finite metric, or a series that
            # never beats chance. Enter the same fix loop as a crashing cell
            # would, routed at the metric-printing cell.
            fail_cell_override = trainability.cell
            stderr_tail = trainability.stderr
            exit_code = 1
            log(stage_id, "smoke_trainability_failed",
                f"iteration {iteration}: notebook executed cleanly but "
                f"{trainability.summary} (cell {fail_cell_override}, "
                f"{trainability.kind}); routing into the smoke fix loop")
            _append_run_event(
                paths,
                "smoke_trainability_failed",
                stage_id=stage_id,
                status="failed",
                summary=f"executed notebook fails the smoke-time sanity "
                        f"pre-check ({trainability.kind}): "
                        f"{trainability.summary}",
                details={"iteration": iteration,
                         "metric_cell": fail_cell_override,
                         "check_kind": trainability.kind,
                         "synthetic_stderr": stderr_tail[:800]},
            )
        if exit_code == 3:
            # Environmental setup error (nbclient/nbformat absent, or no usable
            # Jupyter kernel). The notebook rendered cleanly (3.b) and the
            # package imports cleanly — the ENVIRONMENT, not the artifact, is
            # broken. Per the halt->degrade redesign (type-(I)/type-(D),
            # the halt-to-degrade redesign note (internal, not shipped)) and to match
            # stage_2d's environmental degrade, ship the rendered notebook
            # best-effort and CONTINUE to review + delivery instead of discarding
            # a complete package. Re-dispatching a producer can't fix the env,
            # and a hard halt here strands stages 4/5 + delivery (the
            # bayesian-active-learning halt, 2026-06-24). No retry — retrying
            # won't provision the environment.
            return degrade(
                paths, stage_id,
                reason=f"smoke gate could not run in this environment (exit 3): "
                       f"{stderr_tail[-400:]}",
                what_failed="the notebook could not be smoke-executed in this environment",
                where=str(paths.run_dir / "notebook.ipynb"),
                what_to_do=(
                    "The notebook rendered and the package imports cleanly, but R2C's "
                    "smoke environment is missing a notebook-execution dependency "
                    "(nbclient/nbformat/ipykernel) or a usable Jupyter kernel, so the "
                    "end-to-end run was not verified. Install the package requirements "
                    "(`pip install -r requirements.txt` plus `ipykernel`) and run the "
                    "notebook yourself, or re-run the pipeline once the environment is "
                    "provisioned. The artifact is rendered-but-not-smoke-verified."
                ),
                state=state,
            )
        if exit_code == 2:
            # Pipeline-fault setup error (notebook missing or unparseable) — a
            # genuine defect with nothing to ship. Halt without retry;
            # re-dispatching producers can't recover a missing/corrupt artifact.
            return halt(paths, stage_id,
                        reason="smoke gate setup error (exit 2)",
                        halt_class="internal_contract_violation",
                        context={"stderr": stderr_tail, "note": "pipeline-fault, not environmental"},
                        state=state)

        # exit_code == 1 — cell failed or timed out (or the trainability
        # pre-check synthesized a failure above). Route by section.
        fail_cell = (fail_cell_override if fail_cell_override is not None
                     else _parse_smoke_fail_cell(stderr_tail))
        if fail_cell is None:
            return halt(paths, stage_id,
                        reason="smoke gate failed but could not parse failing cell index",
                        halt_class="internal_contract_violation",
                        retry_count=iteration,
                        context={"stderr": stderr_tail}, state=state)

        try:
            nb = json.loads((paths.run_dir / "notebook.ipynb").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            return halt(paths, stage_id,
                        reason=f"could not read notebook.ipynb for routing: {e}",
                        halt_class="internal_contract_violation",
                        retry_count=iteration, state=state)

        section = cell_index_to_section(fail_cell, nb)

        # (c) — Anti-fixation: if the smoke gate fails at the SAME cell as
        # the previous iteration AND with the SAME exception class, the
        # prior iteration's target_file didn't fix the bug. Add it to
        # failed_targets so this iteration's diagnostician avoids picking
        # it again.
        #
        # If the exception class CHANGED (e.g., RuntimeError → TypeError),
        # the prior fix landed and exposed a new layer — keep the file
        # in scope. The R2C 2026-05-21 bev-distill halt was a false
        # positive on the old guard, which treated same-cell as fixation
        # without checking the exception. Both bugs were genuinely in
        # the notebook, just different layers.
        curr_exc = _extract_exception_class(stderr_tail)
        # Item 20: timeout is a first-class failure kind, not "exception:
        # None" — a repeated timeout at the same cell is fixation (the work
        # was not reduced), and a timeout→exception transition is cascading
        # progress like any other kind change.
        curr_kind = ("timeout" if _smoke_timeout_facts(stderr_tail) is not None
                     else curr_exc)
        if prior_iterations and prior_iterations[-1]["failing_cell"] == fail_cell:
            prior_kind = prior_iterations[-1].get(
                "failure_kind", prior_iterations[-1].get("exception_class"))
            if prior_kind is not None and curr_kind is not None and curr_kind != prior_kind:
                log(stage_id, "smoke_cascade_recognized",
                    f"iteration {iteration}: same cell {fail_cell} but "
                    f"failure kind changed {prior_kind} → {curr_kind}; "
                    f"treating as cascading progress, not fixation. Prior "
                    f"target stays in scope.")
            else:
                prior_target = prior_iterations[-1]["diagnosis"]["target_file"]
                if prior_target not in failed_targets:
                    failed_targets.append(prior_target)
                    detail = ("cell timed out again — the fix did not reduce "
                              "the cell's work"
                              if curr_kind == "timeout"
                              else f"exception unchanged: {curr_exc}")
                    log(stage_id, "smoke_target_failed",
                        f"iteration {iteration}: prior target {prior_target!r} "
                        f"did not unstick cell {fail_cell} "
                        f"({detail}); added to "
                        f"failed_targets (current set: {failed_targets})")

        # Mechanical fallback routing — kept as a backstop. The diagnostician
        # dispatched below overrides this when its analysis succeeds. We log
        # the mechanical pick so the transcript shows whether the diagnostician
        # agreed or overrode.
        producer_by_traceback = smoke_traceback_to_producer(stderr_tail)
        if producer_by_traceback is not None:
            mechanical_producer = producer_by_traceback
            mechanical_basis = "traceback"
        else:
            mechanical_producer = smoke_cell_to_producer(section)
            mechanical_basis = "section_table"
        log(stage_id, "smoke_routing_mechanical",
            f"iteration {iteration}: cell {fail_cell} (§{section}) → "
            f"{mechanical_producer} (basis: {mechanical_basis})")

        failing_cell_source = _extract_failing_cell_source(nb, fail_cell)

        if iteration == STAGE_2_RETRY_CAP:
            # Cap exhausted. Before halting, invoke the halt-judge with the
            # full iteration history so it can either (a) classify the
            # failure and produce a clear human-readable halt diagnosis, or
            # (b) — if high confidence — recommend one final targeted fix
            # dispatch the prior iterations missed (e.g., the BatchNorm
            # placement bug that survived stage 2.d's skip-check and never
            # got picked as a smoke target). Caller halts on low/medium
            # confidence so we don't burn budget on speculative recoveries.
            try:
                outcome = _invoke_judge(
                    state, stage_id=stage_id,
                    validator_label=f"smoke_loop_cap_exhausted:cap={STAGE_2_RETRY_CAP}",
                    stderr_tail=stderr_tail, iteration=iteration,
                    local_dispatchers=_smoke_local_dispatchers(
                        stderr_tail=stderr_tail,
                        failing_cell_source=failing_cell_source,
                        fail_cell=fail_cell, section=section,
                        prior_iterations=prior_iterations,
                        failed_targets=failed_targets,
                    ),
                )
            except (OpencodeClientError, OutOfScopeWritesError, ValueError) as e:
                return halt(
                    paths, stage_id,
                    reason=(
                        f"smoke gate failed after cap={STAGE_2_RETRY_CAP} "
                        f"and halt-judge invocation failed: {e}"
                    ),
                    halt_class=_judge_invocation_halt_class(e),
                    retry_count=iteration,
                    context={"last_failing_cell": fail_cell, "last_section": section,
                             "last_routed_producer": mechanical_producer,
                             "stderr": stderr_tail},
                    state=state,
                )
            if outcome.action == "halt" or outcome.confidence != "high":
                log(stage_id, "judge_cap_degrade",
                    f"cap exhausted; judge classified as {outcome.classification!r} "
                    f"({outcome.confidence}); degrading (notebook exists) and continuing")
                # Fix-loop third option (item 7), give-up point 1: before
                # shipping a broken notebook, the deterministic stub gates
                # may convert the one isolated failing component instead.
                stub_result = _attempt_fix_loop_stub_conversion(
                    state, stage_id,
                    give_up_point="cap exhausted, judge declined recovery",
                    prior_iterations=prior_iterations,
                    failing_cell_source=failing_cell_source,
                    fail_cell=fail_cell, section=section,
                    stderr_tail=stderr_tail)
                if stub_result is not None:
                    return stub_result
                # Type-(D) degrade: the notebook was generated + rendered and runs
                # up to the failing cell. Ship it with a precise failing-cell pointer
                # + failure-kind-honest wording (item 20: a timeout names the
                # budget and wall time, never a traceback).
                what_failed, where, what_to_do = _smoke_degrade_fields(
                    stderr_tail, fail_cell, section)
                return degrade(
                    paths, stage_id,
                    reason=(f"smoke gate failed after cap={STAGE_2_RETRY_CAP}; "
                            f"halt-judge: {outcome.rationale}. stderr: {stderr_tail[-600:]}"),
                    what_failed=what_failed,
                    where=where,
                    what_to_do=what_to_do,
                    state=state,
                )
            # High-confidence recovery: dispatch the judge's pick ONCE more
            # and re-run the smoke gate. Hard halt if it still fails — no
            # additional budget beyond this single extra attempt.
            log(stage_id, "judge_cap_recovery",
                f"cap exhausted; judge high-confidence routed to "
                f"{outcome.target_agent} ({outcome.classification})")
            try:
                outcome.dispatcher(state, [outcome.finding])
            except (OpencodeClientError, OutOfScopeWritesError) as e:
                return halt(
                    paths, stage_id,
                    reason=f"judge-blessed cap-recovery dispatch failed: {e}",
                    halt_class=_dispatch_error_halt_class(e),
                    retry_count=iteration, state=state,
                )
            # Re-render BEFORE re-smoking, like every normal loop iteration.
            # Without this the smoke gate re-executes the STALE notebook:
            # the Rethinking 2026-07-14 cap-recovery landed its blessed fix
            # in notebook_draft.py, "passed" smoke on the pre-fix
            # notebook.ipynb, and delivered a notebook missing its own fix.
            ok_render, render_err = _render_notebook(state)
            if not ok_render:
                return halt(paths, stage_id,
                            reason="render_notebook.py failed after the "
                                   "judge-blessed cap-recovery fix",
                            halt_class="producer_output_invalid",
                            context={"stderr": render_err}, state=state)
            exit_code, _, post_stderr = _run_smoke_gate(state)
            if exit_code == 0:
                log(stage_id, "smoke_passed_after_judge_cap_recovery",
                    "judge-blessed extra iteration unstuck the smoke gate")
                return StageResult(status="completed", stage_id=stage_id,
                                   notes=f"smoke gate clean after judge cap-recovery "
                                         f"(iteration {iteration}+1)")
            # Fix-loop third option (item 7), give-up point 2: the
            # judge-blessed extra is consumed and the failure stands.
            stub_result = _attempt_fix_loop_stub_conversion(
                state, stage_id,
                give_up_point="cap + judge-blessed extra exhausted",
                prior_iterations=prior_iterations,
                failing_cell_source=failing_cell_source,
                fail_cell=fail_cell, section=section,
                stderr_tail=post_stderr)
            if stub_result is not None:
                return stub_result
            what_failed, where, what_to_do = _smoke_degrade_fields(
                post_stderr, fail_cell, section)
            return degrade(
                paths, stage_id,
                reason=(f"smoke gate STILL failed after cap={STAGE_2_RETRY_CAP} + one "
                        f"judge-blessed extra iteration (judge picked {outcome.target_agent}). "
                        f"stderr: {post_stderr[-600:]}"),
                what_failed=what_failed,
                where=where,
                what_to_do=what_to_do,
                state=state,
            )

        # L2 — Diagnostician dispatch. Think-class agent reads the failing cell,
        # traceback, and relevant source files, then writes a structured
        # diagnosis (target_agent, target_file, root_cause, proposed_fix). The
        # driver consumes the diagnosis to (a) pick the right producer for the
        # fix-mode dispatch and (b) inject root_cause + proposed_fix into the
        # SMOKE001 finding so the producer starts with a real diagnosis, not a
        # raw traceback.
        # Clear any prior diagnosis file from a previous iteration — the agent's
        # writeable_paths permit overwriting, but a stale file would mask a
        # silent "agent failed to write" failure.
        diagnosis_path = paths.pipeline_dir / "smoke_diagnosis.json"
        if diagnosis_path.exists():
            diagnosis_path.unlink()
        diag_attempts: list[tuple[str, DispatchResult]] = []
        try:
            diag_attempts.append(("attempt 1", _dispatch_smoke_diagnostician(
                state,
                stderr_tail=stderr_tail,
                failing_cell_source=failing_cell_source,
                failing_cell_index=fail_cell,
                section=section,
                prior_iterations=prior_iterations if prior_iterations else None,
                failed_targets=failed_targets if failed_targets else None,
            )))
        except (OpencodeClientError, OutOfScopeWritesError) as e:
            # pending_finding (item 12, the 07-07 mock-run lesson): the
            # researcher-readable part of a dead dispatch is what it was FOR.
            return halt(paths, stage_id,
                        reason=f"smoke diagnostician dispatch failed: {e}",
                        halt_class=_dispatch_error_halt_class(e),
                        evidence={"pending_finding": (
                            f"diagnose why the demo notebook fails at "
                            f"cell {fail_cell}")},
                        retry_count=iteration, state=state)

        diagnosis, diag_err = _read_smoke_diagnosis(state)
        if diagnosis is None and not diagnosis_path.exists():
            # Missing-output retry, FILE-ABSENT case only (the Think-class
            # stop-early recovery every other Think dispatch already has:
            # decomposer, analyzer, stage-reviewer). This is deterministically
            # distinguishable from the schema-invalid case the OLD
            # unconditional retry mishandled — that case keeps the judge path
            # below. detr 2026-07-04: a 17.7s zero-write diagnostician turn
            # went straight to the judge, which halted a run whose smoke
            # failure had a routable producer fix behind it.
            log(stage_id, "smoke_diagnostician_missing_output_retry",
                f"iteration {iteration}: diagnostician wrote no "
                f"smoke_diagnosis.json; one retry_mode re-dispatch before "
                f"the judge")
            try:
                diag_attempts.append(
                    ("attempt 2 (write-first retry)", _dispatch_smoke_diagnostician(
                        state,
                        stderr_tail=stderr_tail,
                        failing_cell_source=failing_cell_source,
                        failing_cell_index=fail_cell,
                        section=section,
                        prior_iterations=prior_iterations if prior_iterations else None,
                        failed_targets=failed_targets if failed_targets else None,
                        retry_mode=True,
                    )))
            except (OpencodeClientError, OutOfScopeWritesError) as e:
                return halt(paths, stage_id,
                            reason=f"smoke diagnostician missing-output "
                                   f"retry failed: {e}",
                            halt_class=_dispatch_error_halt_class(e),
                            evidence={"pending_finding": (
                                f"diagnose why the demo notebook fails at "
                                f"cell {fail_cell}")},
                            retry_count=iteration, state=state)
            diagnosis, diag_err = _read_smoke_diagnosis(state)
        dispatch_evidence = _diagnostician_dispatch_evidence(diag_attempts)
        if diagnosis is None and not diagnosis_path.exists():
            # Double zero-write (the Think-tier thinking-budget class, detr
            # 2026-07-05: prompt-level write-first cannot preempt a thinking
            # channel that exhausts the output budget before the first tool
            # call). Structural write-first: the driver authors the
            # imperfect-but-honest fallback from facts it already holds and
            # the fix loop PROCEEDS instead of halting. Loud (run event +
            # assumptions.md) and honest (every judgment field says
            # driver-authored). When no honest fallback exists (burned
            # target), fall through to the judge exactly as before.
            fallback = _write_driver_fallback_smoke_diagnosis(
                state,
                mechanical_producer=mechanical_producer,
                stderr_tail=stderr_tail,
                fail_cell=fail_cell,
                section=section,
                exception_class=curr_exc,
                failed_targets=failed_targets,
                dispatch_evidence=dispatch_evidence,
            )
            if fallback is not None:
                log(stage_id, "smoke_diagnosis_driver_fallback",
                    f"iteration {iteration}: both diagnostician turns "
                    f"completed with zero writes; driver authored the "
                    f"fallback diagnosis (target "
                    f"{fallback['target_agent']} / "
                    f"{fallback['target_file']!r}) so the fix loop proceeds")
                _append_run_event(
                    paths, "smoke_diagnosis_driver_fallback",
                    summary=("driver authored a fallback smoke diagnosis "
                             "after two zero-write diagnostician turns"),
                    details={"iteration": iteration,
                             "target_agent": fallback["target_agent"],
                             "target_file": fallback["target_file"],
                             "failing_cell": fail_cell},
                )
                aid = _next_assumption_id(state)
                _append_assumption(
                    state, aid=aid,
                    title="Driver-authored fallback smoke diagnosis",
                    detected=(
                        f"Both automated-debugger analysis turns for notebook "
                        f"cell {fail_cell} finished without producing their "
                        f"diagnosis file (their reasoning exhausted the "
                        f"per-step output budget before writing)."
                    ),
                    action=(
                        f"The driver wrote a coarse fallback diagnosis from "
                        f"the traceback and section routing (fix routed to "
                        f"{fallback['target_agent']} in "
                        f"{fallback['target_file']}) so the fix loop could "
                        f"continue instead of stopping the run."
                    ),
                    reasoning=(
                        "An imperfect but honest routing the pipeline can act "
                        "on beats stopping the run with no diagnosis at all; "
                        "the fallback is labeled driver-authored in every "
                        "field so nothing downstream mistakes it for an "
                        "analyzed diagnosis."
                    ),
                    alternative=(
                        "Stop the run at the first unanalyzed failure and "
                        "have a human debug the failing cell directly."
                    ),
                    override=(
                        "Re-run after fixing the failing cell yourself, or "
                        "raise the model's per-step output budget so the "
                        "analysis turn can complete."
                    ),
                )
                diagnosis, diag_err = _read_smoke_diagnosis(state)
        if diagnosis is None:
            # Judge-routed recovery. Replaces the prior unconditional
            # "missing-output retry" path that mishandled the schema-invalid
            # case (file present but a required field missing → retry prompt
            # told the agent "you didn't write" → agent refused to rewrite).
            # The judge inspects the file's actual state + the read error and
            # decides: repair-dispatch the diagnostician with a structured
            # finding, OR halt with diagnosis.
            log(stage_id, "smoke_diagnostician_judge",
                f"iteration {iteration}: diagnosis unusable ({diag_err}); "
                f"invoking halt-judge")
            judge_stderr = (
                f"{diag_err}\n\n{dispatch_evidence}" if dispatch_evidence
                else diag_err
            )
            try:
                outcome = _invoke_judge(
                    state, stage_id=stage_id,
                    validator_label="schema_validation:smoke_diagnosis",
                    stderr_tail=judge_stderr, iteration=iteration,
                    local_dispatchers=_smoke_local_dispatchers(
                        stderr_tail=stderr_tail,
                        failing_cell_source=failing_cell_source,
                        fail_cell=fail_cell, section=section,
                        prior_iterations=prior_iterations,
                        failed_targets=failed_targets,
                    ),
                )
            except (OpencodeClientError, OutOfScopeWritesError, ValueError) as e:
                return halt(paths, stage_id,
                            reason=f"halt-judge invocation failed on diagnostician output: {e}",
                            halt_class=_judge_invocation_halt_class(e),
                            retry_count=iteration,
                            context={"stderr": stderr_tail,
                                     "diag_err": diag_err,
                                     "dispatch_evidence": dispatch_evidence,
                                     "mechanical_producer": mechanical_producer},
                            state=state)
            if outcome.action == "halt":
                log(stage_id, "judge_halt",
                    f"iteration {iteration}: judge classified diagnosis as "
                    f"{outcome.classification!r} ({outcome.confidence}); halting")
                # No user_message: the judge_halt story leads; the rationale
                # stays behind the collapsible (item 12 rule 3 — this is the
                # exact site whose quoted rationale asserted the detr "900s
                # timeout" that never happened). The structured attempt
                # evidence lets the story state the provable no-write
                # mechanism instead.
                return halt(paths, stage_id,
                            reason=f"halt-judge decided to halt: {outcome.rationale}",
                            halt_class="judge_halt",
                            evidence=_diagnostician_halt_evidence(
                                diag_attempts,
                                diagnosis_file_exists=diagnosis_path.exists()),
                            retry_count=iteration,
                            context={"stderr": stderr_tail,
                                     "diag_err": diag_err,
                                     "dispatch_evidence": dispatch_evidence,
                                     "judge_classification": outcome.classification,
                                     "judge_confidence": outcome.confidence,
                                     "mechanical_producer": mechanical_producer},
                            state=state)
            log(stage_id, "judge_dispatch",
                f"iteration {iteration}: judge routed to {outcome.target_agent} "
                f"({outcome.classification}, {outcome.confidence})")
            try:
                outcome.dispatcher(state, [outcome.finding])
            except (OpencodeClientError, OutOfScopeWritesError) as e:
                return halt(paths, stage_id,
                            reason=f"judge-routed dispatch failed: {e}",
                            halt_class=_dispatch_error_halt_class(e),
                            retry_count=iteration, state=state)
            diagnosis, diag_err = _read_smoke_diagnosis(state)
            if diagnosis is None:
                return halt(paths, stage_id,
                            reason=(
                                f"smoke diagnostician output still unusable after "
                                f"halt-judge repair-dispatch: {diag_err}"
                            ),
                            halt_class="producer_output_invalid",
                            evidence=_diagnostician_halt_evidence(
                                diag_attempts,
                                diagnosis_file_exists=diagnosis_path.exists()),
                            retry_count=iteration,
                            context={"stderr": stderr_tail,
                                     "mechanical_producer": mechanical_producer},
                            state=state)
        diagnosed_target = diagnosis["target_agent"]
        diagnosed_file = diagnosis["target_file"]
        # (c) — Anti-repeat with one-shot consumer-framing retry.
        #
        # If the diagnostician picked a target_file in failed_targets, give it
        # ONE more pass with explicit consumer-framing guidance (the most
        # common misidentification is "the producer returns wrong value" when
        # the actual bug is in a consumer's mutation pattern or index
        # manipulation). The retry forces Tier 2 differential diagnosis
        # (top-3 hypotheses with source quoting per the agent's Step 6).
        # Only halt if the retry STILL picks a forbidden target.
        if diagnosed_file in failed_targets:
            log(stage_id, "smoke_forbidden_target_retry",
                f"iteration {iteration}: diagnostician picked forbidden target "
                f"{diagnosed_file!r}; re-dispatching with consumer-framing prompt")
            try:
                _dispatch_smoke_diagnostician(
                    state,
                    stderr_tail=stderr_tail,
                    failing_cell_source=failing_cell_source,
                    failing_cell_index=fail_cell,
                    section=section,
                    prior_iterations=prior_iterations if prior_iterations else None,
                    failed_targets=failed_targets if failed_targets else None,
                    forbidden_pick_retry=diagnosed_file,
                )
            except (OpencodeClientError, OutOfScopeWritesError) as e:
                return halt(paths, stage_id,
                            reason=f"smoke diagnostician forbidden-target retry dispatch failed: {e}",
                            halt_class=_dispatch_error_halt_class(e),
                            retry_count=iteration, state=state)
            diagnosis, diag_err = _read_smoke_diagnosis(state)
            if diagnosis is None:
                return halt(paths, stage_id,
                            reason=(
                                "smoke diagnostician forbidden-target retry did not "
                                f"produce a valid diagnosis ({diag_err}); halting."
                            ),
                            halt_class="producer_output_invalid",
                            retry_count=iteration,
                            context={"stderr": stderr_tail},
                            state=state)
            diagnosed_target = diagnosis["target_agent"]
            diagnosed_file = diagnosis["target_file"]
            if diagnosed_file in failed_targets:
                return halt(paths, stage_id,
                            reason=(
                                f"diagnostician picked target_file={diagnosed_file!r} "
                                f"for cell {fail_cell}; this file is in the "
                                f"failed_targets set ({failed_targets}) (previously "
                                f"tried for an earlier cell). Re-dispatched with "
                                f"consumer-framing prompt + Tier 2 differential "
                                f"diagnosis (top-3 with source quoting); the "
                                f"diagnostician still picked the same forbidden "
                                f"target. The bug may live in an unsuspected "
                                f"consumer of {diagnosed_file!r}, or the diagnostician "
                                f"is fixated. Halting for human review."
                            ),
                            halt_class="producer_output_invalid",
                            retry_count=iteration,
                            context={"failed_targets": failed_targets,
                                     "diagnosis": diagnosis,
                                     "stderr": stderr_tail,
                                     "note": "forbidden-target retry already attempted"},
                            state=state)
            log(stage_id, "smoke_forbidden_target_recovered",
                f"iteration {iteration}: retry now targeting {diagnosed_file!r} "
                f"(was {diagnosis.get('target_agent', '?')}); proceeding with fix-mode")
        producer = _smoke_target_agent_to_producer_short(diagnosed_target)
        override = "override" if producer != mechanical_producer else "agree"
        log(stage_id, "smoke_routing_diagnosis",
            f"iteration {iteration}: diagnostician → {producer} "
            f"(target_file={diagnosed_file!r}, vs mechanical {mechanical_producer}: {override})")

        try:
            fix_fn = _smoke_producer_to_fix_fn(producer)
            producer_writeable = _smoke_producer_to_writeable_paths(producer)
        except ValueError as e:
            return halt(paths, stage_id, reason=str(e),
                        halt_class="internal_contract_violation",
                        retry_count=iteration, state=state)

        # L1 anchor — deepest owned frame in the producer's writeable paths.
        # Still useful even with the L2 diagnosis: tells the producer agent
        # which line in its own file to start at.
        anchor = smoke_traceback_deepest_owned_frame(stderr_tail, producer_writeable)
        smoke_context = {"traceback_hint": anchor or ""}
        if anchor:
            log(stage_id, "smoke_anchor",
                f"iteration {iteration}: deepest owned frame → {anchor}")

        # Item 20: a timeout failure names itself in the fix dispatch and
        # constrains the ask to work reduction — the bev-distill 07-08 loop
        # asked the notebook-generator to "fix" a cell whose only defect was
        # costing more than the budget, and the fix changed nothing.
        timeout = _smoke_timeout_facts(stderr_tail)
        if timeout is not None:
            failure_line = (
                f"Smoke gate TIMED OUT at notebook cell {fail_cell} "
                f"(§{section} of the notebook layout): the cell ran "
                f"~{timeout['wall_s']:.0f}s against the {timeout['budget_s']}s "
                f"per-cell budget. It did NOT raise — there is no traceback; "
                f"the defect is cost, not a bug."
            )
            fix_constraint = (
                f"\n\nTHIS IS A TIMEOUT, NOT AN ERROR. Constrain your change "
                f"to reducing the cell's work: fewer epochs, a smaller demo "
                f"slice, or splitting the cell so each part fits the "
                f"{timeout['budget_s']}s budget. A logic change that does not "
                f"reduce work will not clear this failure."
            )
        else:
            failure_line = (
                f"Smoke gate failed at notebook cell {fail_cell} "
                f"(§{section} of the notebook layout)."
            )
            fix_constraint = ""
        # A standing data-signal failure leads the finding: with unwinnable
        # demo data, every model-side "fix" burns an iteration against an
        # impossible gate (the Rethinking 2026-07-14 shape).
        data_signal_preamble = (
            f"{data_signal_note}\n\n" if data_signal_note else "")
        finding = {
            "id": "SMOKE001",
            "severity": "critical",
            "description": (
                f"{data_signal_preamble}"
                f"{failure_line}\n\n"
                f"**Root cause (per diagnostician):** {diagnosis['root_cause']}\n\n"
                f"**Diagnostician reasoning:** {diagnosis['reasoning']}\n\n"
                f"**Raw stderr (for reference):**\n{stderr_tail}"
            ),
            "proposed_fix": (
                f"{diagnosis['proposed_fix']}{fix_constraint}\n\n"
                f"Apply this change to `{diagnosis['target_file']}` only. If on "
                f"reading the file you find the diagnostician's proposed fix is "
                f"incorrect, do not silently substitute a different file — your "
                f"writeable_paths allowlist will halt the dispatch. Report the "
                f"disagreement in your status table; the orchestrator will route."
            ),
        }
        data_digest_before = _demo_data_digest(paths)
        try:
            fix_fn(state, [finding], smoke_context=smoke_context)
        except (OpencodeClientError, OutOfScopeWritesError) as e:
            return halt(paths, stage_id,
                        reason=f"smoke fix-mode dispatch to {producer} failed: {e}",
                        halt_class=_dispatch_error_halt_class(e),
                        retry_count=iteration, state=state)

        # Record this iteration for the next iteration's diagnostician. We
        # only capture fields the diagnostician needs to interpret history;
        # we deliberately do NOT include the full stderr (the next iteration
        # will see its own fresh stderr).
        prior_iterations.append({
            "iteration": iteration,
            "failing_cell": fail_cell,
            "section": section,
            "exception_class": curr_exc,
            "failure_kind": curr_kind,
            "mechanical_routing": mechanical_producer,
            "diagnosis": {
                "target_agent": diagnosis["target_agent"],
                "target_file": diagnosis["target_file"],
                "root_cause": diagnosis["root_cause"],
                "proposed_fix": diagnosis["proposed_fix"],
            },
            "producer_dispatched": diagnosed_target,
            # Failure evidence for the stub-gate isolation check (G3): the
            # deepest owned frame in THIS iteration's traceback, None when
            # the failure fired outside method/ (G3 then never fires).
            "deepest_owned_frame": smoke_traceback_deepest_owned_frame(
                stderr_tail, ["method/*.py"]),
        })

        # Re-render (cheap, idempotent — covers the notebook-generator-fix case
        # where notebook_draft.py changed) + re-run the static validator before
        # re-attempting the smoke gate.
        ok, err = _render_notebook(state)
        if not ok:
            return halt(paths, stage_id,
                        reason="render_notebook.py failed after smoke fix-mode",
                        halt_class="producer_output_invalid",
                        retry_count=iteration, context={"stderr": err}, state=state)
        ok, err = _run_stage2_script(
            state, stage_id=stage_id,
            args=["scripts/validate_notebook_output.py"],
        )
        if not ok:
            # A fix is judged on whether it made the static validation
            # state WORSE, not on findings stage 3a already degraded past
            # (the ICRA 2026-07-13 halt: a correct smoke fix landed, then
            # the run died re-litigating the pre-existing accepted
            # failure here).
            new_failures = _notebook_validation_new_failures(state, err)
            if new_failures:
                return halt(paths, stage_id,
                            reason=f"validate_notebook_output.py failed "
                                   f"after smoke fix-mode "
                                   f"({len(new_failures)} failure(s) not "
                                   f"in the stage-3a accepted baseline)",
                            halt_class="producer_output_invalid",
                            retry_count=iteration,
                            context={"stderr": err,
                                     "new_failures": new_failures},
                            state=state)
            _log_accepted_validation_continue(
                state, stage_id, "the smoke loop after the fix")

        # Data-signal check (fix loop only, by design): when THIS iteration's
        # fix changed the demo-data surface, ask deterministically whether
        # the new data can carry the signal the beats-chance gate needs
        # BEFORE burning another smoke iteration on it. Runs after the
        # re-render so the check can read the freshly rendered notebook's own
        # data cells. Any data change resets the note to the fresh result;
        # a standing failure also lands on this iteration's record so the
        # next diagnostician turn knows the gate is unwinnable on this data.
        if _demo_data_digest(paths) != data_digest_before:
            data_signal_note = _post_fix_data_signal_note(state, iteration)
            if data_signal_note and prior_iterations:
                prior_iterations[-1]["data_signal"] = data_signal_note

    # Unreachable: loop returns or halts at cap.
    return halt(paths, stage_id, reason="smoke gate fix loop exited unexpectedly",
                halt_class="internal_contract_violation",
                state=state)


# ---------------------------------------------------------------------------
# Stage 4 — paper-fidelity review (LLM judgment pass over the whole package
#           + notebook). Stage 5 routes the findings.
# ---------------------------------------------------------------------------

_REVIEW_NOTEBOOK_NAME = "notebook_for_review.ipynb"
_REVIEW_OUTPUT_KEEP_CHARS = 1500  # head AND tail kept per long text output


def _strip_notebook_outputs_for_review(nb: dict) -> dict:
    """Outputs-stripped copy of a notebook: image/binary payloads replaced
    with a stub, long text outputs kept head+tail (the smoke-traceback
    lesson: the tail carries the message). Code cells are untouched."""
    def _clip(text: str) -> str:
        keep = _REVIEW_OUTPUT_KEEP_CHARS
        if len(text) <= 2 * keep + 100:
            return text
        return (f"{text[:keep]}\n…[{len(text) - 2 * keep} chars stripped "
                f"for review]…\n{text[-keep:]}")

    for cell in nb.get("cells") or []:
        for output in cell.get("outputs") or []:
            data = output.get("data")
            if isinstance(data, dict):
                for mime in list(data):
                    value = data[mime]
                    size = len(value) if isinstance(value, (str, list)) else 0
                    if not mime.startswith("text/"):
                        data[mime] = (
                            f"[{mime} output stripped for review "
                            f"({size} chars); the rendered notebook.ipynb "
                            f"has the original]")
                    else:
                        joined = "".join(value) if isinstance(value, list) else str(value)
                        data[mime] = _clip(joined)
            if isinstance(output.get("text"), (str, list)):
                text = output["text"]
                joined = "".join(text) if isinstance(text, list) else str(text)
                output["text"] = _clip(joined)
    return nb


def _write_review_notebook(state: PipelineState) -> Path | None:
    """Write `.pipeline/notebook_for_review.ipynb` for the fidelity
    reviewer: identical code cells, stripped outputs. The ICRA 2026-07-14
    stage 4 read the executed notebook whole — a 144KB embedded image made
    ~109k input tokens and the turn dead-stopped twice. Best-effort."""
    nb_path = state.paths.run_dir / "notebook.ipynb"
    if not nb_path.is_file():
        return None
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        stripped = _strip_notebook_outputs_for_review(nb)
        out_path = state.paths.pipeline_dir / _REVIEW_NOTEBOOK_NAME
        out_path.write_text(
            json.dumps(stripped, indent=1) + "\n", encoding="utf-8")
        log("review_hygiene", "notebook_stripped",
            f"{nb_path.stat().st_size} → {out_path.stat().st_size} bytes "
            f"({out_path.name})")
        return out_path
    except Exception as e:  # noqa: BLE001 - hygiene is best-effort
        log("review_hygiene", "notebook_strip_failed",
            f"{type(e).__name__}: {e}")
        return None


def _review_notebook_hygiene_note(review_nb_path: Path | None) -> str:
    if review_nb_path is None:
        return ""
    return (
        f"\n\n**Input hygiene:** when reviewing the notebook, read "
        f"`{review_nb_path}` — an outputs-stripped copy of notebook.ipynb "
        f"with IDENTICAL code and markdown cells (embedded images replaced "
        f"by stubs, long outputs clipped head+tail). The executed "
        f"notebook.ipynb embeds images that can consume your whole context; "
        f"read it directly only if a specific stripped output matters to a "
        f"finding."
    )


def _open_important_findings_note(state: PipelineState) -> str:
    """Stage 4 context: the important-tier findings earlier stage reviews
    raised and no fix loop acted on.

    Second consumer of the `finding_tier_dropped` fix (R2C-068). Without this,
    the fidelity reviewer starts from a blank slate and can certify an element
    the run's own earlier reviewer already declared divergent. On the
    2026-08-06 loop-2 roll it certified "all 8 methodology replication
    contract elements are faithfully implemented" 54 minutes after the stage 2c
    reviewer filed important findings against two of those same elements.

    Context only. The reviewer still reaches its own verdict by reading the
    code; the note tells it what to check, never what to conclude.
    """
    try:
        important, _nice, _caveats = _stage_review_deferred_findings(state)
    except Exception as exc:  # noqa: BLE001 — review context is best-effort
        log("stage_4", "open_important_context_failed", f"{type(exc).__name__}: {exc}")
        return ""
    if not important:
        return ""
    lines = [
        "",
        "",
        f"**Open findings from earlier stage reviews ({len(important)}).** "
        "This run's own stage reviewers raised the findings below and no fix "
        "was routed for them, because the stage fix loops act on critical "
        "findings only. They are unverified as of now: each one may have been "
        "fixed incidentally by later work, may still stand, or may have been "
        "wrong in the first place. Re-read the named code and decide for "
        "yourself. What you must NOT do is certify an element as faithfully "
        "implemented without addressing the objection already on record "
        "against it. If a finding still stands, raise it as your own finding "
        "at whatever severity you judge correct. If the code no longer has "
        "the problem, say so in your summary.",
        "",
    ]
    for fnd in important:
        where = str(fnd.get("location") or "") or str(fnd.get("file") or "")
        check = str(fnd.get("check_id") or "")
        head = f"- `{fnd['id']}` ({fnd.get('raised_at', 'earlier stage')}"
        if check:
            head += f", check `{check}`"
        head += ")"
        if where:
            head += f" — {where}"
        lines.append(head)
        lines.append(f"  - {_clip_for_prompt(fnd.get('description'), 600)}")
        fix = str(fnd.get("proposed_fix") or "").strip()
        if fix:
            lines.append(f"  - Reviewer's proposed fix: {_clip_for_prompt(fix, 400)}")
    return "\n".join(lines)


def _clip_for_prompt(text, limit: int) -> str:
    """One-line clip for prompt context: collapse whitespace, cap length."""
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + " […]"


def _dispatch_paper_fidelity_reviewer(
    state: PipelineState, *, write_first_retry: bool = False,
) -> DispatchResult:
    """`write_first_retry`: set by run_stage_4's missing-output retry. The
    review is a Think-tier artifact like the judge decision — when the first
    turn dies with no Write, an identical re-dispatch tends to die
    identically (detr 2026-07-04 evidence, generalized), so the retry
    inverts to write-first."""
    agent = "r2c-paper-fidelity-reviewer"
    review_path = state.paths.pipeline_dir / "review_report.json"
    review_nb_path = _write_review_notebook(state)
    prompt = build_dispatch_prompt(
        task_summary=STAGE_TASK_SUMMARIES["stage_4_review"],
        paths=build_paths_block(state.paths),
        closing=REVIEWER_CLOSING,
        writeable_paths=WRITEABLE_PATHS[agent],
        think_anchor_output_path=str(review_path),
    ) + _review_notebook_hygiene_note(review_nb_path) \
      + _open_important_findings_note(state)
    if write_first_retry:
        prompt = build_write_first_retry_preamble(
            artifact_name=str(review_path), artifact_kind="review report",
        ) + prompt
    # Branch A opt-in (maintainer-approved 2026-07-03, reversing the 2026-06-25
    # audit exclusion): the detr-distill 2026-07-03 drift — the reviewer
    # overwrote the completed method_explanations.json sidecar with its own
    # equation analysis instead of writing review_report.json — is exactly
    # the wander shape branch A recovers (revert the stray from the content
    # snapshot, one pointed corrective re-dispatch, halt if it drifts
    # again). The fix-mode and incremental dispatches keep the hard halt
    # until they show a live case of their own (two-cases rule).
    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt=prompt,
        timeout_s=PAPER_FIDELITY_REVIEWER_TIMEOUT_S,
        recovery_check_fn=_review_report_valid,
        allow_corrective_redispatch=True,
    )


def _dispatch_paper_fidelity_reviewer_fix(
    state: PipelineState, findings: list[dict]
) -> DispatchResult:
    agent = "r2c-paper-fidelity-reviewer"
    review_path = state.paths.pipeline_dir / "review_report.json"
    prompt = build_fix_mode_prompt(
        target_agent=agent,
        findings=findings,
        paths=build_paths_block(state.paths),
        writeable_paths=WRITEABLE_PATHS[agent],
        think_anchor_output_path=str(review_path),
    )
    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt=prompt,
        timeout_s=PAPER_FIDELITY_REVIEWER_TIMEOUT_S,
        recovery_check_fn=_review_report_valid,
    )


# Late registration — _dispatch_paper_fidelity_reviewer_fix is now defined.
JUDGE_FIX_DISPATCHERS["r2c-paper-fidelity-reviewer"] = _dispatch_paper_fidelity_reviewer_fix


def _dispatch_paper_fidelity_reviewer_incremental(
    state: PipelineState,
    iteration: int,
    prior_finding_ids: list[str],
    fix_dispatched_to: list[str],
) -> DispatchResult:
    """Stage 5 re-review dispatch (post-fix). Uses the incremental template
    instead of the full Stage 4 framing — the reviewer's job here is narrow:
    verify each prior finding by re-reading the file:line it pointed at,
    confirm the change matches proposed_fix, mark resolved or still-present.

    The plan's INCREMENTAL_REREVIEW_TEMPLATE wants "the producer's fix-mode
    status table appended below" as context. To avoid an extra HTTP roundtrip
    fetching each fix-mode dispatch's assistant message text, we pass a
    synthetic placeholder naming which producers ran fix-mode. The reviewer
    can still complete its task by reading the files directly."""
    summary = INCREMENTAL_REREVIEW_TEMPLATE.format(
        iteration=iteration,
        prior_finding_ids=", ".join(prior_finding_ids) or "(none)",
    )
    appended_context = (
        "\n\n**Producer fix-mode dispatches this iteration:**\n"
        + "\n".join(f"  - {a}" for a in fix_dispatched_to)
        + "\n\n(Fix-mode status tables are not captured by the driver; verify "
        "each prior finding by reading the file:line cited in the prior "
        "report and checking against proposed_fix.)"
    )
    agent = "r2c-paper-fidelity-reviewer"
    review_path = state.paths.pipeline_dir / "review_report.json"
    review_nb_path = _write_review_notebook(state)
    prompt = build_dispatch_prompt(
        task_summary=summary + appended_context,
        paths=build_paths_block(state.paths),
        closing=REVIEWER_CLOSING,
        writeable_paths=WRITEABLE_PATHS[agent],
        think_anchor_output_path=str(review_path),
    ) + _review_notebook_hygiene_note(review_nb_path)
    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt=prompt,
        timeout_s=PAPER_FIDELITY_REVIEWER_TIMEOUT_S,
        recovery_check_fn=_review_report_valid,
    )


def _run_review_report_validator(state: PipelineState) -> tuple[bool, str]:
    """validate_review_report.py takes only --run-dir (no --spec)."""
    proc = run_script(
        "stage_4",
        ["scripts/validate_review_report.py", "--run-dir", str(state.paths.run_dir)],
        timeout=60,
    )
    err_tail = _stderr_excerpt(proc.stderr.strip() or proc.stdout.strip())
    ok = proc.returncode == 0
    _append_validation_event(
        state.paths,
        stage_id="stage_4",
        validator="validate_review_report.py",
        ok=ok,
        stderr_tail=err_tail,
        artifacts=[str(state.paths.pipeline_dir / "review_report.json")],
    )
    return ok, err_tail


def run_stage_4(state: PipelineState) -> StageResult:
    """4 — paper-fidelity review. Single LLM pass over the assembled package
    + notebook + spec; the reviewer scans for emergent properties that
    per-stage reviewers can't see by design (cross-stage inconsistencies,
    narrative coherence, scope creep). Cap=1 fix-mode retry on validator
    failure (per the archived v2 orchestrator spec (internal, not shipped) — that's a reviewer prompt bug to surface,
    not something to band-aid)."""
    stage_id = "stage_4"
    paths = state.paths
    skipped = _skip_if_done(paths, stage_id, [
        paths.pipeline_dir / "review_report.json",
    ])
    if skipped is not None:
        return skipped
    try:
        _dispatch_paper_fidelity_reviewer(state)
    except (OpencodeClientError, OutOfScopeWritesError) as e:
        return halt(paths, stage_id,
                    reason=f"paper-fidelity-reviewer dispatch failed: {e}",
                    halt_class=_dispatch_error_halt_class(e),
                    state=state)

    review_path = paths.pipeline_dir / "review_report.json"
    for iteration in range(STAGE_4_RETRY_CAP + 1):
        if not review_path.exists():
            if iteration == STAGE_4_RETRY_CAP:
                return degrade(paths, stage_id,
                            reason="paper-fidelity reviewer did not produce review_report.json after retry",
                            what_failed="paper-fidelity review could not be completed",
                            what_to_do=_REVIEW_DEGRADE_TODO,
                            state=state)
            log(stage_id, "missing_output_retry",
                f"iteration {iteration}: review_report.json missing; "
                f"re-dispatching with the write-first retry preamble")
            try:
                _dispatch_paper_fidelity_reviewer(state, write_first_retry=True)
            except (OpencodeClientError, OutOfScopeWritesError) as e:
                return halt(paths, stage_id,
                            reason=f"paper-fidelity-reviewer retry dispatch failed: {e}",
                            halt_class=_dispatch_error_halt_class(e),
                            retry_count=iteration, state=state)
            continue
        ok, err_tail = _run_review_report_validator(state)
        if ok:
            return StageResult(status="completed", stage_id=stage_id,
                               notes="paper-fidelity review + validator clean")
        if iteration == STAGE_4_RETRY_CAP:
            return degrade(paths, stage_id,
                        reason=f"validate_review_report.py failed after cap=1 fix-mode retry: {err_tail[-500:]}",
                        what_failed="paper-fidelity review report failed validation",
                        where="review_report.json",
                        what_to_do=_REVIEW_DEGRADE_TODO,
                        state=state)
        # Judge-routed validator failure (replaces the prior mechanical
        # "always dispatch reviewer fix-mode" path). The judge inspects the
        # validation error + review_report.json + the spec and decides
        # whether the reviewer can fix the schema violation or whether this
        # is an upstream/pipeline-bug class that should halt with diagnosis.
        try:
            outcome = _invoke_judge(
                state, stage_id=stage_id,
                validator_label="validate_review_report.py",
                stderr_tail=err_tail, iteration=iteration,
            )
        except (OpencodeClientError, OutOfScopeWritesError, ValueError) as e:
            return halt(
                paths, stage_id,
                reason=f"halt-judge invocation failed at validate_review_report.py: {e}",
                halt_class=_judge_invocation_halt_class(e),
                retry_count=iteration,
                context={"stderr": err_tail}, state=state,
            )
        if outcome.action == "halt":
            log(stage_id, "judge_degrade",
                f"iteration {iteration}: judge classified as "
                f"{outcome.classification!r} ({outcome.confidence}); "
                f"degrading (review is non-load-bearing) and continuing")
            return degrade(
                paths, stage_id,
                reason=f"halt-judge decided to halt ({outcome.classification}/"
                       f"{outcome.confidence}): {outcome.rationale}",
                what_failed="paper-fidelity review could not be completed",
                what_to_do=_judge_halt_user_message(outcome.rationale),
                state=state,
            )
        log(stage_id, "judge_dispatch",
            f"iteration {iteration}: judge routed to {outcome.target_agent} "
            f"({outcome.classification}, {outcome.confidence})")
        try:
            outcome.dispatcher(state, [outcome.finding])
        except (OpencodeClientError, OutOfScopeWritesError) as e:
            return halt(paths, stage_id,
                        reason=f"judge-routed fix-mode dispatch failed: {e}",
                        halt_class=_dispatch_error_halt_class(e),
                        retry_count=iteration, state=state)
    return halt(paths, stage_id,
                reason="run_stage_4 loop exited unexpectedly",
                halt_class="internal_contract_violation",
                state=state)


# ---------------------------------------------------------------------------
# Stage 5 — findings routing. Critical findings auto-route to fix-mode;
#           important findings log-only (MVP — interactive ask_user is a
#           follow-up); nice-to-have + human-routed log only.
# ---------------------------------------------------------------------------


def _stage5_fix_dispatcher(target_agent_name: str):
    """Map a target_agent name (from review findings) to the fix-mode
    dispatch function. The set covers every producer Stage 5 can route to."""
    table = {
        "analyzer": _dispatch_analyzer_fix,
        "method-analyzer": _dispatch_analyzer_fix,
        "decomposer": _dispatch_decomposer_fix,
        "architecture-coder": _dispatch_arch_coder_fix,
        "method-coder": _dispatch_method_coder_fix,
        "notebook-generator": _dispatch_notebook_generator_fix,
        # Fully-qualified r2c-* names also accepted (the reviewer may use either form)
        "r2c-method-analyzer": _dispatch_analyzer_fix,
        "r2c-decomposer": _dispatch_decomposer_fix,
        "r2c-architecture-coder": _dispatch_arch_coder_fix,
        "r2c-method-coder": _dispatch_method_coder_fix,
        "r2c-notebook-generator": _dispatch_notebook_generator_fix,
    }
    if target_agent_name in table:
        return table[target_agent_name]
    return None  # caller logs as unroutable; finding stays open


def _ask_user_log_only(
    state: PipelineState, findings: list[dict], audience_label: str
) -> None:
    """MVP for important / nice-to-have / human-routed findings: append to
    a human-readable log file under `<RUN_DIR>/`. The interactive
    SSE-based `ask_user` flow is deferred per the design plan's risk
    mitigation; the log gives the researcher everything they need to
    decide whether to address each finding."""
    if not findings:
        return
    log_path = run_layout.run_path(state.paths.run_dir, run_layout.DEFERRED_FINDINGS_MD)
    header_needed = not log_path.exists()
    with log_path.open("a", encoding="utf-8") as f:
        if header_needed:
            f.write("# Findings deferred to manual review\n\n")
            f.write(
                "Written by `scripts/run_pipeline.py` at Stage 5 routing. "
                "Critical findings auto-routed via fix-mode and are not listed "
                "here. The findings below are surfaced for human judgment.\n\n"
            )
        f.write(f"## {audience_label} ({len(findings)})\n\n")
        for fnd in findings:
            fid = fnd.get("id", "F???")
            sev = fnd.get("severity", "?")
            target = fnd.get("target_agent", "?")
            desc = fnd.get("description", "")
            fix = fnd.get("proposed_fix", "")
            f.write(f"### {fid} ({sev}, target_agent={target})\n\n")
            # A reviewer names the file and, separately, the symbol/line range
            # inside it. Both matter to whoever acts on the finding, so keep
            # them both instead of letting the file shadow the location.
            file_name = str(fnd.get("file") or "")
            location = str(fnd.get("location") or "")
            if not file_name or not location:
                file_loc = file_name or location
            elif location in file_name or file_name in location:
                # One already spells out the other ("method/x.py" against
                # "method/x.py:99"); keep whichever says more.
                file_loc = max(file_name, location, key=len)
            else:
                file_loc = f"{file_name} — {location}"
            if file_loc:
                f.write(f"**Location:** {file_loc}\n\n")
            raised_at = fnd.get("raised_at")
            if raised_at:
                status = str(fnd.get("resolution_status") or "").strip()
                suffix = f" and is still open ({status})" if status else ""
                f.write(
                    f"**Raised by:** the {raised_at} stage review{suffix}. "
                    f"No fix was routed, because the stage fix loops act on "
                    f"critical findings only.\n\n"
                )
            f.write(f"**Description:** {desc}\n\n")
            if fix:
                f.write(f"**Proposed fix:** {fix}\n\n")
            promoted_to = fnd.get("disclosure_promoted_to")
            if promoted_to:
                f.write(f"**Promoted to assumptions.md as {promoted_to}.**\n\n")
            f.write("---\n\n")


# Severity tokens the stage reviewers actually emit. The schema vocabulary is
# critical / important / nice-to-have (schemas/review_report.py), but reviewer
# output has historically also carried the underscore spelling and the
# minor/major synonyms, so each tier normalizes a small alias set instead of
# trusting one spelling.
_NICE_SEVERITY_TOKENS = frozenset({"nice-to-have", "nice_to_have", "minor"})
_IMPORTANT_SEVERITY_TOKENS = frozenset({"important", "major"})

# `resolution_status` values that mean the finding was actually handled. Every
# other value (pending, failed, needs_user, or the field absent) leaves it
# OPEN, and an open important-tier finding must reach the researcher.
_RESOLVED_STATUS_TOKENS = frozenset({"applied", "resolved"})

# Reviewer stages whose own consumption already BLOCKS on the important tier.
# run_stage_2x gates on route_findings.should_halt_stage, so an important
# finding there has exactly four fates: a structured resolution is applied, a
# safe-direction provenance finding is routed to assumptions.md, a needs-user
# finding is routed to assumptions.md, or the run HALTS before any delivery.
# Nothing is dropped, so collecting those findings here would double-report a
# handled one — and the entry would claim on a delivered surface that nobody
# acted on it. The applied resolutions merge in memory, which is why the
# on-disk review file still reads `pending` and cannot be trusted for these
# stages. example_runs/pdwa is the standing evidence: both of its stage-2x
# important findings read `pending` on disk while assumptions.md carries the
# relabel it applied (A004) and the needs-user entry it raised (A003).
#
# Every reviewer stage NOT listed here routes critical findings only (the
# tier-3 arm of run_fix_loop, and run_stage_1's reviewer loop), which is the
# gap this collector closes. Add a stage here only when its consumption starts
# blocking on the important tier.
_IMPORTANT_TIER_BLOCKING_STAGES = frozenset({"stage_2x_params"})


def _stage_review_deferred_findings(
    state: PipelineState,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Collect non-blocking stage-review context for final delivery.

    Returns `(important, nice, caveats)`.

    Stage reviewers can surface useful findings and caveats before Stage 4's
    final paper-fidelity review runs. Stage 5 still lets the final review drive
    repair routing, but a completed stage review's own findings should remain
    visible to the researcher instead of disappearing.

    The important tier is collected here because of the `finding_tier_dropped`
    failure class (R2C-068): the stage fix loops route ONLY critical findings,
    so an important-tier stage-review finding used to fall between "critical
    routes to a fix" and "nice-to-have gets disclosed" and reached no surface
    at all. The 2026-08-06 loop-2 roll closed stage 2c as "2 findings, 0
    critical" and threw away two real algorithm defects that way. Disclosure
    only: carrying a finding here never routes a fix and never moves the label.
    """
    important: list[dict] = []
    nice: list[dict] = []
    caveats: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    # The important tier dedupes WITHOUT the review file name, because a
    # resolution re-ask writes `stage_review_<stage>_resolution.json` beside
    # the original review and both match the glob. The same finding appearing
    # in both must produce one delivered entry, not two.
    seen_important: set[tuple[str, str, str]] = set()
    for path in sorted(state.paths.pipeline_dir.glob("stage_review_*.json")):
        try:
            review = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stage_id = str(review.get("stage_id") or path.stem.removeprefix("stage_review_"))
        for idx, finding in enumerate(review.get("findings") or [], start=1):
            if not isinstance(finding, dict):
                continue
            severity = str(finding.get("severity") or "").lower()
            if severity in _IMPORTANT_SEVERITY_TOKENS:
                if stage_id in _IMPORTANT_TIER_BLOCKING_STAGES:
                    continue  # that stage handles the tier itself
                entry = _normalize_open_important_finding(
                    finding, stage_id=stage_id, review_file=path.name, idx=idx,
                )
                if entry is None:
                    continue  # already resolved — nothing open to disclose
                dedupe_key = (
                    str(entry.get("check_id") or entry.get("id") or ""),
                    entry["description"][:200],
                    str(entry.get("location") or entry.get("file") or ""),
                )
                if dedupe_key in seen_important:
                    continue
                seen_important.add(dedupe_key)
                important.append(entry)
                continue
            if severity not in _NICE_SEVERITY_TOKENS:
                continue
            description = str(finding.get("description") or "")
            dedupe_key = (path.name, description, str(finding.get("file") or finding.get("location") or ""))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            normalized = dict(finding)
            normalized.setdefault("id", f"{stage_id}:nice:{idx}")
            normalized["severity"] = "nice-to-have"
            normalized.setdefault("target_agent", finding.get("target_agent") or "human")
            normalized.setdefault("file", finding.get("file") or path.name)
            nice.append(normalized)
        for idx, caveat in enumerate(review.get("caveats") or [], start=1):
            if not caveat:
                continue
            # A caveat is usually a free string. A reviewer that wants to route
            # a disclosure structurally (item 2) emits a dict carrying `text`
            # (or `note`) plus a `disclosure_request`; keep the text as the
            # description and carry the structured request through.
            disclosure_request = None
            if isinstance(caveat, str):
                description = caveat
            elif isinstance(caveat, dict):
                description = str(caveat.get("text") or caveat.get("note") or "").strip()
                if not description:
                    description = json.dumps(caveat, sort_keys=True)
                disclosure_request = caveat.get("disclosure_request")
            else:
                description = json.dumps(caveat, sort_keys=True)
            dedupe_key = (path.name, description, "caveat")
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            caveats.append({
                "id": f"{stage_id}:caveat:{idx}",
                "severity": "nice-to-have",
                "target_agent": "human",
                "file": path.name,
                "description": description,
                "disclosure_request": disclosure_request,
            })
    return important, nice, caveats


def _normalize_open_important_finding(
    finding: dict, *, stage_id: str, review_file: str, idx: int,
) -> dict | None:
    """One important-tier stage-review finding, shaped for the delivered
    deferred-findings surface. Returns None when the finding is recorded as
    resolved, since a fixed defect is not an open disclosure.

    The entry names its owning producer and the stage review that raised it,
    so a researcher reading the surface knows who wrote the code and where
    the objection came from.
    """
    status = str(finding.get("resolution_status") or "").lower()
    if status in _RESOLVED_STATUS_TOKENS:
        return None
    entry = dict(finding)
    entry["severity"] = "important"
    entry["id"] = str(finding.get("id") or f"{stage_id}:important:{idx}")
    entry["description"] = str(finding.get("description") or "")
    entry["target_agent"] = str(finding.get("target_agent") or "") or "human"
    entry["raised_at"] = stage_id
    entry["raised_in"] = review_file
    # `file` is the source file the reviewer pointed at; fall back to the
    # review file only when the reviewer named none, matching the nice tier.
    entry["file"] = str(finding.get("file") or "") or review_file
    entry["resolution_status"] = status or "pending"
    return entry


# A caveat/finding whose prose ASKS for a researcher surface (a REQUEST-shaped
# verb + surface name in one sentence) is a documentation request the driver
# should honor even when the reviewer forgot the structured disclosure_request
# field (item 2, the GBALD Eq. 13 caveat). Request shape matters, not just verb
# presence: "is already documented in METHOD.md" is a statement and "the
# dropout flag in REPORT.md" uses the verb as a noun — neither may promote.
# The bridge only adds noise to assumptions.md when wrong, never a label
# change, but assumptions.md is a trust surface, so precision stays first.
_DISCLOSURE_REQUEST_RE = re.compile(
    # (a) request modal + passive: "should be documented in assumptions.md"
    r"\b(?:should|must|needs?\s+to|ought\s+to|has\s+to)\s+be\s+(?:\w+ly\s+)?"
    r"(?:documented|disclosed|recorded|flagged|noted)\b[^.]*?"
    r"\b(?:assumptions\.md|method\.md|report\.md)\b"
    # (b) base-form verb at an imperative/modal position: "Document this ...
    #     in assumptions.md", "please flag ... in REPORT.md" — never the bare
    #     word mid-sentence, where flag/record are usually nouns
    r"|(?:^|(?<=[.;:!?]\s)|(?<=\bshould\s)|(?<=\bmust\s)|(?<=\bplease\s)|(?<=\bto\s))"
    r"(?:document|disclose|record|flag|note)\b[^.]*?"
    r"\b(?:assumptions\.md|method\.md|report\.md)\b"
    # (c) asking for a surface entry outright: "needs an assumptions.md entry"
    r"|\b(?:assumptions\.md|method\.md|report\.md)\s+entry\b",
    re.IGNORECASE,
)


def _caveat_requests_disclosure(text: str) -> bool:
    """True when the caveat text asks, in prose, to be documented on a
    researcher-read surface (the legacy shape the text bridge promotes)."""
    return bool(_DISCLOSURE_REQUEST_RE.search(text or ""))


def _promote_reviewer_disclosures(
    state: "PipelineState", items: list[dict]
) -> dict[str, str]:
    """Promote nice-to-have reviewer caveats/findings that request researcher
    disclosure into assumptions.md (item 2 — the 07-06 audit found the GBALD
    Eq. 13 disclosure buried in deferred_findings.md, a file no researcher
    reads).

    An item qualifies via a structured `disclosure_request` (surface
    == "assumptions") or the deterministic text bridge. Each promotion writes
    one assumptions.md entry through the existing writer, emits a
    `reviewer_disclosure_routed` event, and annotates the item in place with
    `disclosure_promoted_to` so the deferred_findings.md entry cross-references
    it. Disclosure only: never changes severity or the delivery label.

    Idempotent via `.pipeline/reviewer_disclosures.json` (keyed by item id):
    a re-run re-annotates already-promoted items with their existing
    assumption id and appends nothing new. Returns {item_id: assumption_id}
    for every promoted item."""
    marker_path = state.paths.pipeline_dir / "reviewer_disclosures.json"
    try:
        promoted = json.loads(marker_path.read_text(encoding="utf-8"))
        if not isinstance(promoted, dict):
            promoted = {}
    except (OSError, json.JSONDecodeError):
        promoted = {}

    changed = False
    for item in items:
        iid = str(item.get("id") or "")
        if not iid:
            continue
        text = str(item.get("description") or "")
        req = item.get("disclosure_request")
        structured = isinstance(req, dict) and req.get("surface") == "assumptions"
        if not (structured or _caveat_requests_disclosure(text)):
            continue
        if iid in promoted:
            item["disclosure_promoted_to"] = promoted[iid]  # keep the cross-ref stable
            continue
        # Unattended-safe: a write failure on this nice-to-have promotion logs
        # and continues (never kills a run), and leaves the item unmarked so a
        # resume retries it.
        try:
            aid = _next_assumption_id(state)
            note = req.get("note") if structured else None
            _append_assumption(
                state,
                aid=aid,
                title=f"Reviewer disclosure — {iid}",
                detected=f"{text}\n\n(Source: {item.get('file') or 'stage review'}.)",
                action="Disclosed to you here; no behavioral change was made to the delivery.",
                reasoning=(
                    note
                    or "A stage reviewer flagged this for your attention and asked that "
                    "it be documented. Routing it here so it reaches a surface the "
                    "delivery report consolidates, instead of only deferred_findings.md."
                ),
                alternative=None,
                override=(
                    "This entry records a disclosure, not a change. If you disagree with "
                    "the delivered behavior, inspect the cited file/location and edit the "
                    "generated package."
                ),
            )
        except Exception as exc:  # noqa: BLE001 — nice-to-have tier never kills a run
            log("stage_5", "disclosure_promotion_failed", f"{iid}: {exc}")
            continue
        _append_run_event(
            state.paths,
            "reviewer_disclosure_routed",
            stage_id="stage_5",
            status="routed",
            summary=f"Routed reviewer caveat {iid} to assumptions.md as {aid}",
            details={
                "caveat_id": iid,
                "assumption_id": aid,
                "via": "structured" if structured else "text_bridge",
            },
        )
        promoted[iid] = aid
        item["disclosure_promoted_to"] = aid
        changed = True

    if changed:
        marker_path.write_text(
            json.dumps(promoted, indent=2) + "\n", encoding="utf-8"
        )
    return dict(promoted)


# ---------------------------------------------------------------------------
# Pass 0 auto-resolution: apply spec-asserted invariant fixes deterministically
# and log to <run_dir>/assumptions.md. Designed for unattended background runs:
# the pipeline applies an expert-default fix rather than halting, and surfaces
# every decision in one auditable file the researcher reads at the end.
# ---------------------------------------------------------------------------


def _next_assumption_id(state: "PipelineState") -> str:
    path = state.paths.run_dir / run_layout.ASSUMPTIONS_MD
    if not path.exists():
        return "A001"
    import re
    content = path.read_text(encoding="utf-8")
    ids = re.findall(r"^## (A\d{3}) ", content, flags=re.MULTILINE)
    if not ids:
        return "A001"
    return f"A{max(int(i[1:]) for i in ids) + 1:03d}"


def _append_assumption(
    state: "PipelineState", *,
    aid: str, title: str, detected: str, action: str,
    reasoning: str, alternative: str | None, override: str,
) -> None:
    path = run_layout.run_path(state.paths.run_dir, run_layout.ASSUMPTIONS_MD)
    header_needed = not path.exists()
    with path.open("a", encoding="utf-8") as f:
        if header_needed:
            f.write("# Pipeline assumptions\n\n")
            f.write(
                "Decisions the system made without human intervention during this run. "
                "The researcher should review each entry to confirm the choice matches "
                "their expectation; alternatives are listed for cases where a different "
                "policy might be preferred. Each entry includes an explicit override path.\n\n"
                "---\n\n"
            )
        f.write(f"## {aid} — {title}\n\n")
        f.write(f"**Detected.** {html.escape(str(detected), quote=False)}\n\n")
        f.write(f"**Action taken.** {html.escape(str(action), quote=False)}\n\n")
        # Last-resort floor: a rendered "**Reasoning.**" with nothing after
        # it reads as a formatting bug to the researcher, whatever the
        # caller failed to pass.
        reasoning_text = str(reasoning or "").strip() or (
            "No reasoning text accompanied this resolution; the Detected "
            "and Action entries above are the full record of this change."
        )
        f.write(f"**Reasoning.** {reasoning_text}\n\n")
        if alternative:
            f.write(f"**Alternative an expert might prefer.** {alternative}\n\n")
        f.write(f"**To override.** {override}\n\n")
        f.write("---\n\n")


def _append_needs_user(state: "PipelineState", finding: dict) -> str:
    aid = _next_assumption_id(state)
    fid = finding.get("id", "F???")
    desc = finding.get("description", "")
    fail = finding.get("_apply_failure")
    detected = f"Finding {fid}: {desc}"
    if fail:
        detected += f"\n\n(Driver attempted auto-resolve but it failed: {fail})"
    _append_assumption(
        state, aid=aid,
        title=f"NEEDS USER ATTENTION — finding {fid}",
        detected=detected,
        action="No automatic action taken; the reviewer flagged this finding as having "
               "no safe expert default for this run's configuration.",
        reasoning="When the reviewer cannot construct a reliable default reconciliation "
                  "(e.g., exotic data preprocessing, contradictory spec entry, regen "
                  "convergence failure), it surfaces here for human judgment rather "
                  "than guessing.",
        alternative=finding.get("proposed_fix"),
        override=f"Inspect {finding.get('file', '?')} at {finding.get('location', '?')}, "
                 f"apply a fix, and re-run the pipeline.",
    )
    return aid


def finalize_assumptions_md(state: "PipelineState") -> None:
    """Post-run readability pass over assumptions.md: prepend a 'how to use'
    intro + a Summary TOC (one severity-iconed line per entry) and sort the
    body so NEEDS-USER entries come first.

    HRI-readiness (hri-readiness-plan Task 2): a chronological list of 6-field
    blocks buried the 1-2 entries that actually need a researcher decision. The
    TOC lets them scan, spot the red entries, and jump to them.

    Idempotent: it re-parses only the `## A### — ` entry blocks (discarding any
    previously-generated intro/TOC), so running it twice — or after more entries
    are appended — reproduces the same structure. No-op if assumptions.md is
    absent or has no entries."""
    import re  # local: stdlib, only needed here

    path = state.paths.run_dir / run_layout.ASSUMPTIONS_MD
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    # Each entry starts at a line "## A### — ...". Split on that boundary
    # (zero-width lookahead) and keep only the entry blocks; this drops any
    # prior intro/TOC/header, which is what makes re-running idempotent.
    blocks = [
        b for b in re.split(r"(?m)^(?=## A\d{3} — )", content)
        if re.match(r"## A\d{3} — ", b)
    ]
    if not blocks:
        return
    parsed: list[tuple[str, str, bool, str]] = []
    for b in blocks:
        m = re.match(r"## (A\d{3}) — (.*)", b)
        aid, title = m.group(1), m.group(2).strip()
        needs_user = "NEEDS USER ATTENTION" in title
        parsed.append((aid, title, needs_user, b.rstrip() + "\n\n"))
    # NEEDS-USER first, then auto-applied; stable by id within each group.
    parsed.sort(key=lambda t: (not t[2], t[0]))

    toc = "\n".join(
        f"- {'🔴' if nu else '🟢'} {aid} — {title}"
        for aid, title, nu, _ in parsed
    )
    needs_n = sum(1 for _, _, nu, _ in parsed if nu)
    auto_n = len(parsed) - needs_n
    intro = (
        "Decisions the system made automatically during this run. "
        "**🔴 entries need your decision; 🟢 entries were applied for you** "
        "(FYI — you can override any of them via the \"To override\" line in "
        "the entry). Scan the summary, then jump to the 🔴 entries first.\n\n"
        f"_{needs_n} need your attention · {auto_n} applied automatically._\n\n"
    )
    rebuilt = (
        "# Pipeline assumptions\n\n"
        + intro
        + "## Summary\n\n"
        + toc
        + "\n\n---\n\n"
        + "".join(body for _, _, _, body in parsed)
    )
    path.write_text(rebuilt, encoding="utf-8")


# ---------------------------------------------------------------------------
# KNOWN_ISSUES.md — the degrade-and-continue researcher report. When a quality
# gate can't be satisfied within the recovery cap but a usable artifact exists,
# the driver logs the issue here and CONTINUES (see degrade() + the halt->degrade
# redesign doc). The researcher gets the notebook + package + this file, which
# they can hand to an AI assistant to finish.
# ---------------------------------------------------------------------------

# The stable heading of the terminal-halt entry that explanation-only
# finalization writes into KNOWN_ISSUES.md. Shared between the writer
# (_finalize_explanation_only_package) and the outcome-aware pruning in
# PipelineState.record so a later invocation that re-runs the halted stage
# can recognize and retire exactly this entry — a halt claim ("the run
# stopped here; treat code artifacts as unreliable") is invocation-scoped
# truth, unlike degrade entries, which describe delivered-artifact quality
# and survive a resume unless their stage later completes cleanly.
_EXPLANATION_ONLY_WHAT_FAILED = (
    "R2C could not produce a reliable code package for this run"
)

_KNOWN_ISSUES_INTRO = (
    "This package was produced end-to-end, but the items below are quality gates "
    "R2C could not fully satisfy automatically within its retry budget. **The "
    "artifacts are delivered as best-effort** — review these before relying on "
    "the output. Each entry says what failed, where, and what to try (including "
    "handing the notebook + this file to an AI assistant like Claude to finish). "
    "R2C never edits your code to force a gate to pass; it surfaces issues here "
    "honestly instead."
)

_KNOWN_ISSUES_STOPPED_INTRO = (
    "This run stopped before R2C could deliver a reliable code package. The "
    "items below record the stop and any quality limits discovered beforehand. "
    "Explanation or partial artifacts may still be available, but they are not "
    "an end-to-end package; use REPORT.md as the authoritative guide to what "
    "exists and what to do next."
)


def _rewrite_known_issues_intro(path: Path, intro: str) -> None:
    """Replace only the generated KNOWN_ISSUES preamble, idempotently.

    An explanation-only halt may follow an earlier degrade, so selecting the
    stopped-run copy only when the file is first created is insufficient.  The
    issue entries and their signatures stay byte-for-byte untouched.
    """
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    replacement = f"# Known issues\n\n{intro}\n\n---\n\n"
    pattern = re.compile(r"\A# Known issues\n\n.*?\n\n---\n\n", re.DOTALL)
    rewritten, count = pattern.subn(replacement, text, count=1)
    if count == 0 and text.startswith("# Known issues\n\n"):
        first_entry = text.find("## ", len("# Known issues\n\n"))
        if first_entry >= 0:
            rewritten = replacement + text[first_entry:]
    if rewritten != text:
        path.write_text(rewritten, encoding="utf-8")


def _normalize_known_issue_text(text: str | None) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Judge-halt reasons embed the judge's free-form rationale, which is reworded
# on every resume of the same underlying issue (fedavg + ROMAN 2026-07-17: the
# same defect produced duplicate KNOWN_ISSUES entries differing only in judge
# prose). Only the deterministic prefix — verdict class, confidence, and the
# failing location — participates in the dedup signature; the full rationale
# still lands in the entry body. Covers all three live reason shapes:
# "(class/conf) at <label>: …", "(class/conf) on persisting <stage>
# findings: …", and the bare "halt-judge decided to halt: …".
_JUDGE_REASON_SIG_RE = re.compile(
    r"^(halt-judge decided to halt"
    r"(?: \([^)]*\))?"
    r"(?: (?:at|on) [^:]*)?"
    r"):"
)


def _known_issue_signature(
    *,
    stage_id: str,
    what_failed: str,
    where: str | None,
    reason: str,
) -> str:
    judge = _JUDGE_REASON_SIG_RE.match(reason or "")
    if judge:
        reason = judge.group(1)
    return " | ".join(
        _normalize_known_issue_text(part)
        for part in (stage_id, what_failed, where or "", reason)
    )


def _known_issue_already_recorded(path: Path, signature: str) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    if f"<!-- issue_signature: {signature} -->" in text:
        return True
    # Back-compat for files written before signatures were embedded: compare
    # normalized section text so repeated identical degrades do not duplicate
    # old entries after a resume.
    normalized_sig_parts = [
        part for part in signature.split(" | ") if part
    ]
    for match in re.finditer(r"(?ms)^## .+?(?=^## |\Z)", text):
        section = _normalize_known_issue_text(match.group(0))
        if all(part in section for part in normalized_sig_parts):
            return True
    return False


def _append_known_issue(
    paths: PipelinePaths, *,
    stage_id: str, what_failed: str, what_to_do: str,
    where: str | None = None, reason: str = "",
) -> None:
    """Append a degrade entry to `<RUN_DIR>/KNOWN_ISSUES.md`. Called by
    degrade(); also safe to call directly. Writes the header on first use."""
    path = run_layout.run_path(paths.run_dir, run_layout.KNOWN_ISSUES_MD)
    signature = _known_issue_signature(
        stage_id=stage_id,
        what_failed=what_failed,
        where=where,
        reason=reason,
    )
    if _known_issue_already_recorded(path, signature):
        return
    header_needed = not path.exists()
    with path.open("a", encoding="utf-8") as f:
        if header_needed:
            f.write("# Known issues\n\n" + _KNOWN_ISSUES_INTRO + "\n\n---\n\n")
        f.write(f"## {stage_id} — {what_failed}\n\n")
        f.write(f"<!-- issue_signature: {signature} -->\n\n")
        if where:
            f.write(f"**Where.** {html.escape(str(where), quote=False)}\n\n")
        f.write(f"**What to do.** {what_to_do}\n\n")
        if reason:
            f.write(f"**Technical detail.** {reason}\n\n")
        f.write("---\n\n")


def _remove_known_issue(
    paths: PipelinePaths, *,
    stage_id: str,
    what_failed: str,
) -> None:
    """Remove an existing known-issue entry with the same stable heading."""
    path = paths.run_dir / run_layout.KNOWN_ISSUES_MD
    if not path.is_file():
        return
    heading = f"## {stage_id} — {what_failed}"
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"(?ms)^{re.escape(heading)}\n\n.*?(?=^## |\Z)"
    )
    new_text = pattern.sub("", text)
    if new_text != text:
        _write_known_issues_or_remove(path, new_text)


def _remove_known_issues_for_stage(paths: PipelinePaths, *, stage_id: str) -> None:
    """Remove stale known-issue entries for a stage that later completed cleanly."""
    path = paths.run_dir / run_layout.KNOWN_ISSUES_MD
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"(?ms)^## {re.escape(stage_id)}\s+[—-]\s+.*?(?=^## |\Z)"
    )
    new_text = pattern.sub("", text)
    if new_text != text:
        _write_known_issues_or_remove(path, new_text)


def _write_known_issues_or_remove(path: Path, text: str) -> None:
    """Persist known issues after pruning, or remove an empty issue file."""
    if re.search(r"(?m)^## ", text):
        path.write_text(text.rstrip() + "\n\n", encoding="utf-8")
    else:
        path.unlink()


def _strip_known_issues_banner(text: str) -> str:
    """Remove any existing 'Known issues' README banner (the banner line plus its
    trailing blank line). Keeps the banner from being orphaned when a prior
    degrade was later resolved and KNOWN_ISSUES.md was pruned away."""
    kept: list[str] = []
    skip_blank = False
    for line in text.splitlines(keepends=True):
        if line.startswith(">") and "KNOWN_ISSUES.md" in line:
            skip_blank = True
            continue
        if skip_blank and not line.strip():
            skip_blank = False
            continue
        skip_blank = False
        kept.append(line)
    return "".join(kept).lstrip("\n")


def finalize_run_report(state: "PipelineState") -> None:
    """End-of-run pass over the package README's 'Known issues' banner.

    If any stage degraded (KNOWN_ISSUES.md exists with entries), prepend a
    prominent banner so the researcher can't miss that the output has unresolved
    quality gates. If a prior degrade was later resolved (the file was pruned or
    removed on a clean resume), strip any stale banner so the README never points
    at a KNOWN_ISSUES.md that no longer exists.

    Trust guardrail: the banner makes best-effort delivery honest (the researcher
    sees up front that N gates couldn't be satisfied, rather than being handed
    output that looks fully verified), and a resolved run never ships a dangling
    reference. This normal end-of-run path restores delivered-package preamble
    copy; the explanation-only wrapper applies stopped-run copy afterward, so a
    successful resume cannot retain stale stopped wording. Idempotent, and a
    no-op write when nothing changes."""
    import re  # local: stdlib
    ki_path = state.paths.run_dir / run_layout.KNOWN_ISSUES_MD
    n_issues = (len(re.findall(r"(?m)^## ", ki_path.read_text(encoding="utf-8")))
                if ki_path.is_file() else 0)
    banner = ""
    if n_issues:
        _rewrite_known_issues_intro(
            ki_path,
            _KNOWN_ISSUES_INTRO,
        )
        # Links are relative to the package README's own location (method/).
        banner = (
            f"> **Known issues:** This package has {n_issues} known issue"
            f"{'' if n_issues == 1 else 's'}. Some quality gates could not be "
            f"satisfied automatically. Start with [`REPORT.md`](../REPORT.md), then "
            f"see [`KNOWN_ISSUES.md`](../{run_layout.KNOWN_ISSUES_MD}) for full details.\n\n"
        )
    readme = run_layout.run_path(state.paths.run_dir, run_layout.PACKAGE_README)
    if readme.is_file():
        original = readme.read_text(encoding="utf-8")
        updated = banner + _strip_known_issues_banner(original)
        if updated != original:
            readme.write_text(updated, encoding="utf-8")
    elif banner:
        readme.write_text(banner, encoding="utf-8")


PROBE_BATTERY_TIMEOUT_S = 900


def _notebook_digest_mismatch(paths: "PipelinePaths") -> str | None:
    """R2C-039 block 2: is the delivered notebook the one smoke verified?

    Compares notebook.ipynb against the `notebook_sha256` recorded on the
    run's LAST smoke event when that event is a pass. Returns a demotion
    message on mismatch, None when the invariant holds or no comparison
    applies: no smoke events (never smoked), a last smoke event that failed
    (the delivery is already degraded/halted on that path, including the
    environmental exit-3 degrade, which discloses rendered-but-not-verified),
    or a pass recorded before this invariant existed (no digest field)."""
    events_path = paths.pipeline_dir / "run_events.jsonl"
    if not events_path.is_file():
        return None
    last_smoke = None
    try:
        for line in events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event_type") in ("smoke_passed", "smoke_failed"):
                last_smoke = ev
    except OSError:
        return None
    if not last_smoke or last_smoke.get("event_type") != "smoke_passed":
        return None
    verified = (last_smoke.get("details") or {}).get("notebook_sha256")
    if not verified:
        return None
    nb_path = paths.run_dir / "notebook.ipynb"
    try:
        current = hashlib.sha256(nb_path.read_bytes()).hexdigest()
    except OSError:
        return ("the smoke gate recorded a verified notebook digest but "
                "notebook.ipynb is unreadable at delivery")
    if current != verified:
        return ("the delivered notebook.ipynb is not the notebook the smoke "
                "gate verified: its digest differs from the one recorded at "
                "the last smoke pass, so it was modified after its last "
                "verified execution")
    return None


def run_delivery_gating(state: "PipelineState") -> dict:
    """Slice 1.5: run the probe battery and derive the two-tier label.

    Never halts delivery — a battery that crashes or times out yields a
    `draft` label with the failure as the reason (absence of verification
    evidence is not verification). Smoke passing has no say here: the
    label comes from behavioral verdicts plus unresolved critical/important
    fidelity findings.

    Partial delivery (§3.3/§3.5): recorded stubbed elements flow into the
    derivation (they can never read `verified`), and every recorded stub's
    surfaces are verified here — a stub whose work order or stub file is
    missing or mute demotes, loudly.
    """
    from delivery_label import (  # local: stdlib-only
        apply_neutral_plan_cap,
        apply_unprobeable_core_novelty_cap,
        derive_delivery_label,
        unprobeable_core_novelty,
    )
    from partial_delivery import load_stubbed_elements, verify_stub_surfaces

    pipeline_dir = state.paths.pipeline_dir
    probe_report: dict | None = None
    battery_error: str | None = None
    try:
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "run_probes.py"),
             "--run-dir", str(state.paths.run_dir)],
            capture_output=True, text=True, timeout=PROBE_BATTERY_TIMEOUT_S,
            cwd=str(REPO_ROOT),
        )
        # Exit 1 = gating failures found: a valid, parseable outcome.
        report_path = pipeline_dir / "probe_report.json"
        if report_path.is_file():
            probe_report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            battery_error = (f"battery exit {proc.returncode}, no report: "
                             f"{(proc.stderr or proc.stdout)[-300:]}")
    except subprocess.TimeoutExpired:
        battery_error = f"battery timed out after {PROBE_BATTERY_TIMEOUT_S}s"
    except Exception as e:  # noqa: BLE001 — gating must not kill delivery
        battery_error = f"battery could not run: {e}"

    review_report: dict | None = None
    review_path = pipeline_dir / "review_report.json"
    if review_path.is_file():
        try:
            review_report = json.loads(review_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

    # Post-smoke demo-success verdict (design 2026-07-16). An absent artifact
    # remains a legacy no-op unless the effective family declares demo_skill;
    # that schema-2 requirement is resolved below and fails closed. A failed
    # verdict demotes through its own first-class reason, never through
    # adjacent-demoter luck.
    demo_verdict: dict | None = None
    demo_verdict_path = pipeline_dir / "demo_verdict.json"
    if demo_verdict_path.is_file():
        try:
            demo_verdict = json.loads(
                demo_verdict_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

    # The taxonomy node id, for `missing_probe_family` on an
    # uncertified — new territory delivery (the growth-engine demand key),
    # plus the approximated-core records (maintainer-approved 2026-07-05): a
    # core_methodology element whose replication_status is anything but
    # must_replicate ships as an approved approximation, and an
    # approximated core can never read verified. Supporting and
    # demo-scale approximations stay disclosures — the filter on role
    # happens HERE, so the label derivation stays pure.
    paradigm: str | None = None
    core_approximations: list[dict] = []
    neutral_build_plan = False
    graph_execution_plan: dict | None = None
    # The parsed spec, kept for binding and the unprobeable core-novelty cap.
    # Missing or malformed input is recorded separately and demotes below: an
    # absent contract is not evidence that archived compatibility applies.
    label_spec: dict | None = None
    label_spec_error: str | None = (
        "method_spec.json is missing, so contribution evidence cannot be "
        "bound to this package's declared methodology obligations"
    )
    spec_path = pipeline_dir / "method_spec.json"
    if spec_path.is_file():
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            label_spec = spec if isinstance(spec, dict) else None
            if label_spec is None:
                label_spec_error = (
                    "method_spec.json is not a JSON object, so contribution "
                    "evidence cannot be bound to methodology obligations"
                )
            else:
                label_spec_error = None
            paradigm = (spec.get("comparison", {})
                        .get("classification", {}).get("id")) or None
            elements = ((spec.get("methodology_replication_contract") or {})
                        .get("elements") or [])
            core_approximations = [
                {"element_id": el.get("element_id"),
                 "replication_status": el.get("replication_status")}
                for el in elements
                if isinstance(el, dict)
                and el.get("role") == "core_methodology"
                and el.get("replication_status") != "must_replicate"
            ]
        except (OSError, json.JSONDecodeError, AttributeError):
            paradigm = None
            core_approximations = []
            label_spec = None
            label_spec_error = (
                "method_spec.json is unreadable or malformed, so contribution "
                "evidence cannot be bound to methodology obligations"
            )
        else:
            # Build-context provenance for the neutral-plan label cap
            # (R2C-032): recompute the plan the run was actually built
            # against (the plan is never persisted; every consumer derives
            # it the same way, e.g. the 2.b halt summarizer above). A
            # resolution failure means no cap — the cap can only narrow a
            # label, so failing open here is failing toward today's
            # behavior, and committed families always carry their own
            # plan_key.
            try:
                from scripts.build_plan import (  # noqa: PLC0415
                    PROVISIONAL_PLAN_KEY,
                    load_build_plan,
                )
                from scripts.taxonomy import run_overlay_dir  # noqa: PLC0415

                plan = load_build_plan(
                    spec, REPO_ROOT,
                    provisional_packs_dir=run_overlay_dir(
                        state.paths.run_dir),
                )
                neutral_build_plan = bool(plan) and (
                    plan.get("plan_key") == PROVISIONAL_PLAN_KEY)
            except Exception:  # noqa: BLE001 — gating must not kill delivery
                neutral_build_plan = False

    methodology = (
        label_spec.get("methodology_replication_contract")
        if isinstance(label_spec, dict) else None
    )
    graph_contract = (
        methodology.get("homogeneous_graph_mechanism")
        if isinstance(methodology, dict) else None
    )
    if isinstance(graph_contract, dict):
        declared_graph_refs = {
            str(row.get("probe_ref") or "")
            for row in (probe_report or {}).get("verdicts") or []
            if isinstance(row, dict)
            and str(row.get("probe_ref") or "").startswith(
                "graph_mechanism."
            )
        }
        declared_graph_refs.add("graph_mechanism.alignment_prerequisite")
        raw_graph_refs = graph_contract.get("probe_refs")
        if isinstance(raw_graph_refs, dict):
            declared_graph_refs.update(
                value for value in raw_graph_refs.values()
                if isinstance(value, str) and value
            )
        try:
            from build_probe_harness import (  # noqa: PLC0415
                _frozen_graph_mechanism_execution_plan,
            )

            graph_execution_plan = _frozen_graph_mechanism_execution_plan(
                state.paths.run_dir,
                {probe_ref: {} for probe_ref in declared_graph_refs},
            )
        except Exception:  # noqa: BLE001 — missing authority fails closed below
            graph_execution_plan = None

    structured_demo_required = False
    try:
        from demo_verdict import structured_demo_requirement  # noqa: PLC0415

        _, structured_demo_required = structured_demo_requirement(
            state.paths.run_dir
        )
    except Exception:  # noqa: BLE001 - malformed/missing artifacts still gate below
        structured_demo_required = False

    stubs, stub_error = load_stubbed_elements(pipeline_dir)
    delivery = derive_delivery_label(probe_report, review_report,
                                     paradigm=paradigm,
                                     stubbed_elements=stubs,
                                     core_approximations=core_approximations,
                                     demo_verdict=demo_verdict,
                                     structured_demo_required=(
                                         structured_demo_required
                                     ),
                                     neutral_build_plan=neutral_build_plan,
                                     spec=label_spec,
                                     graph_execution_plan=graph_execution_plan)
    unprobed_core = unprobeable_core_novelty(probe_report, label_spec)
    if label_spec_error:
        delivery["reasons"].insert(0, {
            "source": "method_spec",
            "id": "artifact_missing_or_invalid",
            "message": label_spec_error,
        })
        delivery["label"] = "draft"
    if battery_error:
        delivery["reasons"].insert(0, {
            "source": "battery", "id": "error", "message": battery_error,
        })
        delivery["label"] = "draft"
    if stub_error:
        # An unreadable partial record could hide a partial state — the
        # fail-closed direction is a demoted label, never a clean one.
        delivery["reasons"].insert(0, {
            "source": "partial_delivery", "id": "artifact_error",
            "message": stub_error,
        })
        delivery["label"] = "draft"
    for problem in verify_stub_surfaces(state.paths.run_dir, stubs):
        # §3.5 criterion 3, enforced at delivery: a stub that does not
        # self-identify is a defect in the package, not a formatting nit.
        delivery["reasons"].append({
            "source": "partial_delivery", "id": "stub_surface",
            "message": problem,
        })
        delivery["label"] = "draft"
    digest_mismatch = _notebook_digest_mismatch(state.paths)
    if digest_mismatch:
        # R2C-039 block 2: a notebook that differs from the smoke-verified
        # one cannot present itself as verified. Same fail-closed direction
        # as the battery/stub errors above.
        delivery["reasons"].append({
            "source": "smoke", "id": "notebook_digest_mismatch",
            "message": digest_mismatch,
        })
        delivery["label"] = "draft"
    if unprobed_core:
        # Same ordering argument as the neutral-plan cap below: the post-hoc
        # blocks above set draft, which sits ABOVE uncertified on the label
        # scale, so a battery error or stub defect would otherwise RAISE a
        # capped label. Idempotent, so the earlier application inside the
        # derivation is not duplicated (R2C-047).
        apply_unprobeable_core_novelty_cap(delivery, unprobed_core)
    if neutral_build_plan:
        # The cap outranks the post-hoc draft demotions above: on the label
        # scale draft sits ABOVE uncertified, so letting a battery error or
        # stub defect set draft here would RAISE a capped label. The
        # demoting reasons those blocks recorded stay in `reasons`,
        # fully disclosed (idempotent; see apply_neutral_plan_cap).
        apply_neutral_plan_cap(delivery, paradigm)
    _append_run_event(
        state.paths,
        "delivery_label_derived",
        stage_id="stage_5",
        status=delivery["label"],
        summary=f"{len(delivery['reasons'])} demoting reason(s), "
                f"{len(delivery['disclosures'])} disclosure(s)"
                + (f", PARTIAL: {len(stubs)} stubbed element(s)"
                   if stubs else ""),
        details={"label": delivery["label"],
                 "partial": bool(delivery.get("partial")),
                 "stubbed_element_ids": [s["element_id"] for s in stubs],
                 "reason_ids": [r["id"] for r in delivery["reasons"]]},
    )
    log("delivery", delivery["label"],
        ("PARTIAL — " if delivery.get("partial") else "")
        + ("; ".join(f"{r['id']}: {r['message'][:80]}"
                     for r in delivery["reasons"]) or "all gates clean"))
    return delivery


_SMOKE_BANNER_MARKER = "<!-- r2c:smoke-failure-banner -->"


def _smoke_failure_facts(state: "PipelineState") -> dict | None:
    """What is known about a failed smoke run, or None when it passed.

    Read from the stage results and the diagnosis the run already recorded.
    `diagnosis_is_fallback` matters on its own: the driver's fallback states
    in its own text that the root cause is unanalyzed, and that honesty lived
    only in `.pipeline/` where no researcher looks."""
    smoke = next((r for r in state.stage_results if r.stage_id == "stage_3c"
                  and r.status in ("degraded", "failed")), None)
    if smoke is None:
        return None
    facts: dict = {"status": smoke.status, "cell": None,
                   "diagnosis_is_fallback": False}
    # The failing cell is recorded in the known-issues entry degrade() just
    # wrote ("notebook.ipynb — cell 29"), which is the only surface that
    # carries it (the StageResult notes are a summary line).
    issues = run_layout.run_path(state.paths.run_dir, run_layout.KNOWN_ISSUES_MD)
    if issues.is_file():
        match = re.search(r"(?m)^\*\*Where\.\*\* notebook\.ipynb\s+[—-]\s+"
                          r"cell (\d+)", issues.read_text(encoding="utf-8"))
        if match:
            facts["cell"] = int(match.group(1))
    diagnosis_path = state.paths.pipeline_dir / "smoke_diagnosis.json"
    if diagnosis_path.is_file():
        try:
            diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            diagnosis = {}
        facts["diagnosis_is_fallback"] = "DRIVER-AUTHORED FALLBACK" in str(
            diagnosis.get("root_cause") or "")
        facts["target_file"] = diagnosis.get("target_file")
    return facts


def finalize_degraded_surfaces(state: "PipelineState") -> None:
    """Make every researcher-facing surface tell the same story (R2C-067).

    Degradation disclosure was written once, into the report and known
    issues, while the surfaces generated EARLIER in the run kept their
    optimistic wording. So the 2026-08-05 delivery disclosed its crash in
    REPORT.md's trouble table while the notebook promised a runnable
    tutorial and the package README said to click Run All to verify. The
    surfaces a researcher opens first were the ones that lied
    (`surface_story_drift`).

    Three deterministic edits, all conditioned on state the run already
    recorded, all idempotent so a resume cannot stack them."""
    facts = _smoke_failure_facts(state)
    if facts is None:
        return
    _stamp_notebook_smoke_banner(state, facts)
    _condition_readme_run_invitation(state)
    _label_fallback_diagnosis_in_known_issues(state, facts)


def _stamp_notebook_smoke_banner(state: "PipelineState", facts: dict) -> None:
    """Put the crash in the notebook's own first cell.

    A researcher opens the notebook, not the report. The 2026-08-05
    notebook's header promised "a runnable tutorial that trains GraphDeepAR"
    and its otherwise thorough what-this-does-NOT-do list omitted the one
    thing that mattered."""
    path = run_layout.run_path(state.paths.run_dir, run_layout.NOTEBOOK_IPYNB)
    if not path.is_file():
        return
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        return
    where = (f"cell {facts['cell']}" if facts.get("cell")
             else "one of its cells")
    text = (
        f"{_SMOKE_BANNER_MARKER}\n"
        f"# This notebook does NOT run end to end\n\n"
        f"R2C executed it before delivery and **{where} raised an "
        f"exception**, so every cell after that point is untested and the "
        f"numbers below it were never produced. Do not read this as a "
        f"working tutorial.\n\n"
        f"Read [`details/KNOWN_ISSUES.md`](details/KNOWN_ISSUES.md) for the "
        f"failure and [`REPORT.md`](REPORT.md) for what the delivery does "
        f"and does not stand behind.\n"
    )
    banner = {"cell_type": "markdown", "metadata": {},
              "source": text.splitlines(keepends=True)}
    kept = [c for c in cells
            if _SMOKE_BANNER_MARKER not in "".join(c.get("source") or [])]
    notebook["cells"] = [banner] + kept
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


def _condition_readme_run_invitation(state: "PipelineState") -> None:
    """Never invite Run All on a notebook known not to run.

    Every paradigm's README template carries the invitation, written at
    stage 2.a hours before the smoke gate has an opinion. This is the seam
    where the run's actual outcome reaches it."""
    readme = run_layout.run_path(state.paths.run_dir, run_layout.PACKAGE_README)
    if not readme.is_file():
        return
    text = readme.read_text(encoding="utf-8")
    replacement = (
        "**Do not click Run All yet.** R2C executed this notebook before "
        "delivery and it raised an exception partway through; see "
        "`../details/KNOWN_ISSUES.md`. Read the cells for the method's "
        "structure, and expect to fix the failure before the run completes."
    )
    new_text = re.sub(
        r"(?m)^Click \*\*Run All\*\*\..*(?:\n(?!\n).*)*", replacement, text)
    if new_text != text:
        readme.write_text(new_text, encoding="utf-8")


def _label_fallback_diagnosis_in_known_issues(
    state: "PipelineState", facts: dict,
) -> None:
    """Say plainly when the shipped diagnosis is routing, not analysis.

    The driver's fallback labels itself honestly inside
    `.pipeline/smoke_diagnosis.json` ("the actual root cause is
    unanalyzed"), and the researcher-facing entry presented its routing as
    a diagnosis. Working from that entry alone, the 2026-08-05 reviewer
    fixed the wrong file first."""
    if not facts.get("diagnosis_is_fallback"):
        return
    path = run_layout.run_path(state.paths.run_dir, run_layout.KNOWN_ISSUES_MD)
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    note = (
        "**Diagnosis provenance.** Symptom and routing only — the root "
        "cause is UNANALYZED. Both diagnostician attempts ended without "
        "writing an analysis, so the fix pointer"
        + (f" (`{facts['target_file']}`)" if facts.get("target_file") else "")
        + " comes from mechanical traceback routing rather than from a "
        "walked data-flow trace. Verify the cause yourself before "
        "trusting the pointer: the failure may originate upstream of the "
        "file named here.\n"
    )
    if note.splitlines()[0] in text:
        return
    marker = "**What to do.**"
    if marker not in text:
        return
    path.write_text(text.replace(marker, note + "\n" + marker, 1),
                    encoding="utf-8")


def finalize_delivery_banner(state: "PipelineState", delivery: dict) -> None:
    """Put the label where the researcher starts reading: atop the README.

    Idempotent — an existing delivery banner line is replaced, not stacked.
    A partial delivery leads with the word PARTIAL in the banner's first
    line (§3.5 criterion 1) and the completeness numbers, whatever the
    label underneath.

    Links inside the banner are relative to the package README's own
    location (method/), one level below the run root.
    """
    readme = run_layout.run_path(state.paths.run_dir, run_layout.PACKAGE_README)
    if delivery.get("partial"):
        from partial_delivery import (completeness_counts,
                                      completeness_statement,
                                      core_gap_clause, stub_display_lines)
        stubs = delivery.get("stubbed_elements") or []
        counts = completeness_counts(state.paths.pipeline_dir, stubs)
        label_gloss = {
            "draft": "draft",
            "uncertified_new_territory": "uncertified, new territory",
        }.get(delivery["label"], delivery["label"])
        core = core_gap_clause(stubs)
        stub_list = "; ".join(stub_display_lines(stubs))
        line = (f"> **PARTIAL delivery — {label_gloss}**: "
                f"{completeness_statement(counts)}. "
                + (f"{core} " if core else "")
                + f"Stubbed: {stub_list}. Each stub raises on use and "
                f"carries a researcher-consumable work order under "
                f"`work_orders/`. Start with [`REPORT.md`](../REPORT.md); the "
                f"structured record is `delivery.stubbed_elements` in "
                f"`../details/final_manifest.json`.\n\n")
    elif delivery["label"] == "verified":
        line = ("> **Delivery: verified** - the probe battery and the "
                "fidelity review found no gating issues. Unchecked surface, "
                "if any, is summarized in [`REPORT.md`](../REPORT.md) and listed "
                "under `delivery.disclosures` in `../details/final_manifest.json`.\n\n")
    elif delivery["label"] == "explanation_only":
        method_md = state.paths.run_dir / "METHOD.md"
        if method_md.is_file():
            from generate_method_md import (  # noqa: PLC0415
                method_md_has_substantive_explanation,
            )
            method_text = method_md.read_text(encoding="utf-8")
            if method_md_has_substantive_explanation(method_text):
                start_clause = (
                    "Start with [`REPORT.md`](../REPORT.md) and "
                    "[`METHOD.md`](../METHOD.md). "
                )
            elif not method_text.strip():
                start_clause = (
                    "Start with [`REPORT.md`](../REPORT.md). "
                    "[`METHOD.md`](../METHOD.md) exists but is empty; no "
                    "substantive method explanation was produced. "
                )
            else:
                start_clause = (
                    "Start with [`REPORT.md`](../REPORT.md). "
                    "[`METHOD.md`](../METHOD.md) contains a decomposition "
                    "source index, not a substantive method explanation. "
                )
        else:
            start_clause = (
                "Start with [`REPORT.md`](../REPORT.md). The method explanation "
                "was not produced before the halt. "
            )
        line = ("> **Delivery: explanation only** - R2C could not produce a "
                f"reliable code package for this run. {start_clause}The "
                "gap rationale is recorded in [`KNOWN_ISSUES.md`](../details/KNOWN_ISSUES.md) "
                "and in `delivery.reasons` in `../details/final_manifest.json`.\n\n")
    elif delivery["label"] == "uncertified_new_territory":
        from delivery_label import (contribution_gap_clause,  # noqa: PLC0415
                                    universal_floor_clause)
        line = (f"> **Delivery: uncertified — new territory** - "
                f"{universal_floor_clause(delivery)} "
                f"{contribution_gap_clause(delivery)} Nothing failed, and "
                "nobody has verified the core mechanism either. What was "
                "and wasn't checked is summarized in "
                "[`REPORT.md`](../REPORT.md).\n\n")
    else:
        n = len(delivery["reasons"])
        line = (f"> **Delivery: draft** - {n} finding"
                f"{'' if n == 1 else 's'} need{'s' if n == 1 else ''} your "
                f"attention before relying on this output. Start with "
                f"[`REPORT.md`](../REPORT.md), then see `delivery.reasons` in "
                f"`../details/final_manifest.json`.\n\n")
    existing = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    if "**Delivery: " in existing or "**PARTIAL delivery" in existing:
        kept = [l for l in existing.splitlines(keepends=True)
                if "**Delivery: " not in l and "**PARTIAL delivery" not in l]
        existing = "".join(kept).lstrip("\n")
    readme.write_text(line + existing, encoding="utf-8")


def _explanation_only_delivery(result: StageResult) -> dict:
    """Delivery verdict for the Section 2.4 fallback package.

    This is distinct from `draft`: draft means code exists but needs review;
    explanation_only means code was not produced or cannot be relied on, while
    METHOD.md / REPORT.md still carry the researcher-facing understanding.
    """
    return {
        "schema_version": "1.0.0",
        "label": "explanation_only",
        "reasons": [{
            "source": "halt",
            "id": result.stage_id,
            "message": result.notes or "pipeline halted before reliable code delivery",
        }],
        "disclosures": [],
        "probe_counts": {},
    }


def _ensure_method_md_on_halt(
    state: "PipelineState", result: StageResult,
) -> None:
    """§6b (failure-path spec): halts that fire before Stage 1.x used to
    skip METHOD.md entirely — breaking the two-tier promise exactly on the
    papers where the explanation is the ONLY deliverable (2026-07-02 sweep:
    ICRA21_HICA's honest feasibility block and SRL's gap-path halt both
    shipped no METHOD.md although the decomposition they need existed).

    When METHOD.md is missing but its required input (paper_map.json)
    exists, run the best-effort stage 1.x generation now, before the
    explanation-only bundle is packaged. run_stage_1x never halts by
    design: dispatch failures degrade to explicit PENDING markers. Wrapped
    so no failure here can mask the halt or abort the packaging."""
    method_md = state.paths.run_dir / "METHOD.md"
    if method_md.is_file():
        return
    if not state.paths.paper_map.is_file():
        log(result.stage_id, "explanation_on_halt_skipped",
            "METHOD.md missing and paper_map.json absent — the halt "
            "precedes the explanation tier's inputs; packaging without "
            "METHOD.md")
        return
    log(result.stage_id, "explanation_on_halt",
        "halt fired before stage_1x completed — generating METHOD.md now "
        "so the explanation tier still ships with the explanation-only "
        "package")
    try:
        stage_1x_result = run_stage_1x(state)
        log(result.stage_id, "explanation_on_halt_done",
            f"stage_1x on the halt path: {stage_1x_result.status} — "
            f"{(stage_1x_result.notes or '')[:300]}")
    except Exception as e:  # noqa: BLE001 — packaging must never mask a halt
        log(result.stage_id, "explanation_on_halt_failed",
            f"{type(e).__name__}: {e}")


def finalize_explanation_only_package(
    state: "PipelineState",
    result: StageResult,
) -> None:
    """Best-effort Section 2.4 fallback package for halted runs.

    A halt remains a halt: the driver exits non-zero and the manifest stays
    blocked when required code artifacts are absent. This helper only ensures the
    researcher still gets the understanding layer: METHOD.md if available,
    REPORT.md as the front door, KNOWN_ISSUES.md as the gap rationale, and a
    machine-readable manifest naming the explanation-only delivery mode.
    """
    delivery = _explanation_only_delivery(result)
    halt_path = (state.paths.pipeline_dir / f"{result.stage_id}.halt")
    where = (
        _safe_path_for_manifest(state.paths, halt_path)
        if halt_path.is_file() else result.stage_id
    )
    what_failed = _EXPLANATION_ONLY_WHAT_FAILED
    stage_results = [*state.stage_results, result]
    halt_artifact = (
        result.halt_artifact if isinstance(result.halt_artifact, dict) else {}
    )
    terminal_action = halt_catalog.terminal_gap_action(
        halt_artifact.get("halt_class"), halt_artifact.get("context")
    )
    if terminal_action:
        issue_action = (
            "Use REPORT.md as the front door; it identifies whether METHOD.md "
            "contains a substantive explanation, only a decomposition index, "
            "or no usable content. "
            f"Scope-decision guidance: {terminal_action} Do "
            "not resume the unchanged run."
        )
    else:
        issue_action = (
            "Use REPORT.md as the front door; it identifies whether METHOD.md "
            "contains a substantive explanation, only a decomposition index, "
            "or no usable content. "
            "Treat any code artifacts as absent or unreliable until the halt "
            "is resolved and the pipeline is resumed."
        )
    _ensure_method_md_on_halt(state, result)
    try:
        _remove_known_issue(
            state.paths,
            stage_id=result.stage_id,
            what_failed=what_failed,
        )
        _append_known_issue(
            state.paths,
            stage_id=result.stage_id,
            what_failed=what_failed,
            where=where,
            what_to_do=issue_action,
            reason=result.notes or "",
        )
        finalize_assumptions_md(state)
        finalize_method_md(state)
        finalize_run_report(state)
        _rewrite_known_issues_intro(
            run_layout.run_path(state.paths.run_dir, run_layout.KNOWN_ISSUES_MD),
            _KNOWN_ISSUES_STOPPED_INTRO,
        )
        finalize_delivery_banner(state, delivery)
        finalize_claims_report(state, delivery, stage_results=stage_results)
        finalize_final_manifest(
            state,
            delivery,
            stage_results=stage_results,
        )
        log(result.stage_id, "explanation_only_package",
            "wrote REPORT.md, KNOWN_ISSUES.md, README banner, and final_manifest.json")
    except Exception as e:  # noqa: BLE001 — fallback packaging must never mask halt
        log(result.stage_id, "explanation_only_package_failed",
            f"{type(e).__name__}: {e}")


def finalize_claims_report(
    state: "PipelineState",
    delivery: dict | None = None,
    stage_results: list[StageResult] | None = None,
) -> None:
    """Render the full REPORT.md front door around the CT-3 claims section.

    Best-effort at this call site: the manifest marks REPORT.md required, so a
    render failure stays visible as a blocked manifest instead of disappearing.
    """
    try:
        from render_run_report import write_run_report  # noqa: PLC0415
        if write_run_report(
            state.paths.run_dir,
            delivery=delivery,
            stage_results=stage_results if stage_results is not None
            else state.stage_results,
        ):
            log("stage_5", "run_report",
                f"wrote {state.paths.run_dir / 'REPORT.md'}")
    except Exception as e:  # noqa: BLE001 — rendering must never break delivery
        log("stage_5", "run_report_skipped", f"{type(e).__name__}: {e}")


def finalize_method_md(state: "PipelineState") -> None:
    """Refresh METHOD.md at delivery so its parameter-provenance table
    reflects the final params.json (stage 1x generates BEFORE stage 2x
    derives parameters). Deterministic regeneration from the same sidecar;
    a refresh failure leaves the stage-1x document in place. No-op when
    stage 1x never produced a METHOD.md."""
    method_md = state.paths.run_dir / "METHOD.md"
    if not method_md.is_file():
        return
    sidecar = state.paths.pipeline_dir / "method_explanations.json"
    args = ["scripts/generate_method_md.py",
            "--run-dir", str(state.paths.run_dir)]
    if sidecar.is_file():
        args += ["--explanations", str(sidecar)]
    proc = run_script("stage_5", args, timeout=60)
    if proc.returncode != 0:
        log("stage_5", "method_md_refresh_failed",
            (proc.stderr or proc.stdout)[-300:])


def finalize_final_manifest(
    state: "PipelineState", delivery: dict | None = None,
    stage_results: list[StageResult] | None = None,
) -> tuple[bool, str]:
    """Write `<RUN_DIR>/final_manifest.json` after delivery finalization.

    The manifest is fail-closed for delivery-critical artifacts. A degraded
    manifest is still deliverable (it records known issues or skipped checks);
    a blocked manifest means required delivery evidence is missing or halted.
    Returns (ok_for_exit_zero, manifest_status_or_error).
    """
    manifest_path = state.paths.run_dir / run_layout.FINAL_MANIFEST_JSON
    try:
        manifest = write_final_manifest(
            state.paths,
            stage_results=stage_results if stage_results is not None
            else state.stage_results,
            delivery=delivery,
        )
    except Exception as e:
        _append_validation_event(
            state.paths,
            stage_id="stage_5",
            validator="validate_final_manifest.py",
            ok=False,
            stderr_tail=str(e),
            artifacts=[str(manifest_path)],
        )
        return False, f"generation failed: {e}"

    ok = manifest.run_status != "blocked"
    _append_validation_event(
        state.paths,
        stage_id="stage_5",
        validator="validate_final_manifest.py",
        ok=ok,
        stderr_tail="; ".join(manifest.diagnostics[-5:]),
        artifacts=[str(manifest_path)],
    )
    log(
        "final_manifest",
        manifest.run_status,
        f"wrote {manifest_path} with {len(manifest.artifacts)} artifact rows",
    )
    # Defer the nested delivery baseline until main() has recorded terminal
    # state and physically released the run lock. This flag is invocation-local:
    # a stale manifest inherited by a resume cannot make an early failure look
    # like a newly finalized delivery.
    state.delivery_baseline_pending = True
    return ok, manifest.run_status


def _delivered_manifest_integrity_violations(run_dir: Path) -> list[str]:
    """Re-verify final_manifest.json's per-artifact hashes against disk.

    Returns one human-readable violation per artifact whose recorded sha256
    no longer matches the file (or whose file vanished after the manifest
    recorded it present). Mutable kinds carry sha256=None in the manifest
    and are skipped. An absent or unreadable manifest returns [] — the
    baseline path already tolerates non-delivered terminal states."""
    manifest_path = run_dir / run_layout.FINAL_MANIFEST_JSON
    if not manifest_path.is_file():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    violations: list[str] = []
    for row in manifest.get("artifacts") or []:
        recorded = row.get("sha256")
        rel = row.get("path")
        if not recorded or not rel or row.get("status") != "present":
            continue
        path = run_dir / rel
        if not path.is_file():
            violations.append(
                f"{rel}: recorded present in final_manifest.json but "
                f"missing from disk at baseline time")
        elif sha256_file(path) != recorded:
            violations.append(
                f"{rel}: content changed after final_manifest.json "
                f"recorded its hash (manifest {recorded[:12]}..., disk "
                f"{sha256_file(path)[:12]}...)")
    return violations


def _finalize_delivered_baseline(state: "PipelineState") -> None:
    """Best-effort terminal provenance snapshot for a finalized delivery.

    Before committing the as-delivered baseline, re-verify the manifest's
    per-artifact hashes against disk (B-13 change 2): the baseline is the
    durable "as delivered" record, so a late overwrite from ANY source —
    hook, concurrent process, stray agent write — must block the commit
    and surface, never get baked in as the delivered state."""
    if not state.delivery_baseline_pending:
        return
    state.delivery_baseline_pending = False
    try:
        violations = _delivered_manifest_integrity_violations(
            state.paths.run_dir)
    except Exception as e:  # noqa: BLE001 - verification is best-effort
        violations = []
        log("final_manifest", "baseline_integrity_check_failed",
            f"{type(e).__name__}: {e}")
    if violations:
        log("final_manifest", "delivered_baseline_blocked",
            f"{len(violations)} artifact(s) changed after the manifest "
            f"recorded them; the as-delivered baseline commit is blocked")
        _append_run_event(
            state.paths,
            "delivered_baseline_blocked",
            status="integrity_violation",
            summary=(
                "final_manifest.json hashes no longer match disk; the "
                "as-delivered git baseline was NOT committed"),
            details={"violations": violations},
        )
        try:
            known_issues = state.paths.run_dir / run_layout.KNOWN_ISSUES_MD
            known_issues.parent.mkdir(parents=True, exist_ok=True)
            with known_issues.open("a", encoding="utf-8") as fh:
                fh.write(
                    "\n## Delivered-baseline integrity violation\n\n"
                    "The following delivered artifacts changed AFTER "
                    "final_manifest.json recorded their hashes, so the "
                    "as-delivered git baseline was not committed. The "
                    "manifest, not the current file content, is the "
                    "delivery-time record.\n\n"
                    + "".join(f"- {v}\n" for v in violations))
        except OSError as e:
            log("final_manifest", "delivered_baseline",
                f"KNOWN_ISSUES append failed: {e}")
        return
    try:
        from scripts.finalize_run_git import ensure_delivered_baseline  # noqa: PLC0415

        git_ok, git_msg = ensure_delivered_baseline(state.paths.run_dir)
        log("final_manifest", "delivered_baseline",
            git_msg if git_ok else f"baseline skipped: {git_msg}")
    except Exception as e:
        log("final_manifest", "delivered_baseline",
            f"baseline skipped: {type(e).__name__}: {e}")


def _existing_rescale_assumption_id(entry: dict) -> str | None:
    text = "\n".join(
        str(entry.get(key) or "")
        for key in ("reasoning", "note")
    )
    match = re.search(r"AUTO-RESOLVED\s+\((A\d{3})\)", text)
    return match.group(1) if match else None


def _infer_rescale_paper_value(entry: dict, resolution: dict, *, current_value, new_value):
    if entry.get("paper_value") is not None:
        return entry["paper_value"]
    note = str(entry.get("note") or "")
    match = re.search(
        r"rescaled\s+from\s+([-+]?\d+(?:\.\d+)?)\s+to\s+[-+]?\d+(?:\.\d+)?",
        note,
        flags=re.IGNORECASE,
    )
    if match:
        try:
            parsed = float(match.group(1))
            return int(parsed) if parsed.is_integer() else parsed
        except ValueError:
            pass
    factor_text = str(resolution.get("factor_derivation") or "")
    if _param_values_equivalent(current_value, new_value) and factor_text:
        match = re.search(r"[-+]?\d+(?:\.\d+)?", factor_text)
        if match:
            try:
                parsed = float(match.group(0))
                if not _param_values_equivalent(parsed, new_value):
                    return int(parsed) if parsed.is_integer() else parsed
            except ValueError:
                pass
    return current_value


def _apply_rescale_param_resolution(
    state: "PipelineState", finding: dict
) -> tuple[bool, str]:
    """Apply kind=rescale_param: edit params.json and log to assumptions.md.
    Returns (ok, assumption_id_or_error)."""
    resolution = finding.get("proposed_resolution") or {}
    param_name = resolution.get("param")
    new_value = resolution.get("new_value")
    if not param_name or new_value is None:
        return False, "rescale_param resolution missing param/new_value"

    params_path = state.paths.pipeline_dir / "params.json"
    if not params_path.exists():
        return False, f"params.json missing at {params_path}"

    data = json.loads(params_path.read_text(encoding="utf-8"))
    params = data.get("params") or {}
    if param_name not in params:
        return False, f"param {param_name!r} not in params.json"

    entry = params[param_name]
    old_value = entry.get("value")
    value_already_rescaled = _param_values_equivalent(old_value, new_value)
    if value_already_rescaled and entry.get("source") == "system_default" and entry.get("paper_value") is not None:
        return True, f"already_resolved:{param_name}"

    pre_edit_snapshot = json.dumps(data, indent=2)
    existing_aid = _existing_rescale_assumption_id(entry)
    aid = existing_aid if value_already_rescaled and existing_aid else _next_assumption_id(state)
    paper_value = _infer_rescale_paper_value(
        entry, resolution, current_value=old_value, new_value=new_value
    )

    aid = _next_assumption_id(state)
    factor_text = resolution.get("factor_derivation", "")
    expert_reasoning = resolution.get("expert_reasoning", "")
    previous_context = []
    if entry.get("reasoning"):
        previous_context.append(f"Previous reasoning: {entry['reasoning']}")
    if entry.get("note"):
        previous_context.append(f"Previous note: {entry['note']}")
    reasoning_parts = [
        f"AUTO-RESOLVED ({aid}): the paper value is {paper_value}, but this run uses {new_value}.",
    ]
    if factor_text:
        reasoning_parts.append(f"Derivation: {factor_text}.")
    if expert_reasoning:
        reasoning_parts.append(expert_reasoning)
    reasoning_parts.append(f"See assumptions.md entry {aid}.")
    reasoning_parts.extend(previous_context)

    entry["value"] = new_value
    entry["source"] = "system_default"
    entry["paper_value"] = paper_value
    entry["reasoning"] = " ".join(str(part).strip() for part in reasoning_parts if str(part).strip())
    # `render_notebook.py` prefers `note` over `reasoning` in the parameter table.
    # After a rescale, `reasoning` is the authoritative researcher-facing text.
    entry.pop("note", None)

    data["params"] = params
    params_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    sys.path.insert(0, str(REPO_ROOT))
    try:
        from schemas.params import Params  # noqa: PLC0415
        Params.model_validate(data)
    except Exception as e:
        params_path.write_text(pre_edit_snapshot + "\n", encoding="utf-8")
        return False, (
            f"resolution produced schema-invalid params.json (reverted): {e}"
        )

    if value_already_rescaled and existing_aid:
        return True, f"already_resolved:{param_name}"

    _append_assumption(
        state, aid=aid,
        title=f"{param_name} rescaled from {paper_value} to {new_value}",
        detected=finding.get("description", ""),
        action=(
            f"Set params.json `params.{param_name}.source` to 'system_default', "
            f"preserved `paper_value` as {paper_value}, and set runtime `value` "
            f"to {new_value}. Derivation: {factor_text}."
        ),
        reasoning=expert_reasoning,
        alternative=resolution.get("alternative"),
        override=(
            f"Edit `<run_dir>/.pipeline/params.json` to set "
            f"`params.{param_name}.value` to your preferred value (or change the "
            f"data preprocessing in `method/data.py` to match the paper's scale), "
            f"then re-run."
        ),
    )
    return True, aid


def _param_values_equivalent(a, b) -> bool:
    if (
        isinstance(a, (int, float)) and not isinstance(a, bool)
        and isinstance(b, (int, float)) and not isinstance(b, bool)
    ):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def _format_resolution_field_value(value: object, limit: int = 60) -> str:
    """Render a proposed-resolution field value for assumptions.md.

    `relabel_param_source.add_fields` can include structured params metadata
    such as numeric `paper_value`; do not assume every value is a string.
    """
    if isinstance(value, str):
        return f"{value[:limit]!r}{'...' if len(value) > limit else ''}"
    try:
        rendered = json.dumps(value, sort_keys=True)
    except TypeError:
        rendered = repr(value)
    return f"{rendered[:limit]}{'...' if len(rendered) > limit else ''}"


def _apply_relabel_param_source_resolution(
    state: "PipelineState", finding: dict
) -> tuple[bool, str]:
    """Apply kind=relabel_param_source: edit a param's provenance fields in
    params.json and log to assumptions.md. Returns (ok, assumption_id_or_error).

    Used by the stage_2x_params auto-resolve path: when the stage-reviewer
    detects a `paper_source_values_match_paper` violation, this is the
    deterministic fix. Re-validates params.json against the Params model
    after applying — if the resolution produced a schema-invalid file
    (e.g., new_source=system_inferred without a `reasoning` field), the
    edit is reverted and we return failure so the caller halts."""
    resolution = finding.get("proposed_resolution") or {}
    param_name = resolution.get("param")
    new_source = resolution.get("new_source")
    if not param_name or not new_source:
        return False, "relabel_param_source resolution missing param/new_source"

    params_path = state.paths.pipeline_dir / "params.json"
    if not params_path.exists():
        return False, f"params.json missing at {params_path}"

    data = json.loads(params_path.read_text(encoding="utf-8"))
    params = data.get("params") or {}
    if param_name not in params:
        return False, f"param {param_name!r} not in params.json"

    pre_edit_snapshot = json.dumps(data, indent=2)
    pre_edit_entry_keys = sorted(params[param_name].keys())
    old_source = params[param_name].get("source")

    # The schemas/ package lives at repo root; the driver puts scripts/ on
    # sys.path at startup, so add REPO_ROOT here (same pattern as the
    # post-edit validation below, hoisted so the field filter can see the
    # model).
    sys.path.insert(0, str(REPO_ROOT))
    from schemas.params import ParamEntry  # noqa: PLC0415

    entry = params[param_name]
    entry["source"] = new_source
    for k in resolution.get("remove_fields") or []:
        entry.pop(k, None)
    # Almost-correct-resolution remap #2 (pdwa 2026-07-07 resume): the
    # reviewer's substantively perfect relabel carried a helpful extra key
    # (`component_sources`) and ParamEntry's extra="forbid" reverted the
    # WHOLE edit, halting the run on a correct resolution. Unknown
    # add_fields keys never enter the entry; their content is preserved
    # verbatim in the assumptions.md record below, so nothing the reviewer
    # said is lost. The post-edit validation still guards everything else.
    dropped_fields: dict[str, object] = {}
    for k, v in (resolution.get("add_fields") or {}).items():
        if k in ParamEntry.model_fields:
            entry[k] = v
        else:
            dropped_fields[k] = v

    # Stale-reasoning guard (fedavg C 2026-07-21): a relabel to source=paper
    # that neither removed nor replaced `reasoning` left the OLD source's
    # justification text ("value from the signature default ...") on an
    # entry now labelled paper-stated — contradictory provenance prose.
    # `paper` is the one source whose schema does not require reasoning, so
    # dropping the stale text is always schema-safe; every other source
    # keeps it (removing it would fail the pydantic gate and revert the
    # whole edit).
    if (
        entry.get("source") == "paper"
        and old_source != "paper"
        and "reasoning" not in (resolution.get("add_fields") or {})
        and "reasoning" not in (resolution.get("remove_fields") or [])
    ):
        entry.pop("reasoning", None)

    # Schema-compat remap: `system_default` asserts the paper STATES a value
    # we deviate from (the schema requires `paper_value`). A relabel to
    # system_default that leaves no paper_value is the reviewer saying "the
    # paper states no value" — which is `system_inferred` by definition, the
    # no-paper-value source kind. Map to the sibling instead of letting the
    # pydantic gate revert-and-halt (bev-distill 2026-07-02: F002's
    # almost-correct resolution halted an otherwise-clean roll).
    source_remapped = False
    if entry.get("source") == "system_default" and entry.get("paper_value") is None:
        source_remapped = True
        new_source = "system_inferred"
        entry["source"] = new_source

    data["params"] = params
    params_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    # Safety net: re-validate against the Params model. If the resolution
    # produced an invalid entry (e.g., new_source=system_inferred without
    # `reasoning`), revert and return failure — the caller halts.
    # The schemas/ package lives at repo root; the driver puts scripts/ on
    # sys.path at startup, so we add REPO_ROOT here (same pattern as the
    # smoke-diagnosis schema import at ~line 2055).
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from schemas.params import Params  # noqa: PLC0415
        Params.model_validate(data)
    except Exception as e:
        # Revert
        params_path.write_text(pre_edit_snapshot + "\n", encoding="utf-8")
        return False, (
            f"resolution produced schema-invalid params.json (reverted): {e}"
        )

    # Second safety net, same revert-and-fail shape: would the RELABELED
    # entry still fail the provenance validator's US-3b arm? Relabeling
    # `source` never clears an inline "paper states name=value" claim left
    # standing in `reasoning`/`note`/`paper_section` — US-3b fires on every
    # source by design — so without this the applier reports success and the
    # very next validator run halts the stage with an "internal
    # inconsistency between applier and validator" (bayesian-active-learning
    # 2026-07-27: a relabel to `system_inferred` kept "The paper uses
    # batch_returns=300", a claim the paper does not support because the
    # paper names that knob b/b'). Failing here routes the finding back as
    # an honest, actionable halt instead of a false resolution.
    paper_md_path = state.paths.pipeline_dir / "paper.md"
    if paper_md_path.is_file():
        from validate_params_provenance import (  # noqa: PLC0415
            unsatisfiable_paper_claims,
        )
        try:
            paper_text = paper_md_path.read_text(encoding="utf-8")
        except OSError:
            paper_text = None
        if paper_text is not None:
            standing = unsatisfiable_paper_claims(entry, paper_text)
            if standing:
                params_path.write_text(pre_edit_snapshot + "\n",
                                       encoding="utf-8")
                claims = ", ".join(f"{n}={v}" for n, v in standing)
                return False, (
                    f"resolution cannot clear the finding (reverted): "
                    f"relabeling {param_name!r} to {new_source!r} leaves an "
                    f"inline paper claim the validator's US-3b arm rejects "
                    f"({claims}). A relabel away from source=paper must also "
                    f"supply a `reasoning` that does not state the paper "
                    f"gives this value"
                )

    aid = _next_assumption_id(state)
    add_fields_summary = ", ".join(
        f"{k}={_format_resolution_field_value(v)}"
        for k, v in (resolution.get("add_fields") or {}).items()
    ) or "(no fields added)"
    remove_fields_summary = ", ".join(resolution.get("remove_fields") or []) or "(none)"
    remap_note = (
        " The reviewer proposed source='system_default', which requires the "
        "paper's own value; this finding establishes the paper states none, "
        "so the driver mapped it to 'system_inferred' (the source kind for "
        "values the paper does not state)."
        if source_remapped else ""
    )
    if dropped_fields:
        dropped_summary = ", ".join(
            f"{k}={_format_resolution_field_value(v)}"
            for k, v in dropped_fields.items())
        remap_note += (
            f" The resolution also proposed non-schema field(s) the params "
            f"entry cannot carry (dropped from the entry, preserved here): "
            f"{dropped_summary}."
        )
    # A resolution artifact without `expert_reasoning` must not render an
    # empty Reasoning line in assumptions.md (SRL 2026-07-29 delivery,
    # R2C-044 rider): fall back to the reasoning text the relabel itself put
    # on the entry, then to a deterministic sentence.
    assumption_reasoning = (
        str(resolution.get("expert_reasoning") or "").strip()
        or str((resolution.get("add_fields") or {}).get("reasoning") or "").strip()
        or (
            f"The reviewer's resolution carried no expert reasoning; the "
            f"provenance relabel from {old_source!r} to {new_source!r} is "
            f"fully described by the finding and action above."
        )
    )
    _append_assumption(
        state, aid=aid,
        title=f"params.{param_name} relabeled {old_source!r} -> {new_source!r}",
        detected=finding.get("description", ""),
        action=(
            f"In params.json, changed `params.{param_name}.source` from "
            f"{old_source!r} to {new_source!r}. Removed fields: "
            f"{remove_fields_summary}. Added/updated fields: {add_fields_summary}."
            f"{remap_note}"
        ),
        reasoning=assumption_reasoning,
        alternative=resolution.get("alternative"),
        override=(
            f"If you have a paper reference that justifies the original "
            f"`source: {old_source}` labeling, edit "
            f"`<run_dir>/.pipeline/params.json` to restore it (with the "
            f"correct `paper_section` reference), then re-run. "
            f"If the script's labeling should be changed permanently, "
            f"update `scripts/derive_params.py` so future runs produce "
            f"the correct provenance directly."
        ),
    )
    return True, aid


def _finding_param_key(location: str) -> str:
    """The bare params.json key for a finding `location`.

    Reviewers write the location field in both bare (`dropout_rate`) and
    prefixed (`params.dropout_rate`) forms — GBALD 2026-07-03 round 2 used
    the prefixed form, so the safe-direction defer missed its lookup and an
    under-claim finding cost a re-ask dispatch it never needed."""
    loc = str(location or "")
    return loc[len("params."):] if loc.startswith("params.") else loc


def _defer_safe_provenance_findings(
    state: "PipelineState", stage_id: str, findings: list[dict],
) -> tuple[list[dict], list[str]]:
    """Split out provenance findings in the SAFE direction (under-claim);
    they route to assumptions.md instead of halting the run.

    Direction is determined from the run's own params.json, not from finding
    prose: a `paper_source_values_match_paper` finding whose target param is
    currently labeled non-paper is the reviewer asking for an UPGRADE to
    paper provenance — the label is more conservative than the paper
    supports, the runtime value is untouched, nothing dishonest ships. Three
    of 2026-06-10's four halts were exactly this shape (GBALD dropout, BADGE
    training threshold + hidden width), each costing a multi-hour resume for
    zero correctness risk. The OVER-claim direction (param labeled paper
    that the paper does not support) still halts: that is the fabrication
    class and needs producer attention, never a run-time relabel."""
    params_path = state.paths.pipeline_dir / "params.json"
    try:
        loaded = json.loads(params_path.read_text(encoding="utf-8"))
        params = loaded.get("params", loaded)
    except (OSError, json.JSONDecodeError):
        return findings, []

    keep: list[dict] = []
    deferred_ids: list[str] = []
    for fnd in findings:
        location = _finding_param_key(fnd.get("location", ""))
        entry = params.get(location)
        source = entry.get("source") if isinstance(entry, dict) else None
        is_underclaim = (
            fnd.get("check_id") == "paper_source_values_match_paper"
            and source is not None
            and source != "paper"
            and not (
                isinstance(entry, dict)
                and entry.get("protocol_role") is not None
            )
        )
        if not is_underclaim:
            keep.append(fnd)
            continue
        fid = fnd.get("id", "F???")
        aid = _next_assumption_id(state)
        _append_assumption(
            state, aid=aid,
            title=f"{location}: provenance label more conservative than the paper",
            detected=(f"Stage-2x review finding {fid}: "
                      f"{fnd.get('description', '(no description)')}"),
            action=(f"Kept the run going. `{location}` is labeled "
                    f"source:{source} while the reviewer reads the paper as "
                    f"stating this value; the runtime value is unchanged, so "
                    f"the label under-claims — the safe direction."),
            reasoning=(
                "An under-claimed label never ships a fabricated paper "
                "quote; halting the run to upgrade a label costs a "
                "multi-hour resume for zero correctness risk. The deriver "
                "templates are the durable fix for the label itself."),
            alternative=("Treat every provenance mismatch as blocking, "
                         "regardless of direction."),
            override=(f"Edit `<run_dir>/.pipeline/params.json` → "
                      f"`params.{location}` (set source / paper_section per "
                      f"the finding), or fix scripts/derive_params.py and "
                      f"re-run the stage."),
        )
        deferred_ids.append(fid)
        log(stage_id, "provenance_underclaim_deferred",
            f"{fid} ({location}): safe-direction provenance finding routed "
            f"to assumptions.md ({aid}) instead of halting")
    return keep, deferred_ids


def _defer_disclosed_demo_scale_findings(
    state: "PipelineState", stage_id: str, findings: list[dict],
) -> tuple[list[dict], list[str]]:
    """Route disclosed demo-scale trade-off findings to assumptions.md instead
    of halting.

    A `system_default_reasoning_explains_deviation` finding fires when the
    stage-2x reviewer judges a system-default param's reasoning to under-justify
    a deviation from taxonomy smoke economics (e.g. an active-learning
    budget-to-pool ratio above the paradigm's max-budget-to-pool ratio, where the
    diversity signal degenerates at smoke scale). When the param is a non-paper
    (system) default whose reasoning already discloses the deviation, this is a
    known smoke-scale economics trade-off the researcher judges, not an
    undefaultable blocker: nothing dishonest ships (the value is not claimed
    from the paper) and the honest record already lives in the param reasoning.
    Per the unattended-run policy (auto-resolve + log to assumptions, halts
    reserved for undefaultable cases) it routes to assumptions.md and the run
    continues. Right-sizing the smoke defaults is the durable deriver-template
    fix. Over-claims (paper-sourced values) and every other finding class are
    untouched and still reach the halt check."""
    params_path = state.paths.pipeline_dir / "params.json"
    try:
        loaded = json.loads(params_path.read_text(encoding="utf-8"))
        params = loaded.get("params", loaded)
    except (OSError, json.JSONDecodeError):
        return findings, []

    typed_axis_shrinks: dict[str, object] = {}
    try:
        spec = json.loads(
            state.paths.method_spec.read_text(encoding="utf-8")
        )
        manifest = load_bundle_manifest(state.paths.run_dir)
    except (OSError, json.JSONDecodeError):
        spec = None
        manifest = None
    if isinstance(spec, dict) and isinstance(manifest, dict):
        typed_axis_shrinks = {
            assessment.param: assessment
            for assessment in protocol_axis_param_shrinks(
                spec, params, manifest,
            )
            if assessment.feasible is False
        }

    keep: list[dict] = []
    deferred_ids: list[str] = []
    for fnd in findings:
        location = _finding_param_key(fnd.get("location", ""))
        entry = params.get(location)
        source = entry.get("source") if isinstance(entry, dict) else None
        reasoning = ((entry.get("reasoning") or entry.get("note"))
                     if isinstance(entry, dict) else None)
        typed_axis_assessment = typed_axis_shrinks.get(location)
        is_typed_axis_shrink = (
            isinstance(entry, dict)
            and typed_axis_assessment is not None
            and entry.get("protocol_role") == "forecast_call_horizon"
            and entry.get("paper_value_status") == "paper_stated"
        )
        is_disclosed_default = (
            fnd.get("check_id") == "system_default_reasoning_explains_deviation"
            and source is not None
            and source != "paper"
            and bool(reasoning)
            # Role-typed protocol findings can be cross-role or false-
            # provenance contradictions, not ordinary smoke economics. Keep
            # them blocking so they cannot be laundered into assumptions.md.
            # The one exception is R2C-081's deterministic insufficient-axis
            # verdict: it carries the exact typed root, provenance root, and
            # all boundary arithmetic in the producer's reasoning.
            and not (
                isinstance(entry, dict)
                and entry.get("protocol_role") is not None
                and not is_typed_axis_shrink
            )
        )
        if not is_disclosed_default:
            keep.append(fnd)
            continue
        fid = fnd.get("id", "F???")
        aid = _next_assumption_id(state)
        if is_typed_axis_shrink:
            title = (
                f"{location}: typed protocol boundary keeps the demo value"
            )
            action = (
                f"Kept the run going with `{location}` as source:{source}; "
                "the converted paper step count remains in paper_value and "
                "the structured R2C-081 verdict proves at least one required "
                f"boundary is insufficient. The parameter record states: "
                f"{reasoning}"
            )
            resolution_reasoning = (
                "This is an enumerated typed-axis resolution recomputed from "
                "comparison.evaluation_protocol, params.json, and "
                "method/example_data/PROVENANCE.json. It is not authorized "
                "by reviewer prose or a generic smoke-economics exception."
            )
            override = (
                f"Provide a compatible bundle whose typed ranges support "
                f"`{location}` at its converted paper step count, then rerun "
                "Stage 2x."
            )
        else:
            title = (
                f"{location}: smoke-scale default deviates from taxonomy "
                "smoke economics"
            )
            action = (
                f"Kept the run going. `{location}` is a system default "
                f"(source:{source}) whose reasoning already discloses the "
                f"deviation; the value is not claimed from the paper, so "
                f"nothing dishonest ships. The trade-off is logged here for "
                f"the researcher to judge. The parameter record states: "
                f"{reasoning}"
            )
            resolution_reasoning = (
                "A disclosed smoke-scale economics trade-off on a non-paper "
                "default is a defaultable decision for an unattended run, not "
                "an undefaultable blocker. Halting costs a multi-hour resume "
                "while the honest record already lives in the param reasoning "
                "and here. Right-sizing the smoke defaults is the durable "
                "deriver fix."
            )
            override = (
                f"Edit scripts/derive_params.py to right-size `{location}` "
                f"(or its inputs) per the finding, then re-run the stage."
            )
        _append_assumption(
            state, aid=aid,
            title=title,
            detected=(f"Stage-2x review finding {fid}: "
                      f"{fnd.get('description', '(no description)')}"),
            action=action,
            reasoning=resolution_reasoning,
            alternative=fnd.get("proposed_fix"),
            override=override,
        )
        deferred_ids.append(fid)
        log(stage_id, "demo_scale_tradeoff_deferred",
            f"{fid} ({location}): disclosed demo-scale trade-off routed to "
            f"assumptions.md ({aid}) instead of halting")
    return keep, deferred_ids


def _try_auto_resolve_stage_review_findings(
    state: "PipelineState", stage_id: str, findings: list[dict],
) -> tuple[list[dict], list[str], list[str]]:
    """Walk stage-reviewer findings; for each one with a structured
    `proposed_resolution`, try to apply it. Returns:
      remaining: findings that were NOT auto-resolved (caller decides halt)
      applied: ids of auto-resolved findings
      failed: ids of findings whose application failed (resolution malformed
              or post-apply validation rejected the edit)

    Findings without `proposed_resolution`, or with an unknown `kind`, pass
    through unchanged. Findings with `resolution_status: "needs_user"` are
    also passed through (the reviewer explicitly declined to auto-resolve)."""
    remaining: list[dict] = []
    applied_ids: list[str] = []
    failed_ids: list[str] = []

    for fnd in findings:
        fid = fnd.get("id", "F???")
        status = fnd.get("resolution_status", "pending")
        resolution = fnd.get("proposed_resolution")

        if status == "needs_user" or not resolution:
            remaining.append(fnd)
            continue

        kind = resolution.get("kind")
        if kind == "relabel_param_source":
            ok, msg = _apply_relabel_param_source_resolution(state, fnd)
            if ok:
                applied_ids.append(fid)
                log(stage_id, "auto_resolved",
                    f"{fid} auto-resolved via relabel_param_source (logged as {msg})")
            else:
                failed_ids.append(fid)
                log(stage_id, "auto_resolve_failed",
                    f"{fid} relabel_param_source apply failed: {msg}")
                # Surface to assumptions.md as needs-user; keep in remaining
                # so the halt check below picks it up.
                _append_needs_user(state, {**fnd, "_apply_failure": msg})
                remaining.append(fnd)
            continue

        # Unknown kind — pass through unchanged. The halt check will halt
        # if the finding is important+, surfacing it for human review.
        remaining.append(fnd)

    return remaining, applied_ids, failed_ids


def _dispatch_stage_reviewer_resolution_reask(
    state: "PipelineState", reviewer_stage_id: str, findings: list[dict],
) -> DispatchResult:
    """One targeted reviewer dispatch asking for structured resolutions on
    the specific findings that would otherwise halt a deterministic-producer
    stage. Writes `stage_review_<reviewer_stage_id>_resolution.json` (a
    separate file, so the original review stays as evidence)."""
    agent = "r2c-stage-reviewer"
    reask_path = (state.paths.pipeline_dir
                  / f"stage_review_{reviewer_stage_id}_resolution.json")
    prompt = build_dispatch_prompt(
        task_summary=STAGE_REVIEW_RESOLUTION_REASK_TEMPLATE.format(
            stage_id=reviewer_stage_id,
            findings_json=json.dumps(findings, indent=2),
        ),
        paths=build_paths_block(state.paths),
        closing=REVIEWER_CLOSING,
        writeable_paths=WRITEABLE_PATHS[agent],
        think_anchor_output_path=str(reask_path),
    )
    return _dispatch_with_scope_check(
        state=state, agent=agent, prompt=prompt, timeout_s=REVIEWER_TIMEOUT_S,
        recovery_check_fn=_stage_review_valid(f"{reviewer_stage_id}_resolution"),
        allow_corrective_redispatch=True,
    )


def _reask_unstructured_blocking_resolutions(
    state: "PipelineState", stage_id: str, reviewer_stage_id: str,
    findings: list[dict],
) -> tuple[list[dict], list[str], list[str]]:
    """ONE reviewer re-ask for blocking findings that carry no structured
    `proposed_resolution` before a deterministic-producer halt.

    The 2026-07-02 GBALD re-roll halted at 2.x on a relabel-shaped finding
    whose reviewer wrote only a prose `proposed_fix` — the resolution
    machinery never got a structured object to apply, so the run died on a
    finding the same reviewer resolves cleanly when asked directly. This
    gives the reviewer exactly one targeted follow-up: attach a structured
    resolution or explicitly mark the finding `needs_user`.

    Returns the same (remaining, applied, failed) contract as
    `_try_auto_resolve_stage_review_findings`. Best-effort throughout: any
    dispatch or parse failure returns the findings unchanged and the caller
    halts exactly as before."""
    blocking = filter_by_severity(findings, min_severity="important")
    candidates = [
        f for f in blocking
        if not f.get("proposed_resolution")
        and f.get("resolution_status", "pending") != "needs_user"
    ]
    if not candidates:
        return findings, [], []
    candidate_ids = {f.get("id") for f in candidates}
    log(stage_id, "resolution_reask",
        f"{len(candidates)} blocking finding(s) lack a structured "
        f"resolution; one reviewer re-ask before halting: "
        + ", ".join(sorted(str(i) for i in candidate_ids)))
    reask_path = (state.paths.pipeline_dir
                  / f"stage_review_{reviewer_stage_id}_resolution.json")
    try:
        _dispatch_stage_reviewer_resolution_reask(
            state, reviewer_stage_id, candidates)
    except (OpencodeClientError, OutOfScopeWritesError) as e:
        log(stage_id, "resolution_reask_failed",
            f"re-ask dispatch failed ({e}); falling through to halt")
        return findings, [], []
    reask, read_err = _read_review_json_or_err(
        reask_path, kind=f"{reviewer_stage_id} resolution re-ask",
        expected_stage_id=reviewer_stage_id,
    )
    if read_err:
        log(stage_id, "resolution_reask_failed",
            f"re-ask output unusable ({read_err}); falling through to halt")
        return findings, [], []

    reask_findings = [f for f in (reask.get("findings") or [])
                      if isinstance(f, dict)]
    by_id = {f.get("id"): f for f in reask_findings}

    # Layer-3 id fallback (ICRA21_HICA 2026-07-08): the re-ask reviewer
    # answered the dispatched synthesized finding (DEF-r_e) under its own
    # original-review numbering (F001), and the strict id-keyed merge
    # dropped a substantively correct relabel — the run halted holding its
    # fix. When a candidate id finds no match, fall back to
    # (check_id, location): both fields come from the validator bullet, are
    # stable across a reviewer's renumbering, and identify the finding in
    # this seam (one finding per param location). The fallback match must
    # be UNIQUE in both directions (one unmatched re-ask answer, one
    # unmatched candidate per key) or it is ignored and the halt proceeds
    # exactly as before. Same genus as the almost-correct-resolution
    # remaps: accept the substance, do not let formatting drop it.
    def _fallback_key(f: dict) -> tuple[str, str] | None:
        check_id, location = f.get("check_id"), f.get("location")
        if (isinstance(check_id, str) and check_id
                and isinstance(location, str) and location):
            return (check_id, location)
        return None

    unmatched_reask = [f for f in reask_findings
                       if f.get("id") not in candidate_ids]
    reask_key_counts: dict[tuple[str, str], int] = {}
    for f in unmatched_reask:
        key = _fallback_key(f)
        if key is not None:
            reask_key_counts[key] = reask_key_counts.get(key, 0) + 1
    by_fallback = {
        key: f for f in unmatched_reask
        if (key := _fallback_key(f)) is not None and reask_key_counts[key] == 1
    }
    unmatched_candidate_key_counts: dict[tuple[str, str], int] = {}
    for f in findings:
        if f.get("id") in candidate_ids and f.get("id") not in by_id:
            key = _fallback_key(f)
            if key is not None:
                unmatched_candidate_key_counts[key] = (
                    unmatched_candidate_key_counts.get(key, 0) + 1)

    to_resolve: list[dict] = []
    passthrough: list[dict] = []
    adopted_ids: list[str] = []
    for fnd in findings:
        fid = fnd.get("id")
        re_fnd = by_id.get(fid) if fid in candidate_ids else None
        if re_fnd is None and fid in candidate_ids:
            key = _fallback_key(fnd)
            if (key is not None and key in by_fallback
                    and unmatched_candidate_key_counts.get(key) == 1):
                re_fnd = by_fallback[key]
                log(stage_id, "resolution_reask_id_fallback",
                    f"re-ask answered finding {fid} under a different id "
                    f"({re_fnd.get('id')!r}); adopted via unique "
                    f"(check_id, location) match {key}")
        update: dict = {}
        if re_fnd:
            # Adopt ONLY the resolution fields; the original finding text
            # stays authoritative (the re-ask must not reword findings).
            if isinstance(re_fnd.get("proposed_resolution"), dict):
                update["proposed_resolution"] = re_fnd["proposed_resolution"]
            if re_fnd.get("resolution_status"):
                update["resolution_status"] = re_fnd["resolution_status"]
        if update:
            adopted_ids.append(str(fid))
            to_resolve.append({**fnd, **update})
        else:
            passthrough.append(fnd)
    if not to_resolve:
        log(stage_id, "resolution_reask_failed",
            "re-ask returned no usable resolution decisions; "
            "falling through to halt")
        return findings, [], []
    log(stage_id, "resolution_reask_merged",
        f"adopted resolution decisions for {len(adopted_ids)} finding(s): "
        + ", ".join(sorted(adopted_ids)))
    remaining, applied_ids, failed_ids = (
        _try_auto_resolve_stage_review_findings(state, stage_id, to_resolve))
    return [*passthrough, *remaining], applied_ids, failed_ids


def _preprocess_pass0_resolutions(
    state: "PipelineState", stage_id: str, crit: list[dict], iteration: int,
) -> tuple[list[dict], set[str], list[str]]:
    """Apply Pass 0 deterministic auto-resolutions and route needs_user findings
    to assumptions.md before the standard Stage 5 routing runs.

    Returns:
      remaining_findings — critical findings still needing standard routing
        (includes regenerate_with_requirement findings, which get their
         `proposed_fix` swapped to the resolution's `guidance` so the existing
         fix-mode dispatcher prompt picks it up).
      dispatched_targets — producer targets implicitly "touched" by
        deterministic appliers (e.g., {'parameter-deriver'} after a rescale);
        used to trigger the matching revalidator + smoke gate.
      auto_resolved_ids — finding IDs handled without re-dispatch (for the
        Stage 5 result notes).
    """
    remaining: list[dict] = []
    dispatched_targets: set[str] = set()
    auto_resolved_ids: list[str] = []

    for fnd in crit:
        fid = fnd.get("id", "F???")
        status = fnd.get("resolution_status", "pending")
        resolution = fnd.get("proposed_resolution")

        if status == "needs_user":
            aid = _append_needs_user(state, fnd)
            log(stage_id, "needs_user",
                f"iteration {iteration}: {fid} surfaced to assumptions.md as {aid}")
            continue

        if not resolution:
            remaining.append(fnd)
            continue

        kind = resolution.get("kind")
        if kind == "rescale_param":
            ok, msg = _apply_rescale_param_resolution(state, fnd)
            if ok:
                dispatched_targets.add("parameter-deriver")
                auto_resolved_ids.append(fid)
                log(stage_id, "auto_resolved",
                    f"iteration {iteration}: {fid} auto-resolved via rescale_param "
                    f"(logged as {msg})")
            else:
                _append_needs_user(state, {**fnd, "_apply_failure": msg})
                log(stage_id, "auto_resolve_failed",
                    f"iteration {iteration}: {fid} rescale_param apply failed: {msg}")
            continue

        if kind == "regenerate_with_requirement":
            modified = dict(fnd)
            modified["proposed_fix"] = (
                resolution.get("guidance") or fnd.get("proposed_fix", "")
            )
            res_target = resolution.get("target_agent")
            if res_target:
                modified["target_agent"] = res_target
            remaining.append(modified)
            continue

        # Unknown kind — fall back to traditional routing rather than halting.
        remaining.append(fnd)

    return remaining, dispatched_targets, auto_resolved_ids


def _run_relevant_revalidators(
    state: PipelineState, stage_id: str, dispatched_targets: set[str]
) -> StageResult | None:
    """After Stage 5 fix-mode dispatches land, re-run the relevant per-
    producer structural validators + re-render notebook (if notebook was
    touched) + smoke gate (always, per the archived v2 orchestrator spec (internal, not shipped) — any producer change can
    break notebook execution). Returns None on success, halted StageResult
    on first failure."""
    # Once any producer fix lands, the prior executed notebook is no longer
    # the source on disk.  Invalidate every artifact derived from that old
    # execution before any validator can halt this path; a failed revalidation
    # must not leave old model/checkpoint evidence beside modified code.
    if dispatched_targets:
        _clear_post_smoke_derived_artifacts(state.paths)

    per_target_validators = [
        ("architecture-coder", ["scripts/validate_architecture_coder_output.py"]),
        ("method-coder",       ["scripts/validate_method_coder_output.py"]),
        ("parameter-deriver",  ["scripts/validate_params_output.py"]),
    ]
    for target, args in per_target_validators:
        if target in dispatched_targets:
            ok, err = _run_stage2_script(state, stage_id=stage_id, args=args)
            if not ok:
                return halt(state.paths, stage_id,
                            reason=f"{args[0]} failed after Stage 5 fix-mode",
                            halt_class="producer_output_invalid",
                            context={"stderr": err}, state=state)

    # Item 26 (ms3d overnight 07-08): a stage-5 fix that touches method/
    # can add a public symbol, but __init__.py's __all__ was finalized at
    # 2d — the ms3d F001 fix added multi_round_self_train to method.py and
    # importing it raised ImportError in the delivered package (caught by
    # the reviewer's own re-review as F005). Re-run the idempotent init
    # finalizer + the import validator whenever a method/-touching producer
    # was dispatched, so the delivered package imports what it exports.
    if dispatched_targets & {"architecture-coder", "method-coder"}:
        proc = run_script(
            stage_id,
            ["scripts/finalize_package_init.py",
             "--spec", str(state.paths.method_spec),
             "--run-dir", str(state.paths.run_dir)],
            timeout=60,
        )
        if proc.returncode != 0:
            return halt(state.paths, stage_id,
                        reason=(f"finalize_package_init.py exit "
                                f"{proc.returncode} after Stage 5 fix-mode"),
                        halt_class="internal_contract_violation",
                        context={"stderr": (proc.stderr or proc.stdout)[-2000:]},
                        state=state)
        log(stage_id, "package_refinalized",
            "stage-5 fix touched method/; __init__.py re-finalized")
        ok, err = _run_stage2_script(
            state, stage_id=stage_id,
            args=["scripts/validate_package_imports.py"],
        )
        if not ok:
            return halt(state.paths, stage_id,
                        reason=("validate_package_imports.py failed after "
                                "Stage 5 fix-mode"),
                        halt_class="producer_output_invalid",
                        context={"stderr": err}, state=state)

    # render_notebook.py consumes both notebook_draft.py and params.json.
    # Any Stage 5 change can invalidate the rendered notebook's embedded
    # params/code, so re-render before notebook validation and smoke.
    if dispatched_targets:
        ok, err = _render_notebook(state)
        if not ok:
            return halt(state.paths, stage_id,
                        reason="render_notebook.py failed after Stage 5 fix-mode",
                        halt_class="producer_output_invalid",
                        context={"stderr": err}, state=state)
    if dispatched_targets:
        ok, err = _run_stage2_script(
            state, stage_id=stage_id,
            args=["scripts/validate_notebook_output.py"],
        )
        if not ok:
            new_failures = _notebook_validation_new_failures(state, err)
            if new_failures:
                return halt(state.paths, stage_id,
                            reason=f"validate_notebook_output.py failed "
                                   f"after Stage 5 fix-mode "
                                   f"({len(new_failures)} failure(s) not "
                                   f"in the stage-3a accepted baseline)",
                            halt_class="producer_output_invalid",
                            context={"stderr": err,
                                     "new_failures": new_failures},
                            state=state)
            _log_accepted_validation_continue(
                state, stage_id, "to the smoke gate after the Stage 5 fix")

        # Smoke gate: full re-execution. The plan calls for this on every
        # post-fix iteration since any producer change can break runtime.
        exit_code, stdout_tail, stderr_tail = _run_smoke_gate(state)
        if exit_code != 0:
            return halt(state.paths, stage_id,
                        reason="smoke gate failed after Stage 5 fix-mode "
                               "(producer fix introduced a runtime regression)",
                        halt_class="producer_output_invalid",
                        context={"stderr": stderr_tail}, state=state)
        # The successful smoke replaced notebook outputs. Re-run the same
        # deterministic post-smoke consumer used by Stage 3.c so the demo,
        # scaling-state, and training-history artifacts are all rebound to the
        # new executed model/checkpoint before Stage 5 probes or reports run.
        _record_demo_verdict(state, stage_id=stage_id)
    return None


def run_stage_5(state: PipelineState) -> StageResult:
    """Stage 5 — findings routing. Outer loop:
      1. Read review_report.json
      2. critical findings → auto-route to fix-mode (one dispatch per
         target_agent group, with their filtered findings)
      3. Re-run relevant producer validators + smoke gate
      4. Re-dispatch paper-fidelity-reviewer in incremental mode
      5. Loop. Cap=3 outer iterations.

    Post-loop: log important + nice-to-have + human-routed findings to
    run-root deferred_findings.md (the MVP fallback for interactive ask_user)."""
    stage_id = "stage_5"
    paths = state.paths
    review_path = paths.pipeline_dir / "review_report.json"
    skipped = _skip_if_done(paths, stage_id, [
        paths.run_dir / run_layout.DEFERRED_FINDINGS_MD,
    ])
    # Skip semantics for Stage 5 are weak: deferred_findings.md only exists
    # if there were any non-critical findings. So skip-if-done sometimes
    # over-skips on clean runs. Keep behavior consistent with other stages
    # and only skip when the file exists AND no halt artifact remains.
    if skipped is not None and review_path.exists():
        return skipped

    if not review_path.exists():
        # Stage 4 degraded without producing a report. Routing is a pure
        # quality stage — the notebook + package are already delivered — so
        # log and finish rather than deny delivery.
        return degrade(paths, stage_id,
                    reason="review_report.json absent at Stage 5 entry (Stage 4 degraded without producing it)",
                    what_failed="findings routing skipped — no paper-fidelity review available",
                    what_to_do=_REVIEW_DEGRADE_TODO,
                    state=state)

    # Truncate any prior `deferred_findings.md` from an earlier invocation.
    # The file is append-only within a single Stage 5 pass (one append per
    # audience: important / nice-to-have / human-routed), but cross-run
    # accumulation would produce duplicate sections. This unlink-on-entry
    # makes each Stage 5 run produce a fresh file.
    deferred_findings_path = paths.run_dir / run_layout.DEFERRED_FINDINGS_MD
    if deferred_findings_path.exists():
        deferred_findings_path.unlink()

    auto_routed_total: list[str] = []  # for the final notes

    for outer_iteration in range(STAGE_5_OUTER_CAP + 1):
        review, read_err = _read_review_json_or_err(
            review_path, kind="stage_5 routing review",
        )
        if read_err:
            return degrade(paths, stage_id,
                        reason=f"stage 5 input review_report.json unusable: {read_err}",
                        what_failed="findings routing skipped — paper-fidelity review unreadable",
                        where="review_report.json",
                        what_to_do=_REVIEW_DEGRADE_TODO,
                        state=state)
        findings = review.get("findings", []) or []
        crit_raw = critical_findings(findings)
        if not crit_raw:
            log(stage_id, "criticals_clear",
                f"iteration {outer_iteration}: no critical findings; "
                f"proceeding to important/nice-to-have routing")
            break
        if outer_iteration == STAGE_5_OUTER_CAP:
            summary = "; ".join(
                f"{f.get('id', '?')}: {(f.get('description') or '')[:120]}"
                for f in crit_raw[:8]
            )
            return degrade(paths, stage_id,
                        reason=f"{len(crit_raw)} critical finding(s) persist after "
                               f"cap={STAGE_5_OUTER_CAP} routing iterations",
                        what_failed=f"{len(crit_raw)} critical paper-fidelity finding(s) could not be auto-resolved",
                        what_to_do=(
                            "These are correctness concerns the reviewer raised that R2C could "
                            "not auto-fix within its retry budget. Review them against the paper "
                            "before relying on the output, or hand the notebook + method/ + this "
                            "list to an AI assistant to resolve. Unresolved findings — " + summary
                        ),
                        state=state)

        # Pass 0 auto-resolution: apply deterministic resolutions (rescale_param)
        # directly, route needs_user findings to assumptions.md, and swap
        # `proposed_fix` for `guidance` on regenerate_with_requirement findings
        # so the existing fix-mode dispatchers receive the binding guidance.
        crit, pre_dispatched_targets, pre_resolved_ids = _preprocess_pass0_resolutions(
            state, stage_id, crit_raw, outer_iteration,
        )
        auto_routed_total.extend(pre_resolved_ids)

        # Group by target_agent. Findings with target_agent=human (or
        # missing/unknown) are not auto-routable; log them and continue.
        groups = group_by_target_agent(crit)
        dispatched_targets: set[str] = set(pre_dispatched_targets)
        for target_name, group in groups.items():
            fix_fn = _stage5_fix_dispatcher(target_name)
            if fix_fn is None:
                log(stage_id, "human_routed",
                    f"iteration {outer_iteration}: {len(group)} critical "
                    f"finding(s) with target_agent={target_name!r} — not "
                    f"auto-routable, surfacing to user")
                _ask_user_log_only(state, group,
                                   f"Critical findings (target_agent={target_name}, "
                                   f"iteration {outer_iteration} — not auto-routable)")
                continue
            log(stage_id, "auto_route",
                f"iteration {outer_iteration}: dispatching {len(group)} "
                f"critical finding(s) to {target_name} in fix-mode")
            try:
                fix_fn(state, group)
                dispatched_targets.add(target_name)
                auto_routed_total.append(f"{target_name}:iter{outer_iteration}")
            except (OpencodeClientError, OutOfScopeWritesError) as e:
                return halt(paths, stage_id,
                            reason=f"fix-mode dispatch to {target_name} failed: {e}",
                            halt_class=_dispatch_error_halt_class(e),
                            retry_count=outer_iteration, state=state)

        if not dispatched_targets:
            # Every critical finding was non-auto-routable (target_agent=human
            # or unknown). Nothing left to do in the outer loop — break out.
            log(stage_id, "no_auto_routable",
                f"iteration {outer_iteration}: all critical findings are "
                f"human-routed; ending outer loop")
            break

        # Re-validate the relevant producers + smoke gate.
        revalidation_halt = _run_relevant_revalidators(
            state, stage_id, dispatched_targets
        )
        if revalidation_halt is not None:
            return revalidation_halt

        # Re-dispatch reviewer in incremental mode for next iteration's read.
        # Pass the FULL prior critical set (crit_raw, including auto-resolved
        # IDs) so the incremental re-reviewer verifies every prior finding —
        # if a rescale didn't actually fix the downstream symptom, the
        # re-review will surface it again.
        try:
            _dispatch_paper_fidelity_reviewer_incremental(
                state,
                iteration=outer_iteration + 1,
                prior_finding_ids=[
                    f.get("id", f"F{i:03d}") for i, f in enumerate(crit_raw)
                ],
                fix_dispatched_to=sorted(dispatched_targets),
            )
        except (OpencodeClientError, OutOfScopeWritesError) as e:
            return halt(paths, stage_id,
                        reason=f"incremental re-review dispatch failed: {e}",
                        halt_class=_dispatch_error_halt_class(e),
                        retry_count=outer_iteration, state=state)
        # Validate the new review_report.
        if not review_path.exists():
            return degrade(paths, stage_id,
                        reason="incremental re-reviewer did not write review_report.json",
                        what_failed="findings routing incomplete — re-review did not complete",
                        what_to_do=_REVIEW_DEGRADE_TODO, state=state)
        ok, err = _run_review_report_validator(state)
        if not ok:
            return degrade(paths, stage_id,
                        reason=f"validate_review_report.py failed after incremental re-review: {err[-400:]}",
                        what_failed="findings routing incomplete — re-review report failed validation",
                        where="review_report.json",
                        what_to_do=_REVIEW_DEGRADE_TODO, state=state)
        # Loop back: next iteration reads the fresh review.

    # Outer loop ended cleanly (no critical findings remain). Handle the
    # rest of the findings via the MVP log-only flow.
    final_review, read_err = _read_review_json_or_err(
        review_path, kind="stage_5 final review",
    )
    if read_err:
        return degrade(paths, stage_id,
                    reason=f"stage 5 final review_report.json unusable: {read_err}",
                    what_failed="findings routing incomplete — final review report unreadable",
                    where="review_report.json",
                    what_to_do=_REVIEW_DEGRADE_TODO, state=state)
    final_findings = final_review.get("findings", []) or []
    important = important_findings(final_findings)
    nice = nice_to_have_findings(final_findings)
    human_routed = [
        f for f in final_findings
        if (f.get("target_agent") or "").lower() in ("human", "user")
        and f.get("severity") != "critical"  # criticals were handled in the loop
    ]
    _ask_user_log_only(state, important, "Important findings")
    _ask_user_log_only(state, nice, "Nice-to-have findings")
    _ask_user_log_only(state, human_routed, "Human-routed findings")
    (stage_review_important, stage_review_nice,
     stage_review_caveats) = _stage_review_deferred_findings(state)
    # Item 2: promote caveats/nice findings that request researcher disclosure
    # into assumptions.md BEFORE logging them, so the deferred_findings.md entry
    # carries the cross-reference. Disclosure only — the label is untouched.
    _promote_reviewer_disclosures(state, stage_review_nice + stage_review_caveats)
    # R2C-068: an important-tier stage-review finding no fix loop acted on is
    # a defect the pipeline found and would otherwise discard. It leads the
    # file, above the nice-to-have tier, because it is the highest-severity
    # thing on this surface.
    if stage_review_important:
        log(stage_id, "stage_review_important_deferred",
            f"{len(stage_review_important)} open important stage-review "
            f"finding(s) carried to the delivered deferred-findings surface: "
            + ", ".join(
                f"{f['id']}@{f.get('raised_at', '?')}"
                for f in stage_review_important))
    _ask_user_log_only(
        state, stage_review_important,
        "Open important findings from earlier stage reviews",
    )
    _ask_user_log_only(
        state, stage_review_nice, "Completed stage-review nice-to-have findings"
    )
    _ask_user_log_only(
        state, stage_review_caveats, "Completed stage-review caveats"
    )

    notes_parts: list[str] = []
    if auto_routed_total:
        notes_parts.append(f"auto-routed {len(auto_routed_total)} group(s)")
    if important:
        notes_parts.append(f"{len(important)} important deferred")
    if nice:
        notes_parts.append(f"{len(nice)} nice-to-have logged")
    if stage_review_important:
        notes_parts.append(
            f"{len(stage_review_important)} open stage-review important disclosed")
    if stage_review_nice:
        notes_parts.append(f"{len(stage_review_nice)} stage-review nice-to-have logged")
    if stage_review_caveats:
        notes_parts.append(f"{len(stage_review_caveats)} stage-review caveat(s) logged")
    notes = "; ".join(notes_parts) or "no findings to route"
    # Readability pass: TOC + NEEDS-USER-first sort so the researcher can spot
    # the entries needing their decision at a glance (hri-readiness Task 2).
    finalize_assumptions_md(state)
    return StageResult(status="completed", stage_id=stage_id, notes=notes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


STAGES: list[str] = [
    "stage_0", "stage_1", "stage_1x",
    "stage_2a", "stage_2b", "stage_2c", "stage_2d", "stage_2x",
    "stage_3a", "stage_3b", "stage_3c",
    "stage_4", "stage_5",
]


def _env_truthy(name: str) -> bool:
    """True iff the env var `name` is set to a truthy token (1/true/yes/on,
    case-insensitive)."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _prime_agent_models(port: int, names: list[str]) -> dict[str, dict | None]:
    """Fetch /agent once and build a {name: model} map, validating that each
    expected agent is present. Fail fast at startup if an agent is missing."""
    available = {a["name"]: a.get("model") for a in list_agents(port=port)}
    missing = [n for n in names if n not in available]
    if missing:
        raise OpencodeClientError(
            f"agents missing on server: {missing} "
            f"(available: {sorted(available.keys())})"
        )
    return {n: available[n] for n in names}


def main(argv: list[str] | None = None) -> int:
    # Repo-root .env first (never overrides the shell), so a fresh machine
    # can configure R2C_MODEL, the Marker knobs, and friends in one file. Every driver
    # env read happens at function time, so main-time loading covers all.
    from env_file import load_env_file  # noqa: PLC0415
    load_env_file()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--paper", help="Paper input (filename, slug, or path).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--server-url",
        default=None,
        help="full opencode server URL (e.g. http://127.0.0.1:39021). "
        "Overrides --port. When omitted, the R2C_SERVER_URL env var "
        "(exported by the r2c opencode plugin) is honored, then --port.",
    )
    parser.add_argument(
        "--dir",
        default=os.getcwd(),
        help="workspace dir for session discovery (default: cwd)",
    )
    parser.add_argument(
        "--stop-after",
        choices=STAGES,
        default=None,
        help="run through this stage and exit (for incremental development)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="start a clean roll: archive an existing delivery for this slug to "
        "<slug>_<n> (nothing deleted) and run from scratch. Item 29 phase 2 — "
        "the deliberate counterpart to same-dir resume. Refused if a run is live "
        "on the slug (kill it first).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="retired: self-tests moved to tests/test_driver_selftest.py",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        # B-07 step 3 (maintainer decision 2026-08-18): the flag is retired; the checks
        # live in pytest. Exit loudly rather than printing nothing and
        # exiting 0 (the vacuous-green trap).
        raise SystemExit(
            "--self-test retired: the driver self-tests moved to "
            "tests/test_driver_selftest.py — run: "
            "python3 -m pytest tests/test_driver_selftest.py"
        )
    if not args.paper:
        parser.error("--paper is required")

    if args.server_url:
        # Pin every opencode_client call to this address. state.port stays
        # as-is for log readability; the URL wins inside the client.
        opencode_client.set_server_url(args.server_url)

    # Defensive guard: refuse to proceed if this looks like a foreground
    # launch inside an opencode bash tool (the deadlock pattern documented
    # in README.md, "How it works"). The `/r2c-run` slash command's `nohup ... &` wrapper
    # is the correct path; this check catches anyone (build agent, manual
    # invocation, future tooling) that bypasses it.
    _check_for_session_deadlock_or_exit()

    # Stage 0 setup resolves the run dir before the lock can exist. Acquire
    # the lock immediately after setup and before PDF parsing, so duplicate
    # launches cannot race on paper.md / parse_quality.json.
    workspace = str(Path(args.dir).resolve())
    paths: PipelinePaths | None = None
    run_lock: RunDirectoryLock | None = None
    stage0_started_event_written = False

    def acquire_stage0_lock(stage0_paths: PipelinePaths) -> None:
        nonlocal paths, run_lock, stage0_started_event_written
        paths = stage0_paths
        run_lock = RunDirectoryLock(
            stage0_paths.pipeline_dir,
            run_id=stage0_paths.slug,
        )
        run_lock.acquire()
        # Run-history ledger: mint this invocation's identity BEFORE the
        # first event append so every event (including run_lock_acquired)
        # reaches the write-through mirror. Best-effort by contract.
        history_pointer = run_history.init_invocation(
            stage0_paths.run_dir,
            slug=stage0_paths.slug,
            paper_input=args.paper,
            input_kind=stage0_paths.input_kind,
            paper_path=stage0_paths.input_path,
            repo_root=stage0_paths.repo_root,
            driver_args={
                "fresh": args.fresh,
                "stop_after": args.stop_after,
            },
        )
        run_started_details: dict = {
            "paper_input": args.paper,
            "workspace": workspace,
        }
        if history_pointer:
            # The segmentation axes historical analysis needs: without the
            # tree commit, cross-time analysis cannot separate "the pipeline
            # changed" from "the model changed". Model ids resolve later
            # (after stage 0) and land in the ledger row instead.
            run_started_details["provenance"] = {
                key: history_pointer.get(key)
                for key in (
                    "invocation_id", "lineage_id", "invocation_index",
                    "tree_commit", "tree_dirty", "host", "paper_sha256",
                    "driver_args", "batch_manifest", "run_tag",
                )
            }
        _append_run_event(
            stage0_paths,
            "run_lock_acquired",
            status="locked",
            summary=f"Acquired run lock for {stage0_paths.slug}",
            details={"lock_path": str(stage0_paths.pipeline_dir / "_lock")},
        )
        _append_run_event(
            stage0_paths,
            "run_started",
            status="running",
            summary=f"R2C run started for {stage0_paths.slug}",
            details=run_started_details,
        )
        _append_run_event(
            stage0_paths,
            "stage_started",
            stage_id="stage_0",
            status="running",
            summary=f"{stage_label('stage_0')} started",
        )
        stage0_started_event_written = True

    try:
        paths, s0 = run_stage_0(
            args.paper,
            workspace=workspace,
            fresh=args.fresh,
            before_parse=acquire_stage0_lock,
        )
    except RunDirectoryLockError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 2
    except Exception:
        if paths is not None and run_lock is not None:
            _append_run_event(
                paths,
                "run_finished",
                status="failed",
                summary="R2C run finished with status=failed",
            )
            _append_run_event(
                paths,
                "run_lock_released",
                status="unlocked",
                summary=f"Released run lock for {paths.slug}",
                details={"lock_path": str(paths.pipeline_dir / "_lock")},
            )
            _finalize_run_history(paths, "failed")
            run_lock.release()
        raise
    if paths is None:
        return 1
    if run_lock is None:
        print("FAIL: run lock was not acquired before Stage 0 parsing", file=sys.stderr)
        return 2

    run_status = "failed"
    state: PipelineState | None = None
    final_notice_state: PipelineState | None = None
    final_notice_payload: dict | None = None
    try:
        if not stage0_started_event_written:
            _append_run_event(
                paths,
                "stage_started",
                stage_id="stage_0",
                status="running",
                summary=f"{stage_label('stage_0')} started",
            )
            stage0_started_event_written = True
        _append_stage_result_event(paths, s0)
        if s0.status == "halted":
            run_status = "halted"
            return 1
        _reconcile_candidate_dispatches(paths, context="startup")

        if args.stop_after == "stage_0":
            # No state object yet; persist a minimal trace manually for inspection.
            (paths.pipeline_dir / "driver_state.json").write_text(
                json.dumps(
                    {
                        "slug": paths.slug,
                        "stages": [
                            {
                                "stage_id": s0.stage_id,
                                "stage_label": stage_label(s0.stage_id),
                                "status": s0.status,
                                "notes": s0.notes,
                            }
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            run_status = "completed"
            return 0

        # Stages 1+ need the opencode session + agent models.
        if not health_check(port=args.port):
            print(
                f"FAIL: opencode server not reachable at "
                f"{opencode_client.resolved_server_address(args.port)}.",
                file=sys.stderr,
            )
            return 2
        try:
            sid = discover_session(directory=workspace, port=args.port)
        except SessionNotFound as e:
            print(f"FAIL: {e}", file=sys.stderr)
            return 2

        try:
            agent_models = _prime_agent_models(
                args.port,
                [
                    "r2c-decomposer",
                    "r2c-method-analyzer",
                    "r2c-architecture-coder",
                    "r2c-method-coder",
                    "r2c-notebook-generator",
                    "r2c-paper-fidelity-reviewer",
                    "r2c-stage-reviewer",
                    "r2c-test-generator",
                ],
            )
        except (OpencodeClientError, OutOfScopeWritesError) as e:
            print(f"FAIL: {e}", file=sys.stderr)
            return 2

        # Cap-burn detection keys on the served model's output limit.
        cap = opencode_client.configure_output_cap(port=args.port)
        log("driver", "output_cap_resolved", f"per-step output cap: {cap} tokens")

        state = PipelineState(
            session_id=sid, port=args.port, paths=paths, agent_models=agent_models,
            workspace=workspace,
        )
        log("driver", "per_dispatch_sessions_on",
            "per-dispatch-session transport is unconditional; each agent runs "
            "in a fresh session nested under the TUI session")
        state.record(s0)

        # Render the pipeline todo list into the TUI side panel via the build
        # agent's TodoWrite. Stage 0 already finished by the time state exists,
        # so seed the list then immediately advance: stage_0 → completed, stage_1
        # → in_progress. The dispatch is best-effort; failure logs and continues.
        _init_pipeline_todos(state)
        _advance_todos(state, "stage_0", s0.status)

        # Stages execute sequentially. A halt at any stage stops the pipeline;
        # --stop-after returns success after the named stage's StageResult is
        # recorded. The substage order matches the archived v2 orchestrator spec (internal, not shipped):
        # a → b → c → d → x.
        stage_fns = [
            ("stage_1", run_stage_1),
            ("stage_1x", run_stage_1x),
            ("stage_2a", run_stage_2a),
            ("stage_2b", run_stage_2b),
            ("stage_2c", run_stage_2c),
            ("stage_2d", run_stage_2d),
            ("stage_2x", run_stage_2x),
            ("stage_3a", run_stage_3a),
            ("stage_3b", run_stage_3b),
            ("stage_3c", run_stage_3c),
            ("stage_4", run_stage_4),
            ("stage_5", run_stage_5),
        ]
        for stage_id, stage_fn in stage_fns:
            _reconcile_candidate_dispatches(
                state.paths,
                context=f"before {stage_id}",
            )
            _append_run_event(
                state.paths,
                "stage_started",
                stage_id=stage_id,
                status="running",
                summary=f"{stage_label(stage_id)} started",
            )
            try:
                result = stage_fn(state)
            except (KeyboardInterrupt, SystemExit):
                # Deliberate stops stay deliberate.
                raise
            except Exception as stage_exc:  # noqa: BLE001 - last-resort guard
                # The class-level floor (stage-exception-guard-design.md):
                # any unhandled exception in any stage function gets a halt
                # artifact + catalog story + explanation-only packaging
                # instead of a bare traceback and a run dir with no front
                # door (the Rethinking-Grouping 2026-07-13 crash shape).
                # Known failure classes keep their own designed handling;
                # this fires only for the unknown ones. rc stays 1.
                result = _last_resort_stage_guard(state, stage_id, stage_exc)
            state.record(result)
            _append_stage_result_event(state.paths, result)
            # Sentinel maintenance: completed stages persist a marker; halts
            # clear it. Skipped stages neither write nor clear (whatever the
            # prior run set is preserved). See _stage_complete_sentinel_path
            # for why this is required in addition to the output-file check.
            if result.status == "completed":
                _write_stage_complete_sentinel(state.paths, stage_id)
                if stage_id == "stage_3c":
                    # Snapshot the upstream file contents so the next run's
                    # smart-skip check can verify nothing changed without
                    # being fooled by idempotent re-writes (R2C 2026-05-22).
                    write_stage_3c_upstream_digest(state.paths)
                elif stage_id == "stage_2d":
                    # Snapshot method/*.py + arch_contract.json + method_spec.json
                    # contents so the next run can detect when stage 2.c
                    # regenerated method.py with API changes that would leave
                    # __init__.py stale (R2C 2026-05-22 cascade).
                    write_stage_2d_upstream_digest(state.paths)
            elif result.status in ("halted", "degraded"):
                # A degraded stage is not clean: clear the complete-sentinel so a
                # re-run re-attempts it (it may recover if the gate is later fixed).
                # Unlike a halt, the pipeline CONTINUES — the issue is logged to
                # KNOWN_ISSUES.md and the best-effort artifact flows downstream.
                _clear_stage_complete_sentinel(state.paths, stage_id)
            _advance_todos(state, stage_id, result.status)
            if result.status == "halted":
                run_status = "halted"
                return 1
            if args.stop_after == stage_id:
                finalize_assumptions_md(state)
                finalize_method_md(state)
                finalize_run_report(state)
                run_status = (
                    "degraded"
                    if any(r.status == "degraded" for r in state.stage_results)
                    else "completed"
                )
                if stage_id == "stage_5":
                    delivery = run_delivery_gating(state)
                    finalize_delivery_banner(state, delivery)
                    finalize_claims_report(state, delivery)
                    manifest_ok, manifest_status = finalize_final_manifest(
                        state, delivery)
                    if manifest_status == "degraded":
                        run_status = "degraded"
                    if not manifest_ok:
                        run_status = "failed"
                        final_notice_state = state
                        final_notice_payload = {
                            "run_status": run_status,
                            "manifest_status": manifest_status,
                            "delivery": delivery,
                        }
                        print(
                            f"FAIL: final_manifest.json status={manifest_status}",
                            file=sys.stderr,
                        )
                        return 2
                    final_notice_state = state
                    final_notice_payload = {
                        "run_status": run_status,
                        "manifest_status": manifest_status,
                        "delivery": delivery,
                    }
                return 0

        # Pipeline reached the end (cleanly or with degraded stages). Finalize the
        # researcher-facing reports: assumptions.md TOC + the KNOWN_ISSUES.md banner
        # on the README, so best-effort delivery is honest about what's unresolved.
        finalize_assumptions_md(state)
        finalize_method_md(state)
        finalize_degraded_surfaces(state)
        finalize_run_report(state)
        run_status = (
            "degraded"
            if any(r.status == "degraded" for r in state.stage_results)
            else "completed"
        )
        delivery = run_delivery_gating(state)
        finalize_delivery_banner(state, delivery)
        finalize_claims_report(state, delivery)
        manifest_ok, manifest_status = finalize_final_manifest(state, delivery)
        if manifest_status == "degraded":
            run_status = "degraded"
        if not manifest_ok:
            run_status = "failed"
            final_notice_state = state
            final_notice_payload = {
                "run_status": run_status,
                "manifest_status": manifest_status,
                "delivery": delivery,
            }
            print(
                f"FAIL: final_manifest.json status={manifest_status}",
                file=sys.stderr,
            )
            return 2
        final_notice_state = state
        final_notice_payload = {
            "run_status": run_status,
            "manifest_status": manifest_status,
            "delivery": delivery,
        }
        return 0
    finally:
        # Insight shadow hook (Block 7 continuation): the best-effort
        # generic-understanding pass, default OFF (R2C_INSIGHT_SHADOW=1).
        # Runs on the common terminal path for every outcome, before the
        # terminal events so its one namespaced record lands inside this
        # invocation's stream. Non-blocking by contract: it never raises,
        # never changes run_status, delivery, or any pre-existing file.
        try:
            from insight_shadow_hook import maybe_run_insight_shadow

            _shadow_report = maybe_run_insight_shadow(
                paths.run_dir, run_id=paths.slug)
            if _shadow_report is not None:
                log("insight_shadow", "completed",
                    f"outcome={_shadow_report.get('outcome')} review="
                    f"{(_shadow_report.get('review') or {}).get('status')}")
        except Exception as e:  # noqa: BLE001 - shadow is non-blocking
            log("insight_shadow", "unexpected_failure",
                f"{type(e).__name__}: {e}")
        _append_run_event(
            paths,
            "run_finished",
            status=run_status,
            summary=f"R2C run finished with status={run_status}",
        )
        _append_run_event(
            paths,
            "run_lock_released",
            status="unlocked",
            summary=f"Released run lock for {paths.slug}",
            details={"lock_path": str(paths.pipeline_dir / "_lock")},
        )
        # Terminal ledger hook AFTER the final events so the mirror holds
        # the complete stream, including run_finished.
        _finalize_run_history(paths, run_status, state)
        if run_lock is not None:
            run_lock.release()
        if state is not None:
            _finalize_delivered_baseline(state)
        if final_notice_state is not None and final_notice_payload is not None:
            try:
                _post_end_of_run_notice(final_notice_state, **final_notice_payload)
            except Exception as e:  # noqa: BLE001 - notice must never mask exit
                log("final_notice", "unexpected_failure",
                    f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    sys.exit(main())
