#!/usr/bin/env python3
"""Run-history ledger: a write-through, mirror-only history of every driver
invocation (the run history ledger design note (internal, not shipped)).

Contract (the same one `_append_run_event` already honors): the ledger can
never block, fail, or slow a run, and nothing in the pipeline ever reads it
back. The run dir stays authoritative for the run; the history dir exists
only for after-the-fact analysis. Every public function here is therefore
best-effort — failures log once and return None.

Layout under `<runs_root>/_history/` (override: R2C_HISTORY_DIR, kill
switch: R2C_HISTORY_DISABLE=1):

    README.md                     DO NOT DELETE sentinel
    ledger.jsonl                  one summary row per driver invocation
    <invocation_id>/
      events.jsonl                byte-identical mirror of run_events.jsonl
      terminal/                   size-capped end-of-run artifact snapshots

Identity: `invocation_id` is minted per driver start; `lineage_id` ties
resumes to the run-dir lifecycle they continue. Both live in
`.pipeline/history_pointer.json`, which survives resumes (same run dir) and
dies with a wipe — exactly the lineage semantics the design wants.

Note on `exit_code` in the ledger row: the driver's finally block does not
see the function's return value, so exit_code is inferred from run_status
(completed/degraded -> 0, halted/failed -> 1). Manifest-write failures that
exit 2 are recorded as run_status="failed" with exit_code 1; run_status is
the authoritative query axis.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LEDGER_SCHEMA_VERSION = "1.0.0"
HISTORY_DIR_ENV = "R2C_HISTORY_DIR"
HISTORY_DISABLE_ENV = "R2C_HISTORY_DISABLE"
BATCH_MANIFEST_ENV = "R2C_BATCH_MANIFEST"
RUN_TAG_ENV = "R2C_RUN_TAG"

POINTER_NAME = "history_pointer.json"
LEDGER_NAME = "ledger.jsonl"
EVENTS_MIRROR_NAME = "events.jsonl"
TERMINAL_DIR_NAME = "terminal"
README_NAME = "README.md"

# Per-file cap for terminal snapshots. Oversized files are copied truncated
# and named loudly in the ledger row's `snapshot_truncated`.
DEFAULT_SNAPSHOT_CAP_BYTES = 512 * 1024

README_TEXT = """\
# r2c run history — DO NOT DELETE

This directory is the ONLY durable record of past R2C runs. Run dirs are
routinely wiped before re-runs (`rm -rf r2c_runs/<slug>` is the working
default); everything here survives those wipes by design.

- `ledger.jsonl` — one JSON line per driver invocation (the query surface;
  DuckDB/pandas read it directly).
- `<invocation_id>/events.jsonl` — the full mirrored event stream for that
  invocation, byte-identical to the run's `run_events.jsonl`.
- `<invocation_id>/terminal/` — size-capped end-of-run artifact snapshots.

Deleting this directory destroys the only historical record of every run
whose run dir has since been wiped.

Design: the run history ledger design note (internal, not shipped)
"""

_mirror_failure_logged = False


def _log(kind: str, message: str) -> None:
    print(f"[history][{kind}] {message}", file=sys.stderr)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def history_disabled() -> bool:
    return os.environ.get(HISTORY_DISABLE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_history_dir(run_dir: Path) -> Path:
    """`R2C_HISTORY_DIR` when set, else a `_history` sibling of the run dir
    (r2c_runs/<slug> -> r2c_runs/_history). The underscore prefix keeps
    `fleet_state.is_run_dir` from ever mistaking it for a run."""
    override = os.environ.get(HISTORY_DIR_ENV, "").strip()
    if override:
        return Path(override)
    return Path(run_dir).resolve().parent / "_history"


def _pipeline_dir_for(run_dir_or_pipeline_dir: Path) -> Path:
    path = Path(run_dir_or_pipeline_dir)
    return path if path.name == ".pipeline" else path / ".pipeline"


def pointer_path(run_dir_or_pipeline_dir: Path) -> Path:
    return _pipeline_dir_for(run_dir_or_pipeline_dir) / POINTER_NAME


def load_pointer(run_dir_or_pipeline_dir: Path) -> dict[str, Any] | None:
    path = pointer_path(run_dir_or_pipeline_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def ensure_history_readme(history_dir: Path) -> None:
    readme = history_dir / README_NAME
    if not readme.exists():
        readme.write_text(README_TEXT, encoding="utf-8")


def mint_invocation_id(slug: str, *, now: str | None = None) -> str:
    stamp = (now or _utc_now_iso()).replace("-", "").replace(":", "")
    return f"{slug}-{stamp}-{secrets.token_hex(3)}"


def collect_git_provenance(repo_root: Path) -> tuple[str | None, bool | None]:
    """Return commit plus tracked-tree dirtiness, or best-effort nulls.

    `tree_dirty` records staged or unstaged changes to tracked files. Untracked
    scratch/run files do not change the committed source tree being measured.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=10,
        )
        if commit.returncode != 0:
            return None, None
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=10,
        )
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        return commit.stdout.strip(), dirty
    except Exception:
        return None, None


def sha256_of(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


def init_invocation(
    run_dir: Path,
    *,
    slug: str,
    paper_input: str | None = None,
    input_kind: str | None = None,
    paper_path: Path | None = None,
    repo_root: Path | None = None,
    driver_args: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Mint this invocation's identity and provenance, persist the pointer,
    and prepare the history dirs. Returns the pointer dict (also the
    run_started enrichment payload), or None when disabled or on failure."""
    if history_disabled():
        return None
    try:
        run_dir = Path(run_dir)
        history_dir = resolve_history_dir(run_dir)
        previous = load_pointer(run_dir)
        started_at = _utc_now_iso()
        invocation_id = mint_invocation_id(slug, now=started_at)
        if previous and isinstance(previous.get("lineage_id"), str):
            lineage_id = previous["lineage_id"]
            prior_index = previous.get("invocation_index")
            invocation_index = (prior_index + 1) if isinstance(prior_index, int) else 2
        else:
            lineage_id = invocation_id.rsplit("-", 1)[0]
            invocation_index = 1
        tree_commit, tree_dirty = collect_git_provenance(
            Path(repo_root) if repo_root is not None else run_dir.parent.parent
        )
        pointer: dict[str, Any] = {
            "ledger_schema_version": LEDGER_SCHEMA_VERSION,
            "invocation_id": invocation_id,
            "lineage_id": lineage_id,
            "invocation_index": invocation_index,
            "slug": slug,
            "history_dir": str(history_dir),
            "started_at": started_at,
            "paper_input": paper_input,
            "input_kind": input_kind,
            "paper_sha256": sha256_of(paper_path) if paper_path else None,
            "tree_commit": tree_commit,
            "tree_dirty": tree_dirty,
            "host": socket.gethostname(),
            "driver_args": driver_args or {},
            "batch_manifest": os.environ.get(BATCH_MANIFEST_ENV) or None,
            "run_tag": os.environ.get(RUN_TAG_ENV) or None,
        }
        invocation_dir = history_dir / invocation_id
        invocation_dir.mkdir(parents=True, exist_ok=True)
        ensure_history_readme(history_dir)
        pointer_file = pointer_path(run_dir)
        pointer_file.parent.mkdir(parents=True, exist_ok=True)
        pointer_file.write_text(
            json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return pointer
    except Exception as e:
        _log("init_failed", f"{type(e).__name__}: {e}")
        return None


def mirror_event_line(pipeline_dir: Path, line: str) -> None:
    """Append one canonical event line to this invocation's mirror.

    Best-effort by contract: any failure logs once per process and is never
    retried. No pointer (unit tests, script-driven appends outside a driver
    run) means no mirror — silently."""
    global _mirror_failure_logged
    if history_disabled():
        return
    try:
        pointer = load_pointer(pipeline_dir)
        if not pointer:
            return
        history_dir = pointer.get("history_dir")
        invocation_id = pointer.get("invocation_id")
        if not history_dir or not invocation_id:
            return
        target = Path(history_dir) / str(invocation_id) / EVENTS_MIRROR_NAME
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(line)
            if not line.endswith("\n"):
                handle.write("\n")
    except Exception as e:
        if not _mirror_failure_logged:
            _mirror_failure_logged = True
            _log("mirror_failed", f"{type(e).__name__}: {e} (further failures silent)")


def tally_message_tokens(messages: list[dict]) -> dict[str, int]:
    """Sum step-finish token counts across a session's assistant messages.
    Shape mirrors opencode's step-finish `tokens` payload; unknown/missing
    fields count as zero.

    Ledger semantics (B-14): opencode's `input` is NET of cache reads —
    verified empirically on the one provider that reports caching (step
    input 9.9M against cache_read 92.1M). The derived `prompt` key is the
    full prompt volume (`input + cache_read`), so the day the serving
    endpoint starts populating `prompt_tokens_details.cached_tokens`,
    cross-window prompt-size comparisons stay meaningful instead of
    `input` silently dropping ~90 percent."""
    totals = {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0}
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        if (message.get("info") or {}).get("role") != "assistant":
            continue
        for part in message.get("parts") or []:
            if not isinstance(part, dict) or part.get("type") != "step-finish":
                continue
            tokens = part.get("tokens") or {}
            if not isinstance(tokens, dict):
                continue
            for key in ("input", "output", "reasoning"):
                value = tokens.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    totals[key] += int(value)
            cache = tokens.get("cache")
            if isinstance(cache, dict):
                for key in ("read", "write"):
                    value = cache.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        totals[f"cache_{key}"] += int(value)
    totals["prompt"] = totals["input"] + totals["cache_read"]
    return totals


def record_dispatch_tokens(
    run_dir_or_pipeline_dir: Path, *, agent: str, messages: list[dict]
) -> None:
    """Accumulate a dispatch's token usage into `.pipeline/token_usage.json`
    (decision 2026-07-14: per-run token totals in the ledger row). The
    dispatch layer already fetches session messages for cap-burn
    classification, so this is a read of data in hand. Best-effort."""
    try:
        tallied = tally_message_tokens(messages)
        if not any(tallied.values()):
            return
        path = _pipeline_dir_for(run_dir_or_pipeline_dir) / "token_usage.json"
        try:
            usage = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(usage, dict):
                usage = {}
        except Exception:
            usage = {}
        totals = usage.get("totals") or {}
        per_agent = usage.get("per_agent") or {}
        agent_totals = per_agent.get(agent) or {}
        for key, value in tallied.items():
            totals[key] = int(totals.get(key, 0)) + value
            agent_totals[key] = int(agent_totals.get(key, 0)) + value
        per_agent[agent] = agent_totals
        usage = {
            "schema_version": "1.0",
            "totals": totals,
            "per_agent": per_agent,
            "dispatches_counted": int(usage.get("dispatches_counted", 0)) + 1,
            "updated_at": _utc_now_iso(),
        }
        path.write_text(json.dumps(usage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as e:
        _log("token_record_failed", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Terminal snapshot + ledger row
# ---------------------------------------------------------------------------

# (snapshot name, run-dir-relative source). Researcher-facing files go
# through run_layout.resolve_existing when available so pre-consolidation
# runs backfill correctly. Never code or notebooks: bulky, and
# r2c_runs/archive plus delivered baselines already cover full-dir
# preservation.
_TERMINAL_ARTIFACTS: list[tuple[str, str]] = [
    ("final_manifest.json", "details/final_manifest.json"),
    ("progress.json", ".pipeline/progress.json"),
    ("judge_decisions.json", ".pipeline/judge_decisions.json"),
    ("assumptions.md", "details/assumptions.md"),
    ("parse_quality.json", ".pipeline/parse_quality.json"),
    ("REPORT.md", "REPORT.md"),
    ("METHOD.md", "METHOD.md"),
    ("paper_map.json", ".pipeline/paper_map.json"),
    ("method_spec.json", ".pipeline/method_spec.json"),
    ("element_tests.json", ".pipeline/element_tests.json"),
]


def _resolve_terminal_source(run_dir: Path, rel: str) -> Path | None:
    try:
        from run_layout import resolve_existing  # scripts/ is on sys.path

        resolved = resolve_existing(run_dir, rel)
        if resolved is not None:
            return resolved
    except Exception:
        pass
    candidate = run_dir / rel
    return candidate if candidate.is_file() else None


def write_terminal_snapshot(
    run_dir: Path,
    invocation_dir: Path,
    *,
    cap_bytes: int = DEFAULT_SNAPSHOT_CAP_BYTES,
) -> tuple[dict[str, int], list[str]]:
    """Copy the terminal artifacts (size-capped) into `terminal/`. Returns
    ({snapshot name: bytes copied}, [names that hit the cap])."""
    run_dir = Path(run_dir)
    terminal_dir = Path(invocation_dir) / TERMINAL_DIR_NAME
    terminal_dir.mkdir(parents=True, exist_ok=True)
    sources = list(_TERMINAL_ARTIFACTS)
    pipeline_dir = run_dir / ".pipeline"
    if pipeline_dir.is_dir():
        for halt in sorted(pipeline_dir.glob("*.halt")):
            sources.append((halt.name, f".pipeline/{halt.name}"))
    snapshot: dict[str, int] = {}
    truncated: list[str] = []
    for name, rel in sources:
        source = _resolve_terminal_source(run_dir, rel)
        if source is None:
            continue
        try:
            with source.open("rb") as handle:
                data = handle.read(cap_bytes + 1)
            if len(data) > cap_bytes:
                data = data[:cap_bytes]
                truncated.append(name)
            (terminal_dir / name).write_bytes(data)
            snapshot[name] = len(data)
        except Exception as e:
            _log("snapshot_failed", f"{name}: {type(e).__name__}: {e}")
    return snapshot, truncated


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _count_events(events_path: Path) -> tuple[int, dict[str, int]]:
    count, by_type = 0, {}
    try:
        with events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                count += 1
                try:
                    event_type = json.loads(line).get("event_type")
                except Exception:
                    continue
                if isinstance(event_type, str):
                    by_type[event_type] = by_type.get(event_type, 0) + 1
    except Exception:
        pass
    return count, by_type


def _assumptions_count(run_dir: Path) -> int | None:
    source = _resolve_terminal_source(run_dir, "details/assumptions.md")
    if source is None:
        return None
    try:
        text = source.read_text(encoding="utf-8")
    except Exception:
        return None
    return sum(1 for line in text.splitlines() if line.startswith("## "))


def _delivery_fields(run_dir: Path) -> tuple[str | None, list[str]]:
    manifest = _load_json(run_dir / "details" / "final_manifest.json")
    if not isinstance(manifest, dict):
        return None, []
    delivery = manifest.get("delivery")
    if not isinstance(delivery, dict):
        return None, []
    label = delivery.get("label") if isinstance(delivery.get("label"), str) else None
    reasons = delivery.get("reasons")
    reason_ids = []
    if isinstance(reasons, list):
        for reason in reasons:
            if isinstance(reason, dict) and isinstance(reason.get("id"), str):
                reason_ids.append(reason["id"])
    return label, reason_ids


def _halt_fields(run_dir: Path) -> tuple[str | None, str | None]:
    pipeline_dir = run_dir / ".pipeline"
    if not pipeline_dir.is_dir():
        return None, None
    halts = sorted(
        (p for p in pipeline_dir.glob("stage_*.halt")),
        key=lambda p: p.stat().st_mtime,
    )
    if not halts:
        return None, None
    newest = halts[-1]
    payload = _load_json(newest)
    halt_class = payload.get("halt_class") if isinstance(payload, dict) else None
    return (halt_class if isinstance(halt_class, str) else None), newest.stem


def _paradigm_and_gap(run_dir: Path, event_type_counts: dict[str, int]) -> tuple[str | None, bool]:
    paradigm = None
    spec = _load_json(run_dir / ".pipeline" / "method_spec.json")
    if isinstance(spec, dict):
        paradigm = (
            ((spec.get("comparison") or {}).get("classification") or {}).get("id")
            if isinstance(spec.get("comparison"), dict)
            else None
        )
        if not isinstance(paradigm, str):
            paradigm = None
    gap_path = bool(event_type_counts.get("provisional_pack_installed")) or (
        run_dir / ".pipeline" / "provisional_pack.json"
    ).is_file()
    return paradigm, gap_path


def _progress_fields(run_dir: Path) -> dict[str, Any]:
    progress = _load_json(run_dir / ".pipeline" / "progress.json")
    if not isinstance(progress, dict):
        return {"stage_durations_s": {}, "terminal_stage": None,
                "stages_completed": [], "progress_started_at": None}
    stages = progress.get("stages") or []
    durations, completed, terminal = {}, [], None
    for row in stages:
        if not isinstance(row, dict):
            continue
        stage_id, status = row.get("stage_id"), row.get("status")
        if not isinstance(stage_id, str):
            continue
        if isinstance(row.get("duration_s"), (int, float)):
            durations[stage_id] = row["duration_s"]
        if status in {"completed", "degraded", "skipped"}:
            completed.append(stage_id)
        if status not in {None, "pending"}:
            terminal = stage_id
    return {
        "stage_durations_s": durations,
        "terminal_stage": progress.get("current") or terminal,
        "stages_completed": completed,
        "progress_started_at": progress.get("started_at"),
    }


_EXIT_CODE_BY_STATUS = {"completed": 0, "degraded": 0, "halted": 1, "failed": 1}


def append_ledger_row(history_dir: Path, row: dict[str, Any]) -> None:
    history_dir = Path(history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    ensure_history_readme(history_dir)
    with (history_dir / LEDGER_NAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True))
        handle.write("\n")


def finalize_invocation(
    run_dir: Path,
    *,
    run_status: str,
    think_model: str | None = None,
    code_model: str | None = None,
    per_dispatch_sessions: bool | None = None,
) -> dict[str, Any] | None:
    """Terminal snapshot + ledger row, from the driver's finally block (and
    the stage-0 exception path). Best-effort: returns the row or None."""
    if history_disabled():
        return None
    try:
        run_dir = Path(run_dir)
        pointer = load_pointer(run_dir)
        if not pointer or not pointer.get("invocation_id"):
            return None
        history_dir = Path(pointer.get("history_dir") or resolve_history_dir(run_dir))
        invocation_id = str(pointer["invocation_id"])
        invocation_dir = history_dir / invocation_id
        snapshot, truncated = write_terminal_snapshot(run_dir, invocation_dir)
        events_file = invocation_dir / EVENTS_MIRROR_NAME
        event_count, event_type_counts = _count_events(
            events_file if events_file.is_file()
            else run_dir / ".pipeline" / "run_events.jsonl"
        )
        progress = _progress_fields(run_dir)
        delivery_label, label_reason_ids = _delivery_fields(run_dir)
        halt_class, halt_stage = _halt_fields(run_dir)
        paradigm, gap_path = _paradigm_and_gap(run_dir, event_type_counts)
        token_usage = _load_json(run_dir / ".pipeline" / "token_usage.json")
        started_at = pointer.get("started_at")
        finished_at = _utc_now_iso()
        wall_seconds = None
        try:
            begin = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
            end = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
            wall_seconds = int((end - begin).total_seconds())
        except Exception:
            pass
        row: dict[str, Any] = {
            "ledger_schema_version": LEDGER_SCHEMA_VERSION,
            "invocation_id": invocation_id,
            "lineage_id": pointer.get("lineage_id"),
            "invocation_index": pointer.get("invocation_index"),
            "slug": pointer.get("slug"),
            "paper_input": pointer.get("paper_input"),
            "paper_sha256": pointer.get("paper_sha256"),
            "input_kind": pointer.get("input_kind"),
            "tree_commit": pointer.get("tree_commit"),
            "tree_dirty": pointer.get("tree_dirty"),
            "think_model": think_model,
            "code_model": code_model,
            "per_dispatch_sessions": per_dispatch_sessions,
            "host": pointer.get("host"),
            "driver_args": pointer.get("driver_args"),
            "batch_manifest": pointer.get("batch_manifest"),
            "run_tag": pointer.get("run_tag"),
            "started_at": started_at,
            "finished_at": finished_at,
            "wall_seconds": wall_seconds,
            "stage_durations_s": progress["stage_durations_s"],
            "run_status": run_status,
            "exit_code": _EXIT_CODE_BY_STATUS.get(run_status),
            "terminal_stage": progress["terminal_stage"],
            "stages_completed": progress["stages_completed"],
            "delivery_label": delivery_label,
            "label_reason_ids": label_reason_ids,
            "halt_class": halt_class,
            "halt_stage": halt_stage,
            "paradigm": paradigm,
            "gap_path": gap_path,
            "event_count": event_count,
            "event_type_counts": event_type_counts,
            "dispatch_failed_count": event_type_counts.get("agent_dispatch_failed", 0),
            "cap_burns": event_type_counts.get("cap_burn_turn_detected", 0),
            "cap_burns_recovered": event_type_counts.get("cap_burn_recovered", 0),
            "truncated_turns": event_type_counts.get("truncated_turn_detected", 0),
            "judge_decisions": event_type_counts.get("judge_decision_recorded", 0),
            "assumptions_recorded": _assumptions_count(run_dir),
            "tokens": (token_usage or {}).get("totals") if isinstance(token_usage, dict) else None,
            "token_dispatches_counted": (
                token_usage.get("dispatches_counted")
                if isinstance(token_usage, dict) else None
            ),
            "events_file": f"{invocation_id}/{EVENTS_MIRROR_NAME}",
            "events_bytes": events_file.stat().st_size if events_file.is_file() else 0,
            "terminal_snapshot": snapshot,
            "snapshot_truncated": truncated,
            "backfilled": False,
        }
        append_ledger_row(history_dir, row)
        return row
    except Exception as e:
        _log("finalize_failed", f"{type(e).__name__}: {e}")
        return None
