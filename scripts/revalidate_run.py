"""Post-delivery re-validation gate with a provenance-rich delta (R2C-019).

The run companion's post-edit gates prove "your adaptation still imports
and runs." This command converts that into "your adaptation still passes
the checks the delivery passed": it re-runs the delivered probe battery
against the adapted tree in ISOLATED output mode and records a delta
against the committed delivery-time probe report, without touching the
delivery-time evidence.

    python3 scripts/revalidate_run.py --run-dir r2c_runs/<slug>

Each invocation writes one numbered results directory under the run's
designated post-delivery home:

    details/post_delivery_validation/<NNN>/
        probe_report.json    # the battery's isolated verdict report
        claims_ledger.json   # the battery's isolated claims ledger
        revalidation.json    # the delta record (provenance below)

This is the SECOND sanctioned exception to the details/ read-only rule,
exactly the way `details/POST_DELIVERY_CHANGES.md` already is. The
delivered label and everything under `.pipeline/` stay frozen
delivery-time truth: the record marks itself `"delivery_truth": false`
and the delta is always AGAINST that baseline, never a replacement.

Provenance carried by every record: the adapted tree's commit (plus a
dirty flag — the companion runs gates BEFORE its commit, so the results
ride in that commit and git maps them), the baseline and current battery
version stamps, and an attribution for any check-set difference. With
matching versions an added or removed check is attributable to the
adaptation; with differing versions it lands in a battery-drift bucket,
so a re-validation with a newer battery can never masquerade as a
regression or an improvement (and verdict flips under a changed battery
are reported with that caveat attached).

Refusals (exit 2): a live run (`.pipeline/progress.json` run_status
"running" — the companion's own read-only rule for live runs), a run
with no git baseline (run `scripts/finalize_run_git.py` first so the
adapted commit is recordable), and a run with no delivery-time
`probe_report.json` (nothing to diff against — the honest answer is
that this delivery predates the battery record, not a fabricated
baseline).

Exit codes: 0 = no regressions against delivery, 1 = at least one
regression, 2 = setup refusal. Checks that were already failing at
delivery and still fail do NOT gate — they are listed separately as
"still failing from delivery".
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_probes import main as run_probes_main
from trusted import trusted_path

RESULTS_HOME = Path("details") / "post_delivery_validation"
BASELINE_REPORT = Path(".pipeline") / "probe_report.json"

# Worst-first verdict order for per-check aggregation and flip direction.
# `not_applicable` is evidence-neutral like a pass for delta direction: a
# conditional check falling outside the declared method shape cannot itself
# create a regression.  The verdict string and report counts remain distinct.
_SEVERITY = {"pass": 0, "not_applicable": 0, "warn": 1, "unprobeable": 2,
             "flag_for_researcher": 3, "fail": 4}

RECORD_NOTE = (
    "This record describes the run AFTER post-delivery edits. It is not "
    "delivery truth: the delivered label and the evidence under "
    ".pipeline/ describe the run as delivered and never change.")


def _run_is_live(run_dir: Path) -> bool:
    progress = run_dir / ".pipeline" / "progress.json"
    if not progress.is_file():
        return False
    try:
        state = json.loads(progress.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return state.get("run_status") == "running"


def adapted_tree_provenance(run_dir: Path) -> dict | None:
    """The adapted tree's commit and dirty flag from the run's OWN git
    repository, or None when the run has no git baseline."""
    if not (run_dir / ".git").exists():
        return None
    try:
        git_dir = str(trusted_path(run_dir))
        head = subprocess.run(
            ["git", "-C", git_dir, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10)
        status = subprocess.run(
            ["git", "-C", git_dir, "status", "--porcelain"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    commit = head.stdout.strip()
    if head.returncode != 0 or not commit:
        return None
    return {
        "commit": commit,
        "dirty": bool(status.returncode == 0 and status.stdout.strip()),
    }


def next_results_dir(run_dir: Path) -> Path:
    """The next numbered results directory (001, 002, ...) under the
    run's post-delivery validation home."""
    home = run_dir / RESULTS_HOME
    taken = [int(p.name) for p in home.iterdir()
             if p.is_dir() and p.name.isdigit()] if home.is_dir() else []
    return home / f"{max(taken, default=0) + 1:03d}"


def _aggregate(report: dict) -> dict[str, dict]:
    """Worst verdict per probe id, with the message of the worst row —
    one check can emit several rows (per-finding fail rows), and the
    delta compares checks, not rows."""
    out: dict[str, dict] = {}
    for v in report.get("verdicts", []):
        pid = v.get("probe_id", "")
        verdict = v.get("verdict", "")
        rank = _SEVERITY.get(verdict, 0)
        if (
            pid not in out
            or rank > out[pid]["rank"]
            or (
                rank == out[pid]["rank"]
                and out[pid]["verdict"] == "not_applicable"
                and verdict == "pass"
            )
        ):
            out[pid] = {"rank": rank, "verdict": verdict,
                        "message": v.get("message", "")}
    return out


def compute_delta(baseline: dict, current: dict) -> dict:
    """The check-level delta between the delivery-time report and the
    re-validation report, with battery drift bucketed separately."""
    base = _aggregate(baseline)
    curr = _aggregate(current)
    common = sorted(set(base) & set(curr))
    added = sorted(set(curr) - set(base))
    removed = sorted(set(base) - set(curr))

    regressions, improvements, still_failing = [], [], []
    unchanged = 0
    for pid in common:
        b, c = base[pid], curr[pid]
        entry = {"check": pid, "at_delivery": b["verdict"],
                 "now": c["verdict"], "message": c["message"]}
        if b["verdict"] == "pass" and c["verdict"] == "not_applicable":
            # The frozen contract/check set did not change. Losing a check
            # that previously executed is coverage loss, even though a
            # genuinely conditional N/A is non-demoting at delivery time.
            regressions.append(entry)
        elif b["verdict"] == "not_applicable" and c["verdict"] == "pass":
            improvements.append(entry)
        elif c["rank"] > b["rank"]:
            regressions.append(entry)
        elif c["rank"] < b["rank"]:
            improvements.append(entry)
        elif b["verdict"] == "fail":
            still_failing.append(entry)
        else:
            unchanged += 1

    base_version = baseline.get("battery_version")
    curr_version = current.get("battery_version")
    if not base_version:
        attribution = "unknown"
        attribution_note = (
            "the delivery-time report carries no battery version stamp, "
            "so a check-set difference cannot be attributed")
    elif base_version == curr_version:
        attribution = "adaptation"
        attribution_note = (
            "the battery matches the delivery-time version, so any "
            "check-set difference is attributable to the edited run")
    else:
        attribution = "battery_changed_since_delivery"
        attribution_note = (
            "the battery version differs from delivery, so added or "
            "removed checks are battery drift, not effects of the edit, "
            "and verdict flips carry the same caveat")

    return {
        "regressions": regressions,
        "improvements": improvements,
        "still_failing_from_delivery": still_failing,
        "unchanged": unchanged,
        "battery_drift": {
            "checks_added": added,
            "checks_removed": removed,
            "attribution": attribution,
            "note": attribution_note,
        },
    }


def summary_line(delta: dict, results_dir_rel: str) -> str:
    """One line for the change-log entry's gate-results field."""
    drift = delta["battery_drift"]
    checked = (delta["unchanged"] + len(delta["regressions"])
               + len(delta["improvements"])
               + len(delta["still_failing_from_delivery"]))
    parts = [f"battery re-run: {checked} delivered checks compared"]
    if delta["regressions"]:
        ids = ", ".join(e["check"] for e in delta["regressions"])
        parts.append(f"{len(delta['regressions'])} regression(s): {ids}")
    else:
        parts.append("no regressions")
    if delta["improvements"]:
        ids = ", ".join(e["check"] for e in delta["improvements"])
        parts.append(f"{len(delta['improvements'])} improvement(s): {ids}")
    if delta["still_failing_from_delivery"]:
        parts.append(f"{len(delta['still_failing_from_delivery'])} still "
                     "failing since delivery")
    if drift["checks_added"] or drift["checks_removed"]:
        parts.append(
            f"check set drifted (+{len(drift['checks_added'])}/"
            f"-{len(drift['checks_removed'])}, {drift['attribution']})")
    elif drift["attribution"] == "battery_changed_since_delivery":
        parts.append("battery newer than delivery")
    parts.append(f"results: {results_dir_rel}")
    return "; ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--n-classes", type=int, default=None,
                        help="chance-level override for the executed-"
                             "notebook check")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"FAIL: run dir not found: {run_dir}", file=sys.stderr)
        return 2
    if _run_is_live(run_dir):
        print("FAIL: this run is live (run_status: running) — a live run "
              "is read-only; re-validate after it terminates.",
              file=sys.stderr)
        return 2
    provenance = adapted_tree_provenance(run_dir)
    if provenance is None:
        print("FAIL: the run has no git baseline, so the adapted state "
              "would be unrecordable. Snapshot the delivered state first:\n"
              f"  python scripts/finalize_run_git.py --run-dir {run_dir}",
              file=sys.stderr)
        return 2
    baseline_path = run_dir / BASELINE_REPORT
    if not baseline_path.is_file():
        print("FAIL: no delivery-time probe report at "
              f"{baseline_path} — this delivery predates the battery "
              "record, so there is no baseline to diff against.",
              file=sys.stderr)
        return 2
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"FAIL: delivery-time probe report unparseable: {e}",
              file=sys.stderr)
        return 2

    results_dir = next_results_dir(run_dir)
    battery_args = ["--run-dir", str(run_dir),
                    "--output-dir", str(results_dir)]
    if args.n_classes is not None:
        battery_args += ["--n-classes", str(args.n_classes)]
    # The battery imports the run's method package; bytecode caches would
    # dirty the run's git status and break the byte-identity guarantee
    # (same rule as the vendored harness entrypoint).
    dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        battery_rc = run_probes_main(battery_args)
    finally:
        sys.dont_write_bytecode = dont_write
    if battery_rc == 2:
        print("FAIL: the probe battery refused to run; see above.",
              file=sys.stderr)
        return 2

    current = json.loads(
        (results_dir / "probe_report.json").read_text(encoding="utf-8"))
    delta = compute_delta(baseline, current)
    results_dir_rel = results_dir.relative_to(run_dir).as_posix()
    summary = summary_line(delta, results_dir_rel)

    record = {
        "record_type": "post_delivery_revalidation",
        "delivery_truth": False,
        "note": RECORD_NOTE,
        "run": run_dir.name,
        "created": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "adapted_tree": provenance,
        "battery": {
            "at_delivery": baseline.get("battery_version"),
            "now": current.get("battery_version"),
        },
        "baseline_report": BASELINE_REPORT.as_posix(),
        "counts": current.get("counts", {}),
        "delta": delta,
        "summary": summary,
    }
    (results_dir / "revalidation.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"re-validation record: {results_dir / 'revalidation.json'}")
    print(summary)
    if provenance["dirty"]:
        print("adapted tree has uncommitted changes — commit the edit and "
              "these results together so git maps them.")
    return 1 if delta["regressions"] else 0


if __name__ == "__main__":
    sys.exit(main())
