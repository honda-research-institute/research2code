#!/usr/bin/env python3
"""Calibration harness: drive the insight semantic reviewer over the
golden corpus and verify the hiring bar.

Design: the insight semantic reviewer design note (internal, not shipped).
The golden corpus (tests/fixtures/generic_insights/semantic_review/) is
the recorded calibration bar: all rejected corruptions must come back
rejected and all accepted accounts accepted, with ZERO tolerance for
accepting a known-bad. A needs_human on an accepted account is a
calibration miss worth investigating but not disqualifying.

The calibration output is itself a kept artifact (per-case verdicts and
reasons, hash-bound to the agent definition and each dispatch prompt) so
promotion evidence is inspectable, and it reruns on any change to the
reviewer's prompt, model, or bundle construction.

Corpus cases never become production heuristics: the reviewer is prompted
with the failure-class genus only; this harness measures it.

The dispatch function is injectable — tests exercise the bar logic with
fake reviewers; the CLI dispatches live against an opencode server using
the per-dispatch-session transport (one fresh session per case).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "scripts") not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from dispatch_templates import build_insight_review_prompt  # noqa: E402
from generic_insights import atomic_write_text  # noqa: E402
from insight_semantic_review import (  # noqa: E402
    build_case_bundle,
    load_acceptance_record,
)

DEFAULT_AGENT = "r2c-insight-semantic-reviewer"
DEFAULT_CORPUS = _REPO_ROOT / "tests/fixtures/generic_insights/semantic_review"
AGENT_DEFINITION = _REPO_ROOT / ".opencode/agents/r2c-insight-semantic-reviewer.md"
RECORD_FILENAME = "generic_insights.semantic_review.json"
REPORT_FILENAME = "calibration_report.json"

# Outcome classes. Plain language: "match" is the reviewer agreeing with
# the recorded corpus verdict; "accepted_known_bad" is the zero-tolerance
# catastrophic case (a recorded corruption came back accepted);
# "undecided_known_bad" and "no_record" fail the bar because the bar
# requires every corruption actively rejected; "rejected_known_good"
# fails the bar (a recorded-good account came back rejected);
# "undecided_known_good" is the sanctioned non-disqualifying miss.
DISQUALIFYING_OUTCOMES = frozenset({
    "accepted_known_bad",
    "undecided_known_bad",
    "rejected_known_good",
    "no_record",
})

# A dispatch function receives (case, prompt, record_path) and must leave
# the reviewer's acceptance record at record_path (or not, on failure).
DispatchFn = Callable[["CalibrationCase", str, Path], None]


@dataclass(frozen=True)
class CalibrationCase:
    case_id: str
    expected_verdict: str  # "accepted" | "rejected" (recorded corpus verdict)
    source_bytes: bytes
    candidate: dict
    source_sha256: str
    candidate_sha256: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_corpus_cases(
    corpus_dir: Path = DEFAULT_CORPUS, repo_root: Path = _REPO_ROOT
) -> list[CalibrationCase]:
    """All corpus cases, hash-verified against their recorded
    expectations. A calibration over a drifted corpus is meaningless, so
    any mismatch between live bytes and recorded hashes fails loudly here
    rather than producing a report bound to the wrong bytes."""
    cases: list[CalibrationCase] = []
    for cdir in sorted(p for p in Path(corpus_dir).iterdir() if p.is_dir()):
        candidate_bytes = (cdir / "candidate.json").read_bytes()
        candidate = json.loads(candidate_bytes)
        expectations = json.loads(
            (cdir / "expectations.json").read_text(encoding="utf-8"))
        source_bytes = (repo_root / candidate["source"]["path"]).read_bytes()
        source_sha = _sha256(source_bytes)
        candidate_sha = _sha256(candidate_bytes)
        if source_sha != expectations["source_sha256"]:
            raise SystemExit(
                f"{cdir.name}: live source hash does not match the recorded "
                "expectation — corpus drifted; refusing to calibrate")
        if candidate_sha != expectations["candidate_sha256"]:
            raise SystemExit(
                f"{cdir.name}: live candidate hash does not match the "
                "recorded expectation — corpus drifted; refusing to calibrate")
        cases.append(CalibrationCase(
            case_id=expectations["case_id"],
            expected_verdict=expectations["verdict"],
            source_bytes=source_bytes,
            candidate=candidate,
            source_sha256=source_sha,
            candidate_sha256=candidate_sha,
        ))
    if not cases:
        raise SystemExit(f"no corpus cases found under {corpus_dir}")
    return cases


def classify_outcome(expected_verdict: str, overall: str | None) -> str:
    """Map (recorded corpus verdict, reviewer overall verdict or None for
    an absent record) to an outcome class."""
    if overall is None:
        return "no_record"
    if expected_verdict == "rejected":
        if overall == "rejected":
            return "match"
        if overall == "accepted":
            return "accepted_known_bad"
        return "undecided_known_bad"
    if overall == "accepted":
        return "match"
    if overall == "rejected":
        return "rejected_known_good"
    return "undecided_known_good"


def run_calibration(
    cases: list[CalibrationCase],
    dispatch: DispatchFn,
    out_dir: Path,
    *,
    agent_name: str = DEFAULT_AGENT,
    binding: dict | None = None,
) -> dict:
    """Dispatch every case, resolve each record with full hash binding,
    classify outcomes, and write the calibration report. Returns the
    report dict. Dispatch failures are recorded per-case (an absent
    record → no_record), never raised — the report is the evidence either
    way."""
    out_dir = Path(out_dir)
    case_results: list[dict] = []
    for case in cases:
        case_dir = out_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        record_path = case_dir / RECORD_FILENAME
        bundle = build_case_bundle(case.source_bytes, case.candidate)
        prompt = build_insight_review_prompt(
            bundle_json=json.dumps(bundle, indent=1, sort_keys=True),
            record_path=str(record_path),
            source_sha256=case.source_sha256,
            candidate_sha256=case.candidate_sha256,
        )
        dispatch_error: str | None = None
        try:
            dispatch(case, prompt, record_path)
        except Exception as exc:  # noqa: BLE001 - report, never crash the bar
            dispatch_error = f"{type(exc).__name__}: {exc}"
        resolution = load_acceptance_record(
            record_path,
            source_sha256=case.source_sha256,
            candidate_sha256=case.candidate_sha256,
            expected_record_ids={case.case_id},
        )
        record = resolution.record
        overall = record.overall_verdict if record else None
        outcome = classify_outcome(case.expected_verdict, overall)
        case_results.append({
            "case_id": case.case_id,
            "expected_verdict": case.expected_verdict,
            "record_present": resolution.present,
            "record_resolution": resolution.reason,
            "dispatch_error": dispatch_error,
            "overall_verdict": overall,
            "verdicts": [
                {
                    "record_id": v.record_id,
                    "verdict": v.verdict,
                    "failure_class": v.failure_class,
                    "reason": v.reason,
                }
                for v in (record.verdicts if record else [])
            ],
            "needs_human": [
                {"record_id": e.record_id, "reason": e.reason}
                for e in (record.needs_human if record else [])
            ],
            "outcome": outcome,
            "prompt_sha256": _sha256(prompt.encode("utf-8")),
        })

    outcomes = [c["outcome"] for c in case_results]
    disqualifying = sorted(
        c["case_id"] for c in case_results
        if c["outcome"] in DISQUALIFYING_OUTCOMES)
    misses = sorted(
        c["case_id"] for c in case_results
        if c["outcome"] == "undecided_known_good")
    report = {
        "schema_version": "1.0.0",
        "agent": agent_name,
        "binding": binding or {},
        "cases": case_results,
        "summary": {
            "total": len(case_results),
            "matches": outcomes.count("match"),
            "known_bad_accepted": outcomes.count("accepted_known_bad"),
            "disqualifying_cases": disqualifying,
            "non_disqualifying_misses": misses,
            "zero_known_bad_accepts":
                outcomes.count("accepted_known_bad") == 0,
            "bar_passed": not disqualifying,
        },
    }
    atomic_write_text(
        out_dir / REPORT_FILENAME,
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def agent_definition_binding(
    agent_definition: Path = AGENT_DEFINITION,
) -> dict:
    """Hashes that bind a calibration report to the reviewer build it
    measured: the agent definition (instructions + declared model) and the
    schema module. Prompt/bundle construction is bound per-case via
    prompt_sha256."""
    binding: dict = {}
    for label, path in (
        ("agent_definition_sha256", agent_definition),
        ("record_schema_sha256", _REPO_ROOT / "schemas/semantic_review.py"),
        ("bundle_builder_sha256",
         _REPO_ROOT / "scripts/insight_semantic_review.py"),
    ):
        try:
            binding[label] = _sha256(Path(path).read_bytes())
        except OSError:
            binding[label] = None
    return binding


def make_live_dispatch(
    *,
    agent: str,
    port: int,
    server_url: str | None,
    timeout_s: float,
    directory: str,
) -> DispatchFn:
    """One fresh opencode session per case (the per-dispatch-session
    transport): no shared conversation state can bleed between cases."""
    import opencode_client  # noqa: PLC0415 - keep the module import-light

    if server_url:
        opencode_client.set_server_url(server_url)

    def dispatch(case: CalibrationCase, prompt: str, record_path: Path) -> None:
        session_id = opencode_client.create_session(
            title=f"r2c insight-review calibration: {case.case_id}",
            directory=directory,
            port=port,
        )
        opencode_client.dispatch_and_wait(
            session_id=session_id, agent=agent, prompt=prompt,
            port=port, timeout_s=timeout_s)

    return dispatch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the insight semantic reviewer over the golden corpus and "
            "verify the calibration bar (zero known-bad accepts, every "
            "corruption rejected, every accepted account accepted)."))
    parser.add_argument("--out", required=True, type=Path,
                        help="output directory for per-case records and "
                             "the calibration report (kept artifact)")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--port", type=int, default=4096)
    parser.add_argument("--server-url", default=None,
                        help="explicit opencode server URL (overrides --port)")
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="per-dispatch timeout in seconds")
    args = parser.parse_args(argv)

    # The reviewer writes each record with its own Write tool, and the
    # opencode server only auto-allows writes inside its project
    # directory. An out-of-tree --out makes every dispatch hang on a
    # permission ask no headless session can answer (each case then burns
    # its full timeout and reads as no_record) — refuse up front.
    out_resolved = args.out.resolve()
    if not out_resolved.is_relative_to(_REPO_ROOT):
        parser.error(
            f"--out {out_resolved} is outside the repo; the reviewer's "
            "writes are only auto-allowed inside the project directory "
            "(use a gitignored path such as r2c_output/)")

    cases = load_corpus_cases(args.corpus)
    dispatch = make_live_dispatch(
        agent=args.agent, port=args.port, server_url=args.server_url,
        timeout_s=args.timeout, directory=str(_REPO_ROOT))
    binding = agent_definition_binding()
    binding["run_started_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report = run_calibration(
        cases, dispatch, args.out, agent_name=args.agent, binding=binding)

    for case in report["cases"]:
        print(f"{case['case_id']}: expected {case['expected_verdict']}, "
              f"got {case['overall_verdict'] or 'no record'} "
              f"-> {case['outcome']}")
    summary = report["summary"]
    print(f"matches {summary['matches']}/{summary['total']}; "
          f"known-bad accepts: {summary['known_bad_accepted']}; "
          f"bar {'PASSED' if summary['bar_passed'] else 'FAILED'}")
    if summary["non_disqualifying_misses"]:
        print("non-disqualifying misses (needs_human on accepted cases): "
              + ", ".join(summary["non_disqualifying_misses"]))
    if summary["disqualifying_cases"]:
        print("disqualifying cases: "
              + ", ".join(summary["disqualifying_cases"]))
    return 0 if summary["bar_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
