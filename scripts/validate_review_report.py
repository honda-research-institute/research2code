"""Stage 4 — validator for review_report.json.

Runs after the paper-fidelity reviewer. Confirms:

  1. `<run_dir>/.pipeline/review_report.json` exists.
  2. The file parses against the `ReviewReport` Pydantic schema.
  3. Finding IDs are unique within the report.
  4. `review_status` is consistent with the findings (`passed` iff no
     critical or important findings exist).

This is a thin gate — most of the validation work is the schema itself
(via Pydantic). The reviewer's value is *judgment*, which a deterministic
validator can't second-guess; we just confirm the report is well-formed and
internally consistent.

Usage:

    python scripts/validate_review_report.py --run-dir <output_dir>

Exit codes:
  0  validation passed
  1  validation failures (schema or consistency)
  2  setup error (missing file)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pydantic import ValidationError  # noqa: E402

from schemas.review_report import ReviewReport  # noqa: E402


def validate(run_dir: Path) -> list[str]:
    errors: list[str] = []

    report_path = run_dir / ".pipeline" / "review_report.json"
    if not report_path.is_file():
        return [f"review_report.json missing at {report_path}"]

    raw = report_path.read_text(encoding="utf-8")
    try:
        report = ReviewReport.model_validate_json(raw)
    except ValidationError as e:
        return [f"review_report.json fails Pydantic validation: {e}"]
    except json.JSONDecodeError as e:
        return [f"review_report.json is not valid JSON: {e}"]

    # Unique IDs
    ids = [f.id for f in report.findings]
    duplicates = {fid for fid in ids if ids.count(fid) > 1}
    if duplicates:
        errors.append(f"findings have duplicate IDs: {sorted(duplicates)}. Each F0NN must be unique.")

    # review_status consistency
    has_critical_or_important = any(f.severity in ("critical", "important") for f in report.findings)
    if has_critical_or_important and report.review_status != "issues_found":
        errors.append(
            "review_status is 'passed' but at least one finding has severity=critical or important. "
            "If any critical/important findings exist, review_status must be 'issues_found'."
        )
    if not has_critical_or_important and report.review_status == "issues_found":
        errors.append(
            "review_status is 'issues_found' but no findings have severity=critical or important. "
            "Set review_status to 'passed' when all findings are nice-to-have or below."
        )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    errors = validate(args.run_dir)
    if errors:
        print(f"FAIL: {len(errors)} validation error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    # Successful — print a brief summary of findings count.
    report = ReviewReport.model_validate_json(
        (args.run_dir / ".pipeline" / "review_report.json").read_text(encoding="utf-8")
    )
    counts = {sev: 0 for sev in ("critical", "important", "nice-to-have")}
    for f in report.findings:
        counts[f.severity] += 1
    summary = f"status={report.review_status}, findings: {counts['critical']} critical, {counts['important']} important, {counts['nice-to-have']} nice-to-have"
    print(f"ok: review_report.json validates ({summary}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
