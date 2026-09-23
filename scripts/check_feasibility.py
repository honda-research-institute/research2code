"""Stage 2 Step 4b — feasibility gate.

Reads `method_spec.json`, walks the methodology replication contract plus
`critical_requirements.blockers`, and decides whether the pipeline can proceed
with generation.

Halt rules:

1. If `replication_feasibility.verdict` derived from
   `methodology_replication_contract` is `not_replicable`, write
   `<output>.halt` and exit 0.
2. Else, if any blocker has `status: "cannot_implement"`, write
`<output>.halt` with the structured halt record and exit 0. The orchestrator
detects the halt file separately (mirroring the Stage 1 halt-check pattern)
and surfaces it to the user.

Otherwise, write `<output>` with a feasibility summary and exit 0.

Usage:
    python scripts/check_feasibility.py <method_spec.json> --output <feasibility_gate.json>

Exit codes:
  0  ran successfully (caller checks for `<output>.halt` to know whether to halt)
  1  IO error or invalid arguments
  2  schema validation failed on the input spec
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pydantic import ValidationError  # noqa: E402

from schemas.method_spec import MethodSpec  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("spec_path", type=Path, help="Path to method_spec.json")
    parser.add_argument("--output", type=Path, required=True, help="Path for feasibility_gate.json")
    args = parser.parse_args()

    if not args.spec_path.exists():
        print(f"error: spec not found: {args.spec_path}", file=sys.stderr)
        return 1

    try:
        spec = MethodSpec.model_validate_json(args.spec_path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        print(f"error: spec failed schema validation:\n{exc}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"error: cannot read spec: {e}", file=sys.stderr)
        return 1

    replication = spec.derive_replication_feasibility()
    replication_verdict = (
        replication.verdict.value
        if replication is not None
        else None
    )

    blockers = spec.critical_requirements.blockers
    cannot = [b for b in blockers if b.status.value == "cannot_implement"]
    must = [b for b in blockers if b.status.value == "must_implement"]
    approx = [b for b in blockers if b.status.value == "can_approximate"]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if replication is not None and replication.verdict.value == "not_replicable":
        halt_record = {
            "status": "halted",
            "reason": (
                "Methodology replication contract: "
                f"{len(replication.blockers)} core methodology element(s) marked "
                "`not_replicable`. Pipeline cannot proceed to generation without "
                "changing the paper, supplying the missing resource, or adding a "
                "new implementation capability."
            ),
            "not_replicable_core_methodology": [
                {
                    "element_id": blocker.element_id,
                    "technical_concept": blocker.technical_concept,
                    "reason": blocker.reason,
                }
                for blocker in replication.blockers
            ],
            "spec_path": str(args.spec_path),
            "feasibility_summary_at_spec": spec.feasibility.value,
            "replication_feasibility_at_spec": replication.verdict.value,
        }
        halt_path = args.output.with_suffix(args.output.suffix + ".halt")
        halt_path.write_text(json.dumps(halt_record, indent=2) + "\n", encoding="utf-8")
        print(
            f"halt: replication_feasibility=not_replicable; wrote {halt_path}",
            file=sys.stderr,
        )
        for blocker in replication.blockers:
            print(
                f"  - {blocker.element_id}: {blocker.reason}",
                file=sys.stderr,
            )
        return 0

    if cannot:
        halt_record = {
            "status": "halted",
            "reason": (
                f"Feasibility gate: {len(cannot)} blocker(s) marked `cannot_implement`. "
                "Pipeline cannot proceed to generation without addressing or overriding them."
            ),
            "cannot_implement_blockers": [
                {
                    "requirement": b.requirement,
                    "reason": b.reason,
                    "resolution_proposed_by_analyzer": b.resolution,
                }
                for b in cannot
            ],
            "spec_path": str(args.spec_path),
            "feasibility_summary_at_spec": spec.feasibility.value,
            "replication_feasibility_at_spec": replication_verdict,
        }
        halt_path = args.output.with_suffix(args.output.suffix + ".halt")
        halt_path.write_text(json.dumps(halt_record, indent=2) + "\n", encoding="utf-8")
        print(
            f"halt: {len(cannot)} cannot_implement blocker(s); wrote {halt_path}",
            file=sys.stderr,
        )
        for b in cannot:
            print(f"  - {b.requirement}: {b.reason}", file=sys.stderr)
        return 0

    summary = {
        "status": "passed",
        "feasibility_at_spec": spec.feasibility.value,
        "replication_feasibility_at_spec": replication_verdict,
        "blocker_counts": {
            "must_implement": len(must),
            "can_approximate": len(approx),
            "cannot_implement": 0,
        },
        "blockers": [
            {
                "requirement": b.requirement,
                "status": b.status.value,
                "reason": b.reason,
                "resolution": b.resolution,
            }
            for b in blockers
        ],
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        f"ok: feasibility=passed (must_implement={len(must)}, can_approximate={len(approx)}); "
        f"wrote {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
