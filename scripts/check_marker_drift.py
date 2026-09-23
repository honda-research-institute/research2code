"""Marker-drift guard over a `pytest --durations=0` output.

Hand-applied `probe_runtime` markers drift asymmetrically: an unmarked file
turning slow self-announces in fast-gate wall time, but a marked file turning
fast silently costs fast-gate coverage forever. This checker runs at phase
boundaries (Track A gate procedure), never inside pytest — per-test timing
asserts are machine-dependent and flaky.

Usage:
    python3 -m pytest -q --durations=0 2>&1 | tee /tmp/durations.txt
    python3 scripts/check_marker_drift.py /tmp/durations.txt

Fails (exit 1) when an UNMARKED file sums to more than SLOW_UNMARKED_S
seconds (it belongs in the probe_runtime tier) or a MARKED file sums to less
than FAST_MARKED_S (it no longer earns its exclusion from the fast gate).
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

# Calibrated on the 2026-08-18 post-A1 baseline: the marked set sums 7.1s+
# per file, the unmarked ceiling is test_dispatch_hardening.py at 5.7s
# (deliberate paced socket reproducers, knowingly left unmarked by the W3
# tier selection). 8.0 sits in the gap between those two populations.
SLOW_UNMARKED_S = 8.0
FAST_MARKED_S = 2.0

# Keep in lockstep with the pytestmark = pytest.mark.probe_runtime files.
PROBE_RUNTIME_FILES = {
    "test_relational_indexing_contract.py",
    "test_run_probes.py",
    "test_motion_planning_probes.py",
    "test_probe_harness_portability.py",
    "test_al_loop_probes.py",
    "test_constructor_contract.py",
    "test_paradigm_gates.py",
    "test_arch_contract_v2_runtime.py",
    "test_time_series_target_scaling_runtime.py",
    "test_time_series_training_history_runtime.py",
    "test_loader_contract_dict_returns.py",
    "test_stage_1.py",
    "test_element_test_step.py",
}

# One --durations=0 row: "0.05s call tests/test_foo.py::test_bar"
_ROW = re.compile(r"^\s*(\d+\.\d+)s\s+\w+\s+tests/([^:]+)::")


def main(path: str) -> int:
    sums: dict[str, float] = defaultdict(float)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line)
        if m:
            sums[m.group(2)] += float(m.group(1))
    if not sums:
        print(f"no --durations=0 rows found in {path}", file=sys.stderr)
        return 1
    problems: list[str] = []
    for fname, total in sorted(sums.items(), key=lambda kv: -kv[1]):
        marked = fname in PROBE_RUNTIME_FILES
        if not marked and total > SLOW_UNMARKED_S:
            problems.append(
                f"UNMARKED {fname} sums to {total:.1f}s "
                f"(> {SLOW_UNMARKED_S}s): add probe_runtime"
            )
        elif marked and total < FAST_MARKED_S:
            problems.append(
                f"MARKED {fname} sums to {total:.1f}s "
                f"(< {FAST_MARKED_S}s): drop probe_runtime"
            )
    for p in problems:
        print(p)
    if not problems:
        print(f"marker drift: clean ({len(sums)} test files measured)")
    return 1 if problems else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
