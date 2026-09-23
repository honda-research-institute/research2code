# R2C orchestrator test suite

Driver-level tests for `scripts/run_pipeline.py` and the judge / sentinel infrastructure. Tests run WITHOUT an opencode server, real LLM calls, or real smoke gates — the `FakeDispatch` fixture substitutes for `run_pipeline.dispatch_agent` and tests queue canned outputs per expected call. Suite-wide autouse fixtures in `conftest.py` keep every test network-free (pip preflight skipped, work-session transport stubbed, halt-post transport stubbed, the halt-post retry delay zeroed).

## Running

```bash
# Install deps (one-time; pytest ships in requirements.txt)
pip install -r requirements.txt

# Full suite (~6 minutes)
pytest

# Fast gate (~2 minutes): everything except the heavy probe/validator tier
pytest -m "not probe_runtime"

# One file
pytest tests/test_canary.py
```

## Tiers and markers

- **`probe_runtime`** marks the 13 files whose tests execute real probe / validator / sklearn work (seconds each). The per-block fast gate excludes them; the full suite, phase-boundary gates, and CI include them. Marker drift is checked at phase boundaries by `scripts/check_marker_drift.py` over `pytest --durations=0` output.
- **`manual_only`** marks tests that need real finished run artifacts on local disk (`r2c_runs/`, opt-in `R2C_LIVE_RUN_TESTS=1`); they are vacuous or skipped without them.
- Dependency-gated files (torch, sklearn, …) use module-level `importorskip`; the gated surface is pinned by `tests/test_dependency_gate_pins.py` — update its literal map in the same diff that adds or removes a gate. The whole `probe_runtime` tier also skips when torch is absent (a conftest autouse fixture), because those tests execute generated torch code.

## Writing a new test

```python
from tests.helpers.state import make_state
from tests.helpers.assertions import assert_stage_halted
from tests.helpers.inject import minimal_method_spec


def test_arch_contract_schema_invalid_routes_to_judge(fake_dispatch, run_dir):
    """When validate_arch_contract.py rejects the contract on schema, the
    fix-loop invokes the halt-judge."""
    state = make_state(run_dir)
    # ... set up upstream artifacts ...
    # ... queue the dispatches the test expects ...
    # ... run the stage under test ...
    # ... assert ...
```

Conventions:

1. **Use the `fake_dispatch` fixture** when the code under test dispatches agents. Teardown asserts no expectations were left unfired.
2. **Use `run_dir` (or `tmp_path`)** for hermetic per-test directories — never a fixed `/tmp` path, and never the live working tree.
3. **Use helpers from `tests/helpers/`** — `state.make_state`, `inject.*`, `assertions.*` — to keep test bodies focused on intent rather than setup boilerplate.

## Directory layout

```
tests/
├── conftest.py             # FakeDispatch, FakeSubprocess + autouse network stubs
├── helpers/
│   ├── state.py            # make_state / make_paths
│   ├── inject.py           # failure injection + minimal-artifact builders
│   ├── assertions.py       # assert_stage_halted, assert_stage_degraded, …
│   └── correspondence_run.py
├── fixtures/               # committed canned outputs (see fixtures/README.md)
├── test_canary.py          # infrastructure smoke
└── test_*.py               # per-stage, per-validator, per-probe suites
```

## CI reality

`.github/workflows/tests.yml` runs `pytest -q` on pushes to `main` / `merged_main` and on pull requests — a push to `dev` runs nothing. The CI box is torch-less, so the dependency-gated set is skipped there (visible as module-level skips, counted by the AST pin test). Local full runs on a workstation with torch are the stronger signal.

## What this suite is NOT

- **Not a smoke-gate replacement**. Smoke gates execute generated code on a real Python kernel; this suite tests the orchestrator's behavior around dispatches and validators. Both are needed.
- **Not a judge-accuracy test**. Driver-behavior tests fake the judge's decision (specify "given decision X, does the driver do Y?"). Judge accuracy ("would the judge actually decide X here?") is a separate concern with hand-labeled gold standards — not in v1.

## When tests fail

Read the failing test's docstring and file header for what behavior is pinned and why. The assertion message should name the specific expectation that broke.
