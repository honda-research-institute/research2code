# Contributing to R2C

Thanks for looking at R2C. This is a research prototype; contributions are
welcome within the conventions below.

## Setup

```bash
pip install -r requirements.txt
python -m pytest -m "not probe_runtime" -q     # fast gate, ~2 min
python -m pytest -q                            # full suite, ~6 min
```

No GPU, no model endpoint, and no opencode server are needed for the test
suite. Tests that execute generated torch code (the `probe_runtime` tier and
a few torch-gated files) skip automatically when torch is absent; tests that
need real finished run directories on local disk skip honestly without them.
With torch installed the full suite runs everything.

## Ground rules

- **Fixes must be generalizable.** The pipeline is paper-agnostic: never
  branch on a paper slug or bake in paper-specific magic values. Treat the
  paper that exposed a bug as the reproducer, not the target, and fix the
  failure class where all papers route through.
- **Root cause over symptom.** Walk a bad value upstream before adding
  detection or recovery layers.
- **Never hand-edit generated run output.** Files under `r2c_runs/<slug>/`
  are producer artifacts; if one is wrong, the fix lives in the producer
  (agent prompt, taxonomy node, validator, or template), never in the
  artifact.
- **Every behavior change lands with a test.** Known-bad plus known-good
  coverage for the failure class it fixes.
- **Method/task knowledge belongs in the taxonomy SSOT**
  (`docs/ssot/taxonomies.yaml`), not in driver code. Run
  `python scripts/validate_taxonomy.py` and
  `python scripts/validate_field_guides.py report` after touching it —
  CI runs both.

## Pull requests

- Keep each PR to one logical change with its tests.
- The CI workflow (`.github/workflows/tests.yml`) is what a fork runs:
  `pip install -r requirements.txt`, the two hygiene validators, then
  `pytest`. It must stay green.
- Commit messages: a subject line that states the behavior change, a body
  that explains why, and test evidence.

## License

By contributing you agree that your contributions are licensed under the
repository's [license](LICENSE.pdf).
