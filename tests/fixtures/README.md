# tests/fixtures/

Canned artifacts for the orchestrator test suite. Committed to git so tests are deterministic and reviewable.

## Conventions

- **Keep fixtures small.** A minimal `method_spec.json` is ~30 lines. We don't need a full bev-distill-sized fixture for most tests.
- **One subdirectory per scenario.** E.g., `fixtures/happy_path_at_stage_2c/` contains the state of a pipeline run at the moment stage 2.c is about to start.
- **Match the live schemas.** When a schema changes, fixtures need regenerating. If you edit `schemas/*.py`, run `pytest` to catch breakage.

## Generated vs hand-authored

- **Hand-authored**: minimal artifacts produced by `tests/helpers/inject.py` builders (e.g., `minimal_method_spec()`). Used inline in test bodies; no on-disk fixture needed.
- **Snapshot from real runs**: any non-trivial artifact (a real `arch_contract.json`, a `notebook_draft.py`) lives here as a file and is regenerated periodically from a known-good run.

## Adding a fixture

1. Identify the smallest input that exercises the failure mode you're testing.
2. If it's tiny (one JSON dict, one file), prefer an `inject.py` builder.
3. If it's bigger (model.py + training.py + arch_contract.json from a real run), snapshot it here under a descriptive subdir.
4. Reference the fixture from the test with a relative path; don't duplicate it across tests.
