# Preserved run evidence (2026-06)

Raw evidence preserved 2026-06-09 from volatile locations (`/tmp` driver logs vanish on
reboot; `r2c_runs/` is gitignored and deleted by default before re-runs). Audit analyses
of this material live in the session record and `docs/`; this directory keeps the
primary sources those analyses cite.

- `driver-logs/` and `transcripts/` — UNTRACKED since 2026-08-19 (os-cleanup
  decision 2): the raw driver logs and orchestrator transcripts stay on local
  disk and in git history (last tracked at commit ba8ca3338), but are no
  longer part of the tracked tree. Their content summaries live in the git
  history of this README.
- `june9-gbald-run/` — the delivered artifacts + full `.pipeline/` of the 2026-06-09
  GBALD run (status: degraded), minus the ~69MB MNIST cache under `method/example_data/`.
  See `../zoo/README.md` ("Current-best reference") for its known residual defects.

These are **frozen** — do not regenerate or edit. New evidence gets a new dated entry.
