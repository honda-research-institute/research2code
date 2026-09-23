---
description: Author a taxonomy-pack proposal for a halted unmatched-paper run via the r2c_paradigms plugin tool.
---

The user invoked `/r2c-paradigms $1`. Call the `r2c_paradigms` tool with `run="$1"`, then report the pid, log path and run dir from the tool's output in one short sentence. Do not run any other tools first; do not analyze the output.

Notes:

- `$1` is a run slug under `r2c_runs/` (for example `ADAM`) or an explicit run directory path. The run must contain `.pipeline/paradigm_gap_report.json` (a Stage-1 paradigm-gap halt).
- The tool spawns the authoring driver detached with this server's own address. Do NOT run `scripts/author_field_guide_proposal.py` in the foreground — it dispatches `r2c-field-guide-author` back into this same session and would deadlock against its own dispatch.
- A second argument (the old port, `$2`) is obsolete: the tool targets the server this session runs in automatically.

The authoring driver creates or reuses a proposal packet under `<run_dir>/.pipeline/paradigm_proposals/<proposal_id>/`, dispatches the author, validates with `scripts/validate_paradigm_proposal.py`, redispatches with validation errors up to the retry cap, and stops before promotion. Promotion remains a maintainer action:

```bash
python3 scripts/apply_paradigm_proposal.py <proposal_dir>
```

Monitoring options: the author's output appearing live in this session; `tail -f <log path>`; `<proposal_dir>/proposal_authoring_log.md` and `<proposal_dir>/validation_report.json`.
