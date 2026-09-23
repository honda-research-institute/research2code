---
description: Run the R2C pipeline on a paper via the r2c_run plugin tool (detached driver, session stays free).
---

The user invoked `/r2c-run` with arguments: `$ARGUMENTS`

Parse those arguments before calling the tool (they may contain more than the paper):

- The paper is the one argument that is not a flag — a slug (e.g. `bayesian-active-learning`) or a `.md`/`.pdf` filename or path under `input_papers/`.
- If the token `--fresh` appears anywhere in the arguments, DO NOT pass it as part of `paper`; instead set `fresh=true` on the `r2c_run` call. If `--fresh` is absent, omit `fresh` (the default is resume).
- Ignore any other stray token (e.g. a legacy port number — obsolete, see the note below).

Then call the `r2c_run` tool with `paper=<the slug or path>` (and `fresh=true` when `--fresh` was present), then report the pid, log path, and watch command from the tool's output in one short sentence. Do not run any other tools first; do not analyze the output.

A fresh roll archives any existing delivery for that slug to `<slug>_<n>` (nothing is ever deleted) and starts from scratch, instead of the default same-dir resume. Without `--fresh`, re-running a slug picks up where it left off. A fresh roll is refused if a run is already live on the slug; kill the running roll first.

Notes:

- The tool spawns the Python driver detached with this server's own address, so the session stays free for the driver's dispatches back into it. Never run `scripts/run_pipeline.py` in the foreground yourself — a foreground driver deadlocks the session against its own dispatches (see README.md, "How it works").
- A legacy port argument is obsolete: the tool targets the server this session runs in automatically, parallel windows included. If one appears in the arguments, ignore it (only the paper and an optional `--fresh` are meaningful).
- If the `r2c_run` tool is not available, this opencode binary predates the r2c plugin — tell the researcher to update opencode, or to launch via the legacy fallback `./r2c-start.sh` (see README.md, "How it works").

After the tool returns, the driver runs autonomously: it skips already-completed stages, dispatches producer agents into this session, writes `.pipeline/progress.json`, updates the side-panel todo list best-effort, and posts any halt as a `🛑 Pipeline halted at <stage>` notice here. Monitoring options: the watch command from the tool output; `tail -f <log path>` from any terminal; halt artifacts at `r2c_runs/<slug>/.pipeline/<stage>.halt`. The side panel is convenient but not authoritative because opencode can delay TodoWrite updates while a long agent dispatch is running.
