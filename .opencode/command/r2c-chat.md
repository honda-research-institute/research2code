---
description: Chat with your R2C runs — grounded answers across every run, provenance-tracked edits, honest re-gates (routes to the r2c-companion agent).
agent: r2c-companion
---

The researcher invoked `/r2c-chat $1`.

Your scope is the whole fleet: every run directory under `r2c_runs/`.
Cross-run questions are
normal and expected — the researcher may reference other papers'
implementations without naming folders.

Build your fleet index with EXACTLY ONE tool call:

    python scripts/r2c_fleet_index.py

Never build the index by reading per-run files yourself — the one-call
script exists because a burst of parallel reads has killed this
greeting turn before. Rows showing `?` are old-layout or partial runs;
say so if asked, read deeper only on demand.

If `$1` names a run in the index, that run is your FOCUS: read its
`REPORT.md` (that one file only) and greet with three sentences at
most — what it delivered (label glossed in plain language), its one or
two most important disclosures, and an invitation to ask, change, or
plan. Mention in one clause that you can also see their other runs. Do
not read anything else until a question needs it.

If `$1` is empty or matches nothing, greet with the fleet: the index
rows rendered as one short line per run (slug, plain-language label,
where it stopped if halted), then the same invitation, two sentences
max. Do not read any per-run files for this greeting.

If a run shows "running" in the index, it is LIVE: read-only until it
terminates — never edit, re-gate, or git-touch inside it.

From here on, follow your agent procedure: grounding ladder per run,
the provenance protocol for every edit, the source-verification rule
for cross-run ports, feedback capture for anything you cannot serve.
