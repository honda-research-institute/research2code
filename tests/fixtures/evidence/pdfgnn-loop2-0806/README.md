# pdfgnn loop-2 stage 2c review (2026-08-06) — the finding_tier_dropped known-bad

Preserved from
`r2c_runs/probabilistic-demand-forecasting-with-graph-neural-networks/.pipeline/`
on 2026-08-06, because `r2c_runs/` is gitignored and deleted by default before
a re-roll.

`stage_review_stage_2c_method.json` is the shaping evidence for R2C-068. The
stage 2c reviewer filed two important-tier findings against the generated
method package and the run closed the stage as "2 findings, 0 critical",
because the stage fix loops route critical findings only. Both findings sat on
disk with `resolution_status: pending` and reached no researcher-facing
surface: not REPORT.md, not `details/assumptions.md`, not
`details/deferred_findings.md`, not `method/README.md`.

The two findings:

- `F001` — the GNN runs once on flattened features and broadcasts one static
  embedding across every timestep, where the paper specifies one embedding per
  context timestep. Filed against the methodology contract element
  `gnn-neighborhood-aggregation`, whose acceptable-approximations list is
  empty.
- `F002` — the Student-t negative log likelihood normalizes with a batch-mean
  degrees-of-freedom where the formula requires per-element df.

Stage 4 then certified "all 8 methodology replication contract elements are
faithfully implemented", naming both elements the 2c reviewer had declared
divergent 54 minutes earlier. An independent code review graded the delivery
down for defects of exactly this class.

Frozen. Do not regenerate or edit.
