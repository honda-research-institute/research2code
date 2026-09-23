# Run Report

Run: `deep-batch-active-learning`

Delivery label: **verified**.

Label scale, least to most reliable: explanation only → uncertified — new territory → draft → **verified**. This run sits at 4 of 4.

Machine-readable manifest: `details/final_manifest.json`.

After this report, open: [`notebook.ipynb`](notebook.ipynb) — the tutorial; `method/` — the importable package; [`METHOD.md`](METHOD.md) — the method explained.

## Verification — what we could and could not confirm

This is the claims ledger. The honest default is untested at smoke scale. A claim is marked verified only when a scale-free behavioral check passes and also fails for a baseline with the contribution removed, so the check is meaningful. Where our run fell short, the ledger says so and attributes it to our own implementation, never to the paper.

**Claims checked:** 2 verified at smoke scale (behavioral properties), 0 contradicted, 0 need a fix on our side, 1 untested at this scale.

| Claim | Verdict | What this means |
|---|---|---|
| BADGE robustness evaluation | Untested at this scale | We did not verify this claim. It is a scale-bound experiment claim, and the paper-map text does not contain one unambiguous metric value that can be matched to the executed notebook. To verify it: a full-scale run on the paper's dataset, training budget, and seed count. |
| the method selects differently from random acquisition | Verified at smoke scale | We verified at smoke scale that the method selects differently from random acquisition. This check fails for the contribution-removed baseline, so it is meaningful. It is a scale-free behavioral property and does not confirm the paper's full-scale headline numbers. |
| every claimed scoring term can change the selection | Verified at smoke scale | We verified at smoke scale that every claimed scoring term can change the selection. This check fails for the contribution-removed baseline, so it is meaningful. It is a scale-free behavioral property and does not confirm the paper's full-scale headline numbers. |

## Automated checks

Every delivered package runs a battery of automated behavioral checks against the generated code. The counts cover all of them; the details below list only the ones that need your attention.

Results: 12 passed, 4 not checked.

### Checks that could not run

These checks did not produce a verdict. Each line names the artifact, environment capability, or method shape that blocked it.

- **active-learning loop bookkeeping is correct.** loop cell needs unknown name 'F' — extend the seeding table or probe by hand — This finding carries probe ref `al_loop.microharness`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.
- **the selector returns valid, unique picks.** no module-level callable matches the core-set naming convention (name containing 'core_set'/'coreset') — the Stage-1 probes cannot bind — This finding carries probe ref `al_stage1.coreset_construction`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.
- **selections are not stuck in the first cluster.** no module-level callable matches the core-set naming convention (name containing 'core_set'/'coreset') — the Stage-1 probes cannot bind — This finding carries probe ref `al_stage1.coreset_construction`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.
- **the paper's selection parameter has a live effect.** no module-level callable matches the core-set naming convention (name containing 'core_set'/'coreset') — the Stage-1 probes cannot bind — This finding carries probe ref `al_stage1.coreset_construction`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.

## Where the run had trouble

The pipeline's quality gates recorded 1 problem in its own output before delivery:

- Stage 1 - Paper Decomposition & Method Analysis: the `validate_paper_map.py` check rejected the produced output (id prefix consistency check failed (1 error(s)):); a later attempt passed the same `validate_paper_map.py` check, and the step completed.

No unresolved failures remained at delivery.

## Issues To Review

These rows consolidate the issue files. The source files remain available for the full details.

| Source | File | Summary |
|---|---|---|
| Method explanation | `METHOD.md` | produced; no pending explanation markers found; the math checks had nothing to verify in 8 of 8 explanation sections (disclosure only, not a defect) |
| Automatic assumptions | `details/assumptions.md` | 1 assumption entry |
| Deferred review findings | `details/deferred_findings.md` | 3 deferred findings |
| Known delivery limits | `details/KNOWN_ISSUES.md` | not produced |
| Per-element tests | `method/tests/README.md` | 5 of 12 eligible elements covered by a per-element test shown to distinguish correct from broken code; 4 weak tests that could not demonstrate that distinction |
