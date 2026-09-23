# Run Report

Run: `deep-batch-active-learning`

Delivery label: **draft, needs attention before relying on the output**.

Label scale, least to most reliable: explanation only → uncertified — new territory → **draft** → verified. This run sits at 3 of 4.

Machine-readable manifest: `details/final_manifest.json`.

After this report, open: [`notebook.ipynb`](notebook.ipynb) — the tutorial; `method/` — the importable package; [`METHOD.md`](METHOD.md) — the method explained.

## Why this label

2 finding(s) hold this delivery below `verified`:

- **Behavioral check (executed notebook outputs are sane)**: executed metric series 'test_accuracy' moves but is unvetted here (no class count for a chance comparison) — surfaced rather than passed silently
- **Paper-fidelity review, critical finding**: The contract still references `AL-kmeanspp-incremental-distances`, but the matched batch_acquisition taxonomy declares no probe with that name. This leaves the core k-MEANS++ incremental-distance requirement unverifiable through the declared integration surface even though the generated implementati…

10 more item(s) are disclosed for awareness without blocking the label:

- loop cell raised AttributeError: 'list' object has no attribute 'detach' — This finding carries known probe ref `al_loop.microharness`, but declares no callable or element anchor that could reach a methodology obligation. The result is unanchored coverage, not certification evidence.
- in-loop training builds a fresh model each round (fresh-weights behavior is the "fresh model weights are verified behaviorally" check's complementary check) — This finding carries probe ref `al_loop.fresh_retrain`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.
- post-acquisition learning-curve points are retrained/evaluated after the label merge — This finding carries probe ref `al_loop.eval_label_alignment`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.
- the delivered demo config exercises the acquisition loop (initial_labeled=100 < pool_size=12000; budget 500 fits the remaining 11900) — This finding carries probe ref `al_loop.demo_config_reachability`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.
- selection differs from the degenerate nulls (first-k, seeded-random idioms) across 3 trial pools — This finding carries probe ref `claims.contribution_floor`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.
- contract holds (4 unique in-range indices) and the selection responds to model weights — This finding carries probe ref `al_loop.acquisition_contract`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.
- no module-level callable matches the core-set naming convention (name containing 'core_set'/'coreset') — the Stage-1 probes cannot bind — This finding carries probe ref `al_stage1.coreset_construction`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.
- no module-level callable matches the core-set naming convention (name containing 'core_set'/'coreset') — the Stage-1 probes cannot bind — This finding carries probe ref `al_stage1.coreset_construction`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.
- no module-level callable matches the core-set naming convention (name containing 'core_set'/'coreset') — the Stage-1 probes cannot bind — This finding carries probe ref `al_stage1.coreset_construction`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.
- the headline demo outcome was not machine-checked: markers are declared for this family but none matched the headline demo section's executed outputs

## Verification — what we could and could not confirm

This is the claims ledger. The honest default is untested at smoke scale. A claim is marked verified only when a scale-free behavioral check passes and also fails for a baseline with the contribution removed, so the check is meaningful. Where our run fell short, the ledger says so and attributes it to our own implementation, never to the paper.

**Claims checked:** 2 verified at smoke scale (behavioral properties), 0 contradicted, 0 need a fix on our side, 1 untested at this scale.

| Claim | Verdict | What this means |
|---|---|---|
| BADGE robustness across datasets, architectures, and batch sizes | Untested at this scale | We did not verify this claim. It is a scale-bound experiment claim, and the paper-map text does not contain one unambiguous metric value that can be matched to the executed notebook. To verify it: a full-scale run on the paper's dataset, training budget, and seed count. |
| the method selects differently from random acquisition | Verified at smoke scale | We verified at smoke scale that the method selects differently from random acquisition. This check fails for the contribution-removed baseline, so it is meaningful. It is a scale-free behavioral property and does not confirm the paper's full-scale headline numbers. |
| every claimed scoring term can change the selection | Verified at smoke scale | We verified at smoke scale that every claimed scoring term can change the selection. This check fails for the contribution-removed baseline, so it is meaningful. It is a scale-free behavioral property and does not confirm the paper's full-scale headline numbers. |

## Automated checks

Every delivered package runs a battery of automated behavioral checks against the generated code. The counts cover all of them; the details below list only the ones that need your attention.

Results: 11 passed, 1 needs researcher judgment, 4 not checked.

### executed notebook outputs are sane

**What this checks.** executed notebook outputs are sane

**Why it matters.** The notebook is the integration test. Diverged, chance-level, or flat outputs mean that successful execution alone is insufficient.

**Result.** 1 needs researcher judgment.

**Recorded result.** executed metric series 'test_accuracy' moves but is unvetted here (no class count for a chance comparison) — surfaced rather than passed silently

**Evidence.** notebook.ipynb: test_accuracy=['0.731', '0.821', '0.843', '0.852', '0.881', '0.887']

**How to read this.** A numeric series moved, though this check lacks the context needed to decide whether that movement is good. A researcher should compare it with the metric definition in the paper.

### Checks that could not run

These checks did not produce a verdict. Each line names the artifact, environment capability, or method shape that blocked it.

- **active-learning loop bookkeeping is correct.** loop cell raised AttributeError: 'list' object has no attribute 'detach' — This finding carries known probe ref `al_loop.microharness`, but declares no callable or element anchor that could reach a methodology obligation. The result is unanchored coverage, not certification evidence.
- **the selector returns valid, unique picks.** no module-level callable matches the core-set naming convention (name containing 'core_set'/'coreset') — the Stage-1 probes cannot bind — This finding carries probe ref `al_stage1.coreset_construction`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.
- **selections are not stuck in the first cluster.** no module-level callable matches the core-set naming convention (name containing 'core_set'/'coreset') — the Stage-1 probes cannot bind — This finding carries probe ref `al_stage1.coreset_construction`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.
- **the paper's selection parameter has a live effect.** no module-level callable matches the core-set naming convention (name containing 'core_set'/'coreset') — the Stage-1 probes cannot bind — This finding carries probe ref `al_stage1.coreset_construction`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.

## Where the run had trouble

The pipeline's quality gates recorded 4 problems in its own output before delivery:

- Stage 1 - Paper Decomposition & Method Analysis: the `validate_method_spec.py --strict` check rejected the produced output (strict: methodology verification-probe cross-check failed:); a later attempt passed the same `validate_method_spec.py --strict` check, and the step completed.
- Stage 2.b - Architecture & Training: the `validate_architecture_coder_output.py` check rejected the produced output (.pipeline/arch_contract.json failed schema validation: 12 validation errors for ArchContractV2); a later attempt passed the same `validate_architecture_coder_output.py` check, and the step completed.
- Stage 2.d - Package Finalization: the `validate_arch_contract.py` check rejected the produced output (FAIL: 4 arch_contract validation error(s):); a later attempt passed the same `validate_arch_contract.py` check, and the step completed.
- Stage 3.a - Notebook Authoring: the `validate_notebook_output.py` check rejected the produced output (notebook missing section `setup_pieces` with heading '## 3. The setup pieces'. Taxonomy notebook_layout declares this section.); a later attempt passed the same `validate_notebook_output.py` check, and the step completed.

No unresolved failures remained at delivery.

## Issues To Review

These rows consolidate the issue files. The source files remain available for the full details.

| Source | File | Summary |
|---|---|---|
| Method explanation | `METHOD.md` | produced; no pending explanation markers found; the math checks had nothing to verify in 3 of 3 explanation sections (disclosure only, not a defect) |
| Automatic assumptions | `details/assumptions.md` | 2 assumption entries; 1 needs researcher attention |
| Deferred review findings | `details/deferred_findings.md` | 6 deferred findings; 1 at the important tier, unresolved — read it before relying on the affected code |
| Known delivery limits | `details/KNOWN_ISSUES.md` | not produced |
| Per-element tests | `method/tests/README.md` | 2 of 5 eligible elements covered by a per-element test shown to distinguish correct from broken code; 1 weak test that could not demonstrate that distinction; 1 generated test could not be verified and did not ship |
