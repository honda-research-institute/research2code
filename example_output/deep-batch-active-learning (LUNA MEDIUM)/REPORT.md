# Run Report

Run: `deep-batch-active-learning`

Delivery label: **draft, needs attention before relying on the output**.

Label scale, least to most reliable: explanation only → uncertified — new territory → **draft** → verified. This run sits at 3 of 4.

Machine-readable manifest: `details/final_manifest.json`.

After this report, open: [`notebook.ipynb`](notebook.ipynb) — the tutorial; `method/` — the importable package; [`METHOD.md`](METHOD.md) — the method explained.

## Why this label

4 finding(s) hold this delivery below `verified`:

- **Paper-fidelity review, important finding**: The paper-map training protocol specifies the Adam variant of SGD, while the structured method spec still records the optimizer as plain SGD. The generated training implementation uses Adam, so the paper-truth contract and the produced package disagree.
- **Paper-fidelity review, important finding**: The notebook says it uses the package's SGD implementation and that the training function uses the supplied SGD protocol, but method/training.py constructs torch.optim.Adam. This gives the researcher the wrong account of the runnable training protocol even though the code follows the paper's Adam wo…
- **Paper-fidelity review, important finding**: The module names the flattened MLP substitution but does not state that changing the paper's image backbones to an MLP changes the learned representation and inductive bias, so acquisition behavior is not a reproduction of the paper's CNN/ResNet benchmark claims. The existing hook warning does not c…
- **Paper-fidelity review, important finding**: The pool-size provenance says that diversity baselines are meaningfully tested at this scale, but this single-method package intentionally implements and runs no baselines. The claim conflicts with the scope contract and can make the smoke setting sound like a comparison experiment.

12 more item(s) are disclosed for awareness without blocking the label:

- batch_size: batch_size is source=paper but paper_section 'Experimental setup' does not look like a paper locator
- initial_labeled: initial_labeled is source=paper but paper_section 'Experimental setup' does not look like a paper locator
- loop bookkeeping invariants hold over 3 adversarial rounds — This finding carries probe ref `al_loop.microharness`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.
- 4 training calls used 4 distinct fresh models — This finding carries probe ref `al_loop.microharness`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.
- in-loop training builds a fresh model each round (fresh-weights behavior is the "fresh model weights are verified behaviorally" check's complementary check) — This finding carries probe ref `al_loop.fresh_retrain`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.
- post-acquisition learning-curve points are retrained/evaluated after the label merge — This finding carries probe ref `al_loop.eval_label_alignment`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.
- the delivered demo config exercises the acquisition loop (initial_labeled=100 < pool_size=12000; budget 500 fits the remaining 11900) — This finding carries probe ref `al_loop.demo_config_reachability`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.
- selection differs from the degenerate nulls (first-k, seeded-random idioms) across 3 trial pools — This finding carries probe ref `claims.contribution_floor`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.
- contract holds (4 unique in-range indices) and the selection responds to model weights — This finding carries probe ref `al_loop.acquisition_contract`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.
- no module-level callable matches the core-set naming convention (name containing 'core_set'/'coreset') — the Stage-1 probes cannot bind — This finding carries probe ref `al_stage1.coreset_construction`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.
- no module-level callable matches the core-set naming convention (name containing 'core_set'/'coreset') — the Stage-1 probes cannot bind — This finding carries probe ref `al_stage1.coreset_construction`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.
- no module-level callable matches the core-set naming convention (name containing 'core_set'/'coreset') — the Stage-1 probes cannot bind — This finding carries probe ref `al_stage1.coreset_construction`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.

## Verification — what we could and could not confirm

This is the claims ledger. The honest default is untested at smoke scale. A claim is marked verified only when a scale-free behavioral check passes and also fails for a baseline with the contribution removed, so the check is meaningful. Where our run fell short, the ledger says so and attributes it to our own implementation, never to the paper.

**Claims checked:** 0 verified at smoke scale, 0 contradicted, 2 need a fix on our side, 5 untested at this scale.

| Claim | Verdict | What this means |
|---|---|---|
| the method selects differently from random acquisition | Needs a fix on our side | Our run did not support the method selects differently from random acquisition. This is most likely our implementation, not a problem with the paper: an unresolved important fidelity finding on this path (method/model.py). We are flagging it for a fix and do not read it as the paper being wrong. |
| every claimed scoring term can change the selection | Needs a fix on our side | Our run did not support every claimed scoring term can change the selection. This is most likely our implementation, not a problem with the paper: an unresolved important fidelity finding on this path (method/model.py). We are flagging it for a fix and do not read it as the paper being wrong. |
| no point is acquired twice across rounds | Untested at this scale | No point is acquired twice across rounds holds, but a trivial baseline (such as random acquisition) also satisfies it, so on its own it does not confirm the paper's contribution. To verify it: an ablation that re-selects, since uniqueness alone is met by random acquisition too. |
| the labeled set grows by the batch size each round | Untested at this scale | The labeled set grows by the batch size each round holds, but a trivial baseline (such as random acquisition) also satisfies it, so on its own it does not confirm the paper's contribution. To verify it: a baseline loop also grows the labeled set, so growth alone is not evidence of the contribution. |
| BADGE robustness across settings | Untested at this scale | We did not verify this claim. It is a scale-bound experiment claim, and the paper-map text does not contain one unambiguous metric value that can be matched to the executed notebook. To verify it: a full-scale run on the paper's dataset, training budget, and seed count. |
| Gradient-space uncertainty and diversity diagnostics | Untested at this scale | We did not verify this claim. It is a scale-bound experiment claim, and the paper-map text does not contain one unambiguous metric value that can be matched to the executed notebook. To verify it: a full-scale run on the paper's dataset, training budget, and seed count. |
| k-MEANS++ sampling efficiency | Untested at this scale | We did not verify this claim. It is a scale-bound experiment claim, and the paper-map text does not contain one unambiguous metric value that can be matched to the executed notebook. To verify it: a full-scale run on the paper's dataset, training budget, and seed count. |

## Automated checks

Every delivered package runs a battery of automated behavioral checks against the generated code. The counts cover all of them; the details below list only the ones that need your attention.

Results: 14 passed, 2 advisory, 3 not checked.

### Checks that could not run

These checks did not produce a verdict. Each line names the artifact, environment capability, or method shape that blocked it.

- **the selector returns valid, unique picks.** no module-level callable matches the core-set naming convention (name containing 'core_set'/'coreset') — the Stage-1 probes cannot bind — This finding carries probe ref `al_stage1.coreset_construction`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.
- **selections are not stuck in the first cluster.** no module-level callable matches the core-set naming convention (name containing 'core_set'/'coreset') — the Stage-1 probes cannot bind — This finding carries probe ref `al_stage1.coreset_construction`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.
- **the paper's selection parameter has a live effect.** no module-level callable matches the core-set naming convention (name containing 'core_set'/'coreset') — the Stage-1 probes cannot bind — This finding carries probe ref `al_stage1.coreset_construction`, but the methodology contract declares no such verification ref. The finding remains unbound; this is a probe-side or analyzer-side coverage gap, not evidence about the paper.

### Advisory checks

These checks surfaced a risk without failing the delivery.

- **paper-sourced parameters name a paper location (2 instances).** batch_size: batch_size is source=paper but paper_section 'Experimental setup' does not look like a paper locator initial_labeled: initial_labeled is source=paper but paper_section 'Experimental setup' does not look like a paper locator

## Where the run had trouble

The pipeline's quality gates recorded 3 problems in its own output before delivery:

- Stage 1 - Paper Decomposition & Method Analysis: the `validate_method_spec.py --strict` check rejected the produced output (schema validation failed (1 error(s)):); a later attempt passed the same `validate_method_spec.py --strict` check, and the step completed.
- Stage 1 - Paper Decomposition & Method Analysis: the `validate_method_spec.py --strict` check rejected the produced output (strict: paper_map cross-check failed:); a later attempt passed the same `validate_method_spec.py --strict` check, and the step completed.
- Stage 3.c - Smoke Execution: the notebook executed but showed no learning signal; the step later completed, but no passing re-check of this learning-signal check was recorded.

No stage remained halted or degraded at delivery, but one or more failed checks lacked a recorded passing re-check.

## Issues To Review

These rows consolidate the issue files. The source files remain available for the full details.

| Source | File | Summary |
|---|---|---|
| Method explanation | `METHOD.md` | produced; no pending explanation markers found; the math checks had nothing to verify in 7 of 7 explanation sections (disclosure only, not a defect) |
| Automatic assumptions | `details/assumptions.md` | 1 assumption entry |
| Deferred review findings | `details/deferred_findings.md` | 10 deferred findings; 6 at the important tier, unresolved — read these before relying on the affected code |
| Known delivery limits | `details/KNOWN_ISSUES.md` | not produced |
| Per-element tests | `method/tests/README.md` | 1 of 7 eligible elements covered by a per-element test shown to distinguish correct from broken code; 5 generated tests could not be verified and did not ship |
| Model-call recoveries | `.pipeline/run_events.jsonl` | 1 model call failed in transport (hang or outage) and was recovered by an automatic retry — disclosure only; the failure stays on the run's dispatch record |
