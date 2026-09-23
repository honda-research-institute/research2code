# pdfgnn split-lineage reductions, 2026-08-07

These files are evidence reductions for R2C-077 piece 3. They preserve the
smallest source shapes needed to design and enforce source-ordered,
interprocedural range lineage without treating the generated run as the fix
target.

The fixtures are ordinary Python so AST-based checks can read them directly.
`current_notebook.py` and `current_trainer.py` are the known-bad pair.
`current_notebook_disjoint.py` and `current_trainer_disjoint.py` keep the same
syntax while making the caller-to-callee boundary genuinely disjoint.
`current_notebook_generated.py` reduces the delivered notebook's opaque bundle
construction, dead-tail trim, resolved config reads, model-builder call, whole
matrix trainer call, and masked reported evaluation. It pins the producer-seam
adapter without copying protocol truth into the analyzer.
`archived_notebook_generated.py` reduces the `_11` notebook's
`last_valid_idx + 1` live-end convention, `nan_to_num` and tensor wrappers,
caller-supplied `val_split_start`, trainer call, and masked metric. Together
with `archived_trainer.py`, it yields fitting `[10,1032)`, selection
`[1029,1033)`, and reported evaluation `[1029,1033)` on the 1,033-step bundle.
`current_notebook_generated_disjoint.py` is the real-validator known-good. It
passes a pre-evaluation slice into `current_trainer_disjoint.py`, leaving
fitting `[10,744)`, selection `[744,826)`, conditioning `[816,826)`, and
reported evaluation `[826,830)` with every target-role pair disjoint.

## Provenance

### Archived trainer leak

`archived_trainer.py` reduces:

- `r2c_runs/probabilistic-demand-forecasting-with-graph-neural-networks_11/method/training.py:219-220`
  for the training-window count.
- The same file at lines 267-292 for `randperm`, the `t_start` and `t_end`
  aliases, the `min()`-bounded target end, and fitting supervision.
- The same file at lines 331-378 for the renamed validation target, checkpoint
  selection, and best-state restoration.

Let `P=context_length`, `K=forecast_horizon`, and
`V=val_split_start`. Under the archived call's normal feasibility condition
`T >= V + K`:

- `window_indices = randperm(V - P)` ranges over `[0, V-P)`.
- `t_end` ranges from `P` through `V-1`.
- Fitting targets have union `[P, V+K-1)`.
- Model-selection targets are `[V, V+K)`.
- Their overlap is `[V, V+K-1)`, with length `K-1`.
- At `K=4`, the overlap has three steps.
- At `K=1`, the fitting range ends at `V` and the selection range starts at
  `V`, so this archived-case control is disjoint.

The selection binding is deliberately named `val_demand_target`, while the
fitting binding remains `demand_target`. A pure rename must not change the
range relationship.

### Current notebook-to-trainer leak

`current_notebook.py` reduces the delivered notebook's:

- Cell 35, id `d07a5567`, mirrored in
  `.pipeline/notebook_draft.py:345-372`, for the 80-percent boundary and the
  whole-matrix trainer call.
- Cell 38, id `8751942d`, mirrored at draft lines 381-397, for inference
  conditioning and the reported actual window.
- Cell 41, id `1c1ffb23`, mirrored at draft lines 453-470, for the boolean-mask
  metric lineage.

`current_trainer.py` reduces:

- The current `method/training.py:154-164` private split calculation.
- Lines 187-228 for the mutable fitting loop.
- Lines 234-282 for the second `t` definition, full-axis model selection,
  checkpoint choice, and returned scored model.

Use the concrete facts recorded by the delivered notebook:

```text
time extent T = 1033
context P = 10
horizon K = 4
val_split = 0.1
time_stride = 1
```

The expected half-open ranges are:

| Consumer | Expected demand-target range |
| --- | --- |
| Fitting supervision | `[10, 930)` |
| Model selection | `[10, 1033)` |
| Inference conditioning | `[816, 826)` |
| Reported evaluation | nonempty boolean subset of `[826, 830)` |

The internal boundary is
`1033 - max(int(1033 * 0.1), 10 + 4) = 930`. The notebook boundary is
`int(1033 * 0.8) = 826`. Fitting and model selection therefore each overlap
the complete reported interval `[826, 830)`. Fitting and model selection also
overlap on `[10, 930)`. Inference conditioning ends at 826, so its overlap with
reported evaluation is empty.

The delivered cell output recorded 120 finite points, exactly 30 entities
times four time steps. Static lineage must still preserve a conservative
subset marker rather than depending on that executed output:

```text
actual
  -> demand_matrix[:, 826:830]
finite_mask
  -> elementwise predicate aligned with actual
actual_finite
  -> subset(root=demand_matrix, envelope=[826,830), mask=finite_mask)
```

The metric branch establishes that `actual_finite` is nonempty. Renaming
`actual_finite`, `finite_mask`, either scoped `train_end`, `batch_target`, or
`val_target` must not change these relationships.

Model selection is identified by a loss flowing into a branch that snapshots
or restores model state. It does not depend on `val_loss`, `best_val_loss`, or
another local spelling. Both the archived and current rename controls replace
the loss, target, and comparison-bound names while preserving the same role
ranges.

## Separately disjoint current-case control

`current_notebook_disjoint.py` and `current_trainer_disjoint.py` keep `K=4`,
mutable loops, tensor conversion, row permutation, and the boolean-mask metric
shape. They change the trainer's consumed ranges:

- The notebook passes `demand_matrix[:, :train_end]` to the trainer, giving
  the trainer formal an available extent of 826.
- Its private selection width is `max(int(826 * 0.1), 14) = 82`.
- Fitting ends at `826 - 82 = 744`.
- The selection loop begins at 744 instead of restarting at `P`.

The expected ranges are therefore:

| Consumer | Expected demand-target range |
| --- | --- |
| Fitting supervision | `[10, 744)` |
| Model selection | `[744, 826)` |
| Inference conditioning | `[816, 826)` |
| Reported evaluation | nonempty boolean subset of `[826, 830)` |

Fitting targets, selection targets, and reported-evaluation targets are
pairwise disjoint. The inference context ends exactly where evaluation begins.
Selection context may read earlier observations, including observations from
the fitting period. That is conditioning history, not a selection target, so a
range summary must retain both consumer role and read kind rather than unioning
all reads under one label.

## Delivered-fleet noise measurement

The report-only stress sweep on 2026-08-07 assigned a conservative 100-step
envelope to target-like formals in every top-level `train`, `fit`, or `retrain`
function under the delivered fleet. It measured 27 functions across 22 run
directories, 18 role observations, six concrete relations, two structured
unresolved reads, and zero analysis errors. All six relations belonged to
pdfgnn variants. No non-forecasting family produced a concrete relation.

(The `delivered_fleet_snapshot.json` copy of those counts was removed in
Track A Block A1 — its only reader was a constant-versus-copy assert, and it
was already 2x stale. The prose above is the record.) `fleet_noise_trainers.py`
preserves two
executable non-forecasting controls derived from the delivered classifier
shapes: training-set accuracy remains a trainer-local progress metric, and
checkpoint selection on a distinct target root creates no range relation.

## Required summary properties

- Definitions are processed in source order and are scoped. In particular,
  the notebook boundary and trainer boundary are different definitions even
  when they share a spelling, and the fitting and selection loops each define
  their own version of `t`.
- Tensor wrappers, aliases, row gathers, permutations, and boolean masks keep
  the original root and the unaffected protocol-axis envelope.
- Trainer summaries identify fitting and selection reads, their formal-axis
  bounds, and the fact that the returned or mutated model is controlled by the
  selection result.
- Call-site instantiation maps formal roots and bounds onto the caller's root,
  slice, scalar arguments, and model attributes.
- Forecast outputs retain the identity of the scored model, so reported actual
  ranges are compared only with the fitting and selection ranges that produced
  that model.
- Metric discovery follows values that reach public trainer/forecast returns
  and explicit evaluation consumers. Pure prediction helpers, conditioning-
  only calls, and trainer-local progress metrics remain outside reported
  evaluation; helper-call bindings are scoped per invocation.
- Producer adaptation deduplicates only equivalent observations from the same
  trainer, formal root, scored model, and caller offset. A second trainer,
  tensor root, model receiver, or sliced-formal binding remains independent.
- Control-flow interpretation preserves normal, return, break, continue,
  explicit-raise, and implicit-raise completions. Local-unbound paths,
  definition-time inputs, ordered assignment/with/delete targets, literal loop
  elements, typed unpack failures, eager comprehensions, and class-local versus
  enclosing bindings are all processed in runtime order.
- Affine and exact iteration domains retain exact bounds when justified;
  sparse indices, early exits, joins, and masks become support envelopes;
  unsupported reads that reach a relevant target consumer remain symbolic and
  unresolved rather than being silently treated as clean.

## Report and enforcement boundary retained by this block

- A boolean-masked range is a conservative support envelope. Relations that
  include it carry `certainty=support`; they do not claim that every element in
  the envelope survives the mask.
- A trainer summary emits fitting and model-selection reads. Reported metrics
  are rooted in the calling analysis surface, which prevents trainer-local
  progress metrics from being mislabeled as researcher-facing evaluation.
- Concrete loader outputs, tensor extents, and model attributes enter as named
  reduction facts. Opaque syntax that reaches a consumer remains a structured
  unresolved row with its role, root, and model identity.
- `scripts/eval_split_lineage.py` remains the report generator. The separate
  enforcement policy compares target reads for fitting, model selection, and
  reported evaluation for the same root and scored model.
- Exact overlaps are definite findings. Overlaps that include a masked support
  envelope fail because disjointness is unresolved, while the message preserves
  the weaker certainty and does not claim an exact selected element set.
- The architecture seam emits only relationships resolved from trainer-local
  facts. The notebook seam owns caller-bound facts and fails closed only for an
  unresolved target read that reaches a relevant temporal consumer.
- The method seam projects training and explicit-evaluation consumers back to
  their `method.py` call sites, so a composed train-and-evaluate function stays
  in the method-coder fix loop while trainer-local progress metrics stay out of
  reported evaluation.
- Caller-resolved relationships whose consumers both live in
  `method/training.py` retain an explicit architecture-coder routing hint at
  the notebook seam. They are not presented as notebook-owned repairs.
- Conditioning history remains a separate read kind. The pre-existing notebook
  range arm continues to own conditioning-versus-evaluation enforcement.
- The current cases contain one trainer call per scored model. Sequential
  retraining of one binding needs model-version and event-order semantics
  before enforcement and is intentionally outside this reduction block.
- Combined mask-and-time indexing, rolling-origin evaluation, final refitting,
  row-set lineage, and a runtime split ledger remain later concrete controls.
  This block does not declare a partition plan or paper-protocol value.
