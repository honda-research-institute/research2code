# Time-Series Forecasting — long-form priors

> Long-form expertise prose for the `time_series_forecasting` taxonomy node
> (`TE-TSF/time_series_forecasting`). Referenced from the node via
> `scaffold_hints.content_file`; the node itself keeps only terse, structured
> priors/checks (schema §3.5). This file biases generation but never gates —
> nothing here produces a `verified` label.
>
> Authored 2026-08-06 (R2C-070) directly from five rolls of pdfgnn audit
> evidence (`audit/reports/2026-08-04-pdfgnn-delivery-quality-review.md`,
> `.../2026-08-05-pdfgnn-day-roll-quality-review.md`,
> `.../2026-08-05-pdfgnn-night3-quality-review.md`,
> `.../2026-08-06-pdfgnn-loop2-quality-review.md`) and the two run-authored
> provisional packs those rolls produced. Every failure shape below was
> observed live, most of them more than once.

## What the family is

Trained models that forecast future values of time series from historical
context. The modern shape is a *global* model: one network trained across many
related series (articles, sensors, locations), often with relational structure
between series (a similarity graph, cross-series attention), and typically with
a *probabilistic* output — distribution parameters, quantiles, or Monte Carlo
samples — because the downstream use (inventory, staffing, capacity) needs
uncertainty, not just a point estimate.

Two parameters are structural, never tuning knobs: the context length P (how
many historical steps the model conditions on) and the forecast horizon K (how
many future steps it predicts). Together with any validation window they fix
the minimum usable time-axis length; a demo dataset that cannot cover
P + K + validation arithmetic cannot exercise the method at all.

## The protocol is the method

For this family, most catastrophic failures are protocol failures, not model
failures. The train-on-history / evaluate-on-held-out-window protocol has
several places to break silently, and every one of them was observed in the
pdfgnn rolls:

1. **Chronology must survive aggregation.** Aggregating daily data to weeks by
   `isocalendar().week` alone folds every year onto one 52-bin calendar —
   January 2017 + 2018 + 2019 become one bucket, and the "time series" is a
   seasonal aggregate with a fabricated axis. Key by year AND period
   (`year*100 + week`), and verify the axis is strictly chronological.
2. **Held out must also mean carrying signal.** A window disjoint from
   training is necessary, not sufficient. Public competition datasets
   (the Kaggle layout) deliberately withhold the target over their final
   period — that period is the competition's own prediction window — so the
   last K steps of the nominal axis can hold no ground truth whatsoever.
   Measure the target column's live extent (the last step that actually
   carries values) and evaluate on the last window that has data.
3. **Leakage has more than one shape.** Observed variants: the evaluation
   window inside the training range; the training target injected into the
   decoder's own input at the forecast positions (train/serve skew — training
   sees the answer, inference sees zeros); and a validation window computed
   with a double-subtracted offset that lands it entirely inside the training
   region, so early stopping selects on training data. Check the executed
   index ranges, not the prose.
4. **Missing values are a policy decision, made at the aggregation seam.**
   The loader maps blank cells to NaN (documented, correct). What happens
   next must be explicit: an aggregation that `+=`s NaN poisons whole
   buckets; a `np.nan_to_num` buried inside the model converts a loud,
   routable crash into a silent NaN metric and fabricated history. Treating
   an unrecorded observation as zero is a modeling assumption — state it
   where the aggregation happens.

## Mechanism fidelity: the three recurring substitutions

These are the family's must-replicate mechanisms and the exact ways they were
silently substituted across rolls:

- **Autoregressive decoding (DeepAR lineage).** The defining property is
  ancestral sampling: the value drawn at step k conditions the state at step
  k+1, so K-step trajectories carry path uncertainty. Observed substitutions:
  feeding the predicted mean forward instead of a sample; a rollout whose
  return signature aliased the mean as the "samples"; and the rollout deleted
  outright, one forward pass emitting K independent marginals. All three
  understate cumulative-horizon uncertainty — the quantity the probabilistic
  claim exists for.
- **Per-series scale handling.** DeepAR-lineage implementations normalize
  targets per series by default. Without it, targets in the hundreds push the
  learned mean toward zero while the scale parameter inflates: the loss falls,
  every surface signal reads healthy, and the delivered forecasts are
  statistically indistinguishable from predicting zero. The scale state is fit
  only on the declared fitting range and keyed by typed stable entity identity.
  The declared `train_model` entry calls the fixed
  `method.target_scaling.fit_target_scaling_state` and `transform_targets`
  helpers with its schema-2 `entity_ids`, `targets`, and explicit
  `fitting_range` inputs, and returns the fitted state under the exact
  `target_scaling_state` mapping key. For an induced relational batch,
  transform the coherently prepared target subset using its prepared output
  entity ids; never rejoin by a transient local row position.
  The public `forecast` call receives both the exact entity ids for its series
  axis and that state; it never joins scales by row position. Before returning,
  it uses the fixed `method.target_scaling.inverse_forecast_output` helper to
  restore original target units for every target-valued field: location and
  samples by scale, distribution scale by absolute scale, and variance by
  squared scale. Dimensionless shape parameters stay unchanged. Absence or
  partial inversion is a mechanism deviation, not a hygiene nit.
- **Real sampling in the returned distribution.** "Samples" must be genuine
  draws with spread consistent with the scale parameter. The observed defect:
  conditional means recorded as Monte Carlo trajectories, with trajectory
  spread four orders of magnitude below sigma — quantile bands built from them
  report near-certainty where the model is highly uncertain.

Graph-augmented members (GraphDeepAR and kin) add two more: the relational
encoder's output must actually reach the decoder on the executed data path
(a constructed-but-never-called projection layer is a failure, not dead
weight), and graph construction details the paper states (self-loops,
neighborhood sampling caps) are part of the mechanism — deleting self-loops
from a similarity graph severs every node's own history from its embedding
and zeroes isolated nodes entirely.

## Training-loop traps specific to this family

- **The time stride is not the batch size.** Series-batch size and
  time-window stride are different axes. `t += batch_size` visited 4 of ~127
  windows per epoch on a real roll — 88% of the history never became a
  training target, and raising batch size for throughput silently cuts
  training data further.
- **Synchronized batching can hold vacuously.** "All series in a batch share
  one time window" is trivially true when only one window is ever built.
  The property matters across many windows.
- **Return structured training evidence.** "Training complete." with no
  numbers shipped on two consecutive rolls and made training unverifiable.
  The declared `train_model` entry uses the fixed
  `method.training_history.record_training_history` helper and returns its
  canonical record under the exact `training_history` mapping key. The record
  binds finite, strictly ordered loss observations to the fitting range,
  stable target-scaling state, seed, configuration id, model id, and
  checkpoint id. A performed single-holdout selection additionally binds its
  metric observations to the exact selection range and deterministically
  selects the best, then earliest, checkpoint. A run with no selection says
  `not_performed`; it does not invent validation evidence.

## Evaluation discipline

Compute the paper's metrics with explicit NaN discipline — assert finiteness
rather than letting NaN propagate. Always print two naive baselines beside the
model: predict-zero and repeat-last-value (persistence). A forecaster that
cannot beat predict-zero has learned nothing; one that loses to persistence
has not cleared the family's minimum-competence comparator. Demonstrated skill
therefore requires a valid evaluation and wins against both under the node's
structured `demo_skill` contract.

The notebook emits exactly one `R2C_DEMO_EVALUATION_JSON: <object>` line from
the executed actuals, model predictions, and pre-window history. The object
uses literal `schema_version: "2.0.0"` and carries one exact
`evaluation_protocol_role` copied from
`comparison.evaluation_protocol.quantities`, plus the target root, fitted
model identity, evaluated checkpoint identity, configuration identity, metric
id, units, aggregation, row and evaluation positions,
actuals, finite mask, model predictions, the recorded model metric,
deterministic identities for the actual values and mask, and the corresponding
predictions, recorded metric, and coidentity fields for both comparators. Use
`test_span` only when the executed evaluation is the typed protocol's test-span
role. Otherwise carry the exact role declared for that evaluation; never infer
one from window length, row count, nearby numeric values, or prose. The
pipeline recomputes every recorded metric and requires numerical agreement.
The identity recipe is exact: prefix `sha256:` to SHA-256 over UTF-8 compact
JSON with sorted keys (`separators=(",", ":")`, `ensure_ascii=False`,
`allow_nan=False`). `actual_values_id` hashes `{actuals, row_ids, units}` and
`finite_mask_id` hashes `{finite_mask, row_ids}`; masked missing values are
JSON null rather than non-finite JSON tokens.
`finite_mask` is exactly the per-row finiteness of `actuals`, never a
model-selected subset. Every repeat-last source is the final position
immediately before the complete evaluation window, so a later horizon cannot
reuse an earlier held-out actual or select an arbitrarily older weak value.
Row and position equality is canonical-JSON typed;
integers do not alias floating identifiers.
The same executed headline section also emits exactly one
`R2C_TARGET_SCALING_STATE_JSON: <object>` line containing the canonical
JSON state returned by the declared fitting call. Use compact sorted-key JSON
with `allow_nan=False`; generated code must not write the durable artifact.
The pipeline rebinds this reported state to its independently minted fitting
lineage and alone writes `.pipeline/target_scaling_state.json`.

The training subsection emits exactly one
`R2C_TRAINING_HISTORY_JSON: <object>` line containing the canonical record
returned by the fixed training-history helper. Generated code never writes
the durable artifact. The pipeline rebinds model, checkpoint, configuration,
fitting and selection ranges, and `target_scaling_state_id` to the executed
evaluation plus its independently owned split and scaling evidence before it
writes `.pipeline/training_history.json`. Loss-descent checks and report
summaries consume that structured artifact only; notebook prose is not an
evidence fallback.

For target scaling, the `R2C_DEMO_EVALUATION_JSON` object additionally carries
one `target_scaling_proof` mapping with literal `schema_version: "1.0.0"`, the
typed `entity_ids` and their exact `entity_id_root`, and both scaled and
original-facing arrays for actuals and model predictions. Each side declares
its closed output role (`location` for target values, or another exact role
when justified) and integer entity axis. Preserve the real array shape and
entity order: do not flatten, reshape, or infer an axis to satisfy the record.
The pipeline applies the fixed inverse helper by typed identity and requires
numerical agreement with the original-facing arrays before those metrics can
count as original-unit evidence.
Repeat-last additionally records one value in `input_values` and one position
in `input_positions` per row, strictly before the corresponding evaluation
position. The pipeline matches `target_root`, `model_id`, and the evaluation
positions to the trusted fitting-to-reported-evaluation relation, recomputes
the metrics from those objects, and joins its split-validity receipt; the
notebook never self-asserts evaluation validity. A final
`Demo verdict: PASS — ...` or `Demo verdict: FAIL — ...` line is useful
presentation derived from the same objects, but is never evidence authority
and cannot override the structured result.

## Demo-scale calibration

Preserve `context_length`, `forecast_horizon`, and the distribution family —
they are the method's identity. Sacrifice epochs, series count, and
embedding/hidden dims first. A demo that trains in ~5 minutes on tens of
series over the paper's real (subsampled) time axis demonstrates the
mechanism; it does not reproduce benchmarks, and the surfaces should say so.
When bundling real data, subsample by entities (series), never by leading rows
of a date-sorted table — the time axis must survive whole, and its live extent
(where the target actually ends) must be stated.
