# Active Learning — long-form priors

> Long-form expertise prose for the `active_learning` taxonomy node
> (`TE-TS/active_learning`). Referenced from the node via
> `scaffold_hints.content_file`; the node itself keeps only terse, structured
> priors/checks (schema §3.5, Open Q D). This file biases generation but never
> gates — nothing here produces a `verified` label.
>
> Decomposed in Phase 1 of the
> [taxonomy migration](../../recentering/taxonomy-migration-implementation-plan.md)
> from the retired active-learning guides. This file is the migration target for
> their *human-readable* reference prose.

## What active learning is

Active learning addresses settings where labeled data is expensive but unlabeled
data is plentiful. An *acquisition function* selects which unlabeled examples to
label next, given a labeled set and a current model. The goal: reach a target
test accuracy with as few labels as possible — equivalently, maximize accuracy
at a fixed label budget.

The canonical loop:

1. Train a model on the current labeled set.
2. Score remaining unlabeled examples with an acquisition function.
3. Select top-k (or sample) examples to label.
4. Add to the labeled set; repeat from 1.

Standard scope: pool-based (full unlabeled pool available) and supervised
classification. Stream-based and structured-prediction settings exist but
pool-based classification dominates the literature.

## Sub-variants

- **`batch_acquisition`** — select batches of B > 1 examples per round. The
  modern deep-learning default. The interesting tension (diversity vs.
  uncertainty) only manifests at B > 1: naïve top-K-uncertainty picks
  near-duplicates and wastes labels. BADGE/CORESET/BatchBALD exist to fix this.
  `batch_size` is part of the algorithmic regime, not a scale knob.
- **`bayesian`** — estimate epistemic uncertainty via posterior approximation
  (MC dropout, ensembles, SVI) and select for expected information gain (BALD,
  BatchBALD, GBALD). The Bayesian framing changes the implementation: the model
  exposes posterior *samples* (T stochastic forward passes, T≈10–2000), so
  acquisition scoring is multi-pass and dropout must stay **active at inference**
  (`model.train()` before MC sampling). A Bayesian method that also batches
  (GBALD, BatchBALD) still classifies as `bayesian` — the Bayesian aspect is the
  load-bearing identity; the batch is mechanical.

Distinguishing Bayesian from non-Bayesian uncertainty (the most common
spec-writing mistake): if the method needs T > 1 stochastic forward passes per
example, it is `bayesian`; a single-pass Confidence/Margin/Entropy on a
deterministic softmax is plain uncertainty sampling.

## Architecture conventions

- **Tabular / OpenML**: 2-layer MLP with ReLU, hidden_dim 256–1024.
- **Image**: ResNet-18, VGG-11, or similar CNNs. **MLP on flattened image data
  is a known compromise** — works mechanically but doesn't reflect a paper that
  specifies a CNN. Acceptable for demos; must be flagged in "Known departures".
- **Text**: small transformer or LSTM, depending on the paper's era.
- Bayesian AL additionally **requires dropout layers active at inference**;
  dropout rate (typically 0.25–0.5) is an uncertainty-calibration knob, not a
  free hyperparameter.

If the paper specifies an architecture, the generated `model.py` must use it or
flag the substitution in the notebook's "what this notebook does NOT do" block.

## Standard datasets

- **Image**: CIFAR-10, CIFAR-100, SVHN, MNIST (CIFAR/SVHN are the modern
  workhorses; ImageNet is rare due to compute cost).
- **Tabular**: OpenML benchmarks (e.g. `letter`, `vehicle`, `phoneme`, `magic`).
- **Text**: AG News, IMDB, 20-newsgroups when present (uncommon in core AL).

## Benchmark selection for multi-benchmark papers

Modern AL papers report across several benchmarks; the spec's `data_setup`
captures *one*. **Prefer the smallest standard benchmark** (MNIST < OpenML
tabular < SVHN < CIFAR-10 < CIFAR-100 < ImageNet): the calibrator scales the
paper's benchmark down to demo runtime, and starting small needs the least
aggressive scaling, keeping the demo closest to the paper's own setup. Capture
the choice in `data_setup.benchmark_name`. All four numeric fields
(`initial_labeled`, `batch_size`, `total_budget`, `num_rounds`) must come from
the *same* benchmark configuration — mixing them across benchmarks produces an
incoherent spec. Override the smallest-benchmark preference only when the
smallest benchmark is clearly a sanity check rather than a headline result.

## Evaluation conventions

- **Learning curves** (x = labels, y = test accuracy) — the standard primary
  visual; plot all methods on one chart.
- **AUC of the learning curve** ("anytime performance") — captures sample
  efficiency throughout, not just at the budget cutoff.
- **Final accuracy at fixed budget** — simplest but most noise-fragile; should
  accompany a learning curve, not replace it.
- **Pairwise penalty matrices** (BADGE-style) — robust to single-cell noise.

## Demo-scale conventions

R2C pipelines run at demo scale; some compromises are acceptable, some are not:

- **Small pool** (~1–10% of paper) — OK *if the pool/budget ratio is preserved*.
  If `total_budget / pool_size` exceeds the ratio floor, the pool exhausts and
  all methods converge (the diversity signal degenerates). The floor is
  paradigm-specific: 0.2 for general AL, **0.05 for batch acquisition** (BADGE's
  diversity advantage needs a much larger pool than budget), **0.4 for Bayesian**
  (per-example posterior scoring lacks the diversity-vs-pool tension).
- **Single seed** — acceptable for *mechanical verification* (does it run?), never
  for a ranking claim. Inter-method differences are routinely smaller than
  inter-seed variance; a ranking claim needs ≥3 seeds.
- **Substitute a simpler architecture** (MLP for ResNet) — acceptable; flag it.
- **Reduce `num_rounds`, NOT `batch_size`** — batch size is part of the
  algorithmic regime. For Bayesian methods, `mc_samples` is likewise a real
  algorithmic knob (floor ≈ 20 for a meaningful BALD estimate, per Gal et al.),
  not a free dial; reducing it changes the uncertainty estimate measurably and
  must be flagged.

## Common pitfalls

- **Too-small pool ratio.** Labeling >20% of the pool exhausts it and degenerates
  the diversity signal; the notebook should flag this when smoke defaults trip it.
- **Single-seed claims.** Differences <1–2 points are within seed noise — treat
  single-seed output as mechanical verification, not a result.
- **Warm-starting between rounds.** Most modern AL (BADGE included) retrains from
  scratch; warm-starting changes the algorithmic regime and is a *silent* failure
  (no crash; the curve still rises).
- **Index-space bugs.** `select_batch` returns positions *into the tensor it was
  passed*, not global pool indices; recover globals through a maintained
  unlabeled-index array, never positional-offset arithmetic. Keep
  labeled/unlabeled disjoint so an already-labeled example can't be re-selected.
  Also silent.
- **Test-set contamination.** Isolate the test set from the pool from the start;
  indices that began as test must never be acquired.
- **Bayesian-specific:** calling `model.eval()` before MC sampling disables
  dropout, collapses BALD scores to zero, and silently degenerates to random
  sampling; distance hyperparameters (e.g. GBALD's `R_0`) are calibrated for the
  paper's data scale and become meaningless under different preprocessing — they
  belong in `critical_requirements.scale_dependent_hyperparameters` with
  schema-1.13 `calibration_context.kind: feature_magnitude` and its closed
  `scale` value declared. Raw-pixel to zero-one uses only the exact
  `paper_value / 255` conversion; every other mismatch remains unresolved rather
  than receiving a guessed factor.
- **Convergence check under the wrong mode.** The flip side of the rule above:
  the `train_until_accuracy` convergence check inside the training loop must run
  under `model.eval()` (then restore `model.train()`). With dropout (or
  batchnorm) active during the measurement, the reported train accuracy is
  depressed, a threshold like 0.99 is never reached, the early-stop is dead code,
  and per-round training silently caps at `max_epochs`. When `max_epochs` is a
  low smoke bound that becomes a few gradient steps on a small labeled set, the
  model never leaves chance and the BALD signal collapses too — a degenerate
  learning curve with no crash. Same `train()`/`eval()` discipline as MC
  sampling, opposite direction: dropout ON for scoring, OFF for the accuracy
  measurement.
- **A ranking score is not a probability.** A geometric/representativeness
  RANKING that scores candidates by an inverse distance (GBALD Eq. 13,
  `max_j R_0/‖x−D_j‖`) must rank by the RAW value, not the capped probability
  model (Eq. 5, `p=1` within `R_0`) — even though the paper writes Eq. 13 with a
  `p(·)` symbol. Capping the ranking ties every within-radius candidate at the
  top, the sort falls back to input order, and the ranking becomes a no-op: the
  method degenerates to plain BALD with no crash. Corollary: a ratio-ranking
  with a constant numerator is scale-invariant, so the RANKING never needs `R_0`
  rescaling — only the Stage-1 capped prior is genuinely scale-dependent. This is
  the actual root cause behind an `R_0`-saturated geometric ranking; rescaling
  `R_0` only papers over a wrongly-capped ranking.
- **A diversity ranking selects the FARTHEST, not the closest.** A
  representativeness/core-set ranking that exists to reduce redundancy selects
  the candidate farthest from the labeled set — max-min, `argmax_x min_j ‖x−D_j‖`
  — not the closest. A geometric probability `R_0/‖x−D‖` measures CLOSENESS (high
  = close = already represented = redundant), so ranking a diversity objective by
  the argmax of it selects the most redundant point — inverted. GBALD's Eq-13
  literal formula has exactly this inversion against its own k-centers definition
  ("nearest distance to D₀ is maximal") and its "reduce sampling nearby data"
  purpose; implement the max-min selection the prose requires, which also removes
  `R_0` from the Stage-2 ranking entirely.
- **A two-stage selector must preselect at least as many candidates as it
  returns (b >= b').** If the method scores a candidate pool of size b
  (`batch_returns`) and then ranks down to b' outputs per round (`batch_size`),
  then b >= b' or the selector can never return a full batch and silently
  throttles acquisition. The preselection count b is paper truth, so it belongs
  in `data_setup.batch_returns` (not just the signature default, which is a
  runtime convenience and cannot see `batch_size`). The deriver reconciles the
  runtime value to satisfy b >= b'. Single-stage selectors (one acquisition
  score, no prefilter, e.g. BADGE) have no `batch_returns` and skip this.

The pipeline produces a **single-method** `method/` package + tutorial notebook.
Standard baselines (Random, uncertainty family, CORESET, BALD/BatchBALD),
multi-method benchmarks, seed sweeps, and pairwise penalty matrices are *not*
generated — they belong to a downstream researcher-run benchmark that uses the
generated package as one entrant. The pipeline enforces only what the generated
package itself can carry: retrain-from-scratch (`training.py`), a fixed held-out
test set (`data.py`), and — for Bayesian methods — MC sampling active at
inference (`method.py`).

## Canonical references

- Settles (2009) — *Active Learning Literature Survey*.
- Ash et al. (2020) — *Deep Batch Active Learning by Diverse, Uncertain Gradient
  Lower Bounds* (BADGE).
- Sener & Savarese (2018) — *Active Learning for CNNs: A Core-Set Approach*
  (CORESET).
- Houlsby et al. (2011) — *Bayesian Active Learning by Disagreement* (BALD);
  Gal & Ghahramani (2016), Gal et al. (2017) for MC dropout; Kirsch et al. (2019)
  for BatchBALD; Cao & Tsang (2022) for GBALD.
