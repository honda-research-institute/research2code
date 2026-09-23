# The bug zoo — known-bad / known-good artifacts from real runs

Curated snapshots of **real pipeline outputs** whose quality status was established by
manual semantic audit (see internal notes (not shipped)). This is the acceptance contract for
the verification work in the recentering plan (internal, not shipped): **every new verification tier must
flag every known-bad artifact below and must NOT flag the known-good ones.** That makes
"does our verification catch real failures" a seconds-fast pytest question instead of a
70-minute live-run question.

Provenance: snapshotted 2026-06-09 from `example_runs/` (committed state of 2026-06-01).
Cross-reference finding IDs against the cross-paper findings note (internal, not shipped).

**Reading the codes:** the probe ids (US-x, UB-x, AL-x, MP-x, KD-x, TSF-x, CT-x) and finding
classes (M-00x) used in the matrices below are machine identifiers — they key
`probe_report.json` and the acceptance tests. The quick-reference glossary at the top
of the probe catalog note (internal, not shipped) maps every one to plain language.

**Scenario types.** `frozen-evidence` = byte-identical snapshot of a real run output.
`reconstruction-needed` = the buggy original lived only in gitignored `r2c_runs/` and was
deleted before snapshotting (the committed `example_runs/` versions are post-fix
regenerations) — a synthetic mutant must be built from the verbatim buggy code quoted in
the analyses, labeled as a reconstruction citing its finding ID. Verified per-file
2026-06-10 against the analyses' quoted code.

## Scenarios

### `gbald-lr74-never-learns/` — frozen-evidence (paper: bayesian-active-learning / GBALD)

| File | Status | Verified expected flags |
|---|---|---|
| `params.json` | **known-bad** | `learning_rate=7.4` — the paper's *section number* leaked into the value, reasoning says "Adam-default convention" (range probe: lr ∉ [1e-6, 1]; reasoning-value consistency probe). `R_0=2000.0` (here `source=spec_default`) against ToTensor [0,1] data — its degeneracy is caught **behaviorally** (UB-2/UB-7, the constant/dead geometric-prior term); the *static* scale-mismatch probe (US-4) needs the spec's `scale_dependent_hyperparameters` declaration + a data sample, which this dir lacks — see the self-contained `gbald-r0-scale-mismatch/` below. `dropout_rate` labeled `source=paper` with `paper_section` admitting "Field-guide convention" (provenance-internal-consistency probe, A-001 class). NOTE: the fabricated 99% quote is NOT in this file (it's in `../evidence/june9-gbald-run/`, below). |
| `notebook.ipynb` | **known-bad** (executed, outputs included) | Learning curve flat at 0.090–0.116 across all rounds on 10-class MNIST — trains to chance (beats-chance probe). Geometric-prior demo prints `1.0000` three times — degenerate scoring (degeneracy probe). The flat-curve outputs ARE the evidence; do not strip outputs. Its §5.1 loop bookkeeping is actually CORRECT (must pass index-space probes — false-positive guard). |
| `method.py` | known-bad (secondary) | Eq. 10 likelihood term is a scalar added uniformly to all candidates — dead for argmax selection (dead-computation/term-ablation probe; self-acknowledged at lines 167–171). With R_0=2000 the geometric prior is constant across the pool (score-degeneracy probe). Snapshot contains only method.py — behavioral probes need the harness stub kit or package completion. |

Pipeline verdict at the time: smoke gate passed "clean (iteration 0)"; stage-2x params
review passed lr=7.4 explicitly; stage 4 caught only R_0 (critical), which was deferred —
the artifact **shipped**.

### `gbald-verified-coreset/` — known-good (paper: bayesian-active-learning / GBALD)

| File | Status | Verified expected flags |
|---|---|---|
| `method.py` | **known-good** | `construct_core_set` harvested verbatim from the 2026-07-03 first VERIFIED GBALD delivery (`r2c_runs/archive/bayesian-active-learning-20260703-first-verified/method/method.py`), so the AL Stage-1 probe family's healthy conformance case survives archive cleanup. Must pass AL-S1-1 (unique in-range indices), AL-S1-2 (picks spread across a cluster-sorted pool), and AL-S1-3 (eta moves the selection). Its lr74 counterpart above is the family's defect case (AL-S1-2 fail: constant-score argmax picks the leading cluster block). |

### `pdwa-omega-degenerate/` — frozen-evidence (paper: pdwa / Enhanced DWA, motion_planning)

| File | Status | Verified expected flags |
|---|---|---|
| `method.py` | **known-bad** (M-005 class) | `evaluate_objective_function` takes `omega` and never reads it (unused-decision-parameter probe, lines 171–355) — all (v,ω) with equal v tie exactly; steering is a seeded random tie-break (lines 506–521; steering-responsiveness probe). Penalty terms added with `+` in a maximized objective — rewards proximity (direction-consistency probe). Per 2026-06-09 decision: correct verdict = **flag-for-researcher → assumptions.md** (faithful transcription of degenerate paper equations), never auto-fix. |
| `notebook.ipynb` | **known-bad** (partial) | Cell prints literal `Collision-free: Yes (no collisions detected in this demo)` with no collision check anywhere in §5 (claimed-result-literal probe; the adjacent computed `Goal reached: {...}` must NOT flag). `env` and `collision_model` loaded in §3 and never passed to the planner (unused-artifact probe). **M-004 is NOT in this snapshot** — this version DOES move obstacles each step (`obs['x'] += dx`); see reconstruction below. |

Pipeline verdict at the time: stage reviews claimed "symbol-by-symbol correctness";
stage 4 passed with **zero findings**.

### `badge-goodmethod-badnotebook/` — frozen-evidence, **method.py good / notebook ALSO GOOD in this snapshot**

| File | Status | Verified expected flags |
|---|---|---|
| `method.py` | **known-GOOD** | Vectorized, faithful, no dead code — the strongest output the pipeline has produced. **Must pass all probes** (false-positive guard for the whole catalog). |
| `notebook.ipynb` | **known-good (post-fix)** | This snapshot is the CORRECTED loop: `chosen = unlabeled_idx[positions]` + set-difference update, and `build_model(...)` fresh each round. The M-002/M-003 bugs are NOT here. It serves as the PASS-side guard for the index-space and warm-start probes. |

### Reconstruction scenarios — synthetic mutants (built 2026-06-10)

Built from the verbatim buggy code quoted in the analyses, by mutating the corrected
frozen-evidence snapshots. Modified cells have outputs stripped (the bug shapes were
never executed here); all surrounding narrative is intact — the claims-vs-behavior gap
is the property under test.

| Scenario | Finding | Verified expected flags |
|---|---|---|
| `badge-badloop-reconstructed/notebook.ipynb` (mutated cell `c53ec107`) | **M-002 + M-003** | Index space: unlabeled pool rebuilt every round as the fixed suffix `x_pool[len(initial_indices):]` (contiguous-prefix assumption), positions mapped via `i - len(initial_indices)`, no set-difference — already-labeled points re-selectable, negative indices wrap silently (AL-1 loop-invariant probe: disjointness, no-reacquisition, added==unlabeled_before[positions] all violated; AL-2 offset-ban). Warm-start: `build_model` only before the loop, one persisted model retrained in-place while comments claim "Retrain from scratch" (AL-3 AST gate, AL-4 fresh-weights probe). |
| `pdwa-frozen-obstacles-reconstructed/notebook.ipynb` (mutated cell `e78a615b`) | **M-004** | The per-step obstacle-update block is deleted: `obstacle_states` set once while the setup comment still says "dynamic obstacles (moving toward the robot's path)" and the narrative claims a dynamic scenario (MP-1 scenario-dynamics probe, both static and behavioral arms). |
| `kd-wrong-op-reconstructed/method.py` | **M-001 class** | `compute_instance_weights` docstring claims `(B, N_queries, 1)`; `torch.diag_embed` on the (B,N,N) similarity matrix actually produces (B,N,N,N) — empirically verified: claimed (2,5,1), actual (2,5,5,5), and `weighted_feature_loss` still computes a finite scalar (silent) (UB-1 shape-claims-vs-execution probe). Symbol-level math matches the "paper"; only the op is wrong. No bev-distill artifact was ever captured, so this is the only M-001 fixture path. |
| `idbrrt-no-path-reconstructed/method.py` (built 2026-07-04) | **planner_no_path_found class** | The live iDb-RRT delivery (breadth run 2026-07-01, gitignored run dir) returned `PlanResult(trajectory=None, status="timeout")` with zero optimization attempts from its OWN demo notebook and from the probe kit's trivial obstacle-free fixture — the planner never demonstrated a successful plan. Compact reconstruction in the paradigm-template interface shape (goal-STATE signature that raises on a 2-dim goal, Environment/UnicycleDynamics classes, Trajectory with parallel states/controls lists). Expected: MP-4 goal-progress probe `flag_for_researcher` via the contract-driven kit (mp_planner_kit); MP-3 honest unprobeable pointing at MP-4. |

### `pdwa-jterms-fresh/` — frozen-evidence (2026-06-10 pdwa run, the first post-recentering matrix row)

| File | Status | Verified expected flags |
|---|---|---|
| `method.py` | **known-bad** (M-005 structural recurrence) | Fresh transcription decomposed the objective into six `compute_j_*` terms and FOUR read no decision variable at all (heading/distance/risk/count depend only on `robot_state`) — the paper's candidate-invariance faithfully reproduced under a brand-new API (`plan` + terms vs May's `select_velocity`). Stage-4 review again produced no finding for it. Probe target: j-term signature/AST analysis (US-5 class) + UB-3 once the term-call adapter exists. Secondary: `plan` builds `rng = np.random.default_rng(seed)` and never uses it (F002, dead seed). |
| `params.json` | **known-bad** (US-3b class) | `beta_1..5` and `d_safe` carry `paper_value=0.1` with reasoning "Paper states beta_X=0.1" / "Paper states d_safe=0.1" on `system_default` entries — the paper defines betas symbolically only and states d_safe = V_max/4. Caught live by the stage-4 review (F003/F004, deferred as nice-to-have, SHIPPED anyway) and now deterministically by US-3b. `dt`'s honest relabeled entry must NOT flag. |
| `stage4_review_report.json` | reference | The review that caught F001–F005 (incl. the M-004 class via the methodology contract) while missing the j-term degeneracy — the comparison baseline for probe-vs-review coverage. |

### `gbald-double-retrain-loop/` — frozen-evidence (2026-06-10 fresh GBALD run, matrix row 2), **known-good loop shape**

| File | Status | Verified expected flags |
|---|---|---|
| notebook.ipynb | loop bookkeeping GOOD (delivered unexecuted) | acquisition-loop invariants (AL-1) and fresh-weights (AL-4) must PASS |

Why it's here: this notebook retrains from scratch both BEFORE acquisition (the
scoring model) and AFTER it (the learning-curve model), so consecutive training
calls alternate growth 0 and +batch_size. The strict every-step-grows form of the
training-growth invariant false-positived on it during the 2026-06-10 matrix-row
audit — this fixture pins the false-positive side (a step of 0 is loop-shape
variation, never a defect on its own). Its loop cell also binds `n_features`,
pinning the microharness seeding-table coverage that was missing that name.

### `gbald-negstride-selector/` — frozen-evidence (2026-06-10 GBALD validation re-run, the zero-halt row)

| File | Status | Verified expected flags |
|---|---|---|
| `method.py` | **known-bad** (crash-on-legal-input) | `select_batch` Step 2 builds a reversed (negative-stride) numpy index array — `np.argsort(bald_scores)[::-1][:batch_returns]` at line 401 — and indexes the torch pool with it at line 402: torch rejects negative-stride numpy arrays, so EVERY call with a non-empty pool crashes (`ValueError`, empirically verified; numpy pools crash earlier on `.device`, so no input convention works). The notebook's own §4 demo of the same composition used `.copy()`; the packaged pluggable didn't. AL-5 crash arm: fail. CT-1 / UB-7 (composite): precise subject-crash unprobeable. |
| `params.json` | **known-bad** (AL-6 class) | `initial_labeled=1000` >= `pool_size=800`: the core-set swallows the whole pool, the unlabeled set is empty before round 1, and every `select_batch` call early-returns `[]` — the delivered demo never executes Stage 2, which is exactly why the smoke gate never reached the crash (AL-6 demo-config-reachability: fail). The deriver's own reasoning prose even notes the budget/pool ratio 1.88 against the field guide's 0.4 cap. |
| `probe_report.json` | reference | The before-state: the battery that reported the crash as three bare unprobeables (UB-7/CT-1/AL-5) — the evidence that motivated the crash arm. After the arm: 2 fail (AL-5 crash, AL-6 reachability), CT-1/UB-7 deferring precisely. |
| `smoke_diagnosis.json` | reference | The 3c diagnosis from the run's smoke loop: the float64 empty-index crash at pool exhaustion, whose fix (`if not pos: break`) made the no-op loop "pass" smoke. The pool-exhaustion root cause is named in its own prose and was not acted on. |

Why it's here: the negative-stride adjudication's two findings ride on each other —
a selector broken for every real input shipped `completed` because the demo config
made the broken path unreachable and the probes filed the crash under "couldn't
probe". The fixture pins both directions: the crash stays a crash (no `.copy()`
sneaks into Step 2) and the config stays unreachable (1000 vs 800).

### `bayesian-reversed-argsort-reconstructed/` — reconstruction (2026-07-29 bayesian-active-learning delivery, RCA 2026-08-03 finding 6)

| File | Status | Verified expected flags |
|---|---|---|
| `method.py` | **known-bad** (silent miscompute) | `select_batch` Step 1 intends "sort candidates descending by BALD score" and argsorts the REVERSED score array — `np.argsort(bald_scores.cpu().numpy()[::-1])`, verbatim from the delivered `method/method.py:419`. Executes cleanly; the ordering is meaningless against the original candidate array. US-10 lint gate (reversed-argsort check): **error**. |

Why it's here: the silent sibling of `gbald-negstride-selector/` — same descending
intent, opposite failure direction. The GBALD roll used the correct idiom
(`np.argsort(x)[::-1]`) and crashed downstream on torch's negative-stride
rejection; the bayesian roll used the broken idiom and sailed through smoke with a
scrambled ranking. The pair is the false-positive contract for the lint: the
correct idiom in the negstride fixture (and in this fixture's own geometric
ranker) must never fire, the reversed operand must always fire.

### Demo-success verdict shapes — reconstructions (built 2026-07-17)

The acceptance fixtures for the `demo_failure_invisible_to_smoke` class
(the demo success semantics design note (internal, not shipped), approved
2026-07-16): a notebook executes cleanly while its headline demonstration
visibly fails in its own printed output. The three failing shapes were
recorded live on 2026-07-14; the original run dirs were gitignored and
cleared, so these are reconstructions carrying the verbatim evidence lines
the design note quotes, on the committed notebook layouts. The verdict pass
(`scripts/demo_verdict.py`) is exercised by `tests/test_demo_verdict.py`.

| Scenario | Status | Verified expected verdict |
|---|---|---|
| `idbrrt-demo-timeout-reconstructed/notebook.ipynb` | **known-bad** | Headline §5.1 cell prints `Planning status: timeout` / `Planning failed after 7 iterations`; §5.2 prints `No trajectory to visualize`. Smoke-clean. Demo verdict: **failed** (failure marker), quoting the status line — and the delivery demotes through the `demo_verdict` reason itself, never through adjacent demoters. |
| `icra-demo-goal-miss-reconstructed/notebook.ipynb` | **known-bad** | Headline cell prints `Status: timeout` ... `Distance to goal: 4.000 m` (the robot spun in place). Demo verdict: **failed** (failure marker). |
| `mp-demo-success-reconstructed/notebook.ipynb` | **known-good** (adjacent-good guard) | Same layout, `Planning status: success` + path stats. Demo verdict: **succeeded**; MUST add zero demoting reasons. |
| `demo-chance-flat-reconstructed/notebook.ipynb` | **known-bad** | The Rethinking-Grouping 2026-07-14 headline shape on the supervised layout: executed accuracy series flat at chance (0.09–0.106 on 10 classes) across every round. Demo verdict: **failed** (beats-chance check, the shared UB-6 rule). |
| `demo-beats-chance-reconstructed/notebook.ipynb` | **known-good** (adjacent-good guard) | Healthy 0.46→0.88 curve (the shape of the real GBALD healthy run). Demo verdict: **succeeded**; MUST add zero demoting reasons. |

Families without `demo_success` markers are pinned in-test (no fixture dir
needed): `undetermined` plus an honest disclosure, and on a COMMITTED family
additionally the kit-coverage finding we own.

Time-series forecasting shapes — reconstructions (built 2026-08-06, R2C-070),
from the loop2 quality review (`audit/reports/2026-08-06-pdfgnn-loop2-quality-review.md`):

| Scenario | Status | Verified expected verdict |
|---|---|---|
| `tsf-demo-nan-metrics-reconstructed/notebook.ipynb` | **known-bad** | The 2026-08-06 pdfgnn loop2 headline shape: the executed §5 evaluation prints `RMSE: nan / MAE: nan / WMAPE: nan` (held-out window with no ground truth). Smoke-clean. Demo verdict: **failed** (metric-NaN failure marker) — the live roll was vacuously `undetermined` because the run-authored pack declared no markers, which is exactly what the committed TE-TSF node closes. |
| `tsf-demo-pass-reconstructed/notebook.ipynb` | **known-good** (adjacent-good guard) | Same layout: per-epoch loss series, finite held-out metrics, predict-zero + repeat-last baselines printed beside the model, and the deterministic `Demo verdict: PASS — ...` line the TE-TSF notebook layout specifies. Demo verdict: **succeeded**; MUST add zero demoting reasons. |

### Forecasting probe-kit shapes — schema-2 reconstructions (built 2026-08-09, R2C-087)

`tsf-probe-reconstructions/cases.json` is the compact authority for the
generated-style packages exercised by the shared kit-conformance registry.
These are synthetic probe inputs reconstructed from audited failure and
representation shapes; they do **not** certify an archived run and they do not
invent graph, partition, or paper-protocol truth.

| Case | Source status | Verified expected probes |
|---|---|---|
| `audited_output_bad` | **known-bad reconstruction** | Combines the generic conditional-means-as-samples class recorded on `_7` with the near-zero magnitude class recorded on the current roll. TSF-1/3/4/5 fail against the coherent synthetic comparison record. No claim is made that one archived roll had all four defects. |
| `relational_self_path_severed` | **known-bad reconstruction** | Night3, archived as `_7`, supplies the executed proof: an isolated series stayed bit-identical under a 100x own-history perturbation. The tiny asymmetric dense package reconstructs that consequence and TSF-3 fails. Current and `_10` remain R2C-084 relational-index-domain bad reproducers, not the source of this own-history claim. |
| `archived_11_sparse_good` | **known-good representation control** | `_11` supplies the sparse/local-endpoint-bounded representation shape. The synthetic package restores the self path; TSF-1/3/4/5 pass and TSF-2 is not applicable. `_11`'s independent-horizon output is not mislabeled ancestral. |
| `archived_7_dense_good` | **known-good representation control** | `_7` supplies the dense representation shape only. The synthetic output fixes `_7`'s conditional-mean sample defect; TSF-1/3/4/5 pass and TSF-2 is not applicable. |
| `synthetic_ancestral_good` | **generic positive control** | Seeded Student-t paths carry adjacent-step feedback on the sparse asymmetric fixture; all five probes pass. `_10` has the corresponding ancestral output shape but remains relational-indexing known-bad, so this control does not certify it. |
| `graph_free_adjacent_good` | **known-good adjacent-family control** | Exact graph-free `None`, live self-history path, genuine samples, valid held-out skill, and same-unit magnitude. TSF-1/3/4/5 pass; TSF-2 is not applicable. |

Unsupported schema-1 runtime grammar plus absent typed evaluation-role evidence
is the unbindable control: all five rows say `unprobeable`; no legacy shape,
constructor, or role guessing supplies a stand-in.

### Scenario-fidelity shapes — reconstructions (built 2026-07-27, R2C-025)

The acceptance fixtures for the **M-006 class** (a demo setup that contradicts
a paper-stated scenario assumption; the researcher-feedback section 3 designs note (internal, not shipped),
Note 2, approved post-review). Labeled reconstructions of the recorded
circles-vs-polygon validity bug on the committed motion-planning notebook
layout: self-contained run dirs (`.pipeline/method_spec.json` with the slice-A
`scenario_assumptions` capture + `notebook.ipynb`). Sections 4 and 5 of every
notebook are poison cells (`raise RuntimeError`), so a **pass** verdict also
proves the setup-slice executor never reaches the method or demo sections.
Exercised by `tests/test_scenario_fidelity.py` through
`probes/scenario_setup.py` and the SC-* detectors in
`probes/motion_planning.py`.

| Scenario | Status | Verified expected verdict |
|---|---|---|
| `mp-scenario-geometry-mismatch-reconstructed/` | **known-bad** (M-006) | Spec captures `obstacle_geometry="circle"`; §3 builds two polygon obstacles (in a list AND behind `env.obstacles`). SC-1: **fail** (`scenario_mismatch`) naming polygon vs circle — demotes. SC-2/SC-3: **unprobeable** (`no_captured_assumption`) — disclosures only. |
| `mp-scenario-geometry-match-reconstructed/` | **known-good** (adjacent-good guard) | Same spec; §3 builds two circles (one explicit `type`, one inferred from x/y/radius keys). SC-1: **pass**; MUST add zero demoting reasons from SC-1. |
| `mp-scenario-unbindable-reconstructed/` | **known-bad** (captured-but-unbindable) | Same spec; §3 hides the scene inside an opaque `_World` object with no `.obstacles` attribute and no obstacle-named list. SC-1: **flag_for_researcher** (`setup_unbindable`) — demotes, never a quiet disclosure (the zero-pedestrians shape). |
| `mp-scenario-absent-assumption-reconstructed/` | **known-good** (disclosure shape) | Circle setup, spec has NO `scenario_assumptions`. SC-1/2/3: **unprobeable** (`no_captured_assumption`) — disclosed without demotion, and the setup slice is never executed. |

Adjacent-family guard (no fixture dir needed, pinned in-test): a
non-declaring family yields no scenario dimension bindings, no SC rows, and a
byte-identical report.

### `gbald-r0-scale-mismatch/` — frozen-evidence (2026-06-12 GBALD attempt-4 run), the US-4 fail-side

Self-contained run dir (`.pipeline/method_spec.json` + `.pipeline/params.json` +
`method/example_data/`) snapshotted at audit time — the live `r2c_runs/` copy was about
to be cleared by the next validation run. The dedicated home for the
**scale-dependent-parameter mismatch** acceptance (US-4), which needs all three inputs
together (declaration + shipped value + measurable data scale).

| File | Status | Verified expected flags |
|---|---|---|
| `.pipeline/method_spec.json` | declaration (real analyzer output) | `critical_requirements.scale_dependent_hyperparameters[0]` declares `R_0` `paper_value=2000`, `assumes_data_scale="raw_pixel_unnormalized"`, divide-by-255 rescale rule in `description` (`formula` is null — the half-built deterministic-rescale contract in `schemas/method_spec.py`). |
| `.pipeline/params.json` | **known-bad** (M-004 class) | `R_0=2000.0` shipped `source=paper` — the raw-[0,255]-pixel calibration, unrescaled (US-4 fail). |
| `method/example_data/mnist_subset.json` | data sample | Faithful compact slice (8×33) of the delivered MNIST subset, preserving the real [0,1] normalized scale (min 0, max ~1) — torch-free JSON so the probe + test run in CI. |

US-4 verdict: **fail** — the data is [0,1]-scaled while `R_0=2000` assumes raw pixels and
ships unrescaled, so the geometric prior `R_0/||x-D_j||` saturates to 1.0 and
`select_batch` collapses to plain BALD. PASS-side guards (constructed in-test from this
fixture): exact `R_0=2000/255` rescaled → pass; the historical rounded
`7.843` remains evidence but is not certified as the exact conversion; the same raw value on
genuinely raw-scaled data → pass.

## Real fabricated-provenance evidence (not in the zoo dirs)

`tests/fixtures/evidence/june9-gbald-run/pipeline/params.json` — `train_until_accuracy=0.99`
labeled `source="paper"`, `paper_section="Section 7.4"`, note "Paper: train until training
accuracy exceeds 99%": **zero occurrences of "99%" in that run's own `pipeline/paper.md`**;
`num_rounds=6` `source=paper` ("Paper uses 6 rounds") is equally unsupported. This is the
fail-side target for the provenance quote-match probe. The PASS-side: the BADGE paper
genuinely contains "until training accuracy exceeds 99%".

## Current-best reference (not part of the zoo)

`tests/fixtures/evidence/june9-gbald-run/` is the 2026-06-09 GBALD run (status:
degraded) — the best end-to-end output to date, preserved because `r2c_runs/` is
gitignored and deleted by default. Known residual defects, useful as probe targets:
stale "R_0=2000" title-cell prose contradicting its own params table (narrative-vs-params
probe); the fabricated provenance above; per-round core-set recomputation whose
acquisitions are discarded (dead-computation probe). Known-good side: `R_0=7.843`
auto-rescaled (scale probe PASS), `lr=0.001` (range + trains-on-synthetic PASS).

## Adding a scenario

1. A scenario earns a dir only after a manual audit (per the paper-analysis template (internal, not shipped))
   establishes its status — fixtures encode *verdicts*, not vibes.
2. Snapshot the smallest file set that carries the evidence (keep executed outputs when
   the outputs are the evidence). If the buggy original is already gone, build a
   reconstruction mutant from the analysis doc's quoted code and label it as such.
3. Add the scenario + verified expected-flags table here, citing the finding ID —
   verify each claimed flag against the actual file before writing it down.
4. Wire the probe regression test to the table.

**Standing lesson (2026-06-10):** snapshot evidence at audit time, not later — the
delete-before-rerun policy plus post-fix regeneration erased the buggy originals for
M-002/M-003/M-004 within days of their discovery.
