# Overnight 2026-07-07 batch — harvested fixtures

Snapshots pulled from the 10-paper overnight batch (ran 23:14 07-07 to
08:21 07-08) so the evidence survives the post-fix re-run, which wipes
its source run dirs. Full run-by-run analysis lives in
the overnight-0707 batch results note (internal, not shipped). These five items were the
ones the re-run leaves without a pinned fixture (items 12b, 21, 23, 24,
25 landed in the 07-08 morning fix block with their fixtures already in
the test suite).

Harvested 2026-07-09 (Opus session) before launching
`r2c_runs/rerun-0708-post-fix-manifest.txt`. The re-run wipes
ICRA21_HICA, detr-distill, iDb-RRT, ADAM, ms3d, Rethinking-Grouping,
ACC2021_MPC_CBF, and SRL. It does NOT wipe bev-distill or fedavg.

These are raw evidence snapshots, not minimal test inputs. When each
item gets built, distill the smallest input that exercises the failure
(an `inject.py` builder or a trimmed JSON) and keep these as the
provenance record the trimmed fixture derives from.

---

## item-19-runtime-cache-detr (queue item 19)

**Source:** `r2c_runs/detr-distill/.pipeline/stage_2b.halt` (dir WIPED).

The architecture-coder ran its own generated package to self-test it
(contract-encouraged), the generated `data.py` cached synthetic data
under `<caller path>/_cache`, and because the caller path resolved
against the working directory the cache landed at the run root as
`_cache/synthetic_detection.pt` (240 KB binary, not copied — the path
string is the fixture). The scope check flagged the stray as an
out-of-scope write and hard-halted with all three declared outputs
already on disk. This is the BADGE 07-05 MNIST-cache genus one path
generalization away.

`stage_2b.halt` records the exact rejection:
`Out-of-scope writes: ['_cache/synthetic_detection.pt']`.

**Test to build:** `_is_runtime_data_cache` (scripts/run_pipeline.py
near line 2115) currently exempts only the literal prefix
`method/example_data/_cache/`. A unit test should assert a run-root
`_cache/synthetic_detection.pt` is exempt while a genuine out-of-scope
authored file (say `method/scratch.py` or a top-level `notes.txt`)
still halts. The structural half (opt the arch-coder dispatch into the
branch-B recovery check `validate_architecture_coder_output.py`) is
source work, not a fixture.

---

## item-20-smoke-timeout (queue item 20)

**Sources:** bev-distill (primary, NOT wiped by the current manifest
but harvested anyway) and Rethinking-Grouping (counter-datum, WIPED).

- `bev-distill.KNOWN_ISSUES.md` — cell 44 timed out at the 180s
  per-cell budget three times and never raised, yet the surface tells
  the researcher "that cell raises an error" and "The traceback is in
  the technical detail below" (there is no traceback), then dumps the
  cell source under the label "stderr:". The embedded issue_signature
  also carries the generated cell's unverified claim "3 epochs stays
  within the 180s per-cell timeout (spec-approved range: 3-5)", which
  it does not for this model at this scale.
- `bev-distill.run_events.jsonl` — the fix loop treated the timeout
  like any smoke failure and asked the notebook-generator to "fix" a
  cell whose only defect is costing more than 180s, and its fix
  changed nothing ("exception unchanged: None" in the driver record).
- `rethinking.smoke_diagnosis.json` + `rethinking.run_events.jsonl` —
  the counter-datum. Cell 14 timed out twice, iteration 1's
  diagnostician overrode to the notebook-generator, and that fix
  cleared it (smoke clean at iteration 2). So the class is fixable
  today, unreliably. The loop-layer fix is about making that reliable.

**Test to build:** a timeout degrade whose KNOWN_ISSUES entry names
the budget and wall time with no traceback language, and a fix dispatch
prompt that names the timeout class and constrains the ask to work
reduction.

---

## item-22-id-fidelity (queue item 22)

**Sources:** ADAM and iDb-RRT (both WIPED). Instance (a), run 1's ICRA
re-ask DEF-r_e to F001, landed with item 12b layer 3 and is pinned
there already.

- `ADAM.paper_map.json` — the correct element ids, including
  `eq-temporal-average` and `concept-temporal-averaging`.
- `ADAM.run_events.jsonl` — instance (b): the explainer anchored to
  `eq-temporal-averaging` (a near-miss of `eq-temporal-average`), the
  fabricated-anchor check rejected it, the retry repeated the same
  near-miss, and METHOD.md shipped 3 pending markers. Instance (c):
  the decomposer's `concept-ema` cited `eq-ema-vt` and `eq-ema-mt`,
  neither of which exists (healed by a judge-routed retry).
- `iDb-RRT.paper_map.json` — instance (d) plus the aggravator: this
  map mixes id prefixes in one artifact (`eq-cost-function` etc.
  alongside `equation-continuous-dynamics`), which invites the near-miss.
- `iDb-RRT.run_events.jsonl` — the decomposer cited
  `eq-dynamics-continuous` where its own map says
  `equation-continuous-dynamics` (healed by retry).

**Test to build:** an explainer retry that converges on the corrected
id in one pass (the ADAM near-miss verbatim is the fixture), driven by
a validator error that names the nearest existing id and a retry prompt
that enumerates the allowed ids inline. A prefix-normalization check in
validate_paper_map is the cheap prophylactic (the iDb-RRT mixed-prefix
map is its fixture).

---

## item-26-stage5-refinalize-ms3d (queue item 26)

**Source:** ms3d (WIPED).

The fidelity loop auto-routed a real fix (the paper's
`multi_round_self_train`, added to method.py at line 971), but
`init_finalizer` runs only at 2d and never again, so the new public
symbol is absent from `__init__.py`'s `__all__` and importing it raises
ImportError. The fidelity reviewer caught the regression of its own fix
chain and deferred it honestly as F005.

- `method.py` — contains `def multi_round_self_train(` (line 971) and
  lists it in the module docstring's symbol table.
- `__init__.py` — imports and `__all__` OMIT multi_round_self_train
  (confirmed absent). This is the import-inconsistency the test must
  detect.
- `deferred_findings.md` — F005 states the regression verbatim: the
  function "exists but is inaccessible from the public API".

**Test to build:** after a stage-5 auto-routed fix that touches
method/, re-run init_finalizer + validate_package_imports before the
final manifest, so a stage-5 fix that adds a public symbol is
importable from the delivered package.

---

## item-27-stage1-bonus-srl (queue item 27)

**Source:** SRL (WIPED).

The decomposer's equation-floor trajectory converged 14 to 8 to 3
failing quotes across the fix-mode retries, but the retry cap ended the
run 3 quotes short, and one of the four iterations (the 8-to-8 plateau)
was a verified item-23 cap-burn death (a dead turn, not fixation). The
halt reason is truthful and precise here ("cascading progress that did
not converge within the cap"), which is the honest contrast the fix
should preserve.

- `stage_1.halt` — the truthful halt reason and the final 3 failing
  equation quotes (eq-objective, eq-constraint-collision,
  eq-value-function).
- `run_events.jsonl` — the full 14/8/8/3 count trajectory across the
  validator loop iterations.

**Test to build:** the SRL trajectory fixture completing (or halting
one blessed iteration later with the truthful cascading reason) once
the stage-1 loop gains a judge-blessed bonus iteration on strictly
monotone progress, mirroring the smoke gate's judge_cap_recovery, and
no bonus iteration granted on a flat signature.

---

## item-23-passage-locator-acc (queue item 23 part 3)

**Source:** ACC2021_MPC_CBF (WIPED by the re-run). Harvested 2026-07-10
(Fable session) when the passage-locator design note was written,
the passage locator design note (internal, not shipped).

The run 5 halt whose fix dispatches burned the 20k output cap
re-reading the paper to fix rejected equation quotes. The paper map
snapshot still fails the equation verbatim-quote floor on 11 elements
against this paper.md, which makes the pair the live reproduction the
locator's unit tests need (the prototype in the design note measured
11 of 11 located, 10 of 11 with the display math in the top candidate,
the one miss being the stitched-quote shape the math-block adjacency
guard covers).

- `paper_map.json` — the decomposer's map with the 11 floor-failing
  equation quotes (verified failing on harvest day by re-running
  `check_equation_quotes` against the paper text).
- `paper.md` — the 50KB Marker output the quotes must be located in.
- `stage_1.halt` — the recorded halt (pre-item-23-parts-1-2 wording,
  kept as the before picture).

**Test to build:** the locator unit tests in the design note's testing
section run against this pair. When the build lands, keep this as the
provenance record and distill nothing — the whole point of the fixture
is a real paper's size and Marker artifacts.
