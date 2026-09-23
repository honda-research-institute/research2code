# Example output: one paper, several models

Purpose: show what an R2C delivery looks like, and what changes when the same paper runs on different models, so a new user can pick a model with the tradeoffs in view.

## What is in this folder

Three complete deliveries of the same paper, as the pipeline left them (nothing hand-edited, except that absolute local paths in the `.pipeline/` bookkeeping files were replaced with `<repo>` and the `method/example_data/` download cache was removed):

| Folder | Model | Effort |
|---|---|---|
| `deep-batch-active-learning (LUNA LOW)` | gpt-5.6-luna | low |
| `deep-batch-active-learning (LUNA MEDIUM)` | gpt-5.6-luna | medium |
| `deep-batch-active-learning (TERRA MEDIUM)` | gpt-5.6-terra | medium |

Each folder is what `r2c_runs/<slug>/` holds after a run. Start with `REPORT.md` (the delivery label and why), then `METHOD.md` (the paper's method explained), `method/` (the generated code and tests), and `notebook.ipynb` (the runnable demo). `details/` and `.pipeline/` hold the per-stage evidence the report cites.

The fourth column below, the self-hosted Qwen 27B reference run, is not included here (it is the model R2C was built on and is not reproducible without that deployment). Its numbers stay in the table for context.

## The paper

`input_papers/deep-batch-active-learning` (Ash et al., "Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds"). Every run used the same pipeline code, the same markdown input, and the default `R2C_TIMEOUT_SCALE`. All four runs happened within a week of each other on the same laptop.

## The runs

| | Qwen 27B (reference) | gpt-5.6-luna, low effort | gpt-5.6-luna, medium effort | gpt-5.6-terra, medium effort |
|---|---|---|---|---|
| Date | 2026-08-31 | 2026-09-04 | 2026-09-04 | 2026-09-04 |
| Where the model ran | self-hosted, the model R2C was built and tuned on | OpenAI API | OpenAI API | OpenAI API |
| Wall time | 4 h 11 min | 19 min | 22 min (see note) | 21 min |
| Median agent turn | 226 s | 19 s | 19 s | 22 s |
| API cost | none (self-hosted) | about $0.44 | about $0.73 | about $5.40 |
| **Delivery label** | **draft** | **draft** | **draft** | **verified** |
| Automated checks | 12 pass, 1 for researcher, 3 could not run | 11 pass, 1 for researcher, 4 could not run | 14 pass, 2 advisory, 3 could not run | 12 pass, 4 could not run |
| Claims found in the paper | 9 (2 verified at smoke scale) | 3 (same 2 verified) | 7 (0 verified, 2 blocked on our side) | 3 (same 2 verified) |
| Fidelity-review findings | 1 nice-to-have | 1 critical | 4 important | 1 nice-to-have |
| Generated code | 746 lines | 448 lines | 484 lines | 501 lines |
| METHOD.md | 6,155 words, 11 equations, 2 algorithms | 1,832 words, 3 equations, 2 algorithms | 2,719 words, 7 equations, 2 algorithms | 3,129 words, 8 equations, 3 algorithms |
| Element tests | 6 files, 579 lines | 4 files, 101 lines | 2 files, 40 lines | 12 files, 335 lines |
| Notebook learning curve (test accuracy) | 0.74 to 0.88 over 7 rounds | 0.73 to 0.89 over 6 rounds | 0.74 to 0.86 over 6 rounds | 0.73 to 0.88 over 6 rounds |
| Validator rejections during the run | 0 | 3 | 2 | 1 |
| Halt-judge dispatches | 3 | 7 | 2 | 1 |

Cost is computed from the run's token ledger (`r2c_runs/_history/ledger.jsonl`) at the provider's list prices. Nearly all prompt tokens are cache reads (4.7M, 8.2M, and 5.7M for the three OpenAI runs); output was 49k, 74k, and 52k tokens. Terra costs ten times luna per token, which is the whole cost gap.

Note on the luna-medium wall time: that run halted at stage 1 because a pipeline configuration file in the repo carried an uncommitted edit made minutes before the run started (an operator mistake, not the model), and the driver's write-scope guard attributed the change to the running agent. The run was resumed with the same command and finished. The 22 minutes include the halt and the redo of stage 1.

## What was the same across all four

- **The algorithm.** Every `method.py` implements BADGE faithfully: the hallucinated-label last-layer gradient embedding, then k-means++ seeding with an incrementally maintained nearest-center distance. The OpenAI versions are terser than Qwen's, not wrong.
- **The notebook.** Same structure (35 or 36 cells), learning curves within a few points of each other, all ending near 0.87 to 0.89 test accuracy on the demo subset.
- **The unbound core-set probes.** Three stage-1 probes look for a callable named like `core_set`; BADGE has none, so they could not bind in any run. That is a probe-naming gap, not a model difference.

## What separated them

**Terra (medium) reached `verified`.** It bound its method elements to the right verification probes, returned `List[int]` from `select_batch` as the family contract asks, wrote twelve element tests, and drew one nice-to-have review finding about a README sentence. The two contribution claims (selection differs from random, every scoring term moves the selection) verified at smoke scale, and no finding held the label down. It made one validator slip (a paper-map id prefix) that the judge fixed in one pass.

**Luna at low effort: correct core, two fidelity misses.**

1. `select_batch` returned a torch tensor instead of `List[int]`, and the notebook then called `.detach()` on it. The loop microharness probe substitutes list returns, so it crashed and the loop-bookkeeping check could not run.
2. The analyzer put a pack-check id where a verification probe reference belongs. Flagged as "important" at stage 1 (not routed for a fix), then escalated to critical by the stage-4 fidelity reviewer. That critical finding is what held the label at draft.

Everything downstream was thinner: three equations explained instead of eleven, three paper claims instead of nine, a fifth of Qwen's test code.

**Luna at medium effort: more words, worse delivery.** The extra reasoning bought a longer METHOD.md, seven claims instead of three, and a passing loop probe (it returned a list this time). It also produced the weakest delivery of the four on the things that matter:

- Four important fidelity findings: the method spec says the optimizer is plain SGD while the paper and the code use Adam, the notebook prose repeats the SGD claim, the MLP-for-CNN substitution is not disclosed as a change in inductive bias, and a provenance note claims baselines were tested when the package runs none.
- Because the substitution finding sits unresolved on `model.py`, the report withheld both contribution claims: they show as "needs a fix on our side" even though the underlying probes passed.
- The analyzer left every verification probe reference empty, the same binding mistake as the low-effort run, in a different shape.
- Two element tests, forty lines, covering one of seven eligible elements.
- The smoke gate first saw no learning signal in the notebook (a later pass completed), and two explainer turns died and had to be resumed.

Low-to-medium effort on the same model did not fix the binding mistake and added spec-versus-code inconsistencies. Spending 60 percent more bought a longer explanation and a worse label.

## Reading the table as a new user

- **Cheapest working path:** luna at low effort. Under half a dollar, twenty minutes, a correct algorithm with a thin explanation and a draft label. Good for "does this pipeline work on my machine".
- **Best hosted result here:** terra at medium effort. About five dollars, twenty minutes, verified label, the most complete tests. Comparable to the Qwen reference in verified content, with a shorter explanation.
- **The reference:** Qwen 27B self-hosted. Twelve times slower, the most thorough explanation and claims coverage, draft label because of one metric-sanity flag every run shares.
- **Spending more reasoning on a small model** did not help. The model tier mattered more than the effort knob.

Four runs of one paper is a small sample. Treat this as a frame of reference, not a benchmark.

## Pipeline follow-ups this exposed

These are pipeline gaps, not model faults:

- A stage-2c check that the pluggable's return type matches the taxonomy's declared interface (`List[int]` here). Would have caught the tensor return before the notebook was written.
- The stage-1 reviewer's probe-reference check is severity "important", so it never routes for a fix, and the same defect resurfaces as critical three stages later. Routing that check class at stage 1 would fix it in one cheap analyzer redispatch. Two of the three OpenAI runs hit exactly this.
- The loop microharness re-executes the notebook's loop cell with a fixed table of known names. Terra's cell used `F` (torch.nn.functional, imported in an earlier cell) and the probe gave up. Seeding the table from the notebook's own imports would make that check run.

## How to add a run to this table

Copy the finished `r2c_runs/<slug>/` folder here, then pull the numbers:

1. Wall time, cost tokens, label, judge count: the run's row in `r2c_runs/_history/ledger.jsonl` (not copied into the run folder) (`wall_seconds`, `tokens`, `delivery_label`, `judge_decisions`). A resumed run has one row per invocation; the last row's `wall_seconds` and `tokens` are cumulative.
2. Median agent turn: `elapsed_s` on `agent_dispatch_completed` events in `.pipeline/run_events.jsonl`.
3. Checks, claims, learning curve, review findings: the "Why this label", "Verification", and "Automated checks" sections of `REPORT.md`.
4. Code, tests, METHOD.md sizes: `wc -l method/*.py method/tests/*.py`, `wc -w METHOD.md`, and the `eq-` and `alg-` anchors in METHOD.md's headings.
5. Validator rejections: the "Where the run had trouble" section of `REPORT.md`.
