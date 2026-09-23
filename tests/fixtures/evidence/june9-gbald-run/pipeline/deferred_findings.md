# Findings deferred to manual review

Written by `scripts/run_pipeline.py` at Stage 5 routing. Critical findings auto-routed via fix-mode and are not listed here. The findings below are surfaced for human judgment.

## Important findings (1)

### F002 (important, target_agent=notebook-generator)

**Location:** notebook.ipynb

**Description:** The title cell prose claims 'R_0=2000.0 calibrated for raw [0,255] pixel scale; MNIST is normalized to [0,1] here, so the geometric prior is approximate'. This is stale after the F001 fix: R_0 was rescaled to 7.843 to match the [0,1] normalization, so the geometric prior is no longer 'approximate' due to scale mismatch. The prose should be updated to reflect the rescaling decision documented in assumptions.md entry A001.

**Proposed fix:** Update the title cell markdown to reflect that R_0 was rescaled from 2000.0 to 7.843 to match the ToTensor() normalization. For example: 'R_0=7.843 (rescaled from paper's 2000.0 to match [0,1] normalization; see assumptions.md A001)'.

---

## Completed stage-review nice-to-have findings (1)

### F002 (nice-to-have, target_agent=method-coder)

**Location:** method/method.py

**Description:** Line 243 creates `rng = np.random.default_rng(seed)` but rng is never used anywhere in construct_ellipsoid_core_set. The only stochastic call in this function is KMeans(random_state=seed, ...) at line 254, which uses seed directly. The rng variable is dead code.

**Proposed fix:** Remove the unused rng variable from line 243.

---

## Completed stage-review caveats (7)

### stage_1_analyzer:caveat:1 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_1_analyzer.json

**Description:** The pluggable_component signature correctly follows the bayesian sub-paradigm override (includes x_labeled). The data_setup values form a consistent MNIST benchmark: initial_labeled=20, batch_size=100, num_rounds=6, total_budget=620 (arithmetic: 20 + 6*100 = 620). No methodology_replication_contract was present in the final spec (removed during fix-loop), so the contract baseline check did not apply.

---

### stage_2c_method:caveat:1 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_2c_method.json

**Description:** The ||L_0 - L||^2 term in construct_ellipsoid_core_set aggregates over all remaining candidates (scalar), rather than being per-candidate. This is consistent with the paper's formulation where L_0 and L are full log-likelihoods over the dataset. The differentiation across candidates comes solely from log p(y|x,theta). There are alternative interpretations but this reading is the natural one.

---

### stage_2c_method:caveat:2 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_2c_method.json

**Description:** The kmeans_centers value (default 10) is not specified by the paper. The field guide notes the paper uses k-means to initialize U, but doesn't state how many centers. The parameterization in construct_ellipsoid_core_set is good engineering practice; the hard-coding in select_batch is a minor API gap (F001).

---

### stage_2x_params:caveat:1 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_2x_params.json

**Description:** The eta param (source: paper) has a terse note ('Paper-stated value recovered from paper_map') and lacks a paper_section field, unlike all other paper-sourced params. Value is correct (0.9, from hyp-gbald-defaults in paper_map → Section 7.9) but the provenance entry is less auditable than it could be.

---

### stage_2x_params:caveat:2 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_2x_params.json

**Description:** The mc_samples param is labelled source: system_default with paper_value: 2000 and the reasoning correctly cites the paper's value. The params model does not enforce a paper_value field for system_default sources, so it is technically missing from schema validation, but the value is present in the file. The reviewer notes this is a positive: the provenance is clear.

---

### stage_3a_notebook:caveat:1 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_3a_notebook.json

**Description:** The notebook correctly avoids referencing initial_labeled (which has used_in_notebook: false in params.json) — the bootstrap uses construct_ellipsoid_core_set instead.

---

### stage_3a_notebook:caveat:2 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_3a_notebook.json

**Description:** The R_0 scale mismatch (R_0=2000 for raw [0,255] pixels but MNIST is normalized to [0,1]) is disclosed in the title departures section (§0 'what this notebook does NOT do') but not explicitly in the §4.2 geometric prior subsection. This is a documentation hygiene issue for a future iteration, not a correctness finding — the notebook's overall approach of disclosing it upfront is adequate.

---

