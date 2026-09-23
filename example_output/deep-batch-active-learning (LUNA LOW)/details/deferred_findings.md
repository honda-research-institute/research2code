# Findings deferred to manual review

Written by `scripts/run_pipeline.py` at Stage 5 routing. Critical findings auto-routed via fix-mode and are not listed here. The findings below are surfaced for human judgment.

## Open important findings from earlier stage reviews (1)

### F001 (important, target_agent=analyzer)

**Location:** .pipeline/method_spec.json — methodology_replication_contract.elements[kmeanspp-gradient-selection].verification_probe_refs

**Raised by:** the stage_1_analyzer stage review and is still open (pending). No fix was routed, because the stage fix loops act on critical findings only.

**Description:** The contract references verification probe `AL-kmeanspp-incremental-distances`, but the matched batch_acquisition taxonomy declares only the exact probes `al_loop.acquisition_contract`, `al_loop.demo_config_reachability`, `al_loop.eval_label_alignment`, `al_loop.fresh_retrain`, `al_loop.microharness`, `al_stage1.coreset_construction`, and `claims.contribution_floor`. This leaves the core k-MEANS++ methodology requirement unverifiable through the declared integration surface.

**Proposed fix:** Re-emit the analyzer spec with the k-MEANS++ element bound to the taxonomy's applicable exact probe, most likely `al_loop.microharness` for incremental-distance behavior, while preserving the contract's paper-grounded requirements, approved approximations, and replication-feasibility status.

---

## Completed stage-review caveats (5)

### stage_1_analyzer:caveat:1 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_1_analyzer.json

**Description:** The paper contains extensive baseline comparisons, but under the single-method scope contract those comparisons are not implementation requirements and were not reviewed as core method elements.

---

### stage_2b_architecture:caveat:1 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_2b_architecture.json

**Description:** The compact MLP is a demonstration architecture rather than the paper's larger image backbones; this is compatible with the pluggable classifier contract, provided downstream selection consumes its embedding output.

---

### stage_2c_method:caveat:1 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_2c_method.json

**Description:** The zero-distance/equidistant k-MEANS++ fallback samples an unselected point because the paper's probability distribution is undefined when the distance sum is zero; this is a documented edge-case handling choice, not a mismatch at nondegenerate paper scale.

---

### stage_2x_params:caveat:1 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_2x_params.json

**Description:** The configuration is intentionally smoke-scaled: five rounds and a 12,000-example pool differ from the paper-scale benchmark, and the params file discloses those deviations.

---

### stage_3a_notebook:caveat:1 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_3a_notebook.json

**Description:** The notebook is intentionally smoke-scaled and explicitly discloses the reduced pool/round budget, one seed, and demo MLP instead of the paper's full benchmark protocol.

---

