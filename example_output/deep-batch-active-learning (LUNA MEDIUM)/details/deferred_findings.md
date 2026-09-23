# Findings deferred to manual review

Written by `scripts/run_pipeline.py` at Stage 5 routing. Critical findings auto-routed via fix-mode and are not listed here. The findings below are surfaced for human judgment.

## Important findings (4)

### F001 (important, target_agent=analyzer)

**Location:** .pipeline/method_spec.json — critical_requirements.training.optimizer, lines 112-119

**Description:** The paper-map training protocol specifies the Adam variant of SGD, while the structured method spec still records the optimizer as plain SGD. The generated training implementation uses Adam, so the paper-truth contract and the produced package disagree.

**Proposed fix:** Record Adam explicitly in critical_requirements.training.optimizer, optionally noting that the paper calls it an Adam variant of SGD.

---

### F002 (important, target_agent=notebook-generator)

**Location:** notebook.ipynb — markdown cells for the opening departures list and §3.3 Training

**Description:** The notebook says it uses the package's SGD implementation and that the training function uses the supplied SGD protocol, but method/training.py constructs torch.optim.Adam. This gives the researcher the wrong account of the runnable training protocol even though the code follows the paper's Adam wording.

**Proposed fix:** Describe the implementation as Adam, and distinguish it from any smoke-scale epoch-cap or architecture departures.

---

### F003 (important, target_agent=architecture-coder)

**Location:** method/model.py — module docstring, lines 1-19

**Description:** The module names the flattened MLP substitution but does not state that changing the paper's image backbones to an MLP changes the learned representation and inductive bias, so acquisition behavior is not a reproduction of the paper's CNN/ResNet benchmark claims. The existing hook warning does not cover this fidelity limitation.

**Proposed fix:** Add a concise Known departures note explaining that the flattened MLP is a smoke-scale architecture substitution and is not benchmark-faithful to the paper's image-backbone results.

---

### F004 (important, target_agent=parameter-deriver)

**Location:** .pipeline/params.json — params.pool_size.reasoning, lines 24-28

**Description:** The pool-size provenance says that diversity baselines are meaningfully tested at this scale, but this single-method package intentionally implements and runs no baselines. The claim conflicts with the scope contract and can make the smoke setting sound like a comparison experiment.

**Proposed fix:** Replace the baseline language with a statement that the pool ratio preserves a meaningful pool for BADGE's own diversity-sensitive acquisition checks.

---

## Open important findings from earlier stage reviews (2)

### F001 (important, target_agent=analyzer)

**Location:** .pipeline/method_spec.json — critical_requirements.training.optimizer, lines 112-120

**Raised by:** the stage_1_analyzer stage review and is still open (pending). No fix was routed, because the stage fix loops act on critical findings only.

**Description:** The paper specifies training with the Adam variant of SGD (paper Section 4, line 138 and paper-map element hyp-training-optimizer), but method_spec.json records the optimizer as plain SGD. This paper-truth field is too generic and can cause downstream training to diverge from the stated protocol.

**Proposed fix:** Set critical_requirements.training.optimizer to Adam, or explicitly record Adam as the optimizer while noting that it is an SGD variant. Preserve the paper's optimizer rather than a broader family label.

---

### F001 (important, target_agent=architecture-coder)

**Location:** method/model.py — module docstring, lines 1-19

**Raised by:** the stage_2b_architecture stage review and is still open (pending). No fix was routed, because the stage fix loops act on critical findings only.

**Description:** The module clearly says it ships a two-layer flattened-input MLP instead of reproducing the paper's benchmark backbone, but it only describes the silent failure of omitting the embedding hook. It does not explain that the MLP-on-image substitution changes the representation/inductive-bias setting and therefore limits fidelity to the paper's CNN benchmark claims.

**Proposed fix:** Add a concise Known departures note to the module docstring stating that the flattened MLP is a smoke-scale substitution for the paper's image backbones, and that its learned representation and resulting acquisition behavior are not a reproduction of the CNN benchmark; retain the existing hook-failure warning. 

---

## Completed stage-review caveats (4)

### stage_2b_architecture:caveat:1 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_2b_architecture.json

**Description:** The supplied spec requires SGD, while the paper's Section 4 experimental text describes Adam; this review judged training protocol against the supplied spec and found the artifact compliant.

---

### stage_2c_method:caveat:1 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_2c_method.json

**Description:** The shipped classifier is an MLP demonstration architecture rather than the paper's benchmark-specific CNN backbones; the reviewed method path nevertheless preserves the required final-layer gradient-embedding mechanism.

---

### stage_2x_params:caveat:1 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_2x_params.json

**Description:** The paper's learning-rate value is conditional on image versus non-image data; the generic parameter entry uses the image-data value 0.001 and documents it as system_inferred rather than encoding a dataset-specific map.

---

### stage_3a_notebook:caveat:1 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_3a_notebook.json

**Description:** The notebook intentionally uses smoke-scale defaults, one seed, five rounds, and a flattened MLP rather than the paper's full SVHN benchmark protocol; these departures are explicitly disclosed in the opening and parameter sections.

---

