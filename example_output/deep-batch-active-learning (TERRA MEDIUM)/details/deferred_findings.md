# Findings deferred to manual review

Written by `scripts/run_pipeline.py` at Stage 5 routing. Critical findings auto-routed via fix-mode and are not listed here. The findings below are surfaced for human judgment.

## Nice-to-have findings (1)

### F001 (nice-to-have, target_agent=package-scaffolder)

**Location:** method/example_data/README.md — lines 3-6 and 27-28

**Description:** The README says the notebook calls load_data("example_data/") and that the MNIST fallback saves mnist_subset.pt at the example_data root. The notebook uses load_data() with its default directory, and data.py stores the fallback at example_data/_cache/mnist_subset.pt, so the suggested deletion or filename workaround does not describe the live loader.

**Proposed fix:** Describe the default load_data() call and the _cache/mnist_subset.pt location; explain that top-level user files take precedence over the cache.

---

## Completed stage-review caveats (2)

### stage_2c_method:caveat:1 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_2c_method.json

**Description:** The implementation uses the closed-form softmax cross-entropy derivative rather than constructing the scalar loss and differentiating it; this is mathematically equivalent to Equation (1).

---

### stage_3a_notebook:caveat:1 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_3a_notebook.json

**Description:** The notebook is intentionally unexecuted at this production stage; the smoke gate must confirm the observed final held-out accuracy exceeds chance.

---

