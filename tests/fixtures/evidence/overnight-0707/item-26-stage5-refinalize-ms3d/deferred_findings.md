# Findings deferred to manual review

Written by `scripts/run_pipeline.py` at Stage 5 routing. Critical findings auto-routed via fix-mode and are not listed here. The findings below are surfaced for human judgment.

## Important findings (3)

### F002 (important, target_agent=method-coder)

**Location:** method/method.py

**Description:** detector_weighing() correctly computes source-target similarity weights (e.g., Waymo=1.476, Lyft=1.476, nuScenes=0.048 for Waymo target). These weights are returned by run_ensemble_detections. However, the KBF fusion call at line 302 passes weights=None, causing uniform weighting across all detection sets. The computed weights are dead code in the runtime path. The acceptable approximation (equal weighting) is permitted by spec.element detector-weighing-by-lidar-density, but it must be disclosed when the weighing code is present but unused.

**Proposed fix:** Either (a) pass the computed detector_weights spread across each detector's VMFIxTTA detection sets to kbf_fuse_boxes, or (b) document the equal-weighting approximation in run_ensemble_detections()'s docstring and the notebook's honesty note, and either remove the detector_weighing() call or make its return value visibly unused.

---

### F003 (important, target_agent=notebook-generator)

**Location:** notebook.ipynb

**Description:** The notebook's 'NOT do' section states 'this notebook runs one round on 50 synthetic frames' and 'Only 1 seed is used — the paper averages over 4 self-training rounds with tuned per-round thresholds.' This characterizes the departure as 'one round is executed' (implying the loop exists but is only called once) rather than 'the loop structure was not implemented in the notebook run.' multi_round_self_train() now exists in method.py but the notebook's §5 only calls generate_pseudo_labels() once and retrain_on_pseudo_labels() once — it does not call multi_round_self_train() or demonstrate the iterative loop. The §5.4 markdown table describes the 4-round structure conceptually but does not execute it.

**Proposed fix:** Add a bullet to the 'NOT do' section that explicitly states the multi-stage self-training LOOPS is not executed: 'The multi_stage_self_train() function exists in method.py and is demonstrated conceptually in §5.4, but the notebook §5 run only executes a single generate_pseudo_labels() pass followed by one retrain_on_pseudo_labels() call. The iterative loop structure with per-round ensemble reduction (4→2 detectors) and threshold relaxation (s_pos: 0.9→0.6, n_pos: 5→2) is described but not executed.'

---

### F005 (important, target_agent=package-scaffolder)

**Location:** method/__init__.py

**Description:** multi_round_self_train() was added to method.py (resolving F001) but is not imported or exported in __init__.py. The __all__ list contains 13 entries but omits multi_round_self_train. A researcher importing from the package (e.g., 'from method import multi_round_self_train') would get an ImportError. The notebook's §1 import cell also does not import it, and cannot because it isn't exported. This is a regression from F001's fix: the function exists but is inaccessible from the public API.

**Proposed fix:** Add 'multi_round_self_train' to the import from .method and to __all__. Per the build plan's content_rule ordering [pluggable_component.name, *method_helpers, *architecture_classes, *training_functions, load_data], it belongs in the method_helpers section, after generate_pseudo_labels. The notebook's §1 import cell should also add the import.

---

## Completed stage-review caveats (7)

### stage_1_analyzer:caveat:1 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_1_analyzer.json

**Description:** The spec correctly blocks pre-trained detectors and large-scale datasets as must_implement / can_approximate respectively. Downstream stages (architecture-coder, method-coder) will need to handle detector-as-input contract and mock data carefully — this is expected and within the approved approximation scope.

---

### stage_2c_method:caveat:1 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_2c_method.json

**Description:** Single-box-per-frame simplification in KBF (no spatial clustering before KDE) is documented at line 141 and is acceptable at smoke scale. At paper scale, kbf_fuse_boxes would need spatial clustering to handle multiple objects per frame.

---

### stage_2c_method:caveat:2 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_2c_method.json

**Description:** The minimum-point filter (< 1 point in box) from Section 4.5 is omitted at smoke scale (noted at lines 673-675). This is acceptable since mock point clouds are dense.

---

### stage_2c_method:caveat:3 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_2c_method.json

**Description:** The tracker uses a simplified greedy nearest-neighbor approach rather than a full Kalman Filter with prediction/update steps. The methodology contract permits 'simplified KF tracker' at smoke scale, and the static/dynamic classification logic (begin-to-end distance + variance) matches the paper.

---

### stage_2x_params:caveat:1 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_2x_params.json

**Description:** The spec_default source for s_pos=0.9, n_pos=5, kbf_bandwidth=1.0 correctly avoids claiming paper provenance, but the paper does discuss these parameters conceptually (s_pos in Section 4.4, N_pos in Section 4.4.1, bandwidth h in Section 4.2). If the paper's GitHub repo or supplementary materials contain concrete values, upstream spec or paper_map should be enriched accordingly.

---

### stage_3a_notebook:caveat:1 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_3a_notebook.json

**Description:** The kalman_track function in method.py uses greedy nearest-neighbor assignment rather than a full Kalman Filter (no predict/update, no state covariance). The notebook §4.5 describes it as 'Kalman Filter-based' matching the function name, but the simplification is acceptable per the contract's demo_scale_implementation allowance. This is a stage_2c_method topic, not a notebook narrative issue.

---

### stage_3a_notebook:caveat:2 (nice-to-have, target_agent=human)

**Location:** stage_review_stage_3a_notebook.json

**Description:** Section 4.9 demo cell passes empty lists for static_vehicle_boxes, tracked_vehicle_boxes, and dynamic_pedestrian_boxes, so only VMFI boxes are exercised in the filtering demo. The full four-source combination is exercised in §5.1's end-to-end run. The §4.9 cell is a standalone demo, not the full pipeline.

---

