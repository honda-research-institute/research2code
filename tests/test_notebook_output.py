"""Tests for scripts/validate_notebook_output.py.

Focus: the deterministic active-learning warm-start gate (M-003). The §5.1
acquisition loop must build a FRESH model each round; a loop that calls
`train_from_scratch` without a `build_model` in the same loop body warm-starts.
LLM reviewers repeatedly missed this, so it's a static check.
"""
from __future__ import annotations

import json

from validate_notebook_output import (
    _active_learning_pluggable_batch_size_errors,
    _bundle_consumption_errors,
    _al_loop_has_unused_final_acquisition,
    _al_loop_has_postmerge_count_for_premerge_eval,
    _al_loop_skips_last_acquisition,
    _al_loop_warm_starts,
    _called_names_within,
    _inverse_distance_markdown_contradictions,
    _markdown_loader_keyword_errors,
    _markdown_runtime_param_table_errors,
    _non_idempotent_cell_errors,
)


# --- the warm-start detector -------------------------------------------------

CORRECT_LOOP = """
labeled_idx = np.asarray(initial_indices)
unlabeled_idx = np.setdiff1d(np.arange(len(x_pool)), labeled_idx)
for r in range(cfg["num_rounds"]):
    batch_positions = select_batch(model, x_pool[unlabeled_idx], cfg["batch_size"], SEED + r)
    chosen = unlabeled_idx[np.asarray(batch_positions)]
    labeled_idx = np.union1d(labeled_idx, chosen)
    unlabeled_idx = np.setdiff1d(unlabeled_idx, chosen)
    model = build_model(input_dim=input_dim, n_classes=n_classes, hidden_dim=cfg["hidden_dim"])
    model = train_from_scratch(model, x_pool[labeled_idx], y_pool[labeled_idx], seed=SEED + r)
    acc = evaluate(model, x_test, y_test)
"""

WARM_START_LOOP = """
model = build_model(input_dim=input_dim, n_classes=n_classes, hidden_dim=cfg["hidden_dim"])
for round_idx in range(cfg["num_rounds"]):
    model = train_from_scratch(model, x_train_current, y_train_current, seed=SEED + round_idx)
    test_accuracy = evaluate(model, x_test, y_test)
    selected = select_batch(model, x_unlabeled_current, cfg["batch_size"], SEED + round_idx)
"""

BOOTSTRAP_ONLY = """
model = build_model(input_dim=input_dim, n_classes=n_classes, hidden_dim=cfg["hidden_dim"])
model = train_from_scratch(model, x_train, y_train, seed=SEED)
acc = evaluate(model, x_test, y_test)
"""

WHILE_WARM_START = """
model = build_model(input_dim, n_classes)
while len(labeled_idx) < budget:
    model = train_from_scratch(model, x_pool[labeled_idx], y_pool[labeled_idx], seed=0)
    chosen = select_batch(model, x_pool[unlabeled_idx], 100, 0)
"""

SKIPS_LAST_ACQUISITION = """
for r in range(cfg["num_rounds"]):
    acc = evaluate(model, x_test, y_test)
    if r < cfg["num_rounds"] - 1:
        batch_positions = select_batch(model, x_pool[unlabeled_idx], cfg["batch_size"], SEED + r)
        chosen = unlabeled_idx[np.asarray(batch_positions)]
        labeled_idx = np.union1d(labeled_idx, chosen)
"""

ACQUIRES_EVERY_ROUND = """
for r in range(cfg["num_rounds"]):
    batch_positions = select_batch(model, x_pool[unlabeled_idx], cfg["batch_size"], SEED + r)
    chosen = unlabeled_idx[np.asarray(batch_positions)]
    labeled_idx = np.union1d(labeled_idx, chosen)
    acc = evaluate(model, x_test, y_test)
"""

NEW_ACQUISITION_ROUND_SKELETON = """
labeled_idx = np.asarray(initial_indices)
unlabeled_idx = np.setdiff1d(np.arange(len(x_pool)), labeled_idx)
learning_curve = []

def train_eval_current(round_seed):
    model = build_model(input_dim=input_dim, n_classes=n_classes, hidden_dim=cfg["hidden_dim"])
    model = train_from_scratch(model, x_pool[labeled_idx], y_pool[labeled_idx], seed=round_seed)
    acc = evaluate(model, x_test, y_test)
    learning_curve.append((len(labeled_idx), acc))
    return model

model = train_eval_current(SEED)
for r in range(cfg["num_rounds"]):
    batch_positions = select_batch(model, x_pool[unlabeled_idx], cfg["batch_size"], SEED + r)
    chosen = unlabeled_idx[np.asarray(batch_positions)]
    labeled_idx = np.union1d(labeled_idx, chosen)
    unlabeled_idx = np.setdiff1d(unlabeled_idx, chosen)
    model = train_eval_current(SEED + r + 1)
"""

UNUSED_FINAL_ACQUISITION = """
learning_curve = []
for r in range(cfg["num_rounds"]):
    model = build_model(input_dim=input_dim, n_classes=n_classes, hidden_dim=cfg["hidden_dim"])
    model = train_from_scratch(model, x_pool[labeled_idx], y_pool[labeled_idx], seed=SEED + r)
    acc = evaluate(model, x_test, y_test)
    learning_curve.append((len(labeled_idx), acc))
    batch_positions = select_batch(model, x_pool[unlabeled_idx], cfg["batch_size"], SEED + r)
    chosen = unlabeled_idx[np.asarray(batch_positions)]
    labeled_idx = np.union1d(labeled_idx, chosen)
    unlabeled_idx = np.setdiff1d(unlabeled_idx, chosen)
"""

PREMERGE_EVAL_POSTMERGE_COUNT = """
learning_curve = []
for r in range(cfg["num_rounds"]):
    model = build_model(input_dim=input_dim, n_classes=n_classes, hidden_dim=cfg["hidden_dim"])
    model = train_from_scratch(model, x_pool[labeled_idx], y_pool[labeled_idx], seed=SEED + r)
    test_acc = evaluate(model, x_test, y_test)
    batch_positions = select_batch(model, x_pool[unlabeled_idx], cfg["batch_size"], SEED + r)
    chosen = unlabeled_idx[np.asarray(batch_positions)]
    labeled_idx = np.union1d(labeled_idx, chosen)
    unlabeled_idx = np.setdiff1d(unlabeled_idx, chosen)
    learning_curve.append((len(labeled_idx), test_acc))
"""

POSTMERGE_RETRAINED_EVAL = """
learning_curve = []
for r in range(cfg["num_rounds"]):
    batch_positions = select_batch(model, x_pool[unlabeled_idx], cfg["batch_size"], SEED + r)
    chosen = unlabeled_idx[np.asarray(batch_positions)]
    labeled_idx = np.union1d(labeled_idx, chosen)
    unlabeled_idx = np.setdiff1d(unlabeled_idx, chosen)
    model = build_model(input_dim=input_dim, n_classes=n_classes, hidden_dim=cfg["hidden_dim"])
    model = train_from_scratch(model, x_pool[labeled_idx], y_pool[labeled_idx], seed=SEED + r)
    test_acc = evaluate(model, x_test, y_test)
    learning_curve.append((len(labeled_idx), test_acc))
"""

SEPARATE_EVAL_POINTS = """
for r in range(cfg["num_rounds"]):
    if r < cfg["num_eval_points"] - 1:
        batch_positions = select_batch(model, x_pool[unlabeled_idx], cfg["batch_size"], SEED + r)
"""


# --- non-idempotent cell gate (queue 11h) -----------------------------------

# The exact GBALD shape: re-running raises 'set' object has no attribute 'tolist'.
NON_IDEMPOTENT_SET_TOLIST = """
labeled_idx = set(labeled_idx.tolist())
all_pool_idx = set(range(len(x_pool)))
unlabeled_idx = list(all_pool_idx - labeled_idx)
"""

# Direct self-conversion without a wrapper is also non-idempotent.
NON_IDEMPOTENT_BARE_TOLIST = """
labeled_idx = labeled_idx.tolist()
"""

# Idempotent reassignments that MUST NOT be flagged (no type-exiting self-conv).
IDEMPOTENT_REASSIGNS = """
labeled_idx = np.union1d(labeled_idx, chosen)
model = model.to(device)
x = x.reshape(-1)
arr = np.asarray(arr)
labeled_idx = set(bootstrap_labeled_idx.tolist())
"""


def test_non_idempotent_set_tolist_detected():
    errors = _non_idempotent_cell_errors([NON_IDEMPOTENT_SET_TOLIST])
    assert any("labeled_idx" in e and "tolist" in e for e in errors), errors


def test_non_idempotent_bare_tolist_detected():
    errors = _non_idempotent_cell_errors([NON_IDEMPOTENT_BARE_TOLIST])
    assert any("labeled_idx" in e for e in errors), errors


def test_idempotent_reassignments_not_flagged():
    # Deriving from a STABLE source (bootstrap_labeled_idx) and ordinary
    # idempotent reassignments must not trip the gate.
    assert _non_idempotent_cell_errors([IDEMPOTENT_REASSIGNS]) == []


def test_idempotent_correct_loop_not_flagged():
    # The canonical correct loop reassigns labeled_idx via np.union1d(self, ...),
    # which is fine (self is a plain arg, not a type-exiting method call).
    assert _non_idempotent_cell_errors([CORRECT_LOOP]) == []


def test_warm_start_loop_detected():
    # trains in-loop, build_model only at bootstrap -> warm-start
    assert _al_loop_warm_starts([WARM_START_LOOP], train_fn="train_from_scratch",
                                build_fn="build_model") is True


def test_fresh_model_each_round_passes():
    # build_model inside the loop body -> retrain-from-scratch, no warm-start
    assert _al_loop_warm_starts([CORRECT_LOOP], train_fn="train_from_scratch",
                                build_fn="build_model") is False


def test_bootstrap_only_is_not_applicable():
    # training happens once, not in a loop -> check N/A (must not false-positive)
    assert _al_loop_warm_starts([BOOTSTRAP_ONLY], train_fn="train_from_scratch",
                                build_fn="build_model") is None


def test_while_loop_warm_start_detected():
    assert _al_loop_warm_starts([WHILE_WARM_START], train_fn="train_from_scratch",
                                build_fn="build_model") is True


def test_split_across_cells_bootstrap_then_warm_loop():
    # realistic: bootstrap build+train in one cell, warm-starting loop in another
    assert _al_loop_warm_starts([BOOTSTRAP_ONLY, WARM_START_LOOP],
                                train_fn="train_from_scratch", build_fn="build_model") is True


def test_unparseable_cell_is_skipped():
    # a placeholder-substitution failure shouldn't crash the detector
    assert _al_loop_warm_starts(["this is not <PYTHON>", CORRECT_LOOP],
                                train_fn="train_from_scratch", build_fn="build_model") is False


# --- the acquisition-budget detector ----------------------------------------

def test_acquisition_guard_that_skips_last_round_detected():
    assert _al_loop_skips_last_acquisition(
        [SKIPS_LAST_ACQUISITION],
        pluggable_name="select_batch",
    ) is True


def test_unconditional_acquisition_each_round_passes():
    assert _al_loop_skips_last_acquisition(
        [ACQUIRES_EVERY_ROUND],
        pluggable_name="select_batch",
    ) is False


def test_non_num_rounds_guard_does_not_false_positive():
    assert _al_loop_skips_last_acquisition(
        [SEPARATE_EVAL_POINTS],
        pluggable_name="select_batch",
    ) is False


def test_acquisition_round_skeleton_passes_budget_shape_checks():
    assert _al_loop_skips_last_acquisition(
        [NEW_ACQUISITION_ROUND_SKELETON],
        pluggable_name="select_batch",
    ) is False
    assert _al_loop_has_unused_final_acquisition(
        [NEW_ACQUISITION_ROUND_SKELETON],
        pluggable_name="select_batch",
    ) is False


def test_unused_final_acquisition_detected():
    assert _al_loop_has_unused_final_acquisition(
        [UNUSED_FINAL_ACQUISITION],
        pluggable_name="select_batch",
    ) is True


def test_premerge_eval_with_postmerge_label_count_detected():
    assert _al_loop_has_postmerge_count_for_premerge_eval(
        [PREMERGE_EVAL_POSTMERGE_COUNT],
        pluggable_name="select_batch",
    ) is True


def test_postmerge_retrain_before_eval_passes_alignment_check():
    assert _al_loop_has_postmerge_count_for_premerge_eval(
        [POSTMERGE_RETRAINED_EVAL],
        pluggable_name="select_batch",
    ) is False


def test_helper_routed_postmerge_retrain_passes_alignment_check():
    assert _al_loop_has_postmerge_count_for_premerge_eval(
        [NEW_ACQUISITION_ROUND_SKELETON],
        pluggable_name="select_batch",
    ) is False


# --- inverse-distance prose consistency -------------------------------------

def test_inverse_distance_farther_markdown_contradiction_detected():
    cells = [
        "Scores are computed as `R_0 / distance` and farther samples receive higher scores.",
        "Other text.",
    ]
    assert _inverse_distance_markdown_contradictions(cells) == [0]


def test_inverse_distance_far_from_markdown_contradiction_detected():
    cells = [
        "The score is `R_0 / distance`, so this picks samples far from the labeled set with high representativeness.",
        "Other text.",
    ]
    assert _inverse_distance_markdown_contradictions(cells) == [0]


def test_inverse_distance_closer_markdown_passes():
    cells = [
        "Scores are computed as `R_0 / distance`; closer samples receive higher scores.",
    ]
    assert _inverse_distance_markdown_contradictions(cells) == []


# --- the call-name extractor -------------------------------------------------

def test_called_names_within_picks_name_and_attribute_calls():
    import ast
    tree = ast.parse("x = build_model(); obj.train_from_scratch(); plain()")
    names = _called_names_within(tree)
    assert {"build_model", "train_from_scratch", "plain"} <= names


# --- params/notebook SSOT checks -------------------------------------------

def test_pluggable_batch_size_must_use_cfg_batch_size():
    errors = _active_learning_pluggable_batch_size_errors(
        [
            """
for r in range(cfg["num_rounds"]):
    selected = select_batch(
        model,
        x_unlabeled,
        batch_size=cfg["batch_output"],
        seed=SEED + r,
    )
"""
        ],
        pluggable_name="select_batch",
        expected_sig="select_batch(model, x_unlabeled, batch_size, seed) -> List[int]",
    )
    assert errors
    assert "batch_output" in errors[0]


def test_pluggable_batch_size_accepts_cfg_batch_size():
    assert _active_learning_pluggable_batch_size_errors(
        [
            """
for r in range(cfg["num_rounds"]):
    selected = select_batch(
        model,
        x_unlabeled,
        batch_size=min(cfg["batch_size"], len(x_unlabeled)),
        seed=SEED + r,
    )
"""
        ],
        pluggable_name="select_batch",
        expected_sig="select_batch(model, x_unlabeled, batch_size, seed) -> List[int]",
    ) == []


def test_runtime_param_table_mismatch_detects_stale_demo_value():
    params = {
        "R_0": {
            "value": 7.843,
            "source": "system_default",
            "paper_value": 2000.0,
        }
    }
    markdown = [
        """
| Parameter | Paper value | Demo value |
|---|---|---|
| `R_0` | 2000.0 | 20 |
"""
    ]
    errors = _markdown_runtime_param_table_errors(markdown, params)
    assert errors
    assert "R_0" in errors[0]
    assert "7.843" in errors[0]


def test_runtime_param_table_accepts_params_value():
    params = {
        "R_0": {
            "value": 7.843,
            "source": "system_default",
            "paper_value": 2000.0,
        }
    }
    markdown = [
        """
| Parameter | Paper value | Demo value |
|---|---|---|
| `R_0` | 2000.0 | 7.843 (rescaled from 2000.0) |
"""
    ]
    assert _markdown_runtime_param_table_errors(markdown, params) == []


def test_disclosed_demo_scale_override_table_is_accepted():
    # A table pairing a spec-default baseline with a demo-value override column
    # openly documents an intentional downscale (per demo_scale_implementation).
    # The demo column is exempt; the baseline column must match params.json.
    params = {
        "max_iterations": {"value": 10000, "source": "spec_default"},
        "initial_primitive_count": {"value": 200, "source": "spec_default"},
    }
    markdown = [
        """
| Parameter | Spec default | Demo value | Ratio |
|---|---|---|---|
| `max_iterations` | 10000 | 500 | 1/20 |
| `initial_primitive_count` | 200 | 50 | 1/4 |
"""
    ]
    assert _markdown_runtime_param_table_errors(markdown, params) == []


def test_disclosed_override_still_anchors_baseline_to_params():
    # The demo column is exempt, but a baseline column that misstates the real
    # deriver value is still caught — the disclosure must name the true value.
    params = {"max_iterations": {"value": 10000, "source": "spec_default"}}
    markdown = [
        """
| Parameter | Spec default | Demo value | Ratio |
|---|---|---|---|
| `max_iterations` | 8000 | 500 | 1/20 |
"""
    ]
    errors = _markdown_runtime_param_table_errors(markdown, params)
    assert errors
    assert "max_iterations" in errors[0]
    assert "Spec default" in errors[0]


def test_undisclosed_demo_value_without_baseline_still_checked():
    # No baseline column beside the demo column: the old contract still holds,
    # so a demo value that drifts from params.json is flagged.
    params = {"max_iterations": {"value": 10000, "source": "spec_default"}}
    markdown = [
        """
| Parameter | Demo value |
|---|---|
| `max_iterations` | 500 |
"""
    ]
    errors = _markdown_runtime_param_table_errors(markdown, params)
    assert errors
    assert "max_iterations" in errors[0]


def test_own_data_loader_example_must_match_method_signature(tmp_path):
    method_dir = tmp_path / "method"
    method_dir.mkdir()
    (method_dir / "data.py").write_text(
        "def load_data(path=None, *, pool_size=5000, n_test=1000, seed=0):\n"
        "    pass\n",
        encoding="utf-8",
    )
    markdown = [
        """
## 6. Use your own data

```python
x_pool, y_pool, x_test, y_test = load_data(data_path="/tmp/data.pt", seed=0)
```
"""
    ]
    errors = _markdown_loader_keyword_errors(markdown, method_dir)
    assert errors
    assert "data_path" in errors[0]


def test_own_data_loader_example_accepts_real_signature(tmp_path):
    method_dir = tmp_path / "method"
    method_dir.mkdir()
    (method_dir / "data.py").write_text(
        "def load_data(path=None, *, pool_size=5000, n_test=1000, seed=0):\n"
        "    pass\n",
        encoding="utf-8",
    )
    markdown = [
        """
## 6. Use your own data

```python
x_pool, y_pool, x_test, y_test = load_data(path="/tmp/data.pt", seed=0)
```
"""
    ]
    assert _markdown_loader_keyword_errors(markdown, method_dir) == []


# --- missing notebook_layout is coverage, not a defect (bev-distill 3.c) -----


def test_missing_notebook_layout_skips_section_check_but_runs_the_rest(tmp_path):
    # bev-distill 2026-07-01 halted at stage 3.c: the KD family declares no
    # notebook_layout on its extends chain, and the validator's early return
    # both failed the notebook on taxonomy coverage AND silently skipped every
    # check after it. A missing section contract must skip ONLY the
    # section-structure check; universal checks (here: the placeholder gate)
    # still run.
    import nbformat

    from validate_notebook_output import validate

    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell("# BEVDistill demo"),
        nbformat.v4.new_code_cell("x = 1  # PLACEHOLDER: fill_in"),
    ]
    nbformat.write(nb, tmp_path / "notebook.ipynb")
    spec = {"comparison": {"classification": {
        "id": "knowledge_distillation/detection/cross_modal"}}}

    errors = validate(spec, tmp_path, tmp_path)

    assert not any("notebook_layout.sections" in e for e in errors), errors
    assert any("PLACEHOLDER" in e for e in errors), (
        "checks after the section check no longer run: " + repr(errors))


# ---------------------------------------------------------------------------
# notebook_data_binding — the demo must consume the bundled dataset
# ---------------------------------------------------------------------------


def _seed_bundle(tmp_path):
    example = tmp_path / "method" / "example_data"
    example.mkdir(parents=True)
    (example / "PROVENANCE.json").write_text("{}", encoding="utf-8")
    return tmp_path


def test_bundle_never_called_is_flagged(tmp_path):
    """The pdfgnn 2026-08-04 shape: load_data imported, never called, demo
    trained on synthetic arrays generated in the notebook."""
    run_dir = _seed_bundle(tmp_path)
    cells = [
        "from method import load_data, forecast\n",
        "import numpy as np\ny = np.random.default_rng(42).normal(size=(20, 50))\n",
    ]
    errs = _bundle_consumption_errors(cells, run_dir)
    assert errs and "notebook_data_binding" in errs[0]


def test_bundle_called_passes(tmp_path):
    run_dir = _seed_bundle(tmp_path)
    cells = ["from method import load_data\ntables = load_data()\n"]
    assert _bundle_consumption_errors(cells, run_dir) == []


def test_bundle_called_via_module_attribute_passes(tmp_path):
    run_dir = _seed_bundle(tmp_path)
    cells = ["import method.data\ntables = method.data.load_data()\n"]
    assert _bundle_consumption_errors(cells, run_dir) == []


def test_no_bundle_no_requirement(tmp_path):
    (tmp_path / "method").mkdir()
    cells = ["import numpy as np\ny = np.zeros(3)\n"]
    assert _bundle_consumption_errors(cells, tmp_path) == []


# ---------------------------------------------------------------------------
# eval_split_range_overlap — the notebook seam of R2C-066's range arm
# ---------------------------------------------------------------------------


def _seed_params(tmp_path, **values):
    pipeline = tmp_path / ".pipeline"
    pipeline.mkdir(parents=True, exist_ok=True)
    (pipeline / "params.json").write_text(json.dumps({
        "params": {k: {"value": v} for k, v in values.items()}}),
        encoding="utf-8")
    return tmp_path


def test_split_leakage_flags_the_night3_shape(tmp_path):
    from validate_notebook_output import _split_leakage_errors

    run_dir = _seed_params(tmp_path, forecast_horizon=26)
    cells = [
        "K = cfg['forecast_horizon']\n"
        "demand_tensor = torch.tensor(demand_history)\n"
        "model = train_model(model, demand_history=demand_tensor)\n",
        "result = forecast(model=model, historical_demand=demand_history)\n",
        "test_actual = demand_history[:, T_total - K:T_total]\n"
        "rmse = np.mean((test_actual - result.predicted_means.numpy()) ** 2)\n",
    ]
    errs = _split_leakage_errors(cells, run_dir)
    assert errs and all("eval_split_range_overlap" in e for e in errs)


def test_split_leakage_silent_on_a_held_out_protocol(tmp_path):
    from validate_notebook_output import _split_leakage_errors

    run_dir = _seed_params(tmp_path, forecast_horizon=26)
    cells = [
        "K = cfg['forecast_horizon']\n"
        "split = T_total - K\n"
        "train_model(model, demand_history=demand_history[:, :split])\n"
        "result = forecast(model, historical_demand=demand_history[:, :split])\n"
        "rmse = np.mean((demand_history[:, split:T_total] - result.means) ** 2)\n",
    ]
    assert _split_leakage_errors(cells, run_dir) == []


# ---------------------------------------------------------------------------
# int_context_float — the certain-fix torch/numpy boundary lint
# ---------------------------------------------------------------------------


def test_int_context_flags_the_night3_cell30_shape():
    from validate_notebook_output import _int_context_errors

    cells = ["axes[0].hist(d.numpy(), bins=range(d.max().item() + 2))\n"]
    errs = _int_context_errors(cells)
    assert errs and "int_context_float" in errs[0] and ".item()" in errs[0]


def test_int_context_flags_true_division():
    from validate_notebook_output import _int_context_errors

    errs = _int_context_errors(["for i in range(n / 2):\n    pass\n"])
    assert errs and "true division" in errs[0]


def test_int_context_accepts_the_sanitized_fix():
    from validate_notebook_output import _int_context_errors

    cells = [
        "bins = range(int(d.max().item()) + 2)\n",
        "for i in range(n // 2):\n    pass\n",
        "for i in range(10):\n    pass\n",
    ]
    assert _int_context_errors(cells) == []


# ---------------------------------------------------------------------------
# calendar_fold_aggregation — chronology must survive aggregation
# ---------------------------------------------------------------------------


def _seed_multiyear_bundle(tmp_path):
    example = tmp_path / "method" / "example_data"
    example.mkdir(parents=True)
    (example / "PROVENANCE.json").write_text(json.dumps({
        "files": [{"file": "sales.csv",
                   "time_axis": {"column": "date", "steps_kept": 1092,
                                 "first_step": "2017-01-02",
                                 "last_step": "2019-12-29"}}]}),
        encoding="utf-8")
    return tmp_path


def test_calendar_fold_flags_week_of_year_groupby_on_multiyear_axis(tmp_path):
    from validate_notebook_output import _chronology_fold_errors

    run_dir = _seed_multiyear_bundle(tmp_path)
    cells = [
        "sales_df['week'] = sales_df['date'].dt.isocalendar().week.astype(int)\n"
        "weekly = sales_df.groupby(['product_id', 'week'])['sales'].sum()\n",
    ]
    errs = _chronology_fold_errors(cells, run_dir)
    assert errs and "calendar_fold_aggregation" in errs[0]


def test_calendar_fold_accepts_year_paired_grouping(tmp_path):
    from validate_notebook_output import _chronology_fold_errors

    run_dir = _seed_multiyear_bundle(tmp_path)
    cells = [
        "sales_df['week'] = sales_df['date'].dt.isocalendar().week\n"
        "sales_df['year'] = sales_df['date'].dt.year\n"
        "weekly = sales_df.groupby(['product_id', 'year', 'week']).sum()\n",
    ]
    assert _chronology_fold_errors(cells, run_dir) == []


def test_calendar_fold_silent_on_a_single_year_axis(tmp_path):
    from validate_notebook_output import _chronology_fold_errors

    example = tmp_path / "method" / "example_data"
    example.mkdir(parents=True)
    (example / "PROVENANCE.json").write_text(json.dumps({
        "files": [{"file": "sales.csv",
                   "time_axis": {"first_step": "2019-01-01",
                                 "last_step": "2019-12-29"}}]}),
        encoding="utf-8")
    cells = [
        "sales_df['week'] = sales_df['date'].dt.isocalendar().week\n"
        "weekly = sales_df.groupby(['product_id', 'week']).sum()\n",
    ]
    assert _chronology_fold_errors(cells, tmp_path) == []


# ---------------------------------------------------------------------------
# undeclared_notebook_dependency — imports must be installable
# ---------------------------------------------------------------------------


def test_undeclared_dependency_flags_the_pandas_gap(tmp_path):
    from validate_notebook_output import _undeclared_dependency_errors

    (tmp_path / "requirements.txt").write_text(
        "numpy>=1.24\ntorch>=2.0\nmatplotlib\njupyter\n", encoding="utf-8")
    cells = ["import pandas as pd\nimport numpy as np\n"]
    errs = _undeclared_dependency_errors(cells, tmp_path)
    assert errs and "pandas" in errs[0] \
        and "undeclared_notebook_dependency" in errs[0]


def test_undeclared_dependency_accepts_stdlib_method_and_aliases(tmp_path):
    from validate_notebook_output import _undeclared_dependency_errors

    (tmp_path / "requirements.txt").write_text(
        "numpy\nscikit-learn\n", encoding="utf-8")
    cells = [
        "import json\nimport numpy as np\nfrom method import forecast\n"
        "from sklearn.metrics import mean_squared_error\n",
    ]
    assert _undeclared_dependency_errors(cells, tmp_path) == []
