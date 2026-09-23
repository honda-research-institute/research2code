"""Zoo acceptance tests for US-4 — scale-dependent-parameter mismatch.

Rows of the acceptance matrix executed for real:
- the snapshotted GBALD R_0 evidence (raw-pixel value on [0,1] data) must FAIL;
- the same run with R_0 rescaled by exact 2000/255 must PASS;
- the same value on genuinely raw-scaled data must PASS;
- a spec with no scale-dependent hyperparameters is N/A (no verdict);
- a missing param / unrecognized assumption / missing data come back
  `unprobeable`, never a silent pass.

The probe reads only files (spec, params, one data sample) and runs no method
code, so these tests need no torch and run in CI.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import probes.scale_mismatch as scale_mismatch_probe

REPO = Path(__file__).resolve().parent.parent
ZOO = REPO / "tests" / "fixtures" / "zoo"
FAIL_FIXTURE = ZOO / "gbald-r0-scale-mismatch"

from probes.scale_mismatch import (  # noqa: E402
    RAW,
    STANDARDIZED,
    ZERO_ONE,
    classify_data_scale,
    measure_data_scale,
    normalize_scale_name,
    probe_geometric_saturation,
    probe_scale_dependent_mismatch,
    values_equal,
)


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("raw_pixel_unnormalized", RAW),
    ("raw [0,255]", RAW),
    ("unnormalized", RAW),
    ("zero_one", ZERO_ONE),
    ("[0,1] normalized", ZERO_ONE),
    ("min-max", ZERO_ONE),
    ("standardized", STANDARDIZED),
    ("z-score", STANDARDIZED),
    ("[-1,1]", STANDARDIZED),
    ("", None),
    ("gibberish scale", None),
    (None, None),
    (2000, None),
])
def test_normalize_scale_name(text, expected):
    assert normalize_scale_name(text) == expected


@pytest.mark.parametrize("min_val,max_abs,frac_neg,expected", [
    (0.0, 1.0, 0.0, ZERO_ONE),       # [0,1] normalized
    (0.0, 255.0, 0.0, RAW),          # raw pixels
    (0.0, 28.0, 0.0, RAW),           # unnormalized distances
    (-2.1, 3.0, 0.40, STANDARDIZED),  # z-score
    (-1.0, 1.0, 0.45, STANDARDIZED),  # [-1,1]
    (0.0, 0.0, 0.0, ZERO_ONE),       # all-zero sample reads as [0,1]
])
def test_classify_data_scale(min_val, max_abs, frac_neg, expected):
    assert classify_data_scale(min_val, max_abs, frac_neg) == expected


def test_values_equal():
    assert values_equal(2000.0, 2000)       # int/float equivalence
    assert values_equal(7.843, 7.843)
    assert not values_equal(2000.0, 7.843)  # the rescaled-vs-raw case
    assert not values_equal(2000.0, 2000.5)  # not fooled by close-but-different
    assert values_equal("full set", "full set")  # non-numeric falls back to ==


# --------------------------------------------------------------------------
# Measurement against the real fixture sample
# --------------------------------------------------------------------------

def test_measure_fixture_is_zero_one():
    scale, reason = measure_data_scale(FAIL_FIXTURE)
    assert scale == ZERO_ONE, reason


# --------------------------------------------------------------------------
# Zoo acceptance matrix
# --------------------------------------------------------------------------

def _r0_verdict(verdicts):
    r0 = [v for v in verdicts if "R_0" in v.message or "R_0" in v.evidence]
    assert len(r0) == 1, [v.to_dict() for v in verdicts]
    return r0[0]


def test_fail_side_gbald_raw_value_on_normalized_data():
    """The headline catch: R_0=2000 (raw-pixel calibration) shipped onto
    [0,1] data → fail, naming the degeneration and the fix."""
    verdicts = probe_scale_dependent_mismatch(FAIL_FIXTURE)
    v = _r0_verdict(verdicts)
    assert v.verdict == "fail", v.to_dict()
    assert v.probe_id == "US-4"
    assert v.finding_class == "M-004"
    assert "rescale" in v.message.lower()


def test_pass_side_rescaled_value(tmp_path):
    """The only supported conversion is the exact raw-pixel /255 value."""
    run = tmp_path / "rescaled"
    shutil.copytree(FAIL_FIXTURE, run)
    params_path = run / ".pipeline" / "params.json"
    doc = json.loads(params_path.read_text())
    doc["params"]["R_0"]["value"] = 2000 / 255
    doc["params"]["R_0"]["source"] = "system_default"
    params_path.write_text(json.dumps(doc, indent=2))

    v = _r0_verdict(probe_scale_dependent_mismatch(run))
    assert v.verdict == "pass", v.to_dict()
    assert "rescaled" in v.message.lower()


def test_nearby_but_inexact_rescale_is_not_certified(tmp_path):
    """The historical rounded 7.843 value is evidence, not an exact formula."""
    run = tmp_path / "rounded"
    shutil.copytree(FAIL_FIXTURE, run)
    params_path = run / ".pipeline" / "params.json"
    doc = json.loads(params_path.read_text())
    doc["params"]["R_0"]["value"] = 7.843
    doc["params"]["R_0"]["source"] = "system_default"
    params_path.write_text(json.dumps(doc, indent=2))

    v = _r0_verdict(probe_scale_dependent_mismatch(run))
    assert v.verdict == "fail", v.to_dict()
    assert "exact" in v.message.lower()


def test_typed_feature_context_matches_legacy_probe_result(tmp_path):
    """A schema-1.13 typed feature context selects the same observer/verdict."""
    legacy = _r0_verdict(probe_scale_dependent_mismatch(FAIL_FIXTURE))
    run = tmp_path / "typed"
    shutil.copytree(FAIL_FIXTURE, run)
    spec_path = run / ".pipeline" / "method_spec.json"
    doc = json.loads(spec_path.read_text())
    entry = doc["critical_requirements"]["scale_dependent_hyperparameters"][0]
    entry.pop("assumes_data_scale")
    entry["calibration_context"] = {
        "kind": "feature_magnitude",
        "scale": "raw_pixel_unnormalized",
    }
    spec_path.write_text(json.dumps(doc, indent=2))

    typed = _r0_verdict(probe_scale_dependent_mismatch(run))
    assert typed.to_dict() == legacy.to_dict()


def test_pass_side_scale_actually_matches(tmp_path):
    """Same raw R_0=2000, but the delivered data is genuinely raw-scaled
    ([0,255]) → pass (the value is at its calibration; FP guard)."""
    run = tmp_path / "rawdata"
    shutil.copytree(FAIL_FIXTURE, run)
    data_path = run / "method" / "example_data" / "mnist_subset.json"
    doc = json.loads(data_path.read_text())
    # scale the [0,1] sample back up to raw pixels
    doc["x_pool"] = [[round(v * 255.0, 1) for v in row] for row in doc["x_pool"]]
    doc["x_test"] = [[round(v * 255.0, 1) for v in row] for row in doc["x_test"]]
    data_path.write_text(json.dumps(doc))

    v = _r0_verdict(probe_scale_dependent_mismatch(run))
    assert v.verdict == "pass", v.to_dict()
    assert "matching" in v.message.lower()


def test_na_when_no_scale_dependent_declared(tmp_path):
    """A spec without scale-dependent hyperparameters yields no verdict (N/A),
    never a synthetic pass/unprobeable that clutters the report."""
    run = tmp_path / "none"
    (run / ".pipeline").mkdir(parents=True)
    (run / ".pipeline" / "method_spec.json").write_text(json.dumps({
        "comparison": {"classification": {"id": "motion_planning"}},
        "critical_requirements": {},
    }))
    assert probe_scale_dependent_mismatch(run) == []


def test_unprobeable_when_param_absent(tmp_path):
    """Declared scale-dependent but missing from params.json → unprobeable,
    not a silent pass."""
    run = tmp_path / "absent"
    shutil.copytree(FAIL_FIXTURE, run)
    params_path = run / ".pipeline" / "params.json"
    doc = json.loads(params_path.read_text())
    del doc["params"]["R_0"]
    params_path.write_text(json.dumps(doc, indent=2))

    v = _r0_verdict(probe_scale_dependent_mismatch(run))
    assert v.verdict == "unprobeable"
    assert "absent" in v.message.lower()


def test_unprobeable_when_no_data_sample(tmp_path):
    """No data sample to measure → unprobeable with a named reason."""
    run = tmp_path / "nodata"
    shutil.copytree(FAIL_FIXTURE, run)
    shutil.rmtree(run / "method")

    v = _r0_verdict(probe_scale_dependent_mismatch(run))
    assert v.verdict == "unprobeable"
    assert "data scale" in v.message.lower()


def test_unprobeable_when_assumption_unrecognized(tmp_path):
    """An assumes_data_scale string the probe can't map → unprobeable, never a
    guess."""
    run = tmp_path / "weird"
    shutil.copytree(FAIL_FIXTURE, run)
    spec_path = run / ".pipeline" / "method_spec.json"
    doc = json.loads(spec_path.read_text())
    doc["critical_requirements"]["scale_dependent_hyperparameters"][0][
        "assumes_data_scale"] = "quantum_furlongs"
    spec_path.write_text(json.dumps(doc, indent=2))

    v = _r0_verdict(probe_scale_dependent_mismatch(run))
    assert v.verdict == "unprobeable"
    assert "not recognized" in v.message.lower()


@pytest.mark.parametrize(
    ("context", "label"),
    [
        ({"calibration_context": {
            "kind": "feature_magnitude", "scale": "unit_norm",
        }}, "unit_norm"),
        ({"calibration_context": {
            "kind": "other",
            "label": "Mixed meters + radians",
            "reason": "No one scalar observation surface exists.",
        }}, "Mixed meters + radians"),
        ({"assumes_data_scale": "148 weekly steps (retail dataset)"},
         "148 weekly steps (retail dataset)"),
        ({"assumes_data_scale": "graph_density_dependent"},
         "graph_density_dependent"),
        ({"assumes_data_scale": "cosine_similarity_normalized"},
         "cosine_similarity_normalized"),
        ({"assumes_data_scale": "meters"}, "meters"),
        ({"assumes_data_scale": "radians"}, "radians"),
        ({"assumes_data_scale": "unit normalized embeddings"},
         "unit normalized embeddings"),
    ],
)
def test_unprobeable_contexts_never_invoke_a_nearby_observer(
    tmp_path, monkeypatch, context, label,
):
    run = tmp_path / "unobservable"
    shutil.copytree(FAIL_FIXTURE, run)
    spec_path = run / ".pipeline" / "method_spec.json"
    doc = json.loads(spec_path.read_text())
    entry = doc["critical_requirements"]["scale_dependent_hyperparameters"][0]
    entry.pop("assumes_data_scale")
    entry.update(context)
    spec_path.write_text(json.dumps(doc, indent=2))

    def unexpected_observer(_run_dir):
        raise AssertionError("an unprobeable context selected an observer")

    monkeypatch.setattr(
        scale_mismatch_probe, "measure_data_scale", unexpected_observer,
    )
    monkeypatch.setattr(
        scale_mismatch_probe, "measure_box_coordinate_scale", unexpected_observer,
    )

    v = _r0_verdict(probe_scale_dependent_mismatch(run))
    assert v.verdict == "unprobeable", v.to_dict()
    assert label in v.message


# --------------------------------------------------------------------------
# US-4b — geometric-ranking scale-invariance (queue 11d Arm B, the folded 11f)
# --------------------------------------------------------------------------
#
# A faithful representativeness ranking (Eq-13, raw R_0/distance) is invariant to
# R_0's magnitude. A ranking that caps within-radius scores at inf (Eq-5's
# probability) is R_0-sensitive. US-4b perturbs R_0 through the actual pluggable.

# A select_batch that BALD-preselects then ranks the preselected candidates by
# representativeness. Two variants differ only in the ranking line.
_CAPPED_RANK = '''    within = (d <= R_0).any(dim=1)
    score = torch.full((len(xs),), float("inf"))   # BUG: caps within-radius (Eq-5)
    outside = ~within
    if outside.any():
        score[outside] = (R_0 / d[outside]).max(dim=1).values'''
_UNCAPPED_RANK = '''    score = (R_0 / d).max(dim=1).values   # raw Eq-13 ratio (scale-invariant)'''

_SELECT_BATCH_TMPL = '''
import torch


def select_batch(model, x_unlabeled, x_labeled, batch_size, seed,
                 R_0=2000.0, batch_returns=16, mc_samples=8,
                 core_set_size=6, eta=0.9):
    with torch.no_grad():
        info = model(x_unlabeled).softmax(dim=1).max(dim=1).values
    k = min(int(batch_returns), len(x_unlabeled))
    top = torch.topk(info, k).indices
    xs = x_unlabeled[top]
    d = torch.norm(xs.unsqueeze(1) - x_labeled.unsqueeze(0), dim=2)
{rank}
    order = torch.argsort(score, descending=True)[:min(int(batch_size), k)]
    return sorted(int(top[i].item()) for i in order)
'''

_NO_R0_SELECT_BATCH = '''
def select_batch(model, x_unlabeled, x_labeled, batch_size, seed):
    return sorted(range(min(int(batch_size), len(x_unlabeled))))
'''

_R0_DESC = "Distance threshold for the geometric probability model R_0/||x-D_j||."


def _us4b(verdicts):
    rows = [v for v in verdicts if v.probe_id == "US-4b"]
    assert len(rows) == 1, [v.to_dict() for v in verdicts]
    return rows[0]


def _al_run(tmp_path, select_batch_src, *, sdh_name="R_0", sdh_desc=_R0_DESC):
    run = tmp_path / "run"
    (run / "method").mkdir(parents=True)
    (run / ".pipeline").mkdir(parents=True)
    (run / "method" / "__init__.py").write_text(select_batch_src, encoding="utf-8")
    spec = {
        "comparison": {"classification": {"id": "active_learning/bayesian"},
                       "pluggable_component": {"name": "select_batch"}},
        "critical_requirements": {"scale_dependent_hyperparameters": [{
            "name": sdh_name, "paper_value": 2000.0,
            "assumes_data_scale": "raw_pixel_unnormalized",
            "description": sdh_desc, "paper_section": "S4.2"}]},
    }
    (run / ".pipeline" / "method_spec.json").write_text(json.dumps(spec))
    (run / ".pipeline" / "params.json").write_text(json.dumps({
        "schema_version": "1.0.0",
        "params": {"R_0": {"value": 7.843, "source": "system_default",
                           "paper_value": 2000.0, "reasoning": "rescaled"}}}))
    return run


def test_us4b_fails_capped_ranking(tmp_path):
    """The capped ranking is R_0-sensitive → fail (M-004). This is the real
    Stage-2 root cause behind the audit's 'GBALD degenerates to BALD'."""
    pytest.importorskip("torch")
    run = _al_run(tmp_path, _SELECT_BATCH_TMPL.format(rank=_CAPPED_RANK))
    v = _us4b(probe_geometric_saturation(run))
    assert v.verdict == "fail", v.to_dict()
    assert v.finding_class == "M-004"
    assert "changes with" in v.message.lower()


def test_us4b_passes_uncapped_ranking(tmp_path):
    """The faithful raw-ratio ranking is R_0-invariant → pass. Crucially this is
    the case the old saturation-fraction probe would have FALSE-failed."""
    pytest.importorskip("torch")
    run = _al_run(tmp_path, _SELECT_BATCH_TMPL.format(rank=_UNCAPPED_RANK))
    v = _us4b(probe_geometric_saturation(run))
    assert v.verdict == "pass", v.to_dict()


def test_us4b_unprobeable_when_param_not_a_pluggable_kwarg(tmp_path):
    """R_0 declared scale-dependent but select_batch takes no R_0 → unprobeable,
    never a guessed pass/fail."""
    pytest.importorskip("torch")
    run = _al_run(tmp_path, _NO_R0_SELECT_BATCH)
    v = _us4b(probe_geometric_saturation(run))
    assert v.verdict == "unprobeable"
    assert "not a parameter" in v.message.lower()


def test_us4b_na_for_non_distance_scale_param(tmp_path):
    """A scale-dependent parameter that is not a distance/radius (e.g. a norm
    bound) is out of scope for the ranking-invariance check → no verdict."""
    run = _al_run(
        tmp_path, _SELECT_BATCH_TMPL.format(rank=_UNCAPPED_RANK),
        sdh_name="w_max", sdh_desc="Upper bound on the L2 norm of the weights.")
    assert probe_geometric_saturation(run) == []


def test_us4b_na_when_no_scale_dependent_declared(tmp_path):
    run = tmp_path / "none"
    (run / ".pipeline").mkdir(parents=True)
    (run / ".pipeline" / "method_spec.json").write_text(json.dumps({
        "comparison": {"classification": {"id": "motion_planning"}},
        "critical_requirements": {},
    }))
    assert probe_geometric_saturation(run) == []


# --------------------------------------------------------------------------
# Grid-coordinate calibration (the bev-distill sigma tag, KD stub-kit fold)
# --------------------------------------------------------------------------

from probes.scale_mismatch import GRID, measure_box_coordinate_scale  # noqa: E402


def _grid_run(
    tmp_path, *, boxes, sigma_value=2.0, paper_value=2.0, typed=False,
):
    """A KD-shaped run declaring sigma calibrated in BEV grid cells, with a
    targets-convention data sample carrying the given box coordinates."""
    run = tmp_path / "gridrun"
    (run / ".pipeline").mkdir(parents=True)
    (run / "method" / "example_data").mkdir(parents=True)
    calibration = (
        {"calibration_context": {
            "kind": "representation_convention",
            "convention": "target_box_grid",
        }}
        if typed else {"assumes_data_scale": "bev_grid_coordinates"}
    )
    (run / ".pipeline" / "method_spec.json").write_text(json.dumps({
        "comparison": {"classification": {"id": "knowledge_distillation"}},
        "critical_requirements": {"scale_dependent_hyperparameters": [{
            "name": "sigma",
            "paper_value": paper_value,
            **calibration,
        }]},
    }))
    (run / ".pipeline" / "params.json").write_text(json.dumps({
        "schema_version": "1.0.0",
        "params": {"sigma": {"value": sigma_value, "source": "paper"}},
    }))
    (run / "method" / "example_data" / "sample.json").write_text(json.dumps({
        "targets_train": [{"boxes": boxes, "labels": [0] * len(boxes)}],
    }))
    return run


def test_normalize_recognizes_grid_coordinates():
    assert normalize_scale_name("bev_grid_coordinates") == GRID
    assert normalize_scale_name("BEV grid cells") == GRID


def test_grid_pass_when_boxes_are_grid_scaled(tmp_path):
    """bev-distill's live shape: sigma=2 (grid-cell calibration) and the
    delivered targets' boxes really are in grid units → pass, not the old
    'not recognized' unprobeable."""
    run = _grid_run(tmp_path, boxes=[[8.0, 12.0, 33.0, 21.0],
                                     [40.0, 9.0, 14.0, 6.0]])
    verdicts = probe_scale_dependent_mismatch(run)
    assert len(verdicts) == 1
    v = verdicts[0]
    assert v.verdict == "pass", v.to_dict()
    assert "matching" in v.message.lower()


def test_typed_target_box_grid_matches_legacy_probe_result(tmp_path):
    legacy_run = _grid_run(
        tmp_path / "legacy",
        boxes=[[8.0, 12.0, 33.0, 21.0], [40.0, 9.0, 14.0, 6.0]],
    )
    typed_run = _grid_run(
        tmp_path / "typed",
        boxes=[[8.0, 12.0, 33.0, 21.0], [40.0, 9.0, 14.0, 6.0]],
        typed=True,
    )
    legacy = probe_scale_dependent_mismatch(legacy_run)
    typed = probe_scale_dependent_mismatch(typed_run)
    assert [v.to_dict() for v in typed] == [v.to_dict() for v in legacy]


def test_grid_mismatch_is_unprobeable_without_declared_conversion(tmp_path):
    """A grid mismatch cannot be converted without representation metadata."""
    run = _grid_run(tmp_path, boxes=[[0.1, 0.4, 0.3, 0.2],
                                     [0.7, 0.2, 0.1, 0.05]])
    verdicts = probe_scale_dependent_mismatch(run)
    assert len(verdicts) == 1
    v = verdicts[0]
    assert v.verdict == "unprobeable", v.to_dict()
    assert not v.finding_class
    assert "will not invent" in v.message.lower()


def test_grid_unprobeable_without_targets_boxes(tmp_path):
    """No targets-convention boxes in the data sample → named unprobeable,
    never a silent pass."""
    run = _grid_run(tmp_path, boxes=[[8.0, 12.0, 33.0, 21.0]])
    (run / "method" / "example_data" / "sample.json").write_text(json.dumps({
        "x_pool": [[0.1, 0.2], [0.3, 0.4]],
    }))
    verdicts = probe_scale_dependent_mismatch(run)
    assert len(verdicts) == 1
    v = verdicts[0]
    assert v.verdict == "unprobeable", v.to_dict()
    assert "boxes" in v.message


def test_measure_box_scale_rejects_negative_coordinates(tmp_path):
    run = _grid_run(tmp_path, boxes=[[-3.0, 5.0, 2.0, 2.0]])
    scale, reason = measure_box_coordinate_scale(run)
    assert scale is None
    assert "negative" in reason
