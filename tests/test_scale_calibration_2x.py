"""Tests for the Stage 2.x deterministic scale calibration (hardening item 1
from the 2026-07-02 GBALD acceptance run).

The concrete case: R_0=2000 declared calibrated for raw [0,255] pixel
distances, shipped onto [0,1]-normalized data. Before this step the rescale
depended on reviewer luck (the 2026-07-02 reviewer certified preservation);
now it is deterministic, and undefined scale pairs go to needs_attention
instead of a guessed factor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from probes.scale_mismatch import probe_scale_dependent_mismatch
from scale_calibration_2x import assess

_PKG_TMPL = '''
import numpy as np


def load_data(pool_size=5000, n_test=1000, seed=0):
    rng = np.random.default_rng(seed)
    x_pool = {pool_expr}
    y_pool = rng.integers(0, 3, pool_size)
    return x_pool, y_pool, x_pool[:n_test], y_pool[:n_test]
'''

_ZERO_ONE_POOL = "rng.random((pool_size, 8))"
_RAW_POOL = "rng.random((pool_size, 8)) * 255.0"


def _make_run(tmp_path: Path, *, pool_expr: str, assumes: str,
              paper_value=2000.0, live_value=2000.0) -> Path:
    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)
    (run / "method").mkdir()
    (run / "method" / "__init__.py").write_text(
        _PKG_TMPL.format(pool_expr=pool_expr))
    (run / ".pipeline" / "method_spec.json").write_text(json.dumps({
        "critical_requirements": {"scale_dependent_hyperparameters": [{
            "name": "R_0",
            "paper_value": paper_value,
            "assumes_data_scale": assumes,
        }]},
    }))
    (run / ".pipeline" / "params.json").write_text(json.dumps({
        "schema_version": "1.0.0",
        "params": {"R_0": {"value": live_value, "source": "paper"}},
    }))
    return run


def _set_typed_context(run: Path, context: dict) -> None:
    path = run / ".pipeline" / "method_spec.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    entry = spec["critical_requirements"]["scale_dependent_hyperparameters"][0]
    entry.pop("assumes_data_scale")
    entry["calibration_context"] = context
    path.write_text(json.dumps(spec), encoding="utf-8")


def _write_target_loader(run: Path, boxes: list[list[float]], *, named=True):
    result = (
        "{'x_pool': x_pool, 'targets_train': "
        f"[{{'boxes': np.asarray({boxes!r}, dtype=float)}}]}}"
        if named else
        f"(x_pool, [{{'boxes': np.asarray({boxes!r}, dtype=float)}}])"
    )
    (run / "method" / "__init__.py").write_text(
        "import numpy as np\n\n"
        "def load_data(pool_size=256, n_test=64, seed=0):\n"
        "    x_pool = np.zeros((pool_size, 4), dtype=float)\n"
        f"    return {result}\n",
        encoding="utf-8",
    )


def _write_forbidden_loader(run: Path) -> None:
    (run / "method" / "__init__.py").write_text(
        "from pathlib import Path\n\n"
        "def load_data(**kwargs):\n"
        "    Path(__file__).with_name('loader_called.marker').write_text('called')\n"
        "    raise RuntimeError('irrelevant loader was called')\n",
        encoding="utf-8",
    )


def test_raw_pixel_value_on_normalized_data_rescales(tmp_path):
    run = _make_run(tmp_path, pool_expr=_ZERO_ONE_POOL,
                    assumes="raw_pixel_unnormalized")
    result = assess(run)
    assert result["applicable"], result
    assert result["measured_scale"] == "zero_one"
    entry = result["entries"][0]
    assert entry["status"] == "rescale", entry
    assert entry["new_value"] == 7.8431372549019605
    assert "255" in entry["derivation"]


def test_matching_scale_stays_at_calibration(tmp_path):
    run = _make_run(tmp_path, pool_expr=_RAW_POOL,
                    assumes="raw_pixel_unnormalized")
    result = assess(run)
    entry = result["entries"][0]
    assert entry["status"] == "at_calibration", entry


def test_exact_rescaled_value_is_left_alone_after_observation(tmp_path):
    run = _make_run(tmp_path, pool_expr=_ZERO_ONE_POOL,
                    assumes="raw_pixel_unnormalized",
                    live_value=2000.0 / 255.0)
    result = assess(run)
    entry = result["entries"][0]
    assert entry["status"] == "already_rescaled", entry
    assert entry["measured"] == "zero_one"


def test_rounded_rescale_needs_attention_after_observation(tmp_path):
    run = _make_run(tmp_path, pool_expr=_ZERO_ONE_POOL,
                    assumes="raw_pixel_unnormalized", live_value=7.843)

    result = assess(run)
    entry = result["entries"][0]

    assert entry["status"] == "needs_attention", entry
    assert entry["measured"] == "zero_one"
    assert "exact" in entry["reason"]


def test_non_pixel_raw_units_are_preserved_unprobeable_not_guessed(tmp_path):
    """Raw NON-pixel units (meters etc.) have no universal 255 factor —
    applying one would be a fabrication. The context stays unprobeable."""
    run = _make_run(tmp_path, pool_expr=_ZERO_ONE_POOL,
                    assumes="raw sensor range, unnormalized meters")
    result = assess(run)
    entry = result["entries"][0]
    assert entry["status"] == "unprobeable", entry
    assert entry["calibration_dispatch"]["label"] == (
        "raw sensor range, unnormalized meters"
    )
    assert "no supported calibration observer" in entry["reason"]


def test_typed_raw_pixel_context_preserves_exact_conversion(tmp_path):
    run = _make_run(tmp_path, pool_expr=_ZERO_ONE_POOL,
                    assumes="legacy-placeholder")
    _set_typed_context(run, {
        "kind": "feature_magnitude",
        "scale": "raw_pixel_unnormalized",
    })

    entry = assess(run)["entries"][0]

    assert entry["status"] == "rescale"
    assert entry["new_value"] == 7.8431372549019605
    assert entry["calibration_dispatch"]["source"] == "typed"


def test_typed_target_box_grid_observes_named_targets_not_features(tmp_path):
    run = _make_run(tmp_path, pool_expr=_ZERO_ONE_POOL,
                    assumes="legacy-placeholder", paper_value=2.0,
                    live_value=2.0)
    _set_typed_context(run, {
        "kind": "representation_convention",
        "convention": "target_box_grid",
    })
    _write_target_loader(run, [[8.0, 12.0, 33.0, 21.0]])

    result = assess(run)
    entry = result["entries"][0]

    assert entry["status"] == "at_calibration"
    assert entry["measured"] == "grid_coordinates"
    assert result["observations"]["target_box_grid"]["measured"] == (
        "grid_coordinates"
    )


def test_same_bev_fixture_routes_stage_2x_and_stage_5_to_target_boxes(tmp_path):
    run = _make_run(tmp_path, pool_expr=_ZERO_ONE_POOL,
                    assumes="legacy-placeholder", paper_value=2.0,
                    live_value=2.0)
    _set_typed_context(run, {
        "kind": "representation_convention",
        "convention": "target_box_grid",
    })
    boxes = [[8.0, 12.0, 33.0, 21.0]]
    _write_target_loader(run, boxes)
    example_data = run / "method" / "example_data"
    example_data.mkdir()
    (example_data / "bev_sample.json").write_text(json.dumps({
        "x_pool": [[0.1, 0.2, 0.3, 0.4]],
        "targets_train": [{"boxes": boxes}],
    }), encoding="utf-8")

    stage_2x = assess(run)
    stage_5 = probe_scale_dependent_mismatch(run)

    assert stage_2x["entries"][0]["status"] == "at_calibration"
    assert set(stage_2x["observations"]) == {"target_box_grid"}
    assert len(stage_5) == 1
    assert stage_5[0].verdict == "pass", stage_5[0].to_dict()
    assert "target_box_grid observer" in stage_5[0].message


def test_target_box_grid_normalized_boxes_are_known_bad_without_conversion(
    tmp_path,
):
    run = _make_run(tmp_path, pool_expr=_RAW_POOL,
                    assumes="bev_grid_coordinates", paper_value=2.0,
                    live_value=2.0)
    _write_target_loader(run, [[0.1, 0.4, 0.3, 0.2]])

    entry = assess(run)["entries"][0]

    assert entry["status"] == "needs_attention"
    assert entry["measured"] == "zero_one"
    assert "no deterministic conversion" in entry["reason"]


def test_changed_target_box_value_cannot_bypass_declared_observer(tmp_path):
    run = _make_run(tmp_path, pool_expr=_RAW_POOL,
                    assumes="bev_grid_coordinates", paper_value=2.0,
                    live_value=1.0)
    _write_target_loader(run, [[0.1, 0.4, 0.3, 0.2]])

    result = assess(run)
    entry = result["entries"][0]

    assert entry["status"] == "needs_attention"
    assert entry["measured"] == "zero_one"
    assert set(result["observations"]) == {"target_box_grid"}
    assert "no deterministic conversion" in entry["reason"]


def test_target_box_grid_never_guesses_tuple_position(tmp_path):
    run = _make_run(tmp_path, pool_expr=_RAW_POOL,
                    assumes="bev_grid_coordinates", paper_value=2.0,
                    live_value=2.0)
    _write_target_loader(run, [[8.0, 12.0, 33.0, 21.0]], named=False)

    entry = assess(run)["entries"][0]

    assert entry["status"] == "unprobeable"
    assert "named targets" in entry["reason"]


@pytest.mark.parametrize(
    "context",
    [
        {"kind": "feature_magnitude", "scale": "unit_norm"},
        {
            "kind": "other",
            "label": "mixed meters/radians",
            "reason": "no common scalar observation surface",
        },
    ],
)
def test_typed_unobservable_contexts_do_not_call_loader(tmp_path, context):
    run = _make_run(tmp_path, pool_expr=_ZERO_ONE_POOL,
                    assumes="legacy-placeholder")
    _set_typed_context(run, context)
    _write_forbidden_loader(run)

    entry = assess(run)["entries"][0]

    assert entry["status"] == "unprobeable"
    assert not (run / "method" / "loader_called.marker").exists()


def test_typed_other_live_value_difference_is_not_certified_as_rescaled(tmp_path):
    run = _make_run(tmp_path, pool_expr=_ZERO_ONE_POOL,
                    assumes="legacy-placeholder", live_value=7.0)
    _set_typed_context(run, {
        "kind": "other",
        "label": "mixed meters/radians",
        "reason": "no common scalar observation surface",
    })
    _write_forbidden_loader(run)

    entry = assess(run)["entries"][0]

    assert entry["status"] == "unprobeable"
    assert entry["calibration_dispatch"]["label"] == "mixed meters/radians"
    assert not (run / "method" / "loader_called.marker").exists()


@pytest.mark.parametrize(
    "legacy",
    [
        "148 weekly steps",
        "graph_density_dependent",
        "cosine_similarity_normalized",
        "meters",
        "radians",
        "system-dependent_norm",
        "unknown calibration",
    ],
)
def test_legacy_other_contexts_preserve_text_and_do_not_call_loader(
    tmp_path, legacy,
):
    run = _make_run(tmp_path, pool_expr=_ZERO_ONE_POOL, assumes=legacy)
    _write_forbidden_loader(run)

    entry = assess(run)["entries"][0]

    assert entry["status"] == "unprobeable"
    assert entry["calibration_dispatch"]["label"] == legacy
    assert not (run / "method" / "loader_called.marker").exists()


def test_no_declarations_is_not_applicable(tmp_path):
    run = _make_run(tmp_path, pool_expr=_ZERO_ONE_POOL,
                    assumes="raw_pixel_unnormalized")
    (run / ".pipeline" / "method_spec.json").write_text(
        json.dumps({"critical_requirements": {}}))
    result = assess(run)
    assert result["applicable"] is False


def test_broken_load_data_is_a_named_na_not_a_crash(tmp_path):
    run = _make_run(tmp_path, pool_expr=_ZERO_ONE_POOL,
                    assumes="raw_pixel_unnormalized")
    (run / "method" / "__init__.py").write_text(
        "def load_data(pool_size=10, n_test=5, seed=0):\n"
        "    raise RuntimeError('boom')\n")
    result = assess(run)
    assert result["applicable"] is False
    assert "boom" in result["reason"]
