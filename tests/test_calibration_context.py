"""Closed pure dispatcher coverage for R2C-091 calibration contexts."""

from __future__ import annotations

import pytest

from probes.calibration_context import dispatch_calibration_context


@pytest.mark.parametrize(
    ("scale", "observer", "expected"),
    [
        ("raw_pixel_unnormalized", "feature_magnitude", "raw"),
        ("pixel_zero_one", "feature_magnitude", "zero_one"),
        ("pixel_centered", "feature_magnitude", "centered"),
        ("standardized", "feature_magnitude", "centered"),
        ("unit_norm", None, None),
    ],
)
def test_typed_feature_contexts_dispatch_exactly(scale, observer, expected):
    route = dispatch_calibration_context({
        "calibration_context": {"kind": "feature_magnitude", "scale": scale},
    })
    assert route.kind == "feature_magnitude"
    assert route.source == "typed"
    assert route.scale == scale
    assert route.observer == observer
    assert route.expected_observation == expected


def test_typed_target_box_grid_selects_only_structural_box_observer():
    route = dispatch_calibration_context({
        "calibration_context": {
            "kind": "representation_convention",
            "convention": "target_box_grid",
        },
    })
    assert route.kind == "representation_convention"
    assert route.convention == "target_box_grid"
    assert route.observer == "target_box_grid"
    assert route.expected_observation == "grid_coordinates"


def test_typed_other_preserves_label_and_reason_byte_for_byte():
    route = dispatch_calibration_context({
        "calibration_context": {
            "kind": "other",
            "label": "Mixed meters + radians",
            "reason": "No one scalar observation surface exists.",
        },
    })
    assert route.kind == "other"
    assert route.observer is None
    assert route.label == "Mixed meters + radians"
    assert route.reason == "No one scalar observation surface exists."


@pytest.mark.parametrize(
    ("legacy", "kind", "scale", "observer"),
    [
        ("raw pixel", "feature_magnitude", "raw_pixel_unnormalized", "feature_magnitude"),
        ("raw [0,255]", "feature_magnitude", "raw_pixel_unnormalized", "feature_magnitude"),
        ("unnormalized", "feature_magnitude", "raw_pixel_unnormalized", "feature_magnitude"),
        ("raw_pixel_unnormalized", "feature_magnitude", "raw_pixel_unnormalized", "feature_magnitude"),
        ("zero-one", "feature_magnitude", "pixel_zero_one", "feature_magnitude"),
        ("[0,1] normalized", "feature_magnitude", "pixel_zero_one", "feature_magnitude"),
        ("min-max", "feature_magnitude", "pixel_zero_one", "feature_magnitude"),
        ("pixel_zero_one", "feature_magnitude", "pixel_zero_one", "feature_magnitude"),
        ("centered", "feature_magnitude", "pixel_centered", "feature_magnitude"),
        ("[-1,1]", "feature_magnitude", "pixel_centered", "feature_magnitude"),
        ("standardized", "feature_magnitude", "standardized", "feature_magnitude"),
        ("z-score", "feature_magnitude", "standardized", "feature_magnitude"),
        ("unit_norm", "feature_magnitude", "unit_norm", None),
        ("target-box grid", "representation_convention", None, "target_box_grid"),
        ("bev_grid_coordinates", "representation_convention", None, "target_box_grid"),
        ("bev_grid_128x128", "representation_convention", None, "target_box_grid"),
        ("BEV grid cells", "representation_convention", None, "target_box_grid"),
    ],
)
def test_closed_legacy_mapping(legacy, kind, scale, observer):
    route = dispatch_calibration_context({"assumes_data_scale": legacy})
    assert route.source == "legacy"
    assert route.kind == kind
    assert route.scale == scale
    assert route.observer == observer


@pytest.mark.parametrize(
    "legacy",
    [
        "148 weekly steps (retail dataset)",
        "weekly observations",
        "graph_density_dependent",
        "cosine_similarity_normalized",
        "meters",
        "physical_units_meters_per_second",
        "radians",
        "system-dependent_norm",
        "unknown experimental calibration",
    ],
)
def test_legacy_noncontexts_preserve_exact_label_as_other(legacy):
    route = dispatch_calibration_context({"assumes_data_scale": legacy})
    assert route.kind == "other"
    assert route.observer is None
    assert route.label == legacy
    assert route.reason == (
        "legacy assumes_data_scale has no supported calibration observer"
    )


def test_normalized_substrings_never_guess_zero_one():
    for legacy in (
        "cosine_similarity_normalized",
        "unit normalized embeddings",
        "normalization-dependent coefficient",
    ):
        route = dispatch_calibration_context({"assumes_data_scale": legacy})
        assert route.kind == "other"
        assert route.expected_observation is None


def test_graph_statistic_is_not_a_supported_typed_arm_yet():
    route = dispatch_calibration_context({
        "calibration_context": {
            "kind": "graph_statistic",
            "statistic": "average_degree",
        },
    })
    assert route.kind == "other"
    assert route.observer is None
    assert route.label == "graph_statistic"
    assert "unsupported" in route.reason


def test_mixed_typed_and_legacy_authority_is_unprobeable():
    route = dispatch_calibration_context({
        "calibration_context": {"kind": "feature_magnitude", "scale": "pixel_zero_one"},
        "assumes_data_scale": "raw_pixel_unnormalized",
    })
    assert route.kind == "other"
    assert route.source == "conflict"
    assert route.observer is None


def test_inactive_null_carrier_does_not_create_a_false_conflict():
    typed = dispatch_calibration_context({
        "calibration_context": {
            "kind": "feature_magnitude",
            "scale": "pixel_zero_one",
        },
        "assumes_data_scale": None,
    })
    legacy = dispatch_calibration_context({
        "calibration_context": None,
        "assumes_data_scale": "raw_pixel_unnormalized",
    })

    assert typed.source == "typed"
    assert typed.scale == "pixel_zero_one"
    assert legacy.source == "legacy"
    assert legacy.scale == "raw_pixel_unnormalized"
