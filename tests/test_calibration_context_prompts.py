"""Prompt guards for the typed calibration-context contract (R2C-091)."""

from pathlib import Path

from dispatch_templates import (
    ANALYZER_CALIBRATION_CONTEXT_GUIDANCE,
    REVIEWER_CALIBRATION_CONTEXT_GUIDANCE,
    STAGE_TASK_SUMMARIES,
    DispatchPaths,
    build_fix_mode_prompt,
)


ROOT = Path(__file__).resolve().parents[1]


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _paths() -> DispatchPaths:
    return DispatchPaths(
        spec="/tmp/run/.pipeline/method_spec.json",
        paper="/tmp/run/.pipeline/paper.md",
        paper_map="/tmp/run/.pipeline/paper_map.json",
        run_dir="/tmp/run",
        taxonomy_source="taxonomy:active_learning/batch_acquisition",
    )


def test_analyzer_surfaces_emit_the_same_closed_typed_context_grammar():
    surfaces = {
        "initial": STAGE_TASK_SUMMARIES["stage_1_analyzer"],
        "chunk_retry": STAGE_TASK_SUMMARIES["stage_1_analyzer_retry"],
        "agent": (
            ROOT / ".opencode" / "agents" / "r2c-method-analyzer.md"
        ).read_text(encoding="utf-8"),
    }

    for name, raw in surfaces.items():
        text = _normalized(raw)
        assert "v1.13+" in text, name
        assert "calibration_context" in text, name
        assert "assumes_data_scale" in text, name
        assert "MUST NOT emit" in text or "none emits legacy" in text, name
        assert "feature_magnitude" in text, name
        for scale in (
            "raw_pixel_unnormalized",
            "pixel_zero_one",
            "pixel_centered",
            "standardized",
            "unit_norm",
        ):
            assert scale in text, (name, scale)
        assert "representation_convention" in text, name
        assert "target_box_grid" in text, name
        assert "target boxes" in text and "feature rows" in text, name
        assert '"kind":"other"' in text or "`other`" in text, name
        assert "label" in text and "reason" in text, name
        assert "unprobeable" in text, name
        assert "not an escape hatch" in text, name
        assert "evaluation_protocol" in text, name
        assert "scale-free" in text.lower(), name
        assert "graph" in text.lower(), name

    assert ANALYZER_CALIBRATION_CONTEXT_GUIDANCE in surfaces["initial"]
    assert ANALYZER_CALIBRATION_CONTEXT_GUIDANCE in surfaces["chunk_retry"]
    assert '"schema_version": "1.14.0"' in surfaces["agent"]
    assert '"kind": "feature_magnitude"' in surfaces["agent"]
    assert '"scale": "raw_pixel_unnormalized"' in surfaces["agent"]


def test_reviewer_surfaces_follow_observer_routing_without_guessing_conversions():
    surfaces = {
        "dispatch": STAGE_TASK_SUMMARIES["stage_4_review"],
        "agent": (
            ROOT
            / ".opencode"
            / "agents"
            / "r2c-paper-fidelity-reviewer.md"
        ).read_text(encoding="utf-8"),
    }

    for name, raw in surfaces.items():
        text = _normalized(raw)
        assert "calibration" in text.lower(), name
        assert "never choose an observer from" in text, name
        assert "feature_magnitude" in text, name
        assert "raw_pixel_unnormalized" in text, name
        assert "pixel_zero_one" in text, name
        assert "1/255" in text, name
        assert "2000/255 = 7.8431372549019605" in text, name
        assert "unit_norm" in text and "unprobeable" in text, name
        assert "target_box_grid" in text, name
        assert "target boxes" in text and "feature rows" in text, name
        assert "other" in text and "unprobeable" in text, name
        assert "temporal" in text.lower(), name
        assert "scale-free" in text.lower(), name
        assert "graph" in text.lower(), name
        assert "producer-contract finding" in text, name

    assert REVIEWER_CALIBRATION_CONTEXT_GUIDANCE in surfaces["dispatch"]
    assert "factor = `2/255`" not in surfaces["agent"]
    assert "Other combinations: derive the factor" not in surfaces["agent"]
    assert "physical-unit scales" not in surfaces["agent"]
    assert "new_value\": 7.8431372549019605" in surfaces["agent"]
    assert "new_value\": 7.843," not in surfaces["agent"]


def test_analyzer_fix_prompt_reinjects_typed_context_guidance_only_for_owner():
    finding = {
        "id": "F001",
        "severity": "critical",
        "description": "A fresh spec emitted an unsupported context kind.",
        "proposed_fix": "Use one supported typed arm.",
    }

    analyzer_prompt = build_fix_mode_prompt(
        target_agent="r2c-method-analyzer",
        findings=[finding],
        paths=_paths(),
    )
    method_prompt = build_fix_mode_prompt(
        target_agent="r2c-method-coder",
        findings=[finding],
        paths=_paths(),
    )

    assert analyzer_prompt.count(ANALYZER_CALIBRATION_CONTEXT_GUIDANCE) == 1
    assert ANALYZER_CALIBRATION_CONTEXT_GUIDANCE not in method_prompt


def test_active_learning_ssot_uses_the_current_typed_calibration_contract():
    taxonomy = (
        ROOT / "docs" / "ssot" / "taxonomies.yaml"
    ).read_text(encoding="utf-8")
    content = (
        ROOT / "docs" / "ssot" / "content" / "active_learning.md"
    ).read_text(encoding="utf-8")

    assert "schema-1.13 typed\n                  calibration_context" in taxonomy
    assert "exact raw-pixel to zero-one paper_value / 255 conversion" in taxonomy
    assert "assumes_data_scale" not in taxonomy
    assert "calibration_context.kind: feature_magnitude" in content
    assert "paper_value / 255" in content
    assert "assumes_data_scale" not in content
