"""Tests for maintainer-side SSOT documentation tooling."""

from __future__ import annotations

import copy
from pathlib import Path


def test_integration_ssot_validates_against_runtime_constants():
    from refresh_ssot_docs import ROOT, load_integration_ssot, validate_ssot_data

    data = load_integration_ssot(ROOT)

    assert validate_ssot_data(data, ROOT) == []


def test_stage_display_name_drift_is_rejected():
    from refresh_ssot_docs import ROOT, load_integration_ssot, validate_ssot_data

    data = copy.deepcopy(load_integration_ssot(ROOT))
    data["stages"][1]["display_name"] = "Stage 1 - Drifted Name"

    errors = validate_ssot_data(data, ROOT)

    assert any("stage stage_1 display_name mismatch" in error for error in errors)


def test_generated_block_splice_preserves_surrounding_prose():
    from refresh_ssot_docs import ROOT, load_integration_ssot, render_integration_status, splice

    body = render_integration_status(load_integration_ssot(ROOT))
    text = "\n".join(
        [
            "# Human Heading",
            "",
            "Human-authored context stays.",
            "",
            "<!-- BEGIN AUTO: integration-status -->",
            "stale generated content",
            "<!-- END AUTO: integration-status -->",
            "",
            "Human-authored footer stays.",
            "",
        ]
    )

    refreshed = splice(text, "integration-status", body)

    assert "# Human Heading" in refreshed
    assert "Human-authored context stays." in refreshed
    assert "Human-authored footer stays." in refreshed
    assert "stale generated content" not in refreshed
    assert "## Runtime Stage Inventory" in refreshed


def test_diff_and_write_target_detect_and_clear_drift(tmp_path: Path):
    from refresh_ssot_docs import ROOT, diff_target, load_integration_ssot
    from refresh_ssot_docs import render_integration_status, write_target

    body = render_integration_status(load_integration_ssot(ROOT))
    target = tmp_path / "target.md"
    target.write_text(
        "\n".join(
            [
                "<!-- BEGIN AUTO: integration-status -->",
                "stale",
                "<!-- END AUTO: integration-status -->",
                "",
            ]
        ),
        encoding="utf-8",
    )

    diff = diff_target(target, "integration-status", body)
    assert "stale" in diff

    assert write_target(target, "integration-status", body) is True
    assert diff_target(target, "integration-status", body) == ""


def test_marker_audit_ignores_inline_and_fenced_examples(tmp_path: Path):
    from refresh_ssot_docs import TargetSpec, audit_auto_markers

    target = tmp_path / "target.md"
    target.write_text(
        "\n".join(
            [
                "<!-- BEGIN AUTO: integration-status -->",
                "generated",
                "<!-- END AUTO: integration-status -->",
                "",
            ]
        ),
        encoding="utf-8",
    )
    examples = tmp_path / "examples.md"
    examples.write_text(
        "\n".join(
            [
                "Inline example: `<!-- BEGIN AUTO: x -->`.",
                "",
                "```markdown",
                "<!-- BEGIN AUTO: y -->",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    errors = audit_auto_markers(
        tmp_path,
        [
            TargetSpec(
                id="integration_status",
                renderer="integration_status",
                target="target.md",
                tag="integration-status",
            )
        ],
    )

    assert errors == []


def test_taxonomy_role_view_file_set_detects_missing_file(tmp_path: Path):
    from refresh_ssot_docs import TargetSpec, diff_file_set

    target = TargetSpec(
        id="taxonomy_role_views",
        renderer="taxonomy_role_views",
        target="docs/generated/field-guides",
        tag="taxonomy-role-views",
        mode="file_set",
    )
    rendered = {
        Path("docs/generated/field-guides/demo/analyzer.md"): "generated\n",
    }

    diff = diff_file_set(tmp_path, target, rendered)

    assert "missing: docs/generated/field-guides/demo/analyzer.md" in diff


def test_taxonomy_role_views_are_ssot_sourced():
    from refresh_ssot_docs import ROOT
    from scripts import taxonomy_role_views

    rendered = taxonomy_role_views.render_role_views(ROOT)

    producer = rendered[Path("docs/generated/field-guides/active_learning/producer.md")]
    analyzer = rendered[Path("docs/generated/field-guides/active_learning/analyzer.md")]
    assert producer.startswith(taxonomy_role_views.ROLE_VIEW_MARKER)
    assert "Source taxonomy node: `TE-TS/active_learning`" in producer
    assert "FIELD_GUIDE.md" not in producer
    assert "## package_manifest" in producer
    assert "## arch_contract_requirements" in producer
    assert "## fingerprint" in analyzer
