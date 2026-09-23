from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_taxonomy_pack_readiness_is_green_for_current_nodes():
    from scripts.validate_field_guides import build_readiness

    readiness = build_readiness(ROOT)

    assert readiness["schema_version"] == "2.0"
    assert readiness["status"] == "ready"
    assert readiness["summary"]["error_count"] == 0
    assert readiness["summary"]["warning_count"] == 0


def test_taxonomy_node_count_change_detector():
    """Deliberate change detector, separate from the green contract above:
    legitimate taxonomy growth trips THIS pin (update the number in the
    same diff that adds the node), never the is-green test."""
    from scripts.validate_field_guides import build_readiness

    readiness = build_readiness(ROOT)
    assert readiness["summary"]["node_count"] == 14


def test_readiness_reports_build_plan_and_taxonomy_coverage():
    from scripts.validate_field_guides import build_readiness

    readiness = build_readiness(ROOT)
    by_id = {row["paradigm"]: row for row in readiness["nodes"]}
    row = by_id["active_learning/bayesian"]
    coverage = {
        (item["consumer"], item["field"]): item["status"]
        for item in row["coverage"]
    }

    assert row["taxonomy_id"] == "TE-TS/active_learning/bayesian"
    assert coverage[("analyzer", "fingerprint.positive_signals")] == "ok"
    assert coverage[("producer/validators", "pluggable_component.signature_template")] == "ok"
    assert coverage[("package validators", "build_plan.package_manifest.files")] == "ok"
    assert coverage[("architecture validator", "build_plan.arch_contract_requirements.required_blocks")] == "ok"
    assert coverage[("notebook validator", "notebook_layout.sections")] == "ok"


def test_generated_readiness_report_is_taxonomy_sourced():
    from scripts.validate_field_guides import build_readiness, render_readiness_report

    report = render_readiness_report(build_readiness(ROOT))

    assert report.startswith("# Taxonomy Pack Readiness\n")
    assert "Generated from `docs/ssot/taxonomies.yaml`" in report
    assert "Legacy `FIELD_GUIDE.md` files are no longer an authoring or runtime source." in report
    assert "## Node Inventory" in report
    assert "active_learning/bayesian" in report


def test_role_views_keep_review_only_focus_out_of_producer_view():
    from scripts import taxonomy_role_views

    rendered = taxonomy_role_views.render_role_views(ROOT)

    producer = rendered[Path("docs/generated/field-guides/active_learning/bayesian/producer.md")]
    reviewer = rendered[Path("docs/generated/field-guides/active_learning/bayesian/reviewer.md")]

    assert "## semantic_checks" not in producer
    assert "AL-bayes-dropout-active" not in producer
    assert "## semantic_checks" in reviewer
    assert "AL-bayes-dropout-active" in reviewer


def test_generated_docs_check_matches_current_files():
    from scripts.validate_field_guides import check_generated_docs

    mismatches, readiness = check_generated_docs(ROOT)

    assert readiness["summary"]["error_count"] == 0
    assert mismatches == []


def test_generated_docs_check_detects_stale_readiness_report(tmp_path):
    from scripts.validate_field_guides import (
        check_generated_docs,
        expected_generated_docs,
    )

    report_path = tmp_path / "readiness.md"
    expected, _ = expected_generated_docs(ROOT, report_path)
    for rel_path, text in expected.items():
        rel_path.parent.mkdir(parents=True, exist_ok=True)
        rel_path.write_text(text, encoding="utf-8")

    mismatches, readiness = check_generated_docs(ROOT, report_path)
    assert readiness["summary"]["error_count"] == 0
    assert mismatches == []

    report_path.write_text(report_path.read_text(encoding="utf-8") + "\nSTALE\n", encoding="utf-8")
    mismatches, _ = check_generated_docs(ROOT, report_path)
    assert [(m.path, m.issue) for m in mismatches] == [(report_path.as_posix(), "stale")]
