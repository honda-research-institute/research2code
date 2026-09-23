"""Analyzer prompt guards for exact probe-to-obligation identities (R2C-087)."""

from pathlib import Path

from dispatch_templates import (
    ANALYZER_VERIFICATION_PROBE_GUIDANCE,
    STAGE_TASK_SUMMARIES,
    DispatchPaths,
    build_fix_mode_prompt,
)


ROOT = Path(__file__).resolve().parents[1]


def _paths() -> DispatchPaths:
    return DispatchPaths(
        spec="/tmp/run/.pipeline/method_spec.json",
        paper="/tmp/run/.pipeline/paper.md",
        paper_map="/tmp/run/.pipeline/paper_map.json",
        run_dir="/tmp/run",
        taxonomy_source="taxonomy:time_series_forecasting",
    )


def test_initial_retry_and_durable_analyzer_contract_match_probe_ref_rule():
    surfaces = {
        "initial": STAGE_TASK_SUMMARIES["stage_1_analyzer"],
        "chunk_retry": STAGE_TASK_SUMMARIES["stage_1_analyzer_retry"],
        "agent": (
            ROOT / ".opencode" / "agents" / "r2c-method-analyzer.md"
        ).read_text(encoding="utf-8"),
    }

    for name, text in surfaces.items():
        assert "verification_probe_refs" in text, name
        assert "semantic_checks[].probe" in text, name
        assert "effective taxonomy node" in text, name
        assert "genuinely" in text and "obligation" in text, name
        assert "shared" in text and "callable" in text, name
        assert "paper_element_ids" in text, name
        assert "[]" in text, name
        assert "invent" in text, name
        assert "archived" in text.lower(), name

    assert ANALYZER_VERIFICATION_PROBE_GUIDANCE in surfaces["initial"]
    assert ANALYZER_VERIFICATION_PROBE_GUIDANCE in surfaces["chunk_retry"]
    assert '"schema_version": "1.14.0"' in surfaces["agent"]
    assert (
        '"verification_probe_refs": ["al_loop.acquisition_contract"]'
        in surfaces["agent"]
    )


def test_analyzer_fix_prompt_reinjects_exact_probe_ref_guidance_only_for_owner():
    finding = {
        "id": "F001",
        "severity": "critical",
        "description": "An obligation declares an unknown probe ref.",
        "proposed_fix": "Use the exact effective-taxonomy identity.",
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

    assert analyzer_prompt.count(ANALYZER_VERIFICATION_PROBE_GUIDANCE) == 1
    assert ANALYZER_VERIFICATION_PROBE_GUIDANCE not in method_prompt
