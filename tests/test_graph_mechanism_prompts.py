"""Producer/reviewer prompt guards for the R2C-088 graph contract."""

from pathlib import Path

from dispatch_templates import (
    ANALYZER_GRAPH_MECHANISM_GUIDANCE,
    GRAPH_MECHANISM_IMPLEMENTATION_GUIDANCE,
    GRAPH_MECHANISM_NOTEBOOK_GUIDANCE,
    REVIEWER_GRAPH_MECHANISM_GUIDANCE,
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


def test_analyzer_surfaces_require_the_typed_paper_grounded_graph_block():
    surfaces = {
        "initial": STAGE_TASK_SUMMARIES["stage_1_analyzer"],
        "chunk_retry": STAGE_TASK_SUMMARIES["stage_1_analyzer_retry"],
        "agent": (
            ROOT / ".opencode" / "agents" / "r2c-method-analyzer.md"
        ).read_text(encoding="utf-8"),
    }

    assert surfaces["initial"].count(ANALYZER_GRAPH_MECHANISM_GUIDANCE) == 1
    assert surfaces["chunk_retry"].count(ANALYZER_GRAPH_MECHANISM_GUIDANCE) == 1
    for name, text in surfaces.items():
        assert "homogeneous_graph_mechanism" in text, name
        assert "alignment_element_id" in text, name
        assert "graph_mechanism.alignment_prerequisite" in text, name
        assert "evaluation_control" in text, name
        assert "paper-justified" in text, name
        assert "R2C-083" in text, name
        assert "top-level" in text, name
        assert "dotted class" in text, name
        assert "invent" in text, name


def test_architecture_and_method_prompts_bind_exact_live_callables_and_params():
    surfaces = {
        "architecture_dispatch": STAGE_TASK_SUMMARIES["stage_2b_architecture"],
        "method_dispatch": STAGE_TASK_SUMMARIES["stage_2c_method"],
        "architecture_agent": (
            ROOT / ".opencode" / "agents" / "r2c-architecture-coder.md"
        ).read_text(encoding="utf-8"),
        "method_agent": (
            ROOT / ".opencode" / "agents" / "r2c-method-coder.md"
        ).read_text(encoding="utf-8"),
    }

    assert GRAPH_MECHANISM_IMPLEMENTATION_GUIDANCE in surfaces[
        "architecture_dispatch"
    ]
    assert GRAPH_MECHANISM_IMPLEMENTATION_GUIDANCE in surfaces["method_dispatch"]
    for name, text in surfaces.items():
        assert "homogeneous_graph_mechanism" in text, name
        assert "qualname" in text, name
        assert "top-level" in text, name
        assert "dotted class" in text, name
        assert "hardcoded" in text or "internal literal" in text, name
        assert "neighbor" in text, name
        assert "live" in text, name
        assert "identity" in text, name


def test_reviewer_keeps_every_graph_evidence_rung_separate():
    surfaces = {
        "dispatch": STAGE_TASK_SUMMARIES["stage_4_review"],
        "agent": (
            ROOT / ".opencode" / "agents" / "r2c-paper-fidelity-reviewer.md"
        ).read_text(encoding="utf-8"),
    }

    assert REVIEWER_GRAPH_MECHANISM_GUIDANCE in surfaces["dispatch"]
    for name, text in surfaces.items():
        text = " ".join(text.split())
        assert "top-level" in text, name
        assert "dotted class" in text, name
        for phrase in (
            "graph construction",
            "entity alignment",
            "mechanism liveness",
            "contribution ablation",
            "held-out skill",
            "paper-scale uplift",
        ):
            assert phrase in text, (name, phrase)


def test_fix_prompts_reinject_graph_contract_guidance_for_each_owner():
    finding = {
        "id": "F001",
        "severity": "critical",
        "description": "The graph contract and generated callable disagree.",
        "proposed_fix": "Honor the exact typed graph contract.",
    }
    analyzer = build_fix_mode_prompt(
        target_agent="r2c-method-analyzer", findings=[finding], paths=_paths(),
    )
    architecture = build_fix_mode_prompt(
        target_agent="r2c-architecture-coder", findings=[finding], paths=_paths(),
    )
    method = build_fix_mode_prompt(
        target_agent="r2c-method-coder", findings=[finding], paths=_paths(),
    )
    notebook = build_fix_mode_prompt(
        target_agent="r2c-notebook-generator",
        findings=[finding],
        paths=_paths(),
    )

    assert analyzer.count(ANALYZER_GRAPH_MECHANISM_GUIDANCE) == 1
    assert architecture.count(GRAPH_MECHANISM_IMPLEMENTATION_GUIDANCE) == 1
    assert method.count(GRAPH_MECHANISM_IMPLEMENTATION_GUIDANCE) == 1
    assert notebook.count(GRAPH_MECHANISM_NOTEBOOK_GUIDANCE) == 1


def test_notebook_prompts_expose_the_supported_live_binding_grammar():
    surfaces = {
        "dispatch": STAGE_TASK_SUMMARIES["stage_3a_notebook"],
        "agent": (
            ROOT / ".opencode" / "agents" / "r2c-notebook-generator.md"
        ).read_text(encoding="utf-8"),
    }

    assert GRAPH_MECHANISM_NOTEBOOK_GUIDANCE in surfaces["dispatch"]
    for name, text in surfaces.items():
        text = " ".join(text.split())
        for phrase in (
            "cfg = unpack(params)",
            "exact public",
            "explicit keyword",
            "feature",
            "threshold/cap",
            "copy()",
            "NumPy array",
            "torch Tensor",
            "clone()",
            "original graph",
            "architecture class",
            "returned model",
            "training",
            "inference",
            "output_root",
        ):
            assert phrase in text, (name, phrase)
