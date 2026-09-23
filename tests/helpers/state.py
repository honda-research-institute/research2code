"""Construct minimal PipelineState/PipelinePaths for tests.

The real construction (`PipelinePaths.from_setup_result`) requires running
`setup_pipeline_dirs.py` against a real paper. Tests don't have a paper, so
we synthesize the equivalent in-memory.
"""

from __future__ import annotations

from pathlib import Path


def make_paths(run_dir: Path, slug: str = "test-paper") -> "PipelinePaths":  # noqa: F821
    """Construct a PipelinePaths anchored at `run_dir`. Creates .pipeline/
    if missing. Paper.md is a placeholder; input_path is the run_dir itself.

    Imports run_pipeline lazily so this module can be imported before the
    sys.path setup in conftest.py runs."""
    from run_pipeline import PipelinePaths  # noqa: PLC0415

    pipeline_dir = run_dir / ".pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    paper_md = pipeline_dir / "paper.md"
    if not paper_md.exists():
        paper_md.write_text("# Placeholder paper\n", encoding="utf-8")
    spec = pipeline_dir / "method_spec.json"
    pmap = pipeline_dir / "paper_map.json"
    gate = pipeline_dir / "feasibility_gate.json"
    return PipelinePaths(
        repo_root=run_dir.parent,
        run_dir=run_dir,
        pipeline_dir=pipeline_dir,
        paper_md=paper_md,
        input_path=run_dir,
        input_kind="markdown",
        slug=slug,
        method_spec=spec,
        paper_map=pmap,
        paper_map_halt=pmap.with_suffix(pmap.suffix + ".halt"),
        method_spec_halt=spec.with_suffix(spec.suffix + ".halt"),
        paradigm_gap_report=pipeline_dir / "paradigm_gap_report.json",
        paradigm_gap_report_md=pipeline_dir / "paradigm_gap_report.md",
        provisional_packs_dir=pipeline_dir / "provisional_packs",
        provisional_pack_manifest=pipeline_dir / "provisional_pack.json",
        feasibility_gate=gate,
        feasibility_halt=gate.with_suffix(gate.suffix + ".halt"),
    )


def make_state(run_dir: Path, slug: str = "test-paper") -> "PipelineState":  # noqa: F821
    """Construct a minimal PipelineState anchored at `run_dir`.

    session_id and port are placeholders — the fake dispatcher doesn't use
    them. agent_models is an empty dict; dispatch_agent normally looks up the
    model server-side, but the fake skips that.
    """
    from run_pipeline import PipelineState  # noqa: PLC0415

    return PipelineState(
        session_id="fake-session",
        port=9999,
        paths=make_paths(run_dir, slug=slug),
        agent_models={},
    )
