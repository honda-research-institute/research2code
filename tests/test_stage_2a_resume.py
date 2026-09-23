"""Stage 2.a resume checks follow package-scaffolder manifest ownership."""

from __future__ import annotations

import json
from pathlib import Path

from tests.helpers.assertions import assert_stage_completed, assert_stage_skipped
from tests.helpers.inject import minimal_method_spec
from tests.helpers.state import make_state


_LEGACY_SCAFFOLDER_OUTPUTS = (
    "method/README.md",
    "method/data.py",
    "method/example_data/README.md",
)

_TSF_SCAFFOLDER_OUTPUTS = (
    "method/data.py",
    "method/example_data/README.md",
    "method/target_scaling.py",
    "method/training_history.py",
    "method/README.md",
)


def _write_outputs(run_dir: Path, relative_paths: tuple[str, ...]) -> None:
    for relative_path in relative_paths:
        target = run_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# prior scaffolder output\n", encoding="utf-8")


def _tsf_state(run_dir: Path):
    state = make_state(run_dir)
    spec = minimal_method_spec(paradigm_id="time_series_forecasting")
    state.paths.method_spec.write_text(
        json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return state


def test_stage_2a_resume_reruns_when_new_tsf_scaffolder_helper_is_missing(
    fake_subprocess,
    run_dir,
):
    """An old complete sentinel cannot hide a manifest migration."""
    import run_pipeline

    state = _tsf_state(run_dir)
    prior_outputs = tuple(
        path for path in _TSF_SCAFFOLDER_OUTPUTS
        if path != "method/training_history.py"
    )
    _write_outputs(run_dir, prior_outputs)
    run_pipeline._write_stage_complete_sentinel(state.paths, "stage_2a")

    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_stage2_script(ok=True)

    result = run_pipeline.run_stage_2a(state)

    assert_stage_completed(result, "stage_2a")
    assert [call["args"][0] for call in fake_subprocess.script_calls] == [
        "scripts/scaffold_package.py",
    ]
    assert [call["args"][0] for call in fake_subprocess.stage2_calls] == [
        "scripts/validate_scaffolder_output.py",
    ]


def test_stage_2a_resume_skips_when_all_manifest_scaffolder_outputs_exist(
    run_dir,
):
    import run_pipeline

    state = _tsf_state(run_dir)
    _write_outputs(run_dir, _TSF_SCAFFOLDER_OUTPUTS)
    run_pipeline._write_stage_complete_sentinel(state.paths, "stage_2a")

    result = run_pipeline.run_stage_2a(state)

    assert_stage_skipped(result, "stage_2a")
    assert result.paths_written == [
        run_dir / relative_path for relative_path in _TSF_SCAFFOLDER_OUTPUTS
    ]


def test_stage_2a_resume_uses_legacy_outputs_when_plan_is_unavailable(
    monkeypatch,
    run_dir,
):
    """A plan-load failure retains the pre-migration resume behavior."""
    import run_pipeline
    from scripts import build_plan

    state = _tsf_state(run_dir)
    _write_outputs(run_dir, _LEGACY_SCAFFOLDER_OUTPUTS)
    run_pipeline._write_stage_complete_sentinel(state.paths, "stage_2a")
    monkeypatch.setattr(build_plan, "load_build_plan", lambda *args, **kwargs: None)

    result = run_pipeline.run_stage_2a(state)

    assert_stage_skipped(result, "stage_2a")
    assert result.paths_written == [
        run_dir / relative_path for relative_path in _LEGACY_SCAFFOLDER_OUTPUTS
    ]
