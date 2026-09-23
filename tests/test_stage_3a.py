"""Stage 3.a notebook authoring recovery behavior."""

from __future__ import annotations

import json

from tests.helpers.assertions import assert_stage_completed
from tests.helpers.state import make_state


def _clean_stage_review() -> str:
    return json.dumps({
        "schema_version": "1.0.0",
        "stage_id": "stage_3a_notebook",
        "review_status": "passed",
        "summary": "All checks pass.",
        "findings": [],
        "checks_summary": [],
        "caveats": [],
    }, indent=2)


def test_stage_3a_reconciles_existing_draft_before_redispatch(
    fake_dispatch, fake_subprocess, run_dir,
):
    """If a prior notebook-generator dispatch wrote notebook_draft.py but the
    driver halted before the stage sentinel, resume should validate/review the
    draft instead of asking the agent to rewrite it."""
    from run_pipeline import run_stage_3a

    state = make_state(run_dir)
    (run_dir / ".pipeline" / "notebook_draft.py").write_text(
        "# %%\nprint('notebook draft')\n", encoding="utf-8",
    )

    fake_subprocess.expect_script(returncode=0)  # render_notebook.py
    fake_subprocess.expect_stage2_script(ok=True)  # validate_notebook_output.py
    fake_subprocess.expect_stage2_script(ok=True)  # validate_package_imports.py
    fake_subprocess.expect_script(returncode=0)  # lint_generated_code.py (US-10 gate)
    fake_dispatch.expect(
        "r2c-stage-reviewer",
        writes={
            ".pipeline/stage_review_stage_3a_notebook.json": _clean_stage_review(),
        },
    )

    result = run_stage_3a(state)

    assert_stage_completed(result, "stage_3a")
    assert [call.agent for call in fake_dispatch.calls] == ["r2c-stage-reviewer"]


def test_notebook_generator_timeout_recovers_when_draft_landed(
    monkeypatch, fake_subprocess, run_dir,
):
    """A blocking opencode POST can time out after the notebook agent wrote the
    canonical file. Treat that as soft-complete only when the changed draft
    passes the normal Stage 3.a validator chain."""
    import run_pipeline
    from opencode_client import DispatchResponseTimeout

    state = make_state(run_dir)

    def _timeout_after_write(*, state, agent: str, prompt: str, timeout_s: int):
        assert agent == "r2c-notebook-generator"
        (state.paths.pipeline_dir / "notebook_draft.py").write_text(
            "# %%\nprint('landed before timeout')\n", encoding="utf-8",
        )
        raise DispatchResponseTimeout("test dispatch response timeout")

    monkeypatch.setattr(run_pipeline, "dispatch_agent", _timeout_after_write)
    fake_subprocess.expect_script(returncode=0)  # render_notebook.py
    fake_subprocess.expect_stage2_script(ok=True)  # validate_notebook_output.py
    fake_subprocess.expect_stage2_script(ok=True)  # validate_package_imports.py
    fake_subprocess.expect_script(returncode=0)  # lint_generated_code.py (US-10 gate)

    result = run_pipeline._dispatch_notebook_generator(state)

    assert result.completed is False
    assert result.error and result.error["recovered_from"] == "DispatchResponseTimeout"
    assert (run_dir / ".pipeline" / "notebook_draft.py").exists()


def test_stage_3a_fix_loop_degrade_reconciles_requirements(
    monkeypatch, fake_dispatch, run_dir,
):
    """A cap-exhaustion degrade from the fix loop continues to delivery with
    the notebook on disk — requirements must be reconciled exactly like on
    the clean-completion path (O2 gap, repo-map verdict 2026-07-04)."""
    import run_pipeline

    state = make_state(run_dir)
    (run_dir / ".pipeline" / "notebook_draft.py").write_text(
        "# %%\nprint('notebook draft')\n", encoding="utf-8",
    )

    calls: list[str] = []
    monkeypatch.setattr(
        run_pipeline, "_reconcile_requirements_with_notebook",
        lambda state, stage_id="stage_3a": calls.append(stage_id),
    )
    degraded = run_pipeline.StageResult(
        status="degraded", stage_id="stage_3a", notes="cap exhausted",
    )
    monkeypatch.setattr(run_pipeline, "run_fix_loop", lambda **kwargs: degraded)

    result = run_pipeline.run_stage_3a(state)

    assert result.status == "degraded"
    assert calls == ["stage_3a"]


def test_stage_3a_fix_loop_halt_does_not_reconcile(
    monkeypatch, fake_dispatch, run_dir,
):
    """A halt from the fix loop stops the run — no reconcile dispatch."""
    import run_pipeline

    state = make_state(run_dir)
    (run_dir / ".pipeline" / "notebook_draft.py").write_text(
        "# %%\nprint('notebook draft')\n", encoding="utf-8",
    )

    calls: list[str] = []
    monkeypatch.setattr(
        run_pipeline, "_reconcile_requirements_with_notebook",
        lambda state, stage_id="stage_3a": calls.append(stage_id),
    )
    halted = run_pipeline.StageResult(
        status="halted", stage_id="stage_3a", notes="validator failed at cap",
    )
    monkeypatch.setattr(run_pipeline, "run_fix_loop", lambda **kwargs: halted)

    result = run_pipeline.run_stage_3a(state)

    assert result.status == "halted"
    assert calls == []
