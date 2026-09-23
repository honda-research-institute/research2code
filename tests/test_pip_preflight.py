"""Item 30 — restricted-network dependency readiness + wheelhouse.

A researcher's first solo run (07-08) died 30+ minutes in at the stage 2.d dependency
install because the corporate network blackholes external PyPI. Stage 2.d
now materializes the dependency manifest, checks local metadata without pip or
network access, and only then asks whether an install source is reachable. The
wheelhouse mechanics remain available for fully-offline installation.

The autouse conftest fixture sets R2C_SKIP_PIP_PREFLIGHT=1 for the whole suite,
so these tests delete it before exercising the preflight, and stub the network
probe rather than making real calls.
"""

from __future__ import annotations

import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest

import run_pipeline


def _paths(repo_root: Path) -> run_pipeline.PipelinePaths:
    run_dir = repo_root / "r2c_runs" / "sample"
    return run_pipeline.PipelinePaths.from_setup_result(
        {
            "repo_root": str(repo_root),
            "input_path": str(repo_root / "input_papers" / "sample.md"),
            "input_kind": "markdown",
            "slug": "sample",
            "run_dir": str(run_dir),
            "pipeline_dir": str(run_dir / ".pipeline"),
            "paper_md_path": str(run_dir / ".pipeline" / "paper.md"),
            "paper_md_present": False,
        }
    )


@pytest.fixture(autouse=True)
def _reset_reachability_cache(monkeypatch):
    monkeypatch.setattr(run_pipeline, "_PIP_INDEX_REACHABLE", None)


# --- _resolve_wheelhouse ----------------------------------------------------


def test_wheelhouse_env_dir_wins(tmp_path, monkeypatch):
    wh = tmp_path / "wheels"
    wh.mkdir()
    monkeypatch.setenv("R2C_WHEELHOUSE", str(wh))
    assert run_pipeline._resolve_wheelhouse(tmp_path) == wh


def test_wheelhouse_env_pointing_at_nonexistent_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("R2C_WHEELHOUSE", str(tmp_path / "nope"))
    assert run_pipeline._resolve_wheelhouse(tmp_path) is None


def test_wheelhouse_conventional_dir_at_repo_root(tmp_path, monkeypatch):
    monkeypatch.delenv("R2C_WHEELHOUSE", raising=False)
    (tmp_path / "r2c_wheelhouse").mkdir()
    assert run_pipeline._resolve_wheelhouse(tmp_path) == tmp_path / "r2c_wheelhouse"


def test_wheelhouse_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("R2C_WHEELHOUSE", raising=False)
    assert run_pipeline._resolve_wheelhouse(tmp_path) is None


# --- _pip_index_reachable (real probe, stubbed transport) -------------------


class _FakeResp:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_index_reachable_on_response(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: _FakeResp())
    assert run_pipeline._pip_index_reachable(timeout=1) is True


def test_index_reachable_on_http_error_status(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", boom)
    # A real HTTP status still proves the index host is reachable.
    assert run_pipeline._pip_index_reachable(timeout=1) is True


def test_index_unreachable_on_url_error(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert run_pipeline._pip_index_reachable(timeout=1) is False


def test_index_unreachable_on_os_error(monkeypatch):
    def boom(req, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert run_pipeline._pip_index_reachable(timeout=1) is False


# --- _pip_wheelhouse_install_args -------------------------------------------


def test_install_args_empty_without_wheelhouse(tmp_path, monkeypatch):
    monkeypatch.delenv("R2C_WHEELHOUSE", raising=False)
    assert run_pipeline._pip_wheelhouse_install_args(tmp_path) == []


def test_install_args_find_links_when_online(tmp_path, monkeypatch):
    wh = tmp_path / "r2c_wheelhouse"
    wh.mkdir()
    monkeypatch.delenv("R2C_WHEELHOUSE", raising=False)
    monkeypatch.setattr(run_pipeline, "_PIP_INDEX_REACHABLE", True)
    assert run_pipeline._pip_wheelhouse_install_args(tmp_path) == [
        "--find-links",
        str(wh),
    ]


def test_install_args_add_no_index_when_offline(tmp_path, monkeypatch):
    wh = tmp_path / "r2c_wheelhouse"
    wh.mkdir()
    monkeypatch.delenv("R2C_WHEELHOUSE", raising=False)
    monkeypatch.setattr(run_pipeline, "_PIP_INDEX_REACHABLE", False)
    assert run_pipeline._pip_wheelhouse_install_args(tmp_path) == [
        "--find-links",
        str(wh),
        "--no-index",
    ]


def test_install_args_no_no_index_when_reachability_unknown(tmp_path, monkeypatch):
    wh = tmp_path / "r2c_wheelhouse"
    wh.mkdir()
    monkeypatch.delenv("R2C_WHEELHOUSE", raising=False)
    monkeypatch.setattr(run_pipeline, "_PIP_INDEX_REACHABLE", None)
    assert run_pipeline._pip_wheelhouse_install_args(tmp_path) == [
        "--find-links",
        str(wh),
    ]


# --- _preflight_pip_for_install (only for locally missing requirements) -----


def test_install_preflight_skipped_by_env(tmp_path, monkeypatch):
    monkeypatch.setattr(run_pipeline, "_PIP_INDEX_REACHABLE", False)
    # Autouse fixture set the skip env; the install-entry check honors it too.
    reason = run_pipeline._preflight_pip_for_install(_paths(tmp_path), stage_id="stage_2d")
    assert reason is None


def test_install_preflight_proceeds_when_cached_reachable(tmp_path, monkeypatch):
    monkeypatch.delenv("R2C_SKIP_PIP_PREFLIGHT", raising=False)
    monkeypatch.setattr(run_pipeline, "_PIP_INDEX_REACHABLE", True)
    # Must not re-probe when a reachable result is already cached.
    monkeypatch.setattr(run_pipeline, "_pip_index_reachable",
                        lambda **k: (_ for _ in ()).throw(AssertionError("re-probed")))
    reason = run_pipeline._preflight_pip_for_install(_paths(tmp_path), stage_id="stage_2d")
    assert reason is None


def test_install_preflight_reprobes_when_cache_empty_on_resume(tmp_path, monkeypatch):
    """A Stage 2.d source decision starts with an empty process-local cache and
    probes only after the caller found unsatisfied local requirements."""
    monkeypatch.delenv("R2C_SKIP_PIP_PREFLIGHT", raising=False)
    monkeypatch.delenv("R2C_WHEELHOUSE", raising=False)
    monkeypatch.setattr(run_pipeline, "_PIP_INDEX_REACHABLE", None)
    monkeypatch.setattr(run_pipeline, "_pip_index_reachable", lambda **k: False)
    reason = run_pipeline._preflight_pip_for_install(_paths(tmp_path), stage_id="stage_2d")
    assert reason is not None
    assert "package index" in reason
    assert run_pipeline._PIP_INDEX_REACHABLE is False


def test_install_preflight_halts_when_unreachable_and_no_wheelhouse(tmp_path, monkeypatch):
    monkeypatch.delenv("R2C_SKIP_PIP_PREFLIGHT", raising=False)
    monkeypatch.delenv("R2C_WHEELHOUSE", raising=False)
    monkeypatch.setattr(run_pipeline, "_PIP_INDEX_REACHABLE", False)
    # A cached "unreachable" verdict is re-probed because the network may have
    # changed; still unreachable means halt.
    monkeypatch.setattr(run_pipeline, "_pip_index_reachable", lambda **k: False)
    reason = run_pipeline._preflight_pip_for_install(_paths(tmp_path), stage_id="stage_2d")
    assert reason is not None
    assert "package index" in reason


def test_install_preflight_reprobes_a_stale_unreachable_verdict(tmp_path, monkeypatch):
    """A stale unreachable verdict is refreshed if the network changes before
    the package-source decision is retried."""
    monkeypatch.delenv("R2C_SKIP_PIP_PREFLIGHT", raising=False)
    monkeypatch.delenv("R2C_WHEELHOUSE", raising=False)
    monkeypatch.setattr(run_pipeline, "_PIP_INDEX_REACHABLE", False)
    monkeypatch.setattr(run_pipeline, "_pip_index_reachable", lambda **k: True)
    reason = run_pipeline._preflight_pip_for_install(_paths(tmp_path), stage_id="stage_2d")
    assert reason is None
    # The refreshed verdict also un-sticks the --no-index install flag choice.
    assert run_pipeline._PIP_INDEX_REACHABLE is True


def test_install_preflight_proceeds_offline_when_wheelhouse_present(tmp_path, monkeypatch):
    monkeypatch.delenv("R2C_SKIP_PIP_PREFLIGHT", raising=False)
    (tmp_path / "r2c_wheelhouse").mkdir()
    monkeypatch.delenv("R2C_WHEELHOUSE", raising=False)
    monkeypatch.setattr(run_pipeline, "_PIP_INDEX_REACHABLE", False)
    monkeypatch.setattr(
        run_pipeline,
        "_pip_index_reachable",
        lambda **k: (_ for _ in ()).throw(AssertionError("index probed")),
    )
    reason = run_pipeline._preflight_pip_for_install(_paths(tmp_path), stage_id="stage_2d")
    assert reason is None
    assert run_pipeline._PIP_INDEX_REACHABLE is False
    assert run_pipeline._pip_wheelhouse_install_args(tmp_path) == [
        "--find-links",
        str(tmp_path / "r2c_wheelhouse"),
        "--no-index",
    ]


# --- run_stage_2d wiring -----------------------------------------------------


def _completed(args, stdout="", stderr="", code=0):
    return subprocess.CompletedProcess(args, code, stdout=stdout, stderr=stderr)


def test_run_stage2d_offline_writes_manifest_then_halts_without_pip(
    tmp_path, monkeypatch
):
    """Fresh corporate-network run: manifest first, then a no-pip transport halt."""
    from tests.helpers.state import make_state  # noqa: PLC0415

    state = make_state(tmp_path / "r2c_runs" / "sample", slug="sample")
    state.paths.repo_root = tmp_path
    monkeypatch.delenv("R2C_SKIP_PIP_PREFLIGHT", raising=False)
    monkeypatch.delenv("R2C_WHEELHOUSE", raising=False)
    monkeypatch.setattr(run_pipeline, "_PIP_INDEX_REACHABLE", False)
    requirements_path = state.paths.run_dir / "requirements.txt"
    calls: list[list[str]] = []

    def fake_run_script(stage_id, args, *, timeout=300):
        calls.append(list(args))
        assert args[0] == "scripts/finalize_package_init.py"
        requirements_path.write_text(
            "r2c-test-dependency-that-is-not-installed>=1.0\n",
            encoding="utf-8",
        )
        (state.paths.run_dir / "method").mkdir(parents=True, exist_ok=True)
        (state.paths.run_dir / "method" / "__init__.py").write_text("# generated\n")
        return _completed(args)

    def unreachable_only_after_manifest(**kwargs):
        assert requirements_path.is_file()
        return False

    monkeypatch.setattr(run_pipeline, "run_script", fake_run_script)
    monkeypatch.setattr(run_pipeline, "_pip_index_reachable", unreachable_only_after_manifest)
    monkeypatch.setattr(
        run_pipeline.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("direct subprocess.run invoked")
        ),
    )
    monkeypatch.setattr(
        run_pipeline.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("direct subprocess.Popen invoked")
        ),
    )
    monkeypatch.setattr(run_pipeline, "_post_halt_to_session", lambda *a, **k: None)

    result = run_pipeline.run_stage_2d(state)

    assert result.status == "halted"
    assert requirements_path.is_file()
    assert calls == [[
        "scripts/finalize_package_init.py",
        "--spec",
        str(state.paths.method_spec),
        "--run-dir",
        str(state.paths.run_dir),
    ]]
    assert not any(call[:3] == ["-m", "pip", "install"] for call in calls)
    message = result.halt_artifact["user_message"]
    context = result.halt_artifact["context"]
    assert "wrote this paper's dependency list" in message
    assert "r2c_runs/sample/requirements.txt" in message
    assert sys.executable in message
    assert "r2c-test-dependency-that-is-not-installed>=1.0" in message
    assert "network where package installation works" in message
    assert "resume the same run" in message
    assert "without invoking pip" in message
    assert "R2C_SKIP_PIP_PREFLIGHT=1" in message
    assert context["requirements_path"] == str(requirements_path)
    assert context["python_executable"] == sys.executable
    assert context["unsatisfied_requirements"] == [
        "r2c-test-dependency-that-is-not-installed>=1.0 (not installed)"
    ]
    assert (state.paths.pipeline_dir / "stage_2d.halt").is_file()


def test_run_stage2d_locally_satisfied_resumes_without_probe_or_pip(
    tmp_path, monkeypatch
):
    """After an off-network install, corporate-network resume stays fully local."""
    from tests.helpers.state import make_state  # noqa: PLC0415

    state = make_state(tmp_path / "r2c_runs" / "sample", slug="sample")
    state.paths.repo_root = tmp_path
    monkeypatch.delenv("R2C_SKIP_PIP_PREFLIGHT", raising=False)
    halt_path = state.paths.pipeline_dir / "stage_2d.halt"
    halt_path.write_text('{"status": "halted"}\n', encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run_script(stage_id, args, *, timeout=300):
        calls.append(list(args))
        assert args[0] == "scripts/finalize_package_init.py"
        (state.paths.run_dir / "requirements.txt").write_text(
            "pytest>=8.0\n",
            encoding="utf-8",
        )
        (state.paths.run_dir / "method").mkdir(parents=True, exist_ok=True)
        (state.paths.run_dir / "method" / "__init__.py").write_text("# generated\n")
        return _completed(args)

    monkeypatch.setattr(run_pipeline, "run_script", fake_run_script)
    monkeypatch.setattr(
        run_pipeline,
        "_pip_index_reachable",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("index probed")),
    )
    monkeypatch.setattr(
        run_pipeline,
        "_run_stage2_script",
        lambda *args, **kwargs: run_pipeline.Stage2ScriptOutcome(
            ok=True,
            error_tail="",
            returncode=0,
        ),
    )
    monkeypatch.setattr(
        run_pipeline.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("direct subprocess.run invoked")
        ),
    )
    monkeypatch.setattr(
        run_pipeline.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("direct subprocess.Popen invoked")
        ),
    )

    result = run_pipeline.run_stage_2d(state)

    assert result.status == "completed"
    assert len(calls) == 1
    assert calls[0][0] == "scripts/finalize_package_init.py"
    assert not any(call[:3] == ["-m", "pip", "install"] for call in calls)
    assert not halt_path.exists()


def test_run_stage2d_finalizer_success_without_manifest_is_contract_halt(
    tmp_path, monkeypatch
):
    from tests.helpers.state import make_state  # noqa: PLC0415

    state = make_state(tmp_path / "r2c_runs" / "sample", slug="sample")
    monkeypatch.setattr(run_pipeline, "_post_halt_to_session", lambda *a, **k: None)
    monkeypatch.setattr(
        run_pipeline,
        "run_script",
        lambda stage_id, args, timeout=300: _completed(args),
    )
    monkeypatch.setattr(
        run_pipeline,
        "_pip_index_reachable",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("index probed")),
    )

    result = run_pipeline.run_stage_2d(state)

    assert result.status == "halted"
    assert result.halt_artifact["halt_class"] == "internal_contract_violation"
    assert "did not create requirements.txt" in result.halt_artifact["reason"]


def test_run_stage2d_checker_error_fails_closed_without_probe_or_pip(
    tmp_path, monkeypatch
):
    from tests.helpers.state import make_state  # noqa: PLC0415

    state = make_state(tmp_path / "r2c_runs" / "sample", slug="sample")
    monkeypatch.setattr(run_pipeline, "_post_halt_to_session", lambda *a, **k: None)
    requirements_path = state.paths.run_dir / "requirements.txt"
    calls: list[list[str]] = []

    def fake_run_script(stage_id, args, *, timeout=300):
        calls.append(list(args))
        requirements_path.write_text("numpy>=1.24\n", encoding="utf-8")
        return _completed(args)

    monkeypatch.setattr(run_pipeline, "run_script", fake_run_script)
    monkeypatch.setattr(
        run_pipeline.dependency_readiness,
        "check_local_requirements",
        lambda path: run_pipeline.dependency_readiness.LocalRequirementsCheck(
            checker_error="metadata backend failed"
        ),
    )
    monkeypatch.setattr(
        run_pipeline,
        "_pip_index_reachable",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("index probed")),
    )

    result = run_pipeline.run_stage_2d(state)

    assert result.status == "halted"
    assert result.halt_artifact["halt_class"] == "internal_contract_violation"
    assert result.halt_artifact["context"]["checker_error"] == (
        "metadata backend failed"
    )
    assert calls == [[
        "scripts/finalize_package_init.py",
        "--spec",
        str(state.paths.method_spec),
        "--run-dir",
        str(state.paths.run_dir),
    ]]
