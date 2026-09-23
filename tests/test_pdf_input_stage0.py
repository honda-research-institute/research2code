from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

import parse_pdf
import progress_state
import run_pipeline
import setup_pipeline_dirs
from pdf_parse_quality import assess_parse_quality
from run_events import RunDirectoryLockError


def _write_current_progress(pipeline_dir: Path, slug: str = "sample") -> None:
    """A prior run always leaves a current progress.json — it is written at
    lock acquisition, before any parse — so a cached-parse resume dir has one.
    Item 29's archive check keys off it to tell a resume from an old-version
    dir; without it these dirs would be moved aside as incompatible."""
    (pipeline_dir / "progress.json").write_text(
        json.dumps(
            {
                "schema_version": progress_state.PROGRESS_SCHEMA_VERSION,
                "slug": slug,
                "stages": [],
            }
        ),
        encoding="utf-8",
    )


def _setup_result(repo: Path, run_dir: Path, pdf_path: Path) -> dict:
    return {
        "repo_root": str(repo),
        "input_path": str(pdf_path),
        "input_kind": "pdf",
        "slug": pdf_path.stem,
        "run_dir": str(run_dir),
        "pipeline_dir": str(run_dir / ".pipeline"),
        "paper_md_path": str(run_dir / ".pipeline" / "paper.md"),
        "paper_md_present": (run_dir / ".pipeline" / "paper.md").is_file(),
    }


def _completed(args: list[str], stdout: str = "", stderr: str = "", code: int = 0):
    return subprocess.CompletedProcess(args, code, stdout=stdout, stderr=stderr)


def _pdf_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quality_report_for(pdf_path: Path, *, passed: bool = True) -> dict:
    return {
        "passed": passed,
        "word_count": 260,
        "char_count": 1600,
        "visible_char_count": 1600,
        "data_image_count": 0,
        "data_image_chars": 0,
        "min_words": 250,
        "reasons": [],
        "source_pdf_path": str(pdf_path.resolve()),
        "source_pdf_sha256": _pdf_sha256(pdf_path),
    }


def test_suffixless_input_prefers_markdown_over_pdf(tmp_path):
    """Markdown-first (maintainer decision 2026-07-07): a pre-parsed .md skips the Marker
    parse — faster, cheaper, and immune to the parse service being down.
    An explicit '<slug>.pdf' still forces the PDF (companion test below)."""
    papers = tmp_path / "input_papers"
    papers.mkdir()
    (papers / "sample.md").write_text("# Markdown shortcut\n", encoding="utf-8")
    (papers / "sample.pdf").write_bytes(b"%PDF-1.4\n")

    resolved = setup_pipeline_dirs._resolve_input("sample", tmp_path)

    assert resolved == (papers / "sample.md").resolve()


def test_explicit_pdf_extension_still_selects_the_pdf(tmp_path):
    papers = tmp_path / "input_papers"
    papers.mkdir()
    (papers / "sample.md").write_text("# Markdown shortcut\n", encoding="utf-8")
    (papers / "sample.pdf").write_bytes(b"%PDF-1.4\n")

    resolved = setup_pipeline_dirs._resolve_input("sample.pdf", tmp_path)

    assert resolved == (papers / "sample.pdf").resolve()


def test_materialize_copies_source_and_its_companion_format(tmp_path):
    """Researcher request 2026-07-07: the run's .pipeline/ carries BOTH
    formats when both exist next to the input — an .md-driven run keeps
    the .pdf it was exported from, and vice versa."""
    papers = tmp_path / "input_papers"
    papers.mkdir()
    md = papers / "sample.md"
    md.write_text("# Markdown shortcut\n", encoding="utf-8")
    (papers / "sample.pdf").write_bytes(b"%PDF-1.4\n")

    result, err = setup_pipeline_dirs.resolve_setup(str(md), tmp_path)
    assert err is None
    result, err = setup_pipeline_dirs.materialize_setup(result)
    assert err is None

    pipeline = tmp_path / "r2c_runs" / "sample" / ".pipeline"
    assert (pipeline / "sample.md").is_file()
    assert (pipeline / "sample.pdf").is_file()
    # And nothing lands at the run-dir top level (reserved for deliverables).
    top = sorted(p.name for p in (tmp_path / "r2c_runs" / "sample").iterdir())
    assert top == [".pipeline"]


def test_setup_pipeline_dirs_resolve_only_does_not_write(tmp_path, monkeypatch, capsys):
    papers = tmp_path / "input_papers"
    papers.mkdir()
    (papers / "sample.md").write_text("# Paper\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["setup_pipeline_dirs.py", "sample", "--resolve-only"],
    )

    assert setup_pipeline_dirs.main() == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)

    assert result["slug"] == "sample"
    assert result["input_kind"] == "markdown"
    assert not (tmp_path / "r2c_runs").exists()


def test_parse_pdf_retries_until_quality_passes(monkeypatch):
    calls: list[tuple[str, bool, float]] = []

    def fake_call(pdf_path: str, use_llm: bool, timeout_s: float) -> str:
        calls.append((pdf_path, use_llm, timeout_s))
        if len(calls) == 1:
            raise RuntimeError("temporary Marker failure")
        if len(calls) == 2:
            return "too short"
        return " ".join(["method"] * 260)

    monkeypatch.delenv("R2C_MARKER_USE_LLM", raising=False)
    monkeypatch.setattr(parse_pdf, "_call_marker", fake_call)

    markdown, quality, attempt = parse_pdf.parse_pdf_with_retries(
        "paper.pdf", retries=2, retry_delay_s=0, min_words=250,
        attempt_timeout_s=480.0,
    )

    assert attempt == 3
    assert quality.passed
    assert len(markdown.split()) == 260
    # use_llm follows DEFAULT_USE_LLM (False) and the per-attempt bound
    # reaches every attempt.
    assert calls == [("paper.pdf", False, 480.0)] * 3


def test_main_without_marker_installed_prints_the_setup_story(
        tmp_path, monkeypatch, capsys):
    """The not-installed case must reach stage_0's stderr as one plain line
    naming the install command, never a traceback."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 stub")
    monkeypatch.setenv("R2C_MARKER_BIN", str(tmp_path / "no-such-marker"))
    import env_file
    monkeypatch.setattr(env_file, "default_env_path",
                        lambda: tmp_path / "absent.env")
    monkeypatch.setattr(
        sys, "argv",
        ["parse_pdf.py", str(pdf), "--output", str(tmp_path / "out.md")])
    with pytest.raises(SystemExit) as exc:
        parse_pdf.main()
    assert exc.value.code == parse_pdf.PARSER_MISSING_EXIT
    err = capsys.readouterr().err
    assert "pip install marker-pdf" in err and ".env.example" in err
    assert "Traceback" not in err


def test_preflight_names_a_missing_llama_backend(tmp_path, monkeypatch):
    """Marker installed but llama.cpp absent must fail before the parse
    starts, with the install recipe, not minutes in with Marker's traceback."""
    marker = _fake_marker(tmp_path, "exit 0")
    monkeypatch.setenv("R2C_MARKER_BIN", str(marker))
    monkeypatch.setenv("LLAMA_CPP_BINARY", str(tmp_path / "no-llama-server"))
    with pytest.raises(parse_pdf.ParsePDFError, match="inference backend.*llama.cpp"):
        parse_pdf.preflight()
    monkeypatch.setenv("LLAMA_CPP_BINARY", str(marker))  # any executable will do
    assert parse_pdf.preflight() == str(marker)


def test_resolve_use_llm_defaults_false(monkeypatch):
    monkeypatch.delenv("R2C_MARKER_USE_LLM", raising=False)
    assert parse_pdf.DEFAULT_USE_LLM is False
    assert parse_pdf.resolve_use_llm() is False
    for raw, expected in [
        ("1", True), ("true", True), ("YES", True), ("on", True),
        ("0", False), ("false", False), ("No", False), ("off", False),
        # Empty/unrecognized fall back to the default (currently False).
        ("", False), ("   ", False), ("bogus", False),
    ]:
        monkeypatch.setenv("R2C_MARKER_USE_LLM", raw)
        assert parse_pdf.resolve_use_llm() is expected, raw


def test_use_llm_env_and_explicit_override_reach_the_call(monkeypatch):
    seen: list[bool] = []

    def fake_call(pdf_path: str, use_llm: bool, timeout_s: float) -> str:
        seen.append(use_llm)
        return " ".join(["method"] * 260)

    monkeypatch.setattr(parse_pdf, "_call_marker", fake_call)
    monkeypatch.setenv("R2C_MARKER_USE_LLM", "true")
    parse_pdf.parse_pdf_with_retries(
        "paper.pdf", retries=0, retry_delay_s=0, min_words=250,
        attempt_timeout_s=0,
    )
    # Explicit argument beats the environment.
    parse_pdf.parse_pdf_with_retries(
        "paper.pdf", retries=0, retry_delay_s=0, min_words=250,
        attempt_timeout_s=0, use_llm=False,
    )
    assert seen == [True, False]


def _fake_marker(tmp_path: Path, body: str) -> Path:
    """A stand-in marker_single on disk: `body` is the shell that runs after
    the arguments are recorded to $ARGS_FILE."""
    script = tmp_path / "marker_single"
    script.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" > '{tmp_path / 'args.txt'}'\n"
        + body + "\n")
    script.chmod(0o755)
    return script


def test_call_marker_kills_hung_attempt(tmp_path, monkeypatch):
    # A hung parse must not eat the whole stage-0 budget: the attempt is
    # killed at the per-attempt bound so the retry loop gets its turn.
    import time
    monkeypatch.setenv("R2C_MARKER_BIN", str(_fake_marker(tmp_path, "sleep 60")))
    t0 = time.monotonic()
    with pytest.raises(parse_pdf.ParsePDFError, match="timed out after 2s"):
        parse_pdf._call_marker("paper.pdf", False, 2.0)
    assert time.monotonic() - t0 < 20


def test_call_marker_reads_output_and_reports_errors(tmp_path, monkeypatch):
    # marker_single writes <output_dir>/<stem>/<stem>.md; the flags are the
    # local-run contract (markdown, no image extraction, LLM assist off).
    ok = _fake_marker(tmp_path, (
        'out=""; while [ $# -gt 0 ]; do [ "$1" = "--output_dir" ] && out="$2"; shift; done\n'
        'mkdir -p "$out/paper" && python3 -c "print(\'method \' * 260)" > "$out/paper/paper.md"'))
    monkeypatch.setenv("R2C_MARKER_BIN", str(ok))
    monkeypatch.delenv("R2C_MARKER_FORCE_OCR", raising=False)
    markdown = parse_pdf._call_marker("paper.pdf", False, 30.0)
    assert len(markdown.split()) == 260
    args = (tmp_path / "args.txt").read_text().split("\n")
    assert args[0] == "paper.pdf"
    assert "--output_format" in args and "markdown" in args
    assert "--disable_image_extraction" in args
    assert "--use_llm" not in args and "--force_ocr" not in args

    parse_pdf._call_marker("paper.pdf", True, 30.0)
    assert "--use_llm" in (tmp_path / "args.txt").read_text().split("\n")

    bad = _fake_marker(tmp_path, 'echo "CUDA out of memory" >&2; exit 3')
    monkeypatch.setenv("R2C_MARKER_BIN", str(bad))
    with pytest.raises(parse_pdf.ParsePDFError, match=r"exited 3[\s\S]*CUDA out of memory"):
        parse_pdf._call_marker("paper.pdf", False, 30.0)


def test_parse_pdf_retries_after_attempt_timeout(monkeypatch):
    # A timed-out attempt is a retryable failure like any other: attempt 2
    # must run with the same per-attempt bound.
    timeouts_seen: list[float] = []

    def fake_call(pdf_path, use_llm, timeout_s):
        timeouts_seen.append(timeout_s)
        if len(timeouts_seen) == 1:
            raise parse_pdf.ParsePDFError(
                "Marker timed out after 480s on paper.pdf"
            )
        return " ".join(["method"] * 260)

    monkeypatch.setattr(parse_pdf, "_call_marker", fake_call)

    markdown, quality, attempt = parse_pdf.parse_pdf_with_retries(
        "paper.pdf", retries=2, retry_delay_s=0, min_words=250,
        attempt_timeout_s=480.0,
    )

    assert attempt == 2
    assert quality.passed
    assert timeouts_seen == [480.0, 480.0]


def test_stage0_budget_covers_every_parse_attempt(monkeypatch):
    monkeypatch.delenv("R2C_MARKER_RETRIES", raising=False)
    monkeypatch.delenv("R2C_MARKER_ATTEMPT_TIMEOUT_S", raising=False)
    monkeypatch.delenv("R2C_MARKER_RETRY_DELAY_S", raising=False)
    assert parse_pdf.total_budget_s() == 2 * 1500.0 + 2.0 + 60.0
    monkeypatch.setenv("R2C_MARKER_ATTEMPT_TIMEOUT_S", "0")
    assert parse_pdf.total_budget_s() == 24 * 3600.0
    assert run_pipeline._parse_pdf_budget_s() == 24 * 3600.0


def test_describe_error_never_renders_empty():
    # 2026-07-02 batch log: "Attempt 1/3 failed: " with no error text — a
    # bare connection exception carries no message. Fall back to repr.
    class BareError(Exception):
        pass

    assert parse_pdf._describe_error(BareError()) == "BareError()"
    assert parse_pdf._describe_error(RuntimeError("boom")) == "boom"
    assert parse_pdf._describe_error(None) == "unknown error"


def test_parse_quality_rejects_short_or_error_output():
    short = assess_parse_quality("Internal Server Error", min_words=10)
    assert not short.passed
    assert "service error page" in " ".join(short.reasons)

    valid_error_paper = assess_parse_quality(
        "This paper studies Internal Server Error handling in deployed systems. "
        * 45,
        min_words=250,
    )
    assert valid_error_paper.passed

    image_only = assess_parse_quality(
        '<img src="data:image/png;base64,AAAA" alt="">\nword word',
        min_words=10,
    )
    assert not image_only.passed
    assert image_only.data_image_count == 1
    assert image_only.word_count < 10

    many_empty_images = assess_parse_quality(
        '<img src="data:image/png;base64,AAAA" alt="">\n' * 20,
        min_words=10,
    )
    assert not many_empty_images.passed
    assert many_empty_images.word_count == 0

    invalid_minimum = assess_parse_quality("method", min_words=-1)
    assert not invalid_minimum.passed
    assert invalid_minimum.min_words > 0

    enough_text = assess_parse_quality(" ".join(["method"] * 20), min_words=10)
    assert enough_text.passed


def test_run_stage0_pdf_parse_then_quality_gate(tmp_path, monkeypatch):
    repo = tmp_path
    pdf = repo / "input_papers" / "sample.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"%PDF-1.4\n")
    run_dir = repo / "r2c_runs" / "sample"
    (run_dir / ".pipeline").mkdir(parents=True)
    _write_current_progress(run_dir / ".pipeline")
    setup = _setup_result(repo, run_dir, pdf)
    calls: list[list[str]] = []

    def fake_run_script(stage_id: str, args: list[str], *, timeout: int = 300):
        calls.append(args)
        if args[0] == "scripts/setup_pipeline_dirs.py":
            return _completed(args, stdout=json.dumps(setup))
        if args[0] == "scripts/parse_pdf.py":
            (run_dir / ".pipeline" / "paper.md").write_text(
                " ".join(["method"] * 260),
                encoding="utf-8",
            )
            return _completed(args, stdout="parsed")
        if args[0] == "scripts/pdf_parse_quality.py":
            (run_dir / ".pipeline" / "parse_quality.json").write_text(
                json.dumps({"passed": True}),
                encoding="utf-8",
            )
            return _completed(args, stdout='{"passed": true}')
        raise AssertionError(args)

    monkeypatch.setattr(run_pipeline, "run_script", fake_run_script)

    paths, result = run_pipeline.run_stage_0("sample", workspace=str(repo))

    assert paths is not None
    assert result.status == "completed"
    assert [call[0] for call in calls] == [
        "scripts/setup_pipeline_dirs.py",
        "scripts/setup_pipeline_dirs.py",
        "scripts/parse_pdf.py",
        "scripts/pdf_parse_quality.py",
    ]
    assert "--resolve-only" in calls[0]
    assert "--resolve-only" not in calls[1]
    assert paths.paper_md in result.paths_written
    assert (run_dir / ".pipeline" / "parse_quality.json") in result.paths_written


def test_run_stage0_acquires_lock_before_pdf_parse(tmp_path, monkeypatch):
    repo = tmp_path
    pdf = repo / "input_papers" / "sample.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"%PDF-1.4\n")
    run_dir = repo / "r2c_runs" / "sample"
    (run_dir / ".pipeline").mkdir(parents=True)
    _write_current_progress(run_dir / ".pipeline")
    setup = _setup_result(repo, run_dir, pdf)
    calls: list[str] = []

    def fake_run_script(stage_id: str, args: list[str], *, timeout: int = 300):
        if args[0] == "scripts/setup_pipeline_dirs.py":
            calls.append("resolve" if "--resolve-only" in args else "setup")
            return _completed(args, stdout=json.dumps(setup))
        if args[0] == "scripts/parse_pdf.py":
            calls.append("parse")
            assert calls == ["resolve", "lock", "setup", "parse"]
            (run_dir / ".pipeline" / "paper.md").write_text(
                " ".join(["method"] * 260),
                encoding="utf-8",
            )
            return _completed(args, stdout="parsed")
        if args[0] == "scripts/pdf_parse_quality.py":
            calls.append("quality")
            (run_dir / ".pipeline" / "parse_quality.json").write_text(
                json.dumps({"passed": True}),
                encoding="utf-8",
            )
            return _completed(args, stdout='{"passed": true}')
        raise AssertionError(args)

    monkeypatch.setattr(run_pipeline, "run_script", fake_run_script)

    paths, result = run_pipeline.run_stage_0(
        "sample",
        workspace=str(repo),
        before_parse=lambda paths: calls.append("lock"),
    )

    assert paths is not None
    assert result.status == "completed"
    assert calls == ["resolve", "lock", "setup", "parse", "quality"]


def test_run_stage0_lock_contention_stops_before_setup_materialization(
    tmp_path, monkeypatch,
):
    repo = tmp_path
    pdf = repo / "input_papers" / "sample.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"%PDF-1.4\n")
    run_dir = repo / "r2c_runs" / "sample"
    setup = _setup_result(repo, run_dir, pdf)
    calls: list[list[str]] = []

    def fake_run_script(stage_id: str, args: list[str], *, timeout: int = 300):
        calls.append(args)
        assert args[0] == "scripts/setup_pipeline_dirs.py"
        assert "--resolve-only" in args
        return _completed(args, stdout=json.dumps(setup))

    monkeypatch.setattr(run_pipeline, "run_script", fake_run_script)

    with pytest.raises(RunDirectoryLockError):
        run_pipeline.run_stage_0(
            "sample",
            workspace=str(repo),
            before_parse=lambda paths: (_ for _ in ()).throw(
                RunDirectoryLockError("locked")
            ),
        )

    assert len(calls) == 1
    assert not (run_dir / ".pipeline" / "setup_result.json").exists()
    assert not (run_dir / ".pipeline" / "paper.md").exists()


def test_run_stage0_pdf_quality_gate_runs_on_cached_parse(tmp_path, monkeypatch):
    repo = tmp_path
    pdf = repo / "input_papers" / "sample.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"%PDF-1.4\n")
    run_dir = repo / "r2c_runs" / "sample"
    (run_dir / ".pipeline").mkdir(parents=True)
    (run_dir / ".pipeline" / "paper.md").write_text(
        " ".join(["method"] * 260),
        encoding="utf-8",
    )
    (run_dir / ".pipeline" / "parse_quality.json").write_text(
        json.dumps(_quality_report_for(pdf)),
        encoding="utf-8",
    )
    _write_current_progress(run_dir / ".pipeline")
    setup = _setup_result(repo, run_dir, pdf)
    calls: list[list[str]] = []

    def fake_run_script(stage_id: str, args: list[str], *, timeout: int = 300):
        calls.append(args)
        if args[0] == "scripts/setup_pipeline_dirs.py":
            return _completed(args, stdout=json.dumps(setup))
        if args[0] == "scripts/pdf_parse_quality.py":
            return _completed(args, stdout='{"passed": true}')
        raise AssertionError(args)

    monkeypatch.setattr(run_pipeline, "run_script", fake_run_script)

    paths, result = run_pipeline.run_stage_0("sample", workspace=str(repo))

    assert paths is not None
    assert result.status == "completed"
    assert [call[0] for call in calls] == [
        "scripts/setup_pipeline_dirs.py",
        "scripts/setup_pipeline_dirs.py",
        "scripts/pdf_parse_quality.py",
    ]
    assert "--resolve-only" in calls[0]
    assert "--resolve-only" not in calls[1]
    assert "--source-pdf" in calls[-1]


def test_run_stage0_archives_incompatible_dir_and_starts_fresh(tmp_path, monkeypatch):
    """Item 29 end to end: an old-version run dir (a valid-looking cached
    parse but NO current progress.json) is moved aside at stage 0 and the run
    re-parses into a fresh dir instead of reusing the stale cache. Nothing is
    deleted. Contrast with the cached-parse test above, where a compatible dir
    is reused and the parse is skipped."""
    repo = tmp_path
    pdf = repo / "input_papers" / "sample.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"%PDF-1.4\n")
    run_dir = repo / "r2c_runs" / "sample"
    (run_dir / ".pipeline").mkdir(parents=True)
    # A cache that WOULD be reused if the dir were compatible...
    (run_dir / ".pipeline" / "paper.md").write_text("# stale parse\n", encoding="utf-8")
    (run_dir / ".pipeline" / "parse_quality.json").write_text(
        json.dumps(_quality_report_for(pdf)),
        encoding="utf-8",
    )
    # ...but no progress.json (an older-version dir), plus a top-level file the
    # archive must preserve.
    (run_dir / "REPORT.md").write_text("# old delivery\n", encoding="utf-8")
    setup = _setup_result(repo, run_dir, pdf)
    calls: list[list[str]] = []

    def fake_run_script(stage_id: str, args: list[str], *, timeout: int = 300):
        calls.append(args)
        if args[0] == "scripts/setup_pipeline_dirs.py":
            return _completed(args, stdout=json.dumps(setup))
        if args[0] == "scripts/parse_pdf.py":
            (run_dir / ".pipeline" / "paper.md").write_text(
                " ".join(["method"] * 260),
                encoding="utf-8",
            )
            return _completed(args, stdout="parsed")
        if args[0] == "scripts/pdf_parse_quality.py":
            return _completed(args, stdout='{"passed": true}')
        raise AssertionError(args)

    monkeypatch.setattr(run_pipeline, "run_script", fake_run_script)

    paths, result = run_pipeline.run_stage_0("sample", workspace=str(repo))

    assert result.status == "completed"
    # The old dir was moved aside exactly once, never deleted, content intact.
    archives = list((repo / "r2c_runs").glob("sample_pre-upgrade-*"))
    assert len(archives) == 1
    assert (archives[0] / "REPORT.md").read_text(encoding="utf-8") == "# old delivery\n"
    # The stale cache did NOT satisfy the run — a fresh parse ran instead.
    assert "scripts/parse_pdf.py" in [call[0] for call in calls]
    assert "stale parse" not in paths.paper_md.read_text(encoding="utf-8")


def test_run_stage0_pdf_reparses_cached_markdown_without_quality_report(
    tmp_path, monkeypatch,
):
    repo = tmp_path
    pdf = repo / "input_papers" / "sample.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"%PDF-1.4\n")
    run_dir = repo / "r2c_runs" / "sample"
    (run_dir / ".pipeline").mkdir(parents=True)
    (run_dir / ".pipeline" / "paper.md").write_text(
        "# Old markdown shortcut\n",
        encoding="utf-8",
    )
    _write_current_progress(run_dir / ".pipeline")
    setup = _setup_result(repo, run_dir, pdf)
    calls: list[list[str]] = []

    def fake_run_script(stage_id: str, args: list[str], *, timeout: int = 300):
        calls.append(args)
        if args[0] == "scripts/setup_pipeline_dirs.py":
            return _completed(args, stdout=json.dumps(setup))
        if args[0] == "scripts/parse_pdf.py":
            (run_dir / ".pipeline" / "paper.md").write_text(
                " ".join(["method"] * 260),
                encoding="utf-8",
            )
            (run_dir / ".pipeline" / "parse_quality.json").write_text(
                json.dumps({"passed": True}),
                encoding="utf-8",
            )
            return _completed(args, stdout="parsed")
        if args[0] == "scripts/pdf_parse_quality.py":
            return _completed(args, stdout='{"passed": true}')
        raise AssertionError(args)

    monkeypatch.setattr(run_pipeline, "run_script", fake_run_script)

    paths, result = run_pipeline.run_stage_0("sample", workspace=str(repo))

    assert paths is not None
    assert result.status == "completed"
    assert [call[0] for call in calls] == [
        "scripts/setup_pipeline_dirs.py",
        "scripts/setup_pipeline_dirs.py",
        "scripts/parse_pdf.py",
        "scripts/pdf_parse_quality.py",
    ]
    assert "--resolve-only" in calls[0]
    assert "--resolve-only" not in calls[1]
    assert "Old markdown shortcut" not in paths.paper_md.read_text(encoding="utf-8")


def test_run_stage0_pdf_reparses_mismatched_quality_report(tmp_path, monkeypatch):
    repo = tmp_path
    pdf = repo / "input_papers" / "sample.pdf"
    other_pdf = repo / "input_papers" / "other.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"%PDF-1.4\nnew source\n")
    other_pdf.write_bytes(b"%PDF-1.4\nold source\n")
    run_dir = repo / "r2c_runs" / "sample"
    (run_dir / ".pipeline").mkdir(parents=True)
    (run_dir / ".pipeline" / "paper.md").write_text(
        "# Old paper text\n",
        encoding="utf-8",
    )
    (run_dir / ".pipeline" / "parse_quality.json").write_text(
        json.dumps(_quality_report_for(other_pdf)),
        encoding="utf-8",
    )
    _write_current_progress(run_dir / ".pipeline")
    setup = _setup_result(repo, run_dir, pdf)
    calls: list[list[str]] = []

    def fake_run_script(stage_id: str, args: list[str], *, timeout: int = 300):
        calls.append(args)
        if args[0] == "scripts/setup_pipeline_dirs.py":
            return _completed(args, stdout=json.dumps(setup))
        if args[0] == "scripts/parse_pdf.py":
            (run_dir / ".pipeline" / "paper.md").write_text(
                " ".join(["method"] * 260),
                encoding="utf-8",
            )
            (run_dir / ".pipeline" / "parse_quality.json").write_text(
                json.dumps(_quality_report_for(pdf)),
                encoding="utf-8",
            )
            return _completed(args, stdout="parsed")
        if args[0] == "scripts/pdf_parse_quality.py":
            return _completed(args, stdout='{"passed": true}')
        raise AssertionError(args)

    monkeypatch.setattr(run_pipeline, "run_script", fake_run_script)

    paths, result = run_pipeline.run_stage_0("sample", workspace=str(repo))

    assert paths is not None
    assert result.status == "completed"
    assert [call[0] for call in calls] == [
        "scripts/setup_pipeline_dirs.py",
        "scripts/setup_pipeline_dirs.py",
        "scripts/parse_pdf.py",
        "scripts/pdf_parse_quality.py",
    ]
    assert "--resolve-only" in calls[0]
    assert "--resolve-only" not in calls[1]
    assert "Old paper text" not in paths.paper_md.read_text(encoding="utf-8")


def test_run_stage0_pdf_halts_when_quality_gate_fails(tmp_path, monkeypatch):
    repo = tmp_path
    pdf = repo / "input_papers" / "sample.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"%PDF-1.4\n")
    run_dir = repo / "r2c_runs" / "sample"
    (run_dir / ".pipeline").mkdir(parents=True)
    (run_dir / ".pipeline" / "paper.md").write_text("too short", encoding="utf-8")
    (run_dir / ".pipeline" / "parse_quality.json").write_text(
        json.dumps(_quality_report_for(pdf)),
        encoding="utf-8",
    )
    _write_current_progress(run_dir / ".pipeline")
    setup = _setup_result(repo, run_dir, pdf)

    def fake_run_script(stage_id: str, args: list[str], *, timeout: int = 300):
        if args[0] == "scripts/setup_pipeline_dirs.py":
            return _completed(args, stdout=json.dumps(setup))
        if args[0] == "scripts/pdf_parse_quality.py":
            return _completed(args, stdout="{}", stderr="too short", code=1)
        raise AssertionError(args)

    monkeypatch.setattr(run_pipeline, "run_script", fake_run_script)

    paths, result = run_pipeline.run_stage_0("sample", workspace=str(repo))

    assert paths is not None
    assert result.status == "halted"
    assert result.notes == "PDF parse quality gate failed"
    assert (run_dir / ".pipeline" / "stage_0.halt").is_file()


def test_input_paper_markdown_has_no_embedded_base64_payloads():
    """The default `.md` for a paper carries prose, never image binaries.

    A converter that embeds figures as base64 produces a file that is
    almost entirely image data (ROMAN25 was 98% payload), and the producer
    agents open paper.md with their own read tools rather than receiving it
    inlined — so the analyzer reads a screen of base64 instead of the paper.

    The convention (maintainer decision 2026-07-27): keep the original bytes, do not edit
    them. Rename the payload-carrying file to `<stem>_images.md` and write a
    stripped copy back to `<stem>.md`, which stays the default the pipeline
    runs. The archive form is exempt here; every default must be clean.
    """
    offenders = []
    for path in Path("input_papers").glob("*.md"):
        if path.stem.endswith("_images"):
            continue  # sanctioned archive of the original converter output
        text = path.read_text(encoding="utf-8")
        if "data:image" in text or ";base64," in text:
            offenders.append(str(path))
    assert offenders == [], (
        "embedded base64 in a default input paper — archive it as "
        "<stem>_images.md and strip the copy that keeps the default name"
    )


def test_replace_payloads_swaps_each_data_uri_for_a_figure_placeholder():
    """Known-bad: both payload forms stripped, prose byte-identical outside."""
    from pdf_parse_quality import replace_payloads_with_placeholders

    text = (
        "Intro prose.\n"
        '<img src="data:image/png;base64,AAAA" alt="Fig 1">\n'
        "Middle prose.\n"
        "![caption](data:image/jpeg;base64,BBBB)\n"
        "Closing prose.\n"
    )
    stripped, count = replace_payloads_with_placeholders(text)
    assert count == 2
    assert "data:image" not in stripped
    assert '<img src="figure-1.png" alt="Fig 1">' in stripped
    assert "![caption](figure-2.png)" in stripped
    for line in ("Intro prose.", "Middle prose.", "Closing prose."):
        assert line in stripped


def test_replace_payloads_leaves_clean_text_unchanged():
    """Known-good: no payload means no rewrite, and the strip is idempotent."""
    from pdf_parse_quality import replace_payloads_with_placeholders

    clean = "Prose with a normal figure ![f](figure-1.png) reference.\n"
    stripped, count = replace_payloads_with_placeholders(clean)
    assert count == 0
    assert stripped == clean


def test_run_stage0_strips_payloads_from_cache_hit_paper_md(tmp_path, monkeypatch):
    """A run dir whose banked paper.md carries base64 gets stripped + archived
    on resume, without invalidating the parse cache (Interaction_Tax 08-27:
    90% of paper.md was payload and the analyzer read 830k tokens of it)."""
    repo = tmp_path
    pdf = repo / "input_papers" / "sample.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"%PDF-1.4\n")
    run_dir = repo / "r2c_runs" / "sample"
    (run_dir / ".pipeline").mkdir(parents=True)
    prose = " ".join(["method"] * 260)
    payload_md = (
        f"{prose}\n"
        '<img src="data:image/png;base64,' + "A" * 5000 + '" alt="Fig 1">\n'
    )
    (run_dir / ".pipeline" / "paper.md").write_text(payload_md, encoding="utf-8")
    (run_dir / ".pipeline" / "parse_quality.json").write_text(
        json.dumps(_quality_report_for(pdf)),
        encoding="utf-8",
    )
    _write_current_progress(run_dir / ".pipeline")
    setup = _setup_result(repo, run_dir, pdf)

    def fake_run_script(stage_id: str, args: list[str], *, timeout: int = 300):
        if args[0] == "scripts/setup_pipeline_dirs.py":
            return _completed(args, stdout=json.dumps(setup))
        if args[0] == "scripts/pdf_parse_quality.py":
            return _completed(args, stdout='{"passed": true}')
        raise AssertionError(args)  # parse_pdf.py must NOT run: cache is valid

    monkeypatch.setattr(run_pipeline, "run_script", fake_run_script)

    paths, result = run_pipeline.run_stage_0("sample", workspace=str(repo))

    assert result.status == "completed"
    stripped = paths.paper_md.read_text(encoding="utf-8")
    assert "data:image" not in stripped
    assert '<img src="figure-1.png" alt="Fig 1">' in stripped
    assert prose in stripped
    archive = paths.paper_md.with_name("paper_images.md")
    assert archive.read_text(encoding="utf-8") == payload_md


def test_image_archives_keep_the_original_bytes_beside_the_default():
    """Each `<stem>_images.md` archive has a live default `<stem>.md`."""
    for archive in Path("input_papers").glob("*_images.md"):
        default = archive.with_name(f"{archive.stem[:-len('_images')]}.md")
        assert default.is_file(), (
            f"{archive.name} has no stripped default beside it; the archive "
            f"is never the file the pipeline runs"
        )
