"""Tests for scripts/refresh_derived_blocks.py — the shared marked-block
refresh engine (foundation for Japan-feedback notes 4 and 5).

Covers: a marked block updates when the delivered source changes, unmarked
content stays byte-identical, an unknown block kind fails honestly,
idempotence (running twice changes nothing), the notebook markdown-cell
path, and the block-kind registry seam notes 4/5 will register into.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import refresh_derived_blocks as rdb
from refresh_derived_blocks import (
    DerivedBlockError,
    END_MARKER,
    RefreshSetupError,
    format_begin_marker,
    main,
    make_block,
    refresh_markdown_file,
    refresh_notebook,
    refresh_run_documents,
    refresh_text,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_METHOD_V1 = '''\
def foo(x):
    """Add one."""
    return x + 1


class C:
    def m(self, y):
        return y * 2
'''

_METHOD_V2 = _METHOD_V1.replace("x + 1", "x + 2")


def _foo_marker() -> str:
    return format_begin_marker(
        "source_listing",
        {"file": "method/method.py", "qualname": "foo"})


def _method_md(marker: str) -> str:
    return "\n".join([
        "# Fixture method doc",
        "",
        "Prose above the block.",
        "",
        marker,
        "(stale content to be replaced)",
        END_MARKER,
        "",
        "Prose below the block.",
        "",
    ])


@pytest.fixture
def run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "method").mkdir(parents=True)
    (run_dir / "method" / "method.py").write_text(
        _METHOD_V1, encoding="utf-8")
    (run_dir / "METHOD.md").write_text(
        _method_md(_foo_marker()), encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# The core contract: refresh only the marked block, honestly, idempotently
# ---------------------------------------------------------------------------


def test_marked_block_updates_when_source_changes(run):
    md = run / "METHOD.md"
    seen, changed = refresh_markdown_file(md, run)
    assert (seen, changed) == (1, 1)
    text = md.read_text(encoding="utf-8")
    assert "x + 1" in text
    assert "(stale content to be replaced)" not in text
    assert "method/method.py:1" in text  # the location line

    # The companion edits the delivered code; the refresh picks it up.
    (run / "method" / "method.py").write_text(_METHOD_V2, encoding="utf-8")
    seen, changed = refresh_markdown_file(md, run)
    assert (seen, changed) == (1, 1)
    text = md.read_text(encoding="utf-8")
    assert "x + 2" in text
    assert "x + 1" not in text


def test_unmarked_content_is_byte_identical(run):
    md = run / "METHOD.md"
    before = md.read_text(encoding="utf-8").split("\n")
    refresh_markdown_file(md, run)
    after = md.read_text(encoding="utf-8").split("\n")
    begin_b, end_b = before.index(_foo_marker()), before.index(END_MARKER)
    begin_a, end_a = after.index(_foo_marker()), after.index(END_MARKER)
    # Everything outside the block — including both marker lines — is
    # byte-identical; only the interior was rewritten.
    assert after[:begin_a + 1] == before[:begin_b + 1]
    assert after[end_a:] == before[end_b:]


def test_idempotent_second_run_changes_nothing(run):
    md = run / "METHOD.md"
    refresh_markdown_file(md, run)
    frozen = md.read_bytes()
    seen, changed = refresh_markdown_file(md, run)
    assert (seen, changed) == (1, 0)
    assert md.read_bytes() == frozen


def test_unknown_kind_fails_honestly(run):
    marker = format_begin_marker("no_such_kind", {"x": 1})
    (run / "METHOD.md").write_text(_method_md(marker), encoding="utf-8")
    with pytest.raises(DerivedBlockError) as exc:
        refresh_markdown_file(run / "METHOD.md", run)
    msg = str(exc.value)
    assert "no_such_kind" in msg
    assert "source_listing" in msg  # names the registered kinds
    # The failed refresh left the document untouched.
    assert "(stale content to be replaced)" in \
        (run / "METHOD.md").read_text(encoding="utf-8")


def test_unmatched_markers_fail(run):
    with pytest.raises(DerivedBlockError, match="no end"):
        refresh_text(_foo_marker() + "\ninterior", run)
    with pytest.raises(DerivedBlockError, match="without a begin"):
        refresh_text("text\n" + END_MARKER, run)
    nested = "\n".join(
        [_foo_marker(), _foo_marker(), "x", END_MARKER, END_MARKER])
    with pytest.raises(DerivedBlockError, match="nest"):
        refresh_text(nested, run)


def test_invalid_marker_payload_fails(run):
    bad = "<!-- derived-block: begin {not json} -->\nx\n" + END_MARKER
    with pytest.raises(DerivedBlockError, match="invalid JSON"):
        refresh_text(bad, run)


# ---------------------------------------------------------------------------
# The built-in source_listing kind
# ---------------------------------------------------------------------------


def test_source_listing_handles_method_qualnames(run):
    block = make_block(
        "source_listing",
        {"file": "method/method.py", "qualname": "C.m"}, run)
    assert "def m(self, y):" in block
    assert "`C.m`" in block


def test_source_listing_missing_function_fails_honestly(run):
    with pytest.raises(DerivedBlockError, match="gone_fn"):
        make_block("source_listing",
                   {"file": "method/method.py", "qualname": "gone_fn"}, run)


def test_source_listing_missing_file_fails_honestly(run):
    with pytest.raises(DerivedBlockError, match="not found"):
        make_block("source_listing",
                   {"file": "method/nope.py", "qualname": "foo"}, run)


def test_source_listing_incomplete_spec_fails(run):
    with pytest.raises(DerivedBlockError, match="spec"):
        make_block("source_listing", {"file": "method/method.py"}, run)


def test_make_block_is_idempotent_from_birth(run):
    # A block minted through the engine round-trips unchanged: producers
    # (notes 4/5) inject exactly what a later refresh would derive.
    doc = "before\n" + make_block(
        "source_listing",
        {"file": "method/method.py", "qualname": "foo"}, run) + "\nafter\n"
    new_doc, seen, changed = refresh_text(doc, run)
    assert (seen, changed) == (1, 0)
    assert new_doc == doc


# ---------------------------------------------------------------------------
# Notebook markdown cells
# ---------------------------------------------------------------------------


def _write_notebook(path: Path, marker: str) -> None:
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    nb = new_notebook(cells=[
        new_markdown_cell("# Demo notebook"),
        new_markdown_cell("\n".join([
            "How `foo` is implemented:",
            "",
            marker,
            "(stale listing)",
            END_MARKER,
        ])),
        new_code_cell("from method.method import foo\nfoo(1)"),
    ])
    nbformat.write(nb, str(path))


def test_notebook_marked_cell_updates_others_untouched(run):
    import nbformat

    nb_path = run / "notebook.ipynb"
    _write_notebook(nb_path, _foo_marker())
    before = nbformat.read(str(nb_path), as_version=4)

    seen, changed = refresh_notebook(nb_path, run)
    assert (seen, changed) == (1, 1)

    after = nbformat.read(str(nb_path), as_version=4)
    assert "def foo(x):" in after.cells[1].source
    assert "(stale listing)" not in after.cells[1].source
    # Unmarked cells are untouched.
    assert after.cells[0].source == before.cells[0].source
    assert after.cells[2].source == before.cells[2].source
    assert len(after.cells) == len(before.cells)


def test_notebook_refresh_is_idempotent(run):
    nb_path = run / "notebook.ipynb"
    _write_notebook(nb_path, _foo_marker())
    refresh_notebook(nb_path, run)
    frozen = nb_path.read_bytes()
    seen, changed = refresh_notebook(nb_path, run)
    assert (seen, changed) == (1, 0)
    # Nothing changed, so the file was not rewritten at all.
    assert nb_path.read_bytes() == frozen


# ---------------------------------------------------------------------------
# The registry seam notes 4/5 will use
# ---------------------------------------------------------------------------


def test_registered_kind_is_used_and_duplicates_rejected(run, monkeypatch):
    # Isolate the registry so the test kind never leaks into other tests.
    monkeypatch.setattr(rdb, "_BLOCK_KINDS", dict(rdb._BLOCK_KINDS))

    def _derive_echo(spec: dict, _run_dir: Path) -> str:
        return f"echo: {spec['text']}"

    rdb.register_block_kind("echo", _derive_echo)
    with pytest.raises(ValueError, match="already registered"):
        rdb.register_block_kind("echo", _derive_echo)

    doc = "\n".join([
        format_begin_marker("echo", {"text": "hello"}),
        "stale",
        END_MARKER,
    ])
    new_doc, seen, changed = refresh_text(doc, run)
    assert (seen, changed) == (1, 1)
    assert "echo: hello" in new_doc


# ---------------------------------------------------------------------------
# Run-level entrypoint + CLI
# ---------------------------------------------------------------------------


def test_refresh_run_documents_default_targets(run):
    # Only METHOD.md exists; notebook.ipynb is silently absent by default.
    results = refresh_run_documents(run)
    assert results == {"METHOD.md": (1, 1)}

    _write_notebook(run / "notebook.ipynb", _foo_marker())
    results = refresh_run_documents(run)
    assert results == {"METHOD.md": (1, 0), "notebook.ipynb": (1, 1)}


def test_refresh_run_documents_explicit_missing_target_is_setup_error(run):
    with pytest.raises(RefreshSetupError, match="not found"):
        refresh_run_documents(run, ["nope.md"])


def test_cli_refresh(run, capsys):
    assert main(["--run-dir", str(run)]) == 0
    out = capsys.readouterr().out
    assert "METHOD.md: 1 derived block(s), 1 rewritten" in out


def test_cli_unknown_kind_exits_one(run, capsys):
    marker = format_begin_marker("no_such_kind", {})
    (run / "METHOD.md").write_text(_method_md(marker), encoding="utf-8")
    assert main(["--run-dir", str(run)]) == 1
    assert "no_such_kind" in capsys.readouterr().err


def test_cli_missing_run_dir_exits_two(tmp_path, capsys):
    assert main(["--run-dir", str(tmp_path / "nope")]) == 2
    assert "error:" in capsys.readouterr().err


def test_marker_payload_is_deterministic():
    # sort_keys in the marker payload: two producers minting the same spec
    # emit byte-identical markers (stable diffs in the run's git history).
    a = format_begin_marker("k", {"b": 1, "a": 2})
    b = format_begin_marker("k", {"a": 2, "b": 1})
    assert a == b
    payload = json.loads(a[len("<!-- derived-block: begin "):-len(" -->")])
    assert payload == {"kind": "k", "spec": {"a": 2, "b": 1}}
