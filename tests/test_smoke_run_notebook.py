"""Tests for the notebook smoke runner's artifact semantics."""

from __future__ import annotations

import json

from nbformat.v4 import new_output

import smoke_run_notebook


def _write_notebook(path):
    path.write_text(
        json.dumps({
            "cells": [{
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": "print('hello')\n",
            }],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }) + "\n",
        encoding="utf-8",
    )


def test_smoke_success_saves_executed_notebook(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    nb_path = run_dir / "notebook.ipynb"
    _write_notebook(nb_path)

    class FakeNotebookClient:
        code_cells_executed = 1

        def __init__(self, nb, **kwargs):
            self.nb = nb

        def execute(self):
            self.nb.cells[0].execution_count = 1
            self.nb.cells[0].outputs = [
                new_output("stream", name="stdout", text="hello\n")
            ]

    monkeypatch.setattr(smoke_run_notebook, "NotebookClient", FakeNotebookClient)
    monkeypatch.setattr(smoke_run_notebook, "_NOTEBOOK_EXEC_IMPORT_ERROR", None)

    rc = smoke_run_notebook.main([
        "--run-dir", str(run_dir),
        "--timeout", "1",
    ])

    assert rc == 0
    saved = json.loads(nb_path.read_text(encoding="utf-8"))
    assert saved["cells"][0]["execution_count"] == 1
    assert "".join(saved["cells"][0]["outputs"][0]["text"]) == "hello\n"


def _write_notebook_with_cells(path, cells):
    path.write_text(
        json.dumps({
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }) + "\n",
        encoding="utf-8",
    )


def _code_cell(source, tags=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": tags} if tags else {},
        "outputs": [],
        "source": source,
    }


def _ok_reply_client(monkeypatch, error_output_cells):
    """Fake client: every reply ok, but the given cells record an error
    output — the display-formatter-swallowed class (SRL cell 59)."""

    class FakeNotebookClient:
        code_cells_executed = 1

        def __init__(self, nb, **kwargs):
            self.nb = nb

        def execute(self):
            for i, cell in enumerate(self.nb.cells):
                if cell.cell_type != "code":
                    continue
                cell.execution_count = 1
                if i in error_output_cells:
                    cell.outputs = [new_output(
                        "error", ename="ValueError",
                        evalue="markevery is iterable but not a valid form",
                        traceback=["Traceback (most recent call last):",
                                   "ValueError: markevery ..."])]
                else:
                    cell.outputs = [new_output("stream", name="stdout",
                                               text="ok\n")]

    monkeypatch.setattr(smoke_run_notebook, "NotebookClient",
                        FakeNotebookClient)
    monkeypatch.setattr(smoke_run_notebook, "_NOTEBOOK_EXEC_IMPORT_ERROR",
                        None)


def test_recorded_error_output_fails_gate_and_does_not_save(
        tmp_path, monkeypatch, capsys):
    # RCA 2026-08-03 finding 2 (R2C-039 block 1): an ok-reply execution
    # carrying a recorded error output must fail like an in-cell crash,
    # with a parseable failing-cell index, and must not be saved.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    nb_path = run_dir / "notebook.ipynb"
    _write_notebook_with_cells(nb_path, [
        _code_cell("import numpy as np\n"),
        _code_cell("ax.plot(x, y, markevery=(0.0, 0.1))\n"),
    ])
    before = nb_path.read_text(encoding="utf-8")
    _ok_reply_client(monkeypatch, error_output_cells={1})

    rc = smoke_run_notebook.main(["--run-dir", str(run_dir), "--timeout", "1"])

    err = capsys.readouterr().err
    assert rc == 1
    assert nb_path.read_text(encoding="utf-8") == before
    assert "(failing cell index: 1)" in err
    assert "ValueError" in err and "markevery" in err
    assert "recorded an error-type output" in err
    assert "notebook execution failed at cell 1" in err


def test_tagged_stub_cell_error_output_passes(tmp_path, monkeypatch):
    # The partial-delivery exemption: only the stub renderer mints the
    # raises-exception tag, and exactly those cells may record an error.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    nb_path = run_dir / "notebook.ipynb"
    _write_notebook_with_cells(nb_path, [
        _code_cell("raise NotImplementedError('stubbed element')\n",
                   tags=["raises-exception"]),
    ])
    _ok_reply_client(monkeypatch, error_output_cells={0})

    rc = smoke_run_notebook.main(["--run-dir", str(run_dir), "--timeout", "1"])

    assert rc == 0
    saved = json.loads(nb_path.read_text(encoding="utf-8"))
    assert saved["cells"][0]["outputs"][0]["output_type"] == "error"


def test_stream_stderr_warning_still_passes(tmp_path, monkeypatch):
    # Scope boundary: stream outputs are not gated (warnings are legitimate).
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    nb_path = run_dir / "notebook.ipynb"
    _write_notebook_with_cells(nb_path, [_code_cell("print('x')\n")])

    class FakeNotebookClient:
        code_cells_executed = 1

        def __init__(self, nb, **kwargs):
            self.nb = nb

        def execute(self):
            self.nb.cells[0].execution_count = 1
            self.nb.cells[0].outputs = [new_output(
                "stream", name="stderr",
                text="UserWarning: divide by zero encountered\n")]

    monkeypatch.setattr(smoke_run_notebook, "NotebookClient",
                        FakeNotebookClient)
    monkeypatch.setattr(smoke_run_notebook, "_NOTEBOOK_EXEC_IMPORT_ERROR",
                        None)

    rc = smoke_run_notebook.main(["--run-dir", str(run_dir), "--timeout", "1"])

    assert rc == 0


def test_smoke_failure_does_not_save_partial_execution(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    nb_path = run_dir / "notebook.ipynb"
    _write_notebook(nb_path)
    before = nb_path.read_text(encoding="utf-8")

    class FakeNotebookClient:
        code_cells_executed = 1

        def __init__(self, nb, **kwargs):
            self.nb = nb

        def execute(self):
            self.nb.cells[0].execution_count = 1
            self.nb.cells[0].outputs = [
                new_output("stream", name="stdout", text="partial\n")
            ]
            raise RuntimeError("boom")

    monkeypatch.setattr(smoke_run_notebook, "NotebookClient", FakeNotebookClient)
    monkeypatch.setattr(smoke_run_notebook, "_NOTEBOOK_EXEC_IMPORT_ERROR", None)

    rc = smoke_run_notebook.main([
        "--run-dir", str(run_dir),
        "--timeout", "1",
    ])

    assert rc == 2
    assert nb_path.read_text(encoding="utf-8") == before


def test_missing_nbclient_dependency_exits_setup_error(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_notebook(run_dir / "notebook.ipynb")

    monkeypatch.setattr(
        smoke_run_notebook,
        "_NOTEBOOK_EXEC_IMPORT_ERROR",
        ModuleNotFoundError("No module named 'nbclient'"),
    )

    rc = smoke_run_notebook.main([
        "--run-dir", str(run_dir),
        "--timeout", "1",
    ])

    # Missing nbclient/nbformat is environmental (exit 3), not a pipeline-fault
    # setup error (exit 2): the rendered notebook exists, the env does not have
    # the runtime. The driver degrades-and-continues on exit 3.
    assert rc == 3
    assert "missing notebook execution dependency" in capsys.readouterr().err


def _write_mixed_notebook(path):
    """Markdown/code interleaving in the GBALD shape: code cells at
    positions 1, 3, and 5, markdown between."""
    def md(text):
        return {"cell_type": "markdown", "metadata": {}, "source": text}

    def code(src):
        return {"cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": src}

    path.write_text(
        json.dumps({
            "cells": [
                md("# title"),
                code("import x\n"),
                md("## params"),
                code("y = 1\n"),
                md("## loop"),
                code("while True: pass  # the genuinely slow cell\n"),
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }) + "\n",
        encoding="utf-8",
    )


def _fake_timeout_client(
    code_cells_executed_count,
    kernel_is_alive,
    *,
    last_started_cell_index=None,
    reported_cell_index=None,
):
    class _KC:
        def is_alive(self):
            return kernel_is_alive

    class FakeNotebookClient:
        code_cells_executed = code_cells_executed_count

        def __init__(self, nb, **kwargs):
            self.nb = nb
            self.kc = _KC()
            self.on_cell_start = kwargs.get("on_cell_start")

        def execute(self):
            if last_started_cell_index is not None and self.on_cell_start:
                self.on_cell_start(
                    cell=self.nb.cells[last_started_cell_index],
                    cell_index=last_started_cell_index,
                )
            error = smoke_run_notebook.CellTimeoutError("budget expired")
            if reported_cell_index is not None:
                error.cell_index = reported_cell_index
            raise error

    return FakeNotebookClient


def test_timeout_count_maps_to_code_cell_not_markdown(tmp_path, monkeypatch, capsys):
    """CellTimeoutError without a cell_index: the executed-code-cell COUNT
    must map through code-cell positions to the hung CODE cell. Using it as
    a raw index landed on markdown and misclassified a slow-cell timeout as
    a kernel death (the 2026-07-02 GBALD run, twice)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_mixed_notebook(run_dir / "notebook.ipynb")

    # nbclient increments the count when it sends the third code cell, before
    # waiting for its reply. No callback is fired by this legacy-style fake,
    # so the count fallback must subtract one before mapping code positions.
    monkeypatch.setattr(smoke_run_notebook, "NotebookClient",
                        _fake_timeout_client(3, kernel_is_alive=True))
    monkeypatch.setattr(smoke_run_notebook, "_NOTEBOOK_EXEC_IMPORT_ERROR", None)

    rc = smoke_run_notebook.main(["--run-dir", str(run_dir), "--timeout", "1"])
    err = capsys.readouterr().err

    assert rc == 1
    assert "TIMED OUT at cell 5" in err
    assert "genuinely slow cell" in err          # names the real source
    assert "environmental failure" not in err    # no false kill claim
    assert "appears to have died" not in err


def test_timeout_prefers_last_started_cell_over_next_reported_cell(
    tmp_path, monkeypatch, capsys
):
    """The progress callback beats nbclient's next-cell timeout attribution."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_mixed_notebook(run_dir / "notebook.ipynb")

    # Cell 3 was sent and never completed. The timeout exception points at
    # cell 5, the next code cell after markdown cell 4 (the live detr shape).
    monkeypatch.setattr(
        smoke_run_notebook,
        "NotebookClient",
        _fake_timeout_client(
            2,
            kernel_is_alive=True,
            last_started_cell_index=3,
            reported_cell_index=5,
        ),
    )
    monkeypatch.setattr(smoke_run_notebook, "_NOTEBOOK_EXEC_IMPORT_ERROR", None)

    rc = smoke_run_notebook.main(["--run-dir", str(run_dir), "--timeout", "1"])
    err = capsys.readouterr().err

    assert rc == 1
    assert "TIMED OUT at cell 3" in err
    assert "y = 1" in err
    assert "genuinely slow cell" not in err
    assert "(failing cell index: 3)" in err


def test_dead_kernel_still_reports_environmental(tmp_path, monkeypatch, capsys):
    """A genuinely dead kernel keeps the environmental reading."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_mixed_notebook(run_dir / "notebook.ipynb")

    monkeypatch.setattr(smoke_run_notebook, "NotebookClient",
                        _fake_timeout_client(3, kernel_is_alive=False))
    monkeypatch.setattr(smoke_run_notebook, "_NOTEBOOK_EXEC_IMPORT_ERROR", None)

    rc = smoke_run_notebook.main(["--run-dir", str(run_dir), "--timeout", "1"])
    err = capsys.readouterr().err

    assert rc == 1
    assert "alive=False" in err
    assert "environmental failure" in err


def test_alive_kernel_with_markdown_attribution_hedges(tmp_path, monkeypatch, capsys):
    """When nbclient itself attributes the timeout to a markdown cell and the
    kernel is ALIVE, the message must hedge (attribution ambiguous), never
    assert 'environmental, not a notebook bug'."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_mixed_notebook(run_dir / "notebook.ipynb")

    Fake = _fake_timeout_client(0, kernel_is_alive=True)

    class FakeWithIndex(Fake):
        def execute(self):
            e = smoke_run_notebook.CellTimeoutError("budget expired")
            e.cell_index = 2  # markdown position, straight from the exception
            raise e

    monkeypatch.setattr(smoke_run_notebook, "NotebookClient", FakeWithIndex)
    monkeypatch.setattr(smoke_run_notebook, "_NOTEBOOK_EXEC_IMPORT_ERROR", None)

    rc = smoke_run_notebook.main(["--run-dir", str(run_dir), "--timeout", "1"])
    err = capsys.readouterr().err

    assert rc == 1
    assert "NOT confirmed as an environmental kill" in err
    assert "prime suspect" in err
    assert "environmental failure, not a notebook bug" not in err


def test_format_failure_keeps_exception_message_and_traceback_tail():
    """The ACC 2026-07-14 blindness: a >2000-char traceback was sliced
    head-only, dropping the exception message Python puts at the END, and
    the smoke diagnostician burned 4 of 4 dispatches hunting for it. The
    failure excerpt must lead with ename/evalue first-class and keep both
    ends of the traceback."""
    import nbformat

    nb = nbformat.v4.new_notebook()
    frames = [f'  File "method/x.py", line {i}, in f{i}\n    f{i + 1}()'
              for i in range(60)]
    tb = ["Traceback (most recent call last):", *frames,
          "RuntimeError: Matrix product with incompatible dimensions. "
          "Lhs is 4x1 and rhs is 4x1."]
    cell = nbformat.v4.new_code_cell("plan()")
    cell.outputs = [new_output(
        "error", ename="RuntimeError",
        evalue="Matrix product with incompatible dimensions. "
               "Lhs is 4x1 and rhs is 4x1.",
        traceback=tb)]
    nb.cells = [cell]

    msg = smoke_run_notebook._format_failure(nb, 0, Exception("boom"))
    # ename/evalue lead, first-class and untruncatable by the tb excerpt.
    assert "RuntimeError: Matrix product with incompatible dimensions" in msg
    # Both ends of the traceback survive, the cut is marked.
    assert "Traceback (most recent call last):" in msg
    assert "middle truncated" in msg
    assert "Lhs is 4x1 and rhs is 4x1" in msg.split("error traceback")[1]


def test_format_failure_short_traceback_unchanged():
    import nbformat

    nb = nbformat.v4.new_notebook()
    cell = nbformat.v4.new_code_cell("x")
    cell.outputs = [new_output(
        "error", ename="ValueError", evalue="bad",
        traceback=["Traceback:", "ValueError: bad"])]
    nb.cells = [cell]
    msg = smoke_run_notebook._format_failure(nb, 0, Exception("boom"))
    assert "ValueError: bad" in msg
    assert "middle truncated" not in msg
