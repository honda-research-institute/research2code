"""Stage 3 — runtime smoke gate for the generated notebook.

Runs after `scripts/validate_notebook_output.py` confirms the notebook
satisfies its structural contract. This script EXECUTES the notebook
end-to-end with nbclient and fails the pipeline on any cell error or
timeout. Catches the failure class that purely-static validators miss:
type errors at dispatch sites, undefined symbols, runtime schema drift
between the package and the notebook, etc.

Strategy:

  1. Open `<run_dir>/notebook.ipynb`.
  2. Execute it in a Python 3 kernel from `<run_dir>` as cwd (so relative
     paths in the notebook resolve correctly — e.g., `method/example_data/`).
  3. Apply a per-cell timeout (default: 180s). A smoke run is meant to be
     fast (a healthy run finishes well under a minute); the budget leaves
     headroom for a first-run dataset download + a CPU train-from-scratch
     round, while still failing a runaway / mis-scaled cell quickly.
  4. On any cell exception, exit 1 with the failing cell index, the cell
     source (truncated), and the last ~30 lines of the traceback.
  5. After a clean execution, scan the executed cells for recorded
     error-type outputs (exceptions the display formatter swallowed while
     the cell reply stayed ok, e.g. at figure-draw time). An untagged hit
     fails exactly like an in-cell crash; cells tagged `raises-exception`
     (minted only by the partial-delivery stub renderer) are exempt.
  6. On success, save the executed notebook back to `notebook.ipynb` and
     exit 0 with a one-line summary (cell count, total time).

The delivered notebook is the researcher-facing artifact, so a successful
smoke run persists its executed outputs. Failed executions are not saved; the
on-disk notebook then stays at the last clean render or clean execution.

Usage:

    python scripts/smoke_run_notebook.py --run-dir <output_dir> [--timeout 180]

Exit codes:
  0  notebook executed end-to-end without error
  1  one or more cells failed (producer-routable; the notebook is the suspect)
  2  pipeline-fault setup error (notebook missing or unparseable) — nothing to ship
  3  environmental setup error (nbclient/nbformat not importable, or no usable
     Jupyter kernel) — the rendered notebook exists; the driver degrades and
     continues to delivery rather than discarding a complete package. Split out
     from exit 2 so the orchestrator can tell "the environment is wrong" from
     "the artifact is wrong" (R2C halt->degrade redesign, type-(I) vs type-(T)).
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_NOTEBOOK_EXEC_IMPORT_ERROR: ImportError | None = None
try:
    import nbformat  # noqa: E402
except ImportError as e:  # pragma: no cover - exercised by monkeypatch in tests
    _NOTEBOOK_EXEC_IMPORT_ERROR = e
    nbformat = None  # type: ignore[assignment]

try:
    from nbclient import NotebookClient  # noqa: E402
    from nbclient.exceptions import CellExecutionError, CellTimeoutError  # noqa: E402
except ImportError as e:  # pragma: no cover - exercised in envs without nbclient
    if _NOTEBOOK_EXEC_IMPORT_ERROR is None:
        _NOTEBOOK_EXEC_IMPORT_ERROR = e
    NotebookClient = None  # type: ignore[assignment]

    class CellExecutionError(Exception):  # type: ignore[no-redef]
        pass

    class CellTimeoutError(Exception):  # type: ignore[no-redef]
        pass


def _truncate(s: str, n: int = 800) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"\n... (truncated; full length {len(s)})"


def _head_tail(s: str, n: int = 2000) -> str:
    """Head+tail excerpt with the cut marked. Python tracebacks put the
    exception message and deepest frame at the END, so a head-only slice
    hides exactly what a diagnostician needs (ACC 2026-07-14: every
    diagnosis prompt carried call-site frames and a bare "RuntimeError"
    with no message, and 4 of 4 blinded dispatches burned their output
    budget hunting for it)."""
    if len(s) <= n:
        return s
    marker = f"\n... (middle truncated; full length {len(s)}) ...\n"
    keep = n - len(marker)
    head = s[: keep // 2]
    tail = s[-(keep - len(head)):]
    return head + marker + tail


def _cell_preview(source: str, n: int = 70) -> str:
    """One-line preview of a cell's source for progress logging.

    Strips leading blank lines and comments-only lines so the preview shows
    the first meaningful statement (which is more useful than e.g. the file's
    docstring or import block when scanning a long log).
    """
    if not source:
        return "(empty)"
    text = source if isinstance(source, str) else "".join(source)
    # First non-blank, non-comment line
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") and not stripped.startswith("#%"):
            continue  # skip comment lines, but keep cell-magic-style lines
        if len(stripped) > n:
            return stripped[: n - 1] + "…"
        return stripped
    # Fallback: use the first line as-is
    first = text.splitlines()[0] if text else ""
    return first[: n - 1] + "…" if len(first) > n else first


def _make_progress_callbacks(
    n_total_cells: int, n_code_cells: int,
) -> tuple[dict, dict]:
    """Build nbclient progress callbacks and expose their attribution state.

    Logs `start: <preview>` before each code cell and `done: Xs` after.
    Markdown cells are silent (they're instant and add noise). Output is
    line-flushed to stderr so progress is visible live even when stdout is
    captured by a wrapping process (e.g., the orchestrator's Bash tool).

    nbclient passes `cell` and `cell_index` as kwargs to these callbacks,
    plus `execute_reply` to `on_cell_executed`.

    The exact last-started notebook index is also returned to the timeout
    handler.  nbclient can report the next cell after an interrupted timeout;
    the callback is the source of truth for which code cell was actually sent.
    """
    state = {
        "code_idx": 0,
        "cell_t0": None,
        "last_started_code_cell_index": None,
    }

    def on_cell_start(cell, cell_index):
        if cell.cell_type == "code":
            state["code_idx"] += 1
            state["cell_t0"] = time.time()
            state["last_started_code_cell_index"] = cell_index
            preview = _cell_preview(cell.source)
            print(
                f"  [code {state['code_idx']:>2}/{n_code_cells}  "
                f"cell {cell_index:>2}/{n_total_cells - 1}]  start: {preview}",
                file=sys.stderr,
                flush=True,
            )

    def on_cell_executed(cell, cell_index, execute_reply):
        if cell.cell_type == "code" and state["cell_t0"] is not None:
            elapsed = time.time() - state["cell_t0"]
            print(
                f"  [code {state['code_idx']:>2}/{n_code_cells}  "
                f"cell {cell_index:>2}/{n_total_cells - 1}]   done: {elapsed:6.1f}s",
                file=sys.stderr,
                flush=True,
            )
            state["cell_t0"] = None

    callbacks = {
        "on_cell_start": on_cell_start,
        "on_cell_executed": on_cell_executed,
    }
    return callbacks, state


def _resolve_kernel(requested: str) -> str:
    """Return the Jupyter kernel name to execute the notebook with.

    The smoke gate must run in the SAME interpreter the driver and validators
    used — the `run_pipeline.py` PYTHON_CMD/sys.executable invariant — or it
    proves nothing about THIS package's environment. The historical default,
    a hardcoded ``python3`` kernelspec, can be absent (then nbclient raises
    ``NoSuchKernel``, the run-3 failure mode) or can point at an unrelated venv.

    So prefer a registered kernelspec whose interpreter resolves to the current
    ``sys.executable``. If introspection fails or nothing matches, return
    ``requested`` unchanged and let NotebookClient surface ``NoSuchKernel``,
    which `main` classifies as an environmental (exit 3) failure. This never
    returns an empty/None value, so a faked NotebookClient (tests) is unaffected.
    """
    try:
        from jupyter_client.kernelspec import KernelSpecManager

        ksm = KernelSpecManager()
        specs = ksm.find_kernel_specs()
    except Exception:
        return requested
    current = str(Path(sys.executable).resolve())
    for name in specs:
        try:
            argv = ksm.get_kernel_spec(name).argv or []
        except Exception:
            continue
        if argv and Path(str(argv[0])).name.startswith("python"):
            try:
                if str(Path(argv[0]).resolve()) == current:
                    return name
            except Exception:
                continue
    return requested


def _unsanctioned_error_outputs(nb) -> list[int]:
    """Cell indices carrying error-type outputs without the sanctioned tag.

    nbclient's allow_errors=False only fails on a cell whose execute REPLY
    is an error. An exception that fires at figure-draw time inside the
    display formatter is swallowed into a recorded error OUTPUT while the
    reply stays ok (SRL 2026-07-29 cell 59: a deferred-validation matplotlib
    kwarg shipped as a recorded ValueError under a clean smoke verdict; RCA
    2026-08-03, finding 2). The `raises-exception` tag is exempt: only the
    partial-delivery stub renderer mints it, paired with a validated stub
    record. Stream outputs (stderr warnings) are not gated."""
    hits = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        if "raises-exception" in (cell.get("metadata", {}).get("tags") or []):
            continue
        if any(o.get("output_type") == "error" for o in cell.get("outputs", [])):
            hits.append(i)
    return hits


def _format_failure(nb, cell_idx: int, exc: Exception) -> str:
    cell = nb.cells[cell_idx] if 0 <= cell_idx < len(nb.cells) else None
    parts = [f"notebook execution failed at cell {cell_idx}"]

    if cell is not None:
        src = cell.get("source", "")
        parts.append(f"\n--- failing cell source ({cell.get('cell_type')}) ---")
        parts.append(_truncate(src))

        outputs = cell.get("outputs", [])
        for out in outputs:
            if out.get("output_type") == "error":
                # The exception name + message lead, first-class: they are
                # the single most diagnostic line and must survive any
                # excerpting (they are also clean text, while the traceback
                # strings carry ANSI color codes).
                ename = str(out.get("ename") or "").strip()
                evalue = str(out.get("evalue") or "").strip()
                if ename or evalue:
                    parts.append("\n--- error ---")
                    parts.append(_truncate(f"{ename}: {evalue}", 600))
                tb = out.get("traceback") or []
                tb_text = "\n".join(tb)
                parts.append("\n--- error traceback ---")
                parts.append(_head_tail(tb_text, 2000))
                break
        else:
            # No error output recorded; fall back to the exception text.
            parts.append("\n--- exception ---")
            parts.append(_truncate(str(exc), 1000))

    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute the generated notebook end-to-end as a smoke gate.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing notebook.ipynb (also used as cwd during execution).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-cell timeout in seconds (default: 180). A smoke run should be fast; this leaves headroom for a first-run dataset download + a CPU train-from-scratch round while failing a runaway / mis-scaled cell quickly.",
    )
    parser.add_argument(
        "--kernel",
        default="python3",
        help="Jupyter kernel name (default: python3).",
    )
    args = parser.parse_args(argv)

    if _NOTEBOOK_EXEC_IMPORT_ERROR is not None:
        print(
            "error: missing notebook execution dependency: "
            f"{_NOTEBOOK_EXEC_IMPORT_ERROR}",
            file=sys.stderr,
        )
        # Environmental, not the artifact's fault: nbclient/nbformat are absent
        # from the smoke environment. Exit 3 so the driver degrades + continues
        # rather than discarding a fully rendered notebook (ENV-1).
        return 3

    nb_path = args.run_dir / "notebook.ipynb"
    if not nb_path.exists():
        print(f"error: notebook not found at {nb_path}", file=sys.stderr)
        return 2

    try:
        nb = nbformat.read(nb_path, as_version=4)
    except Exception as e:
        print(f"error: cannot parse {nb_path}: {e}", file=sys.stderr)
        return 2

    n_cells = len(nb.cells)
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"smoke-run: {nb_path} ({n_cells} cells, {n_code} code) — per-cell timeout {args.timeout}s")

    kernel_name = _resolve_kernel(args.kernel)
    if kernel_name != args.kernel:
        print(f"smoke-run: resolved kernel '{args.kernel}' -> '{kernel_name}' "
              f"(bound to {sys.executable})")

    progress_callbacks, progress_state = _make_progress_callbacks(n_cells, n_code)
    client = NotebookClient(
        nb,
        timeout=args.timeout,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(args.run_dir)}},
        allow_errors=False,
        **progress_callbacks,
    )

    t0 = time.time()
    try:
        client.execute()
    except CellTimeoutError as e:
        # Distinguish the two underlying causes:
        #   (i)  the kernel is alive and a code cell genuinely overran the per-cell budget
        #   (ii) the kernel died (OOM-kill, crash) BEFORE producing a reply, and nbclient
        #        sat waiting on the next cell's send — the reported cell_index is then the
        #        cell that *would* have been sent next, not the one that actually hung. If
        #        that cell is markdown, that's the giveaway: markdown cells don't execute.
        cell_idx = progress_state.get("last_started_code_cell_index")
        if cell_idx is None:
            cell_idx = getattr(e, "cell_index", None)
        if cell_idx is None:
            # No callback/index evidence is available (mainly older/faked
            # nbclient implementations). nbclient increments this count when
            # it SENDS a code cell, before waiting for the reply, so convert
            # the one-based started count to a zero-based code-cell position.
            code_positions = [i for i, c in enumerate(nb.cells)
                              if c.cell_type == "code"]
            count = int(getattr(client, "code_cells_executed", 0) or 0)
            started_position = max(count - 1, 0)
            cell_idx = (code_positions[min(started_position, len(code_positions) - 1)]
                        if code_positions else 0)
        elapsed = time.time() - t0

        kernel_alive = True
        try:
            kc = getattr(client, "kc", None)
            if kc is not None and hasattr(kc, "is_alive"):
                kernel_alive = bool(kc.is_alive())
        except Exception:
            kernel_alive = True  # if we can't tell, don't claim it died

        reported_cell = nb.cells[cell_idx] if 0 <= cell_idx < len(nb.cells) else None
        reported_is_markdown = (
            reported_cell is not None and reported_cell.cell_type == "markdown"
        )

        if not kernel_alive or reported_is_markdown:
            # The kernel is gone (or the timeout is on a non-executing cell, which means
            # the kernel died before it could acknowledge the previous one).
            # Only a dead kernel supports the environmental reading — with the
            # kernel still alive, say what is actually known instead of
            # asserting "environmental, not a notebook bug" (that confident
            # wording steered the diagnostician away from a real mis-scaled
            # cell on the 2026-07-02 GBALD run).
            last_code_idx = None
            for i in range(min(cell_idx, len(nb.cells) - 1), -1, -1):
                if nb.cells[i].cell_type == "code":
                    last_code_idx = i
                    break
            if kernel_alive:
                cause = (
                    "The kernel is still alive, so this is NOT confirmed as an "
                    "environmental kill. The timeout attribution is ambiguous — "
                    "treat the last code cell sent (below) as the prime suspect "
                    "for a genuine per-cell-budget overrun, and an external "
                    "kill/OOM as the fallback reading."
                )
            else:
                cause = (
                    "Most likely cause: OS-level kill (OOM under memory "
                    "pressure, or external SIGKILL). This is an environmental "
                    "failure, not a notebook bug. Free memory (close other "
                    "apps) and retry, or run on a beefier machine."
                )
            print(
                f"notebook execution FAILED — kernel channel broke "
                f"after {elapsed:.1f}s (alive={kernel_alive}; "
                f"nbclient was waiting on cell {cell_idx}, which is "
                f"{'markdown' if reported_is_markdown else 'code'}).\n"
                f"{cause}\n"
                f"Last code cell sent to the kernel: "
                f"{f'cell {last_code_idx}' if last_code_idx is not None else 'unknown'}.",
                file=sys.stderr,
            )
            if last_code_idx is not None:
                src = nb.cells[last_code_idx].get("source", "")
                print(
                    f"\n--- last code cell sent (cell {last_code_idx}) ---\n{_truncate(src)}",
                    file=sys.stderr,
                )
            # Tail-anchored index repeat: the driver reads only the TAIL of
            # this stderr, and the header line scrolled out of the window
            # once the failure excerpt grew (detr 2026-07-15 03:xx halt:
            # "smoke gate failed but could not parse failing cell index").
            print(f"\n(failing cell index: "
                  f"{last_code_idx if last_code_idx is not None else cell_idx})",
                  file=sys.stderr)
            return 1

        # Genuine cell timeout — kernel alive, cell is code, ran past the budget.
        print(
            f"notebook execution TIMED OUT at cell {cell_idx} "
            f"after {args.timeout}s (wall {elapsed:.1f}s).\n"
            f"This cell ran longer than the per-cell budget — either:\n"
            f"  (a) the cell legitimately needs more time on this hardware "
            f"(re-run with `--timeout <larger>`), or\n"
            f"  (b) the cell has an infinite loop / pathological complexity "
            f"(inspect the source and route to the appropriate producer for fix).",
            file=sys.stderr,
        )
        if reported_cell is not None:
            src = reported_cell.get("source", "")
            print(f"\n--- timed-out cell source ---\n{_truncate(src)}", file=sys.stderr)
        # Tail-anchored index repeat (see the kernel-death branch).
        print(f"\n(failing cell index: {cell_idx})", file=sys.stderr)
        return 1
    except CellExecutionError as e:
        # nbclient stores the failing cell index on the exception
        cell_idx = getattr(e, "cell_index", None)
        if cell_idx is None:
            # Fallback: scan for the first cell with an error output
            for i, cell in enumerate(nb.cells):
                if cell.cell_type == "code" and any(
                    o.get("output_type") == "error" for o in cell.get("outputs", [])
                ):
                    cell_idx = i
                    break
        cell_idx = cell_idx if cell_idx is not None else -1
        print(_format_failure(nb, cell_idx, e), file=sys.stderr)
        # Tail-anchored index repeat (see the kernel-death branch).
        print(f"\n(failing cell index: {cell_idx})", file=sys.stderr)
        return 1
    except Exception as e:
        # NoSuchKernel (and any "no such kernel" report) is an environmental
        # failure — the interpreter the driver ran under has no usable Jupyter
        # kernel registered. The notebook itself is fine, so exit 3 (degrade +
        # continue) rather than 2, and point the operator at the fix (ENV-5).
        if type(e).__name__ == "NoSuchKernel" or "No such kernel" in str(e):
            print(f"error: no usable Jupyter kernel: {e}", file=sys.stderr)
            print(
                "Register a kernel for this interpreter "
                f"(`{sys.executable} -m ipykernel install --user`) or install "
                "ipykernel, then re-run.",
                file=sys.stderr,
            )
            return 3
        print(f"error: unexpected execution failure: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 2

    # Post-execution error-output gate: execution "succeeded" (every reply
    # ok), but a recorded exception in the outputs still means a broken
    # artifact would ship. Fail exactly like an in-cell crash so the smoke
    # diagnostician fix loop takes it, and do not save.
    error_cells = _unsanctioned_error_outputs(nb)
    if error_cells:
        cell_idx = error_cells[0]
        print(
            f"cell {cell_idx} executed with an ok reply but recorded an "
            f"error-type output (an exception swallowed by the display "
            f"formatter, e.g. at figure-draw time) — a recorded exception "
            f"must not ship as a passed smoke run.",
            file=sys.stderr,
        )
        print(
            _format_failure(nb, cell_idx, RuntimeError(
                "recorded error output under an ok execute reply")),
            file=sys.stderr,
        )
        if len(error_cells) > 1:
            print(f"(further cells with recorded error outputs: "
                  f"{error_cells[1:]})", file=sys.stderr)
        # Tail-anchored index repeat (see the kernel-death branch).
        print(f"\n(failing cell index: {cell_idx})", file=sys.stderr)
        return 1

    try:
        nbformat.write(nb, nb_path)
    except Exception as e:
        print(f"error: executed notebook could not be saved to {nb_path}: {e}", file=sys.stderr)
        return 2

    elapsed = time.time() - t0
    print(f"smoke-run: passed ({n_code} code cells executed cleanly in {elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
