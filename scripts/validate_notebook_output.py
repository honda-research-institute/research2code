"""Stage 3 — validator for the generated notebook.

Runs after `scripts/render_notebook.py` produces `<run_dir>/notebook.ipynb`.
Confirms the notebook satisfies the structural and content invariants the
taxonomy node's `notebook_layout` declares, plus a few hard contract rules:

  1. notebook.ipynb exists and parses as a valid nbformat v4 notebook.
  2. Every section ID declared in `notebook_layout.sections` is represented
     by at least one cell whose markdown heading matches the section's title.
  3. **No PLACEHOLDER markers remain** — they should all have been substituted
     by render_notebook.py.
  4. The §1 setup code cell exists and includes `%matplotlib inline`.
  5. The §0 install code cell uses `%pip install` (NOT `!pip install`).
  6. The params dict cell exists with a `params = {` line, contains every
     parameter declared in params.json, and the values match (mechanical
     check via AST).
  7. **Used-param coverage**: every parameter in params.json with
     `used_in_notebook: true` is referenced as `cfg["<name>"]` (or
     `cfg['<name>']`) in at least one code cell. Conversely, no param with
     `used_in_notebook: false` is referenced.
  8. §5.1's acquisition loop cell calls `select_batch(...)` (heuristic AST
     check; we look for the pluggable function name from spec).

This is a deterministic gate. On failure, exit 1 with a structured error
list on stderr.

Usage:

    python scripts/validate_notebook_output.py --spec <method_spec.json> --run-dir <output_dir>

Exit codes:
  0  validation passed (zero errors)
  1  validation failures
  2  setup error (missing inputs)
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbformat  # noqa: E402

from scripts import taxonomy  # noqa: E402
from scripts.evaluation_protocol_rendering import (  # noqa: E402
    render_evaluation_protocol_block,
)
from demo_data_provenance import (  # noqa: E402
    disclosure_for,
    provenance_path,
    read_provenance,
)
from schemas.method_spec import (  # noqa: E402
    evaluation_protocol_candidate_values,
)

# Notebook AST analysis for AL loops lives in the dependency-free
# probes.nb_ast module (the portable harness vendors the probes package;
# this validator keeps nbformat/taxonomy). Re-imported here so existing
# consumers and tests keep one implementation and one import path.
from probes.nb_ast import (  # noqa: E402,F401
    _al_loop_has_postmerge_count_for_premerge_eval,
    _al_loop_warm_starts,
    _called_names_within,
    _cfg_key,
    _expand_through_helpers,
    _integer_literal,
    _is_label_index_target,
    _is_num_rounds_expr,
    _iterates_num_rounds,
    _notebook_call_graph,
    _stmt_appends_learning_curve,
    _stmt_calls_expanded,
    _stmt_calls_name,
    _stmt_mutates_label_indices,
)


PLACEHOLDER_PATTERN = re.compile(r"PLACEHOLDER:\s*\S+")


def _heading_prefix(template: str) -> str:
    """Strip a heading template down to the section-number prefix.

    The taxonomy notebook_layout section titles can contain `{method_name}` placeholders
    (e.g., '## 4. The {method_name} method ⭐'). We match by the prefix up to
    the first `{` since the slot is method-specific.
    """
    template = template.strip()
    if "{" in template:
        return template.split("{", 1)[0].rstrip()
    return template


_ANCHOR_LINE_RE = re.compile(r'^\s*<a\s+id\s*=\s*"[^"]*"\s*>\s*</a>\s*$')


def _first_meaningful_line(cell_source: str) -> str:
    """Return the first non-blank, non-anchor line of a markdown cell.

    The render script injects `<a id="…"></a>` anchor lines above ## / ###
    headings to make TOC links portable across Jupyter renderers; this helper
    skips those so heading-detection sees the heading itself.
    """
    for line in (cell_source or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _ANCHOR_LINE_RE.match(line):
            continue
        return stripped
    return ""


def _has_heading(cell_source: str, heading: str) -> bool:
    """Cell starts with the heading prefix (allowing slot substitutions in templates)."""
    return _first_meaningful_line(cell_source).startswith(_heading_prefix(heading))


def _find_cell_with_heading(cells: list, heading: str):
    """Find the first markdown cell whose first line starts with the heading's prefix."""
    for c in cells:
        if c.cell_type == "markdown" and _has_heading(c.source, heading):
            return c
    return None


def _all_code_cells(cells: list) -> list[str]:
    return [c.source for c in cells if c.cell_type == "code"]


def _params_dict_in_code(code_cells: list[str]) -> dict | None:
    """Find the cell containing `params = {...}` and parse the dict literal.
    Returns the params dict (just names → entry-dict) or None if not found.
    """
    for src in code_cells:
        if not src.lstrip().startswith("params = {") and "\nparams = {" not in src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "params" and isinstance(node.value, ast.Dict):
                        try:
                            return ast.literal_eval(node.value)
                        except (ValueError, SyntaxError):
                            return None
    return None


def _extract_cfg_keys(code_cells: list[str]) -> set[str]:
    """Find every `cfg["X"]` or `cfg['X']` reference across code cells. Returns the set of X."""
    found: set[str] = set()
    pattern = re.compile(r"""cfg\[\s*['"]([^'"]+)['"]\s*\]""")
    for src in code_cells:
        for m in pattern.finditer(src):
            found.add(m.group(1))
    return found


def _cfg_keys_in_expr(node: ast.AST) -> set[str]:
    keys: set[str] = set()
    for inner in ast.walk(node):
        key = _cfg_key(inner)
        if key is not None:
            keys.add(key)
    return keys


def _call_name(node: ast.Call) -> str:
    fn = node.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return ""


def _signature_param_names(sig_str: str) -> list[str]:
    try:
        tree = ast.parse(f"def {sig_str}:\n    pass\n")
    except (SyntaxError, ValueError):
        return []
    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        return []
    args = tree.body[0].args
    return [arg.arg for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]]


def _active_learning_pluggable_batch_size_errors(
    code_cells: list[str],
    *,
    pluggable_name: str,
    expected_sig: str,
) -> list[str]:
    """Validate notebook call sites use params["batch_size"] as selector count."""
    if not pluggable_name:
        return []
    expected_params = _signature_param_names(expected_sig)
    batch_size_pos = (
        expected_params.index("batch_size")
        if "batch_size" in expected_params
        else None
    )
    errors: list[str] = []
    for cell_idx, src in enumerate(code_cells):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) != pluggable_name:
                continue
            batch_expr: ast.AST | None = None
            for kw in node.keywords:
                if kw.arg == "batch_size":
                    batch_expr = kw.value
                    break
            if batch_expr is None and batch_size_pos is not None and len(node.args) > batch_size_pos:
                batch_expr = node.args[batch_size_pos]
            if batch_expr is None:
                errors.append(
                    f"notebook code cell {cell_idx} calls `{pluggable_name}(...)` "
                    "without passing the required `batch_size` argument."
                )
                continue
            cfg_keys = _cfg_keys_in_expr(batch_expr)
            legacy_keys = sorted(cfg_keys & {"batch_output", "batch_outputs"})
            if legacy_keys:
                errors.append(
                    f"notebook code cell {cell_idx} passes legacy output-count "
                    f"config key(s) {legacy_keys} "
                    f"as `{pluggable_name}`'s `batch_size`. Active-learning "
                    "selectors must use `cfg[\"batch_size\"]`; `batch_returns` may "
                    "only control a larger candidate prefilter."
                )
            elif cfg_keys and "batch_size" not in cfg_keys:
                errors.append(
                    f"notebook code cell {cell_idx} passes `{ast.unparse(batch_expr)}` "
                    f"as `{pluggable_name}`'s `batch_size`; selector output counts "
                    "must derive from `cfg[\"batch_size\"]`."
                )
            elif not cfg_keys and not (
                isinstance(batch_expr, ast.Name) and batch_expr.id == "batch_size"
            ):
                errors.append(
                    f"notebook code cell {cell_idx} passes literal/expression "
                    f"`{ast.unparse(batch_expr)}` as `{pluggable_name}`'s "
                    "`batch_size`; use `cfg[\"batch_size\"]` so params.json is the "
                    "single runtime source of truth."
                )
    return errors


# Conversion methods that EXIT a type: the result is a value whose type no
# longer has that method (tolist -> list has no .tolist; numpy -> ndarray has
# no .numpy; toarray/todense -> dense array has no .toarray). Reassigning a
# variable to the result of one of these called ON ITSELF is non-idempotent:
# the first run flips the variable's type, the second run calls a now-missing
# method and raises. (GBALD: `labeled_idx = set(labeled_idx.tolist())` raised
# `'set' object has no attribute 'tolist'` on re-run, making the AL-1
# loop-bookkeeping probe unprobeable.)
_TYPE_EXITING_CONVERSIONS = {"tolist", "numpy", "toarray", "todense"}


def _receiver_root_name(node: ast.AST) -> str | None:
    """Return the root Name id of an attribute/subscript receiver chain."""
    cur = node
    while isinstance(cur, (ast.Attribute, ast.Subscript)):
        cur = cur.value
    return cur.id if isinstance(cur, ast.Name) else None


def _reassigns_self_via_type_exiting_conversion(rhs: ast.AST, name: str) -> str | None:
    """If `rhs` contains a `name.<conv>(...)` call where <conv> exits the type,
    return the conversion method name; else None."""
    for inner in ast.walk(rhs):
        if (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr in _TYPE_EXITING_CONVERSIONS
            and _receiver_root_name(inner.func.value) == name
        ):
            return inner.func.attr
    return None


def _non_idempotent_cell_errors(code_cells: list[str]) -> list[str]:
    """Flag top-level cell statements that reassign a variable from a
    type-exiting conversion of its own previous value.

    Narrow on purpose: this is a backstop gate, and the broad principle ("don't
    reassign a variable from a transform of its own previous value") lives in
    the notebook-generator rule R11. Here we only flag the high-confidence,
    re-execution-breaking shape `x = ... x.tolist()/.numpy()/... ...` so the
    gate never false-blocks legitimate idempotent reassignments (`x = x.cuda()`,
    `x = x.reshape(...)`, `x = np.union1d(x, y)`). Cells re-run in a live kernel
    (and the probe harness re-executes the loop cell), so a non-idempotent cell
    is a real defect, not a style nit."""
    errors: list[str] = []
    for cell_idx, src in enumerate(code_cells):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for stmt in tree.body:  # top-level cell statements only (skip loop accumulation)
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name) or stmt.value is None:
                continue
            conv = _reassigns_self_via_type_exiting_conversion(stmt.value, target.id)
            if conv is not None:
                errors.append(
                    f"notebook code cell {cell_idx}: `{target.id} = ... {target.id}.{conv}() ...` "
                    f"reassigns `{target.id}` from a type-exiting conversion of its own previous "
                    f"value. This is not idempotent: re-running the cell finds `{target.id}` already "
                    f"converted, so `.{conv}()` raises (the GBALD run hit `'set' object has no "
                    f"attribute 'tolist'`, which made the AL-1 loop-bookkeeping probe unprobeable). "
                    f"Derive `{target.id}` from a STABLE source (e.g. the immutable bootstrap result) "
                    f"so the cell re-runs cleanly, per rule R11."
                )
    return errors


def _extract_method_imports(code_cells: list[str]) -> dict[str, set[str]]:
    """Parse `from method import ...` and `from method.<sub> import ...` statements.

    Returns: {module_path: {symbol, ...}}. Module paths are "method", "method.method",
    "method.model", etc. — whichever submodule the import line names.
    """
    out: dict[str, set[str]] = {}
    for src in code_cells:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if not node.module or not node.module.startswith("method"):
                continue
            names = {alias.name for alias in node.names if alias.name != "*"}
            out.setdefault(node.module, set()).update(names)
    return out


def _read_package_all(method_dir: Path) -> set[str] | None:
    """Read `<method_dir>/__init__.py`'s `__all__` literal, return the set of names.

    Returns None if __init__.py is missing or doesn't have a parseable __all__.
    """
    init_path = method_dir / "__init__.py"
    if not init_path.is_file():
        return None
    try:
        tree = ast.parse(init_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                try:
                    return set(ast.literal_eval(node.value))
                except (ValueError, SyntaxError):
                    return None
    return None


def _read_module_top_level_names(module_path: Path) -> set[str] | None:
    """Read a method/<sub>.py file and return the set of top-level public names
    (functions and classes whose name doesn't start with `_`)."""
    if not module_path.is_file():
        return None
    try:
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return None
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                out.add(node.name)
    return out


def _is_num_rounds_minus(node: ast.AST, amount: int) -> bool:
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Sub)
        and _is_num_rounds_expr(node.left)
        and _integer_literal(node.right) == amount
    )


def _guard_skips_last_round(test: ast.AST, loop_var: str) -> bool:
    """Return True for guards equivalent to `r < num_rounds - 1`.

    This is the failure shape that makes AL notebooks run N displayed rounds
    while performing only N-1 acquisitions. The check is intentionally narrow:
    it recognizes the common static form and avoids interpreting arbitrary
    control flow.
    """
    if not isinstance(test, ast.Compare):
        return False
    if not isinstance(test.left, ast.Name) or test.left.id != loop_var:
        return False
    if len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    comparator = test.comparators[0]
    if isinstance(test.ops[0], ast.Lt):
        return _is_num_rounds_minus(comparator, 1)
    if isinstance(test.ops[0], ast.LtE):
        return _is_num_rounds_minus(comparator, 2)
    return False


def _al_loop_skips_last_acquisition(code_cells: list[str], *, pluggable_name: str) -> bool:
    """Detect AL loops that only call the acquisition function on N-1 rounds."""
    for src in code_cells:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for loop in ast.walk(tree):
            if not isinstance(loop, ast.For):
                continue
            if not isinstance(loop.target, ast.Name):
                continue
            loop_var = loop.target.id
            guarded_acquisition = False
            unguarded_acquisition = False
            for stmt in loop.body:
                if (
                    isinstance(stmt, ast.If)
                    and _guard_skips_last_round(stmt.test, loop_var)
                    and pluggable_name in _called_names_within(stmt)
                ):
                    guarded_acquisition = True
                    continue
                if pluggable_name in _called_names_within(stmt):
                    unguarded_acquisition = True
            if guarded_acquisition and not unguarded_acquisition:
                return True
    return False


def _al_loop_has_unused_final_acquisition(code_cells: list[str], *, pluggable_name: str) -> bool:
    """Detect AL loops that acquire every round but only evaluate before acquisition.

    Failure shape:
      for r in range(cfg["num_rounds"]):
          learning_curve.append(...)
          select_batch(...)

    With `num_rounds` defined as acquisition rounds, that loop acquires a final
    batch whose retrained model is never evaluated or plotted. Correct notebooks
    either append after acquisition inside the loop or put evaluation in a helper
    called after the labeled/unlabeled update.
    """
    for src in code_cells:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for loop in ast.walk(tree):
            if not isinstance(loop, ast.For) or not _iterates_num_rounds(loop):
                continue
            first_unconditional_acq_idx: int | None = None
            append_indices: list[int] = []
            for idx, stmt in enumerate(loop.body):
                if _stmt_appends_learning_curve(stmt):
                    append_indices.append(idx)
                if isinstance(stmt, ast.If) and isinstance(loop.target, ast.Name):
                    if _guard_skips_last_round(stmt.test, loop.target.id):
                        continue
                if first_unconditional_acq_idx is None and _stmt_calls_name(stmt, pluggable_name):
                    first_unconditional_acq_idx = idx
            if first_unconditional_acq_idx is None:
                continue
            has_eval_before = any(idx < first_unconditional_acq_idx for idx in append_indices)
            has_eval_after = any(idx > first_unconditional_acq_idx for idx in append_indices)
            if has_eval_before and not has_eval_after:
                return True
    return False


_INVERSE_DISTANCE_RE = re.compile(
    r"(?is)(?:\b1\s*/\s*(?:min_)?(?:dist|distance)|\bR\s*[_-]?\s*0\s*/[^.\n]{0,120}(?:dist|distance)|/[^.\n]{0,120}(?:dist|distance))"
)
_FARTHER_BETTER_RE = re.compile(
    r"(?is)(?:"
    r"\bfarther\b[^.\n]{0,120}\b(?:higher|larger|greater|more\s+representative|higher\s+score|selected|preferred|favored|better)\b"
    r"|"
    r"\b(?:higher|larger|greater)\s+distances?\b"
    r"|"
    r"\b(?:higher|larger|greater)\b[^.\n]{0,120}\b(?:farther|more\s+distant)\b"
    r"|"
    r"\bmore\s+distant\b[^.\n]{0,120}\b(?:higher|larger|greater|representative|selected|preferred|favored|better)\b"
    r"|"
    r"\bfar\s+from\b[^.\n]{0,160}\b(?:high|higher|larger|greater|more\s+representative|higher\s+score|selected|preferred|favored|better)\b"
    r"|"
    r"\b(?:high|higher|larger|greater)\b[^.\n]{0,160}\bfar\s+from\b"
    r")"
)


def _inverse_distance_markdown_contradictions(markdown_cells: list[str]) -> list[int]:
    """Return markdown cell indices that claim inverse-distance rewards farther samples."""
    hits: list[int] = []
    for idx, src in enumerate(markdown_cells):
        if _INVERSE_DISTANCE_RE.search(src or "") and _FARTHER_BETTER_RE.search(src or ""):
            hits.append(idx)
    return hits


_TABLE_SEPARATOR_RE = re.compile(r"^\s*:?-{3,}:?\s*$")
_NUMBER_TOKEN_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _is_table_separator(cells: list[str]) -> bool:
    return bool(cells) and all(_TABLE_SEPARATOR_RE.fullmatch(cell) for cell in cells)


def _clean_param_cell(cell: str) -> str:
    cleaned = re.sub(r"`([^`]+)`", r"\1", cell)
    cleaned = cleaned.replace("**", "").strip()
    return cleaned


def _cell_mentions_value(cell: str, value: object) -> bool:
    if isinstance(value, bool):
        return str(value).lower() in cell.lower()
    if isinstance(value, (int, float)):
        numeric_text = re.sub(r"(?<=\d)\s+(?=\d{3}\b)", "", cell.replace(",", ""))
        for token in _NUMBER_TOKEN_RE.findall(numeric_text):
            try:
                candidate = float(token)
            except ValueError:
                continue
            actual = float(value)
            tolerance = max(1e-9, abs(actual) * 1e-3)
            if abs(candidate - actual) <= tolerance:
                return True
        return False
    normalized = str(value).strip("'\"")
    return normalized in cell


def _markdown_runtime_param_table_errors(
    markdown_cells: list[str],
    params_json: dict,
) -> list[str]:
    """Find runtime/demo/system value table cells that drift from params.json.

    A "demo value"/"smoke value" column normally must equal the params.json
    value the notebook actually runs. The exception is a *disclosed* demo-scale
    override table: when the same table also carries a baseline column (spec
    default / full value / benchmark value / default), the notebook is openly
    documenting an intentional downscale from the baseline it names, per the
    method spec's demo_scale_implementation. That is the same disclosed
    demo-scale trade-off Stage 5 routes to assumptions.md instead of halting, so
    the override column is exempt here too. The baseline column is still
    anchored to params.json (the disclosure must name the real value the deriver
    produced), and an undisclosed demo/smoke column — one with no baseline
    beside it — must still match params.json, so genuine staleness is still
    caught.
    """
    errors: list[str] = []
    runtime_header_names = {"system value", "demo value", "runtime value", "smoke value"}
    # Columns whose value is an intentionally reduced demo/smoke figure.
    override_header_names = {"demo value", "smoke value"}
    # Columns naming the un-reduced baseline a demo override is measured against.
    baseline_header_names = {
        "spec default", "spec value", "default", "default value",
        "full value", "benchmark value",
    }
    for cell_idx, source in enumerate(markdown_cells):
        rows = [_split_table_row(line) for line in (source or "").splitlines()]
        rows = [row for row in rows if row]
        if len(rows) < 3:
            continue
        for idx, header in enumerate(rows[:-1]):
            if idx + 1 >= len(rows) or not _is_table_separator(rows[idx + 1]):
                continue
            header_l = [h.lower().strip() for h in header]
            baseline_cols = [
                i for i, name in enumerate(header_l)
                if name in baseline_header_names
            ]
            has_baseline = bool(baseline_cols)
            # In a disclosed override table the demo/smoke column is an
            # intentional, documented downscale — exempt it. Without a baseline
            # beside it, the same column must still match params.json.
            runtime_cols = [
                i for i, name in enumerate(header_l)
                if name in runtime_header_names
                and not (has_baseline and name in override_header_names)
            ]
            if not runtime_cols and not has_baseline:
                continue
            for row in rows[idx + 2:]:
                if len(row) < len(header) or _is_table_separator(row):
                    continue
                name = _clean_param_cell(row[0])
                if name not in params_json:
                    continue
                entry = params_json.get(name) or {}
                if not isinstance(entry, dict) or "value" not in entry:
                    continue
                value = entry.get("value")
                source_kind = entry.get("source")
                cols_to_check = list(runtime_cols)
                # Anchor the baseline column to params.json's value. For a
                # spec/system default that value IS the baseline; for a
                # paper-sourced param the full figure lives in paper_value, so
                # skip baseline anchoring there to avoid a false mismatch (the
                # demo column is already exempt).
                if has_baseline and source_kind != "paper":
                    cols_to_check += baseline_cols
                for col in cols_to_check:
                    cell = row[col]
                    if source_kind == "paper" and cell.strip() in {"—", "-", ""}:
                        continue
                    if not _cell_mentions_value(cell, value):
                        errors.append(
                            f"notebook markdown cell {cell_idx} has stale runtime "
                            f"value for `{name}` in table column `{header[col]}`: "
                            f"{cell!r}; params.json has {value!r}."
                        )
    return errors


def _evaluation_protocol_render_errors(
    spec: dict,
    params_data: dict,
    markdown_cells: list[str],
) -> list[str]:
    """Require one deterministic role-separated protocol truth surface."""
    expected = render_evaluation_protocol_block(
        spec,
        params_data,
        heading="### Evaluation protocol",
    )
    if not expected:
        return []

    normalized: list[str] = []
    for source in markdown_cells:
        rendered = (source or "").strip()
        # render_notebook's TOC pass prepends a deterministic anchor to every
        # heading after placeholder expansion. The protocol bytes following
        # that navigation-only prefix must still match exactly.
        rendered = re.sub(r'^<a id="[^"]+"></a>\s*', "", rendered)
        normalized.append(rendered)

    exact = [
        index for index, rendered in enumerate(normalized)
        if rendered == expected.strip()
    ]
    errors: list[str] = []
    if len(exact) != 1:
        errors.append(
            "notebook is missing or has altered the deterministic evaluation "
            "protocol block. Paper context, one-call forecast horizon, "
            "validation span, test span, and runtime/demo values must remain "
            "role-separated; rerun render_notebook.py from the typed "
            "method_spec and params."
        )

    protocol_heading = re.compile(
        r"^\s*#{2,6}\s+evaluation\s+protocol\s*#*\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    competing = [
        index for index, rendered in enumerate(normalized)
        if protocol_heading.search(rendered) and index not in exact
    ]
    if len(exact) == 1 and competing:
        errors.append(
            "notebook contains competing hand-written evaluation-protocol "
            f"section(s) in markdown cell(s) {competing}. Keep exactly the "
            "deterministic typed block; a second K, validation-span, or "
            "test-span summary can reintroduce fabricated paper truth."
        )

    if len(exact) == 1:
        protocol = ((spec.get("comparison") or {}).get("evaluation_protocol")
                    or {})
        quantities = protocol.get("quantities") or []
        entries = params_data.get("params", params_data)
        def _same_number(left: object, right: object) -> bool:
            return (
                isinstance(left, (int, float))
                and not isinstance(left, bool)
                and isinstance(right, (int, float))
                and not isinstance(right, bool)
                and float(left) == float(right)
            )

        narrative_errors: list[str] = []
        for cell_index, rendered in enumerate(normalized):
            if cell_index in exact:
                continue
            # Markdown generators wrap prose freely. Reconstitute logical
            # paragraphs/sentences before checking claims so a line break or
            # long qualifier cannot separate K from its asserted value.
            chunks = re.split(
                r"\n\s*\n|^\s*[-*+]\s+",
                rendered,
                flags=re.MULTILINE,
            )
            logical_spans: list[str] = []
            for chunk in chunks:
                chunk = " ".join(chunk.split())
                if not chunk or chunk.lstrip().startswith("#"):
                    continue
                logical_spans.extend(
                    part.strip()
                    for part in re.split(r"(?<=[.!?;])\s+", chunk)
                    if part.strip()
                )

            for span in logical_spans:
                for quantity in quantities:
                    if not isinstance(quantity, dict):
                        continue
                    candidates = evaluation_protocol_candidate_values(
                        quantity, span
                    )
                    if not candidates:
                        continue

                    name = quantity.get("parameter_name")
                    runtime = (
                        entries.get(name, {}).get("value")
                        if isinstance(entries, dict)
                        and isinstance(name, str)
                        and isinstance(entries.get(name), dict)
                        else None
                    )
                    status = quantity.get("paper_value_status")
                    paper_value = quantity.get("value")
                    paper_attribution = re.search(
                        r"\b(?:paper|study|authors?)\b[^.\n]{0,120}"
                        r"\b(?:uses?|sets?|states?|reports?|specifies?)\b"
                        r"|\b(?:paper[- ]stated|according to the paper)\b",
                        span,
                        flags=re.IGNORECASE,
                    ) is not None
                    runtime_attribution = re.search(
                        r"\b(?:demo|runtime|system|implementation|configured)\b"
                        r"|\b(?:we|this notebook)\s+(?:use|uses|set|sets)\b",
                        span,
                        flags=re.IGNORECASE,
                    ) is not None

                    if paper_attribution and runtime_attribution:
                        valid = False
                    elif paper_attribution:
                        valid = status == "paper_stated" and all(
                            _same_number(candidate, paper_value)
                            for candidate in candidates
                        )
                    elif runtime_attribution:
                        valid = all(
                            _same_number(candidate, runtime)
                            for candidate in candidates
                        )
                    else:
                        # An unqualified number is safe only when paper and
                        # runtime are literally the same stated fact. Otherwise
                        # its provenance is ambiguous outside the canonical
                        # deterministic block.
                        valid = (
                            status == "paper_stated"
                            and _same_number(runtime, paper_value)
                            and all(
                                _same_number(candidate, paper_value)
                                for candidate in candidates
                            )
                        )
                    if not valid:
                        narrative_errors.append(
                            "notebook markdown cell "
                            f"{cell_index} contains a competing numeric claim "
                            f"for protocol role={quantity.get('role')!r}, "
                            f"paper_names={quantity.get('paper_names')!r}, "
                            f"paper_symbols={quantity.get('paper_symbols')!r}: "
                            f"{span!r}. Candidate values {candidates!r} do not "
                            "match their paper/runtime attribution; keep "
                            "numeric protocol claims in the deterministic "
                            "block."
                        )
        errors.extend(dict.fromkeys(narrative_errors))
    return errors


def _function_signature_from_method_dir(method_dir: Path, fn_name: str) -> tuple[set[str], bool] | None:
    for path in sorted(method_dir.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name != fn_name:
                continue
            args = node.args
            params = {
                arg.arg
                for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]
            }
            has_varkw = args.kwarg is not None
            return params, has_varkw
    return None


def _markdown_loader_keyword_errors(markdown_cells: list[str], method_dir: Path) -> list[str]:
    """Validate own-data loader examples against the generated loader signature."""
    errors: list[str] = []
    for loader in ("load_data", "load_problem", "load_target_data"):
        signature = _function_signature_from_method_dir(method_dir, loader)
        if signature is None:
            continue
        allowed, has_varkw = signature
        if has_varkw:
            continue
        call_re = re.compile(rf"\b{re.escape(loader)}\s*\((.*?)\)", re.DOTALL)
        for cell_idx, src in enumerate(markdown_cells):
            if "own data" not in (src or "").lower() and loader not in (src or ""):
                continue
            for match in call_re.finditer(src or ""):
                body = match.group(1)
                for kw in re.findall(r"(?<![\w.])([A-Za-z_]\w*)\s*=", body):
                    if kw not in allowed:
                        errors.append(
                            f"notebook markdown cell {cell_idx} shows `{loader}(..., "
                            f"{kw}=...)`, but method package signature accepts only "
                            f"{sorted(allowed)}. Update the own-data example to match "
                            "method/data.py."
                        )
    return errors


def _partial_stub_errors(run_dir: Path, markdown_cells: list[str],
                         code_cells: list[str]) -> list[str]:
    """§3.5 criterion 4: for every recorded stub the notebook must carry a
    markdown notice naming it AND a raising code cell naming it — no mocked
    outputs, no skipped-but-green cells."""
    from scripts.partial_delivery import load_stubbed_elements  # noqa: PLC0415

    errors: list[str] = []
    stubs, stub_error = load_stubbed_elements(run_dir / ".pipeline")
    if stub_error:
        errors.append(
            f"partial-delivery record is unreadable ({stub_error}) — a "
            f"notebook cannot be validated complete while the partial "
            f"state is unknown")
    for rec in stubs:
        eid = rec["element_id"]
        if not any("PARTIAL" in src and eid in src for src in markdown_cells):
            errors.append(
                f"stubbed component `{eid}`: no markdown cell says PARTIAL "
                f"and names it — the notebook must state the stub before "
                f"the cell that would exercise it (partial delivery §3.5)")
        # The raising cell is required only for stubs with a CODE
        # representation (stub_path set). A gate-recorded obligation the
        # package never carries as a file has no cell that could exercise
        # it — the notice (satisfied by the injected PARTIAL banner at
        # minimum) is the whole criterion-4 duty there.
        if rec.get("stub_path") and not any(
            eid in c and "NotImplementedError" in c for c in code_cells
        ):
            errors.append(
                f"stubbed component `{eid}`: no code cell raises for it — "
                f"a notebook must not run silently past a stub (partial "
                f"delivery §3.5); use the "
                f"`# %% PLACEHOLDER: component_stub:{eid}` marker")
    return errors


def _bundle_consumption_errors(code_cells: list[str], run_dir: Path) -> list[str]:
    """notebook_data_binding — every provisioned tier must flow via load_data.

    The pdfgnn 2026-08-04 delivery imported the loader but never called it.
    This check remains tier-neutral mechanically while its diagnostic preserves
    the material distinction between paper-cited public data and the supported
    family-owned synthetic fallback.
    """
    if not provenance_path(run_dir).is_file():
        return []
    for src in code_cells:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else (
                    fn.attr if isinstance(fn, ast.Attribute) else None)
                if name == "load_data":
                    return []
    manifest = read_provenance(run_dir) or {}
    disclosure = disclosure_for(manifest)
    if disclosure.is_synthetic:
        source_clause = (
            "carries the family-owned synthetic fallback "
            "(PROVENANCE.json tier family_owned_synthetic). The demo must "
            "run on those bundled fixture tables"
        )
        replacement_clause = (
            "Do not generate a second stand-in or describe this fixture as "
            "real/from-paper data"
        )
    elif disclosure.is_public:
        source_clause = (
            "carries a bundled demo-scale extract of the paper's own dataset "
            "(PROVENANCE.json present). The demo must run on the bundled real "
            "tables"
        )
        replacement_clause = (
            "Synthetic generation is acceptable only for quantities the "
            "bundle genuinely lacks, disclosed in prose, never as a "
            "replacement for the bundled data"
        )
    else:
        source_clause = (
            "carries provisioned tables with an unrecognized provenance tier. "
            "The demo must run on those bundled tables"
        )
        replacement_clause = (
            "Do not replace the provisioned tables or claim an origin that "
            "PROVENANCE.json does not establish"
        )
    return [
        "notebook never CALLS load_data() although method/example_data/ "
        f"{source_clause}: load them via load_data() in the data section and "
        "derive the method's inputs from those tables. "
        f"{replacement_clause} (notebook_data_binding)."
    ]


_SYNTHETIC_POSITIVE_EVIDENCE_PATTERNS = tuple(
    re.compile(pattern) for pattern in (
        r"bundled demo data[^\n.]{0,80}\breal\b",
        r"\b(?:this|the) (?:demo|notebook|bundle|fixture) "
        r"(?:uses|loads|runs on|contains|is) real data\b",
        r"\bpaper's own cited (?:data|dataset|source)\b",
        r"\bdataset (?:the|this) paper cites\b",
        r"\bpaper-cited public data\b",
        r"\b(?:data|dataset|bundle|fixture) (?:was |is )?"
        r"(?:fetched|taken|drawn|obtained) from (?:the )?paper\b",
        r"\b(?:data|dataset|bundle|fixture|tables?) "
        r"(?:(?:comes?|originates?) from|(?:is|are) sourced from) "
        r"(?:the )?paper\b",
        r"\b(?:this|the) (?:data|dataset|bundle|fixture|tables?) "
        r"(?:is|are|uses|contains) (?:actual |real )?(?:the )?"
        r"paper(?:'s)? (?:data|dataset)\b",
        r"\bthese (?:data|tables?) (?:are )?(?:actual |real )?"
        r"paper data\b",
        r"\bactual paper data\b",
        r"\breal paper data\b",
        r"\bmissing counts/dtypes are real\b",
        r"\b(?:these|the|demo|reported|its) "
        r"(?:results?|numbers?|metrics?|outputs?) "
        r"(?:are|provide|constitute|represent|serve as) "
        r"(?:(?:strong|valid|legitimate|actual|credible|meaningful) )?"
        r"(?:paper[- ]comparable|(?:the )?paper(?:'s)? benchmark(?: results?)?"
        r"|benchmark evidence|benchmark results?)\b",
        r"\b(?:these|the|demo|reported|its) "
        r"(?:results?|numbers?|metrics?|outputs?) "
        r"(?:match(?:es|ed)?|agree(?:s|d)? with|reproduce(?:s|d)?|"
        r"replicate(?:s|d)?) (?:the )?paper(?:'s)? "
        r"(?:benchmark|results?|numbers?|metrics?)\b",
        r"\b(?:are|provide|constitute|represent|serve as) "
        r"(?!(?:not|never|no)\b)"
        r"(?:(?:strong|valid|legitimate|actual|credible|meaningful) )?"
        r"benchmark evidence\b",
        r"\bbenchmark evidence[^\n.,;:]{0,40}\b(?:is|looks?|remains?) "
        r"(?:strong|valid|legitimate|actual|credible|meaningful)\b",
        r"\b(?:these|the|demo|reported|its) "
        r"(?:results?|numbers?|metrics?|outputs?) "
        r"(?:confirm(?:s|ed)?|support(?:s|ed)?|corroborate(?:s|d)?) "
        r"(?:the )?paper(?:'s)? (?:benchmark|results?|numbers?|metrics?)\b",
    )
)


def _claim_is_negated(text: str, start: int) -> bool:
    context = text[max(0, start - 96):start]
    # Only a nearby negation in the same clause can govern the matched claim.
    # An earlier disclosure such as "not paper-comparable, but these results
    # are benchmark evidence" must not launder the later positive assertion.
    context = re.split(
        r"[.!?;:,\n]|\b(?:and|but|yet|however|although)\b", context,
    )[-1]
    return bool(
        re.search(
            r"\b(?:not|never|no|isn't|wasn't|aren't|weren't)\b\s*"
            r"(?:(?:a|an|the)\s+)?"
            r"(?:(?:actually|genuinely|really)\s+)?$",
            context,
        )
        or re.search(
            r"\b(?:cannot|can't)\s+(?:be\s+)?"
            r"(?:(?:considered|called)\s+|treated\s+as\s+)?$",
            context,
        )
    )


def _synthetic_evidence_limit_is_stated(prose: str) -> bool:
    """Require a result-scoped limit inside one prose clause."""

    result_noun = r"(?:results?|numbers?|metrics?|outputs?)"
    limit = r"(?:paper[- ]comparable|benchmark evidence|benchmark results?)"
    negation = r"(?:no|not|never|cannot|can't|isn't|aren't|wasn't|weren't)"
    clauses = re.split(
        r"[.!?;:,\n]|\b(?:but|however|yet|although|while)\b", prose,
    )
    for clause in clauses:
        if re.search(
            rf"\b{result_noun}\b[^\n]{{0,100}}\bmechanics only\b"
            rf"|\bmechanics only\b[^\n]{{0,100}}\b{result_noun}\b",
            clause,
        ):
            return True
        if re.search(
            rf"\b{result_noun}\b[^\n]{{0,100}}\b{negation}\b"
            rf"[^\n]{{0,60}}\b{limit}\b",
            clause,
        ):
            return True
        if re.search(
            rf"\b{negation}\b[^\n]{{0,60}}\b{limit}\b"
            rf"[^\n]{{0,100}}\b{result_noun}\b",
            clause,
        ):
            return True
    return False


def _bundle_provenance_honesty_errors(
    markdown_cells: list[str], run_dir: Path
) -> list[str]:
    """Synthetic notebooks must name their tier and its evidence limit."""

    path = provenance_path(run_dir)
    if not path.is_file():
        return []
    manifest = read_provenance(run_dir)
    if manifest is None:
        return [
            "method/example_data/PROVENANCE.json is unreadable; notebook data "
            "origin cannot be disclosed honestly (demo_data_provenance)."
        ]
    disclosure = disclosure_for(manifest)
    if not disclosure.is_synthetic:
        return []

    prose = "\n".join(markdown_cells).lower()
    prose = prose.replace("’", "'").replace("‑", "-").replace("–", "-").replace("—", "-")
    errors: list[str] = []
    positive_claims = [
        match.group(0)
        for pattern in _SYNTHETIC_POSITIVE_EVIDENCE_PATTERNS
        for match in pattern.finditer(prose)
        if not _claim_is_negated(prose, match.start())
    ]
    if positive_claims:
        errors.append(
            "notebook describes a family-owned synthetic fallback as real or "
            "from-paper data, or upgrades its outputs to paper-comparable or "
            "benchmark evidence. Remove the positive evidence claim and "
            "disclose the synthetic tier instead (demo_data_provenance)."
        )

    tier_matches = re.finditer(
        r"\bfamily[- ]owned synthetic (?:offline )?fallback\b", prose,
    )
    names_tier = any(
        not _claim_is_negated(prose, match.start()) for match in tier_matches
    )
    states_limit = _synthetic_evidence_limit_is_stated(prose)
    if not names_tier or not states_limit:
        errors.append(
            "notebook must state that its data are a family-owned synthetic "
            "fallback and that resulting numbers are not paper-comparable or "
            "benchmark evidence (demo_data_provenance)."
        )
    return errors


def _stitched_code(code_cells: list[str]) -> ast.Module | None:
    """The notebook's code cells as one module, magics stripped — the
    executed notebook IS this module, so cross-cell dataflow checks parse
    it whole."""
    source = "\n".join(
        "\n".join(line for line in src.splitlines()
                  if not line.lstrip().startswith(("%", "!")))
        for src in code_cells)
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _split_leakage_errors(code_cells: list[str], run_dir: Path) -> list[str]:
    """eval_split_range_overlap (R2C-066, the range arm): the notebook's
    training and inference-conditioning reads must not provably overlap
    the windows its metrics are computed on. The night3 pdfgnn notebook
    trained on the full history, conditioned forecast() on the full
    history, and scored against demand_history[:, T-K:T] — three
    reviewers independently called the resulting metrics meaningless."""
    tree = _stitched_code(code_cells)
    if tree is None:
        return []
    from scripts.eval_split_ranges import find_notebook_split_leakage

    params: dict = {}
    params_path = run_dir / ".pipeline" / "params.json"
    if params_path.is_file():
        try:
            loaded = json.loads(params_path.read_text(encoding="utf-8"))
            values = loaded.get("params", loaded)
            if isinstance(values, dict):
                params = {k: (v.get("value") if isinstance(v, dict) else v)
                          for k, v in values.items()}
        except (OSError, ValueError):
            params = {}
    return [f"notebook evaluation leaks fitted data: {f.message()}"
            for f in find_notebook_split_leakage(tree, params)]


def _reachable_split_lineage_errors(
    code_cells: list[str], spec: dict, run_dir: Path, repo_root: Path,
) -> list[str]:
    """R2C-077 interprocedural range enforcement at the caller seam.

    Stage 3a is the first producer seam where the notebook call, trainer body,
    resolved demo parameters, and measured bundle extent coexist.  Keep the
    older notebook-only range arm above as an independent control.
    """
    tree = _stitched_code(code_cells)
    if tree is None:
        return []
    from scripts.build_plan import load_build_plan  # noqa: PLC0415
    from scripts.eval_split_validation import (  # noqa: PLC0415
        notebook_split_lineage_errors,
    )
    from scripts.taxonomy import run_overlay_dir  # noqa: PLC0415

    build_plan = load_build_plan(
        spec, repo_root, provisional_packs_dir=run_overlay_dir(run_dir)
    )
    return notebook_split_lineage_errors(spec, build_plan, run_dir, tree)


def _training_history_notebook_flow_errors(
    code_cells: list[str], spec: dict, run_dir: Path, repo_root: Path,
) -> list[str]:
    """Early producer gate for R2C-090's bounded notebook value flow.

    Unsupported interprocedural grammar is intentionally deferred to the
    typed post-smoke evidence adapter, where it is recorded as pipeline-owned
    without consuming a producer retry.  This static validator reports only
    supported direct disagreements.
    """
    from scripts.build_plan import load_build_plan  # noqa: PLC0415
    from scripts.taxonomy import run_overlay_dir  # noqa: PLC0415
    from scripts.time_series_training_history import (  # noqa: PLC0415
        TrainingHistoryCoverageError,
        TrainingHistoryProducerError,
    )
    from scripts.training_history_notebook_flow import (  # noqa: PLC0415
        training_history_notebook_applicable,
        validate_training_history_architecture_contract,
        validate_training_history_notebook_flow,
    )

    try:
        build_plan = load_build_plan(
            spec, repo_root, provisional_packs_dir=run_overlay_dir(run_dir)
        )
    except Exception:  # another validator owns unresolved plan construction
        return []
    if not training_history_notebook_applicable(run_dir, build_plan):
        return []
    try:
        validate_training_history_architecture_contract(run_dir, build_plan)
        validate_training_history_notebook_flow(code_cells, build_plan)
    except TrainingHistoryProducerError as exc:
        return [
            "notebook structured training-history value flow disagrees with "
            f"the schema-2 execution contract ({exc.code}): {exc}"
        ]
    except TrainingHistoryCoverageError:
        return []
    return []


def _graph_constructor_notebook_flow_errors(
    code_cells: list[str], spec: dict, run_dir: Path
) -> list[str]:
    """Reject supported constructor routes that bypass the declared helper.

    The live-use analyser has a deliberately closed grammar.  Coverage gaps
    are deferred to the frozen HG receipt, where they become unprobeable;
    only a disagreement inside supported notebook grammar consumes a producer
    retry here.
    """

    from scripts.graph_callable_liveness import (  # noqa: PLC0415
        GraphCallableCoverageError,
        GraphCallableProducerError,
        graph_callable_identities,
        prove_constructor_notebook_flow,
    )

    try:
        if graph_callable_identities(spec) is None:
            return []
        contract = json.loads(
            (run_dir / ".pipeline" / "arch_contract.json").read_text(
                encoding="utf-8"
            )
        )
        prove_constructor_notebook_flow(
            code_cells, spec, contract, run_dir / "method"
        )
    except GraphCallableProducerError as exc:
        return [
            "notebook graph-constructor value flow disagrees with the exact "
            f"homogeneous-graph callable contract ({exc.code}): {exc}"
        ]
    except GraphCallableCoverageError:
        return []
    except (OSError, json.JSONDecodeError):
        # The existing architecture/notebook setup gates own missing or invalid
        # artifacts.  Do not relabel those failures as graph producer defects.
        return []
    return []


def _int_context_unsafe(node: ast.AST) -> list[str]:
    """Sub-expressions that make an int context fail at runtime: a bare
    `.item()` (float for any float tensor) or true division, unless an
    `int(...)` wrapper sanitizes the subtree."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id == "int":
        return []
    hits: list[str] = []
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
            and node.func.attr == "item":
        hits.append("`.item()` (float for a float tensor)")
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        hits.append("true division (always float)")
    for child in ast.iter_child_nodes(node):
        hits.extend(_int_context_unsafe(child))
    return hits


def _int_context_errors(code_cells: list[str]) -> list[str]:
    """torch/numpy boundary lint, the certain-fix arm: `range()` over an
    expression that is float by construction dies with \"'float' object
    cannot be interpreted as an integer\" — the night3 cell-30 smoke
    failure (`range(degrees.max().item() + 2)`), one full smoke iteration
    spent on a defect visible in the source. Fix is always `int(...)`.
    Noise floor measured at landing: 0 of 67 range() calls across every
    delivered notebook."""
    errors: list[str] = []
    for index, src in enumerate(code_cells):
        module = _stitched_code([src])
        if module is None:
            continue
        for node in ast.walk(module):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == "range":
                for arg in node.args:
                    unsafe = _int_context_unsafe(arg)
                    if unsafe:
                        errors.append(
                            f"code cell {index}: `{ast.unparse(node)}` — "
                            f"range() needs ints but the argument contains "
                            f"{'; '.join(unsafe)}. Wrap the expression in "
                            f"int(...) (int_context_float)."
                        )
    return errors


# Calendar-position attributes: grouping a multi-year time axis by one of
# these ALONE folds the years onto each other and destroys chronology.
_CALENDAR_POSITION_ATTRS = frozenset(
    {"week", "weekofyear", "month", "quarter", "dayofweek", "day_of_week",
     "dayofyear"})


def _bundle_spans_multiple_years(run_dir: Path) -> bool:
    provenance = run_dir / "method" / "example_data" / "PROVENANCE.json"
    if not provenance.is_file():
        return False
    try:
        manifest = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    for entry in manifest.get("files") or []:
        axis = entry.get("time_axis")
        if not isinstance(axis, dict):
            continue
        first = str(axis.get("first_step") or "")[:4]
        last = str(axis.get("last_step") or "")[:4]
        if first.isdigit() and last.isdigit() and first != last:
            return True
    return False


def _calendar_position_names(tree: ast.Module) -> dict[str, str]:
    """Names/columns assigned from a calendar-position attribute, mapped
    to the attribute they carry (`df[\"week\"] = ....isocalendar().week`
    → {\"week\": \"week\"})."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        attr = None
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Attribute) \
                    and sub.attr in _CALENDAR_POSITION_ATTRS:
                attr = sub.attr
                break
        if attr is None:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            out[target.id] = attr
        elif isinstance(target, ast.Subscript) \
                and isinstance(target.slice, ast.Constant) \
                and isinstance(target.slice.value, str):
            out[target.slice.value] = attr
    return out


def _groupby_key_names(call: ast.Call) -> list[str]:
    names: list[str] = []
    for arg in list(call.args) + [kw.value for kw in call.keywords]:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            names.append(arg.value)
        elif isinstance(arg, (ast.List, ast.Tuple)):
            names.extend(e.value for e in arg.elts
                         if isinstance(e, ast.Constant)
                         and isinstance(e.value, str))
    return names


def _chronology_fold_errors(code_cells: list[str], run_dir: Path) -> list[str]:
    """calendar_fold_aggregation: on a bundle whose time axis spans more
    than one year, aggregating by a calendar POSITION (week-of-year,
    month, ...) without the year folds the years onto one calendar and
    the \"time series\" stops being time. The night3 notebook summed
    January 2017 + 2018 + 2019 into \"week 1\" and every downstream
    number ran on that artifact. Aggregate chronologically: group by
    (year, position) or resample on the datetime itself."""
    if not _bundle_spans_multiple_years(run_dir):
        return []
    tree = _stitched_code(code_cells)
    if tree is None:
        return []
    position_names = _calendar_position_names(tree)
    if not position_names:
        return []
    errors: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "groupby"):
            continue
        keys = _groupby_key_names(node)
        folded = [k for k in keys if k in position_names]
        if folded and not any("year" in k.lower() for k in keys):
            errors.append(
                f"notebook aggregates a MULTI-YEAR time axis by calendar "
                f"position alone: groupby({keys}) where "
                f"{', '.join(f'`{k}`' for k in folded)} carries "
                f"{', '.join(sorted({position_names[k] for k in folded}))} "
                f"— the bundled data spans multiple years, so this folds "
                f"the years onto one calendar and destroys chronology. "
                f"Group by (year, position) or resample on the datetime "
                f"column (calendar_fold_aggregation)."
            )
    return errors


# Import-name to requirement-name aliases for the dependency check.
_IMPORT_TO_REQUIREMENT = {
    "sklearn": "scikit-learn", "cv2": "opencv-python", "PIL": "pillow",
    "yaml": "pyyaml", "skimage": "scikit-image", "bs4": "beautifulsoup4",
}


def _undeclared_dependency_errors(code_cells: list[str],
                                  run_dir: Path) -> list[str]:
    """undeclared_notebook_dependency: every top-level module the notebook
    imports must be installable from requirements.txt (or be stdlib, or
    the delivered package). The night3 notebook imported pandas in its
    data cell while requirements.txt listed numpy/torch/matplotlib/jupyter
    — the advertised Run All dies with ImportError on a clean machine."""
    requirements_path = run_dir / "requirements.txt"
    if not requirements_path.is_file():
        return []
    try:
        req_lines = requirements_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    declared = set()
    for line in req_lines:
        name = re.split(r"[<>=!~\[;\s]", line.strip(), maxsplit=1)[0].lower()
        if name and not name.startswith("#"):
            declared.add(name.replace("_", "-"))
    stdlib = getattr(sys, "stdlib_module_names", frozenset())
    imported: dict[str, int] = {}
    for index, src in enumerate(code_cells):
        module = _stitched_code([src])
        if module is None:
            continue
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.setdefault(alias.name.split(".")[0], index)
            elif isinstance(node, ast.ImportFrom) and node.module \
                    and node.level == 0:
                imported.setdefault(node.module.split(".")[0], index)
    errors: list[str] = []
    for module_name, index in sorted(imported.items()):
        if module_name in stdlib or module_name == "method":
            continue
        requirement = _IMPORT_TO_REQUIREMENT.get(
            module_name, module_name).lower().replace("_", "-")
        if requirement not in declared:
            errors.append(
                f"code cell {index} imports `{module_name}` but "
                f"requirements.txt does not declare `{requirement}` — the "
                f"README's Run All dies with ImportError on a clean "
                f"environment. Add it to requirements.txt or drop the "
                f"import (undeclared_notebook_dependency)."
            )
    return errors


def validate(spec: dict, run_dir: Path, repo_root: Path) -> list[str]:
    errors: list[str] = []

    # 1. notebook.ipynb exists + parses
    nb_path = run_dir / "notebook.ipynb"
    if not nb_path.is_file():
        return [f"notebook.ipynb missing at {nb_path}"]
    try:
        nb = nbformat.read(nb_path, as_version=4)
    except Exception as e:  # nbformat raises various exception types
        return [f"notebook.ipynb fails to parse as a v4 notebook: {e}"]

    cells = nb.cells

    # 2. notebook_layout — section-by-section check from the taxonomy node.
    comparison_paradigm = (spec.get("comparison") or {}).get("classification") or {}
    paradigm_id = comparison_paradigm.get("id") or ""
    # Gap-path serves() overlay: a gap paper's provisional classification is
    # served by the run's own installed pack (keyed on run_dir); the layout
    # then resolves through the pack's extends chain. Non-gap runs have no
    # pack dir and resolve the committed view byte-identically.
    tax = taxonomy.load_taxonomy(
        provisional_packs_dir=taxonomy.run_overlay_dir(run_dir))
    node = taxonomy.serves(paradigm_id, tax) if paradigm_id else None
    if node is None:
        return [f"no taxonomy notebook_layout for comparison.classification.id={paradigm_id!r}"]
    notebook_layout = taxonomy.load_notebook_layout(paradigm_id=paradigm_id, taxonomy=tax)
    layout_source = f"the matched taxonomy node ({paradigm_id})"

    sections = notebook_layout.get("sections") or []
    if not sections:
        # No section contract declared anywhere on this node's extends chain.
        # Only some families carry one (AL and MP do; KD does not yet) — that
        # is a taxonomy-coverage fact, not a notebook defect, so skip the
        # section-structure check and KEEP RUNNING the universal checks below.
        # The old early return halted bev-distill at stage 3.c (2026-07-01)
        # and also silently skipped every check after it (placeholders,
        # idempotency gate, runtime-value tables).
        print(f"note: {layout_source} declares no `notebook_layout.sections`; "
              f"section-structure check skipped, universal checks still run",
              file=sys.stderr)

    # The `title` section is special — its title is method-specific (paper title), so we
    # can't string-match its heading. Just check that there's a top-level # cell at index 0.
    if sections and sections[0].get("id") == "title":
        if not (cells and cells[0].cell_type == "markdown" and cells[0].source.lstrip().startswith("# ")):
            errors.append("notebook missing title cell at index 0 (markdown starting with `# `)")

    for section in sections[1:]:
        sid = section.get("id")
        title = (section.get("title") or "").strip()
        if not title:
            continue
        if not _find_cell_with_heading(cells, title):
            errors.append(
                f"notebook missing section `{sid}` with heading {title!r}. "
                f"Taxonomy notebook_layout declares this section."
            )

    # 3. No PLACEHOLDER markers remain
    for i, c in enumerate(cells):
        if PLACEHOLDER_PATTERN.search(c.source or ""):
            errors.append(
                f"notebook cell {i} still contains a PLACEHOLDER marker: "
                f"{c.source.splitlines()[0]!r}. The render script should have substituted it."
            )

    code_cells = _all_code_cells(cells)
    code_text_blob = "\n".join(code_cells)
    markdown_cells = [c.source for c in cells if c.cell_type == "markdown"]

    # Partial delivery (§3.5 criterion 4): the notebook can never run
    # silently past a stubbed component.
    errors.extend(_partial_stub_errors(run_dir, markdown_cells, code_cells))

    # Non-idempotent cell reassignments (queue 11h): break re-execution and the
    # AL-1 loop probe. Paradigm-agnostic — applies to every generated notebook.
    errors.extend(_non_idempotent_cell_errors(code_cells))

    # 4. §1 setup includes %matplotlib inline
    if "%matplotlib inline" not in code_text_blob:
        errors.append("notebook §1 setup is missing `%matplotlib inline` — plots will not render inline")

    # 5. §0 install uses %pip not !pip
    if "!pip install" in code_text_blob:
        errors.append(
            "notebook contains `!pip install` — should use `%pip install` instead. The bang form "
            "uses a subshell whose `pip` may not be on PATH (common on macOS); the percent magic "
            "uses the kernel's Python."
        )
    if "%pip install" not in code_text_blob:
        errors.append("notebook §0 install is missing `%pip install` cell")

    # 6. params dict — exists, contains every param from params.json, values match
    params_path = run_dir / ".pipeline" / "params.json"
    if not params_path.is_file():
        errors.append(f"params.json not found at {params_path}")
    else:
        try:
            params_data = json.loads(params_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            errors.append(f"params.json invalid JSON: {e}")
            return errors
        params_json = params_data.get("params", {})

        params_in_nb = _params_dict_in_code(code_cells)
        if params_in_nb is None:
            errors.append(
                "notebook §2 is missing the `params = {...}` cell, or the dict couldn't be "
                "AST-parsed (placeholder substitution failed?)."
            )
        else:
            # Names match
            json_names = set(params_json.keys())
            nb_names = set(params_in_nb.keys())
            missing_in_nb = json_names - nb_names
            extra_in_nb = nb_names - json_names
            if missing_in_nb:
                errors.append(
                    f"notebook params dict missing parameters from params.json: {sorted(missing_in_nb)}"
                )
            if extra_in_nb:
                errors.append(
                    f"notebook params dict has extra parameters not in params.json: {sorted(extra_in_nb)}"
                )
            # Values match for shared names
            for name in json_names & nb_names:
                json_value = params_json[name].get("value")
                nb_value = params_in_nb[name].get("value") if isinstance(params_in_nb[name], dict) else None
                if json_value != nb_value:
                    errors.append(
                        f"params.{name}.value mismatch: params.json={json_value!r}, "
                        f"notebook={nb_value!r}"
                    )

            errors.extend(_markdown_runtime_param_table_errors(markdown_cells, params_json))

        errors.extend(
            _evaluation_protocol_render_errors(
                spec, params_data, markdown_cells
            )
        )

        # 7. used_in_notebook coverage
        cfg_keys = _extract_cfg_keys(code_cells)
        for name, entry in params_json.items():
            used = entry.get("used_in_notebook", True)
            if used and name not in cfg_keys:
                # Some params may be passed through to a helper that receives
                # **kwargs (e.g. a training/build helper, or a planner's risk
                # weights), so a missing literal `cfg[name]` reference isn't
                # necessarily a bug. Don't error; the reviewer agent can flag
                # it if it matters.
                pass
            if not used and name in cfg_keys:
                errors.append(
                    f"notebook references `cfg[{name!r}]` but params.json marks it "
                    f"used_in_notebook=False. The notebook should not use this param at runtime — "
                    f"see params.json's `unused_reason` for why."
                )
        legacy_output_keys = sorted(cfg_keys & {"batch_output", "batch_outputs"})
        if paradigm_id.startswith("active_learning") and legacy_output_keys:
            errors.append(
                f"notebook references legacy output-count config key(s) "
                f"{legacy_output_keys}. "
                "Active-learning output count must be the single `batch_size` param; "
                "`batch_returns` may remain the larger candidate prefilter."
            )

    # 8. The notebook must invoke the paper's pluggable component somewhere.
    # (Paradigm-neutral: the section that calls it is the AL acquisition loop
    # for active_learning, the planning/run section for motion_planning, the
    # distillation-loss demo for KD — the common invariant is that the
    # notebook actually exercises the contribution, not just imports it.)
    pluggable_name = (spec.get("comparison") or {}).get("pluggable_component", {}).get("name")
    if pluggable_name and f"{pluggable_name}(" not in code_text_blob:
        errors.append(
            f"notebook does not call `{pluggable_name}(...)` in any code cell — the "
            f"notebook must invoke the paper's pluggable component (the section that "
            f"runs the method end-to-end)."
        )

    # 8b. Active-learning retrain-from-scratch (M-003). The §5.1 acquisition loop
    # must build a FRESH model each round; reusing one model object across rounds
    # is warm-starting, which retrain-from-scratch AL (BADGE included) forbids.
    # This is a *deterministic* gate on purpose: the bug is statically detectable
    # and is silent at runtime (no crash, learning curve still rises), and LLM
    # reviewers have repeatedly waved it through (the stage-2b and stage-3a
    # semantic checks both missed it) — so it can't be left to the reviewer.
    # `build_model`/`train_from_scratch` are the AL package-manifest training-
    # function names (a paradigm convention, not a paper-specific value). Gated on
    # the paper actually retraining from scratch — a deliberately warm-starting
    # paper (protocol mentions "warm") is the rare exception and is skipped.
    paradigm_id = ((spec.get("comparison") or {}).get("classification") or {}).get("id", "")
    protocol = ((spec.get("critical_requirements") or {}).get("training") or {}).get("protocol", "") or ""
    warm_start_intended = "warm" in protocol.lower()
    if paradigm_id.startswith("active_learning") and not warm_start_intended:
        expected_sig = (
            (spec.get("comparison") or {})
            .get("pluggable_component", {})
            .get("signature", "")
        )
        if pluggable_name:
            errors.extend(
                _active_learning_pluggable_batch_size_errors(
                    code_cells,
                    pluggable_name=pluggable_name,
                    expected_sig=expected_sig,
                )
            )
        if _al_loop_warm_starts(code_cells, train_fn="train_from_scratch", build_fn="build_model") is True:
            errors.append(
                "notebook §5.1 acquisition loop calls `train_from_scratch(...)` inside the "
                "round loop but never calls `build_model(...)` there — it reuses one model "
                "object across rounds, i.e. WARM-STARTS. Active learning retrains from scratch "
                "each round (the paper's training protocol; warm-starting changes the "
                "algorithmic regime). Build a fresh model at the top of every round."
            )
        if pluggable_name and _al_loop_skips_last_acquisition(code_cells, pluggable_name=pluggable_name):
            errors.append(
                "notebook §5.1 acquisition loop iterates over `num_rounds` but calls the "
                f"acquisition function `{pluggable_name}(...)` only under a guard equivalent "
                "to `round < num_rounds - 1`. That performs one fewer acquisition batch than "
                "the notebook's displayed budget implies. Either acquire once per stated "
                "round, or split the notebook into explicit `num_acquisition_rounds` and "
                "`num_eval_points` parameters so the prose, plots, and labeled-budget math "
                "match the code."
            )
        if pluggable_name and _al_loop_has_unused_final_acquisition(code_cells, pluggable_name=pluggable_name):
            errors.append(
                "notebook §5.1 acquisition loop appends learning-curve evaluation before "
                f"calling `{pluggable_name}(...)` and has no post-acquisition evaluation "
                "inside the `num_rounds` loop. That acquires a final batch that is never "
                "retrained/evaluated/plotted. Treat `num_rounds` as acquisition rounds: "
                "evaluate the initial labeled set once, then after every acquisition append "
                "the post-acquisition evaluation."
            )
        if pluggable_name and _al_loop_has_postmerge_count_for_premerge_eval(
            code_cells,
            pluggable_name=pluggable_name,
        ):
            errors.append(
                "notebook §5.1 acquisition loop updates the labeled/unlabeled sets "
                "and then appends a learning-curve point before retraining on the "
                "updated labeled set. That labels a pre-acquisition model evaluation "
                "with the post-acquisition label count. Evaluate the initial set once, "
                "then after every acquisition retrain/evaluate the merged labeled set "
                "before appending that budget point."
            )

    inverse_distance_cells = _inverse_distance_markdown_contradictions(markdown_cells)
    for idx in inverse_distance_cells:
        errors.append(
            f"notebook markdown cell {idx} describes an inverse-distance score while also "
            "claiming farther/more distant samples receive higher score or preference. "
            "For scores like `R_0 / distance` ranked descending, lower distance receives "
            "the higher score; fix the prose or the ranking formula."
        )

    errors.extend(_markdown_loader_keyword_errors(markdown_cells, run_dir / "method"))

    # 9. Cross-stage integration: notebook imports must resolve against the method package
    # — every `from method import X` symbol must be in __init__.py's __all__; every
    # `from method.<sub> import X` must reference an actual top-level name in method/<sub>.py.
    # Catches signature drift between the package and the notebook before the smoke gate runs.
    method_dir = run_dir / "method"
    imports = _extract_method_imports(code_cells)
    all_set = _read_package_all(method_dir)

    for module_path, symbols in imports.items():
        if module_path == "method":
            if all_set is None:
                errors.append(
                    f"notebook imports `from method import {sorted(symbols)}` but "
                    f"{method_dir / '__init__.py'} is missing or has no parseable `__all__`"
                )
                continue
            missing = sorted(symbols - all_set)
            if missing:
                errors.append(
                    f"notebook imports `from method import {missing}` but those names "
                    f"are not in `method/__init__.py`'s `__all__` ({sorted(all_set)}). "
                    f"Either add them to __all__ or import from a submodule."
                )
        else:
            # e.g., "method.method" -> method/method.py
            sub = module_path.split(".", 1)[1]
            sub_path = method_dir / f"{sub}.py"
            top_names = _read_module_top_level_names(sub_path)
            if top_names is None:
                errors.append(
                    f"notebook imports `from {module_path} import {sorted(symbols)}` but "
                    f"{sub_path} is missing or fails to parse"
                )
                continue
            missing = sorted(symbols - top_names)
            if missing:
                errors.append(
                    f"notebook imports `from {module_path} import {missing}` but those names "
                    f"are not defined at top-level in {sub_path}. Top-level public names: {sorted(top_names)}"
                )

    errors.extend(_bundle_consumption_errors(code_cells, run_dir))
    errors.extend(_bundle_provenance_honesty_errors(markdown_cells, run_dir))
    errors.extend(_split_leakage_errors(code_cells, run_dir))
    errors.extend(_training_history_notebook_flow_errors(
        code_cells, spec, run_dir, repo_root
    ))
    errors.extend(_graph_constructor_notebook_flow_errors(
        code_cells, spec, run_dir
    ))
    errors.extend(_reachable_split_lineage_errors(
        code_cells, spec, run_dir, repo_root
    ))
    errors.extend(_int_context_errors(code_cells))
    errors.extend(_chronology_fold_errors(code_cells, run_dir))
    errors.extend(_undeclared_dependency_errors(code_cells, run_dir))

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()

    if not args.spec.is_file():
        print(f"error: spec not found: {args.spec}", file=sys.stderr)
        return 2

    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"error: spec is not valid JSON ({e})", file=sys.stderr)
        return 2

    errors = validate(spec, args.run_dir, args.repo_root)
    if errors:
        print(f"FAIL: {len(errors)} validation error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"ok: notebook at {args.run_dir / 'notebook.ipynb'} validates against the layout + spec.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
