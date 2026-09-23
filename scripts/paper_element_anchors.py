"""Where the generated code says which part of the paper it implements.

Generated modules carry `# paper-element: <id>` comment anchors naming paper-map
elements. Three consumers read them and each used to reach a different module
for the same scan:

- the coder gates, which validate every anchor against the paper map's closed
  id set;
- the anchor join table, which lifts anchors into a run-level artifact so a
  surface can ask "which function implements element X";
- and now the probe battery, which asks the reverse question: given the
  callables a behavioral check actually exercised, which paper-map elements do
  they implement (R2C-072). That answer is the last hop in the chain that lets
  a probe finding reach the methodology contract obligation it bears on.

The scan lives here because the probe battery is VENDORED into the portable
harness a researcher runs outside this repo. Before this module, the scan sat
inside the architecture-coder validator, which the harness cannot import, so
stamping findings there would have made the harness adjudicate differently from
the delivered report. This module is stdlib-only and dependency-free, so it
vendors cleanly. `validate_architecture_coder_output` and
`build_anchor_join_table` import from here rather than keeping their own copies.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

# Slug grammar: permissive enough for legitimate slugs such as `alg-kmeans++`
# without swallowing trailing prose after whitespace.
PAPER_ELEMENT_PATTERN = re.compile(
    r"#\s*paper-element:\s*([A-Za-z][A-Za-z0-9_+\-]*)")

# Anchor placement relative to its enclosing scope, in primary-site preference
# order (lower ranks first). Consumed by the join table's primary selection.
PLACEMENT_RANK = {"leading": 0, "body": 1, "class_body": 2, "module": 2}


def extract_paper_element_ids(file_path: Path) -> list[tuple[int, str]]:
    """Return [(line_number, id), ...] for every `# paper-element: <id>`
    annotation in one file. A missing file yields nothing rather than raising:
    every caller treats an unscannable file as carrying no anchors."""
    file_path = Path(file_path)
    if not file_path.is_file():
        return []
    out: list[tuple[int, str]] = []
    for lineno, line in enumerate(
            file_path.read_text(encoding="utf-8").splitlines(), start=1):
        m = PAPER_ELEMENT_PATTERN.search(line)
        if m:
            out.append((lineno, m.group(1)))
    return out


class Scope:
    """One enclosing scope: a function/method (kind "function") or a class
    body (kind "class"), with its dotted qualname and line span."""

    __slots__ = ("qualname", "kind", "start", "end", "first_body_line", "def_line")

    def __init__(self, qualname: str, kind: str, start: int, end: int,
                 first_body_line: int, def_line: int) -> None:
        self.qualname = qualname
        self.kind = kind
        self.start = start
        self.end = end
        # First line of the first non-docstring statement. Anchor lines
        # BEFORE it are "leading" (docstring-embedded anchors included —
        # real deliveries carry them there, ms3d 2026-07-22).
        self.first_body_line = first_body_line
        self.def_line = def_line

    def contains(self, line: int) -> bool:
        return self.start <= line <= self.end


def first_non_docstring_line(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> int:
    """Line of the first statement that isn't the docstring. A body that is
    ONLY a docstring makes every interior anchor leading."""
    body = node.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if not body:
        return (node.end_lineno or node.lineno) + 1
    return body[0].lineno


def collect_scopes(tree: ast.Module) -> list[Scope]:
    """Every function/method and class scope in the module, with dotted
    qualnames (`Class.method`, `outer.inner`)."""
    scopes: list[Scope] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = f"{prefix}{child.name}"
                scopes.append(Scope(
                    qualname, "function", child.lineno,
                    child.end_lineno or child.lineno,
                    first_non_docstring_line(child), child.lineno,
                ))
                visit(child, f"{qualname}.")
            elif isinstance(child, ast.ClassDef):
                qualname = f"{prefix}{child.name}"
                scopes.append(Scope(
                    qualname, "class", child.lineno,
                    child.end_lineno or child.lineno,
                    child.lineno, child.lineno,
                ))
                visit(child, f"{qualname}.")
            else:
                visit(child, prefix)

    visit(tree, "")
    return scopes


def attribute_site(line: int, scopes: list[Scope]) -> dict:
    """The (qualname, function_line, placement) attribution for one anchor
    line. Innermost enclosing FUNCTION wins; a class body without an
    enclosing function attributes to the class; otherwise module level."""
    functions = [s for s in scopes if s.kind == "function" and s.contains(line)]
    if functions:
        innermost = max(functions, key=lambda s: s.start)
        placement = "leading" if line < innermost.first_body_line else "body"
        return {
            "qualname": innermost.qualname,
            "function_line": innermost.def_line,
            "placement": placement,
        }
    classes = [s for s in scopes if s.kind == "class" and s.contains(line)]
    if classes:
        innermost = max(classes, key=lambda s: s.start)
        return {
            "qualname": innermost.qualname,
            "function_line": innermost.def_line,
            "placement": "class_body",
        }
    return {"qualname": None, "function_line": None, "placement": "module"}


def element_ids_by_callable(method_dir: Path) -> dict[str, tuple[str, ...]]:
    """`{callable qualname: (paper-map element ids anchored inside it, ...)}`.

    The reverse of the anchor join table's question, and the last hop of the
    probe-to-contract binding (R2C-072): a behavioral check declares the
    callables it exercised, and this says which parts of the paper those
    callables implement.

    Deliberately NOT validated against the paper map here. The coder gates and
    the join table already refuse a run whose anchors name unknown ids, and the
    contract join treats an id it cannot resolve as unmatched anyway, so
    re-checking here would only add a way for a probe to crash on a run that
    already failed its own gate.

    Keys are BARE qualnames (`select_batch`, `Selector.score`), unqualified by
    file, because a probe knows the callable it invoked and not which module
    the package put it in. A qualname defined in two files therefore carries the
    union of both files' anchors, which is the honest reading: the probe cannot
    tell them apart either.

    Module-level anchors are excluded. They belong to the file rather than to
    any callable, so attributing them to whatever a probe exercised would bind
    findings to elements the checked code never touches.
    """
    method_dir = Path(method_dir)
    if not method_dir.is_dir():
        return {}
    out: dict[str, set] = {}
    for py in sorted(method_dir.glob("*.py")):
        anchors = extract_paper_element_ids(py)
        if not anchors:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError:
            # Attribution needs a parseable file. A package that does not parse
            # has bigger problems and its own gate; here it simply contributes
            # no bindings.
            continue
        scopes = collect_scopes(tree)
        for lineno, element_id in anchors:
            site = attribute_site(lineno, scopes)
            qualname = site["qualname"]
            if not qualname:
                continue
            out.setdefault(qualname, set()).add(element_id)
    return {q: tuple(sorted(ids)) for q, ids in sorted(out.items())}


def element_ids_for_callables(
    method_dir: Path, qualnames: "list[str] | tuple[str, ...]",
) -> tuple[str, ...]:
    """The paper-map element ids implemented by the given callables, merged and
    deduplicated.

    A qualname with no anchors contributes nothing, so a probe that exercised
    unannotated code stays unbound. That is the honest outcome: there is no
    recorded link between that code and the paper.

    A bare method name also matches its dotted form (`score` finds
    `Selector.score`) when the bare name is not itself defined, so a probe that
    invoked a method through an instance does not have to know the class.
    """
    table = element_ids_by_callable(method_dir)
    found: set = set()
    for raw in qualnames or ():
        name = str(raw or "")
        if not name:
            continue
        if name in table:
            found.update(table[name])
            continue
        suffix = f".{name}"
        for qualname, ids in table.items():
            if qualname.endswith(suffix):
                found.update(ids)
    return tuple(sorted(found))
