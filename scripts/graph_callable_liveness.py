"""Deterministic live-use evidence for homogeneous-graph callables.

The graph mechanism contract names directly probeable top-level helpers.  A
direct probe proves the helper's behaviour, but not that the delivered package
uses that helper.  This module closes that narrower identity gap without
inventing another semantic contract:

* graph construction is proven through one supported notebook value flow from
  the exact constructor output into the R2C-084 fitting and inference graph
  parameters; and
* message passing is proven through the exact R2C-084 architecture ``forward``
  path, where the declared helper receives the declared graph and neighbour
  signal and its result reaches the returned model value.

The implementation is deliberately dependency-light so the receipt assessment
can be vendored with the portable probe harness.  Unsupported Python grammar is
reported as coverage, never converted into a producer defect.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ALLOWED_GRAPH_CALLABLE_MODULES = frozenset({
    "method.model",
    "method.training",
    "method.method",
})
RECEIPT_VERSION = "1.6"
NUMERIC_OPERAND_DISPATCH_PRECONDITION = (
    "exact_base_numeric_arrays_on_frozen_fixture"
)


class GraphCallableLivenessError(ValueError):
    """Base typed error carrying a stable reason code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class GraphCallableProducerError(GraphCallableLivenessError):
    """The generated package disagrees with an otherwise supported contract."""


class GraphCallableCoverageError(GraphCallableLivenessError):
    """The current deterministic analyser cannot prove this honest shape."""


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GraphCallableCoverageError(
            "graph_liveness_contract_unavailable",
            f"{label} must be an object",
        )
    return value


def _exact_identifier(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or not value.isidentifier()
    ):
        raise GraphCallableCoverageError(
            "graph_liveness_contract_unavailable",
            f"{label} must be one exact Python identifier",
        )
    return value


def _callable_identity(value: object, *, label: str) -> dict[str, str]:
    raw = _mapping(value, label=label)
    if set(raw) != {"module", "qualname"}:
        raise GraphCallableCoverageError(
            "graph_callable_identity_malformed",
            f"{label} must contain exactly module and qualname",
        )
    module = raw.get("module")
    if module not in ALLOWED_GRAPH_CALLABLE_MODULES:
        raise GraphCallableProducerError(
            "graph_callable_module_not_allowed",
            f"{label}.module {module!r} is outside the exact v1 owner set",
        )
    qualname = _exact_identifier(raw.get("qualname"), label=f"{label}.qualname")
    if qualname.startswith("_"):
        raise GraphCallableProducerError(
            "graph_callable_not_public",
            f"{label}.qualname must name a public top-level helper",
        )
    return {"module": str(module), "qualname": qualname}


def _mechanism(method_spec: Mapping[str, Any]) -> Mapping[str, Any] | None:
    methodology = method_spec.get("methodology_replication_contract")
    if not isinstance(methodology, Mapping):
        return None
    value = methodology.get("homogeneous_graph_mechanism")
    return value if isinstance(value, Mapping) else None


def graph_callable_identities(
    method_spec: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str]] | None:
    """Return exact constructor/message identities, or ``None`` if inactive."""

    mechanism = _mechanism(method_spec)
    if mechanism is None:
        return None
    construction = _mapping(
        mechanism.get("construction"), label="homogeneous graph construction"
    )
    message = _mapping(
        mechanism.get("message_passing"), label="homogeneous graph message passing"
    )
    return (
        _callable_identity(
            construction.get("callable"), label="construction.callable"
        ),
        _callable_identity(message.get("callable"), label="message_passing.callable"),
    )


def _module_file(method_dir: Path, module_name: str) -> Path:
    relative = module_name.removeprefix("method.")
    if not relative or "." in relative:
        raise GraphCallableCoverageError(
            "graph_callable_module_grammar_unsupported",
            f"cannot map graph callable module {module_name!r} to one package file",
        )
    return method_dir / f"{relative}.py"


def _parse_file(path: Path, *, code: str) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except OSError as exc:
        raise GraphCallableProducerError(
            code, f"required generated source is unreadable: {path}: {exc}"
        ) from exc
    except SyntaxError as exc:
        raise GraphCallableProducerError(
            code, f"required generated source does not parse: {path}: {exc}"
        ) from exc


def _stitched_notebook(code_cells: Sequence[str]) -> ast.Module:
    if any(
        line.lstrip().startswith(("%", "!"))
        for cell in code_cells
        for line in str(cell).splitlines()
    ):
        raise GraphCallableCoverageError(
            "graph_constructor_notebook_grammar_unsupported",
            "notebook magics and shell escapes execute outside the closed graph "
            "liveness grammar",
        )
    source = "\n".join(str(cell) for cell in code_cells)
    try:
        return ast.parse(source)
    except SyntaxError as exc:
        raise GraphCallableCoverageError(
            "graph_constructor_notebook_grammar_unsupported",
            f"stitched notebook cannot be parsed: {exc}",
        ) from exc


def notebook_code_cells(run_dir: Path) -> list[str]:
    """Read code cells from the rendered notebook without importing nbformat."""

    path = Path(run_dir) / "notebook.ipynb"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphCallableCoverageError(
            "graph_constructor_notebook_unavailable",
            f"rendered notebook is unavailable or invalid: {exc}",
        ) from exc
    cells = payload.get("cells") if isinstance(payload, Mapping) else None
    if not isinstance(cells, list):
        raise GraphCallableCoverageError(
            "graph_constructor_notebook_unavailable",
            "rendered notebook has no cells list",
        )
    out: list[str] = []
    for cell in cells:
        if not isinstance(cell, Mapping) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source")
        if isinstance(source, list) and all(isinstance(line, str) for line in source):
            out.append("".join(source))
        elif isinstance(source, str):
            out.append(source)
    return out


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }


def _expression_path(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        prefix = _expression_path(node.value)
        return (*prefix, node.attr) if prefix else ()
    return ()


def _call_path(node: ast.Call) -> tuple[str, ...]:
    return _expression_path(node.func)


def _expression_contains_route(
    node: ast.AST,
    routes: set[tuple[str, ...]],
) -> bool:
    return any(
        _expression_path(observed) in routes
        for observed in ast.walk(node)
        if isinstance(observed, (ast.Name, ast.Attribute))
    )


_DYNAMIC_NAMESPACE_CALLS = frozenset({
    "globals",
    "locals",
    "vars",
    "exec",
    "eval",
    "setattr",
    "delattr",
})
_DYNAMIC_NAMESPACE_NAMES = _DYNAMIC_NAMESPACE_CALLS | frozenset({
    "__builtins__",
    "__import__",
    "attrgetter",
    "compile",
    "gc",
    "getattr",
    "get_ipython",
    "import_module",
    "importlib",
    "inspect",
    "methodcaller",
    "operator",
    "runpy",
    "subprocess",
    "ctypes",
    "sys",
})
_DYNAMIC_NAMESPACE_MODULES = frozenset({
    "IPython",
    "builtins",
    "ctypes",
    "gc",
    "importlib",
    "inspect",
    "operator",
    "runpy",
    "subprocess",
    "sys",
})
_UNCONDITIONAL_DYNAMIC_IMPORT_MODULES = frozenset({"IPython", "gc", "inspect"})


def _dynamic_namespace_import_bindings(
    statements: Sequence[ast.stmt],
) -> set[str]:
    bindings: set[str] = set()
    for node in statements:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in _DYNAMIC_NAMESPACE_MODULES:
                    bindings.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".", 1)[0]
            if module not in _DYNAMIC_NAMESPACE_MODULES:
                continue
            if module in _UNCONDITIONAL_DYNAMIC_IMPORT_MODULES:
                bindings.update(alias.asname or alias.name for alias in node.names)
            else:
                bindings.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if alias.name in _DYNAMIC_NAMESPACE_NAMES
                )
    return bindings


def _reject_dynamic_namespace(
    tree: ast.AST,
    *,
    label: str,
    module_tree: ast.Module | None = None,
) -> None:
    dynamic_names = set(_DYNAMIC_NAMESPACE_NAMES)
    if module_tree is not None:
        dynamic_names.update(_dynamic_namespace_import_bindings(module_tree.body))
    dynamic = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in dynamic_names:
            dynamic = True
            break
        if isinstance(node, ast.Import) and any(
            alias.name.split(".", 1)[0] in _DYNAMIC_NAMESPACE_MODULES
            for alias in node.names
        ):
            dynamic = True
            break
        if (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").split(".", 1)[0]
            in _DYNAMIC_NAMESPACE_MODULES
            and (
                (node.module or "").split(".", 1)[0]
                in _UNCONDITIONAL_DYNAMIC_IMPORT_MODULES
                or any(alias.name in _DYNAMIC_NAMESPACE_NAMES for alias in node.names)
            )
        ):
            dynamic = True
            break
    if dynamic:
        raise GraphCallableCoverageError(
            f"{label}_dynamic_namespace_unsupported",
            "dynamic namespace or attribute mutation is outside graph liveness v1",
        )


def _has_dynamic_namespace(tree: ast.AST) -> bool:
    try:
        _reject_dynamic_namespace(tree, label="graph_live")
    except GraphCallableCoverageError:
        return True
    return False


def _root_reexports_exact(method_dir: Path, module_name: str, qualname: str) -> bool:
    init_path = method_dir / "__init__.py"
    tree = _parse_file(init_path, code="graph_constructor_root_export_missing")
    expected_module = module_name.removeprefix("method.")
    origins = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module == expected_module
        and any(
            alias.name == qualname
            and (alias.asname or alias.name) == qualname
            for alias in node.names
        )
    ]
    if len(origins) != 1:
        return False
    try:
        init_nodes = _module_init_nodes(tree)
        init_reachable = _module_init_reachable_functions(tree)
        if init_reachable:
            return False
        execution_nodes = [
            *init_nodes,
            *(
                node
                for function in init_reachable
                for node in _function_scope_nodes(function)
            ),
        ]
        _reject_local_method_imports(
            tree,
            execution_nodes,
            code="graph_constructor_root_export_missing",
            method_dir=method_dir,
        )
    except GraphCallableCoverageError:
        return False
    for node in execution_nodes:
        if isinstance(node, ast.Name) and node.id == qualname:
            return False
        paths: list[tuple[str, ...]] = []
        if isinstance(node, ast.stmt):
            paths.extend(_module_statement_bound_paths(node))
        if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            root = _value_root_name(node)
            if root is not None:
                paths.append((root, "<descendant>"))
        if any(path and path[0] == qualname for path in paths):
            return False
    origin_line = origins[0].lineno
    for statement in tree.body:
        if statement.lineno <= origin_line:
            continue
        if any(
            path and path[0] == qualname
            for path in _module_statement_bound_paths(statement)
        ):
            return False
        if _has_dynamic_namespace(statement):
            return False
        for call in (
            node for node in ast.walk(statement) if isinstance(node, ast.Call)
        ):
            receiver = (
                [call.func.value]
                if isinstance(call.func, ast.Attribute)
                else []
            )
            values = (
                *receiver,
                *call.args,
                *(keyword.value for keyword in call.keywords),
            )
            if any(qualname in _expr_names(value) for value in values):
                return False
    return True


def _require_declarative_package_init(method_dir: Path) -> None:
    """Package import may only bind exports and literal metadata in liveness v1."""

    tree = _parse_file(
        Path(method_dir) / "__init__.py",
        code="graph_package_init_unavailable",
    )
    try:
        init_nodes = _module_init_nodes(tree)
        reachable = _module_init_reachable_functions(tree)
        _reject_local_method_imports(
            tree,
            init_nodes,
            code="graph_package_init_execution_unsupported",
            method_dir=Path(method_dir),
        )
    except GraphCallableCoverageError as exc:
        raise GraphCallableCoverageError(
            "graph_package_init_execution_unsupported",
            "method package initialization executes outside the closed export "
            "grammar",
        ) from exc
    if reachable or _has_dynamic_namespace(tree):
        raise GraphCallableCoverageError(
            "graph_package_init_execution_unsupported",
            "method package initialization may bind exports and literal metadata "
            "but cannot call producer-defined code or dynamic namespaces",
        )


def _exact_import_path(
    tree: ast.Module,
    *,
    module_name: str,
    qualname: str,
    method_dir: Path,
    label: str,
) -> tuple[tuple[str, ...], int]:
    routes: list[tuple[tuple[str, ...], int]] = []
    module_parts = tuple(module_name.split("."))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            if node.module == module_name:
                for alias in node.names:
                    if alias.name == qualname:
                        routes.append(((alias.asname or qualname,), node.lineno))
            elif node.module == "method" and _root_reexports_exact(
                method_dir, module_name, qualname
            ):
                for alias in node.names:
                    if alias.name == qualname:
                        routes.append(((alias.asname or qualname,), node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != module_name:
                    continue
                path = (
                    (alias.asname, qualname)
                    if alias.asname
                    else (*module_parts, qualname)
                )
                routes.append((path, node.lineno))
    if not routes:
        visible_same_name_route = any(
            (
                isinstance(node, ast.ImportFrom)
                and any(alias.name == qualname for alias in node.names)
            )
            or (
                isinstance(node, ast.Call)
                and _call_path(node)
                and _call_path(node)[-1] == qualname
            )
            for node in ast.walk(tree)
        )
        if visible_same_name_route:
            raise GraphCallableProducerError(
                f"{label}_route_disagreement",
                f"notebook visibly routes {qualname!r} somewhere other than "
                f"exact {module_name}:{qualname}",
            )
        raise GraphCallableCoverageError(
            f"{label}_exact_import_missing",
            f"version one cannot prove an indirect call to {module_name}:{qualname}",
        )
    if len(routes) != 1:
        raise GraphCallableCoverageError(
            f"{label}_import_grammar_unsupported",
            f"version one proves one exact import route for {module_name}:{qualname}",
        )
    return routes[0]


def _target_bound_paths(target: ast.AST) -> list[tuple[str, ...]]:
    if isinstance(target, (ast.Tuple, ast.List)):
        return [
            path
            for element in target.elts
            for path in _target_bound_paths(element)
        ]
    path = _expression_path(target)
    return [path] if path else []


def _bound_paths(statement: ast.stmt) -> list[tuple[str, ...]]:
    """Return visible module bindings created or replaced by one statement."""

    if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        return [path for target in targets for path in _target_bound_paths(target)]
    if isinstance(statement, ast.Delete):
        return [
            path
            for target in statement.targets
            for path in _target_bound_paths(target)
        ]
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [(statement.name,)]
    if isinstance(statement, (ast.For, ast.AsyncFor)):
        return _target_bound_paths(statement.target)
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        return [
            path
            for item in statement.items
            if item.optional_vars is not None
            for path in _target_bound_paths(item.optional_vars)
        ]
    if isinstance(statement, ast.Import):
        return [
            ((alias.asname or alias.name.split(".", 1)[0]),)
            for alias in statement.names
        ]
    if isinstance(statement, ast.ImportFrom):
        return [
            ((alias.asname or alias.name),)
            for alias in statement.names
            if alias.name != "*"
        ]
    return []


def _module_statement_bound_paths(statement: ast.stmt) -> list[tuple[str, ...]]:
    """Bindings one top-level statement can create in module scope."""

    def visit(node: ast.AST) -> list[tuple[str, ...]]:
        paths: list[tuple[str, ...]] = []
        if isinstance(node, ast.stmt):
            paths.extend(_bound_paths(node))
        if isinstance(node, ast.NamedExpr):
            paths.extend(_target_bound_paths(node.target))
        if isinstance(node, ast.ExceptHandler) and isinstance(node.name, str):
            paths.append((node.name,))
        if isinstance(node, (ast.MatchAs, ast.MatchStar)):
            name = node.name
            if isinstance(name, str):
                paths.append((name,))
        if isinstance(node, ast.MatchMapping) and isinstance(node.rest, str):
            paths.append((node.rest,))
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            return paths
        for child in ast.iter_child_nodes(node):
            paths.extend(visit(child))
        return paths

    return visit(statement)


def _route_rebound_before_call(
    tree: ast.Module,
    *,
    route: tuple[str, ...],
    imported_at: int,
    call: ast.Call,
) -> bool:
    for statement in tree.body:
        if statement.lineno <= imported_at or statement.lineno >= call.lineno:
            continue
        bound_paths = _module_statement_bound_paths(statement)
        if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            bound_paths.extend(
                (root, "<descendant>")
                for target in targets
                if (root := _value_root_name(target)) is not None
                and not isinstance(target, ast.Name)
            )
        if isinstance(statement, ast.Import) and len(route) > 1:
            safe_package_roots = {
                alias.name.split(".", 1)[0]
                for alias in statement.names
                if alias.asname is None and "." in alias.name
            }
            bound_paths = [
                bound
                for bound in bound_paths
                if not (bound == (route[0],) and route[0] in safe_package_roots)
            ]
        for bound in bound_paths:
            replaces_route = (
                len(bound) <= len(route) and route[: len(bound)] == bound
            )
            mutates_route = (
                len(bound) > len(route) and bound[: len(route)] == route
            )
            if replaces_route or mutates_route:
                return True
    return False


_CONTROL_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.Match,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.comprehension,
)


def _require_top_level_call(
    call: ast.Call, parents: Mapping[ast.AST, ast.AST], *, code: str, label: str
) -> None:
    node: ast.AST | None = call
    while node is not None and not isinstance(node, ast.Module):
        node = parents.get(node)
        if isinstance(node, _CONTROL_NODES):
            raise GraphCallableCoverageError(
                code,
                f"{label} occurs under unsupported conditional or nested control flow",
            )


def _assigned_name_for_selector(
    tree: ast.Module,
    call: ast.Call,
    selector: Mapping[str, Any],
    parents: Mapping[ast.AST, ast.AST],
) -> str:
    kind = selector.get("kind")
    parent = parents.get(call)
    if kind == "return_value":
        if isinstance(parent, (ast.Assign, ast.AnnAssign)):
            targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
            if len(targets) == 1 and isinstance(targets[0], ast.Name):
                return targets[0].id
    elif kind == "tuple_item":
        index = selector.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise GraphCallableCoverageError(
                "graph_constructor_output_selector_unsupported",
                "tuple graph output selector has no exact integer index",
            )
        if isinstance(parent, ast.Assign) and len(parent.targets) == 1:
            target = parent.targets[0]
            if (
                isinstance(target, (ast.Tuple, ast.List))
                and not any(isinstance(element, ast.Starred) for element in target.elts)
                and index < len(target.elts)
            ):
                selected = target.elts[index]
                if isinstance(selected, ast.Name):
                    return selected.id
        if isinstance(parent, ast.Subscript):
            try:
                selected_index = ast.literal_eval(parent.slice)
            except Exception:  # noqa: BLE001 - unsupported AST literal
                selected_index = None
            assignment = parents.get(parent)
            if (
                selected_index == index
                and isinstance(assignment, ast.Assign)
                and len(assignment.targets) == 1
                and isinstance(assignment.targets[0], ast.Name)
            ):
                return assignment.targets[0].id
    elif kind == "mapping_item":
        key = selector.get("key")
        if isinstance(parent, ast.Subscript):
            try:
                selected_key = ast.literal_eval(parent.slice)
            except Exception:  # noqa: BLE001 - unsupported AST literal
                selected_key = None
            assignment = parents.get(parent)
            if (
                selected_key == key
                and isinstance(assignment, ast.Assign)
                and len(assignment.targets) == 1
                and isinstance(assignment.targets[0], ast.Name)
            ):
                return assignment.targets[0].id
    raise GraphCallableCoverageError(
        "graph_constructor_output_selector_unsupported",
        "version one requires the selected constructor output to bind one name "
        "directly by return, tuple unpack/index, or mapping index",
    )


def _assigned_call_name(
    call: ast.Call,
    parents: Mapping[ast.AST, ast.AST],
    *,
    label: str,
) -> str:
    parent = parents.get(call)
    if isinstance(parent, ast.Assign) and parent.value is call:
        if len(parent.targets) == 1 and isinstance(parent.targets[0], ast.Name):
            return parent.targets[0].id
    if (
        isinstance(parent, ast.AnnAssign)
        and parent.value is call
        and isinstance(parent.target, ast.Name)
    ):
        return parent.target.id
    raise GraphCallableCoverageError(
        f"{label}_result_binding_unsupported",
        f"version one requires the exact {label} call result in one local name",
    )


def _root_parameter(root: object, *, prefix: str, label: str) -> str:
    if not isinstance(root, str) or not root.startswith(prefix) or root == prefix:
        raise GraphCallableCoverageError(
            "graph_liveness_relational_binding_unavailable",
            f"{label} must be one exact root below {prefix!r}",
        )
    value = root[len(prefix):]
    return _exact_identifier(value, label=label)


def _relational_bindings(
    arch_contract: Mapping[str, Any], mechanism: Mapping[str, Any]
) -> dict[str, str]:
    relational = _mapping(
        arch_contract.get("relational_indexing"), label="relational_indexing"
    )
    execution = _mapping(relational.get("execution"), label="relational execution")
    fitting = _mapping(execution.get("fitting"), label="relational fitting")
    inference = _mapping(execution.get("inference"), label="relational inference")
    architecture_block = _exact_identifier(
        inference.get("architecture_block"), label="inference architecture block"
    )
    fitting_graph = _root_parameter(
        fitting.get("graph_input_root"),
        prefix="training_loop.input.",
        label="fitting graph root",
    )
    fitting_model = _root_parameter(
        fitting.get("model_input_root"),
        prefix="training_loop.input.",
        label="fitting model root",
    )
    construction = _mapping(
        mechanism.get("construction"), label="homogeneous graph construction"
    )
    feature_root = construction.get("feature_input_root")
    fitting_coindexed = _mapping(
        fitting.get("coindexed_input_roots"), label="fitting coindexed roots"
    )
    feature_typed_root = fitting_coindexed.get(feature_root)
    construction_feature = _root_parameter(
        feature_typed_root,
        prefix="training_loop.input.",
        label="construction feature root",
    )
    inference_prefix = f"architecture.{architecture_block}.forward.input."
    inference_graph = _root_parameter(
        inference.get("graph_input_root"),
        prefix=inference_prefix,
        label="inference graph root",
    )
    message = _mapping(
        mechanism.get("message_passing"), label="homogeneous graph message passing"
    )
    neighbor_root = message.get("neighbor_signal_root")
    coindexed = _mapping(
        inference.get("coindexed_input_roots"), label="inference coindexed roots"
    )
    neighbor_typed_root = coindexed.get(neighbor_root)
    inference_neighbor = _root_parameter(
        neighbor_typed_root,
        prefix=inference_prefix,
        label="inference neighbor-signal root",
    )
    output_root = inference.get("output_coindexed_root")
    if not isinstance(output_root, str) or not output_root:
        raise GraphCallableCoverageError(
            "graph_liveness_relational_binding_unavailable",
            "inference output_coindexed_root is unavailable",
        )
    return {
        "architecture_block": architecture_block,
        "fitting_model_parameter": fitting_model,
        "fitting_graph_parameter": fitting_graph,
        "construction_feature_parameter": construction_feature,
        "inference_graph_parameter": inference_graph,
        "inference_neighbor_parameter": inference_neighbor,
        "inference_input_logical_roots": sorted(coindexed),
        "inference_output_root": output_root,
    }


def _expr_names(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def _top_level_statement(
    node: ast.AST,
    parents: Mapping[ast.AST, ast.AST],
) -> ast.stmt:
    current = node
    while current in parents and not isinstance(
        parents[current], (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        current = parents[current]
    if not isinstance(current, ast.stmt):
        raise GraphCallableCoverageError(
            "graph_live_flow_grammar_unsupported",
            "live graph call has no supported straight-line statement owner",
        )
    return current


def _direct_aliases_before_node(
    body: Sequence[ast.stmt],
    *,
    seed: str,
    node: ast.AST,
    parents: Mapping[ast.AST, ast.AST],
    label: str,
    after_node: ast.AST | None = None,
    allowed_call_paths: frozenset[tuple[str, ...]] = frozenset(),
) -> set[str]:
    """Track one provenance token through direct straight-line name aliases.

    Assignment replaces provenance.  Any transformation of a proven value is
    outside the closed static grammar: treating arbitrary arithmetic or calls
    as dependency-preserving would certify ``graph * 0`` and similar dead
    paths.
    """

    target_statement = _top_level_statement(node, parents)
    start_statement = (
        _top_level_statement(after_node, parents)
        if after_node is not None
        else None
    )
    aliases = {seed}
    started = start_statement is None
    for statement in body:
        if not started:
            if statement is start_statement:
                started = True
            continue
        if statement is target_statement:
            return aliases
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            value = statement.value
            if value is None:
                continue
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            target_names = {
                name for target in targets for name in _target_names(target)
            }
            mutated_roots = {
                root
                for target in targets
                if (root := _value_root_name(target)) is not None
                and not isinstance(target, ast.Name)
            }
            if mutated_roots & aliases:
                raise GraphCallableCoverageError(
                    f"{label}_mutation_unsupported",
                    f"{label} provenance is mutated through an item or attribute write",
                )
            direct_source = (
                value.id
                if isinstance(value, ast.Name) and value.id in aliases
                else None
            )
            if direct_source is None and _expr_names(value) & aliases:
                call_paths = {
                    _call_path(call)
                    for call in ast.walk(value)
                    if isinstance(call, ast.Call)
                }
                if not call_paths or not call_paths <= allowed_call_paths:
                    raise GraphCallableCoverageError(
                        f"{label}_alias_transform_unsupported",
                        f"{label} provenance passes through an unsupported transform",
                    )
            aliases.difference_update(target_names)
            if direct_source is not None:
                aliases.update(target_names)
            continue
        if isinstance(statement, ast.AugAssign):
            if _expr_names(statement) & aliases:
                raise GraphCallableCoverageError(
                    f"{label}_alias_transform_unsupported",
                    f"{label} provenance is mutated by an augmented assignment",
                )
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            if (
                _expr_names(statement.value) & aliases
                and _call_path(statement.value) not in allowed_call_paths
            ):
                raise GraphCallableCoverageError(
                    f"{label}_mutation_unsupported",
                    f"{label} provenance escapes into an opaque callable",
                )
        if isinstance(statement, _CONTROL_NODES):
            names = _expr_names(statement)
            bound = {
                path[0]
                for path in _module_statement_bound_paths(statement)
                if path
            }
            if names & aliases or bound & aliases:
                raise GraphCallableCoverageError(
                    f"{label}_control_flow_unsupported",
                    f"{label} provenance crosses unsupported control flow",
                )
        rebound = {
            path[0]
            for path in _module_statement_bound_paths(statement)
            if len(path) == 1
        }
        aliases.difference_update(rebound)
    raise GraphCallableCoverageError(
        f"{label}_statement_order_unsupported",
        f"cannot locate the supported statement containing {label}",
    )


def _direct_keyword_alias(
    call: ast.Call,
    *,
    parameter: str,
    aliases: set[str],
    label: str,
) -> bool:
    matches = [keyword for keyword in call.keywords if keyword.arg == parameter]
    if len(matches) != 1:
        return False
    value = matches[0].value
    if isinstance(value, ast.Name):
        return value.id in aliases
    if _expr_names(value) & aliases:
        raise GraphCallableCoverageError(
            f"{label}_alias_transform_unsupported",
            f"{label} keyword {parameter!r} transforms the proven value",
        )
    return False


def _cfg_key(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
        return None
    if node.value.id != "cfg":
        return None
    try:
        key = ast.literal_eval(node.slice)
    except Exception:  # noqa: BLE001 - unsupported AST literal
        return None
    return key if isinstance(key, str) else None


def _value_root_name(node: ast.AST) -> str | None:
    current = node
    while isinstance(current, (ast.Attribute, ast.Subscript)):
        current = current.value
    return current.id if isinstance(current, ast.Name) else None


_EFFECTFUL_ARGUMENT_NODES = (
    ast.Await,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Lambda,
    ast.ListComp,
    ast.NamedExpr,
    ast.SetComp,
    ast.Yield,
    ast.YieldFrom,
)


def _reject_effectful_call_arguments(
    call: ast.Call,
    *,
    code: str,
    label: str,
) -> None:
    """Keep protected-call argument evaluation inside a side-effect-free v1.

    Python evaluates arguments in source order.  A walrus expression or nested
    opaque call in an earlier role can therefore replace a later graph/model
    binding even when each final keyword looks like a direct name.  Closed v1
    requires those effects to be named in preceding straight-line statements.
    """

    values = (*call.args, *(keyword.value for keyword in call.keywords))
    if any(
        isinstance(node, (ast.Call, *_EFFECTFUL_ARGUMENT_NODES))
        or (
            isinstance(node, (ast.Name, ast.Attribute, ast.Subscript))
            and isinstance(node.ctx, (ast.Store, ast.Del))
        )
        for value in values
        for node in ast.walk(value)
    ):
        raise GraphCallableCoverageError(
            code,
            f"{label} arguments must be side-effect-free direct values; nested "
            "calls, assignment expressions, and writes are outside graph "
            "liveness v1",
        )


def _exact_keyword_call_values(
    call: ast.Call,
    *,
    parameters: set[str],
    code: str,
    label: str,
) -> dict[str, ast.AST]:
    keyword_names = [keyword.arg for keyword in call.keywords]
    if (
        call.args
        or any(name is None for name in keyword_names)
        or len(keyword_names) != len(parameters)
        or set(keyword_names) != parameters
    ):
        raise GraphCallableCoverageError(
            code,
            f"{label} must bind exactly the declared keyword set "
            f"{sorted(parameters)!r}",
        )
    _reject_effectful_call_arguments(call, code=code, label=label)
    return {
        str(keyword.arg): keyword.value
        for keyword in call.keywords
        if keyword.arg is not None
    }


def _numeric_runtime_import_bindings(tree: ast.Module) -> set[str]:
    """Names that can mutate process-global NumPy/Torch dispatch state."""

    bindings: set[str] = set()
    for statement in ast.walk(tree):
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name.split(".", 1)[0] in {"numpy", "torch"}:
                    bindings.add(alias.asname or alias.name.split(".", 1)[0])
        elif (
            isinstance(statement, ast.ImportFrom)
            and (statement.module or "").split(".", 1)[0] in {"numpy", "torch"}
        ):
            bindings.update(
                alias.asname or alias.name
                for alias in statement.names
                if alias.name != "*"
            )
    return bindings


def _reject_numeric_runtime_binding_mutation(
    tree: ast.Module,
    nodes: Sequence[ast.AST],
    *,
    code: str,
) -> None:
    """Freeze imported numeric modules/types used by the liveness proof."""

    if any(
        isinstance(statement, ast.ImportFrom)
        and (statement.module or "").split(".", 1)[0] in {"numpy", "torch"}
        and any(alias.name == "*" for alias in statement.names)
        for statement in ast.walk(tree)
    ):
        raise GraphCallableCoverageError(
            code,
            "star imports from NumPy or Torch hide mutable numeric dispatch "
            "bindings from the closed liveness grammar",
        )
    bindings = _numeric_runtime_import_bindings(tree)
    if not bindings:
        return
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if (
            isinstance(node, (ast.Attribute, ast.Subscript))
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and _value_root_name(node) in bindings
        ):
            raise GraphCallableCoverageError(
                code,
                "imported NumPy/Torch module or type state is mutated through "
                "a descendant target",
            )
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.id in bindings
        ):
            raise GraphCallableCoverageError(
                code,
                "an imported NumPy/Torch module or type binding is rebound",
            )
        value: ast.AST | None = None
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            value = node.value
        elif isinstance(node, ast.NamedExpr):
            value = node.value
        elif isinstance(node, ast.Return):
            value = node.value
        if value is not None and _contains_numeric_namespace_value(
            value, bindings
        ):
            raise GraphCallableCoverageError(
                code,
                "an imported NumPy/Torch module or type escapes through an "
                "alias or return value",
            )
        if isinstance(node, ast.Call):
            path = _call_path(node)
            if (
                path
                and path[0] in bindings
                and any(
                    component in {
                        "__delattr__", "__dict__", "__setattr__",
                    }
                    for component in path[1:]
                )
            ):
                raise GraphCallableCoverageError(
                    code,
                    "imported NumPy/Torch mutable namespace state is outside "
                    "graph liveness v1",
                )
            if any(
                _contains_numeric_namespace_value(value, bindings)
                for value in (
                    *node.args,
                    *(keyword.value for keyword in node.keywords),
                )
            ):
                raise GraphCallableCoverageError(
                    code,
                    "an imported NumPy/Torch module or type enters an opaque "
                    "callable",
                )


def _verify_cfg_authority(tree: ast.Module, call: ast.Call) -> None:
    canonical = 0
    aliases: set[str] = set()
    for statement in tree.body:
        if statement.lineno >= call.lineno:
            break
        if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            if any(_value_root_name(target) == "cfg" for target in targets):
                value = statement.value
                is_canonical = (
                    len(targets) == 1
                    and isinstance(targets[0], ast.Name)
                    and targets[0].id == "cfg"
                    and isinstance(value, ast.Call)
                    and _call_path(value) == ("unpack",)
                    and len(value.args) == 1
                    and isinstance(value.args[0], ast.Name)
                    and value.args[0].id == "params"
                    and not value.keywords
                )
                if is_canonical and canonical == 0:
                    canonical += 1
                    aliases = {"cfg"}
                    continue
                raise GraphCallableProducerError(
                    "graph_constructor_cfg_authority_rebound",
                    "the render-owned cfg = unpack(params) authority is rebound or "
                    "mutated before graph construction",
                )
        if canonical != 1:
            continue
        if any(
            path and path[0] in aliases
            for path in _module_statement_bound_paths(statement)
        ):
            raise GraphCallableProducerError(
                "graph_constructor_cfg_authority_rebound",
                "the render-owned cfg mapping or one of its direct aliases is "
                "rebound or mutated before graph construction",
            )
        for opaque_call in (
            node for node in ast.walk(statement) if isinstance(node, ast.Call)
        ):
            receiver = (
                [opaque_call.func.value]
                if isinstance(opaque_call.func, ast.Attribute)
                else []
            )
            values = (
                *receiver,
                *opaque_call.args,
                *(keyword.value for keyword in opaque_call.keywords),
            )
            if any(_expr_names(value) & aliases for value in values):
                raise GraphCallableCoverageError(
                    "graph_constructor_cfg_authority_escape_unsupported",
                    "the render-owned cfg mapping or one of its direct aliases "
                    "enters an opaque call before graph construction",
                )
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            target_names = {
                name for target in targets for name in _target_names(target)
            }
            source_is_alias = (
                isinstance(statement.value, ast.Name)
                and statement.value.id in aliases
            )
            aliases.difference_update(target_names)
            if source_is_alias:
                aliases.update(target_names)
    if canonical != 1:
        raise GraphCallableCoverageError(
            "graph_constructor_cfg_authority_unavailable",
            "version one requires the render-owned cfg = unpack(params) binding",
        )


def _constructor_live_bindings(
    call: ast.Call,
    construction: Mapping[str, Any],
    *,
    feature_binding: str,
) -> dict[str, Any]:
    if call.args or any(keyword.arg is None for keyword in call.keywords):
        raise GraphCallableCoverageError(
            "graph_constructor_call_binding_grammar_unsupported",
            "version one requires explicit constructor keyword bindings",
        )
    by_name: dict[str, list[ast.AST]] = {}
    for keyword in call.keywords:
        if keyword.arg is not None:
            by_name.setdefault(keyword.arg, []).append(keyword.value)

    feature_parameter = _exact_identifier(
        construction.get("feature_parameter"), label="construction feature parameter"
    )
    feature_root = construction.get("feature_input_root")
    if not isinstance(feature_root, str) or not feature_root:
        raise GraphCallableCoverageError(
            "graph_constructor_feature_root_grammar_unsupported",
            "feature_input_root is unavailable for exact R2C-084 binding",
        )
    feature_values = by_name.get(feature_parameter, [])
    if (
        len(feature_values) != 1
        or not isinstance(feature_values[0], ast.Name)
        or feature_values[0].id != feature_binding
    ):
        raise GraphCallableProducerError(
            "graph_constructor_feature_binding_mismatch",
            "the exact constructor feature keyword does not consume the local "
            f"binding for {feature_root!r}",
        )

    parameter_bindings: dict[str, dict[str, str]] = {}
    threshold = _mapping(
        construction.get("threshold"), label="construction threshold"
    )
    parameter_contracts = [
        _mapping(threshold.get("parameter"), label="construction threshold parameter")
    ]
    cap = _mapping(construction.get("cap"), label="construction cap")
    if cap.get("kind") != "none":
        parameter_contracts.append(
            _mapping(cap.get("parameter"), label="construction cap parameter")
        )
    for binding in parameter_contracts:
        params_name = _exact_identifier(
            binding.get("params_name"), label="construction params name"
        )
        callable_parameter = _exact_identifier(
            binding.get("callable_parameter"),
            label="construction callable parameter",
        )
        values = by_name.get(callable_parameter, [])
        if len(values) != 1:
            raise GraphCallableProducerError(
                "graph_constructor_parameter_binding_mismatch",
                "the exact constructor parameter keyword is missing or duplicated",
            )
        observed_cfg_key = _cfg_key(values[0])
        if observed_cfg_key != params_name:
            if isinstance(values[0], ast.Constant) or observed_cfg_key is not None:
                raise GraphCallableProducerError(
                    "graph_constructor_parameter_binding_mismatch",
                    "the exact constructor parameter keyword must consume "
                    f"cfg[{params_name!r}] rather than a literal or another key",
                )
            raise GraphCallableCoverageError(
                "graph_constructor_parameter_binding_grammar_unsupported",
                "version one cannot prove an indirect alias of "
                f"cfg[{params_name!r}]",
            )
        parameter_bindings[params_name] = {
            "callable_parameter": callable_parameter,
            "notebook_cfg_key": params_name,
        }
    extra_parameters = sorted(set(by_name) - {
        feature_parameter,
        *(binding["callable_parameter"] for binding in parameter_bindings.values()),
    })
    if extra_parameters:
        raise GraphCallableCoverageError(
            "graph_constructor_call_binding_grammar_unsupported",
            "version one proves only the exact declared feature and graph-parameter "
            f"keywords; extra keywords={extra_parameters!r}",
        )
    return {
        "feature_input_root": feature_root,
        "feature_parameter": feature_parameter,
        "feature_binding": feature_binding,
        "parameter_bindings": parameter_bindings,
    }


def _future_annotations_enabled(tree: ast.Module) -> bool:
    return any(
        isinstance(statement, ast.ImportFrom)
        and statement.module == "__future__"
        and any(alias.name == "annotations" for alias in statement.names)
        for statement in tree.body
    )


def _reject_executable_function_header(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    tree: ast.Module,
) -> None:
    if function.decorator_list or getattr(function, "type_params", ()):
        raise GraphCallableCoverageError(
            "graph_live_callable_header_unsupported",
            "decorated or generic graph-path callables are outside the closed "
            "liveness grammar",
        )
    defaults = (*function.args.defaults, *function.args.kw_defaults)
    for default in defaults:
        if default is None:
            continue
        try:
            ast.literal_eval(default)
        except Exception as exc:  # noqa: BLE001 - closed header grammar
            raise GraphCallableCoverageError(
                "graph_live_callable_header_unsupported",
                "callable defaults must be inert literals in graph liveness v1",
            ) from exc
    annotations = [
        argument.annotation
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
            *([function.args.vararg] if function.args.vararg is not None else []),
            *([function.args.kwarg] if function.args.kwarg is not None else []),
        )
        if argument.annotation is not None
    ]
    if function.returns is not None:
        annotations.append(function.returns)
    if annotations and not _future_annotations_enabled(tree):
        raise GraphCallableCoverageError(
            "graph_live_callable_header_unsupported",
            "runtime-evaluated annotations are outside the closed liveness grammar",
        )


def _function_def(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(matches) != 1:
        raise GraphCallableProducerError(
            "graph_live_callable_definition_mismatch",
            f"expected exactly one top-level definition of {name!r}",
        )
    function = matches[0]
    _reject_executable_function_header(function, tree)
    _reject_deferred_callable(function)
    return function


def _reject_deferred_callable(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> None:
    if isinstance(function, ast.AsyncFunctionDef) or any(
        isinstance(node, (ast.Yield, ast.YieldFrom))
        for node in _function_scope_nodes(function)
    ):
        raise GraphCallableCoverageError(
            "graph_live_deferred_callable_unsupported",
            "async and generator callables do not execute synchronously under "
            "graph liveness v1",
        )
    if any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))
        for node in _function_scope_nodes(function)
    ):
        raise GraphCallableCoverageError(
            "graph_live_nested_callable_unsupported",
            "nested callables and classes are outside the closed synchronous "
            "graph liveness grammar",
        )


def _is_literal_assignment(statement: ast.stmt) -> bool:
    if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
        return False
    value = statement.value
    if value is None:
        return True
    try:
        ast.literal_eval(value)
    except Exception:  # noqa: BLE001 - closed declarative grammar
        return False
    return True


def _declarative_class_body(class_node: ast.ClassDef, tree: ast.Module) -> bool:
    if class_node.decorator_list or class_node.keywords:
        return False
    if any(
        _expression_path(base)
        not in {("object",), ("nn", "Module"), ("torch", "nn", "Module")}
        for base in class_node.bases
    ):
        return False
    for statement in class_node.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                _reject_executable_function_header(statement, tree)
            except GraphCallableCoverageError:
                return False
            continue
        if isinstance(statement, ast.Pass):
            continue
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        if isinstance(statement, ast.AnnAssign) and not _future_annotations_enabled(tree):
            return False
        if _is_literal_assignment(statement):
            continue
        return False
    return True


def _has_exact_torch_module_base(
    tree: ast.Module,
    class_node: ast.ClassDef,
) -> bool:
    if len(class_node.bases) != 1:
        return False
    base = _expression_path(class_node.bases[0])
    origins: list[tuple[tuple[str, ...], int]] = []
    for statement in tree.body:
        if statement.lineno >= class_node.lineno:
            break
        if isinstance(statement, ast.ImportFrom) and statement.module == "torch":
            for alias in statement.names:
                if alias.name == "nn":
                    origins.append(((alias.asname or "nn", "Module"), statement.lineno))
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name == "torch.nn" and alias.asname:
                    origins.append(((alias.asname, "Module"), statement.lineno))
                elif alias.name in {"torch", "torch.nn"} and alias.asname is None:
                    origins.append((("torch", "nn", "Module"), statement.lineno))
    matching = [(route, line) for route, line in origins if route == base]
    if len(matching) != 1:
        return False
    route, origin_line = matching[0]
    for statement in tree.body:
        if statement.lineno <= origin_line or statement.lineno >= class_node.lineno:
            continue
        for bound in _module_statement_bound_paths(statement):
            if (
                len(bound) <= len(route)
                and route[: len(bound)] == bound
            ) or (
                len(bound) > len(route)
                and bound[: len(route)] == route
            ):
                return False
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if statement.decorator_list:
                return False
            continue
        if isinstance(statement, ast.ClassDef):
            if not _declarative_class_body(statement, tree):
                return False
            continue
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ) or _is_literal_assignment(statement):
            continue
        return False
    return True


def _pluggable_graph_parameter(
    method_dir: Path,
    arch_contract: Mapping[str, Any],
    inference_graph_parameter: str,
    inference_neighbor_parameter: str,
    fitting_model_parameter: str,
    *,
    protected_qualnames: set[str],
) -> tuple[str, str, str, str]:
    pluggable = _mapping(
        arch_contract.get("pluggable_component"), label="pluggable component"
    )
    pluggable_name = _exact_identifier(
        pluggable.get("name"), label="pluggable component name"
    )
    training = _mapping(arch_contract.get("training_loop"), label="training loop")
    training_inputs = _mapping(training.get("input"), label="training loop inputs")
    pluggable_inputs = _mapping(
        pluggable.get("input"), label="pluggable component inputs"
    )
    model_descriptor = training_inputs.get(fitting_model_parameter)
    pluggable_model_parameters = [
        name
        for name, descriptor in pluggable_inputs.items()
        if descriptor == model_descriptor
    ]
    if len(pluggable_model_parameters) != 1:
        raise GraphCallableCoverageError(
            "graph_constructor_pluggable_model_binding_unsupported",
            "version one requires one pluggable input whose typed descriptor "
            "equals the R2C-084 fitting model root",
        )
    pluggable_model_parameter = _exact_identifier(
        pluggable_model_parameters[0], label="pluggable model parameter"
    )
    tree = _parse_file(
        method_dir / "method.py", code="graph_pluggable_source_unavailable"
    )
    function = _function_def(tree, pluggable_name)
    _require_final_module_binding(
        tree,
        route=(pluggable_name,),
        origin_line=function.lineno,
        label="graph_pluggable",
        reject_descendant_bindings=True,
        reject_opaque_escape=True,
    )
    _reject_cross_callable_references(
        tree,
        function,
        protected_qualnames=protected_qualnames - {pluggable_name},
        locally_owned_qualnames=_locally_owned_protected_qualnames(
            tree, protected_qualnames - {pluggable_name}
        ),
        label="graph_pluggable",
        method_dir=method_dir,
    )
    if function.decorator_list:
        raise GraphCallableCoverageError(
            "graph_pluggable_dynamic_dispatch_unsupported",
            "decorated pluggable dispatch is outside graph liveness v1",
        )
    parameters = [
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    ]
    if pluggable_model_parameter not in parameters:
        raise GraphCallableProducerError(
            "graph_constructor_pluggable_model_parameter_missing",
            "the declared pluggable omits the typed architecture-model input "
            f"{pluggable_model_parameter!r}",
        )
    parents = _parents(tree)
    model_calls = [
        call
        for call in (node for node in ast.walk(function) if isinstance(node, ast.Call))
        if _expression_path(call.func) in (
            (pluggable_model_parameter,),
            (pluggable_model_parameter, "forward"),
        )
    ]
    if len(model_calls) != 1:
        raise GraphCallableCoverageError(
            "graph_constructor_inference_grammar_unsupported",
            "version one proves exactly one architecture-model call in the "
            "pluggable component",
        )
    architecture = _mapping(
        arch_contract.get("architecture"), label="architecture"
    )
    relational = _mapping(
        arch_contract.get("relational_indexing"),
        label="relational indexing",
    )
    relational_execution = _mapping(
        relational.get("execution"), label="relational execution"
    )
    inference = _mapping(
        relational_execution.get("inference"),
        label="relational inference execution",
    )
    block_name = _exact_identifier(
        inference.get("architecture_block"),
        label="relational architecture block",
    )
    block = _mapping(architecture.get(block_name), label="architecture block")
    forward = _mapping(block.get("forward"), label="architecture forward")
    forward_inputs = _mapping(
        forward.get("input"), label="architecture forward inputs"
    )
    expected_forward_parameters = {
        _exact_identifier(name, label="architecture forward input")
        for name in forward_inputs
    }
    for call in model_calls:
        _exact_keyword_call_values(
            call,
            parameters=expected_forward_parameters,
            code="graph_constructor_inference_grammar_unsupported",
            label="the exact pluggable architecture-model call",
        )
    for call in model_calls:
        current: ast.AST = call
        while current in parents and parents[current] is not function:
            current = parents[current]
            if isinstance(current, _CONTROL_NODES):
                raise GraphCallableCoverageError(
                    "graph_constructor_inference_grammar_unsupported",
                    "the pluggable model call occurs under unsupported control flow",
                )
    graph_candidates: list[str] = []
    neighbor_candidates: list[str] = []
    for parameter in parameters:
        reaches_graph = False
        reaches_neighbor = False
        for call in model_calls:
            model_aliases = _direct_aliases_before_node(
                function.body,
                seed=pluggable_model_parameter,
                node=call,
                parents=parents,
                label="graph_constructor_pluggable_model",
            )
            if pluggable_model_parameter not in model_aliases:
                continue
            aliases = _direct_aliases_before_node(
                function.body,
                seed=parameter,
                node=call,
                parents=parents,
                label="graph_constructor_pluggable_graph",
            )
            if _direct_keyword_alias(
                call,
                parameter=inference_graph_parameter,
                aliases=aliases,
                label="graph_constructor_pluggable_graph",
            ):
                reaches_graph = True
            if _direct_keyword_alias(
                call,
                parameter=inference_neighbor_parameter,
                aliases=aliases,
                label="graph_constructor_pluggable_neighbor_signal",
            ):
                reaches_neighbor = True
        if reaches_graph:
            graph_candidates.append(parameter)
        if reaches_neighbor:
            neighbor_candidates.append(parameter)
    if not graph_candidates:
        raise GraphCallableProducerError(
            "graph_constructor_inference_path_missing",
            "the declared pluggable does not carry any input into the exact "
            f"model-forward graph parameter {inference_graph_parameter!r}",
        )
    if len(graph_candidates) != 1:
        raise GraphCallableCoverageError(
            "graph_constructor_inference_grammar_unsupported",
            "version one requires exactly one pluggable input to reach the "
            "model-forward graph parameter",
        )
    if not neighbor_candidates:
        raise GraphCallableProducerError(
            "graph_constructor_neighbor_inference_path_missing",
            "the declared pluggable does not carry any input into the exact "
            "model-forward neighbor-signal parameter "
            f"{inference_neighbor_parameter!r}",
        )
    if len(neighbor_candidates) != 1 or (
        neighbor_candidates[0] == graph_candidates[0]
    ):
        raise GraphCallableCoverageError(
            "graph_constructor_inference_grammar_unsupported",
            "version one requires distinct exact pluggable inputs to reach the "
            "model-forward graph and neighbor-signal parameters",
        )
    model_call_path = _expression_path(model_calls[0].func)
    class_name = _exact_identifier(
        block.get("class_name"), label="architecture class"
    )
    model_tree = _parse_file(
        method_dir / "model.py", code="graph_model_source_unavailable"
    )
    classes = [
        node
        for node in model_tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        raise GraphCallableProducerError(
            "graph_live_architecture_class_missing",
            f"method/model.py does not define exactly one {class_name!r}",
        )
    _require_final_module_binding(
        model_tree,
        route=(class_name,),
        origin_line=classes[0].lineno,
        label="graph_architecture_class",
        reject_descendant_bindings=True,
        reject_opaque_escape=True,
    )
    if model_call_path == (pluggable_model_parameter,):
        if any(
            ("__call__",) in _module_statement_bound_paths(statement)
            for statement in classes[0].body
        ):
            raise GraphCallableCoverageError(
                "graph_pluggable_dynamic_dispatch_unsupported",
                "custom architecture __call__ can bypass the analyzed forward",
            )
        if not _has_exact_torch_module_base(model_tree, classes[0]):
            raise GraphCallableCoverageError(
                "graph_pluggable_model_call_dispatch_unsupported",
                "implicit model calls require one exact torch.nn.Module base; "
                "otherwise call the analyzed forward explicitly",
            )
    forwards = [
        node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "forward"
    ]
    if len(forwards) != 1:
        raise GraphCallableProducerError(
            "graph_live_forward_missing",
            "the architecture class must define one exact forward method",
        )
    _require_straight_line_live_body(forwards[0])
    initial_env = {parameter: {parameter} for parameter in parameters}
    returned, _ = _analyse_function_body(
        function,
        initial_env,
        {},
        model_call_path,
        None,
        stack=(pluggable_name,),
    )
    if _HELPER_TOKEN not in returned:
        raise GraphCallableProducerError(
            "graph_constructor_pluggable_result_discarded",
            "the exact architecture-model call result does not reach the "
            "pluggable return",
        )
    output_key_tokens = {
        dependency.removeprefix(_HELPER_OUTPUT_KEY_PREFIX)
        for dependency in returned
        if dependency.startswith(_HELPER_OUTPUT_KEY_PREFIX)
    }
    final_return_is_mapping = (
        bool(function.body)
        and isinstance(function.body[-1], ast.Return)
        and isinstance(function.body[-1].value, ast.Dict)
    )
    output_root = inference.get("output_coindexed_root")
    if not isinstance(output_root, str) or "." not in output_root:
        raise GraphCallableCoverageError(
            "graph_constructor_pluggable_output_root_unsupported",
            "the pluggable output has no exact coindexed output-root leaf",
        )
    if (
        (output_key_tokens or final_return_is_mapping)
        and output_root.rsplit(".", 1)[-1] not in output_key_tokens
    ):
        raise GraphCallableProducerError(
            "graph_constructor_pluggable_output_root_mismatch",
            "the exact architecture result does not reach the declared pluggable "
            "output leaf",
        )
    return (
        pluggable_name,
        pluggable_model_parameter,
        graph_candidates[0],
        neighbor_candidates[0],
    )


def _exact_top_level_calls(
    tree: ast.Module,
    *,
    call_path: tuple[str, ...],
    imported_at: int,
    parents: Mapping[ast.AST, ast.AST],
    label: str,
) -> list[ast.Call]:
    exact = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_path(node) == call_path
    ]
    for call in exact:
        _require_top_level_call(
            call,
            parents,
            code=f"{label}_control_flow_unsupported",
            label=label,
        )
        if call.lineno <= imported_at:
            raise GraphCallableCoverageError(
                f"{label}_import_grammar_unsupported",
                f"{label} call occurs before its exact import binding",
            )
        if _route_rebound_before_call(
            tree, route=call_path, imported_at=imported_at, call=call
        ):
            raise GraphCallableProducerError(
                f"{label}_binding_shadowed",
                f"the exact {label} import binding is replaced before its live call",
            )
    return exact


def _fitting_graph_copy_binding(
    body: Sequence[ast.stmt],
    *,
    selected_name: str,
    constructor_call: ast.Call,
    fitting_call: ast.Call,
    parents: Mapping[ast.AST, ast.AST],
    required_method: str,
) -> tuple[str, ast.Call, tuple[str, ...]]:
    constructor_statement = _top_level_statement(constructor_call, parents)
    fitting_statement = _top_level_statement(fitting_call, parents)
    between = False
    matches: list[tuple[str, ast.Call, tuple[str, ...]]] = []
    for statement in body:
        if statement is constructor_statement:
            between = True
            continue
        if statement is fitting_statement:
            break
        if not between or not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        value = statement.value
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        if (
            not isinstance(value, ast.Call)
            or value.args
            or value.keywords
            or len(targets) != 1
            or not isinstance(targets[0], ast.Name)
        ):
            continue
        path = _call_path(value)
        if path == (selected_name, required_method):
            matches.append((targets[0].id, value, path))
    if len(matches) != 1:
        raise GraphCallableCoverageError(
            "graph_constructor_fitting_copy_unsupported",
            "version one requires one no-argument "
            f"{required_method}() of the exact typed constructor graph for fitting",
        )
    return matches[0]


def _constructor_return_partition(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    selector: Mapping[str, Any],
) -> tuple[ast.AST, list[ast.AST]]:
    returns = [
        node
        for node in _function_scope_nodes(function)
        if isinstance(node, ast.Return)
    ]
    if (
        len(returns) != 1
        or not function.body
        or function.body[-1] is not returns[0]
        or returns[0].value is None
    ):
        raise GraphCallableCoverageError(
            "graph_constructor_return_flow_unsupported",
            "the exact constructor requires one unconditional final return",
        )
    value = returns[0].value
    kind = selector.get("kind")
    if kind == "return_value":
        return value, []
    if kind == "tuple_item" and isinstance(value, (ast.Tuple, ast.List)):
        index = selector.get("index")
        if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(value.elts):
            return value.elts[index], [
                element
                for observed_index, element in enumerate(value.elts)
                if observed_index != index
            ]
    if kind == "mapping_item" and isinstance(value, ast.Dict):
        key = selector.get("key")
        literal_keys = [
            observed_key.value
            for observed_key in value.keys
            if isinstance(observed_key, ast.Constant)
            and isinstance(observed_key.value, str)
        ]
        if (
            len(literal_keys) == len(value.keys)
            and len(literal_keys) == len(set(literal_keys))
            and literal_keys.count(key) == 1
        ):
            selected_index = literal_keys.index(key)
            return value.values[selected_index], [
                observed
                for observed_index, observed in enumerate(value.values)
                if observed_index != selected_index
            ]
    raise GraphCallableCoverageError(
        "graph_constructor_output_type_unsupported",
        "the constructor return does not expose the selected graph in the closed "
        "typed-array grammar",
    )


def _constructor_selected_expression(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    selector: Mapping[str, Any],
) -> ast.AST:
    selected, _ = _constructor_return_partition(function, selector)
    return selected


def _array_namespace_bindings(tree: ast.Module) -> dict[str, tuple[str, int]]:
    observed: dict[str, list[tuple[str, int]]] = {"numpy": [], "torch": []}
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name in observed:
                    observed[alias.name].append(
                        (alias.asname or alias.name, statement.lineno)
                    )
        if (
            isinstance(statement, ast.ImportFrom)
            and statement.module in observed
            and any(alias.name == "__dict__" for alias in statement.names)
        ):
            raise GraphCallableCoverageError(
                "graph_constructor_output_type_unsupported",
                "numeric module mutable state cannot be imported into the graph "
                "constructor module",
            )
    if any(len(origins) > 1 for origins in observed.values()):
        raise GraphCallableCoverageError(
            "graph_constructor_output_type_unsupported",
            "version one requires one exact module import for each numeric array "
            "family; aliases and reimports cannot establish stable provenance",
        )
    roots = [namespace for origins in observed.values() for namespace, _ in origins]
    if len(roots) != len(set(roots)):
        raise GraphCallableCoverageError(
            "graph_constructor_output_type_unsupported",
            "NumPy and torch cannot share one module binding in the typed-array "
            "provenance grammar",
        )
    return {
        namespace: (family, line)
        for family, origins in observed.items()
        for namespace, line in origins
    }


def _supported_numeric_call_path(
    path: tuple[str, ...],
    namespace_bindings: Mapping[str, tuple[str, int]],
) -> bool:
    if not path or path[0] not in namespace_bindings:
        return False
    family = namespace_bindings[path[0]][0]
    direct_terminals = {
        "numpy": {
            "array", "asarray", "column_stack", "concatenate", "empty",
            "fill_diagonal", "float16", "float32", "float64", "full",
            "hstack", "maximum", "nonzero", "ones", "stack", "vstack",
            "where", "zeros",
        },
        "torch": {
            "as_tensor", "cat", "clamp", "diag", "diagonal", "empty",
            "eye", "full", "nonzero", "norm", "ones", "stack", "tensor",
            "where", "zeros",
        },
    }
    if len(path) == 2 and path[-1] in direct_terminals[family]:
        return True
    return len(path) == 3 and path[1:] == ("linalg", "norm")


def _contains_numeric_namespace_value(
    node: ast.AST | None,
    namespaces: set[str],
) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Name):
        return node.id in namespaces
    if isinstance(node, ast.Attribute):
        path = _expression_path(node)
        if not path or path[0] not in namespaces:
            return False
        return path[-1] not in {
            "bool_",
            "complex64",
            "complex128",
            "float16",
            "float32",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
        }
    if isinstance(node, ast.Call):
        return any(
            _contains_numeric_namespace_value(value, namespaces)
            for value in (
                *node.args,
                *(keyword.value for keyword in node.keywords),
            )
        )
    return any(
        _contains_numeric_namespace_value(child, namespaces)
        for child in ast.iter_child_nodes(node)
    )


def _reject_numeric_namespace_impersonation(
    module_tree: ast.Module,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    namespace_bindings: Mapping[str, tuple[str, int]],
) -> None:
    namespaces = set(namespace_bindings)
    if not namespaces:
        return
    parameter_names = {
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
            *([function.args.vararg] if function.args.vararg is not None else []),
            *([function.args.kwarg] if function.args.kwarg is not None else []),
        )
    }
    if parameter_names & namespaces:
        raise GraphCallableCoverageError(
            "graph_constructor_output_type_unsupported",
            "constructor parameters cannot shadow exact numeric array namespaces",
        )
    families = {family for family, _ in namespace_bindings.values()}
    for node in _protected_execution_nodes(module_tree, function):
        if isinstance(node, ast.Import) and any(
            alias.name in families for alias in node.names
        ):
            raise GraphCallableCoverageError(
                "graph_constructor_output_type_unsupported",
                "numeric array namespaces cannot be reimported inside executable "
                "graph-constructor scope",
            )
        if (
            isinstance(node, ast.ImportFrom)
            and node.module in families
            and any(alias.name == "__dict__" for alias in node.names)
        ):
            raise GraphCallableCoverageError(
                "graph_constructor_output_type_unsupported",
                "numeric module mutable state cannot be imported into the graph "
                "constructor scope",
            )
        direct_paths: list[tuple[str, ...]] = []
        value: ast.AST | None = None
        if (
            isinstance(node, (ast.Attribute, ast.Subscript))
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and _value_root_name(node) in namespaces
        ):
            raise GraphCallableCoverageError(
                "graph_constructor_output_type_unsupported",
                "the exact numeric array namespace is mutated through a "
                "descendant target",
            )
        if isinstance(node, (ast.stmt, ast.ExceptHandler)):
            direct_paths.extend(_module_statement_bound_paths(node))
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            direct_paths.extend(_bound_paths(node))
            targets = (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            direct_paths.extend(
                (root, "<descendant>")
                for target in targets
                if (root := _value_root_name(target)) is not None
                and not isinstance(target, ast.Name)
            )
            value = node.value
        elif isinstance(node, ast.NamedExpr):
            direct_paths.extend(_target_bound_paths(node.target))
            value = node.value
        elif isinstance(node, ast.Delete):
            direct_paths.extend(_bound_paths(node))
        elif isinstance(node, ast.Return):
            value = node.value
        if any(path and path[0] in namespaces for path in direct_paths):
            raise GraphCallableCoverageError(
                "graph_constructor_output_type_unsupported",
                "the exact numeric array namespace is rebound or mutated after import",
            )
        if value is not None and _contains_numeric_namespace_value(
            value, namespaces
        ):
            raise GraphCallableCoverageError(
                "graph_constructor_output_type_unsupported",
                "numeric array namespace objects cannot be aliased or escaped",
            )
        if isinstance(node, ast.Call) and any(
            _contains_numeric_namespace_value(value, namespaces)
            for value in (
                *node.args,
                *(keyword.value for keyword in node.keywords),
            )
        ):
            raise GraphCallableCoverageError(
                "graph_constructor_output_type_unsupported",
                "numeric array namespace objects cannot enter opaque callables",
            )
        if isinstance(node, ast.Call):
            call_path = _call_path(node)
            if call_path and call_path[0] in namespace_bindings:
                if not _supported_numeric_call_path(
                    call_path, namespace_bindings
                ):
                    raise GraphCallableCoverageError(
                        "graph_constructor_output_type_unsupported",
                        "numeric namespace receiver calls outside the typed-array "
                        "grammar can mutate or impersonate array producers",
                    )


def _constructor_graph_array_kind(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    module_tree: ast.Module,
    selector: Mapping[str, Any],
    *,
    array_parameters: frozenset[str] = frozenset(),
    scalar_parameters: frozenset[str] = frozenset(),
) -> str:
    namespace_bindings = _array_namespace_bindings(module_tree)
    _reject_numeric_namespace_impersonation(
        module_tree, function, namespace_bindings
    )
    env: dict[str, str | None] = {
        **{name: "array" for name in array_parameters},
        **{name: "scalar" for name in scalar_parameters},
    }
    fresh_env: dict[str, bool] = {
        **{name: False for name in array_parameters},
        **{name: True for name in scalar_parameters},
    }
    origin_env: dict[str, frozenset[tuple[str, str]]] = {
        **{
            name: frozenset({("input", name)})
            for name in array_parameters
        },
        **{name: frozenset() for name in scalar_parameters},
    }
    escaped_origins: set[tuple[str, str]] = set()
    def merge(kinds: Sequence[str | None]) -> str | None:
        if any(kind is None for kind in kinds):
            return None
        concrete = {kind for kind in kinds if kind in {"numpy", "torch"}}
        if len(concrete) > 1:
            return None
        if concrete:
            return next(iter(concrete))
        if "array" in kinds:
            return "array"
        return "scalar" if kinds else None

    def resolve(node: ast.AST | None) -> str | None:
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return env.get(node.id)
        if isinstance(node, ast.Constant):
            return "scalar" if isinstance(node.value, (bool, int, float, complex)) else None
        if isinstance(node, (ast.List, ast.Tuple)):
            element_kinds = [resolve(element) for element in node.elts]
            if element_kinds and all(kind == "numpy" for kind in element_kinds):
                return "numpy_sequence"
            if element_kinds and all(kind == "torch" for kind in element_kinds):
                return "torch_sequence"
            if len(element_kinds) == 1 and element_kinds[0] in {
                "numpy_sequence", "torch_sequence"
            }:
                return element_kinds[0]
            return None
        if isinstance(node, ast.Subscript):
            return None
        if isinstance(node, ast.Attribute):
            path = _expression_path(node)
            if (
                path
                and path[0] in namespace_bindings
                and path[-1] in {
                    "bool_", "complex64", "complex128", "float16", "float32",
                    "float64", "int8", "int16", "int32", "int64", "uint8",
                }
            ):
                return "scalar"
            return resolve(node.value) if node.attr == "T" else None
        if isinstance(node, ast.UnaryOp):
            return resolve(node.operand)
        if isinstance(node, ast.BinOp):
            return merge((resolve(node.left), resolve(node.right)))
        if isinstance(node, ast.Compare):
            return merge(
                tuple(resolve(value) for value in (node.left, *node.comparators))
            )
        if isinstance(node, ast.BoolOp):
            return None
        if isinstance(node, ast.IfExp):
            body_kind = resolve(node.body)
            return body_kind if body_kind == resolve(node.orelse) else None
        if not isinstance(node, ast.Call):
            return None
        path = _call_path(node)
        if path and path[0] in namespace_bindings:
            if not _supported_numeric_call_path(path, namespace_bindings):
                return None
            family = namespace_bindings[path[0]][0]
            producers = {
                "numpy": {
                    "array", "asarray", "empty", "full", "ones", "zeros",
                },
                "torch": {
                    "as_tensor", "empty", "eye", "full", "ones", "tensor", "zeros",
                },
            }
            preserving = {
                "numpy": {
                    "column_stack", "concatenate", "hstack", "maximum",
                    "norm", "stack", "vstack",
                },
                "torch": {
                    "cat", "clamp", "diag", "diagonal", "nonzero", "norm", "stack",
                },
            }
            if len(path) >= 2 and path[-1] in producers[family]:
                return family
            if len(path) >= 2 and path[-1] == "nonzero" and family == "numpy":
                return "numpy_sequence" if len(node.args) == 1 and not node.keywords else None
            if len(path) >= 2 and path[-1] == "nonzero" and family == "torch":
                as_tuple = next(
                    (
                        keyword.value
                        for keyword in node.keywords
                        if keyword.arg == "as_tuple"
                    ),
                    None,
                )
                if (
                    len(node.args) != 1
                    or any(keyword.arg != "as_tuple" for keyword in node.keywords)
                    or (
                        as_tuple is not None
                        and not (
                            isinstance(as_tuple, ast.Constant)
                            and as_tuple.value is False
                        )
                    )
                ):
                    return None
                return family
            if len(path) >= 2 and path[-1] == "where":
                if len(node.args) == 1 and not node.keywords:
                    return family + "_sequence"
                if len(node.args) != 3 or node.keywords:
                    return None
                argument_kinds = [resolve(value) for value in node.args]
                if all(
                    kind in {family, "array", "scalar"}
                    for kind in argument_kinds
                ) and any(kind in {family, "array"} for kind in argument_kinds):
                    return family
                return None
            if (
                len(path) >= 2
                and path[-1] == "norm"
                and family == "numpy"
                and not any(keyword.arg == "axis" for keyword in node.keywords)
            ):
                return None
            if len(path) >= 2 and path[-1] in preserving[family]:
                argument_kinds = [
                    resolve(value)
                    for value in (
                        *node.args,
                        *(keyword.value for keyword in node.keywords),
                    )
                    if not isinstance(value, ast.Constant)
                ]
                if argument_kinds and all(
                    kind in {family, family + "_sequence", "array", "scalar"}
                    for kind in argument_kinds
                ) and any(
                    kind in {family, family + "_sequence", "array"}
                    for kind in argument_kinds
                ):
                    return family
        if isinstance(node.func, ast.Attribute):
            receiver_kind = resolve(node.func.value)
            allowed_methods = {
                "numpy": {"astype", "copy"},
                "torch": {
                    "clamp", "clone", "contiguous", "float", "long", "max",
                    "nonzero", "norm", "sum", "t", "to", "transpose",
                    "unsqueeze",
                },
            }
            if (
                receiver_kind in allowed_methods
                and node.func.attr in allowed_methods[receiver_kind]
            ):
                if (
                    receiver_kind == "torch"
                    and node.func.attr == "max"
                    and (
                        node.keywords
                        or len(node.args) > 1
                        or (node.args and resolve(node.args[0]) != "torch")
                    )
                ):
                    return None
                if receiver_kind == "torch" and node.func.attr == "nonzero":
                    as_tuple = next(
                        (
                            keyword.value
                            for keyword in node.keywords
                            if keyword.arg == "as_tuple"
                        ),
                        None,
                    )
                    if (
                        node.args
                        or any(keyword.arg != "as_tuple" for keyword in node.keywords)
                        or (
                            as_tuple is not None
                            and not (
                                isinstance(as_tuple, ast.Constant)
                                and as_tuple.value is False
                            )
                        )
                    ):
                        return None
                argument_kinds = [
                    resolve(value)
                    for value in (
                        *node.args,
                        *(keyword.value for keyword in node.keywords),
                    )
                    if not isinstance(value, ast.Constant)
                ]
                if all(
                    kind in {receiver_kind, "scalar"}
                    for kind in argument_kinds
                ):
                    return receiver_kind
        return None

    def literal_bool_keyword(call: ast.Call, name: str) -> bool | None:
        matches = [keyword.value for keyword in call.keywords if keyword.arg == name]
        if not matches:
            return None
        if len(matches) != 1 or not isinstance(matches[0], ast.Constant):
            return False
        return matches[0].value if isinstance(matches[0].value, bool) else False

    def is_fresh(node: ast.AST | None) -> bool:
        """Whether an expression cannot share array storage with an input."""

        if node is None:
            return False
        if isinstance(node, ast.Name):
            return fresh_env.get(node.id, False)
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, (ast.List, ast.Tuple)):
            return all(is_fresh(element) for element in node.elts)
        if isinstance(node, ast.Attribute):
            return node.attr == "T" and is_fresh(node.value)
        if isinstance(node, ast.UnaryOp):
            return is_fresh(node.operand)
        if isinstance(node, (ast.BinOp, ast.Compare)):
            return resolve(node) in {"numpy", "torch"}
        if isinstance(node, ast.IfExp):
            return is_fresh(node.body) and is_fresh(node.orelse)
        if not isinstance(node, ast.Call):
            return False
        if any(
            keyword.arg is None or keyword.arg in {"like", "out"}
            for keyword in node.keywords
        ):
            return False
        path = _call_path(node)
        if path and path[0] in namespace_bindings:
            if not _supported_numeric_call_path(path, namespace_bindings):
                return False
            family = namespace_bindings[path[0]][0]
            terminal = path[-1]
            if terminal in {"asarray", "as_tensor"}:
                return len(node.args) == 1 and is_fresh(node.args[0])
            if family == "numpy" and terminal == "array":
                copy_value = literal_bool_keyword(node, "copy")
                if copy_value is False:
                    return bool(node.args) and is_fresh(node.args[0])
                return copy_value is not False
            if (family, terminal) in {
                ("numpy", "empty"),
                ("numpy", "full"),
                ("numpy", "ones"),
                ("numpy", "zeros"),
                ("torch", "empty"),
                ("torch", "eye"),
                ("torch", "full"),
                ("torch", "ones"),
                ("torch", "tensor"),
                ("torch", "zeros"),
            }:
                return True
            if family == "torch" and terminal == "diagonal":
                return bool(node.args) and is_fresh(node.args[0])
            if family == "torch" and terminal == "norm" and len(node.args) > 4:
                return False
            if family == "numpy" and (
                (terminal == "maximum" and len(node.args) != 2)
                or (
                    terminal in {"concatenate", "stack"}
                    and not 1 <= len(node.args) <= 2
                )
                or (
                    terminal in {"column_stack", "hstack", "vstack"}
                    and len(node.args) != 1
                )
            ):
                return False
            if resolve(node) in {
                "numpy",
                "torch",
                "numpy_sequence",
                "torch_sequence",
            }:
                return True
            return False
        if not isinstance(node.func, ast.Attribute):
            return False
        receiver = node.func.value
        receiver_kind = resolve(receiver)
        method = node.func.attr
        if method in {"copy", "clone"} and receiver_kind in {"numpy", "torch"}:
            return not node.args and not node.keywords
        if method == "astype" and receiver_kind == "numpy":
            if len(node.args) > 1:
                return is_fresh(receiver)
            copy_value = literal_bool_keyword(node, "copy")
            if copy_value is False:
                return is_fresh(receiver)
            return copy_value is not False
        if method in {
            "contiguous",
            "float",
            "long",
            "t",
            "to",
            "transpose",
            "unsqueeze",
        } and receiver_kind == "torch":
            return is_fresh(receiver)
        if method in {"clamp", "max", "nonzero", "norm", "sum"} \
                and receiver_kind == "torch" and resolve(node) == "torch":
            return True
        return False

    def fresh_token(node: ast.AST) -> tuple[str, str]:
        return (
            "fresh",
            f"{getattr(node, 'lineno', 0)}:{getattr(node, 'col_offset', 0)}",
        )

    def origins(node: ast.AST | None) -> frozenset[tuple[str, str]]:
        """Storage generations reachable through an expression or container."""

        if node is None or isinstance(node, ast.Constant):
            return frozenset()
        if isinstance(node, ast.Name):
            return origin_env.get(node.id, frozenset())
        if isinstance(node, ast.IfExp):
            return origins(node.body) | origins(node.orelse)
        if isinstance(node, ast.UnaryOp):
            return origins(node.operand)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return frozenset().union(*(origins(element) for element in node.elts))
        if isinstance(node, ast.Dict):
            return frozenset().union(
                *(origins(value) for value in (*node.keys, *node.values))
            )
        if isinstance(node, ast.Attribute) and node.attr == "T":
            return origins(node.value)
        if isinstance(node, ast.Call):
            path = _call_path(node)
            if path and _supported_numeric_call_path(path, namespace_bindings):
                family = namespace_bindings[path[0]][0]
                terminal = path[-1]
                if terminal in {"asarray", "as_tensor"}:
                    return origins(node.args[0]) if len(node.args) == 1 else frozenset()
                if family == "numpy" and terminal == "array" \
                        and literal_bool_keyword(node, "copy") is False:
                    return origins(node.args[0]) if node.args else frozenset()
                if family == "torch" and terminal == "diagonal":
                    return origins(node.args[0]) if node.args else frozenset()
            if isinstance(node.func, ast.Attribute):
                receiver = node.func.value
                receiver_kind = resolve(receiver)
                method = node.func.attr
                if method == "astype" and receiver_kind == "numpy":
                    if len(node.args) > 1 \
                            or literal_bool_keyword(node, "copy") is False:
                        return origins(receiver)
                if method in {
                    "contiguous",
                    "float",
                    "long",
                    "t",
                    "to",
                    "transpose",
                    "unsqueeze",
                } and receiver_kind == "torch":
                    return origins(receiver)
            if is_fresh(node):
                return frozenset({fresh_token(node)})
            values: tuple[ast.AST, ...] = (
                *(
                    (node.func.value,)
                    if isinstance(node.func, ast.Attribute)
                    else ()
                ),
                *node.args,
                *(keyword.value for keyword in node.keywords),
            )
            return frozenset().union(*(origins(value) for value in values))
        if is_fresh(node):
            return frozenset({fresh_token(node)})
        return frozenset().union(
            *(origins(child) for child in ast.iter_child_nodes(node))
        )

    def closed_numeric_call(call: ast.Call) -> bool:
        path = _call_path(call)
        if path and path[0] in namespace_bindings:
            return True
        if not isinstance(call.func, ast.Attribute):
            return False
        receiver_kind = resolve(call.func.value)
        return receiver_kind in {"numpy", "torch"} and call.func.attr in {
            "astype",
            "clamp",
            "clone",
            "contiguous",
            "copy",
            "fill_",
            "fill_diagonal_",
            "float",
            "long",
            "max",
            "nonzero",
            "norm",
            "sum",
            "t",
            "to",
            "transpose",
            "unsqueeze",
            "zero_",
        }

    if any(
        isinstance(node, (ast.Global, ast.Nonlocal))
        for node in _function_scope_nodes(function)
    ):
        raise GraphCallableCoverageError(
            "graph_constructor_output_escape_unsupported",
            "global and nonlocal graph-constructor bindings are outside the "
            "closed storage-isolation grammar",
        )

    for statement in function.body:
        for call in (
            node for node in ast.walk(statement) if isinstance(node, ast.Call)
        ):
            if closed_numeric_call(call):
                continue
            values: tuple[ast.AST, ...] = (
                *(
                    (call.func.value,)
                    if isinstance(call.func, ast.Attribute)
                    else ()
                ),
                *call.args,
                *(keyword.value for keyword in call.keywords),
            )
            escaped_origins.update(
                frozenset().union(*(origins(value) for value in values))
            )
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            value_kind = resolve(statement.value)
            value_fresh = is_fresh(statement.value)
            value_origins = origins(statement.value)
            for target in targets:
                if isinstance(target, ast.Name):
                    env[target.id] = value_kind
                    fresh_env[target.id] = value_fresh
                    origin_env[target.id] = value_origins
                elif (
                    isinstance(target, (ast.Tuple, ast.List))
                    and value_kind in {"numpy_sequence", "torch_sequence"}
                    and all(isinstance(element, ast.Name) for element in target.elts)
                ):
                    element_kind = value_kind.removesuffix("_sequence")
                    for element in target.elts:
                        env[element.id] = element_kind
                        fresh_env[element.id] = value_fresh
                        origin_env[element.id] = value_origins
                else:
                    escaped_origins.update(value_origins)
                    for path in _target_bound_paths(target):
                        if path:
                            env[path[0]] = None
                            fresh_env[path[0]] = False
                            origin_env[path[0]] = frozenset()
            continue
        if isinstance(statement, ast.AugAssign):
            if isinstance(statement.target, (ast.Attribute, ast.Subscript)):
                escaped_origins.update(origins(statement.value))
            for path in _target_bound_paths(statement.target):
                if path:
                    env[path[0]] = None
                    fresh_env[path[0]] = False
                    origin_env[path[0]] = frozenset()
            continue
        if isinstance(statement, ast.Return):
            continue
        if isinstance(statement, _CONTROL_NODES):
            escaped_origins.update(
                frozenset().union(
                    *(
                        origins(node)
                        for node in ast.walk(statement)
                        if isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Load)
                    )
                )
            )
        for path in _module_statement_bound_paths(statement):
            if path:
                env[path[0]] = None
                fresh_env[path[0]] = False
                origin_env[path[0]] = frozenset()

    selected_expression = _constructor_selected_expression(function, selector)
    kind = resolve(selected_expression)
    if kind not in {"numpy", "torch"}:
        raise GraphCallableCoverageError(
            "graph_constructor_output_type_unsupported",
            "the selected graph must be one exact NumPy array or torch Tensor "
            "construction before its fitting snapshot can be certified",
        )
    if not is_fresh(selected_expression):
        raise GraphCallableCoverageError(
            "graph_constructor_output_not_fresh",
            "the selected graph may share storage with a constructor input; "
            "version one requires an allocating operation or exact copy/clone",
        )
    selected_origins = origins(selected_expression)
    if len(selected_origins) != 1 or any(
        origin_kind != "fresh" for origin_kind, _ in selected_origins
    ):
        raise GraphCallableCoverageError(
            "graph_constructor_output_not_fresh",
            "the selected graph does not resolve to one isolated fresh-storage "
            "generation",
        )
    _, unselected_expressions = _constructor_return_partition(function, selector)
    unselected_origins = frozenset().union(
        *(origins(expression) for expression in unselected_expressions)
    )
    if selected_origins & unselected_origins:
        raise GraphCallableCoverageError(
            "graph_constructor_output_alias_unsupported",
            "the selected graph also appears in an unselected constructor return "
            "leaf or metadata container",
        )
    if selected_origins & escaped_origins:
        raise GraphCallableCoverageError(
            "graph_constructor_output_escape_unsupported",
            "the selected graph escapes through an opaque call, control flow, or "
            "nonlocal storage target",
        )
    return kind


def _architecture_model_binding(
    tree: ast.Module,
    *,
    arch_contract: Mapping[str, Any],
    method_dir: Path,
    fitting_call: ast.Call,
    parents: Mapping[ast.AST, ast.AST],
) -> tuple[str, ast.Call, str, tuple[str, ...], int]:
    relational = _mapping(
        arch_contract.get("relational_indexing"), label="relational indexing"
    )
    execution = _mapping(relational.get("execution"), label="relational execution")
    inference = _mapping(execution.get("inference"), label="relational inference")
    block_name = _exact_identifier(
        inference.get("architecture_block"), label="architecture block"
    )
    architecture = _mapping(arch_contract.get("architecture"), label="architecture")
    block = _mapping(architecture.get(block_name), label="architecture block")
    class_name = _exact_identifier(block.get("class_name"), label="architecture class")
    class_path, imported_at = _exact_import_path(
        tree,
        module_name="method.model",
        qualname=class_name,
        method_dir=method_dir,
        label="graph_architecture_model",
    )
    calls = _exact_top_level_calls(
        tree,
        call_path=class_path,
        imported_at=imported_at,
        parents=parents,
        label="graph_architecture_model",
    )
    if len(calls) != 1 or calls[0].lineno >= fitting_call.lineno:
        raise GraphCallableCoverageError(
            "graph_architecture_model_construction_unsupported",
            "version one requires exactly one direct top-level construction of "
            "the declared architecture class before fitting",
        )
    return (
        _assigned_call_name(calls[0], parents, label="graph_architecture_model"),
        calls[0],
        class_name,
        class_path,
        imported_at,
    )


_IDENTITY_PRESERVING_MODEL_METHODS = frozenset({
    "buffers",
    "cpu",
    "cuda",
    "eval",
    "forward",
    "load_state_dict",
    "named_buffers",
    "named_parameters",
    "parameters",
    "requires_grad_",
    "state_dict",
    "to",
    "train",
    "zero_grad",
})
_MODEL_IDENTITY_RETURNING_METHODS = frozenset({
    "cpu",
    "cuda",
    "eval",
    "requires_grad_",
    "to",
    "train",
})
_DISPATCH_SENSITIVE_MODEL_ATTRIBUTES = frozenset({
    "__call__",
    "__class__",
    "__delattr__",
    "__dict__",
    "__getattr__",
    "__getattribute__",
    "__setattr__",
    "forward",
})
_FRAMEWORK_DISPATCH_METHODS = (
    _IDENTITY_PRESERVING_MODEL_METHODS - {"forward"}
) | frozenset({
    "__call__",
    "__delattr__",
    "_apply",
    "_call_impl",
    "_slow_forward",
    "_wrapped_call_impl",
    "register_forward_hook",
    "register_forward_pre_hook",
    "__setattr__",
})
_FRAMEWORK_DISPATCH_EXTENSION_POINTS = frozenset({
    "_load_from_state_dict",
    "_named_members",
    "_save_to_state_dict",
    "children",
    "get_extra_state",
    "modules",
    "named_children",
    "named_modules",
    "set_extra_state",
})
_FRAMEWORK_DISPATCH_METHODS = (
    _FRAMEWORK_DISPATCH_METHODS | _FRAMEWORK_DISPATCH_EXTENSION_POINTS
)
_FORWARD_HOOK_APIS = frozenset({
    "register_forward_hook",
    "register_forward_pre_hook",
    "register_module_forward_hook",
    "register_module_forward_pre_hook",
})
_FRAMEWORK_DISPATCH_INTROSPECTION = frozenset({
    "__bases__",
    "__class__",
    "__dict__",
    "__getattribute__",
    "__mro__",
    "__subclasses__",
    "mro",
})
_FRAMEWORK_DISPATCH_DESCRIPTOR_ACCESS = frozenset({
    "__delete__",
    "__delattr__",
    "__get__",
    "__set__",
    "__setattr__",
})


def _identity_preserving_model_value(
    value: ast.AST | None,
    aliases: set[str],
) -> bool:
    if isinstance(value, ast.Name):
        return value.id in aliases
    if not isinstance(value, ast.Call):
        return False
    path = _expression_path(value.func)
    return (
        len(path) == 2
        and path[0] in aliases
        and path[1] in _MODEL_IDENTITY_RETURNING_METHODS
    )


def _function_scope_nodes(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.AST]:
    nodes: list[ast.AST] = []
    pending = list(reversed(function.body))
    while pending:
        node = pending.pop()
        nodes.append(node)
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        pending.extend(reversed(list(ast.iter_child_nodes(node))))
    return nodes


def _reject_forward_hook_access(
    module_tree: ast.Module,
    nodes: Sequence[ast.AST],
    *,
    code: str,
) -> None:
    imported_bindings = {
        alias.asname or alias.name
        for statement in module_tree.body
        if isinstance(statement, ast.ImportFrom)
        and statement.module in {
            "torch.nn.modules.module",
            "torch.nn.modules",
            "torch.nn",
        }
        for alias in statement.names
        if alias.name in _FORWARD_HOOK_APIS
    }
    if any(
        (
            isinstance(node, ast.Attribute)
            and node.attr in _FORWARD_HOOK_APIS
        )
        or (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in imported_bindings
        )
        for node in nodes
    ):
        raise GraphCallableCoverageError(
            code,
            "forward-hook registration or aliasing can replace the analyzed "
            "model input or output",
        )


def _framework_dispatch_bindings(
    tree: ast.Module,
    nodes: Sequence[ast.AST],
) -> tuple[set[str], set[str], bool]:
    """Collect torch namespace and hook aliases from every executable import."""

    bindings: set[str] = set()
    hook_bindings: set[str] = set()
    star_dispatch_import = False
    imports = {
        id(node): node
        for node in (*tree.body, *nodes)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    }.values()
    for statement in imports:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name == "torch" or alias.name.startswith("torch."):
                    bound = alias.asname or alias.name.split(".", 1)[0]
                    bindings.add(bound)
                    if "hook" in alias.name.lower():
                        hook_bindings.add(bound)
            continue
        module = statement.module or ""
        if module != "torch" and not module.startswith("torch."):
            continue
        for alias in statement.names:
            if alias.name == "*":
                if module == "torch.nn" or module.startswith("torch.nn."):
                    star_dispatch_import = True
                continue
            bound = alias.asname or alias.name
            if alias.name in {"Module", "module", "modules", "nn"}:
                bindings.add(bound)
            if "hook" in alias.name.lower():
                hook_bindings.add(bound)
    return bindings, hook_bindings, star_dispatch_import


def _contains_framework_namespace_value(
    node: ast.AST | None,
    bindings: set[str],
) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Name):
        return node.id in bindings
    if isinstance(node, ast.Attribute):
        path = _expression_path(node)
        return bool(path and path[0] in bindings)
    if isinstance(node, ast.Call):
        return any(
            _contains_framework_namespace_value(value, bindings)
            for value in (
                *node.args,
                *(keyword.value for keyword in node.keywords),
            )
        )
    return any(
        _contains_framework_namespace_value(child, bindings)
        for child in ast.iter_child_nodes(node)
    )


def _reject_framework_dispatch_impersonation(
    tree: ast.Module,
    nodes: Sequence[ast.AST],
    *,
    code: str,
) -> None:
    if any(
        (
            isinstance(node, ast.Attribute)
            and node.attr in (
                _FRAMEWORK_DISPATCH_INTROSPECTION
                | _FRAMEWORK_DISPATCH_DESCRIPTOR_ACCESS
            )
        )
        or (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == "type"
        )
        for node in nodes
    ):
        raise GraphCallableCoverageError(
            code,
            "generic class introspection or descriptor access can recover or "
            "replace protected numeric or model dispatch",
        )
    bindings, hook_bindings, star_dispatch_import = (
        _framework_dispatch_bindings(tree, nodes)
    )
    if star_dispatch_import:
        raise GraphCallableCoverageError(
            code,
            "star imports from torch model-dispatch modules are outside the "
            "closed graph liveness grammar",
        )
    if not bindings and not hook_bindings:
        return
    protected_values = bindings | hook_bindings
    for node in nodes:
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, (ast.Load, ast.Store, ast.Del))
            and node.id in hook_bindings
        ):
            raise GraphCallableCoverageError(
                code,
                "framework hook state or registration aliases can replace the "
                "analyzed model input or output",
            )
        if (
            isinstance(node, (ast.Attribute, ast.Subscript))
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and _value_root_name(node) in bindings
        ):
            raise GraphCallableCoverageError(
                code,
                "framework Module dispatch or hook state is mutated through an "
                "imported torch namespace",
            )
        if isinstance(node, ast.Attribute):
            path = _expression_path(node)
            if (
                path
                and path[0] in bindings
                and (
                    any("hook" in component for component in path[1:])
                    or any(
                        component in _FRAMEWORK_DISPATCH_INTROSPECTION
                        for component in path[1:]
                    )
                )
            ):
                raise GraphCallableCoverageError(
                    code,
                    "framework hook state or class-dispatch introspection is "
                    "outside the closed model dispatch grammar",
                )
        value: ast.AST | None = None
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            value = node.value
        elif isinstance(node, ast.NamedExpr):
            value = node.value
        elif isinstance(node, ast.Return):
            value = node.value
        if value is not None and _contains_framework_namespace_value(
            value, protected_values
        ):
            raise GraphCallableCoverageError(
                code,
                "framework namespace objects cannot be aliased or escaped from "
                "the closed dispatch grammar",
            )
        if isinstance(node, ast.Call) and any(
            _contains_framework_namespace_value(value, protected_values)
            for value in (
                *node.args,
                *(keyword.value for keyword in node.keywords),
            )
        ):
            raise GraphCallableCoverageError(
                code,
                "framework namespace or hook state cannot enter an opaque "
                "callable",
            )


def _module_init_nodes(module_tree: ast.Module) -> list[ast.AST]:
    """Return executable import-time nodes under the closed module grammar."""

    nodes: list[ast.AST] = []
    allowed_bases = {
        (),
        ("object",),
        ("nn", "Module"),
        ("torch", "nn", "Module"),
    }
    module_functions = {
        node.name
        for node in module_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for statement in module_tree.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _reject_executable_function_header(statement, module_tree)
            continue
        if isinstance(statement, ast.ClassDef):
            if (
                statement.decorator_list
                or statement.keywords
                or getattr(statement, "type_params", ())
                or any(
                    _expression_path(base) not in allowed_bases
                    for base in statement.bases
                )
            ):
                raise GraphCallableCoverageError(
                    "graph_live_module_init_unsupported",
                    "class decorators, metaclass keywords, generics, or unknown "
                    "bases can execute outside the closed module grammar",
                )
            for class_statement in statement.body:
                if isinstance(
                    class_statement, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    _reject_executable_function_header(
                        class_statement, module_tree
                    )
                    continue
                if isinstance(class_statement, ast.Pass) or (
                    isinstance(class_statement, ast.Expr)
                    and isinstance(class_statement.value, ast.Constant)
                    and isinstance(class_statement.value.value, str)
                ):
                    continue
                if _is_literal_assignment(class_statement):
                    nodes.extend(ast.walk(class_statement))
                    continue
                raise GraphCallableCoverageError(
                    "graph_live_module_init_unsupported",
                    "descriptor construction and nonliteral class-body execution "
                    "are outside the closed graph liveness module grammar",
                )
            continue
        if isinstance(statement, ast.Pass) or (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        if _is_literal_assignment(statement) or (
            isinstance(statement, (ast.Assign, ast.AnnAssign))
            and isinstance(statement.value, ast.Lambda)
        ):
            nodes.extend(ast.walk(statement))
            continue
        calls = [
            node for node in ast.walk(statement) if isinstance(node, ast.Call)
        ]
        if (
            isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr))
            and calls
            and all(
                len(_call_path(call)) == 1
                and _call_path(call)[0] in module_functions
                for call in calls
            )
        ):
            nodes.extend(ast.walk(statement))
            continue
        raise GraphCallableCoverageError(
            "graph_live_module_init_unsupported",
            "import-time object construction or control flow is outside the "
            "closed graph liveness module grammar",
        )
    return nodes


_CORE_METHOD_MODULES = frozenset({
    "method",
    "method.method",
    "method.model",
    "method.training",
})
_CORE_METHOD_LEAVES = frozenset({"method", "model", "training"})


def _is_local_method_import(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(
            alias.name == "method" or alias.name.startswith("method.")
            for alias in node.names
        )
    if not isinstance(node, ast.ImportFrom):
        return False
    return node.level > 0 or node.module == "method" or (
        isinstance(node.module, str) and node.module.startswith("method.")
    )


def _top_level_method_import_is_core(node: ast.Import | ast.ImportFrom) -> bool:
    if isinstance(node, ast.Import):
        return all(
            alias.name in _CORE_METHOD_MODULES
            for alias in node.names
            if alias.name == "method" or alias.name.startswith("method.")
        )
    if node.level > 0:
        if node.module is None:
            return all(alias.name in _CORE_METHOD_LEAVES for alias in node.names)
        return node.module in _CORE_METHOD_LEAVES
    return node.module in _CORE_METHOD_MODULES - {"method"}


def _imports_producer_local_module(node: ast.AST, method_dir: Path | None) -> bool:
    if method_dir is None or not isinstance(node, (ast.Import, ast.ImportFrom)):
        return False
    module_names: list[str] = []
    if isinstance(node, ast.Import):
        module_names.extend(alias.name for alias in node.names)
    elif node.level == 0 and node.module:
        module_names.append(node.module)
        module_names.extend(
            f"{node.module}.{alias.name}"
            for alias in node.names
            if alias.name != "*"
        )
    roots = (Path(method_dir).parent, Path(method_dir))
    for module_name in module_names:
        parts = module_name.split(".")
        head = parts[0]
        if head == "method":
            continue
        if any(
            root.joinpath(*parts).with_suffix(".py").is_file()
            or root.joinpath(*parts).is_dir()
            for root in roots
        ):
            return True
    return False


def _reject_local_method_imports(
    module_tree: ast.Module,
    execution_nodes: Sequence[ast.AST],
    *,
    code: str,
    method_dir: Path | None = None,
) -> None:
    if any(
        isinstance(statement, (ast.Import, ast.ImportFrom))
        and (
            (
                _is_local_method_import(statement)
                and not _top_level_method_import_is_core(statement)
            )
            or _imports_producer_local_module(statement, method_dir)
        )
        for statement in module_tree.body
    ):
        raise GraphCallableCoverageError(
            code,
            "imports from producer-controlled run-local modules or method-package "
            "siblings are outside "
            "the closed graph liveness module grammar",
        )
    if any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            _is_local_method_import(node)
            or _imports_producer_local_module(node, method_dir)
        )
        for node in execution_nodes
    ):
        raise GraphCallableCoverageError(
            code,
            "lazy run-local or method-package imports can execute unproved sibling "
            "side effects",
        )


def _reject_notebook_method_imports(
    tree: ast.Module,
    *,
    allowed_names_by_module: Mapping[str, frozenset[str]],
    method_dir: Path,
) -> None:
    parents = _parents(tree)
    protected_names: set[str] = set()
    module_bindings: list[tuple[tuple[str, ...], str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and (
            _is_local_method_import(node)
            or _imports_producer_local_module(node, method_dir)
        ) and not isinstance(parents.get(node), ast.Module):
            raise GraphCallableCoverageError(
                "graph_constructor_notebook_local_import_unsupported",
                "nested producer-controlled imports execute outside the closed "
                "notebook grammar",
            )
        if _imports_producer_local_module(node, method_dir):
            raise GraphCallableCoverageError(
                "graph_constructor_notebook_local_import_unsupported",
                "the notebook imports an unproved producer-controlled run-local "
                "module",
            )
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != "method" and not alias.name.startswith("method."):
                    continue
                if alias.name not in allowed_names_by_module:
                    raise GraphCallableCoverageError(
                        "graph_constructor_notebook_local_import_unsupported",
                        "the notebook imports an unproved method-package sibling",
                    )
                binding = (
                    (alias.asname,)
                    if alias.asname
                    else tuple(alias.name.split("."))
                )
                module_bindings.append((binding, alias.name))
        elif isinstance(node, ast.ImportFrom) and _is_local_method_import(node):
            if node.level > 0 or node.module not in allowed_names_by_module:
                raise GraphCallableCoverageError(
                    "graph_constructor_notebook_local_import_unsupported",
                    "the notebook imports an unproved method-package sibling",
                )
            allowed_names = allowed_names_by_module[node.module]
            if any(alias.name not in allowed_names for alias in node.names):
                raise GraphCallableCoverageError(
                    "graph_constructor_notebook_local_import_unsupported",
                    "the notebook import is outside the exact constructor, fitting, "
                    "architecture, and pluggable routes",
                )
            protected_names.update(alias.asname or alias.name for alias in node.names)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in protected_names
        ):
            parent = parents.get(node)
            if not (isinstance(parent, ast.Call) and parent.func is node):
                raise GraphCallableCoverageError(
                    "graph_constructor_notebook_route_escape_unsupported",
                    "a protected imported callable or class escapes its exact "
                    "top-level call site",
                )
        if not isinstance(node, ast.Attribute) or not isinstance(node.ctx, ast.Load):
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.Attribute) and parent.value is node:
            continue
        path = _expression_path(node)
        candidates = [
            (binding, module)
            for binding, module in module_bindings
            if len(path) >= len(binding) and path[: len(binding)] == binding
        ]
        if not candidates:
            continue
        binding, module = max(candidates, key=lambda item: len(item[0]))
        relative = path[len(binding):]
        if (
            len(relative) != 1
            or relative[0] not in allowed_names_by_module[module]
            or not (isinstance(parent, ast.Call) and parent.func is node)
        ):
            raise GraphCallableCoverageError(
                "graph_constructor_notebook_route_escape_unsupported",
                "a core method module exposes an unproved member or protected "
                "route outside its exact top-level call site",
            )


def _module_init_reachable_functions(
    module_tree: ast.Module,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    definitions = {
        node.name: node
        for node in module_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    pending = [
        definitions[path[0]]
        for node in _module_init_nodes(module_tree)
        if isinstance(node, ast.Call)
        for path in [_call_path(node)]
        if len(path) == 1 and path[0] in definitions
    ]
    reachable: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        _reject_deferred_callable(current)
        reachable.append(current)
        for node in _function_scope_nodes(current):
            if not isinstance(node, ast.Call):
                continue
            path = _call_path(node)
            if len(path) == 1 and path[0] in definitions:
                pending.append(definitions[path[0]])
    return reachable


def _reachable_module_functions(
    module_tree: ast.Module,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    definitions = {
        node.name: node
        for node in module_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    reachable: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    pending = [
        function,
        *_module_init_reachable_functions(module_tree),
    ]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        _reject_deferred_callable(current)
        reachable.append(current)
        for node in _function_scope_nodes(current):
            if not isinstance(node, ast.Call):
                continue
            path = _call_path(node)
            if len(path) == 1 and path[0] in definitions:
                pending.append(definitions[path[0]])
    return reachable


def _protected_runtime_bindings(
    module_tree: ast.Module,
    protected_qualnames: set[str],
    *,
    function: ast.FunctionDef | ast.AsyncFunctionDef | None = None,
    locally_owned_qualnames: frozenset[str] = frozenset(),
) -> set[str]:
    bindings: set[str] = set(locally_owned_qualnames)
    protected_modules = {"method", "method.model", "method.method"}
    statements: list[ast.AST] = list(module_tree.body)
    if function is not None:
        for reachable in _reachable_module_functions(module_tree, function):
            statements.extend(_function_scope_nodes(reachable))
    for statement in statements:
        if isinstance(statement, ast.ImportFrom):
            absolute = statement.module or ""
            relative_owner = statement.level == 1 and absolute in {"model", "method"}
            absolute_owner = statement.level == 0 and absolute in protected_modules
            if relative_owner or absolute_owner:
                for alias in statement.names:
                    if alias.name == "*":
                        bindings.update(protected_qualnames)
                    elif alias.name in {"model", "method"} | protected_qualnames:
                        bindings.add(alias.asname or alias.name)
            if statement.level == 1 and statement.module is None:
                for alias in statement.names:
                    if alias.name in {"model", "method"} | protected_qualnames:
                        bindings.add(alias.asname or alias.name)
        elif isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name in protected_modules:
                    bindings.add(alias.asname or alias.name.split(".", 1)[0])
    return bindings


def _protected_execution_nodes(
    module_tree: ast.Module,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.AST]:
    nodes: list[ast.AST] = []
    for reachable in _reachable_module_functions(module_tree, function):
        nodes.extend(_function_scope_nodes(reachable))
    nodes.extend(_module_init_nodes(module_tree))
    return nodes


def _reject_producer_architecture_class_references(
    module_tree: ast.Module,
    *,
    class_name: str,
    code: str,
) -> None:
    """Keep every producer-defined callable outside protected class authority.

    Framework methods such as ``Module.train`` recursively invoke locally
    registered child modules.  Auditing only the architecture class therefore
    misses a child override that mutates or escapes the parent architecture
    class when inherited traversal reaches it.  The v1 proof grammar has no
    need for producer helpers or class methods to hold that class as data, so
    reject the reference at every producer-defined callable boundary.
    """

    callables: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for statement in module_tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            callables.append(statement)
        elif isinstance(statement, ast.ClassDef) and statement.name != class_name:
            callables.extend(
                child
                for child in statement.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
    locally_owned = frozenset({class_name})
    for function in callables:
        try:
            _reject_dynamic_namespace(
                function,
                label="graph_producer_architecture",
                module_tree=module_tree,
            )
        except GraphCallableCoverageError as exc:
            raise GraphCallableCoverageError(
                code,
                "a producer-defined helper or class method uses reflective "
                "class recovery outside the closed model-dispatch grammar",
            ) from exc
        protected_bindings = _protected_runtime_bindings(
            module_tree,
            {class_name},
            locally_owned_qualnames=locally_owned,
        )
        nodes = _function_scope_nodes(function)
        for node in nodes:
            if isinstance(node, ast.ImportFrom):
                absolute = node.module or ""
                if (
                    (node.level == 1 and absolute in {"model", "method"})
                    or (node.level == 1 and node.module is None)
                    or (
                        node.level == 0
                        and absolute in {"method", "method.model", "method.method"}
                    )
                ):
                    protected_bindings.update(
                        alias.asname or alias.name
                        for alias in node.names
                        if alias.name in {class_name, "model", "method"}
                    )
            elif isinstance(node, ast.Import):
                protected_bindings.update(
                    alias.asname or alias.name.split(".", 1)[0]
                    for alias in node.names
                    if alias.name in {"method", "method.model", "method.method"}
                )
        if any(
            isinstance(node, ast.Name) and node.id in protected_bindings
            for node in nodes
        ):
            raise GraphCallableCoverageError(
                code,
                "a producer-defined helper or class method reaches the protected "
                "architecture class outside the analyzed instance path",
            )


def _reject_cross_callable_references(
    module_tree: ast.Module,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    protected_qualnames: set[str],
    locally_owned_qualnames: frozenset[str],
    label: str,
    method_dir: Path,
) -> None:
    _reject_dynamic_namespace(function, label=label, module_tree=module_tree)
    for module_function in (
        node
        for node in module_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        _reject_executable_function_header(module_function, module_tree)
    protected_bindings = _protected_runtime_bindings(
        module_tree,
        protected_qualnames,
        function=function,
        locally_owned_qualnames=locally_owned_qualnames,
    )
    execution_nodes = _protected_execution_nodes(module_tree, function)
    _reject_local_method_imports(
        module_tree,
        execution_nodes,
        code=f"{label}_cross_callable_effect_unsupported",
        method_dir=method_dir,
    )
    _reject_framework_dispatch_impersonation(
        module_tree,
        execution_nodes,
        code=f"{label}_cross_callable_effect_unsupported",
    )
    _reject_numeric_runtime_binding_mutation(
        module_tree,
        execution_nodes,
        code=f"{label}_cross_callable_effect_unsupported",
    )
    _reject_forward_hook_access(
        module_tree,
        execution_nodes,
        code=f"{label}_cross_callable_effect_unsupported",
    )
    for node in execution_nodes:
        if isinstance(node, ast.Name) and node.id in protected_bindings:
            raise GraphCallableCoverageError(
                f"{label}_cross_callable_effect_unsupported",
                "the callable reaches another protected graph-path binding",
            )
        direct_paths: list[tuple[str, ...]] = []
        if isinstance(node, ast.stmt):
            direct_paths.extend(_bound_paths(node))
        if isinstance(node, ast.NamedExpr):
            direct_paths.extend(_target_bound_paths(node.target))
        if any(path and path[0] in protected_bindings for path in direct_paths):
            raise GraphCallableCoverageError(
                f"{label}_cross_callable_effect_unsupported",
                "the callable mutates another protected graph-path binding",
            )
    module_functions = {
        node.name
        for node in module_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    parents = _parents(module_tree)
    for node in execution_nodes:
        if not isinstance(node, ast.Name) or node.id not in module_functions:
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.Call) and parent.func is node:
            continue
        raise GraphCallableCoverageError(
            f"{label}_callable_alias_unsupported",
            "module-local callable aliases are outside the closed graph "
            "liveness call graph",
        )
    for class_node in (
        node for node in module_tree.body if isinstance(node, ast.ClassDef)
    ):
        if class_node.name in locally_owned_qualnames:
            continue
        for statement in class_node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(statement):
                if isinstance(node, ast.Name) and node.id in protected_bindings:
                    raise GraphCallableCoverageError(
                        f"{label}_cross_callable_effect_unsupported",
                        "executable class-body code reaches a protected graph-path "
                        "binding",
                    )


def _locally_owned_protected_qualnames(
    module_tree: ast.Module,
    protected_qualnames: set[str],
) -> frozenset[str]:
    return frozenset(
        node.name
        for node in module_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name in protected_qualnames
    )


def _prove_training_model_identity(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    model_parameter: str,
    module_tree: ast.Module,
    protected_qualnames: set[str],
    model_callable_attributes: set[str],
    model_data_attributes: set[str],
    allow_direct_model_call: bool,
    method_dir: Path,
) -> None:
    parameters = {
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    }
    if model_parameter not in parameters:
        raise GraphCallableProducerError(
            "graph_fitting_model_parameter_missing",
            "the exact fitting function omits its declared architecture-model input",
        )
    if function.decorator_list:
        raise GraphCallableCoverageError(
            "graph_fitting_model_identity_grammar_unsupported",
            "decorated fitting dispatch is outside graph liveness v1",
        )
    _reject_dynamic_namespace(
        function,
        label="graph_fitting_model",
        module_tree=module_tree,
    )
    protected_bindings = _protected_runtime_bindings(
        module_tree, protected_qualnames, function=function
    )
    execution_nodes = _protected_execution_nodes(module_tree, function)
    _reject_local_method_imports(
        module_tree,
        execution_nodes,
        code="graph_fitting_model_identity_grammar_unsupported",
        method_dir=method_dir,
    )
    _reject_framework_dispatch_impersonation(
        module_tree,
        execution_nodes,
        code="graph_fitting_model_identity_grammar_unsupported",
    )
    _reject_numeric_runtime_binding_mutation(
        module_tree,
        execution_nodes,
        code="graph_fitting_model_identity_grammar_unsupported",
    )
    _reject_forward_hook_access(
        module_tree,
        execution_nodes,
        code="graph_fitting_model_identity_grammar_unsupported",
    )
    for node in execution_nodes:
        if isinstance(node, ast.Name) and node.id in protected_bindings:
            raise GraphCallableCoverageError(
                "graph_fitting_protected_callable_mutation_unsupported",
                "the fitting function reaches another graph-path callable or "
                "architecture binding",
            )
        direct_paths: list[tuple[str, ...]] = []
        if isinstance(node, ast.stmt):
            direct_paths.extend(_bound_paths(node))
        if isinstance(node, ast.NamedExpr):
            direct_paths.extend(_target_bound_paths(node.target))
        if any(path and path[0] in protected_bindings for path in direct_paths):
            raise GraphCallableCoverageError(
                "graph_fitting_protected_callable_mutation_unsupported",
                "the fitting function mutates another protected graph-path binding",
            )

    aliases = {model_parameter}
    changed = True
    while changed:
        changed = False
        for statement in _function_scope_nodes(function):
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            if (
                len(targets) == 1
                and isinstance(targets[0], ast.Name)
                and _identity_preserving_model_value(statement.value, aliases)
                and targets[0].id not in aliases
            ):
                aliases.add(targets[0].id)
                changed = True

    for statement in _function_scope_nodes(function):
        direct_paths: list[tuple[str, ...]] = []
        if isinstance(statement, ast.stmt):
            direct_paths.extend(_bound_paths(statement))
        if isinstance(statement, ast.NamedExpr):
            direct_paths.extend(_target_bound_paths(statement.target))
        if isinstance(statement, ast.ExceptHandler) and isinstance(
            statement.name, str
        ):
            direct_paths.append((statement.name,))
        if isinstance(statement, (ast.MatchAs, ast.MatchStar)) and isinstance(
            statement.name, str
        ):
            direct_paths.append((statement.name,))
        if isinstance(statement, ast.MatchMapping) and isinstance(
            statement.rest, str
        ):
            direct_paths.append((statement.rest,))
        bound_roots = {path[0] for path in direct_paths if path}
        protected_bound = bound_roots & aliases
        if not protected_bound:
            continue
        allowed_targets: set[str] = set()
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            if _identity_preserving_model_value(statement.value, aliases):
                allowed_targets = {
                    target.id for target in targets if isinstance(target, ast.Name)
                }
        if protected_bound - allowed_targets:
            raise GraphCallableProducerError(
                "graph_fitting_model_identity_not_preserved",
                "the fitting function rebinds or mutates the declared model identity",
            )

    parents = _parents(function)
    for node in _function_scope_nodes(function):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            continue
        if node.id not in aliases:
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.Return) and parent.value is node:
            continue
        if isinstance(parent, (ast.Assign, ast.AnnAssign)) and parent.value is node:
            continue
        current: ast.AST = node
        while isinstance(parents.get(current), ast.Attribute) \
                and parents[current].value is current:
            current = parents[current]
        call = parents.get(current)
        if isinstance(call, ast.Call) and call.func is current:
            path = _expression_path(current)
            if (path == (node.id,) and allow_direct_model_call) or (
                len(path) == 2
                and path[0] == node.id
                and path[1] in _IDENTITY_PRESERVING_MODEL_METHODS
            ):
                continue
        path = _expression_path(current)
        if (
            len(path) >= 2
            and path[0] == node.id
            and not set(path[1:]) & _DISPATCH_SENSITIVE_MODEL_ATTRIBUTES
            and path[1] not in model_callable_attributes
            and path[1] in model_data_attributes
            and isinstance(current, ast.Attribute)
            and isinstance(current.ctx, ast.Load)
            and not (isinstance(call, ast.Call) and call.func is current)
        ):
            continue
        raise GraphCallableCoverageError(
            "graph_fitting_model_identity_grammar_unsupported",
            "the fitting model escapes the supported identity-preserving call and "
            "return grammar",
        )

    returns = [
        node
        for node in _function_scope_nodes(function)
        if isinstance(node, ast.Return)
    ]
    if not returns or any(
        not isinstance(node.value, ast.Name) or node.value.id not in aliases
        for node in returns
    ):
        raise GraphCallableProducerError(
            "graph_fitting_model_identity_not_preserved",
            "every fitting return must preserve the exact architecture-model input",
        )


def _guard_training_model_method_overrides(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    model_parameter: str,
    model_tree: ast.Module,
    class_name: str,
) -> None:
    """Reject class overrides of framework methods trusted by fitting v1."""

    classes = [
        node
        for node in model_tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        raise GraphCallableProducerError(
            "graph_live_architecture_class_missing",
            f"method/model.py does not define exactly one {class_name!r}",
        )
    _reject_architecture_init_instance_escape(
        classes[0],
        code="graph_fitting_model_method_override_unsupported",
    )
    _reject_producer_architecture_class_references(
        model_tree,
        class_name=class_name,
        code="graph_fitting_model_method_override_unsupported",
    )
    aliases = {model_parameter}
    changed = True
    while changed:
        changed = False
        for statement in _function_scope_nodes(function):
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            if (
                len(targets) == 1
                and isinstance(targets[0], ast.Name)
                and _identity_preserving_model_value(statement.value, aliases)
                and targets[0].id not in aliases
            ):
                aliases.add(targets[0].id)
                changed = True
    trusted_calls = {
        path[1]
        for node in _function_scope_nodes(function)
        if isinstance(node, ast.Call)
        for path in [_call_path(node)]
        if len(path) == 2
        and path[0] in aliases
        and path[1] in _IDENTITY_PRESERVING_MODEL_METHODS
        and path[1] != "forward"
    }
    overridden = {
        method
        for statement in classes[0].body
        for method in _FRAMEWORK_DISPATCH_METHODS
        if (method,) in _module_statement_bound_paths(statement)
    }
    instance_overrides = {
        node.attr
        for class_method in classes[0].body
        if isinstance(class_method, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in _function_scope_nodes(class_method)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and isinstance(node.ctx, (ast.Store, ast.Del))
        and node.attr in _FRAMEWORK_DISPATCH_METHODS
    }
    if _FRAMEWORK_DISPATCH_METHODS:
        init_method = next(
            (
                node
                for node in classes[0].body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "__init__"
            ),
            None,
        )
        if init_method is not None and _has_dynamic_namespace(init_method):
            instance_overrides.update(_FRAMEWORK_DISPATCH_METHODS)
    overridden.update(instance_overrides)
    if overridden:
        raise GraphCallableCoverageError(
            "graph_fitting_model_method_override_unsupported",
            "the architecture class overrides framework model methods whose "
            "identity-preserving behavior is trusted during fitting: "
            f"{sorted(overridden)!r}",
        )


def _reject_architecture_init_instance_escape(
    class_node: ast.ClassDef,
    *,
    code: str,
) -> None:
    """Prevent inherited framework traversal from reaching the parent instance."""

    init_method = next(
        (
            statement
            for statement in class_node.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and statement.name == "__init__"
        ),
        None,
    )
    if init_method is None:
        return
    for node in _function_scope_nodes(init_method):
        if isinstance(node, ast.Call):
            path = _call_path(node)
            values = (*node.args, *(keyword.value for keyword in node.keywords))
            if (path and path[0] == "self") or any(
                "self" in _expr_names(value) for value in values
            ):
                raise GraphCallableCoverageError(
                    code,
                    "model initialization cannot pass the live architecture "
                    "instance into producer-controlled callables",
                )
        value: ast.AST | None = None
        targets: Sequence[ast.AST] = ()
        if isinstance(node, ast.Assign):
            value = node.value
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            value = node.value
            targets = (node.target,)
        elif isinstance(node, ast.NamedExpr):
            value = node.value
            targets = (node.target,)
        elif isinstance(node, ast.Return):
            value = node.value
        if value is None or "self" not in _expr_names(value):
            continue
        direct_instance_attributes = bool(targets) and all(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            for target in targets
        )
        if not direct_instance_attributes:
            raise GraphCallableCoverageError(
                code,
                "model initialization cannot store the live architecture "
                "instance into producer-controlled aliases or child state",
            )


def prove_constructor_notebook_flow(
    code_cells: Sequence[str],
    method_spec: Mapping[str, Any],
    arch_contract: Mapping[str, Any],
    method_dir: Path,
) -> dict[str, Any]:
    """Prove the selected constructor output feeds fitting and inference."""

    mechanism = _mechanism(method_spec)
    if mechanism is None:
        return {"status": "not_applicable", "reason": "graph_mechanism_absent"}
    _require_declarative_package_init(Path(method_dir))
    construction = _mapping(mechanism.get("construction"), label="construction")
    identity = _callable_identity(
        construction.get("callable"), label="construction.callable"
    )
    selector = _mapping(
        construction.get("output_selector"), label="construction output selector"
    )
    bindings = _relational_bindings(arch_contract, mechanism)
    tree = _stitched_notebook(code_cells)
    _reject_dynamic_namespace(tree, label="graph_constructor")
    _reject_numeric_runtime_binding_mutation(
        tree,
        list(ast.walk(tree)),
        code="graph_constructor_dynamic_namespace_unsupported",
    )
    _reject_framework_dispatch_impersonation(
        tree,
        list(ast.walk(tree)),
        code="graph_constructor_dynamic_namespace_unsupported",
    )
    if any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))
        for node in ast.walk(tree)
    ):
        raise GraphCallableCoverageError(
            "graph_constructor_notebook_local_callable_unsupported",
            "notebook-local callables can mutate protected graph dispatch outside "
            "the closed liveness grammar",
        )
    parents = _parents(tree)
    constructor_tree = _parse_file(
        _module_file(Path(method_dir), identity["module"]),
        code="graph_constructor_source_unavailable",
    )
    try:
        constructor_function = _function_def(
            constructor_tree, identity["qualname"]
        )
    except GraphCallableProducerError as exc:
        raise GraphCallableProducerError(
            "graph_constructor_route_disagreement",
            "the declared constructor is not one exact top-level function in "
            f"{identity['module']}",
        ) from exc
    _require_final_module_binding(
        constructor_tree,
        route=(identity["qualname"],),
        origin_line=constructor_function.lineno,
        label="graph_constructor_callable",
        reject_descendant_bindings=True,
        reject_opaque_escape=True,
    )
    constructor_array_parameters = frozenset({
        _exact_identifier(
            construction.get("feature_parameter"),
            label="construction feature parameter",
        )
    })
    constructor_scalar_parameters = {
        _exact_identifier(
            _mapping(
                _mapping(construction.get("threshold"), label="construction threshold").get(
                    "parameter"
                ),
                label="construction threshold parameter",
            ).get("callable_parameter"),
            label="construction threshold callable parameter",
        )
    }
    construction_cap = _mapping(construction.get("cap"), label="construction cap")
    if construction_cap.get("kind") != "none":
        constructor_scalar_parameters.add(
            _exact_identifier(
                _mapping(
                    construction_cap.get("parameter"),
                    label="construction cap parameter",
                ).get("callable_parameter"),
                label="construction cap callable parameter",
            )
        )
    constructor_graph_kind = _constructor_graph_array_kind(
        constructor_function,
        constructor_tree,
        selector,
        array_parameters=constructor_array_parameters,
        scalar_parameters=frozenset(constructor_scalar_parameters),
    )
    constructor_path, constructor_imported_at = _exact_import_path(
        tree,
        module_name=identity["module"],
        qualname=identity["qualname"],
        method_dir=Path(method_dir),
        label="graph_constructor",
    )
    calls = _exact_top_level_calls(
        tree,
        call_path=constructor_path,
        imported_at=constructor_imported_at,
        parents=parents,
        label="graph_constructor",
    )
    if len(calls) != 1:
        raise GraphCallableCoverageError(
            "graph_constructor_call_missing"
            if not calls
            else "graph_constructor_call_grammar_unsupported",
            "version one requires exactly one top-level call to the exact graph constructor",
        )
    _verify_cfg_authority(tree, calls[0])
    live_bindings = _constructor_live_bindings(
        calls[0],
        construction,
        feature_binding=bindings["construction_feature_parameter"],
    )
    selected_name = _assigned_name_for_selector(tree, calls[0], selector, parents)

    training = _mapping(arch_contract.get("training_loop"), label="training loop")
    training_name = _exact_identifier(
        training.get("function_name"), label="training loop function name"
    )
    training_tree = _parse_file(
        Path(method_dir) / "training.py", code="graph_fitting_source_unavailable"
    )
    training_function = _function_def(training_tree, training_name)
    _require_final_module_binding(
        training_tree,
        route=(training_name,),
        origin_line=training_function.lineno,
        label="graph_fitting_callable",
        reject_descendant_bindings=True,
        reject_opaque_escape=True,
    )
    message_contract = _mapping(
        mechanism.get("message_passing"), label="message passing"
    )
    message_identity = _callable_identity(
        message_contract.get("callable"), label="message_passing.callable"
    )
    pluggable_contract = _mapping(
        arch_contract.get("pluggable_component"), label="pluggable component"
    )
    pluggable_name = _exact_identifier(
        pluggable_contract.get("name"), label="pluggable component name"
    )
    architecture = _mapping(
        arch_contract.get("architecture"), label="architecture"
    )
    architecture_block = _mapping(
        architecture.get(bindings["architecture_block"]),
        label="architecture block",
    )
    architecture_class_name = _exact_identifier(
        architecture_block.get("class_name"), label="architecture class"
    )
    constructor_protected = {
        message_identity["qualname"],
        training_name,
        pluggable_name,
        architecture_class_name,
    }
    _reject_notebook_method_imports(
        tree,
        allowed_names_by_module={
            "method": frozenset({
                identity["qualname"],
                training_name,
                pluggable_name,
                architecture_class_name,
            }),
            "method.model": frozenset(
                {architecture_class_name}
                | (
                    {identity["qualname"]}
                    if identity["module"] == "method.model"
                    else set()
                )
            ),
            "method.training": frozenset(
                {training_name}
                | (
                    {identity["qualname"]}
                    if identity["module"] == "method.training"
                    else set()
                )
            ),
            "method.method": frozenset(
                {pluggable_name}
                | (
                    {identity["qualname"]}
                    if identity["module"] == "method.method"
                    else set()
                )
            ),
        },
        method_dir=Path(method_dir),
    )
    _reject_cross_callable_references(
        constructor_tree,
        constructor_function,
        protected_qualnames=constructor_protected,
        locally_owned_qualnames=_locally_owned_protected_qualnames(
            constructor_tree, constructor_protected
        ),
        label="graph_constructor_source",
        method_dir=Path(method_dir),
    )
    model_tree = _parse_file(
        Path(method_dir) / "model.py", code="graph_model_source_unavailable"
    )
    model_classes = [
        node
        for node in model_tree.body
        if isinstance(node, ast.ClassDef) and node.name == architecture_class_name
    ]
    if len(model_classes) != 1:
        raise GraphCallableProducerError(
            "graph_live_architecture_class_missing",
            "method/model.py does not define exactly one "
            f"{architecture_class_name!r}",
        )
    model_callable_attributes = set(_IDENTITY_PRESERVING_MODEL_METHODS)
    model_callable_attributes.update(
        node.name
        for node in model_classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    model_data_attributes: set[str] = set()
    init_method = next(
        (
            node
            for node in model_classes[0].body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "__init__"
        ),
        None,
    )
    if init_method is not None:
        literal_data_attributes: dict[str, bool] = {}
        for statement in _function_scope_nodes(init_method):
            if not isinstance(
                statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)
            ):
                continue
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            direct_attributes = {
                target.attr
                for target in targets
                if isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr not in model_callable_attributes
            }
            if not direct_attributes:
                continue
            value = statement.value
            literal = value is not None
            if literal:
                try:
                    ast.literal_eval(value)
                except Exception:  # noqa: BLE001 - closed inert-data grammar
                    literal = False
            for attribute in direct_attributes:
                literal_data_attributes[attribute] = (
                    literal_data_attributes.get(attribute, True) and literal
                )
        model_data_attributes.update(
            attribute
            for attribute, literal in literal_data_attributes.items()
            if literal
        )
    _prove_training_model_identity(
        training_function,
        model_parameter=bindings["fitting_model_parameter"],
        module_tree=training_tree,
        protected_qualnames={
            identity["qualname"],
            message_identity["qualname"],
            pluggable_name,
            architecture_class_name,
        },
        model_callable_attributes=model_callable_attributes,
        model_data_attributes=model_data_attributes,
        allow_direct_model_call=_has_exact_torch_module_base(
            model_tree, model_classes[0]
        ) and not any(
            ("__call__",) in _module_statement_bound_paths(statement)
            for statement in model_classes[0].body
        ),
        method_dir=Path(method_dir),
    )
    _guard_training_model_method_overrides(
        training_function,
        model_parameter=bindings["fitting_model_parameter"],
        model_tree=model_tree,
        class_name=architecture_class_name,
    )
    training_path, training_imported_at = _exact_import_path(
        tree,
        module_name="method.training",
        qualname=training_name,
        method_dir=Path(method_dir),
        label="graph_fitting",
    )
    fitting_calls = _exact_top_level_calls(
        tree,
        call_path=training_path,
        imported_at=training_imported_at,
        parents=parents,
        label="graph_fitting",
    )
    if len(fitting_calls) != 1:
        raise GraphCallableCoverageError(
            "graph_constructor_fitting_call_unsupported",
            "version one proves one exact top-level fitting call",
        )
    (
        architecture_model_name,
        architecture_model_call,
        architecture_class,
        architecture_class_path,
        architecture_class_imported_at,
    ) = _architecture_model_binding(
            tree,
            arch_contract=arch_contract,
            method_dir=Path(method_dir),
            fitting_call=fitting_calls[0],
            parents=parents,
        )
    pluggable_path, pluggable_imported_at = _exact_import_path(
        tree,
        module_name="method.method",
        qualname=pluggable_name,
        method_dir=Path(method_dir),
        label="graph_inference",
    )
    fitting_copy_name, fitting_copy_call, fitting_copy_path = (
        _fitting_graph_copy_binding(
            tree.body,
            selected_name=selected_name,
            constructor_call=calls[0],
            fitting_call=fitting_calls[0],
            parents=parents,
            required_method="copy" if constructor_graph_kind == "numpy" else "clone",
        )
    )
    selected_aliases_at_copy = _direct_aliases_before_node(
        tree.body,
        seed=selected_name,
        node=fitting_copy_call,
        parents=parents,
        label="graph_constructor_fitting_copy_source",
        after_node=calls[0],
    )
    if selected_name not in selected_aliases_at_copy:
        raise GraphCallableProducerError(
            "graph_constructor_fitting_copy_source_missing",
            "the fitting snapshot receiver is no longer the selected constructor "
            "graph at the exact copy call",
        )
    training_inputs = _mapping(training.get("input"), label="training loop inputs")
    expected_fitting_keywords = {
        _exact_identifier(name, label="training loop input")
        for name in training_inputs
    }
    for call in fitting_calls:
        _exact_keyword_call_values(
            call,
            parameters=expected_fitting_keywords,
            code="graph_constructor_fitting_grammar_unsupported",
            label="the exact fitting call",
        )
    fitting_matches = []
    for call in fitting_calls:
        selected_aliases_at_fitting = _direct_aliases_before_node(
            tree.body,
            seed=selected_name,
            node=call,
            parents=parents,
            label="graph_constructor_fitting_original",
            after_node=calls[0],
            allowed_call_paths=frozenset({fitting_copy_path}),
        )
        if any(
            _expr_names(keyword.value) & selected_aliases_at_fitting
            for keyword in call.keywords
        ):
            raise GraphCallableProducerError(
                "graph_constructor_fitting_original_escape",
                "the mutable inference graph reaches fitting instead of only its "
                "exact snapshot",
            )
        graph_aliases = _direct_aliases_before_node(
            tree.body,
            seed=fitting_copy_name,
            node=call,
            parents=parents,
            label="graph_constructor_fitting",
            after_node=fitting_copy_call,
        )
        graph_matches = _direct_keyword_alias(
            call,
            parameter=bindings["fitting_graph_parameter"],
            aliases=graph_aliases,
            label="graph_constructor_fitting",
        )
        feature_aliases = _direct_aliases_before_node(
            tree.body,
            seed=live_bindings["feature_binding"],
            node=call,
            parents=parents,
            label="graph_constructor_feature",
            after_node=calls[0],
        )
        feature_matches = _direct_keyword_alias(
            call,
            parameter=bindings["construction_feature_parameter"],
            aliases=feature_aliases,
            label="graph_constructor_feature",
        )
        model_aliases = _direct_aliases_before_node(
            tree.body,
            seed=architecture_model_name,
            node=call,
            parents=parents,
            label="graph_architecture_model",
            after_node=architecture_model_call,
        )
        model_matches = _direct_keyword_alias(
            call,
            parameter=bindings["fitting_model_parameter"],
            aliases=model_aliases,
            label="graph_architecture_model",
        )
        protected_routes = {
            constructor_path,
            training_path,
            pluggable_path,
            architecture_class_path,
        }
        if any(
            _expression_contains_route(keyword.value, protected_routes)
            for keyword in call.keywords
        ):
            raise GraphCallableProducerError(
                "graph_constructor_fitting_protected_route_escape",
                "a protected constructor, fitting, model-class, or pluggable "
                "route escapes through a fitting argument",
            )
        designated_aliases = {
            bindings["fitting_graph_parameter"]: graph_aliases,
            bindings["construction_feature_parameter"]: feature_aliases,
            bindings["fitting_model_parameter"]: model_aliases,
        }
        if any(
            _expr_names(keyword.value) & aliases
            for parameter, aliases in designated_aliases.items()
            for keyword in call.keywords
            if keyword.arg != parameter
        ):
            raise GraphCallableProducerError(
                "graph_constructor_fitting_role_escape",
                "a protected graph, feature, or architecture-model value reaches "
                "a fitting parameter outside its declared R2C-084 role",
            )
        if graph_matches and feature_matches and model_matches:
            fitting_matches.append(call)
    if not fitting_matches:
        if any(
            call.args or any(keyword.arg is None for keyword in call.keywords)
            for call in fitting_calls
        ):
            raise GraphCallableCoverageError(
                "graph_constructor_fitting_grammar_unsupported",
                "positional or expanded-keyword fitting bindings are outside "
                "graph constructor liveness v1",
            )
        raise GraphCallableProducerError(
            "graph_constructor_fitting_path_missing",
            "the selected constructor output and its exact feature authority do "
            "not reach the R2C-084 fitting graph/feature parameters "
            f"{bindings['fitting_graph_parameter']!r} and "
            f"{bindings['construction_feature_parameter']!r}",
        )
    if len(fitting_matches) != 1:
        raise GraphCallableCoverageError(
            "graph_constructor_fitting_grammar_unsupported",
            "version one proves one exact top-level fitting call",
        )
    fitting_model_result = _assigned_call_name(
        fitting_matches[0], parents, label="graph_fitting"
    )

    (
        pluggable_name,
        pluggable_model_parameter,
        pluggable_graph_parameter,
        pluggable_neighbor_signal_parameter,
    ) = _pluggable_graph_parameter(
        Path(method_dir),
        arch_contract,
        bindings["inference_graph_parameter"],
        bindings["inference_neighbor_parameter"],
        bindings["fitting_model_parameter"],
        protected_qualnames={
            identity["qualname"],
            message_identity["qualname"],
            training_name,
            architecture_class_name,
        },
    )
    neighbor_signal_root = message_contract.get("neighbor_signal_root")
    if not isinstance(neighbor_signal_root, str) or "." not in neighbor_signal_root:
        raise GraphCallableCoverageError(
            "graph_constructor_neighbor_binding_unsupported",
            "the declared neighbor-signal logical root has no exact notebook "
            "binding leaf",
        )
    neighbor_signal_binding = _exact_identifier(
        neighbor_signal_root.rsplit(".", 1)[-1],
        label="neighbor-signal notebook binding",
    )
    inference_calls = _exact_top_level_calls(
        tree,
        call_path=pluggable_path,
        imported_at=pluggable_imported_at,
        parents=parents,
        label="graph_inference",
    )
    expected_pluggable_keywords = {
        _exact_identifier(name, label="pluggable component input")
        for name in _mapping(
            pluggable_contract.get("input"),
            label="pluggable component inputs",
        )
    }
    for call in inference_calls:
        _exact_keyword_call_values(
            call,
            parameters=expected_pluggable_keywords,
            code="graph_constructor_inference_grammar_unsupported",
            label="the exact pluggable inference call",
        )
    inference_matches = []
    for call in inference_calls:
        if _route_rebound_before_call(
            tree,
            route=architecture_class_path,
            imported_at=architecture_class_imported_at,
            call=call,
        ):
            raise GraphCallableProducerError(
                "graph_architecture_model_binding_shadowed",
                "the declared architecture class binding is mutated before inference",
            )
        _direct_aliases_before_node(
            tree.body,
            seed=architecture_model_name,
            node=call,
            parents=parents,
            label="graph_architecture_model",
            after_node=architecture_model_call,
            allowed_call_paths=frozenset({training_path}),
        )
        graph_aliases = _direct_aliases_before_node(
            tree.body,
            seed=selected_name,
            node=call,
            parents=parents,
            label="graph_constructor_inference",
            after_node=calls[0],
            allowed_call_paths=frozenset({fitting_copy_path}),
        )
        model_aliases = _direct_aliases_before_node(
            tree.body,
            seed=fitting_model_result,
            node=call,
            parents=parents,
            label="graph_constructor_inference_model",
            after_node=fitting_matches[0],
        )
        graph_matches = _direct_keyword_alias(
            call,
            parameter=pluggable_graph_parameter,
            aliases=graph_aliases,
            label="graph_constructor_inference",
        )
        model_matches = _direct_keyword_alias(
            call,
            parameter=pluggable_model_parameter,
            aliases=model_aliases,
            label="graph_constructor_inference_model",
        )
        neighbor_matches = _direct_keyword_alias(
            call,
            parameter=pluggable_neighbor_signal_parameter,
            aliases={neighbor_signal_binding},
            label="graph_constructor_inference_neighbor_signal",
        )
        if graph_matches and model_matches and neighbor_matches:
            inference_matches.append(call)
    if not inference_calls:
        raise GraphCallableCoverageError(
            "graph_constructor_inference_call_unsupported",
            "version one cannot prove an indirect pluggable call",
        )
    if not inference_matches:
        if any(
            call.args or any(keyword.arg is None for keyword in call.keywords)
            for call in inference_calls
        ):
            raise GraphCallableCoverageError(
                "graph_constructor_inference_grammar_unsupported",
                "positional or expanded-keyword inference bindings are outside "
                "graph constructor liveness v1",
            )
        raise GraphCallableProducerError(
            "graph_constructor_inference_path_missing",
            "the selected constructor output and declared neighbor-signal root "
            "do not reach the pluggable inputs that feed "
            f"{bindings['inference_graph_parameter']!r} and "
            f"{bindings['inference_neighbor_parameter']!r}",
        )
    if len(inference_calls) != 1 or len(inference_matches) != 1:
        raise GraphCallableCoverageError(
            "graph_constructor_inference_grammar_unsupported",
            "version one proves one exact top-level inference call",
        )
    inference_result = _assigned_call_name(
        inference_matches[0], parents, label="graph_inference"
    )
    expected_inference_result = bindings["inference_output_root"].rsplit(".", 1)[-1]
    if inference_result != expected_inference_result:
        raise GraphCallableProducerError(
            "graph_constructor_inference_output_sink_mismatch",
            "the exact pluggable result does not bind the declared inference "
            f"output leaf {expected_inference_result!r}",
        )
    inference_statement = _top_level_statement(inference_matches[0], parents)
    sink_aliases = {inference_result}
    after_inference = False
    for statement in tree.body:
        if statement is inference_statement:
            after_inference = True
            continue
        if not after_inference:
            continue
        direct_alias_targets: set[str] = set()
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            if isinstance(statement.value, ast.Name) and statement.value.id in sink_aliases:
                direct_alias_targets = {
                    target.id for target in targets if isinstance(target, ast.Name)
                }
        sink_loads = {
            node.id
            for node in ast.walk(statement)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in sink_aliases
        }
        bare_display = (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Name)
            and statement.value.id in sink_aliases
        )
        if sink_loads and not direct_alias_targets and not bare_display:
            raise GraphCallableCoverageError(
                "graph_constructor_inference_output_sink_escape_unsupported",
                "the declared inference output sink enters a container, attribute, "
                "transform, or opaque alias after the exact pluggable call",
            )
        bound_paths = _module_statement_bound_paths(statement)
        if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            bound_paths.extend(
                (root, "<descendant>")
                for target in targets
                if (root := _value_root_name(target)) is not None
                and not isinstance(target, ast.Name)
            )
        if any(
            path
            and path[0] in sink_aliases
            and path[0] not in direct_alias_targets
            for path in bound_paths
        ):
            raise GraphCallableProducerError(
                "graph_constructor_inference_output_sink_rebound",
                "the declared inference output sink is rebound or mutated after "
                "the exact pluggable call",
            )
        for call in (
            node for node in ast.walk(statement) if isinstance(node, ast.Call)
        ):
            receiver = (
                [call.func.value] if isinstance(call.func, ast.Attribute) else []
            )
            values = (
                *receiver,
                *call.args,
                *(keyword.value for keyword in call.keywords),
            )
            if any(_expr_names(value) & sink_aliases for value in values):
                raise GraphCallableCoverageError(
                    "graph_constructor_inference_output_sink_escape_unsupported",
                    "the declared inference output sink enters an opaque callable "
                    "after the exact pluggable call",
                )
        sink_aliases.update(direct_alias_targets)
    return {
        "status": "pass",
        "reason": "constructor_output_reaches_fitting_and_inference",
        "callable": identity,
        "output_selector": dict(selector),
        "selected_binding": selected_name,
        "fitting_copy_binding": fitting_copy_name,
        "fitting_copy_method": fitting_copy_path[-1],
        "fitting_graph_array_kind": constructor_graph_kind,
        "numeric_operand_dispatch_precondition": (
            NUMERIC_OPERAND_DISPATCH_PRECONDITION
        ),
        **live_bindings,
        "fitting_graph_parameter": bindings["fitting_graph_parameter"],
        "fitting_model_parameter": bindings["fitting_model_parameter"],
        "architecture_class": architecture_class,
        "architecture_model_binding": architecture_model_name,
        "fitting_model_identity_preserved": True,
        "fitting_model_result_binding": fitting_model_result,
        "pluggable_callable": f"method.method:{pluggable_name}",
        "pluggable_model_parameter": pluggable_model_parameter,
        "pluggable_graph_parameter": pluggable_graph_parameter,
        "pluggable_neighbor_signal_parameter": (
            pluggable_neighbor_signal_parameter
        ),
        "inference_graph_parameter": bindings["inference_graph_parameter"],
        "inference_neighbor_signal_parameter": (
            bindings["inference_neighbor_parameter"]
        ),
        "neighbor_signal_root": neighbor_signal_root,
        "neighbor_signal_binding": neighbor_signal_binding,
        "inference_output_binding": inference_result,
        "pluggable_result_consumed": True,
        "phases": ["fitting", "inference"],
    }


_HELPER_TOKEN = "__r2c_graph_helper_output__"
_HELPER_OUTPUT_KEY_PREFIX = "__r2c_graph_helper_output_key__:"
_UNSAFE_DEPENDENCY_PREFIX = "__r2c_unsafe_dependency__:"


def _unsafe_dependencies(values: set[str]) -> set[str]:
    return {
        value
        if value == _HELPER_TOKEN or value.startswith(_UNSAFE_DEPENDENCY_PREFIX)
        else _UNSAFE_DEPENDENCY_PREFIX + value
        for value in values
    }


def _relative_import_path(
    tree: ast.Module, *, owner_module: str, identity: Mapping[str, str]
) -> tuple[tuple[str, ...], int] | None:
    if identity["module"] == owner_module:
        function = _function_def(tree, identity["qualname"])
        return (identity["qualname"],), function.lineno
    expected = identity["module"].removeprefix("method.")
    routes: list[tuple[tuple[str, ...], int]] = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module
        if node.level == 1 and module == expected:
            for alias in node.names:
                if alias.name == identity["qualname"]:
                    routes.append(
                        ((alias.asname or identity["qualname"],), node.lineno)
                    )
        if node.level == 0 and module == identity["module"]:
            for alias in node.names:
                if alias.name == identity["qualname"]:
                    routes.append(
                        ((alias.asname or identity["qualname"],), node.lineno)
                    )
    if len(routes) > 1:
        raise GraphCallableCoverageError(
            "graph_message_live_import_grammar_unsupported",
            "version one requires one exact message-helper binding in method.model",
        )
    return routes[0] if routes else None


def _require_final_module_binding(
    tree: ast.Module,
    *,
    route: tuple[str, ...],
    origin_line: int,
    label: str,
    reject_descendant_bindings: bool = False,
    reject_opaque_escape: bool = False,
) -> None:
    for statement in tree.body:
        if statement.lineno <= origin_line:
            continue
        for bound in _module_statement_bound_paths(statement):
            replaces_route = (
                len(bound) <= len(route) and route[: len(bound)] == bound
            )
            mutates_route = (
                reject_descendant_bindings
                and len(bound) > len(route)
                and bound[: len(route)] == route
            )
            if replaces_route or mutates_route:
                raise GraphCallableProducerError(
                    f"{label}_binding_shadowed",
                    f"the exact {label} binding is shadowed elsewhere in its module",
                )
        if reject_opaque_escape:
            protected_root = route[0]
            if _has_dynamic_namespace(statement):
                raise GraphCallableCoverageError(
                    f"{label}_escape_unsupported",
                    f"dynamic namespace mutation can replace the exact {label}",
                )
            for call in (
                node for node in ast.walk(statement) if isinstance(node, ast.Call)
            ):
                receiver = (
                    [call.func.value]
                    if isinstance(call.func, ast.Attribute)
                    else []
                )
                values = (
                    *receiver,
                    *call.args,
                    *(keyword.value for keyword in call.keywords),
                )
                if any(protected_root in _expr_names(value) for value in values):
                    raise GraphCallableCoverageError(
                        f"{label}_escape_unsupported",
                        f"the exact {label} object enters an opaque module-level "
                        "call after its definition",
                    )


def _bind_call_sources(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    call: ast.Call,
    env: Mapping[str, set[str]],
) -> dict[str, set[str]]:
    parameters = [
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
        if argument.arg != "self"
    ]
    _reject_effectful_call_arguments(
        call,
        code="graph_message_self_call_grammar_unsupported",
        label=f"the self-method call to {function.name!r}",
    )
    keyword_names = [keyword.arg for keyword in call.keywords]
    if (
        any(name is None for name in keyword_names)
        or len(call.args) > len(parameters)
        or len(keyword_names) != len(set(keyword_names))
        or any(name not in parameters for name in keyword_names)
        or any(name in parameters[: len(call.args)] for name in keyword_names)
        or len(call.args) + len(keyword_names) != len(parameters)
    ):
        raise GraphCallableCoverageError(
            "graph_message_self_call_grammar_unsupported",
            f"the self-method call to {function.name!r} must bind each declared "
            "input exactly once",
        )
    bound: dict[str, set[str]] = {parameter: set() for parameter in parameters}
    positional = list(call.args)
    for parameter, value in zip(parameters, positional):
        bound[parameter] = _expression_dependencies(value, env, {}, None, None)[0]
    for keyword in call.keywords:
        if keyword.arg in bound:
            bound[keyword.arg] = _expression_dependencies(
                keyword.value, env, {}, None, None
            )[0]
    return bound


def _expression_dependencies(
    node: ast.AST,
    env: Mapping[str, set[str]],
    methods: Mapping[str, ast.FunctionDef | ast.AsyncFunctionDef],
    helper_path: tuple[str, ...] | None,
    helper_parameters: tuple[str, str] | None,
    *,
    stack: tuple[str, ...] = (),
) -> tuple[set[str], list[dict[str, Any]]]:
    if isinstance(node, ast.Name):
        return set(env.get(node.id, {node.id})), []
    if isinstance(node, ast.Constant):
        return set(), []
    if isinstance(node, ast.Call):
        if helper_path is not None and _call_path(node) == helper_path:
            if helper_parameters is None:
                _reject_effectful_call_arguments(
                    node,
                    code="graph_constructor_inference_grammar_unsupported",
                    label="the exact architecture-model call",
                )
                return {_HELPER_TOKEN}, []
            graph_parameter, signal_parameter = helper_parameters
            by_name = _exact_keyword_call_values(
                node,
                parameters={graph_parameter, signal_parameter},
                code="graph_message_helper_call_grammar_unsupported",
                label="the exact message-helper call",
            )
            graph_deps, _ = _expression_dependencies(
                by_name[graph_parameter], env, methods, helper_path, helper_parameters,
                stack=stack,
            )
            signal_deps, _ = _expression_dependencies(
                by_name[signal_parameter], env, methods, helper_path, helper_parameters,
                stack=stack,
            )
            if any(
                dependency.startswith(_UNSAFE_DEPENDENCY_PREFIX)
                for dependency in graph_deps | signal_deps
            ):
                raise GraphCallableCoverageError(
                    "graph_message_input_flow_unsupported",
                    "the exact graph or neighbor-signal input reaches the helper "
                    "through a non-transparent transform",
                )
            return {_HELPER_TOKEN, *graph_deps, *signal_deps}, [
                {
                    "call_lineno": node.lineno,
                    "graph_dependencies": sorted(graph_deps),
                    "neighbor_dependencies": sorted(signal_deps),
                }
            ]
        path = _call_path(node)
        if len(path) == 2 and path[0] == "self" and path[1] in methods:
            method_name = path[1]
            if method_name in stack:
                raise GraphCallableCoverageError(
                    "graph_message_recursive_grammar_unsupported",
                    "recursive model-method flow is outside graph liveness v1",
                )
            bound = _bind_call_sources(methods[method_name], node, env)
            return _analyse_function_body(
                methods[method_name],
                bound,
                methods,
                helper_path,
                helper_parameters,
                stack=(*stack, method_name),
            )
        dependencies: set[str] = set()
        evidence: list[dict[str, Any]] = []
        for child in (*node.args, *(keyword.value for keyword in node.keywords)):
            child_deps, child_evidence = _expression_dependencies(
                child, env, methods, helper_path, helper_parameters, stack=stack
            )
            dependencies.update(child_deps)
            evidence.extend(child_evidence)
        if _HELPER_TOKEN in dependencies:
            raise GraphCallableCoverageError(
                "graph_message_result_flow_unsupported",
                "the message-helper result enters an opaque callable transform",
            )
        return _unsafe_dependencies(dependencies), evidence
    if isinstance(node, ast.Dict):
        dependencies: set[str] = set()
        evidence: list[dict[str, Any]] = []
        entries: list[tuple[ast.AST | None, set[str]]] = []
        for key, value in zip(node.keys, node.values):
            if key is not None:
                key_deps, key_evidence = _expression_dependencies(
                    key, env, methods, helper_path, helper_parameters, stack=stack
                )
                dependencies.update(
                    dependency
                    for dependency in key_deps
                    if not dependency.startswith(_HELPER_OUTPUT_KEY_PREFIX)
                )
                evidence.extend(key_evidence)
            value_deps, value_evidence = _expression_dependencies(
                value, env, methods, helper_path, helper_parameters, stack=stack
            )
            entries.append((key, value_deps))
            dependencies.update(
                dependency
                for dependency in value_deps
                if not dependency.startswith(_HELPER_OUTPUT_KEY_PREFIX)
            )
            evidence.extend(value_evidence)
        helper_entries = [
            (key, value_deps)
            for key, value_deps in entries
            if _HELPER_TOKEN in value_deps
        ]
        if helper_entries:
            literal_keys = [
                key.value
                for key, _ in entries
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
            helper_key, helper_deps = helper_entries[0]
            if (
                len(helper_entries) != 1
                or len(literal_keys) != len(entries)
                or len(literal_keys) != len(set(literal_keys))
                or not isinstance(helper_key, ast.Constant)
                or not isinstance(helper_key.value, str)
                or any(
                    dependency.startswith(_HELPER_OUTPUT_KEY_PREFIX)
                    for dependency in helper_deps
                )
            ):
                raise GraphCallableCoverageError(
                    "graph_message_result_flow_unsupported",
                    "a certified output mapping requires one direct helper value "
                    "under unique literal string keys with no unpacking or nesting",
                )
            dependencies.add(_HELPER_OUTPUT_KEY_PREFIX + helper_key.value)
        dependencies = {
            dependency
            if dependency == _HELPER_TOKEN
            or dependency.startswith(_HELPER_OUTPUT_KEY_PREFIX)
            else _UNSAFE_DEPENDENCY_PREFIX + dependency
            if not dependency.startswith(_UNSAFE_DEPENDENCY_PREFIX)
            else dependency
            for dependency in dependencies
        }
        return dependencies, evidence
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        children = list(node.elts)
    else:
        children = list(ast.iter_child_nodes(node))
    dependencies: set[str] = set()
    evidence: list[dict[str, Any]] = []
    for child in children:
        child_deps, child_evidence = _expression_dependencies(
            child, env, methods, helper_path, helper_parameters, stack=stack
        )
        dependencies.update(child_deps)
        evidence.extend(child_evidence)
    if _HELPER_TOKEN in dependencies and isinstance(
        node, (ast.List, ast.Tuple, ast.Set)
    ):
        raise GraphCallableCoverageError(
            "graph_message_result_flow_unsupported",
            "sequence/set containment cannot identify one exact declared output leaf",
        )
    if _HELPER_TOKEN in dependencies and not isinstance(
        node, (ast.List, ast.Tuple, ast.Set)
    ):
        raise GraphCallableCoverageError(
            "graph_message_result_flow_unsupported",
            "the message-helper result enters a non-transparent expression",
        )
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        dependencies = {
            dependency
            if dependency == _HELPER_TOKEN
            else _UNSAFE_DEPENDENCY_PREFIX + dependency
            if not dependency.startswith(_UNSAFE_DEPENDENCY_PREFIX)
            else dependency
            for dependency in dependencies
        }
    else:
        dependencies = _unsafe_dependencies(dependencies)
    return dependencies, evidence


def _target_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for element in target.elts for name in _target_names(element)]
    return []


def _guard_opaque_input_calls(
    statement: ast.stmt,
    *,
    env: Mapping[str, set[str]],
    methods: Mapping[str, ast.FunctionDef | ast.AsyncFunctionDef],
    helper_path: tuple[str, ...] | None,
    protected_sources: set[str],
) -> None:
    for call in (node for node in ast.walk(statement) if isinstance(node, ast.Call)):
        path = _call_path(call)
        if path == helper_path or (
            len(path) == 2 and path[0] == "self" and path[1] in methods
        ):
            continue
        dependencies: set[str] = set()
        receiver = (
            [call.func.value] if isinstance(call.func, ast.Attribute) else []
        )
        for value in (
            *receiver,
            *call.args,
            *(keyword.value for keyword in call.keywords),
        ):
            child_dependencies, _ = _expression_dependencies(
                value, env, {}, None, None
            )
            dependencies.update(child_dependencies)
        if _HELPER_TOKEN in dependencies:
            raise GraphCallableCoverageError(
                "graph_message_result_flow_unsupported",
                "the message-helper result enters an opaque receiver call",
            )
        normalized = {
            dependency.removeprefix(_UNSAFE_DEPENDENCY_PREFIX)
            for dependency in dependencies
        }
        if normalized & protected_sources:
            raise GraphCallableCoverageError(
                "graph_message_input_escape_unsupported",
                "a protected graph or neighbor-signal value enters an opaque "
                "call before the exact message helper",
            )


def _require_straight_line_live_body(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> None:
    if any(
        not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Return, ast.Expr))
        for statement in function.body
    ):
        raise GraphCallableCoverageError(
            "graph_live_statement_unsupported",
            "version one proves only straight-line assignments, expressions, and "
            "one final return in analyzed model/pluggable functions",
        )


def _analyse_function_body(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    initial_env: Mapping[str, set[str]],
    methods: Mapping[str, ast.FunctionDef | ast.AsyncFunctionDef],
    helper_path: tuple[str, ...] | None,
    helper_parameters: tuple[str, str] | None,
    *,
    stack: tuple[str, ...],
) -> tuple[set[str], list[dict[str, Any]]]:
    _require_straight_line_live_body(function)
    return_nodes = [
        node
        for node in _function_scope_nodes(function)
        if isinstance(node, ast.Return)
    ]
    if (
        len(return_nodes) != 1
        or not function.body
        or function.body[-1] is not return_nodes[0]
    ):
        raise GraphCallableCoverageError(
            "graph_live_return_flow_unsupported",
            "version one requires one unconditional final return from every "
            "analyzed model/pluggable function",
        )
    if helper_path is not None and len(helper_path) == 1:
        helper_name = helper_path[0]
        parameters = {
            argument.arg
            for argument in (
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            )
        }
        local_stores = {
            path[0]
            for statement in function.body
            for path in _module_statement_bound_paths(statement)
            if len(path) == 1
        }
        helper_parameter_is_seed = (
            helper_parameters is None
            and helper_name in parameters
            and initial_env.get(helper_name) == {helper_name}
        )
        if (
            (helper_name in parameters and not helper_parameter_is_seed)
            or helper_name in local_stores
        ):
            raise GraphCallableProducerError(
                "graph_message_binding_shadowed",
                "the exact message-helper binding is shadowed inside the live "
                f"model method {function.name!r}",
            )
    env = {name: set(values) for name, values in initial_env.items()}
    protected_sources = {
        value.removeprefix(_UNSAFE_DEPENDENCY_PREFIX)
        for values in initial_env.values()
        for value in values
        if value != _HELPER_TOKEN
    }
    returns: set[str] = set()
    evidence: list[dict[str, Any]] = []
    for statement in function.body:
        _guard_opaque_input_calls(
            statement,
            env=env,
            methods=methods,
            helper_path=helper_path,
            protected_sources=protected_sources,
        )
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            value = statement.value
            if value is None:
                continue
            deps, found = _expression_dependencies(
                value, env, methods, helper_path, helper_parameters, stack=stack
            )
            targets = (
                statement.targets
                if isinstance(statement, ast.Assign)
                else [statement.target]
            )
            mutated_roots = {
                root
                for target in targets
                if (root := _value_root_name(target)) is not None
                and not isinstance(target, ast.Name)
            }
            if any(
                _HELPER_TOKEN in env.get(root, set())
                or {
                    dependency.removeprefix(_UNSAFE_DEPENDENCY_PREFIX)
                    for dependency in env.get(root, set())
                } & protected_sources
                for root in mutated_roots
            ):
                raise GraphCallableCoverageError(
                    "graph_message_input_mutation_unsupported",
                    "a protected graph, neighbor signal, or message-helper "
                    "result is mutated on the live path",
                )
            if _HELPER_TOKEN in deps and any(
                not isinstance(target, ast.Name) for target in targets
            ):
                raise GraphCallableCoverageError(
                    "graph_message_result_flow_unsupported",
                    "the message-helper result uses an unsupported assignment target",
                )
            for target in targets:
                for name in _target_names(target):
                    env[name] = set(deps)
            evidence.extend(found)
        elif isinstance(statement, ast.Return) and statement.value is not None:
            deps, found = _expression_dependencies(
                statement.value, env, methods, helper_path, helper_parameters, stack=stack
            )
            returns.update(deps)
            evidence.extend(found)
        elif isinstance(statement, ast.Expr):
            _, found = _expression_dependencies(
                statement.value, env, methods, helper_path, helper_parameters, stack=stack
            )
            evidence.extend(found)
        elif any(
            helper_path is not None
            and isinstance(node, ast.Call)
            and _call_path(node) == helper_path
            for node in ast.walk(statement)
        ):
            raise GraphCallableCoverageError(
                "graph_message_control_flow_unsupported",
                "the exact message helper is nested under unsupported control flow",
            )
    return returns, evidence


def _message_output_binding(
    tree: ast.Module,
    *,
    helper_path: tuple[str, ...],
    evidence: Mapping[str, Any],
    output_root: str,
    final_output_root: str,
) -> str:
    if "." not in output_root:
        raise GraphCallableCoverageError(
            "graph_message_output_root_grammar_unsupported",
            "version one requires a dotted coindexed output root",
        )
    expected_binding = output_root.rsplit(".", 1)[1]
    if not expected_binding.isidentifier():
        raise GraphCallableCoverageError(
            "graph_message_output_root_grammar_unsupported",
            "the coindexed output root has no exact Python-identifier leaf",
        )
    lineno = evidence.get("call_lineno")
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and node.lineno == lineno
        and _call_path(node) == helper_path
    ]
    if len(calls) != 1:
        raise GraphCallableCoverageError(
            "graph_message_output_binding_unsupported",
            "cannot resolve one exact message-helper call for output binding",
        )
    parents = _parents(tree)
    parent = parents.get(calls[0])
    if isinstance(parent, (ast.Assign, ast.AnnAssign)):
        targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
        if len(targets) == 1 and isinstance(targets[0], ast.Name):
            observed = targets[0].id
            if observed != expected_binding:
                raise GraphCallableProducerError(
                    "graph_message_output_root_mismatch",
                    "the direct message-helper result binds local "
                    f"{observed!r}, not the exact coindexed output root "
                    f"{output_root!r}",
                )
            return observed
    if isinstance(parent, ast.Return) and output_root == final_output_root:
        return "<direct_return>"
    raise GraphCallableCoverageError(
        "graph_message_output_binding_unsupported",
        "version one requires the direct helper result to bind the exact output-root "
        "leaf, or directly return the declared final inference root",
    )


def _guard_reachable_self_method_bindings(
    class_node: ast.ClassDef,
    methods: Mapping[str, ast.FunctionDef | ast.AsyncFunctionDef],
    module_tree: ast.Module,
    method_dir: Path,
) -> None:
    """Refuse descriptor shadowing on any self-method traversed from forward."""

    _reject_architecture_init_instance_escape(
        class_node,
        code="graph_message_dynamic_dispatch_unsupported",
    )
    _reject_producer_architecture_class_references(
        module_tree,
        class_name=class_node.name,
        code="graph_message_dynamic_dispatch_unsupported",
    )
    reachable: set[str] = {"forward"}
    pending = ["forward"]
    while pending:
        owner = pending.pop()
        function = methods.get(owner)
        if function is None:
            continue
        for call in (
            node for node in ast.walk(function) if isinstance(node, ast.Call)
        ):
            path = _call_path(call)
            if len(path) != 2 or path[0] != "self" or path[1] not in methods:
                continue
            name = path[1]
            if name not in reachable:
                reachable.add(name)
                pending.append(name)

    if not reachable:
        return
    if class_node.decorator_list or class_node.keywords:
        raise GraphCallableCoverageError(
            "graph_message_dynamic_dispatch_unsupported",
            "class decorators or class keywords can replace forward dispatch",
        )
    if not _declarative_class_body(class_node, module_tree):
        raise GraphCallableCoverageError(
            "graph_message_dynamic_dispatch_unsupported",
            "executable class-body descriptors or decorated methods are outside "
            "the closed model-dispatch grammar",
        )
    allowed_bases = {
        (),
        ("object",),
        ("nn", "Module"),
        ("torch", "nn", "Module"),
    }
    if any(_expression_path(base) not in allowed_bases for base in class_node.bases):
        raise GraphCallableCoverageError(
            "graph_message_dynamic_dispatch_unsupported",
            "an unrecognized model base can replace forward or self-method dispatch",
        )
    if {"__getattribute__", "__getattr__"} & set(methods):
        raise GraphCallableCoverageError(
            "graph_message_dynamic_dispatch_unsupported",
            "custom attribute lookup can replace a traversed self-method",
        )
    if any(methods[name].decorator_list for name in reachable):
        raise GraphCallableCoverageError(
            "graph_message_dynamic_dispatch_unsupported",
            "decorated self-method dispatch is outside graph liveness v1",
        )
    dynamic_class_bindings = {
        "__call__",
        "__delattr__",
        "__getattribute__",
        "__getattr__",
        "__new__",
        "__setattr__",
        "_call_impl",
        "_wrapped_call_impl",
    } | set(_FRAMEWORK_DISPATCH_EXTENSION_POINTS)
    if any(
        path == (name,)
        for statement in class_node.body
        for path in _module_statement_bound_paths(statement)
        for name in dynamic_class_bindings
    ):
        raise GraphCallableCoverageError(
            "graph_message_dynamic_dispatch_unsupported",
            "dynamic attribute lookup or construction can replace the analyzed model",
        )
    for statement in class_node.body:
        for name in reachable:
            if statement is methods[name]:
                continue
            if (name,) in _module_statement_bound_paths(statement):
                raise GraphCallableCoverageError(
                    "graph_message_method_binding_shadowed",
                    f"class-level binding shadows traversed self-method {name!r}",
                )
    for function in methods.values():
        for node in ast.walk(function):
            if not isinstance(node, ast.Attribute) or node.attr not in reachable:
                continue
            if not isinstance(node.value, ast.Name) or node.value.id != "self":
                continue
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                raise GraphCallableCoverageError(
                    "graph_message_method_binding_shadowed",
                    f"instance binding shadows traversed self-method {node.attr!r}",
                )
    guarded_methods = set(reachable)
    if "__init__" in methods:
        guarded_methods.add("__init__")
    guarded_reachable_functions = [
        reachable
        for name in guarded_methods
        for reachable in _reachable_module_functions(module_tree, methods[name])
    ]
    guarded_execution_nodes = [
        *_module_init_nodes(module_tree),
        *(
            node
            for reachable in guarded_reachable_functions
            for node in _function_scope_nodes(reachable)
        ),
    ]
    _reject_local_method_imports(
        module_tree,
        guarded_execution_nodes,
        code="graph_message_dynamic_dispatch_unsupported",
        method_dir=method_dir,
    )
    _reject_framework_dispatch_impersonation(
        module_tree,
        guarded_execution_nodes,
        code="graph_message_dynamic_dispatch_unsupported",
    )
    _reject_numeric_runtime_binding_mutation(
        module_tree,
        guarded_execution_nodes,
        code="graph_message_dynamic_dispatch_unsupported",
    )
    _reject_forward_hook_access(
        module_tree,
        guarded_execution_nodes,
        code="graph_message_dynamic_dispatch_unsupported",
    )
    if any(
        isinstance(node, ast.Name) and node.id == class_node.name
        for reachable in guarded_reachable_functions
        if reachable not in methods.values()
        for node in _function_scope_nodes(reachable)
    ):
        raise GraphCallableCoverageError(
            "graph_message_dynamic_dispatch_unsupported",
            "a forward-reachable local callable touches the declaring model class",
        )
    hook_methods = {
        "register_forward_hook",
        "register_forward_pre_hook",
        "register_full_backward_hook",
        "register_full_backward_pre_hook",
    }
    for name in guarded_methods:
        function = methods[name]
        _reject_executable_function_header(function, module_tree)
        _reject_deferred_callable(function)
        _reject_dynamic_namespace(
            function,
            label="graph_message",
            module_tree=module_tree,
        )
        if name == "__init__":
            if any(
                isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                and statement.value is not None
                and class_node.name in _expr_names(statement.value)
                for statement in _function_scope_nodes(function)
            ):
                raise GraphCallableCoverageError(
                    "graph_message_dynamic_dispatch_unsupported",
                    "model initialization cannot store the declaring class in "
                    "instance data used outside the analyzed forward path",
                )
            for call in (
                node
                for node in _function_scope_nodes(function)
                if isinstance(node, ast.Call)
            ):
                path = _call_path(call)
                if path and path[0] == "self":
                    raise GraphCallableCoverageError(
                        "graph_message_dynamic_dispatch_unsupported",
                        "model initialization cannot dispatch through self-bound "
                        "callables in the closed liveness grammar",
                    )
                values = (
                    *call.args,
                    *(keyword.value for keyword in call.keywords),
                )
                if any("self" in _expr_names(value) for value in values):
                    raise GraphCallableCoverageError(
                        "graph_message_dynamic_dispatch_unsupported",
                        "model initialization cannot pass the live instance into "
                        "opaque callables",
                    )
            for node in _function_scope_nodes(function):
                if not isinstance(node, ast.Attribute):
                    continue
                path = _expression_path(node)
                if not path or path[0] != "self" or len(path) < 2:
                    continue
                if (
                    path[1] == "__dict__"
                    or "hook" in path[1]
                    or path[1] in _FRAMEWORK_DISPATCH_METHODS
                    or path[1] == "forward"
                ):
                    raise GraphCallableCoverageError(
                        "graph_message_dynamic_dispatch_unsupported",
                        "model initialization touches protected instance dispatch "
                        "state",
                    )
        if any(
            isinstance(node, ast.Call)
            and len(_call_path(node)) == 2
            and _call_path(node)[0] == "self"
            and _call_path(node)[1] in hook_methods
            for node in _function_scope_nodes(function)
        ):
            raise GraphCallableCoverageError(
                "graph_message_dynamic_dispatch_unsupported",
                "model forward hooks can replace the analyzed return path",
            )
        if any(
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and "hook" in node.attr
            and isinstance(node.ctx, (ast.Store, ast.Del))
            for node in _function_scope_nodes(function)
        ):
            raise GraphCallableCoverageError(
                "graph_message_dynamic_dispatch_unsupported",
                "model hook state can replace the analyzed return path",
            )
        for statement in function.body:
            if any(
                path
                and path[0] == class_node.name
                for path in _module_statement_bound_paths(statement)
            ):
                raise GraphCallableCoverageError(
                    "graph_message_method_binding_shadowed",
                    "a reachable model method mutates its declaring class binding",
                )
            for call in (
                node
                for node in ast.walk(statement)
                if isinstance(node, ast.Call)
            ):
                receiver = (
                    [call.func.value]
                    if isinstance(call.func, ast.Attribute)
                    else []
                )
                values = (
                    *receiver,
                    *call.args,
                    *(keyword.value for keyword in call.keywords),
                )
                if any(class_node.name in _expr_names(value) for value in values):
                    raise GraphCallableCoverageError(
                        "graph_message_dynamic_dispatch_unsupported",
                        "the declaring model class enters an opaque reachable call",
                    )


def prove_message_live_path(
    method_spec: Mapping[str, Any],
    arch_contract: Mapping[str, Any],
    method_dir: Path,
) -> dict[str, Any]:
    """Prove the exact helper feeds the Stage-2.d-exercised model return."""

    mechanism = _mechanism(method_spec)
    if mechanism is None:
        return {"status": "not_applicable", "reason": "graph_mechanism_absent"}
    _require_declarative_package_init(Path(method_dir))
    message = _mapping(mechanism.get("message_passing"), label="message passing")
    identity = _callable_identity(message.get("callable"), label="message_passing.callable")
    graph_parameter = _exact_identifier(
        message.get("graph_parameter"), label="message graph parameter"
    )
    signal_parameter = _exact_identifier(
        message.get("neighbor_signal_parameter"), label="message signal parameter"
    )
    output_root = message.get("output_root")
    if not isinstance(output_root, str) or not output_root:
        raise GraphCallableCoverageError(
            "graph_message_output_root_unavailable",
            "message_passing.output_root is unavailable",
        )
    bindings = _relational_bindings(arch_contract, mechanism)
    if output_root in bindings["inference_input_logical_roots"]:
        raise GraphCallableProducerError(
            "graph_message_output_root_is_input",
            "message_passing.output_root names an R2C-084 inference input root, "
            "not an output of the declared helper",
        )
    architecture = _mapping(arch_contract.get("architecture"), label="architecture")
    block = _mapping(
        architecture.get(bindings["architecture_block"]), label="architecture block"
    )
    class_name = _exact_identifier(block.get("class_name"), label="architecture class")
    model_path = Path(method_dir) / "model.py"
    model_tree = _parse_file(model_path, code="graph_model_source_unavailable")
    classes = [
        node for node in model_tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        raise GraphCallableProducerError(
            "graph_live_architecture_class_missing",
            f"method/model.py does not define exactly one {class_name!r}",
        )
    architecture_class = classes[0]
    _require_final_module_binding(
        model_tree,
        route=(class_name,),
        origin_line=architecture_class.lineno,
        label="graph_architecture_class",
        reject_descendant_bindings=True,
        reject_opaque_escape=True,
    )
    methods = {
        node.name: node
        for node in architecture_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    forward = methods.get("forward")
    if forward is None:
        raise GraphCallableProducerError(
            "graph_live_forward_missing",
            f"architecture class {class_name!r} has no exact forward method",
        )
    _guard_reachable_self_method_bindings(
        architecture_class, methods, model_tree, Path(method_dir)
    )

    owner_path = _module_file(Path(method_dir), identity["module"])
    owner_tree = _parse_file(owner_path, code="graph_message_source_unavailable")
    owner_function = _function_def(owner_tree, identity["qualname"])
    construction_identity = _callable_identity(
        _mapping(mechanism.get("construction"), label="construction").get("callable"),
        label="construction.callable",
    )
    training_name = _exact_identifier(
        _mapping(arch_contract.get("training_loop"), label="training loop").get(
            "function_name"
        ),
        label="training loop function name",
    )
    pluggable_name = _exact_identifier(
        _mapping(
            arch_contract.get("pluggable_component"), label="pluggable component"
        ).get("name"),
        label="pluggable component name",
    )
    helper_protected = {
        construction_identity["qualname"],
        training_name,
        pluggable_name,
        class_name,
    }
    _require_final_module_binding(
        owner_tree,
        route=(identity["qualname"],),
        origin_line=owner_function.lineno,
        label="graph_message_callable",
        reject_descendant_bindings=True,
        reject_opaque_escape=True,
    )
    _reject_cross_callable_references(
        owner_tree,
        owner_function,
        protected_qualnames=helper_protected,
        locally_owned_qualnames=_locally_owned_protected_qualnames(
            owner_tree, helper_protected
        ),
        label="graph_message_source",
        method_dir=Path(method_dir),
    )
    helper_binding = _relative_import_path(
        model_tree, owner_module="method.model", identity=identity
    )
    if helper_binding is None:
        raise GraphCallableProducerError(
            "graph_message_live_import_missing",
            "method/model.py does not bind the exact declared message helper",
        )
    helper_path, helper_origin_line = helper_binding
    _require_final_module_binding(
        model_tree,
        route=helper_path,
        origin_line=helper_origin_line,
        label="graph_message",
        reject_descendant_bindings=True,
        reject_opaque_escape=True,
    )

    forward_parameters = [
        argument.arg
        for argument in (
            *forward.args.posonlyargs,
            *forward.args.args,
            *forward.args.kwonlyargs,
        )
    ]
    initial_env = {name: {name} for name in forward_parameters}
    returned, evidence = _analyse_function_body(
        forward,
        initial_env,
        methods,
        helper_path,
        (graph_parameter, signal_parameter),
        stack=("forward",),
    )
    if not evidence:
        if any(
            isinstance(node, ast.Call) and _call_path(node) == helper_path
            for node in ast.walk(model_tree)
        ):
            raise GraphCallableCoverageError(
                "graph_message_call_grammar_unsupported",
                "the exact message helper is called outside the supported direct "
                "architecture.forward or self-method flow",
            )
        raise GraphCallableProducerError(
            "graph_message_live_call_missing",
            "the exact declared message helper is not reached from architecture.forward",
        )
    if len(evidence) != 1:
        raise GraphCallableCoverageError(
            "graph_message_call_grammar_unsupported",
            "version one proves one exact live message-helper call",
        )
    observed = evidence[0]
    if observed["graph_dependencies"] != [bindings["inference_graph_parameter"]]:
        raise GraphCallableProducerError(
            "graph_message_live_graph_mismatch",
            "the exact message helper does not consume the R2C-084 inference graph root",
        )
    if observed["neighbor_dependencies"] != [
        bindings["inference_neighbor_parameter"]
    ]:
        raise GraphCallableProducerError(
            "graph_message_live_signal_mismatch",
            "the exact message helper does not consume the R2C-084 neighbor-signal root",
        )
    if _HELPER_TOKEN not in returned:
        raise GraphCallableProducerError(
            "graph_message_result_discarded",
            "architecture.forward calls the exact message helper but its result "
            "does not reach the declared model return",
        )
    output_binding = _message_output_binding(
        model_tree,
        helper_path=helper_path,
        evidence=observed,
        output_root=output_root,
        final_output_root=bindings["inference_output_root"],
    )
    output_key_tokens = {
        dependency.removeprefix(_HELPER_OUTPUT_KEY_PREFIX)
        for dependency in returned
        if dependency.startswith(_HELPER_OUTPUT_KEY_PREFIX)
    }
    final_return_is_mapping = (
        bool(forward.body)
        and isinstance(forward.body[-1], ast.Return)
        and isinstance(forward.body[-1].value, ast.Dict)
    )
    if (
        output_root == bindings["inference_output_root"]
        and (output_key_tokens or final_return_is_mapping)
        and output_root.rsplit(".", 1)[-1] not in output_key_tokens
    ):
        raise GraphCallableProducerError(
            "graph_message_output_root_mismatch",
            "the declared final output leaf does not carry the exact message-helper "
            "result",
        )
    return {
        "status": "pass",
        "reason": "message_helper_consumed_by_fitting_and_inference_forward",
        "callable": identity,
        "architecture_block": bindings["architecture_block"],
        "architecture_class": class_name,
        "forward_callable": f"method.model:{class_name}.forward",
        "helper_graph_parameter": graph_parameter,
        "helper_neighbor_signal_parameter": signal_parameter,
        "graph_parameter": bindings["inference_graph_parameter"],
        "neighbor_signal_parameter": bindings["inference_neighbor_parameter"],
        "output_root": output_root,
        "output_binding": output_binding,
        "helper_result_consumed": True,
        "phases": ["fitting", "inference"],
    }


def _digest(paths: Sequence[Path], root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): (
            "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        )
        for path in paths
    }


def stage_2d_authority_digests(run_dir: Path) -> dict[str, str]:
    """Mirror the driver's existing Stage 2.d authority set."""

    root = Path(run_dir)
    method_dir = root / "method"
    files = (
        sorted(
            path
            for path in method_dir.rglob("*.py")
            if path != method_dir / "__init__.py"
        )
        if method_dir.is_dir()
        else []
    )
    files.extend(
        path for path in (
            root / ".pipeline" / "arch_contract.json",
            root / ".pipeline" / "method_spec.json",
        ) if path.is_file()
    )
    return _digest(files, root)


def stage_3c_authority_digests(run_dir: Path) -> dict[str, str]:
    """Mirror the driver's existing Stage 3.c notebook-smoke authority set."""

    root = Path(run_dir)
    method_dir = root / "method"
    files = sorted(method_dir.rglob("*.py")) if method_dir.is_dir() else []
    files.extend(
        path for path in (
            root / ".pipeline" / "notebook_draft.py",
            root / "notebook.ipynb",
            root / ".pipeline" / "params.json",
            root / "requirements.txt",
        ) if path.is_file()
    )
    return _digest(files, root)


def _valid_digest_map(value: object) -> bool:
    return isinstance(value, Mapping) and bool(value) and all(
        isinstance(relative, str)
        and bool(relative)
        and isinstance(digest, str)
        and len(digest) == 71
        and digest.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in digest[7:])
        for relative, digest in value.items()
    )


def assess_liveness_receipt(
    run_dir: Path,
    plan: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    """Fail-closed live/portable assessment of a frozen liveness receipt."""

    receipt = plan.get("callable_liveness_receipt")
    if not isinstance(receipt, Mapping):
        return False, "graph_callable_liveness_receipt_missing", {}
    expected_keys = {
        "receipt_version", "status", "verified", "reason", "construction",
        "message_passing", "authority_digests",
    }
    if set(receipt) != expected_keys or receipt.get("receipt_version") != RECEIPT_VERSION:
        return False, "graph_callable_liveness_receipt_shape_mismatch", {}
    if (
        receipt.get("status") != "pass"
        or receipt.get("verified") is not True
        or receipt.get("reason") != "graph_callable_liveness_verified"
    ):
        return (
            False,
            str(receipt.get("reason") or "graph_callable_liveness_unverified"),
            {},
        )
    construction = receipt.get("construction")
    message = receipt.get("message_passing")
    plan_construction = plan.get("construction")
    plan_execution = plan.get("execution")
    if not all(isinstance(value, Mapping) for value in (
        construction, message, plan_construction, plan_execution
    )):
        return False, "graph_callable_liveness_identity_mismatch", {}
    expected_construction_keys = {
        "status",
        "reason",
        "callable",
        "output_selector",
        "selected_binding",
        "fitting_copy_binding",
        "fitting_copy_method",
        "fitting_graph_array_kind",
        "numeric_operand_dispatch_precondition",
        "feature_input_root",
        "feature_parameter",
        "feature_binding",
        "parameter_bindings",
        "fitting_graph_parameter",
        "fitting_model_parameter",
        "architecture_class",
        "architecture_model_binding",
        "fitting_model_identity_preserved",
        "fitting_model_result_binding",
        "pluggable_callable",
        "pluggable_model_parameter",
        "pluggable_graph_parameter",
        "pluggable_neighbor_signal_parameter",
        "inference_graph_parameter",
        "inference_neighbor_signal_parameter",
        "neighbor_signal_root",
        "neighbor_signal_binding",
        "inference_output_binding",
        "pluggable_result_consumed",
        "phases",
    }
    expected_message_keys = {
        "status",
        "reason",
        "callable",
        "architecture_block",
        "architecture_class",
        "forward_callable",
        "helper_graph_parameter",
        "helper_neighbor_signal_parameter",
        "graph_parameter",
        "neighbor_signal_parameter",
        "output_root",
        "output_binding",
        "helper_result_consumed",
        "phases",
    }
    if (
        set(construction) != expected_construction_keys
        or set(message) != expected_message_keys
        or construction.get("status") != "pass"
        or construction.get("reason") != (
            "constructor_output_reaches_fitting_and_inference"
        )
        or construction.get("pluggable_result_consumed") is not True
        or construction.get("fitting_model_identity_preserved") is not True
        or construction.get("fitting_graph_array_kind") not in {"numpy", "torch"}
        or construction.get("numeric_operand_dispatch_precondition") != (
            NUMERIC_OPERAND_DISPATCH_PRECONDITION
        )
        or construction.get("inference_output_binding") != (
            str(plan_execution.get("output_root", "")).rsplit(".", 1)[-1]
        )
        or construction.get("neighbor_signal_root") != (
            plan_execution.get("neighbor_signal_root")
        )
        or construction.get("inference_graph_parameter") != (
            message.get("graph_parameter")
        )
        or construction.get("inference_neighbor_signal_parameter") != (
            message.get("neighbor_signal_parameter")
        )
        or message.get("status") != "pass"
        or message.get("reason") != (
            "message_helper_consumed_by_fitting_and_inference_forward"
        )
    ):
        return False, "graph_callable_liveness_proof_shape_mismatch", {}
    if (
        construction.get("callable") != plan_construction.get("callable")
        or message.get("callable") != plan_execution.get("callable")
        or construction.get("output_selector") != (
            plan_construction.get("output_selector")
        )
        or construction.get("feature_input_root") != (
            plan_construction.get("feature_input_root")
        )
        or construction.get("feature_parameter") != (
            plan_construction.get("feature_parameter")
        )
        or message.get("helper_graph_parameter") != (
            plan_execution.get("graph_parameter")
        )
        or message.get("helper_neighbor_signal_parameter") != (
            plan_execution.get("neighbor_signal_parameter")
        )
        or message.get("output_root") != plan_execution.get("output_root")
    ):
        return False, "graph_callable_liveness_identity_mismatch", {}
    expected_parameter_bindings: dict[str, dict[str, str]] = {}
    for raw_parameter in (
        (plan_construction.get("threshold") or {}).get("parameter"),
        (plan_construction.get("cap") or {}).get("parameter"),
    ):
        if not isinstance(raw_parameter, Mapping):
            continue
        params_name = raw_parameter.get("params_name")
        callable_parameter = raw_parameter.get("callable_parameter")
        if isinstance(params_name, str) and isinstance(callable_parameter, str):
            expected_parameter_bindings[params_name] = {
                "callable_parameter": callable_parameter,
                "notebook_cfg_key": params_name,
            }
    if construction.get("parameter_bindings") != expected_parameter_bindings:
        return False, "graph_callable_liveness_parameter_binding_mismatch", {}
    if (
        construction.get("phases") != ["fitting", "inference"]
        or message.get("phases") != ["fitting", "inference"]
        or message.get("helper_result_consumed") is not True
    ):
        return False, "graph_callable_liveness_phase_mismatch", {}
    authority = receipt.get("authority_digests")
    if not isinstance(authority, Mapping) or set(authority) != {"stage_2d", "stage_3c"}:
        return False, "graph_callable_liveness_authority_malformed", {}
    if not _valid_digest_map(authority.get("stage_2d")) or not _valid_digest_map(
        authority.get("stage_3c")
    ):
        return False, "graph_callable_liveness_authority_malformed", {}
    try:
        current = {
            "stage_2d": stage_2d_authority_digests(Path(run_dir)),
            "stage_3c": stage_3c_authority_digests(Path(run_dir)),
        }
    except OSError as exc:
        return False, "graph_callable_liveness_authority_unreadable", {
            "exception": type(exc).__name__,
        }
    expected = {
        "stage_2d": dict(authority["stage_2d"]),
        "stage_3c": dict(authority["stage_3c"]),
    }
    missing_stage_receipts = [
        stage
        for stage in ("stage_2d", "stage_3c")
        if not (
            Path(run_dir) / ".pipeline" / f"{stage}.complete"
        ).is_file()
    ]
    if missing_stage_receipts:
        return False, "graph_callable_liveness_stage_receipt_missing", {
            "missing_stages": missing_stage_receipts,
        }
    if current != expected:
        changed = sorted({
            f"{stage}:{relative}"
            for stage in ("stage_2d", "stage_3c")
            for relative in set(current[stage]) | set(expected[stage])
            if current[stage].get(relative) != expected[stage].get(relative)
        })
        return False, "graph_callable_liveness_receipt_stale", {
            "changed_authority_paths": changed,
        }
    root = Path(run_dir)
    try:
        recomputed_method_spec = json.loads(
            (root / ".pipeline" / "method_spec.json").read_text(
                encoding="utf-8"
            )
        )
        recomputed_arch_contract = json.loads(
            (root / ".pipeline" / "arch_contract.json").read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(recomputed_method_spec, Mapping) or not isinstance(
            recomputed_arch_contract, Mapping
        ):
            raise TypeError("liveness proof authorities must be JSON objects")
        recomputed_construction = prove_constructor_notebook_flow(
            notebook_code_cells(root),
            recomputed_method_spec,
            recomputed_arch_contract,
            root / "method",
        )
        recomputed_message = prove_message_live_path(
            recomputed_method_spec,
            recomputed_arch_contract,
            root / "method",
        )
    except GraphCallableLivenessError as exc:
        return False, "graph_callable_liveness_proof_recomputation_failed", {
            "proof_reason_code": exc.code,
        }
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        return False, "graph_callable_liveness_proof_recomputation_failed", {
            "exception": type(exc).__name__,
        }
    if (
        dict(construction) != recomputed_construction
        or dict(message) != recomputed_message
    ):
        changed_proof_fields = sorted({
            *(f"construction.{key}" for key in (
                set(construction) | set(recomputed_construction)
            ) if construction.get(key) != recomputed_construction.get(key)),
            *(f"message_passing.{key}" for key in (
                set(message) | set(recomputed_message)
            ) if message.get(key) != recomputed_message.get(key)),
        })
        return False, "graph_callable_liveness_receipt_proof_mismatch", {
            "changed_proof_fields": changed_proof_fields,
        }
    return True, "graph_callable_liveness_verified", {
        "construction_callable": construction.get("callable"),
        "message_callable": message.get("callable"),
        "authority_paths": {
            stage: sorted(expected[stage]) for stage in ("stage_2d", "stage_3c")
        },
    }


__all__ = [
    "ALLOWED_GRAPH_CALLABLE_MODULES",
    "GraphCallableCoverageError",
    "GraphCallableLivenessError",
    "GraphCallableProducerError",
    "NUMERIC_OPERAND_DISPATCH_PRECONDITION",
    "RECEIPT_VERSION",
    "assess_liveness_receipt",
    "graph_callable_identities",
    "notebook_code_cells",
    "prove_constructor_notebook_flow",
    "prove_message_live_path",
    "stage_2d_authority_digests",
    "stage_3c_authority_digests",
]
