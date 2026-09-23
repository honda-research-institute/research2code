"""Stage 2.d static validator for `<RUN_DIR>/.pipeline/arch_contract.json`.

Two-tier check, both deterministic:

  T1 — Schema validation.  The file exists, parses as JSON, and validates
       through the versioned reader against the legacy or typed architecture
       contract schema.

  T2 — Paradigm completeness.  The spec-derived taxonomy build plan carries
       an `arch_contract_requirements.required_blocks` block declaring which
       interior fields the contract must populate for this paradigm (e.g.,
       KD requires architecture.teacher in addition to architecture.student;
       AL only requires architecture.model). This validator reads the build
       plan and checks each declared requirement against the contract.

       The same requirements block may carry `family_components`: the
       pack-declared legal names for the contract's one extension container
       (arch-contract headroom, approved 2026-07-16; SRL's reward_function
       2026-07-15, the shape ADAM's optimizer_state was hard-added for
       2026-07-08). Enforced in BOTH directions: a declared-required
       component that is missing fails, and a present component the pack
       never declared fails naming the unknown key and the declared set.

Companion `validate_arch_contract_runtime.py` runs after this one and
performs the runtime dry-run.

Both validators run AFTER `validate_package_imports.py` at Stage 2.d.
Producer-owned failures retain the bounded judge/fix route; trusted
pipeline-owned coverage failures bypass producer retries and use Stage 2.d's
terminal degrade route.

Usage:

    python scripts/validate_arch_contract.py --spec <method_spec.json> --run-dir <output_dir>

Exit codes:
  0  validation passed
  1  producer-owned, mixed, or unstructured validation failures
  2  setup error (missing or unreadable spec)
  3  exclusively pipeline-owned semantic coverage failures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from schemas.arch_contract_v2 import ArchContractV2  # noqa: E402
from schemas.graph_wiring import mechanism_marker_element_ids  # noqa: E402
from scripts.arch_contract_semantics import (  # noqa: E402
    AnyArchContract,
    SemanticIssue,
    format_semantic_issue,
    load_arch_contract,
    semantic_issue_exit_code,
)
from scripts.build_plan import load_build_plan  # noqa: E402
from scripts.taxonomy import run_overlay_dir  # noqa: E402


def _walk_dotted_path(obj: Any, path: str) -> tuple[bool, Any]:
    """Resolve `a.b.c` against a nested dict/object. Returns (found, value)."""
    parts = path.split(".")
    cursor: Any = obj
    for p in parts:
        if isinstance(cursor, dict) and p in cursor:
            cursor = cursor[p]
        elif hasattr(cursor, p):
            cursor = getattr(cursor, p)
        else:
            return False, None
    return True, cursor


def _spec_or_mapping_value(spec: dict[str, Any], mapping: dict[str, Any], target: str) -> tuple[bool, Any]:
    if target.startswith("spec."):
        return _walk_dotted_path(spec, target[len("spec."):])
    return _walk_dotted_path(mapping, target)


def _check_required_block(
    contract: AnyArchContract, block_path: str, block_spec: dict[str, Any]
) -> list[str]:
    """Verify one entry from `arch_contract_requirements.required_blocks` is
    satisfied by the contract. Returns a list of human-readable error
    strings (empty on success)."""
    errors: list[str] = []
    found, value = _walk_dotted_path(contract, block_path)
    if not found or value is None:
        errors.append(
            f"arch_contract.{block_path}: required by the paradigm's "
            f"arch_contract_requirements, but missing from the contract."
        )
        return errors

    if isinstance(block_spec, dict):
        # Sub-block requirement (e.g., required_subblocks: [class_name, forward.input.x])
        for sub in block_spec.get("required_subblocks", []) or []:
            sub_found, _ = _walk_dotted_path(value, sub)
            if not sub_found:
                errors.append(
                    f"arch_contract.{block_path}.{sub}: required by the "
                    f"paradigm's arch_contract_requirements, but missing."
                )
        # min_keys: dict must have at least these keys
        for key in block_spec.get("min_keys", []) or []:
            if not isinstance(value, dict) or key not in value:
                errors.append(
                    f"arch_contract.{block_path}: required key '{key}' "
                    f"missing from this dict-shaped block."
                )
        # exact_keys: dict must match the declared runtime/scaffold shape.
        exact_keys = block_spec.get("exact_keys", []) or []
        if exact_keys:
            expected = set(exact_keys)
            actual = set(value) if isinstance(value, dict) else set()
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            if missing or extra:
                detail = []
                if missing:
                    detail.append(f"missing {missing}")
                if extra:
                    detail.append(f"extra {extra}")
                errors.append(
                    f"arch_contract.{block_path}: keys must match the "
                    f"paradigm runtime contract exactly ({sorted(expected)}); "
                    + ", ".join(detail)
                    + ". Reconcile arch_contract.json with the scaffolded "
                    "loader return shape."
                )
        # type hint (advisory only — pydantic already enforced)
        # must_equal: validated below in the loop's special-case branch
    return errors


def _v2_requirement_path(path: str) -> tuple[str | None, str | None]:
    """Translate one legacy taxonomy requirement path into the v2 view.

    Taxonomy build plans remain the generic cross-version authority.  Version
    2 gives typed values different field names, so the static validator adapts
    only paths whose meaning is preserved.  In particular, v1's structured
    ``forward.output_keys`` contract has no v2 equivalent: an opaque output
    describes one value but cannot certify named members of that value.
    """
    parts = path.split(".")
    if path == "pluggable_component.batch_dict_shape":
        return None, format_semantic_issue(
            SemanticIssue(
                code="unsupported_validator_feature",
                message=(
                    "legacy pluggable_component.batch_dict_shape requirements "
                    "describe keys inside one structured batch, while schema "
                    "version 2 input descriptors describe callable parameters; "
                    "an opaque batch parameter cannot certify interior keys"
                ),
                roots=[
                    (
                        "arch_contract_requirements.required_blocks."
                        "pluggable_component.batch_dict_shape"
                    ),
                    "arch_contract.pluggable_component.input",
                    "scripts/validate_arch_contract.py",
                ],
                values={
                    "source_requirement_path": path,
                    "typed_input_root": "pluggable_component.input",
                },
            )
        )
    if any(
        part == "output_keys" and index > 0 and parts[index - 1] == "forward"
        for index, part in enumerate(parts)
    ):
        output_keys_index = parts.index("output_keys")
        typed_output_root = ".".join(parts[:output_keys_index] + ["output"])
        return None, format_semantic_issue(
            SemanticIssue(
                code="unsupported_validator_feature",
                message=(
                    f"architecture requirement {path!r} cannot be represented "
                    "by schema_version='2.0.0': typed forward.output describes "
                    "one value and an opaque descriptor cannot certify "
                    "structured output_keys"
                ),
                roots=[
                    "arch_contract_requirements.required_blocks." + path,
                    "arch_contract." + typed_output_root,
                    "scripts/validate_arch_contract.py",
                ],
                values={
                    "source_requirement_path": path,
                    "typed_output_root": typed_output_root,
                },
            )
        )

    mapped = list(parts)
    for index, part in enumerate(parts):
        if index > 0 and parts[index - 1] == "forward" and part in {
            "output_type",
            "output_shape",
        }:
            mapped[index] = "output"
        elif (
            parts[0] == "pluggable_component"
            and part == "input_shapes"
        ):
            mapped[index] = "input"
        elif parts[0] == "pluggable_component" and part == "output_shape":
            mapped[index] = "output"
        elif parts[0] == "training_loop" and part == "input_shapes":
            mapped[index] = "input"
    return ".".join(mapped), None


def _v2_requirements_view(
    required_blocks: dict[str, Any],
) -> tuple[list[tuple[str, Any]], list[str]]:
    """Return v2 lookup entries without mutating the taxonomy build plan."""
    entries: list[tuple[str, Any]] = []
    errors: list[str] = []
    for source_path, source_spec in required_blocks.items():
        block_path, path_error = _v2_requirement_path(source_path)
        if path_error is not None:
            errors.append(path_error)
            continue
        assert block_path is not None

        if not isinstance(source_spec, dict):
            entries.append((block_path, source_spec))
            continue

        block_spec = dict(source_spec)
        required_subblocks: list[str] = []
        for sub_path in source_spec.get("required_subblocks", []) or []:
            full_source_path = f"{source_path}.{sub_path}"
            full_v2_path, sub_error = _v2_requirement_path(full_source_path)
            if sub_error is not None:
                errors.append(sub_error)
                continue
            assert full_v2_path is not None
            prefix = f"{block_path}."
            if not full_v2_path.startswith(prefix):
                errors.append(
                    f"arch_contract requirement {full_source_path!r} has no "
                    "safe schema_version='2.0.0' requirements view."
                )
                continue
            mapped_sub_path = full_v2_path[len(prefix):]
            if mapped_sub_path not in required_subblocks:
                required_subblocks.append(mapped_sub_path)
        if "required_subblocks" in block_spec:
            block_spec["required_subblocks"] = required_subblocks
        entries.append((block_path, block_spec))
    return entries, errors


def _check_family_components(
    contract: AnyArchContract, declared: dict[str, Any], pack_hint: str
) -> list[str]:
    """Both-direction check of `family_components` against the pack's
    declaration (`arch_contract_requirements.family_components`).

    `pack_hint` names where the declaration lives (the run's provisional
    pack for gap runs, the committed taxonomy node otherwise) so the error
    points at BOTH remedies: declare the component in the pack, or remove
    it from the contract. Fail-loud on undeclared names is where the
    container's typo safety lives — a misspelled component name can never
    pass as an ignored extra."""
    errors: list[str] = []
    present = contract.family_components or {}

    for name in sorted(declared):
        decl = declared[name] if isinstance(declared[name], dict) else {}
        if name not in present:
            if decl.get("required", True):
                errors.append(
                    f"arch_contract.family_components.{name}: declared "
                    f"required by the paradigm's family_components block, but "
                    f"missing from the contract. Either emit the component in "
                    f"arch_contract.json, or relax/remove the declaration in "
                    f"{pack_hint}."
                )
            continue
        # Legacy components always expose ``entries``; v2 deliberately uses a
        # disjoint single-value/multi-entry union.  A single-value component
        # satisfies a component-presence declaration but cannot satisfy a
        # named-entry declaration.
        component_entries = getattr(present[name], "entries", None)
        entries = (
            component_entries if isinstance(component_entries, dict) else {}
        )
        for entry in decl.get("required_entries", []) or []:
            if entry not in entries:
                errors.append(
                    f"arch_contract.family_components.{name}.entries.{entry}: "
                    f"required by the paradigm's family_components "
                    f"declaration, but missing. Either emit the entry under "
                    f"the {name!r} component's `entries` map, or relax/remove "
                    f"the required_entries declaration in {pack_hint}."
                )

    declared_names = sorted(declared)
    for name in sorted(present):
        if name in declared:
            continue
        declared_desc = (
            f"declared set: {declared_names}"
            if declared_names
            else "this paradigm declares no family components"
        )
        errors.append(
            f"arch_contract.family_components.{name}: not declared by the "
            f"paradigm ({declared_desc}). Either declare {name!r} in "
            f"{pack_hint} (the pack owns the legal component names), or "
            f"remove it from the contract."
        )
    return errors


def _check_relational_indexing_contract(
    spec: dict[str, Any], contract: AnyArchContract
) -> list[str]:
    """Condition the graph contract on paper-grounded method truth.

    Activation comes only from the methodology element's typed marker.  A
    graph-looking callable, parameter, or prose phrase has no authority here.
    Unsupported relational forms become structured pipeline-owned coverage
    issues. Stage 2.d routes them directly without spending an architecture
    producer retry.
    """
    methodology = spec.get("methodology_replication_contract") or {}
    elements = methodology.get("elements") or []
    homogeneous_ids: list[str] = []
    unsupported: list[tuple[int, str, str]] = []
    for element_index, raw in enumerate(elements):
        if not isinstance(raw, dict):
            continue
        marker = raw.get("relational_structure")
        if not isinstance(marker, dict):
            continue
        element_id = str(raw.get("element_id") or "<missing-element-id>")
        kind = marker.get("kind")
        if kind == "homogeneous_graph":
            homogeneous_ids.append(element_id)
        elif kind == "unsupported":
            unsupported.append(
                (
                    element_index,
                    element_id,
                    str(marker.get("unsupported_kind") or "unspecified"),
                )
            )

    # R2C-092: the graph-mechanism block's alignment, construction, and
    # message-passing ids carry the marker by derivation, so the stored
    # JSON may omit it; activation and the exact-match check below must
    # read the same derived set every other consumer does.
    mechanism = (
        methodology.get("homogeneous_graph_mechanism")
        if isinstance(methodology, dict) else None
    )
    if isinstance(mechanism, dict):
        known_ids = {
            str(raw.get("element_id"))
            for raw in elements
            if isinstance(raw, dict)
        }
        for element_id in mechanism_marker_element_ids(mechanism):
            if element_id in known_ids and element_id not in homogeneous_ids:
                homogeneous_ids.append(element_id)

    errors: list[str] = []
    for element_index, element_id, unsupported_kind in unsupported:
        root = (
            "methodology_replication_contract.elements"
            f"[{element_index}].relational_structure"
        )
        errors.append(
            format_semantic_issue(
                SemanticIssue(
                    code="unsupported_validator_feature",
                    message=(
                        f"methodology element {element_id!r} declares relational "
                        f"form {unsupported_kind!r}, outside relational_indexing "
                        "version 1's homogeneous-graph validation boundary"
                    ),
                    roots=[root, "scripts/validate_arch_contract.py"],
                    values={
                        "element_id": element_id,
                        "unsupported_kind": unsupported_kind,
                        "supported_boundary": "homogeneous_graph",
                    },
                )
            )
        )

    relational = contract.relational_indexing
    if not homogeneous_ids:
        if relational is not None:
            errors.append(
                "arch_contract.relational_indexing is present, but no "
                "methodology_replication_contract element declares "
                "relational_structure.kind='homogeneous_graph'. Remove the "
                "inferred graph contract or correct the analyzer-owned method "
                "marker from paper evidence."
            )
        return errors

    if relational is None:
        errors.append(
            "arch_contract.relational_indexing is required because method_spec "
            "declares homogeneous-graph methodology element(s) "
            f"{sorted(homogeneous_ids)!r}. Architecture-coder must declare "
            "roles, roots, endpoint spaces, phase batch modes, degree "
            "semantics, output order, and the preparation callable."
        )
        return errors

    expected = set(homogeneous_ids)
    actual = set(relational.methodology_element_ids)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        detail: list[str] = []
        if missing:
            detail.append(f"missing graph element ids {missing}")
        if extra:
            detail.append(f"non-graph element ids {extra}")
        errors.append(
            "arch_contract.relational_indexing.methodology_element_ids must "
            "exactly match the method spec's homogeneous-graph elements; "
            + ", ".join(detail)
            + "."
        )
    if contract.training_loop is None:
        errors.append(
            "arch_contract.training_loop is required with relational_indexing: "
            "the producer seam must prove the declared graph-batch preparation "
            "callable is reachable from the fitting entry point."
        )
    elif (
        relational.preparation_callable.name
        == contract.training_loop.function_name
    ):
        errors.append(
            "arch_contract.relational_indexing.preparation_callable.name must "
            "name a pure graph-batch helper, not the training-loop entry point "
            f"{contract.training_loop.function_name!r}."
        )
    return errors


def validate(spec: dict, run_dir: Path) -> list[str]:
    errors: list[str] = []

    contract_path = run_dir / ".pipeline" / "arch_contract.json"
    if not contract_path.is_file():
        errors.append(
            f".pipeline/arch_contract.json missing at {contract_path}. "
            f"Architecture-coder must produce this artifact at Stage 2.b. "
            f"See schemas/arch_contract.py / schemas/arch_contract_v2.py for "
            f"the structure and the matched "
            f"taxonomy build plan for required fields."
        )
        return errors

    # T1 — Pydantic schema check.
    try:
        raw = json.loads(contract_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        errors.append(f"arch_contract.json is not valid JSON: {e}")
        return errors
    try:
        contract = load_arch_contract(raw)
    except Exception as e:  # pydantic.ValidationError
        errors.append(f"arch_contract.json failed schema validation: {e}")
        return errors

    # T2 — Paradigm completeness check.
    spec_paradigm_id = (spec.get("comparison") or {}).get("classification", {}).get("id")
    if contract.paradigm_id != spec_paradigm_id:
        errors.append(
            f"arch_contract.paradigm_id={contract.paradigm_id!r} does not "
            f"match spec.comparison.classification.id={spec_paradigm_id!r}. "
            f"Architecture-coder must copy the paradigm id from the spec."
        )

    errors.extend(_check_relational_indexing_contract(spec, contract))

    overlay_dir = run_overlay_dir(run_dir)
    build_plan = load_build_plan(
        spec, ROOT, provisional_packs_dir=overlay_dir)
    if build_plan is None:
        errors.append(
            format_semantic_issue(
                SemanticIssue(
                    code="unsupported_validator_feature",
                    message=(
                        "no taxonomy build plan exists for "
                        "comparison.classification.id="
                        f"{spec_paradigm_id!r}; the pipeline cannot evaluate "
                        "paradigm-specific architecture requirements"
                    ),
                    roots=[
                        "spec.comparison.classification.id",
                        (
                            str(overlay_dir)
                            if overlay_dir is not None
                            else "docs/ssot/taxonomies.yaml"
                        ),
                        "scripts/validate_arch_contract.py",
                    ],
                    values={"paradigm_id": spec_paradigm_id},
                )
            )
        )
        return errors
    schema = build_plan.get("arch_contract_requirements")
    build_plan_merged: dict[str, Any] = {
        "pluggable_component": (build_plan.get("pluggable_component") or {})
    }
    # Family-component names are enforced in both directions even when the
    # paradigm declares no other requirements: an undeclared extension block
    # must fail loudly regardless.
    pack_hint = (
        f"the run's provisional pack under {overlay_dir}"
        if overlay_dir is not None
        else "the family's taxonomy node (docs/ssot/taxonomies.yaml)"
    )
    errors.extend(_check_family_components(
        contract, (schema or {}).get("family_components") or {}, pack_hint))
    if not schema:
        # The paradigm has no extra arch-contract requirements. Treat as an
        # authoring gap, not the architecture-coder's fault; the universal
        # schema check already passed.
        print(
            f"warning: paradigm {spec_paradigm_id!r} has no "
            f"arch-contract requirement block; skipping paradigm-completeness check.",
            file=sys.stderr,
        )
        return errors

    required_blocks = schema.get("required_blocks", {}) or {}
    if not isinstance(required_blocks, dict):
        errors.append("spec-derived build plan: arch_contract required_blocks is not a dict.")
        return errors

    requirement_entries: list[tuple[str, Any]] = list(required_blocks.items())
    if isinstance(contract, ArchContractV2):
        requirement_entries, view_errors = _v2_requirements_view(required_blocks)
        errors.extend(view_errors)

    for block_path, block_spec in requirement_entries:
        # Handle must_equal at top level (paths like pluggable_component.name).
        if isinstance(block_spec, dict) and "must_equal" in block_spec:
            found, value = _walk_dotted_path(contract, block_path)
            target = block_spec["must_equal"]
            # Resolve the `must_equal` target. Precedence:
            #   1. For `pluggable_component.*`, the SPEC is authoritative — it
            #      carries the analyzer's per-paper, variant-aware choice. An
            #      optimization_based MPC paper picks `solve_step`, not the
            #      paradigm default `plan`; older build metadata couldn't express that
            #      as a single value, but the spec can (PA-FG3).
            #   2. Else, a dotted path resolving against the build-plan mapping
            #      (e.g. a paradigm-wide constant) (PA-D5).
            #   3. Else, treat the target as a literal.
            target_value, resolved = target, False
            if isinstance(target, str) and target.startswith("pluggable_component."):
                sub = target[len("pluggable_component."):]
                sp_found, sp_val = _walk_dotted_path(
                    (spec.get("comparison") or {}).get("pluggable_component") or {}, sub
                )
                if sp_found:
                    target_value, resolved = sp_val, True
            if not resolved and isinstance(target, str):
                plan_found, plan_val = _spec_or_mapping_value(spec, build_plan_merged, target)
                if plan_found:
                    target_value, resolved = plan_val, True
            if not found or value != target_value:
                errors.append(
                    f"arch_contract.{block_path}={value!r} must equal "
                    f"{target_value!r} (per spec-derived build plan)."
                )
            continue
        errors.extend(_check_required_block(contract, block_path, block_spec or {}))

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.spec.is_file():
        print(f"error: spec not found: {args.spec}", file=sys.stderr)
        return 2
    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"error: spec is not valid JSON ({e})", file=sys.stderr)
        return 2

    errors = validate(spec, args.run_dir)
    if errors:
        print(f"FAIL: {len(errors)} arch_contract validation error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return semantic_issue_exit_code(errors)
    print(f"ok: arch_contract.json at {args.run_dir} validates against schema + paradigm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
