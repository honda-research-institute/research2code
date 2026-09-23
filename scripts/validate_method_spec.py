"""Validate a method_spec.json against the canonical schema.

Two modes:

  Default — structural validation only (pydantic model_validate). Catches
  typos, missing fields, wrong types, unknown fields. Fast.

  --strict — additionally cross-checks the spec's pluggable_component
  signature against the matched taxonomy node
  (`pluggable_component.contract.forbidden_param_names`). Catches signatures
  that would break the harness's variadic-kwargs forwarding contract. It also
  rejects duplicate glossary/scale-lane paper-value carriers by exact
  declared identity and runs the spec-internal promise/contract consistency
  check: a function-kind
  system_provides promise that name-extends the pluggable component's single
  declared entrypoint must be declared on the contract surface itself, or it
  is an analyzer over-promise no Stage 2 producer can keep (the ADAM
  2026-07-21 genus: `optimize_adamax` promised beside a contract that routes
  the AdaMax variant through `optimize(..., variant=...)`).
  Methodology elements that declare `verification_probe_refs` are also checked
  by exact identity against the matched effective taxonomy node's closed
  `semantic_checks[].probe` set. Omitted refs remain valid for archived specs.

  --require-methodology-contract — additionally requires the Step 1
  methodology-fidelity fields introduced during the main/v3 integration:
  `methodology_replication_contract`, `methodology_contract_pack`, and
  `replication_feasibility`.

  --require-current-schema — requires the producer artifact to declare the
  exact schema version implemented by this checkout. The pipeline uses this
  for fresh Stage 1 output; omit it when replaying archived artifacts.

  NOTE: prior versions of --strict also enforced field-guide
  `required_baselines` coverage in `comparison.standard_baselines`. That
  check has been removed: v2 produces a single-method package + notebook
  with no baseline implementations, so coverage no longer makes sense.

Usage:

    python scripts/validate_method_spec.py path/to/method_spec.json
    python scripts/validate_method_spec.py path/to/method_spec.json --strict
    python scripts/validate_method_spec.py path/to/method_spec.json --require-methodology-contract
    python scripts/validate_method_spec.py path/to/method_spec.json --require-current-schema

Exit codes:
  0  valid
  1  schema validation failed
  2  strict / methodology-contract requirement failed
  3  file not found / unreadable / unparseable
"""

from __future__ import annotations

import argparse
import ast
import json
import keyword
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pydantic import ValidationError  # noqa: E402

from schemas.graph_wiring import derived_graph_probe_refs  # noqa: E402
from schemas.method_spec import (  # noqa: E402
    EvaluationProtocolQuantity,
    EvaluationProtocolRole,
    EvaluationProtocolScheme,
    MethodSpec,
    SCHEMA_VERSION,
    VERIFICATION_PROBE_REFS_SCHEMA_VERSION,
    evaluation_protocol_candidate_values,
    evaluation_protocol_label_matches,
)
from scripts import taxonomy  # noqa: E402
from scripts.signature_ast import (  # noqa: E402
    _parse_signature_string_param_names,
)

_CORE_METHODOLOGY_REQUIRED_DETAIL_FIELDS = (
    "forbidden_substitutions",
    "required_controls",
    "fairness_checks",
    "verification_expectations",
)


# ---------------------------------------------------------------------------
# Taxonomy contract cross-check
# ---------------------------------------------------------------------------


def cross_check_symbol_kinds_against_build_plan(
    spec: MethodSpec, spec_path: Path, repo_root: Path,
    provisional_packs_dir: Path | None = None,
) -> list[str]:
    """A promised symbol_kind must agree with the taxonomy build plan.

    bayesian-active-learning 2026-09-01: the spec promised `build_model`
    with symbol_kind='class' while the matched build plan declares
    `build_model` in method/training.py as kind='function' (a factory
    returning the model class). The architecture coder followed the build
    plan — its shape authority — so the kind re-check at 2.d halted the
    run on a contradiction NO producer could fix: two authorities over one
    symbol, visible in the spec alone, after a 2.b degrade and a full
    generation pass. Both surfaces resolve at Stage 1, so the mismatch
    fails here, where it costs one analyzer fix loop instead of a dead
    run.

    Conservative by construction: entries without symbol_kind, symbols the
    build plan does not declare, and specs without a derivable build plan
    are untouched (same posture as the 2.b naming bridge)."""
    try:
        from scripts.build_plan import load_build_plan  # noqa: PLC0415

        raw = json.loads(spec_path.read_text(encoding="utf-8"))
        plan = load_build_plan(
            raw, repo_root, provisional_packs_dir=provisional_packs_dir)
    except Exception:
        return []
    if not plan:
        return []
    plan_kinds: dict[str, tuple[str, str]] = {}
    manifest = plan.get("package_manifest") or {}
    for file_entry in manifest.get("files") or []:
        if not isinstance(file_entry, dict):
            continue
        for sym in file_entry.get("public_symbols") or []:
            if not isinstance(sym, dict):
                continue
            name, kind = sym.get("name"), sym.get("kind")
            if name and kind in ("class", "function"):
                plan_kinds.setdefault(
                    str(name), (str(kind), str(file_entry.get("path"))))
    errors: list[str] = []
    for entry in spec.try_it_out.system_provides:
        if entry.symbol is None or entry.symbol_kind is None:
            continue
        planned = plan_kinds.get(entry.symbol)
        if planned is None or planned[0] == entry.symbol_kind:
            continue
        plan_kind, plan_path = planned
        errors.append(
            f"try_it_out.system_provides entry {entry.name!r} declares "
            f"symbol_kind='{entry.symbol_kind}' for `{entry.symbol}`, but "
            f"the matched taxonomy build plan declares `{entry.symbol}` in "
            f"{plan_path} as kind='{plan_kind}'. The build plan is the "
            f"shape authority the stage-2 coders follow, so this promise "
            f"cannot be kept and would halt the run at the 2.d kind "
            f"re-check, after generation. Fix the SPEC: set "
            f"symbol_kind='{plan_kind}' for `{entry.symbol}`, or promise "
            f"the class surface under its own class name as a separate "
            f"entry if one is genuinely intended."
        )
    return errors


def cross_check_field_guide(
    spec: MethodSpec, repo_root: Path, provisional_packs_dir: Path | None = None
) -> list[str]:
    """Confirm the pluggable signature respects the matched taxonomy node's
    `pluggable_component.contract.forbidden_param_names`.

    Phase 4.3 removed the legacy field-guide path-existence gate; strict mode is
    keyed only by `comparison.classification.id`. Baseline-coverage cross-check
    was removed in v2. `provisional_packs_dir` lets a gap-path run's own
    provisional pack serve the classification (the serves() overlay design);
    without it a provisional classification correctly reads unserved."""
    paradigm_id = spec.comparison.classification.id
    if not paradigm_id:
        return ["spec.comparison.classification.id missing"]
    tax = taxonomy.load_taxonomy(repo_root, provisional_packs_dir=provisional_packs_dir)
    node = taxonomy.serves(paradigm_id, tax)
    if node is None:
        message = (
            f"comparison.classification.id {paradigm_id!r} is not served by "
            "docs/ssot/taxonomies.yaml"
        )
        # Truthfulness (SRL 2026-07-13): when a provisional pack exists but
        # the overlay skipped it, "not served" alone hides the load-bearing
        # fact — a pack was authored and rejected. Name each skip so the
        # halt stderr (and the judge reading it) sees the real mechanism.
        skips = [d for d in tax.diagnostics
                 if d.code.startswith("provisional_")
                 and d.code != "provisional_populates_reserved"]
        if provisional_packs_dir is not None and skips:
            details = "; ".join(f"{d.code}: {d.message}" for d in skips)
            message += (
                f" — provisional pack(s) were present but did NOT graft: "
                f"{details}"
            )
        return [message]
    contract_block = taxonomy.load_pluggable_component_contract(
        paradigm_id=paradigm_id, taxonomy=tax
    )
    source = f"the matched taxonomy node ({paradigm_id})"

    errors: list[str] = []

    # Pluggable-component contract — `forbidden_param_names`
    contract = contract_block.get("contract", {}) if isinstance(contract_block, dict) else {}
    forbidden = contract.get("forbidden_param_names", []) if isinstance(contract, dict) else []
    forbidden_set = {n for n in forbidden if isinstance(n, str)}
    if forbidden_set:
        spec_sig = spec.comparison.pluggable_component.signature
        spec_param_names = _parse_signature_string_param_names(spec_sig)
        if spec_param_names is None:
            errors.append(
                f"could not parse spec.comparison.pluggable_component.signature {spec_sig!r} "
                "to check against the `forbidden_param_names` contract"
            )
        else:
            hits = sorted(set(spec_param_names) & forbidden_set)
            if hits:
                errors.append(
                    f"spec.comparison.pluggable_component.signature declares forbidden "
                    f"parameter name(s) {hits!r}: {source} "
                    f"lists {sorted(forbidden_set)!r} in `pluggable_component.contract.forbidden_param_names`. "
                    f"Each paradigm-extra must be declared as a named keyword parameter with a "
                    f"default (e.g., `mc_samples: int = 100`) — never bundled into a `config` dict."
                )

    return errors


def _resolved_taxonomy_node(
    paradigm_id: str, tax: taxonomy.Taxonomy
) -> taxonomy.VariantNode | taxonomy.FamilyNode | None:
    """Resolve a populated classification in the forms the taxonomy accepts.

    Method specs normally carry the canonical legacy id, which ``serves``
    resolves (including run-local provisional overlays). Some runtime callers
    carry a registered variant slug, taxonomy id, or alias instead; the
    taxonomy's public ``variant`` lookup owns those identities. Never accept a
    reserved/unpopulated candidate.
    """
    node = taxonomy.serves(paradigm_id, tax)
    if node is not None:
        return node
    candidate = tax.variant(paradigm_id)
    if candidate is not None and candidate.is_populated:
        return candidate
    return None


def cross_check_verification_probe_refs(
    spec: MethodSpec,
    repo_root: Path,
    provisional_packs_dir: Path | None = None,
) -> list[str]:
    """Check methodology probe bindings against the effective taxonomy set.

    A methodology element may name only exact ``semantic_checks[].probe``
    values declared by its resolved node after inheritance and any run-local
    provisional overlay are applied. The field is optional, so a legacy spec
    with no declared refs incurs no taxonomy requirement here.
    """
    contract = spec.methodology_replication_contract
    if contract is None:
        return []

    def _version(value: object) -> tuple[int, ...]:
        parts = str(value or "").split(".")
        if not parts or any(not part.isdigit() for part in parts):
            return ()
        return tuple(int(part) for part in parts)

    errors: list[str] = []
    # R2C-087 activated this field at v1.12. Keep that boundary frozen when
    # later independent schema revisions advance SCHEMA_VERSION.
    reference_schema = _version(VERIFICATION_PROBE_REFS_SCHEMA_VERSION)
    if reference_schema and _version(spec.schema_version) >= reference_schema:
        for index, element in enumerate(contract.elements):
            if "verification_probe_refs" not in element.model_fields_set:
                errors.append(
                    "methodology_replication_contract.elements"
                    f"[{index}] {element.element_id!r} omits "
                    "verification_probe_refs under schema "
                    f"{spec.schema_version}; fresh reference-aware specs must "
                    "emit the field explicitly (use [] when no declared "
                    "probe genuinely verifies this obligation)"
                )
    if errors:
        return errors
    if not any(
        element.verification_probe_refs for element in contract.elements
    ):
        return []

    paradigm_id = spec.comparison.classification.id
    if not paradigm_id:
        return [
            "cannot validate methodology verification_probe_refs because "
            "spec.comparison.classification.id is missing"
        ]

    tax = taxonomy.load_taxonomy(
        repo_root, provisional_packs_dir=provisional_packs_dir
    )
    node = _resolved_taxonomy_node(paradigm_id, tax)
    if node is None:
        return [
            "cannot validate methodology verification_probe_refs because "
            f"comparison.classification.id {paradigm_id!r} does not resolve "
            "to a populated taxonomy node"
        ]

    if isinstance(node, taxonomy.VariantNode):
        checks = taxonomy.effective_node(tax, node.slug).get(
            "semantic_checks", []
        )
    else:
        checks = node.fields.get("semantic_checks", [])
    allowed_refs = {
        probe
        for check in checks
        if isinstance(check, dict)
        and isinstance((probe := check.get("probe")), str)
        and bool(probe)
    }

    # R2C-092: graph_mechanism.* refs on their owning elements are derived
    # from the homogeneous_graph_mechanism block, not authored by the
    # producer, so a taxonomy node that lacks those registrations is a
    # pipeline coverage gap and must not read as producer blame.
    derived_by_element = derived_graph_probe_refs(
        contract.homogeneous_graph_mechanism
    )
    for index, element in enumerate(contract.elements):
        unknown = sorted(set(element.verification_probe_refs) - allowed_refs)
        if not unknown:
            continue
        derived_here = set(derived_by_element.get(element.element_id, ()))
        producer_unknown = [ref for ref in unknown if ref not in derived_here]
        pipeline_unknown = [ref for ref in unknown if ref in derived_here]
        if producer_unknown:
            errors.append(
                "methodology_replication_contract.elements"
                f"[{index}] {element.element_id!r} declares unknown "
                f"verification_probe_refs {producer_unknown!r} for taxonomy "
                f"node {paradigm_id!r}; allowed exact refs are "
                f"{sorted(allowed_refs)!r}"
            )
        if pipeline_unknown:
            errors.append(
                "pipeline coverage gap (not a producer error): taxonomy node "
                f"{paradigm_id!r} does not register the graph probe(s) "
                f"{pipeline_unknown!r} that the declared "
                "homogeneous_graph_mechanism derives onto element "
                f"{element.element_id!r}; register the graph_mechanism "
                "semantic checks on the node (or its pack) before running "
                "this family"
            )
    return errors


# ---------------------------------------------------------------------------
# Spec-internal promise/contract consistency (the ADAM 2026-07-21 genus)
# ---------------------------------------------------------------------------


_IDENTIFIER_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _signature_string_variant_params(sig_str: str) -> list[str]:
    """Parameter names in a spec signature string whose declared default is a
    string literal — the machine-visible footprint of a contract that routes
    behavioral variants through a parameter (e.g. `variant: str = 'adam'`).
    Used only to enrich the consistency error message; a signature that does
    not parse yields [] and never fails anything here."""
    try:
        tree = ast.parse(f"def {sig_str}:\n    pass\n")
    except (SyntaxError, ValueError):
        return []
    if not tree.body or not isinstance(
        tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        return []
    args = tree.body[0].args
    names: list[str] = []
    positional = list(args.posonlyargs) + list(args.args)
    defaults = list(args.defaults)
    for arg, default in zip(positional[len(positional) - len(defaults):], defaults):
        if isinstance(default, ast.Constant) and isinstance(default.value, str):
            names.append(arg.arg)
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        if (
            default is not None
            and isinstance(default, ast.Constant)
            and isinstance(default.value, str)
        ):
            names.append(arg.arg)
    return names


def cross_check_promise_contract_consistency(spec: MethodSpec) -> list[str]:
    """Spec-internal cross-check: promised function symbols must be
    reconcilable with the spec's own pluggable-component contract.

    The ADAM 2026-07-21 case: the analyzer promised TWO function symbols
    (`optimize`, `optimize_adamax`) while its own pluggable_component
    contract declared ONE `optimize()` entrypoint with a
    `variant: str = 'adam'` kwarg. The method-coder correctly followed the
    contract, so the second promise was unkeepable — caught only by the 2.d
    naming-bridge re-check, after generation. Both surfaces are authored by
    the Stage 1 analyzer, so the contradiction is visible in the spec ALONE
    and belongs here, where failing costs minutes.

    Deterministic rule, shaped by the diff between that case and every
    green-run spec on record: an ADDITIONAL function-kind promise whose
    symbol name-extends the single declared entrypoint (`<entrypoint>_*`)
    claims a sibling variant of the entrypoint surface itself, so the
    contract must declare that symbol somewhere on its own surface (name,
    signature, or description). Absent that, the promise duplicates behavior
    the contract routes through the single entrypoint (typically via a
    variant parameter) and no Stage 2 producer can keep it. Helper promises
    with their own name stems (an `extract_entities` beside a `run_pipeline`
    entrypoint) are untouched; so are name-only promises, class promises,
    and legacy specs without symbol_kind — conservative by construction,
    same posture as the 2.b bridge (this check is the promise-SET
    counterpart of its per-symbol kind-awareness).
    """
    component = spec.comparison.pluggable_component
    entrypoint = component.name
    if not entrypoint.isidentifier() or keyword.iskeyword(entrypoint):
        # Placeholder/prose component names get no enforcement — the same
        # legacy tolerance the 2.b bridge grants invalid declared symbols.
        return []

    surface_tokens = set(
        _IDENTIFIER_TOKEN_RE.findall(
            " ".join((component.name, component.signature, component.description))
        )
    )
    variant_params = _signature_string_variant_params(component.signature)
    if variant_params:
        params = ", ".join(f"`{p}`" for p in variant_params)
        variant_hint = (
            f" — its string-valued parameter(s) {params} already look like "
            f"the variant switch"
        )
    else:
        variant_hint = ""

    errors: list[str] = []
    for entry in spec.try_it_out.system_provides:
        symbol = entry.symbol
        if symbol is None or entry.symbol_kind != "function":
            continue
        if symbol == entrypoint or not symbol.startswith(entrypoint + "_"):
            continue
        if symbol in surface_tokens:
            # The contract itself declares the sibling entrypoint (e.g. a
            # multi-entrypoint signature surface) — the promise is
            # reconcilable, nothing to enforce.
            continue
        errors.append(
            f"try_it_out.system_provides entry {entry.name!r} promises the "
            f"function symbol `{symbol}`, a name-extension of the pluggable "
            f"component's single declared entrypoint `{entrypoint}` that the "
            f"contract itself never declares (the symbol appears nowhere on "
            f"the component's name/signature/description surface). The spec "
            f"internally contradicts itself: the method-coder builds the "
            f"`{entrypoint}()` surface from the pluggable_component "
            f"contract{variant_hint}, so a separate promise for a variant "
            f"the contract handles through the single entrypoint is an "
            f"over-promise that cannot be kept — it would fail the "
            f"naming-bridge re-check at 2.d, after generation. Both surfaces "
            f"are authored by the Stage 1 analyzer; fix the spec, not the "
            f"package: either drop the `{symbol}` system_provides entry (the "
            f"capability stays reachable through `{entrypoint}`'s "
            f"parameters), or declare `{symbol}` as a second entrypoint on "
            f"the pluggable_component contract surface."
        )
    return errors


def cross_check_scenario_assumptions(
    spec: MethodSpec,
    repo_root: Path,
    provisional_packs_dir: Path | None = None,
) -> list[str]:
    """Join captured scenario assumptions to the matched node vocabulary.

    Slice A is visibility-only, so this checks declaration ownership and
    spelling without requiring detectors or comparing against a generated
    setup. A non-declaring family remains unchanged unless an analyzer
    incorrectly emits assumptions for it.
    """
    assumptions = spec.scenario_assumptions
    if not assumptions:
        return []

    paradigm_id = spec.comparison.classification.id
    tax = taxonomy.load_taxonomy(
        repo_root, provisional_packs_dir=provisional_packs_dir
    )
    dimensions = taxonomy.load_scenario_assumption_dimensions(
        paradigm_id, tax
    )
    declared_ids = {
        str(entry.get("id"))
        for entry in dimensions
        if isinstance(entry, dict) and entry.get("id")
    }
    if not declared_ids:
        return [
            "scenario_assumptions were captured, but the matched taxonomy "
            f"node ({paradigm_id}) declares no "
            "scenario_assumption_dimensions"
        ]

    unknown = sorted(set(assumptions) - declared_ids)
    if unknown:
        return [
            "scenario_assumptions contains dimension id(s) not declared by "
            f"the matched taxonomy node ({paradigm_id}): {unknown}. "
            f"declared ids are {sorted(declared_ids)}"
        ]
    return []


def cross_check_evaluation_protocol(
    spec: MethodSpec,
    repo_root: Path,
    provisional_packs_dir: Path | None = None,
) -> list[str]:
    """Require and role-check fresh temporal protocol carrier bindings.

    ``comparison.evaluation_protocol`` stays optional in Pydantic so legacy
    artifacts remain readable. Fresh strict validation uses the matched
    taxonomy node's ``params_derivation.protocol_role`` metadata as an
    independent authority: a spec cannot make ``forecast_horizon`` mean
    ``test_span`` merely by labeling it that way itself.
    """
    paradigm_id = spec.comparison.classification.id
    tax = taxonomy.load_taxonomy(
        repo_root, provisional_packs_dir=provisional_packs_dir
    )
    declarations = taxonomy.load_params_derivation(paradigm_id, tax)
    declared_roles = {
        name: str(entry.get("protocol_role"))
        for name, entry in declarations.items()
        if entry.get("protocol_role")
    }
    is_tsf = paradigm_id in {
        "time_series_forecasting",
        "graph_time_series_forecasting",
        "TE-TSF/time_series_forecasting",
    }
    protocol = spec.comparison.evaluation_protocol
    if protocol is None:
        if is_tsf or declared_roles:
            return [
                "comparison.evaluation_protocol is required for fresh "
                f"time-series output ({paradigm_id!r}); provide exactly one "
                "context_length, forecast_call_horizon, validation_span, "
                "and test_span quantity plus the evaluation scheme. Use "
                "paper_value_status='paper_unspecified' with value=null "
                "instead of promoting nearby notation."
            ]
        return []
    if not is_tsf and not declared_roles:
        return [
            "comparison.evaluation_protocol is present for non-forecasting "
            f"taxonomy node {paradigm_id!r}, which declares no temporal "
            "protocol roles. Remove the cross-family protocol block instead "
            "of publishing forecasting semantics for this method family."
        ]

    errors: list[str] = []
    bound = {
        quantity.parameter_name: quantity
        for quantity in protocol.quantities
        if quantity.parameter_name is not None
    }
    legal_roles = {role.value for role in EvaluationProtocolRole}
    for name, role in sorted(declared_roles.items()):
        if role not in legal_roles:
            errors.append(
                f"taxonomy params_derivation.{name}.protocol_role={role!r} "
                f"is invalid; expected one of {sorted(legal_roles)!r}"
            )
            continue
        quantity = bound.get(name)
        if quantity is None:
            errors.append(
                f"taxonomy params_derivation.{name}.protocol_role={role!r} "
                "declares a runtime protocol carrier, but no "
                "comparison.evaluation_protocol quantity binds "
                f"parameter_name={name!r}"
            )
            continue
        if quantity.role.value != role:
            errors.append(
                f"comparison.evaluation_protocol parameter_name={name!r} "
                f"binds role={quantity.role.value!r}, value={quantity.value!r}, "
                f"unit={quantity.unit!r}, granularity="
                f"{quantity.granularity!r}, but taxonomy "
                f"params_derivation.{name}.protocol_role={role!r}; a "
                "cross-role carrier join is forbidden"
            )

    for name, quantity in sorted(bound.items()):
        declared_role = declared_roles.get(name)
        if declared_role is None:
            errors.append(
                "comparison.evaluation_protocol binds parameter_name="
                f"{name!r} for role={quantity.role.value!r}, but the matched "
                f"taxonomy node ({paradigm_id!r}) has no corresponding "
                f"params_derivation.{name}.protocol_role declaration; do not "
                "invent validation/test runtime carriers"
            )
        if (
            quantity.paper_value_status.value == "paper_stated"
            and quantity.value is not None
        ):
            raw_steps = float(quantity.value) / float(quantity.granularity)
            rounded = round(raw_steps)
            if rounded <= 0 or abs(raw_steps - rounded) > 1e-9:
                errors.append(
                    "comparison.evaluation_protocol step-count binding is "
                    "not a positive integral physical-to-step conversion: "
                    f"parameter_name={name!r}, role={quantity.role.value!r}, "
                    f"value={quantity.value!r} {quantity.unit}, "
                    f"granularity={quantity.granularity!r} "
                    f"{quantity.unit}/step gives {raw_steps:g} steps. "
                    "Keep the carrier closed rather than rounding boundary "
                    "arithmetic."
                )
    return errors


def cross_check_parameter_carrier_exclusivity(spec: MethodSpec) -> list[str]:
    """Refuse duplicate paper-value carriers for one declared parameter.

    The scale lane and glossary have different consumers.  Join them only by
    the scale-lane entry's exact, case-sensitive, stripped ``name`` against
    the glossary entry's exact stripped ``name`` and declared ``aliases``.
    Values, formulas, token fragments, and numeric coincidence are never
    identity evidence.

    A glossary entry with ``paper_value=null`` is meaning-only and may sit
    beside the scale lane.  A non-null glossary value beside a matching scale
    entry leaves two paper-truth carriers, regardless of whether the lane
    carries the same value, a different value, or a formula.
    """
    errors: list[str] = []
    glossary = spec.critical_requirements.param_glossary
    scale_lane = spec.critical_requirements.scale_dependent_hyperparameters

    for lane_index, lane_entry in enumerate(scale_lane):
        lane_label = lane_entry.name.strip()
        if not lane_label:
            continue
        lane_root = (
            "critical_requirements.scale_dependent_hyperparameters"
            f"[{lane_index}]"
        )
        for glossary_index, glossary_entry in enumerate(glossary):
            if glossary_entry.paper_value is None:
                continue
            glossary_labels = [
                label.strip()
                for label in (glossary_entry.name, *glossary_entry.aliases)
                if label.strip()
            ]
            if lane_label not in glossary_labels:
                continue
            glossary_root = (
                f"critical_requirements.param_glossary[{glossary_index}]"
            )
            errors.append(
                "paper_value_carrier_duplication: "
                f"{lane_root}.name={lane_entry.name!r} exactly matches "
                f"{glossary_root} stripped labels={glossary_labels!r}; "
                f"{lane_root}.paper_value={lane_entry.paper_value!r}, "
                f"{lane_root}.formula={lane_entry.formula!r}, and "
                f"{glossary_root}.paper_value="
                f"{glossary_entry.paper_value!r} leave two paper-value "
                "carriers. Correction: a scale-free constant belongs in "
                "the glossary only; a data-scale-dependent parameter "
                "belongs in the scale-dependent lane only, with any "
                "glossary entry kept meaning-only (paper_value=null)."
            )
    return errors


def cross_check_parameter_carrier_consumability(spec: MethodSpec) -> list[str]:
    """Refuse populated glossary values with no exact runtime identity.

    R2C-064's duplicate-carrier check runs first and remains independent.
    This check addresses the later consumer seam only.  Descriptive
    scale-lane names retain legacy tolerance because that lane has older
    family-specific consumers and no alias field; the new refusal is scoped to
    glossary values, whose schema explicitly provides derivation aliases.

    A glossary entry whose labels join a bound evaluation-protocol carrier
    needs no alias of its own: the quantity's ``parameter_name`` IS the exact
    runtime identity, and R2C-081/082 derivation delivers the (schema-checked
    agreeing) value through it. Demanding a second runtime path for a value
    that already has the blessed one burned the 2026-08-11 attempt-6 fix
    loop on the pdfgnn 'context length' glossary entry.
    """
    protocol = spec.comparison.evaluation_protocol
    bound_quantities = [
        quantity
        for quantity in (protocol.quantities if protocol is not None else [])
        if isinstance(quantity.parameter_name, str)
        and quantity.parameter_name.strip()
    ]
    errors: list[str] = []
    glossary = spec.critical_requirements.param_glossary
    for glossary_index, glossary_entry in enumerate(glossary):
        if glossary_entry.paper_value is None:
            continue
        labels = [
            label.strip()
            for label in (glossary_entry.name, *glossary_entry.aliases)
            if label.strip()
        ]
        if any(
            label.isidentifier() and not keyword.iskeyword(label)
            for label in labels
        ):
            continue
        if any(
            evaluation_protocol_label_matches(quantity, label)
            for quantity in bound_quantities
            for label in labels
        ):
            continue
        errors.append(
            "paper_value_carrier_unnameable: "
            "critical_requirements.param_glossary"
            f"[{glossary_index}] carries paper_value="
            f"{glossary_entry.paper_value!r}, but its exact stripped "
            f"name and aliases {labels!r} expose no valid Python "
            "identifier. Declare one exact runtime alias so deterministic "
            "parameter derivation can consume the value."
        )
    return errors


def cross_check_paper_map(spec: MethodSpec, spec_path: Path) -> list[str]:
    """Confirm every paper-element ID the spec references resolves in paper_map.json.

    The spec's `core_method.key_elements` is a list of paper-element IDs that describe
    the method's core algorithm. Other code (architecture-coder, method-coder, reviewer)
    relies on these IDs being valid; an unresolved ID means the analyzer wrote a typo
    or referenced an element it never put in the paper_map.

    This is a Stage 1 integration check (T3): does spec's view of the paper match
    paper_map's catalog?
    """
    paper_map_path = spec_path.parent / "paper_map.json"
    if not paper_map_path.is_file():
        return [
            f"paper_map.json missing at {paper_map_path} — cannot cross-check spec's "
            f"paper-element ID references. Stage 1 must produce both method_spec.json "
            f"AND paper_map.json."
        ]

    try:
        paper_map = json.loads(paper_map_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"paper_map.json at {paper_map_path} is not valid JSON: {e}"]

    elements = paper_map.get("elements") or []
    elements_by_id = {
        el.get("id"): el
        for el in elements
        if isinstance(el, dict) and el.get("id")
    }
    valid_ids = set(elements_by_id)
    if not valid_ids:
        return [f"paper_map.json at {paper_map_path} has no `elements` with `id` fields"]

    errors: list[str] = []
    referenced = list(spec.core_method.key_elements or [])
    missing = [ref for ref in referenced if ref not in valid_ids]
    if missing:
        errors.append(
            f"spec.core_method.key_elements references paper-element ID(s) "
            f"{missing!r} that don't exist in paper_map.json (valid IDs: "
            f"{sorted(valid_ids)[:10]}{'…' if len(valid_ids) > 10 else ''}). "
            f"Either the analyzer wrote a typo or the paper_map is incomplete."
        )
    errors.extend(_check_contract_crosswalk(spec, valid_ids))
    errors.extend(_check_evaluation_protocol_crosswalk(spec, elements_by_id))
    errors.extend(_check_evaluation_protocol_reader_narratives(
        spec, list(elements_by_id.values())
    ))
    return errors


def _check_evaluation_protocol_reader_narratives(
    spec: MethodSpec, paper_elements: list[dict],
) -> list[str]:
    """Reject producer prose that republishes incompatible protocol truth.

    The typed protocol is the authority, but several reader surfaces copy
    analyzer/decomposer prose verbatim: ``core_method.summary`` reaches every
    package README and METHOD.md, while paper-map names/descriptions reach
    METHOD.md and REPORT.md.  Validating only the typed table would therefore
    leave a second, contradictory truth surface beside it.

    This gate intentionally uses the strict grammatical binder rather than
    the broad conflict-candidate scan.  A sentence saying that a K-step
    forecast is scored on a 26-week test set is not itself a claim that K=26;
    ``K = 26`` or ``forecast horizon is 26`` is.  Explicit demo/runtime prose
    is left to the parameter/output gates, which have the runtime value needed
    to adjudicate it.
    """
    protocol = spec.comparison.evaluation_protocol
    if protocol is None:
        return []

    records: list[tuple[str, str]] = [
        ("core_method.summary", spec.core_method.summary),
        ("paper_claims.method_description", spec.paper_claims.method_description),
        ("paper_claims.claimed_results", spec.paper_claims.claimed_results),
        ("paper_claims.benchmark_scale", spec.paper_claims.benchmark_scale),
        ("comparison.description", spec.comparison.description),
        (
            "comparison.pluggable_component.description",
            spec.comparison.pluggable_component.description,
        ),
    ]
    for index, element in enumerate(paper_elements):
        element_id = element.get("id", index)
        for field in ("name", "description"):
            value = element.get(field)
            if isinstance(value, str):
                records.append(
                    (f"paper_map.elements[{element_id!r}].{field}", value)
                )

    errors: list[str] = []
    for path, text in records:
        runtime_attribution = re.search(
            r"\b(?:demo|runtime|system|implementation|configured)\b"
            r"|\b(?:we|this (?:package|notebook|demo))\s+"
            r"(?:use|uses|set|sets)\b",
            text,
            flags=re.IGNORECASE,
        ) is not None
        for quantity in protocol.quantities:
            candidates = evaluation_protocol_candidate_values(quantity, text)
            if not candidates:
                continue
            value = quantity.value
            valid = (
                not runtime_attribution
                and quantity.paper_value_status.value == "paper_stated"
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and all(float(candidate) == float(value) for candidate in candidates)
            )
            if not valid:
                errors.append(
                    f"{path} publishes protocol role={quantity.role.value!r} "
                    f"numeric claim(s) {candidates!r} in {text!r}, but the "
                    f"typed paper fact is status="
                    f"{quantity.paper_value_status.value!r}, value={value!r}. "
                    "Reader-facing summaries copied into README.md, "
                    "METHOD.md, or REPORT.md may not contradict the typed "
                    "evaluation protocol."
                )
    return list(dict.fromkeys(errors))


def _protocol_scheme_records(
    scheme: object,
) -> list[tuple[str, str, list[str], str, str]]:
    return [
        (
            "comparison.evaluation_protocol.scheme",
            scheme.evidence_quote,
            scheme.paper_element_ids,
            scheme.paper_section,
            "paper_element_ids",
        ),
    ]


def _protocol_quantity_records(
    index: int, quantity: object,
) -> list[tuple[str, str, list[str], str, str]]:
    return [
        (
            f"comparison.evaluation_protocol.quantities[{index}]"
            f" (role={quantity.role.value!r})",
            quantity.evidence_quote,
            quantity.paper_element_ids,
            quantity.paper_section,
            "paper_element_ids",
        ),
        (
            f"comparison.evaluation_protocol.quantities[{index}]"
            f" (role={quantity.role.value!r}).axis_evidence",
            quantity.axis_evidence_quote,
            quantity.axis_paper_element_ids,
            quantity.axis_paper_section,
            "axis_paper_element_ids",
        ),
    ]


def _check_evaluation_protocol_crosswalk(
    spec: MethodSpec, elements_by_id: dict[str, dict],
) -> list[str]:
    """Every protocol evidence identity resolves to matching paper-map text."""
    protocol = spec.comparison.evaluation_protocol
    if protocol is None:
        return []
    records = _protocol_scheme_records(protocol.scheme)
    for index, quantity in enumerate(protocol.quantities):
        records.extend(_protocol_quantity_records(index, quantity))
    return _protocol_grounding_record_errors(records, elements_by_id)


def _protocol_grounding_record_errors(
    records: list[tuple[str, str, list[str], str, str]],
    elements_by_id: dict[str, dict],
) -> list[str]:
    errors: list[str] = []
    for path, quote, element_ids, section, id_field in records:
        missing = [
            ref for ref in element_ids if ref not in elements_by_id
        ]
        if missing:
            errors.append(
                f"{path}.{id_field} contains {missing!r} that do not "
                "exist in paper_map.json; role, value, unit, granularity, "
                "and evidence must reach the same paper identity"
            )
            continue
        # A quote may join contiguous passages with the elision marker
        # "[...]" (protocol evidence, 2026-08-10). Overlap is judged per
        # fragment: a joined quote is never a substring of any single
        # element, so the whole-string containment test structurally fails
        # every fragmented quote (pdfgnn Attempt 3 halt). Containment is
        # word-sequence based rather than byte based because paper-map
        # elements normalize table punctuation ('Context length: 10') while
        # evidence quotes keep the markdown pipes ('| Context length | 10 |')
        # — same verbatim words, different separators.
        def _word_seq(text: str) -> list[str]:
            return re.findall(r"[a-z0-9]+", text.casefold())

        def _longest_common_run(a: list[str], b: list[str]) -> int:
            if not a or not b:
                return 0
            best = 0
            prev = [0] * (len(b) + 1)
            for i in range(1, len(a) + 1):
                current = [0] * (len(b) + 1)
                for j in range(1, len(b) + 1):
                    if a[i - 1] == b[j - 1]:
                        current[j] = prev[j - 1] + 1
                        if current[j] > best:
                            best = current[j]
                prev = current
            return best

        def _grounds(fragment: list[str], source: list[str]) -> bool:
            # Paper-map elements are summaries with their own elisions, so
            # full containment fails correct citations (pdfgnn Attempt 4,
            # 2026-08-10: an element sharing an eight-word run with the
            # quote was rejected). A shared contiguous word run that is
            # long, or covers most of the shorter side, grounds the
            # citation; genuinely unrelated elements share only stray words.
            run = _longest_common_run(fragment, source)
            shorter = min(len(fragment), len(source))
            if shorter == 0:
                return False
            return run >= 4 or run >= 0.6 * shorter

        fragment_words = [
            _word_seq(fragment)
            for fragment in quote.split("[...]")
            if fragment.strip()
        ]
        for ref in element_ids:
            source_text = elements_by_id[ref].get("source_text")
            source_words = (
                _word_seq(source_text)
                if isinstance(source_text, str)
                else []
            )
            if not source_words or not any(
                _grounds(fragment, source_words)
                for fragment in fragment_words
            ):
                errors.append(
                    f"{path} cites paper_element_id={ref!r}, but that "
                    "element's verbatim source_text does not overlap the "
                    "record's evidence quote; a valid-but-wrong paper-map "
                    "ID cannot ground role, value, unit, or granularity. "
                    "Cite the element that covers the quoted passage, or "
                    "declare an empty list when the paper map covers none "
                    "of it (the honest state; it costs downstream "
                    "adjudication, never invents a link)"
                )
            element_section = elements_by_id[ref].get("section")
            def _section_parts(value: str) -> tuple[str | None, list[str], str]:
                folded = " ".join(value.casefold().split())
                kind_match = re.search(
                    r"\b(section|sec|figure|fig|table|appendix)\b",
                    folded,
                )
                kind = kind_match.group(1) if kind_match else None
                kind = {
                    "sec": "section", "fig": "figure",
                }.get(kind, kind)
                numbers = re.findall(r"\d+(?:\.\d+)*", folded)
                title = re.sub(
                    r"[^a-z]+", " ", re.sub(
                        r"\b(?:section|sec|figure|fig|table|appendix)\b|"
                        r"\d+(?:\.\d+)*",
                        " ",
                        folded,
                    )
                ).strip()
                return kind, numbers, " ".join(title.split())

            declared_kind, declared_numbers, declared_title = _section_parts(
                section
            )
            element_kind, element_numbers, element_title = _section_parts(
                element_section if isinstance(element_section, str) else ""
            )
            kinds_compatible = not (
                declared_kind and element_kind
                and declared_kind != element_kind
            )
            numbers_compatible = True
            if declared_numbers and element_numbers:
                numbers_compatible = any(
                    left == right
                    or left.startswith(right + ".")
                    or right.startswith(left + ".")
                    for left in declared_numbers
                    for right in element_numbers
                )
            titles_compatible = (
                declared_title == element_title
                if declared_title or element_title
                else True
            )
            compatible_section = bool(element_section) and kinds_compatible and (
                (declared_numbers and element_numbers and numbers_compatible)
                or titles_compatible
            )
            if not compatible_section:
                errors.append(
                    f"{path} declares paper section {section!r}, but cited "
                    f"paper_element_id={ref!r} belongs to "
                    f"{element_section!r}; evidence location and identity "
                    "must agree"
                )
    return errors


def best_effort_protocol_grounding_errors(
    raw_spec: dict, spec_path: Path,
) -> list[str]:
    """Protocol-grounding errors computable while the spec still fails schema.

    The grounding layer's inputs (one protocol record plus paper_map.json)
    are independently parseable per record, so the layer must not hide
    behind an unrelated schema failure: the 2026-08-11 pdfgnn attempt-3
    roll burned its whole fix budget on schema floors, and five grounding
    errors were revealed only after the last retry cleared them (error
    trajectory 2 -> 1 -> 1 -> 5). Scheme and quantities are validated
    individually here, keeping their original indices, so a quantity whose
    sibling still fails schema gets its citations checked in the same
    dispatch. Records that fail their own schema are skipped; their schema
    errors are already in the report this rides along with.
    """
    comparison = raw_spec.get("comparison")
    raw_protocol = (
        comparison.get("evaluation_protocol")
        if isinstance(comparison, dict) else None
    )
    if not isinstance(raw_protocol, dict):
        return []
    paper_map_path = spec_path.parent / "paper_map.json"
    if not paper_map_path.is_file():
        return []
    try:
        paper_map = json.loads(paper_map_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    elements_by_id = {
        el.get("id"): el
        for el in (paper_map.get("elements") or [])
        if isinstance(el, dict) and el.get("id")
    }
    if not elements_by_id:
        return []

    records: list[tuple[str, str, list[str], str, str]] = []
    raw_scheme = raw_protocol.get("scheme")
    if isinstance(raw_scheme, dict):
        try:
            scheme = EvaluationProtocolScheme.model_validate(raw_scheme)
        except ValidationError:
            pass
        else:
            records.extend(_protocol_scheme_records(scheme))
    raw_quantities = raw_protocol.get("quantities")
    for index, raw_quantity in enumerate(
        raw_quantities if isinstance(raw_quantities, list) else []
    ):
        if not isinstance(raw_quantity, dict):
            continue
        try:
            quantity = EvaluationProtocolQuantity.model_validate(raw_quantity)
        except ValidationError:
            continue
        records.extend(_protocol_quantity_records(index, quantity))
    if not records:
        return []
    return _protocol_grounding_record_errors(records, elements_by_id)


def _check_contract_crosswalk(spec: MethodSpec, valid_ids: set) -> list[str]:
    """Every methodology contract element's `paper_element_ids` must resolve.

    The crosswalk is what lets a behavioral finding about a piece of generated
    code reach the contract obligation that code implements (R2C-072). Generated
    code carries paper_map IDs; the contract carries its own IDs; nothing joined
    them until this field. An unresolvable ID here is worse than a missing one:
    it looks like a working link and binds nothing, which is exactly the silent
    failure the whole field exists to remove.

    The field is optional, so an empty crosswalk is not an error. It costs the
    element its contract adjudication and nothing else.
    """
    contract = getattr(spec, "methodology_replication_contract", None)
    if contract is None:
        return []
    errors: list[str] = []
    for element in contract.elements or []:
        missing = [ref for ref in (element.paper_element_ids or [])
                   if ref not in valid_ids]
        if missing:
            errors.append(
                f"methodology contract element {element.element_id!r} lists "
                f"paper_element_ids {missing!r} that don't exist in "
                f"paper_map.json (valid IDs: {sorted(valid_ids)[:10]}"
                f"{'…' if len(valid_ids) > 10 else ''}). This crosswalk is how "
                f"a behavioral finding on the generated code reaches this "
                f"contract element, so a dangling ID silently binds nothing."
            )
    return errors


def require_methodology_contract(spec: MethodSpec) -> list[str]:
    """Require the methodology-fidelity fields for new analyzer outputs.

    The Pydantic model keeps these fields optional so committed legacy
    artifacts and migration-era tests can still parse. This opt-in validator
    mode is the deterministic gate for contexts where a fresh Stage 1 analyzer
    output must carry the main/v3 methodology contract.
    """
    missing = []
    if spec.methodology_replication_contract is None:
        missing.append("methodology_replication_contract")
    if spec.methodology_contract_pack is None:
        missing.append("methodology_contract_pack")
    if spec.replication_feasibility is None:
        missing.append("replication_feasibility")
    if not missing:
        return []
    return [
        "methodology-fidelity contract required but missing field(s): "
        + ", ".join(missing)
    ]


def methodology_core_detail_errors(raw_spec: object) -> list[str]:
    """Return all empty support-detail fields on core methodology elements.

    Pydantic reports model-level validators one element at a time. For Stage 1
    fix-mode, that hides sibling empty fields and causes one-field-per-retry
    repair loops. This raw-JSON pass is intentionally narrow: it only expands
    the already-required core methodology detail invariant, leaving all other
    schema/type checks to Pydantic.
    """
    if not isinstance(raw_spec, dict):
        return []
    contract = raw_spec.get("methodology_replication_contract")
    if not isinstance(contract, dict):
        return []
    elements = contract.get("elements")
    if not isinstance(elements, list):
        return []

    errors: list[str] = []
    for index, element in enumerate(elements):
        if not isinstance(element, dict):
            continue
        if element.get("role") != "core_methodology":
            continue
        element_id = element.get("element_id")
        element_label = element_id if isinstance(element_id, str) else f"index-{index}"
        for field_name in _CORE_METHODOLOGY_REQUIRED_DETAIL_FIELDS:
            value = element.get(field_name)
            if not value:
                errors.append(
                    "methodology_replication_contract.elements."
                    f"{index}.{field_name}: core methodology element "
                    f"{element_label!r} requires non-empty {field_name}"
                )
    return errors


def _format_methodology_core_detail_errors(errors: list[str]) -> str:
    lines = [
        "methodology_replication_contract core detail completeness failed "
        f"({len(errors)} missing field(s)):"
    ]
    lines.extend(f"  - {error}" for error in errors)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_pydantic_errors(exc: ValidationError) -> str:
    lines = [f"schema validation failed ({exc.error_count()} error(s)):"]
    for err in exc.errors():
        loc = ".".join(str(x) for x in err["loc"])
        lines.append(f"  - {loc}: {err['msg']} [{err['type']}]")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a method_spec.json against the canonical schema.",
    )
    parser.add_argument("spec_path", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Cross-check against the matched taxonomy node.",
    )
    parser.add_argument(
        "--require-methodology-contract",
        action="store_true",
        help=(
            "Require methodology_replication_contract, methodology_contract_pack, "
            "and replication_feasibility. Use for fresh analyzer outputs; omit "
            "for legacy artifacts."
        ),
    )
    parser.add_argument(
        "--require-current-schema",
        action="store_true",
        help=(
            "Require schema_version to equal this checkout's current schema. "
            "Use for fresh analyzer outputs; omit for archived artifacts."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help="Repo root for resolving the taxonomy SSOT (default: this repo).",
    )
    parser.add_argument(
        "--provisional-packs-dir",
        type=Path,
        default=None,
        help=(
            "A run's .pipeline/provisional_packs/ directory. Installed packs "
            "are overlaid onto the taxonomy so a gap-path run's own "
            "provisional classification can pass the strict cross-check."
        ),
    )
    parser.add_argument(
        "--paper-md",
        type=Path,
        default=None,
        help=(
            "paper.md for verbatim evidence floors: every "
            "critical_requirements.param_glossary meaning_quote and every "
            "scenario_assumptions and comparison.evaluation_protocol "
            "evidence_quote must be a "
            "whitespace-normalized verbatim passage of the paper. Skipped "
            "when omitted for legacy validation."
        ),
    )
    args = parser.parse_args()

    if not args.spec_path.exists():
        print(f"error: file not found: {args.spec_path}", file=sys.stderr)
        return 3

    try:
        text = args.spec_path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"error: cannot read {args.spec_path}: {e}", file=sys.stderr)
        return 3

    try:
        raw_spec = json.loads(text)  # surface JSON syntax errors before pydantic
    except json.JSONDecodeError as e:
        print(f"error: {args.spec_path} is not valid JSON: {e}", file=sys.stderr)
        return 3

    core_detail_errors = methodology_core_detail_errors(raw_spec)
    try:
        spec = MethodSpec.model_validate_json(text)
    except ValidationError as exc:
        if core_detail_errors:
            print(_format_methodology_core_detail_errors(core_detail_errors), file=sys.stderr)
            print("", file=sys.stderr)
        print(_format_pydantic_errors(exc), file=sys.stderr)
        # Layered gates reveal one fix dispatch at a time: report the
        # grounding errors of every individually-parseable protocol record
        # alongside the schema errors, so a producer fixing a quote floor
        # sees in the same dispatch that its citations must move with it.
        # Strict-only, mirroring the gate that enforces grounding on a
        # schema-valid spec.
        grounding_errors = (
            best_effort_protocol_grounding_errors(raw_spec, args.spec_path)
            if args.strict
            else []
        )
        if grounding_errors:
            print("", file=sys.stderr)
            print(
                "strict: paper_map cross-check failures already visible on "
                "schema-valid protocol records (fix together with the "
                "schema errors above):",
                file=sys.stderr,
            )
            for error in grounding_errors:
                print(f"  - {error}", file=sys.stderr)
        return 1
    if core_detail_errors:
        print(_format_methodology_core_detail_errors(core_detail_errors), file=sys.stderr)
        return 1

    declared_schema_version = raw_spec.get("schema_version")
    if (
        args.require_current_schema
        and declared_schema_version != SCHEMA_VERSION
    ):
        print(
            "current-schema requirement failed: "
            "method_spec.json must explicitly declare schema_version "
            f"{SCHEMA_VERSION!r}, but its raw value is "
            f"{declared_schema_version!r}; "
            f"fresh analyzer output must declare {SCHEMA_VERSION!r}",
            file=sys.stderr,
        )
        return 2

    glossary = spec.critical_requirements.param_glossary
    scenario_assumptions = spec.scenario_assumptions or {}
    evaluation_protocol = spec.comparison.evaluation_protocol
    if (glossary or scenario_assumptions or evaluation_protocol) and args.paper_md \
            and args.paper_md.is_file():
        # The meaning-quote floor (param-glossary design 2026-07-21): the
        # same whitespace-normalized verbatim substring check the paper
        # map's equation floor uses. A paraphrase fails at extraction
        # time, never at a reader's trust surface.
        # All quote floors report together in one pass: a floor that
        # fail-fasts hides the next floor's failures and burns one fix
        # dispatch per revelation (SRL 2026-07-28: three iterations each
        # surfaced a new layer, and the oscillation judge halted the run
        # before the producer ever saw the full failure set).
        def _nws(t: str) -> str:
            return " ".join((t or "").split())
        haystack = _nws(args.paper_md.read_text(encoding="utf-8"))

        floor_failed = False
        quote_errors = [
            f"  - param_glossary[{entry.name!r}]: meaning_quote is not a "
            f"verbatim passage of the paper (whitespace-normalized "
            f"substring check): {_nws(entry.meaning_quote)[:100]!r}"
            for entry in glossary
            if _nws(entry.meaning_quote) not in haystack
        ]
        if quote_errors:
            print("param-glossary meaning-quote floor failed:",
                  file=sys.stderr)
            print("\n".join(quote_errors), file=sys.stderr)
            floor_failed = True

        protocol_quotes: list[tuple[str, str]] = []
        if evaluation_protocol is not None:
            protocol_quotes = [
                ("scheme.evidence_quote", evaluation_protocol.scheme.evidence_quote),
                *[
                    (
                        f"quantities[{index}] role={quantity.role.value!r}.evidence_quote",
                        quantity.evidence_quote,
                    )
                    for index, quantity in enumerate(
                        evaluation_protocol.quantities
                    )
                ],
                *[
                    (
                        f"quantities[{index}] role={quantity.role.value!r}.axis_evidence_quote",
                        quantity.axis_evidence_quote,
                    )
                    for index, quantity in enumerate(
                        evaluation_protocol.quantities
                    )
                ],
            ]
        # A protocol quote may join several contiguous passages with the
        # standard elision marker "[...]" (identity and value evidence for
        # one quantity can live in different paper locations, e.g. prose
        # notation plus an appendix table row). Each fragment must itself be
        # a verbatim passage; the marker never excuses a paraphrase.
        quote_errors = [
            f"  - comparison.evaluation_protocol.{path} "
            "is not a verbatim passage of the paper "
            "(whitespace-normalized substring check, applied to each "
            "'[...]'-separated fragment): "
            f"{_nws(fragment)[:100]!r}"
            for path, quote in protocol_quotes
            for fragment in _nws(quote).split("[...]")
            if _nws(fragment) not in haystack
        ]
        if quote_errors:
            print(
                "evaluation-protocol evidence-quote floor failed:",
                file=sys.stderr,
            )
            print("\n".join(quote_errors), file=sys.stderr)
            floor_failed = True

        quote_errors = [
            f"  - scenario_assumptions[{dimension_id!r}]: evidence_quote "
            f"is not a verbatim passage of the paper "
            f"(whitespace-normalized substring check): "
            f"{_nws(entry.evidence_quote)[:100]!r}"
            for dimension_id, entry in scenario_assumptions.items()
            if _nws(entry.evidence_quote) not in haystack
        ]
        if quote_errors:
            print(
                "scenario-assumption evidence-quote floor failed:",
                file=sys.stderr,
            )
            print("\n".join(quote_errors), file=sys.stderr)
            floor_failed = True

        if floor_failed:
            return 1

    print(
        f"ok: {args.spec_path} validates against schema v{spec.schema_version} "
        f"(paradigm={spec.comparison.classification.id})"
    )

    if args.strict:
        # Spec-internal consistency first: it needs no taxonomy or sibling
        # artifact, and a spec that contradicts itself should fail on its
        # own terms before any cross-artifact check gets a say (ADAM
        # 2026-07-21: this exact contradiction previously survived to the
        # 2.d naming-bridge re-check, after a full generation pass).
        errors = cross_check_promise_contract_consistency(spec)
        if errors:
            print(
                "strict: promise/contract consistency cross-check failed:",
                file=sys.stderr,
            )
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            return 2
        print("strict: promise/contract consistency cross-check passed")

        errors = cross_check_evaluation_protocol(
            spec,
            args.repo_root,
            provisional_packs_dir=args.provisional_packs_dir,
        )
        if errors:
            print(
                "strict: evaluation-protocol cross-check failed:",
                file=sys.stderr,
            )
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            return 2
        if spec.comparison.evaluation_protocol is not None:
            print("strict: evaluation-protocol cross-check passed")

        errors = cross_check_parameter_carrier_exclusivity(spec)
        if errors:
            print(
                "strict: parameter-carrier exclusivity cross-check failed:",
                file=sys.stderr,
            )
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            return 2
        print("strict: parameter-carrier exclusivity cross-check passed")

        errors = cross_check_parameter_carrier_consumability(spec)
        if errors:
            print(
                "strict: parameter-carrier consumability cross-check failed:",
                file=sys.stderr,
            )
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            return 2
        print("strict: parameter-carrier consumability cross-check passed")

        errors = cross_check_field_guide(
            spec, args.repo_root, provisional_packs_dir=args.provisional_packs_dir
        )
        if errors:
            print("strict: taxonomy contract cross-check failed:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            return 2
        print("strict: taxonomy contract cross-check passed")

        errors = cross_check_symbol_kinds_against_build_plan(
            spec, args.spec_path, args.repo_root,
            provisional_packs_dir=args.provisional_packs_dir,
        )
        if errors:
            print(
                "strict: symbol-kind build-plan cross-check failed:",
                file=sys.stderr,
            )
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            return 2
        print("strict: symbol-kind build-plan cross-check passed")

        errors = cross_check_verification_probe_refs(
            spec,
            args.repo_root,
            provisional_packs_dir=args.provisional_packs_dir,
        )
        if errors:
            print(
                "strict: methodology verification-probe cross-check failed:",
                file=sys.stderr,
            )
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            return 2
        if spec.methodology_replication_contract is not None and any(
            element.verification_probe_refs
            for element in spec.methodology_replication_contract.elements
        ):
            print(
                "strict: methodology verification-probe cross-check passed"
            )

        errors = cross_check_scenario_assumptions(
            spec,
            args.repo_root,
            provisional_packs_dir=args.provisional_packs_dir,
        )
        if errors:
            print(
                "strict: scenario-assumption taxonomy cross-check failed:",
                file=sys.stderr,
            )
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            return 2
        if spec.scenario_assumptions:
            print(
                "strict: scenario-assumption taxonomy cross-check passed"
            )

        errors = cross_check_paper_map(spec, args.spec_path)
        if errors:
            print("strict: paper_map cross-check failed:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            return 2
        print("strict: paper_map cross-check passed")

    if args.require_methodology_contract:
        errors = require_methodology_contract(spec)
        if errors:
            print("methodology-contract requirement failed:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            return 2
        print("methodology-contract requirement passed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
