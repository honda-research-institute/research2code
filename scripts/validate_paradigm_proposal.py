#!/usr/bin/env python3
"""Run gate on the stage-1 gap path (imported via the authoring driver).
Not repo-hygiene despite the neighbors.

Validate a candidate taxonomy-pack proposal packet.

Proposal directories are run artifacts under
`<run_dir>/.pipeline/paradigm_proposals/<proposal_id>/`. This validator checks
that a candidate pack can be reviewed without first copying it into the SSOT.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from schemas.paradigm_proposal import (  # noqa: E402
    BUILD_PLAN_SOURCE_INHERIT,
    BUILD_PLAN_SOURCE_NEUTRAL,
    BuildPlanChoice,
    FamilyComponentDeclaration,
    ParadigmProposal,
    ProposalValidationResult,
)
from scripts import taxonomy  # noqa: E402


def validate_proposal(
    proposal_dir: Path,
    repo_root: Path = ROOT,
    *,
    allow_update_existing: bool = False,
) -> ProposalValidationResult:
    proposal_dir = proposal_dir.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    proposal: ParadigmProposal | None = None

    proposal_path = proposal_dir / "proposal.json"
    pack_path = proposal_dir / "pack.yaml"

    if not proposal_dir.is_dir():
        errors.append(f"proposal directory not found: {proposal_dir}")
        return _result(proposal_dir, None, errors, warnings)

    if not proposal_path.is_file():
        errors.append("proposal.json missing")
    else:
        try:
            proposal_raw = json.loads(proposal_path.read_text(encoding="utf-8"))
            proposal = ParadigmProposal.model_validate(proposal_raw)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"proposal.json invalid: {exc}")

    pack: dict[str, Any] | None = None
    if not pack_path.is_file():
        errors.append("pack.yaml missing")
    else:
        try:
            raw_pack = yaml.safe_load(pack_path.read_text(encoding="utf-8")) or {}
            if not isinstance(raw_pack, dict):
                errors.append("pack.yaml must contain a mapping")
            else:
                pack = raw_pack
        except Exception as exc:  # noqa: BLE001
            errors.append(f"pack.yaml invalid: {exc}")

    if errors:
        return _result(proposal_dir, proposal, errors, warnings)

    assert proposal is not None and pack is not None

    proposal = proposal.model_copy(update={"pack": pack})
    _check_pack_alignment(proposal, pack, errors)
    _check_target_taxonomy(repo_root, proposal, errors, warnings, allow_update_existing)
    _check_pack_shape(pack, errors, warnings, decision=proposal.decision)

    return _result(proposal_dir, proposal, errors, warnings)


def _check_pack_alignment(
    proposal: ParadigmProposal,
    pack: dict[str, Any],
    errors: list[str],
) -> None:
    if pack.get("legacy_paradigm") != proposal.target_paradigm_id:
        errors.append(
            "pack.yaml legacy_paradigm does not match proposal "
            f"target_paradigm_id: {pack.get('legacy_paradigm')!r} != "
            f"{proposal.target_paradigm_id!r}"
        )
    if pack.get("extends") != proposal.extends:
        errors.append(
            "pack.yaml extends does not match proposal extends: "
            f"{pack.get('extends')!r} != {proposal.extends!r}"
        )
    if proposal.target_taxonomy_id and pack.get("taxonomy_id") != proposal.target_taxonomy_id:
        errors.append(
            "pack.yaml taxonomy_id does not match proposal target_taxonomy_id: "
            f"{pack.get('taxonomy_id')!r} != {proposal.target_taxonomy_id!r}"
        )


def _check_target_taxonomy(
    repo_root: Path,
    proposal: ParadigmProposal,
    errors: list[str],
    warnings: list[str],
    allow_update_existing: bool,
) -> None:
    tax = taxonomy.load_taxonomy(repo_root)
    if tax.node_for_legacy(proposal.target_paradigm_id) is not None and not allow_update_existing:
        errors.append(f"target taxonomy legacy_paradigm already exists: {proposal.target_paradigm_id}")
    elif tax.node_for_legacy(proposal.target_paradigm_id) is not None:
        warnings.append(f"target taxonomy legacy_paradigm exists and would be updated: {proposal.target_paradigm_id}")
    if proposal.decision == "new_subparadigm_needed":
        if proposal.extends is None or tax.node_for_legacy(proposal.extends) is None:
            errors.append(f"extends parent not found in taxonomy: {proposal.extends}")
    if proposal.decision == "new_top_level_needed" and proposal.extends is not None:
        errors.append("new top-level proposal must not extend an existing taxonomy node")


def _check_pack_shape(
    pack: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    *,
    decision: str | None = None,
) -> None:
    if pack.get("status") != "provisional":
        errors.append("pack.yaml status must be provisional")
    if not isinstance(pack.get("taxonomy_id"), str) or not pack.get("taxonomy_id"):
        errors.append("pack.yaml taxonomy_id missing")
    # Placeholder values fail exactly like missing ones — the 2026-07-06
    # fedavg pack shipped a literal 'TODO: describe the pluggable
    # interface' hint that passed the old emptiness check, read as a
    # carried contract, and died three stages later. Same predicate as the
    # overlay's populated-status rule so the two floors cannot drift.
    from taxonomy import is_placeholder_contract_value

    fingerprint = pack.get("fingerprint")
    if not isinstance(fingerprint, dict) or is_placeholder_contract_value(
            fingerprint.get("what_it_is")):
        errors.append("pack.yaml fingerprint.what_it_is missing or a "
                      "placeholder (write the real method fingerprint)")
    hints = pack.get("scaffold_hints")
    if not isinstance(hints, dict) or is_placeholder_contract_value(
            hints.get("interface_hint")):
        errors.append("pack.yaml scaffold_hints.interface_hint missing or a "
                      "placeholder (write the real pluggable interface, e.g. "
                      "'train_federated(model, clients, rounds, seed) -> "
                      "TrainedModel')")
    # The scaffolder interpolates the hint INSIDE a generated module's
    # docstring, so a hint carrying its own triple-quoted docstring
    # terminates that docstring early and the module no longer parses (the
    # DomIndOnto 2026-07-21 stage-2a halt: signature plus a full NumPy-style
    # docstring shipped as one hint). Hard fail, same floor philosophy as
    # the placeholder check — the authoring retry loop is where this gets
    # fixed, not three stages later in a live run.
    elif any(terminator in str(hints.get("interface_hint"))
             for terminator in ('"""', "'''")):
        errors.append("pack.yaml scaffold_hints.interface_hint contains a "
                      "docstring terminator (\"\"\" or ''') and cannot be "
                      "safely interpolated into the scaffolded module's "
                      "docstring (write the signature only — multi-line is "
                      "fine — and drop the embedded docstring)")
    for key in ("semantic_checks", "smoke_bugs"):
        if key in pack and not isinstance(pack[key], list):
            errors.append(f"pack.yaml {key} must be a list when present")
    if not pack.get("semantic_checks") and not pack.get("smoke_bugs"):
        warnings.append("pack.yaml has no semantic_checks or smoke_bugs yet")
    _check_retired_arch_contract_schema(pack, errors)
    _check_family_components(pack, errors, decision=decision)
    _check_build_plan(pack, errors, warnings, decision=decision)
    _check_params_derivation(pack, errors, warnings)


def _check_retired_arch_contract_schema(
    pack: dict[str, Any], errors: list[str]
) -> None:
    """Refuse a pack that declares `scaffold_hints.arch_contract_schema`.

    R2C-049. That block is retired, and a pack carrying it is actively
    dangerous rather than merely redundant. Nothing in the pipeline validates
    or consumes it: the build plan's real requirements come from
    `scripts/build_plan.py` (deliberately minimal for provisional packs), and
    the committed-node schema declared the block superseded by the
    spec-derived build plan. Its only readers are model eyes, through the
    producer role view.

    The cost of that is what makes this a hard fail. The universal
    `ArchContract` skeleton is `extra='forbid'` on every model, so a block
    demanding fields the skeleton does not declare asks the architecture coder
    for a contract pydantic will reject. It cannot satisfy both. The
    2026-08-03 probabilistic-demand-forecasting run degraded at stage 2b with
    12 `extra_forbidden` errors for exactly this reason, and the coder
    followed the pack over its own prompt rule, because a pack is presented to
    it as the paradigm's authority.

    Refusing at authoring time is cheap: the gap path's authoring loop feeds
    validation errors back to the author for up to three iterations, so the
    refusal names the surviving surface for each kind of content instead of
    dead-ending a gap run."""
    hints = pack.get("scaffold_hints")
    if not isinstance(hints, dict) or "arch_contract_schema" not in hints:
        return
    errors.append(
        "pack.yaml scaffold_hints.arch_contract_schema is RETIRED and must be "
        "removed (R2C-049). Nothing in the pipeline reads it, and a block "
        "demanding fields the universal ArchContract skeleton does not declare "
        "hands the architecture coder a contract pydantic rejects "
        "(extra='forbid' on every model), which is unsatisfiable. Express the "
        "same needs on the surfaces that have real consumers: declare "
        "family-specific top-level contract blocks under `family_components` "
        "(typed, and enforced in both directions at stage 2.d); declare "
        "training facts such as optimizer, loss, and epochs under `priors` "
        "and `params_derivation`; and declare the pluggable's signature and "
        "return type under `pluggable_component.signature_template` / "
        "`return_type`, where they already live."
    )


def _check_params_derivation(
    pack: dict[str, Any], errors: list[str], warnings: list[str]
) -> None:
    """Type-check the pack's `params_derivation` declaration at authoring time.

    Plan item 9 (2026-07-21): this block is what lets the deterministic
    parameter deriver satisfy the pack's own stage_2x_params review checks
    (the DomIndOnto config-paths halt and the fedavg derived-statistic
    halt). A malformed declaration must fail inside the proposal-authoring
    retry loop, not as an unresolvable 2.x reviewer halt in a live run.
    Additionally: a pack that declares a stage_2x_params semantic check
    but no params_derivation gets a warning — that combination is exactly
    the producer-blind review focus that halted both 2026-07-21 runs."""
    block = pack.get("params_derivation")
    if block is not None:
        if not isinstance(block, dict):
            errors.append(
                "pack.yaml params_derivation must be a mapping of param name "
                "-> {kind: config_path|derived_statistic|suppress, ...} when "
                "present"
            )
            return
        for name, entry in block.items():
            if not (isinstance(name, str) and name.isidentifier()):
                errors.append(
                    f"pack.yaml params_derivation key {name!r} must be a "
                    "python-identifier param name (it becomes a params.json key)"
                )
                continue
            if not isinstance(entry, dict):
                errors.append(
                    f"pack.yaml params_derivation.{name} must be a mapping "
                    "with a `kind`")
                continue
            kind = entry.get("kind")
            if kind == "config_path":
                if not entry.get("reasoning"):
                    errors.append(
                        f"pack.yaml params_derivation.{name} (config_path) "
                        "must carry `reasoning` — what the path configures "
                        "and where the method consumes it")
            elif kind == "derived_statistic":
                if not entry.get("formula"):
                    errors.append(
                        f"pack.yaml params_derivation.{name} "
                        "(derived_statistic) must carry `formula`")
                inputs = entry.get("inputs")
                if not isinstance(inputs, dict) or not inputs:
                    errors.append(
                        f"pack.yaml params_derivation.{name} "
                        "(derived_statistic) must carry a non-empty `inputs` "
                        "mapping of formula symbol -> `params.<name>` or "
                        "`spec.<dotted.path>`")
                else:
                    for sym, ref in inputs.items():
                        if not str(ref or "").startswith(("params.", "spec.")):
                            errors.append(
                                f"pack.yaml params_derivation.{name} input "
                                f"{sym!r} must reference `params.<name>` or "
                                f"`spec.<dotted.path>`, got {ref!r}")
            elif kind == "suppress":
                if not entry.get("reason"):
                    errors.append(
                        f"pack.yaml params_derivation.{name} (suppress) must "
                        "carry `reason`")
            else:
                errors.append(
                    f"pack.yaml params_derivation.{name} kind {kind!r} is not "
                    "one of ['config_path', 'derived_statistic', 'suppress']")
    _check_2x_checks_have_producers(pack, errors, warnings)


# Snake_case tokens that appear in pack prose as pipeline surface names,
# spec paths, or params-entry vocabulary rather than as parameter names.
# Without this guard the named-parameter arm below would demand a
# params_derivation entry for a check that merely says where its evidence
# lives or which provenance label it expects — for example the common
# hygiene check "values pulled from conventions are source:
# system_inferred", which requires no parameter at all.
_NON_PARAM_SNAKE_TOKENS = frozenset({
    # pipeline and spec surfaces
    "arch_contract", "critical_requirements", "family_components",
    "load_data", "method_spec", "paper_map", "param_glossary",
    "params_derivation", "pluggable_component", "return_type",
    "scale_dependent_hyperparameters", "semantic_checks",
    "signature_template", "silent_failure", "train_test_split",
    # params.json entry vocabulary: provenance labels and field names
    "code_binding", "paper_says", "paper_section", "paper_value",
    "spec_default", "system_default", "system_inferred", "unused_reason",
    "used_in_notebook",
})

_SNAKE_TOKEN = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")

# Parameter names the deterministic deriver emits from its own generic
# readers, with no pack declaration needed. Grounded in the literal
# `params[...]` assignments and the spec-path registry in
# scripts/derive_params.py, so a pack may reference any of these in a check
# without declaring a producer for it. Listed generously on purpose: an
# extra name here only costs some of this floor's teeth, while a missing
# name would false-fail a pack whose check is genuinely satisfiable.
_GENERICALLY_DERIVED_PARAMS = frozenset({
    "batch_returns", "batch_size", "dropout_rate", "hidden_dim",
    "initial_labeled", "learning_rate", "max_epochs", "mc_samples",
    "n_test", "num_epochs", "num_rounds", "pool_size", "seed",
    "total_budget", "train_size", "train_until_accuracy",
})


def _named_params_in_check(text: str) -> list[str]:
    """Parameter-looking identifiers a semantic check names, in order."""
    seen: list[str] = []
    for token in _SNAKE_TOKEN.findall(text or ""):
        if token in _NON_PARAM_SNAKE_TOKENS or token in seen:
            continue
        seen.append(token)
    return seen


def _pack_signature_names(pack: dict[str, Any]) -> set[str]:
    """Identifiers the pack's OWN declared interface carries.

    A keyword extra on the pluggable signature is derived from that
    signature by `_add_paradigm_extras`, so a pack whose interface declares
    `learning_rate: float = 0.001` has a real producer for it and must not
    be asked to declare one again (the fedavg shape).
    """
    texts = [
        str((pack.get("scaffold_hints") or {}).get("interface_hint") or ""),
        str((pack.get("pluggable_component") or {}).get(
            "signature_template") or ""),
    ]
    names: set[str] = set()
    for text in texts:
        names.update(_SNAKE_TOKEN.findall(text))
    return names


def _check_2x_checks_have_producers(
    pack: dict[str, Any], errors: list[str], warnings: list[str]
) -> None:
    """A stage_2x_params check needs a declared producer for what it demands.

    R2C-055 (maintainer-approved 2026-08-05). This combination previously
    emitted a non-blocking warning and went on to halt three live runs at
    the 2.x review on entries no producer could emit: two on 2026-07-21
    (the DomIndOnto config-paths halt and the fedavg derived-statistic
    halt) and pdfgnn on 2026-08-05, whose pack demanded similarity_cutoff
    in (0,1) with no params_derivation block at all. The install-time
    warning predicted that halt verbatim 74 minutes before it happened.

    Refusing at authoring keeps the fix inside the proposal retry loop
    instead of surfacing three stages into a live run, which is the policy
    already approved for the family-components floor (R2C-032) and for an
    unsatisfiable architecture contract (R2C-049).

    Two arms:

    - Declaration floor, the R2C-032 shape: a pack with a stage_2x_params
      check must DECLARE `params_derivation`. An explicit empty mapping is
      the deliberate "my checks need no specific parameter entries"
      answer; absence is an authoring error.
    - Named-parameter floor: a check that NAMES a parameter must have a
      producer for that name. This is what stops an empty mapping from
      being a free pass for a check that names its own requirement. A
      check naming no parameter keeps the declaration floor only.

    A name counts as having a producer when the pack declares it under
    `params_derivation` (any kind, including `suppress` — dropping a param
    is a declared decision), when the deriver's generic readers emit it
    anyway, or when the pack's own pluggable signature carries it as a
    keyword extra. Those last two are why fedavg's check may reference
    `learning_rate` while declaring only its five bespoke params.
    """
    block = pack.get("params_derivation")
    declared_names = set(block) if isinstance(block, dict) else set()
    produced_names = (
        declared_names
        | _GENERICALLY_DERIVED_PARAMS
        | _pack_signature_names(pack)
    )
    checks = [
        c for c in pack.get("semantic_checks") or []
        if isinstance(c, dict) and c.get("stage") == "stage_2x_params"
    ]
    if not checks:
        return
    if block is None:
        errors.append(
            "pack.yaml declares a stage_2x_params semantic check but no "
            "params_derivation block. The 2.x review halts on entries no "
            "producer can emit, so declare the parameter entries the check "
            "requires (kind: config_path for a paper-stated or configured "
            "value, derived_statistic for a computed one, suppress to drop "
            "one deliberately), or declare `params_derivation: {}` to state "
            "that these checks need no specific parameter entries"
        )
    for check in checks:
        check_id = str(check.get("id") or "<unnamed>")
        named = _named_params_in_check(str(check.get("check") or ""))
        missing = [n for n in named if n not in produced_names]
        if not missing:
            continue
        errors.append(
            f"pack.yaml semantic check {check_id} (stage_2x_params) requires "
            f"parameter(s) {', '.join(missing)} that params_derivation does "
            f"not declare, so the deterministic deriver has no instruction "
            f"to emit them and the 2.x review will halt on their absence. "
            f"Declare each one under params_derivation (kind: config_path "
            f"for a paper-stated or configured value, derived_statistic for "
            f"a computed one, suppress to drop it deliberately), or reword "
            f"the check so it does not require a parameter entry"
        )


def _check_family_components(
    pack: dict[str, Any], errors: list[str], *, decision: str | None = None
) -> None:
    """Type-check the pack's `family_components` declaration at authoring time.

    The declared names become the ONLY legal `arch_contract.family_components`
    keys the Stage 2.d static validator accepts (arch-contract headroom,
    approved 2026-07-16 — the SRL 2026-07-15 reward_function halts). Failing
    a malformed declaration here keeps the fix inside the proposal-authoring
    retry loop instead of surfacing three stages later in a live run.

    R2C-032 (approved 2026-07-28): every new-subparadigm pack must DECLARE
    the family components its interface implies — absence is an authoring
    error, an explicit empty mapping (`family_components: {}`) is the
    deliberate "the universal skeleton covers this method" answer. The
    2026-07-28 SRL run cleared classification on a pack with no declaration
    and died at stage 2b when the contract's reward_function was rejected
    as undeclared; the fix belongs in the authoring loop, not three stages
    into a live run. Installed packs are untouched — this floor runs only
    when a proposal is (re)validated. Top-level packs stay silent here (the
    design rule is scoped to sub-paradigms; a brand-new family may honestly
    not know its components yet)."""
    components = pack.get("family_components")
    if components is None:
        if decision == "new_subparadigm_needed":
            errors.append(
                "pack.yaml must declare family_components for a new "
                "sub-paradigm: the top-level arch-contract components its "
                "interface implies (e.g. a value-learning RL pack declares "
                "reward_function), or an explicit empty mapping "
                "(`family_components: {}`) when the universal contract "
                "skeleton fully covers the method. An undeclared surface "
                "rejects every family-specific component at stage 2.b/2.d "
                "(the 2026-07-28 SRL halt)."
            )
        return
    if not isinstance(components, dict):
        errors.append(
            "pack.yaml family_components must be a mapping of component name "
            "-> declaration ({required, description, required_entries}) when "
            "present"
        )
        return
    for name, declaration in components.items():
        if not (isinstance(name, str) and name.isidentifier()):
            errors.append(
                f"pack.yaml family_components key {name!r} must be a "
                "python-identifier component name (it becomes a top-level "
                "arch_contract.family_components key)"
            )
            continue
        try:
            FamilyComponentDeclaration.model_validate(declaration or {})
        except Exception as exc:  # noqa: BLE001 — pydantic.ValidationError
            errors.append(f"pack.yaml family_components.{name} invalid: {exc}")


def _check_build_plan(
    pack: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    *,
    decision: str | None = None,
) -> None:
    """Type-check the pack's `build_plan` routing choice at authoring time.

    R2C-032 (approved 2026-07-28): the pack author chooses between
    inheriting the parent's static build plan and taking the neutral
    provisional plan. The choice must be explicit for a new sub-paradigm —
    the 2026-07-28 SRL run inherited the motion-planning manifest
    implicitly and halted at stage 2b on classes an RL method cannot have.
    A `neutral` choice caps the delivery label at uncertified — new
    territory (one honest certification tier for opting out of family
    conventions); `inherit_parent` keeps the full range. Installed packs
    with no declaration keep today's implicit ancestor walk — this floor
    runs only inside the proposal-authoring loop."""
    block = pack.get("build_plan")
    if block is None:
        if decision == "new_subparadigm_needed":
            errors.append(
                "pack.yaml must declare build_plan.source for a new "
                "sub-paradigm: `inherit_parent` when the parent family's "
                "package manifest genuinely describes this method's build "
                "shape, or `neutral` to take the generic provisional plan "
                "(which caps the delivery label at uncertified — new "
                "territory). The 2026-07-28 SRL stage-2b halt is what an "
                "unexamined inherit costs."
            )
        return
    if not isinstance(block, dict):
        errors.append(
            "pack.yaml build_plan must be a mapping "
            "({source: inherit_parent | neutral, reasoning}) when present"
        )
        return
    try:
        choice = BuildPlanChoice.model_validate(block)
    except Exception as exc:  # noqa: BLE001 — pydantic.ValidationError
        errors.append(f"pack.yaml build_plan invalid: {exc}")
        return
    if choice.source == BUILD_PLAN_SOURCE_INHERIT:
        if not pack.get("extends"):
            errors.append(
                "pack.yaml build_plan.source is inherit_parent but the pack "
                "extends nothing — a top-level pack has no parent plan to "
                "inherit; declare `neutral`"
            )
            return
        legacy_paradigm = str(pack.get("legacy_paradigm") or "")
        # Active-learning descendants inherit the AL family's derived plan,
        # not a STATIC_PLAN_BY_PARADIGM entry — the walk below would read
        # them as plan-less. Mirrors load_build_plan's AL branch exactly.
        from scripts.build_plan import (  # noqa: PLC0415 — avoid import cost
            PROVISIONAL_PLAN_KEY,
            _static_plan_key,
        )

        resolved = _static_plan_key(legacy_paradigm, provisional=True)
        if resolved == PROVISIONAL_PLAN_KEY:
            errors.append(
                "pack.yaml build_plan.source is inherit_parent but no "
                "ancestor of "
                f"{pack.get('legacy_paradigm')!r} carries a static build "
                "plan — the declaration would silently degrade to the "
                "neutral provisional plan at runtime; declare `neutral` "
                "and take the label cap honestly"
            )
    elif choice.source == BUILD_PLAN_SOURCE_NEUTRAL and not choice.reasoning:
        warnings.append(
            "pack.yaml build_plan.source is neutral with no reasoning — "
            "say why the parent manifest does not fit (this lands in the "
            "promotion review)"
        )


def _result(
    proposal_dir: Path,
    proposal: ParadigmProposal | None,
    errors: list[str],
    warnings: list[str],
) -> ProposalValidationResult:
    return ProposalValidationResult(
        proposal_dir=str(proposal_dir),
        valid=not errors,
        errors=errors,
        warnings=warnings,
        proposal=proposal,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("proposal_dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--allow-update-existing", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args(argv)

    result = validate_proposal(
        args.proposal_dir,
        args.repo_root,
        allow_update_existing=args.allow_update_existing,
    )
    if args.write_report:
        report_path = args.proposal_dir / "validation_report.json"
        report_path.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    elif result.valid:
        msg = f"proposal OK: {args.proposal_dir}"
        if result.warnings:
            msg += f" ({len(result.warnings)} warning(s))"
        print(msg)
    else:
        print(f"proposal FAILED: {args.proposal_dir}", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        for warning in result.warnings:
            print(f"  warning: {warning}", file=sys.stderr)
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
