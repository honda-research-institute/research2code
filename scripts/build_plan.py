"""Spec-derived build plan for taxonomy-served paradigms.

Re-centering §5.3 moves the binding package/architecture contract into the
taxonomy/build-plan path. The taxonomy pack still supplies an interface prior
(`scaffold_hints.interface_hint`) and universal contract hints, but the actual
per-run build plan is derived from the method spec: pluggable name/signature,
paper-required model methods, and the pipeline's producer ownership map.

Only populated taxonomy nodes produce a build plan. Unserved/reserved nodes
return ``None`` and fail deterministically at the consumer gate instead of
falling back to legacy guide-backed manifest/schema loaders.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from schemas.method_spec import normalized_seed_param
from schemas.paradigm_proposal import (
    BUILD_PLAN_SOURCE_INHERIT,
    BUILD_PLAN_SOURCE_NEUTRAL,
)
from scripts import taxonomy


ROOT = Path(__file__).resolve().parent.parent
BUILD_PLAN_SCHEMA_VERSION = "1.0"


def load_build_plan(
    spec: dict[str, Any],
    repo_root: Path = ROOT,
    provisional_packs_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Return a spec-derived build plan for taxonomy-served nodes, else None.

    `provisional_packs_dir` is the gap-path serves() overlay
    (the gap path serves overlay design note (internal, not shipped)): pass the run's
    own `.pipeline/provisional_packs/` (via `taxonomy.run_overlay_dir`) so a
    gap paper's provisional classification resolves here the same way it
    does in the stage-1 strict cross-check. Committed behavior is
    byte-identical when the argument is None."""
    classification = (spec.get("comparison") or {}).get("classification") or {}
    paradigm_id = classification.get("id")
    if not isinstance(paradigm_id, str) or not paradigm_id:
        return None

    try:
        tax = taxonomy.load_taxonomy(
            repo_root, provisional_packs_dir=provisional_packs_dir)
    except FileNotFoundError:
        tax = taxonomy.load_taxonomy(
            ROOT, provisional_packs_dir=provisional_packs_dir)
    node = taxonomy.serves(paradigm_id, tax)
    if node is None:
        return None

    # Authoring-time plan choice (R2C-032): a PROVISIONAL pack may declare
    # `build_plan.source`. `neutral` routes to the generic provisional plan
    # before any family-specific branch — opting out of the parent's build
    # conventions is the whole point of the declaration (the SRL 2026-07-28
    # stage-2b halt: the inherited motion-planning manifest demanded classes
    # an RL method cannot have). `inherit_parent` and an absent declaration
    # both fall through to the existing ancestor walk, so every installed
    # pack that predates the field behaves byte-identically, and committed
    # nodes never reach this read.
    provisional = bool(getattr(node, "provisional", False))
    declared_source = (
        _declared_build_plan_source(node, tax) if provisional else None)
    if declared_source == BUILD_PLAN_SOURCE_NEUTRAL:
        return derive_static_build_plan(
            spec,
            paradigm_id,
            node,
            plan_key=PROVISIONAL_PLAN_KEY,
            tax=tax,
            inherit_family_contracts=False,
        )

    plan_key = _static_plan_key(paradigm_id, provisional=provisional)
    if plan_key is None and not provisional:
        # The spec's classification may carry a registered taxonomy ALIAS
        # (pool_based_active_learning) rather than the canonical id; serves()
        # resolved the node, so key the static table on the node's own
        # canonical legacy id (the B-02 spin-off: an alias-id spec reached
        # no build plan at all).
        canonical = getattr(node, "legacy_paradigm", None)
        if isinstance(canonical, str) and canonical and canonical != paradigm_id:
            plan_key = _static_plan_key(canonical, provisional=False)
    if plan_key is not None:
        return derive_static_build_plan(
            spec, paradigm_id, node, plan_key=plan_key, tax=tax)
    return None


def _declared_build_plan_source(
    node: "taxonomy.VariantNode | taxonomy.FamilyNode",
    tax: "taxonomy.Taxonomy | None",
) -> str | None:
    """The pack-declared `build_plan.source`, or None when absent/unknown.

    Read through the implementation bucket, the same surface
    `family_components` flows through. Unknown vocabulary reads as
    undeclared (the authoring validator rejects it at proposal time; a
    hand-edited installed pack degrades to today's walk instead of
    crashing a live run)."""
    block = taxonomy.node_implementation(node, tax).get("build_plan")
    if not isinstance(block, dict):
        return None
    source = block.get("source")
    if source in (BUILD_PLAN_SOURCE_INHERIT, BUILD_PLAN_SOURCE_NEUTRAL):
        return source
    return None


def _static_plan_key(paradigm_id: str, *, provisional: bool) -> str | None:
    """The STATIC_PLAN_BY_PARADIGM key that serves `paradigm_id`.

    Committed nodes match by exact key ONLY — their behavior is pinned. A
    provisional node (a run-local gap pack) has no static entry by
    definition, so it falls back to its nearest committed ancestor's plan:
    the pack's contract is inheritance-based (`extends` a committed family),
    and the build plan follows the same parent-first resolution. The walk is
    purely structural on the id path; nothing is paper- or family-specific."""
    if paradigm_id in STATIC_PLAN_BY_PARADIGM:
        return paradigm_id
    if not provisional:
        return None
    parts = paradigm_id.split("/")
    for i in range(len(parts) - 1, 0, -1):
        candidate = "/".join(parts[:i])
        if candidate in STATIC_PLAN_BY_PARADIGM:
            return candidate
    # A provisional pack with no committed ancestor (a new top level, or a
    # populated reserved stub whose family carries no plan — the SRL and
    # fedavg shapes) gets the generic provisional plan: the same shared
    # scaffolder/coder ownership rows every family has, no per-family rows
    # until promotion (item 4). Unreachable for committed nodes.
    return PROVISIONAL_PLAN_KEY


def derive_static_build_plan(
    spec: dict[str, Any],
    paradigm_id: str,
    node: taxonomy.VariantNode | taxonomy.FamilyNode,
    plan_key: str | None = None,
    tax: "taxonomy.Taxonomy | None" = None,
    inherit_family_contracts: bool = True,
) -> dict[str, Any]:
    """Build non-AL migrated plans from the node contract and static ownership map.

    `plan_key` is the STATIC_PLAN_BY_PARADIGM entry to base the plan on
    (defaults to `paradigm_id`; differs only for provisional gap nodes,
    which inherit their nearest committed ancestor's plan). `tax` is the
    taxonomy view `node` came from — required for provisional nodes, whose
    inheritance resolves only against their overlay. A provisional child's
    explicit neutral source also sets ``inherit_family_contracts=False`` so
    parent runtime subcontracts cannot leak back into the neutral plan."""
    base = STATIC_PLAN_BY_PARADIGM[plan_key or paradigm_id]
    package_manifest = deepcopy(base["package_manifest"])
    arch_contract = deepcopy(base["arch_contract_requirements"])
    _merge_node_family_components(arch_contract, node, tax)
    _merge_pack_declared_empty_files(package_manifest, node, tax)
    pluggable = _pluggable_component_from_spec_or_node(
        spec, paradigm_id, node, plan_key=plan_key, tax=tax)

    for entry in package_manifest.get("files", []):
        if entry.get("path") != "method/method.py":
            continue
        for symbol in entry.get("public_symbols") or []:
            name = symbol.get("name")
            if name == "<pluggable_component.name>" or name == base.get("default_pluggable_name"):
                symbol["name"] = pluggable["name"]
                symbol["signature"] = pluggable["signature"]
                symbol["contract"] = deepcopy(pluggable["contract"])

    # Merge the spec's paper-derived model methods into the manifest so the
    # 2.b architecture gate enforces the spec interface (bev-distill
    # 2026-07-02 F001/F002: enforced NOWHERE, the classes drifted from the
    # paper-derived interface, and the stage-4 fidelity review demoted the
    # delivery). HOW they merge depends on how many classes the manifest
    # declares:
    #
    #   - exactly one class symbol (the AL shape this merge was born from):
    #     merge into that class's required_methods, deduped by method name —
    #     unchanged legacy behavior.
    #   - multiple class symbols (KD student/teacher, MP dynamics/collision):
    #     the spec's required_model_methods carry NO class attribution, so
    #     requiring them of EVERY class corrupts the requirements (overnight
    #     2026-07-08: iDb-RRT's step_jacobian was demanded of the collision
    #     model, physically meaningless, and a correct two-class package
    #     halted at 2.b; detr-distill failed on the same shape). They land
    #     in the entry-level `any_class_required_methods` bucket instead:
    #     the 2.b gate requires each on AT LEAST ONE public class. Which
    #     class *should* own a method is paradigm knowledge and belongs in
    #     the kit's per-class required_methods (queue items 6/13); the
    #     stage-4 fidelity review remains the backstop for wrong-class
    #     placement. Both paths dedupe by method NAME so a convention that
    #     already carries the method is not re-required.
    spec_methods = _required_model_method_signatures(spec)
    signature_collisions: list[dict[str, str]] = []
    if spec_methods:
        for entry in package_manifest.get("files", []):
            if entry.get("path") != "method/model.py":
                continue
            class_symbols = [
                s for s in (entry.get("public_symbols") or [])
                if s.get("kind") == "class"
            ]
            if len(class_symbols) == 1:
                required = class_symbols[0].setdefault("required_methods", [])
                by_name = {_method_name(s): i for i, s in enumerate(required)}
                for signature in spec_methods:
                    name = _method_name(signature)
                    if name not in by_name:
                        by_name[name] = len(required)
                        required.append(signature)
                        continue
                    existing = required[by_name[name]]
                    if existing != signature:
                        # Same-name-different-signature collision (decided by
                        # maintainer decision 2026-08-18): the paper-declared signature wins,
                        # the collision is flagged for the run to log and
                        # record in assumptions.md. Silent name-dedup used to
                        # drop the spec-declared interface here — the exact
                        # "spec interface enforced nowhere" class.
                        required[by_name[name]] = signature
                        signature_collisions.append({
                            "method": name,
                            "manifest_signature": existing,
                            "spec_signature": signature,
                            "resolution": "paper_declared",
                        })
            elif class_symbols:
                by_name = {
                    _method_name(s): s
                    for symbol in class_symbols
                    for s in (symbol.get("required_methods") or [])
                }
                bucket = entry.setdefault("any_class_required_methods", [])
                by_name.update({_method_name(s): s for s in bucket})
                for signature in spec_methods:
                    name = _method_name(signature)
                    if name not in by_name:
                        bucket.append(signature)
                    elif by_name[name] != signature:
                        # Multi-class manifests carry no class attribution for
                        # spec methods (the iDb-RRT wrong-class corruption),
                        # so the collision is flagged WITHOUT mutating the
                        # class-declared signature; the fidelity review is
                        # the placement backstop.
                        signature_collisions.append({
                            "method": name,
                            "manifest_signature": by_name[name],
                            "spec_signature": signature,
                            "resolution": "flagged_only_multi_class",
                        })

    resolved_plan = {
        "schema_version": BUILD_PLAN_SCHEMA_VERSION,
        "source": "spec+taxonomy",
        "paradigm_id": paradigm_id,
        "taxonomy_id": node.taxonomy_id if isinstance(node, taxonomy.VariantNode) else node.id,
        # Which STATIC_PLAN_BY_PARADIGM entry this plan was based on — the
        # build-context provenance the delivery-label cap keys on (R2C-032):
        # `_provisional` means the neutral plan carried this run, so the
        # delivery caps at uncertified — new territory. Additive; no plan
        # consumer reads unknown keys and plans are never persisted, so
        # committed-family behavior is unchanged.
        "plan_key": plan_key or paradigm_id,
        "interface_hint": (
            taxonomy.node_expertise(node, tax).get("interface_hint")
            if node is not None else None
        ),
        "pluggable_component": pluggable,
        "package_manifest": package_manifest,
        "arch_contract_requirements": arch_contract,
    }
    if signature_collisions:
        # Structured record for the run to log and write to assumptions.md;
        # build_plan is a pure library with no run-dir access.
        resolved_plan["signature_collisions"] = signature_collisions
    # Only a committed family that owns an exact runtime route carries this
    # block. It is copied from the static plan rather than inferred from a
    # one-block architecture, parameter spelling, or discovered build_* name.
    if "runtime_execution" in base:
        resolved_plan["runtime_execution"] = deepcopy(
            base["runtime_execution"]
        )
    # Scaling is an independent family policy. It is never inferred from the
    # runtime construction route or from a generated implementation.
    if "target_scaling" in base:
        resolved_plan["target_scaling"] = deepcopy(
            base["target_scaling"]
        )
    if "target_scaling_execution" in base:
        resolved_plan["target_scaling_execution"] = deepcopy(
            base["target_scaling_execution"]
        )
    if "training_history_execution" in base:
        resolved_plan["training_history_execution"] = deepcopy(
            base["training_history_execution"]
        )
    # The family owns the offline table grammar in its taxonomy build-plan
    # declaration. K remains owned by the taxonomy's separate protocol-role
    # declaration: copying that demo value here gives Stage 2a one closed
    # resolved contract without minting a second table or horizon authority in
    # Python. The node read also preserves explicit provisional inheritance.
    declared_build_plan = taxonomy.node_implementation(node, tax).get(
        "build_plan"
    )
    declared_offline_demo_data = (
        declared_build_plan.get("offline_demo_data")
        if isinstance(declared_build_plan, dict)
        else None
    )
    if inherit_family_contracts and declared_offline_demo_data is not None:
        offline_demo_data = deepcopy(declared_offline_demo_data)
        horizon_entry = taxonomy.load_params_derivation(
            paradigm_id, tax
        ).get("forecast_horizon") or {}
        horizon = horizon_entry.get("demo_value")
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or horizon <= 0
        ):
            raise ValueError(
                "build_plan.offline_demo_data requires a positive integral "
                "params_derivation.forecast_horizon.demo_value"
            )
        offline_demo_data["forecast_horizon_demo_value"] = horizon
        resolved_plan["offline_demo_data"] = offline_demo_data
    return resolved_plan


def _merge_pack_declared_empty_files(
    package_manifest: dict[str, Any],
    node: "taxonomy.VariantNode | taxonomy.FamilyNode",
    tax: "taxonomy.Taxonomy | None",
) -> None:
    """Honor a pack's explicit `public_symbols: []` file declarations.

    A provisional pack's own package_manifest is binding where it
    explicitly declares a file EMPTY: `public_symbols: []` means the
    paradigm exports nothing from that file, so any placeholder rows the
    base plan carries for that path are dropped. The DomIndOnto KBP run
    (2026-07-29, stage 2d halt): the pack declared training.py
    producer-less with no public symbols, the architecture-coder
    correctly wrote a docstring-only placeholder, and the neutral plan's
    `<training_functions>` row made the finalizer demand functions of a
    file the paradigm deliberately leaves empty. An empty declaration
    can only RELAX a placeholder into nothing-required; it never adds an
    enforced symbol. Non-empty pack manifest entries stay authoring
    documentation until a second concrete case fixes the merge's shape.
    """
    declared = taxonomy.node_implementation(node, tax).get(
        "package_manifest") if node is not None else None
    if not isinstance(declared, dict):
        return
    empty_paths = {
        str(entry.get("path"))
        for entry in (declared.get("files") or [])
        if isinstance(entry, dict) and entry.get("public_symbols") == []
    }
    if not empty_paths:
        return
    for entry in package_manifest.get("files", []):
        if entry.get("path") in empty_paths and "public_symbols" in entry:
            entry["public_symbols"] = []


def _merge_node_family_components(
    arch_contract_requirements: dict[str, Any],
    node: "taxonomy.VariantNode | taxonomy.FamilyNode | None",
    tax: "taxonomy.Taxonomy | None",
) -> None:
    """Fold the node's pack-declared `family_components` into the plan's
    `arch_contract_requirements` (in place).

    Arch-contract headroom (approved 2026-07-16): a committed taxonomy node
    or a run-local provisional pack declares the LEGAL top-level
    `family_components` names for its architecture contract; the Stage 2.d
    static validator enforces the declared set in both directions, the same
    way `required_blocks` flows today. Node declarations override a static
    plan's per component name (child-wins, matching taxonomy inheritance).
    Plans for nodes that declare nothing are byte-identical to before."""
    if node is None:
        return
    declared = taxonomy.node_implementation(node, tax).get("family_components")
    if not isinstance(declared, dict) or not declared:
        return
    merged = dict(arch_contract_requirements.get("family_components") or {})
    merged.update(deepcopy(declared))
    arch_contract_requirements["family_components"] = merged


def _pluggable_component_from_spec_or_node(
    spec: dict[str, Any],
    paradigm_id: str,
    node: taxonomy.VariantNode | taxonomy.FamilyNode,
    plan_key: str | None = None,
    tax: "taxonomy.Taxonomy | None" = None,
) -> dict[str, Any]:
    comparison = spec.get("comparison") or {}
    spec_pluggable = comparison.get("pluggable_component") or {}
    node_pluggable = (
        taxonomy.node_implementation(node, tax).get("pluggable_component")
        if node is not None else {}
    )
    node_contract = (node_pluggable or {}).get("contract") or {}
    pluggable_name = (
        spec_pluggable.get("name")
        or (node_pluggable or {}).get("name")
        or STATIC_PLAN_BY_PARADIGM[plan_key or paradigm_id]["default_pluggable_name"]
    )
    pluggable_signature = (
        spec_pluggable.get("signature")
        or (node_pluggable or {}).get("signature_template")
        or ""
    )
    return {
        "name": pluggable_name,
        "signature": pluggable_signature,
        "seed_param": normalized_seed_param(spec_pluggable.get("seed_param"))
        or normalized_seed_param(node_contract.get("seed_param")) or "seed",
        "contract": deepcopy(node_contract),
    }


def _method_name(signature: str) -> str:
    """The bare method name of a `name(self, ...) -> T` signature string."""
    return signature.split("(", 1)[0].strip()


def _required_model_method_signatures(spec: dict[str, Any]) -> list[str]:
    critical = spec.get("critical_requirements") or {}
    methods = critical.get("required_model_methods") or []
    out: list[str] = []
    for method in methods:
        signature = method.get("signature") if isinstance(method, dict) else None
        if isinstance(signature, str) and signature.strip():
            out.append(signature.strip())
    return out


COMMON_VALIDATORS = {
    "after_package_scaffolder": "scripts/validate_scaffolder_output.py",
    "after_architecture_coder": "scripts/validate_architecture_coder_output.py",
    "after_method_coder": "scripts/validate_method_coder_output.py",
    "after_init_finalizer": [
        "scripts/validate_package_imports.py",
        "scripts/validate_arch_contract.py",
        "scripts/validate_arch_contract_runtime.py",
    ],
}


AL_DEPENDENCY_GRAPH = {
    "method/data.py": [],
    "method/model.py": [],
    "method/method.py": ["method/model.py"],
    "method/training.py": ["method/model.py"],
    "method/__init__.py": [
        "method/data.py",
        "method/method.py",
        "method/model.py",
        "method/training.py",
    ],
}


# Hoisted verbatim from the retired derive_active_learning_build_plan
# (B-02, decision 6): the manifest literal is the plan; the only spec-derived
# parts (pluggable name/signature/contract, required model methods) resolve
# through derive_static_build_plan's shared substitution and merge loops.
AL_FILES = [
    {
        "path": "method/data.py",
        "kind": "paradigm_fixed",
        "produced_by": "package_scaffolder",
        "public_symbols": [
            {
                "name": "load_data",
                "kind": "function",
                "signature": (
                    "load_data(path: str | os.PathLike | None = None, *, "
                    "pool_size: int = 5000, n_test: int = 1000, seed: int = 0) "
                    "-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]"
                ),
            }
        ],
    },
    {"path": "method/example_data/README.md", "kind": "paradigm_fixed", "produced_by": "package_scaffolder"},
    {
        "path": "method/model.py",
        "kind": "paradigm_shaped",
        "produced_by": "architecture_coder",
        "public_symbols": [
            {
                "name": "<ArchitectureClass>",
                "kind": "class",
                "inherits": "torch.nn.Module",
                "required_methods": ["forward(self, x: torch.Tensor) -> torch.Tensor"],
            }
        ],
    },
    {
        "path": "method/training.py",
        "kind": "paradigm_fixed",
        "produced_by": "architecture_coder",
        "public_symbols": [
            {
                "name": "build_model",
                "kind": "function",
                "signature": (
                    "build_model(input_dim: int, n_classes: int, **arch_kwargs) "
                    "-> <ArchitectureClass>"
                ),
            },
            {
                "name": "train_from_scratch",
                "kind": "function",
                "signature": (
                    "train_from_scratch(model: <ArchitectureClass>, x_train: torch.Tensor, "
                    "y_train: torch.Tensor, *, learning_rate: float, max_epochs: int, "
                    "train_until_accuracy: float, seed: int) -> <ArchitectureClass>"
                ),
            },
        ],
    },
    {
        "path": "method/method.py",
        "kind": "method_shaped",
        "produced_by": "method_coder",
        "public_symbols": [
            {
                # Substituted by derive_static_build_plan against
                # default_pluggable_name: "select_batch" (the loop at the
                # placeholder-substitution site fills name, signature, and
                # contract from the spec-or-node pluggable resolution).
                "name": "<pluggable_component.name>",
                "kind": "function",
                "signature": "<pluggable_component.signature>",
                "contract": {},
            },
            {
                "name": "<method_helpers>",
                "kind": "function",
                "all_top_level_functions_are_public": True,
            },
        ],
    },
    {
        "path": "method/__init__.py",
        "kind": "derived",
        "produced_by": "init_finalizer",
        "content_rule": (
            "Re-export every public symbol declared above. __all__ ordering: "
            "[pluggable_component.name, *method_helpers, *architecture_classes, "
            "*training_functions, load_data]."
        ),
    },
    {
        "path": "requirements.txt",
        "kind": "derived",
        "produced_by": "init_finalizer",
        "content_rule": (
            "Computed by AST-walking imports across method/ and notebook imports; "
            "base requirements are owned by finalize_package_init.py."
        ),
    },
    {
        "path": "method/README.md",
        "kind": "paradigm_fixed",
        "produced_by": "package_scaffolder",
    },
]


AL_ARCH_CONTRACT_REQUIREMENTS = {
    "description": (
        "Spec-derived active-learning architecture contract requirements. "
        "Concrete shapes remain paper/code-derived; this only validates the "
        "universal AL contract surface."
    ),
    "required_blocks": {
        "data_loader.load_data_returns": {
            "type": "dict[str, shape_string]",
            "min_keys": ["x_pool", "y_pool", "x_test", "y_test"],
            "exact_keys": ["x_pool", "y_pool", "x_test", "y_test"],
        },
        "architecture.model": {
            "required_subblocks": [
                "class_name",
                "forward.input",
                "forward.output_type",
            ],
        },
        # Kept verbatim: this spelling and the bare
        # "pluggable_component.name" resolve identically on the populated
        # path but diverge on the no-pluggable edge (red-team verified);
        # harmonizing is deliberately NOT done.
        "pluggable_component.name": {
            "must_equal": "spec.comparison.pluggable_component.name",
        },
        "pluggable_component.input_shapes": {
            "type": "dict[str, shape_string]",
            "min_keys": ["x_unlabeled"],
        },
        "pluggable_component.output_shape": {
            "type": "shape_string",
        },
        "training_loop.function_name": {
            "must_equal": "train_from_scratch",
        },
        "training_loop.input_shapes": {
            "type": "dict[str, shape_string]",
            "min_keys": ["x_train", "y_train"],
        },
    },
    "symbol_conventions": {
        "B": "runtime batch size inside model forward",
        "N": "training-set size",
        "N_test": "test-set size",
        "N_pool": "unlabeled pool size",
        "n_features": "input feature dim",
        "n_classes": "number of classes",
        "hidden_dim": "model hidden width when applicable",
    },
}


KD_DEPENDENCY_GRAPH = {
    "method/data.py": [],
    "method/model.py": [],
    "method/method.py": ["method/model.py"],
    "method/training.py": ["method/model.py", "method/method.py"],
    "method/__init__.py": [
        "method/data.py",
        "method/method.py",
        "method/model.py",
        "method/training.py",
    ],
}


def _manifest(description: str, files: list[dict[str, Any]], dependency_graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "description": description,
        "files": files,
        "dependency_graph": deepcopy(dependency_graph),
        "validators": deepcopy(COMMON_VALIDATORS),
    }


KD_ROOT_FILES = [
    {
        "path": "method/data.py",
        "kind": "paradigm_fixed",
        "produced_by": "package_scaffolder",
        "public_symbols": [
            {
                "name": "load_data",
                "kind": "function",
                "signature": (
                    "load_data(path: str | os.PathLike | None = None, *, "
                    "train_size: int = 1000, n_test: int = 200, seed: int = 0) "
                    "-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]"
                ),
            }
        ],
    },
    {"path": "method/example_data/README.md", "kind": "paradigm_fixed", "produced_by": "package_scaffolder"},
    {
        "path": "method/model.py",
        "kind": "paradigm_shaped",
        "produced_by": "architecture_coder",
        "public_symbols": [
            {
                "name": "<StudentClass>",
                "kind": "class",
                "inherits": "torch.nn.Module",
                "required_methods": ["forward(self, x: torch.Tensor) -> torch.Tensor"],
            },
            {
                "name": "<TeacherClass>",
                "kind": "class",
                "inherits": "torch.nn.Module",
                "required_methods": ["forward(self, x: torch.Tensor) -> torch.Tensor"],
            },
        ],
    },
    {
        "path": "method/training.py",
        "kind": "paradigm_fixed",
        "produced_by": "architecture_coder",
        "public_symbols": [
            {"name": "build_student", "kind": "function", "signature": "build_student(input_dim: int, n_classes: int, **arch_kwargs) -> <StudentClass>"},
            {"name": "build_teacher", "kind": "function", "signature": "build_teacher(input_dim: int, n_classes: int, **arch_kwargs) -> <TeacherClass>"},
            {
                "name": "train_with_distillation",
                "kind": "function",
                "signature": (
                    "train_with_distillation(student: <StudentClass>, teacher: <TeacherClass>, "
                    "x_train: torch.Tensor, y_train: torch.Tensor, *, distillation_loss_fn: Callable, "
                    "learning_rate: float, num_epochs: int, batch_size: int, seed: int, "
                    "**distillation_kwargs) -> <StudentClass>"
                ),
            },
        ],
    },
    {
        "path": "method/method.py",
        "kind": "method_shaped",
        "produced_by": "method_coder",
        "public_symbols": [
            {"name": "<pluggable_component.name>", "kind": "function", "signature": "from spec.comparison.pluggable_component.signature", "contract": "from pluggable_component.contract"},
            {"name": "<method_helpers>", "kind": "function", "all_top_level_functions_are_public": True},
        ],
    },
    {"path": "method/__init__.py", "kind": "derived", "produced_by": "init_finalizer"},
    {"path": "requirements.txt", "kind": "derived", "produced_by": "init_finalizer"},
    {"path": "method/README.md", "kind": "paradigm_fixed", "produced_by": "package_scaffolder"},
]


KD_DETECTION_FILES = [
    {
        "path": "method/data.py",
        "kind": "paradigm_fixed",
        "produced_by": "package_scaffolder",
        "public_symbols": [
            {
                "name": "load_data",
                "kind": "function",
                "signature": (
                    "load_data(path: str | os.PathLike | None = None, *, "
                    "train_size: int = 200, n_test: int = 50, seed: int = 0) "
                    "-> Tuple[List[Image], List[Dict], List[Image], List[Dict]]"
                ),
            }
        ],
    },
    {"path": "method/example_data/README.md", "kind": "paradigm_fixed", "produced_by": "package_scaffolder"},
    {
        "path": "method/model.py",
        "kind": "paradigm_shaped",
        "produced_by": "architecture_coder",
        "public_symbols": [
            {
                "name": "<StudentClass>",
                "kind": "class",
                "inherits": "torch.nn.Module",
                "required_methods": [
                    "forward(self, x: torch.Tensor, *, query_prior: torch.Tensor | None = None) -> Dict[str, torch.Tensor]"
                ],
            },
            {
                "name": "<TeacherClass>",
                "kind": "class",
                "inherits": "torch.nn.Module",
                "required_methods": [
                    "forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]",
                    "get_query_embeddings(self) -> torch.Tensor",
                ],
            },
        ],
    },
    {
        "path": "method/training.py",
        "kind": "paradigm_fixed",
        "produced_by": "architecture_coder",
        "public_symbols": [
            {"name": "build_student", "kind": "function", "signature": "build_student(num_classes: int, num_queries: int = 100, **arch_kwargs) -> <StudentClass>"},
            {"name": "build_teacher", "kind": "function", "signature": "build_teacher(num_classes: int, num_queries: int = 100, **arch_kwargs) -> <TeacherClass>"},
            {
                "name": "train_with_distillation",
                "kind": "function",
                "signature": (
                    "train_with_distillation(student: <StudentClass>, teacher: <TeacherClass>, "
                    "images_train: List[Image], targets_train: List[Dict], *, "
                    "distillation_loss_fn: Callable, learning_rate: float, num_epochs: int, "
                    "batch_size: int, seed: int, **distillation_kwargs) -> <StudentClass>"
                ),
            },
        ],
    },
    {
        "path": "method/method.py",
        "kind": "method_shaped",
        "produced_by": "method_coder",
        "public_symbols": [
            {"name": "<pluggable_component.name>", "kind": "function", "signature": "from spec.comparison.pluggable_component.signature", "contract": "from pluggable_component.contract"},
            {"name": "<method_helpers>", "kind": "function", "all_top_level_functions_are_public": True},
        ],
    },
    {"path": "method/__init__.py", "kind": "derived", "produced_by": "init_finalizer"},
    {"path": "requirements.txt", "kind": "derived", "produced_by": "init_finalizer"},
    {"path": "method/README.md", "kind": "paradigm_fixed", "produced_by": "package_scaffolder"},
]


def _manifest_symbol(files: list, path: str, name: str) -> dict:
    """Path+name-keyed lookup into a files manifest (B-11: positional
    indices break silently when rows are added; a missing path/name raises
    at import, loudly)."""
    for entry in files:
        if entry.get("path") != path:
            continue
        for symbol in entry.get("public_symbols") or []:
            if symbol.get("name") == name:
                return symbol
    raise KeyError(f"{path}::{name} not in manifest files")


KD_CROSS_MODAL_FILES = deepcopy(KD_DETECTION_FILES)
_manifest_symbol(KD_CROSS_MODAL_FILES, "method/data.py", "load_data")["signature"] = (
    "load_data(path: str | os.PathLike | None = None, *, train_size: int = 200, "
    "n_test: int = 50, seed: int = 0) -> Tuple[List[Image], List[Tensor], "
    "List[Dict], List[Image], List[Tensor], List[Dict]]"
)
_manifest_symbol(KD_CROSS_MODAL_FILES, "method/model.py", "<StudentClass>")["required_methods"] = [
    "forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]",
    "forward_with_bev_features(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]",
]
_manifest_symbol(KD_CROSS_MODAL_FILES, "method/model.py", "<TeacherClass>")["required_methods"] = [
    "forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]",
    "forward_with_bev_features(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]",
]
_manifest_symbol(KD_CROSS_MODAL_FILES, "method/training.py", "build_student")["signature"] = "build_student(num_classes: int, **arch_kwargs) -> <StudentClass>"
_manifest_symbol(KD_CROSS_MODAL_FILES, "method/training.py", "build_teacher")["signature"] = "build_teacher(num_classes: int, **arch_kwargs) -> <TeacherClass>"
_manifest_symbol(KD_CROSS_MODAL_FILES, "method/training.py", "train_with_distillation")["signature"] = (
    "train_with_distillation(student: <StudentClass>, teacher: <TeacherClass>, "
    "student_inputs_train: List[Image], teacher_inputs_train: List[Tensor], "
    "targets_train: List[Dict], *, distillation_loss_fn: Callable, learning_rate: float, "
    "num_epochs: int, batch_size: int, seed: int, **distillation_kwargs) -> <StudentClass>"
)


VIT_FILES = [
    {
        "path": "method/data.py",
        "kind": "paradigm_fixed",
        "produced_by": "package_scaffolder",
        "public_symbols": [
            {
                "name": "load_data",
                "kind": "function",
                "signature": (
                    "load_data(path: str | os.PathLike | None = None, *, img_size: int = 32, "
                    "train_transform=None, test_transform=None, seed: int = 0, batch_size: int = 256) "
                    "-> Tuple[DataLoader, DataLoader]"
                ),
            }
        ],
    },
    {"path": "method/example_data/README.md", "kind": "paradigm_fixed", "produced_by": "package_scaffolder"},
    {
        "path": "method/model.py",
        "kind": "paradigm_shaped",
        "produced_by": "architecture_coder",
        "public_symbols": [
            {"name": "<ViTBackbone>", "kind": "class", "inherits": "torch.nn.Module", "required_methods": ["forward(self, x: torch.Tensor) -> torch.Tensor"]}
        ],
    },
    {
        "path": "method/method.py",
        "kind": "method_shaped",
        "produced_by": "method_coder",
        "public_symbols": [
            {"name": "group_tokens", "kind": "function", "signature": "from spec.comparison.pluggable_component.signature", "contract": "from pluggable_component.contract"},
            {"name": "generate_random_tensor", "kind": "function", "signature": "generate_random_tensor(h: int, w: int, num_heads: int = 1, *, seed: int = 0) -> torch.Tensor"},
            {"name": "<method_helpers>", "kind": "function", "all_top_level_functions_are_public": True},
        ],
    },
    {
        "path": "method/training.py",
        "kind": "paradigm_shaped",
        "produced_by": "architecture_coder",
        "public_symbols": [
            {"name": "build_model", "kind": "function", "signature": "build_model(input_dim: int = 3, img_size: int = 32, n_classes: int = 10, **arch_kwargs) -> <ViTBackbone>"},
            {"name": "train_epoch", "kind": "function", "signature": "train_epoch(model: <ViTBackbone>, loader: DataLoader, optimizer: torch.optim.Optimizer, *, seed: int = 0) -> Dict[str, float]"},
            {"name": "evaluate", "kind": "function", "signature": "evaluate(model: <ViTBackbone>, loader: DataLoader) -> Dict[str, float]"},
        ],
    },
    {"path": "method/__init__.py", "kind": "derived", "produced_by": "init_finalizer"},
    {"path": "requirements.txt", "kind": "derived", "produced_by": "init_finalizer"},
    {"path": "method/README.md", "kind": "paradigm_fixed", "produced_by": "package_scaffolder"},
]


DA_FILES = [
    {
        "path": "method/data.py",
        "kind": "paradigm_fixed",
        "produced_by": "package_scaffolder",
        "public_symbols": [
            {"name": "load_target_data", "kind": "function", "signature": "load_target_data(name: str = 'smoke_target', *, n_frames: int = 50, seed: int = 0) -> TargetDataset"},
            {"name": "load_source_detectors", "kind": "function", "signature": "load_source_detectors(names: List[str] = None, *, ensemble_size: int = 3) -> List[Detector]"},
        ],
    },
    {"path": "method/example_data/README.md", "kind": "paradigm_fixed", "produced_by": "package_scaffolder"},
    {
        "path": "method/model.py",
        "kind": "paradigm_shaped",
        "produced_by": "architecture_coder",
        "public_symbols": [
            {
                "name": "<DetectorClass>",
                "kind": "class",
                "inherits": "torch.nn.Module",
                "required_methods": [
                    "forward(self, point_cloud: Tensor) -> Dict[str, Tensor]",
                    "predict(self, point_cloud: Tensor) -> List[BoundingBox3D]",
                ],
            }
        ],
    },
    {
        "path": "method/training.py",
        "kind": "paradigm_shaped",
        "produced_by": "architecture_coder",
        "public_symbols": [
            {"name": "build_detector", "kind": "function", "signature": "build_detector(num_classes: int = 3, **arch_kwargs) -> <DetectorClass>"},
            {"name": "retrain_on_pseudo_labels", "kind": "function", "signature": "retrain_on_pseudo_labels(detector, target_data, pseudo_labels, *, n_epochs: int = 1, seed: int = 0) -> <DetectorClass>"},
        ],
    },
    {
        "path": "method/method.py",
        "kind": "method_shaped",
        "produced_by": "method_coder",
        "public_symbols": [
            {"name": "<pluggable_component.name>", "kind": "function", "signature": "from spec.comparison.pluggable_component.signature", "contract": "from pluggable_component.contract"},
            {"name": "<method_helpers>", "kind": "function", "all_top_level_functions_are_public": True},
        ],
    },
    {"path": "method/__init__.py", "kind": "derived", "produced_by": "init_finalizer"},
    {"path": "requirements.txt", "kind": "derived", "produced_by": "init_finalizer"},
    {"path": "method/README.md", "kind": "paradigm_fixed", "produced_by": "package_scaffolder"},
]


MP_FILES = [
    {
        "path": "method/data.py",
        "kind": "paradigm_fixed",
        "produced_by": "package_scaffolder",
        "public_symbols": [
            {
                "name": "load_environment",
                "kind": "function",
                "signature": "load_environment(name: str = 'two_rooms_simple', *, seed: int = 0) -> Environment",
            },
            {
                "name": "load_problem",
                "kind": "function",
                "signature": (
                    "load_problem(name: str = 'two_rooms_simple', *, seed: int = 0) "
                    "-> Tuple[torch.Tensor, torch.Tensor, Environment]"
                ),
            },
        ],
    },
    {"path": "method/example_data/README.md", "kind": "paradigm_fixed", "produced_by": "package_scaffolder"},
    {
        "path": "method/model.py",
        "kind": "paradigm_shaped",
        "produced_by": "architecture_coder",
        "public_symbols": [
            {
                "name": "<SystemDynamics>",
                "kind": "class",
                "required_methods": [
                    "step(self, state: Tensor, control: Tensor, dt: float) -> Tensor",
                    "step_jacobian(self, state: Tensor, control: Tensor, dt: float) -> Tuple[Tensor, Tensor]",
                ],
            },
            {
                "name": "<CollisionModel>",
                "kind": "class",
                "required_methods": [
                    "is_in_collision(self, state: Tensor) -> bool",
                    "distance_to_obstacle(self, state: Tensor) -> float",
                ],
            },
        ],
    },
    {
        "path": "method/training.py",
        "kind": "paradigm_fixed",
        "produced_by": "architecture_coder",
        "public_symbols": [
            {
                "name": "precompute_motion_primitives",
                "kind": "function",
                "signature": (
                    "precompute_motion_primitives(dynamics: <SystemDynamics>, *, "
                    "n_primitives: int = 100, seed: int = 0) -> List[MotionPrimitive]"
                ),
            },
        ],
    },
    {
        "path": "method/method.py",
        "kind": "method_shaped",
        "produced_by": "method_coder",
        "public_symbols": [
            {"name": "<pluggable_component.name>", "kind": "function", "signature": "from spec.comparison.pluggable_component.signature", "contract": "from pluggable_component.contract"},
            {"name": "<method_helpers>", "kind": "function", "all_top_level_functions_are_public": True},
        ],
    },
    {"path": "method/__init__.py", "kind": "derived", "produced_by": "init_finalizer"},
    {"path": "requirements.txt", "kind": "derived", "produced_by": "init_finalizer"},
    {"path": "method/README.md", "kind": "paradigm_fixed", "produced_by": "package_scaffolder"},
]


MP_DEPENDENCY_GRAPH = {
    "method/data.py": [],
    "method/model.py": [],
    "method/method.py": ["method/model.py"],
    "method/training.py": ["method/model.py"],
    "method/__init__.py": ["method/data.py", "method/method.py", "method/model.py", "method/training.py"],
}


MP_ARCH_CONTRACT_REQUIREMENTS = {
    "required_blocks": {
        "data_loader.load_data_returns": {"type": "dict[str, shape_string]"},
        "architecture.dynamics": {"required_subblocks": ["class_name", "forward.input"]},
        "architecture.collision_model": {"required_subblocks": ["class_name", "forward.input"]},
        "pluggable_component.name": {"must_equal": "pluggable_component.name"},
        "pluggable_component.input_shapes": {"type": "dict[str, shape_string]"},
        "pluggable_component.output_shape": {"type": "shape_string"},
    },
    "symbol_conventions": {
        "state_dim": "dimension of the state vector",
        "control_dim": "dimension of the control input",
        "B": "runtime batch size when dynamics are vectorized",
    },
}


# RL collision avoidance (CLC-MP/rl_collision_avoidance, promoted from the
# SRL run's provisional pack, R2C-029). Starts from MP_FILES and reshapes the
# producer-owned rows for a method that TRAINS a network instead of running a
# classical planner:
#   - data.py / example_data / README rows are byte-identical to MP_FILES.
#     The scenario surface is the same planning-problem shape (start, goal,
#     environment — the SRL arch contract's load_data_returns), and keeping
#     the scaffolder-owned rows identical is what lets the node inherit
#     paradigms/motion_planning/templates (test_paradigm_templates validates
#     those templates against THIS manifest).
#   - model.py declares the two classes the SRL contract/spec actually
#     carried: a value network (MultiAgentValueNet — an nn.Module per the
#     spec's pytorch framework) and a policy that queries it for velocity
#     actions (SA_CADRLPolicy). Both are placeholders like <SystemDynamics>;
#     with two class symbols, spec required_model_methods land in
#     any_class_required_methods, unchanged.
#   - training.py becomes a real producer row (the family trains): a builder
#     plus the Algorithm 1 training entry point, replacing
#     precompute_motion_primitives.
#   - method.py keeps the pluggable row verbatim (plan/PlanResult interface
#     semantics inherit; the spec's pluggable name/signature wins at
#     derivation time, exactly as for every family).
RL_CA_FILES = [
    {
        "path": "method/data.py",
        "kind": "paradigm_fixed",
        "produced_by": "package_scaffolder",
        "public_symbols": [
            {
                "name": "load_environment",
                "kind": "function",
                "signature": "load_environment(name: str = 'two_rooms_simple', *, seed: int = 0) -> Environment",
            },
            {
                "name": "load_problem",
                "kind": "function",
                "signature": (
                    "load_problem(name: str = 'two_rooms_simple', *, seed: int = 0) "
                    "-> Tuple[torch.Tensor, torch.Tensor, Environment]"
                ),
            },
        ],
    },
    {"path": "method/example_data/README.md", "kind": "paradigm_fixed", "produced_by": "package_scaffolder"},
    {
        "path": "method/model.py",
        "kind": "paradigm_shaped",
        "produced_by": "architecture_coder",
        "public_symbols": [
            {
                "name": "<ValueNetwork>",
                "kind": "class",
                "inherits": "torch.nn.Module",
                "required_methods": [
                    "forward(self, joint_state: torch.Tensor) -> torch.Tensor",
                ],
            },
            {
                # The policy wraps the value net for action selection; the SRL
                # coder's SA_CADRLPolicy exposed forward -> (B, 2) velocities.
                # No nn.Module requirement: a policy may be a plain wrapper.
                "name": "<Policy>",
                "kind": "class",
                "required_methods": [
                    "forward(self, joint_state: torch.Tensor) -> torch.Tensor",
                ],
            },
        ],
    },
    {
        "path": "method/training.py",
        "kind": "paradigm_shaped",
        "produced_by": "architecture_coder",
        "public_symbols": [
            {
                "name": "build_value_network",
                "kind": "function",
                "signature": "build_value_network(n_agents: int, **arch_kwargs) -> <ValueNetwork>",
            },
            {
                # Algorithm 1's loop: epsilon-greedy trajectory generation,
                # experience assimilation, RMSprop updates over n_episodes.
                "name": "train_policy",
                "kind": "function",
                "signature": (
                    "train_policy(value_net: <ValueNetwork>, *, n_episodes: int, "
                    "seed: int, **training_kwargs) -> <Policy>"
                ),
            },
        ],
    },
    {
        "path": "method/method.py",
        "kind": "method_shaped",
        "produced_by": "method_coder",
        "public_symbols": [
            {"name": "<pluggable_component.name>", "kind": "function", "signature": "from spec.comparison.pluggable_component.signature", "contract": "from pluggable_component.contract"},
            {"name": "<method_helpers>", "kind": "function", "all_top_level_functions_are_public": True},
        ],
    },
    {"path": "method/__init__.py", "kind": "derived", "produced_by": "init_finalizer"},
    {"path": "requirements.txt", "kind": "derived", "produced_by": "init_finalizer"},
    {"path": "method/README.md", "kind": "paradigm_fixed", "produced_by": "package_scaffolder"},
]


# Differs from MP_DEPENDENCY_GRAPH in one edge: method.py depends on
# training.py as well as model.py, because the family's pluggable is the
# end-to-end train-then-plan entry point (the SRL spec's train_and_plan
# drives training). Producer order is unchanged (architecture_coder writes
# model.py + training.py at 2.b before method_coder writes method.py at 2.c).
RL_CA_DEPENDENCY_GRAPH = {
    "method/data.py": [],
    "method/model.py": [],
    "method/training.py": ["method/model.py"],
    "method/method.py": ["method/model.py", "method/training.py"],
    "method/__init__.py": ["method/data.py", "method/method.py", "method/model.py", "method/training.py"],
}


RL_CA_ARCH_CONTRACT_REQUIREMENTS = {
    "required_blocks": {
        # min_keys mirror the SRL contract's actual load_data_returns
        # (start / goal / environment) — the same planning-problem surface
        # load_problem scaffolds.
        "data_loader.load_data_returns": {"type": "dict[str, shape_string]", "min_keys": ["start", "goal", "environment"]},
        # The two architecture blocks the SRL contract carried (value_net:
        # MultiAgentValueNet, policy: SA_CADRLPolicy) — replacing the MP
        # planner blocks (architecture.dynamics / architecture.collision_model)
        # the halt showed an RL method cannot satisfy.
        "architecture.value_net": {"required_subblocks": ["class_name", "forward.input", "forward.output_type"]},
        "architecture.policy": {"required_subblocks": ["class_name", "forward.input", "forward.output_type"]},
        "pluggable_component.name": {"must_equal": "pluggable_component.name"},
        "pluggable_component.input_shapes": {"type": "dict[str, shape_string]"},
        "pluggable_component.output_shape": {"type": "shape_string"},
        # Presence-only (required_subblocks), deliberately NO must_equal on
        # the function name: in the SRL contract training_loop.function_name
        # is the pluggable itself (train_and_plan) because training is
        # encapsulated in the end-to-end entry point. Pinning a fixed name
        # the way AL (train_from_scratch) and KD (train_with_distillation) do
        # would have failed the authoring run's own contract; a fixed family
        # name is a two-paper abstraction we don't have yet.
        "training_loop": {"required_subblocks": ["function_name", "input_shapes"]},
    },
    # family_components.reward_function is declared on the taxonomy node
    # (docs/ssot/taxonomies.yaml) and merged into this plan per run by
    # _merge_node_family_components — do not restate it here.
    "symbol_conventions": {
        "n_agents": "number of agents the trained network is configured for (the paper's n)",
        "n_obs": "number of observed neighbor agents; joint state is 6 + 8*n_obs per Equations (7)-(8)",
        "B": "runtime batch size through the value network",
    },
}


STOCH_FILES = [
    {
        "path": "method/data.py",
        "kind": "paradigm_fixed",
        "produced_by": "package_scaffolder",
        "public_symbols": [
            {
                "name": "make_quadratic_objective",
                "kind": "function",
                "signature": "make_quadratic_objective(dim: int = 4, *, noise_std: float = 0.0, seed: int = 0) -> tuple",
            },
            {
                "name": "make_synthetic_classification",
                "kind": "function",
                "signature": "make_synthetic_classification(n_samples: int = 128, n_features: int = 4, *, seed: int = 0) -> tuple",
            },
        ],
    },
    {"path": "method/example_data/README.md", "kind": "paradigm_fixed", "produced_by": "package_scaffolder"},
    {
        "path": "method/model.py",
        "kind": "paradigm_shaped",
        "produced_by": "architecture_coder",
        "public_symbols": [
            {
                "name": "ObjectiveModel",
                "kind": "class",
                "inherits": "torch.nn.Module",
                "required_methods": ["forward(self, x) -> Tensor"],
            },
        ],
    },
    {
        "path": "method/training.py",
        "kind": "paradigm_shaped",
        "produced_by": "architecture_coder",
        "public_symbols": [
            {
                "name": "build_model",
                "kind": "function",
                "signature": "build_model(input_dim: int, output_dim: int = 1, **arch_kwargs) -> ObjectiveModel",
            },
            {
                "name": "run_optimization_trace",
                "kind": "function",
                "signature": (
                    "run_optimization_trace(objective, initial_params, data, optimizer, "
                    "*, num_steps: int = 100, seed: int = 0) -> dict"
                ),
            },
        ],
    },
    {
        "path": "method/method.py",
        "kind": "method_shaped",
        "produced_by": "method_coder",
        "public_symbols": [
            {"name": "optimize", "kind": "function", "signature": "from spec.comparison.pluggable_component.signature", "contract": "from pluggable_component.contract"},
            {
                "name": "optimizer_step",
                "kind": "function",
                "signature": (
                    "optimizer_step(params, grads, state, *, step_size: float, "
                    "beta1: float, beta2: float, epsilon: float) -> tuple"
                ),
            },
        ],
    },
    {"path": "method/__init__.py", "kind": "derived", "produced_by": "init_finalizer"},
    {"path": "requirements.txt", "kind": "derived", "produced_by": "init_finalizer"},
    {"path": "method/README.md", "kind": "paradigm_fixed", "produced_by": "package_scaffolder"},
]


STOCH_DEPENDENCY_GRAPH = {
    "method/data.py": [],
    "method/model.py": [],
    "method/method.py": [],
    "method/training.py": ["method/model.py", "method/method.py"],
}


# The generic plan for provisional families with no committed ancestor
# (item 4). Ownership mirrors the shared rows every family has; the shaped
# rows stay deliberately loose — the coders receive their real contracts
# from the spec and the authored pack, not from this manifest. model.py's
# class count is flexible (a brand-new family's component count is unknown
# by definition) and training.py's functions are producer-defined
# placeholders; the 2b validator honors both (see
# validate_architecture_coder_output.py).
PROVISIONAL_PLAN_KEY = "_provisional"

PROVISIONAL_FILES = [
    {
        "path": "method/data.py",
        "kind": "paradigm_fixed",
        "produced_by": "package_scaffolder",
        "public_symbols": [
            {
                "name": "load_data",
                "kind": "function",
                "signature": (
                    "load_data(path: str | os.PathLike | None = None, *, "
                    "seed: int = 0) -> dict"
                ),
            }
        ],
    },
    {"path": "method/example_data/README.md", "kind": "paradigm_fixed", "produced_by": "package_scaffolder"},
    {
        "path": "method/model.py",
        "kind": "paradigm_shaped",
        "produced_by": "architecture_coder",
        "class_count": "flexible",
        "public_symbols": [
            {"name": "<MethodComponentClass>", "kind": "class"},
        ],
    },
    {
        "path": "method/training.py",
        "kind": "paradigm_shaped",
        "produced_by": "architecture_coder",
        "public_symbols": [
            {"name": "<training_functions>", "kind": "function"},
        ],
    },
    {
        "path": "method/method.py",
        "kind": "method_shaped",
        "produced_by": "method_coder",
        "public_symbols": [
            {"name": "<pluggable_component.name>", "kind": "function", "signature": "from spec.comparison.pluggable_component.signature", "contract": "from pluggable_component.contract"},
            {"name": "<method_helpers>", "kind": "function", "all_top_level_functions_are_public": True},
        ],
    },
    {"path": "method/__init__.py", "kind": "derived", "produced_by": "init_finalizer"},
    {"path": "requirements.txt", "kind": "derived", "produced_by": "init_finalizer"},
    {"path": "method/README.md", "kind": "paradigm_fixed", "produced_by": "package_scaffolder"},
]

PROVISIONAL_DEPENDENCY_GRAPH = {
    "method/data.py": [],
    "method/model.py": [],
    "method/method.py": ["method/model.py"],
    "method/training.py": ["method/model.py", "method/method.py"],
    "method/__init__.py": [
        "method/data.py",
        "method/method.py",
        "method/model.py",
        "method/training.py",
    ],
}

# Time-series forecasting (R2C-070, committed 2026-08-06). Shaped from the
# structure the five pdfgnn rolls actually delivered under the provisional
# plan, with the audit-driven tightenings: ForecastResult is defined in
# method.py (the loop2 delivery defined it in training.py and the export was
# unreachable — R2C-053/054 territory), and training.py owns build_model +
# train_model so the training entry point is a named, validatable surface.
TSF_FILES = [
    {
        "path": "method/data.py",
        "kind": "paradigm_fixed",
        "produced_by": "package_scaffolder",
        "public_symbols": [
            {
                "name": "load_data",
                "kind": "function",
                "signature": (
                    "load_data(path: str | os.PathLike | None = None, *, "
                    "seed: int = 0) -> dict"
                ),
            }
        ],
    },
    {"path": "method/example_data/README.md", "kind": "paradigm_fixed", "produced_by": "package_scaffolder"},
    {
        "path": "method/target_scaling.py",
        "kind": "paradigm_fixed",
        "produced_by": "package_scaffolder",
        "public_symbols": [
            {"name": "fit_target_scaling_state", "kind": "function"},
            {"name": "transform_targets", "kind": "function"},
            {"name": "inverse_forecast_output", "kind": "function"},
        ],
    },
    {
        "path": "method/training_history.py",
        "kind": "paradigm_fixed",
        "produced_by": "package_scaffolder",
        "public_symbols": [
            {"name": "record_training_history", "kind": "function"},
        ],
    },
    {
        "path": "method/model.py",
        "kind": "paradigm_shaped",
        "produced_by": "architecture_coder",
        "class_count": "flexible",
        "public_symbols": [
            {"name": "<ForecastModelClass>", "kind": "class"},
        ],
    },
    {
        "path": "method/training.py",
        "kind": "paradigm_shaped",
        "produced_by": "architecture_coder",
        "public_symbols": [
            {"name": "build_model", "kind": "function"},
            {"name": "train_model", "kind": "function"},
        ],
    },
    {
        "path": "method/method.py",
        "kind": "method_shaped",
        "produced_by": "method_coder",
        "public_symbols": [
            {"name": "<pluggable_component.name>", "kind": "function", "signature": "from spec.comparison.pluggable_component.signature", "contract": "from pluggable_component.contract"},
            {"name": "ForecastResult", "kind": "class"},
            {"name": "<method_helpers>", "kind": "function", "all_top_level_functions_are_public": True},
        ],
    },
    {"path": "method/__init__.py", "kind": "derived", "produced_by": "init_finalizer"},
    {"path": "requirements.txt", "kind": "derived", "produced_by": "init_finalizer"},
    {"path": "method/README.md", "kind": "paradigm_fixed", "produced_by": "package_scaffolder"},
]

TSF_DEPENDENCY_GRAPH = {
    "method/data.py": [],
    "method/target_scaling.py": [],
    "method/training_history.py": [],
    "method/model.py": [],
    "method/method.py": ["method/model.py", "method/target_scaling.py"],
    "method/training.py": [
        "method/model.py",
        "method/method.py",
        "method/target_scaling.py",
        "method/training_history.py",
    ],
    "method/__init__.py": [
        "method/data.py",
        "method/method.py",
        "method/model.py",
        "method/target_scaling.py",
        "method/training_history.py",
        "method/training.py",
    ],
}


# Exact family-owned bridge from the schema-2 contract to the callable that
# forecasting probes execute. ``model`` is otherwise necessarily opaque in
# the pluggable signature, and neither a sole architecture block nor that
# parameter spelling is authority to inject a constructed object.
TSF_RUNTIME_EXECUTION = {
    "schema_version": "1.0",
    "construction": {
        "architecture_root": "architecture.model",
        "route": {
            "kind": "builder",
            "module": "method.training",
            "callable": "build_model",
        },
    },
    "pluggable_call": {
        "module": "method.method",
        "callable": "forecast",
        "positional_parameters": [
            "model",
            "history",
            "static_features",
            "time_varying_features",
            "graph",
            "entity_ids",
            "target_scaling_state",
        ],
        "keyword_parameters": ["seed"],
        "input_bindings": {
            "pluggable_component.input.model": {
                "kind": "constructed_model",
                "architecture_root": "architecture.model",
            },
            "pluggable_component.input.history": {
                "kind": "typed_fixture",
                "semantic_role": "history",
                "relational_root": "batch.demand",
            },
            "pluggable_component.input.static_features": {
                "kind": "typed_fixture",
                "semantic_role": "static_features",
                "relational_root": "batch.static_features",
            },
            "pluggable_component.input.time_varying_features": {
                "kind": "typed_fixture",
                "semantic_role": "time_varying_features",
                "relational_root": "batch.time_varying_features",
            },
            "pluggable_component.input.graph": {
                "kind": "relational_graph_or_graph_free_none",
                "semantic_role": "graph",
            },
            "pluggable_component.input.entity_ids": {
                "kind": "target_scaling_entity_ids",
                "semantic_role": "stable_entity_ids",
                "relational_root": "batch.entity_ids",
            },
            "pluggable_component.input.target_scaling_state": {
                "kind": "target_scaling_state",
                "semantic_role": "target_scaling_state",
            },
            "pluggable_component.input.seed": {
                "kind": "typed_fixture",
                "semantic_role": "seed",
            },
        },
        "additional_inputs": "typed_paradigm_extras_by_exact_name",
    },
    "output_grammar": {
        "contract_root": "pluggable_component.output",
        "kind": "attribute_record",
        "module": "method.method",
        "type_name": "ForecastResult",
        "fields": {
            "mean": "mean",
            "variance": "variance",
            "samples": "samples",
            "distribution_params": "distribution_params",
        },
        # The committed family definition has shipped both spellings. The
        # key sets are closed: consumers select one exact set and never infer
        # a scale parameter from arbitrary mapping names.
        "distribution_params": {
            "kind": "student_t",
            "closed": True,
            "allowed_key_sets": [
                {"location": "mu", "scale": "scale", "degrees_of_freedom": "df"},
                {"location": "mu", "scale": "sigma", "degrees_of_freedom": "df"},
            ],
        },
    },
    "conditional_probes": {
        "autoregressive_path_dependence": {
            "probe_ref": "time_series_forecasting.path_dependence",
            "authority": "methodology_verification_probe_refs",
        },
    },
}


# Closed version-one forecasting policy for training-only per-series scaling.
# This is a static contract only. The build-plan loader does not fit or write
# scale state, and consumers must use the separately proven fitting range.
TSF_TARGET_SCALING = {
    "schema_version": "1.0.0",
    "mode": "per_series_affine",
    "supported_modes": ["none", "per_series_affine"],
    "fitting_role": "fitting",
    "entity_id_root": "batch.entity_ids",
    "target_root": "batch.targets",
    "target_entity_axis": 0,
    "target_protocol_axis": 1,
    "statistic": "mean_absolute",
    "centering": "none",
    "epsilon": 1.0e-8,
    "nonfinite_policy": "reject",
    "zero_series_policy": (
        "unit_scale_when_all_absolute_values_lte_epsilon"
    ),
    "state_artifact": ".pipeline/target_scaling_state.json",
    "output_inversion": {
        "location": "multiply_scale",
        "samples": "multiply_scale",
        "distribution_scale": "multiply_absolute_scale",
        "variance": "multiply_squared_scale",
        "unitless_shape": "unchanged",
    },
}


# Exact family-owned crosswalk from the scaling policy's logical roots to the
# generated schema-2 fitting call.  The architecture producer declares these
# ordinary typed inputs; it does not invent another root spelling or write the
# final pipeline-owned state artifact.  ``fitting_range`` remains opaque in
# ArchContractV2 because the trusted R2C-077 carrier is a structured pipeline
# object, not a tensor shape or a paper-authored partition declaration.
TSF_TARGET_SCALING_EXECUTION = {
    "schema_version": "1.0.0",
    "helper": {
        "module": "method.target_scaling",
        "fit_callable": "fit_target_scaling_state",
        "transform_callable": "transform_targets",
        "inverse_callable": "inverse_forecast_output",
    },
    "training_call": {
        "module": "method.training",
        "callable": "train_model",
        "model_input_root": "training_loop.input.model",
        "entity_ids_input_root": "training_loop.input.entity_ids",
        "targets_input_root": "training_loop.input.targets",
        "fitting_range_input_root": "training_loop.input.fitting_range",
        "state_result": {
            "kind": "mapping_key",
            "key": "target_scaling_state",
        },
    },
}


# Exact family-owned crosswalk for the single-run/single-holdout history
# carrier.  Ranges and config identity remain opaque schema-2 boundaries;
# Stage 2.d supplies only its bounded validator fixture, while live acceptance
# rebinds the returned record to pipeline-minted lineage and scaling state.
TSF_TRAINING_HISTORY_EXECUTION = {
    "schema_version": "1.0.0",
    "helper": {
        "module": "method.training_history",
        "record_callable": "record_training_history",
    },
    "training_call": {
        "module": "method.training",
        "callable": "train_model",
        "fitting_range_input_root": "training_loop.input.fitting_range",
        "selection_range_input_root": "training_loop.input.selection_range",
        "seed_input_root": "training_loop.input.seed",
        "config_id_input_root": "training_loop.input.config_id",
        "history_result": {
            "kind": "mapping_key",
            "key": "training_history",
        },
    },
}


STATIC_PLAN_BY_PARADIGM: dict[str, dict[str, Any]] = {
    PROVISIONAL_PLAN_KEY: {
        "default_pluggable_name": None,
        "package_manifest": _manifest(
            "Generic package ownership for provisional (gap-path) families.",
            PROVISIONAL_FILES,
            PROVISIONAL_DEPENDENCY_GRAPH,
        ),
        # Deliberately minimal: a brand-new family's contract blocks are
        # unknown; the pluggable name equality is the one invariant every
        # pack carries. The fidelity review remains the backstop.
        "arch_contract_requirements": {
            "required_blocks": {
                "pluggable_component.name": {"must_equal": "pluggable_component.name"},
            },
        },
    },
    "active_learning": {
        "default_pluggable_name": "select_batch",
        "package_manifest": _manifest(
            "Spec-derived structural contract for method/ files. Producer "
            "ownership is fixed by the pipeline; pluggable names/signatures and "
            "additional model methods come from method_spec.json.",
            AL_FILES, AL_DEPENDENCY_GRAPH),
        "arch_contract_requirements": deepcopy(AL_ARCH_CONTRACT_REQUIREMENTS),
    },
    "active_learning/bayesian": {
        "default_pluggable_name": "select_batch",
        "package_manifest": _manifest(
            "Spec-derived structural contract for method/ files. Producer "
            "ownership is fixed by the pipeline; pluggable names/signatures and "
            "additional model methods come from method_spec.json.",
            AL_FILES, AL_DEPENDENCY_GRAPH),
        "arch_contract_requirements": deepcopy(AL_ARCH_CONTRACT_REQUIREMENTS),
    },
    "active_learning/batch_acquisition": {
        "default_pluggable_name": "select_batch",
        "package_manifest": _manifest(
            "Spec-derived structural contract for method/ files. Producer "
            "ownership is fixed by the pipeline; pluggable names/signatures and "
            "additional model methods come from method_spec.json.",
            AL_FILES, AL_DEPENDENCY_GRAPH),
        "arch_contract_requirements": deepcopy(AL_ARCH_CONTRACT_REQUIREMENTS),
    },
    "knowledge_distillation": {
        "default_pluggable_name": "compute_distillation_loss",
        "package_manifest": _manifest("Knowledge-distillation package ownership.", KD_ROOT_FILES, KD_DEPENDENCY_GRAPH),
        "arch_contract_requirements": {
            "required_blocks": {
                "data_loader.load_data_returns": {"type": "dict[str, shape_string]", "min_keys": ["x_train", "y_train", "x_test", "y_test"]},
                "architecture.student": {"required_subblocks": ["class_name", "forward.input", "forward.output_type"]},
                "architecture.teacher": {"required_subblocks": ["class_name", "forward.input", "forward.output_type"]},
                "pluggable_component.name": {"must_equal": "pluggable_component.name"},
                "pluggable_component.batch_dict_shape": {"type": "dict[str, shape_string]", "min_keys": ["inputs"]},
                "training_loop.function_name": {"must_equal": "train_with_distillation"},
                "training_loop.input_shapes": {"type": "dict[str, shape_string]", "min_keys": ["x_train", "y_train"]},
            },
        },
    },
    "knowledge_distillation/detection": {
        "default_pluggable_name": "compute_distillation_loss",
        "package_manifest": _manifest("Detection-KD package ownership.", KD_DETECTION_FILES, KD_DEPENDENCY_GRAPH),
        "arch_contract_requirements": {
            "required_blocks": {
                "data_loader.load_data_returns": {"type": "dict[str, shape_string]", "min_keys": ["images_train", "targets_train", "images_test", "targets_test"]},
                "architecture.student": {"required_subblocks": ["class_name", "forward.input", "forward.output_type"]},
                "architecture.teacher": {"required_subblocks": ["class_name", "forward.input", "forward.output_type"]},
                "pluggable_component.name": {"must_equal": "pluggable_component.name"},
                "pluggable_component.batch_dict_shape": {"type": "dict[str, shape_string]", "min_keys": ["inputs", "targets"]},
                "training_loop.function_name": {"must_equal": "train_with_distillation"},
                "training_loop.input_shapes": {"type": "dict[str, shape_string]", "min_keys": ["images_train", "targets_train"]},
            },
        },
    },
    "knowledge_distillation/detection/cross_modal": {
        "default_pluggable_name": "compute_distillation_loss",
        "package_manifest": _manifest("Cross-modal detection-KD package ownership.", KD_CROSS_MODAL_FILES, KD_DEPENDENCY_GRAPH),
        "arch_contract_requirements": {
            "required_blocks": {
                "data_loader.load_data_returns": {"type": "dict[str, shape_string]", "min_keys": ["student_inputs_train", "teacher_inputs_train", "targets_train", "student_inputs_test", "teacher_inputs_test", "targets_test"]},
                "architecture.student": {"required_subblocks": ["class_name", "forward.input", "forward.output_type"]},
                "architecture.teacher": {"required_subblocks": ["class_name", "forward.input", "forward.output_type"]},
                "pluggable_component.name": {"must_equal": "pluggable_component.name"},
                "pluggable_component.batch_dict_shape": {"type": "dict[str, shape_string]", "min_keys": ["student_inputs", "teacher_inputs", "targets"]},
                "training_loop.function_name": {"must_equal": "train_with_distillation"},
                "training_loop.input_shapes": {"type": "dict[str, shape_string]", "min_keys": ["student_inputs_train", "teacher_inputs_train", "targets_train"]},
            },
        },
    },
    "vision_transformer": {
        "default_pluggable_name": "group_tokens",
        "package_manifest": _manifest(
            "Vision-transformer package ownership.",
            VIT_FILES,
            {
                "method/data.py": [],
                "method/model.py": ["method/method.py"],
                "method/method.py": [],
                "method/training.py": ["method/model.py"],
                "method/__init__.py": ["method/data.py", "method/method.py", "method/model.py", "method/training.py"],
            },
        ),
        "arch_contract_requirements": {
            "required_blocks": {
                "data_loader.load_data_returns": {"type": "dict[str, shape_string]", "min_keys": ["train_loader", "test_loader"]},
                "architecture.model": {"required_subblocks": ["class_name", "forward.input", "forward.output_type"]},
                "pluggable_component.name": {"must_equal": "group_tokens"},
                "pluggable_component.input_shapes": {"type": "dict[str, shape_string]", "min_keys": ["tokens"]},
                "pluggable_component.output_shape": {"type": "shape_string"},
                "training_loop.function_name": {"must_equal": "train_epoch"},
                "training_loop.input_shapes": {"type": "dict[str, shape_string]", "min_keys": ["x", "y"]},
            },
        },
    },
    "domain_adaptation": {
        "default_pluggable_name": "generate_pseudo_labels",
        "package_manifest": _manifest(
            "Domain-adaptation package ownership.",
            DA_FILES,
            {
                "method/data.py": [],
                "method/model.py": [],
                "method/method.py": ["method/model.py", "method/data.py"],
                "method/training.py": ["method/model.py"],
                "method/__init__.py": ["method/data.py", "method/method.py", "method/model.py", "method/training.py"],
            },
        ),
        "arch_contract_requirements": {
            "required_blocks": {
                "data_loader.load_data_returns": {"type": "dict[str, shape_string]"},
                "architecture.detector": {"required_subblocks": ["class_name", "forward.input", "forward.output_keys.pred_boxes", "forward.output_keys.pred_scores", "forward.output_keys.pred_classes"]},
                "pluggable_component.name": {"must_equal": "pluggable_component.name"},
                "pluggable_component.input_shapes": {"type": "dict[str, shape_string]"},
                "pluggable_component.output_shape": {"type": "shape_string"},
            },
        },
    },
    "motion_planning": {
        "default_pluggable_name": "plan",
        "package_manifest": _manifest("Motion-planning package ownership.", MP_FILES, MP_DEPENDENCY_GRAPH),
        "arch_contract_requirements": deepcopy(MP_ARCH_CONTRACT_REQUIREMENTS),
    },
    "motion_planning/sampling_based": {
        "default_pluggable_name": "plan",
        "package_manifest": _manifest("Sampling-based motion-planning package ownership.", MP_FILES, MP_DEPENDENCY_GRAPH),
        "arch_contract_requirements": deepcopy(MP_ARCH_CONTRACT_REQUIREMENTS),
    },
    "motion_planning/optimization_based": {
        "default_pluggable_name": "plan",
        "package_manifest": _manifest("Optimization-based motion-planning package ownership.", MP_FILES, MP_DEPENDENCY_GRAPH),
        "arch_contract_requirements": deepcopy(MP_ARCH_CONTRACT_REQUIREMENTS),
    },
    # Promoted from the SRL run's provisional pack (R2C-029, 2026-07-28).
    # This entry is the promotion's HARD BLOCKER fix: committed nodes resolve
    # by exact match only (_static_plan_key), so without it the node would
    # serve while load_build_plan returned None and readiness went red. It is
    # also the fix for 3 of the 4 stage_2b.halt failures (the planner-shaped
    # MP_FILES manifest an RL method cannot satisfy); the fourth
    # (family_components.reward_function undeclared) is fixed by the node's
    # declaration in taxonomies.yaml, merged per run by
    # _merge_node_family_components.
    "motion_planning/rl_collision_avoidance": {
        "default_pluggable_name": "train_and_plan",
        "package_manifest": _manifest(
            "RL collision-avoidance package ownership.",
            RL_CA_FILES,
            RL_CA_DEPENDENCY_GRAPH,
        ),
        "arch_contract_requirements": deepcopy(RL_CA_ARCH_CONTRACT_REQUIREMENTS),
    },
    "stochastic_optimization": {
        "default_pluggable_name": "optimize",
        "package_manifest": _manifest(
            "Stochastic-optimization package ownership.",
            STOCH_FILES,
            STOCH_DEPENDENCY_GRAPH,
        ),
        "arch_contract_requirements": {
            "required_blocks": {
                "data_loader.load_data_returns": {"type": "dict[str, shape_string]", "min_keys": ["x", "y"]},
                "architecture.objective_model": {"required_subblocks": ["class_name", "forward.input", "forward.output_type"]},
                "optimizer_state": {"required_subblocks": ["step", "params", "first_moment", "second_moment"]},
                "pluggable_component.name": {"must_equal": "pluggable_component.name"},
                "pluggable_component.input_shapes": {"type": "dict[str, shape_string]", "min_keys": ["initial_params"]},
                "pluggable_component.output_shape": {"type": "shape_string"},
            },
            "symbol_conventions": {
                "P": "number of optimized scalar parameters",
                "B": "runtime minibatch size",
                "T": "number of optimization steps",
            },
        },
    },
    # Committed 2026-08-06 (R2C-070): the forecasting family's own plan lifts
    # the neutral-plan delivery cap that held every pdfgnn roll at
    # uncertified_new_territory. Requirements stay within the universal
    # ArchContract skeleton (extra='forbid'): the pack-arch-schema
    # contradiction class (2026-08-03 live finding) is a pack demanding
    # contract fields pydantic forbids, and every path below exists on the
    # skeleton.
    "time_series_forecasting": {
        "default_pluggable_name": "forecast",
        "runtime_execution": deepcopy(TSF_RUNTIME_EXECUTION),
        "target_scaling": deepcopy(TSF_TARGET_SCALING),
        "target_scaling_execution": deepcopy(
            TSF_TARGET_SCALING_EXECUTION
        ),
        "training_history_execution": deepcopy(
            TSF_TRAINING_HISTORY_EXECUTION
        ),
        "package_manifest": _manifest(
            "Time-series-forecasting package ownership.",
            TSF_FILES,
            TSF_DEPENDENCY_GRAPH,
        ),
        "arch_contract_requirements": {
            "required_blocks": {
                "data_loader.load_data_returns": {"type": "dict[str, shape_string]"},
                "architecture.model": {"required_subblocks": ["class_name", "forward.input", "forward.output_type"]},
                "pluggable_component.name": {"must_equal": "pluggable_component.name"},
                "training_loop.function_name": {"must_equal": "train_model"},
                "training_loop.input_shapes": {
                    "type": "dict[str, shape_string]",
                    "min_keys": [
                        "model",
                        "entity_ids",
                        "targets",
                        "fitting_range",
                        "selection_range",
                        "seed",
                        "config_id",
                    ],
                },
            },
            "symbol_conventions": {
                "N": "number of series (articles/sensors/locations)",
                "T": "number of time steps on the realized axis",
                "P": "context length (historical steps conditioned on)",
                "K": "forecast horizon (future steps predicted)",
            },
        },
    },
}
