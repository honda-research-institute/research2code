"""Single loader/API for the taxonomy SSOT (`docs/ssot/taxonomies.yaml`).

Runtime result of the paradigm -> taxonomy migration
(the taxonomy migration implementation plan note (internal, not shipped)). This module has
two layers:

1. **The new API** -- :func:`load_taxonomy` parses and validates the SSOT into a
   :class:`Taxonomy`: method-axis nodes (Root -> Family -> Variant -> sub-variant)
   with intra-file inheritance, the task-domain axis, alias tables, and
   :func:`assert_classification_tuple` for ``(method_variant, task_domain)`` pairs.

2. **The compatibility renderer** -- mirrors the historical consumer surface
   (``merged_stage_review_focus``, ``merged_common_smoke_bugs`` ...) from the
   taxonomy node itself. Phase 4.1 retired the guide-file fallback; these
   helpers no longer walk markdown guide files.

Schema contract: ``docs/ssot/taxonomy-node-schema.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SSOT_PATH = Path("docs/ssot/taxonomies.yaml")

METHOD_ROOT_IDS = ("TE", "CLC", "SC", "AD", "PP", "OM", "GEN", "DA", "INF")
REVIEW_STAGE_IDS = (
    "stage_1_analyzer",
    "stage_2b_architecture",
    "stage_2c_method",
    "stage_2x_params",
    "stage_3a_notebook",
)

# Fields a child node inherits from its ancestors (root <- family <- variant <-
# sub-variant). Mapping fields are shallow-merged; the check lists are
# concatenated then de-duplicated by `id` (parent first), matching the legacy
# additive `merged_stage_review_focus` semantics.
_INHERITABLE_MAPPING_FIELDS = (
    "fingerprint",
    "priors",
    "provenance_params",
    "tier_heuristic",
    "smoke_economics",
    "scaffold_hints",
    "pluggable_component",
    # Scaffolded ahead of their consumers so the node can carry/serve/lint them
    # (configurable-node-schema-design.md): `notebook_layout` populates in Phase
    # 1.6, `paradigm_extras` in Phase 5.7. Behavior-neutral until a node declares
    # them — no layer carries them today, so `effective_node` adds no key.
    "notebook_layout",
    "paradigm_extras",
    # `model_defaults` (Phase 5.7b): per-paradigm system_inferred fallbacks for
    # model-architecture params (hidden_dim, dropout_rate) — same merge semantics
    # as paradigm_extras (shallow-merge by name; a sub-variant adds/overrides).
    "model_defaults",
    # `demo_success` (demo-success-semantics design, approved 2026-07-16):
    # the family's declared success/failure markers for the headline demo
    # section, read by the deterministic post-smoke verdict pass
    # (scripts/demo_verdict.py). Kit territory, next to smoke_bugs; committed
    # families and provisional/gap packs share this one declaration surface.
    "demo_success",
    # `demo_skill` (R2C-086): the family-owned, machine-readable comparison
    # policy for deciding whether an executed demo demonstrated task skill.
    # This remains separate from coarse execution markers in `demo_success`:
    # a printed PASS is presentation, while the structured evaluation record
    # is recomputed against this contract.  Provisional packs use the same
    # inherited declaration surface.
    "demo_skill",
    # `family_components` (arch-contract headroom, approved 2026-07-16): the
    # pack-declared legal names for `arch_contract.family_components` blocks
    # (with per-name `required`/`required_entries`/`description`). Flows into
    # the build plan's `arch_contract_requirements` and is enforced in both
    # directions by `scripts/validate_arch_contract.py`. Shallow-merge by
    # component name, so a child family adds/overrides components. Added for
    # SRL's `reward_function` (2026-07-15, rejected twice by the universal
    # schema) and the shape ADAM's `optimizer_state` was hard-added for
    # (2026-07-08).
    "family_components",
    # `params_derivation` (plan item 9, 2026-07-21): the node/pack-declared
    # parameter entries the deterministic deriver must emit (or drop) for
    # this paradigm, keyed by param name. Entry kinds: `config_path`
    # (pipeline configuration the pluggable component requires — the
    # DomIndOnto KBP halt), `derived_statistic` (a formula over other
    # params/spec facts the review focus demands documented — the fedavg
    # `u = nE/(KB)` halt), and `suppress` (drop an inapplicable
    # convention param such as hidden_dim on a non-ML paradigm). This is
    # the machine-readable half of `stage_review_focus.stage_2x_params`:
    # the reviewer enforces prose checks, and this block is what lets the
    # producer satisfy them. Shallow-merge by param name.
    "params_derivation",
    # `package_manifest` (DomIndOnto KBP 2026-07-29 stage-2d halt): the pack
    # author's per-file ownership declaration. Consumed today ONLY for its
    # explicitly-empty `public_symbols: []` entries, which drop the neutral
    # plan's placeholder requirements for files the paradigm deliberately
    # leaves symbol-less (a KBP pipeline has no training functions). Non-empty
    # entries are authoring documentation until a second concrete case fixes
    # the merge's shape. No committed node declares this field.
    "package_manifest",
    # `build_plan` (R2C-032, approved 2026-07-28): the pack author's declared
    # build-plan routing (`{source: inherit_parent | neutral, reasoning}`),
    # read by `scripts/build_plan.py::load_build_plan` for PROVISIONAL nodes
    # only — committed resolution stays exact-match pinned. A pack that
    # declares nothing keeps the implicit ancestor walk, byte-identical to
    # before (the 2026-07-28 SRL stage-2b halt: an RL gap pack silently
    # inherited the motion-planning manifest it could never satisfy).
    "build_plan",
)
_INHERITABLE_CHECK_LIST_FIELDS = ("semantic_checks", "smoke_bugs")
_INHERITABLE_DIMENSION_LIST_FIELDS = ("scenario_assumption_dimensions",)
_INHERITABLE_DOMAIN_CHECK_FIELD = "domain_checks"

# The named deterministic checks a `demo_success.checks[]` entry may reference
# (executed by scripts/demo_verdict.py). One vocabulary, owned here next to the
# field declaration so the SSOT lint and the verdict pass can never disagree.
DEMO_CHECK_KINDS = ("beats_chance",)

# Closed vocabularies for the family-owned `demo_skill` contract.  The
# taxonomy lint owns the nested shape; these constants keep its grammar and
# downstream consumers from inventing slightly different spellings.
DEMO_SKILL_SCHEMA_VERSION = "1.0.0"
DEMO_SKILL_METRIC_DIRECTIONS = ("lower_is_better", "higher_is_better")
DEMO_SKILL_DECISION_POLICIES = ("all_required",)
DEMO_SKILL_COMPARATOR_ROLES = (
    "degeneracy_floor",
    "minimum_competence",
    "context",
)
DEMO_SKILL_COMPARATOR_IMPLEMENTATIONS = (
    "predict_zero",
    "repeat_last_pre_window",
    "constant_prediction",
)


class TaxonomyError(RuntimeError):
    """Raised when a taxonomy operation cannot be resolved (unknown slug, bad tuple)."""


@dataclass(frozen=True)
class TaxonomyDiagnostic:
    severity: Literal["error", "warning"]
    code: str
    message: str
    node: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.node:
            payload["node"] = self.node
        return payload


@dataclass(frozen=True)
class VariantNode:
    slug: str
    taxonomy_id: str
    name: str
    root_id: str
    family_id: str
    parent_slug: str | None
    status: str
    legacy_paradigm: str | None
    aliases: tuple[str, ...]
    fields: dict[str, Any]
    sub_variants: dict[str, "VariantNode"] = field(default_factory=dict)
    # True only for run-local provisional-pack nodes grafted by
    # `load_taxonomy(provisional_packs_dir=...)`; never set by the SSOT.
    provisional: bool = False

    @property
    def is_populated(self) -> bool:
        return self.status == "populated"


@dataclass(frozen=True)
class FamilyNode:
    id: str
    name: str
    root_id: str
    description: str
    legacy_paradigm: str | None
    status: str
    fields: dict[str, Any]
    variants: dict[str, VariantNode]

    @property
    def is_populated(self) -> bool:
        return self.status == "populated"


@dataclass(frozen=True)
class RootNode:
    id: str
    name: str
    archetype: str
    families: dict[str, FamilyNode]


@dataclass(frozen=True)
class DomainNode:
    id: str
    name: str
    leaves: dict[str, str]  # leaf slug -> leaf name


@dataclass
class Taxonomy:
    repo_root: Path
    source_path: Path
    roots: dict[str, RootNode]
    task_domains: dict[str, DomainNode]
    aliases: dict[str, str]
    universal_checks: list[dict[str, Any]]
    diagnostics: tuple[TaxonomyDiagnostic, ...]
    # Flat indices (built once at load).
    _variants_by_slug: dict[str, VariantNode] = field(default_factory=dict)
    _variants_by_taxonomy_id: dict[str, VariantNode] = field(default_factory=dict)
    _node_by_legacy: dict[str, VariantNode | FamilyNode] = field(default_factory=dict)
    _leaf_to_domain: dict[str, str] = field(default_factory=dict)

    # -- diagnostics ------------------------------------------------------
    @property
    def errors(self) -> list[TaxonomyDiagnostic]:
        return [d for d in self.diagnostics if d.severity == "error"]

    @property
    def warnings(self) -> list[TaxonomyDiagnostic]:
        return [d for d in self.diagnostics if d.severity == "warning"]

    # -- method-axis lookup ----------------------------------------------
    def resolve_alias(self, slug: str) -> str:
        """Return the canonical variant slug for an alias, else the slug itself."""
        return self.aliases.get(slug, slug)

    def variant(self, slug_or_id: str) -> VariantNode | None:
        """Resolve a variant by slug, `taxonomy_id`, or alias. None if unknown."""
        if slug_or_id in self._variants_by_taxonomy_id:
            return self._variants_by_taxonomy_id[slug_or_id]
        canonical = self.resolve_alias(slug_or_id)
        return self._variants_by_slug.get(canonical)

    def node_for_legacy(self, paradigm_id: str) -> VariantNode | FamilyNode | None:
        """Return the node that supersedes a `paradigms/` id (variant or family)."""
        return self._node_by_legacy.get(paradigm_id)

    # -- task-domain lookup ----------------------------------------------
    def task_domain(self, id_or_leaf: str) -> tuple[str, str | None] | None:
        """Resolve a domain id or leaf slug to `(domain_id, leaf_slug | None)`."""
        if id_or_leaf in self.task_domains:
            return (id_or_leaf, None)
        if id_or_leaf in self._leaf_to_domain:
            return (self._leaf_to_domain[id_or_leaf], id_or_leaf)
        return None

    @property
    def all_variants(self) -> list[VariantNode]:
        return list(self._variants_by_slug.values())


# =============================================================================
# Loading + validation
# =============================================================================


@lru_cache(maxsize=None)
def _load_committed_taxonomy(repo_root: Path, source_path: Path | str) -> Taxonomy:
    path = Path(source_path)
    abs_path = path if path.is_absolute() else repo_root / path
    raw = yaml.safe_load(abs_path.read_text(encoding="utf-8")) or {}
    return _build_taxonomy(repo_root, abs_path, raw)


def load_taxonomy(
    repo_root: Path = ROOT,
    source_path: Path | str = DEFAULT_SSOT_PATH,
    provisional_packs_dir: Path | str | None = None,
) -> Taxonomy:
    """Parse + index the SSOT once (cached per `(repo_root, source_path)`).

    `provisional_packs_dir` is the gap-path overlay
    (the gap path serves overlay design note (internal, not shipped)): when a run
    directory's `.pipeline/provisional_packs/` is passed, each installed
    `pack.yaml` is grafted onto the committed view as a node marked
    `provisional=True`, so `serves()` can answer for the run's own gap
    classification. The overlay view is built fresh on every call and is
    NEVER cached — one run's provisional pack must not leak into another
    run or into the committed view.
    """
    committed = _load_committed_taxonomy(repo_root, source_path)
    if provisional_packs_dir is None:
        return committed
    return _overlay_provisional_packs(committed, Path(provisional_packs_dir))


# Tests clear the committed cache through the public name.
load_taxonomy.cache_clear = _load_committed_taxonomy.cache_clear  # type: ignore[attr-defined]


def load_taxonomy_uncached(repo_root: Path, raw: dict[str, Any], source_path: Path | None = None) -> Taxonomy:
    """Build a Taxonomy directly from a parsed mapping (for tests / in-memory edits)."""
    return _build_taxonomy(repo_root, source_path or (repo_root / DEFAULT_SSOT_PATH), raw)


def run_overlay_dir(run_dir: Path | str | None) -> Path | None:
    """The run-local provisional-packs dir, when the gap path installed one.

    Shared key for every downstream consumer that takes `--run-dir`
    (scaffolder, stage-2/3 validators, derive_params, the init finalizer):
    a run's overlay lives at `<run_dir>/.pipeline/provisional_packs/` and
    exists exactly when stage 1's gap path authored + installed a pack.
    Returns None for non-gap runs, so `load_taxonomy(...,
    provisional_packs_dir=run_overlay_dir(run_dir))` is byte-identical to
    the committed view for them (the leak guard's cheap half)."""
    if run_dir is None:
        return None
    d = Path(run_dir) / ".pipeline" / "provisional_packs"
    return d if d.is_dir() else None


PROVISIONAL_GROUP_ID = "PROVISIONAL"

# The committed family-neutral template skeleton (repo-root-relative) served
# to run-local provisional packs that declare/inherit no templates_dir of
# their own (item 4 — stage-2a scaffolding for provisional paradigms).
PROVISIONAL_TEMPLATES_DIR = "paradigms/_provisional/templates"


def is_placeholder_contract_value(value: object) -> bool:
    """True for empty or template-placeholder contract text.

    The 2026-07-06 fedavg pack shipped `interface_hint: 'TODO: describe
    the pluggable interface this paper needs.'` — non-empty, so it passed
    every emptiness check and would have marked the pack populated with
    no usable contract. One predicate, used by BOTH enforcement points
    (overlay populated-status here, authoring floor in
    validate_paradigm_proposal) so they cannot drift apart."""
    text = str(value or "").strip()
    return not text or text.lower().startswith("todo")


def _overlay_provisional_packs(committed: Taxonomy, packs_dir: Path) -> Taxonomy:
    """Return a NEW Taxonomy with each installed provisional pack grafted on.

    The committed instance is never mutated. A pack grafts as a variant of
    the family its `extends` resolves to; a pack with NO `extends` (a new
    top-level proposal, e.g. federated learning 2026-07-06) grafts under a
    synthetic run-local PROVISIONAL group, where inheritance contributes
    nothing and the pack must be self-contained. Either way it reads
    populated (and therefore served) only when it carries the contract
    fields the gap path authored (`fingerprint.what_it_is` +
    `scaffold_hints.interface_hint`, real values — placeholders do not
    count; the same floor `validate_paradigm_proposal.py` enforces) — a
    stub pack stays unserved and fails at the consumer gate. Packs that
    would shadow a committed slug/id/legacy-paradigm are skipped with a
    diagnostic — the overlay may only ADD nodes, never change committed
    answers — with ONE carve-out: a pack whose only collision is a
    committed `status: reserved` placeholder POPULATES that placeholder
    for the run (adopting its committed identity), because reserved
    entries are name-stubs that exist precisely to be filled in and an
    empty stub is not a committed answer (SRL 2026-07-13).
    """
    pack_paths = sorted(packs_dir.glob("*/pack.yaml")) if packs_dir.is_dir() else []
    if not pack_paths:
        return committed

    diagnostics = list(committed.diagnostics)
    roots = dict(committed.roots)
    variants_by_slug = dict(committed._variants_by_slug)
    variants_by_taxonomy_id = dict(committed._variants_by_taxonomy_id)
    node_by_legacy = dict(committed._node_by_legacy)

    def _skip(code: str, message: str, node: str | None = None) -> None:
        diagnostics.append(TaxonomyDiagnostic("warning", code, message, node))

    for pack_path in pack_paths:
        try:
            pack = yaml.safe_load(pack_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            _skip("provisional_unreadable", f"cannot load {pack_path}: {exc}")
            continue
        if not isinstance(pack, dict):
            _skip("provisional_unreadable", f"{pack_path} is not a mapping")
            continue

        legacy = pack.get("legacy_paradigm")
        taxonomy_id = pack.get("taxonomy_id") or legacy or ""
        slug = str(taxonomy_id).rsplit("/", 1)[-1].strip()
        if not (isinstance(legacy, str) and legacy and slug):
            _skip("provisional_incomplete",
                  f"{pack_path} lacks legacy_paradigm/taxonomy_id", str(taxonomy_id) or None)
            continue

        if not pack.get("extends"):
            # New TOP-LEVEL proposal: no committed parent exists (that is
            # the whole point of the gap decision), so the pack grafts
            # under a synthetic run-local PROVISIONAL group. The halt-judge
            # diagnosed the old behavior precisely on the 2026-07-06
            # fedavg run: extends-null packs were silently skipped here,
            # so serves() never answered for a correctly-authored pack.
            prov_root = roots.get(PROVISIONAL_GROUP_ID)
            if prov_root is None:
                prov_family = FamilyNode(
                    id=PROVISIONAL_GROUP_ID,
                    name="Provisional (run-local overlay)",
                    root_id=PROVISIONAL_GROUP_ID,
                    description="Synthetic group for top-level provisional "
                                "packs; exists only in overlay views.",
                    legacy_paradigm=None,
                    status="reserved",
                    fields={},
                    variants={},
                )
                prov_root = RootNode(
                    id=PROVISIONAL_GROUP_ID,
                    name="Provisional (run-local overlay)",
                    archetype="provisional",
                    families={prov_family.id: prov_family},
                )
                roots[PROVISIONAL_GROUP_ID] = prov_root
            parent = prov_root.families[PROVISIONAL_GROUP_ID]
        else:
            parent = node_by_legacy.get(str(pack.get("extends") or ""))
            if isinstance(parent, VariantNode) and parent.parent_slug is not None:
                _skip("provisional_extends_unresolved",
                      f"{pack_path} extends {pack.get('extends')!r}, a nested "
                      "sub-variant; overlay grafting supports family and "
                      "top-level-variant parents", str(taxonomy_id))
                continue
            if not isinstance(parent, (FamilyNode, VariantNode)):
                _skip("provisional_extends_unresolved",
                      f"{pack_path} extends {pack.get('extends')!r}, which is "
                      "not a committed family or variant", str(taxonomy_id))
                continue

        fields = {
            k: v
            for k, v in pack.items()
            if k in (
                *_INHERITABLE_MAPPING_FIELDS,
                *_INHERITABLE_CHECK_LIST_FIELDS,
                *_INHERITABLE_DIMENSION_LIST_FIELDS,
                _INHERITABLE_DOMAIN_CHECK_FIELD,
            )
        }
        fingerprint = pack.get("fingerprint") or {}
        hints = pack.get("scaffold_hints") or {}
        carries_contract = bool(
            isinstance(fingerprint, dict)
            and not is_placeholder_contract_value(fingerprint.get("what_it_is"))
            and isinstance(hints, dict)
            and not is_placeholder_contract_value(hints.get("interface_hint"))
        )

        collisions = [
            node for node in (
                variants_by_slug.get(slug),
                variants_by_taxonomy_id.get(str(taxonomy_id)),
                node_by_legacy.get(legacy),
            ) if node is not None
        ]
        if collisions:
            # A provisional pack may POPULATE a committed RESERVED
            # placeholder — that is what reserved entries exist for (the
            # SRL 2026-07-13 halt: the gap path authored a correct pack for
            # `single_agent_rl`, and the unconditional shadow skip blocked
            # it because the SSOT reserves that name as an empty stub). The
            # graft adopts the COMMITTED identity (taxonomy_id, family
            # position, aliases) so downstream consumers see the real node,
            # not a PROVISIONAL/* synonym. Populated committed nodes stay
            # protected exactly as before, and every ambiguous shape skips:
            # collisions on different nodes, nested placeholders,
            # placeholders with children, or a different legacy claim.
            target = collisions[0]
            overlayable = (
                all(n is target for n in collisions)
                and isinstance(target, VariantNode)
                and target.status == "reserved"
                and target.parent_slug is None
                and not target.sub_variants
                and target.legacy_paradigm in (None, legacy)
            )
            if not overlayable:
                statuses = ", ".join(sorted({
                    f"{getattr(n, 'taxonomy_id', '?')} ({n.status})"
                    for n in collisions
                }))
                _skip("provisional_shadows_committed",
                      f"{pack_path} would shadow an existing slug/taxonomy_id/"
                      f"legacy_paradigm ({statuses}); overlay nodes may only "
                      "add, or populate a single reserved placeholder",
                      str(taxonomy_id))
                continue
            node = VariantNode(
                slug=target.slug,
                taxonomy_id=target.taxonomy_id,
                name=pack.get("name", target.name),
                root_id=target.root_id,
                family_id=target.family_id,
                parent_slug=None,
                status="populated" if carries_contract else "provisional",
                legacy_paradigm=legacy,
                aliases=target.aliases,
                fields=fields,
                provisional=True,
            )
            family = roots[target.root_id].families[target.family_id]
            new_family = replace(
                family, variants={**family.variants, target.slug: node})
            root = roots[target.root_id]
            roots[target.root_id] = replace(
                root, families={**root.families, new_family.id: new_family})
            variants_by_slug[node.slug] = node
            variants_by_taxonomy_id[node.taxonomy_id] = node
            node_by_legacy[legacy] = node
            diagnostics.append(TaxonomyDiagnostic(
                "warning", "provisional_populates_reserved",
                f"{pack_path} populates the committed reserved placeholder "
                f"{target.taxonomy_id} for this run (run-local overlay; the "
                "committed SSOT is unchanged)", target.taxonomy_id))
            continue

        is_family_parent = isinstance(parent, FamilyNode)
        node = VariantNode(
            slug=slug,
            taxonomy_id=str(taxonomy_id),
            name=pack.get("name", slug),
            root_id=parent.root_id,
            family_id=parent.id if is_family_parent else parent.family_id,
            parent_slug=None if is_family_parent else parent.slug,
            status="populated" if carries_contract else "provisional",
            legacy_paradigm=legacy,
            aliases=(),
            fields=fields,
            provisional=True,
        )

        if is_family_parent:
            new_family = replace(parent, variants={**parent.variants, slug: node})
        else:
            # Top-level variant parent: the pack grafts as its sub-variant so
            # inheritance walks parent-first exactly like a committed child.
            new_parent = replace(
                parent, sub_variants={**parent.sub_variants, slug: node}
            )
            family = roots[parent.root_id].families[parent.family_id]
            new_family = replace(
                family, variants={**family.variants, parent.slug: new_parent}
            )
            variants_by_slug[parent.slug] = new_parent
            variants_by_taxonomy_id[new_parent.taxonomy_id] = new_parent
            if parent.legacy_paradigm:
                node_by_legacy[parent.legacy_paradigm] = new_parent
        root = roots[node.root_id]
        roots[node.root_id] = replace(
            root, families={**root.families, new_family.id: new_family}
        )
        variants_by_slug[slug] = node
        variants_by_taxonomy_id[node.taxonomy_id] = node
        node_by_legacy[legacy] = node
        # Later packs graft onto the updated family, and downstream lookups
        # (node_for_legacy on `extends`) must see the replaced instance.
        if is_family_parent and parent.legacy_paradigm:
            node_by_legacy[parent.legacy_paradigm] = new_family

    return Taxonomy(
        repo_root=committed.repo_root,
        source_path=committed.source_path,
        roots=roots,
        task_domains=committed.task_domains,
        aliases=committed.aliases,
        universal_checks=committed.universal_checks,
        diagnostics=tuple(diagnostics),
        _variants_by_slug=variants_by_slug,
        _variants_by_taxonomy_id=variants_by_taxonomy_id,
        _node_by_legacy=node_by_legacy,
        _leaf_to_domain=committed._leaf_to_domain,
    )


def _build_taxonomy(repo_root: Path, source_path: Path, raw: dict[str, Any]) -> Taxonomy:
    diagnostics: list[TaxonomyDiagnostic] = []
    roots: dict[str, RootNode] = {}
    variants_by_slug: dict[str, VariantNode] = {}
    variants_by_taxonomy_id: dict[str, VariantNode] = {}
    node_by_legacy: dict[str, VariantNode | FamilyNode] = {}

    def _err(code: str, message: str, node: str | None = None) -> None:
        diagnostics.append(TaxonomyDiagnostic("error", code, message, node))

    def _register_variant(v: VariantNode) -> None:
        if v.slug in variants_by_slug:
            _err("variant_duplicate", f"duplicate variant slug {v.slug!r}", v.taxonomy_id)
        else:
            variants_by_slug[v.slug] = v
        if v.taxonomy_id in variants_by_taxonomy_id:
            _err("taxonomy_id_duplicate", f"duplicate taxonomy_id {v.taxonomy_id!r}", v.taxonomy_id)
        else:
            variants_by_taxonomy_id[v.taxonomy_id] = v
        if v.legacy_paradigm:
            if v.legacy_paradigm in node_by_legacy:
                _err("legacy_duplicate", f"two nodes claim legacy_paradigm {v.legacy_paradigm!r}", v.taxonomy_id)
            else:
                node_by_legacy[v.legacy_paradigm] = v

    def _build_variant(slug: str, body: dict[str, Any], root_id: str, family_id: str, parent_slug: str | None) -> VariantNode:
        body = body or {}
        sub_raw = body.get("sub_variants") or {}
        fields = {
            k: v
            for k, v in body.items()
            if k
            in (
                *_INHERITABLE_MAPPING_FIELDS,
                *_INHERITABLE_CHECK_LIST_FIELDS,
                *_INHERITABLE_DIMENSION_LIST_FIELDS,
                _INHERITABLE_DOMAIN_CHECK_FIELD,
            )
        }
        taxonomy_id = body.get("taxonomy_id") or f"{family_id}/{slug}"
        node = VariantNode(
            slug=slug,
            taxonomy_id=taxonomy_id,
            name=body.get("name", slug),
            root_id=root_id,
            family_id=family_id,
            parent_slug=parent_slug,
            status=body.get("status", "reserved"),
            legacy_paradigm=body.get("legacy_paradigm"),
            aliases=tuple(body.get("aliases") or ()),
            fields=fields,
            sub_variants={
                sub_slug: _build_variant(sub_slug, sub_body, root_id, family_id, slug)
                for sub_slug, sub_body in sub_raw.items()
            },
        )
        return node

    def _walk_register(v: VariantNode) -> None:
        _register_variant(v)
        for sub in v.sub_variants.values():
            _walk_register(sub)

    method_roots = raw.get("method_roots") or {}
    for root_id, root_body in method_roots.items():
        root_body = root_body or {}
        if root_id not in METHOD_ROOT_IDS:
            _err("root_unknown", f"method root {root_id!r} is not one of {METHOD_ROOT_IDS}", root_id)
        families: dict[str, FamilyNode] = {}
        for fam_id, fam_body in (root_body.get("families") or {}).items():
            fam_body = fam_body or {}
            fam_fields = {
                k: v
                for k, v in fam_body.items()
                if k
                in (
                    *_INHERITABLE_MAPPING_FIELDS,
                    *_INHERITABLE_CHECK_LIST_FIELDS,
                    *_INHERITABLE_DIMENSION_LIST_FIELDS,
                    _INHERITABLE_DOMAIN_CHECK_FIELD,
                )
            }
            variants: dict[str, VariantNode] = {}
            for v_slug, v_body in (fam_body.get("variants") or {}).items():
                variant = _build_variant(v_slug, v_body, root_id, fam_id, None)
                variants[v_slug] = variant
                _walk_register(variant)
            family = FamilyNode(
                id=fam_id,
                name=fam_body.get("name", fam_id),
                root_id=root_id,
                description=str(fam_body.get("description", "")),
                legacy_paradigm=fam_body.get("legacy_paradigm"),
                status=fam_body.get("status", "reserved"),
                fields=fam_fields,
                variants=variants,
            )
            families[fam_id] = family
            if family.legacy_paradigm:
                if family.legacy_paradigm in node_by_legacy:
                    _err("legacy_duplicate", f"two nodes claim legacy_paradigm {family.legacy_paradigm!r}", fam_id)
                else:
                    node_by_legacy[family.legacy_paradigm] = family
        roots[root_id] = RootNode(
            id=root_id,
            name=root_body.get("name", root_id),
            archetype=str(root_body.get("archetype", "")),
            families=families,
        )

    # Task domains.
    task_domains: dict[str, DomainNode] = {}
    leaf_to_domain: dict[str, str] = {}
    for dom_id, dom_body in (raw.get("task_domains") or {}).items():
        dom_body = dom_body or {}
        leaves = {slug: (leaf or {}).get("name", slug) for slug, leaf in (dom_body.get("leaves") or {}).items()}
        task_domains[dom_id] = DomainNode(id=dom_id, name=dom_body.get("name", dom_id), leaves=leaves)
        for leaf_slug in leaves:
            if leaf_slug in leaf_to_domain and leaf_to_domain[leaf_slug] != dom_id:
                _err("leaf_duplicate", f"leaf {leaf_slug!r} appears under multiple domains", leaf_slug)
            leaf_to_domain[leaf_slug] = dom_id

    # Aliases.
    aliases: dict[str, str] = {}
    for alias, canonical in (raw.get("method_family_aliases") or {}).items():
        if alias in variants_by_slug:
            _err("alias_shadows_variant", f"alias {alias!r} collides with a canonical variant slug")
        if canonical not in variants_by_slug:
            _err("alias_unresolved", f"alias {alias!r} -> unknown variant {canonical!r}")
        aliases[alias] = canonical

    # Per-variant `aliases:` list (register into the alias table).
    for v in list(variants_by_slug.values()):
        for alias in v.aliases:
            if alias in variants_by_slug and alias != v.slug:
                _err("alias_shadows_variant", f"alias {alias!r} collides with a canonical variant slug", v.taxonomy_id)
            aliases.setdefault(alias, v.slug)

    tax = Taxonomy(
        repo_root=repo_root,
        source_path=source_path,
        roots=roots,
        task_domains=task_domains,
        aliases=aliases,
        universal_checks=list(raw.get("universal_checks") or []),
        diagnostics=tuple(diagnostics),
        _variants_by_slug=variants_by_slug,
        _variants_by_taxonomy_id=variants_by_taxonomy_id,
        _node_by_legacy=node_by_legacy,
        _leaf_to_domain=leaf_to_domain,
    )
    return tax


# =============================================================================
# Intra-file inheritance + effective node
# =============================================================================


def _ancestor_field_layers(taxonomy: Taxonomy, variant: VariantNode) -> list[dict[str, Any]]:
    """Return inheritable-field dicts parent-first: root, family, variant chain."""
    family = taxonomy.roots[variant.root_id].families[variant.family_id]
    layers: list[dict[str, Any]] = [family.fields]
    # Walk the variant/sub-variant chain root-most first.
    chain: list[VariantNode] = []
    current: VariantNode | None = variant
    # Build slug->node map within the family (including nested) to walk parents.
    by_slug: dict[str, VariantNode] = {}

    def _index(v: VariantNode) -> None:
        by_slug[v.slug] = v
        for sub in v.sub_variants.values():
            _index(sub)

    for top in family.variants.values():
        _index(top)
    while current is not None:
        chain.append(current)
        current = by_slug.get(current.parent_slug) if current.parent_slug else None
    layers.extend(node.fields for node in reversed(chain))
    return layers


def effective_node(taxonomy: Taxonomy, variant_slug: str, task_domain: str | None = None) -> dict[str, Any]:
    """Merge a variant's inheritable fields parent-first; overlay `task_domain`.

    Shallow-merge for mapping fields (child key replaces parent). Check-list
    fields (`semantic_checks`, `smoke_bugs`) are concatenated then de-duplicated
    by `id` (parent first). When `task_domain` is given, that domain's entries
    from each layer's `domain_checks` are appended to `semantic_checks`.
    """
    variant = taxonomy.variant(variant_slug)
    if variant is None:
        raise TaxonomyError(f"unknown method variant {variant_slug!r}")
    layers = _ancestor_field_layers(taxonomy, variant)

    merged: dict[str, Any] = {}
    for key in _INHERITABLE_MAPPING_FIELDS:
        for layer in layers:
            block = layer.get(key)
            if isinstance(block, dict):
                merged.setdefault(key, {}).update(block)
            elif block is not None:
                merged[key] = block

    for key in _INHERITABLE_CHECK_LIST_FIELDS:
        merged[key] = _concat_dedupe_checks(layer.get(key) for layer in layers)

    for key in _INHERITABLE_DIMENSION_LIST_FIELDS:
        merged[key] = _concat_dedupe_checks(layer.get(key) for layer in layers)

    # Domain overlay.
    if task_domain is not None:
        resolved = taxonomy.task_domain(task_domain)
        if resolved is None:
            raise TaxonomyError(f"unknown task domain {task_domain!r}")
        dom_id, leaf = resolved
        extra: list[dict[str, Any]] = []
        for layer in layers:
            dom_block = layer.get(_INHERITABLE_DOMAIN_CHECK_FIELD) or {}
            for key in (dom_id, leaf):
                if key and isinstance(dom_block.get(key), list):
                    extra.extend(c for c in dom_block[key] if isinstance(c, dict))
        if extra:
            merged["semantic_checks"] = _concat_dedupe_checks([merged.get("semantic_checks"), extra])

    return merged


def load_pack(
    paradigm_id: str,
    repo_root: Path = ROOT,
    task_domain: str | None = None,
) -> dict[str, Any] | None:
    """Return the effective pack for a taxonomy-served legacy paradigm id.

    Re-centering §5.2's "pack loader" is the same object as a taxonomy node per
    `the paradigm to taxonomy migration note (internal, not shipped)`: load the SSOT once,
    merge inherited node fields additively, and expose the result under a pack
    shaped API. Unknown or unpopulated paradigms return ``None``.
    """
    tax = load_taxonomy(repo_root)
    node = serves(paradigm_id, tax)
    if node is None:
        return None
    if isinstance(node, VariantNode):
        fields = effective_node(tax, node.slug, task_domain=task_domain)
    else:
        fields = dict(node.fields)
    return {
        "pack": paradigm_id,
        "taxonomy_id": node.taxonomy_id if isinstance(node, VariantNode) else node.id,
        "legacy_paradigm": node.legacy_paradigm,
        **fields,
    }


def _concat_dedupe_checks(blocks) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for block in blocks:
        if not isinstance(block, list):
            continue
        for check in block:
            if not isinstance(check, dict):
                continue
            cid = check.get("id")
            if isinstance(cid, str) and cid:
                if cid in seen:
                    continue
                seen.add(cid)
            out.append(check)
    return out


# =============================================================================
# Classification-tuple validation
# =============================================================================


def assert_classification_tuple(
    taxonomy: Taxonomy,
    method_variant: str,
    task_domain: str | None = None,
) -> tuple[VariantNode, tuple[str, str | None] | None]:
    """Validate a `(method_variant, task_domain)` pair; raise on unknown slugs.

    Returns the resolved `(VariantNode, (domain_id, leaf))`. `task_domain` may be
    None (method-only classification) but, if given, must resolve.
    """
    variant = taxonomy.variant(method_variant)
    if variant is None:
        raise TaxonomyError(
            f"unknown method variant {method_variant!r} (not a slug, taxonomy_id, or alias)"
        )
    resolved_domain: tuple[str, str | None] | None = None
    if task_domain is not None:
        resolved_domain = taxonomy.task_domain(task_domain)
        if resolved_domain is None:
            raise TaxonomyError(f"unknown task domain {task_domain!r}")
    return variant, resolved_domain


# =============================================================================
# Taxonomy-serving helpers
# =============================================================================


def serves(paradigm_id: str, taxonomy: Taxonomy | None = None) -> VariantNode | FamilyNode | None:
    """Return the populated taxonomy node that serves `paradigm_id`, else None.

    Resolution order: the legacy_paradigm registry first, then the variant
    lookup (slug, taxonomy_id, or registered alias). A runtime
    classification may use any registered taxonomy alias, not only a
    legacy_paradigm id — before the fallback, an alias-id spec
    (pool_based_active_learning) reached no build plan at all while
    load_demo_skill carried its own private copy of the same fallback
    (the B-02 spin-off; two concrete cases, so the fallback lives here
    for every serves() consumer)."""
    tax = taxonomy or load_taxonomy()
    node = tax.node_for_legacy(paradigm_id)
    if node is None:
        node = tax.variant(paradigm_id)
    if node is None:
        return None
    return node if node.is_populated else None


def _paradigm_of(guide) -> str | None:
    if isinstance(guide, str):
        return guide
    return getattr(guide, "paradigm", None)


def registered_paradigm_ids(repo_root: Path = ROOT) -> list[str]:
    """Legacy paradigm ids registered by the taxonomy SSOT.

    Phase 2.6 orchestrator seam: callers that only need the registered literal
    ids read the SSOT directly instead of building a guide-file catalog.
    """
    tax = load_taxonomy(repo_root)
    return sorted(tax._node_by_legacy)


def merged_stage_review_focus(catalog, guide, stage_id: str) -> dict[str, Any]:
    """Stage-review focus rendered from the taxonomy node."""
    node = serves(_paradigm_of(guide) or "")
    if node is None:
        return {}
    return _node_stage_review_focus(node, stage_id)


def merged_all_stage_review_focus(
    catalog, guide, stage_ids: tuple[str, ...] = REVIEW_STAGE_IDS
) -> dict[str, dict[str, Any]]:
    """All non-empty per-stage review blocks from the taxonomy node.

    Mirrors the historical all-stage shape but renders from taxonomy nodes only.
    """
    return {
        stage_id: block
        for stage_id in stage_ids
        if (block := merged_stage_review_focus(catalog, guide, stage_id))
    }


def merged_common_smoke_bugs(catalog, guide) -> dict[str, Any]:
    """Smoke-bug catalog rendered from the taxonomy node."""
    node = serves(_paradigm_of(guide) or "")
    if node is None:
        return {}
    return _node_smoke_bugs(node)


def effective_blocks(catalog, guide) -> dict[str, Any]:
    """Effective blocks rendered from the taxonomy node."""
    pid = _paradigm_of(guide) or ""
    node = serves(pid)
    if node is None:
        return {}
    blocks: dict[str, Any] = {}
    for key in (
        *_INHERITABLE_MAPPING_FIELDS,
        *_INHERITABLE_DIMENSION_LIST_FIELDS,
    ):
        value = _node_effective_field(node, key)
        if value:
            blocks[key] = value
    blocks["stage_review_focus"] = merged_all_stage_review_focus(catalog, guide)
    blocks["common_smoke_bugs"] = merged_common_smoke_bugs(catalog, guide)
    return blocks


def load_pluggable_component_contract(
    paradigm_id: str | None = None, taxonomy: Taxonomy | None = None
) -> dict[str, Any]:
    """Pluggable-component contract block from the taxonomy node.

    Pass `taxonomy` to resolve against an overlay view (the gap-path
    provisional packs); default resolves the committed SSOT.
    """
    pid = paradigm_id
    if pid:
        node = serves(pid, taxonomy)
        if node is not None:
            block = _node_effective_field(node, "pluggable_component", taxonomy)
            if isinstance(block, dict) and block:
                return block
    return {}


def load_notebook_layout(
    paradigm_id: str | None = None, taxonomy: Taxonomy | None = None
) -> dict[str, Any]:
    """Notebook-section layout block rendered from a taxonomy node.

    Pass `taxonomy` to resolve against an overlay view (the gap-path
    provisional packs); default resolves the committed SSOT."""
    pid = paradigm_id
    if pid:
        node = serves(pid, taxonomy)
        if node is not None:
            block = _node_effective_field(node, "notebook_layout", taxonomy)
            if isinstance(block, dict) and block:
                return block
    return {}


def load_demo_success(
    paradigm_id: str | None = None, taxonomy: Taxonomy | None = None
) -> dict[str, Any]:
    """Demo success/failure marker block rendered from a taxonomy node.

    The declaration surface for the post-smoke demo-success verdict
    (scripts/demo_verdict.py): `failure_markers` / `success_markers` are
    line-regex markers matched against the headline demo section's executed
    outputs, `checks` names deterministic checks from the shared vocabulary
    (today: `beats_chance`). A family without the block yields `{}` — the
    verdict pass then records `undetermined` (honest, and a kit-coverage
    finding on a committed family). Pass `taxonomy` to resolve against an
    overlay view (the gap-path provisional packs use the same surface)."""
    pid = paradigm_id
    if pid:
        node = serves(pid, taxonomy)
        if node is not None:
            block = _node_effective_field(node, "demo_success", taxonomy)
            if isinstance(block, dict) and block:
                return block
    return {}


def load_demo_skill(
    paradigm_id: str | None = None, taxonomy: Taxonomy | None = None
) -> dict[str, Any]:
    """Family-owned structured demo-skill comparison contract.

    This is deliberately independent of :func:`load_demo_success`: the latter
    describes coarse executed-output markers, while this block declares the
    metric, eligibility floor, comparators, and decision policy used to
    recompute demonstrated skill from structured executed evidence.  A family
    without a meaningful comparison contract yields ``{}``; no forecasting
    baseline is inferred for adjacent families.  Pass ``taxonomy`` for a
    run-local provisional-pack overlay.
    """
    pid = paradigm_id
    if pid:
        tax = taxonomy or load_taxonomy()
        # serves() owns the alias fallback (slug/taxonomy_id/alias after the
        # legacy registry) and still refuses reserved/unpopulated nodes.
        node = serves(pid, tax)
        if node is not None:
            block = _node_effective_field(node, "demo_skill", tax)
            if isinstance(block, dict) and block:
                return block
    return {}


def resolve_templates_dir(
    paradigm_id: str | None = None,
    repo_root: Path | None = None,
    taxonomy: Taxonomy | None = None,
) -> Path | None:
    """Most-specific ``templates/`` dir from node ``scaffold_hints.templates_dir``.

    Pass `taxonomy` to resolve against an overlay view (the gap-path
    provisional packs — a provisional node inherits its parent's
    templates_dir unless the pack overrides it).

    Provisional nodes with NO declared or inherited templates_dir fall back
    to the committed family-neutral skeleton (item 4, the fedavg 2026-07-07
    stage-2a halt): a brand-new family has no family-specific committed
    templates by definition. Committed nodes never reach the fallback —
    silence there is a repo bug and stays a halt."""
    pid = paradigm_id
    if pid:
        node = serves(pid, taxonomy)
        if node is not None:
            hints = _node_effective_field(node, "scaffold_hints", taxonomy) or {}
            rel = hints.get("templates_dir")
            root = Path(repo_root) if repo_root is not None else ROOT
            if rel:
                cand = root / rel
                return cand if cand.is_dir() else None
            if getattr(node, "provisional", False):
                cand = root / PROVISIONAL_TEMPLATES_DIR
                return cand if cand.is_dir() else None
    return None


# How a node's own templates get the demo its data (R2C-074).
#
# Until 2026-08-06 nothing declared this and one proxy stood in for it: demo
# dataset acquisition fired only on GAP-PATH runs, on the reasoning that a
# committed family "owns its data story". That held while every committed
# family's template downloaded its own data, and it broke the moment a family
# was promoted OUT of the gap path with the gap path's file-only loader still
# in its templates: the forecasting family was committed on 2026-08-05 and its
# next run silently fetched nothing, because promotion moved it across the gate
# without giving it what the gate assumed.
#
# So the assumption is a declaration now. Absent means today's behavior exactly,
# which keeps every family that has not been examined unchanged.
DEMO_DATA_TEMPLATE_DOWNLOADS = "template_downloads"
DEMO_DATA_NEEDS_ACQUISITION = "needs_acquisition"
DEMO_DATA_SOURCES = (DEMO_DATA_TEMPLATE_DOWNLOADS, DEMO_DATA_NEEDS_ACQUISITION)


def demo_data_source(
    paradigm_id: str | None = None,
    taxonomy: Taxonomy | None = None,
) -> str | None:
    """Node `scaffold_hints.demo_data_source`, or None when undeclared.

    `template_downloads` — the node's own data loader fetches what the demo
    needs, so bundling the paper's dataset would duplicate it.
    `needs_acquisition` — the loader reads local files only, so a run of this
    family first tries to bundle the paper's own cited public dataset. If that
    attempt cannot supply data and the resolved build plan declares a supported
    family-owned offline fallback, Stage 2a materializes that typed fallback;
    otherwise the run refuses rather than inventing data.
    None — undeclared, and every caller keeps its pre-declaration behavior.
    """
    if not paradigm_id:
        return None
    node = serves(paradigm_id, taxonomy)
    if node is None:
        return None
    hints = _node_effective_field(node, "scaffold_hints", taxonomy) or {}
    value = hints.get("demo_data_source")
    return value if value in DEMO_DATA_SOURCES else None


def demo_scale_threshold(
    repo_root: Path,
    paradigm_id: str | None = None,
    taxonomy: Taxonomy | None = None,
) -> float | None:
    """Return node `smoke_economics.max_budget_to_pool_ratio`, or None."""
    pid = paradigm_id
    if pid:
        node = serves(pid, taxonomy)
        if node is not None:
            econ = _node_effective_field(node, "smoke_economics", taxonomy) or {}
            ratio = econ.get("max_budget_to_pool_ratio")
            if ratio is not None:
                try:
                    return float(ratio)
                except (TypeError, ValueError):
                    return None
    return None


def load_smoke_economics(
    paradigm_id: str | None, taxonomy: Taxonomy | None = None
) -> dict[str, Any] | None:
    """A served node's merged smoke-economics block, or ``None``.

    This is the Phase 5.7 deterministic-consumer API for derive_params. Reserved
    paradigms return ``None`` so their existing Python literal fallback behavior
    is unchanged until they migrate. Pass `taxonomy` to resolve against an
    overlay view (the gap-path provisional packs).
    """
    if not paradigm_id:
        return None
    node = serves(paradigm_id, taxonomy)
    if node is None:
        return None
    block = _node_effective_field(node, "smoke_economics", taxonomy)
    return block if isinstance(block, dict) and block else None


def load_paradigm_extra(
    paradigm_id: str | None, name: str, taxonomy: Taxonomy | None = None
) -> tuple[object, str] | None:
    """A served node's smoke default for a pluggable-signature extra (Phase 5.7).

    Returns ``(value, reasoning_template)`` from the node's ``paradigm_extras[name]``
    when ``paradigm_id`` is served by a ``populated`` node that declares it, else
    ``None``. The fallback source for these defaults is hardcoded Python
    (``derive_params.KNOWN_EXTRAS_BY_PARADIGM``), not a guide file: the node wins
    for served paradigms, and the Python table remains the fallback for the
    provisional/future ones. The
    ``reasoning_template`` keeps its ``{value}``/``{paper_value}`` ``.format()`` slots
    verbatim for the consumer to fill.
    """
    if not paradigm_id:
        return None
    node = serves(paradigm_id, taxonomy)
    if node is None:
        return None
    extras = _node_effective_field(node, "paradigm_extras", taxonomy)
    if not isinstance(extras, dict):
        return None
    entry = extras.get(name)
    if isinstance(entry, dict) and "value" in entry and entry.get("reasoning_template"):
        return entry["value"], entry["reasoning_template"]
    return None


def load_model_default(
    paradigm_id: str | None, name: str, taxonomy: Taxonomy | None = None
) -> tuple[object, str] | None:
    """A served node's *system_inferred* fallback for a model-architecture param (Phase 5.7b).

    Returns ``(value, reasoning_template)`` from the node's ``model_defaults[name]``
    when ``paradigm_id`` is served by a ``populated`` node that declares it, else
    ``None``. These are the evidence-driven model params (``hidden_dim``,
    ``dropout_rate``) in ``derive_params._add_model_params``: the node owns ONLY the
    ``source: system_inferred`` fallback arm — the ``source: paper`` arm is computed
    per-run from the spec's own evidence and is never node-stored. Like
    ``load_paradigm_extra``, the fallback source is hardcoded Python (the literal
    fallbacks in ``_add_model_params``), so lookup is node first and then Python
    literal for provisional/future ids. The ``reasoning_template`` keeps its ``{value}``
    ``.format()`` slot verbatim for the consumer to fill.
    """
    if not paradigm_id:
        return None
    node = serves(paradigm_id, taxonomy)
    if node is None:
        return None
    defaults = _node_effective_field(node, "model_defaults", taxonomy)
    if not isinstance(defaults, dict):
        return None
    entry = defaults.get(name)
    if isinstance(entry, dict) and "value" in entry and entry.get("reasoning_template"):
        return entry["value"], entry["reasoning_template"]
    return None


def load_params_derivation(
    paradigm_id: str | None, taxonomy: Taxonomy | None = None
) -> dict[str, dict[str, Any]]:
    """The served node's ``params_derivation`` declarations (plan item 9).

    Returns the effective param-name -> entry mapping (inheritance
    resolved, malformed entries dropped), or ``{}`` when the paradigm is
    unserved or declares nothing. Unlike ``paradigm_extras`` /
    ``model_defaults`` there is no hardcoded Python fallback: the whole
    point of the surface is that the PARADIGM declares what the
    deterministic deriver could never guess (required configuration
    paths, derived statistics, inapplicable convention params), so an
    empty declaration means the deriver's generic behavior is already
    correct for the paradigm.
    """
    if not paradigm_id:
        return {}
    node = serves(paradigm_id, taxonomy)
    if node is None:
        return {}
    block = _node_effective_field(node, "params_derivation", taxonomy)
    if not isinstance(block, dict):
        return {}
    return {
        str(name): entry
        for name, entry in block.items()
        if isinstance(entry, dict) and entry.get("kind")
    }


def load_scenario_assumption_dimensions(
    paradigm_id: str | None, taxonomy: Taxonomy | None = None
) -> list[dict[str, Any]]:
    """Return the served node's scenario-assumption dimension declarations.

    Each entry declares the dimension id and normalization guidance the
    analyzer may use in ``method_spec.scenario_assumptions`` (capture, slice
    A) plus the `detector` executor ref and authored `check`/`silent_failure`
    context the probe battery binds at verification time (enforcement, slice
    B). The taxonomy lint makes a declared dimension without a detector
    authoring-invalid.
    """
    if not paradigm_id:
        return []
    node = serves(paradigm_id, taxonomy)
    if node is None:
        return []
    block = _node_effective_field(
        node, "scenario_assumption_dimensions", taxonomy
    )
    if not isinstance(block, list):
        return []
    return [
        entry
        for entry in block
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    ]


# -----------------------------------------------------------------------------
# Node renderers (taxonomy-served branch).
# -----------------------------------------------------------------------------


def _node_effective_field(
    node: VariantNode | FamilyNode, key: str, taxonomy: Taxonomy | None = None
) -> Any:
    """The effective value of `key` for a served node, resolving inheritance.

    Asymmetric by node kind (and intentionally so): a `VariantNode` resolves
    through `effective_node` (root←family←variant←sub_variant merge), while a
    `FamilyNode` reads `node.fields` directly — a family has no parent layers to
    merge, so its declared fields *are* its effective fields. The bucket loaders
    (`load_notebook_layout`, `load_pluggable_component_contract`, …) share this
    helper, so both kinds resolve consistently across them. A provisional
    overlay node only resolves against the overlay `taxonomy` it came from.
    """
    if isinstance(node, VariantNode):
        return effective_node(taxonomy or load_taxonomy(), node.slug).get(key)
    # Family-level node: family fields are already effective (no variant layer).
    return node.fields.get(key)


def _node_stage_review_focus(node: VariantNode | FamilyNode, stage_id: str) -> dict[str, Any]:
    node_checks = [
        c
        for c in (_node_effective_field(node, "semantic_checks") or [])
        if isinstance(c, dict) and c.get("stage") == stage_id
    ]
    by_id = {c.get("id"): c for c in node_checks}

    # Phase 2.1: compose the canonical top-level `universal_checks` for this stage
    # with the node's own. Universal checks come first (in declared order); a node
    # check with the same id OVERRIDES the universal (node-wins), so a node keeps its
    # specialized/probe-wired version while inheriting the generic ones it no longer
    # restates. Node-specific checks (ids not in the universal set) follow.
    universals = [
        u
        for u in load_taxonomy().universal_checks
        if isinstance(u, dict) and u.get("stage") == stage_id
    ]
    ordered: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for u in universals:
        uid = u.get("id")
        ordered.append(by_id.get(uid, u))  # node override wins
        seen.add(uid)
    for c in node_checks:
        if c.get("id") not in seen:
            ordered.append(c)

    # Map node check shape -> legacy `check_id` shape consumers expect.
    checks: list[dict[str, Any]] = []
    for check in ordered:
        legacy = dict(check)
        if "id" in legacy and "check_id" not in legacy:
            legacy["check_id"] = legacy["id"]
        checks.append(legacy)
    if not checks:
        return {}
    return {"description": None, "inputs": [], "semantic_checks": checks}


def _node_smoke_bugs(node: VariantNode | FamilyNode) -> dict[str, Any]:
    bugs_raw = _node_effective_field(node, "smoke_bugs") or []
    out: dict[str, Any] = {}
    for bug in bugs_raw:
        if isinstance(bug, dict) and bug.get("id"):
            out[bug["id"]] = bug
    return out


# =============================================================================
# Context bucket views (the configurable-system seam).
# -----------------------------------------------------------------------------
# Each consumer asks the node for *its* bucket of context instead of reading raw
# blocks. Flat node fields are assembled into four typed
# views; `semantic_checks` and `smoke_economics` intentionally appear in two
# views by design (expertise+testing, parameters+testing) — see
# the configurable node schema design note (internal, not shipped) §1.
# =============================================================================


def _node_view(
    node: VariantNode | FamilyNode, keys: tuple[str, ...],
    extra: dict[str, Any] | None = None, taxonomy: Taxonomy | None = None,
) -> dict[str, Any]:
    """Assemble a bucket view: effective node fields for `keys`, present-only."""
    view: dict[str, Any] = {}
    for key in keys:
        val = _node_effective_field(node, key, taxonomy)
        if val:
            view[key] = val
    if extra:
        view.update({k: v for k, v in extra.items() if v})
    return view


def node_expertise(
    node: VariantNode | FamilyNode, taxonomy: Taxonomy | None = None
) -> dict[str, Any]:
    """Expertise context — classification + review knowledge (design §2.1).

    Pass `taxonomy` when `node` came from an overlay view (a run-local
    provisional pack): a provisional node's inheritance only resolves
    against the overlay it was grafted into; the committed view raises."""
    hints = _node_effective_field(node, "scaffold_hints", taxonomy) or {}
    return _node_view(
        node,
        (
            "fingerprint", "priors", "semantic_checks",
            "scenario_assumption_dimensions",
        ),
        {"interface_hint": hints.get("interface_hint"), "content_file": hints.get("content_file")},
        taxonomy,
    )


def node_implementation(
    node: VariantNode | FamilyNode, taxonomy: Taxonomy | None = None
) -> dict[str, Any]:
    """Implementation context — the build/codegen contract (design §2.2).
    `taxonomy` as in `node_expertise`."""
    hints = _node_effective_field(node, "scaffold_hints", taxonomy) or {}
    return _node_view(
        node,
        ("pluggable_component", "notebook_layout", "demo_skill", "family_components",
         "build_plan", "package_manifest"),
        {"templates_dir": hints.get("templates_dir")}, taxonomy,
    )


def node_parameters(
    node: VariantNode | FamilyNode, taxonomy: Taxonomy | None = None
) -> dict[str, Any]:
    """Parameters context — derive_params floors + defaults (design §2.3).
    `taxonomy` as in `node_expertise`."""
    return _node_view(
        node,
        ("smoke_economics", "paradigm_extras", "model_defaults",
         "params_derivation", "provenance_params", "tier_heuristic"),
        None, taxonomy,
    )


def node_testing(node: VariantNode | FamilyNode) -> dict[str, Any]:
    """Testing context — probes + smoke failure modes (design §2.4)."""
    return _node_view(
        node,
        (
            "semantic_checks", "scenario_assumption_dimensions", "smoke_bugs",
            "smoke_economics", "demo_success", "demo_skill",
        ),
    )


_BUCKET_VIEWS = {
    "expertise": node_expertise,
    "implementation": node_implementation,
    "parameters": node_parameters,
    "testing": node_testing,
}


def bucket_coverage(node: VariantNode | FamilyNode) -> dict[str, list[str]]:
    """Which context each bucket currently provides for `node`.

    Returns `{bucket: [field names present]}` — the mechanical, authorable
    "is this node ready to serve X?" check (design §4 mechanism 4). A bucket with
    an empty list provides no context yet (consumers fall back). Descriptive, not
    a pass/fail: completeness-to-spec firms up as `notebook_layout` (1.6) and
    `paradigm_extras` (5.7) migrate.
    """
    return {bucket: sorted(view(node).keys()) for bucket, view in _BUCKET_VIEWS.items()}


def _repo_root_for(path: Path) -> Path:
    """Nearest ancestor containing a `paradigms/` dir, else the module repo root."""
    resolved = path.resolve()
    for ancestor in resolved.parents:
        if (ancestor / "paradigms").is_dir():
            return ancestor
    return ROOT
