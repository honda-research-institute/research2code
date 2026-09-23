# Taxonomy node schema

> The per-node contract for [`docs/ssot/taxonomies.yaml`](taxonomies.yaml) — the
> single SSOT that replaces the 12 `paradigms/**/FIELD_GUIDE.md` files. Authored
> in Phase 0 of the
> [migration implementation plan](../recentering/taxonomy-migration-implementation-plan.md);
> field meanings trace to the
> [pack schema](../recentering/paradigm-pack-schema.md) and the
> [taxonomy philosophy](../recentering/taxonomy-philosophy.md).
>
> This document is normative for `scripts/taxonomy.py` (the loader) and
> `scripts/validate_taxonomy.py` (the lint). When the two disagree, the
> validator wins and this doc is the bug.

## 1. The two axes

A classification is a **`(method_variant, task_domain)` tuple**, never a single
slug (philosophy P3). The SSOT therefore carries two independent indices:

- **Method axis** — `Root → Family → Variant`. Three levels, coarse-to-fine
  (P1/P2). The *root* is the archetype (what kind of contribution this is); the
  *family* is the mechanism class; the *variant* is the specific method.
- **Task-domain axis** — `Domain → leaf`. What the method is *applied to*. The
  same method (a distillation loss) lands different `domain_checks` depending on
  whether the domain is 2D classification or camera-lidar 3D detection — this is
  the two-axis proof case the migration validates.

## 2. Top-level SSOT shape

```yaml
schema_version: "2.0"

method_roots:          # mapping: ROOT_ID -> root node
  TE: { ... }          # one of TE CLC SC AD PP OM GEN DA INF

task_domains:          # mapping: DOMAIN_ID -> domain node
  CV: { ... }

method_family_aliases: # mapping: legacy/alias slug -> canonical variant slug
  test_time_adaptation: entropy_minimization_tta

universal_checks: []   # pipeline-wide checks, referenced (not duplicated) by
                       # nodes; landed in Phase 2.1, reserved-empty in Phase 0.
```

Mappings (not lists) are keyed by id so the loader can resolve in O(1) and the
lint can detect duplicates structurally. **Slugs are human-readable** — v3's
opaque numeric variant codes (`TE-TS-02`) and short-code directories are not
ported. Root ids (`TE`…`INF`) and family ids (`TE-TS`, `CLC-MP`, `OM-OPT`) keep
their codes because they name archetypes/mechanism-classes, not leaves.

## 3. Method-axis nodes

### 3.1 Root node

```yaml
TE:
  name: Train and Evaluate          # required, human title
  archetype: >                      # required, P1 — what KIND of contribution
    A method that changes how a model is trained or what objective/data it sees.
  families: { ... }                 # required mapping FAMILY_ID -> family node
```

### 3.2 Family node

```yaml
TE-TS:
  name: Training Strategy
  description: >                     # required
    Methods that change the training loop or data regime rather than the loss
    or architecture.
  # Family-level fields below are INHERITED by every variant unless the variant
  # overrides them (intra-file inheritance — replaces cross-file `extends:`).
  fingerprint: { ... }               # optional at family level (see §3.4)
  priors: { ... }
  semantic_checks: [ ... ]
  variants: { ... }                  # required mapping VARIANT_SLUG -> variant node
```

### 3.3 Variant node

The leaf. Carries only what *specializes* its family (intra-file inheritance:
the loader merges `root ← family ← variant`, and the variant overrides only the
keys it sets).

```yaml
active_learning:
  name: Active Learning
  status: reserved                   # reserved | populated  (see §6)
  legacy_paradigm: active_learning   # the paradigms/ id this node supersedes
  taxonomy_id: TE-TS/active_learning # stable id; root/family/variant path
  aliases: [pool_based_active_learning]

  fingerprint: { ... }               # §3.4 — detection + negative space
  priors: { ... }                    # §3.5 — expertise / scaling rules
  semantic_checks: [ ... ]           # §3.6 — THE richest field; behavioral probes
  domain_checks: { ... }             # §3.7 — per-task-domain check overlays
  provenance_params: [ ... ]         # params whose value must trace to the paper
  tier_heuristic: >                  # prose: how to pick smoke vs. full tier
  smoke_bugs: [ ... ]                # §3.8 — failure modes the diagnostician hunts
  smoke_economics: { ... }           # §3.9 — budget floors / clamps
  scaffold_hints: { ... }            # §3.10 — codegen pointers + long-form content
  pluggable_component: { ... }       # §3.11 — the build/codegen contract (Phase 1.5)
  family_components: { ... }         # §3.16 — legal arch-contract extension blocks
  params_derivation: { ... }         # §3.17 — deriver-consumable 2.x param requirements
  scenario_assumption_dimensions: [ ... ] # §3.18, family-scoped capture vocabulary
  demo_skill: { ... }                # §3.19 — structured task-skill comparison policy
  build_plan: { ... }                # §3.20 — routing + family runtime contracts
```

### 3.4 `fingerprint` — detection **and negative space** (philosophy P6/§6)

```yaml
fingerprint:
  what_it_is: >                      # required when populated; one tight paragraph.
    Iteratively selects the most informative unlabeled points to label next.
  positive_signals:                  # cues that route a paper HERE
    - "an acquisition function scoring unlabeled points"
    - "a label budget and successive query rounds"
  route_elsewhere:                   # NEGATIVE SPACE — cues that route AWAY.
    - signal: "selection is fixed/random, no model-in-the-loop"
      to: TE-DS/data_sampling
  worked_examples:                   # papers/methods that anchor the node
    - "BADGE (Ash et al. 2020)"
```

`route_elsewhere[].to` must resolve to a real variant `taxonomy_id` or variant
slug; the lint rejects orphans (Open Q: no dangling negative-space targets).

### 3.5 `priors`

Expertise that biases generation but never gates: hyperparameter scaling rules,
demo-scale numbers, known-good defaults. Free-form mapping; long prose belongs in
a referenced markdown file (`scaffold_hints.content_file`), not inlined (Open Q D).

### 3.6 `semantic_checks` — the essence (philosophy P6)

A list of declarative behavioral checks executed by the **one generic probe
harness** (`scripts/probes/`), not per-family executable code. Each entry:

```yaml
- id: AL-1                           # required, stable, unique within the node
  stage: stage_3a_notebook           # required; one of the REVIEW_STAGE_IDS
  check: >                           # required; what the probe asserts
    Final accuracy exceeds chance (1 / num_classes).
  silent_failure: >                  # required; what passing-but-wrong looks like
    A miswired loop reports high "accuracy" by evaluating on training labels.
  severity: error                    # error | warning
  status: seed                       # seed | observed  (see §6) — required
  probe: al_loop.above_chance        # optional; executor ref in scripts/probes/
```

**`status` is mandatory on every check.** `seed` checks disclose only — they
never produce a `verified` label and never block delivery until a real run
promotes them to `observed` (Phase 5.6). The lint fails if `status` is missing.

### 3.7 `domain_checks` — the second axis

Per-task-domain overlays merged on top of `semantic_checks` when the
classification tuple names that domain:

```yaml
domain_checks:
  CV:                                # keyed by task_domain id (must resolve)
    - id: AL-CV-1
      stage: stage_3a_notebook
      check: "Selected indices are valid image ids within the pool."
      silent_failure: "..."
      severity: warning
      status: seed
```

### 3.8 `smoke_bugs`

Failure modes the smoke-diagnostician hunts at build time. Same `status`
discipline (`seed|observed`).

```yaml
smoke_bugs:
  - id: AL-SB-1
    symptom: "accuracy flat across rounds"
    likely_cause: "model re-initialized but optimizer state leaked"
    status: seed
```

### 3.9 `smoke_economics`

Budget floors and clamp rules consumed by `derive_params`. Ratio floors apply to
derived scale knobs such as `pool_size`; per-param floors apply only when the
emitted smoke value would otherwise fall below a deterministic lower bound.
Where a floor has no deterministic target, consumers may surface it as advisory
context instead.

```yaml
smoke_economics:
  max_budget_to_pool_ratio: 0.5
  floors:
    mc_samples:
      absolute_min: 20
      reason: "below this BALD's posterior estimate is not meaningfully tested"
  protected_params: [num_classes]    # never clamped for budget
```

### 3.10 `scaffold_hints`

Codegen pointers (interface hints, file layout cues), the pointer to long-form
prose, and the pointer to the on-disk paradigm-fixed templates the package
scaffolder renders:

```yaml
scaffold_hints:
  interface_hint: "select(model, pool, k) -> list[int]"
  content_file: docs/ssot/content/active_learning.md   # long-form priors prose
  templates_dir: paradigms/supervised_ml/active_learning/templates  # repo-relative
```

`templates_dir` is a repo-relative path to the `templates/` tree the
`package_scaffolder` mirrors into the run directory (replacing the legacy
"walk the `extends:` chain for the nearest `templates/` dir" discovery). The
lint requires it to resolve to an existing directory. It is inherited like any
other `scaffold_hints` key, so sub-variants share the parent's templates unless
they declare their own.

### 3.11 `pluggable_component` — the build/codegen contract (Phase 1.5)

The shape the paradigm's *pluggable function* must have so the harness/notebook
can call it uniformly. This is the build-contract surface migrated in Phase 1.5
(the review surface — §3.6 — was Phase 1). A **top-level variant field**, not
nested under `scaffold_hints`; inherited and overridable like any mapping field
(a sub-variant that threads an extra positional redeclares the whole block, and
the override replaces it wholesale — see §5).

```yaml
pluggable_component:
  name: select_batch                 # the function symbol the validator imports
  return_type: "List[int]"
  signature_template: "select_batch(model, x_unlabeled, batch_size, seed, *paradigm_extras_by_name) -> List[int]"
  contract:                          # required when pluggable_component is present
    fixed_positional_args: [model, x_unlabeled, batch_size]   # always positional, in order
    seed_param: seed                                          # harness passes the per-round seed
    paradigm_extras_forwarding: by_name_from_config           # named kwargs with defaults
    forbidden_param_names: [config]   # a `config: dict` arg cannot be filled by the variadic-kwargs contract
  rationale: >                        # prose: why this signature/contract holds
    ...
```

The lint enforces, on any node that declares a `pluggable_component`,
`signature_template` + `return_type` + the four `contract` sub-keys (with
`fixed_positional_args` / `forbidden_param_names` as lists). The adapter
(`scripts/taxonomy.py:load_pluggable_component_contract`) serves this block from
the node when the paradigm is `populated`; unpopulated ids return no contract
and must route through the provisional-pack gap path.
The `arch_contract_schema` / `package_manifest` blocks are **not** migrated here
— they are superseded by the recentering-plan §5.3 spec-derived build plan.

## 4. Task-domain nodes

```yaml
task_domains:
  CV:
    name: Computer Vision
    leaves:                          # mapping LEAF_SLUG -> leaf node
      image_classification:
        name: Image Classification
        domain_checks: [ ... ]       # optional domain-wide checks (any method)
      detection_3d_camera_lidar_fusion:
        name: 3D Detection (camera-lidar fusion)
```

A classification tuple's `task_domain` may name either a domain id (`CV`) or a
leaf slug (`image_classification`); the loader resolves both.

## 5. Intra-file inheritance (replaces cross-file `extends:`)

The loader computes a variant's **effective node** by merging, parent-first:

1. the root node's inheritable fields,
2. the family node's inheritable fields,
3. the variant node's own fields (overrides win).

Merge semantics preserve the migration-era consumer contract:
**shallow** at the top level — a child mapping replaces a parent key wholesale;
lists of checks and scenario-assumption dimensions are concatenated then
de-duplicated by `id` (parent first).
This is what lets a `bayesian` variant add the `mc_samples` check without
restating the family's whole check list.

**Sub-variants.** Where a legacy paradigm nested deeper than three levels
(`active_learning/batch_acquisition`, `knowledge_distillation/detection/cross_modal`),
a variant may carry a `sub_variants:` mapping of the same shape as a variant.
Inheritance simply extends the chain — `root ← family ← variant ← sub_variant ←
…` — and each sub-variant overrides only what it specializes. This replaces the
legacy multi-hop `extends:` chain without adding a fixed depth limit.

## 6. `status`: `reserved` → `populated`, `seed` → `observed`

Two orthogonal status fields, both load-bearing for the strangler migration:

- **Node `status`** (`reserved | populated`): a *reserved* node is scaffold only
  — it never serves runtime consumers. Since Phase 4 retired the legacy
  `FIELD_GUIDE.md` fallback, callers get taxonomy data only for populated nodes;
  unpopulated ids resolve to empty/`None` and must use the provisional-pack gap
  path when they need executable context.
- **Check `status`** (`seed | observed`): a *seed* check is advisory — disclosed
  to agents, never gating. A real run that reproduces the check promotes it to
  *observed* (Phase 5.6), at which point it can gate. The lint requires the field
  on every check and rejects `verified`-implying use of seeds.

## 7. What the loader guarantees / the lint enforces

- Every `method_root` id is one of `TE CLC SC AD PP OM GEN DA INF`.
- Every variant has a unique `taxonomy_id` and unique slug across the SSOT.
- Every `aliases[]` and `method_family_aliases` entry resolves to a real variant
  and is unique (no alias collides with a canonical slug).
- Every `route_elsewhere[].to` resolves to a real variant.
- Every `domain_checks` key resolves to a real task-domain id or leaf.
- Every `semantic_checks[]` / `smoke_bugs[]` entry sets `status`.
- Every `populated` node carries a `legacy_paradigm` (or is explicitly marked
  net-new) and the fields its consumers read.
- No two `populated` nodes claim the same `legacy_paradigm`.
- Any node declaring a `pluggable_component` (§3.11) carries `signature_template`,
  `return_type`, and the four `contract` sub-keys.
- Any `scaffold_hints.templates_dir` (§3.10) resolves to an existing directory.
- Any node declaring a `notebook_layout` (§3.12) carries a non-empty `sections` list,
  each entry with an `id`.
- Any node declaring `paradigm_extras` (§3.13) maps each extra name to a
  `{value, reasoning_template}` entry.
- Any node declaring `model_defaults` (§3.14) maps each param name to a
  `{value, reasoning_template}` entry.
- Any node declaring `demo_success` (§3.15) carries at least one valid marker
  (`pattern` compiles as a regex, `gloss` present) or a `checks[]` entry whose
  `kind` is in the shared vocabulary (`taxonomy.DEMO_CHECK_KINDS`).
- Any node declaring `scenario_assumption_dimensions` (§3.18) carries a
  non-empty list of unique ids, each with `display_name`, `description`,
  `normalized_value_guidance`, a family-scoped `detector` executor ref, and
  authored `check` and `silent_failure` context.
- Any node declaring `demo_skill` (§3.19) carries the exact schema-versioned
  metric, eligibility, decision-policy, and comparator grammar. Comparator ids
  are unique, thresholds are finite and non-negative, and at least one
  comparator is required.

## 8. Context bucket views (the configurable-system seam)

Per [configurable-node-schema-design.md](../recentering/configurable-node-schema-design.md),
consumers do not read raw node blocks — they ask the node for **their bucket of context**.
The loader exposes four typed views over the flat fields above, plus a coverage query:

| Accessor (`scripts/taxonomy.py`) | Surfaces | Bucket |
| --- | --- | --- |
| `node_expertise(node)` | `fingerprint`, `priors`, `semantic_checks`, `scenario_assumption_dimensions`, `scaffold_hints.interface_hint`/`content_file` | classification + review |
| `node_implementation(node)` | `pluggable_component`, `scaffold_hints.templates_dir`, `notebook_layout`, `demo_skill`, `family_components`, `build_plan`, `package_manifest` | build/codegen contract |
| `node_parameters(node)` | `smoke_economics`, `paradigm_extras`, `model_defaults`, `params_derivation`, `provenance_params`, `tier_heuristic` | param derivation |
| `node_testing(node)` | `semantic_checks`, `scenario_assumption_dimensions`, `smoke_bugs`, `smoke_economics`, `demo_success`, `demo_skill` | probes + smoke + demo verdict |
| `bucket_coverage(node)` | `{bucket: [fields present]}` — the mechanical "ready to serve X?" check | — |

Each view returns **present fields only** (an omitted block ⇒ the consumer falls back —
absent context is never a crash). `semantic_checks` and `smoke_economics` appear in two
views by design (a field can be both knowledge and a probe, both a floor and a scale).
The field membership lives in the per-bucket key tuples inside `scripts/taxonomy.py`'s four `node_*` view functions; the four bucket field-shape
lints in §7 are the per-bucket completeness gate.

Bucket-story status (os-cleanup, 2026-08-19): the duplicate `_BUCKET_FIELDS`
map this section once named as the single source is dead and deleted; the
four accessors above ARE the surface. Two of them (`node_parameters`,
`node_testing`) currently have no production consumer — they are parked
pending an actual consumer need, not a promise of one. `notebook_layout`, `paradigm_extras`, and
`demo_skill` are inheritable node fields, so the views/lint/coverage work the moment a node
populates them.

### 3.12 `notebook_layout` (implementation; Phase 1.6)

Notebook section contract the notebook-generator emits against and
`validate_notebook_output` checks. A top-level `description` plus an ordered
`sections[]`; each section carries `{id, title, kind: agnostic|paradigm|method_specific,
content}` and may add `subsections[]` (same `{id, title, content}` shape) and
`generator_instructions`. The validator's hard requirement (and the §7 lint) is only
that `sections` is non-empty and every entry has an `id`; the richer fields are guidance
the generator agent interprets. Top-level variant field; inherited like any mapping field
(AL declares it on the parent; sub-variants inherit).

### 3.13 `paradigm_extras` (parameters; Phase 5.7)

Per-paradigm smoke-default values + reasoning for the method's pluggable-signature extras,
replacing `derive_params`' hardcoded `KNOWN_EXTRAS_BY_PARADIGM`. Maps `<extra_name> ->
{value, reasoning_template}`. Top-level (design §7 Q2, revisable); merged by extra-name down
the chain so a sub-variant overrides a single extra without restating the rest.

### 3.14 `model_defaults` (parameters; Phase 5.7b)

Per-paradigm **`system_inferred` fallbacks** for model-architecture params (`hidden_dim`,
`dropout_rate`), replacing the hardcoded else-branches in `derive_params._add_model_params`.
Maps `<param_name> -> {value, reasoning_template}` (same shape as `paradigm_extras`; the
`reasoning_template` keeps its `{value}` `.format()` slot). **The node owns ONLY the
fallback arm**: the `source: paper` and CNN-substitution arms are computed per-run from the
spec's own evidence and are never node-stored — so a `model_defaults` entry is always
emitted as `source: system_inferred`. Served via `taxonomy.load_model_default`; the consumer
keeps Python literals only as defensive fallback for provisional/future ids.
`hidden_dim` lives on the parent `active_learning` node
(AL-wide — every neural sub-variant inherits it); `dropout_rate` on the `bayesian`
sub-variant only.

### 3.15 `demo_success` (testing; demo-success semantics, 2026-07-16)

Success/failure markers for the family's **headline demo section** — the input
to the deterministic post-smoke verdict pass (`scripts/demo_verdict.py`), which
records `succeeded` / `failed` / `undetermined` after a clean smoke execution.
Kit territory, next to `smoke_bugs`; committed families and provisional/gap
packs use the same declaration surface (the pack overlay grafts it like any
inheritable mapping field). Served via `taxonomy.load_demo_success`.

```yaml
demo_success:
  headline_section_id: running       # optional; defaults to `running`
  failure_markers:                   # line regexes over the section's executed outputs
    - pattern: '(?im)^.*\bplanning\s+status\s*:\s*(?:timeout|infeasible|error).*$'
      gloss: "the planner reports it did not find a plan"
  success_markers:
    - pattern: '(?im)^.*\bstatus\s*:\s*success\b.*$'
      gloss: "the planner reports success"
  checks:                            # named deterministic checks (DEMO_CHECK_KINDS)
    - kind: beats_chance             # accuracy series vs 1/num_classes + shared margin
```

Evaluation order is failure markers, then success markers, then checks — a
headline demo that prints a failure line does not demonstrably succeed, so the
failure reading wins. A family without the block yields `undetermined` (honest;
on a committed family the verdict pass additionally records a kit-coverage
finding we own). `verdict=failed` is a first-class delivery demoter, never a
halt (`delivery_label.py`).
### 3.16 `family_components` (implementation; arch-contract headroom, 2026-07-16)

The pack-declared legal names for the architecture contract's one typed extension
container, `arch_contract.family_components`. The universal ArchContract skeleton
stays `extra='forbid'`; a family that needs a top-level component block the skeleton
cannot know in advance (single-agent RL's `reward_function`, SRL 2026-07-15; the shape
stochastic_optimization's `optimizer_state` was hard-added for, ADAM 2026-07-08)
declares it here. Maps `<component_name> -> {required, description, required_entries}`:

```yaml
family_components:
  reward_function:
    required: true                      # default true; false = legal but optional
    description: "Reward the policy optimizes, as method.py computes it."
    required_entries: []                # entry names a multi-slot block must carry
```

Flows through the spec-derived build plan into `arch_contract_requirements` and is
enforced by `scripts/validate_arch_contract.py` in **both directions**: a
declared-required name missing from the contract fails, and a present name the pack
never declared fails naming the unknown key and the declared set (typo safety for the
container). Merged by component name down the chain, like `paradigm_extras`.
Provisional (gap-path) packs use the exact same field;
`validate_paradigm_proposal.py` type-checks their declarations at authoring time.

### 3.17 `params_derivation` (parameters; plan item 9, 2026-07-21)

The machine-readable half of `stage_review_focus.stage_2x_params`: the parameter
entries the deterministic deriver (`scripts/derive_params.py`) must emit or drop
for this paradigm. Exists because a review check is prose the reviewer enforces
but no script can act on — the DomIndOnto (missing KBP configuration paths) and
fedavg (missing derived `u = nE/(KB)`) 2026-07-21 halts were both the reviewer
failing params.json for entries the producer had no way to produce. Maps
`<param_name> -> {kind, ...}` with three kinds:

```yaml
params_derivation:
  ontology_path:
    kind: config_path                  # pipeline configuration, not an ML hyperparameter
    reasoning: "Path to the domain ontology the extraction pipeline loads."
    # optional: demo_value, paper_section, used_in_notebook
  u_expected_updates:
    kind: derived_statistic            # paper-defined statistic, derived + documented
    formula: "n * E / (K * B)"         # arithmetic only (safe-evaluated, never exec)
    inputs:                            # symbol -> params.<name> | spec.<dotted.path>
      E: params.E
      B: params.B
      n: spec.critical_requirements.data_setup.dataset_size
      K: spec.critical_requirements.data_setup.num_clients
    paper_section: "Section 2 (eq-expected-updates)"
  hidden_dim:
    kind: suppress                     # drop an inapplicable convention param
    reason: "This family has no neural model; hidden_dim is meaningless."
```

Consumed last in `derive_params.derive()` via `taxonomy.load_params_derivation`:
`suppress` drops the named entry wherever it came from; the other kinds never
overwrite an entry an earlier derivation path already sourced. A
`derived_statistic` whose inputs cannot all be resolved emits an honest
documented entry (formula + which inputs were unavailable) with `value: null`
instead of guessing. Emitted entries are `source: system_inferred` with
`used_in_notebook: false` by default (`config_path` flips to used when it
carries a `demo_value` or declares `used_in_notebook: true`). Merged by param
name down the chain. Provisional (gap-path) packs use the exact same field;
`validate_paradigm_proposal.py` type-checks declarations at authoring time and
warns when a pack declares a `stage_2x_params` check with no
`params_derivation` block.

### 3.18 `scenario_assumption_dimensions` (expertise + testing, scenario fidelity slices A+B)

The family-scoped vocabulary the analyzer may use when capturing
`method_spec.scenario_assumptions`, plus the runtime detector binding the probe
battery enforces at verification time. Each entry declares one stable dimension
id, plain-language normalization guidance, and a detector executor ref with its
authored explanation context:

```yaml
scenario_assumption_dimensions:
  - id: obstacle_geometry
    display_name: "Obstacle geometry"
    description: "Geometry families the paper permits for obstacles."
    normalized_value_guidance: >
      Use one lowercase shape family or a flat list when the paper names
      several, for example "circle" or ["circle", "polygon"].
    detector: motion_planning.scenario_geometry
    check: >
      The executed demo's obstacle shapes stay within the geometry families
      the paper states.
    silent_failure: "A circles-only paper is demonstrated on polygon obstacles."
```

Capture (slice A, 2026-07-23) uses the id and guidance for strict id joining,
report visibility, and notebook-generator guidance. Enforcement (slice B,
2026-07-27) binds `detector` exactly like `semantic_checks[].probe`: the probe
battery executes the notebook's setup slice in a bounded subprocess, the
family-owned detector observes the instantiated setup, and the comparison
against the captured paper value emits a scenario-fidelity verdict (SC-*
probe ids, finding class M-006 — a demo setup that contradicts a paper-stated
scenario assumption). A proven mismatch fails and demotes, a captured
assumption the detector cannot bind flags for researcher judgment and demotes,
and a dimension with no captured assumption discloses as unprobeable without
demoting. Declaring a dimension without a detector is authoring-invalid: the
declaration itself is the family's opt-in to runtime comparison and demotion.

The list inherits by id so a motion-planning sub-variant can add a dimension
without restating the family vocabulary. Families that omit the field provide
no capture ids, produce no scenario-assumption report section, and see no
scenario verdicts — their reports stay byte-identical.

### 3.19 `demo_skill` (implementation + testing; R2C-086)

The family-owned policy for deriving demonstrated task skill from one
structured executed-evaluation record. It does not describe whether the
notebook executed (`demo_success` owns coarse execution markers), whether the
split is valid (the pipeline's split receipt owns that), or whether the
paper's mechanism is behaviorally live (probes own that). A prose PASS/FAIL
line is never input authority.

```yaml
demo_skill:
  schema_version: "1.0.0"
  primary_metric:
    id: rmse
    direction: lower_is_better       # lower_is_better | higher_is_better
    aggregation: root_mean_squared_error
    units: target_units
  eligibility:
    minimum_finite_rows: 1
    require_nonzero_actuals: true
  decision_policy: all_required
  comparators:
    - id: predict_zero
      role: degeneracy_floor          # degeneracy_floor | minimum_competence | context
      implementation: predict_zero
      required: true
      absolute_margin: 0.0
      tolerance: 1.0e-12
    - id: repeat_last
      role: minimum_competence
      implementation: repeat_last_pre_window
      required: true
      absolute_margin: 0.0
      tolerance: 1.0e-12
```

All keys above are strict: the lint rejects unknown or missing fields, unknown
vocabulary values, duplicate comparator ids, non-boolean `required`, invalid
eligibility values, and negative or non-finite thresholds. At least one
comparator must be required; a family without a meaningful task-skill
comparator omits the contract and receives no inferred forecasting policy.

The block is inherited and served by `taxonomy.load_demo_skill`, including for
run-local provisional-pack overlays. It appears in both implementation and
testing views: the producer needs the contract to emit compatible structured
evidence, and deterministic evaluation consumes the same declaration. For
forecasting, predict-zero is the required degeneracy floor and repeat-last is
the required minimum-competence comparator. Both are evaluated over identical
actuals, finite mask, units, row ids, evaluation positions, and aggregation;
the finite mask is recomputed as the exact finiteness mask of actuals, and
repeat-last uses the final position immediately before the complete evaluation
window. Comparator
implementations are a closed executable vocabulary (`predict_zero`,
`repeat_last_pre_window`, and `constant_prediction`); a constant comparator's
finite `constant_value` lives in the family contract, never the executed
record.

The corresponding executed record is not part of this taxonomy mapping. The
producer emits it as one `R2C_DEMO_EVALUATION_JSON: <object>` output line. Its
top level includes `schema_version`, `target_root`, `model_id`, metric and row
metadata, exact actuals/mask/model predictions plus the recorded model metric
with deterministic identity hashes, and comparator objects with their own
recorded metric and the same coidentity fields. The pipeline recomputes every
metric and requires numerical agreement. The pipeline
and producer share one hash recipe: `sha256:` plus SHA-256 over UTF-8 compact
JSON with sorted keys and no NaN tokens; actual identity covers
`{actuals, row_ids, units}` and mask identity covers
`{finite_mask, row_ids}`. Masked missing values are JSON null. The pipeline
rejects a false mask entry for a finite actual and compares row/position
identity using canonical JSON types, so integer and floating identifiers do
not alias. It also requires every repeat-last source position to equal the
final position immediately before the complete evaluation window. The pipeline
uses `target_root`, `model_id`, and evaluation positions to bind the record to
the trusted fitting/evaluation relation. The notebook does not put an
`evaluation_validity` assertion in the record; that authority remains with the
pipeline-owned split receipt.

### 3.20 `build_plan` (implementation; R2C-032 and R2C-052)

This implementation-bucket mapping holds declarations consumed while resolving
the per-run build plan. A provisional node may declare `source` as
`inherit_parent` or `neutral` (R2C-032). A committed data-dependent family may
declare a family-owned `offline_demo_data` subcontract. The first supported
grammar is the closed time-series panel schema:

```yaml
build_plan:
  offline_demo_data:
    schema_id: time_series_panel_v1
    synthetic_feasible: true
    generator_id: time_series_offline_fallback
    generator_version: 1.0.0
    entity_count: 8
    family_fitting_prefix_steps: 24
    supported_cadences: [week, day, hour, minute, second]
    tables:
      series.csv:
        columns:
          - {name: series_id, dtype: string, semantic_role: entity_id}
          - {name: static_level, dtype: float64, semantic_role: static_numeric_level}
          - {name: static_amplitude, dtype: float64, semantic_role: static_numeric_amplitude}
        primary_key: [series_id]
      observations.csv:
        columns:
          - {name: series_id, dtype: string, semantic_role: entity_id}
          - {name: timestamp, dtype: datetime_iso8601_utc, semantic_role: time_index}
          - {name: target, dtype: float64, semantic_role: target}
          - {name: season_sin, dtype: float64, semantic_role: known_time_varying_covariate}
          - {name: season_cos, dtype: float64, semantic_role: known_time_varying_covariate}
          - {name: time_fraction, dtype: float64, semantic_role: known_time_varying_covariate}
        primary_key: [series_id, timestamp]
```

The table names, column order, dtypes, semantic roles, entity count, generator
identity/version, fitting prefix, and cadence vocabulary are exact. Callers
cannot substitute another shape. `family_fitting_prefix_steps` is an explicit
system-owned smoke-fixture capacity (two cycles of the generator's 12-step
season), not a paper protocol fact or a fitting-partition claim. The resolved
plan adds `forecast_horizon_demo_value` from the
same node's positive integral
`params_derivation.forecast_horizon.demo_value`; it is intentionally absent
from `offline_demo_data` in the SSOT so the system-owned K default has only one
authority. Families or schemas outside this declared grammar remain unsupported
until their own typed contract is added.


## 9. Extending the taxonomy: the real per-family surface list

A new method family is NOT one YAML node. The full surface, in dependency
order, with the failure you get when a step is skipped. Worked examples with
every surface populated today: `knowledge_distillation/detection` and (since
the B-02 migration) `active_learning`.

1. **The node itself** in `docs/ssot/taxonomies.yaml` — root → family →
   variant, `status: populated`, `legacy_paradigm`, `fingerprint.what_it_is`.
   Skipped: `scripts/validate_taxonomy.py` lint errors
   (`populated_no_legacy`, `populated_no_fingerprint`), and `serves()`
   returns nothing so every downstream consumer falls back.
   If the family needs a NEW METHOD ROOT: the root ids are a hard-coded
   tuple, `METHOD_ROOT_IDS` in `scripts/taxonomy.py`. Skipped: the lint
   rejects the whole SSOT with `root_unknown`.
2. **Templates** under `paradigms/<path>/templates/method/` plus
   `scaffold_hints.templates_dir` on the node. Every template lives under
   `templates/method/` — a template one level up renders into the RUN ROOT
   instead of the package (the cross_modal misplaced-README class;
   `tests/test_paradigm_templates.py` now closes both directions:
   declared-to-rendered and rendered-to-declared). Skipped: stage 2a
   scaffolds nothing, or the lint fails with `templates_dir_missing`.
3. **`scaffold_hints.demo_data_source`** (`template_downloads` |
   `needs_acquisition`). The lint only WARNS when undeclared
   (`demo_data_source_undeclared`) — but an undeclared family that gets
   promoted out of the gap path can silently lose its dataset acquisition
   (the R2C-074 incident). Declare it.
4. **The static build-plan entry** in `scripts/build_plan.py`'s
   `STATIC_PLAN_BY_PARADIGM`. THE EXACT-KEY TRAP: committed ids match by
   exact key only — a family-only entry does not serve its variants (the
   motion-planning readiness blocker), so add one entry per served id
   (see the three `active_learning*` keys). Skipped: `load_build_plan`
   returns None and runs take the no-plan path; the scaffolder validator
   and 2.b architecture gate silently lose their manifest.
5. **Probe catalogs** under `scripts/probes/catalogs/` registered in
   `scripts/probes/catalog.py`. Skipped: only the universal probes run, and
   a family pass cannot certify the paper's contribution (the
   probe-binding rule: an unbound pass is non-certifying).
6. **Paradigm-name branches** in `scripts/derive_params.py`,
   `scripts/run_probes.py`, and `scripts/validate_notebook_output.py`
   (`paradigm_id.startswith(...)` sites). Audit them for whether the new
   family needs its own arm. Skipped: family-specific derivations and
   validations silently do not run — nothing fails.
7. **The `EXPECTED_LEGACY_PARADIGMS` allowlist** in
   `tests/test_taxonomy.py` (via `POPULATED_PARADIGMS`). Skipped: the
   suite does not know the family exists, so nothing pins that it stays
   served; updating it is part of the same diff that adds the node.
8. **Generated-docs refresh**: run `python3 scripts/refresh_ssot_docs.py`
   and `python3 scripts/validate_field_guides.py`. Skipped:
   `tests/test_ssot_docs.py` fails on drift, and the node-count change
   detector in `tests/test_field_guides.py` trips (update its pin in the
   same diff — that is its job).

Section 3's field reference tells you what a node can carry; this list is
what a CONTRIBUTOR has to touch. The pack-authoring path (a provisional
gap-pack that later promotes) shares steps 1-3 through the pack schema and
inherits its build plan from the nearest committed ancestor until a static
entry exists (step 4).
