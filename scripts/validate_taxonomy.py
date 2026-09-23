"""Repo-hygiene check. The driver never calls this.

Lint the taxonomy SSOT (`docs/ssot/taxonomies.yaml`).

Phase 0 of the paradigm -> taxonomy migration. Runs the structural diagnostics
the loader produces (duplicate slugs/ids, unresolved aliases, unknown roots)
**plus** the deeper node-level lint from the schema contract
(`docs/ssot/taxonomy-node-schema.md` §7):

  - every method root id is one of TE CLC SC AD PP OM GEN DA INF
  - every variant has a unique taxonomy_id and slug (loader)
  - every alias resolves and does not shadow a canonical slug (loader)
  - every `route_elsewhere[].to` resolves to a real variant
  - every `domain_checks` key resolves to a real task-domain id or leaf
  - every `semantic_checks[]` / `smoke_bugs[]` entry sets a valid `status`
    (`seed | observed`) and (for semantic_checks) `id`/`stage`/`severity`
  - every declared `demo_skill` block has the exact schema-versioned metric,
    eligibility, decision-policy, and comparator grammar
  - every `populated` node carries a `legacy_paradigm` and `fingerprint.what_it_is`
  - no two `populated` nodes claim the same `legacy_paradigm` (loader)

Also validates `(method_variant, task_domain)` tuples passed on the CLI.

Usage:
    python scripts/validate_taxonomy.py                 # lint the SSOT
    python scripts/validate_taxonomy.py --check active_learning:CV   # validate a tuple

Exit codes:
    0  no errors
    1  one or more lint errors (or an invalid --check tuple)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.taxonomy import (  # noqa: E402
    DEMO_CHECK_KINDS,
    DEMO_DATA_SOURCES,
    DEMO_SKILL_COMPARATOR_IMPLEMENTATIONS,
    DEMO_SKILL_COMPARATOR_ROLES,
    DEMO_SKILL_DECISION_POLICIES,
    DEMO_SKILL_METRIC_DIRECTIONS,
    DEMO_SKILL_SCHEMA_VERSION,
    Taxonomy,
    TaxonomyDiagnostic,
    TaxonomyError,
    VariantNode,
    assert_classification_tuple,
    bucket_coverage,
    load_taxonomy,
)

_DEMO_CHECK_KINDS = set(DEMO_CHECK_KINDS)
_DEMO_SKILL_COMPARATOR_IMPLEMENTATIONS = set(
    DEMO_SKILL_COMPARATOR_IMPLEMENTATIONS
)
_DEMO_SKILL_COMPARATOR_ROLES = set(DEMO_SKILL_COMPARATOR_ROLES)
_DEMO_SKILL_DECISION_POLICIES = set(DEMO_SKILL_DECISION_POLICIES)
_DEMO_SKILL_METRIC_DIRECTIONS = set(DEMO_SKILL_METRIC_DIRECTIONS)

_VALID_STATUS = {"seed", "observed"}
_VALID_SEVERITY = {"error", "warning"}
_REVIEW_STAGE_IDS = {
    "stage_1_analyzer",
    "stage_2b_architecture",
    "stage_2c_method",
    "stage_2x_params",
    "stage_3a_notebook",
}


def lint(taxonomy: Taxonomy) -> list[TaxonomyDiagnostic]:
    """Return all diagnostics: loader structural ones + node-level lint."""
    diags: list[TaxonomyDiagnostic] = list(taxonomy.diagnostics)

    def err(code: str, message: str, node: str | None = None) -> None:
        diags.append(TaxonomyDiagnostic("error", code, message, node))

    def warn(code: str, message: str, node: str | None = None) -> None:
        diags.append(TaxonomyDiagnostic("warning", code, message, node))

    def _lint_check(check: dict, node_id: str, kind: str) -> None:
        if not isinstance(check, dict):
            err("check_malformed", f"{kind} entry is not a mapping", node_id)
            return
        status = check.get("status")
        if status not in _VALID_STATUS:
            err("check_status_missing", f"{kind} {check.get('id')!r} status must be seed|observed (got {status!r})", node_id)
        if kind == "semantic_checks":
            if not check.get("id"):
                err("check_id_missing", "semantic_checks entry missing `id`", node_id)
            if check.get("stage") not in _REVIEW_STAGE_IDS:
                err("check_stage_invalid", f"semantic_checks {check.get('id')!r} stage {check.get('stage')!r} not a review stage", node_id)
            if check.get("severity") not in _VALID_SEVERITY:
                err("check_severity_invalid", f"semantic_checks {check.get('id')!r} severity {check.get('severity')!r} invalid", node_id)

    def _lint_pluggable_component(pc: dict, node_id: str) -> None:
        if not isinstance(pc, dict):
            err("pluggable_malformed", "pluggable_component must be a mapping", node_id)
            return
        for key in ("signature_template", "return_type"):
            if not isinstance(pc.get(key), str) or not pc.get(key):
                err("pluggable_field_missing", f"pluggable_component missing `{key}`", node_id)
        contract = pc.get("contract")
        if not isinstance(contract, dict):
            err("pluggable_contract_missing", "pluggable_component missing `contract` mapping", node_id)
            return
        for key in ("fixed_positional_args", "seed_param", "paradigm_extras_forwarding", "forbidden_param_names"):
            if key not in contract:
                err("pluggable_contract_key_missing", f"pluggable_component.contract missing `{key}`", node_id)
        for list_key in ("fixed_positional_args", "forbidden_param_names"):
            if list_key in contract and not isinstance(contract[list_key], list):
                err("pluggable_contract_shape", f"pluggable_component.contract.{list_key} must be a list", node_id)

    # --- per-bucket field-shape lint (configurable-node-schema-design.md §4) ---
    # "A node that serves bucket X carries X's required fields." These fire only
    # on a *declared* block, so a node mid-migration (which simply omits the
    # not-yet-migrated field) stays clean; bucket *provision* is reported
    # separately by `taxonomy.bucket_coverage`.

    def _lint_notebook_layout(nl: dict, node_id: str) -> None:
        # IMPLEMENTATION bucket (Phase 1.6).
        if not isinstance(nl, dict):
            err("notebook_layout_malformed", "notebook_layout must be a mapping", node_id)
            return
        sections = nl.get("sections")
        if not isinstance(sections, list) or not sections:
            err("notebook_layout_sections_missing", "notebook_layout must carry a non-empty `sections` list", node_id)
            return
        for sec in sections:
            if not isinstance(sec, dict) or not sec.get("id"):
                err("notebook_layout_section_id_missing", "notebook_layout.sections entry missing `id`", node_id)

    def _lint_paradigm_extras(pe: dict, node_id: str) -> None:
        # PARAMETERS bucket (Phase 5.7).
        if not isinstance(pe, dict):
            err("paradigm_extras_malformed", "paradigm_extras must be a mapping name -> {value, reasoning_template}", node_id)
            return
        for name, entry in pe.items():
            if not isinstance(entry, dict) or "value" not in entry or not entry.get("reasoning_template"):
                err("paradigm_extras_entry_shape", f"paradigm_extras[{name!r}] must carry `value` and `reasoning_template`", node_id)

    def _lint_demo_success(ds: dict, node_id: str) -> None:
        # TESTING bucket (demo-success-semantics design, 2026-07-16). The block
        # feeds the deterministic post-smoke verdict pass, so a malformed
        # marker must fail here (author-time) rather than silently yield
        # `undetermined` on every run.
        import re as _re  # noqa: PLC0415 - lint-local

        if not isinstance(ds, dict):
            err("demo_success_malformed", "demo_success must be a mapping", node_id)
            return
        known_keys = {"headline_section_id", "success_markers", "failure_markers", "checks"}
        for key in ds:
            if key not in known_keys:
                err("demo_success_unknown_key",
                    f"demo_success key {key!r} is not one of {sorted(known_keys)}", node_id)
        markers_or_checks = False
        for list_key in ("success_markers", "failure_markers"):
            entries = ds.get(list_key)
            if entries is None:
                continue
            if not isinstance(entries, list):
                err("demo_success_markers_shape", f"demo_success.{list_key} must be a list", node_id)
                continue
            for entry in entries:
                if not isinstance(entry, dict) or not entry.get("pattern") or not entry.get("gloss"):
                    err("demo_success_marker_entry",
                        f"demo_success.{list_key} entries must carry `pattern` (regex) and `gloss` (plain language)", node_id)
                    continue
                try:
                    _re.compile(str(entry["pattern"]))
                except _re.error as exc:
                    err("demo_success_marker_regex",
                        f"demo_success.{list_key} pattern {entry['pattern']!r} is not a valid regex: {exc}", node_id)
                    continue
                markers_or_checks = True
        checks = ds.get("checks")
        if checks is not None:
            if not isinstance(checks, list):
                err("demo_success_checks_shape", "demo_success.checks must be a list", node_id)
            else:
                for entry in checks:
                    kind = entry.get("kind") if isinstance(entry, dict) else None
                    if kind not in _DEMO_CHECK_KINDS:
                        err("demo_success_check_kind",
                            f"demo_success.checks entry kind {kind!r} is not a known check "
                            f"(known: {sorted(_DEMO_CHECK_KINDS)})", node_id)
                        continue
                    markers_or_checks = True
        if not markers_or_checks:
            err("demo_success_empty",
                "demo_success declares no valid marker or check — drop the block "
                "or add at least one failure/success marker or a named check", node_id)
        hs = ds.get("headline_section_id")
        if hs is not None and (not isinstance(hs, str) or not hs):
            err("demo_success_headline_section",
                "demo_success.headline_section_id must be a non-empty section id string", node_id)

    def _lint_demo_skill(contract: object, node_id: str) -> None:
        """Strict author-time grammar for family-owned skill comparisons.

        A malformed policy cannot be repaired from notebook prose at runtime:
        the evaluator must know exactly which metric and comparator semantics
        the family owns before it reads executed evidence.
        """

        def exact_keys(
            value: object,
            *,
            path: str,
            required: set[str],
            code: str,
        ) -> dict | None:
            if not isinstance(value, dict):
                err(code, f"{path} must be a mapping", node_id)
                return None
            missing = sorted(required - set(value))
            unknown = sorted(set(value) - required, key=repr)
            if missing:
                err(code, f"{path} missing required keys {missing!r}", node_id)
            if unknown:
                err(code, f"{path} has unknown keys {unknown!r}", node_id)
            return value

        top = exact_keys(
            contract,
            path="demo_skill",
            required={
                "schema_version",
                "primary_metric",
                "eligibility",
                "decision_policy",
                "comparators",
            },
            code="demo_skill_shape",
        )
        if top is None:
            return
        if top.get("schema_version") != DEMO_SKILL_SCHEMA_VERSION:
            err(
                "demo_skill_schema_version",
                "demo_skill.schema_version must be "
                f"{DEMO_SKILL_SCHEMA_VERSION!r}",
                node_id,
            )

        metric = exact_keys(
            top.get("primary_metric"),
            path="demo_skill.primary_metric",
            required={"id", "direction", "aggregation", "units"},
            code="demo_skill_primary_metric",
        )
        if metric is not None:
            for key in ("id", "aggregation", "units"):
                if not isinstance(metric.get(key), str) or not metric.get(key):
                    err(
                        "demo_skill_primary_metric",
                        f"demo_skill.primary_metric.{key} must be a non-empty string",
                        node_id,
                    )
            direction = metric.get("direction")
            if not isinstance(direction, str) \
                    or direction not in _DEMO_SKILL_METRIC_DIRECTIONS:
                err(
                    "demo_skill_metric_direction",
                    "demo_skill.primary_metric.direction must be one of "
                    f"{sorted(_DEMO_SKILL_METRIC_DIRECTIONS)!r}",
                    node_id,
                )

        eligibility = exact_keys(
            top.get("eligibility"),
            path="demo_skill.eligibility",
            required={"minimum_finite_rows", "require_nonzero_actuals"},
            code="demo_skill_eligibility",
        )
        if eligibility is not None:
            minimum = eligibility.get("minimum_finite_rows")
            if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
                err(
                    "demo_skill_minimum_finite_rows",
                    "demo_skill.eligibility.minimum_finite_rows must be a positive integer",
                    node_id,
                )
            if not isinstance(eligibility.get("require_nonzero_actuals"), bool):
                err(
                    "demo_skill_require_nonzero_actuals",
                    "demo_skill.eligibility.require_nonzero_actuals must be boolean",
                    node_id,
                )

        decision_policy = top.get("decision_policy")
        if not isinstance(decision_policy, str) \
                or decision_policy not in _DEMO_SKILL_DECISION_POLICIES:
            err(
                "demo_skill_decision_policy",
                "demo_skill.decision_policy must be one of "
                f"{sorted(_DEMO_SKILL_DECISION_POLICIES)!r}",
                node_id,
            )

        comparators = top.get("comparators")
        if not isinstance(comparators, list) or not comparators:
            err(
                "demo_skill_comparators",
                "demo_skill.comparators must be a non-empty list",
                node_id,
            )
            return
        seen_ids: set[str] = set()
        required_count = 0
        common_comparator_keys = {
            "id",
            "role",
            "implementation",
            "required",
            "absolute_margin",
            "tolerance",
        }
        for index, raw in enumerate(comparators):
            path = f"demo_skill.comparators[{index}]"
            comparator_keys = set(common_comparator_keys)
            if isinstance(raw, dict) \
                    and raw.get("implementation") == "constant_prediction":
                comparator_keys.add("constant_value")
            comparator = exact_keys(
                raw,
                path=path,
                required=comparator_keys,
                code="demo_skill_comparator_shape",
            )
            if comparator is None:
                continue
            comparator_id = comparator.get("id")
            if not isinstance(comparator_id, str) or not comparator_id:
                err(
                    "demo_skill_comparator_id",
                    f"{path}.id must be a non-empty string",
                    node_id,
                )
            elif comparator_id in seen_ids:
                err(
                    "demo_skill_comparator_id_duplicate",
                    f"demo_skill.comparators repeats id {comparator_id!r}",
                    node_id,
                )
            else:
                seen_ids.add(comparator_id)
            role = comparator.get("role")
            if not isinstance(role, str) \
                    or role not in _DEMO_SKILL_COMPARATOR_ROLES:
                err(
                    "demo_skill_comparator_role",
                    f"{path}.role must be one of "
                    f"{sorted(_DEMO_SKILL_COMPARATOR_ROLES)!r}",
                    node_id,
                )
            implementation = comparator.get("implementation")
            if not isinstance(implementation, str) \
                    or implementation not in _DEMO_SKILL_COMPARATOR_IMPLEMENTATIONS:
                err(
                    "demo_skill_comparator_implementation",
                    f"{path}.implementation must be one of "
                    f"{sorted(_DEMO_SKILL_COMPARATOR_IMPLEMENTATIONS)!r}",
                    node_id,
                )
            if implementation == "constant_prediction":
                constant = comparator.get("constant_value")
                if (
                    isinstance(constant, bool)
                    or not isinstance(constant, (int, float))
                    or not math.isfinite(float(constant))
                ):
                    err(
                        "demo_skill_comparator_constant",
                        f"{path}.constant_value must be a finite number",
                        node_id,
                    )
            required_value = comparator.get("required")
            if not isinstance(required_value, bool):
                err(
                    "demo_skill_comparator_required",
                    f"{path}.required must be boolean",
                    node_id,
                )
            elif required_value:
                required_count += 1
            for key in ("absolute_margin", "tolerance"):
                value = comparator.get(key)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or value < 0
                ):
                    err(
                        "demo_skill_comparator_threshold",
                        f"{path}.{key} must be a finite non-negative number",
                        node_id,
                    )
        if required_count == 0:
            err(
                "demo_skill_required_comparator_missing",
                "demo_skill must declare at least one required comparator; "
                "families without one should omit the contract",
                node_id,
            )

    def _lint_scenario_assumption_dimensions(
        dimensions: object, node_id: str
    ) -> None:
        # EXPERTISE/TESTING bridge (scenario-fidelity slices A+B). Each
        # dimension is capture vocabulary AND a runtime detector binding:
        # since slice B (2026-07-27) a declared dimension with no `detector`
        # ref (plus its authored `check`/`silent_failure` context) is
        # authoring-invalid — declaring a dimension opts the family into
        # runtime comparison and demotion.
        if not isinstance(dimensions, list) or not dimensions:
            err(
                "scenario_assumption_dimensions_shape",
                "scenario_assumption_dimensions must be a non-empty list",
                node_id,
            )
            return
        seen: set[str] = set()
        for entry in dimensions:
            if not isinstance(entry, dict):
                err(
                    "scenario_assumption_dimension_entry",
                    "scenario_assumption_dimensions entries must be mappings",
                    node_id,
                )
                continue
            dimension_id = entry.get("id")
            if not isinstance(dimension_id, str) or not dimension_id:
                err(
                    "scenario_assumption_dimension_id",
                    "scenario_assumption_dimensions entry missing non-empty `id`",
                    node_id,
                )
                continue
            if dimension_id in seen:
                err(
                    "scenario_assumption_dimension_duplicate",
                    f"scenario_assumption_dimensions repeats id {dimension_id!r}",
                    node_id,
                )
            seen.add(dimension_id)
            for key in (
                "display_name",
                "description",
                "normalized_value_guidance",
                "detector",
                "check",
                "silent_failure",
            ):
                if not isinstance(entry.get(key), str) or not entry[key].strip():
                    err(
                        "scenario_assumption_dimension_field",
                        f"scenario_assumption_dimensions[{dimension_id!r}] "
                        f"missing non-empty `{key}`",
                        node_id,
                    )
            detector = entry.get("detector")
            if isinstance(detector, str) and detector.strip() and "." not in detector:
                err(
                    "scenario_assumption_dimension_detector",
                    f"scenario_assumption_dimensions[{dimension_id!r}] "
                    f"detector {detector!r} must be a family-scoped executor "
                    f"ref like 'motion_planning.scenario_geometry'",
                    node_id,
                )

    def _lint_model_defaults(md: dict, node_id: str) -> None:
        # PARAMETERS bucket (Phase 5.7b). Each entry is the *system_inferred*
        # fallback for a model-architecture param; same shape as paradigm_extras.
        if not isinstance(md, dict):
            err("model_defaults_malformed", "model_defaults must be a mapping name -> {value, reasoning_template}", node_id)
            return
        for name, entry in md.items():
            if not isinstance(entry, dict) or "value" not in entry or not entry.get("reasoning_template"):
                err("model_defaults_entry_shape", f"model_defaults[{name!r}] must carry `value` and `reasoning_template`", node_id)

    def _lint_params_derivation(pd: dict, node_id: str) -> None:
        # PARAMETERS bucket (plan item 9, 2026-07-21). The machine-readable
        # half of stage_review_focus.stage_2x_params: entries the deriver
        # emits (config_path, derived_statistic) or drops (suppress). A
        # malformed entry must fail at author time — at run time the deriver
        # silently skips what it cannot read, and the 2.x reviewer would
        # then halt the run on the missing entry (the DomIndOnto/fedavg
        # 2026-07-21 class this surface exists to close).
        if not isinstance(pd, dict):
            err("params_derivation_malformed",
                "params_derivation must be a mapping param name -> {kind, ...}", node_id)
            return
        protocol_roles: dict[str, str] = {}
        legal_protocol_roles = {
            "context_length",
            "forecast_call_horizon",
            "validation_span",
            "test_span",
        }
        for name, entry in pd.items():
            if not isinstance(entry, dict):
                err("params_derivation_entry_shape",
                    f"params_derivation[{name!r}] must be a mapping with a `kind`", node_id)
                continue
            kind = entry.get("kind")
            if kind == "config_path":
                if not entry.get("reasoning"):
                    err("params_derivation_config_path_reasoning",
                        f"params_derivation[{name!r}] (config_path) must carry `reasoning` "
                        "— what the path configures and where the method consumes it", node_id)
            elif kind == "derived_statistic":
                if not entry.get("formula"):
                    err("params_derivation_formula_missing",
                        f"params_derivation[{name!r}] (derived_statistic) must carry `formula`", node_id)
                inputs = entry.get("inputs")
                if not isinstance(inputs, dict) or not inputs:
                    err("params_derivation_inputs_missing",
                        f"params_derivation[{name!r}] (derived_statistic) must carry a non-empty "
                        "`inputs` mapping of formula symbol -> `params.<name>` or `spec.<dotted.path>`", node_id)
                else:
                    for sym, ref in inputs.items():
                        if not str(ref or "").startswith(("params.", "spec.")):
                            err("params_derivation_input_ref",
                                f"params_derivation[{name!r}] input {sym!r} must reference "
                                f"`params.<name>` or `spec.<dotted.path>`, got {ref!r}", node_id)
            elif kind == "suppress":
                if not entry.get("reason"):
                    err("params_derivation_suppress_reason",
                        f"params_derivation[{name!r}] (suppress) must carry `reason`", node_id)
            else:
                err("params_derivation_kind",
                    f"params_derivation[{name!r}] kind {kind!r} is not one of "
                    "['config_path', 'derived_statistic', 'suppress']", node_id)

            protocol_role = entry.get("protocol_role")
            if protocol_role is None:
                continue
            if kind != "config_path":
                err(
                    "params_derivation_protocol_role_kind",
                    f"params_derivation[{name!r}].protocol_role is valid only "
                    "for kind='config_path' runtime carriers",
                    node_id,
                )
            if protocol_role not in legal_protocol_roles:
                err(
                    "params_derivation_protocol_role",
                    f"params_derivation[{name!r}].protocol_role "
                    f"{protocol_role!r} is not one of "
                    f"{sorted(legal_protocol_roles)!r}",
                    node_id,
                )
                continue
            prior = protocol_roles.get(str(protocol_role))
            if prior is not None:
                err(
                    "params_derivation_protocol_role_duplicate",
                    f"params_derivation entries {prior!r} and {name!r} both "
                    f"claim protocol_role={protocol_role!r}; one scientific "
                    "role cannot bind two runtime carriers",
                    node_id,
                )
            else:
                protocol_roles[str(protocol_role)] = str(name)

    def _lint_offline_demo_data(
        offline: object, params_derivation: object, node_id: str
    ) -> None:
        """Validate the one supported family-owned offline table grammar."""
        if not isinstance(offline, dict):
            err(
                "offline_demo_data_malformed",
                "build_plan.offline_demo_data must be a mapping",
                node_id,
            )
            return
        expected_keys = {
            "schema_id",
            "synthetic_feasible",
            "generator_id",
            "generator_version",
            "entity_count",
            "family_fitting_prefix_steps",
            "supported_cadences",
            "tables",
        }
        if set(offline) != expected_keys:
            err(
                "offline_demo_data_keys",
                "build_plan.offline_demo_data keys must be exactly "
                f"{sorted(expected_keys)!r}",
                node_id,
            )
        expected_scalars = {
            "schema_id": "time_series_panel_v1",
            "synthetic_feasible": True,
            "generator_id": "time_series_offline_fallback",
            "generator_version": "1.0.0",
            "entity_count": 8,
            "family_fitting_prefix_steps": 24,
            "supported_cadences": [
                "week", "day", "hour", "minute", "second"
            ],
        }
        for key, expected in expected_scalars.items():
            if offline.get(key) != expected:
                err(
                    "offline_demo_data_value",
                    f"build_plan.offline_demo_data.{key} must equal "
                    f"{expected!r}",
                    node_id,
                )

        expected_tables = {
            "series.csv": {
                "columns": [
                    {
                        "name": "series_id",
                        "dtype": "string",
                        "semantic_role": "entity_id",
                    },
                    {
                        "name": "static_level",
                        "dtype": "float64",
                        "semantic_role": "static_numeric_level",
                    },
                    {
                        "name": "static_amplitude",
                        "dtype": "float64",
                        "semantic_role": "static_numeric_amplitude",
                    },
                ],
                "primary_key": ["series_id"],
            },
            "observations.csv": {
                "columns": [
                    {
                        "name": "series_id",
                        "dtype": "string",
                        "semantic_role": "entity_id",
                    },
                    {
                        "name": "timestamp",
                        "dtype": "datetime_iso8601_utc",
                        "semantic_role": "time_index",
                    },
                    {
                        "name": "target",
                        "dtype": "float64",
                        "semantic_role": "target",
                    },
                    {
                        "name": "season_sin",
                        "dtype": "float64",
                        "semantic_role": "known_time_varying_covariate",
                    },
                    {
                        "name": "season_cos",
                        "dtype": "float64",
                        "semantic_role": "known_time_varying_covariate",
                    },
                    {
                        "name": "time_fraction",
                        "dtype": "float64",
                        "semantic_role": "known_time_varying_covariate",
                    },
                ],
                "primary_key": ["series_id", "timestamp"],
            },
        }
        if offline.get("tables") != expected_tables:
            err(
                "offline_demo_data_tables",
                "build_plan.offline_demo_data.tables must match the closed "
                "time_series_panel_v1 table and column grammar",
                node_id,
            )

        params = (
            params_derivation
            if isinstance(params_derivation, dict)
            else {}
        )
        horizon_entry = params.get("forecast_horizon")
        horizon = (
            horizon_entry.get("demo_value")
            if isinstance(horizon_entry, dict)
            else None
        )
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or horizon <= 0
        ):
            err(
                "offline_demo_data_forecast_horizon",
                "build_plan.offline_demo_data requires a positive integral "
                "params_derivation.forecast_horizon.demo_value",
                node_id,
            )

    def _lint_node_fields(node_id: str, fields: dict, status: str, legacy: str | None) -> None:
        for check in fields.get("semantic_checks") or []:
            _lint_check(check, node_id, "semantic_checks")
        for bug in fields.get("smoke_bugs") or []:
            _lint_check(bug, node_id, "smoke_bugs")
        pc = fields.get("pluggable_component")
        if pc is not None:
            _lint_pluggable_component(pc, node_id)
        nl = fields.get("notebook_layout")
        if nl is not None:
            _lint_notebook_layout(nl, node_id)
        # `paradigm_extras` placement is not yet ratified (design §7 Q2); accept
        # either the recommended top-level home or nested under smoke_economics.
        pe = fields.get("paradigm_extras") or (fields.get("smoke_economics") or {}).get("paradigm_extras")
        if pe is not None:
            _lint_paradigm_extras(pe, node_id)
        md = fields.get("model_defaults")
        if md is not None:
            _lint_model_defaults(md, node_id)
        pd = fields.get("params_derivation")
        if pd is not None:
            _lint_params_derivation(pd, node_id)
        build_plan = fields.get("build_plan")
        if isinstance(build_plan, dict) and "offline_demo_data" in build_plan:
            _lint_offline_demo_data(
                build_plan["offline_demo_data"], pd, node_id
            )
        ds = fields.get("demo_success")
        if ds is not None:
            _lint_demo_success(ds, node_id)
        demo_skill = fields.get("demo_skill")
        if demo_skill is not None:
            _lint_demo_skill(demo_skill, node_id)
        scenario_dimensions = fields.get("scenario_assumption_dimensions")
        if scenario_dimensions is not None:
            _lint_scenario_assumption_dimensions(
                scenario_dimensions, node_id
            )
        # Retired-key guard (B-11 item 6): the SSOT must refuse
        # scaffold_hints.arch_contract_schema exactly like the pack-authoring
        # validator does. CALL the one existing rule — a copied body would be
        # two authorities over one rule, the R2C-049 diagnosis pattern.
        from validate_paradigm_proposal import (  # noqa: PLC0415
            _check_retired_arch_contract_schema,
        )

        retired_errors: list[str] = []
        _check_retired_arch_contract_schema(fields, retired_errors)
        for message in retired_errors:
            err("retired_arch_contract_schema", message, node_id)
        hints = fields.get("scaffold_hints") or {}
        templates_dir = hints.get("templates_dir")
        if templates_dir and not (taxonomy.repo_root / templates_dir).is_dir():
            err(
                "templates_dir_missing",
                f"scaffold_hints.templates_dir {templates_dir!r} is not an existing directory",
                node_id,
            )
        if templates_dir:
            source = hints.get("demo_data_source")
            if source is None:
                # A warning, not an error (R2C-074): the six pre-existing
                # families are undeclared and keep their behavior until each
                # one is examined on its own evidence. What this makes
                # impossible is the SILENT version — a family promoted out of
                # the gap path taking the gap path's file-only loader with it
                # and losing its dataset acquisition without anything saying so.
                warn(
                    "demo_data_source_undeclared",
                    "scaffold_hints declares templates_dir but not "
                    "demo_data_source, so whether a run of this family can "
                    "obtain its own demo data is inferred rather than stated "
                    "(template_downloads | needs_acquisition)",
                    node_id,
                )
            elif source not in DEMO_DATA_SOURCES:
                err(
                    "demo_data_source_invalid",
                    f"scaffold_hints.demo_data_source {source!r} is not one of "
                    f"{list(DEMO_DATA_SOURCES)}",
                    node_id,
                )
        for dom_key, dom_checks in (fields.get("domain_checks") or {}).items():
            if taxonomy.task_domain(dom_key) is None:
                err("domain_check_unresolved", f"domain_checks key {dom_key!r} is not a task domain or leaf", node_id)
            for check in dom_checks or []:
                _lint_check(check, node_id, "semantic_checks")
        for route in (fields.get("fingerprint") or {}).get("route_elsewhere") or []:
            target = route.get("to") if isinstance(route, dict) else None
            if target and taxonomy.variant(target) is None:
                err("route_elsewhere_orphan", f"route_elsewhere.to {target!r} does not resolve to a variant", node_id)
        if status == "populated":
            if not legacy:
                err("populated_no_legacy", "populated node must declare legacy_paradigm", node_id)
            if not (fields.get("fingerprint") or {}).get("what_it_is"):
                err("populated_no_fingerprint", "populated node must carry fingerprint.what_it_is", node_id)

    def _walk_variant(v: VariantNode) -> None:
        _lint_node_fields(v.taxonomy_id, v.fields, v.status, v.legacy_paradigm)
        for sub in v.sub_variants.values():
            _walk_variant(sub)

    for root in taxonomy.roots.values():
        for family in root.families.values():
            _lint_node_fields(family.id, family.fields, family.status, family.legacy_paradigm)
            for variant in family.variants.values():
                _walk_variant(variant)

    # Top-level universal_checks (Phase 2.1): each entry must satisfy the same
    # semantic_checks shape (id, review stage, severity, status) — they compose into
    # every served node's stage-review via `taxonomy._node_stage_review_focus`.
    seen_universal_ids: set[str] = set()
    for check in taxonomy.universal_checks:
        _lint_check(check, "<universal_checks>", "semantic_checks")
        cid = check.get("id") if isinstance(check, dict) else None
        if isinstance(cid, str):
            if cid in seen_universal_ids:
                err("universal_check_duplicate_id", f"universal_checks has duplicate id {cid!r}", "<universal_checks>")
            seen_universal_ids.add(cid)

    return diags


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lint the taxonomy SSOT and validate classification tuples.")
    parser.add_argument(
        "--ssot",
        default="docs/ssot/taxonomies.yaml",
        help="Path to the SSOT yaml (default: docs/ssot/taxonomies.yaml).",
    )
    parser.add_argument(
        "--check",
        action="append",
        default=[],
        metavar="VARIANT[:DOMAIN]",
        help="Validate a (method_variant, task_domain) tuple; repeatable.",
    )
    args = parser.parse_args(argv)

    taxonomy = load_taxonomy(ROOT, args.ssot)
    diags = lint(taxonomy)
    errors = [d for d in diags if d.severity == "error"]
    warnings = [d for d in diags if d.severity == "warning"]

    for d in warnings:
        loc = f" [{d.node}]" if d.node else ""
        print(f"warning: {d.code}: {d.message}{loc}")
    for d in errors:
        loc = f" [{d.node}]" if d.node else ""
        print(f"error: {d.code}: {d.message}{loc}", file=sys.stderr)

    tuple_errors = 0
    for spec in args.check:
        variant, _, domain = spec.partition(":")
        try:
            v, resolved = assert_classification_tuple(taxonomy, variant, domain or None)
            dom_str = resolved[0] if resolved else "—"
            print(f"ok: {v.taxonomy_id}  x  {dom_str}")
        except TaxonomyError as exc:
            print(f"error: invalid tuple {spec!r}: {exc}", file=sys.stderr)
            tuple_errors += 1

    n_nodes = len(taxonomy.all_variants)
    print(
        f"\ntaxonomy: {len(taxonomy.roots)} roots, {n_nodes} variants, "
        f"{len(taxonomy.task_domains)} task domains, {len(taxonomy.aliases)} aliases — "
        f"{len(errors)} error(s), {len(warnings)} warning(s)."
    )

    # Informational: per-node context-bucket provision (configurable-system view).
    populated = sorted(
        (v for v in taxonomy.all_variants if v.is_populated), key=lambda n: n.taxonomy_id
    )
    if populated:
        print("\nnode context coverage (fields served per bucket):")
        for v in populated:
            cov = bucket_coverage(v)
            cells = "  ".join(
                f"{b}={len(cov[b])}" for b in ("expertise", "implementation", "parameters", "testing")
            )
            print(f"  {v.taxonomy_id}: {cells}")

    return 1 if (errors or tuple_errors) else 0


if __name__ == "__main__":
    raise SystemExit(main())
