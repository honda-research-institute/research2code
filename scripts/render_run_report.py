"""Assemble the researcher-facing run front door at ``<RUN_DIR>/REPORT.md``.

This module owns the Section 2.2 report wrapper around the CT-3 claims ledger.
The claims renderer remains the trust-critical verdict section; this assembler
adds the surrounding run status, probe summary, stage-category failures, and
issue-source overview.

Stdlib-only so finalization can import it anywhere the driver can run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from generate_method_md import method_md_has_substantive_explanation
from evaluation_protocol_rendering import render_evaluation_protocol_block
from demo_data_provenance import (
    SYNTHETIC_TIER,
    compact_value,
    disclosure_for,
    provenance_path,
    read_provenance,
)
import halt_catalog
from probes.catalog import (
    PROBE_GLOSS,
    failure_reading,
    probe_label,
    probe_record,
)
from render_claims_report import CODE_RE, FINDING_RE, render_claims_report
import run_events
import run_layout


STAGE_CATEGORIES = {
    "stage_0": "setup_analysis",
    "stage_1": "setup_analysis",
    "stage_1x": "setup_analysis",
    "stage_2a": "package_generation",
    "stage_2b": "package_generation",
    "stage_2c": "package_generation",
    "stage_2d": "package_generation",
    "stage_2x": "package_generation",
    "stage_3a": "notebook_smoke",
    "stage_3b": "notebook_smoke",
    "stage_3c": "notebook_smoke",
    "stage_4": "review_routing",
    "stage_5": "review_routing",
}

STAGE_CATEGORY_GLOSS = {
    "setup_analysis": "method understanding",
    "package_generation": "code generation",
    "notebook_smoke": "notebook execution",
    "review_routing": "review and verification",
}

# Plain-language activity per stage for the halt block's headline (spec §3:
# "no stage ids in the headline"). Owned by the halt catalog since queue
# item 12 so the TUI notice and this report share one gloss; re-exported
# here for existing importers.
STAGE_ACTIVITY = halt_catalog.STAGE_ACTIVITY

VERDICT_GLOSS = {
    "pass": "passed",
    "not_applicable": "not applicable",
    "fail": "needs attention",
    "flag_for_researcher": "needs researcher judgment",
    "warn": "advisory",
    "unprobeable": "not checked",
}

# The scrub grammar is owned by render_claims_report (one definition for
# every researcher-facing scrub path). This module keeping its own copy is
# how stage-scoped ids (AL-S1-1 style) stayed bare on the OTHER report
# paths when only this copy was widened (2026-07-22 design review).
_CODE_RE = CODE_RE
_FINDING_RE = FINDING_RE


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _demo_data_provenance_block(run_dir: Path) -> str:
    """Render the data origin without upgrading synthetic data to evidence.

    Synthetic disclosure is fixed copy rather than manifest-supplied prose:
    a malformed or over-enthusiastic ``honesty_note`` must never turn a
    family-owned fixture into purported real or paper-comparable data.
    """

    path = provenance_path(run_dir)
    if not path.is_file():
        acquisition_path = (
            run_dir / ".pipeline" / "dataset_acquisition.json"
        )
        if not acquisition_path.is_file():
            return ""
        acquisition = _read_json(acquisition_path)
        if not isinstance(acquisition, dict):
            return "\n".join([
                "## Demo data provenance",
                "",
                "The pipeline's dataset-acquisition record could not be "
                "read. No claim is made about a demo-data source.",
            ])
        fallback = acquisition.get("offline_fallback")
        fallback = fallback if isinstance(fallback, dict) else None
        recorded_generation = (
            fallback is not None and fallback.get("status") == "generated"
        )
        availability = (
            "The acquisition ledger records a generated family-owned "
            "synthetic fallback, but its required `PROVENANCE.json` is missing. "
            "No complete provenance-bearing demo bundle can be verified, so "
            "no demo-data results should be interpreted."
            if recorded_generation
            else "Neither a paper-cited public bundle nor a supported "
            "family-owned synthetic fallback materialized. No demo-data "
            "results should be interpreted."
        )
        out = [
            "## Demo data provenance",
            "",
            "**Source tier:** No provisioned demo data (`none`).",
            "",
            availability,
            "",
            f"**Provisioning status:** `{acquisition.get('status', 'unknown')}`.",
        ]
        error = acquisition.get("error")
        if isinstance(error, str) and error.strip():
            out.append(f"**Public-tier error:** {error.strip()}")
        if fallback is not None:
            fallback_status = str(fallback.get("status") or "unknown")
            code = str(
                fallback.get("code") or f"offline_fallback_{fallback_status}"
            )
            reason = str(fallback.get("reason") or "no reason recorded")
            if fallback.get("status") == "generated":
                out.append(
                    "**Offline fallback completion:** The acquisition record "
                    f"says `{code}`, but the required `PROVENANCE.json` "
                    "completion marker is missing. Treat the recorded "
                    "generation as incomplete and make no data-result claim."
                )
            else:
                out.append(
                    f"**Offline fallback refusal:** `{code}` — {reason}"
                )
        attempts = acquisition.get("attempts")
        attempts = attempts if isinstance(attempts, list) else []
        rendered_attempts = []
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            host = str(attempt.get("host") or "unresolved source")
            status = str(attempt.get("status") or "unknown")
            reason = str(attempt.get("reason") or "no reason recorded")
            rendered_attempts.append(
                f"- `{host}` — `{status}`: {reason}"
            )
        if rendered_attempts:
            out.extend(["", "**Public-source attempts:**", ""])
            out.extend(rendered_attempts)
        return "\n".join(out)
    manifest = read_provenance(run_dir)
    if manifest is None:
        return "\n".join([
            "## Demo data provenance",
            "",
            "The bundled `PROVENANCE.json` could not be read. No claim is "
            "made about where the demo data came from; inspect the manifest "
            "before interpreting notebook results.",
        ])

    disclosure = disclosure_for(manifest)
    out = [
        "## Demo data provenance",
        "",
        f"**Source tier:** {disclosure.tier_label} "
        f"(`{disclosure.tier}`).",
        "",
        disclosure.origin_statement,
        "",
        f"**Interpretation limit.** {disclosure.honesty_statement}",
    ]

    if disclosure.tier == SYNTHETIC_TIER:
        metadata: list[tuple[str, Any]] = []
        schema = manifest.get("schema_id")
        schema_version = manifest.get("schema_version")
        if schema is not None or schema_version is not None:
            schema_text = str(schema or "unspecified")
            if schema_version is not None:
                schema_text += f" (schema version {schema_version})"
            metadata.append(("Schema", schema_text))
        generator = manifest.get("generator")
        if isinstance(generator, dict):
            metadata.append(("Generator", generator))
        for key, label in (
            ("bundle_digest", "Bundle digest"),
            ("entity_ids", "Stable entity ids"),
            ("cadence", "Cadence"),
            ("protocol_capacity", "Usable protocol capacity"),
            ("generation_mechanics", "Generation mechanics"),
        ):
            value = manifest.get(key)
            if value not in (None, "", [], {}):
                metadata.append((label, value))
        if metadata:
            out.extend(["", "**Recorded generation details:**", ""])
            out.extend(
                f"- **{label}:** `{compact_value(value)}`"
                for label, value in metadata
            )
    return "\n".join(out)


def _escape_cell(text: object) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", " ").strip()


def _scrub_codes(text: object) -> str:
    """Gloss probe codes in researcher-facing messages.

    Codes never appear bare on researcher surfaces (settled design), but
    the glosses are verb phrases written for the report tables' "what
    this means" column, and bare in-place substitution broke grammar
    whenever a message used the code attributively — the 2026-07-05 pdwa
    REPORT rendered "nonadditive composition is equations are checked
    for internal consistency review territory". Substituting the quoted
    noun form 'the "<gloss>" check' stays parseable in every sentence
    position."""
    raw = str(text or "")

    def repl(match: re.Match[str]) -> str:
        gloss = PROBE_GLOSS.get(match.group(0))
        if gloss is None:
            return "this check"
        return f'the "{gloss}" check'

    scrubbed = _CODE_RE.sub(repl, raw)
    return _FINDING_RE.sub("the linked finding", scrubbed)


def _probe_label(probe_id: object) -> str:
    return probe_label(probe_id)


# The full label vocabulary, least to most reliable, for the header's
# scale line (researcher feedback 2026-07: a bare label gave no sense of where the run
# sits overall — he guessed "3 out of 5"). Ordering rationale:
# explanation_only delivers no reliable code; uncertified_new_territory
# passed the universal floor but NOBODY has checked the paper's core
# contribution (unknown coverage); draft has enumerated findings against
# checks that exist and ran; verified passed everything we check.
# Enumerated known issues rank above unknown coverage.
_LABEL_SCALE = ("explanation_only", "uncertified_new_territory",
                "draft", "verified")
_LABEL_SCALE_NAMES = {
    "explanation_only": "explanation only",
    "uncertified_new_territory": "uncertified — new territory",
    "draft": "draft",
    "verified": "verified",
}


def _label_scale_line(delivery: dict[str, Any] | None) -> str:
    label = str((delivery or {}).get("label") or "") if isinstance(
        delivery, dict) else ""
    if label not in _LABEL_SCALE:
        return ""
    rank = _LABEL_SCALE.index(label) + 1
    steps = " → ".join(
        f"**{_LABEL_SCALE_NAMES[step]}**" if step == label
        else _LABEL_SCALE_NAMES[step]
        for step in _LABEL_SCALE)
    return (f"Label scale, least to most reliable: {steps}. "
            f"This run sits at {rank} of {len(_LABEL_SCALE)}.")


def _plain_delivery_label(delivery: dict[str, Any] | None) -> str:
    if not delivery or not isinstance(delivery, dict):
        return "not recorded yet"
    label = str(delivery.get("label") or "unknown")
    if label == "verified":
        return "verified"
    if label == "draft":
        return "draft, needs attention before relying on the output"
    if label == "explanation_only":
        return "explanation only, code was not produced or is not reliable"
    if label == "uncertified_new_territory":
        # Wording guardrail (design note §5): always the full phrase,
        # never bare "uncertified". Both clauses derive from the delivery
        # itself — the detr-distill 2026-07-05 header claimed "every
        # universal check passed, and checks ... don't exist yet" while a
        # universal check could not run and the (existing) KD checks
        # failed to bind, contradicting the report body one screen below.
        from delivery_label import (contribution_gap_clause,  # noqa: PLC0415
                                    universal_floor_clause)
        floor = universal_floor_clause(delivery).rstrip(".")
        floor = floor[0].lower() + floor[1:]
        gap = contribution_gap_clause(delivery).rstrip(".")
        gap = gap[0].lower() + gap[1:]
        return f"uncertified — new territory: {floor}; {gap}"
    return label


def _header(run_dir: Path, delivery: dict[str, Any] | None) -> str:
    slug = run_dir.name
    partial = bool(delivery and delivery.get("partial"))
    # §3.5 criteria 1 + 2: PARTIAL in the first heading, the completeness
    # numbers and the stub names before ANYTHING else about the package.
    out = ["# PARTIAL delivery — Run Report" if partial else "# Run Report",
           ""]
    if partial:
        from partial_delivery import (completeness_counts,  # noqa: PLC0415
                                      completeness_statement,
                                      core_gap_clause, stub_display_lines)
        stubs = delivery.get("stubbed_elements") or []
        counts = completeness_counts(run_dir / ".pipeline", stubs)
        out.extend([
            f"**This package is PARTIAL: {completeness_statement(counts)}.**",
            "",
        ])
        core = core_gap_clause(stubs)
        if core:
            out.extend([f"**{core}**", ""])
        out.append("Stubbed (NOT implemented) components:")
        out.append("")
        out.extend(f"- {line}" for line in stub_display_lines(stubs))
        out.append("")
    out.extend([
        f"Run: `{slug}`",
        "",
        f"Delivery label: **{_plain_delivery_label(delivery)}**.",
        "",
    ])
    scale = _label_scale_line(delivery)
    if scale:
        out.extend([scale, ""])
    out.extend([
        f"Machine-readable manifest: `{run_layout.FINAL_MANIFEST_JSON}`.",
        "",
    ])
    # Route the reader onward when there is a delivered package to open.
    # Halted / explanation-only runs get the "What you still have" block
    # from the halt section instead.
    label = str((delivery or {}).get("label") or "")
    if label in {"verified", "draft", "uncertified_new_territory"}:
        deliverables = []
        if (run_dir / run_layout.NOTEBOOK_IPYNB).is_file():
            deliverables.append(
                f"[`notebook.ipynb`]({run_layout.NOTEBOOK_IPYNB}) — the tutorial")
        if (run_dir / "method").is_dir():
            deliverables.append("`method/` — the importable package")
        method_md = run_dir / run_layout.METHOD_MD
        if method_md.is_file():
            method_state, _ = _method_explanation_state(_read_text(method_md))
            if method_state == "index_only":
                deliverables.append(
                    f"[`METHOD.md`]({run_layout.METHOD_MD}) — decomposition "
                    "source index; substantive method explanation unavailable"
                )
            elif method_state == "empty":
                deliverables.append(
                    f"[`METHOD.md`]({run_layout.METHOD_MD}) — file is empty; "
                    "substantive method explanation unavailable"
                )
            else:
                deliverables.append(
                    f"[`METHOD.md`]({run_layout.METHOD_MD}) — the method explained")
        if deliverables:
            out.extend([
                "After this report, open: " + "; ".join(deliverables) + ".",
                "",
            ])
    return "\n".join(out)


def _scenario_assumptions_block(run_dir: Path) -> str:
    """Render paper-stated demo constraints when Stage 1 captured them.

    Absent or legacy fields produce no section, preserving the report for
    non-declaring families. This block stays descriptive: the runtime
    comparison against the executed setup is the scenario-fidelity checks'
    job (SC-* rows in the verification section, R2C-025 slice B).
    """
    spec = _read_json(run_dir / ".pipeline" / "method_spec.json")
    assumptions = spec.get("scenario_assumptions") if isinstance(
        spec, dict
    ) else None
    if not isinstance(assumptions, dict) or not assumptions:
        return ""

    core = spec.get("core_method")
    core = core if isinstance(core, dict) else {}
    method_name = str(core.get("name") or "the paper's method").strip()
    method_summary = str(core.get("summary") or "").strip()
    description = f"The notebook demonstrates **{method_name}** at smoke scale."
    if method_summary:
        description += f" {method_summary}"

    out = [
        "## Demo setup",
        "",
        description,
        "",
        "### The paper's stated demo assumptions",
        "",
        "These text-grounded statements guide scenario construction and make "
        "the intended setup visible. The scenario-fidelity checks in the "
        "verification section compare each stated assumption against the "
        "demo setup the notebook actually executes.",
        "",
    ]
    for dimension_id, raw_entry in assumptions.items():
        if not isinstance(raw_entry, dict):
            continue
        label = str(dimension_id).replace("_", " ").strip().capitalize()
        value = raw_entry.get("normalized_value")
        normalized = json.dumps(
            value, ensure_ascii=False, sort_keys=True
        )
        location = _escape_cell(raw_entry.get("paper_location")) or (
            "paper location unavailable"
        )
        quote = " ".join(
            str(raw_entry.get("evidence_quote") or "").split()
        )
        out.extend([
            f"- **{label}**",
            f"  - Normalized paper constraint: `{normalized}`",
            f"  - Evidence from {location}: “{quote}”",
        ])
    out.append("")
    return "\n".join(out).strip()


def _evaluation_protocol_block(run_dir: Path) -> str:
    """Render typed paper protocol facts beside the demo's runtime choices."""
    spec = _read_json(run_dir / ".pipeline" / "method_spec.json")
    params = _read_json(run_dir / ".pipeline" / "params.json")
    return render_evaluation_protocol_block(spec, params)


_DEMO_AXIS_LABELS = (
    ("execution", "Execution"),
    ("evaluation_validity", "Evaluation validity"),
    ("graph_construction", "Graph construction"),
    ("graph_alignment", "Graph/entity alignment"),
    ("mechanism", "Mechanism evidence"),
    ("contribution", "Contribution ablation"),
    ("skill", "Demo task skill"),
    ("paper_benchmark", "Paper benchmark reproduction"),
)

_OPTIONAL_GRAPH_AXES = frozenset({
    "graph_construction", "graph_alignment", "contribution",
})

_DEMO_AXIS_DEFAULTS = {
    ("execution", "completed"): "the package ran to completion",
    ("evaluation_validity", "valid"): (
        "the scored rows are bound to a valid held-out evaluation"
    ),
    ("graph_construction", "demonstrated"): (
        "graph parameters and construction semantics agree"
    ),
    ("graph_alignment", "demonstrated"): (
        "entity identity remains aligned through the graph path"
    ),
    ("mechanism", "demonstrated"): (
        "bound behavioral probes demonstrated the declared mechanism"
    ),
    ("contribution", "demonstrated"): (
        "the real mechanism passed a check that its declared null failed"
    ),
    ("skill", "demonstrated"): (
        "the model beat every required family comparator"
    ),
    ("paper_benchmark", "not_assessed"): (
        "demo-scale evidence does not reproduce the paper benchmark"
    ),
}


def _homogeneous_graph_evidence_required(
    run_dir: Path, delivery: dict[str, Any] | None,
) -> bool:
    if isinstance(delivery, dict) and delivery.get(
        "graph_evidence_applicability"
    ) == "homogeneous_graph_v1":
        return True
    spec = _read_json(run_dir / ".pipeline" / "method_spec.json")
    methodology = (
        spec.get("methodology_replication_contract")
        if isinstance(spec, dict) else None
    )
    return (
        isinstance(methodology, dict)
        and isinstance(methodology.get("homogeneous_graph_mechanism"), dict)
    )


def _demo_evidence_block(
    run_dir: Path, delivery: dict[str, Any] | None,
) -> str:
    """Render R2C-086's orthogonal evidence axes as their own surface."""
    candidate = (
        delivery.get("demo_verdict")
        if isinstance(delivery, dict)
        and isinstance(delivery.get("demo_verdict"), dict)
        else None
    )
    if candidate is None:
        stored = _read_json(run_dir / ".pipeline" / "demo_verdict.json")
        candidate = stored if isinstance(stored, dict) else None
    raw_axes = (
        candidate.get("evidence_status") if isinstance(candidate, dict) else None
    )
    graph_required = _homogeneous_graph_evidence_required(run_dir, delivery)
    if not isinstance(raw_axes, dict) and not graph_required:
        return ""
    axes = dict(raw_axes) if isinstance(raw_axes, dict) else {}

    missing_graph_message: str | None = None
    if (
        isinstance(candidate, dict)
        and candidate.get("decided_by") == "delivery_graph_evidence_fallback"
    ):
        missing_graph_message = str(
            candidate.get("reason")
            or "the graph delivery has no complete structured evidence record"
        )
    if graph_required:
        graph_axis_keys = (
            "graph_construction", "graph_alignment", "mechanism", "contribution",
        )
        if not all(isinstance(axes.get(key), dict) for key in graph_axis_keys):
            missing_graph_message = missing_graph_message or (
                "The homogeneous graph contract is present, but its complete "
                "structured graph evidence record is missing."
            )
            missing_reason = [{
                "code": "structured_graph_evidence_missing",
                "message": (
                    "no complete structured graph evidence record was delivered"
                ),
            }]
            for key in graph_axis_keys:
                axes[key] = {
                    "status": "undetermined",
                    "reasons": [dict(missing_reason[0])],
                }
        unresolved_defaults = {
            "execution": "undetermined",
            "evaluation_validity": "unresolved",
            "skill": "undetermined",
            "paper_benchmark": "undetermined",
        }
        for key, status in unresolved_defaults.items():
            if not isinstance(axes.get(key), dict):
                axes[key] = {
                    "status": status,
                    "reasons": [{
                        "code": "structured_demo_evidence_missing",
                        "message": (
                            "no complete structured demo evidence record was delivered"
                        ),
                    }],
                }

    graph_evidence = graph_required or any(
        key in axes for key in _OPTIONAL_GRAPH_AXES
    )

    out = [
        "## Demo evidence",
        "",
    ]
    if missing_graph_message:
        out.extend([missing_graph_message, ""])
    out.extend([
        (
            "These answers stay separate: running successfully is not proof of "
            "a valid evaluation, graph construction, entity alignment, "
            "mechanism liveness, contribution, task skill, or paper-scale "
            "benchmark reproduction."
            if graph_evidence
            else
            "These answers stay separate: running successfully is not proof of "
            "a valid evaluation, mechanism behavior, task skill, or paper-scale "
            "benchmark reproduction."
        ),
        "",
        "| Question | Status | Evidence basis |",
        "|---|---|---|",
    ])
    for key, label in _DEMO_AXIS_LABELS:
        if key in _OPTIONAL_GRAPH_AXES and key not in axes:
            continue
        if key == "mechanism" and graph_evidence:
            label = "Mechanism liveness"
        axis = axes.get(key)
        axis = axis if isinstance(axis, dict) else {}
        status = str(axis.get("status") or "undetermined")
        messages = []
        for reason in axis.get("reasons") or []:
            if isinstance(reason, dict) and reason.get("message"):
                messages.append(str(reason["message"]))
        basis = " ".join(messages) or _DEMO_AXIS_DEFAULTS.get(
            (key, status), "no conclusive evidence was recorded"
        )
        out.append(
            f"| {label} | **{status.replace('_', ' ')}** | "
            f"{_escape_cell(_scrub_codes(basis))} |"
        )
    out.append("")
    return "\n".join(out)


def _history_range_text(value: object) -> str:
    if not isinstance(value, dict):
        return "not recorded"
    return (
        f"`{_escape_cell(value.get('target_root'))}` "
        f"[{value.get('start')}, {value.get('stop')})"
    )


def _training_history_block(run_dir: Path) -> str:
    """Render bounded facts from the validated R2C-090 artifact.

    The full event series stays in the linked JSON.  REPORT.md carries the
    exact identities and ranges plus concise start/best/end statistics, which
    makes the model-selection story inspectable without turning the front door
    into a dump of every epoch.
    """
    rel_path = ".pipeline/training_history.json"
    artifact = run_dir / rel_path
    if not artifact.is_file():
        return ""
    raw = _read_json(artifact)
    try:
        from scripts.time_series_training_history import (  # noqa: PLC0415
            validate_training_history,
        )

        history = validate_training_history(raw)
    except Exception as exc:  # noqa: BLE001 - report the invalid evidence
        return "\n".join([
            "## Training and model selection",
            "",
            "The structured training-history artifact is present but could "
            "not be validated, so this report does not infer a trajectory "
            f"from notebook text ({type(exc).__name__}: {_escape_cell(exc)}).",
            "",
            f"Artifact: [`{rel_path}`]({rel_path})",
        ])

    digest = str(history["record_digest"])
    if history["status"] == "not_applicable":
        return "\n".join([
            "## Training and model selection",
            "",
            "**Not applicable.** " + _escape_cell(history["reason"]),
            "",
            f"Artifact: [`{rel_path}`]({rel_path})  ",
            f"Record digest: `{digest}`",
        ])

    observations = history["loss_observations"]
    values = [float(row["value"]) for row in observations]
    sample_weight = sum(int(row["sample_weight"]) for row in observations)
    selection = history["selection"]
    if selection["status"] == "performed":
        selection_summary = (
            f"performed on {_history_range_text(history['selection_range'])}; "
            f"`{_escape_cell(selection['metric_id'])}` "
            f"({str(selection['direction']).replace('_', ' ')}), "
            f"{len(selection['observations'])} observation(s), selected "
            f"`{_escape_cell(selection['selected_checkpoint_id'])}` by "
            f"`{_escape_cell(selection['tie_break'])}`"
        )
    else:
        selection_summary = (
            "not performed; " + _escape_cell(selection["reason"])
        )

    return "\n".join([
        "## Training and model selection",
        "",
        "This is the pipeline-validated history emitted by the executed "
        "training call; notebook prose is not used as evidence.",
        "",
        "| Field | Recorded evidence |",
        "|---|---|",
        f"| Model | `{_escape_cell(history['model_id'])}` |",
        f"| Evaluated checkpoint | `{_escape_cell(history['checkpoint_id'])}` |",
        f"| Configuration | `{_escape_cell(history['config_id'])}` |",
        f"| Seed | `{history['seed']}` |",
        f"| Target-scaling state | `{_escape_cell(history['target_scaling_state_id'])}` |",
        f"| Fitting targets | {_history_range_text(history['fitting_range'])} |",
        "| Training loss | "
        f"{len(values)} {history['loss_index_kind']}(s); "
        f"start `{values[0]:.6g}`, best `{min(values):.6g}`, "
        f"end `{values[-1]:.6g}`; aggregate sample weight "
        f"`{sample_weight}` |",
        f"| Model selection | {selection_summary} |",
        "",
        f"Artifact: [`{rel_path}`]({rel_path})  ",
        f"Record digest: `{digest}`",
    ])


def _reason_lead(reason: dict[str, Any]) -> str:
    source = str(reason.get("source") or "")
    if source == "probe":
        return f"Behavioral check ({_probe_label(reason.get('id'))})"
    if source == "fidelity_review":
        severity = str(reason.get("severity") or "").strip()
        if severity:
            return f"Paper-fidelity review, {severity} finding"
        return "Paper-fidelity review finding"
    if source == "partial_delivery":
        return "Missing component (partial delivery)"
    if source == "approximated_core":
        return "Approximated core mechanism"
    if source == "demo_verdict":
        if reason.get("id") == "demo_evaluation_invalid":
            return "Demo evaluation validity"
        if reason.get("id") == "demo_skill_not_demonstrated":
            return "Demo task skill"
        return "Headline demo outcome"
    return "Verification finding"


def _proximity_line(reasons: list[dict[str, Any]],
                    probe_counts: dict[str, Any]) -> str:
    """The size-and-shape-of-the-gap sentence (failure-path spec §4).

    The 2026-07-02 GBALD lesson: an honest draft whose report led with the
    label and dense findings, while "you are one small fix away" had to be
    assembled by the reader. State the gap's size first, then the findings.
    When every demoter shares one evidence trace, say so — that is the
    single highest-value sentence in the report."""
    passed = 0
    if isinstance(probe_counts, dict):
        try:
            passed = int(probe_counts.get("pass") or 0)
        except (TypeError, ValueError):
            passed = 0
    everything_else = (
        f" Everything else we checked passed ({passed} checks)."
        if passed else ""
    )
    if len(reasons) == 1:
        return ("One specific finding holds this delivery below `verified`; "
                "the details are below." + everything_else)
    evidence = {str(r.get("evidence") or "").strip() for r in reasons}
    evidence.discard("")
    if len(evidence) == 1 and all(r.get("evidence") for r in reasons):
        where = next(iter(evidence))
        return (f"All {len(reasons)} findings below trace to one place "
                f"(`{where}`) — likely a single underlying defect."
                + everything_else)
    return (f"{len(reasons)} finding(s) hold this delivery below "
            f"`verified`:")


def _new_territory_block(delivery: dict[str, Any],
                         run_dir: Path | None) -> list[str]:
    """The uncertified — new territory rationale (design note §4/§5).

    Always the two-sided truth: what passed (the universal floor) and what
    NOBODY has checked (the contribution) — never phrasing that implies
    the only gap is paperwork. The floor sentence derives from what
    actually ran (delivery_label.universal_floor_clause), so a run whose
    notebook never executed cannot read "every universal check passed".
    The contribution sentence likewise derives from the disclosures
    (delivery_label.contribution_gap_clause): a family whose kit exists
    but could not bind (detr-distill 2026-07-05) must not read "checks
    don't exist yet" above a probe table full of that family's rows."""
    from delivery_label import (contribution_checks_blocked,  # noqa: PLC0415
                                contribution_gap_clause,
                                universal_floor_clause)

    kit_blocked = contribution_checks_blocked(delivery) > 0
    out = [
        f"{universal_floor_clause(delivery)} "
        f"{contribution_gap_clause(delivery)}",
        "",
    ]
    passed_glosses: list[str] = []
    if run_dir is not None:
        report = _read_json(run_dir / ".pipeline" / "probe_report.json")
        verdicts = report.get("verdicts") if isinstance(report, dict) else None
        for v in verdicts or []:
            if isinstance(v, dict) and v.get("verdict") == "pass":
                passed_glosses.append(_probe_label(v.get("probe_id")))
    if passed_glosses:
        out.extend([
            "What WAS checked and passed:",
            "",
        ])
        out.extend(f"- {g}" for g in passed_glosses)
        out.append("")
    if kit_blocked:
        out.extend([
            "What nobody has checked: whether the paper's core mechanism "
            "behaves as claimed. Not because a check failed — the checks "
            "exist for this paper family, but none could bind to this "
            "package's delivered interfaces (each 'not checked' row below "
            "names what could not be synthesized). Treat the "
            "core-contribution code with the same scrutiny you would give "
            "a colleague's first draft.",
            "",
            "Making these checks bind to this package is a fix on our "
            "side, not a new build — the disclosures name exactly where "
            "the checks and the delivered interfaces disagree.",
            "",
        ])
    else:
        out.extend([
            "What nobody has checked: whether the paper's core mechanism "
            "behaves as claimed. Not because a check failed — checks for "
            "this paper family have not been built yet. Treat the "
            "core-contribution code with the same scrutiny you would give "
            "a colleague's first draft.",
            "",
            "Certification for a new family is buildable on request (the "
            "knowledge-distillation checks took one design day).",
            "",
        ])
    return out


def _why_this_label(delivery: dict[str, Any] | None,
                    run_dir: Path | None = None) -> str:
    """The demote rationale, rendered in the report itself.

    bev-distill 2026-07-02: REPORT.md said "draft, needs attention" while
    the five findings that demoted it lived only in final_manifest.json and
    the stage-failure section read all-clear, so a researcher saw a demoted
    label with no visible justification. Every demoting reason (and every
    non-blocking disclosure) renders here, right under the label."""
    if not delivery or not isinstance(delivery, dict):
        return ""
    reasons = [r for r in delivery.get("reasons") or [] if isinstance(r, dict)]
    disclosures = [
        d for d in delivery.get("disclosures") or [] if isinstance(d, dict)]
    label = str(delivery.get("label") or "")
    if label == "uncertified_new_territory":
        out = ["## Why this label", ""]
        out.extend(_new_territory_block(delivery, run_dir))
        if disclosures:
            out.extend([
                f"{len(disclosures)} item(s) are disclosed for awareness "
                f"without blocking the label:",
                "",
            ])
            for disclosure in disclosures:
                out.append(f"- {_scrub_codes(disclosure.get('message'))}")
            out.append("")
        return "\n".join(out).strip()
    if label == "verified" or not (reasons or disclosures):
        return ""
    out = ["## Why this label", ""]
    if reasons:
        out.extend([
            _proximity_line(reasons, delivery.get("probe_counts") or {}),
            "",
        ])
        for reason in reasons:
            out.append(f"- **{_reason_lead(reason)}**: "
                       f"{_scrub_codes(reason.get('message'))}")
        out.append("")
    if disclosures:
        out.extend([
            f"{len(disclosures)} more item(s) are disclosed for awareness "
            f"without blocking the label:",
            "",
        ])
        for disclosure in disclosures:
            out.append(f"- {_scrub_codes(disclosure.get('message'))}")
        out.append("")
    return "\n".join(out).strip()


def _load_claims_section(run_dir: Path) -> str:
    ledger_path = run_dir / ".pipeline" / "claims_ledger.json"
    if not ledger_path.is_file():
        return "\n".join([
            "## Verification - what we could and could not confirm",
            "",
            "The claims ledger was not available, so claim-level verification could not be rendered for this run.",
        ])
    ledger = _read_json(ledger_path)
    if not isinstance(ledger, dict):
        return "\n".join([
            "## Verification - what we could and could not confirm",
            "",
            "The claims ledger could not be read, so claim-level verification could not be rendered for this run.",
        ])
    return render_claims_report(ledger).strip()


def _unique_text(values: list[object]) -> list[str]:
    """Return stable, scrubbed, non-empty text values."""
    out: list[str] = []
    for value in values:
        text = " ".join(_scrub_codes(value).split())
        if text and text not in out:
            out.append(text)
    return out


def _probe_groups(verdicts: list[Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group verdict rows by check id while preserving first-seen order."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in verdicts:
        if not isinstance(item, dict):
            continue
        probe_id = str(item.get("probe_id") or "")
        grouped.setdefault(probe_id, []).append(item)
    return list(grouped.items())


def _group_result_line(items: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for item in items:
        verdict = str(item.get("verdict") or "unknown")
        counts[verdict] = counts.get(verdict, 0) + 1
    return ", ".join(
        f"{count} {VERDICT_GLOSS.get(verdict, verdict)}"
        for verdict, count in counts.items()
    )


def _full_probe_group(probe_id: str, items: list[dict[str, Any]]) -> list[str]:
    """Expanded what/why/evidence/reading treatment for consequential rows."""
    record = probe_record(probe_id) or {}
    out = [
        f"### {_probe_label(probe_id)}",
        "",
        f"**What this checks.** "
        f"{str(record.get('what') or _probe_label(probe_id))}",
        "",
        f"**Why it matters.** "
        f"{str(record.get('why') or 'This check protects a delivery claim.')}",
        "",
        f"**Result.** {_group_result_line(items)}.",
        "",
    ]

    pack_checks = _unique_text([item.get("pack_check") for item in items])
    pack_whys = _unique_text([item.get("pack_why") for item in items])
    if pack_checks:
        out.extend([
            f"**Paper-family focus.** {' '.join(pack_checks)}",
            "",
        ])
    if pack_whys:
        out.extend([
            f"**Why this matters for this family.** {' '.join(pack_whys)}",
            "",
        ])

    messages = _unique_text([item.get("message") for item in items])
    if len(messages) == 1:
        out.extend([f"**Recorded result.** {messages[0]}", ""])
    elif messages:
        out.extend(["**Recorded instances.**", ""])
        out.extend(f"- {message}" for message in messages)
        out.append("")

    evidence: list[str] = []
    for item in items:
        raw = str(item.get("evidence") or "").strip()
        message = str(item.get("message") or "")
        scrubbed = _scrub_codes(raw).strip()
        if scrubbed and raw not in message and scrubbed not in evidence:
            evidence.append(scrubbed)
    if len(evidence) == 1:
        out.extend([f"**Evidence.** {evidence[0]}", ""])
    elif evidence:
        out.extend(["**Evidence.**", ""])
        out.extend(f"- {value}" for value in evidence)
        out.append("")

    readings = _unique_text([
        failure_reading(
            probe_id,
            item.get("verdict"),
            item.get("reason"),
        )
        for item in items
    ])
    if len(readings) == 1:
        out.extend([f"**How to read this.** {readings[0]}", ""])
    elif readings:
        out.extend(["**How to read these results.**", ""])
        out.extend(f"- {reading}" for reading in readings)
        out.append("")
    return out


def _compact_probe_groups(
    title: str,
    introduction: str,
    groups: list[tuple[str, list[dict[str, Any]]]],
) -> list[str]:
    if not groups:
        return []
    out = [f"### {title}", "", introduction, ""]
    for probe_id, items in groups:
        messages = _unique_text([item.get("message") for item in items])
        detail = " ".join(messages) or "No detail was recorded."
        count = f" ({len(items)} instances)" if len(items) > 1 else ""
        out.append(f"- **{_probe_label(probe_id)}{count}.** {detail}")
    out.append("")
    return out


def _probe_summary(run_dir: Path, delivery: dict[str, Any] | None) -> str:
    report = _read_json(run_dir / ".pipeline" / "probe_report.json")
    verdicts = report.get("verdicts") if isinstance(report, dict) else None
    if not isinstance(verdicts, list):
        if delivery and isinstance(delivery, dict):
            reasons = [
                r for r in delivery.get("reasons") or []
                if isinstance(r, dict) and r.get("source") == "battery"
            ]
            if reasons:
                reason = _scrub_codes(reasons[0].get("message", ""))
                return "\n".join([
                    "## Automated checks",
                    "",
                    f"The automated checks did not produce a readable report. {reason}",
                    "",
                ])
        return "\n".join([
            "## Automated checks",
            "",
            "No automated-check report was available for this run.",
            "",
        ])

    counts: dict[str, int] = {}
    for item in verdicts:
        if not isinstance(item, dict):
            continue
        verdict = str(item.get("verdict") or "unknown")
        counts[verdict] = counts.get(verdict, 0) + 1

    count_parts = []
    for key in (
        "pass", "not_applicable", "fail", "flag_for_researcher", "warn",
        "unprobeable",
    ):
        n = counts.get(key, 0)
        if n:
            count_parts.append(f"{n} {VERDICT_GLOSS.get(key, key)}")
    scope_line = (
        "Every delivered package runs a battery of automated behavioral "
        "checks against the generated code. The counts cover all of "
        "them; the details below list attention items separately from "
        "checks that did not apply to the declared method shape."
        if counts.get("not_applicable")
        else
        "Every delivered package runs a battery of automated behavioral "
        "checks against the generated code. The counts cover all of "
        "them; the details below list only the ones that need your "
        "attention."
    )
    out = [
        "## Automated checks",
        "",
        scope_line,
        "",
        "Results: " + (", ".join(count_parts) if count_parts else "none recorded") + ".",
        "",
    ]

    visible = [
        v for v in verdicts
        if isinstance(v, dict)
        and v.get("verdict") in {
            "fail", "flag_for_researcher", "warn", "unprobeable",
            "not_applicable",
        }
    ]
    if not visible:
        out.append("No probe failures, researcher flags, warnings, or unchecked checks were recorded.")
        out.append("")
        return "\n".join(out)

    grouped = _probe_groups(visible)
    expanded = [
        (probe_id, items)
        for probe_id, items in grouped
        if any(
            item.get("verdict") in {"fail", "flag_for_researcher"}
            for item in items
        )
    ]
    expanded_ids = {probe_id for probe_id, _ in expanded}
    unchecked = [
        (
            probe_id,
            [item for item in items
             if item.get("verdict") == "unprobeable"],
        )
        for probe_id, items in grouped
        if probe_id not in expanded_ids
        and any(item.get("verdict") == "unprobeable" for item in items)
    ]
    advisory = [
        (
            probe_id,
            [item for item in items if item.get("verdict") == "warn"],
        )
        for probe_id, items in grouped
        if probe_id not in expanded_ids
        and any(item.get("verdict") == "warn" for item in items)
    ]
    not_applicable = [
        (
            probe_id,
            [item for item in items
             if item.get("verdict") == "not_applicable"],
        )
        for probe_id, items in grouped
        if probe_id not in expanded_ids
        and any(item.get("verdict") == "not_applicable" for item in items)
    ]

    for probe_id, items in expanded:
        out.extend(_full_probe_group(probe_id, items))
    out.extend(_compact_probe_groups(
        "Checks that could not run",
        "These checks did not produce a verdict. Each line names the "
        "artifact, environment capability, or method shape that blocked it.",
        unchecked,
    ))
    out.extend(_compact_probe_groups(
        "Advisory checks",
        "These checks surfaced a risk without failing the delivery.",
        advisory,
    ))
    out.extend(_compact_probe_groups(
        "Checks that did not apply",
        "These conditional checks were outside the declared method shape by "
        "design. They did not fail, and they did not leave expected evidence "
        "missing.",
        not_applicable,
    ))
    return "\n".join(out)


def _rows_from_stage_results(stage_results: list[Any] | None) -> list[dict[str, str]]:
    rows = []
    for result in stage_results or []:
        stage_id = str(getattr(result, "stage_id", "") or "")
        if not stage_id:
            continue
        rows.append({
            "stage_id": stage_id,
            "stage_label": stage_id,
            "stage_category": STAGE_CATEGORIES.get(stage_id, ""),
            "status": str(getattr(result, "status", "") or ""),
            "notes": str(getattr(result, "notes", "") or ""),
        })
    return rows


def _rows_from_driver_state(run_dir: Path) -> list[dict[str, str]]:
    data = _read_json(run_dir / ".pipeline" / "driver_state.json")
    if not isinstance(data, dict):
        return []
    rows = []
    for stage in data.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        stage_id = str(stage.get("stage_id") or "")
        if not stage_id:
            continue
        rows.append({
            "stage_id": stage_id,
            "stage_label": str(stage.get("stage_label") or stage_id),
            "stage_category": STAGE_CATEGORIES.get(stage_id, ""),
            "status": str(stage.get("status") or ""),
            "notes": str(stage.get("notes") or ""),
        })
    return rows


_RESOLVED_FAILURE_EVENTS = frozenset({
    "validation_failed", "smoke_failed", "smoke_trainability_failed",
})


def _read_run_events(run_dir: Path) -> list[dict]:
    path = run_dir / ".pipeline" / "run_events.jsonl"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    events = []
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _describe_catch(event: dict, *, rechecked: bool) -> str:
    """One plain-language line for a failure in a later-completed stage.

    Stage completion is not evidence that the rejected check itself passed.
    ``rechecked`` is therefore derived from an exact later pass event, not
    from the stage's terminal status.
    """
    stage = str(event.get("stage_label") or event.get("stage_id")
                or "a pipeline step")
    raw_details = event.get("details")
    details = raw_details if isinstance(raw_details, dict) else {}
    if event.get("event_type") == "smoke_failed":
        tail = ("a later notebook execution passed the same check, and the "
                "step completed." if rechecked else
                "the step later completed, but no passing re-check of this "
                "execution check was recorded.")
        return f"{stage}: a notebook execution attempt failed; {tail}"
    if event.get("event_type") == "smoke_trainability_failed":
        return (f"{stage}: the notebook executed but showed no learning "
                f"signal; the step later completed, but no passing re-check "
                f"of this learning-signal check was recorded.")
    validator = str(details.get("validator") or "an automated check")
    lines = [ln.strip().lstrip("- ").strip() for ln in
             str(details.get("stderr_tail") or "").strip().splitlines()
             if ln.strip()]
    # Counting boilerplate ("FAIL: 2 validation error(s):") says nothing a
    # researcher can use; the substance is on the next line.
    while lines and re.fullmatch(r"FAIL: \d+ (?:lint |validation )?error\(s\):?",
                                 lines[0]):
        lines.pop(0)
    excerpt = _scrub_codes(lines[0]) if lines else ""
    if len(excerpt) > 140:
        excerpt = excerpt[:140] + "…"
    if excerpt.count("`") % 2:
        # A truncated excerpt with a dangling backtick breaks the markdown
        # of everything after it on the line.
        excerpt = excerpt.replace("`", "'")
    tail = f" ({excerpt})" if excerpt else ""
    resolution = (
        f"a later attempt passed the same `{validator}` check, and the step "
        f"completed."
        if rechecked else
        f"the step later completed, but no passing re-check of the "
        f"`{validator}` check was recorded."
    )
    return (f"{stage}: the `{validator}` check rejected the produced "
            f"output{tail}; {resolution}")


def _has_matching_recheck(failure: dict, window: list[dict]) -> bool:
    """Whether ``window`` records the failed check itself passing.

    The caller limits ``window`` to events before the first later completion
    of this stage. This prevents another validator, a later stage invocation,
    or stage completion alone from being presented as proof of correction.
    """
    stage_id = failure.get("stage_id")
    if not stage_id:
        return False
    event_type = failure.get("event_type")
    if event_type == "validation_failed":
        details = failure.get("details")
        validator = (details.get("validator")
                     if isinstance(details, dict) else None)
        if not validator:
            return False
        return any(
            event.get("event_type") == "validation_passed"
            and event.get("stage_id") == stage_id
            and isinstance(event.get("details"), dict)
            and event["details"].get("validator") == validator
            for event in window
        )
    if event_type == "smoke_failed":
        return any(
            event.get("event_type") == "smoke_passed"
            and event.get("stage_id") == stage_id
            for event in window
        )
    # No smoke_trainability_passed event exists. A smoke_passed event is
    # written before the learning-signal check runs, so it is not proof.
    return False


def _resolved_catches(run_dir: Path) -> list[tuple[str, bool]]:
    """Failure events in the DELIVERING invocation whose stage later
    completed, paired with whether the failed check itself passed before
    that completion. This keeps the problems visible without inferring repair
    from stage status (R2C-077), and prevents the blanket clean sentence from
    shipping over caught failures (RCA 2026-08-03 finding 8). Excluded:
    baseline-accepted validation failures (the degraded row owns them; their
    accepting event lands right after the validator's own failure event) and
    dispatch recoveries (their own disclosure row)."""
    sliced = run_events.current_invocation_slice(_read_run_events(run_dir))
    catches = []
    for i, event in enumerate(sliced):
        if event.get("event_type") not in _RESOLVED_FAILURE_EVENTS:
            continue
        stage_id = event.get("stage_id")
        later_events = sliced[i + 1:]
        completion_offset = next((
            j for j, later in enumerate(later_events)
            if later.get("event_type") == "stage_completed"
            and later.get("stage_id") == stage_id
        ), None)
        if completion_offset is None:
            continue
        window = later_events[:completion_offset + 1]
        follower = next((later for later in later_events
                         if later.get("stage_id") == stage_id), None)
        if (follower is not None and follower.get("event_type")
                == "accepted_validation_failures_only"):
            continue
        rechecked = _has_matching_recheck(event, window)
        catches.append((_describe_catch(event, rechecked=rechecked),
                        rechecked))
    return catches


def _stage_failures(run_dir: Path, stage_results: list[Any] | None) -> str:
    rows = _rows_from_stage_results(stage_results) or _rows_from_driver_state(run_dir)
    failures = [
        r for r in rows
        if r.get("status") in {"halted", "degraded", "missing"}
    ]
    catches = _resolved_catches(run_dir)
    out = ["## Where the run had trouble", ""]
    if not failures and not catches:
        # Both signals clean: terminal stage status AND the event stream.
        out.extend([
            "No method understanding, code generation, notebook execution, or review failures were recorded.",
            "",
        ])
        return "\n".join(out)

    if catches:
        n = len(catches)
        out.extend([
            f"The pipeline's quality gates recorded {n} "
            f"problem{'s' if n != 1 else ''} in its own output before "
            f"delivery:",
            "",
        ])
        out.extend(f"- {line}" for line, _ in catches)
        out.append("")

    if not failures:
        if catches and not all(rechecked for _, rechecked in catches):
            out.extend([
                "No stage remained halted or degraded at delivery, but one "
                "or more failed checks lacked a recorded passing re-check.",
                "",
            ])
        else:
            out.extend([
                "No unresolved failures remained at delivery.",
                "",
            ])
        return "\n".join(out)

    out.extend([
        "Each row is a pipeline step that stopped, was degraded, or went "
        "missing, grouped by what that step was doing.",
        "",
        "| Category | Stage | Status | Notes |", "|---|---|---|---|"])
    for row in failures:
        category = STAGE_CATEGORY_GLOSS.get(row.get("stage_category"), "unknown")
        out.append(
            f"| {_escape_cell(category)} | {_escape_cell(row.get('stage_label') or row.get('stage_id'))} | "
            f"{_escape_cell(row.get('status'))} | {_escape_cell(_scrub_codes(row.get('notes')))} |"
        )
    out.append("")
    return "\n".join(out)


def _count_assumptions(text: str) -> tuple[int, int]:
    titles = re.findall(r"(?m)^## A\d{3}\s+[-\u2014]\s+(.*)$", text)
    needs = sum(1 for title in titles if "NEEDS USER ATTENTION" in title)
    return len(titles), needs


def _count_deferred(text: str) -> int:
    return len(re.findall(r"(?m)^###\s+", text))


def _count_deferred_important(text: str) -> int:
    """Deferred entries at the important tier. The front door names them
    separately because "3 deferred findings" reads as housekeeping, while an
    important-tier finding is a defect a reviewer objected to and nobody
    fixed. The entry heading is written by run_pipeline._ask_user_log_only as
    `### <id> (<severity>, target_agent=<agent>)`.
    """
    return len(re.findall(r"(?m)^###\s+\S+\s+\(important[,)]", text))


def _count_known_issues(text: str) -> int:
    return len(re.findall(r"(?m)^##\s+", text))


def _count_pending_method_explanations(text: str) -> int:
    return len(re.findall(r"\(explanation pending\b", text, flags=re.IGNORECASE))


def _method_explanation_state(text: str) -> tuple[str, int]:
    """Classify METHOD.md without conflating zero pending slots with content."""
    if not text.strip():
        return "empty", 0
    pending = _count_pending_method_explanations(text)
    if pending:
        return "pending", pending
    if not method_md_has_substantive_explanation(text):
        return "index_only", 0
    return "substantive", 0


def _not_checkable_disclosure(run_dir: Path) -> str:
    """Item 3 disclosure: what the math-sanity pass could NOT vouch for —
    claims it could not machine-check, and (the ADAM 2026-07-14 blind spot)
    explanation entries whose math carried no checkable claims at all, so
    'math checks passed' is honest about its coverage. Disclosure only —
    never a demotion — read from the explanation sidecar's `math_sanity`
    totals. Empty string when the pass did not run or covered everything."""
    sidecar = _read_json(run_dir / ".pipeline" / "method_explanations.json")
    totals = sidecar.get("math_sanity") if isinstance(sidecar, dict) else None
    if not isinstance(totals, dict):
        return ""
    parts = []
    n = totals.get("not_checkable")
    if isinstance(n, int) and n > 0:
        parts.append(f"{n} mechanism claim{'s were' if n != 1 else ' was'} "
                     f"not machine-checkable")
    claim_free = totals.get("entries_claim_free")
    entries_total = totals.get("entries_total")
    if isinstance(claim_free, int) and claim_free > 0:
        of_total = (f" of {entries_total}"
                    if isinstance(entries_total, int) and entries_total > 0
                    else "")
        parts.append(
            f"the math checks had nothing to verify in {claim_free}"
            f"{of_total} explanation section"
            f"{'s' if claim_free != 1 else ''}")
    if not parts:
        return ""
    return f"; {'; '.join(parts)} (disclosure only, not a defect)"


def _count_recovered_dispatches(run_dir: Path) -> int:
    """Recovered transport failures (recovery-ladder rungs 1 and 3) during
    the current (most recent) driver invocation: counted from run events so
    an unattended run's operator sees the flap without reading the event log.

    Scoped to the last `run_started` boundary because the event log is
    append-only across resume legs — counting the whole file presented an
    earlier dead attempt's recovery as part of the current delivery (SRL
    2026-07-15). Earlier legs' recoveries remain in the event log and run
    history; this row describes the delivering attempt. A log with no
    boundary falls back to the whole-file count (legacy conservatism).
    Unreadable events → 0 (disclosure only)."""
    path = run_dir / ".pipeline" / "run_events.jsonl"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return 0
    events = []
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return sum(
        1 for event in run_events.current_invocation_slice(events)
        if event.get("event_type") == "agent_dispatch_recovered"
    )


def _issue_summary(run_dir: Path) -> str:
    rows = []

    method_md = run_dir / "METHOD.md"
    method_text = _read_text(method_md)
    if method_md.is_file():
        method_state, pending = _method_explanation_state(method_text)
        if method_state == "pending":
            status = (
                f"{pending} pending explanation{'s' if pending != 1 else ''}; "
                "run should be treated as degraded until they are resolved"
            )
        elif method_state == "index_only":
            status = (
                "decomposition source index produced; substantive method "
                "explanation unavailable"
            )
        elif method_state == "empty":
            status = (
                "file is empty; substantive method explanation unavailable"
            )
        else:
            status = "produced; no pending explanation markers found"
        status += _not_checkable_disclosure(run_dir)
    else:
        status = "not produced"
    rows.append(("Method explanation", "METHOD.md", status))

    assumptions = run_layout.resolve_existing(run_dir, run_layout.ASSUMPTIONS_MD)
    assumptions_text = _read_text(assumptions) if assumptions else None
    if assumptions_text:
        total, needs = _count_assumptions(assumptions_text)
        if total:
            status = f"{total} assumption entr{'y' if total == 1 else 'ies'}"
            if needs:
                status += f"; {needs} need{'s' if needs == 1 else ''} researcher attention"
        else:
            status = "file present; no assumption entries found"
    else:
        status = "not produced"
    rows.append(("Automatic assumptions", run_layout.ASSUMPTIONS_MD, status))

    deferred_rel = run_layout.DEFERRED_FINDINGS_MD
    deferred_path = run_layout.resolve_existing(run_dir, deferred_rel)
    deferred_text = _read_text(deferred_path) if deferred_path else None
    if deferred_path is not None:
        deferred_rel = deferred_path.relative_to(run_dir).as_posix()
    if deferred_text:
        n = _count_deferred(deferred_text)
        status = f"{n} deferred finding{'s' if n != 1 else ''}" if n else "file present; no deferred findings found"
        n_important = _count_deferred_important(deferred_text)
        if n_important:
            status += (
                f"; {n_important} at the important tier, unresolved — "
                f"read {'these' if n_important != 1 else 'it'} before "
                f"relying on the affected code"
            )
    else:
        status = "not produced"
    rows.append(("Deferred review findings", deferred_rel, status))

    known = run_layout.resolve_existing(run_dir, run_layout.KNOWN_ISSUES_MD)
    known_text = _read_text(known) if known else None
    if known_text:
        n = _count_known_issues(known_text)
        status = f"{n} known issue{'s' if n != 1 else ''}" if n else "file present; no known issue entries found"
    else:
        status = "not produced"
    rows.append(("Known delivery limits", run_layout.KNOWN_ISSUES_MD, status))

    element_tests_path = run_dir / ".pipeline" / "element_tests.json"
    if element_tests_path.is_file():
        try:
            et = json.loads(element_tests_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            et = None
        if et is not None:
            et_statuses = et.get("statuses", {})
            tested = sum(1 for s in et_statuses.values()
                         if s.get("status") == "tested")
            weak = sum(1 for s in et_statuses.values()
                       if s.get("status") == "weak")
            eligible = len(et.get("eligible", []))
            dropped = len(et.get("dropped", {}))
            status = (
                f"{tested} of {eligible} eligible element"
                f"{'s' if eligible != 1 else ''} covered by a per-element "
                "test shown to distinguish correct from broken code"
            )
            if weak:
                status += (
                    f"; {weak} weak test{'s' if weak != 1 else ''} that "
                    "could not demonstrate that distinction"
                )
            if dropped:
                status += (
                    f"; {dropped} generated test{'s' if dropped != 1 else ''} "
                    "could not be verified and did not ship"
                )
            rows.append(
                ("Per-element tests", "method/tests/README.md", status))

    adoptions_path = run_dir / ".pipeline" / "quote_adoptions.json"
    if adoptions_path.is_file():
        try:
            adoptions = json.loads(
                adoptions_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            adoptions = None
        if adoptions:
            listed = "; ".join(
                f"`{a.get('element_id')}` at similarity "
                f"{a.get('similarity')}" for a in adoptions)
            # Two adoption surfaces share this sidecar: equation quotes
            # (no surface key) and method-spec quotes (surface="method_spec").
            spec_n = sum(1 for a in adoptions
                         if a.get("surface") == "method_spec")
            eq_n = len(adoptions) - spec_n
            kinds = []
            if eq_n:
                kinds.append(
                    f"{eq_n} equation quote{'s' if eq_n != 1 else ''}")
            if spec_n:
                kinds.append(
                    f"{spec_n} method-spec quote{'s' if spec_n != 1 else ''}")
            rows.append((
                "Adopted paper passages",
                ".pipeline/quote_adoptions.json",
                f"{' and '.join(kinds)} replaced "
                f"with the paper's own bytes after the producer's "
                f"transcription failed the verbatim floor: {listed}. "
                f"Each adoption has an assumptions.md entry with the "
                f"original quote preserved",
            ))

    recovered = _count_recovered_dispatches(run_dir)
    if recovered:
        rows.append((
            "Model-call recoveries",
            ".pipeline/run_events.jsonl",
            f"{recovered} model call{'s' if recovered != 1 else ''} failed "
            f"in transport (hang or outage) and {'were' if recovered != 1 else 'was'} "
            f"recovered by an automatic retry — disclosure only; the "
            f"failure{'s stay' if recovered != 1 else ' stays'} on the "
            f"run's dispatch record",
        ))

    out = [
        "## Issues To Review",
        "",
        "These rows consolidate the issue files. The source files remain available for the full details.",
        "",
        "| Source | File | Summary |",
        "|---|---|---|",
    ]
    for source, rel_path, status in rows:
        out.append(f"| {source} | `{rel_path}` | {_escape_cell(status)} |")
    out.append("")
    return "\n".join(out)


def _halt_block(run_dir: Path, stage_results: list[Any] | None) -> str:
    """The stopped-run block (failure-path spec §3): four mandatory parts in
    plain language, rendered at the top of REPORT.md when the run halted.

    Gated on a halted stage in the SAME rows _stage_failures reads, so a
    resumed run that later delivers never renders a stale block off a
    leftover halt artifact. The halt artifact adds the researcher-actionable
    message when the halt recorded one; the technical reason stays visible
    but below the plain-language story. The plain-language rewrite of
    engineering reasons is deterministic wording for now — the spec's
    LLM rewrite is the deferred piece."""
    rows = _rows_from_stage_results(stage_results) or _rows_from_driver_state(run_dir)
    halted = [r for r in rows if r.get("status") == "halted"]
    if not halted:
        return ""
    stage_id = halted[-1]["stage_id"]
    artifact = _read_json(run_dir / ".pipeline" / f"{stage_id}.halt")
    artifact = artifact if isinstance(artifact, dict) else {}
    user_message = str(artifact.get("user_message") or "").strip()
    reason = str(artifact.get("reason") or halted[-1].get("notes") or "").strip()
    terminal_action = halt_catalog.terminal_gap_action(
        artifact.get("halt_class"), artifact.get("context")
    )
    activity = STAGE_ACTIVITY.get(stage_id, "processing this paper")
    category = STAGE_CATEGORY_GLOSS.get(
        STAGE_CATEGORIES.get(stage_id, ""), "pipeline processing")
    # Halt catalog (queue item 12): a classified halt renders the catalog's
    # story — the same one the TUI notice showed, so the two surfaces never
    # tell different stories. A terminal structured gap action wins over stale
    # hand-set prose; otherwise user_message wins, and artifacts with neither
    # render the stock wording unchanged.
    story = None
    if terminal_action or not user_message:
        evidence = artifact.get("evidence")
        story = halt_catalog.render_halt_story(
            artifact.get("halt_class")
            or ("paradigm_mismatch" if terminal_action else None),
            stage_id=stage_id,
            evidence=evidence if isinstance(evidence, dict) else None)

    out = ["## This run stopped — it needs a decision", ""]
    if story is not None:
        out.append(f"**What happened.** {story.what_happened} "
                   f"It did not crash: stopping was the deliberate choice.")
    else:
        out.append(f"**What happened.** The run stopped while {activity}. "
                   f"It did not crash: stopping was the deliberate choice.")
    out.append("")
    if terminal_action and story is not None:
        out.append(f"**Why we stopped instead of guessing.** "
                   f"{story.why_stopped} (Failure area: {category}.)")
    elif user_message:
        out.append(f"**Why we stopped instead of guessing.** {user_message} "
                   f"(Failure area: {category}.)")
    elif story is not None:
        out.append(f"**Why we stopped instead of guessing.** "
                   f"{story.why_stopped} (Failure area: {category}.)")
    else:
        out.append(
            f"**Why we stopped instead of guessing.** The run hit an internal "
            f"error that automatic recovery could not resolve, and guessing "
            f"past it would risk delivering wrong code silently. "
            f"(Failure area: {category}.)")
    out.append("")
    have = []
    method_md = run_dir / "METHOD.md"
    if method_md.is_file():
        method_state, _ = _method_explanation_state(_read_text(method_md))
        if method_state == "index_only":
            have.append("[`METHOD.md`](METHOD.md) — a decomposition source "
                        "index; a substantive plain-language method "
                        "explanation was not produced")
        elif method_state == "empty":
            have.append("[`METHOD.md`](METHOD.md) — the file is empty; a "
                        "substantive plain-language method explanation was "
                        "not produced")
        else:
            have.append("[`METHOD.md`](METHOD.md) — the plain-language "
                        "explanation of the paper's method")
    if (run_dir / ".pipeline" / "paper_map.json").is_file():
        have.append("`.pipeline/paper_map.json` — the structured map of the "
                    "paper's sections and claims")
    if (run_dir / "notebook.ipynb").is_file():
        have.append("[`notebook.ipynb`](notebook.ipynb) — the demo notebook "
                    "as of the stop (unverified)")
    if (run_dir / "method").is_dir() and any((run_dir / "method").rglob("*.py")):
        # Gated on actual code: a run that stops before scaffolding leaves
        # method/ holding only a README banner, and calling that "generated
        # code" misleads the reader (iDb-RRT 2026-07-13, stage-1 stop).
        have.append("`method/` — generated code as of the stop; treat it as "
                    "unreliable until the stop is resolved")
    if (run_dir / run_layout.ASSUMPTIONS_MD).is_file():
        have.append(f"[`assumptions.md`]({run_layout.ASSUMPTIONS_MD}) — every judgment "
                    "call made automatically so far")
    out.append("**What you still have.** This report stays the front door; "
               "the understanding tier ships regardless of the stop:")
    out.append("")
    out.extend(f"- {item}" for item in have)
    if not have:
        out.append("- this report (the stop fired before other artifacts "
                   "were produced)")
    out.append("")
    slug = run_dir.name
    if terminal_action:
        next_step = (
            f"{terminal_action} This is a terminal scope decision for the "
            "current input; do not resume the unchanged run. To ask about "
            f"the evidence first, use `/r2c-chat {slug}`."
        )
    elif user_message:
        next_step = ("Do what the message above asks, then resume the run: "
                     f"`/r2c-run {slug}` in the opencode window. Completed "
                     "stages are never redone.")
    elif story is not None:
        next_step = (f"{story.what_next} To resume: `/r2c-run {slug}` in "
                     f"the opencode window. To understand what happened "
                     f"first, ask the run chat: `/r2c-chat {slug}` answers "
                     "questions about this run from its own records.")
    else:
        next_step = ("Resume the run first — most stops like this are "
                     f"transient: `/r2c-run {slug}` in the opencode window. "
                     "If it stops at the same place twice, report it with "
                     "the technical detail below attached. To understand "
                     f"what happened first, ask the run chat: `/r2c-chat "
                     f"{slug}` answers questions about this run from its "
                     "own records.")
    out.append(f"**What to do next.** {next_step}")
    out.append("")
    if reason:
        out.extend([
            "<details>",
            "<summary>Technical detail (for engineering)</summary>",
            "",
            f"**Stopped at:** `{stage_id}`. **Recorded reason:** "
            f"{_scrub_codes(reason)}",
            "",
            f"**Halt artifact:** `.pipeline/{stage_id}.halt`",
            "",
            "</details>",
        ])
    return "\n".join(out).strip()


def render_run_report(
    run_dir: Path,
    *,
    delivery: dict[str, Any] | None = None,
    stage_results: list[Any] | None = None,
) -> str:
    sections = [
        _header(run_dir, delivery).strip(),
        _halt_block(run_dir, stage_results),
        _scenario_assumptions_block(run_dir),
        _evaluation_protocol_block(run_dir),
        _demo_data_provenance_block(run_dir),
        _demo_evidence_block(run_dir, delivery),
        _training_history_block(run_dir),
        _why_this_label(delivery, run_dir),
        _load_claims_section(run_dir),
        _probe_summary(run_dir, delivery).strip(),
        _stage_failures(run_dir, stage_results).strip(),
        _issue_summary(run_dir).strip(),
    ]
    return "\n\n".join(section for section in sections if section) + "\n"


def write_run_report(
    run_dir,
    *,
    delivery: dict[str, Any] | None = None,
    stage_results: list[Any] | None = None,
) -> bool:
    run_dir = Path(run_dir)
    (run_dir / "REPORT.md").write_text(
        render_run_report(run_dir, delivery=delivery, stage_results=stage_results),
        encoding="utf-8",
    )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    # Standalone re-render (archived/curated runs): the driver isn't here to
    # pass the live delivery verdict, so recover it from the stored manifest.
    delivery = None
    manifest_path = run_layout.resolve_existing(run_dir, run_layout.FINAL_MANIFEST_JSON)
    if manifest_path is not None:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(manifest.get("delivery"), dict):
                delivery = manifest["delivery"]
        except (OSError, ValueError):
            delivery = None
    write_run_report(run_dir, delivery=delivery)
    print(f"wrote {run_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
