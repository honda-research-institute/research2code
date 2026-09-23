"""Post-smoke demo-success verdict (demo-success-semantics design, 2026-07-16).

The reusable failure class: **demo_failure_invisible_to_smoke** — a notebook
executes every cell cleanly while its headline demonstration visibly fails in
its own printed output ("Planning status: timeout", a chance-flat accuracy
series), and no pipeline surface names that. The smoke gate answers "did the
notebook execute"; the researcher asks "did the demo demonstrably work". This
pass owns the second question deterministically:

1. After a clean smoke execution, read the EXECUTED notebook.
2. Locate the family's declared headline demo section (the taxonomy
   `notebook_layout` already declares it; `demo_success.headline_section_id`
   may override the default `running` id). Per the maintainer's 2026-07-16 decision the
   verdict reads the HEADLINE section only — the researcher's first click.
3. When the family declares `demo_skill`, parse its one structured executed
   evaluation record, join it to the split owner's validity receipt, and
   recompute every required comparator.  Printed PASS/FAIL prose is only a
   presentation trace.  Families without that structured contract retain the
   legacy marker/check reading.
4. Record execution, evaluation validity, mechanism evidence, task skill, and
   paper-benchmark status separately.  The legacy three-valued `verdict` is a
   compatibility projection, never the structured authority.

`failed` is NOT a halt — it becomes a first-class delivery demoter
(`delivery_label.py`, reason id `demo_failed`) with a plain-language
disclosure. `undetermined` on a family without markers is an honest
disclosure; on a COMMITTED family it additionally records a kit-coverage
finding we own (the maintainer's 2026-07-16 decision). Probes are unchanged: this
verdict is the cheap always-on floor beneath them, and where both exist and
disagree the delivery derivation records a probe-bug disclosure.

Evaluation order is failure markers, then success markers, then named checks —
a headline demo that prints a failure line does not demonstrably succeed, so
the failure reading wins.

Usage (the driver calls `record_demo_verdict` in-process at stage 3.c):

    python scripts/demo_verdict.py --run-dir r2c_runs/<slug>

Output: `.pipeline/demo_verdict.json` plus a one-line summary on stdout.
Exit 0 whatever the verdict — the verdict is a delivery input, never a gate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

SCHEMA_VERSION = "2.0.0"
DEMO_VERDICT_JSON = "demo_verdict.json"  # under <run>/.pipeline/
TARGET_SCALING_EVIDENCE_SCHEMA_VERSION = "1.0.0"
TARGET_SCALING_PROOF_SCHEMA_VERSION = "1.0.0"
TRAINING_HISTORY_EVIDENCE_SCHEMA_VERSION = "1.0.0"
_TARGET_SCALING_PROOF_KEYS = {
    "schema_version",
    "state_digest",
    "entity_ids",
    "entity_id_root",
    "scaled_actuals",
    "original_actuals",
    "actual_output_role",
    "actual_entity_axis",
    "scaled_predictions",
    "original_predictions",
    "prediction_output_role",
    "prediction_entity_axis",
}
VERDICTS = ("succeeded", "failed", "undetermined")
# The layout section that IS the headline demo unless the marker block
# overrides it — every committed notebook_layout names its end-to-end
# demonstration section `running` (AL §5, MP §5, SO §5).
DEFAULT_HEADLINE_SECTION_ID = "running"
# The layout section that carries the demo DATA setup (every committed
# layout: "## 3. The setup pieces"). The data-signal check executes setup
# sections only, bounded by this section when the layout declares it.
SETUP_PIECES_SECTION_ID = "setup_pieces"
# Exclusive fallback boundary when a layout declares neither a setup_pieces
# nor a headline section: sections 1..3, the shape of every committed layout.
_DEFAULT_SETUP_BOUNDARY = 4

# Structured family evidence (R2C-086).  The prefix is emitted by the
# executed notebook from live objects; its JSON payload, not the notebook's
# later PASS/FAIL prose, is the comparison authority.
from demo_skill_evidence import RECORD_PREFIX as DEMO_EVALUATION_PREFIX


# ---------------------------------------------------------------------------
# Notebook readers (executed-output side; format-matched to probes.universal)
# ---------------------------------------------------------------------------


def cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src or "")


def cell_output_text(cell: dict) -> str:
    """One concatenated text blob for a code cell's executed outputs."""
    parts: list[str] = []
    for out in cell.get("outputs") or []:
        if out.get("output_type") == "stream":
            text = out.get("text", "")
            parts.append("".join(text) if isinstance(text, list) else str(text))
        else:
            data = (out.get("data") or {}).get("text/plain", "")
            parts.append("".join(data) if isinstance(data, list) else str(data))
    return "\n".join(p for p in parts if p)


_HEADING_NUMBER_RE = re.compile(r"^(\d+)\.")
_HEADING_SECTION_RE = re.compile(r"^§\s*(\d+)")


def heading_section_number(line: str) -> int | None:
    """Section number when `line` is a `## N.` / `## §N` markdown heading.

    Stricter than route_findings.cell_index_to_section on purpose: only lines
    that actually start with `#` count, so a numbered LIST item ("1. do this")
    inside a markdown cell can never move the section pointer. Multi-digit
    section numbers ("## 10.") bind too."""
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return None
    text = stripped.lstrip("#").lstrip()
    m = _HEADING_NUMBER_RE.match(text) or _HEADING_SECTION_RE.match(text)
    return int(m.group(1)) if m else None


def _title_section_number(title: str) -> int | None:
    """Section number declared by a notebook_layout section title."""
    return heading_section_number(title.strip())


def headline_section_cells(
    nb: dict, section_number: int
) -> list[tuple[int, dict]]:
    """(index, cell) for every CODE cell inside notebook section N.

    Section membership follows the `## N.` heading convention the layout
    titles declare and the notebook validator enforces; `### N.M` subsection
    headings keep the pointer at N, so the whole headline demo (loop +
    plots + stats) is in scope."""
    current: int | None = None
    out: list[tuple[int, dict]] = []
    for idx, cell in enumerate(nb.get("cells") or []):
        if cell.get("cell_type") == "markdown":
            for line in cell_source(cell).splitlines():
                number = heading_section_number(line)
                if number is not None:
                    current = number
        elif cell.get("cell_type") == "code" and current == section_number:
            out.append((idx, cell))
    return out


def _resolve_headline_section(
    layout: dict, demo_success: dict
) -> tuple[int | None, str, str]:
    """(section_number, section_id, reason-when-unresolvable).

    The notebook layout is the declaration: the section whose id matches
    `demo_success.headline_section_id` (default `running`) names the headline
    demo, and its declared `## N.` title carries the section number the
    executed notebook is matched on."""
    section_id = str(
        demo_success.get("headline_section_id") or DEFAULT_HEADLINE_SECTION_ID)
    sections = layout.get("sections") if isinstance(layout, dict) else None
    if not isinstance(sections, list) or not sections:
        return None, section_id, (
            "the family's notebook_layout declares no sections, so the "
            "headline demo section cannot be located")
    for section in sections:
        if isinstance(section, dict) and section.get("id") == section_id:
            title = str(section.get("title") or "")
            number = _title_section_number(title)
            if number is None:
                return None, section_id, (
                    f"the notebook_layout section {section_id!r} has no "
                    f"`## N.`-numbered title to match executed cells on")
            return number, section_id, ""
    return None, section_id, (
        f"the family's notebook_layout declares no section with id "
        f"{section_id!r}")


# ---------------------------------------------------------------------------
# Marker + check evaluation (pure)
# ---------------------------------------------------------------------------


def _match_markers(
    markers: list, cells: list[tuple[int, dict]]
) -> tuple[str, int, str] | None:
    """First marker match: (quoted evidence line, cell index, gloss).

    Markers are evaluated in declared order; within a marker, cells in
    notebook order. The regex is matched per output so multi-line-aware
    patterns work; the quoted evidence is the matched line."""
    for marker in markers or []:
        if not isinstance(marker, dict) or not marker.get("pattern"):
            continue
        try:
            pattern = re.compile(str(marker["pattern"]))
        except re.error:
            continue  # the SSOT lint rejects these at author time
        for idx, cell in cells:
            text = cell_output_text(cell)
            if not text:
                continue
            m = pattern.search(text)
            if m:
                line = next(
                    (ln for ln in m.group(0).splitlines() if ln.strip()),
                    m.group(0),
                ).strip()
                return line, idx, str(marker.get("gloss") or "")
    return None


def extract_executed_evaluation_record(
    cells: list[tuple[int, dict]],
) -> tuple[dict[str, Any] | None, int | None, str]:
    """Read exactly one structured evaluation JSON line from executed output.

    The record remains untrusted until :func:`bind_split_validity_receipt`
    joins it to the pipeline-owned lineage proof.  Multiple records are
    ambiguous rather than "last one wins"; malformed JSON likewise yields an
    explicit undetermined reason.
    """
    matches: list[tuple[dict[str, Any], int]] = []
    malformed: list[str] = []
    for index, cell in cells:
        for line in cell_output_text(cell).splitlines():
            if not line.startswith(DEMO_EVALUATION_PREFIX):
                continue
            raw = line[len(DEMO_EVALUATION_PREFIX):].strip()
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                malformed.append(f"cell {index}: {exc.msg}")
                continue
            if not isinstance(value, dict):
                malformed.append(f"cell {index}: record is not a JSON object")
                continue
            matches.append((value, index))
    if malformed:
        return None, None, (
            "the structured executed-evaluation line is malformed ("
            + "; ".join(malformed) + ")"
        )
    if not matches:
        return None, None, (
            f"the headline section emitted no {DEMO_EVALUATION_PREFIX.strip()} "
            "structured record"
        )
    if len(matches) != 1:
        return None, None, (
            f"the headline section emitted {len(matches)} structured "
            "evaluation records; exactly one is required"
        )
    record, index = matches[0]
    return record, index, ""


def _beats_chance_verdict(
    nb: dict, cells: list[tuple[int, dict]]
) -> tuple[str, str, int | None] | None:
    """(verdict, evidence, cell index) from the shared beats-chance rule, or
    None when not evaluable here (no headline accuracy series / no class
    count) — the battery's UB-6 then owns the disclosure.

    One rule, three enforcement points (UB-6 battery verdict, the smoke-time
    trainability pre-check, and this headline verdict): the regexes and the
    margin are imported from probes.universal, never restated. Both readings
    need >= 2 metric points: a single matched number is too weak either way —
    one early round-0 print must never demote a delivery, and one config echo
    ("target accuracy: 0.95", which the shared regex matches) must never
    certify a demo as succeeded."""
    try:
        from probes.universal import (  # noqa: PLC0415
            CHANCE_MARGIN, _METRIC_RE, detect_n_classes)
    except ImportError:
        return None
    n_classes = detect_n_classes(nb)
    if not n_classes:
        return None
    series: list[tuple[float, str, int]] = []  # (value, line, cell index)
    for idx, cell in cells:
        for line in cell_output_text(cell).splitlines():
            for m in _METRIC_RE.finditer(line):
                series.append((float(m.group(1)), line.strip(), idx))
    if len(series) < 2:
        return None
    chance = 1.0 / n_classes
    peak_value, peak_line, peak_cell = max(series, key=lambda item: item[0])
    if peak_value >= chance + CHANCE_MARGIN:
        return (
            "succeeded",
            f"{peak_line} (peak {peak_value:.3f} vs chance {chance:.2f})",
            peak_cell,
        )
    return (
        "failed",
        f"{peak_line} (accuracy peaks at {peak_value:.3f} while chance "
        f"is {chance:.2f} for {n_classes} classes)",
        peak_cell,
    )


def derive_demo_verdict(
    nb: dict,
    demo_success: dict,
    layout: dict,
    *,
    paradigm: str = "",
    family_kit_committed: bool = False,
    demo_skill: dict | None = None,
    executed_evaluation: dict | None = None,
    evaluation_validity: dict | None = None,
    evaluation_record_cell: int | None = None,
) -> dict[str, Any]:
    """The pure three-valued verdict over an executed notebook.

    Returns the demo_verdict payload (schema above). Never raises on content:
    every not-decidable path is an `undetermined` with a plain-language
    `reason`, because the verdict is a delivery input and a missing signal is
    a disclosure, never a crash."""

    def _undetermined(
        reason: str, *, markers_declared: bool, kit_gap: bool = False
    ) -> dict[str, Any]:
        payload = _base_payload(paradigm, family_kit_committed)
        payload["verdict"] = "undetermined"
        payload["markers_declared"] = markers_declared
        payload["reason"] = reason
        if family_kit_committed and kit_gap:
            # the maintainer's 2026-07-16 decision: missing markers on a COMMITTED
            # family are a coverage finding we own, not a silent disclosure.
            # A committed family whose markers exist but whose LAYOUT cannot
            # name a headline section is the same class of kit-side gap.
            # Notebook-side binding failures (section absent from the
            # executed notebook, no outputs) stay plain disclosures — those
            # are artifact facts, not kit coverage.
            payload["kit_coverage_finding"] = {
                "id": "demo_markers_missing",
                "family": paradigm,
                "message": (
                    f"coverage gap we own: the committed {paradigm or 'family'} "
                    f"kit cannot machine-check its headline demo outcome "
                    f"({reason}); extend the taxonomy node's demo_success "
                    f"declaration (or its notebook_layout) so the post-smoke "
                    f"verdict can bind"),
            }
        return payload

    if not demo_success:
        return _undetermined(
            f"the {paradigm or 'matched'} family declares no demo "
            f"success/failure markers yet",
            markers_declared=False,
            kit_gap=True,
        )

    number, section_id, section_err = _resolve_headline_section(
        layout, demo_success)
    if number is None:
        return _undetermined(section_err, markers_declared=True, kit_gap=True)

    cells = headline_section_cells(nb, number)
    if not cells:
        return _undetermined(
            f"the executed notebook has no code cells under the headline "
            f"demo section (`## {number}.`, layout id {section_id!r})",
            markers_declared=True,
        )
    if not any(cell_output_text(cell) for _, cell in cells):
        return _undetermined(
            f"the headline demo section (`## {number}.`) has no executed "
            f"outputs to read",
            markers_declared=True,
        )

    payload = _base_payload(paradigm, family_kit_committed)
    payload["markers_declared"] = True
    payload["headline_section_id"] = section_id
    payload["headline_section_number"] = number

    # R2C-086 structured families: marker prose remains a presentation trace,
    # but the pipeline recomputes skill from executed objects and a trusted
    # split-validity receipt.  Non-structured families continue through the
    # marker/check path below unchanged.
    if isinstance(demo_skill, dict) and demo_skill:
        from demo_skill_evidence import derive_demo_skill_evidence  # noqa: PLC0415

        evidence = derive_demo_skill_evidence(
            demo_skill, executed_evaluation, evaluation_validity,
        )
        skill = evidence["skill"]
        skill_status = str(skill.get("status") or "undetermined")
        legacy_projection = {
            "demonstrated": "succeeded",
            "not_demonstrated": "failed",
        }.get(skill_status, "undetermined")
        marker_match = (
            _match_markers(demo_success.get("failure_markers") or [], cells)
            or _match_markers(demo_success.get("success_markers") or [], cells)
        )
        presentation = None
        if marker_match is not None:
            marker_line, marker_cell, marker_gloss = marker_match
            presentation = {
                "line": marker_line,
                "cell": marker_cell,
                "gloss": marker_gloss,
            }
        reasons = skill.get("reasons") or []
        reason = "; ".join(
            str(item.get("message") or item.get("code") or "")
            for item in reasons if isinstance(item, dict)
        ).strip("; ") or None
        payload.update({
            "verdict": legacy_projection,
            "evidence_line": evidence.get("presentation_headline"),
            "evidence_cell": evaluation_record_cell,
            "decided_by": "structured_demo_skill",
            "marker_gloss": (
                "task skill is recomputed from the family contract, exact "
                "executed rows, and pipeline-owned evaluation-validity receipt"
            ),
            "reason": reason,
            "structured_evaluation_present": isinstance(
                executed_evaluation, dict
            ),
            # Portable inputs for the held-out-skill probe (R2C-087).  The
            # manifest projection deliberately omits these row-level values;
            # they remain in the run's pipeline artifact beside the notebook
            # output they came from.
            "demo_skill_contract": demo_skill,
            "executed_evaluation": executed_evaluation,
            "evaluation_validity_receipt": evaluation_validity,
            "evaluation_protocol_role": evidence.get(
                "evaluation_protocol_role"
            ),
            "presentation_marker": presentation,
            "comparisons": evidence.get("comparisons") or [],
            "evidence_status": {
                "execution": evidence["execution"],
                "evaluation_validity": evidence["evaluation_validity"],
                "mechanism": {
                    "status": "undetermined",
                    "reasons": [{
                        "code": "behavioral_probes_pending",
                        "message": (
                            "mechanism evidence is populated only from bound "
                            "behavioral probes after post-smoke evaluation"
                        ),
                    }],
                },
                "skill": evidence["skill"],
                "paper_benchmark": {
                    "status": "not_assessed",
                    "reasons": [{
                        "code": "smoke_scale_not_paper_benchmark",
                        "message": (
                            "demo-scale task skill does not establish the "
                            "paper's benchmark result"
                        ),
                    }],
                },
            },
        })
        return payload

    # Failure markers first: a headline demo that prints a failure line does
    # not demonstrably succeed, whatever else it also prints.
    failure = _match_markers(demo_success.get("failure_markers") or [], cells)
    if failure is not None:
        line, idx, gloss = failure
        payload.update(verdict="failed", evidence_line=line,
                       evidence_cell=idx, decided_by="failure_marker",
                       marker_gloss=gloss)
        return payload

    success = _match_markers(demo_success.get("success_markers") or [], cells)
    if success is not None:
        line, idx, gloss = success
        payload.update(verdict="succeeded", evidence_line=line,
                       evidence_cell=idx, decided_by="success_marker",
                       marker_gloss=gloss)
        return payload

    check_kinds = [
        str(c.get("kind"))
        for c in demo_success.get("checks") or []
        if isinstance(c, dict) and c.get("kind")
    ]
    if "beats_chance" in check_kinds:
        result = _beats_chance_verdict(nb, cells)
        if result is not None:
            verdict, evidence, idx = result
            payload.update(verdict=verdict, evidence_line=evidence,
                           evidence_cell=idx, decided_by="beats_chance",
                           marker_gloss=(
                               "the headline accuracy series is compared "
                               "against chance (1/num_classes) plus the "
                               "shared margin"))
            return payload

    payload["verdict"] = "undetermined"
    payload["reason"] = (
        "markers are declared for this family but none matched the headline "
        "demo section's executed outputs")
    return payload


def _base_payload(paradigm: str, family_kit_committed: bool) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds").replace("+00:00", "Z"),
        "paradigm": paradigm,
        "verdict": "undetermined",
        "evidence_line": None,
        "evidence_cell": None,
        "decided_by": None,
        "marker_gloss": None,
        "markers_declared": False,
        "family_kit_committed": family_kit_committed,
        "kit_coverage_finding": None,
        "reason": None,
    }


def structured_demo_requirement(run_dir: Path) -> tuple[str, bool]:
    """Return ``(paradigm, required)`` for the run's effective taxonomy.

    Historical schema-1 artifacts predate ``demo_skill`` and remain valid
    compatibility inputs.  This query is for *absence*: once a run's effective
    family declares the structured contract, a missing schema-2 artifact can no
    longer be mistaken for a legacy family that never required one.
    """
    run_dir = Path(run_dir)
    try:
        spec = json.loads(
            (run_dir / ".pipeline" / "method_spec.json").read_text(
                encoding="utf-8"
            )
        )
        paradigm = str(
            ((spec.get("comparison") or {}).get("classification") or {})
            .get("id") or ""
        )
    except (OSError, json.JSONDecodeError, AttributeError):
        return "", False
    if not paradigm:
        return "", False

    import taxonomy  # noqa: PLC0415

    try:
        tax = taxonomy.load_taxonomy(
            provisional_packs_dir=taxonomy.run_overlay_dir(run_dir)
        )
        required = bool(taxonomy.load_demo_skill(paradigm, tax))
    except Exception:  # noqa: BLE001 - unknown effective policy must fail closed
        # The spec names a family, but its effective taxonomy could not answer
        # whether structured evidence is required. Treating that infrastructure
        # failure as "no contract" recreates the missing-artifact escape this
        # query exists to close.
        required = True
    return paradigm, required


def record_unresolved_structured_demo_verdict(
    run_dir: Path,
    *,
    paradigm: str,
    error: BaseException,
) -> dict[str, Any]:
    """Persist a fail-closed schema-2 record when the verdict pass crashes.

    Smoke has already completed when the driver calls this path, so execution
    remains a true fact.  The comparison and validity axes stay unresolved;
    claims consumers can therefore suppress the notebook's printed metrics
    without turning a pipeline-owned evidence-pass error into producer output.
    """
    detail = f"{type(error).__name__}: {error}"[:500]
    payload = _base_payload(paradigm, family_kit_committed=False)
    payload.update({
        "verdict": "undetermined",
        "decided_by": "structured_demo_skill_error",
        "reason": (
            "the pipeline-owned structured demo-evidence pass could not "
            f"complete ({detail})"
        ),
        "structured_evaluation_present": False,
        "evidence_status": {
            "execution": {
                "status": "completed",
                "reasons": [],
            },
            "evaluation_validity": {
                "status": "unresolved",
                "reasons": [{
                    "code": "structured_demo_evidence_pass_error",
                    "message": (
                        "the pipeline-owned structured demo-evidence pass "
                        "failed before it could bind evaluation validity"
                    ),
                }],
            },
            "mechanism": {
                "status": "undetermined",
                "reasons": [{
                    "code": "structured_demo_evidence_pass_error",
                    "message": "mechanism evidence was not derived",
                }],
            },
            "skill": {
                "status": "undetermined",
                "reasons": [{
                    "code": "structured_demo_evidence_pass_error",
                    "message": "task skill was not derived",
                }],
            },
            "paper_benchmark": {
                "status": "not_assessed",
                "reasons": [{
                    "code": "smoke_scale_not_paper_benchmark",
                    "message": (
                        "demo-scale execution does not establish the paper's "
                        "benchmark result"
                    ),
                }],
            },
        },
    })
    out_path = Path(run_dir) / ".pipeline" / DEMO_VERDICT_JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _trusted_split_receipt(run_dir: Path, nb: dict) -> dict[str, Any]:
    """Re-run the split owner's analysis and return its structured receipt."""
    try:
        import ast  # noqa: PLC0415

        from scripts.build_plan import load_build_plan  # noqa: PLC0415
        from scripts.eval_split_validation import (  # noqa: PLC0415
            notebook_split_validity_receipt,
        )
        from scripts.taxonomy import run_overlay_dir  # noqa: PLC0415
        from validate_notebook_output import _stitched_code  # noqa: PLC0415

        spec_path = run_dir / ".pipeline" / "method_spec.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if not isinstance(spec, dict):
            raise ValueError("method_spec.json is not an object")
        code_cells = [
            cell_source(cell)
            for cell in nb.get("cells") or []
            if isinstance(cell, dict) and cell.get("cell_type") == "code"
        ]
        tree = _stitched_code(code_cells)
        if not isinstance(tree, ast.Module):
            raise ValueError("executed notebook code cannot be parsed")
        build_plan = load_build_plan(
            spec, ROOT, provisional_packs_dir=run_overlay_dir(run_dir)
        )
        return notebook_split_validity_receipt(
            spec, build_plan, run_dir, tree,
        )
    except Exception as exc:  # noqa: BLE001 - unresolved is the fail-closed path
        return {
            "schema_version": "1.0.0",
            "status": "unresolved",
            "validator": "eval_split_lineage",
            "reasons": ["split_validity_receipt_unavailable"],
            "evidence": {
                "error": f"{type(exc).__name__}: {exc}",
            },
        }


def _effective_build_plan(run_dir: Path) -> dict[str, Any] | None:
    """Load the same spec-derived plan used by pipeline validators."""
    from scripts.build_plan import load_build_plan  # noqa: PLC0415
    from scripts.taxonomy import run_overlay_dir  # noqa: PLC0415

    spec = json.loads(
        (run_dir / ".pipeline" / "method_spec.json").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(spec, dict):
        raise ValueError("method_spec.json is not an object")
    return load_build_plan(
        spec,
        ROOT,
        provisional_packs_dir=run_overlay_dir(run_dir),
    )


def _training_history_applicable(
    run_dir: Path,
    build_plan: Mapping[str, object] | None,
) -> bool:
    """Whether this run has the typed boundary R2C-090 can validate.

    The static forecasting plan also serves legacy/schema-1 packages.  Those
    packages predate the typed training-loop identity carrier, so the plan arm
    alone must not retroactively require evidence they cannot bind.  For
    schema 2, however, the plan arm is authoritative: a missing training loop
    is an invalid producer contract rather than a non-applicable history.
    """
    from scripts.training_history_notebook_flow import (  # noqa: PLC0415
        training_history_notebook_applicable,
    )

    return training_history_notebook_applicable(run_dir, build_plan)


def _executed_output_text(nb: dict) -> str:
    """All executed code-cell output, separated at cell boundaries."""
    return "\n".join(
        text
        for cell in nb.get("cells") or []
        if isinstance(cell, dict) and cell.get("cell_type") == "code"
        for text in [cell_output_text(cell)]
        if text
    )


def _target_scaling_original_unit_proof(
    executed_evaluation: Mapping[str, object] | None,
    state: Mapping[str, object],
) -> dict[str, Any]:
    """Validate one exact nested proof without borrowing flat metric rows."""
    from scripts.target_scaling_post_smoke import (  # noqa: PLC0415
        validate_original_unit_comparison,
    )
    from scripts.time_series_target_scaling import (  # noqa: PLC0415
        TargetScalingProducerError,
    )

    if not isinstance(executed_evaluation, Mapping):
        raise TargetScalingProducerError(
            "target_scaling_original_unit_proof_missing",
            "the executed evaluation must carry target_scaling_proof",
        )
    proof = executed_evaluation.get("target_scaling_proof")
    if not isinstance(proof, Mapping):
        raise TargetScalingProducerError(
            "target_scaling_original_unit_proof_missing",
            "executed_evaluation.target_scaling_proof must be a mapping",
        )
    keys = set(proof)
    if keys != _TARGET_SCALING_PROOF_KEYS:
        raise TargetScalingProducerError(
            "target_scaling_original_unit_proof_shape",
            "executed_evaluation.target_scaling_proof has unsupported shape; "
            f"missing={sorted(_TARGET_SCALING_PROOF_KEYS - keys)}, "
            f"unknown={sorted(keys - _TARGET_SCALING_PROOF_KEYS)}",
        )
    if proof.get("schema_version") != TARGET_SCALING_PROOF_SCHEMA_VERSION:
        raise TargetScalingProducerError(
            "target_scaling_original_unit_proof_schema",
            "executed_evaluation.target_scaling_proof.schema_version must "
            f"equal {TARGET_SCALING_PROOF_SCHEMA_VERSION!r}",
        )
    state_digest = state.get("state_digest")
    if proof.get("state_digest") != state_digest:
        raise TargetScalingProducerError(
            "target_scaling_original_unit_state_disagreement",
            "executed_evaluation.target_scaling_proof.state_digest disagrees "
            "with the validated training-only state",
        )
    return validate_original_unit_comparison(
        state,
        entity_ids=proof["entity_ids"],
        entity_id_root=proof["entity_id_root"],
        scaled_actuals=proof["scaled_actuals"],
        original_actuals=proof["original_actuals"],
        actual_output_role=proof["actual_output_role"],
        actual_entity_axis=proof["actual_entity_axis"],
        scaled_predictions=proof["scaled_predictions"],
        original_predictions=proof["original_predictions"],
        prediction_output_role=proof["prediction_output_role"],
        prediction_entity_axis=proof["prediction_entity_axis"],
    )


def _target_scaling_evidence(
    run_dir: Path,
    nb: dict,
    *,
    build_plan: Mapping[str, object] | None,
    split_receipt: Mapping[str, object] | None,
    executed_evaluation: Mapping[str, object] | None,
) -> dict[str, Any] | None:
    """Validate and persist target-scaling evidence for applicable plans.

    Typed target-scaling failures become data in the demo-verdict artifact so
    responsibility and retry ownership survive the post-smoke adapter.  An
    unrelated I/O or infrastructure failure still raises to the driver's
    existing fail-closed verdict fallback.
    """
    if not isinstance(build_plan, Mapping) \
            or "target_scaling" not in build_plan:
        return None

    from scripts.target_scaling_post_smoke import (  # noqa: PLC0415
        STATE_ARTIFACT,
        persist_post_smoke_target_scaling_state,
        validate_post_smoke_target_scaling_state,
    )
    from scripts.time_series_target_scaling import (  # noqa: PLC0415
        TargetScalingError,
    )

    output_text = _executed_output_text(nb)
    try:
        state = validate_post_smoke_target_scaling_state(
            output_text,
            build_plan=build_plan,
            split_receipt=split_receipt,
        )
        comparison = _target_scaling_original_unit_proof(
            executed_evaluation,
            state,
        )
        # Persistence repeats the pure checks so the write boundary cannot be
        # called independently with evidence weaker than the proof above.
        persisted = persist_post_smoke_target_scaling_state(
            run_dir,
            output_text,
            build_plan=build_plan,
            split_receipt=split_receipt,
        )
    except TargetScalingError as exc:
        status = {
            "producer": "invalid",
            "pipeline": "unsupported",
            "upstream": "upstream_invalid",
        }.get(exc.owner, "unsupported")
        return {
            "schema_version": TARGET_SCALING_EVIDENCE_SCHEMA_VERSION,
            "status": status,
            "responsibility": exc.owner,
            "consume_producer_retry": exc.consume_producer_retry,
            "reason": {
                "code": exc.code,
                "message": str(exc),
            },
        }
    return {
        "schema_version": TARGET_SCALING_EVIDENCE_SCHEMA_VERSION,
        "status": "valid",
        "responsibility": "pipeline_validation",
        "consume_producer_retry": False,
        "state_artifact": STATE_ARTIFACT.as_posix(),
        "state_digest": persisted["state_digest"],
        "original_unit_comparison": comparison,
    }


def _apply_target_scaling_evidence(
    payload: dict[str, Any],
    evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    """Attach scaling evidence and fail closed on non-valid applicable data."""
    if evidence is None:
        return payload
    payload["target_scaling_evidence"] = evidence
    if evidence.get("status") == "valid":
        return payload

    reason = evidence.get("reason")
    if not isinstance(reason, dict):
        reason = {
            "code": "target_scaling_evidence_unresolved",
            "message": "target scaling evidence could not be validated",
        }
    axes = payload.get("evidence_status")
    if not isinstance(axes, dict):
        axes = {}
        payload["evidence_status"] = axes
    evaluation = axes.get("evaluation_validity")
    if not isinstance(evaluation, dict):
        evaluation = {"status": "unresolved", "reasons": []}
        axes["evaluation_validity"] = evaluation
    reasons = evaluation.get("reasons")
    if not isinstance(reasons, list):
        reasons = []
    reasons.append({
        "code": reason.get("code"),
        "message": reason.get("message"),
        "responsibility": evidence.get("responsibility"),
        "consume_producer_retry": evidence.get("consume_producer_retry"),
    })
    evaluation["reasons"] = reasons
    if evidence.get("status") in {"invalid", "upstream_invalid"}:
        evaluation["status"] = "invalid"
    elif evaluation.get("status") != "invalid":
        evaluation["status"] = "unresolved"

    skill = axes.get("skill")
    if not isinstance(skill, dict):
        skill = {"status": "undetermined", "reasons": []}
        axes["skill"] = skill
    skill["status"] = "undetermined"
    skill["reasons"] = [{
        "code": "target_scaling_evidence_not_valid",
        "message": (
            "task skill cannot be concluded until target scaling state and "
            "original-unit outputs are valid"
        ),
    }]
    payload["verdict"] = "undetermined"
    payload["reason"] = str(reason.get("message") or reason.get("code") or "")
    return payload


def _training_history_evidence(
    run_dir: Path,
    nb: dict,
    *,
    build_plan: Mapping[str, object] | None,
    split_receipt: Mapping[str, object] | None,
    executed_evaluation: Mapping[str, object] | None,
) -> dict[str, Any] | None:
    """Persist one structured history for plans that declare the fixed seam."""
    if not _training_history_applicable(run_dir, build_plan):
        return None

    from scripts.time_series_training_history import (  # noqa: PLC0415
        TrainingHistoryError,
    )
    from scripts.training_history_post_smoke import (  # noqa: PLC0415
        TRAINING_HISTORY_ARTIFACT,
        persist_post_smoke_training_history,
    )
    from scripts.training_history_notebook_flow import (  # noqa: PLC0415
        validate_training_history_architecture_contract,
        validate_training_history_notebook_flow,
    )

    scaling_path = run_dir / ".pipeline" / "target_scaling_state.json"
    target_scaling_state: Mapping[str, object] | None = None
    try:
        candidate = json.loads(scaling_path.read_text(encoding="utf-8"))
        if isinstance(candidate, Mapping):
            target_scaling_state = candidate
    except (OSError, json.JSONDecodeError):
        pass

    try:
        validate_training_history_architecture_contract(run_dir, build_plan)
        notebook_flow = validate_training_history_notebook_flow(
            [
                cell_source(cell)
                for cell in nb.get("cells") or []
                if isinstance(cell, dict) and cell.get("cell_type") == "code"
            ],
            build_plan,
        )
        persisted = persist_post_smoke_training_history(
            run_dir,
            _executed_output_text(nb),
            split_receipt=split_receipt,
            target_scaling_state=target_scaling_state,
            executed_evaluation=executed_evaluation,
            # Presence of this schema-2 plan arm declares a real training
            # phase.  The core's not-applicable arm remains available to
            # genuinely non-training families, but cannot satisfy this seam.
            require_recorded=True,
        )
    except TrainingHistoryError as exc:
        status = {
            "producer": "invalid",
            "pipeline": "unsupported",
            "upstream": "upstream_invalid",
        }.get(exc.owner, "unsupported")
        return {
            "schema_version": TRAINING_HISTORY_EVIDENCE_SCHEMA_VERSION,
            "status": status,
            "responsibility": exc.owner,
            "consume_producer_retry": exc.consume_producer_retry,
            "reason": {"code": exc.code, "message": str(exc)},
        }

    result: dict[str, Any] = {
        "schema_version": TRAINING_HISTORY_EVIDENCE_SCHEMA_VERSION,
        "status": "valid",
        "responsibility": "pipeline_validation",
        "consume_producer_retry": False,
        "history_artifact": TRAINING_HISTORY_ARTIFACT.as_posix(),
        "record_digest": persisted["record_digest"],
        "model_id": persisted["model_id"],
        "checkpoint_id": persisted["checkpoint_id"],
        "config_id": persisted["config_id"],
        "target_scaling_state_id": persisted["target_scaling_state_id"],
        "notebook_value_flow": notebook_flow,
    }
    return result


def _apply_training_history_evidence(
    payload: dict[str, Any],
    evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    """Attach training evidence and fail closed when an applicable record fails."""
    if evidence is None:
        return payload
    payload["training_history_evidence"] = evidence
    if evidence.get("status") == "valid":
        return payload

    reason = evidence.get("reason")
    if not isinstance(reason, dict):
        reason = {
            "code": "training_history_evidence_unresolved",
            "message": "structured training history could not be validated",
        }
    axes = payload.get("evidence_status")
    if not isinstance(axes, dict):
        axes = {}
        payload["evidence_status"] = axes
    evaluation = axes.get("evaluation_validity")
    if not isinstance(evaluation, dict):
        evaluation = {"status": "unresolved", "reasons": []}
        axes["evaluation_validity"] = evaluation
    reasons = evaluation.get("reasons")
    if not isinstance(reasons, list):
        reasons = []
    reasons.append({
        "code": reason.get("code"),
        "message": reason.get("message"),
        "responsibility": evidence.get("responsibility"),
        "consume_producer_retry": evidence.get("consume_producer_retry"),
    })
    evaluation["reasons"] = reasons
    if evidence.get("status") in {"invalid", "upstream_invalid"}:
        evaluation["status"] = "invalid"
    elif evaluation.get("status") != "invalid":
        evaluation["status"] = "unresolved"

    skill = axes.get("skill")
    if not isinstance(skill, dict):
        skill = {"status": "undetermined", "reasons": []}
        axes["skill"] = skill
    skill["status"] = "undetermined"
    skill["reasons"] = [{
        "code": "training_history_evidence_not_valid",
        "message": (
            "task skill cannot be concluded until training and selected-model "
            "history identifies the evaluated checkpoint"
        ),
    }]
    payload["verdict"] = "undetermined"
    payload["reason"] = str(reason.get("message") or reason.get("code") or "")
    return payload


def _typed_evaluation_protocol(run_dir: Path) -> dict[str, Any] | None:
    """Validated R2C-082 protocol, or None when exact identity is unavailable.

    Role binding consumes this typed source only.  It does not scan protocol
    prose or infer a role from the executed window length.
    """
    try:
        from schemas.method_spec import EvaluationProtocol  # noqa: PLC0415

        raw = json.loads(
            (run_dir / ".pipeline" / "method_spec.json").read_text(
                encoding="utf-8"
            )
        )
        protocol = (raw.get("comparison") or {}).get("evaluation_protocol")
        return EvaluationProtocol.model_validate(protocol).model_dump(mode="json")
    except Exception:  # noqa: BLE001 - missing/malformed truth is unresolved
        return None


# ---------------------------------------------------------------------------
# Run-dir wrapper (the driver's small call)
# ---------------------------------------------------------------------------


def _paradigm_of_run(run_dir: Path) -> str:
    spec_path = run_dir / ".pipeline" / "method_spec.json"
    if not spec_path.is_file():
        return ""
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        return str(((spec.get("comparison") or {}).get("classification")
                    or {}).get("id") or "")
    except (OSError, json.JSONDecodeError, AttributeError):
        return ""


def run_demo_context(run_dir: Path) -> tuple[str, dict, dict, bool]:
    """(paradigm, demo_success, notebook_layout, family_kit_committed) for a
    run dir, resolved through the run's provisional-pack overlay — the one
    resolution shared by the verdict pass and the data-signal check."""
    import taxonomy  # noqa: PLC0415 - scripts/ path is set at module import

    paradigm = _paradigm_of_run(run_dir)
    tax = taxonomy.load_taxonomy(
        provisional_packs_dir=taxonomy.run_overlay_dir(run_dir))
    node = taxonomy.serves(paradigm, tax) if paradigm else None
    committed = node is not None and not getattr(node, "provisional", False)
    demo_success = taxonomy.load_demo_success(paradigm, tax) if paradigm else {}
    layout = taxonomy.load_notebook_layout(paradigm, tax) if paradigm else {}
    return paradigm, demo_success, layout, committed


def setup_section_boundary(layout: dict, demo_success: dict) -> int:
    """Exclusive upper bound for the notebook's SETUP sections.

    The data-signal check may execute setup cells only, and the data-swap
    digest hashes exactly the same sources — so both resolve the boundary
    from the layout SSOT instead of hardcoding section numbers: through the
    declared `setup_pieces` section when the layout names one, else strictly
    below the declared headline demo section, else the committed layouts'
    common shape (sections 1..3). Always capped at the headline section, so
    demo cells are structurally out of reach even in a compact layout."""
    headline, _, _ = _resolve_headline_section(layout, demo_success)
    setup_number = None
    for section in (layout.get("sections") or []) if isinstance(layout, dict) else []:
        if isinstance(section, dict) and section.get("id") == SETUP_PIECES_SECTION_ID:
            setup_number = _title_section_number(str(section.get("title") or ""))
    if setup_number is not None:
        bound = setup_number + 1
    elif headline is not None:
        bound = headline
    else:
        bound = _DEFAULT_SETUP_BOUNDARY
    if headline is not None:
        bound = min(bound, headline)
    return bound


def setup_section_source_blob(run_dir: Path) -> str | None:
    """Concatenated code-cell SOURCES of the rendered notebook's setup
    sections (1 up to `setup_section_boundary`, §0 install excluded).

    This is the data-relevant slice of the notebook the fix loop hashes to
    detect a demo-data swap: hashing the whole draft would fire the
    data-signal subprocess on every cosmetic notebook fix (review finding 4).
    Returns None when there is no readable rendered notebook — the caller
    then falls back to hashing the whole draft (still swap-safe, just
    coarser)."""
    nb_path = Path(run_dir) / "notebook.ipynb"
    if not nb_path.is_file():
        return None
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    _, demo_success, layout, _ = run_demo_context(Path(run_dir))
    bound = setup_section_boundary(layout, demo_success)
    sources: list[str] = []
    section: int | None = None
    for cell in nb.get("cells") or []:
        if cell.get("cell_type") == "markdown":
            for line in cell_source(cell).splitlines():
                number = heading_section_number(line)
                if number is not None:
                    section = number
        elif (cell.get("cell_type") == "code"
              and section is not None and 1 <= section < bound):
            sources.append(cell_source(cell))
    return "\n# --- cell ---\n".join(sources)


def record_demo_verdict(run_dir: Path) -> dict[str, Any] | None:
    """Compute the verdict for a run dir and persist `.pipeline/demo_verdict.json`.

    Returns the payload, or None when there is no notebook to read. Legacy
    families retain the historical no-artifact behavior; a family declaring
    structured evidence is independently required to produce schema 2 and
    therefore fails closed at delivery if this function cannot write it."""
    run_dir = Path(run_dir)
    nb_path = run_dir / "notebook.ipynb"
    if not nb_path.is_file():
        return None
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    # Gap-path overlay: a provisional pack declares markers on the same
    # surface, and its node resolves only against the run's own overlay view.
    paradigm, demo_success, layout, committed = run_demo_context(run_dir)
    import taxonomy  # noqa: PLC0415
    tax = taxonomy.load_taxonomy(
        provisional_packs_dir=taxonomy.run_overlay_dir(run_dir))
    demo_skill = taxonomy.load_demo_skill(paradigm, tax) if paradigm else {}
    build_plan = _effective_build_plan(run_dir)
    executed_evaluation = None
    record_cell = None
    receipt = None
    split_receipt = None
    scaling_applicable = (
        isinstance(build_plan, Mapping)
        and "target_scaling" in build_plan
    )
    history_applicable = _training_history_applicable(run_dir, build_plan)
    if demo_skill or scaling_applicable or history_applicable:
        split_receipt = _trusted_split_receipt(run_dir, nb)
    if demo_skill:
        number, _, _ = _resolve_headline_section(layout, demo_success)
        cells = headline_section_cells(nb, number) if number is not None else []
        executed_evaluation, record_cell, record_error = (
            extract_executed_evaluation_record(cells)
        )
        from demo_skill_evidence import bind_split_validity_receipt  # noqa: PLC0415
        receipt = bind_split_validity_receipt(
            split_receipt, executed_evaluation,
            evaluation_protocol=_typed_evaluation_protocol(run_dir),
        )
        if executed_evaluation is None and receipt.get("status") == "unresolved":
            receipt["reasons"] = [
                "executed_evaluation_record_missing",
                *(receipt.get("reasons") or []),
            ]
            receipt.setdefault("evidence", {})["record_error"] = record_error

    target_scaling = _target_scaling_evidence(
        run_dir,
        nb,
        build_plan=build_plan,
        split_receipt=split_receipt,
        executed_evaluation=executed_evaluation,
    )
    training_history = _training_history_evidence(
        run_dir,
        nb,
        build_plan=build_plan,
        split_receipt=split_receipt,
        executed_evaluation=executed_evaluation,
    )

    payload = derive_demo_verdict(
        nb, demo_success, layout,
        paradigm=paradigm,
        # A family nobody serves is missing_probe_family territory (the
        # growth-engine demand key), not a kit-coverage finding — the finding
        # is reserved for kits that EXIST and cannot bind.
        family_kit_committed=committed,
        demo_skill=demo_skill,
        executed_evaluation=executed_evaluation,
        evaluation_validity=receipt,
        evaluation_record_cell=record_cell,
    )
    payload = _apply_target_scaling_evidence(payload, target_scaling)
    payload = _apply_training_history_evidence(payload, training_history)
    out_path = run_dir / ".pipeline" / DEMO_VERDICT_JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.run_dir.is_dir():
        print(f"FAIL: run dir not found: {args.run_dir}", file=sys.stderr)
        return 2
    payload = record_demo_verdict(args.run_dir)
    if payload is None:
        print("demo-verdict: no executed notebook.ipynb to read; "
              "nothing recorded")
        return 0
    detail = payload.get("evidence_line") or payload.get("reason") or ""
    print(f"demo-verdict: {payload['verdict']}"
          + (f" — {detail}" if detail else "")
          + f" (.pipeline/{DEMO_VERDICT_JSON})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
