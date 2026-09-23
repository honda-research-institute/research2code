"""One-command probe battery over a finished (or halted) run dir.

Consolidates the recentering verification tiers built so far into a single
audit command — the same battery that was run ad hoc for the pdwa matrix row:

    python3 scripts/run_probes.py --run-dir r2c_runs/<slug>

- US-10  lint gate over method/*.py + notebook_draft.py
- US-1/2/3/3b  params provenance probes (against the run's own paper.md)
- CT-3   claims ledger from paper_map (every experiment claim explicitly
         statused; writes .pipeline/claims_ledger.json)
- UB-6   executed-notebook output sanity
- UB-4/UB-7  objective direction-consistency + term-ranking power
         (motion_planning method packages, additive-chain analysis)
- UB-7   composite term-channel perturbation (active_learning selectors)
- CT-1   contribution floor: selector vs degenerate nulls (active_learning)
- MP-1   scenario-dynamics (motion_planning runs)
- MP-3/MP-4  steering-responsiveness + goal-progress on the method package's
         planner (motion_planning runs)
- AL-1/2/4 + AL-5 + UB-5  acquisition-loop microharness, selector contract
         (incl. the crash arm: a selector that crashes inside its own code
         on contract-conformant pools is a finding, never silently
         unprobeable), trains-on-synthetic (active_learning runs, torch
         needed)
- AL-6   demo-config reachability: the delivered params must let the
         acquisition loop actually run (the empty-pool no-op-demo class)

Paradigm and pluggable-component name come from the run's method_spec
(probe targets are never guessed — the 2026-06-10 lesson). Missing artifacts
become `unprobeable` verdicts, never crashes and never silent passes.

Output: human summary on stdout + a chat-readable verdict report at
<run>/.pipeline/probe_report.json (override with --report). Exit 1 when any
gating failure exists, else 0 — wiring-ready for the driver and for CI use
on preserved fixtures.

Isolated output mode (--output-dir, R2C-019): redirects BOTH the verdict
report and the CT-3 claims ledger into a caller-selected directory and
leaves the run dir byte-identical. This is the portable-harness /
post-delivery re-validation contract: delivery-time truth under
<run>/.pipeline/ is frozen evidence, and a researcher (or the run
companion's post-edit gate) re-running the battery must produce a
SEPARATE result to diff against the baseline, never overwrite it. The
default (no --output-dir) behavior is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from probes import ProbeReport, ProbeVerdict
from probes.catalogs.run_probes import PROBE_CATALOG as _PROBE_CATALOG
from probes.narrative import probe_narrative_vs_params
from probes.universal import probe_executed_notebook, probe_loss_descent

from lint_generated_code import default_targets, lint_paths
from validate_params_provenance import check_params

try:  # Repo execution versus the vendored flat harness layout.
    from graph_callable_liveness import (  # type: ignore[import-not-found]
        ALLOWED_GRAPH_CALLABLE_MODULES,
        assess_liveness_receipt,
    )
except ImportError:  # pragma: no cover - exercised by package-style imports
    from scripts.graph_callable_liveness import (
        ALLOWED_GRAPH_CALLABLE_MODULES,
        assess_liveness_receipt,
    )

PROBE_CATALOG = _PROBE_CATALOG

_TRAINING_HISTORY_FIXED_AUTHORITY_PATHS = (
    "requirements.txt",
    "notebook.ipynb",
    ".pipeline/method_spec.json",
    ".pipeline/arch_contract.json",
    ".pipeline/params.json",
    ".pipeline/target_scaling_state.json",
    ".pipeline/demo_verdict.json",
    ".pipeline/training_history.json",
)

_GRAPH_ALIGNMENT_PROBE_REF = "graph_mechanism.alignment_prerequisite"
_GRAPH_PROBE_ROWS = (
    ("HG-1", _GRAPH_ALIGNMENT_PROBE_REF),
    ("HG-2", "graph_mechanism.parameter_agreement"),
    ("HG-3", "graph_mechanism.construction_semantics"),
    ("HG-4", "graph_mechanism.topology_sensitivity"),
    ("HG-5", "graph_mechanism.neighbor_sensitivity"),
    ("HG-6", "graph_mechanism.permutation_equivalence"),
    ("HG-7", "graph_mechanism.contribution_ablation"),
)
_GRAPH_PROBE_REFS = frozenset(ref for _, ref in _GRAPH_PROBE_ROWS)
_GRAPH_RECEIPT_VERSION = "1.0"
_GRAPH_ALIGNMENT_VALIDATOR = "validate_arch_contract_runtime.py"
_GRAPH_RECEIPT_KEYS = frozenset({
    "receipt_version",
    "probe_ref",
    "status",
    "reason",
    "verified",
    "validator",
    "element_id",
    "fixture_scope",
    "representation",
    "entity_ids",
    "authority_digests",
})


def _training_history_authority_digests(
    run_dir: Path,
) -> dict[str, str | None]:
    """Hash the closed delivery-time authority behind structured history."""
    root = Path(run_dir)
    relatives = set(_TRAINING_HISTORY_FIXED_AUTHORITY_PATHS)
    method = root / "method"
    if method.is_dir():
        relatives.update(
            path.relative_to(root).as_posix()
            for path in method.rglob("*.py")
            if path.is_file()
            and "__pycache__" not in path.relative_to(method).parts
            and path.suffix != ".pyc"
        )
    example_data = method / "example_data"
    if example_data.is_dir():
        relatives.update(
            path.relative_to(root).as_posix()
            for path in example_data.rglob("*")
            if path.is_file() and path.name != "README.md"
        )
    result: dict[str, str | None] = {}
    for relative in sorted(relatives):
        try:
            result[relative] = hashlib.sha256(
                (root / relative).read_bytes()
            ).hexdigest()
        except OSError:
            result[relative] = None
    return result


def _lint_tier(run_dir: Path, report: ProbeReport) -> None:
    targets = default_targets(run_dir)
    if not targets:
        report.add(ProbeVerdict("US-10", "unprobeable",
                                "no method/*.py or notebook_draft.py found",
                                tier="static"))
        return
    findings, engine = lint_paths(targets)
    errors = [f for f in findings if f["severity"] == "error"]
    for f in errors:
        report.add(ProbeVerdict(
            "US-10", "fail", f["message"], tier="static",
            evidence=f"{Path(f['file']).name}:{f['line']}"))
    if not errors:
        report.add(ProbeVerdict(
            "US-10", "pass",
            f"{len(targets)} file(s) lint-clean "
            f"({len(findings)} warning(s); engine: {engine})", tier="static"))


def _provenance_tier(run_dir: Path, report: ProbeReport) -> None:
    params_path = run_dir / ".pipeline" / "params.json"
    paper_path = run_dir / ".pipeline" / "paper.md"
    if not params_path.is_file():
        report.add(ProbeVerdict("US-1", "unprobeable", "no params.json",
                                tier="static"))
        return
    params = json.loads(params_path.read_text(encoding="utf-8"))
    params = params.get("params", params)
    paper = paper_path.read_text(encoding="utf-8") if paper_path.is_file() else None
    findings = check_params(params, paper)
    errors = [f for f in findings if f["severity"] == "error"]
    for f in errors:
        report.add(ProbeVerdict(f["probe"], "fail",
                                f"{f['param']}: {f['message']}", tier="static"))
    for f in findings:
        if f["severity"] == "warn":
            report.add(ProbeVerdict(f["probe"], "warn",
                                    f"{f['param']}: {f['message']}",
                                    tier="static"))
    if not errors:
        note = "" if paper else " (US-3 skipped: no paper.md)"
        report.add(ProbeVerdict(
            "US-1", "pass",
            f"params provenance probes clean over {len(params)} entries{note}",
            tier="static"))


def _paradigm_and_pluggable(run_dir: Path) -> tuple[str, str]:
    spec_path = run_dir / ".pipeline" / "method_spec.json"
    if not spec_path.is_file():
        return "", ""
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        comparison = spec.get("comparison", {})
        paradigm = comparison.get("classification", {}).get("id", "")
        pluggable = comparison.get("pluggable_component", {}).get("name", "")
        return paradigm, pluggable
    except (json.JSONDecodeError, AttributeError):
        return "", ""


def _node_declared_probe_context(
    paradigm: str,
) -> dict[str, dict[str, str]] | None:
    """Probe executor refs and authored taxonomy context, or None.

    Phase 1 of the taxonomy migration: a populated node's `semantic_checks`
    carry `probe:` executor refs (`al_loop.microharness`, ...), so the NODE
    declares which behavioural probes apply and the harness here only knows HOW
    to run them.  Verification explainability keeps the node's check id, check
    text, and silent-failure rationale on the emitted verdict, so the report
    renderer never needs to load taxonomy YAML.

    Returns None when no node serves the paradigm (still `reserved`) — the
    caller then falls back to dispatching the full family battery, preserving
    Phase-0 behaviour.

    Scenario-fidelity slice B: the node's `scenario_assumption_dimensions`
    bind detectors exactly like semantic checks bind probes, so their
    `detector` refs merge into the same declared context. Scenario entries
    additionally carry the `dimension` id — the join key against
    `method_spec.scenario_assumptions` — which is how the dispatch (and the
    vendored harness's frozen copy of this context) recognizes them.
    """
    try:
        from scripts import taxonomy as tx  # noqa: PLC0415
    except ImportError:
        return None
    node = tx.serves(paradigm)
    if node is None:
        return None
    if isinstance(node, tx.VariantNode):
        checks = tx.effective_node(tx.load_taxonomy(), node.slug).get("semantic_checks") or []
    else:  # FamilyNode: read its own checks directly (no variant layer).
        checks = node.fields.get("semantic_checks") or []
    out: dict[str, dict[str, str]] = {}
    for check in checks:
        if not isinstance(check, dict) or not check.get("probe"):
            continue
        ref = str(check["probe"])
        context = {
            "id": str(check.get("id") or ""),
            "check": str(check.get("check") or ""),
            "why": str(check.get("silent_failure") or ""),
        }
        if ref in out and out[ref] != context:
            raise ValueError(
                f"taxonomy node {paradigm!r} maps probe ref {ref!r} "
                "to multiple check records"
            )
        out[ref] = context
    for dimension in tx.load_scenario_assumption_dimensions(paradigm):
        ref = str(dimension.get("detector") or "")
        if not ref:
            continue
        dimension_id = str(dimension.get("id") or "")
        context = {
            "id": f"scenario-{dimension_id}",
            "check": str(dimension.get("check") or ""),
            "why": str(dimension.get("silent_failure") or ""),
            "dimension": dimension_id,
        }
        if ref in out and out[ref] != context:
            raise ValueError(
                f"taxonomy node {paradigm!r} maps probe ref {ref!r} "
                "to multiple check records"
            )
        out[ref] = context
    return out


def _node_declared_probe_refs(paradigm: str) -> set[str] | None:
    """Compatibility view of the taxonomy-declared executor refs."""
    context = _node_declared_probe_context(paradigm)
    return None if context is None else set(context)


def _stamp_pack_context(
    verdict: ProbeVerdict,
    ref: str,
    declared_context: dict[str, dict[str, str]] | None,
) -> ProbeVerdict:
    """Attach the exact family executor ref and its authored context.

    This wrapper is the single taxonomy-gated return seam.  Stamping the ref
    here keeps pass, fail, unprobeable, and conditional outcomes equally
    bindable; universal checks that bypass the wrapper remain honestly
    unbound.
    """
    normalized_ref = str(ref or "").strip()
    existing_ref = str(verdict.probe_ref or "").strip()
    if existing_ref and existing_ref != normalized_ref:
        raise ValueError(
            f"probe verdict {verdict.probe_id!r} already carries ref "
            f"{existing_ref!r}; taxonomy dispatch attempted to stamp "
            f"different ref {normalized_ref!r}"
        )
    verdict.probe_ref = normalized_ref
    context = (declared_context or {}).get(ref)
    if context:
        verdict.pack_check_id = context["id"]
        verdict.pack_check = context["check"]
        verdict.pack_why = context["why"]
    return verdict


def _probe_enabled(ref: str, declared: set[str] | None) -> bool:
    """A probe runs if the node declares its ref, or if no node serves the
    paradigm yet (declared is None → fall back to the full battery)."""
    return declared is None or ref in declared


def _scenario_dimension_bindings(
    declared_context: dict[str, dict[str, str]] | None,
) -> list[tuple[str, str]]:
    """(dimension_id, detector_ref) pairs from the declared context.

    Scenario entries are the ones carrying a `dimension` key (see
    `_node_declared_probe_context`); non-declaring families yield an empty
    list, so their batteries and reports stay byte-identical."""
    out = [
        (str(context["dimension"]), ref)
        for ref, context in (declared_context or {}).items()
        if isinstance(context, dict) and context.get("dimension")
    ]
    return sorted(out)


def _live_setup_boundary(run_dir: Path) -> int | None:
    """The layout-resolved setup-section boundary, or None outside the repo.

    In-repo batteries resolve it from the taxonomy through the run overlay
    (the demo-verdict pass's own resolution); the vendored harness carries a
    frozen copy instead, so an ImportError here is expected there."""
    try:
        from scripts import demo_verdict  # noqa: PLC0415
    except ImportError:
        return None
    try:
        _, demo_success, layout, _ = demo_verdict.run_demo_context(run_dir)
        return demo_verdict.setup_section_boundary(layout, demo_success)
    except Exception:  # noqa: BLE001 — a broken spec must not kill the battery
        return None


def _live_training_history_required(run_dir: Path) -> bool:
    """Whether the live run has R2C-090's typed schema-2 boundary."""
    try:
        from scripts import demo_verdict  # noqa: PLC0415

        plan = demo_verdict._effective_build_plan(Path(run_dir))
        return bool(
            demo_verdict._training_history_applicable(Path(run_dir), plan)
        )
    except Exception:  # noqa: BLE001 - unresolved applicability is not proof
        return False


def _frozen_training_history_staleness(
    run_dir: Path,
    frozen_gating: dict,
) -> ProbeVerdict | None:
    """Fail closed when delivery-time history sources no longer match.

    The portable harness cannot regenerate pipeline-owned post-smoke evidence.
    It may reuse the delivered record only while the complete bounded method,
    data, configuration, executed-notebook, and evidence authority remains
    byte-identical to the delivery freeze.
    """
    expected = frozen_gating.get("training_history_authority_digests")
    if not isinstance(expected, dict) or not expected or any(
        not isinstance(value, str) or len(value) != 64
        for value in expected.values()
    ):
        return ProbeVerdict(
            "UB-9",
            "unprobeable",
            "the frozen harness has no complete delivery-time digest manifest "
            "for the training-history authority",
            evidence="training_history_authority_digests missing or malformed",
            reason="training_history_authority_digest_unavailable",
        )

    actual = _training_history_authority_digests(run_dir)
    changed = [
        (
            f"{relative}=added" if relative not in expected else
            f"{relative}=removed" if relative not in actual else
            f"{relative}=changed"
        )
        for relative in sorted(set(expected) | set(actual))
        if expected.get(relative) != actual.get(relative)
        or (relative in expected) != (relative in actual)
    ]
    if not changed:
        return None
    return ProbeVerdict(
        "UB-9",
        "unprobeable",
        "the delivered training-history record is stale because its bounded "
        "source/data/config/artifact authority changed after delivery",
        evidence=", ".join(changed),
        reason="training_history_authority_stale",
    )


def _graph_stage_2d_authority_digests(
    run_dir: Path,
) -> dict[str, str]:
    """Recompute the exact upstream set certified by Stage 2.d.

    This mirrors the driver's Stage 2.d digest contract without importing the
    driver (the vendored probe harness does not carry it).  ``__init__.py`` is
    a Stage 2.d output; every other top-level ``method/*.py`` file plus the two
      structured validator inputs is upstream authority. Nested package modules
      are included; only the root ``method/__init__.py`` Stage 2.d output is
      excluded.
    """
    root = Path(run_dir)
    files: list[Path] = []
    method_dir = root / "method"
    if method_dir.is_dir():
        files.extend(sorted(
            path for path in method_dir.rglob("*.py")
            if path != method_dir / "__init__.py"
        ))
    files.extend(
        path
        for path in (
            root / ".pipeline" / "arch_contract.json",
            root / ".pipeline" / "method_spec.json",
        )
        if path.is_file()
    )
    return {
        path.relative_to(root).as_posix(): (
            "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        )
        for path in files
    }


def _graph_receipt_metadata(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Structural identities a graph-alignment receipt must certify."""
    alignment = plan.get("alignment")
    fixture = plan.get("fixture")
    return {
        "receipt_version": _GRAPH_RECEIPT_VERSION,
        "probe_ref": _GRAPH_ALIGNMENT_PROBE_REF,
        "element_id": (
            alignment.get("element_id")
            if isinstance(alignment, Mapping) else None
        ),
        "fixture_scope": (
            fixture.get("scope") if isinstance(fixture, Mapping) else None
        ),
        "representation": plan.get("representation"),
        "entity_ids": (
            fixture.get("entity_ids") if isinstance(fixture, Mapping) else None
        ),
    }


def _unprobeable_graph_plan(reason: str) -> dict[str, Any]:
    """Closed disposition returned when live graph-plan freezing fails."""
    return {
        "schema_version": "1.0",
        "status": "unprobeable",
        "reason": reason,
        "representation": None,
        "construction": None,
        "parameter_authority": {
            "status": "unprobeable",
            "reason": reason,
            "bindings": {},
        },
        "execution": None,
        "ablation": None,
        "groundings": {},
        "alignment": None,
        "fixture": None,
        "alignment_runtime_receipt": None,
    }


def _graph_grounding(
    plan: Mapping[str, Any],
    probe_ref: str,
) -> tuple[list[str], list[str], bool]:
    """Read only a grounding carrying the requested exact executor ref."""
    groundings = plan.get("groundings")
    raw = groundings.get(probe_ref) if isinstance(groundings, Mapping) else None
    if not isinstance(raw, Mapping) or raw.get("probe_ref") != probe_ref:
        return [], [], False

    def strings(value: object) -> list[str]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return []
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip() and item not in out:
                out.append(item)
        return out

    return (
        strings(raw.get("element_ids")),
        strings(raw.get("bound_callables")),
        True,
    )


def _graph_receipt_assessment(
    run_dir: Path,
    plan: Mapping[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    """Re-verify a live or portable alignment receipt against this run."""
    receipt = plan.get("alignment_runtime_receipt")
    if not isinstance(receipt, Mapping):
        return False, "alignment_runtime_receipt_missing", {}
    if set(receipt) != _GRAPH_RECEIPT_KEYS:
        return False, "alignment_runtime_receipt_shape_mismatch", {}
    expected_metadata = _graph_receipt_metadata(plan)
    mismatched_fields = sorted(
        key
        for key, expected in expected_metadata.items()
        if receipt.get(key) != expected
    )
    if receipt.get("validator") != _GRAPH_ALIGNMENT_VALIDATOR:
        mismatched_fields.append("validator")
    if receipt.get("reason") != "stage_2d_runtime_alignment_verified":
        mismatched_fields.append("reason")
    if mismatched_fields:
        return (
            False,
            "alignment_runtime_receipt_identity_mismatch",
            {"mismatched_fields": sorted(set(mismatched_fields))},
        )
    if receipt.get("status") != "pass" or receipt.get("verified") is not True:
        return (
            False,
            str(receipt.get("reason") or "alignment_runtime_receipt_unverified"),
            {
                "receipt_status": receipt.get("status"),
                "verified": receipt.get("verified"),
            },
        )
    if not (Path(run_dir) / ".pipeline" / "stage_2d.complete").is_file():
        return (
            False,
            "alignment_runtime_stage_receipt_missing",
            {"missing_stage": "stage_2d"},
        )
    authority = receipt.get("authority_digests")
    if not isinstance(authority, Mapping) or not authority or any(
        not isinstance(relative, str)
        or not relative
        or not isinstance(digest, str)
        or not digest.startswith("sha256:")
        or len(digest) != 71
        or digest != digest.lower()
        or any(
            character not in "0123456789abcdef"
            for character in digest.removeprefix("sha256:")
        )
        for relative, digest in authority.items()
    ):
        return False, "alignment_runtime_receipt_authority_malformed", {}
    try:
        current = _graph_stage_2d_authority_digests(run_dir)
    except OSError as exc:
        return (
            False,
            "alignment_runtime_receipt_authority_unreadable",
            {"exception": type(exc).__name__},
        )
    expected = dict(authority)
    if current != expected:
        changed = sorted(
            relative
            for relative in set(current) | set(expected)
            if current.get(relative) != expected.get(relative)
        )
        return (
            False,
            "alignment_runtime_receipt_stale",
            {"changed_authority_paths": changed},
        )
    return (
        True,
        "alignment_runtime_receipt_verified",
        {
            "validator": _GRAPH_ALIGNMENT_VALIDATOR,
            "authority_paths": sorted(expected),
        },
    )


def _graph_verdict(
    probe_id: str,
    probe_ref: str,
    status: str,
    message: str,
    *,
    evidence: Mapping[str, Any] | None = None,
    element_ids: Sequence[str] = (),
    bound_callables: Sequence[str] = (),
    reason: str,
) -> ProbeVerdict:
    return ProbeVerdict(
        probe_id,
        status,
        message,
        evidence=json.dumps(
            dict(evidence or {}),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        element_ids=list(element_ids),
        bound_callables=list(bound_callables),
        reason=reason,
        probe_ref=probe_ref,
    )


def _graph_alignment_verdict(
    run_dir: Path,
    plan: Mapping[str, Any],
) -> ProbeVerdict:
    """Emit HG-1 from structural alignment plus runtime receipt authority."""
    from probes.graph_mechanism import graph_plan_trace  # noqa: PLC0415

    plan_status = str(plan.get("status") or "")
    plan_reason = str(plan.get("reason") or "graph plan has no disposition")
    element_ids, bound_callables, grounded = _graph_grounding(
        plan, _GRAPH_ALIGNMENT_PROBE_REF
    )
    base_evidence: dict[str, Any] = {
        "plan_status": plan_status,
        "plan_reason": plan_reason,
        "trace": graph_plan_trace(plan),
    }
    if plan_status == "not_applicable":
        return _graph_verdict(
            "HG-1", _GRAPH_ALIGNMENT_PROBE_REF, "not_applicable", plan_reason,
            evidence=base_evidence,
            element_ids=element_ids,
            bound_callables=bound_callables,
            reason=plan_reason,
        )
    if plan_status == "unprobeable":
        return _graph_verdict(
            "HG-1", _GRAPH_ALIGNMENT_PROBE_REF, "unprobeable", plan_reason,
            evidence=base_evidence,
            element_ids=element_ids,
            bound_callables=bound_callables,
            reason=plan_reason,
        )

    alignment = plan.get("alignment")
    alignment_status = (
        alignment.get("status") if isinstance(alignment, Mapping) else None
    )
    alignment_reason = (
        alignment.get("reason") if isinstance(alignment, Mapping) else None
    )
    base_evidence["alignment_status"] = alignment_status
    if alignment_status == "fail":
        reason = str(alignment_reason or "relational_alignment_failed")
        return _graph_verdict(
            "HG-1", _GRAPH_ALIGNMENT_PROBE_REF, "fail", reason,
            evidence=base_evidence,
            element_ids=element_ids,
            bound_callables=bound_callables,
            reason=reason,
        )
    if alignment_status not in ("ready", "pass"):
        reason = str(
            alignment_reason or "relational_alignment_prerequisite_unresolved"
        )
        return _graph_verdict(
            "HG-1", _GRAPH_ALIGNMENT_PROBE_REF, "unprobeable", reason,
            evidence=base_evidence,
            element_ids=element_ids,
            bound_callables=bound_callables,
            reason=reason,
        )
    if not grounded or not element_ids or not bound_callables:
        reason = "alignment_exact_grounding_missing"
        return _graph_verdict(
            "HG-1", _GRAPH_ALIGNMENT_PROBE_REF, "unprobeable", reason,
            evidence=base_evidence,
            element_ids=element_ids,
            bound_callables=bound_callables,
            reason=reason,
        )
    alignment_element = alignment.get("element_id")
    if alignment_element not in element_ids:
        reason = "alignment_grounding_element_mismatch"
        return _graph_verdict(
            "HG-1", _GRAPH_ALIGNMENT_PROBE_REF, "unprobeable", reason,
            evidence={
                **base_evidence,
                "alignment_element_id": alignment_element,
                "grounded_element_ids": element_ids,
            },
            element_ids=element_ids,
            bound_callables=bound_callables,
            reason=reason,
        )
    verified, reason, receipt_evidence = _graph_receipt_assessment(
        run_dir, plan
    )
    evidence = {**base_evidence, **receipt_evidence}
    return _graph_verdict(
        "HG-1",
        _GRAPH_ALIGNMENT_PROBE_REF,
        "pass" if verified else "unprobeable",
        (
            "Stage 2.d runtime evidence certifies exact relational alignment"
            if verified else reason
        ),
        evidence=evidence,
        element_ids=element_ids,
        bound_callables=bound_callables,
        reason=reason,
    )


def _exact_generated_callable(identity: object) -> Callable[..., object] | None:
    """Resolve one exact supported owner/module identity, with no aliases."""
    if not isinstance(identity, Mapping) or set(identity) != {
        "module", "qualname"
    }:
        return None
    module_name = identity.get("module")
    qualname = identity.get("qualname")
    if (
        not isinstance(module_name, str)
        or not module_name.strip()
        or module_name != module_name.strip()
        or module_name not in ALLOWED_GRAPH_CALLABLE_MODULES
        or not isinstance(qualname, str)
        or not qualname.strip()
        or qualname != qualname.strip()
        or "." in qualname
    ):
        return None
    try:
        target: object = importlib.import_module(module_name)
        for segment in qualname.split("."):
            if not segment or segment == "<locals>":
                return None
            target = getattr(target, segment)
    except Exception:  # noqa: BLE001 - generated modules may fail arbitrarily
        return None
    return target if callable(target) else None


def _graph_callable_key(identity: object) -> str | None:
    if not isinstance(identity, Mapping) or set(identity) != {
        "module", "qualname"
    }:
        return None
    module = identity.get("module")
    qualname = identity.get("qualname")
    if not isinstance(module, str) or not module.strip() \
            or module != module.strip() \
            or module not in ALLOWED_GRAPH_CALLABLE_MODULES \
            or not isinstance(qualname, str) or not qualname.strip() \
            or qualname != qualname.strip() or "." in qualname:
        return None
    return f"{module}:{qualname}"


def _graph_backend_array(
    value: object,
    *,
    tensor_backend: str,
    integral: bool,
) -> object:
    """Materialize one graph-probe input in the exact declared CPU backend."""
    if tensor_backend == "numpy":
        import numpy as np  # noqa: PLC0415

        candidate = value
        if (
            isinstance(candidate, np.ndarray)
            and type(candidate) is not np.ndarray
        ):
            raise ValueError(
                "graph probe numeric inputs must use exact base NumPy arrays; "
                "ndarray subclass dispatch is outside the frozen-fixture proof"
            )
        detach = getattr(candidate, "detach", None)
        if callable(detach):
            candidate = detach()
        cpu = getattr(candidate, "cpu", None)
        if callable(cpu):
            candidate = cpu()
        as_numpy = getattr(candidate, "numpy", None)
        if callable(as_numpy):
            candidate = as_numpy()
        array = np.asarray(
            candidate,
            dtype=np.int64 if integral else np.float32,
        )
        if type(array) is not np.ndarray:
            raise ValueError(
                "graph probe tensorization did not produce an exact base NumPy "
                "array"
            )
        return array
    if tensor_backend == "torch":
        import torch  # noqa: PLC0415

        dtype = torch.long if integral else torch.float32
        if isinstance(value, torch.Tensor):
            if type(value) is not torch.Tensor:
                raise ValueError(
                    "graph probe numeric inputs must use exact base Torch "
                    "tensors; Tensor subclass dispatch is outside the frozen-"
                    "fixture proof"
                )
            array = value.detach().to(device="cpu", dtype=dtype)
        else:
            array = torch.as_tensor(value, dtype=dtype, device="cpu")
        if type(array) is not torch.Tensor:
            raise ValueError(
                "graph probe tensorization did not produce an exact base Torch "
                "tensor"
            )
        return array
    raise ValueError(
        f"unsupported graph tensor_backend {tensor_backend!r}"
    )


def _graph_constructor_adapter(
    function: Callable[..., object],
    construction: Mapping[str, Any],
    *,
    tensor_backend: str,
) -> Callable[..., object] | None:
    feature_parameter = construction.get("feature_parameter")
    threshold = construction.get("threshold")
    threshold_binding = (
        threshold.get("parameter") if isinstance(threshold, Mapping) else None
    )
    threshold_parameter = (
        threshold_binding.get("callable_parameter")
        if isinstance(threshold_binding, Mapping) else None
    )
    threshold_name = (
        threshold_binding.get("params_name")
        if isinstance(threshold_binding, Mapping) else None
    )
    cap = construction.get("cap")
    cap_binding = (
        cap.get("parameter")
        if isinstance(cap, Mapping) and cap.get("kind") != "none" else None
    )
    cap_parameter = (
        cap_binding.get("callable_parameter")
        if isinstance(cap_binding, Mapping) else None
    )
    cap_name = (
        cap_binding.get("params_name")
        if isinstance(cap_binding, Mapping) else None
    )
    selector = construction.get("output_selector")
    required_names = [feature_parameter, threshold_parameter, threshold_name]
    if any(not isinstance(value, str) or not value.strip()
           for value in required_names) or not isinstance(selector, Mapping):
        return None
    callable_parameters = [str(feature_parameter), str(threshold_parameter)]
    if cap_binding is not None:
        if any(not isinstance(value, str) or not value.strip()
               for value in (cap_parameter, cap_name)):
            return None
        callable_parameters.append(str(cap_parameter))
    if len(callable_parameters) != len(set(callable_parameters)):
        return None

    def select(raw: object) -> object:
        kind = selector.get("kind")
        if kind == "return_value":
            return raw
        if kind == "tuple_item":
            index = selector.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("tuple_item graph selector has no exact index")
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
                raise ValueError("graph constructor did not return a sequence")
            return raw[index]
        if kind == "mapping_item":
            key = selector.get("key")
            if not isinstance(key, str) or not isinstance(raw, Mapping):
                raise ValueError("graph constructor did not return a mapping")
            return raw[key]
        raise ValueError(f"unsupported exact graph output selector {kind!r}")

    def adapter(*, feature_input, threshold, cap, seed):
        del seed
        kwargs = {
            str(feature_parameter): _graph_backend_array(
                feature_input,
                tensor_backend=tensor_backend,
                integral=False,
            ),
            str(threshold_parameter): threshold,
        }
        parameter_values = {str(threshold_name): threshold}
        if cap_binding is not None:
            kwargs[str(cap_parameter)] = cap
            parameter_values[str(cap_name)] = cap
        raw = function(**kwargs)
        return {
            "graph": select(raw),
            "parameter_values": parameter_values,
        }

    return adapter


def _graph_execution_adapter(
    function: Callable[..., object],
    bindings: Mapping[str, Any],
    *,
    representation: str,
    tensor_backend: str,
) -> Callable[..., object] | None:
    graph_parameter = bindings.get("graph_parameter")
    signal_parameter = bindings.get("neighbor_signal_parameter")
    if (
        not isinstance(graph_parameter, str)
        or not graph_parameter.strip()
        or not isinstance(signal_parameter, str)
        or not signal_parameter.strip()
        or graph_parameter == signal_parameter
    ):
        return None

    def adapter(*, graph, neighbor_signal, entity_ids, seed):
        del entity_ids, seed
        return function(**{
            graph_parameter: _graph_backend_array(
                graph,
                tensor_backend=tensor_backend,
                integral=representation == "sparse_edge_index",
            ),
            signal_parameter: _graph_backend_array(
                neighbor_signal,
                tensor_backend=tensor_backend,
                integral=False,
            ),
        })

    return adapter


def _graph_callable_adapters(
    plan: Mapping[str, Any],
) -> dict[str, Callable[..., object]]:
    """Build normalized adapters only for exact identities in the plan."""
    construction = plan.get("construction")
    execution = plan.get("execution")
    if not isinstance(construction, Mapping) or not isinstance(
        execution, Mapping
    ):
        return {}
    tensor_backend = execution.get("tensor_backend")
    representation = plan.get("representation")
    if tensor_backend not in ("numpy", "torch") or representation not in (
        "sparse_edge_index", "dense_adjacency"
    ):
        return {}
    construction_identity = construction.get("callable")
    execution_identity = execution.get("callable")
    construction_key = _graph_callable_key(construction_identity)
    execution_key = _graph_callable_key(execution_identity)
    if not construction_key or not execution_key \
            or construction_key == execution_key:
        return {}
    adapters: dict[str, Callable[..., object]] = {}
    construction_callable = _exact_generated_callable(construction_identity)
    if construction_callable is not None:
        adapter = _graph_constructor_adapter(
            construction_callable,
            construction,
            tensor_backend=str(tensor_backend),
        )
        if adapter is not None:
            adapters[construction_key] = adapter
    execution_callable = _exact_generated_callable(execution_identity)
    if execution_callable is not None:
        adapter = _graph_execution_adapter(
            execution_callable,
            execution,
            representation=str(representation),
            tensor_backend=str(tensor_backend),
        )
        if adapter is not None:
            adapters[execution_key] = adapter

    ablation = plan.get("ablation")
    ablation_identity = (
        ablation.get("callable") if isinstance(ablation, Mapping) else None
    )
    ablation_key = _graph_callable_key(ablation_identity)
    if ablation_key and ablation_key != construction_key:
        ablation_callable = _exact_generated_callable(ablation_identity)
        if ablation_callable is not None:
            adapter = _graph_execution_adapter(
                ablation_callable,
                ablation,
                representation=str(representation),
                tensor_backend=str(tensor_backend),
            )
            if adapter is not None:
                adapters[ablation_key] = adapter
    return adapters


def _graph_result_verdict(result: Any) -> ProbeVerdict:
    evidence = result.evidence if isinstance(result.evidence, Mapping) else {}
    reason_code = evidence.get("reason_code")
    reason = (
        reason_code.strip()
        if isinstance(reason_code, str) and reason_code.strip()
        else result.reason
    )
    return _graph_verdict(
        result.probe_id,
        result.probe_ref,
        result.status,
        result.reason,
        evidence=evidence,
        element_ids=result.element_ids,
        bound_callables=result.bound_callables,
        reason=reason,
    )


def _run_graph_mechanism_consumers(
    run_dir: Path,
    plan: Mapping[str, Any],
) -> list[ProbeVerdict]:
    """Run HG-1 separately, then HG-2..HG-7 behind that prerequisite."""
    from probes.graph_mechanism import (  # noqa: PLC0415
        run_graph_mechanism_probes,
    )

    hg1 = _graph_alignment_verdict(run_dir, plan)
    plan_status = plan.get("status")
    fixture = plan.get("fixture")
    injected_fixture = fixture if isinstance(fixture, Mapping) else {}
    execution_plan = json.loads(json.dumps(plan, allow_nan=False))
    if plan_status not in ("not_applicable", "unprobeable"):
        if hg1.verdict != "pass":
            execution_plan["status"] = "unprobeable"
            execution_plan["reason"] = (
                f"HG-1 alignment prerequisite did not pass: {hg1.reason}"
            )
            execution_plan["alignment"] = {
                "status": "blocked",
                "reason": hg1.reason,
            }
        else:
            alignment = execution_plan.get("alignment")
            if (
                isinstance(alignment, dict)
                and alignment.get("status") != "pass"
            ):
                alignment["status"] = "pass"
                alignment["reason"] = "alignment_runtime_receipt_verified"

    if execution_plan.get("status") == "ready" and hg1.verdict == "pass":
        live_verified, live_reason, live_evidence = assess_liveness_receipt(
            run_dir, execution_plan
        )
        if not live_verified:
            execution_plan["status"] = "unprobeable"
            execution_plan["reason"] = live_reason
            execution_plan["callable_liveness_assessment"] = {
                "verified": False,
                "reason": live_reason,
                "evidence": live_evidence,
            }

    # The executor consumes the receipt-qualified copy, not the pre-HG-1
    # planner payload. Re-emit the independently passing prerequisite against
    # that exact copy so every passing HG row bears the same canonical plan
    # and fixture digests.
    if execution_plan.get("status") == "ready" and hg1.verdict == "pass":
        hg1 = _graph_alignment_verdict(run_dir, execution_plan)

    adapters: dict[str, Callable[..., object]] = {}
    if execution_plan.get("status") == "ready" \
            and hg1.verdict == "pass":
        from probes.package_loader import (  # noqa: PLC0415
            ProbeLoadError, imported_method_package,
        )
        try:
            with imported_method_package(run_dir):
                adapters = _graph_callable_adapters(execution_plan)
                results = run_graph_mechanism_probes(
                    execution_plan, adapters, injected_fixture
                )
        except ProbeLoadError:
            results = run_graph_mechanism_probes(
                execution_plan, {}, injected_fixture
            )
    else:
        results = run_graph_mechanism_probes(
            execution_plan, adapters, injected_fixture
        )
    return [hg1, *[_graph_result_verdict(result) for result in results]]


def run_battery(
    run_dir: Path,
    n_classes: int | None = None,
    *,
    ledger_path: Path | None = None,
    frozen_gating: dict | None = None,
) -> ProbeReport:
    """Run the full battery. `ledger_path` redirects the CT-3 claims-ledger
    write (isolated output mode); None keeps the delivery-time default of
    <run>/.pipeline/claims_ledger.json.

    `frozen_gating` is the vendored-harness contract (R2C-019): the freeze
    file built at delivery time carries the node-declared probe context the
    taxonomy served THEN, so a battery running outside this repository
    reproduces the delivered check set exactly. Without it, a vendored
    runner that merely failed the taxonomy import would silently fall back
    to the full battery — a DIFFERENT probe set than the delivered report.
    A frozen `declared_context` of null round-trips as the full-battery
    fallback, exactly what a then-reserved paradigm delivered."""
    report = ProbeReport(target=str(run_dir))
    paradigm, pluggable = _paradigm_and_pluggable(run_dir)
    if frozen_gating is not None:
        declared_context = frozen_gating.get("declared_context")
    else:
        declared_context = _node_declared_probe_context(paradigm)
    declared = None if declared_context is None else set(declared_context)

    def add_gated(verdict: ProbeVerdict, ref: str) -> None:
        report.add(_stamp_pack_context(verdict, ref, declared_context))

    _lint_tier(run_dir, report)
    _provenance_tier(run_dir, report)
    report.add(probe_narrative_vs_params(run_dir))

    # Scale-dependent hyperparameter tier: US-4 (static scale-vs-calibration)
    # plus US-4b (behavioral de-saturation, queue 11d Arm B / the folded 11f).
    # Both return [] when the spec declares no scale-dependent hyperparameters,
    # so this is a no-op for paradigms and fixtures that have none.
    from probes.scale_mismatch import (  # noqa: PLC0415
        probe_geometric_saturation, probe_scale_dependent_mismatch)
    for v in probe_scale_dependent_mismatch(run_dir):
        report.add(v)
    for v in probe_geometric_saturation(run_dir):
        report.add(v)

    nb_path = run_dir / "notebook.ipynb"
    if nb_path.is_file():
        for v in probe_executed_notebook(nb_path, n_classes=n_classes):
            report.add(v)
    else:
        report.add(ProbeVerdict(
            "UB-6", "unprobeable", "no notebook.ipynb",
            reason="notebook_missing"))

    history_required = (
        bool(frozen_gating.get("training_history_required"))
        if frozen_gating is not None
        else _live_training_history_required(run_dir)
    )
    if history_required:
        stale = (
            _frozen_training_history_staleness(run_dir, frozen_gating)
            if frozen_gating is not None else None
        )
        report.add(stale or probe_loss_descent(run_dir))

    if (
        paradigm.startswith("motion_planning")
        and nb_path.is_file()
        and _probe_enabled("motion_planning.scenario_dynamics", declared)
    ):
        from probes.motion_planning import probe_scenario_dynamics_static  # noqa: PLC0415
        names = (pluggable,) if pluggable else None
        kwargs = {"planner_names": names} if names else {}
        add_gated(
            probe_scenario_dynamics_static(nb_path, **kwargs),
            "motion_planning.scenario_dynamics",
        )

    if paradigm.startswith("motion_planning") and (run_dir / "method").is_dir():
        import inspect  # noqa: PLC0415

        from probes.motion_planning import (  # noqa: PLC0415
            probe_goal_progress, probe_steering_responsiveness)
        from probes.package_loader import (  # noqa: PLC0415
            ProbeLoadError, imported_method_package)
        from probes.term_ablation import (  # noqa: PLC0415
            probe_direction_consistency, probe_term_ranking_power)
        try:
            with imported_method_package(run_dir) as method:
                fn = getattr(method, pluggable, None) if pluggable else None
                if fn is None:
                    report.add(ProbeVerdict(
                        "UB-4", "unprobeable",
                        f"pluggable {pluggable!r} not found in method package"))
                else:
                    mod = inspect.getmodule(fn) or method
                    report.add(probe_direction_consistency(mod, pluggable))
                    report.add(probe_term_ranking_power(mod, pluggable))
                    steering_on = _probe_enabled(
                        "motion_planning.steering_responsiveness", declared)
                    goal_on = _probe_enabled(
                        "motion_planning.goal_progress", declared)
                    if steering_on or goal_on:
                        # Contract-driven problem instance from the package's
                        # own Environment/dynamics classes (queue item 12) —
                        # built once from the full package (Environment lives
                        # in data.py, not in the pluggable's own module).
                        from probes.motion_planning import mp_planner_kit  # noqa: PLC0415
                        kit = mp_planner_kit(method, pluggable)
                        if steering_on:
                            add_gated(
                                probe_steering_responsiveness(
                                    mod, pluggable, kit=kit),
                                "motion_planning.steering_responsiveness",
                            )
                        if goal_on:
                            add_gated(
                                probe_goal_progress(mod, pluggable, kit=kit),
                                "motion_planning.goal_progress",
                            )
        except ProbeLoadError as e:
            report.add(ProbeVerdict("UB-4", "unprobeable", str(e)))

    # Scenario-fidelity enforcement (R2C-025 slice B): families that declare
    # scenario dimensions opted into a runtime comparison of the executed
    # demo setup against the paper's captured assumptions. Family-generic on
    # purpose — the bindings come from the declared context, so non-declaring
    # families produce no verdicts and stay byte-identical, and the vendored
    # harness reproduces the delivered set through its frozen context plus
    # the frozen setup boundary.
    scenario_dims = _scenario_dimension_bindings(declared_context)
    if scenario_dims and nb_path.is_file():
        if frozen_gating is not None:
            boundary = frozen_gating.get("scenario_setup_boundary")
        else:
            boundary = _live_setup_boundary(run_dir)
        from probes.scenario_setup import scenario_fidelity_verdicts  # noqa: PLC0415
        for verdict, ref in scenario_fidelity_verdicts(
                run_dir, scenario_dims, setup_boundary=boundary):
            add_gated(verdict, ref)

    if paradigm.startswith("active_learning"):
        # The populated active_learning node declares which behavioural probes
        # apply via `semantic_checks[].probe`; `declared` gates dispatch below.
        # None => no node serves this paradigm yet => run the full battery.
        if nb_path.is_file() and _probe_enabled("al_loop.microharness", declared):
            from probes.al_loop import run_al_loop_microharness  # noqa: PLC0415
            for v in run_al_loop_microharness(
                    nb_path, pluggable_name=pluggable or "select_batch"):
                add_gated(v, "al_loop.microharness")

        # AL-3 — the deterministic fresh-retrain gate under its catalog id
        # (calls the stage validator's own AST gate; returns None for
        # deliberately warm-starting papers, mirroring the validator's
        # protocol exception).
        params_path = run_dir / ".pipeline" / "params.json"
        raw_params = {}
        if params_path.is_file():
            loaded = json.loads(params_path.read_text(encoding="utf-8"))
            raw_params = loaded.get("params", loaded)

        if _probe_enabled("al_loop.fresh_retrain", declared):
            from probes.al_loop import probe_fresh_retrain_static  # noqa: PLC0415
            protocol = ""
            spec_path = run_dir / ".pipeline" / "method_spec.json"
            if spec_path.is_file():
                try:
                    full_spec = json.loads(spec_path.read_text(encoding="utf-8"))
                    protocol = ((full_spec.get("critical_requirements") or {})
                                .get("training") or {}).get("protocol", "") or ""
                except json.JSONDecodeError:
                    pass
            al3 = probe_fresh_retrain_static(nb_path, protocol)
            if al3 is not None:
                add_gated(al3, "al_loop.fresh_retrain")

        if nb_path.is_file() and _probe_enabled("al_loop.eval_label_alignment", declared):
            from probes.al_loop import probe_eval_label_alignment_static  # noqa: PLC0415
            add_gated(
                probe_eval_label_alignment_static(
                    nb_path,
                    pluggable_name=pluggable or "select_batch",
                ),
                "al_loop.eval_label_alignment",
            )

        # AL-6 is static (params.json arithmetic only) — it must report even
        # when the method package was never generated (a run halted between
        # params derivation and codegen is exactly where the no-op-demo
        # misconfig is still detectable). Outside the method-dir guard by
        # design (2026-06-11 adversarial-review catch: the guard silently
        # skipped it, violating the never-silent invariant above).
        if _probe_enabled("al_loop.demo_config_reachability", declared):
            from probes.al_loop import probe_demo_config_reachability  # noqa: PLC0415
            if raw_params:
                add_gated(
                    probe_demo_config_reachability(raw_params),
                    "al_loop.demo_config_reachability",
                )
            else:
                add_gated(
                    ProbeVerdict(
                        "AL-6", "unprobeable",
                        "no params.json in the run dir"),
                    "al_loop.demo_config_reachability",
                )

        if (run_dir / "method").is_dir():
            from probes.package_loader import (  # noqa: PLC0415
                ProbeLoadError, imported_method_package)
            from probes.trainability import (  # noqa: PLC0415
                flat_live_params, probe_trains_on_synthetic)
            live = flat_live_params(raw_params) if raw_params else {}
            report.add(probe_trains_on_synthetic(run_dir, live_params=live))
            try:
                with imported_method_package(run_dir) as method:
                    name = pluggable or "select_batch"
                    from probes.term_ablation import probe_al_selector_terms  # noqa: PLC0415
                    report.add(probe_al_selector_terms(method, name))
                    if _probe_enabled("claims.contribution_floor", declared):
                        from probes.claims import probe_contribution_floor_al  # noqa: PLC0415
                        add_gated(
                            probe_contribution_floor_al(method, name),
                            "claims.contribution_floor",
                        )
                    if _probe_enabled("al_loop.acquisition_contract", declared):
                        from probes.al_loop import probe_acquisition_contract  # noqa: PLC0415
                        add_gated(
                            probe_acquisition_contract(method, name),
                            "al_loop.acquisition_contract",
                        )
                    if _probe_enabled("al_stage1.coreset_construction",
                                      declared):
                        from probes.al_stage1 import run_al_stage1_probes  # noqa: PLC0415
                        for v in run_al_stage1_probes(method):
                            add_gated(v, "al_stage1.coreset_construction")
            except ProbeLoadError as e:
                report.add(ProbeVerdict("UB-7", "unprobeable", str(e)))
                if _probe_enabled("claims.contribution_floor", declared):
                    add_gated(
                        ProbeVerdict("CT-1", "unprobeable", str(e)),
                        "claims.contribution_floor",
                    )
                if _probe_enabled("al_loop.acquisition_contract", declared):
                    add_gated(
                        ProbeVerdict("AL-5", "unprobeable", str(e)),
                        "al_loop.acquisition_contract",
                    )
                if _probe_enabled("al_stage1.coreset_construction", declared):
                    for pid in ("AL-S1-1", "AL-S1-2", "AL-S1-3"):
                        add_gated(
                            ProbeVerdict(pid, "unprobeable", str(e)),
                            "al_stage1.coreset_construction",
                        )

    if paradigm.startswith("knowledge_distillation"):
        from probes.kd import (  # noqa: PLC0415
            kd_loss_kit, probe_kd_teacher_influence, probe_kd_temperature,
            probe_teacher_signal_influence, probe_temperature_sensitivity,
            requires_model_batch_kit)
        from probes.package_loader import (  # noqa: PLC0415
            ProbeLoadError, imported_method_package, load_module_from_path)
        method_py = run_dir / "method" / "method.py"
        try:
            mod = load_module_from_path(method_py)
            loss_fn = getattr(mod, pluggable, None)
            if loss_fn is None:
                report.add(ProbeVerdict(
                    "KD-1", "unprobeable",
                    f"pluggable {pluggable!r} not found in method.py"))
            elif not requires_model_batch_kit(loss_fn):
                # Two-tensor loss: the original fast path, untouched.
                report.add(probe_teacher_signal_influence(loss_fn))
                kd2 = probe_temperature_sensitivity(loss_fn)
                if kd2 is not None:
                    report.add(kd2)
            else:
                # Model-and-batch loss: the contract-driven kit builds
                # student/teacher via the package's own builders and the
                # batch from arch_contract.json's declared shapes.
                arch_path = run_dir / ".pipeline" / "arch_contract.json"
                arch_contract = None
                if arch_path.is_file():
                    try:
                        arch_contract = json.loads(
                            arch_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        arch_contract = None
                with imported_method_package(run_dir) as package:
                    kit = kd_loss_kit(package, pluggable,
                                      arch_contract=arch_contract)
                    if isinstance(kit, str):
                        report.add(ProbeVerdict("KD-1", "unprobeable", kit))
                        report.add(ProbeVerdict("KD-2", "unprobeable", kit))
                    else:
                        report.add(probe_kd_teacher_influence(kit))
                        kd2 = probe_kd_temperature(kit)
                        if kd2 is not None:
                            report.add(kd2)
        except ProbeLoadError as e:
            report.add(ProbeVerdict("KD-1", "unprobeable", str(e)))

    # Forecasting dispatch is reference-driven rather than classification-
    # string-driven.  The committed node and its aliases declare the same
    # exact refs, while unrelated families and old frozen harnesses with no
    # such declarations remain byte-for-byte outside this kit.
    from probes.time_series_forecasting import (  # noqa: PLC0415
        PROBE_REFS as TSF_PROBE_REFS,
        run_time_series_forecasting_probes,
    )
    tsf_refs = set(TSF_PROBE_REFS).intersection(
        set(declared_context or {})
    )
    if tsf_refs:
        if frozen_gating is not None:
            execution_plan = frozen_gating.get(
                "forecasting_execution_plan"
            )
            demo_skill_evidence = frozen_gating.get("demo_skill_evidence")
            probe_groundings = frozen_gating.get(
                "forecasting_probe_groundings"
            )
        else:
            execution_plan = None
            probe_groundings = None
            # One shared freezer shape feeds live and portable execution.  It
            # reads only the run's structured demo artifact and never mutates
            # it; importing lazily avoids making this dependency part of
            # unrelated batteries.
            from build_probe_harness import (  # noqa: PLC0415
                _frozen_demo_skill_evidence,
            )
            demo_skill_evidence = _frozen_demo_skill_evidence(run_dir)
        for verdict in run_time_series_forecasting_probes(
            run_dir,
            enabled_refs=tsf_refs,
            execution_plan=execution_plan,
            demo_skill_evidence=demo_skill_evidence,
            probe_groundings=probe_groundings,
        ):
            add_gated(verdict, verdict.probe_ref)

    # Homogeneous-graph dispatch is also exact-reference driven.  HG-1 is a
    # runtime prerequisite over a Stage 2.d receipt; the numerical executor
    # owns only HG-2..HG-7 and cannot promote structural alignment by itself.
    graph_refs = _GRAPH_PROBE_REFS.intersection(
        set(declared_context or {})
    )
    if graph_refs:
        if frozen_gating is not None:
            graph_plan = frozen_gating.get(
                "graph_mechanism_execution_plan"
            )
            if not isinstance(graph_plan, Mapping):
                graph_plan = _unprobeable_graph_plan(
                    "frozen_graph_mechanism_execution_plan_missing"
                )
        else:
            from build_probe_harness import (  # noqa: PLC0415
                _frozen_graph_mechanism_execution_plan,
            )
            graph_plan = _frozen_graph_mechanism_execution_plan(
                run_dir, declared_context
            )
            if not isinstance(graph_plan, Mapping):
                graph_plan = _unprobeable_graph_plan(
                    "live_graph_mechanism_execution_plan_missing"
                )
        if graph_refs == {_GRAPH_ALIGNMENT_PROBE_REF}:
            graph_verdicts = [
                _graph_alignment_verdict(run_dir, graph_plan)
            ]
        else:
            graph_verdicts = _run_graph_mechanism_consumers(
                run_dir, graph_plan
            )
        for verdict in graph_verdicts:
            if verdict.probe_ref in graph_refs:
                add_gated(verdict, verdict.probe_ref)

    if not paradigm:
        report.add(ProbeVerdict(
            "battery", "warn",
            "no paradigm id in method_spec — paradigm tiers skipped",
            tier="static"))

    # Spec conditioning BEFORE the claims ledger, because the ledger reads
    # these verdicts and its label has to agree with the branch each one
    # carries (R2C-047).
    _apply_spec_conditioning(run_dir, report)

    # CT-3 claims ledger LAST: it reads the verdicts the other probes just
    # produced to synthesize the scale-free behavioral rows (the only rows that
    # can earn verified). Lifted experiment rows stay scale-bound and untested.
    from probes.claims import probe_claims_ledger  # noqa: PLC0415
    report.add(probe_claims_ledger(run_dir, report=report,
                                   ledger_path=ledger_path))
    return report


def _apply_spec_conditioning(run_dir: Path, report: ProbeReport) -> None:
    """Ask the run's own methodology contract about every finding (R2C-047).

    One pass over the finished verdicts rather than a spec argument threaded
    through every probe signature: the join needs only the element ids the
    verdict already carries, so every family gets conditioned at once and no
    probe has to learn about the contract.

    Each bound verdict records which branch adjudicates it and carries the
    contract's answer in its own message. The `contract_silent` branch changes
    nothing but the stamp, which is the recorded neutrality decision holding
    where it belongs: the probe cannot tell our miswiring from the paper's
    design, so the researcher adjudicates.

    The pass also resolves each verdict's declared callables into element ids
    first (R2C-072), because a probe knows which functions it exercised and
    nothing downstream does. A verdict that declares no callables and carries no
    ids stays unbound.
    """
    from probe_spec_join import (  # noqa: PLC0415
        disclosure, join_verdict, unbound_gap_note,
    )

    spec_path = run_dir / ".pipeline" / "method_spec.json"
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        spec = None
    for verdict in report.verdicts:
        if verdict.spec_branch:
            continue  # already conditioned (a re-run over the same report)
        _resolve_bound_callables(run_dir, verdict)
        join = join_verdict(verdict, spec)
        verdict.spec_branch = join.branch
        # A qualified pass needs no adjudication prose: there is nothing for
        # the contract to contradict or excuse. An otherwise-passing row that
        # failed exact binding is different: preserve that gap in the battery
        # artifact so downstream surfaces cannot mistake "ran" for "proved".
        if verdict.verdict == "pass":
            addition = unbound_gap_note(join) or ""
            if addition:
                verdict.message = f"{verdict.message} — {addition}"
            continue
        addition = disclosure(join) or (unbound_gap_note(join) or "")
        if addition:
            verdict.message = f"{verdict.message} — {addition}"


def _resolve_bound_callables(run_dir: Path, verdict) -> None:
    """Turn a verdict's declared callables into paper-map element ids (R2C-072).

    Additive only: ids a probe set itself are kept and the resolved ones are
    merged in, so a probe that already knows its elements is never overruled by
    the annotations. Silent when the probe declared nothing, when the generated
    package carries no annotations, or when the package cannot be scanned, and
    each of those leaves the verdict exactly as unbound as it was.
    """
    declared = list(getattr(verdict, "bound_callables", None) or [])
    if not declared:
        return
    try:
        from paper_element_anchors import (  # noqa: PLC0415
            element_ids_for_callables,
        )

        resolved = element_ids_for_callables(run_dir / "method", declared)
    except Exception:  # noqa: BLE001 — binding must never break the battery
        return
    if not resolved:
        return
    existing = list(verdict.element_ids or [])
    verdict.element_ids = existing + [i for i in resolved if i not in existing]


def _live_battery_version() -> str:
    """The checkout's commit for the report version stamp, best-effort.

    Returns "unversioned" when git is unavailable (the vendored layout has
    no repository; its stamp comes from the freeze file instead)."""
    import subprocess  # noqa: PLC0415
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent, capture_output=True,
            text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return "unversioned"
    commit = out.stdout.strip()
    return commit if out.returncode == 0 and commit else "unversioned"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--report", help="verdict JSON path "
                        "(default <run>/.pipeline/probe_report.json)")
    parser.add_argument("--output-dir", help="isolated output mode: write "
                        "the verdict report AND the CT-3 claims ledger into "
                        "this directory instead of the run's .pipeline/ — "
                        "the run dir stays byte-identical (post-delivery "
                        "re-validation must diff against delivery-time "
                        "truth, never overwrite it)")
    parser.add_argument("--frozen-gating", help="path to a frozen probe-"
                        "gating file built at delivery time (the vendored-"
                        "harness contract): the battery uses ITS declared "
                        "probe context instead of loading the taxonomy, so "
                        "an outside-repository run reproduces the delivered "
                        "check set exactly, never a fallback superset")
    parser.add_argument("--n-classes", type=int, default=None,
                        help="chance-level override for UB-6")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"FAIL: run dir not found: {run_dir}", file=sys.stderr)
        return 2

    frozen_gating = None
    if args.frozen_gating:
        frozen_path = Path(args.frozen_gating)
        if not frozen_path.is_file():
            print(f"FAIL: frozen gating file not found: {frozen_path}",
                  file=sys.stderr)
            return 2
        try:
            frozen_gating = json.loads(
                frozen_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"FAIL: frozen gating file unparseable: {e}",
                  file=sys.stderr)
            return 2

    output_dir = Path(args.output_dir) if args.output_dir else None
    ledger_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = output_dir / "claims_ledger.json"

    report = run_battery(run_dir, n_classes=args.n_classes,
                         ledger_path=ledger_path,
                         frozen_gating=frozen_gating)

    if args.report:
        report_path = Path(args.report)
    elif output_dir is not None:
        report_path = output_dir / "probe_report.json"
    else:
        report_path = run_dir / ".pipeline" / "probe_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    # Version stamp (R2C-019): two probe reports are comparable only when a
    # difference is attributable — code change versus battery change. The
    # vendored harness carries its build commit in the freeze file; in-repo
    # runs stamp the live checkout's commit best-effort.
    payload["battery_version"] = (
        frozen_gating.get("battery_version") if frozen_gating else None
    ) or _live_battery_version()
    report_path.write_text(json.dumps(payload, indent=2) + "\n",
                           encoding="utf-8")

    counts = report.counts()
    print(f"probe battery over {run_dir}")
    print("  " + "  ".join(f"{k}={v}" for k, v in counts.items() if v))
    for v in report.verdicts:
        if v.verdict in ("fail", "flag_for_researcher"):
            print(f"  {v.verdict.upper()} [{v.probe_id}] {v.message[:110]}")
        elif v.verdict == "unprobeable":
            print(f"  unprobeable [{v.probe_id}] {v.message[:90]}")
    print(f"report: {report_path}")
    return 1 if report.gating_failures else 0


if __name__ == "__main__":
    sys.exit(main())
