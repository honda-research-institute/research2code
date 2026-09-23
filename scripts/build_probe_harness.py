"""Build the portable check harness for a delivered run (R2C-019).

Vendoring is a delivery-time BUILD step, not a copy. The adversarial
review of the harness design proved all three failure modes this script
exists to prevent: a copied battery dies at import time on repo-internal
imports, a battery that merely dropped those imports would silently run a
DIFFERENT probe set than the delivered report (losing the taxonomy loader
flips the runner into its run-everything fallback), and the battery used
to write into the run's internal artifacts (fixed by the runner's
isolated output mode, which this harness always uses).

What the build does:

1. Copies the probe runner, the probes package, and the two
   dependency-free helpers it needs (the lint gate and the params
   provenance validator with its name-anchor vocabulary) into a
   self-contained directory.
2. Freezes the run's node-declared probe gating into frozen_gating.json,
   so the vendored battery reproduces the delivered check set exactly. A
   paradigm no node serves freezes as null, which round-trips as the
   full-battery fallback that run actually delivered.
3. Stamps the battery version (this checkout's commit) into the freeze
   file and the manifest, so any two probe reports are comparable and a
   difference is attributable to code change versus battery change.
4. Writes the researcher-facing entrypoint (run_checks.py), a README
   that enumerates which checks apply to THIS run, a requirements file,
   and a manifest with per-file digests.

Usage:

    python3 scripts/build_probe_harness.py --run-dir r2c_runs/<slug> \
        [--out <dir>]          # default <run>/validation/harness

Exit codes: 0 built, 2 setup error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_probes import (  # noqa: E402
    _live_setup_boundary,
    _live_training_history_required,
    _node_declared_probe_context,
    _paradigm_and_pluggable,
    _scenario_dimension_bindings,
)

FROZEN_GATING_SCHEMA_VERSION = "1.2"

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

# Repo file -> harness-relative destination. The probes package is copied
# as a tree separately.
_VENDORED_FILES = {
    "scripts/run_probes.py": "run_probes.py",
    "scripts/lint_generated_code.py": "lint_generated_code.py",
    "scripts/validate_params_provenance.py": "validate_params_provenance.py",
    "schemas/param_text.py": "param_text.py",
    "scripts/harness_preflight.py": "harness_preflight.py",
    # The spec-conditioned disclosure (R2C-047). A researcher re-running the
    # checks on their own machine must get the same three-way adjudication the
    # delivered report carried, and the module is pure stdlib over the run's
    # own method_spec.json, so it vendors cleanly.
    "scripts/probe_spec_join.py": "probe_spec_join.py",
    # R2C-092's derived graph-mechanism wiring. The join above merges the
    # derived graph refs, so the portable battery must resolve ownership
    # from the same single derivation surface. Stdlib-only by design.
    "schemas/graph_wiring.py": "graph_wiring.py",
    # R2C-086's stdlib-only comparison primitive.  The held-out-skill probe
    # must recompute the frozen comparison rather than trust verdict prose.
    "scripts/demo_skill_evidence.py": "demo_skill_evidence.py",
    # The `# paper-element:` scan (R2C-072). The battery stamps a verdict with
    # the paper elements the callables it exercised implement, which is what
    # lets the contract adjudicate the finding. A researcher re-running the
    # checks must reach the same adjudication, so the scan vendors alongside
    # the join it feeds. Stdlib-only and dependency-free by design.
    "scripts/paper_element_anchors.py": "paper_element_anchors.py",
    # Dependency-light replay of the schema-2 fixture plan frozen below.  The
    # portable harness never re-resolves Pydantic contracts or guesses
    # constructor/input shapes.
    "scripts/typed_fixture.py": "typed_fixture.py",
    # R2C-088 binds direct graph-helper probe behavior to the exact generated
    # call paths that the delivered notebook and Stage 2.d runtime exercised.
    # The portable runner rechecks the receipt against the same source bytes.
    "scripts/graph_callable_liveness.py": "graph_callable_liveness.py",
    # UB-9 reads only the pipeline-owned R2C-090 artifact.  Vendor the closed
    # validator and its R2C-089 state validator so the portable battery proves
    # the same structured record instead of falling back to notebook prose.
    "scripts/time_series_training_history.py": (
        "scripts/time_series_training_history.py"
    ),
    "scripts/time_series_target_scaling.py": (
        "scripts/time_series_target_scaling.py"
    ),
}

# Hardware-statement scan (the note 6 quote-or-absence rule): the block
# carries the paper's OWN sentence or nothing. Specific hardware model
# tokens always qualify; bare GPU/CPU mentions qualify only in an
# experiment-context sentence, so a related-work aside ("GPU-accelerated
# methods exist") never ships as this paper's hardware statement.
_HW_MODEL_RE = re.compile(
    r"\b(NVIDIA|GeForce|GTX|RTX|Quadro|TITAN|[VAH]100|TPU[s]?|"
    r"Xeon|Ryzen|i[3579][- ]\d{4,5}\w*|EPYC|Apple M\d)\b")
_HW_GENERIC_RE = re.compile(r"\b(GPU[s]?|CPU[s]?|RAM)\b")
_HW_CONTEXT_RE = re.compile(
    r"\b(train|trained|training|experiment|implementation|implemented|"
    r"evaluate[d]?|runs?|took|takes|hours?|minutes?|ms\b|seconds?)\b",
    re.IGNORECASE)


def _nearest_header(text: str, pos: int) -> str | None:
    headers = re.findall(r"^#{1,6}\s+(.+)$", text[:pos], re.MULTILINE)
    if not headers:
        return None
    header = headers[-1].strip().rstrip("#").strip()
    if re.search(r"section|appendix|chapter", header, re.IGNORECASE):
        return header
    return f"Section {header}"


def paper_hardware_statement(paper_text: str | None) -> dict | None:
    """The first sentence in which the paper states its hardware, verbatim
    with a section locator, or None (an honest absence). Never a
    paraphrase and never an invented figure."""
    if not paper_text:
        return None
    for match in re.finditer(r"[^.!?\n]{10,400}[.!?]", paper_text):
        sentence = match.group(0)
        model_hit = _HW_MODEL_RE.search(sentence)
        generic_hit = _HW_GENERIC_RE.search(sentence)
        if not model_hit and not (generic_hit
                                  and _HW_CONTEXT_RE.search(sentence)):
            continue
        cleaned = " ".join(sentence.split()).strip()
        return {"text": cleaned, "section": _nearest_header(
            paper_text, match.start())}
    return None


def _build_hardware_block(run_dir: Path) -> dict:
    """Quotes and structured spec fields only (no inventions): the compute
    class and time estimate the analyzer recorded, plus the paper's own
    hardware sentence when one exists."""
    spec_path = run_dir / ".pipeline" / "method_spec.json"
    compute_class = None
    estimated_time = None
    if spec_path.is_file():
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            deps = spec.get("dependencies") or {}
            compute_class = deps.get("compute")
            estimated_time = deps.get("estimated_time")
        except json.JSONDecodeError:
            pass
    paper_path = run_dir / ".pipeline" / "paper.md"
    paper_text = None
    if paper_path.is_file():
        try:
            paper_text = paper_path.read_text(encoding="utf-8")
        except OSError:
            pass
    return {
        "compute_class": compute_class,
        "estimated_time": estimated_time,
        "paper_statement": paper_hardware_statement(paper_text),
    }


def _frozen_demo_skill_evidence(run_dir: Path) -> dict | None:
    """Freeze the exact R2C-086 inputs, including typed role resolution.

    Runs without structured demo evidence return null.  A malformed or
    incomplete schema-2 artifact freezes as an explicit unresolved payload so
    harness construction never crashes and a portable TSF-4 replay cannot
    silently treat missing role identity as held-out evidence.
    """
    path = run_dir / ".pipeline" / "demo_verdict.json"
    if not path.is_file():
        return None
    try:
        verdict = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "schema_version": "1.0.0",
            "status": "unresolved",
            "reasons": ["demo_verdict_unreadable"],
            "family_contract": None,
            "executed_record": None,
            "validity_receipt": None,
            "evaluation_protocol_role": None,
        }
    if not isinstance(verdict, dict) or not str(
        verdict.get("schema_version") or ""
    ).startswith("2."):
        return None
    if verdict.get("decided_by") not in {
        "structured_demo_skill", "structured_demo_skill_error",
    }:
        return None

    contract = verdict.get("demo_skill_contract")
    record = verdict.get("executed_evaluation")
    receipt = verdict.get("evaluation_validity_receipt")
    verdict_carrier = verdict.get("evaluation_protocol_role")
    receipt_carrier = receipt.get("evaluation_protocol_role") \
        if isinstance(receipt, dict) else None
    carrier = receipt_carrier if isinstance(receipt_carrier, dict) else None
    reasons = list(receipt.get("reasons") or []) \
        if isinstance(receipt, dict) else []
    missing = []
    if not isinstance(contract, dict):
        missing.append("demo_skill_contract_missing")
    if not isinstance(record, dict):
        missing.append("executed_evaluation_record_missing")
    if not isinstance(receipt, dict):
        missing.append("evaluation_validity_receipt_missing")
    if isinstance(record, dict) and record.get("schema_version") == "2.0.0" \
            and not isinstance(carrier, dict):
        missing.append("executed_evaluation_protocol_role_unresolved")
    if isinstance(carrier, dict) and (
        not isinstance(verdict_carrier, dict) or verdict_carrier != carrier
    ):
        missing.append("evaluation_protocol_role_carrier_mismatch")
        carrier = None
    if isinstance(record, dict) and record.get("schema_version") == "2.0.0" \
            and isinstance(carrier, dict) \
            and record.get("evaluation_protocol_role") != carrier.get("role"):
        missing.append("evaluation_protocol_role_carrier_mismatch")
        carrier = None
    for reason in missing:
        if reason not in reasons:
            reasons.append(reason)

    return {
        "schema_version": "1.0.0",
        "status": "ready" if not missing else "unresolved",
        "reasons": reasons,
        "family_contract": contract if isinstance(contract, dict) else None,
        "executed_record": record if isinstance(record, dict) else None,
        "validity_receipt": receipt if isinstance(receipt, dict) else None,
        "evaluation_protocol_role": carrier if isinstance(carrier, dict) else None,
    }


def _frozen_forecasting_execution_plan(
    run_dir: Path,
    declared_context: dict | None,
) -> dict | None:
    """Freeze one exact schema-2 TSF plan, or a named coverage result.

    The build machine may resolve Pydantic contracts and bundle facts.  The
    researcher's harness receives only JSON-normalized construction, typed
    fixtures, closed output grammar, applicability, and R2C-084 relational
    identity tables.  An archived schema-1 run or unsupported relational
    grammar remains runnable: every declared TSF method probe reports the
    frozen unresolved reason instead of attempting a legacy adapter.
    """

    from probes.time_series_forecasting import PROBE_REFS  # noqa: PLC0415

    if not set(PROBE_REFS).intersection(set(declared_context or {})):
        return None
    try:
        from scripts.arch_contract_runtime_plan import (  # noqa: PLC0415
            normalize_schema2_forecasting_plan,
        )
        from scripts.build_plan import load_build_plan  # noqa: PLC0415
        from scripts.taxonomy import run_overlay_dir  # noqa: PLC0415
        from scripts.validate_arch_contract_runtime import (  # noqa: PLC0415
            _load_typed_bundle_dimension_facts,
        )

        spec = json.loads((run_dir / ".pipeline" / "method_spec.json").read_text(
            encoding="utf-8"
        ))
        contract = json.loads((
            run_dir / ".pipeline" / "arch_contract.json"
        ).read_text(encoding="utf-8"))
        plan = load_build_plan(
            spec,
            ROOT,
            provisional_packs_dir=run_overlay_dir(run_dir),
        )
        if plan is None:
            raise ValueError("no exact forecasting build plan")
        return normalize_schema2_forecasting_plan(
            contract,
            plan,
            bundle_facts=_load_typed_bundle_dimension_facts(run_dir),
            method_spec=spec,
        )
    except Exception as exc:  # noqa: BLE001 - freeze the coverage gap
        return {
            "schema_version": "1.0",
            "status": "unresolved",
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _frozen_forecasting_probe_groundings(
    run_dir: Path,
    declared_context: dict | None,
) -> dict[str, dict[str, list[str]]] | None:
    """Freeze exact-ref methodology identities independently of runtime."""

    from probes.time_series_forecasting import PROBE_REFS  # noqa: PLC0415

    if not set(PROBE_REFS).intersection(set(declared_context or {})):
        return None
    try:
        from scripts.arch_contract_runtime_plan import (  # noqa: PLC0415
            normalize_forecasting_probe_groundings,
        )

        spec = json.loads((run_dir / ".pipeline" / "method_spec.json").read_text(
            encoding="utf-8"
        ))
        return normalize_forecasting_probe_groundings(spec)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _stage_2d_alignment_receipt(
    run_dir: Path,
    plan: dict,
) -> dict:
    """Bind R2C-084 runtime success to the exact validated source bytes.

    Normalizing the relational contract proves that its declarations are
    internally resolvable; it does not prove the generated preparation,
    fitting, and inference callables honored them.  Stage 2.d owns that
    runtime proof.  Its completion sentinel is trustworthy only while the
    paired upstream digest still matches the method sources and contracts.
    """

    base = {
        "receipt_version": "1.0",
        "probe_ref": "graph_mechanism.alignment_prerequisite",
        "status": "unprobeable",
        "verified": False,
        "reason": "stage_2d_runtime_validation_receipt_missing",
        "validator": "validate_arch_contract_runtime.py",
        "element_id": None,
        "fixture_scope": None,
        "representation": plan.get("representation"),
        "entity_ids": [],
        "authority_digests": {},
    }
    alignment = plan.get("alignment")
    fixture = plan.get("fixture")
    if isinstance(alignment, dict):
        base["element_id"] = alignment.get("element_id")
    if isinstance(fixture, dict):
        base["fixture_scope"] = fixture.get("scope")
        entity_ids = fixture.get("entity_ids")
        if not isinstance(entity_ids, list):
            source = fixture.get("source")
            if not isinstance(source, dict):
                alignment_source = fixture.get("alignment_source")
                source = (
                    alignment_source.get("source")
                    if isinstance(alignment_source, dict) else None
                )
            entity_ids = (
                source.get("stable_entity_ids")
                if isinstance(source, dict) else None
            )
        if isinstance(entity_ids, list):
            base["entity_ids"] = list(entity_ids)

    pipeline_dir = run_dir / ".pipeline"
    sentinel = pipeline_dir / "stage_2d.complete"
    digest_path = pipeline_dir / "stage_2d.upstream_digest.json"
    if not sentinel.is_file():
        return base
    if not digest_path.is_file():
        base["reason"] = "stage_2d_runtime_validation_digest_missing"
        return base
    try:
        stored = json.loads(digest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        base["reason"] = "stage_2d_runtime_validation_digest_unreadable"
        return base
    if not isinstance(stored, dict) or not stored or any(
        not isinstance(relative, str)
        or not isinstance(digest, str)
        or not digest.startswith("sha256:")
        for relative, digest in stored.items()
    ):
        base["reason"] = "stage_2d_runtime_validation_digest_malformed"
        return base
    try:
        # Reuse the driver's exact smart-skip authority rather than creating a
        # second definition of which Stage 2.d inputs invalidate validation.
        from scripts.run_pipeline import (  # noqa: PLC0415
            _compute_stage_2d_upstream_digest,
        )

        current = _compute_stage_2d_upstream_digest(SimpleNamespace(
            run_dir=run_dir,
            pipeline_dir=pipeline_dir,
        ))
    except (OSError, TypeError, ValueError) as exc:
        base["reason"] = (
            "stage_2d_runtime_validation_digest_unavailable:"
            f"{type(exc).__name__}"
        )
        return base
    if stored != current:
        base["reason"] = "stage_2d_runtime_validation_authority_drift"
        return base

    base.update({
        "status": "pass",
        "verified": True,
        "reason": "stage_2d_runtime_alignment_verified",
        "authority_digests": dict(sorted(stored.items())),
    })
    return base


def _stored_graph_liveness_authority(
    run_dir: Path,
    stage_id: str,
    current: dict[str, str],
) -> tuple[dict[str, str] | None, str | None]:
    """Read one existing stage receipt and reject missing or drifted bytes."""

    pipeline_dir = run_dir / ".pipeline"
    if not (pipeline_dir / f"{stage_id}.complete").is_file():
        return None, f"graph_callable_liveness_{stage_id}_receipt_missing"
    digest_path = pipeline_dir / f"{stage_id}.upstream_digest.json"
    try:
        stored = json.loads(digest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"graph_callable_liveness_{stage_id}_digest_missing"
    except (OSError, json.JSONDecodeError):
        return None, f"graph_callable_liveness_{stage_id}_digest_unreadable"
    if not isinstance(stored, dict) or not stored or any(
        not isinstance(relative, str)
        or not relative
        or not isinstance(digest, str)
        or len(digest) != 71
        or not digest.startswith("sha256:")
        or any(
            character not in "0123456789abcdef"
            for character in digest.removeprefix("sha256:")
        )
        for relative, digest in stored.items()
    ):
        return None, f"graph_callable_liveness_{stage_id}_digest_malformed"
    if stored != current:
        return None, f"graph_callable_liveness_{stage_id}_authority_drift"
    return dict(sorted(stored.items())), None


def _graph_callable_liveness_receipt(
    run_dir: Path,
    plan: dict,
    spec: dict,
    contract: dict,
) -> dict:
    """Freeze live-use proof against existing Stage 2.d/3.c authority."""

    from scripts.graph_callable_liveness import (  # noqa: PLC0415
        RECEIPT_VERSION,
        GraphCallableCoverageError,
        GraphCallableProducerError,
        notebook_code_cells,
        prove_constructor_notebook_flow,
        prove_message_live_path,
        stage_2d_authority_digests,
        stage_3c_authority_digests,
    )

    base = {
        "receipt_version": RECEIPT_VERSION,
        "status": "unprobeable",
        "verified": False,
        "reason": "graph_callable_liveness_unverified",
        "construction": {
            "status": "unprobeable",
            "reason": "graph_constructor_liveness_unverified",
        },
        "message_passing": {
            "status": "unprobeable",
            "reason": "graph_message_liveness_unverified",
        },
        "authority_digests": {"stage_2d": {}, "stage_3c": {}},
    }
    alignment_receipt = plan.get("alignment_runtime_receipt")
    if not isinstance(alignment_receipt, dict) \
            or alignment_receipt.get("verified") is not True:
        base["reason"] = (
            str(alignment_receipt.get("reason"))
            if isinstance(alignment_receipt, dict)
            and alignment_receipt.get("reason")
            else "stage_2d_runtime_validation_receipt_missing"
        )
        return base

    try:
        current_authority = {
            "stage_2d": stage_2d_authority_digests(run_dir),
            "stage_3c": stage_3c_authority_digests(run_dir),
        }
    except OSError as exc:
        base["reason"] = (
            "graph_callable_liveness_authority_unreadable:"
            f"{type(exc).__name__}"
        )
        return base
    for stage_id in ("stage_2d", "stage_3c"):
        stored, reason = _stored_graph_liveness_authority(
            run_dir, stage_id, current_authority[stage_id]
        )
        if reason is not None:
            base["reason"] = reason
            return base
        base["authority_digests"][stage_id] = stored

    try:
        construction = prove_constructor_notebook_flow(
            notebook_code_cells(run_dir), spec, contract, run_dir / "method"
        )
        base["construction"] = construction
    except GraphCallableProducerError as exc:
        base["status"] = "fail"
        base["reason"] = exc.code
        base["construction"] = {"status": "fail", "reason": exc.code}
        return base
    except GraphCallableCoverageError as exc:
        base["reason"] = exc.code
        base["construction"] = {
            "status": "unprobeable",
            "reason": exc.code,
        }
        return base

    try:
        message = prove_message_live_path(spec, contract, run_dir / "method")
        base["message_passing"] = message
    except GraphCallableProducerError as exc:
        base["status"] = "fail"
        base["reason"] = exc.code
        base["message_passing"] = {"status": "fail", "reason": exc.code}
        return base
    except GraphCallableCoverageError as exc:
        base["reason"] = exc.code
        base["message_passing"] = {
            "status": "unprobeable",
            "reason": exc.code,
        }
        return base

    base.update({
        "status": "pass",
        "verified": True,
        "reason": "graph_callable_liveness_verified",
    })
    return base


def _frozen_graph_mechanism_execution_plan(
    run_dir: Path,
    declared_context: dict | None,
) -> dict | None:
    """Freeze one exact homogeneous-graph plan and its alignment receipt."""

    try:
        from scripts.graph_mechanism_runtime_plan import (  # noqa: PLC0415
            ALIGNMENT_PROBE_REF,
            CANONICAL_PROBE_REFS,
            normalize_graph_mechanism_runtime_plan,
        )
    except ImportError as exc:
        return {
            "schema_version": "1.0",
            "status": "unprobeable",
            "reason": f"graph_plan_import_failed:{type(exc).__name__}: {exc}",
        }
    graph_refs = {ALIGNMENT_PROBE_REF, *CANONICAL_PROBE_REFS.values()}
    if not graph_refs.intersection(set(declared_context or {})):
        return None
    try:
        pipeline_dir = run_dir / ".pipeline"
        spec = json.loads((pipeline_dir / "method_spec.json").read_text(
            encoding="utf-8"
        ))
        params = json.loads((pipeline_dir / "params.json").read_text(
            encoding="utf-8"
        ))
        contract = json.loads((pipeline_dir / "arch_contract.json").read_text(
            encoding="utf-8"
        ))
        plan = normalize_graph_mechanism_runtime_plan(spec, params, contract)
        if not isinstance(plan, dict):
            raise TypeError("graph plan normalizer did not return a mapping")

        receipt = _stage_2d_alignment_receipt(run_dir, plan)
        plan["alignment_runtime_receipt"] = receipt
        alignment = plan.get("alignment")
        if receipt["verified"] and isinstance(alignment, dict):
            # Static normalization deliberately stops at ready.  Only the
            # exact Stage 2.d runtime receipt promotes HG-1 to pass.
            alignment["status"] = "pass"
            alignment["reason"] = None
        if plan.get("status") == "ready":
            plan["callable_liveness_receipt"] = (
                _graph_callable_liveness_receipt(
                    run_dir, plan, spec, contract
                )
            )
        json.dumps(plan, allow_nan=False)
        return plan
    except Exception as exc:  # noqa: BLE001 - freeze the coverage gap
        return {
            "schema_version": "1.0",
            "status": "unprobeable",
            "reason": f"{type(exc).__name__}: {exc}",
        }


_ENTRYPOINT = '''\
#!/usr/bin/env python3
"""Run this delivery's frozen check battery — portable entrypoint.

Executes the same probe set that produced the delivered report, against
this run's shipped artifacts, and writes its results into a SEPARATE
output directory. Delivery-time evidence under .pipeline/ is never
modified; compare the new report against the delivered one to see what an
edit changed.

    python3 run_checks.py [--run-dir <run>] [--output-dir <dir>]

Defaults: the run directory this harness ships inside, and
<run>/validation/results/ for output. Requires numpy; some checks report
themselves unprobeable without torch instead of failing.
"""

import sys
from pathlib import Path

# The harness ships inside the run's own git repository; bytecode caches
# would litter its status during companion sessions, so never write them.
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the frozen validation battery for this delivery.")
    parser.add_argument("--run-dir", default=str(HERE.parent.parent),
                        help="the delivered run directory "
                             "(default: the one this harness ships inside)")
    parser.add_argument("--output-dir", default=None,
                        help="where results go "
                             "(default <run>/validation/results)")
    parser.add_argument("--n-classes", type=int, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or str(
        Path(args.run_dir) / "validation" / "results")

    # Capability preflight first: quoted paper-scale requirements (or an
    # honest absence), this machine's measurements, and a loud refusal of
    # paper-scale expectations the machine cannot meet. The demo-scale
    # checks below always proceed.
    from harness_preflight import run_preflight
    print(run_preflight(HERE, Path(output_dir)))
    print()

    from run_probes import main as battery_main
    return battery_main([
        "--run-dir", args.run_dir,
        "--output-dir", output_dir,
        "--frozen-gating", str(HERE / "frozen_gating.json"),
    ])


if __name__ == "__main__":
    sys.exit(main())
'''

_REQUIREMENTS = """\
# Required by the check battery's fixtures and probes.
numpy
# Optional: the acquisition-loop and training probes use torch when
# present and report themselves unprobeable when it is absent.
"""


def _battery_version() -> str:
    """This checkout's commit, with a -dirty marker when the vendored
    sources have uncommitted changes ("unversioned" without git)."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10)
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "scripts", "schemas"],
            cwd=ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return "unversioned"
    commit = head.stdout.strip()
    if head.returncode != 0 or not commit:
        return "unversioned"
    if status.returncode == 0 and status.stdout.strip():
        return commit + "-dirty"
    return commit


def _training_history_authority_digests(
    run_dir: Path,
) -> dict[str, str | None]:
    """Freeze the closed delivery-time authority behind history evidence."""
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
        path = root / relative
        try:
            result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            result[relative] = None
    return result


def _render_readme(paradigm: str, declared_context: dict | None,
                   battery_version: str) -> str:
    lines = [
        "# Validation harness for this delivery",
        "",
        "This directory is a frozen, self-contained copy of the check",
        "battery that produced the delivered report. Run it any time —",
        "including after you edit the delivered code — to re-check the run",
        "without touching the delivery-time evidence:",
        "",
        "    python3 run_checks.py",
        "",
        "Results land in `validation/results/` next to this directory",
        "(`--output-dir` overrides). The delivered report and claims ledger",
        "under `.pipeline/` are never modified, so every re-run stays",
        "comparable against the delivery-time baseline.",
        "",
        "Requires Python 3.10+ with numpy (`pip install -r",
        "requirements.txt`). Checks that need torch report themselves as",
        "unprobeable when it is absent instead of failing.",
        "",
        f"Battery version: `{battery_version}` (the pipeline commit this",
        "harness was frozen at; every report it writes carries the same",
        "stamp, so differing results are attributable to code changes,",
        "never to a drifted check set).",
        "",
        "## Checks frozen for this run",
        "",
    ]
    if declared_context is None:
        lines += [
            f"This run's method family (`{paradigm or 'unknown'}`) had no",
            "curated check declaration at delivery time, so the battery",
            "runs its full generic set, exactly as the delivered report",
            "did.",
        ]
    else:
        lines += [
            f"The `{paradigm}` family declared these behavioral checks at",
            "delivery time. The battery runs exactly this set plus the",
            "universal static tiers (lint, parameter provenance, claims",
            "ledger, executed-notebook sanity).",
            "",
        ]
        for ref in sorted(declared_context):
            ctx = declared_context[ref] or {}
            check_id = ctx.get("id") or ref
            check = (ctx.get("check") or "").strip()
            lines.append(f"- **{check_id}** (`{ref}`)"
                         + (f": {check}" if check else ""))
    lines.append("")
    return "\n".join(lines)


def build_harness(run_dir: Path, out_dir: Path) -> dict:
    """Build the harness; returns the manifest dict."""
    paradigm, _ = _paradigm_and_pluggable(run_dir)
    declared_context = _node_declared_probe_context(paradigm)
    battery_version = _battery_version()

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    for src_rel, dst_rel in _VENDORED_FILES.items():
        destination = out_dir / dst_rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / src_rel, destination)
    shutil.copytree(
        ROOT / "scripts" / "probes", out_dir / "probes",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    training_history_required = _live_training_history_required(run_dir)
    frozen = {
        "schema_version": FROZEN_GATING_SCHEMA_VERSION,
        "battery_version": battery_version,
        "source_run": run_dir.name,
        "paradigm": paradigm,
        "declared_context": declared_context,
        # Exact inputs for the stdlib-only R2C-086 primitive.  Schema-1
        # records remain readable but carry no typed role proof; schema-2
        # records can prove a role only through the validated method spec.
        "demo_skill_evidence": _frozen_demo_skill_evidence(run_dir),
        "forecasting_execution_plan": _frozen_forecasting_execution_plan(
            run_dir, declared_context
        ),
        "forecasting_probe_groundings": (
            _frozen_forecasting_probe_groundings(run_dir, declared_context)
        ),
        # R2C-088 graph probes share one normalized execution plan in live and
        # portable paths.  HG-1 is promoted only by the exact Stage 2.d
        # runtime-validation receipt; the vendored runner rechecks its source
        # digests before allowing any mechanism evidence.
        "graph_mechanism_execution_plan": (
            _frozen_graph_mechanism_execution_plan(run_dir, declared_context)
        ),
        # Freeze applicability as well as implementation.  A future taxonomy
        # checkout must not add or remove UB-9 from a delivered battery.
        "training_history_required": training_history_required,
        # The portable battery cannot mint a new pipeline-owned post-smoke
        # record.  It may reuse UB-9 evidence only while the exact bounded
        # source/data/config/artifact authority remains identical to delivery.
        "training_history_authority_digests": (
            _training_history_authority_digests(run_dir)
            if training_history_required else None
        ),
        # Scenario-fidelity slice B: the setup-section boundary the scenario
        # detectors slice the notebook by. Frozen because the vendored
        # battery cannot resolve the taxonomy layout outside the repository;
        # null when the family declares no scenario dimensions.
        "scenario_setup_boundary": (
            _live_setup_boundary(run_dir)
            if _scenario_dimension_bindings(declared_context) else None
        ),
    }
    (out_dir / "frozen_gating.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (out_dir / "hardware_block.json").write_text(
        json.dumps(_build_hardware_block(run_dir), indent=2,
                   sort_keys=True) + "\n",
        encoding="utf-8")
    (out_dir / "run_checks.py").write_text(_ENTRYPOINT, encoding="utf-8")
    (out_dir / "requirements.txt").write_text(_REQUIREMENTS,
                                              encoding="utf-8")
    (out_dir / "README.md").write_text(
        _render_readme(paradigm, declared_context, battery_version),
        encoding="utf-8")

    files = {
        p.relative_to(out_dir).as_posix():
            hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out_dir.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    manifest = {
        "battery_version": battery_version,
        "source_run": run_dir.name,
        "paradigm": paradigm,
        "files": files,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default=None,
                        help="harness destination "
                             "(default <run>/validation/harness)")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"FAIL: run dir not found: {run_dir}", file=sys.stderr)
        return 2
    out_dir = Path(args.out) if args.out else (
        run_dir / "validation" / "harness")

    manifest = build_harness(run_dir, out_dir)
    print(f"built harness at {out_dir}")
    print(f"  battery_version: {manifest['battery_version']}")
    print(f"  paradigm: {manifest['paradigm'] or '(none)'}")
    print(f"  files: {len(manifest['files'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
