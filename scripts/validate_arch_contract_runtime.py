"""Stage 2.d runtime dry-run validator for the architecture contract.

Runs after `validate_arch_contract.py` (which verifies structure). This
validator does the part the static check can't: actually instantiate the
classes the contract declares, call their forward passes with test
tensors constructed from the contract's shape strings, and surface any
RuntimeError as an "architecture-coder's contract is inconsistent with its
code" halt.

This is the single check that would have caught all three bev-distill
smoke failures before Stage 3.c:
  - `compute_3d_box_iou` indexing col 8 on 8-col input: dry-run call with
    (B, max_objects, 9) gt_boxes → IndexError surfaces here, not at smoke.
  - Notebook 2D data → model expects 4D: dry-run uses the contract's
    declared (B, 3, H, W) for student input → if model.forward can't
    accept that, halt now.
  - Teacher backbone Linear on 3D point cloud: dry-run calls teacher
    with (B, N_points, 3) per the contract → RuntimeError surfaces here.

Strategy

  1. Read `.pipeline/arch_contract.json` (validated by the prior step).
  2. Resolve constructor arguments and fixtures before importing generated
     code. Version 2 uses semantic identities, checked expression trees, exact
     bundle facts, containers, and dtypes through one shared resolver. Version
     1.1 retains its declared constructor resolver; legacy 1.0 shape strings
     retain their historical binding ladder.
  3. Spawn a subprocess (`cwd=run_dir`) that imports the method package,
     instantiates each architecture block via its builder
     (`build_student`, `build_teacher`, `build_model`, etc. — discovered
     from method/training.py's top-level functions), and calls forward
     with constructed test tensors.
  4. Any exception → fail with that exception's message in stderr.

The subprocess pattern matches `validate_package_imports.py` — keeps the
package's heavyweight imports out of the validator's process and matches
the conditions under which the notebook will run.

Deterministic gate. Failures use the ownership-aware exit protocol documented
below and print a structured error list on stderr.

Usage:

    python scripts/validate_arch_contract_runtime.py --spec <method_spec.json> --run-dir <output_dir>

Exit codes:
  0  validation passed
  1  producer-owned, mixed, or unstructured validation failures
  2  setup error
  3  exclusively pipeline-owned semantic coverage failures
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# The dry-run subprocess always runs with this env var set: template-derived
# data loaders refuse network downloads under it (raising a RuntimeError that
# names the var), and the runner treats that refusal like a missing optional
# dependency — synthetic-fixture retry, else a clean skip.
OFFLINE_ENV = "R2C_OFFLINE"
DRY_RUN_TIMEOUT_ENV = "R2C_DRY_RUN_TIMEOUT_S"
DEFAULT_DRY_RUN_TIMEOUT_S = 150.0
_PIPELINE_REACHABILITY_PREFIX = "pipeline_validator_coverage:"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from schemas.arch_contract import ArchContract  # noqa: E402
from schemas.arch_contract_v2 import ArchContractV2  # noqa: E402
from schemas.method_spec import (  # noqa: E402
    evaluation_protocol_identity_labels,
    evaluation_protocol_quantity_for_param,
)
from scripts.bundle_axis_floor import (  # noqa: E402
    ProtocolAxisUnresolved,
    protocol_quantity_steps,
)
from scripts.arch_contract_semantics import (  # noqa: E402
    AnyArchContract,
    SEMANTIC_ISSUE_PREFIX,
    SemanticIssue,
    bundle_dimension_facts,
    format_semantic_issue,
    load_arch_contract,
    resolve_contract,
    semantic_issue_exit_code,
)
from scripts.arch_contract_runtime_plan import (  # noqa: E402
    RuntimePlanError,
    normalize_schema2_forecasting_plan,
)
from scripts.build_plan import load_build_plan  # noqa: E402
from scripts.taxonomy import run_overlay_dir  # noqa: E402
from scripts.time_series_target_scaling import (  # noqa: E402
    TargetScalingCoverageError,
)
from scripts.time_series_training_history import (  # noqa: E402
    TrainingHistoryCoverageError,
)


def _legacy_untrusted_error(message: str) -> str:
    """Remove reserved ownership envelopes from legacy producer text."""
    if SEMANTIC_ISSUE_PREFIX in message:
        return (
            "generated legacy package emitted text resembling the reserved "
            "semantic-issue channel; treated as an unstructured producer failure"
        )
    return message


# Default small concrete values for symbolic dims. Used by the dry-run to
# construct test tensors. Any symbol not in this table falls back to 8
# (small enough to be cheap, large enough to expose most shape issues).
# Field guides may extend this via `arch_contract_requirements.symbol_conventions`
# in the future; for now, the universal defaults cover the paradigms we ship.
DRY_RUN_BINDINGS: dict[str, int] = {
    "B": 2,
    "N": 16,
    "N_test": 8,
    "N_pool": 16,
    "N_init": 4,
    "H": 32,
    "W": 32,
    "C": 3,
    "N_points": 16,
    "N_queries": 8,
    "n_classes": 3,
    "n_features": 8,
    "max_objects": 4,
    "hidden_dim": 8,
    "bev_channels": 8,
    "bev_h": 8,
    "bev_w": 8,
}
DEFAULT_BINDING = 8


# Shape strings matching `(SYM1, SYM2, ...)` — comma-separated tokens.
_SHAPE_TUPLE_RE = re.compile(r"^\(\s*([^)]*)\s*\)$")


def _parse_shape_tokens(shape_str: str) -> tuple[str, ...] | None:
    """Parse a tuple-shaped contract string without resolving symbols.

    Used for contract-to-contract consistency checks where symbol spelling can
    legitimately differ (`input_dim` vs `n_features`) but rank and concrete
    numeric axes must agree.
    """
    m = _SHAPE_TUPLE_RE.match(str(shape_str or "").strip())
    if not m:
        return None
    inner = m.group(1).strip()
    if not inner or inner == "opaque":
        return None
    return tuple(p.strip() for p in inner.split(",") if p.strip())


def _is_collection_axis(token: str) -> bool:
    t = token.strip()
    return t in {"B", "N", "batch", "batch_size"} or t.startswith("N_")


def _sample_shape_tokens(shape_str: str) -> tuple[str, ...] | None:
    tokens = _parse_shape_tokens(shape_str)
    if tokens is None:
        return None
    if tokens and _is_collection_axis(tokens[0]):
        return tokens[1:]
    return tokens


def _numeric_token(token: str) -> int | None:
    t = token.strip()
    if t.isdigit() or (t.startswith("-") and t[1:].isdigit()):
        return int(t)
    return None


def _sample_shapes_compatible(
    left: tuple[str, ...], right: tuple[str, ...]
) -> tuple[bool, str]:
    if len(left) != len(right):
        return (
            False,
            f"sample ranks differ ({len(left)} vs {len(right)})",
        )
    for idx, (a, b) in enumerate(zip(left, right), start=1):
        a_num = _numeric_token(a)
        b_num = _numeric_token(b)
        if a_num is not None and b_num is not None and a_num != b_num:
            return (
                False,
                f"sample axis {idx} has conflicting concrete sizes "
                f"({a_num} vs {b_num})",
            )
    return True, ""


def _normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _is_label_like_name(name: str) -> bool:
    n = _normalized_name(name)
    parts = set(n.split("_"))
    if n in {
        "y",
        "label",
        "labels",
        "target",
        "targets",
        "class",
        "classes",
        "n_classes",
        "num_classes",
        "seed",
        "batch_size",
    }:
        return True
    return bool(
        parts
        & {
            "y",
            "label",
            "labels",
            "target",
            "targets",
            "class",
            "classes",
            "idx",
            "index",
            "indices",
            "id",
            "ids",
            "mask",
        }
    )


def _is_feature_like_data_name(name: str) -> bool:
    if _is_label_like_name(name):
        return False
    n = _normalized_name(name)
    return n.startswith("x") or any(
        term in n
        for term in (
            "input",
            "image",
            "feature",
            "point",
            "token",
            "sample",
            "observation",
            "state",
            "bev",
        )
    )


def _is_generic_model_input_name(name: str) -> bool:
    return _normalized_name(name) in {
        "x",
        "input",
        "inputs",
        "data",
        "features",
        "samples",
        "image",
        "images",
        "tokens",
        "point_cloud",
        "point_clouds",
    }


def _data_shape_entries(contract: ArchContract) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for name, shape_str in contract.data_loader.load_data_returns.items():
        sample = _sample_shape_tokens(shape_str)
        if sample is None:
            continue
        entries.append(
            {
                "name": name,
                "normalized": _normalized_name(name),
                "path": f"data_loader.load_data_returns.{name}",
                "shape": shape_str,
                "sample": sample,
            }
        )
    return entries


def _coherent_feature_reference(
    entries: list[dict[str, Any]]
) -> dict[str, Any] | None:
    feature_entries = [
        entry for entry in entries if _is_feature_like_data_name(entry["name"])
    ]
    if not feature_entries:
        return None
    ref = feature_entries[0]
    for entry in feature_entries[1:]:
        ok, _ = _sample_shapes_compatible(ref["sample"], entry["sample"])
        if not ok:
            # Cross-modal loaders can legitimately expose different feature
            # shapes (e.g. image student + point-cloud teacher). In that case,
            # only exact/block-specific name matches are safe.
            return None
    return ref


def _candidate_data_entries(
    *,
    consumer_name: str,
    block_key: str | None,
    data_entries: list[dict[str, Any]],
    feature_ref: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    by_norm = {entry["normalized"]: entry for entry in data_entries}
    wanted: list[str] = [_normalized_name(consumer_name)]
    n = _normalized_name(consumer_name)

    aliases = {
        "x_unlabeled": ["x_pool", "x_unlabeled", "unlabeled", "unlabeled_inputs"],
        "x_candidates": ["x_pool", "x_unlabeled", "x_candidates"],
        "x_labeled": ["x_train", "x_labeled", "labeled", "labeled_inputs"],
        "x_train": ["x_train"],
        "x_test": ["x_test"],
        "y_train": ["y_train"],
        "y_test": ["y_test"],
    }
    wanted.extend(aliases.get(n, []))

    if block_key and _is_generic_model_input_name(consumer_name):
        b = _normalized_name(block_key)
        wanted.extend(
            [
                f"{b}_input",
                f"{b}_inputs",
                f"{b}_image",
                f"{b}_images",
                f"{b}_point_cloud",
                f"{b}_point_clouds",
                f"{b}_tokens",
                f"{b}_features",
            ]
        )

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in wanted:
        entry = by_norm.get(_normalized_name(key))
        if entry and entry["path"] not in seen:
            candidates.append(entry)
            seen.add(entry["path"])

    if candidates:
        return candidates
    if _is_generic_model_input_name(consumer_name) and feature_ref is not None:
        return [feature_ref]
    return []


def _check_consumer_shape(
    *,
    errors: list[str],
    consumer_path: str,
    consumer_name: str,
    consumer_shape: str,
    block_key: str | None,
    data_entries: list[dict[str, Any]],
    feature_ref: dict[str, Any] | None,
) -> None:
    consumer_sample = _sample_shape_tokens(consumer_shape)
    if consumer_sample is None:
        return
    for entry in _candidate_data_entries(
        consumer_name=consumer_name,
        block_key=block_key,
        data_entries=data_entries,
        feature_ref=feature_ref,
    ):
        ok, reason = _sample_shapes_compatible(entry["sample"], consumer_sample)
        if not ok:
            errors.append(
                f"arch_contract.{consumer_path}={consumer_shape!r} is "
                f"incompatible with arch_contract.{entry['path']}="
                f"{entry['shape']!r}: {reason}. Reconcile "
                "data_loader.load_data_returns with the declared consumer "
                "input shape."
            )


def _check_contract_shape_consistency(contract: ArchContract) -> list[str]:
    """Check that declared data-loader shapes agree with declared consumers.

    The subprocess dry-run verifies model code against architecture shapes and
    load_data code against data_loader shapes separately. This check closes the
    gap where both halves are internally valid but disagree with each other
    (for example, flat `(N, n_features)` data feeding an image-shaped
    `(B, 3, H, W)` model).
    """
    errors: list[str] = []
    data_entries = _data_shape_entries(contract)
    feature_ref = _coherent_feature_reference(data_entries)

    for block_key, block in contract.architecture.items():
        for name, shape_str in block.forward.input.items():
            _check_consumer_shape(
                errors=errors,
                consumer_path=f"architecture.{block_key}.forward.input.{name}",
                consumer_name=name,
                consumer_shape=shape_str,
                block_key=block_key,
                data_entries=data_entries,
                feature_ref=feature_ref,
            )
        for method_name, sig in block.additional_methods.items():
            for name, shape_str in sig.input.items():
                _check_consumer_shape(
                    errors=errors,
                    consumer_path=(
                        f"architecture.{block_key}.additional_methods."
                        f"{method_name}.input.{name}"
                    ),
                    consumer_name=name,
                    consumer_shape=shape_str,
                    block_key=block_key,
                    data_entries=data_entries,
                    feature_ref=feature_ref,
                )

    if contract.pluggable_component.input_shapes:
        for name, shape_str in contract.pluggable_component.input_shapes.items():
            _check_consumer_shape(
                errors=errors,
                consumer_path=f"pluggable_component.input_shapes.{name}",
                consumer_name=name,
                consumer_shape=shape_str,
                block_key=None,
                data_entries=data_entries,
                feature_ref=feature_ref,
            )

    if contract.training_loop:
        for name, shape_str in contract.training_loop.input_shapes.items():
            _check_consumer_shape(
                errors=errors,
                consumer_path=f"training_loop.input_shapes.{name}",
                consumer_name=name,
                consumer_shape=shape_str,
                block_key=None,
                data_entries=data_entries,
                feature_ref=feature_ref,
            )

    return errors


def _load_bindings_from_params(run_dir: Path) -> dict[str, int]:
    """Read params.json (if present) and extract any int-valued entries.
    Params override DRY_RUN_BINDINGS where the symbol names match — so the
    dry-run tests against actual smoke values when params.json declares
    them. Missing params.json is fine; we fall back to DRY_RUN_BINDINGS."""
    params_path = run_dir / ".pipeline" / "params.json"
    if not params_path.is_file():
        return {}
    try:
        raw = json.loads(params_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    out: dict[str, int] = {}
    # params.json's shape: {param_name: {"value": ..., "source": ..., "reasoning": ...}}.
    for k, v in (raw or {}).items():
        val = v.get("value") if isinstance(v, dict) else v
        if isinstance(val, int) and not isinstance(val, bool):
            out[k] = val
        # Special remappings for common params → symbol names.
        if k == "batch_size" and isinstance(val, int):
            out["B"] = val
    return out


# Legacy schema-1.0 spellings that meant "how long is the time axis". Current
# constructor contracts use the single semantic name `time_axis_steps`; this
# alias set remains only so archived shape strings keep their prior behavior.
_TIME_AXIS_SYMBOLS = frozenset({
    "T", "T_total", "T_max", "T_steps", "n_steps", "num_steps", "n_time_steps",
    "num_time_steps", "time_steps", "seq_len", "seq_length", "sequence_length",
    "L", "n_timesteps",
})
BUNDLE_TIME_AXIS_DIMENSION = "time_axis_steps"


def _load_bundle_dimension_facts(run_dir: Path) -> dict[str, int]:
    """Return semantic dimensions explicitly measured by bundle provenance.

    The current provenance contract measures only the realized time-axis
    length. It does not measure static- or time-varying-feature widths, so
    this function must not infer those values from column counts or shape
    spellings.
    """
    provenance = run_dir / "method" / "example_data" / "PROVENANCE.json"
    if not provenance.is_file():
        return {}
    try:
        manifest = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    steps = 0
    for entry in manifest.get("files") or []:
        axis = entry.get("time_axis") if isinstance(entry, dict) else None
        if isinstance(axis, dict) and isinstance(axis.get("steps_kept"), int):
            steps = max(steps, axis["steps_kept"])
    if steps <= 0:
        return {}
    return {BUNDLE_TIME_AXIS_DIMENSION: steps}


def _load_typed_bundle_dimension_facts(run_dir: Path) -> dict[str, Any]:
    """Load version-2 bundle facts without discarding their evidence roots.

    The legacy adapter above intentionally returns plain integers for its
    spelling table.  Version 2 consumes the shared semantic fact envelope so
    an exact measured value keeps its provenance root through disagreement
    reporting.
    """
    provenance = run_dir / "method" / "example_data" / "PROVENANCE.json"
    if not provenance.is_file():
        return {}
    try:
        manifest = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(manifest, dict):
        return {}
    return bundle_dimension_facts(manifest)


def _runtime_issue_roots(message: str) -> list[str]:
    """Name the contract and generated-code roots implicated by a dry run."""
    label = message.split(":", 1)[0].split(" via ", 1)[0].strip()
    if label.startswith("architecture."):
        code_root = (
            "method/training.py"
            if "constructor_args via builder" in message
            else "method/model.py"
        )
        return [label, code_root]
    if label.startswith("data_loader"):
        return [label, "method/data.py"]
    if label.startswith("training_loop"):
        return [label, "method/training.py"]
    if label.startswith("pluggable_component"):
        return [label, "method"]
    if label.startswith("relational_indexing"):
        return ["relational_indexing", "method/training.py"]
    return ["arch_contract", "method"]


def _contract_code_issue(message: str) -> SemanticIssue:
    return SemanticIssue(
        code="contract_code_disagreement",
        message=message,
        roots=_runtime_issue_roots(message),
    )


def _reachability_issue(message: str) -> SemanticIssue:
    """Keep bounded-validator coverage gaps out of producer retry routing."""

    if message.startswith(_PIPELINE_REACHABILITY_PREFIX):
        return SemanticIssue(
            code="unsupported_validator_feature",
            message=message[len(_PIPELINE_REACHABILITY_PREFIX):].lstrip(),
            roots=[
                "build_plan.training_history_execution",
                "scripts/validate_arch_contract_runtime.py",
            ],
        )
    return _contract_code_issue(message)


def _runtime_plan_issues(error: RuntimePlanError) -> list[SemanticIssue]:
    """Preserve normalization ownership instead of flattening every failure.

    Generated contract/code disagreements remain producer-owned.  A semantic
    resolver coverage envelope or malformed family-owned build-plan grammar is
    pipeline-owned and must not consume a producer retry.
    """

    message = str(error)
    semantic_prefix = "schema-2 runtime plan resolution failed: "
    if message.startswith(semantic_prefix):
        try:
            payload = json.loads(message[len(semantic_prefix):])
            issues = [SemanticIssue.model_validate(item) for item in payload]
        except (json.JSONDecodeError, TypeError, ValueError):
            issues = []
        if issues:
            return issues

    pipeline_owned = isinstance(
        error.__cause__, (
            TargetScalingCoverageError,
            TrainingHistoryCoverageError,
        )
    ) or message.startswith("build_plan.") or message.startswith(
        "target_scaling_execution must use"
    ) or message.startswith(
        "target_scaling_execution.helper must equal"
    ) or message.startswith(
        "target_scaling_execution.training_call has unsupported shape"
    ) or message.startswith(
        "target scaling validator-only fixture cannot be frozen without guessing"
    ) or message.startswith(
        "training_history_execution must use"
    ) or message.startswith(
        "training_history_execution.helper must equal"
    ) or message.startswith(
        "training_history_execution.training_call has unsupported shape"
    ) or message.startswith(
        "training-history validator-only fixture cannot be frozen without guessing"
    )
    if pipeline_owned:
        plan_root = (
            "build_plan.training_history_execution"
            if "training_history" in message or "training-history" in message
            else "build_plan.target_scaling_execution"
        )
        return [SemanticIssue(
            code="unsupported_validator_feature",
            message=message,
            roots=[
                plan_root,
                "scripts/validate_arch_contract_runtime.py",
            ],
        )]
    return [_contract_code_issue(message)]


def _host_runtime_capability_issues(
    contract: ArchContractV2,
    typed_fixtures: dict[str, dict[str, Any]],
) -> list[SemanticIssue]:
    """Check runtime libraries on the trusted validator side before import."""
    roots_by_backend: dict[str, list[str]] = {"torch": [], "numpy": []}
    for root, fixture in typed_fixtures.items():
        kind = fixture.get("kind")
        if kind == "tensor":
            roots_by_backend["torch"].append(root)
        elif kind in {"ndarray", "scalar"}:
            roots_by_backend["numpy"].append(root)

    if contract.relational_indexing is not None:
        # The relational oracle compares exact induced graphs through numpy
        # regardless of the generated callable's declared tensor backend.
        roots_by_backend["numpy"].append("relational_indexing")
        roots_by_backend[
            contract.relational_indexing.preparation_callable.tensor_backend
        ].append("relational_indexing.preparation_callable.tensor_backend")

    issues: list[SemanticIssue] = []
    for backend, roots in roots_by_backend.items():
        if roots and importlib.util.find_spec(backend) is None:
            issues.append(
                SemanticIssue(
                    code="unsupported_validator_feature",
                    message=(
                        f"typed fixtures require {backend}, but the Stage 2.d "
                        "validation environment cannot locate that runtime"
                    ),
                    roots=[
                        *roots,
                        "scripts/validate_arch_contract_runtime.py",
                    ],
                )
            )
    return issues


def _resolve_relational_execution_plan(
    contract: ArchContractV2,
    *,
    resolved_dimensions: dict[str, Any],
    typed_fixtures: dict[str, dict[str, Any]],
    permitted_opaque_fitting_roots: frozenset[str] = frozenset(),
) -> tuple[dict[str, Any] | None, list[SemanticIssue]]:
    """Resolve the schema-2 relational call crosswalk without name guessing.

    The model-free oracle remains the broad representation gate. This helper
    closes only the real-call gap: it proves that every value needed to call
    fitting and inference has one exact typed root, and turns that declaration
    into a small portable plan for the subprocess. Generated code is not
    imported here, so unsupported grammar cannot consume a producer retry.
    """
    relational = contract.relational_indexing
    if relational is None:
        return None, []

    execution = relational.execution
    if execution is None:
        return None, [
            SemanticIssue(
                code="incomplete_generated_contract",
                message=(
                    "schema-2 homogeneous relational_indexing requires an "
                    "explicit execution crosswalk for the real fitting and "
                    "inference calls; no model or parameter name is guessed"
                ),
                roots=["relational_indexing.execution"],
            )
        ]

    issues: list[SemanticIssue] = []

    def producer(message: str, *roots: str) -> None:
        issues.append(
            SemanticIssue(
                code="incomplete_generated_contract",
                message=message,
                roots=list(roots),
            )
        )

    def unsupported(message: str, *roots: str) -> None:
        issues.append(
            SemanticIssue(
                code="unsupported_validator_feature",
                message=message,
                roots=[*roots, "scripts/validate_arch_contract_runtime.py"],
            )
        )

    dimension = resolved_dimensions.get(execution.entity_dimension)
    if dimension is None:
        producer(
            "relational_indexing.execution.entity_dimension names an "
            f"unresolved dimension {execution.entity_dimension!r}",
            "relational_indexing.execution.entity_dimension",
            f"dimensions.{execution.entity_dimension}",
        )
        entity_count = None
    else:
        entity_count = int(dimension.value)
        if entity_count < 3:
            unsupported(
                "the real relational training arm needs at least three "
                "entities for an asymmetric induced-subset fixture; resolved "
                f"{execution.entity_dimension!r}={entity_count}",
                "relational_indexing.execution.entity_dimension",
                f"dimensions.{execution.entity_dimension}",
            )

    fitting = execution.fitting
    inference = execution.inference
    fitting_mode = relational.phase_batch_modes.fitting
    inference_mode = relational.phase_batch_modes.inference
    if contract.training_loop is None:
        producer(
            "relational_indexing.execution.fitting requires a declared "
            "training_loop",
            "relational_indexing.execution.fitting",
            "training_loop",
        )

    architecture_block = contract.architecture.get(inference.architecture_block)
    if architecture_block is None:
        producer(
            "relational_indexing.execution.inference.architecture_block names "
            f"unknown block {inference.architecture_block!r}",
            "relational_indexing.execution.inference.architecture_block",
            f"architecture.{inference.architecture_block}",
        )

    fitting_prefix = "training_loop.input."
    inference_prefix = (
        f"architecture.{inference.architecture_block}.forward.input."
    )

    fitting_role_roots = [
        fitting.model_input_root,
        fitting.source_entity_ids_input_root,
        fitting.batch_entity_ids_input_root,
        fitting.graph_input_root,
        *(
            [fitting.degree_input_root]
            if fitting.degree_input_root is not None else []
        ),
        *fitting.coindexed_input_roots.values(),
        *(
            [fitting.one_epoch_input_root]
            if fitting.one_epoch_input_root is not None else []
        ),
    ]
    inference_role_roots = [
        inference.graph_input_root,
        *(
            [inference.degree_input_root]
            if inference.degree_input_root is not None else []
        ),
        *inference.coindexed_input_roots.values(),
    ]
    collapsed_opaque_roots = sorted(
        {
            root
            for roots in (fitting_role_roots, inference_role_roots)
            for root in roots
            if roots.count(root) > 1
            and (typed_fixtures.get(root) or {}).get("kind") == "opaque"
        }
    )
    bound_opaque_roots = sorted(
        {
            root
            for root in [
                *fitting_role_roots,
                *inference_role_roots,
            ]
            if root != fitting.model_input_root
            and (typed_fixtures.get(root) or {}).get("kind") == "opaque"
        }
    )
    bound_fitting_roots = set(fitting_role_roots)
    bound_inference_roots = set(inference_role_roots)
    unbound_opaque_roots: list[str] = []
    if contract.training_loop is not None:
        for name in contract.training_loop.input:
            root = fitting_prefix + name
            if (
                root not in bound_fitting_roots
                and root not in permitted_opaque_fitting_roots
                and (typed_fixtures.get(root) or {}).get("kind") == "opaque"
            ):
                unbound_opaque_roots.append(root)
    if architecture_block is not None:
        for name in architecture_block.forward.input:
            root = inference_prefix + name
            if (
                root not in bound_inference_roots
                and (typed_fixtures.get(root) or {}).get("kind") == "opaque"
            ):
                unbound_opaque_roots.append(root)
    opaque_grammar_roots = sorted(
        set(
            collapsed_opaque_roots
            + bound_opaque_roots
            + unbound_opaque_roots
        )
    )
    if opaque_grammar_roots:
        reason = (
            "one opaque structured input collapses multiple relational roles"
            if collapsed_opaque_roots
            else "an opaque callable input has no deterministic relational fixture"
        )
        return None, [
            SemanticIssue(
                code="unsupported_validator_feature",
                message=(
                    "the schema-2 real relational call cannot construct that "
                    f"relational grammar without guessing because {reason}: "
                    f"{opaque_grammar_roots!r}"
                ),
                roots=[
                    *opaque_grammar_roots,
                    "relational_indexing.execution",
                    "scripts/validate_arch_contract_runtime.py",
                ],
            )
        ]

    def dimension_depends_on(identity: str, target: str, seen=None) -> bool:
        if identity == target:
            return True
        seen = set(seen or ())
        if identity in seen:
            return False
        seen.add(identity)
        definition = contract.dimensions.get(identity)
        if definition is None:
            return False
        payload = definition.expression.model_dump(exclude_none=True)

        def referenced_dimensions(value):
            if isinstance(value, dict):
                if value.get("kind") == "reference" and isinstance(
                    value.get("dimension"), str
                ):
                    yield value["dimension"]
                for nested in value.values():
                    yield from referenced_dimensions(nested)
            elif isinstance(value, list):
                for nested in value:
                    yield from referenced_dimensions(nested)

        return any(
            dimension_depends_on(reference, target, seen)
            for reference in referenced_dimensions(payload)
        )

    def descriptor_dimensions(descriptor) -> list[str]:
        return [
            use.dimension
            for use in (getattr(descriptor, "dimensions", None) or [])
        ]

    fitting_excluded = {
        fitting.model_input_root,
        fitting.source_entity_ids_input_root,
        fitting.batch_entity_ids_input_root,
        fitting.graph_input_root,
        *(
            [fitting.degree_input_root]
            if fitting.degree_input_root is not None else []
        ),
        *(
            [fitting.one_epoch_input_root]
            if fitting.one_epoch_input_root is not None else []
        ),
    }
    fitting_candidates: set[str] = set()
    composite_fitting_roots: list[str] = []
    if contract.training_loop is not None:
        for name, descriptor in contract.training_loop.input.items():
            root = fitting_prefix + name
            if root in fitting_excluded:
                continue
            dimensions = descriptor_dimensions(descriptor)
            direct_axes = [
                index
                for index, identity in enumerate(dimensions)
                if identity == execution.entity_dimension
            ]
            dependent_axes = [
                index
                for index, identity in enumerate(dimensions)
                if dimension_depends_on(identity, execution.entity_dimension)
            ]
            if len(direct_axes) == 1:
                fitting_candidates.add(root)
            elif dependent_axes:
                composite_fitting_roots.append(root)
    if composite_fitting_roots:
        return None, [
            SemanticIssue(
                code="unsupported_validator_feature",
                message=(
                    "the schema-2 fitting surface contains an array whose "
                    "entity domain is composite or appears on multiple axes; "
                    "the current relational grammar requires one direct entity "
                    f"axis: {sorted(composite_fitting_roots)!r}"
                ),
                roots=[
                    *sorted(composite_fitting_roots),
                    "relational_indexing.execution.entity_dimension",
                    "scripts/validate_arch_contract_runtime.py",
                ],
            )
        ]
    mapped_fitting_roots = set(fitting.coindexed_input_roots.values())
    if mapped_fitting_roots != fitting_candidates:
        producer(
            "fitting.coindexed_input_roots must close the typed fitting input "
            "surface on the direct source entity dimension; missing="
            f"{sorted(fitting_candidates - mapped_fitting_roots)!r}, extra="
            f"{sorted(mapped_fitting_roots - fitting_candidates)!r}",
            "relational_indexing.execution.fitting.coindexed_input_roots",
            "training_loop.input",
        )
    for logical_root, typed_root in fitting.coindexed_input_roots.items():
        parameter_name = typed_root[len(fitting_prefix):]
        descriptor = (
            contract.training_loop.input.get(parameter_name)
            if contract.training_loop is not None else None
        )
        dimensions = descriptor_dimensions(descriptor)
        direct_axes = [
            index for index, identity in enumerate(dimensions)
            if identity == execution.entity_dimension
        ]
        expected_axis = relational.coindexed_roots[logical_root]
        if direct_axes != [expected_axis]:
            producer(
                f"fitting co-indexed root {logical_root!r} declares entity "
                f"axis {expected_axis}, but typed input {typed_root!r} uses "
                f"direct semantic axes {direct_axes!r}",
                f"relational_indexing.coindexed_roots.{logical_root}",
                typed_root,
            )

    output_logical_root = inference.output_coindexed_root
    output_typed_root = (
        f"architecture.{inference.architecture_block}.forward.output"
    )
    output_descriptor = (
        architecture_block.forward.output
        if architecture_block is not None else None
    )
    if getattr(output_descriptor, "kind", None) == "opaque":
        return None, [
            SemanticIssue(
                code="unsupported_validator_feature",
                message=(
                    "the declared relational inference output is opaque, so "
                    "Stage 2.d cannot bind its entity axis to ordered stable ids"
                ),
                roots=[
                    output_typed_root,
                    "relational_indexing.execution.inference.output_coindexed_root",
                    "scripts/validate_arch_contract_runtime.py",
                ],
            )
        ]
    output_dimensions = descriptor_dimensions(output_descriptor)
    output_axis = relational.coindexed_roots[output_logical_root]
    if len(output_dimensions) <= output_axis:
        producer(
            "the declared architecture forward output has no entity axis "
            f"{output_axis} for output root {output_logical_root!r}; typed "
            f"dimensions={output_dimensions!r}",
            "relational_indexing.execution.inference.output_coindexed_root",
            output_typed_root,
        )
        inference_entity_dimension = None
    else:
        inference_entity_dimension = output_dimensions[output_axis]
    if inference_entity_dimension is not None and (
        output_dimensions.count(inference_entity_dimension) > 1
    ):
        return None, [
            SemanticIssue(
                code="unsupported_validator_feature",
                message=(
                    "the declared relational inference output carries the "
                    "entity dimension on multiple axes, so Stage 2.d cannot "
                    "bind one exact output row order"
                ),
                roots=[
                    output_typed_root,
                    "relational_indexing.execution.inference.output_coindexed_root",
                    "scripts/validate_arch_contract_runtime.py",
                ],
            )
        ]
    if (
        inference_entity_dimension is not None
        and inference_mode != "induced_subgraph"
        and inference_entity_dimension != execution.entity_dimension
    ):
        producer(
            "a full-graph inference output must use the canonical semantic "
            f"entity dimension {execution.entity_dimension!r}; got "
            f"{inference_entity_dimension!r}",
            "relational_indexing.execution.entity_dimension",
            "relational_indexing.phase_batch_modes.inference",
            output_typed_root,
        )

    if architecture_block is not None and inference_entity_dimension is not None:
        for logical_root, typed_root in inference.coindexed_input_roots.items():
            parameter_name = typed_root[len(inference_prefix):]
            descriptor = architecture_block.forward.input.get(parameter_name)
            dimensions = descriptor_dimensions(descriptor)
            expected_axis = relational.coindexed_roots[logical_root]
            direct_axes = [
                index
                for index, identity in enumerate(dimensions)
                if identity == inference_entity_dimension
            ]
            if direct_axes != [expected_axis]:
                producer(
                    f"inference co-indexed root {logical_root!r} declares "
                    f"entity axis {expected_axis}, but typed input "
                    f"{typed_root!r} uses the output-bound entity dimension "
                    f"{inference_entity_dimension!r} on axes {direct_axes!r}",
                    f"relational_indexing.coindexed_roots.{logical_root}",
                    typed_root,
                    output_typed_root,
                )

    inference_excluded = {
        inference.graph_input_root,
        *(
            [inference.degree_input_root]
            if inference.degree_input_root is not None else []
        ),
    }
    inference_candidates: set[str] = set()
    composite_inference_roots: list[str] = []
    if architecture_block is not None and inference_entity_dimension is not None:
        for name, descriptor in architecture_block.forward.input.items():
            root = inference_prefix + name
            if root in inference_excluded:
                continue
            dimensions = descriptor_dimensions(descriptor)
            direct_axes = [
                index
                for index, identity in enumerate(dimensions)
                if identity == inference_entity_dimension
            ]
            dependent_axes = [
                index
                for index, identity in enumerate(dimensions)
                if dimension_depends_on(identity, inference_entity_dimension)
            ]
            if len(direct_axes) == 1:
                inference_candidates.add(root)
            elif dependent_axes:
                composite_inference_roots.append(root)
    if composite_inference_roots:
        return None, [
            SemanticIssue(
                code="unsupported_validator_feature",
                message=(
                    "the schema-2 inference surface contains an array whose "
                    "entity domain is composite or appears on multiple axes; "
                    "the current relational grammar requires one direct entity "
                    f"axis: {sorted(composite_inference_roots)!r}"
                ),
                roots=[
                    *sorted(composite_inference_roots),
                    "relational_indexing.execution.inference",
                    "scripts/validate_arch_contract_runtime.py",
                ],
            )
        ]
    mapped_inference_roots = set(inference.coindexed_input_roots.values())
    if (
        inference_entity_dimension is not None
        and mapped_inference_roots != inference_candidates
    ):
        producer(
            "inference.coindexed_input_roots must close the typed model-forward "
            "surface on its direct entity dimension; missing="
            f"{sorted(inference_candidates - mapped_inference_roots)!r}, extra="
            f"{sorted(mapped_inference_roots - inference_candidates)!r}",
            "relational_indexing.execution.inference.coindexed_input_roots",
            f"architecture.{inference.architecture_block}.forward.input",
        )

    def parameter(root: str, *, prefix: str, binding_root: str) -> str | None:
        if not root.startswith(prefix) or root == prefix:
            # Pydantic normally catches this. Keep the runtime resolver closed
            # for callers that construct model instances programmatically.
            producer(
                f"{binding_root} must be a full canonical root below {prefix!r}",
                binding_root,
            )
            return None
        name = root[len(prefix):]
        if "." in name or not name:
            producer(
                f"{binding_root} must name one direct callable input; got {root!r}",
                binding_root,
                root,
            )
            return None
        if root not in typed_fixtures:
            producer(
                f"{binding_root} points to undeclared typed input {root!r}",
                binding_root,
                root,
            )
            return None
        return name

    fitting_roots: dict[str, str] = {
        "model": fitting.model_input_root,
        "source_entity_ids": fitting.source_entity_ids_input_root,
        "batch_entity_ids": fitting.batch_entity_ids_input_root,
        "graph": fitting.graph_input_root,
    }
    if fitting.degree_input_root is not None:
        fitting_roots["degrees"] = fitting.degree_input_root
    inference_roots: dict[str, str] = {"graph": inference.graph_input_root}
    if inference.degree_input_root is not None:
        inference_roots["degrees"] = inference.degree_input_root

    fitting_parameters = {
        role: parameter(
            root,
            prefix=fitting_prefix,
            binding_root=f"relational_indexing.execution.fitting.{role}_input_root",
        )
        for role, root in fitting_roots.items()
    }
    inference_parameters = {
        role: parameter(
            root,
            prefix=inference_prefix,
            binding_root=f"relational_indexing.execution.inference.{role}_input_root",
        )
        for role, root in inference_roots.items()
    }
    one_epoch_parameter = (
        parameter(
            fitting.one_epoch_input_root,
            prefix=fitting_prefix,
            binding_root=(
                "relational_indexing.execution.fitting.one_epoch_input_root"
            ),
        )
        if fitting.one_epoch_input_root is not None else None
    )

    fitting_coindexed: dict[str, str | None] = {}
    for logical_root, typed_root in fitting.coindexed_input_roots.items():
        fitting_coindexed[logical_root] = parameter(
            typed_root,
            prefix=fitting_prefix,
            binding_root=(
                "relational_indexing.execution.fitting.coindexed_input_roots"
                f"[{logical_root!r}]"
            ),
        )
    inference_coindexed: dict[str, str | None] = {}
    for logical_root, typed_root in inference.coindexed_input_roots.items():
        inference_coindexed[logical_root] = parameter(
            typed_root,
            prefix=inference_prefix,
            binding_root=(
                "relational_indexing.execution.inference.coindexed_input_roots"
                f"[{logical_root!r}]"
            ),
        )

    fitting_keys = set(fitting_coindexed)
    inference_keys = set(inference_coindexed)
    if not inference_keys <= fitting_keys:
        producer(
            "every inference co-indexed root must also have a fitting binding "
            "so its entity convention can be compared; "
            f"inference_only={sorted(inference_keys - fitting_keys)!r}",
            "relational_indexing.execution.fitting.coindexed_input_roots",
            "relational_indexing.execution.inference.coindexed_input_roots",
        )

    all_fitting_roots = [
        *fitting_roots.values(),
        *fitting.coindexed_input_roots.values(),
        *(
            [fitting.one_epoch_input_root]
            if fitting.one_epoch_input_root is not None else []
        ),
    ]
    all_inference_roots = [
        *inference_roots.values(),
        *inference.coindexed_input_roots.values(),
    ]
    for label, roots in (
        ("fitting", all_fitting_roots),
        ("inference", all_inference_roots),
    ):
        duplicates = sorted({root for root in roots if roots.count(root) > 1})
        if duplicates:
            producer(
                f"relational execution {label} roles must use distinct typed "
                f"inputs; duplicate roots={duplicates!r}",
                f"relational_indexing.execution.{label}",
                *duplicates,
            )

    model_fixture = typed_fixtures.get(fitting.model_input_root)
    if model_fixture is not None and model_fixture.get("kind") != "opaque":
        producer(
            "the fitting model input must be declared opaque and supplied by "
            "the exact constructed architecture block",
            "relational_indexing.execution.fitting.model_input_root",
            fitting.model_input_root,
        )

    if fitting.one_epoch_input_root is not None:
        epoch_fixture = typed_fixtures.get(fitting.one_epoch_input_root)
        if (epoch_fixture or {}).get("kind") == "opaque":
            return None, [
                SemanticIssue(
                    code="unsupported_validator_feature",
                    message=(
                        "the declared one-epoch control is opaque and cannot be "
                        "overridden to one without guessing"
                    ),
                    roots=[
                        fitting.one_epoch_input_root,
                        "relational_indexing.execution.fitting.one_epoch_input_root",
                        "scripts/validate_arch_contract_runtime.py",
                    ],
                )
            ]
        if (
            (epoch_fixture or {}).get("kind") != "scalar"
            or (epoch_fixture or {}).get("dtype") not in {"int32", "int64"}
        ):
            producer(
                "one_epoch_input_root must name an integer scalar typed input",
                "relational_indexing.execution.fitting.one_epoch_input_root",
                fitting.one_epoch_input_root,
            )

    backend = relational.preparation_callable.tensor_backend

    def array_fixture(
        root: str,
        *,
        role: str,
        integer: bool = False,
        expected_rank: int | None = None,
    ) -> dict[str, Any] | None:
        fixture = typed_fixtures.get(root)
        if fixture is None:
            return None
        kind = fixture.get("kind")
        expected_kind = "tensor" if backend == "torch" else "ndarray"
        if kind == "opaque":
            unsupported(
                f"relational execution root {root!r} is opaque; the current "
                "validator cannot construct that relational grammar without "
                "guessing",
                role,
                root,
            )
            return None
        if kind != expected_kind:
            producer(
                f"relational execution root {root!r} must use declared "
                f"tensor_backend={backend!r} ({expected_kind}); got {kind!r}",
                role,
                root,
            )
            return None
        shape = fixture.get("shape")
        if not isinstance(shape, (list, tuple)):
            producer(
                f"relational execution root {root!r} has no resolved array shape",
                role,
                root,
            )
            return None
        if expected_rank is not None and len(shape) != expected_rank:
            producer(
                f"relational execution root {root!r} must have rank "
                f"{expected_rank}; got shape {tuple(shape)!r}",
                role,
                root,
            )
        if integer and fixture.get("dtype") not in {"int32", "int64"}:
            producer(
                f"relational execution root {root!r} must use an integer dtype; "
                f"got {fixture.get('dtype')!r}",
                role,
                root,
            )
        return fixture

    source_ids_fixture = array_fixture(
        fitting.source_entity_ids_input_root,
        role="relational_indexing.execution.fitting.source_entity_ids_input_root",
        integer=True,
        expected_rank=1,
    )
    batch_ids_fixture = array_fixture(
        fitting.batch_entity_ids_input_root,
        role="relational_indexing.execution.fitting.batch_entity_ids_input_root",
        integer=True,
        expected_rank=1,
    )
    fitting_graph_fixture = array_fixture(
        fitting.graph_input_root,
        role="relational_indexing.execution.fitting.graph_input_root",
        integer=relational.representation == "sparse_edge_index",
        expected_rank=2,
    )
    inference_graph_fixture = array_fixture(
        inference.graph_input_root,
        role="relational_indexing.execution.inference.graph_input_root",
        integer=relational.representation == "sparse_edge_index",
        expected_rank=2,
    )

    source_ids_descriptor = (
        contract.training_loop.input.get(
            fitting.source_entity_ids_input_root[len(fitting_prefix):]
        )
        if contract.training_loop is not None else None
    )
    batch_ids_descriptor = (
        contract.training_loop.input.get(
            fitting.batch_entity_ids_input_root[len(fitting_prefix):]
        )
        if contract.training_loop is not None else None
    )
    source_id_dimensions = descriptor_dimensions(source_ids_descriptor)
    batch_id_dimensions = descriptor_dimensions(batch_ids_descriptor)
    if source_id_dimensions != [execution.entity_dimension]:
        producer(
            "source stable-id input must use exactly the canonical semantic "
            f"entity dimension {execution.entity_dimension!r}; got "
            f"{source_id_dimensions!r}",
            "relational_indexing.execution.entity_dimension",
            fitting.source_entity_ids_input_root,
        )
    fitting_entity_dimension = (
        batch_id_dimensions[0] if len(batch_id_dimensions) == 1 else None
    )
    if fitting_entity_dimension is None:
        producer(
            "batch stable-id input must declare exactly one semantic entity "
            f"dimension; got {batch_id_dimensions!r}",
            fitting.batch_entity_ids_input_root,
        )
    elif (
        fitting_mode != "induced_subgraph"
        and fitting_entity_dimension != execution.entity_dimension
    ):
        producer(
            "a full-graph fitting batch-id input must use the canonical "
            f"semantic entity dimension {execution.entity_dimension!r}; got "
            f"{fitting_entity_dimension!r}",
            "relational_indexing.execution.entity_dimension",
            "relational_indexing.phase_batch_modes.fitting",
            fitting.batch_entity_ids_input_root,
        )

    def check_graph_dimension_identity(
        fixture: dict[str, Any] | None,
        descriptor,
        *,
        root: str,
        expected_dimension: str | None,
        role: str,
    ) -> None:
        if fixture is None or expected_dimension is None:
            return
        if relational.representation == "sparse_edge_index":
            indexed_dimension = (fixture.get("constraint") or {}).get(
                "indexed_dimension"
            )
            if indexed_dimension != expected_dimension:
                producer(
                    f"sparse relational graph {root!r} must index the exact "
                    f"semantic entity dimension {expected_dimension!r}; got "
                    f"{indexed_dimension!r}",
                    role,
                    root,
                )
        else:
            dimensions = descriptor_dimensions(descriptor)
            if dimensions != [expected_dimension, expected_dimension]:
                producer(
                    f"dense relational graph {root!r} must use the exact "
                    f"semantic entity dimension {expected_dimension!r} on "
                    f"both axes; got {dimensions!r}",
                    role,
                    root,
                )

    fitting_graph_descriptor = (
        contract.training_loop.input.get(
            fitting.graph_input_root[len(fitting_prefix):]
        )
        if contract.training_loop is not None else None
    )
    inference_graph_descriptor = (
        architecture_block.forward.input.get(
            inference.graph_input_root[len(inference_prefix):]
        )
        if architecture_block is not None else None
    )
    check_graph_dimension_identity(
        fitting_graph_fixture,
        fitting_graph_descriptor,
        root=fitting.graph_input_root,
        expected_dimension=execution.entity_dimension,
        role="relational_indexing.execution.fitting.graph_input_root",
    )
    check_graph_dimension_identity(
        inference_graph_fixture,
        inference_graph_descriptor,
        root=inference.graph_input_root,
        expected_dimension=inference_entity_dimension,
        role="relational_indexing.execution.inference.graph_input_root",
    )

    def require_same_array_convention(
        fitting_fixture: dict[str, Any] | None,
        inference_fixture: dict[str, Any] | None,
        *,
        fitting_root: str,
        inference_root: str,
        label: str,
    ) -> None:
        if fitting_fixture is None or inference_fixture is None:
            return
        fields = ("kind", "dtype", "device")
        mismatched = {
            field: (
                fitting_fixture.get(field),
                inference_fixture.get(field),
            )
            for field in fields
            if fitting_fixture.get(field) != inference_fixture.get(field)
        }
        if mismatched:
            producer(
                f"fitting and inference {label} inputs use different array "
                f"conventions: {mismatched!r}",
                fitting_root,
                inference_root,
            )

    require_same_array_convention(
        fitting_graph_fixture,
        inference_graph_fixture,
        fitting_root=fitting.graph_input_root,
        inference_root=inference.graph_input_root,
        label="graph",
    )
    if (
        source_ids_fixture is not None
        and batch_ids_fixture is not None
        and any(
            source_ids_fixture.get(field) != batch_ids_fixture.get(field)
            for field in ("kind", "dtype", "device")
        )
    ):
        producer(
            "source and batch stable-id inputs must use the same typed array "
            "convention",
            fitting.source_entity_ids_input_root,
            fitting.batch_entity_ids_input_root,
        )

    if entity_count is not None and source_ids_fixture is not None:
        source_shape = tuple(source_ids_fixture.get("shape") or ())
        if source_shape != (entity_count,):
            producer(
                "source stable-id input must use the declared canonical entity "
                f"dimension exactly; expected {(entity_count,)!r}, got "
                f"{source_shape!r}",
                "relational_indexing.execution.entity_dimension",
                fitting.source_entity_ids_input_root,
            )

    def positions_for(mode: str, requested_count: int | None) -> list[int] | None:
        if entity_count is None:
            return None
        if mode == "canonical_full_graph":
            return list(range(entity_count))
        if mode == "coherent_full_graph_permutation":
            positions: list[int] = []
            priority = [entity_count - 1, 2, 0, entity_count - 2, 1]
            for candidate in [*priority, *range(entity_count)]:
                if candidate not in positions:
                    positions.append(candidate)
                if len(positions) == entity_count:
                    break
            return positions
        if mode == "induced_subgraph":
            if requested_count is None:
                return None
            if requested_count <= 0 or requested_count >= entity_count:
                producer(
                    "an induced_subgraph batch must be a non-empty strict "
                    f"subset of the source entity domain; source={entity_count}, "
                    f"batch={requested_count}",
                    "relational_indexing.phase_batch_modes",
                    fitting.batch_entity_ids_input_root,
                )
                return None
            priority = [entity_count - 1, 2, 0]
            positions: list[int] = []
            for candidate in [*priority, *range(entity_count)]:
                if candidate not in positions:
                    positions.append(candidate)
                if len(positions) == requested_count:
                    break
            return positions
        unsupported(
            f"unsupported relational phase batch mode {mode!r}",
            "relational_indexing.phase_batch_modes",
        )
        return None

    batch_shape = (
        tuple(batch_ids_fixture.get("shape") or ())
        if batch_ids_fixture is not None else ()
    )
    fitting_requested = batch_shape[0] if len(batch_shape) == 1 else None
    fitting_positions = positions_for(fitting_mode, fitting_requested)
    if (
        fitting_positions is not None
        and fitting_requested is not None
        and len(fitting_positions) != fitting_requested
    ):
        producer(
            "fitting batch stable-id descriptor disagrees with the declared "
            f"phase mode; expected {len(fitting_positions)} ids, got "
            f"shape {batch_shape!r}",
            "relational_indexing.phase_batch_modes.fitting",
            fitting.batch_entity_ids_input_root,
        )

    # The output crosswalk is the phase-local identity authority even for a
    # graph-only model. It supplies the exact inference entity-axis extent;
    # mapped forward inputs, graph bounds, and degrees must agree below.
    output_fixture = typed_fixtures.get(output_typed_root)
    output_shape = tuple((output_fixture or {}).get("shape") or ())
    inference_requested = (
        output_shape[output_axis]
        if len(output_shape) > output_axis else None
    )
    if (
        inference_mode != "induced_subgraph"
        and entity_count is not None
        and inference_requested is not None
        and inference_requested != entity_count
    ):
        producer(
            "a full-graph inference output must align with the canonical "
            f"entity domain; expected {entity_count}, got "
            f"shape {output_shape!r} on axis {output_axis}",
            "relational_indexing.phase_batch_modes.inference",
            output_typed_root,
        )
    inference_positions = positions_for(inference_mode, inference_requested)

    def check_graph_shape(
        fixture: dict[str, Any] | None,
        *,
        root: str,
        expected_nodes: int | None,
        role: str,
    ) -> None:
        if fixture is None or expected_nodes is None:
            return
        shape = tuple(fixture.get("shape") or ())
        if relational.representation == "sparse_edge_index":
            if len(shape) == 2 and shape[0] != 2:
                producer(
                    f"sparse relational graph {root!r} must have shape (2, E); "
                    f"got {shape!r}",
                    role,
                    root,
                )
            constraint = fixture.get("constraint") or {}
            if (
                constraint.get("kind") != "index"
                or fixture.get("constraint_upper_bound") != expected_nodes
            ):
                producer(
                    f"sparse relational graph {root!r} must declare an index "
                    f"constraint bounded by its {expected_nodes}-entity endpoint "
                    "domain",
                    role,
                    root,
                )
        elif len(shape) == 2 and shape != (expected_nodes, expected_nodes):
            producer(
                f"dense relational graph {root!r} must be square on its entity "
                f"domain; expected {(expected_nodes, expected_nodes)!r}, got "
                f"{shape!r}",
                role,
                root,
            )

    check_graph_shape(
        fitting_graph_fixture,
        root=fitting.graph_input_root,
        expected_nodes=entity_count,
        role="relational_indexing.execution.fitting.graph_input_root",
    )
    check_graph_shape(
        inference_graph_fixture,
        root=inference.graph_input_root,
        expected_nodes=(
            len(inference_positions) if inference_positions is not None else None
        ),
        role="relational_indexing.execution.inference.graph_input_root",
    )

    if fitting.degree_input_root is not None:
        fitting_degree_descriptor = (
            contract.training_loop.input.get(
                fitting.degree_input_root[len(fitting_prefix):]
            )
            if contract.training_loop is not None else None
        )
        inference_degree_descriptor = (
            architecture_block.forward.input.get(
                inference.degree_input_root[len(inference_prefix):]
            )
            if architecture_block is not None else None
        )
        fitting_degree_dimensions = descriptor_dimensions(
            fitting_degree_descriptor
        )
        inference_degree_dimensions = descriptor_dimensions(
            inference_degree_descriptor
        )
        if fitting_degree_dimensions != [execution.entity_dimension]:
            producer(
                "fitting degree input must use exactly the canonical semantic "
                f"entity dimension {execution.entity_dimension!r}; got "
                f"{fitting_degree_dimensions!r}",
                "relational_indexing.execution.entity_dimension",
                fitting.degree_input_root,
            )
        if (
            inference_entity_dimension is not None
            and inference_degree_dimensions != [inference_entity_dimension]
        ):
            producer(
                "inference degree input must use exactly the output-bound "
                f"semantic entity dimension {inference_entity_dimension!r}; "
                f"got {inference_degree_dimensions!r}",
                "relational_indexing.execution.inference.output_coindexed_root",
                inference.degree_input_root,
            )
        fitting_degree = array_fixture(
            fitting.degree_input_root,
            role="relational_indexing.execution.fitting.degree_input_root",
            expected_rank=1,
        )
        inference_degree = array_fixture(
            inference.degree_input_root,
            role="relational_indexing.execution.inference.degree_input_root",
            expected_rank=1,
        )
        require_same_array_convention(
            fitting_degree,
            inference_degree,
            fitting_root=fitting.degree_input_root,
            inference_root=inference.degree_input_root,
            label="degree",
        )
        for fixture, expected, root, role in (
            (
                fitting_degree,
                entity_count,
                fitting.degree_input_root,
                "relational_indexing.execution.fitting.degree_input_root",
            ),
            (
                inference_degree,
                len(inference_positions) if inference_positions is not None else None,
                inference.degree_input_root,
                "relational_indexing.execution.inference.degree_input_root",
            ),
        ):
            if fixture is not None and expected is not None and tuple(
                fixture.get("shape") or ()
            ) != (expected,):
                producer(
                    f"degree input {root!r} must align with its entity domain; "
                    f"expected {(expected,)!r}, got "
                    f"{tuple(fixture.get('shape') or ())!r}",
                    role,
                    root,
                )

    for phase, mappings, expected_count in (
        ("fitting", fitting.coindexed_input_roots, entity_count),
        (
            "inference",
            inference.coindexed_input_roots,
            len(inference_positions) if inference_positions is not None else None,
        ),
    ):
        for logical_root, typed_root in mappings.items():
            fixture = array_fixture(
                typed_root,
                role=(
                    f"relational_indexing.execution.{phase}."
                    f"coindexed_input_roots[{logical_root!r}]"
                ),
            )
            axis = relational.coindexed_roots[logical_root]
            shape = tuple((fixture or {}).get("shape") or ())
            if fixture is not None and len(shape) <= axis:
                producer(
                    f"co-indexed input {typed_root!r} has no declared entity "
                    f"axis {axis}; got shape {shape!r}",
                    f"relational_indexing.coindexed_roots.{logical_root}",
                    typed_root,
                )
            elif (
                fixture is not None
                and expected_count is not None
                and shape[axis] != expected_count
            ):
                producer(
                    f"co-indexed input {typed_root!r} does not align with the "
                    f"{phase} entity domain on axis {axis}; expected "
                    f"{expected_count}, got shape {shape!r}",
                    f"relational_indexing.coindexed_roots.{logical_root}",
                    typed_root,
                )

    for logical_root in sorted(fitting_keys & inference_keys):
        fitting_root = fitting.coindexed_input_roots[logical_root]
        inference_root = inference.coindexed_input_roots[logical_root]
        fitting_fixture = typed_fixtures.get(fitting_root)
        inference_fixture = typed_fixtures.get(inference_root)
        require_same_array_convention(
            fitting_fixture,
            inference_fixture,
            fitting_root=fitting_root,
            inference_root=inference_root,
            label=f"co-indexed root {logical_root!r}",
        )
        if fitting_fixture is None or inference_fixture is None:
            continue
        axis = relational.coindexed_roots[logical_root]
        fitting_shape = tuple(fitting_fixture.get("shape") or ())
        inference_shape = tuple(inference_fixture.get("shape") or ())
        fitting_dimensions = descriptor_dimensions(
            contract.training_loop.input[
                fitting_root[len(fitting_prefix):]
            ]
        )
        inference_dimensions = descriptor_dimensions(
            architecture_block.forward.input[
                inference_root[len(inference_prefix):]
            ]
        )
        if (
            fitting_dimensions[:axis] + fitting_dimensions[axis + 1:]
            != inference_dimensions[:axis] + inference_dimensions[axis + 1:]
        ):
            producer(
                "fitting and inference co-indexed inputs must preserve exact "
                "non-entity semantic dimensions; got "
                f"{fitting_dimensions!r} and {inference_dimensions!r}",
                fitting_root,
                inference_root,
            )
        if len(fitting_shape) <= axis or len(inference_shape) <= axis:
            continue
        if (
            fitting_shape[:axis] + fitting_shape[axis + 1:]
            != inference_shape[:axis] + inference_shape[axis + 1:]
        ):
            producer(
                "fitting and inference co-indexed inputs must differ only on "
                f"their declared entity axis {axis}; got {fitting_shape!r} and "
                f"{inference_shape!r}",
                fitting_root,
                inference_root,
            )
    if issues:
        return None, issues
    if (
        entity_count is None
        or fitting_positions is None
        or inference_positions is None
        or any(value is None for value in fitting_parameters.values())
        or any(value is None for value in inference_parameters.values())
        or any(value is None for value in fitting_coindexed.values())
        or any(value is None for value in inference_coindexed.values())
        or (
            fitting.one_epoch_input_root is not None
            and one_epoch_parameter is None
        )
    ):
        return None, issues

    return {
        "entity_count": entity_count,
        "fitting_mode": fitting_mode,
        "inference_mode": inference_mode,
        "fitting_positions": fitting_positions,
        "inference_positions": inference_positions,
        "architecture_block": inference.architecture_block,
        "fitting_parameters": fitting_parameters,
        "one_epoch_parameter": one_epoch_parameter,
        "one_epoch_root": fitting.one_epoch_input_root,
        "inference_parameters": inference_parameters,
        "fitting_coindexed": fitting_coindexed,
        "inference_coindexed": inference_coindexed,
        "fitting_roots": {
            "model": fitting.model_input_root,
            "source_entity_ids": fitting.source_entity_ids_input_root,
            "batch_entity_ids": fitting.batch_entity_ids_input_root,
            "graph": fitting.graph_input_root,
            "degrees": fitting.degree_input_root,
            "coindexed": dict(fitting.coindexed_input_roots),
        },
        "inference_roots": {
            "graph": inference.graph_input_root,
            "degrees": inference.degree_input_root,
            "coindexed": dict(inference.coindexed_input_roots),
        },
    }, []


def _read_capability_report(path: Path) -> list[SemanticIssue]:
    """Read the runner-owned capability channel, never generated stderr."""
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [
            SemanticIssue(
                code="unsupported_validator_feature",
                message=f"Stage 2.d capability report is unreadable: {exc}",
                roots=["scripts/validate_arch_contract_runtime.py"],
            )
        ]
    if not isinstance(payload, list) or any(
        not isinstance(entry, str) for entry in payload
    ):
        return [
            SemanticIssue(
                code="unsupported_validator_feature",
                message="Stage 2.d capability report has an invalid envelope",
                roots=["scripts/validate_arch_contract_runtime.py"],
            )
        ]

    issues: list[SemanticIssue] = []
    for entry in payload:
        root, separator, message = entry.partition("|")
        if not separator or not root.strip() or not message.strip():
            issues.append(
                SemanticIssue(
                    code="unsupported_validator_feature",
                    message=(
                        "Stage 2.d capability report contains a malformed "
                        f"entry: {entry!r}"
                    ),
                    roots=["scripts/validate_arch_contract_runtime.py"],
                )
            )
            continue
        issues.append(
            SemanticIssue(
                code="unsupported_validator_feature",
                message=message.strip(),
                roots=[
                    root.strip(),
                    "scripts/validate_arch_contract_runtime.py",
                ],
            )
        )
    return issues


def _load_bindings_from_bundle(run_dir: Path) -> dict[str, int]:
    """Expand the semantic bundle fact into legacy shape-symbol aliases.

    This is the schema-1.0 compatibility adapter for R2C-063. Current
    constructor declarations consume :func:`_load_bundle_dimension_facts`
    directly, so a local feature symbol named ``L`` is not overwritten merely
    because legacy shape contracts also used ``L`` for time.

    The dry-run's numbers came only from the contract's declarations, so a
    model consistent with its own contract passed even when both disagreed
    with the data the delivery ships. The 2026-08-05 pdfgnn roll spent its
    entire smoke budget on that disagreement: the contract's arithmetic was
    self-consistent, and the bundle had 4 time steps.

    Only fires when a bundle exists with recorded axis facts, so every
    non-bundle paradigm keeps byte-identical behavior."""
    facts = _load_bundle_dimension_facts(run_dir)
    steps = facts.get(BUNDLE_TIME_AXIS_DIMENSION)
    if steps is None:
        return {}
    return {symbol: steps for symbol in _TIME_AXIS_SYMBOLS}


def _load_bindings_from_spec(spec: dict) -> dict[str, int]:
    """Bind symbols the PAPER pins to the paper's own values.

    A contract symbol named for a paper parameter (`P` lags, `K` horizon)
    binding to the generic default tests the model at a size the delivery
    will never see. The paper's stated value is both more realistic and
    already on disk, under the entry's name and each of its aliases.

    For a parameter explicitly bound by ``comparison.evaluation_protocol``,
    that role-typed fact is the sole paper-value authority. A legacy glossary
    or scale-lane number cannot fill a paper-unspecified protocol value or
    override a paper-stated one. Glossary aliases may carry the typed value to
    a contract symbol; fuzzy scale-lane matching is suppression-only."""
    crit = spec.get("critical_requirements")
    if not isinstance(crit, dict):
        return {}

    comparison = spec.get("comparison")
    protocol = comparison.get("evaluation_protocol") \
        if isinstance(comparison, dict) else None
    quantities = protocol.get("quantities") \
        if isinstance(protocol, dict) else None
    temporal_roles = {
        "context_length", "forecast_call_horizon",
        "validation_span", "test_span",
    }
    protocol_bindings: dict[str, dict] = {}
    if isinstance(quantities, list):
        for raw in quantities:
            if not isinstance(raw, dict):
                continue
            name = raw.get("parameter_name")
            role = raw.get("role")
            if not isinstance(name, str) or not name or role not in temporal_roles:
                continue
            quantity = evaluation_protocol_quantity_for_param(spec, name)
            if isinstance(quantity, dict):
                protocol_bindings[name] = quantity

    out: dict[str, int] = {}
    if not protocol_bindings:
        for key in ("param_glossary", "scale_dependent_hyperparameters"):
            for entry in crit.get(key) or []:
                if not isinstance(entry, dict):
                    continue
                value = entry.get("paper_value")
                if isinstance(value, bool) or not isinstance(value, int) \
                        or value <= 0:
                    continue
                for name in [entry.get("name")] + list(entry.get("aliases") or []):
                    if isinstance(name, str) and name.isidentifier():
                        out[name] = value
        return out

    def _typed_dimension(quantity: dict) -> int | None:
        if quantity.get("paper_value_status") != "paper_stated":
            return None
        try:
            return protocol_quantity_steps(
                quantity,
                path="comparison.evaluation_protocol.quantities[*]",
            )
        except ProtocolAxisUnresolved:
            return None

    # Exact parameter_name bindings authorize typed values. The glossary may
    # add spellings for contract symbols, but cannot authorize the number.
    typed: dict[str, int] = {}
    protocol_labels: dict[str, set[str]] = {
        name: evaluation_protocol_identity_labels(quantity)
        for name, quantity in protocol_bindings.items()
    }
    for name, quantity in protocol_bindings.items():
        value = _typed_dimension(quantity)
        if value is not None:
            for label in evaluation_protocol_identity_labels(quantity):
                if label.isidentifier():
                    typed[label] = value

    for entry in crit.get("param_glossary") or []:
        if not isinstance(entry, dict):
            continue
        labels = [entry.get("name"), *list(entry.get("aliases") or [])]
        labels = [label for label in labels if isinstance(label, str)]
        matched = [name for name in protocol_bindings if name in labels]
        if matched:
            for name in matched:
                protocol_labels[name].update(labels)
            # Strict spec validation rejects one glossary entry joining more
            # than one protocol carrier. If malformed input reaches this
            # consumer, suppress the ambiguous legacy value instead of
            # selecting a role.
            if len(matched) == 1:
                value = _typed_dimension(protocol_bindings[matched[0]])
                if value is not None:
                    for label in labels:
                        if label.isidentifier():
                            typed[label] = value
            continue

        value = entry.get("paper_value")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            continue
        for label in labels:
            if label.isidentifier():
                out[label] = value

    def _lane_name(value: str) -> str:
        prefix = value.split("(", 1)[0]
        return re.sub(r"[^a-z0-9]+", "_", prefix.lower()).strip("_")

    normalized_protocol_labels = {
        name: {_lane_name(label) for label in labels}
        for name, labels in protocol_labels.items()
    }
    for entry in crit.get("scale_dependent_hyperparameters") or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        lane_name = _lane_name(name) if isinstance(name, str) else ""
        if lane_name and any(
            lane_name in labels for labels in normalized_protocol_labels.values()
        ):
            continue
        value = entry.get("paper_value")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            continue
        for label in [name, *list(entry.get("aliases") or [])]:
            if isinstance(label, str) and label.isidentifier():
                out[label] = value

    # Typed protocol facts win any accidental spelling collision with an
    # unrelated legacy entry.
    out.update(typed)
    return out


def _resolve_constructor_args(
    contract: ArchContract,
    spec: dict,
    run_dir: Path,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Resolve schema-1.1 constructor kwargs without importing run code.

    Current construction is deterministic on the first Stage-2d pass: it uses
    explicit built-in dimensions, paper/spec facts, and semantic bundle facts.
    It deliberately does not read ``params.json`` (written later at Stage 2x)
    and never substitutes ``DEFAULT_BINDING`` for an unknown declaration.
    Schema 1.0 returns an empty mapping here and keeps its runner-side adapter.
    """
    if contract.schema_version == "1.0.0":
        return {}, []

    dimensions = {
        **DRY_RUN_BINDINGS,
        **_load_bindings_from_spec(spec),
        **_load_bundle_dimension_facts(run_dir),
    }
    resolved: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for block_name, block in contract.architecture.items():
        # The schema's version cross-check guarantees non-None for 1.1.
        declarations = block.constructor_args or {}
        block_args: dict[str, Any] = {}
        for arg_name, declaration in declarations.items():
            path = (
                f"arch_contract.architecture.{block_name}.constructor_args."
                f"{arg_name}"
            )
            if "literal" in declaration.model_fields_set:
                block_args[arg_name] = declaration.literal
                continue
            dimension = declaration.dimension
            if dimension not in dimensions:
                errors.append(
                    f"{path}.dimension={dimension!r} is unresolved. Schema "
                    "1.1.0 constructor dimensions do not read params.json and "
                    "do not fall back to DEFAULT_BINDING=8; declare a literal "
                    "or a dimension grounded by the spec or an explicit "
                    "bundle measurement."
                )
                continue
            block_args[arg_name] = dimensions[dimension]
        resolved[block_name] = block_args
    return resolved, errors


def _discover_builder_function_names(run_dir: Path) -> set[str]:
    """AST-walk method/training.py for top-level function names. Builders
    are conventionally named `build_*` (build_model, build_student,
    build_teacher); we collect all of them. The dry-run script picks the
    right one for each architecture block based on its name."""
    training_py = run_dir / "method" / "training.py"
    if not training_py.is_file():
        return set()
    try:
        tree = ast.parse(training_py.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("build_")
    }


def _constant_truth(node: ast.AST) -> bool | None:
    """Return a literal condition's truth value, otherwise ``None``.

    This intentionally does not evaluate names, calls, comparisons, or other
    code. It is only the bounded proof needed to reject laundering through
    ``if False`` (and equivalent literal constants) without pretending to
    solve Python control flow.
    """
    if isinstance(node, ast.Constant):
        return bool(node.value)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
        try:
            return bool(ast.literal_eval(node))
        except (ValueError, TypeError):
            return None
    return None


def _block_definitely_terminates(statements: list[ast.stmt]) -> bool:
    for statement in statements:
        if _statement_definitely_terminates(statement):
            return True
    return False


def _statement_definitely_terminates(statement: ast.stmt) -> bool:
    if isinstance(statement, (ast.Return, ast.Raise)):
        return True
    if isinstance(statement, ast.If):
        truth = _constant_truth(statement.test)
        if truth is True:
            return _block_definitely_terminates(statement.body)
        if truth is False:
            return _block_definitely_terminates(statement.orelse)
        return bool(statement.orelse) and (
            _block_definitely_terminates(statement.body)
            and _block_definitely_terminates(statement.orelse)
        )
    return False


class _StaticallyReachableCallCollector(ast.NodeVisitor):
    """Collect calls from non-statically-dead code in one top-level function.

    Nested function/class/lambda bodies are separate scopes and never count.
    Literal-dead branches and statements after a definite return/raise are
    skipped. Unknown branches and loop bodies remain *possibly reachable* and
    are inspected; this is a bounded static reachability check, not a claim
    that every runtime invocation executes the call.
    """

    def __init__(self) -> None:
        self.calls: set[str] = set()
        self.call_nodes: list[ast.Call] = []
        self.assignments: list[
            ast.Assign | ast.AnnAssign | ast.NamedExpr
        ] = []
        self.returns: list[ast.Return] = []

    def collect(self, function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
        self._visit_block(function.body)
        return self.calls

    def _visit_block(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            self.visit(statement)
            if _statement_definitely_terminates(statement):
                break

    def visit_Call(self, node: ast.Call) -> None:
        self.call_nodes.append(node)
        if isinstance(node.func, ast.Name):
            self.calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            dotted = _dotted_call_name(node.func)
            if dotted is not None:
                self.calls.add(dotted)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.assignments.append(node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.assignments.append(node)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.assignments.append(node)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self.returns.append(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Nested definitions are not executed merely because their enclosing
        # function is called. Their bodies cannot launder reachability.
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        truth = _constant_truth(node.test)
        if truth is True:
            self._visit_block(node.body)
        elif truth is False:
            self._visit_block(node.orelse)
        else:
            self._visit_block(node.body)
            self._visit_block(node.orelse)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.visit(node.test)
        truth = _constant_truth(node.test)
        if truth is True:
            self.visit(node.body)
        elif truth is False:
            self.visit(node.orelse)
        else:
            self.visit(node.body)
            self.visit(node.orelse)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        literal_empty = (
            isinstance(node.iter, (ast.Tuple, ast.List, ast.Set))
            and not node.iter.elts
        )
        if not literal_empty:
            self._visit_block(node.body)
        self._visit_block(node.orelse)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit(node.iter)
        self._visit_block(node.body)
        self._visit_block(node.orelse)

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        if _constant_truth(node.test) is not False:
            self._visit_block(node.body)
        self._visit_block(node.orelse)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
        self._visit_block(node.body)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        for item in node.items:
            self.visit(item.context_expr)
        self._visit_block(node.body)

    def visit_Try(self, node: ast.Try) -> None:
        self._visit_block(node.body)
        for handler in node.handlers:
            if handler.type is not None:
                self.visit(handler.type)
            self._visit_block(handler.body)
        self._visit_block(node.orelse)
        self._visit_block(node.finalbody)

    def visit_Match(self, node: ast.Match) -> None:
        self.visit(node.subject)
        for case in node.cases:
            if case.guard is not None:
                self.visit(case.guard)
            self._visit_block(case.body)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        for value in node.values:
            self.visit(value)
            truth = _constant_truth(value)
            if isinstance(node.op, ast.And) and truth is False:
                break
            if isinstance(node.op, ast.Or) and truth is True:
                break


def _dotted_call_name(node: ast.AST) -> str | None:
    """Return a simple dotted call spelling without evaluating expressions."""

    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def _reachable_top_level_calls(
    tree: ast.Module,
    *,
    entry: str,
) -> tuple[set[str], dict[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    calls = {
        name: _StaticallyReachableCallCollector().collect(node)
        for name, node in functions.items()
    }
    reachable_functions: set[str] = set()
    frontier = [entry]
    while frontier:
        name = frontier.pop()
        if name in reachable_functions:
            continue
        reachable_functions.add(name)
        frontier.extend(calls.get(name, set()) - reachable_functions)
    reachable_calls = {
        call
        for name in reachable_functions
        for call in calls.get(name, set())
    }
    return reachable_calls, functions


def _imported_callable_spellings(
    tree: ast.Module,
    *,
    module: str,
    callable_name: str,
) -> set[str]:
    """Find exact spellings that resolve to one declared imported helper."""

    spellings: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            imported_module = node.module or ""
            if node.level and imported_module == module.rsplit(".", 1)[-1]:
                imported_module = module
            if imported_module == module:
                for alias in node.names:
                    if alias.name == callable_name:
                        spellings.add(alias.asname or alias.name)
                continue
            module_parent, module_leaf = module.rsplit(".", 1)
            relative_parent = node.level and imported_module == ""
            absolute_parent = not node.level and imported_module == module_parent
            if relative_parent or absolute_parent:
                for alias in node.names:
                    if alias.name == module_leaf:
                        spellings.add(
                            (alias.asname or alias.name) + "." + callable_name
                        )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != module:
                    continue
                if alias.asname:
                    spellings.add(alias.asname + "." + callable_name)
                else:
                    spellings.add(module + "." + callable_name)
    return spellings


def _call_spelling(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return _dotted_call_name(node.func)
    return None


def _assignment_value(
    node: ast.Assign | ast.AnnAssign | ast.NamedExpr,
) -> ast.expr | None:
    return node.value


def _simple_assignment_targets(
    node: ast.Assign | ast.AnnAssign | ast.NamedExpr,
) -> set[str]:
    raw_targets: list[ast.expr]
    if isinstance(node, ast.Assign):
        raw_targets = list(node.targets)
    else:
        raw_targets = [node.target]
    return {
        target.id
        for target in raw_targets
        if isinstance(target, ast.Name)
    }


def _contains_node(root: ast.AST | None, target: ast.AST) -> bool:
    return root is not None and any(node is target for node in ast.walk(root))


def _loads_any(root: ast.AST | None, names: set[str]) -> bool:
    if root is None or not names:
        return False
    return any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in names
        for node in ast.walk(root)
    )


def _call_is_on_parameter(node: ast.Call, parameter: str) -> bool:
    current: ast.AST = node.func
    while isinstance(current, ast.Attribute):
        current = current.value
    return isinstance(current, ast.Name) and current.id == parameter


def _call_consumes_names(node: ast.Call, names: set[str]) -> bool:
    return any(_loads_any(argument, names) for argument in node.args) or any(
        _loads_any(keyword.value, names) for keyword in node.keywords
    )


def _reachable_function_collectors(
    tree: ast.Module,
    *,
    entry: str,
) -> dict[str, _StaticallyReachableCallCollector]:
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    collectors: dict[str, _StaticallyReachableCallCollector] = {}
    for name, function in functions.items():
        collector = _StaticallyReachableCallCollector()
        collector.collect(function)
        collectors[name] = collector
    reachable: set[str] = set()
    frontier = [entry]
    while frontier:
        name = frontier.pop()
        if name in reachable or name not in collectors:
            continue
        reachable.add(name)
        frontier.extend(collectors[name].calls - reachable)
    return {name: collectors[name] for name in reachable}


def _helper_result_reaches_supported_sink(
    tree: ast.Module,
    *,
    entry: str,
    helper_spellings: set[str],
    sink_kind: str,
    model_parameter: str | None = None,
) -> tuple[bool, int | None]:
    """Prove each reachable helper return reaches one closed supported sink.

    The bounded fitting grammar accepts either a direct argument to the exact
    constructed-model parameter or a value-flow into a loss followed by
    ``.backward()``.  The forecast grammar accepts only a value that reaches
    the declared entry's returned object.  Calls whose result is discarded or
    overwritten therefore cannot satisfy the proof.
    """

    collectors = _reachable_function_collectors(tree, entry=entry)
    helper_calls = [
        (collector, call)
        for collector in collectors.values()
        for call in collector.call_nodes
        if _call_spelling(call) in helper_spellings
    ]
    if not helper_calls:
        return False, None

    for collector, helper_call in helper_calls:
        if sink_kind == "forecast_return" and any(
            _contains_node(return_node.value, helper_call)
            for return_node in collector.returns
        ):
            continue
        if sink_kind == "fitting" and isinstance(model_parameter, str) and any(
            outer is not helper_call
            and _contains_node(outer, helper_call)
            and _call_is_on_parameter(outer, model_parameter)
            for outer in collector.call_nodes
        ):
            continue

        initial_assignments = [
            assignment
            for assignment in collector.assignments
            if _contains_node(_assignment_value(assignment), helper_call)
        ]
        if len(initial_assignments) != 1:
            return False, getattr(helper_call, "lineno", None)
        initial_assignment = initial_assignments[0]
        tainted = _simple_assignment_targets(initial_assignment)
        if not tainted:
            return False, getattr(helper_call, "lineno", None)

        def position(node: ast.AST) -> tuple[int, int]:
            return (
                int(getattr(node, "lineno", -1)),
                int(getattr(node, "col_offset", -1)),
            )

        events: list[tuple[tuple[int, int], int, ast.AST]] = []
        for assignment in collector.assignments:
            if assignment is not initial_assignment:
                events.append((position(assignment), 0, assignment))
        for call in collector.call_nodes:
            if call is not helper_call:
                events.append((position(call), 1, call))
        for return_node in collector.returns:
            events.append((position(return_node), 2, return_node))
        start = position(initial_assignment)
        consumed = False
        for event_position, _, event in sorted(events, key=lambda item: (item[0], item[1])):
            if event_position <= start:
                continue
            if isinstance(event, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                targets = _simple_assignment_targets(event)
                if not targets:
                    continue
                value_is_tainted = _loads_any(_assignment_value(event), tainted)
                tainted.difference_update(targets)
                if value_is_tainted:
                    tainted.update(targets)
                continue
            if isinstance(event, ast.Return):
                if sink_kind == "forecast_return" and _loads_any(
                    event.value, tainted
                ):
                    consumed = True
                    break
                continue
            if not isinstance(event, ast.Call):
                continue
            if sink_kind == "fitting":
                if (
                    isinstance(model_parameter, str)
                    and _call_is_on_parameter(event, model_parameter)
                    and _call_consumes_names(event, tainted)
                ):
                    consumed = True
                    break
                if (
                    isinstance(event.func, ast.Attribute)
                    and event.func.attr == "backward"
                    and _loads_any(event.func.value, tainted)
                ):
                    consumed = True
                    break
        if not consumed:
            return False, getattr(helper_call, "lineno", None)
    return True, None


def _node_position(node: ast.AST) -> tuple[int, int]:
    return (
        int(getattr(node, "lineno", -1)),
        int(getattr(node, "col_offset", -1)),
    )


def _resolve_simple_name_assignment(
    expression: ast.expr | None,
    collector: _StaticallyReachableCallCollector,
    *,
    before: ast.AST,
) -> ast.expr | None:
    """Resolve a bounded chain of simple-name assignments before one return."""

    current = expression
    seen: set[str] = set()
    while isinstance(current, ast.Name) and current.id not in seen:
        seen.add(current.id)
        candidates = [
            assignment
            for assignment in collector.assignments
            if _node_position(assignment) < _node_position(before)
            and current.id in _simple_assignment_targets(assignment)
        ]
        if not candidates:
            break
        current = _assignment_value(max(candidates, key=_node_position))
    return current


def _forecast_return_role_expressions(
    collector: _StaticallyReachableCallCollector,
    *,
    output_grammar: dict[str, Any],
) -> tuple[ast.Return, dict[str, list[ast.expr]]] | None:
    type_name = output_grammar.get("type_name")
    fields = output_grammar.get("fields") or {}
    distribution = output_grammar.get("distribution_params") or {}
    allowed_key_sets = distribution.get("allowed_key_sets") or []
    if not isinstance(type_name, str) or not isinstance(fields, dict):
        return None
    field_order = [
        fields.get("mean"),
        fields.get("variance"),
        fields.get("samples"),
        fields.get("distribution_params"),
    ]
    if any(not isinstance(field, str) for field in field_order):
        return None
    supported: list[tuple[ast.Return, dict[str, list[ast.expr]]]] = []
    for return_node in collector.returns:
        result_expression = _resolve_simple_name_assignment(
            return_node.value, collector, before=return_node
        )
        if not isinstance(result_expression, ast.Call):
            continue
        spelling = _call_spelling(result_expression)
        if not isinstance(spelling, str) or spelling.rsplit(".", 1)[-1] != type_name:
            continue
        values: dict[str, ast.expr] = {}
        for index, argument in enumerate(result_expression.args):
            if index < len(field_order):
                values[str(field_order[index])] = argument
        for keyword in result_expression.keywords:
            if isinstance(keyword.arg, str):
                values[keyword.arg] = keyword.value
        if any(field not in values for field in field_order):
            continue
        distribution_expression = _resolve_simple_name_assignment(
            values[str(fields["distribution_params"])],
            collector,
            before=return_node,
        )
        if not isinstance(distribution_expression, ast.Dict):
            continue
        distribution_values = {
            key.value: value
            for key, value in zip(
                distribution_expression.keys,
                distribution_expression.values,
            )
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        key_set = next(
            (
                candidate
                for candidate in allowed_key_sets
                if isinstance(candidate, dict)
                and set(distribution_values) == set(candidate.values())
            ),
            None,
        )
        if key_set is None:
            continue
        supported.append((return_node, {
            "location": [
                values[str(fields["mean"])],
                distribution_values[str(key_set["location"])],
            ],
            "variance": [values[str(fields["variance"])]],
            "samples": [values[str(fields["samples"])]],
            "distribution_scale": [
                distribution_values[str(key_set["scale"])]
            ],
        }))
    if len(supported) != 1:
        return None
    return supported[0]


def _helper_call_reaches_expressions(
    collector: _StaticallyReachableCallCollector,
    helper_call: ast.Call,
    *,
    expressions: list[ast.expr],
    return_node: ast.Return,
) -> bool:
    if all(_contains_node(expression, helper_call) for expression in expressions):
        return True
    initial_assignments = [
        assignment
        for assignment in collector.assignments
        if _contains_node(_assignment_value(assignment), helper_call)
    ]
    if len(initial_assignments) != 1:
        return False
    initial_assignment = initial_assignments[0]
    tainted = _simple_assignment_targets(initial_assignment)
    if not tainted:
        return False
    start = _node_position(initial_assignment)
    stop = _node_position(return_node)
    for assignment in sorted(collector.assignments, key=_node_position):
        position = _node_position(assignment)
        if assignment is initial_assignment or position <= start or position >= stop:
            continue
        targets = _simple_assignment_targets(assignment)
        if not targets:
            continue
        value_is_tainted = _loads_any(_assignment_value(assignment), tainted)
        tainted.difference_update(targets)
        if value_is_tainted:
            tainted.update(targets)
    return all(
        _contains_node(expression, helper_call) or _loads_any(expression, tainted)
        for expression in expressions
    )


def _forecast_inverse_roles_are_closed(
    tree: ast.Module,
    *,
    entry: str,
    helper_spellings: set[str],
    output_grammar: dict[str, Any],
) -> tuple[bool, str]:
    collectors = _reachable_function_collectors(tree, entry=entry)
    collector = collectors.get(entry)
    if collector is None:
        return False, "declared forecast entry is absent"
    role_expressions = _forecast_return_role_expressions(
        collector, output_grammar=output_grammar
    )
    if role_expressions is None:
        return False, (
            "forecast must have one supported closed ForecastResult return with "
            "an inline or simple-name closed distribution mapping"
        )
    return_node, expressions_by_role = role_expressions
    calls_by_role: dict[str, list[ast.Call]] = {
        role: [] for role in expressions_by_role
    }
    for reachable_collector in collectors.values():
        for call in reachable_collector.call_nodes:
            if _call_spelling(call) not in helper_spellings:
                continue
            output_role = next(
                (
                    keyword.value.value
                    for keyword in call.keywords
                    if keyword.arg == "output_role"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ),
                None,
            )
            if output_role not in calls_by_role:
                return False, (
                    "every inverse call must declare one literal required "
                    "output_role"
                )
            if reachable_collector is not collector:
                return False, (
                    "inverse calls in a separate helper function are outside "
                    "the bounded forecast value-flow grammar"
                )
            calls_by_role[output_role].append(call)
    missing = sorted(role for role, calls in calls_by_role.items() if not calls)
    if missing:
        return False, f"missing required inverse output role(s) {missing!r}"
    for role, calls in calls_by_role.items():
        for call in calls:
            if not _helper_call_reaches_expressions(
                collector,
                call,
                expressions=expressions_by_role[role],
                return_node=return_node,
            ):
                return False, (
                    f"inverse output_role={role!r} at line "
                    f"{getattr(call, 'lineno', '?')} does not reach its exact "
                    "ForecastResult field"
                )
    return True, ""


def _mapping_key_expression(
    expression: ast.expr | None,
    *,
    key: str,
) -> ast.expr | None:
    """Return one exact value expression from a bounded mapping constructor."""

    mapping = expression
    if isinstance(mapping, ast.Call):
        spelling = _call_spelling(mapping)
        leaf = spelling.rsplit(".", 1)[-1] if spelling else ""
        if leaf not in {"dict", "UserDict"}:
            return None
        if mapping.args:
            if len(mapping.args) != 1 or mapping.keywords:
                return None
            mapping = mapping.args[0]
        else:
            matches = [
                keyword.value
                for keyword in mapping.keywords
                if keyword.arg == key
            ]
            return matches[0] if len(matches) == 1 else None
    if not isinstance(mapping, ast.Dict):
        return None
    matches = [
        value
        for raw_key, value in zip(mapping.keys, mapping.values)
        if isinstance(raw_key, ast.Constant) and raw_key.value == key
    ]
    return matches[0] if len(matches) == 1 else None


def _training_history_result_flow_is_closed(
    tree: ast.Module,
    *,
    entry: str,
    helper_spellings: set[str],
    result_key: str,
) -> tuple[bool, str]:
    """Prove the fixed record reaches one exact returned mapping key."""

    collectors = _reachable_function_collectors(tree, entry=entry)
    entry_collector = collectors.get(entry)
    if entry_collector is None:
        return False, "declared fitting entry is absent"
    helper_calls = [
        (collector, call)
        for collector in collectors.values()
        for call in collector.call_nodes
        if _call_spelling(call) in helper_spellings
    ]
    if not helper_calls:
        return False, "fixed recorder is not reached"
    if any(collector is not entry_collector for collector, _ in helper_calls):
        return False, (
            "recorder calls in a separate helper function are outside the "
            "bounded training-result value-flow grammar"
        )

    supported_returns: list[tuple[ast.Return, ast.expr]] = []
    for return_node in entry_collector.returns:
        result_expression = _resolve_simple_name_assignment(
            return_node.value,
            entry_collector,
            before=return_node,
        )
        value_expression = _mapping_key_expression(
            result_expression,
            key=result_key,
        )
        if value_expression is not None:
            supported_returns.append((return_node, value_expression))
    if len(supported_returns) != 1:
        return False, (
            "fitting must have one supported dict/UserDict return containing "
            f"exact key {result_key!r}"
        )
    return_node, value_expression = supported_returns[0]
    for _, call in helper_calls:
        if not _helper_call_reaches_expressions(
            entry_collector,
            call,
            expressions=[value_expression],
            return_node=return_node,
        ):
            return False, (
                f"recorder return at line {getattr(call, 'lineno', '?')} "
                f"does not reach exact returned mapping key {result_key!r}"
            )
    return True, ""


def _check_training_history_reachability(
    contract: AnyArchContract,
    run_dir: Path,
    execution_plan: dict[str, Any] | None,
) -> list[str]:
    """Require the fixed recorder and its result on the live fitting path."""

    if not execution_plan:
        return []
    history = execution_plan.get("training_history") or {}
    helper = history.get("helper") or {}
    training = history.get("training_call") or {}
    module = str(helper.get("module") or "")
    callable_name = str(helper.get("record_callable") or "")
    entry = str(training.get("callable") or "")
    result_key = str((training.get("history_result") or {}).get("key") or "")
    training_path = run_dir / "method" / "training.py"
    if not training_path.is_file():
        return [
            "training_history_execution training_loop: method/training.py is "
            "missing"
        ]
    try:
        tree = ast.parse(training_path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [
            "training_history_execution training_loop: cannot parse "
            f"method/training.py ({exc})"
        ]
    reachable_calls, functions = _reachable_top_level_calls(tree, entry=entry)
    if entry not in functions:
        return [
            "training_history_execution training_loop: declared entry "
            f"{entry!r} has no top-level definition in method/training.py"
        ]
    spellings = _imported_callable_spellings(
        tree,
        module=module,
        callable_name=callable_name,
    )
    if not spellings or not (spellings & reachable_calls):
        return [
            "training_history_execution training_loop: declared entry "
            f"{entry!r} does not reach fixed {module}.{callable_name} through "
            "a non-statically-dead top-level call path"
        ]
    closed, detail = _training_history_result_flow_is_closed(
        tree,
        entry=entry,
        helper_spellings=spellings,
        result_key=result_key,
    )
    if closed:
        return []
    if detail.startswith(
        "recorder calls in a separate helper function are outside"
    ):
        return [
            _PIPELINE_REACHABILITY_PREFIX
            + " training_history_execution training_loop: "
            + detail
        ]
    return [
        "training_history_execution training_loop: fixed "
        f"{module}.{callable_name} does not flow into the declared fitting "
        f"result: {detail}; call-and-discard or a different result key is not "
        "supported"
    ]


def _check_target_scaling_reachability(
    contract: AnyArchContract,
    run_dir: Path,
    execution_plan: dict[str, Any] | None,
) -> list[str]:
    """Require fixed fit/transform/inverse helpers on live declared paths."""

    if not execution_plan:
        return []
    scaling = execution_plan.get("target_scaling") or {}
    helper = scaling.get("helper") or {}
    training = scaling.get("training_call") or {}
    checks = (
        (
            run_dir / "method" / "training.py",
            str(training.get("callable") or ""),
            (
                str(helper.get("fit_callable") or ""),
                str(helper.get("transform_callable") or ""),
            ),
            "training_loop",
        ),
        (
            run_dir / "method" / "method.py",
            str((execution_plan.get("forecast_call") or {}).get("callable") or ""),
            (str(helper.get("inverse_callable") or ""),),
            "pluggable_component",
        ),
    )
    module = str(helper.get("module") or "")
    model_parameter = str(
        ((training.get("parameters") or {}).get("model")) or ""
    )
    errors: list[str] = []
    for path, entry, required, root in checks:
        if not path.is_file():
            errors.append(
                f"target_scaling_execution {root}: {path.relative_to(run_dir)} "
                "is missing"
            )
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            errors.append(
                f"target_scaling_execution {root}: cannot parse "
                f"{path.relative_to(run_dir)} ({exc})"
            )
            continue
        reachable_calls, functions = _reachable_top_level_calls(
            tree, entry=entry
        )
        if entry not in functions:
            errors.append(
                f"target_scaling_execution {root}: declared entry {entry!r} "
                f"has no top-level definition in {path.relative_to(run_dir)}"
            )
            continue
        for callable_name in required:
            spellings = _imported_callable_spellings(
                tree, module=module, callable_name=callable_name
            )
            if not spellings or not (spellings & reachable_calls):
                errors.append(
                    f"target_scaling_execution {root}: declared entry "
                    f"{entry!r} does not reach fixed {module}."
                    f"{callable_name} through a non-statically-dead top-level "
                    "call path"
                )
                continue
            if callable_name == helper.get("inverse_callable"):
                closed, detail = _forecast_inverse_roles_are_closed(
                    tree,
                    entry=entry,
                    helper_spellings=spellings,
                    output_grammar=dict(
                        execution_plan.get("output_grammar") or {}
                    ),
                )
                if not closed:
                    errors.append(
                        f"target_scaling_execution {root}: fixed {module}."
                        f"{callable_name} does not close every target-valued "
                        f"ForecastResult role: {detail}; call-and-discard or "
                        "partial inversion is not supported"
                    )
                continue
            sink_kind = (
                "fitting"
                if callable_name == helper.get("transform_callable")
                else None
            )
            if sink_kind is None:
                continue
            flows, line = _helper_result_reaches_supported_sink(
                tree,
                entry=entry,
                helper_spellings=spellings,
                sink_kind=sink_kind,
                model_parameter=model_parameter or None,
            )
            if flows:
                continue
            if sink_kind == "fitting":
                expected_sink = (
                    "the exact constructed-model call or a loss value followed "
                    "by .backward()"
                )
            else:
                expected_sink = "the object returned by the forecast entry"
            location = f" at line {line}" if isinstance(line, int) else ""
            errors.append(
                f"target_scaling_execution {root}: fixed {module}."
                f"{callable_name} return{location} does not flow into "
                f"{expected_sink}; call-and-discard is not a supported "
                "target-scaling grammar"
            )
    return errors


def _check_relational_preparation_reachability(
    contract: AnyArchContract, run_dir: Path
) -> list[str]:
    """Require the declared pure preparation seam on the fitting path.

    Executing a dead helper would certify scaffolding instead of the training
    code. This bounded intra-module call graph counts only calls in top-level
    functions' non-statically-dead bodies: nested definitions, literal-false
    branches, and code after definite termination do not create edges. It does
    not claim an unknown branch or loop executes on every invocation.
    """
    relational = contract.relational_indexing
    if relational is None:
        return []
    if contract.training_loop is None:
        # The structural validator reports the owner-specific contract error.
        return []

    training_path = run_dir / "method" / "training.py"
    if not training_path.is_file():
        return [
            "relational_indexing preparation reachability: method/training.py "
            "is missing; architecture-coder owns the declared preparation "
            "callable and fitting entry point."
        ]
    try:
        tree = ast.parse(training_path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [
            "relational_indexing preparation reachability: cannot parse "
            f"method/training.py ({exc}); the package syntax validator should "
            "route this architecture-coder output first."
        ]

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    target = relational.preparation_callable.name
    entry = contract.training_loop.function_name
    if target not in functions:
        return [
            "arch_contract.relational_indexing.preparation_callable names "
            f"method.training.{target}, but method/training.py has no top-level "
            "callable with that name. Architecture-coder must implement the "
            "declared pure batch-preparation seam."
        ]
    if entry not in functions:
        return [
            f"arch_contract.training_loop.function_name={entry!r} has no "
            "top-level definition in method/training.py; cannot prove the "
            f"relational preparation callable {target!r} reaches fitting."
        ]

    calls: dict[str, set[str]] = {}
    for name, node in functions.items():
        calls[name] = _StaticallyReachableCallCollector().collect(node)

    reachable: set[str] = set()
    frontier = [entry]
    while frontier:
        name = frontier.pop()
        if name in reachable:
            continue
        reachable.add(name)
        frontier.extend(calls.get(name, set()) - reachable)
    if target not in reachable:
        return [
            f"arch_contract.training_loop.function_name={entry!r} does not "
            "reach relational_indexing.preparation_callable "
            f"method.training.{target} through the non-statically-dead "
            "top-level call graph. A nested or literal-dead fixture helper "
            "cannot certify fitting; call it from a reachable declared "
            "training path so node roots and graph endpoints share one mapping."
        ]
    return []


# Fixed, paradigm-agnostic runner body. Executed via subprocess with a
# preamble (built by `_build_dry_run_script`) that defines CONTRACT, BINDINGS,
# BLOCK_BUILDERS, CONSTRUCTOR_ARGS, BUILDER_KWARG_CANDIDATES, DEFAULT_BINDING.
# Plain string (not an f-string) so its dict/regex literals need no escaping.
_DRY_RUN_RUNNER = r'''
_errors = []
_skips = []
_unsupported = []

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    import numpy as _np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

import importlib
from collections.abc import Mapping as _RuntimeMapping
_IS_V2 = CONTRACT.get("schema_version") == "2.0.0"
if _IS_V2:
    try:
        import method
        _method_import_error = None
    except Exception as _e:
        method = None
        _method_import_error = _e
else:
    import method
    _method_import_error = None
try:
    _training_mod = importlib.import_module("method.training")
    _training_import_error = None
except Exception as _e:
    _training_mod = None
    _training_import_error = _e
try:
    _data_mod = importlib.import_module("method.data")
    _data_import_error = None
except Exception as _e:
    _data_mod = None
    _data_import_error = _e
_target_scaling_plan = (FORECASTING_EXECUTION_PLAN or {}).get(
    "target_scaling"
)
if _target_scaling_plan:
    _target_scaling_helper = _target_scaling_plan.get("helper") or {}
    try:
        _target_scaling_mod = importlib.import_module(
            _target_scaling_helper.get("module")
        )
        _target_scaling_import_error = None
    except Exception as _e:
        _target_scaling_mod = None
        _target_scaling_import_error = _e
else:
    _target_scaling_mod = None
    _target_scaling_import_error = None
_training_history_plan = (FORECASTING_EXECUTION_PLAN or {}).get(
    "training_history"
)
if _training_history_plan:
    _training_history_helper = _training_history_plan.get("helper") or {}
    try:
        _training_history_mod = importlib.import_module(
            _training_history_helper.get("module")
        )
        _training_history_import_error = None
    except Exception as _e:
        _training_history_mod = None
        _training_history_import_error = _e
else:
    _training_history_mod = None
    _training_history_import_error = None


def _is_environment_gap(error):
    if error is None:
        return False
    if "R2C_OFFLINE" in str(error) or isinstance(error, FileNotFoundError):
        return True
    if isinstance(error, ModuleNotFoundError):
        missing = getattr(error, "name", None)
        return not (
            isinstance(missing, str)
            and (missing == "method" or missing.startswith("method."))
        )
    if isinstance(error, ImportError):
        # ``cannot import name`` is normally a broken generated package
        # export, not an absent validation-environment dependency.
        return "cannot import name" not in str(error)
    return False


if _IS_V2:
    _needs_training_module = bool(
        CONTRACT.get("training_loop")
        or CONTRACT.get("relational_indexing")
        or any(BLOCK_BUILDERS.values())
    )
    _needs_data_module = bool(
        (CONTRACT.get("data_loader") or {}).get("load_data_returns")
    )
    for _module_root, _module_name, _module_error, _module_required in (
        ("architecture", "method", _method_import_error, True),
        (
            "training_loop", "method.training", _training_import_error,
            _needs_training_module,
        ),
        ("data_loader", "method.data", _data_import_error, _needs_data_module),
        (
            "target_scaling_execution",
            "method.target_scaling",
            _target_scaling_import_error,
            bool(_target_scaling_plan),
        ),
        (
            "training_history_execution",
            "method.training_history",
            _training_history_import_error,
            bool(_training_history_plan),
        ),
    ):
        if _module_required and _is_environment_gap(_module_error):
            _unsupported.append(
                _module_root + "|" + _module_name
                + " could not be imported in the Stage 2.d environment ("
                + type(_module_error).__name__ + ": "
                + str(_module_error)[:160] + ")"
            )


def _dims(inner):
    out = []
    for p in (x.strip() for x in inner.split(",")):
        if not p:
            continue
        if p.lstrip("-").isdigit():
            out.append(int(p))
        else:
            out.append(BINDINGS.get(p, DEFAULT_BINDING))
    return tuple(out)


def _zeros(dims, kind):
    """A zeros array of the requested kind. (ok, value)."""
    if kind == "ndarray":
        if not _HAS_NUMPY:
            return False, None
        return True, _np.zeros(dims)
    if not _HAS_TORCH:
        return False, None
    return True, torch.zeros(dims)


def _typed_zeros(dims, kind, dtype, label, device=None):
    """Construct one exact typed v2 array fixture. (ok, value)."""
    if kind == "ndarray":
        if not _HAS_NUMPY:
            _unsupported.append(
                label + "|declared ndarray fixture requires numpy, but the "
                "Stage 2.d validation environment cannot import it"
            )
            return False, None
        _dtype = {
            "float32": _np.float32,
            "float64": _np.float64,
            "int32": _np.int32,
            "int64": _np.int64,
            "bool": _np.bool_,
        }.get(dtype)
        return (False, None) if _dtype is None else (True, _np.zeros(dims, dtype=_dtype))
    if kind == "tensor":
        if not _HAS_TORCH:
            _unsupported.append(
                label + "|declared tensor fixture requires torch, but the "
                "Stage 2.d validation environment cannot import it"
            )
            return False, None
        _dtype = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool,
        }.get(dtype)
        return (
            (False, None)
            if _dtype is None
            else (True, torch.zeros(dims, dtype=_dtype, device=device or "cpu"))
        )
    return False, None


def _synth_typed(spec, label):
    kind = (spec or {}).get("kind")
    if kind == "opaque":
        return False, None
    if not (spec or {}).get("synthesizable"):
        _errors.append(label + ": resolved typed fixture is not synthesizable")
        return False, None
    if kind in ("tensor", "ndarray"):
        if spec.get("shape") is None:
            _errors.append(label + ": resolved typed array fixture has no shape")
            return False, None
        return _typed_zeros(
            tuple(spec.get("shape") or ()), kind, spec.get("dtype"), label,
            spec.get("device"),
        )
    if kind == "scalar":
        if not _HAS_NUMPY:
            _unsupported.append(
                label + "|declared exact-width scalar fixture requires numpy, "
                "but the Stage 2.d validation environment cannot import it"
            )
            return False, None
        _dtype = {
            "float32": _np.float32,
            "float64": _np.float64,
            "int32": _np.int32,
            "int64": _np.int64,
            "bool": _np.bool_,
        }.get(spec.get("dtype"))
        if _dtype is None:
            return False, None
        return True, _dtype(spec.get("scalar_value"))
    return False, None


def _typed_dtype_name(value):
    if _HAS_TORCH and isinstance(value, torch.Tensor):
        return str(value.dtype).removeprefix("torch.")
    if _HAS_NUMPY and isinstance(value, (_np.ndarray, _np.generic)):
        return str(value.dtype)
    return None


def _check_typed_value(value, spec, label):
    """Check one returned runtime value against its resolved v2 descriptor."""
    kind = (spec or {}).get("kind")
    if kind == "opaque":
        return
    if kind == "tensor":
        if not (_HAS_TORCH and isinstance(value, torch.Tensor)):
            _errors.append(label + ": contract says kind=tensor but got " + type(value).__name__)
            return
    elif kind == "ndarray":
        if not (_HAS_NUMPY and isinstance(value, _np.ndarray)):
            _errors.append(label + ": contract says kind=ndarray but got " + type(value).__name__)
            return
    elif kind == "scalar":
        if (isinstance(value, (dict, list, tuple))
                or (_HAS_TORCH and isinstance(value, torch.Tensor))
                or (_HAS_NUMPY and isinstance(value, _np.ndarray))):
            _errors.append(label + ": contract says kind=scalar but got container " + type(value).__name__)
            return
        expected_dtype = spec.get("dtype")
        if expected_dtype == "bool":
            valid = type(value) is bool or (_HAS_NUMPY and isinstance(value, _np.bool_))
        elif expected_dtype in ("int32", "int64"):
            valid = (type(value) is int
                     or (_HAS_NUMPY and isinstance(value, _np.integer)))
        else:
            valid = (type(value) is float
                     or (_HAS_NUMPY and isinstance(value, _np.floating)))
        if not valid:
            _errors.append(label + ": scalar dtype " + repr(expected_dtype)
                           + " rejects " + type(value).__name__)
            return
        actual_dtype = _typed_dtype_name(value)
        if actual_dtype is not None and actual_dtype != expected_dtype:
            _errors.append(label + ": contract declares dtype=" + str(expected_dtype)
                           + " but got dtype=" + str(actual_dtype))
        return
    else:
        _errors.append(label + ": missing or unsupported resolved fixture kind " + repr(kind))
        return

    expected_shape = tuple(spec.get("shape") or ())
    actual_shape = tuple(value.shape)
    if actual_shape != expected_shape:
        _errors.append(label + ": contract declares shape " + repr(expected_shape)
                       + " but got " + repr(actual_shape))
    actual_dtype = _typed_dtype_name(value)
    if actual_dtype != spec.get("dtype"):
        _errors.append(label + ": contract declares dtype=" + str(spec.get("dtype"))
                       + " but got dtype=" + str(actual_dtype))
    if kind == "tensor":
        expected_device = spec.get("device") or "cpu"
        actual_device = str(value.device)
        if actual_device != expected_device:
            _errors.append(
                label + ": contract declares device=" + expected_device
                + " but got device=" + actual_device
            )
            # Non-CPU devices such as meta cannot necessarily be materialized
            # for value-bound checks below. Container, shape, dtype, and device
            # disagreement have already been recorded.
            return
    constraint = spec.get("constraint") or {}
    element_count = (
        int(value.numel())
        if _HAS_TORCH and isinstance(value, torch.Tensor)
        else int(value.size)
    )
    if constraint.get("kind") in ("index", "class_id") and element_count:
        if _HAS_TORCH and isinstance(value, torch.Tensor):
            minimum = int(value.min().item())
            maximum = int(value.max().item())
        else:
            minimum = int(value.min())
            maximum = int(value.max())
        upper_bound = spec.get("constraint_upper_bound")
        if minimum < 0 or upper_bound is None or maximum >= upper_bound:
            _errors.append(
                label + ": " + str(constraint.get("kind"))
                + " values must stay in [0, " + str(upper_bound)
                + "); observed min=" + str(minimum) + ", max=" + str(maximum)
            )


if _IS_V2:
    # Resolve/synthesize every declared typed surface, including metadata-only
    # optimizer and family-extension slots.  Opaque values are intentional
    # non-execution declarations; every other supported descriptor must have a
    # realizable fixture even when no public callable consumes it in this run.
    for _fixture_root, _fixture_spec in TYPED_FIXTURES.items():
        if _fixture_spec.get("kind") == "opaque":
            continue
        if not _fixture_spec.get("synthesizable"):
            _errors.append(
                _fixture_root + ": resolved typed fixture is not synthesizable"
            )
            continue
        _fixture_ok, _ = _synth_typed(_fixture_spec, _fixture_root)
        if not _fixture_ok and not any(
            _entry.startswith(_fixture_root + "|") for _entry in _unsupported
        ):
            _errors.append(
                _fixture_root + ": resolved typed fixture could not be synthesized"
            )


def _synth(shape_str, kind="tensor"):
    """(ok, value). ok=False -> unsynthesizable (opaque/free-form) -> skip method.

    `kind` picks the array family for array-shaped inputs: torch tensors
    (the default, supervised-ML paradigms) or numpy arrays. Pure-numpy
    packages fail at runtime when fed tensors (np operations reject them),
    which rejected a correct ROMAN25 motion-planning package at 2.d twice
    (researcher request, 2026-07-14)."""
    s = (shape_str or "").strip()
    m = re.match(r"^list\[\((.*)\)\]$", s)
    if m:
        ok, val = _zeros(_dims(m.group(1)), kind)
        return (True, [val]) if ok else (False, None)
    if s in ("scalar", "float"):
        return True, 0.0
    if s == "int":
        return True, 1
    if s.startswith("(") and s.endswith(")"):
        inner = s[1:-1].strip()
        if inner == "":
            return _zeros((), kind)
        if inner == "opaque":
            return False, None
        return _zeros(_dims(inner), kind)
    return False, None


def _sig_array_kind(sig):
    """Which array family a signature's inputs should be synthesized as.

    Per-signature: an ndarray-returning method gets numpy inputs, a
    tensor-returning one gets tensors. Methods whose return type carries
    no array signal (bool/float/dict/...) inherit the contract default."""
    t = (sig.get("output_type") or "")
    if t == "ndarray":
        return "ndarray"
    if t == "tensor":
        return "tensor"
    return _DEFAULT_KIND


def _build_inputs(input_dict, kind="tensor", label=None):
    """Positional args in declared order. (ok, args) or (False, offending_name)."""
    args = []
    for name, shp in input_dict.items():
        if _IS_V2:
            fixture_root = str(label) + ".input." + name
            fixture = TYPED_FIXTURES.get(fixture_root)
            if fixture is None:
                _errors.append(fixture_root + ": no resolved typed fixture")
                return False, name
            ok, val = _synth_typed(fixture, fixture_root)
        else:
            ok, val = _synth(shp, kind)
        if not ok:
            return False, name
        args.append(val)
    return True, args


def _build_typed_call(input_dict, callable_obj, label, value_overrides=None):
    """Build an exact v2 call from named descriptors and explicit overrides."""
    try:
        signature = inspect.signature(callable_obj)
    except Exception as exc:
        _errors.append(
            label + ": cannot inspect callable signature ("
            + type(exc).__name__ + ": " + str(exc) + ")"
        )
        return False, "signature", [], {}

    parameters = signature.parameters
    has_var_keyword = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    missing = sorted(
        name
        for name, parameter in parameters.items()
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and parameter.default is inspect.Parameter.empty
        and name not in input_dict
    )
    unexpected = sorted(
        name
        for name in input_dict
        if name not in parameters and not has_var_keyword
    )
    variadic = sorted(
        name
        for name in input_dict
        if name in parameters
        and parameters[name].kind == inspect.Parameter.VAR_POSITIONAL
    )
    if missing:
        _errors.append(
            label + ": typed contract omits required callable input(s) "
            + repr(missing) + "; declare each input by its exact parameter name"
        )
    if unexpected:
        _errors.append(
            label + ": typed contract declares unexpected callable input(s) "
            + repr(unexpected) + "; callable signature " + str(signature)
            + " has no **kwargs"
        )
    if variadic:
        _errors.append(
            label + ": typed contract cannot bind *args parameter(s) by name "
            + repr(variadic)
        )
    if missing or unexpected or variadic:
        return False, "signature", [], {}

    value_overrides = value_overrides or {}
    unexpected_overrides = sorted(set(value_overrides) - set(input_dict))
    if unexpected_overrides:
        _errors.append(
            label + ": relational value override(s) are not declared typed "
            "inputs " + repr(unexpected_overrides)
        )
        return False, "signature", [], {}

    values = {}
    for name in input_dict:
        if name in value_overrides:
            value = value_overrides[name]
            fixture_root = label + ".input." + name
            fixture = TYPED_FIXTURES.get(fixture_root)
            if fixture is None:
                _errors.append(fixture_root + ": no resolved typed fixture")
                return False, name, [], {}
            if fixture.get("kind") != "opaque":
                _check_typed_value(value, fixture, fixture_root)
            values[name] = value
            continue
        fixture_root = label + ".input." + name
        fixture = TYPED_FIXTURES.get(fixture_root)
        if fixture is None:
            _errors.append(fixture_root + ": no resolved typed fixture")
            return False, name, [], {}
        ok, value = _synth_typed(fixture, fixture_root)
        if not ok:
            return False, name, [], {}
        values[name] = value

    args = []
    kwargs = {}
    seen_omitted_positional_default = False
    for name, parameter in parameters.items():
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            if name in values:
                if seen_omitted_positional_default:
                    _errors.append(
                        label + ": positional-only input " + repr(name)
                        + " follows an undeclared optional positional-only "
                        "parameter; the typed mapping cannot skip that slot"
                    )
                    return False, name, [], {}
                args.append(values[name])
            elif parameter.default is not inspect.Parameter.empty:
                seen_omitted_positional_default = True
        elif name in values:
            kwargs[name] = values[name]
    for name, value in values.items():
        if name not in parameters:
            kwargs[name] = value
    return True, None, args, kwargs


def _check_output(out, sig, label):
    if _IS_V2:
        if sig.get("output") is None:
            return
        fixture = TYPED_FIXTURES.get(label + ".output")
        if fixture is None:
            _errors.append(label + ".output: no resolved typed fixture")
            return
        _check_typed_value(out, fixture, label + ".output")
        return
    t = sig.get("output_type")
    if t == "tensor":
        if not (_HAS_TORCH and isinstance(out, torch.Tensor)):
            _errors.append(label + ": contract says output_type=tensor but got " + type(out).__name__)
    elif t == "ndarray":
        if not (_HAS_NUMPY and isinstance(out, _np.ndarray)):
            _errors.append(label + ": contract says output_type=ndarray but got " + type(out).__name__)
    elif t == "dict":
        keys = list((sig.get("output_keys") or {}).keys())
        if not isinstance(out, dict):
            _errors.append(label + ": contract says output_type=dict but got " + type(out).__name__)
        else:
            missing = [k for k in keys if k not in out]
            if missing:
                _errors.append(label + " output dict missing keys: " + repr(missing) + "; got: " + repr(list(out.keys())))
    elif t == "tuple":
        n = len(sig.get("output_shapes") or [])
        if not isinstance(out, tuple):
            _errors.append(label + ": contract says output_type=tuple but got " + type(out).__name__)
        elif len(out) != n:
            _errors.append(label + " output tuple has " + str(len(out)) + " elements; contract declared " + str(n))
    elif t == "list":
        if not isinstance(out, list):
            _errors.append(label + ": contract says output_type=list but got " + type(out).__name__)
    elif t in ("bool", "float", "int", "scalar"):
        # Lenient scalar check: a scalar return must not be a container
        # (tensor/ndarray/list/dict/tuple). numpy/torch scalar types pass.
        if (isinstance(out, (dict, list, tuple))
                or (_HAS_TORCH and isinstance(out, torch.Tensor) and out.ndim > 0)
                or (_HAS_NUMPY and isinstance(out, _np.ndarray) and out.ndim > 0)):
            _errors.append(label + ": contract says output_type=" + str(t) + " (scalar) but got a container " + type(out).__name__)


def _exercise_method(inst, callable_obj, sig, label):
    if callable_obj is None:
        _skips.append(label + ": no resolvable callable; skipped (static validators cover existence)")
        return
    if _IS_V2:
        ok, args_or_name, args, kwargs = _build_typed_call(
            sig.get("input") or {}, callable_obj, label
        )
    else:
        ok, args_or_name = _build_inputs(
            sig.get("input") or {}, _sig_array_kind(sig), label=label
        )
        args = args_or_name if ok else []
        kwargs = {}
    if not ok:
        _skips.append(label + ": input " + repr(args_or_name) + " not synthesizable (opaque/free-form); call skipped")
        return
    try:
        out = callable_obj(*args, **kwargs)
    except Exception as _e:
        _errors.append(label + " call raised: " + type(_e).__name__ + ": " + str(_e))
        return
    _check_output(out, sig, label)


def _synth_local_dataset(loader_returns, sig):
    """Write a tiny contract-shaped dataset under a temp dir so load_data reads
    it instead of falling through to a torchvision/MNIST download (audit
    2026-06-25 R1). Returns the temp dir path, or None if it can't be built
    (loader takes no `path` kwarg, or a return shape is opaque/non-tensor).
    Only ranks are checked by the dry-run, so each axis gets size 2; the JSON
    keys are the contract's return names, which is exactly what the generated
    load_data reads for its local-data path."""
    if "path" not in sig.parameters or not loader_returns:
        return None
    import json as _json
    import tempfile as _tempfile

    def _nest_shape(dims, leaf):
        if not dims:
            return leaf
        return [_nest_shape(dims[1:], leaf) for _ in range(dims[0])]

    payload = {}
    for _name, _shape in loader_returns.items():
        if _IS_V2:
            _fixture = TYPED_FIXTURES.get(
                "data_loader.load_data_returns." + _name
            )
            if not _fixture or _fixture.get("kind") == "opaque":
                return None
            if _fixture.get("kind") == "scalar":
                payload[_name] = _fixture.get("scalar_value")
            else:
                _dtype = _fixture.get("dtype")
                _leaf = False if _dtype == "bool" else (
                    0 if _dtype in ("int32", "int64") else 0.0
                )
                payload[_name] = _nest_shape(
                    tuple(_fixture.get("shape") or ()), _leaf
                )
            continue
        _s = str(_shape or "").strip()
        if not (_s.startswith("(") and _s.endswith(")")):
            return None
        _inner = _s[1:-1].strip()
        if _inner in ("", "opaque"):
            return None
        payload[_name] = _nest_shape(
            tuple(2 for p in _inner.split(",") if p.strip()), 0.0
        )
    _d = _tempfile.mkdtemp(prefix="r2c_synth_data_")
    with open(_d + "/data.json", "w", encoding="utf-8") as _f:
        _json.dump(payload, _f)
    return _d


# Contract-level array-family default for signatures whose return type
# carries no array signal (bool/float/dict/...): a contract that declares
# ndarray returns and NO tensor returns is a pure-numpy package, so those
# methods get numpy inputs too. Anything else keeps the torch default.
def _contract_output_types():
    _types = set()
    for _b in (CONTRACT.get("architecture") or {}).values():
        if not isinstance(_b, dict):
            continue
        _fw = _b.get("forward")
        if isinstance(_fw, dict):
            _types.add(_fw.get("output_type") or "")
        for _m in (_b.get("additional_methods") or {}).values():
            if isinstance(_m, dict):
                _types.add(_m.get("output_type") or "")
    return _types

_OUT_TYPES = _contract_output_types()
_DEFAULT_KIND = ("ndarray" if ("ndarray" in _OUT_TYPES and "tensor" not in _OUT_TYPES)
                 else "tensor")


# --- relational_indexing pure preparation-callable gate ---
_rel = CONTRACT.get("relational_indexing")
if _rel:
    _prep_decl = _rel.get("preparation_callable") or {}
    _prep_name = _prep_decl.get("name")
    _prep_backend = _prep_decl.get("tensor_backend")
    _prep_fn = getattr(_training_mod, _prep_name, None) if _training_mod else None
    _rel_prefix = (
        "relational_indexing[entity_axis=" + repr(_rel.get("entity_axis"))
        + ", graph_root=" + repr(_rel.get("graph_root"))
        + ", roots=" + repr(sorted((_rel.get("coindexed_roots") or {}).keys()))
        + "]"
    )

    def _rel_error(case, message):
        _errors.append(_rel_prefix + " case=" + case + ": " + message)

    def _rel_backend_array(value, dtype):
        if _prep_backend == "torch":
            if not _HAS_TORCH:
                return None
            _dtype = torch.long if dtype == "int64" else torch.float32
            return torch.as_tensor(value, dtype=_dtype)
        if _prep_backend == "numpy":
            if not _HAS_NUMPY:
                return None
            _dtype = _np.int64 if dtype == "int64" else _np.float32
            return _np.asarray(value, dtype=_dtype)
        return None

    def _rel_backend_ok(value):
        if _prep_backend == "torch":
            return _HAS_TORCH and isinstance(value, torch.Tensor)
        if _prep_backend == "numpy":
            return _HAS_NUMPY and isinstance(value, _np.ndarray)
        return False

    def _rel_numpy(value):
        if _HAS_TORCH and isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if _HAS_NUMPY and isinstance(value, _np.ndarray):
            return value
        return None

    def _rel_take(value, positions, axis):
        return _np.take(value, _np.asarray(positions, dtype=_np.int64), axis=axis)

    def _rel_expected_graph(source_graph, positions):
        if _rel.get("representation") == "dense_adjacency":
            idx = _np.asarray(positions, dtype=_np.int64)
            return source_graph[_np.ix_(idx, idx)]
        mapping = _np.full((5,), -1, dtype=_np.int64)
        mapping[_np.asarray(positions, dtype=_np.int64)] = _np.arange(
            len(positions), dtype=_np.int64
        )
        src = source_graph[0]
        dst = source_graph[1]
        keep = (mapping[src] >= 0) & (mapping[dst] >= 0)
        return _np.stack((mapping[src[keep]], mapping[dst[keep]])).astype(
            _np.int64, copy=False
        )

    def _rel_expected_degree(graph, count):
        kind = _rel.get("degree_kind")
        if _rel.get("representation") == "dense_adjacency":
            axis = 0 if kind == "in_degree" else 1
            return _np.count_nonzero(graph, axis=axis).astype(_np.float32)
        endpoint_row = 1 if kind == "in_degree" else 0
        if graph.shape[1] == 0:
            return _np.zeros((count,), dtype=_np.float32)
        return _np.bincount(
            graph[endpoint_row], minlength=count
        ).astype(_np.float32)

    def _rel_check_array(case, label, actual, expected, *, integral=False):
        if not _rel_backend_ok(actual):
            _rel_error(
                case,
                label + " must use declared tensor_backend="
                + repr(_prep_backend) + "; got " + type(actual).__name__,
            )
            return
        actual_np = _rel_numpy(actual)
        if integral and actual_np.dtype.kind not in "iu":
            _rel_error(
                case,
                label + " must have integer dtype; got " + str(actual_np.dtype),
            )
        if actual_np.shape != expected.shape or not _np.array_equal(actual_np, expected):
            _rel_error(
                case,
                label + " does not preserve the declared identity mapping; expected "
                + repr(expected.tolist()) + " but got " + repr(actual_np.tolist()),
            )

    def _rel_check_sparse_graph(case, actual, expected, batch_size):
        label = (
            "graph " + repr(_rel.get("graph_root")) + " with "
            + repr(_rel.get("edge_orientation")) + " orientation"
        )
        if not _rel_backend_ok(actual):
            _rel_error(
                case,
                label + " must use declared tensor_backend="
                + repr(_prep_backend) + "; got " + type(actual).__name__,
            )
            return
        actual_np = _rel_numpy(actual)
        if actual_np.dtype.kind not in "iu":
            _rel_error(
                case,
                label + " must have integer dtype; got " + str(actual_np.dtype),
            )
        if actual_np.ndim != 2 or actual_np.shape[0] != 2:
            _rel_error(
                case, "sparse graph must have shape (2, E); got "
                + repr(actual_np.shape),
            )
            return
        if actual_np.size and (
            actual_np.min() < 0 or actual_np.max() >= batch_size
        ):
            _rel_error(
                case,
                "sparse graph endpoints escape local_batch_positions "
                + "[0, " + str(batch_size) + "); observed min="
                + str(actual_np.min()) + ", max=" + str(actual_np.max()),
            )
        # Edge-index column order is not semantic. Compare the sorted directed
        # columns instead; duplicates remain in the lists, so multiplicity and
        # source-to-destination orientation are still exact.
        actual_edges = sorted(tuple(edge) for edge in actual_np.T.tolist())
        expected_edges = sorted(tuple(edge) for edge in expected.T.tolist())
        if actual_edges != expected_edges:
            _rel_error(
                case,
                label + " has the wrong directed edge multiset (column order "
                "is ignored, multiplicity is preserved); expected "
                + repr(expected_edges) + " but got " + repr(actual_edges),
            )

    def _rel_fixture_root(root, axis, root_index, source_ids, source_degrees):
        if root == _rel.get("stable_entity_id_root"):
            return source_ids.copy()
        if root == _rel.get("degree_root"):
            return source_degrees.copy()
        shape = [2] * (axis + 1)
        shape[axis] = 5
        values = _np.arange(_np.prod(shape), dtype=_np.float32).reshape(shape)
        return values + _np.float32((root_index + 1) * 1000)

    def _exercise_relational_case(case, positions):
        source_ids = _np.asarray([101, 205, 309, 412, 518], dtype=_np.int64)
        sparse = _np.asarray(
            # 0->2 appears twice so the equality check also pins edge
            # multiplicity while remaining insensitive to column order.
            [[0, 0, 0, 1, 2, 4, 3], [1, 2, 2, 2, 4, 0, 1]],
            dtype=_np.int64,
        )
        if _rel.get("representation") == "sparse_edge_index":
            source_graph = sparse
        else:
            source_graph = _np.zeros((5, 5), dtype=_np.float32)
            for edge_number, (src, dst) in enumerate(zip(sparse[0], sparse[1]), start=1):
                # Unique directed weights make a silent transpose observable.
                source_graph[src, dst] = _np.float32(edge_number)

        if _rel.get("degree_semantics") == "source_graph":
            # Deliberately not derivable from the fixture edges: source-graph
            # structural metadata must be mapped, not silently recomputed.
            source_degrees = _np.asarray([11, 13, 17, 19, 23], dtype=_np.float32)
        else:
            source_degrees = _rel_expected_degree(source_graph, 5)

        roots = _rel.get("coindexed_roots") or {}
        source_roots = {
            root: _rel_fixture_root(root, axis, index, source_ids, source_degrees)
            for index, (root, axis) in enumerate(roots.items())
        }
        batch_ids = source_ids[_np.asarray(positions, dtype=_np.int64)]
        graph_dtype = (
            "int64" if _rel.get("representation") == "sparse_edge_index"
            else "float32"
        )
        kwargs = {
            "source_entity_ids": _rel_backend_array(source_ids, "int64"),
            "batch_entity_ids": _rel_backend_array(batch_ids, "int64"),
            "coindexed": {
                root: _rel_backend_array(value, "int64" if value.dtype.kind in "iu" else "float32")
                for root, value in source_roots.items()
            },
            "graph": _rel_backend_array(source_graph, graph_dtype),
            "degrees": (
                _rel_backend_array(source_degrees, "float32")
                if _rel.get("degree_root") is not None else None
            ),
        }
        try:
            result = _prep_fn(**kwargs)
        except Exception as exc:
            _rel_error(
                case,
                "preparation callable method.training." + str(_prep_name)
                + " raised " + type(exc).__name__ + ": " + str(exc),
            )
            return
        if not isinstance(result, dict):
            _rel_error(
                case,
                "preparation callable must return a dict with observable "
                "identity mappings; got " + type(result).__name__,
            )
            return
        required = {
            "local_to_source", "source_to_local", "coindexed", "graph",
            "degrees", "output_entity_ids",
        }
        missing = sorted(required - set(result))
        if missing:
            _rel_error(
                case,
                "preparation result is missing required key(s) " + repr(missing)
                + "; no fallback identity or row-order inference is allowed",
            )
            return

        expected_positions = list(positions)
        if _rel.get("output_order") == "canonical_source_order":
            expected_positions = sorted(expected_positions)
        expected_local_to_source = _np.asarray(expected_positions, dtype=_np.int64)
        expected_source_to_local = _np.full((5,), -1, dtype=_np.int64)
        expected_source_to_local[expected_local_to_source] = _np.arange(
            len(expected_positions), dtype=_np.int64
        )
        expected_graph = _rel_expected_graph(source_graph, expected_positions)
        expected_ids = source_ids[expected_local_to_source]

        _rel_check_array(
            case, "local_to_source", result["local_to_source"],
            expected_local_to_source, integral=True,
        )
        _rel_check_array(
            case, "source_to_local", result["source_to_local"],
            expected_source_to_local, integral=True,
        )
        _rel_check_array(
            case, "output_entity_ids", result["output_entity_ids"],
            expected_ids, integral=True,
        )
        graph_np = _rel_numpy(result["graph"])
        if _rel.get("representation") == "sparse_edge_index":
            _rel_check_sparse_graph(
                case, result["graph"], expected_graph, len(expected_positions)
            )
        else:
            _rel_check_array(
                case,
                "graph " + repr(_rel.get("graph_root")) + " with "
                + repr(_rel.get("edge_orientation")) + " orientation",
                result["graph"], expected_graph,
            )
            if graph_np is not None and graph_np.shape != (
                len(expected_positions), len(expected_positions)
            ):
                _rel_error(
                    case,
                    "dense graph must have local shape (B, B)="
                    + repr((len(expected_positions), len(expected_positions)))
                    + "; got " + repr(graph_np.shape),
                )

        returned_roots = result.get("coindexed")
        if not isinstance(returned_roots, dict):
            _rel_error(case, "coindexed must be a dict keyed by declared roots")
        else:
            missing_roots = sorted(set(roots) - set(returned_roots))
            extra_roots = sorted(set(returned_roots) - set(roots))
            if missing_roots or extra_roots:
                _rel_error(
                    case,
                    "coindexed roots differ from the contract; missing="
                    + repr(missing_roots) + ", extra=" + repr(extra_roots),
                )
            for root, axis in roots.items():
                if root not in returned_roots:
                    continue
                if root == _rel.get("degree_root") and _rel.get("degree_semantics") == "induced_graph":
                    expected = _rel_expected_degree(expected_graph, len(expected_positions))
                else:
                    expected = _rel_take(source_roots[root], expected_positions, axis)
                _rel_check_array(
                    case, "coindexed root " + repr(root), returned_roots[root],
                    expected,
                    integral=(expected.dtype.kind in "iu"),
                )

        if _rel.get("degree_root") is None:
            if result.get("degrees") is not None:
                _rel_error(case, "degrees must be None when degree_root is null")
        else:
            if _rel.get("degree_semantics") == "source_graph":
                expected_degree = source_degrees[expected_local_to_source]
            else:
                expected_degree = _rel_expected_degree(expected_graph, len(expected_positions))
            _rel_check_array(
                case,
                "degree root " + repr(_rel.get("degree_root")) + " under "
                + repr(_rel.get("degree_semantics")) + " semantics",
                result["degrees"], expected_degree,
            )

    if _prep_fn is None:
        if not (_IS_V2 and _is_environment_gap(_training_import_error)):
            _rel_error(
                "setup",
                "declared preparation callable method.training." + str(_prep_name)
                + " is not importable; architecture-coder owns this callable",
            )
    elif _prep_backend not in {"torch", "numpy"}:
        _rel_error("setup", "unsupported tensor_backend=" + repr(_prep_backend))
    elif (_prep_backend == "torch" and not _HAS_TORCH) or (
        _prep_backend == "numpy" and not _HAS_NUMPY
    ):
        _rel_error(
            "setup",
            "declared tensor_backend=" + repr(_prep_backend)
            + " is unavailable; the preparation seam cannot be certified",
        )
    else:
        _exercise_relational_case("canonical_full", [0, 1, 2, 3, 4])
        _exercise_relational_case("coherent_permutation", [4, 2, 0, 3, 1])
        _exercise_relational_case("induced_subset_[4,2,0]", [4, 2, 0])
        # Nodes 4 and 3 have no edge between them in the asymmetric source
        # graph: this exercises the exact empty edge-index / zero-adjacency
        # shape without replacing the source graph with a different fixture.
        _exercise_relational_case("empty_induced_edges_[4,3]", [4, 3])

# --- data_loader.load_data dry-run (only when method/data.py declares load_data) ---
_loader_returns = (CONTRACT.get("data_loader") or {}).get("load_data_returns") or {}
_load_data_fn = getattr(_data_mod, "load_data", None) if _data_mod is not None else None
if _load_data_fn is None:
    if _loader_returns:
        if _IS_V2 and _is_environment_gap(_data_import_error):
            _skips.append(
                "data_loader: method.data import is blocked by a recorded "
                "Stage 2.d environment capability gap"
            )
        elif _IS_V2:
            _errors.append(
                "data_loader.load_data_returns: typed contract declares "
                "loader outputs, but method.data.load_data is not importable"
                + (
                    " (method.data import raised "
                    + type(_data_import_error).__name__ + ": "
                    + str(_data_import_error)[:160] + ")"
                    if _data_import_error is not None else ""
                )
            )
        else:
            _skips.append("data_loader: method/data.py has no `load_data` (paradigm uses a different loader, e.g. load_problem); return-shape dry-run skipped")
else:
    _n_expected = len(_loader_returns)
    _names = list(_loader_returns.keys())
    _shape_strs = list(_loader_returns.values())
    _sig = inspect.signature(_load_data_fn)
    _ld_kwargs = {}
    if not _IS_V2:
        for _name in ("pool_size", "train_size", "n_test"):
            if _name in _sig.parameters:
                _ld_kwargs[_name] = 4
        if "seed" in _sig.parameters:
            _ld_kwargs["seed"] = 0
    _ld_result = None
    try:
        _ld_result = _load_data_fn(**_ld_kwargs)
    except Exception as _e:
        # Two environment-shaped failures get the synthetic-fixture retry, not
        # an error: a missing optional data dependency (e.g. torchvision on a
        # torchvision-absent host, audit 2026-06-25 R1), and an offline
        # download refusal (the loader honored R2C_OFFLINE, which this dry-run
        # always sets — the 2026-07-13 network-stall fix). Both are gaps in
        # the environment, not code defects. Anything else is a real error.
        _env_gap = (isinstance(_e, (ModuleNotFoundError, ImportError, FileNotFoundError))
                    or "R2C_OFFLINE" in str(_e))
        if not _env_gap:
            _errors.append("data_loader.load_data(" + repr(_ld_kwargs) + ") raised: " + type(_e).__name__ + ": " + str(_e))
        else:
            if isinstance(_e, FileNotFoundError):
                _gap_desc = "no local dataset available (download-free loader)"
            elif "R2C_OFFLINE" in str(_e):
                _gap_desc = "network download refused (offline dry-run)"
            else:
                _gap_desc = "optional data dependency missing"
            _synth_dir = _synth_local_dataset(_loader_returns, _sig)
            if _synth_dir is not None:
                try:
                    _ld_result = _load_data_fn(path=_synth_dir, **_ld_kwargs)
                except Exception as _e2:
                    if _IS_V2 and _is_environment_gap(_e2):
                        _unsupported.append(
                            "data_loader|" + _gap_desc + " ("
                            + type(_e).__name__ + ") and the exact typed "
                            "synthetic-fixture retry remains unavailable ("
                            + type(_e2).__name__ + ": " + str(_e2)[:120] + ")"
                        )
                    elif _IS_V2:
                        _errors.append(
                            "data_loader.load_data: " + _gap_desc + " ("
                            + type(_e).__name__ + ") but the exact typed "
                            "synthetic-fixture retry raised "
                            + type(_e2).__name__ + ": " + str(_e2)[:120]
                        )
                    else:
                        _skips.append("data_loader.load_data: " + _gap_desc + " (" + type(_e).__name__ + ") and the synthetic-fixture retry failed (" + type(_e2).__name__ + ": " + str(_e2)[:120] + "); return-shape dry-run skipped")
            else:
                if _IS_V2:
                    _unsupported.append(
                        "data_loader|" + _gap_desc + " ("
                        + type(_e).__name__ + ": " + str(_e)[:120]
                        + ") and the typed contract does not provide a "
                        "supported local-loader fixture path"
                    )
                else:
                    _skips.append("data_loader.load_data: " + _gap_desc + " (" + type(_e).__name__ + ": " + str(_e)[:120] + ") and no synthetic fixture could be built from the contract; return-shape dry-run skipped")
    if _ld_result is not None:
        # `load_data_returns` is a NAME -> shape map, and two return
        # structures satisfy it: a tuple (positional returns, the committed
        # supervised-ML family interface) or a dict keyed by those names (the
        # generic gap-family interface — its data.py template returns a dict
        # of named arrays by design). Until 2026-08-04 only the tuple form
        # was accepted, so the pipeline's own scaffold contradicted its own
        # dry-run the first time a gap run's loader actually returned data
        # (pdfgnn: R2C-052's bundling made load_data succeed, exposing the
        # latent mismatch as a stage-2d degrade).
        _entries = None
        if isinstance(_ld_result, dict):
            _missing_keys = [_k for _k in _names if _k not in _ld_result]
            if _missing_keys:
                _errors.append("data_loader.load_data: contract declares named returns " + repr(_names) + " but the returned dict is missing " + repr(_missing_keys) + " (actual keys: " + repr(sorted(str(_k) for _k in _ld_result.keys())[:12]) + "). Declare load_data_returns using the loader's real table names.")
            else:
                _entries = [(_k, _loader_returns[_k], _ld_result[_k]) for _k in _names]
        elif isinstance(_ld_result, tuple):
            if len(_ld_result) != _n_expected:
                _errors.append("data_loader.load_data: contract declares " + str(_n_expected) + " returns " + repr(_names) + " but function returned " + str(len(_ld_result)) + " values. Reconcile arch_contract.json with method/data.py.")
            else:
                _entries = list(zip(_names, _shape_strs, _ld_result))
        else:
            _errors.append("data_loader.load_data: contract declares " + str(_n_expected) + " returns; function returned a " + type(_ld_result).__name__ + ". Return a tuple (positional returns) or a dict keyed by the declared names.")
        if _entries:
            for _i, (_name, _shape_str, _val) in enumerate(_entries):
                if _IS_V2:
                    _fixture_root = "data_loader.load_data_returns." + _name
                    _fixture = TYPED_FIXTURES.get(_fixture_root)
                    if _fixture is None:
                        _errors.append(_fixture_root + ": no resolved typed fixture")
                    else:
                        _check_typed_value(_val, _fixture, _fixture_root)
                    continue
                if not hasattr(_val, "shape"):
                    continue
                _actual = tuple(_val.shape)
                if _shape_str.startswith("(") and _shape_str.endswith(")"):
                    _inner = _shape_str[1:-1].strip()
                    if _inner in ("", "opaque"):
                        continue
                    _parts = [p.strip() for p in _inner.split(",") if p.strip()]
                    if len(_parts) != len(_actual):
                        _errors.append("data_loader.load_data return #" + str(_i) + " (" + _name + "): contract declares rank " + str(len(_parts)) + " (shape " + repr(_shape_str) + ") but tensor has rank " + str(len(_actual)) + " (actual " + repr(_actual) + ").")


_CONSTRUCTION_FAILED = object()
_constructed_architecture = {}


def _construct_declared(_callable, _label, _kwargs):
    """Validate an exact keyword declaration, then invoke only that mapping."""
    try:
        _sig = inspect.signature(_callable)
    except Exception as _e:
        _errors.append(
            _label + ": cannot inspect constructor signature ("
            + type(_e).__name__ + ": " + str(_e) + ")"
        )
        return _CONSTRUCTION_FAILED

    _parameters = _sig.parameters
    _has_var_keyword = any(
        _p.kind == inspect.Parameter.VAR_KEYWORD
        for _p in _parameters.values()
    )
    _required_positional_only = sorted(
        _name
        for _name, _p in _parameters.items()
        if _p.kind == inspect.Parameter.POSITIONAL_ONLY
        and _p.default is inspect.Parameter.empty
    )
    _declared_positional_only = sorted(
        _name
        for _name in _kwargs
        if _name in _parameters
        and _parameters[_name].kind == inspect.Parameter.POSITIONAL_ONLY
    )
    _missing = sorted(
        _name
        for _name, _p in _parameters.items()
        if _p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and _p.default is inspect.Parameter.empty
        and _name not in _kwargs
    )
    _unexpected = sorted(
        _name
        for _name in _kwargs
        if (
            _name not in _parameters
            or _parameters[_name].kind == inspect.Parameter.VAR_POSITIONAL
        )
        and not _has_var_keyword
    )

    _invalid = False
    if _required_positional_only:
        _errors.append(
            _label + ": required positional-only parameter(s) "
            + repr(_required_positional_only)
            + " cannot be satisfied by the contract's keyword-only "
            + "constructor mapping"
        )
        _invalid = True
    if _declared_positional_only:
        _errors.append(
            _label + ": constructor_args declares positional-only "
            + "parameter(s) " + repr(_declared_positional_only)
            + "; declared arguments are passed only as keywords"
        )
        _invalid = True
    if _missing:
        _errors.append(
            _label + ": missing required keyword argument(s) " + repr(_missing)
            + "; declare each one in this architecture block's constructor_args"
        )
        _invalid = True
    if _unexpected:
        _errors.append(
            _label + ": unexpected constructor_args keyword(s) "
            + repr(_unexpected) + "; callable signature " + str(_sig)
            + " has no **kwargs"
        )
        _invalid = True
    if _invalid:
        return _CONSTRUCTION_FAILED

    try:
        _built = _callable(**_kwargs)
    except Exception as _e:
        _errors.append(
            _label + "(" + repr(_kwargs) + ") raised: "
            + type(_e).__name__ + ": " + str(_e)
        )
        return _CONSTRUCTION_FAILED
    if _built is None:
        _errors.append(
            _label + "(" + repr(_kwargs)
            + ") returned None; expected a constructed model instance"
        )
        return _CONSTRUCTION_FAILED
    return _built


# --- architecture blocks ---
for _block_key, _block in (CONTRACT.get("architecture") or {}).items():
    _class_name = _block.get("class_name")
    _builder = BLOCK_BUILDERS.get(_block_key)
    _inst = None
    _current_constructor = CONTRACT.get("schema_version") in ("1.1.0", "2.0.0")
    if _builder:
        # Builder path (supervised-ML): a builder exists, so failure is a real error.
        _fn = (
            getattr(method, _builder, None) if method is not None else None
        ) or (getattr(_training_mod, _builder, None) if _training_mod else None)
        if _fn is None:
            if not (
                _IS_V2
                and (
                    _is_environment_gap(_method_import_error)
                    or _is_environment_gap(_training_import_error)
                )
            ):
                _errors.append("architecture." + _block_key + ": builder " + repr(_builder) + " not found on method/method.training")
            continue
        if _current_constructor:
            _bk = CONSTRUCTOR_ARGS.get(_block_key, {})
            _inst = _construct_declared(
                _fn,
                "architecture." + _block_key + ".constructor_args via builder " + _builder,
                _bk,
            )
            if _inst is _CONSTRUCTION_FAILED:
                continue
        else:
            _sig = inspect.signature(_fn)
            _bk = {_n: _v for _n, _v in BUILDER_KWARG_CANDIDATES.items() if _n in _sig.parameters}
            try:
                _inst = _fn(**_bk)
            except Exception as _e:
                _errors.append("architecture." + _block_key + ": " + _builder + "(" + repr(_bk) + ") raised: " + type(_e).__name__ + ": " + str(_e))
                continue
    else:
        _cls = (
            getattr(method, _class_name, None) if method is not None else None
        )
        if _cls is None:
            if not (_IS_V2 and _is_environment_gap(_method_import_error)):
                _errors.append("architecture." + _block_key + ": class " + repr(_class_name) + " not importable from method package")
            continue
        if _current_constructor:
            _bk = CONSTRUCTOR_ARGS.get(_block_key, {})
            _inst = _construct_declared(
                _cls,
                "architecture." + _block_key + ".constructor_args via class " + str(_class_name),
                _bk,
            )
            if _inst is _CONSTRUCTION_FAILED:
                continue
        else:
            # Legacy no-builder path: construction failure remains a SKIP.
            # Archived non-ML contracts never had constructor metadata, so a
            # required class argument cannot honestly be synthesized here.
            try:
                _inst = _cls()
            except Exception as _e:
                _skips.append("architecture." + _block_key + ": cannot construct " + str(_class_name) + "() with no args (" + type(_e).__name__ + "); runtime dry-run skipped (static validators cover existence)")
                continue

    _constructed_architecture[_block_key] = _inst

    # Exercise the primary forward slot. Resolve to __call__ or
    # an explicit `forward` attribute; if neither exists (e.g. a planner whose
    # primary method is `step`), skip (the additional_methods below cover the
    # real named methods, and static validators cover existence).
    _fwd = _block.get("forward")
    _relational_bound_block = (
        (RELATIONAL_EXECUTION_PLAN or {}).get("architecture_block")
    )
    if _fwd and _block_key != _relational_bound_block:
        _callable = None
        if callable(_inst) and not isinstance(_inst, type):
            _callable = _inst
        elif hasattr(_inst, "forward"):
            _callable = getattr(_inst, "forward")
        _exercise_method(_inst, _callable, _fwd, "architecture." + _block_key + ".forward")

    # Exercise each additional_method by its real declared name.
    for _mname, _msig in (_block.get("additional_methods") or {}).items():
        _mfn = getattr(_inst, _mname, None)
        if _mfn is None:
            _errors.append("architecture." + _block_key + "." + _mname + ": contract declares this method but " + str(_class_name) + " has no such attribute")
            continue
        _method_label = "architecture." + _block_key + "." + _mname
        if _IS_V2:
            _method_label = (
                "architecture." + _block_key + ".additional_methods." + _mname
            )
        _exercise_method(_inst, _mfn, _msig, _method_label)


def _rel_execution_array(value, fixture, label):
    """Cast exact relational values using one declared typed fixture."""
    kind = (fixture or {}).get("kind")
    dtype = (fixture or {}).get("dtype")
    if kind == "tensor":
        if not _HAS_TORCH:
            return None
        torch_dtype = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool,
        }.get(dtype)
        if torch_dtype is None:
            _errors.append(label + ": unsupported relational tensor dtype " + repr(dtype))
            return None
        return torch.as_tensor(value, dtype=torch_dtype, device=fixture.get("device") or "cpu")
    if kind == "ndarray":
        if not _HAS_NUMPY:
            return None
        numpy_dtype = {
            "float32": _np.float32,
            "float64": _np.float64,
            "int32": _np.int32,
            "int64": _np.int64,
            "bool": _np.bool_,
        }.get(dtype)
        if numpy_dtype is None:
            _errors.append(label + ": unsupported relational ndarray dtype " + repr(dtype))
            return None
        return _np.asarray(value, dtype=numpy_dtype)
    _errors.append(label + ": relational value requires an array fixture; got " + repr(kind))
    return None


def _rel_values_equal(left, right):
    left_np = _rel_numpy(left)
    right_np = _rel_numpy(right)
    return (
        left_np is not None
        and right_np is not None
        and left_np.shape == right_np.shape
        and _np.array_equal(left_np, right_np)
    )


def _rel_check_real_typed_convention(case, label, value, typed_root):
    """Check typed container/dtype/device without fixing dynamic graph extents."""
    spec = TYPED_FIXTURES.get(typed_root) or {}
    kind = spec.get("kind")
    if kind == "tensor":
        if not (_HAS_TORCH and isinstance(value, torch.Tensor)):
            _rel_error(
                case,
                label + " must use typed container tensor from "
                + repr(typed_root) + "; got " + type(value).__name__,
            )
            return
        if str(value.device) != (spec.get("device") or "cpu"):
            _rel_error(
                case,
                label + " must use device=" + repr(spec.get("device") or "cpu")
                + " from " + repr(typed_root) + "; got " + str(value.device),
            )
            return
    elif kind == "ndarray":
        if not (_HAS_NUMPY and isinstance(value, _np.ndarray)):
            _rel_error(
                case,
                label + " must use typed container ndarray from "
                + repr(typed_root) + "; got " + type(value).__name__,
            )
            return
    else:
        _rel_error(
            case,
            label + " has unsupported typed convention " + repr(kind)
            + " at " + repr(typed_root),
        )
        return
    actual_dtype = _typed_dtype_name(value)
    if actual_dtype != spec.get("dtype"):
        _rel_error(
            case,
            label + " must use dtype=" + repr(spec.get("dtype"))
            + " from " + repr(typed_root) + "; got " + repr(actual_dtype),
        )


def _rel_real_positions(plan, phase):
    return list(plan.get(phase + "_positions") or [])


def _rel_real_source_fixture(plan):
    """Build one asymmetric source graph from the declared typed roots."""
    count = int(plan["entity_count"])
    fitting_roots = plan["fitting_roots"]
    source_ids_np = _np.asarray(
        [101, 205, 309, 412, 518]
        + [1000 + index for index in range(max(0, count - 5))],
        dtype=_np.int64,
    )[:count]

    graph_spec = TYPED_FIXTURES[fitting_roots["graph"]]
    numpy_dtypes = {
        "float32": _np.float32,
        "float64": _np.float64,
        "int32": _np.int32,
        "int64": _np.int64,
        "bool": _np.bool_,
    }
    source_ids_np = source_ids_np.astype(
        numpy_dtypes[TYPED_FIXTURES[
            fitting_roots["source_entity_ids"]
        ]["dtype"]],
        copy=False,
    )
    if _rel.get("representation") == "sparse_edge_index":
        edge_count = int((graph_spec.get("shape") or [2, 0])[1])
        candidates = [
            (0, 1), (0, 2), (0, 2), (1, 2), (2, min(4, count - 1)),
            (min(4, count - 1), 0), (min(3, count - 1), 1),
        ]
        candidates = [
            edge for edge in candidates
            if edge[0] < count and edge[1] < count
        ]
        if edge_count and not candidates:
            _rel_error(
                "real_training_call",
                "cannot construct an asymmetric sparse fixture for "
                + str(count) + " entities",
            )
            return None
        edges = [
            candidates[index % len(candidates)]
            for index in range(edge_count)
        ]
        source_graph_np = (
            _np.asarray(edges, dtype=_np.int64).T
            if edges else _np.empty((2, 0), dtype=_np.int64)
        )
    else:
        source_graph_np = _np.zeros((count, count), dtype=_np.float32)
        candidates = [
            (0, 1), (0, 2), (1, 2), (2, min(4, count - 1)),
            (min(4, count - 1), 0), (min(3, count - 1), 1),
        ]
        for edge_number, (source, destination) in enumerate(candidates, start=1):
            if source < count and destination < count:
                source_graph_np[source, destination] = _np.float32(edge_number)
    source_graph_np = source_graph_np.astype(
        numpy_dtypes[graph_spec["dtype"]], copy=False
    )

    if _rel.get("degree_semantics") == "source_graph":
        source_degrees_np = _np.asarray(
            [11 + 2 * index for index in range(count)], dtype=_np.float32
        )
    else:
        source_degrees_np = _rel_expected_degree(source_graph_np, count)
    if fitting_roots.get("degrees"):
        source_degrees_np = source_degrees_np.astype(
            numpy_dtypes[TYPED_FIXTURES[
                fitting_roots["degrees"]
            ]["dtype"]],
            copy=False,
        )

    source_roots_np = {
        _rel.get("stable_entity_id_root"): source_ids_np,
    }
    if _rel.get("degree_root") is not None:
        source_roots_np[_rel.get("degree_root")] = source_degrees_np
    for root_index, (logical_root, typed_root) in enumerate(
        fitting_roots["coindexed"].items()
    ):
        spec = TYPED_FIXTURES[typed_root]
        shape = tuple(spec.get("shape") or ())
        axis = int((_rel.get("coindexed_roots") or {})[logical_root])
        if logical_root == _rel.get("stable_entity_id_root"):
            values = source_ids_np.copy()
        elif logical_root == _rel.get("degree_root"):
            values = source_degrees_np.copy()
        elif (
            _target_scaling_plan
            and logical_root
            == (_target_scaling_plan.get("policy") or {}).get("target_root")
            and typed_root
            == ((_target_scaling_plan.get("training_call") or {}).get(
                "roots"
            ) or {}).get("targets")
        ):
            values = _np.asarray(
                (_target_scaling_plan.get("validation_fixture") or {}).get(
                    "targets"
                )
            )
        else:
            size = int(_np.prod(shape))
            values = _np.arange(size).reshape(shape) + (root_index + 1) * 1000
            constraint = spec.get("constraint") or {}
            upper_bound = spec.get("constraint_upper_bound")
            if constraint.get("kind") in {"index", "class_id"} and upper_bound:
                values = values % int(upper_bound)
            if spec.get("dtype") == "bool":
                values = values % 2 == 0
            numpy_dtype = numpy_dtypes.get(spec.get("dtype"))
            if numpy_dtype is not None:
                values = _np.asarray(values, dtype=numpy_dtype)
        if len(shape) <= axis or shape[axis] != count:
            _rel_error(
                "real_training_call",
                "typed co-indexed fixture " + repr(typed_root)
                + " does not expose the declared source entity axis",
            )
            return None
        source_roots_np[logical_root] = values

    source_ids = _rel_execution_array(
        source_ids_np,
        TYPED_FIXTURES[fitting_roots["source_entity_ids"]],
        fitting_roots["source_entity_ids"],
    )
    source_graph = _rel_execution_array(
        source_graph_np, graph_spec, fitting_roots["graph"]
    )
    source_degrees = (
        _rel_execution_array(
            source_degrees_np,
            TYPED_FIXTURES[fitting_roots["degrees"]],
            fitting_roots["degrees"],
        )
        if fitting_roots.get("degrees") else None
    )
    source_roots = {}
    for logical_root, values in source_roots_np.items():
        if logical_root == _rel.get("stable_entity_id_root"):
            source_roots[logical_root] = source_ids
        elif logical_root == _rel.get("degree_root"):
            source_roots[logical_root] = source_degrees
        else:
            typed_root = fitting_roots["coindexed"][logical_root]
            source_roots[logical_root] = _rel_execution_array(
                values, TYPED_FIXTURES[typed_root], typed_root
            )
    return {
        "source_ids_np": source_ids_np,
        "source_graph_np": source_graph_np,
        "source_degrees_np": source_degrees_np,
        "source_roots_np": source_roots_np,
        "source_ids": source_ids,
        "source_graph": source_graph,
        "source_degrees": source_degrees,
        "source_roots": source_roots,
    }


def _rel_expected_real_graph(source_graph, positions, source_count):
    if _rel.get("representation") == "dense_adjacency":
        indices = _np.asarray(positions, dtype=_np.int64)
        return source_graph[_np.ix_(indices, indices)]
    mapping = _np.full((source_count,), -1, dtype=_np.int64)
    mapping[_np.asarray(positions, dtype=_np.int64)] = _np.arange(
        len(positions), dtype=_np.int64
    )
    source = source_graph[0]
    destination = source_graph[1]
    keep = (mapping[source] >= 0) & (mapping[destination] >= 0)
    return _np.stack(
        (mapping[source[keep]], mapping[destination[keep]])
    ).astype(source_graph.dtype, copy=False)


def _rel_validate_real_result(case, result, fixture, positions, plan):
    if not isinstance(result, dict):
        _rel_error(
            case,
            "preparation callable must return a dict during the real call; got "
            + type(result).__name__,
        )
        return False
    required = {
        "local_to_source", "source_to_local", "coindexed", "graph",
        "degrees", "output_entity_ids",
    }
    missing = sorted(required - set(result))
    if missing:
        _rel_error(
            case,
            "real preparation result is missing required key(s) " + repr(missing),
        )
        return False

    expected_positions = list(positions)
    if _rel.get("output_order") == "canonical_source_order":
        expected_positions = sorted(expected_positions)
    source_count = int(len(fixture["source_ids_np"]))
    expected_local = _np.asarray(expected_positions, dtype=_np.int64)
    expected_source = _np.full((source_count,), -1, dtype=_np.int64)
    expected_source[expected_local] = _np.arange(
        len(expected_local), dtype=_np.int64
    )
    expected_ids = fixture["source_ids_np"][expected_local]
    _rel_check_array(
        case, "local_to_source", result["local_to_source"],
        expected_local, integral=True,
    )
    _rel_check_array(
        case, "source_to_local", result["source_to_local"],
        expected_source, integral=True,
    )
    _rel_check_array(
        case, "output_entity_ids", result["output_entity_ids"],
        expected_ids, integral=True,
    )
    _rel_check_real_typed_convention(
        case,
        "output_entity_ids",
        result["output_entity_ids"],
        plan["fitting_roots"]["source_entity_ids"],
    )

    expected_graph = _rel_expected_real_graph(
        fixture["source_graph_np"], expected_positions, source_count
    )
    if _rel.get("representation") == "sparse_edge_index":
        _rel_check_sparse_graph(
            case, result["graph"], expected_graph, len(expected_positions)
        )
    else:
        _rel_check_array(
            case,
            "graph " + repr(_rel.get("graph_root")) + " with "
            + repr(_rel.get("edge_orientation")) + " orientation",
            result["graph"], expected_graph,
        )
    _rel_check_real_typed_convention(
        case,
        "prepared graph",
        result["graph"],
        plan["inference_roots"]["graph"],
    )

    returned_roots = result.get("coindexed")
    if not isinstance(returned_roots, dict):
        _rel_error(case, "real preparation coindexed value must be a dict")
        return False
    expected_root_names = set(fixture["source_roots_np"])
    if set(returned_roots) != expected_root_names:
        _rel_error(
            case,
            "real preparation coindexed roots differ from the explicit fitting "
            "binding; expected=" + repr(sorted(expected_root_names))
            + ", got=" + repr(sorted(returned_roots)),
        )
    for logical_root, source_value in fixture["source_roots_np"].items():
        if logical_root not in returned_roots:
            continue
        axis = int((_rel.get("coindexed_roots") or {})[logical_root])
        if (
            logical_root == _rel.get("degree_root")
            and _rel.get("degree_semantics") == "induced_graph"
        ):
            expected = _rel_expected_degree(
                expected_graph, len(expected_positions)
            ).astype(source_value.dtype, copy=False)
        else:
            expected = _rel_take(source_value, expected_positions, axis)
        _rel_check_array(
            case,
            "coindexed root " + repr(logical_root),
            returned_roots[logical_root],
            expected,
            integral=expected.dtype.kind in "iu",
        )
        if logical_root == _rel.get("stable_entity_id_root"):
            typed_root = plan["fitting_roots"]["source_entity_ids"]
        elif logical_root == _rel.get("degree_root"):
            typed_root = plan["fitting_roots"]["degrees"]
        else:
            typed_root = (
                plan["inference_roots"]["coindexed"].get(logical_root)
                or plan["fitting_roots"]["coindexed"][logical_root]
            )
        _rel_check_real_typed_convention(
            case,
            "coindexed root " + repr(logical_root),
            returned_roots[logical_root],
            typed_root,
        )

    if _rel.get("degree_root") is None:
        if result.get("degrees") is not None:
            _rel_error(case, "degrees must be None when degree_root is null")
    else:
        if _rel.get("degree_semantics") == "source_graph":
            expected_degrees = fixture["source_degrees_np"][expected_local]
        else:
            expected_degrees = _rel_expected_degree(
                expected_graph, len(expected_positions)
            ).astype(fixture["source_degrees_np"].dtype, copy=False)
        _rel_check_array(
            case,
            "degree root " + repr(_rel.get("degree_root")) + " under "
            + repr(_rel.get("degree_semantics")) + " semantics",
            result["degrees"],
            expected_degrees,
        )
        _rel_check_real_typed_convention(
            case,
            "prepared degrees",
            result["degrees"],
            plan["inference_roots"]["degrees"],
        )
    return True


def _rel_bound_call(callable_obj, args, kwargs, label):
    try:
        signature = inspect.signature(callable_obj)
        return dict(signature.bind(*args, **kwargs).arguments)
    except Exception as exc:
        _errors.append(
            label + ": cannot bind captured call ("
            + type(exc).__name__ + ": " + str(exc) + ")"
        )
        return {}


def _ts_error(case, message):
    _errors.append(
        "target_scaling_execution case=" + case + ": " + message
    )


def _th_error(case, message):
    _errors.append(
        "training_history_execution case=" + case + ": " + message
    )


def _ts_values_equal(left, right):
    if (
        (_HAS_TORCH and isinstance(left, torch.Tensor))
        or (_HAS_NUMPY and isinstance(left, _np.ndarray))
        or (_HAS_TORCH and isinstance(right, torch.Tensor))
        or (_HAS_NUMPY and isinstance(right, _np.ndarray))
    ):
        def as_numpy(value):
            if _HAS_TORCH and isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            if _HAS_NUMPY and isinstance(value, _np.ndarray):
                return value
            return None
        left_np = as_numpy(left)
        right_np = as_numpy(right)
        return (
            left_np is not None
            and right_np is not None
            and left_np.shape == right_np.shape
            and _np.array_equal(left_np, right_np)
        )
    return left == right


def _ts_runtime_fixture():
    plan = _target_scaling_plan or {}
    training = plan.get("training_call") or {}
    roots = training.get("roots") or {}
    validation = plan.get("validation_fixture") or {}
    ids_root = roots.get("entity_ids")
    targets_root = roots.get("targets")
    if (
        validation.get("scope") != "schema2_runtime_validator_only"
        or ids_root not in TYPED_FIXTURES
        or targets_root not in TYPED_FIXTURES
        or not isinstance(validation.get("entity_ids"), list)
        or not isinstance(validation.get("targets"), list)
        or not isinstance(validation.get("fitting_range"), dict)
        or not isinstance(validation.get("state"), dict)
    ):
        _ts_error(
            "fixture",
            "frozen validator-only identity, targets, range, or state is malformed",
        )
        return None
    ids = _rel_execution_array(
        validation["entity_ids"], TYPED_FIXTURES[ids_root], ids_root
    )
    targets = _rel_execution_array(
        validation["targets"], TYPED_FIXTURES[targets_root], targets_root
    )
    if ids is None or targets is None:
        _ts_error("fixture", "typed scaling inputs could not be materialized")
        return None
    _check_typed_value(ids, TYPED_FIXTURES[ids_root], ids_root)
    _check_typed_value(targets, TYPED_FIXTURES[targets_root], targets_root)
    return {
        "entity_ids": ids,
        "targets": targets,
        "fitting_range": dict(validation["fitting_range"]),
        "state": dict(validation["state"]),
    }


def _th_runtime_fixture():
    plan = _training_history_plan or {}
    training = plan.get("training_call") or {}
    roots = training.get("roots") or {}
    validation = plan.get("validation_fixture") or {}
    inputs = validation.get("inputs")
    record = validation.get("record")
    seed_root = roots.get("seed")
    if (
        validation.get("scope") != "schema2_fixed_helper_oracle_only"
        or not isinstance(inputs, dict)
        or not isinstance(record, dict)
        or seed_root not in TYPED_FIXTURES
    ):
        _th_error(
            "fixture",
            "frozen validator-only history inputs or record are malformed",
        )
        return None
    seed_spec = TYPED_FIXTURES[seed_root]
    ok, seed_value = _synth_typed(seed_spec, seed_root)
    if not ok or int(seed_value) != int(inputs.get("seed")):
        _th_error(
            "fixture",
            "frozen history seed disagrees with its exact typed scalar root",
        )
        return None
    materialized = json.loads(json.dumps(inputs))
    # The fixed record schema requires a JSON integer. A Python int still
    # satisfies the schema-2 scalar descriptor checked by _build_typed_call.
    materialized["seed"] = int(inputs["seed"])
    # The frozen performed record is a pure helper oracle. Schema 2 currently
    # has no typed selection-target carrier from which the real fitting call
    # could compute held-out metric observations, so Stage 2.d exercises the
    # honest no-selection arm and leaves performed selection to live lineage
    # acceptance.
    materialized["selection_range"] = None
    return {
        "inputs": materialized,
    }


def _ts_patch_training_reference(original, replacement):
    patched = []
    for module in (
        _target_scaling_mod,
        _training_history_mod,
        _training_mod,
    ):
        if module is None:
            continue
        for name, value in list(vars(module).items()):
            if value is original:
                setattr(module, name, replacement)
                patched.append((module, name, original))
    return patched


def _ts_restore_references(patched):
    for module, name, original in reversed(patched):
        setattr(module, name, original)


def _exercise_target_scaling_training_call(
    training_fn,
    training_sig,
    *,
    value_overrides,
    required_transform=None,
):
    """Execute one declared fitting call while observing the fixed helper."""
    if not _target_scaling_plan:
        return False, True
    helper = _target_scaling_plan.get("helper") or {}
    training = _target_scaling_plan.get("training_call") or {}
    parameters = training.get("parameters") or {}
    fixture = _ts_runtime_fixture()
    if fixture is None:
        return True, False
    history_fixture = None
    if _training_history_plan:
        history_fixture = _th_runtime_fixture()
        if history_fixture is None:
            return True, False
    if _target_scaling_mod is None:
        _ts_error(
            "real_training_call",
            "fixed method.target_scaling helper module is not importable ("
            + type(_target_scaling_import_error).__name__ + ": "
            + str(_target_scaling_import_error) + ")",
        )
        return True, False
    fit_name = helper.get("fit_callable")
    transform_name = helper.get("transform_callable")
    original_fit = getattr(_target_scaling_mod, fit_name, None)
    original_transform = getattr(_target_scaling_mod, transform_name, None)
    if not callable(original_fit) or not callable(original_transform):
        _ts_error(
            "real_training_call",
            "fixed target-scaling fit or transform callable is absent",
        )
        return True, False
    original_history = None
    if _training_history_plan:
        if _training_history_mod is None:
            _th_error(
                "real_training_call",
                "fixed method.training_history helper module is not importable ("
                + type(_training_history_import_error).__name__ + ": "
                + str(_training_history_import_error) + ")",
            )
            return True, False
        history_helper = _training_history_plan.get("helper") or {}
        original_history = getattr(
            _training_history_mod,
            history_helper.get("record_callable"),
            None,
        )
        if not callable(original_history):
            _th_error(
                "real_training_call",
                "fixed record_training_history callable is absent",
            )
            return True, False

    overrides = dict(value_overrides or {})
    exact_overrides = {
        parameters.get("entity_ids"): fixture["entity_ids"],
        parameters.get("targets"): fixture["targets"],
        parameters.get("fitting_range"): fixture["fitting_range"],
    }
    if history_fixture is not None:
        history_parameters = (
            _training_history_plan.get("training_call") or {}
        ).get("parameters") or {}
        exact_overrides.update({
            history_parameters.get("fitting_range"): history_fixture[
                "inputs"
            ]["fitting_range"],
            history_parameters.get("selection_range"): history_fixture[
                "inputs"
            ]["selection_range"],
            history_parameters.get("seed"): history_fixture["inputs"]["seed"],
            history_parameters.get("config_id"): history_fixture[
                "inputs"
            ]["config_id"],
        })
    if any(not isinstance(name, str) or not name for name in exact_overrides):
        _ts_error(
            "real_training_call", "frozen training parameter crosswalk is malformed"
        )
        return True, False
    for name, value in exact_overrides.items():
        if name in overrides and not _ts_values_equal(overrides[name], value):
            _ts_error(
                "real_training_call",
                "fixed forecasting input " + repr(name)
                + " disagrees with the relational/source fixture",
            )
            return True, False
        overrides[name] = value

    fit_calls = []
    transform_calls = []
    history_calls = []

    def capture_fit(*args, **kwargs):
        bound = _rel_bound_call(
            original_fit, args, kwargs,
            "target_scaling_execution fit helper",
        )
        result = original_fit(*args, **kwargs)
        fit_calls.append((bound, result))
        return result

    def capture_transform(*args, **kwargs):
        bound = _rel_bound_call(
            original_transform, args, kwargs,
            "target_scaling_execution transform helper",
        )
        result = original_transform(*args, **kwargs)
        transform_calls.append((bound, result))
        return result

    def capture_history(*args, **kwargs):
        bound = _rel_bound_call(
            original_history, args, kwargs,
            "training_history_execution record helper",
        )
        result = original_history(*args, **kwargs)
        history_calls.append((bound, result))
        return result

    patched = []
    try:
        patched.extend(_ts_patch_training_reference(original_fit, capture_fit))
        patched.extend(
            _ts_patch_training_reference(original_transform, capture_transform)
        )
        if original_history is not None:
            patched.extend(
                _ts_patch_training_reference(original_history, capture_history)
            )
        ok, offending, args, kwargs = _build_typed_call(
            training_sig.get("input") or {},
            training_fn,
            "training_loop",
            value_overrides=overrides,
        )
        if not ok:
            _ts_error(
                "real_training_call",
                "declared fitting input " + repr(offending)
                + " could not be built from the target-scaling crosswalk",
            )
            return True, False
        try:
            training_output = training_fn(*args, **kwargs)
        except Exception as exc:
            _ts_error(
                "real_training_call",
                "declared fitting entry raised " + type(exc).__name__ + ": "
                + str(exc),
            )
            return True, False
        _check_output(training_output, training_sig, "training_loop")
    finally:
        _ts_restore_references(patched)

    exact_fit_results = []
    for bound, result in fit_calls:
        if (
            _ts_values_equal(bound.get("entity_ids"), fixture["entity_ids"])
            and _ts_values_equal(bound.get("targets"), fixture["targets"])
            and bound.get("fitting_range") == fixture["fitting_range"]
        ):
            exact_fit_results.append(result)
    if not fit_calls or len(exact_fit_results) != len(fit_calls):
        _ts_error(
            "real_training_call",
            "every fixed fit_target_scaling_state call must receive the exact "
            "typed stable ids, targets, and validator-only fitting range",
        )
        return True, False
    if any(result != fixture["state"] for result in exact_fit_results):
        _ts_error(
            "real_training_call",
            "fixed fitting helper returned state that disagrees with the "
            "frozen closed-policy oracle",
        )
        return True, False

    def coherent_transform(bound):
        expected = required_transform or {
            "entity_ids": fixture["entity_ids"],
            "targets": fixture["targets"],
        }
        return (
            _ts_values_equal(
                bound.get("entity_ids"), expected.get("entity_ids")
            )
            and _ts_values_equal(bound.get("values"), expected.get("targets"))
            and bound.get("state") == fixture["state"]
            and bound.get("entity_axis", 0) == 0
        )

    exact_transforms = [
        result for bound, result in transform_calls if coherent_transform(bound)
    ]
    if not transform_calls or len(exact_transforms) != len(transform_calls):
        _ts_error(
            "real_training_call",
            "every fixed transform_targets call must consume the exact "
            "validated fitting targets and stable ids in their declared "
            "output order, plus fitted state and declared entity axis",
        )
        return True, False
    state_result = training.get("state_result") or {}
    state_key = state_result.get("key")
    if (
        not isinstance(training_output, _RuntimeMapping)
        or state_key not in training_output
        or training_output[state_key] != fixture["state"]
    ):
        _ts_error(
            "real_training_call",
            "declared fitting result does not expose the exact validated state "
            "under mapping key " + repr(state_key),
        )
        return True, False
    if history_fixture is not None:
        if not isinstance(training_output, _RuntimeMapping) \
                or "model" not in training_output:
            _th_error(
                "real_training_call",
                "declared fitting result must retain the trained model beside "
                "target_scaling_state and training_history",
            )
            return True, False
        expected_inputs = history_fixture["inputs"]
        if len(history_calls) != 1:
            _th_error(
                "real_training_call",
                "declared fitting must reach fixed record_training_history "
                "exactly once",
            )
            return True, False
        history_bound, returned_history = history_calls[0]
        independent_inputs = {
            "model_id": expected_inputs["fitting_range"]["model_id"],
            "seed": expected_inputs["seed"],
            "config_id": expected_inputs["config_id"],
            "target_scaling_state": expected_inputs["target_scaling_state"],
            "fitting_range": expected_inputs["fitting_range"],
        }
        if set(history_bound) != set(expected_inputs) or any(
            not _ts_values_equal(history_bound.get(name), expected)
            for name, expected in independent_inputs.items()
        ):
            _th_error(
                "real_training_call",
                "fixed record_training_history must receive the exact "
                "pipeline-owned model identity, seed, config, scaling state, "
                "and fitting range",
            )
            return True, False
        history_validator = getattr(
            _training_history_mod, "validate_training_history", None
        )
        if not callable(history_validator):
            _th_error(
                "real_training_call",
                "fixed training-history canonical validator is absent",
            )
            return True, False
        try:
            canonical_history = history_validator(returned_history)
        except Exception as exc:
            _th_error(
                "real_training_call",
                "fixed recorder returned a noncanonical history ("
                + type(exc).__name__ + ": " + str(exc) + ")",
            )
            return True, False
        if canonical_history != returned_history:
            _th_error(
                "real_training_call",
                "fixed recorder return changed under canonical validation",
            )
            return True, False
        selection = returned_history.get("selection") or {}
        selection_status = selection.get("status")
        if selection_status == "performed":
            if history_bound.get("selection_range") != expected_inputs[
                "selection_range"
            ]:
                _th_error(
                    "real_training_call",
                    "performed selection must retain the exact pipeline-owned "
                    "selection range",
                )
                return True, False
        elif selection_status == "not_performed":
            if history_bound.get("selection_range") is not None:
                _th_error(
                    "real_training_call",
                    "not_performed selection must pass selection_range=None",
                )
                return True, False
        else:
            _th_error(
                "real_training_call",
                "fixed recorder returned an unsupported selection status",
            )
            return True, False
        history_result = (
            _training_history_plan.get("training_call") or {}
        ).get("history_result") or {}
        history_key = history_result.get("key")
        if (
            not isinstance(training_output, _RuntimeMapping)
            or history_key not in training_output
            or training_output[history_key] != returned_history
        ):
            _th_error(
                "real_training_call",
                "declared fitting result does not expose the exact validated "
                "record under mapping key " + repr(history_key),
            )
            return True, False
    return True, True


def _rel_model_consumes_prepared(call, prepared, plan):
    parameters = plan["inference_parameters"]
    graph_name = parameters["graph"]
    if graph_name not in call or call[graph_name] is not prepared["graph"]:
        return False
    degree_name = parameters.get("degrees")
    if degree_name and (
        degree_name not in call
        or call[degree_name] is not prepared["degrees"]
    ):
        return False
    returned_roots = prepared.get("coindexed") or {}
    for logical_root, parameter_name in plan["inference_coindexed"].items():
        if (
            logical_root not in returned_roots
            or parameter_name not in call
            or call[parameter_name] is not returned_roots[logical_root]
        ):
            return False
    return True


def _exercise_relational_training_call(training_fn, training_sig):
    plan = RELATIONAL_EXECUTION_PLAN or {}
    if not plan:
        return False
    model = _constructed_architecture.get(plan.get("architecture_block"))
    if model is None:
        _rel_error(
            "real_training_call",
            "declared architecture block was not constructed; fitting was not called",
        )
        return True
    forward = getattr(model, "forward", None)
    if not callable(forward):
        _unsupported.append(
            "relational_indexing.execution.inference.architecture_block|"
            "the current real-call validator can instrument an exact forward "
            "method, but the constructed model exposes only a different callable grammar"
        )
        return True
    fixture = _rel_real_source_fixture(plan)
    if fixture is None:
        return True

    fitting_positions = _rel_real_positions(plan, "fitting")
    batch_ids_np = fixture["source_ids_np"][
        _np.asarray(fitting_positions, dtype=_np.int64)
    ]
    batch_ids_root = plan["fitting_roots"]["batch_entity_ids"]
    batch_ids = _rel_execution_array(
        batch_ids_np, TYPED_FIXTURES[batch_ids_root], batch_ids_root
    )
    parameters = plan["fitting_parameters"]
    overrides = {
        parameters["model"]: model,
        parameters["source_entity_ids"]: fixture["source_ids"],
        parameters["batch_entity_ids"]: batch_ids,
        parameters["graph"]: fixture["source_graph"],
    }
    if parameters.get("degrees"):
        overrides[parameters["degrees"]] = fixture["source_degrees"]
    for logical_root, parameter_name in plan["fitting_coindexed"].items():
        overrides[parameter_name] = fixture["source_roots"][logical_root]
    if plan.get("one_epoch_parameter"):
        epoch_spec = TYPED_FIXTURES[plan["one_epoch_root"]]
        epoch_value = (
            _np.int32(1)
            if epoch_spec.get("dtype") == "int32" else _np.int64(1)
        )
        overrides[plan["one_epoch_parameter"]] = epoch_value

    preparation_calls = []
    model_calls = []
    original_preparation = _prep_fn
    original_forward = forward

    def capture_preparation(*args, **kwargs):
        bound = _rel_bound_call(
            original_preparation, args, kwargs,
            "relational_indexing real fitting preparation",
        )
        result = original_preparation(*args, **kwargs)
        preparation_calls.append((bound, result))
        return result

    def capture_forward(*args, **kwargs):
        model_calls.append(
            _rel_bound_call(
                original_forward, args, kwargs,
                "relational_indexing real fitting model call",
            )
        )
        return original_forward(*args, **kwargs)

    try:
        setattr(model, "forward", capture_forward)
    except Exception as exc:
        _unsupported.append(
            "relational_indexing.execution.inference.architecture_block|"
            "the current real-call validator cannot instrument the declared "
            "forward method (" + type(exc).__name__ + ": " + str(exc) + ")"
        )
        return True
    try:
        setattr(_training_mod, _prep_name, capture_preparation)
        expected_transform_positions = list(fitting_positions)
        if _rel.get("output_order") == "canonical_source_order":
            expected_transform_positions = sorted(expected_transform_positions)
        scaling_target_root = (
            ((_target_scaling_plan or {}).get("policy") or {}).get(
                "target_root"
            )
        )
        required_transform = None
        if isinstance(scaling_target_root, str):
            source_target = fixture["source_roots_np"].get(
                scaling_target_root
            )
            target_axis = (_rel.get("coindexed_roots") or {}).get(
                scaling_target_root
            )
            if source_target is None or not isinstance(target_axis, int):
                _ts_error(
                    "real_training_call",
                    "the target-scaling root is absent from the exact relational "
                    "fitting fixture",
                )
                return True
            required_transform = {
                "entity_ids": fixture["source_ids_np"][_np.asarray(
                    expected_transform_positions, dtype=_np.int64
                )],
                "targets": _np.take(
                    source_target,
                    _np.asarray(expected_transform_positions, dtype=_np.int64),
                    axis=target_axis,
                ),
            }
        scaling_handled, scaling_ok = _exercise_target_scaling_training_call(
            training_fn,
            training_sig,
            value_overrides=overrides,
            required_transform=required_transform,
        )
        if scaling_handled:
            if not scaling_ok:
                return True
        else:
            ok, offending, args, kwargs = _build_typed_call(
                training_sig.get("input") or {},
                training_fn,
                "training_loop",
                value_overrides=overrides,
            )
            if not ok:
                _rel_error(
                    "real_training_call",
                    "declared fitting input " + repr(offending)
                    + " could not be built from the typed relational crosswalk",
                )
                return True
            try:
                training_output = training_fn(*args, **kwargs)
            except Exception as exc:
                _rel_error(
                    "real_training_call",
                    "declared fitting entry raised " + type(exc).__name__ + ": "
                    + str(exc),
                )
                return True
            _check_output(training_output, training_sig, "training_loop")
    finally:
        setattr(_training_mod, _prep_name, original_preparation)
        setattr(model, "forward", original_forward)

    exact_preparations = []
    expected_preparation = {
        "source_entity_ids": fixture["source_ids"],
        "batch_entity_ids": batch_ids,
        "graph": fixture["source_graph"],
        "degrees": fixture["source_degrees"],
        "coindexed": fixture["source_roots"],
    }
    for bound, result in preparation_calls:
        exact = True
        for name in ("source_entity_ids", "batch_entity_ids", "graph"):
            if name not in bound or not _rel_values_equal(
                bound[name], expected_preparation[name]
            ):
                exact = False
        if _rel.get("degree_root") is not None and (
            "degrees" not in bound
            or not _rel_values_equal(
                bound["degrees"], expected_preparation["degrees"]
            )
        ):
            exact = False
        received_roots = bound.get("coindexed")
        if not isinstance(received_roots, dict) or set(received_roots) != set(
            expected_preparation["coindexed"]
        ):
            exact = False
        elif any(
            not _rel_values_equal(
                received_roots[logical_root],
                expected_preparation["coindexed"][logical_root],
            )
            for logical_root in received_roots
        ):
            exact = False
        if exact:
            exact_preparations.append(result)

    if not preparation_calls:
        _rel_error(
            "real_training_call",
            "declared fitting entry did not execute the preparation callable "
            "with the exact source graph, stable entity ids, requested batch "
            "ids, degrees, and co-indexed roots",
        )
        return True
    if len(exact_preparations) != len(preparation_calls):
        _rel_error(
            "real_training_call",
            "every preparation call made by the declared fitting entry must "
            "use the exact source graph, stable ids, requested batch ids, "
            "degrees, and co-indexed roots; exact_calls="
            + str(len(exact_preparations)) + ", total_calls="
            + str(len(preparation_calls)),
        )
        return True
    validated_preparations = []
    for prepared in exact_preparations:
        if _rel_validate_real_result(
            "real_fitting_" + str(plan.get("fitting_mode")),
            prepared,
            fixture,
            fitting_positions,
            plan,
        ):
            validated_preparations.append(prepared)
    if len(validated_preparations) != len(exact_preparations):
        return True

    every_model_call_is_prepared = bool(model_calls) and all(
        any(
            _rel_model_consumes_prepared(call, prepared, plan)
            for prepared in validated_preparations
        )
        for call in model_calls
    )
    every_preparation_is_consumed = all(
        any(
            _rel_model_consumes_prepared(call, prepared, plan)
            for call in model_calls
        )
        for prepared in validated_preparations
    )
    if not every_model_call_is_prepared or not every_preparation_is_consumed:
        graph_parameter = plan["inference_parameters"]["graph"]
        observed_graphs = [
            _rel_numpy(call.get(graph_parameter))
            for call in model_calls
            if call.get(graph_parameter) is not None
        ]
        batch_size = len(fitting_positions)
        escaped = next(
            (
                graph
                for graph in observed_graphs
                if graph is not None
                and _rel.get("representation") == "sparse_edge_index"
                and graph.size
                and (graph.min() < 0 or graph.max() >= batch_size)
            ),
            None,
        )
        if escaped is not None:
            _rel_error(
                "real_training_call",
                "fitting passed sparse graph endpoints outside "
                "local_batch_positions [0, " + str(batch_size)
                + "); observed min=" + str(escaped.min())
                + ", max=" + str(escaped.max())
                + "; requested local_to_source="
                + repr(list(fitting_positions)),
            )
        else:
            expected_local = list(fitting_positions)
            if _rel.get("output_order") == "canonical_source_order":
                expected_local = sorted(expected_local)
            expected_ids = fixture["source_ids_np"][
                _np.asarray(expected_local, dtype=_np.int64)
            ].tolist()
            _rel_error(
                "real_training_call",
                "the fitting model call did not consume the prepared graph, "
                "degree vector, and co-indexed roots named by the inference "
                "crosswalk; source and local entity domains disagree for "
                "local_to_source=" + repr(expected_local)
                + " and output_entity_ids=" + repr(expected_ids),
            )

    inference_positions = _rel_real_positions(plan, "inference")
    inference_ids_np = fixture["source_ids_np"][
        _np.asarray(inference_positions, dtype=_np.int64)
    ]
    inference_ids = _rel_execution_array(
        inference_ids_np, TYPED_FIXTURES[batch_ids_root], batch_ids_root
    )
    try:
        inference_prepared = original_preparation(
            source_entity_ids=fixture["source_ids"],
            batch_entity_ids=inference_ids,
            coindexed=fixture["source_roots"],
            graph=fixture["source_graph"],
            degrees=fixture["source_degrees"],
        )
    except Exception as exc:
        _rel_error(
            "real_inference_call",
            "declared preparation callable raised "
            + type(exc).__name__ + ": " + str(exc),
        )
        return True
    if not _rel_validate_real_result(
        "real_inference_" + str(plan.get("inference_mode")),
        inference_prepared,
        fixture,
        inference_positions,
        plan,
    ):
        return True

    inference_overrides = {
        plan["inference_parameters"]["graph"]: inference_prepared["graph"],
    }
    if plan["inference_parameters"].get("degrees"):
        inference_overrides[
            plan["inference_parameters"]["degrees"]
        ] = inference_prepared["degrees"]
    for logical_root, parameter_name in plan["inference_coindexed"].items():
        inference_overrides[parameter_name] = (
            inference_prepared["coindexed"][logical_root]
        )
    architecture_block = (
        (CONTRACT.get("architecture") or {})[plan["architecture_block"]]
    )
    ok, offending, args, kwargs = _build_typed_call(
        (architecture_block.get("forward") or {}).get("input") or {},
        model,
        "architecture." + plan["architecture_block"] + ".forward",
        value_overrides=inference_overrides,
    )
    if not ok:
        _rel_error(
            "real_inference_call",
            "declared inference input " + repr(offending)
            + " could not be built from the typed relational crosswalk",
        )
        return True
    prior_training_mode = getattr(model, "training", None)
    eval_method = getattr(model, "eval", None)
    train_method = getattr(model, "train", None)
    try:
        if callable(eval_method):
            eval_method()
        inference_context = (
            torch.no_grad()
            if _HAS_TORCH else contextlib.nullcontext()
        )
        with inference_context:
            inference_output = model(*args, **kwargs)
    except Exception as exc:
        _rel_error(
            "real_inference_call",
            "declared model inference raised " + type(exc).__name__ + ": "
            + str(exc),
        )
        return True
    finally:
        if isinstance(prior_training_mode, bool) and callable(train_method):
            train_method(prior_training_mode)
    _check_output(
        inference_output,
        architecture_block.get("forward") or {},
        "architecture." + plan["architecture_block"] + ".forward",
    )
    return True


if _IS_V2:
    # Version 2 gives the pluggable and training seams the same typed callable
    # contract as architecture methods.  Resolve public functions without
    # guessing positional values; an opaque input intentionally skips the
    # call, while every supported typed input reaches the declared callable.
    _pluggable = CONTRACT.get("pluggable_component") or {}
    _pluggable_name = _pluggable.get("name")
    _pluggable_fn = (
        getattr(method, _pluggable_name, None)
        if method is not None and isinstance(_pluggable_name, str) else None
    )
    if _pluggable_fn is None and _training_mod is not None and isinstance(
        _pluggable_name, str
    ):
        _pluggable_fn = getattr(_training_mod, _pluggable_name, None)
    if _pluggable_fn is None:
        if not (
            _is_environment_gap(_method_import_error)
            or _is_environment_gap(_training_import_error)
        ):
            _errors.append(
                "pluggable_component.name=" + repr(_pluggable_name)
                + ": declared callable is not importable from method or "
                "method.training"
            )
    else:
        _exercise_method(
            None, _pluggable_fn, _pluggable, "pluggable_component"
        )

    _training = CONTRACT.get("training_loop")
    if isinstance(_training, dict):
        _training_name = _training.get("function_name")
        _training_fn = (
            getattr(_training_mod, _training_name, None)
            if _training_mod is not None and isinstance(_training_name, str)
            else None
        )
        if _training_fn is None and isinstance(_training_name, str):
            _training_fn = (
                getattr(method, _training_name, None)
                if method is not None else None
            )
        if _training_fn is None:
            if not (
                _is_environment_gap(_training_import_error)
                or _is_environment_gap(_method_import_error)
            ):
                _errors.append(
                    "training_loop.function_name=" + repr(_training_name)
                    + ": declared callable is not importable from method.training "
                    "or method"
                )
        else:
            if not _exercise_relational_training_call(
                _training_fn, _training
            ):
                _scaling_parameters = (
                    (_target_scaling_plan or {}).get("training_call") or {}
                ).get("parameters") or {}
                _scaling_block = (
                    (FORECASTING_EXECUTION_PLAN or {}).get("construction") or {}
                ).get("architecture_block")
                _scaling_model = _constructed_architecture.get(_scaling_block)
                _scaling_overrides = {}
                if _target_scaling_plan:
                    _model_parameter = _scaling_parameters.get("model")
                    if not isinstance(_model_parameter, str) or _scaling_model is None:
                        _ts_error(
                            "real_training_call",
                            "declared forecasting model was not constructed for "
                            "the target-scaling fitting entry",
                        )
                    else:
                        _scaling_overrides[_model_parameter] = _scaling_model
                _scaling_handled, _ = _exercise_target_scaling_training_call(
                    _training_fn,
                    _training,
                    value_overrides=_scaling_overrides,
                )
                if not _scaling_handled:
                    _exercise_method(
                        None, _training_fn, _training, "training_loop"
                    )


if CAPABILITY_REPORT_PATH:
    try:
        with open(CAPABILITY_REPORT_PATH, "w", encoding="utf-8") as _report_file:
            json.dump(sorted(set(_unsupported)), _report_file)
    except Exception as _report_error:
        _errors.append(
            "Stage 2.d could not write its trusted capability report: "
            + type(_report_error).__name__ + ": " + str(_report_error)
        )

for _s in _skips:
    print("SKIP: " + _s, file=sys.stderr)
if _errors or _unsupported:
    for _err in _errors:
        print(_err, file=sys.stderr)
    sys.exit(1)
print("arch contract runtime: ok")
sys.exit(0)
'''


def _build_dry_run_script(
    contract: Any,
    bindings: dict[str, int],
    builder_names: set[str],
    constructor_args: dict[str, dict[str, Any]],
    typed_fixtures: dict[str, dict[str, Any]] | None = None,
    relational_execution_plan: dict[str, Any] | None = None,
    forecasting_execution_plan: dict[str, Any] | None = None,
    capability_report_path: Path | None = None,
) -> str:
    """Construct a Python script that, run via subprocess from the run dir,
    exercises every architecture block the contract declares — paradigm-
    agnostically.

    The script embeds the contract, bindings, resolved constructor arguments,
    and builder names as JSON literals and interprets them with a generic
    runner (rather than hardcoding a `build_*` + `forward(x:tensor)` shape).
    Current construction validates and passes exactly the declared keyword
    mapping; legacy construction retains the candidate adapter. Construction
    tries a builder if one exists, else direct class construction. Each
    declared method (the `forward` slot + every `additional_methods` entry) is
    called with synthesized inputs. Version 2 binds exact declared names to
    positional-only, positional-or-keyword, and keyword-only parameters;
    legacy contracts retain ordered positional calls whose input names needn't
    match. Inputs that can't be synthesized because they are explicitly opaque
    cause that method's dry-run to be SKIPPED (logged, not failed); a method
    that IS exercised and raises is a real error.

    Supervised-ML paradigms (AL/KD) keep full coverage: build_model/
    build_student exist (builder path), the primary method is `forward`/
    `__call__` (tensor input), and load_data exists. Non-ML paradigms
    (motion_planning) get: direct class construction, additional-method
    exercise where inputs are tensor-synthesizable, and graceful skips for
    opaque inputs / a primary method that isn't named `forward`."""
    # Map architecture block keys to builder names (build_<key>, or the sole
    # build_* only when the contract itself has one block). In a multi-block
    # contract an unmatched block must use its declared class: treating the
    # one discovered builder as global can silently construct the wrong role.
    # None → direct class construction.
    block_to_builder: dict[str, str | None] = {}
    for block_key in contract.architecture:
        candidate = f"build_{block_key}"
        if candidate in builder_names:
            block_to_builder[block_key] = candidate
        elif len(contract.architecture) == 1 and len(builder_names) == 1:
            block_to_builder[block_key] = next(iter(builder_names))
        else:
            block_to_builder[block_key] = None

    contract_payload = contract.model_dump(
        exclude_none=contract.schema_version == "2.0.0"
    )
    contract_json = json.dumps(contract_payload, default=str)
    bindings_json = json.dumps(bindings)
    block_builders_json = json.dumps(block_to_builder)
    constructor_args_json = json.dumps(constructor_args)
    typed_fixtures_json = json.dumps(typed_fixtures or {}, default=str)
    relational_execution_plan_json = json.dumps(
        relational_execution_plan or {}, default=str
    )
    forecasting_execution_plan_json = json.dumps(
        forecasting_execution_plan or {}, default=str
    )
    capability_report_json = json.dumps(
        str(capability_report_path) if capability_report_path is not None else ""
    )
    builder_kwarg_candidates = {
        "n_classes": bindings.get("n_classes", DEFAULT_BINDING),
        "num_classes": bindings.get("n_classes", DEFAULT_BINDING),
        "input_dim": bindings.get("n_features", DEFAULT_BINDING),
        "hidden_dim": bindings.get("hidden_dim", DEFAULT_BINDING),
        "n_features": bindings.get("n_features", DEFAULT_BINDING),
        "bev_channels": bindings.get("bev_channels", DEFAULT_BINDING),
        "bev_h": bindings.get("bev_h", DEFAULT_BINDING),
        "bev_w": bindings.get("bev_w", DEFAULT_BINDING),
        "num_queries": bindings.get("N_queries", DEFAULT_BINDING),
    }
    candidates_json = json.dumps(builder_kwarg_candidates)

    # Preamble injects the per-run JSON literals; the runner body is a fixed
    # paradigm-agnostic template (module constant `_DRY_RUN_RUNNER`). Keeping
    # the runner as a plain string (not an f-string) avoids brace-escaping
    # hazards in the embedded Python.
    preamble = (
        "import sys, json, re, inspect, contextlib\n"
        f"CONTRACT = json.loads({contract_json!r})\n"
        f"BINDINGS = json.loads({bindings_json!r})\n"
        f"BLOCK_BUILDERS = json.loads({block_builders_json!r})\n"
        f"CONSTRUCTOR_ARGS = json.loads({constructor_args_json!r})\n"
        f"TYPED_FIXTURES = json.loads({typed_fixtures_json!r})\n"
        "RELATIONAL_EXECUTION_PLAN = "
        f"json.loads({relational_execution_plan_json!r})\n"
        "FORECASTING_EXECUTION_PLAN = "
        f"json.loads({forecasting_execution_plan_json!r})\n"
        f"CAPABILITY_REPORT_PATH = json.loads({capability_report_json!r})\n"
        f"BUILDER_KWARG_CANDIDATES = json.loads({candidates_json!r})\n"
        f"DEFAULT_BINDING = {DEFAULT_BINDING}\n"
    )
    return preamble + _DRY_RUN_RUNNER


def _graph_message_live_path_errors(
    spec: dict,
    raw_contract: dict,
    run_dir: Path,
    *,
    typed_contract: bool,
) -> list[str]:
    """Reject supported model-forward paths that bypass graph messaging.

    Stage 2.d already exercises the R2C-084 fitting and inference roots.  This
    companion static proof binds both exercised routes to the exact message
    helper and returned value.  Closed-grammar coverage gaps remain
    pipeline-owned and are recorded as unprobeable by the portable receipt.
    """

    from scripts.graph_callable_liveness import (  # noqa: PLC0415
        GraphCallableCoverageError,
        GraphCallableProducerError,
        graph_callable_identities,
        prove_message_live_path,
    )

    try:
        if graph_callable_identities(spec) is None:
            return []
        prove_message_live_path(spec, raw_contract, run_dir / "method")
    except GraphCallableProducerError as exc:
        message = (
            "homogeneous-graph message live-use disagrees with the exact "
            f"R2C-084 forward contract ({exc.code}): {exc}"
        )
        return [
            format_semantic_issue(_contract_code_issue(message))
            if typed_contract
            else message
        ]
    except GraphCallableCoverageError:
        return []
    return []


def validate(spec: dict, run_dir: Path) -> list[str]:
    errors: list[str] = []
    contract_path = run_dir / ".pipeline" / "arch_contract.json"
    if not contract_path.is_file():
        errors.append(
            f"arch_contract.json missing at {contract_path}. "
            "validate_arch_contract.py should have already halted this case; "
            "this runtime validator depends on the structured check passing first."
        )
        return errors

    try:
        raw = json.loads(contract_path.read_text(encoding="utf-8"))
        contract = load_arch_contract(raw)
    except Exception as e:
        errors.append(
            f"arch_contract.json could not be parsed/validated; cannot do "
            f"runtime dry-run: {e}"
        )
        return errors

    errors.extend(
        _graph_message_live_path_errors(
            spec,
            raw,
            run_dir,
            typed_contract=isinstance(contract, ArchContractV2),
        )
    )

    typed_fixtures: dict[str, dict[str, Any]] = {}
    relational_execution_plan: dict[str, Any] | None = None
    forecasting_execution_plan: dict[str, Any] | None = None
    if isinstance(contract, ArchContractV2):
        bundle_facts = _load_typed_bundle_dimension_facts(run_dir)
        build_plan: dict[str, Any] | None = None
        permitted_opaque_fitting_roots: frozenset[str] = frozenset()
        if contract.paradigm_id == "time_series_forecasting":
            build_plan = load_build_plan(
                spec,
                ROOT,
                provisional_packs_dir=run_overlay_dir(run_dir),
            )
            if build_plan is None:
                return [format_semantic_issue(_contract_code_issue(
                    "target_scaling_execution: the time-series forecasting "
                    "schema-2 contract has no served family build plan"
                ))]
            raw_execution = build_plan.get("target_scaling_execution")
            raw_training = (
                raw_execution.get("training_call")
                if isinstance(raw_execution, dict) else None
            )
            raw_range_root = (
                raw_training.get("fitting_range_input_root")
                if isinstance(raw_training, dict) else None
            )
            raw_history = build_plan.get("training_history_execution")
            raw_history_training = (
                raw_history.get("training_call")
                if isinstance(raw_history, dict) else None
            )
            permitted_opaque_fitting_roots = frozenset(
                root
                for root in (
                    raw_range_root,
                    raw_history_training.get("selection_range_input_root")
                    if isinstance(raw_history_training, dict) else None,
                    raw_history_training.get("config_id_input_root")
                    if isinstance(raw_history_training, dict) else None,
                )
                if isinstance(root, str)
            )
        resolution = resolve_contract(
            contract,
            bundle_facts,
        )
        if resolution.issues:
            # Semantic contradictions and validator coverage gaps are decided
            # before importing generated code.  This prevents an unresolved
            # dimension or unsupported fixture from becoming a plausible
            # default value or a misleading model exception.
            return [format_semantic_issue(issue) for issue in resolution.issues]
        constructor_args = resolution.constructor_args
        typed_fixtures = {
            root: fixture.model_dump(exclude_none=True)
            for root, fixture in resolution.fixtures.items()
        }
        capability_issues = _host_runtime_capability_issues(
            contract, typed_fixtures
        )
        if capability_issues:
            return [format_semantic_issue(issue) for issue in capability_issues]
        relational_execution_plan, relational_execution_issues = (
            _resolve_relational_execution_plan(
                contract,
                resolved_dimensions=resolution.dimensions,
                typed_fixtures=typed_fixtures,
                permitted_opaque_fitting_roots=(
                    permitted_opaque_fitting_roots
                ),
            )
        )
        if relational_execution_issues:
            return [
                format_semantic_issue(issue)
                for issue in relational_execution_issues
            ]
        if build_plan is not None:
            try:
                forecasting_execution_plan = (
                    normalize_schema2_forecasting_plan(
                        contract,
                        build_plan,
                        bundle_facts=bundle_facts,
                        method_spec=spec,
                    )
                )
            except RuntimePlanError as exc:
                return [
                    format_semantic_issue(issue)
                    for issue in _runtime_plan_issues(exc)
                ]
        # Version 2 has no spelling ladder.  Every executable value comes from
        # the resolved fixture plan and every constructor value comes from the
        # same semantic evaluator.
        bindings: dict[str, int] = {}
    else:
        constructor_args, constructor_errors = _resolve_constructor_args(
            contract, spec, run_dir
        )
        if constructor_errors:
            # An unresolved declaration is an incomplete generated contract
            # and remains producer-owned under the ordinary Stage-2d fix path.
            return constructor_errors

        # Build the legacy shape-string table, weakest source first: generic
        # defaults, paper values, derived params, then bundle aliases.
        bindings = {
            **DRY_RUN_BINDINGS,
            **_load_bindings_from_spec(spec),
            **_load_bindings_from_params(run_dir),
            **_load_bindings_from_bundle(run_dir),
        }
        errors.extend(_check_contract_shape_consistency(contract))

    reachability_errors = _check_relational_preparation_reachability(
        contract, run_dir
    )
    reachability_errors.extend(
        _check_target_scaling_reachability(
            contract, run_dir, forecasting_execution_plan
        )
    )
    reachability_errors.extend(
        _check_training_history_reachability(
            contract, run_dir, forecasting_execution_plan
        )
    )
    if isinstance(contract, ArchContractV2):
        errors.extend(
            format_semantic_issue(_reachability_issue(message))
            for message in reachability_errors
        )
    else:
        errors.extend(reachability_errors)

    # Builders are OPTIONAL: supervised-ML paradigms have `build_*` functions
    # in training.py; non-ML paradigms (motion_planning) construct their
    # classes directly. The dry-run script falls back to direct construction
    # when no builder is declared for a block, so an empty set is fine here.
    builder_names = _discover_builder_function_names(run_dir)

    # Construct + run the dry-run script. R2C_OFFLINE asks package code not to
    # touch the network: current template-derived downloaders refuse loudly
    # (the runner then retries with a synthetic local dataset), but legacy or
    # custom loaders may not honor the flag. The subprocess timeout bounds
    # anything that still hangs — 150s default, below the driver's 180s
    # run_script wall so the failure carries a precise message instead of a
    # bare rc=124 (the Rethinking-Grouping 2026-07-13 stall: torchvision
    # present, ~150MB CIFAR-10 download).
    try:
        timeout_s = float(os.environ.get(DRY_RUN_TIMEOUT_ENV, "") or DEFAULT_DRY_RUN_TIMEOUT_S)
    except ValueError:
        timeout_s = DEFAULT_DRY_RUN_TIMEOUT_S
    capability_issues: list[SemanticIssue] = []
    with tempfile.TemporaryDirectory(prefix="r2c_arch_runtime_") as temp_dir:
        capability_report_path = (
            Path(temp_dir) / "capabilities.json"
            if isinstance(contract, ArchContractV2)
            else None
        )
        script = _build_dry_run_script(
            contract,
            bindings,
            builder_names,
            constructor_args,
            typed_fixtures=typed_fixtures,
            relational_execution_plan=relational_execution_plan,
            forecasting_execution_plan=forecasting_execution_plan,
            capability_report_path=capability_report_path,
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(run_dir),
                capture_output=True,
                text=True,
                env={**os.environ, OFFLINE_ENV: "1"},
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            partial = e.stderr or b""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            timeout_message = (
                f"runtime dry-run timed out after {timeout_s:.0f}s. The dry-run "
                f"was launched with {OFFLINE_ENV}=1, but package code that does "
                f"not honor this flag can still access the network. The timeout "
                f"may be a dataset download, other blocking I/O, or slow/unbounded "
                f"package work."
                + (f" Partial stderr: {partial[-500:]}" if partial.strip() else "")
            )
            if relational_execution_plan is not None:
                timeout_message = (
                    "relational_indexing.execution.fitting: the supported "
                    "declared one-epoch fitting call did not complete on the "
                    f"tiny typed fixture within {timeout_s:.0f}s; generated "
                    "fitting, preparation, or model code disagrees with the "
                    "bounded execution contract"
                    + (
                        f". Partial stderr: {partial[-500:]}"
                        if partial.strip() else ""
                    )
                )
            errors.append(
                format_semantic_issue(_contract_code_issue(timeout_message))
                if isinstance(contract, ArchContractV2)
                else timeout_message
            )
            return errors
        if capability_report_path is not None:
            if capability_report_path.is_file():
                capability_issues = _read_capability_report(
                    capability_report_path
                )
            elif proc.returncode == 0:
                errors.append(
                    format_semantic_issue(
                        _contract_code_issue(
                            "typed runtime exited before writing its completion "
                            "report; generated package import or execution may "
                            "have terminated the validator subprocess"
                        )
                    )
                )

    errors.extend(format_semantic_issue(issue) for issue in capability_issues)
    if proc.returncode != 0:
        # stderr contains structured per-failure lines plus `SKIP:` lines.
        # SKIP lines are informational (a method that couldn't be exercised,
        # not a contract violation) — they don't constitute a failure. Only
        # non-SKIP lines are errors.
        for line in (proc.stderr or "").strip().splitlines():
            if not line or line.startswith("SKIP:"):
                continue
            if isinstance(contract, ArchContractV2):
                errors.append(format_semantic_issue(_contract_code_issue(line)))
            else:
                # Legacy generated stderr is arbitrary producer text. Prefix
                # it before the outer validator prints its own list so an
                # exact semantic envelope emitted by package code cannot be
                # mistaken for a validator-created ownership record.
                errors.append(
                    "generated package stderr: " + _legacy_untrusted_error(line)
                )
        if not errors:
            empty_message = (
                f"runtime dry-run exited {proc.returncode} with empty stderr; "
                f"stdout: {proc.stdout!r}"
            )
            errors.append(
                format_semantic_issue(_contract_code_issue(empty_message))
                if isinstance(contract, ArchContractV2)
                else empty_message
            )
    return errors


def _declared_schema_version(run_dir: Path) -> str | None:
    """Read the contract version before generated code can execute.

    Legacy child stderr is intentionally untrusted: a generated package can
    print arbitrary bytes, including a valid-looking semantic issue envelope.
    Version 2 wraps every child failure in a validator-created producer issue
    before the exit classifier sees it.
    """
    contract_path = run_dir / ".pipeline" / "arch_contract.json"
    try:
        raw = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    version = raw.get("schema_version", "1.0.0")
    return version if isinstance(version, str) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.spec.is_file():
        print(f"error: spec not found: {args.spec}", file=sys.stderr)
        return 2
    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"error: spec is not valid JSON ({e})", file=sys.stderr)
        return 2

    declared_version = _declared_schema_version(args.run_dir)
    errors = validate(spec, args.run_dir)
    if declared_version != "2.0.0":
        errors = [_legacy_untrusted_error(error) for error in errors]
    if errors:
        print(f"FAIL: {len(errors)} runtime dry-run error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return semantic_issue_exit_code(
            errors,
            trust_pipeline_ownership=declared_version == "2.0.0",
        )
    print(f"ok: arch_contract runtime dry-run cleared at {args.run_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
