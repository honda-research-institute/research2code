"""Stage 2.x — validator for parameter-deriver output.

Runs after `derive_params.py`. Confirms that:

  1. `<run_dir>/.pipeline/params.json` exists.
  2. The file parses against the `Params` Pydantic schema (which enforces the
     "required fields per source" invariants — paper / system_default /
     system_inferred / spec_default).
  3. Every parameter the spec's `comparison.pluggable_component.signature` declares
     as a kwarg-with-default is represented in `params` (the notebook would otherwise
     have no way to set it).
  4. Standard AL parameters (`batch_size`, `num_rounds`, `initial_labeled`,
     `learning_rate`, `max_epochs`, `train_until_accuracy`, `pool_size`, `hidden_dim`)
     are all present.

Deterministic gate. On failure, exit 1 with a structured error list on stderr.

Usage:

    python scripts/validate_params_output.py --spec <method_spec.json> --run-dir <output_dir>

Exit codes:
  0  validation passed (zero errors)
  1  validation failures
  2  setup error (missing spec / params.json)
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pydantic import ValidationError  # noqa: E402

from schemas.params import Params  # noqa: E402
from scripts.bundle_axis_floor import (  # noqa: E402
    ProtocolAxisUnresolved,
    load_bundle_manifest,
    protocol_axis_param_shrinks,
    protocol_quantity_steps,
)


# Required-params sets per paradigm. The notebook's §3 setup cells expect these
# specific names; if missing, downstream rendering breaks at notebook-build time.
#
# This is currently keyed on a coarse paradigm prefix (active_learning/* vs.
# knowledge_distillation/*) because each paradigm's notebook layout uses
# different param names. The longer-term move is to declare `required_params`
# in the taxonomy node and have this script read it; for now the branching is
# explicit so adding a new paradigm is a visible diff here.
REQUIRED_PARAMS_BY_PARADIGM: dict[str, set[str]] = {
    "active_learning": {
        "batch_size",
        "num_rounds",
        "initial_labeled",
        "pool_size",
        "learning_rate",
        "max_epochs",
        "train_until_accuracy",
        "hidden_dim",
    },
    "knowledge_distillation": {
        "batch_size",
        "num_epochs",
        "learning_rate",
        "train_size",
        "n_test",
        "hidden_dim",
    },
}


def _required_params_for(paradigm_id: str) -> set[str]:
    """Return the required-params set for the given paradigm.

    Match by prefix so sub-paradigms (e.g., `active_learning/batch_acquisition`)
    inherit the parent's required-params. Returns empty set for unknown paradigms
    (so the check becomes a no-op rather than a false failure).
    """
    for prefix, required in REQUIRED_PARAMS_BY_PARADIGM.items():
        if paradigm_id.startswith(prefix):
            return required
    return set()


def _parse_signature_kw_defaults(signature: str) -> set[str]:
    """Return names of kwarg-with-default params in the signature, skipping fixed positionals.

    SKIP is the union of `pluggable_component.contract.fixed_positional_args`
    across paradigms (active_learning parent + bayesian override, KD,
    motion_planning), plus `seed`. An unused name in the union is harmless (it
    just won't match a given method's signature). Eventual cleaner form: read
    the matched taxonomy node's `fixed_positional_args` rather than a union.
    """
    SKIP = {"model", "x_unlabeled", "x_labeled", "batch_size", "seed",
            "student", "teacher", "batch",  # KD positionals
            "start", "goal", "environment", "dynamics"}  # motion_planning positionals
    try:
        stub = f"def {signature}:\n    pass\n"
        tree = ast.parse(stub)
    except (SyntaxError, ValueError):
        return set()
    if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()
    func = tree.body[0]
    args = func.args

    out: set[str] = set()

    n_pos_no_default = len(args.args) - len(args.defaults)
    for i, arg in enumerate(args.args):
        if arg.arg in SKIP:
            continue
        if i - n_pos_no_default < 0:
            continue
        out.add(arg.arg)

    for i, arg in enumerate(args.kwonlyargs):
        if arg.arg in SKIP:
            continue
        if args.kw_defaults[i] is None:
            continue
        out.add(arg.arg)

    return out


def validate(spec: dict, run_dir: Path) -> list[str]:
    errors: list[str] = []

    params_path = run_dir / ".pipeline" / "params.json"
    if not params_path.is_file():
        return [f"params.json missing at {params_path}"]

    raw = params_path.read_text(encoding="utf-8")
    try:
        Params.model_validate_json(raw)
    except ValidationError as e:
        errors.append(f"params.json fails Pydantic validation: {e}")
        return errors
    except json.JSONDecodeError as e:
        errors.append(f"params.json is not valid JSON: {e}")
        return errors

    # Now do cross-checks against the spec.
    parsed = json.loads(raw)
    params = parsed.get("params", {})

    # Required-params check, scoped to paradigm
    paradigm_id = (spec.get("comparison") or {}).get("classification", {}).get("id", "")
    required = _required_params_for(paradigm_id)
    missing_required = required - set(params.keys())
    if missing_required:
        errors.append(
            f"params.json missing required parameters for paradigm '{paradigm_id}': "
            f"{sorted(missing_required)}. The notebook's §3 setup cells expect these."
        )

    # Every signature kwarg-with-default is represented
    pc = (spec.get("comparison") or {}).get("pluggable_component") or {}
    sig = pc.get("signature") or ""
    sig_extras = _parse_signature_kw_defaults(sig)
    missing_extras = sig_extras - set(params.keys())
    if missing_extras:
        errors.append(
            f"params.json missing paradigm-extras declared in pluggable_component.signature: "
            f"{sorted(missing_extras)}. The notebook would have no way to set these for the "
            f"per-round acquisition call."
        )

    # Provenance probes as part of the deterministic gate (wired 2026-06-10).
    # Error-severity findings block: implausible values (a learning rate of
    # 7.4) and fabricated paper quotes (the "exceeds 99%" class that shipped
    # in the june9 run and reproduced on the fresh GBALD run, both caught
    # only at audit time before this wiring). Warn-severity findings (e.g. a
    # paper-sourced entry missing its section locator) stay non-blocking —
    # the stage reviewer and the audit battery still see them.
    errors.extend(_provenance_errors(params, run_dir))
    errors.extend(_evaluation_protocol_errors(spec, params))
    errors.extend(_runtime_usage_errors(params, run_dir))
    errors.extend(_batch_acquisition_invariant_errors(params, paradigm_id))
    manifest = load_bundle_manifest(run_dir)
    if manifest is not None:
        errors.extend(
            shrink.validation_message()
            for shrink in protocol_axis_param_shrinks(spec, params, manifest)
            if shrink.feasible
        )

    return errors


def _values_equivalent(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return left == right


def _evaluation_protocol_errors(spec: dict, params: dict) -> list[str]:
    """Cross-check role, value, unit, granularity, and evidence as one join."""
    protocol = ((spec.get("comparison") or {}).get("evaluation_protocol")
                or {})
    quantities = protocol.get("quantities") or []
    if not isinstance(quantities, list):
        return []

    errors: list[str] = []
    bound_names: set[str] = set()
    for index, quantity in enumerate(quantities):
        if not isinstance(quantity, dict):
            continue
        name = quantity.get("parameter_name")
        if not isinstance(name, str) or not name:
            continue
        bound_names.add(name)
        role = quantity.get("role")
        entry = params.get(name)
        path = f"comparison.evaluation_protocol.quantities[{index}]"
        if not isinstance(entry, dict):
            errors.append(
                f"evaluation_protocol_carrier: {path} binds "
                f"parameter_name={name!r}, role={role!r}, but params.json "
                "has no such carrier"
            )
            continue

        expected = {
            "protocol_role": role,
            "protocol_value": quantity.get("value"),
            "protocol_unit": quantity.get("unit"),
            "protocol_granularity": quantity.get("granularity"),
            "protocol_axis_says": quantity.get("axis_evidence_quote"),
            "protocol_axis_section": quantity.get("axis_paper_section"),
            "protocol_axis_element_ids": quantity.get(
                "axis_paper_element_ids"
            ),
            "paper_value_status": quantity.get("paper_value_status"),
            "paper_says": quantity.get("evidence_quote"),
            "paper_section": quantity.get("paper_section"),
            "paper_element_ids": quantity.get("paper_element_ids"),
        }
        for field, value in expected.items():
            actual = entry.get(field)
            if not _values_equivalent(actual, value):
                errors.append(
                    f"evaluation_protocol_carrier: params.{name}.{field}="
                    f"{actual!r} does not match {path}.{field}={value!r} "
                    f"for role={role!r}; value, role, unit, granularity, and "
                    "evidence must travel together"
                )

        status = quantity.get("paper_value_status")
        paper_quantity = quantity.get("value")
        runtime_value = entry.get("value")
        if (
            isinstance(runtime_value, bool)
            or not isinstance(runtime_value, (int, float))
            or runtime_value <= 0
        ):
            errors.append(
                f"evaluation_protocol_carrier: params.{name}.value="
                f"{runtime_value!r} is not a positive numeric quantity for "
                f"role={role!r}; booleans cannot equal temporal value 1"
            )
            continue
        if status == "paper_unspecified":
            if entry.get("paper_value") is not None:
                errors.append(
                    f"evaluation_protocol_carrier: params.{name}.paper_value="
                    f"{entry.get('paper_value')!r}, but {path} marks "
                    f"role={role!r} paper_unspecified; nearby notation or a "
                    "validation/test span cannot authorize paper provenance"
                )
            if entry.get("source") not in {"system_inferred", "spec_default"}:
                errors.append(
                    f"evaluation_protocol_carrier: params.{name}.source="
                    f"{entry.get('source')!r} falsely attributes a "
                    f"paper-unspecified {role!r} runtime value"
                )
            continue

        if status != "paper_stated":
            continue
        try:
            paper_steps = protocol_quantity_steps(quantity, path=path)
        except ProtocolAxisUnresolved as exc:
            errors.append(
                "evaluation_protocol_carrier: cannot derive a step-count "
                f"paper value for params.{name}: {exc}"
            )
            continue

        conversion = (
            f"{paper_quantity!r} {quantity.get('unit')!r} / "
            f"{quantity.get('granularity')!r} "
            f"{quantity.get('unit')!r}/step = {paper_steps} runtime steps"
        )
        if _values_equivalent(runtime_value, paper_steps):
            if entry.get("source") != "paper":
                errors.append(
                    f"evaluation_protocol_carrier: params.{name} matches the "
                    f"paper-stated {role!r} converted value ({conversion}) "
                    "but source="
                    f"{entry.get('source')!r}, expected 'paper'"
                )
            if entry.get("paper_value") is not None:
                errors.append(
                    f"evaluation_protocol_carrier: params.{name}.paper_value="
                    f"{entry.get('paper_value')!r} is stale because runtime "
                    f"value {runtime_value!r} already matches the "
                    f"paper-stated {role!r} conversion ({conversion}); remove "
                    "the competing value"
                )
        elif (
            entry.get("source") != "system_default"
            or not _values_equivalent(entry.get("paper_value"), paper_steps)
        ):
            errors.append(
                f"evaluation_protocol_carrier: params.{name} runtime value "
                f"{runtime_value!r} differs from paper-stated {role!r} "
                f"conversion ({conversion}); expected "
                "source='system_default' and the exact converted step count "
                "in paper_value"
            )

    for name, entry in params.items():
        if not isinstance(entry, dict) or entry.get("protocol_role") is None:
            continue
        if name not in bound_names:
            errors.append(
                f"evaluation_protocol_carrier: params.{name} claims "
                f"protocol_role={entry.get('protocol_role')!r} without an "
                "explicit comparison.evaluation_protocol parameter binding"
            )
    return errors


def _batch_acquisition_invariant_errors(params: dict, paradigm_id: str) -> list[str]:
    """Enforce the two-stage acquisition invariant b >= b' for active learning.

    A two-stage selector (GBALD: BALD prefilter → geometric ranking; any
    method that preselects a candidate pool and then ranks down to the final
    batch) takes the top `batch_returns` candidates and returns `batch_size` of
    them. If `batch_returns < batch_size` the selector can never return the
    requested batch — it silently throttles acquisition to `batch_returns`
    labels per round, flattening the headline learning curve with no crash
    (GBALD's F002: batch_returns=30 < batch_size=100 acquired 30/round, not
    100). The invariant only fires when BOTH knobs are present, so it is a
    no-op for single-stage selectors (e.g. BADGE) that declare no
    `batch_returns`. Raising it here makes Stage 2.x route the violation
    through the producer fix loop instead of deferring it as an audit-time
    finding.
    """
    if not paradigm_id.startswith("active_learning"):
        return []
    br = params.get("batch_returns")
    bs = params.get("batch_size")
    if not isinstance(br, dict) or not isinstance(bs, dict):
        return []
    br_val, bs_val = br.get("value"), bs.get("value")
    numeric = (int, float)
    if isinstance(br_val, bool) or isinstance(bs_val, bool):
        return []
    if not isinstance(br_val, numeric) or not isinstance(bs_val, numeric):
        return []
    if br_val >= bs_val:
        return []
    return [
        f"batch_acquisition_invariant: batch_returns ({br_val}) < batch_size "
        f"({bs_val}). A two-stage active-learning selector preselects "
        f"`batch_returns` candidates and returns `batch_size` of them, so "
        f"`batch_returns` must be >= `batch_size` (the paper invariant b >= b'). "
        f"As shipped the selector can return at most {br_val} positions per round, "
        f"silently throttling acquisition to {br_val} labels/round instead of "
        f"{bs_val}. Set batch_returns >= batch_size (e.g. the paper's prefilter "
        f"size, or at minimum batch_size)."
    ]


_MODEL_RUNTIME_PARAMS = {"hidden_dim", "dropout_rate"}


def _source_mentions_name(path: Path, name: str) -> bool:
    if not path.is_file():
        return False
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.arg) and node.arg == name:
            return True
    return False


def _runtime_usage_errors(params: dict, run_dir: Path) -> list[str]:
    """Check model-shape params against the generated runtime source.

    This gate is intentionally skipped when method source is absent so the
    provenance-only unit tests and pre-code explanation-only packages do not
    fabricate a method package. In the real Stage 2.x path, method/model.py
    and method/training.py already exist.
    """
    method_dir = run_dir / "method"
    runtime_files = [method_dir / "model.py", method_dir / "training.py"]
    if not any(path.is_file() for path in runtime_files):
        return []

    errors: list[str] = []
    for name in sorted(_MODEL_RUNTIME_PARAMS):
        entry = params.get(name)
        if not isinstance(entry, dict):
            continue
        if entry.get("used_in_notebook") is False:
            continue
        if any(_source_mentions_name(path, name) for path in runtime_files):
            continue
        errors.append(
            f"param_runtime_drift: params.json marks `{name}` usable, but "
            "method/model.py and method/training.py do not expose or read that "
            "name. Either wire the model/runtime to consume it, mark it "
            "`used_in_notebook=false` with `unused_reason`, or omit it."
        )

    return errors


def _provenance_errors(params: dict, run_dir: Path) -> list[str]:
    from scripts.validate_params_provenance import check_params  # noqa: PLC0415

    paper_path = run_dir / ".pipeline" / "paper.md"
    paper = (paper_path.read_text(encoding="utf-8")
             if paper_path.is_file() else None)
    findings = check_params(params, paper)
    return [
        f"provenance probe {f['probe']}: {f['param']}: {f['message']}"
        for f in findings if f["severity"] == "error"
    ]


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

    errors = validate(spec, args.run_dir)
    if errors:
        print(f"FAIL: {len(errors)} validation error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"ok: parameter-deriver output at {args.run_dir} validates against the schema + spec.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
