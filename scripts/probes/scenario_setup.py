"""Scenario-fidelity enforcement — the setup-slice inspection harness.

Slice B of the scenario-fidelity design (R2C-025): a family that declares
`scenario_assumption_dimensions` with `detector` refs opts its runs into a
runtime comparison between the paper's captured scenario assumptions
(`method_spec.scenario_assumptions`, slice A) and the demo setup the notebook
actually executes.

Why runtime inspection: notebook setup cells mostly say
``load_problem("two_rooms_simple")`` — the cell TEXT contains a scene name and
nothing about circles or pedestrian counts, and real deliveries (the pdwa
shape) keep their moving obstacles in notebook-local lists rather than in the
paradigm Environment object. So this module executes the notebook's setup
slice (sections 1 up to the layout's setup boundary — the demo sections are
structurally out of reach) in a bounded child subprocess, and family-owned
OBSERVERS read the instantiated namespace. Pure COMPARATORS then judge the
observation against the captured paper value, so the judgement logic is
unit-testable without any subprocess.

Verdict scheme (probe vocabulary, delivery-label semantics):
- proven contradiction        -> fail                (demotes)
- confirmed match             -> pass
- captured but unbindable     -> flag_for_researcher (demotes; the
  zero-pedestrians shape must not ship as a quiet disclosure)
- no captured assumption      -> unprobeable         (disclosure, no demotion)

Families that declare no dimensions produce no verdicts at all, keeping their
reports byte-identical.

This module is dependency-free beyond the probes package so the vendored
harness (R2C-019) can run it outside the repository; the setup-section
boundary is resolved from the taxonomy in-repo and travels in the harness's
frozen-gating file for vendored runs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    from probes import ProbeVerdict
except ImportError:  # direct __main__ invocation without probes on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from probes import ProbeVerdict
from trusted import trusted_path, trusted_text

# Mirrors scripts/demo_verdict._DEFAULT_SETUP_BOUNDARY: the committed layouts'
# common shape (setup sections 1..3). Kept as a local constant because the
# vendored harness cannot import scripts/; in-repo callers pass the
# layout-resolved boundary and vendored callers pass the frozen one, so this
# fallback only covers a missing boundary on both paths.
DEFAULT_SETUP_BOUNDARY = 4

SCENARIO_SETUP_TIMEOUT_S = 180

# A demo setup that contradicts a paper-stated scenario assumption (polygon
# obstacles for a circles-only paper, zero pedestrians for a crowded-scenario
# paper). Registered in the probe catalog note (internal, not shipped).
SCENARIO_FINDING_CLASS = "M-006"


@dataclass
class ScenarioComparison:
    """A comparator's judgement of one observation against one paper value."""

    verdict: str
    message: str
    evidence: str = ""
    reason: str = ""


@dataclass
class ScenarioDetector:
    """One family-dimension detector: how to observe the executed setup and
    how to compare the observation with the captured paper value."""

    probe_id: str
    observe: Callable[[dict], dict] = field(repr=False)
    compare: Callable[[Any, dict], ScenarioComparison] = field(repr=False)


def resolve_scenario_detector(ref: str) -> ScenarioDetector | None:
    """The registered detector for a family-scoped executor ref, or None.

    Refs look like ``motion_planning.scenario_geometry``: the segment before
    the first dot names the family module under ``probes/``, whose
    ``SCENARIO_DETECTORS`` mapping owns the binding.
    """
    ref = str(ref or "")
    if "." not in ref:
        return None
    family = ref.split(".", 1)[0]
    try:
        module = importlib.import_module(f"probes.{family}")
    except ImportError:
        return None
    registry = getattr(module, "SCENARIO_DETECTORS", None)
    detector = registry.get(ref) if isinstance(registry, dict) else None
    return detector if isinstance(detector, ScenarioDetector) else None


# ---------------------------------------------------------------------------
# Setup-slice walking (child side)
# ---------------------------------------------------------------------------

# Mirror of scripts/demo_verdict.heading_section_number, kept dependency-free
# so the vendored harness can run without repo imports; agreement between the
# two implementations is pinned by tests/test_scenario_fidelity.py.
_HEADING_NUMBER_RE = re.compile(r"^(\d+)\.")
_HEADING_SECTION_RE = re.compile(r"^§\s*(\d+)")


def _heading_section_number(line: str) -> int | None:
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return None
    text = stripped.lstrip("#").lstrip()
    m = _HEADING_NUMBER_RE.match(text) or _HEADING_SECTION_RE.match(text)
    return int(m.group(1)) if m else None


def _cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src or "")


def setup_code_cells(nb: dict, boundary: int) -> list[tuple[int, int, str]]:
    """(cell_index, section, source) for code cells in setup sections
    1 <= section < boundary. Section 0 (install) is excluded and the demo
    sections are structurally unreachable, same slicing as the data-signal
    check."""
    out: list[tuple[int, int, str]] = []
    section: int | None = None
    for idx, cell in enumerate(nb.get("cells") or []):
        if cell.get("cell_type") == "markdown":
            for line in _cell_source(cell).splitlines():
                number = _heading_section_number(line)
                if number is not None:
                    section = number
        elif (cell.get("cell_type") == "code"
              and section is not None and 1 <= section < boundary):
            out.append((idx, section, _cell_source(cell)))
    return out


def _child_main(argv: list[str] | None = None) -> int:
    """Execute the setup slice and print per-detector observations as the
    last stdout line (setup cells may print freely above it)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--boundary", type=int, default=DEFAULT_SETUP_BOUNDARY)
    parser.add_argument("--detector", action="append", default=[])
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    nb = json.loads((run_dir / "notebook.ipynb").read_text(encoding="utf-8"))

    os.chdir(run_dir)
    sys.path.insert(0, str(run_dir))
    namespace: dict = {"__name__": "__r2c_scenario_setup__"}
    setup_error: dict | None = None
    for idx, section, src in setup_code_cells(nb, args.boundary):
        cleaned = "\n".join(
            line for line in src.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        try:
            exec(compile(cleaned, f"<setup-cell-{idx}>", "exec"), namespace)  # noqa: S102
        except BaseException as e:  # noqa: BLE001 — generated cells fail freely
            setup_error = {
                "cell_index": idx,
                "section": section,
                "error": f"{type(e).__name__}: {e}",
            }
            break

    observations: dict[str, dict] = {}
    for ref in args.detector:
        detector = resolve_scenario_detector(ref)
        if detector is None:
            observations[ref] = {
                "bound": False,
                "reason": f"no executor registered for detector {ref!r}",
            }
            continue
        try:
            observations[ref] = detector.observe(namespace)
        except Exception as e:  # noqa: BLE001 — observers face arbitrary objects
            observations[ref] = {
                "bound": False,
                "reason": f"observer raised {type(e).__name__}: {e}",
            }

    # Leading newline: a setup cell that printed without a trailing newline
    # must not concatenate onto the result line the parent parses.
    sys.stdout.write("\n" + json.dumps(
        {"observations": observations, "setup_error": setup_error},
        default=str,
    ) + "\n")
    return 0


# ---------------------------------------------------------------------------
# Bounded inspection + verdict synthesis (parent side)
# ---------------------------------------------------------------------------


def _last_json_line(stdout: str) -> dict | None:
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "observations" in payload:
            return payload
    return None


def inspect_setup_slice(
    run_dir: Path,
    detector_refs: list[str],
    *,
    setup_boundary: int | None = None,
    timeout_s: float = SCENARIO_SETUP_TIMEOUT_S,
) -> dict:
    """Run the child inspector as a bounded subprocess.

    Returns ``{"observations": {ref: obs}, "setup_error": ...}`` on success,
    or ``{"error": "<why the setup could not be inspected at all>"}``.
    """
    # Resolve before building the command: the child chdirs into the run
    # dir, so a relative --run-dir would double-join against its own cwd.
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "notebook.ipynb").is_file():
        return {"error": "no notebook.ipynb"}
    boundary = setup_boundary if setup_boundary else DEFAULT_SETUP_BOUNDARY

    safe_run_dir = str(trusted_path(run_dir))
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--run-dir", safe_run_dir, "--boundary", trusted_text(str(boundary)),
    ]
    for ref in detector_refs:
        cmd += ["--detector", trusted_text(ref)]
    env = dict(os.environ)
    # The child imports the generated package and executes generated setup
    # cells: keep bytecode caches out of the run tree (the R2C-019
    # byte-identical contract) and plotting headless.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("MPLBACKEND", "Agg")
    probes_parent = str(Path(__file__).resolve().parent.parent)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        probes_parent + os.pathsep + existing if existing else probes_parent
    )
    try:
        proc = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True,
            timeout=timeout_s, cwd=safe_run_dir, env=env, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": (f"setup-slice execution exceeded the "
                          f"{timeout_s:.0f}s budget")}
    except OSError as e:
        return {"error": f"could not launch the setup-slice inspector: {e}"}

    payload = _last_json_line(proc.stdout)
    if payload is None:
        tail = (proc.stderr or "").strip().splitlines()
        detail = f": {tail[-1]}" if tail else ""
        return {"error": (f"setup-slice inspector produced no result "
                          f"(exit {proc.returncode}{detail})")}
    return payload


def _spec_assumptions(run_dir: Path) -> dict[str, dict]:
    spec_path = Path(run_dir) / ".pipeline" / "method_spec.json"
    if not spec_path.is_file():
        return {}
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    assumptions = spec.get("scenario_assumptions") if isinstance(spec, dict) else None
    if not isinstance(assumptions, dict):
        return {}
    return {k: v for k, v in assumptions.items() if isinstance(v, dict)}


def _dimension_label(dimension_id: str) -> str:
    return str(dimension_id).replace("_", " ").strip()


def scenario_fidelity_verdicts(
    run_dir: Path,
    dimension_bindings: list[tuple[str, str]],
    *,
    setup_boundary: int | None = None,
    timeout_s: float = SCENARIO_SETUP_TIMEOUT_S,
) -> list[tuple[ProbeVerdict, str]]:
    """(verdict, detector_ref) pairs for every declared scenario dimension.

    ``dimension_bindings`` is the node's declaration: (dimension_id,
    detector_ref) pairs. The setup slice executes at most once, and only when
    at least one declared dimension has a captured paper assumption.
    """
    run_dir = Path(run_dir)
    assumptions = _spec_assumptions(run_dir)

    resolved: list[tuple[str, str, ScenarioDetector]] = []
    for dimension_id, ref in dimension_bindings:
        detector = resolve_scenario_detector(ref)
        if detector is None:
            # Registry drift is a coding error, not a run property: raise
            # loudly (the battery's fail-closed path) instead of silently
            # skipping a declared enforcement surface. A structural test pins
            # every committed declaration to a registered executor.
            raise ValueError(
                f"scenario dimension {dimension_id!r} declares detector "
                f"{ref!r} but no executor is registered for it"
            )
        resolved.append((dimension_id, ref, detector))

    out: list[tuple[ProbeVerdict, str]] = []
    captured: list[tuple[str, str, ScenarioDetector, dict]] = []
    for dimension_id, ref, detector in resolved:
        entry = assumptions.get(dimension_id)
        if not isinstance(entry, dict) or "normalized_value" not in entry:
            out.append((ProbeVerdict(
                detector.probe_id, "unprobeable",
                f"the paper states no {_dimension_label(dimension_id)} "
                f"assumption — nothing to compare against the executed demo "
                f"setup",
                finding_class=SCENARIO_FINDING_CLASS,
                reason="no_captured_assumption",
            ), ref))
            continue
        captured.append((dimension_id, ref, detector, entry))

    if not captured:
        return out

    inspection = inspect_setup_slice(
        run_dir, [ref for _, ref, _, _ in captured],
        setup_boundary=setup_boundary, timeout_s=timeout_s,
    )
    if "observations" not in inspection:
        why = str(inspection.get("error") or "unknown inspection failure")
        for dimension_id, ref, detector, _ in captured:
            out.append((ProbeVerdict(
                detector.probe_id, "flag_for_researcher",
                f"the executed demo setup could not be inspected ({why}) — "
                f"researcher judgment needed on the paper's stated "
                f"{_dimension_label(dimension_id)} assumption",
                finding_class=SCENARIO_FINDING_CLASS,
                reason="setup_uninspectable",
            ), ref))
        return out

    observations = inspection.get("observations") or {}
    setup_error = inspection.get("setup_error")
    error_note = ""
    if isinstance(setup_error, dict):
        error_note = (f" (setup cell {setup_error.get('cell_index')} raised "
                      f"{setup_error.get('error')} before the slice finished)")

    for dimension_id, ref, detector, entry in captured:
        label = _dimension_label(dimension_id)
        obs = observations.get(ref)
        if not isinstance(obs, dict):
            out.append((ProbeVerdict(
                detector.probe_id, "flag_for_researcher",
                f"the setup-slice inspector returned no observation for "
                f"{label}{error_note} — researcher judgment needed",
                finding_class=SCENARIO_FINDING_CLASS,
                reason="setup_uninspectable",
            ), ref))
            continue
        comparison = detector.compare(entry.get("normalized_value"), obs)
        verdict, message, reason = (
            comparison.verdict, comparison.message, comparison.reason,
        )
        if error_note:
            # An incomplete setup keeps affirmative CONTRADICTIONS (the
            # offending objects exist as constructed) but an affirmative
            # match over a partial namespace would be claims-ahead-of-reality
            # — demote it to researcher judgment and say why.
            if verdict == "pass":
                verdict = "flag_for_researcher"
                reason = "setup_incomplete"
                message = (f"the records constructed before the setup failure "
                           f"match the paper's stated {label} assumption, but "
                           f"the setup slice did not finish "
                           f"executing{error_note} — researcher judgment "
                           f"needed on the full setup")
            else:
                message += error_note
        out.append((ProbeVerdict(
            detector.probe_id, verdict, message,
            evidence=comparison.evidence,
            finding_class=SCENARIO_FINDING_CLASS,
            reason=reason,
        ), ref))
    return out


if __name__ == "__main__":  # pragma: no cover — exercised via subprocess
    # Delegate to the canonically-imported module: running this file as
    # __main__ would otherwise create a SECOND ScenarioDetector class whose
    # isinstance check rejects every registry entry built against the
    # probes.scenario_setup copy.
    from probes.scenario_setup import _child_main as _canonical_child_main

    sys.exit(_canonical_child_main())
