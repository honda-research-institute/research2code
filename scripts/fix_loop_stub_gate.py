"""Fix-loop third option — deterministic reachability gates (queue item 7).

At the smoke loop's terminal give-up points the driver MAY convert the single
failing component to a stub and continue toward a partial delivery, instead of
degrading with a broken notebook. "May" is decided here, by deterministic
gates — never by an agent's judgment call and never by the halt-judge
free-associating (fix-loop-third-option-design.md).

Firing points (the maintainer's 2026-07-10 call, resolving the design's G1 ambiguity):
BOTH terminal give-up points in the cap-exhaustion block — the
judge-declines-recovery degrade AND the judge-blessed-extra-still-failed
degrade. Being at either point IS G1 ("in the final cap iteration and about
to give up"); the gates below never fire earlier.

The gates (ALL must hold):
- G1 exhaustion    — positional (see above), recorded for the audit trail.
- G2 breadth       — >= 2 DISTINCT fix targets attempted across iterations.
                     One target retried N times is fixation, not
                     unfixability.
- G3 isolation     — every iteration's failure evidence (traceback deepest
                     owned frame) points into ONE owned component file, the
                     same file the diagnosis targets. A failure that wandered
                     is an integration problem, not a stubbable component.
- G4 plausibility  — the recorded diagnosis class must be consistent with the
                     failing cell's source shape (the 2026-07-06 detr audit:
                     a bare matplotlib plot cell "fixed" for pathological
                     complexity). A cataloged MISMATCH halts the run — the
                     diagnosis chain is broken and stubbing would hide a
                     machinery bug behind a partial delivery.
- G5 one per run   — no fix-loop stub record already exists (records with a
                     stub_path; the feasibility gate's records carry none).

File-to-element mapping (maintainer-approved default, 2026-07-10): the failing
component is FILE-granular. role=core exactly when the file is
method/method.py (the pluggable-component home), else supporting; element_id
is the contract's unique core element for the core home, else synthesized
from the file path. The basis string travels with the record so a mislabel
is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CORE_COMPONENT_FILE = "method/method.py"

# G4 catalog: diagnosis-class keywords -> source shapes at least one of which
# must appear in the failing cell. Classes outside the catalog PASS the
# screen (no opinion) — G4 only vetoes a cataloged mismatch, it never guesses.
_G4_CATALOG: list[tuple[str, tuple[str, ...], tuple[str, ...], str]] = [
    ("complexity/infinite-loop",
     ("complexity", "infinite loop", "infinite_loop", "runaway",
      "unbounded", "timeout", "never terminates"),
     ("for ", "while ", "recursi", ".fit(", "train", "range("),
     "a loop, recursion, or unbounded library call"),
    ("shape-mismatch",
     ("shape mismatch", "shape_mismatch", "dimension mismatch",
      "dimension_mismatch", "input_shape", "wrong shape", "broadcast"),
     ("torch", "np.", "numpy", "shape", "reshape", ".view(", "matmul",
      "@", "array", "tensor", ".size("),
     "tensor/array operations"),
]


@dataclass
class GateVerdict:
    fire: bool = False
    halt_on_g4: bool = False
    element_id: str | None = None
    role: str | None = None
    component_file: str | None = None
    mapping_basis: str | None = None
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return "; ".join(self.reasons)


def _synthesize_id(failing_file: str) -> str:
    """'method/training.py' -> 'method-training' (ids appear on researcher
    surfaces and name the work-order file, so no slashes or suffixes)."""
    out = failing_file[:-3] if failing_file.endswith(".py") else failing_file
    return out.replace("/", "-")


def map_failing_file_to_element(
    contract_elements: list[dict] | None, failing_file: str
) -> tuple[str, str, str]:
    """Returns (element_id, role, basis). See the module docstring for the
    approved default this implements."""
    from partial_delivery import contract_role
    if failing_file == CORE_COMPONENT_FILE:
        cores = [el for el in contract_elements or []
                 if isinstance(el, dict) and el.get("element_id")
                 and contract_role(el.get("role")) == "core"]
        if len(cores) == 1:
            return (str(cores[0]["element_id"]), "core",
                    "contract core element (unique)")
        return (_synthesize_id(failing_file), "core",
                f"synthesized from file {failing_file} ({len(cores)} "
                f"contract core elements, no unique pick)")
    return (_synthesize_id(failing_file), "supporting",
            f"synthesized from file {failing_file}")


def frame_file(frame_anchor: str | None) -> str | None:
    """'method/method.py:175 in fn' -> 'method/method.py'."""
    if not frame_anchor:
        return None
    return str(frame_anchor).split(":", 1)[0].strip() or None


def diagnosis_plausibility_screen(
    diagnosis_text: str, cell_source: str
) -> tuple[bool, str]:
    """G4. Returns (consistent, verdict_line). Only cataloged classes are
    screened; an uncataloged diagnosis passes with a no-opinion verdict."""
    diag = (diagnosis_text or "").lower()
    src = (cell_source or "").lower()
    for name, class_keys, shapes, shape_desc in _G4_CATALOG:
        if any(k in diag for k in class_keys):
            if any(s in src for s in shapes):
                return True, (f"G4 pass: {name} diagnosis and the failing "
                              f"cell contains {shape_desc}")
            return False, (f"G4 MISMATCH: the diagnosis reads as {name} but "
                           f"the failing cell's source contains no "
                           f"{shape_desc} — the diagnosis chain is broken; "
                           f"stubbing would hide a machinery bug")
    return True, "G4 pass: diagnosis class not in the static catalog (no opinion)"


def evaluate_stub_gates(
    *,
    give_up_point: str,
    prior_iterations: list[dict],
    current_frame: str | None,
    cell_source: str,
    existing_stubs: list[dict],
    contract_elements: list[dict] | None,
) -> GateVerdict:
    """Evaluate G1-G5. `fire` is True only when every gate holds; a G4
    cataloged mismatch additionally sets `halt_on_g4` (the caller halts
    instead of degrading). Any other failed gate leaves both flags False —
    the caller behaves exactly as today."""
    v = GateVerdict()
    v.reasons.append(f"G1 pass: at terminal give-up point ({give_up_point})")

    # G2 — breadth over ATTEMPTED targets (per-iteration diagnosis targets).
    targets = {str((it.get("diagnosis") or {}).get("target_file") or "")
               for it in prior_iterations}
    targets.discard("")
    if len(targets) < 2:
        v.reasons.append(
            f"G2 FAIL: only {len(targets)} distinct fix target(s) attempted "
            f"({sorted(targets) or 'none'}) — retrying one target is "
            f"fixation, not demonstrated unfixability")
        return v
    v.reasons.append(f"G2 pass: {len(targets)} distinct targets attempted "
                     f"({sorted(targets)})")

    # G3 — isolation: every recorded frame plus the current one, ONE file.
    frames = [it.get("deepest_owned_frame") for it in prior_iterations]
    frames.append(current_frame)
    files = {frame_file(f) for f in frames}
    if None in files or len(files) != 1:
        missing = sum(1 for f in frames if frame_file(f) is None)
        v.reasons.append(
            "G3 FAIL: failure evidence does not isolate one owned component "
            + (f"({missing} iteration(s) recorded no owned frame)" if missing
               else f"(frames span {sorted(f for f in files if f)})"))
        return v
    component_file = files.pop()
    v.reasons.append(f"G3 pass: every failure frame is in {component_file}")

    # G5 before G4: a second exhaustion halts-as-today without consulting the
    # screen (the screen's halt is reserved for a broken diagnosis chain).
    if any(r.get("stub_path") for r in existing_stubs or []):
        v.reasons.append("G5 FAIL: a fix-loop stub record already exists — "
                         "one conversion per run")
        return v
    v.reasons.append("G5 pass: no prior fix-loop stub record")

    # G4 — plausibility screen over the last recorded diagnosis.
    last_diag = (prior_iterations[-1].get("diagnosis") or {}) \
        if prior_iterations else {}
    diag_text = " ".join(str(last_diag.get(k) or "")
                         for k in ("root_cause", "proposed_fix"))
    ok, verdict = diagnosis_plausibility_screen(diag_text, cell_source)
    v.reasons.append(verdict)
    if not ok:
        v.halt_on_g4 = True
        return v

    element_id, role, basis = map_failing_file_to_element(
        contract_elements, component_file)
    v.fire = True
    v.element_id = element_id
    v.role = role
    v.component_file = component_file
    v.mapping_basis = basis
    v.reasons.append(f"mapping: {element_id} role={role} ({basis})")
    return v


def replace_draft_cell_with_stub_marker(
    draft_text: str, failing_cell_source: str, element_id: str
) -> str | None:
    """Replace the draft cell whose body matches the failing notebook cell
    with the criterion-4 placeholder marker. Returns the new draft text, or
    None when the failing cell cannot be located UNIQUELY — the caller must
    then not fire (an ambiguous edit could stub the wrong cell).

    Matching is whitespace-normalized: the renderer preserves cell bodies, so
    the failing .ipynb cell source equals a draft cell body up to trailing
    whitespace."""
    def norm(s: str) -> str:
        return "\n".join(line.rstrip() for line in (s or "").strip().splitlines())

    target = norm(failing_cell_source)
    if not target:
        return None
    lines = draft_text.splitlines()
    # Draft cells are delimited by `# %%` markers (jupytext percent format).
    boundaries = [i for i, ln in enumerate(lines) if ln.startswith("# %%")]
    if not boundaries:
        return None
    matches = []
    for n, start in enumerate(boundaries):
        end = boundaries[n + 1] if n + 1 < len(boundaries) else len(lines)
        body = "\n".join(lines[start + 1:end])
        if norm(body) == target:
            matches.append((start, end))
    if len(matches) != 1:
        return None
    start, end = matches[0]
    replacement = [f"# %% PLACEHOLDER: component_stub:{element_id}"]
    new_lines = lines[:start] + replacement + lines[end:]
    return "\n".join(new_lines) + ("\n" if draft_text.endswith("\n") else "")
