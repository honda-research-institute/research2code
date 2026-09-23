"""Partial delivery — the stubbed-components artifact and its formats.

Design: the partial delivery design note (internal, not shipped) (APPROVED 2026-07-05;
the §3.5 unambiguity constraint is BINDING: a partial package must be
impossible to mistake for a complete one, on every surface, without
cross-referencing anything). This module owns the pieces that make a stub
explicit instead of silent:

- the canonical run artifact ``.pipeline/stubbed_elements.json`` —
  producers (the feasibility gate, a future fix-loop conversion) append
  records here; the delivery gate loads them and the label derivation
  turns them into the recorded `partial_delivery` cause,
- the work-order markdown format (§3.1: the precise interface, the paper
  anchor with verbatim quotes, why the pipeline could not build it, what
  was verified around it — machine-consumable on purpose: a researcher
  pastes it plus the package into their own model and asks for the
  missing implementation),
- the stub-module format (§3.5 criterion 3: the docstring AND the body
  both self-identify, so a researcher who opens the file directly cannot
  mistake it for an implementation),
- delivery-time surface verification: a recorded stub whose files are
  missing or fail to self-identify becomes a demoting reason, loudly —
  never a silent downgrade.

No producer is wired yet (this block builds the delivery side; the
producer decision points — contract-driven stubbing at the feasibility
gate, the fix-loop third option — are their own build blocks per the
design's §3.4 risk note).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ARTIFACT_NAME = "stubbed_elements.json"
ARTIFACT_SCHEMA_VERSION = "1.0"

# The self-identification markers verification checks for (criterion 3).
# Deliberately plain words, not codes: they are researcher-facing.
_STUB_MARKER = "NOT IMPLEMENTED"
_PARTIAL_MARKER = "PARTIAL"


def contract_role(role: str | None) -> str:
    """Map a methodology-contract role onto the two-way stub ranking.

    The contract ranks four roles; the label and the surfaces only need
    the §3.2 split: the core mechanism dominates, everything else is
    supporting.
    """
    return "core" if str(role or "") == "core_methodology" else "supporting"


def load_stubbed_elements(pipeline_dir: Path) -> tuple[list[dict], str | None]:
    """Read the canonical artifact. Returns (records, error).

    Missing file means no stubs (the overwhelmingly common case) — never
    an error. A file that exists but cannot be read or fails the schema
    IS an error: the caller demotes on it, because an unreadable partial
    record could hide a partial state (§3.5 criterion 5).
    """
    path = Path(pipeline_dir) / ARTIFACT_NAME
    if not path.is_file():
        return [], None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return [], f"{ARTIFACT_NAME} unreadable: {type(e).__name__}: {e}"
    records = data.get("stubbed_elements") if isinstance(data, dict) else None
    if not isinstance(records, list):
        return [], f"{ARTIFACT_NAME} malformed: no stubbed_elements list"
    try:
        from schemas.final_manifest import StubbedElement
        validated = [
            StubbedElement.model_validate(r).model_dump(exclude_none=True)
            for r in records
        ]
    except Exception as e:  # noqa: BLE001 — schema failure must demote, not crash
        return [], f"{ARTIFACT_NAME} failed schema validation: {e}"
    return validated, None


def record_stubbed_element(
    pipeline_dir: Path,
    *,
    element_id: str,
    role: str,
    work_order: str,
    stub_path: str | None = None,
) -> dict:
    """Producer API: append (or replace, keyed on element_id) one record.

    ``role`` accepts either the two-way ranking (`core`/`supporting`) or a
    raw methodology-contract role. Validation is fail-loud: producers must
    never write a record the delivery gate would refuse.
    """
    from schemas.final_manifest import StubbedElement

    if role not in ("core", "supporting"):
        role = contract_role(role)
    record = StubbedElement(
        element_id=element_id, role=role, work_order=work_order,
        stub_path=stub_path,
    ).model_dump(exclude_none=True)
    path = Path(pipeline_dir) / ARTIFACT_NAME
    existing, error = load_stubbed_elements(pipeline_dir)
    if error:
        raise ValueError(f"refusing to append to a bad artifact — {error}")
    kept = [r for r in existing if r["element_id"] != element_id]
    kept.append(record)
    kept.sort(key=lambda r: (r["role"] != "core", r["element_id"]))
    path.write_text(
        json.dumps({
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "stubbed_elements": kept,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return record


def remove_stubbed_element(pipeline_dir: Path, element_id: str) -> bool:
    """Producer API: withdraw one record (a corrected spec on resume can
    un-stub an element). Returns True when a record was removed. The
    artifact file is deleted entirely when the last record goes — no
    empty-list file left to read as vestigially partial."""
    path = Path(pipeline_dir) / ARTIFACT_NAME
    existing, error = load_stubbed_elements(pipeline_dir)
    if error:
        raise ValueError(f"refusing to edit a bad artifact — {error}")
    kept = [r for r in existing if r["element_id"] != element_id]
    if len(kept) == len(existing):
        return False
    if not kept:
        path.unlink(missing_ok=True)
        return True
    path.write_text(
        json.dumps({
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "stubbed_elements": kept,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return True


# ---------------------------------------------------------------------------
# Shared surface helpers (§3.5 criteria 1 + 2): every entry surface renders
# the SAME numbers and the same stub names from these, so no surface can
# drift into softer wording than another.
# ---------------------------------------------------------------------------


def completeness_counts(pipeline_dir: Path, stubs: list[dict]) -> dict:
    """{"stubbed": K, "total": M | None, "implemented": N | None}.

    M is the methodology-replication-contract element count (the pipeline's
    own per-element knowledge, design §2.1); when the contract is absent
    (older specs) the totals are honestly None rather than guessed.
    """
    k = len(stubs or [])
    total = None
    spec_path = Path(pipeline_dir) / "method_spec.json"
    if spec_path.is_file():
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            elements = ((spec.get("methodology_replication_contract") or {})
                        .get("elements") or [])
            if elements:
                total = len(elements)
        except (OSError, json.JSONDecodeError, AttributeError):
            total = None
    implemented = max(total - k, 0) if total is not None else None
    return {"stubbed": k, "total": total, "implemented": implemented}


def completeness_statement(counts: dict) -> str:
    """The criterion-2 sentence: numbers, not adjectives."""
    k = counts.get("stubbed") or 0
    total = counts.get("total")
    if total:
        return (f"{counts['implemented']} of {total} components "
                f"implemented, {k} stubbed")
    return (f"{k} component{'' if k == 1 else 's'} NOT implemented, "
            f"shipped as self-identifying stub{'' if k == 1 else 's'}")


def stub_display_lines(stubs: list[dict]) -> list[str]:
    """One line per stub, core first: name, ranking, work-order pointer."""
    lines = []
    for rec in stubs or []:
        what = ("the paper's core mechanism" if rec.get("role") == "core"
                else "supporting component")
        lines.append(f"`{rec.get('element_id')}` — {what}. "
                     f"Work order: `{rec.get('work_order')}`")
    return lines


def core_gap_clause(stubs: list[dict]) -> str | None:
    """§3.2: a stubbed core element forces the headline. None when no core
    element is stubbed."""
    cores = [r.get("element_id") for r in stubs or []
             if r.get("role") == "core"]
    if not cores:
        return None
    names = ", ".join(f"`{c}`" for c in cores)
    return (f"The paper's core mechanism ({names}) is NOT implemented — "
            f"this package is a scaffold around it, not a draft of it.")


# ---------------------------------------------------------------------------
# The work-order format (§3.1) — the stub boundary as a WORK ORDER, not a
# TODO. Acceptance bar: a researcher pasting this file plus the package into
# a frontier model should need to add nothing before asking for the missing
# implementation.
# ---------------------------------------------------------------------------


def render_work_order(
    *,
    element_id: str,
    role: str,
    interface: dict | None = None,
    paper_anchor: dict | None = None,
    why_not_built: str,
    verified_neighborhood: list[str] | None = None,
    fix_history: str | None = None,
) -> str:
    """Render one work order. Every section renders even when its input is
    missing — an honest "not recorded" beats a silently absent section.

    ``interface``: {"signature", "arguments": [str], "returns",
    "call_sites": [str]} — taken from the build plan and architecture
    contract the pipeline already derived.
    ``paper_anchor``: {"section", "references": [str], "quotes": [str]} —
    quotes must be copied verbatim from paper.md (the closed-set
    discipline: copy tokens, never paraphrase from memory).
    """
    is_core = role in ("core", "core_methodology")
    what = "the paper's CORE mechanism" if is_core else "a supporting component"
    out = [
        f"# PARTIAL delivery work order — `{element_id}` is NOT implemented",
        "",
        f"> This package is PARTIAL: `{element_id}` ({what}) was NOT built "
        f"by R2C. This file is a machine-consumable work order — paste it, "
        f"together with the package, into your own model and ask for the "
        f"missing implementation.",
        "",
        "## The precise interface",
        "",
    ]
    interface = interface or {}
    if interface.get("signature"):
        out.extend(["```python", str(interface["signature"]).strip(), "```", ""])
    else:
        out.extend(["The interface was not recorded — derive it from the "
                    "call sites below and the architecture contract "
                    "(`.pipeline/arch_contract.json`).", ""])
    if interface.get("arguments"):
        out.append("Arguments (shapes and dtypes):")
        out.append("")
        out.extend(f"- {a}" for a in interface["arguments"])
        out.append("")
    if interface.get("returns"):
        out.extend([f"Returns: {interface['returns']}", ""])
    if interface.get("call_sites"):
        out.append("Call sites that consume this component:")
        out.append("")
        out.extend(f"- `{c}`" for c in interface["call_sites"])
        out.append("")
    out.extend(["## The paper anchor", ""])
    paper_anchor = paper_anchor or {}
    if paper_anchor.get("section"):
        out.extend([f"Paper location: {paper_anchor['section']}", ""])
    if paper_anchor.get("references"):
        out.extend(f"- {r}" for r in paper_anchor["references"])
        out.append("")
    quotes = [q for q in paper_anchor.get("quotes") or [] if str(q).strip()]
    if quotes:
        out.append("Load-bearing quotes, copied verbatim from the paper:")
        out.append("")
        for q in quotes:
            out.extend(f"> {line}" for line in str(q).strip().splitlines())
            out.append("")
    else:
        out.extend(["No verbatim quotes were recorded for this element — "
                    "read the paper section named above before "
                    "implementing.", ""])
    out.extend([
        "## Why R2C could not build it",
        "",
        why_not_built.strip() or "The reason was not recorded.",
        "",
        "## What was verified around it",
        "",
    ])
    neighborhood = [n for n in verified_neighborhood or [] if str(n).strip()]
    if neighborhood:
        out.append("The boundary you are filling into carries these "
                   "component-level verdicts:")
        out.append("")
        out.extend(f"- {n}" for n in neighborhood)
        out.append("")
    else:
        out.extend(["No component-level verdicts were recorded for the "
                    "neighboring components — treat the boundary as "
                    "unverified.", ""])
    if fix_history and fix_history.strip():
        out.extend([
            "## Fix-loop history",
            "",
            "This stub came from an exhausted fix loop; the full history "
            "and the final diagnosis:",
            "",
            fix_history.strip(),
            "",
        ])
    return "\n".join(out)


# ---------------------------------------------------------------------------
# The stub-module format (§3.5 criterion 3): docstring AND body both say the
# component is NOT implemented; execution raises with the work-order pointer.
# ---------------------------------------------------------------------------


_DEF_RE = re.compile(r"^\s*def\s+(\w+)\s*(\(.*\))\s*(?:->\s*([^:]+))?:?\s*$")


def render_stub_module(
    *,
    element_id: str,
    work_order: str,
    signatures: list[str] | None = None,
) -> str:
    """Python source for a self-identifying stub module.

    ``signatures`` are the declared interface defs (e.g. ``def match(cost:
    "Tensor") -> "Tensor"``); each becomes a named function that raises.
    A module-level ``__getattr__`` catches every other access, so no import
    path through this module can run silently.
    """
    message = (f"{_STUB_MARKER}: `{element_id}` is a {_PARTIAL_MARKER} "
               f"delivery stub, not an implementation. "
               f"See the work order: {work_order}")
    out = [
        '"""' + f"{_STUB_MARKER} — {_PARTIAL_MARKER} DELIVERY STUB.",
        "",
        f"The component `{element_id}` was NOT built by R2C. This module is",
        "a placeholder that fails loudly on any use. Do NOT treat it as an",
        "implementation.",
        "",
        f"Work order (interface, paper anchor, and why): {work_order}",
        '"""',
        "",
        f'STUB_ELEMENT_ID = "{element_id}"',
        f'WORK_ORDER = "{work_order}"',
        "",
        "",
        "def _raise():",
        f'    raise NotImplementedError(\n        {message!r})',
        "",
    ]
    for sig in signatures or []:
        sig = str(sig).strip().rstrip(":")
        m = _DEF_RE.match(sig)
        if not m:
            continue
        out.extend([
            "",
            f"{sig}:",
            f'    """{_STUB_MARKER} — raises. See WORK_ORDER."""',
            "    _raise()",
            "",
        ])
    out.extend([
        "",
        "def __getattr__(name):",
        "    # Dunder lookups keep normal semantics so import machinery and",
        "    # inspection tooling behave; every real attribute access raises.",
        '    if name.startswith("__") and name.endswith("__"):',
        "        raise AttributeError(name)",
        "    _raise()",
        "",
    ])
    return "\n".join(out)


def write_stub(
    run_dir: Path,
    *,
    element_id: str,
    role: str,
    stub_rel_path: str,
    work_order_markdown: str,
    signatures: list[str] | None = None,
) -> dict:
    """Producer convenience: write the work order + the stub module, then
    record the element in the canonical artifact. Returns the record."""
    run_dir = Path(run_dir)
    work_order_rel = f"work_orders/{element_id}.md"
    work_order_path = run_dir / work_order_rel
    work_order_path.parent.mkdir(parents=True, exist_ok=True)
    work_order_path.write_text(work_order_markdown, encoding="utf-8")
    stub_path = run_dir / stub_rel_path
    stub_path.parent.mkdir(parents=True, exist_ok=True)
    stub_path.write_text(
        render_stub_module(element_id=element_id, work_order=work_order_rel,
                           signatures=signatures),
        encoding="utf-8",
    )
    return record_stubbed_element(
        run_dir / ".pipeline",
        element_id=element_id,
        role=role,
        work_order=work_order_rel,
        stub_path=stub_rel_path,
    )


# ---------------------------------------------------------------------------
# Delivery-time verification (§3.5 criterion 3, enforced): a recorded stub
# whose surfaces are missing or mute becomes a demoting reason.
# ---------------------------------------------------------------------------


def verify_stub_surfaces(run_dir: Path, stubs: list[dict]) -> list[str]:
    """Check every recorded stub self-identifies. Returns problem strings
    (empty means every surface holds)."""
    run_dir = Path(run_dir)
    problems = []
    for rec in stubs or []:
        eid = rec.get("element_id", "?")
        wo_rel = str(rec.get("work_order") or "")
        if not wo_rel:
            problems.append(f"stub `{eid}` records no work-order pointer")
        else:
            wo = run_dir / wo_rel
            if not wo.is_file():
                problems.append(
                    f"stub `{eid}`: work order `{wo_rel}` is missing")
            else:
                first = (wo.read_text(encoding="utf-8").splitlines() or [""])[0]
                if _PARTIAL_MARKER.lower() not in first.lower():
                    problems.append(
                        f"stub `{eid}`: work order `{wo_rel}` does not say "
                        f"{_PARTIAL_MARKER} in its first line")
        stub_rel = rec.get("stub_path")
        if not stub_rel:
            continue
        sp = run_dir / str(stub_rel)
        if not sp.is_file():
            problems.append(f"stub `{eid}`: stub file `{stub_rel}` is missing")
            continue
        text = sp.read_text(encoding="utf-8")
        if _STUB_MARKER.lower() not in text.lower():
            problems.append(
                f"stub `{eid}`: stub file `{stub_rel}` does not say "
                f"{_STUB_MARKER}")
        if "NotImplementedError" not in text:
            problems.append(
                f"stub `{eid}`: stub file `{stub_rel}` does not raise — a "
                f"stub that can run silently is forbidden")
        if wo_rel and wo_rel not in text:
            problems.append(
                f"stub `{eid}`: stub file `{stub_rel}` does not point at "
                f"its work order")
    return problems
