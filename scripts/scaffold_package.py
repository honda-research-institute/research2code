"""Stage 2.a — package_scaffolder.

Produces the paradigm_fixed files under `<RUN_DIR>/` declared in the matched
taxonomy build plan's `package_manifest.files` block where `kind:
paradigm_fixed` and `produced_by: package_scaffolder`. Specifically (for
active_learning today):

  - `<RUN_DIR>/method/data.py`              from templates/method/data.py.template
  - `<RUN_DIR>/method/example_data/README.md` from templates/method/example_data/README.md.template
  - `<RUN_DIR>/method/README.md`            from templates/method/README.md.template

The script is deterministic — no LLM calls — and resolves the template directory
from the taxonomy node's `scaffold_hints.templates_dir`, then mirrors its tree
under `<RUN_DIR>/` (after stripping the `.template` suffix and rendering
`{{ slot }}` placeholders from the spec).

Slot rendering uses simple `{{ name }}` and `{{name}}` substitution — no Jinja2
dependency. Available slots are populated from `method_spec.json`:

  - paper_title       : spec.paper.title
  - paper_authors     : spec.paper.authors
  - method_name       : spec.core_method.name
  - method_summary    : spec.core_method.summary
  - run_slug          : the basename of the run directory (e.g., "deep-batch-active-learning")
  - paradigm_id       : spec.comparison.classification.id

Templates that have no slots are still passed through the renderer; missing
keys are left as literal `{{ name }}` and the script logs a warning.

Usage:

    python scripts/scaffold_package.py --spec <method_spec.json> --run-dir <output_dir>

Exit codes:
  0  files written (or already up to date)
  1  setup error (missing spec / templates dir / unparseable spec)
  2  template referenced an unknown slot
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import taxonomy  # noqa: E402


SLOT_PATTERN = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")

# Both Python docstring terminator spellings — pack-authored interface text
# containing either cannot sit inside a generated module's docstring.
_DOCSTRING_TERMINATORS = ('"""', "'''")


def _sanitize_interface_hint(value: str) -> str:
    """Make pack-authored interface text safe to interpolate inside a module
    docstring.

    The authoring floor (validate_paradigm_proposal.py) already hard-fails a
    hint carrying a docstring terminator; this is the deterministic second
    surface at the interpolation site (DomIndOnto 2026-07-21: a gap pack
    shipped its hint as a signature PLUS a full triple-quoted docstring, the
    embedded quotes closed data.py's module docstring early, and the file no
    longer parsed). Everything from the first terminator on is dropped —
    that is where the embedded docstring starts — keeping the signature. If
    the value LEADS with a terminator, the terminator sequences themselves
    are removed instead so real content survives. Terminator-free values
    (the committed and known-good provisional shapes) pass through
    unchanged. Idempotent: the result never contains a terminator.
    """
    positions = [idx for idx in (value.find(t) for t in _DOCSTRING_TERMINATORS)
                 if idx != -1]
    if not positions:
        return value
    print(
        "warning: pack interface hint carries a docstring terminator "
        "(\"\"\" or '''); sanitizing before docstring interpolation — the "
        "proposal should have failed the authoring floor",
        file=sys.stderr,
    )
    head = value[:min(positions)].strip()
    if head:
        return head
    for terminator in _DOCSTRING_TERMINATORS:
        value = value.replace(terminator, "")
    return value.strip()


def _render(text: str, slots: dict[str, str]) -> tuple[str, list[str]]:
    """Substitute `{{ name }}` placeholders with values from slots.

    Returns (rendered_text, missing_slot_names).
    """
    missing: list[str] = []

    def repl(match: re.Match) -> str:
        name = match.group(1)
        if name in slots:
            return slots[name]
        missing.append(name)
        return match.group(0)  # leave literal so the user sees it on inspection

    out = SLOT_PATTERN.sub(repl, text)
    return out, missing


def _build_slots(
    spec: dict,
    run_dir: Path,
    *,
    node=None,
    tax=None,
) -> dict[str, str]:
    """Pull slot values from the spec. Adds run_slug from the run_dir basename.

    For a provisional node (a run-local gap pack), the authored pack supplies
    additional slots for the neutral skeleton: the declared interface (the
    pack's pluggable component / interface hint, floor-validated to be real
    content and sanitized here for docstring interpolation), the family id,
    and the pack's one-line method statement. The
    slots are provisional-only by design: a committed template referencing
    one still exits 2 loudly (slot/pack drift stays visible)."""
    paper = spec.get("paper") or {}
    core = spec.get("core_method") or {}
    paradigm = (spec.get("comparison") or {}).get("classification") or {}
    slots = {
        "paper_title": str(paper.get("title", "")).strip(),
        "paper_authors": str(paper.get("authors", "")).strip(),
        "method_name": str(core.get("name", "")).strip(),
        "method_summary": str(core.get("summary", "")).strip(),
        "run_slug": run_dir.name,
        "paradigm_id": str(paradigm.get("id", "")).strip(),
    }
    if node is not None and getattr(node, "provisional", False):
        expertise = taxonomy.node_expertise(node, tax)
        hint = _sanitize_interface_hint(
            str(expertise.get("interface_hint") or "").strip())
        fingerprint = expertise.get("fingerprint") or {}
        pluggable = taxonomy.load_pluggable_component_contract(
            slots["paradigm_id"] or None, tax)
        signature = _sanitize_interface_hint(str(
            pluggable.get("signature_template") or hint
        ).strip())
        name = str(pluggable.get("name") or "").strip()
        if not name and "(" in hint:
            name = hint.split("(", 1)[0].strip()
        slots.update({
            "interface_signature": signature,
            "interface_name": name,
            "method_statement": str(fingerprint.get("what_it_is") or "").strip(),
            "family_id": str(
                getattr(node, "taxonomy_id", getattr(node, "id", ""))
                or slots["paradigm_id"]
            ).strip(),
        })
    return slots


def scaffold(spec_path: Path, run_dir: Path, repo_root: Path) -> int:
    """Render the paradigm's templates into run_dir. Return 0 on success."""
    if not spec_path.is_file():
        print(f"error: spec not found: {spec_path}", file=sys.stderr)
        return 1

    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"error: spec is not valid JSON ({e})", file=sys.stderr)
        return 1

    comparison_paradigm = (spec.get("comparison") or {}).get("classification") or {}
    paradigm_id = comparison_paradigm.get("id")
    # Gap-path serves() overlay: a gap paper's provisional classification is
    # served by the run's own installed pack, keyed on the run dir (SRL
    # 2026-07-04: this exact call was the stage-2a wall). Non-gap runs have
    # no pack dir and resolve the committed view byte-identically.
    tax = taxonomy.load_taxonomy(
        repo_root, provisional_packs_dir=taxonomy.run_overlay_dir(run_dir))
    node = taxonomy.serves(paradigm_id, tax) if paradigm_id else None

    if node is None:
        print(f"error: no taxonomy node serves paradigm {paradigm_id!r}", file=sys.stderr)
        return 1
    templates_dir = taxonomy.resolve_templates_dir(
        paradigm_id=paradigm_id, repo_root=repo_root, taxonomy=tax)
    if templates_dir is None:
        print(
            f"error: no `templates/` directory for taxonomy-served paradigm "
            f"{paradigm_id!r} (set scaffold_hints.templates_dir on its node)",
            file=sys.stderr,
        )
        return 1

    slots = _build_slots(spec, run_dir, node=node, tax=tax)
    print(f"templates dir: {templates_dir}")
    print(f"slots: {slots}")

    run_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    all_missing: dict[Path, list[str]] = {}

    for template_path in sorted(templates_dir.rglob("*.template")):
        rel = template_path.relative_to(templates_dir)
        target_rel = rel.with_suffix("")  # strip `.template`
        target = run_dir / target_rel
        target.parent.mkdir(parents=True, exist_ok=True)

        rendered, missing = _render(template_path.read_text(encoding="utf-8"), slots)
        if missing:
            all_missing[template_path] = missing

        target.write_text(rendered, encoding="utf-8")
        written.append(target)

    # Report
    print(f"\nwrote {len(written)} file(s):")
    for w in written:
        print(f"  {w.relative_to(run_dir.parent)}")

    if all_missing:
        print("\nWARNING: some templates referenced unknown slots:", file=sys.stderr)
        for tp, names in all_missing.items():
            print(f"  {tp.relative_to(templates_dir)}: {sorted(set(names))}", file=sys.stderr)
        return 2

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True, help="path to method_spec.json")
    parser.add_argument("--run-dir", type=Path, required=True, help="output directory (typically <REPO>/r2c_runs/<slug>)")
    parser.add_argument("--repo-root", type=Path, default=ROOT, help="repo root for resolving the taxonomy SSOT (default: this repo)")
    args = parser.parse_args()

    return scaffold(args.spec, args.run_dir, args.repo_root)


if __name__ == "__main__":
    sys.exit(main())
