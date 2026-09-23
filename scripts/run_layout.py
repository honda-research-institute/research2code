"""Canonical researcher-facing layout of a run directory.

Consolidated 2026-07-07 (maintainer-approved "four deliverables" layout): the
top level of `r2c_runs/<slug>/` holds exactly the things a researcher
reads or runs —

    REPORT.md          the front door: label, findings, claims
    METHOD.md          the paper's method explained
    notebook.ipynb     the tutorial
    requirements.txt   read verbatim by the notebook's install cell
    method/            the importable package (its own README.md inside)
    details/           supporting evidence, linked from REPORT.md
    .pipeline/         driver internals, never researcher-facing

Every writer and reader of these paths imports the constants below so
the layout has one owner. Readers that must also open runs produced
before the consolidation (curated example archives, a researcher's old
local runs in the fleet view) resolve through LEGACY_ALIASES via
resolve_existing().
"""

from __future__ import annotations

from pathlib import Path

DETAILS_DIRNAME = "details"

# Top-level deliverables (unchanged by the consolidation).
REPORT_MD = "REPORT.md"
METHOD_MD = "METHOD.md"
NOTEBOOK_IPYNB = "notebook.ipynb"
REQUIREMENTS_TXT = "requirements.txt"

# The package carries its own quick-start README.
PACKAGE_README = "method/README.md"

# Supporting evidence, consolidated under details/.
ASSUMPTIONS_MD = f"{DETAILS_DIRNAME}/assumptions.md"
DEFERRED_FINDINGS_MD = f"{DETAILS_DIRNAME}/deferred_findings.md"
KNOWN_ISSUES_MD = f"{DETAILS_DIRNAME}/KNOWN_ISSUES.md"
FINAL_MANIFEST_JSON = f"{DETAILS_DIRNAME}/final_manifest.json"

# Current rel path -> pre-consolidation locations, newest first.
LEGACY_ALIASES: dict[str, tuple[str, ...]] = {
    PACKAGE_README: ("README.md",),
    ASSUMPTIONS_MD: ("assumptions.md",),
    DEFERRED_FINDINGS_MD: ("deferred_findings.md", ".pipeline/deferred_findings.md"),
    KNOWN_ISSUES_MD: ("KNOWN_ISSUES.md",),
    FINAL_MANIFEST_JSON: ("final_manifest.json",),
}


def run_path(run_dir: Path, rel: str) -> Path:
    """The CURRENT-layout absolute path for writing. Creates details/
    (and any other parent) so append-mode writers can open immediately."""
    path = Path(run_dir) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_existing(run_dir: Path, rel: str) -> Path | None:
    """For READERS: the current-layout path if present, else the newest
    legacy alias that exists, else None. Never creates anything."""
    run_dir = Path(run_dir)
    candidates = (rel, *LEGACY_ALIASES.get(rel, ()))
    for candidate in candidates:
        path = run_dir / candidate
        if path.exists():
            return path
    return None
