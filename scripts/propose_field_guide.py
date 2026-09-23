#!/usr/bin/env python3
"""Create a candidate taxonomy-pack proposal packet from a gap report.

This is maintainer scaffolding only. It reads a Stage 1
`.pipeline/paradigm_gap_report.json`, creates a proposal packet under the run's
`.pipeline/paradigm_proposals/` directory, and stops. Nothing is copied into
the SSOT until `scripts/apply_paradigm_proposal.py` is run explicitly after
review.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from schemas.paradigm_gap import GapDecision, ParadigmGapReport  # noqa: E402
from schemas.paradigm_proposal import (  # noqa: E402
    ParadigmProposal,
    ProposalEvidence,
)
from scripts import taxonomy  # noqa: E402


PROPOSAL_ROOT = Path(".pipeline/paradigm_proposals")


class ProposalScaffoldError(RuntimeError):
    """Raised when a proposal packet cannot be scaffolded safely."""


def create_proposal_packet(
    run_dir: Path,
    repo_root: Path = ROOT,
    *,
    proposal_id: str | None = None,
    gap_report_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    run_dir = run_dir.resolve()
    repo_root = repo_root.resolve()
    gap_path = gap_report_path or run_dir / ".pipeline" / "paradigm_gap_report.json"
    report = _load_gap_report(gap_path)
    if report.decision not in {
        GapDecision.new_subparadigm_needed,
        GapDecision.new_top_level_needed,
    }:
        raise ProposalScaffoldError(
            "gap report decision does not request a new pack: "
            f"{report.decision.value}"
        )

    tax = taxonomy.load_taxonomy(repo_root)
    proposal = _proposal_from_gap_report(report, tax, run_dir, repo_root)
    if proposal_id is not None:
        proposal = proposal.model_copy(update={"proposal_id": _clean_proposal_id(proposal_id)})

    proposal_dir = run_dir / PROPOSAL_ROOT / proposal.proposal_id
    if proposal_dir.exists() and not overwrite:
        raise ProposalScaffoldError(f"proposal already exists: {proposal_dir}")
    proposal_dir.mkdir(parents=True, exist_ok=True)

    (proposal_dir / "proposal.json").write_text(
        json.dumps(proposal.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    (proposal_dir / "pack.yaml").write_text(
        yaml.safe_dump(proposal.pack, sort_keys=False),
        encoding="utf-8",
    )
    (proposal_dir / "proposal.md").write_text(
        _proposal_markdown(proposal, report),
        encoding="utf-8",
    )
    (proposal_dir / "coupling_report.md").write_text(
        _coupling_report(proposal),
        encoding="utf-8",
    )
    return proposal_dir


def _load_gap_report(path: Path) -> ParadigmGapReport:
    if not path.is_file():
        raise ProposalScaffoldError(f"gap report not found: {path}")
    try:
        return ParadigmGapReport.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except Exception as exc:  # noqa: BLE001
        raise ProposalScaffoldError(f"gap report invalid: {exc}") from exc


def _proposal_from_gap_report(
    report: ParadigmGapReport,
    tax: taxonomy.Taxonomy,
    run_dir: Path,
    repo_root: Path,
) -> ParadigmProposal:
    assert report.proposed_new_paradigm_id is not None
    proposal_id = _default_proposal_id(report.proposed_new_paradigm_id)
    source_slug = report.paper_slug or run_dir.name
    pack = _pack_scaffold(report, tax)
    return ParadigmProposal(
        proposal_id=proposal_id,
        source_paper_slug=source_slug,
        decision=report.decision.value,
        target_paradigm_id=report.proposed_new_paradigm_id,
        target_taxonomy_id=pack.get("taxonomy_id"),
        extends=report.proposed_parent_paradigm,
        title=_title_for_paradigm(report.proposed_new_paradigm_id),
        scope_summary=report.paper_paradigm_summary,
        pack=pack,
        evidence=[
            ProposalEvidence(
                paper_section=item.paper_section,
                quote_or_observation=item.quote_or_observation,
                relevance=item.relevance,
            )
            for item in report.paper_evidence
        ],
        coupling_warnings=_initial_coupling_warnings(report),
        validation_command=(
            "python3 scripts/validate_paradigm_proposal.py "
            f"{run_dir / PROPOSAL_ROOT / proposal_id}"
        ),
    )


def _pack_scaffold(report: ParadigmGapReport, tax: taxonomy.Taxonomy) -> dict:
    assert report.proposed_new_paradigm_id is not None
    parent_pack = (
        taxonomy.load_pack(report.proposed_parent_paradigm, tax.repo_root)
        if report.proposed_parent_paradigm else None
    )
    parent_hint = (parent_pack or {}).get("scaffold_hints", {}).get("interface_hint")
    taxonomy_id = _default_taxonomy_id(report, tax)
    return {
        "schema_version": "1.0",
        "status": "provisional",
        "legacy_paradigm": report.proposed_new_paradigm_id,
        "extends": report.proposed_parent_paradigm,
        "taxonomy_id": taxonomy_id,
        "fingerprint": {
            "what_it_is": report.paper_paradigm_summary,
            "not_this": [],
        },
        "scaffold_hints": {
            "interface_hint": parent_hint or "TODO: describe the pluggable interface this paper needs.",
        },
        "priors": {
            "source_paper": report.paper_slug,
            "evidence": [item.model_dump(mode="json") for item in report.paper_evidence],
        },
        "semantic_checks": [],
        "smoke_bugs": [],
        "open_questions": [
            "Maintainer must decide whether this provisional pack earns a canonical taxonomy node.",
        ],
    }


def _default_taxonomy_id(report: ParadigmGapReport, tax: taxonomy.Taxonomy) -> str | None:
    proposed = report.proposed_new_paradigm_id
    if not proposed:
        return None
    if report.decision == GapDecision.new_subparadigm_needed and report.proposed_parent_paradigm:
        parent = tax.node_for_legacy(report.proposed_parent_paradigm)
        parent_tax_id = getattr(parent, "taxonomy_id", getattr(parent, "id", None))
        if parent_tax_id:
            return f"{parent_tax_id}/{proposed.rsplit('/', 1)[-1]}"
    return f"PROVISIONAL/{proposed}"


def _initial_coupling_warnings(report: ParadigmGapReport) -> list[str]:
    warnings = [
        "Review pipeline coupling before promotion: analyzer classification, package scaffolding, validators, stage reviewer, notebook generator, and smoke diagnostician.",
        "Keep the pack small: put enforceable behavior in semantic_checks/smoke_bugs only when a consumer can actually use it.",
    ]
    if report.decision == GapDecision.new_top_level_needed:
        warnings.append(
            "New top-level packs need a full method-family home and build-plan design before canonical promotion."
        )
    else:
        warnings.append(
            "Sub-paradigm packs should usually inherit parent contracts and only override documented differences."
        )
    return warnings


def _proposal_markdown(proposal: ParadigmProposal, report: ParadigmGapReport) -> str:
    lines = [
        f"# Pack Proposal: {proposal.title}",
        "",
        f"- Proposal id: `{proposal.proposal_id}`",
        f"- Decision: `{proposal.decision}`",
        f"- Source paper: `{proposal.source_paper_slug}`",
        f"- Target paradigm id: `{proposal.target_paradigm_id}`",
        f"- Target taxonomy id: `{proposal.target_taxonomy_id or '-'}`",
        f"- Extends: `{proposal.extends or '-'}`",
        "",
        "## Scope",
        "",
        proposal.scope_summary,
        "",
        "## Evidence",
        "",
    ]
    for item in proposal.evidence:
        lines.extend(
            [
                f"- `{item.paper_section}`: {item.quote_or_observation}",
                f"  - Relevance: {item.relevance}",
            ]
        )
    lines.extend(
        [
            "",
            "## Source Gap Decision",
            "",
            f"- Confidence: {report.confidence:.2f}",
            f"- Recommended next action: {report.recommended_next_action}",
            "",
            "## Required Next Steps",
            "",
            "1. Fill `pack.yaml` from the source paper and relevant parent/sibling taxonomy packs.",
            "2. Keep authoring in this proposal directory until validation passes.",
            "3. Run the validation command listed in `proposal.json`.",
            "4. Review `coupling_report.md` and update any required consumer surface.",
            "5. Promote with `scripts/apply_paradigm_proposal.py` only after review.",
            "",
        ]
    )
    return "\n".join(lines)


def _coupling_report(proposal: ParadigmProposal) -> str:
    lines = [
        "# Coupling Report",
        "",
        "Review this before promoting the proposal into `docs/ssot/taxonomies.yaml`.",
        "",
        "## Candidate",
        "",
        f"- Target paradigm id: `{proposal.target_paradigm_id}`",
        f"- Target taxonomy id: `{proposal.target_taxonomy_id or '-'}`",
        f"- Extends: `{proposal.extends or '-'}`",
        "",
        "## Consumer Checklist",
        "",
        "- Analyzer: classification scope and `pluggable_component` contract are clear.",
        "- Scaffolder/package validators: build-plan and templates implications are explicit.",
        "- Architecture validator: arch-contract requirements are explicit if this pack will codegen.",
        "- Stage reviewer: `semantic_checks` adds concrete checks instead of aspirational prose.",
        "- Notebook generator: `notebook_layout.sections` is present if the pack will codegen.",
        "- Smoke diagnostician: `smoke_bugs` is present when smoke-bug coverage is claimed.",
        "",
        "## Initial Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in proposal.coupling_warnings)
    lines.append("")
    return "\n".join(lines)

def _title_for_paradigm(paradigm_id: str) -> str:
    leaf = paradigm_id.rsplit("/", maxsplit=1)[-1]
    return " ".join(part.capitalize() for part in re.split(r"[_-]+", leaf))


def _clean_proposal_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.replace("/", "-")).strip("-")
    if not cleaned:
        cleaned = "proposal"
    return cleaned.lower()


def _default_proposal_id(value: str) -> str:
    cleaned = _clean_proposal_id(value)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    if cleaned.startswith(stamp):
        return cleaned
    return f"{stamp}-{cleaned}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--gap-report", type=Path)
    parser.add_argument("--proposal-id")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        proposal_dir = create_proposal_packet(
            args.run_dir,
            args.repo_root,
            proposal_id=args.proposal_id,
            gap_report_path=args.gap_report,
            overwrite=args.overwrite,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"proposal scaffold FAILED: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"proposal_dir": str(proposal_dir)}, indent=2))
    else:
        print(f"proposal scaffold written: {proposal_dir}")
        print(f"validate with: python3 scripts/validate_paradigm_proposal.py {proposal_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
