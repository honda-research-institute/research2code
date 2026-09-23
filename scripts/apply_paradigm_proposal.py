#!/usr/bin/env python3
"""Promote a reviewed taxonomy-pack proposal into `docs/ssot/proposed_packs/`.

This command is maintainer tooling. It copies a validated proposal pack into
the source tree for review before a canonical `docs/ssot/taxonomies.yaml` edit.
It does not commit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from schemas.paradigm_proposal import ParadigmProposal  # noqa: E402
from scripts.validate_paradigm_proposal import validate_proposal  # noqa: E402


PROPOSED_PACKS_DIR = Path("docs/ssot/proposed_packs")


def apply_proposal(
    proposal_dir: Path,
    repo_root: Path = ROOT,
    *,
    update_existing: bool = False,
    refresh_generated: bool = True,
) -> list[str]:
    result = validate_proposal(
        proposal_dir,
        repo_root,
        allow_update_existing=update_existing,
    )
    if not result.valid or result.proposal is None:
        details = "\n".join(f"  - {error}" for error in result.errors)
        raise RuntimeError(f"proposal validation failed:\n{details}")
    proposal = result.proposal
    written: list[str] = []
    target_dir = repo_root / PROPOSED_PACKS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{proposal.proposal_id}.yaml"
    if target.exists() and not update_existing:
        raise RuntimeError(f"proposed pack already exists: {target.relative_to(repo_root)}")

    pack = yaml.safe_load((proposal_dir / "pack.yaml").read_text(encoding="utf-8")) or {}
    payload = {
        "proposal": {
            "proposal_id": proposal.proposal_id,
            "source_paper_slug": proposal.source_paper_slug,
            "source_gap_report": proposal.source_gap_report,
            "decision": proposal.decision,
            "target_paradigm_id": proposal.target_paradigm_id,
            "target_taxonomy_id": proposal.target_taxonomy_id,
            "extends": proposal.extends,
            "title": proposal.title,
        },
        "pack": pack,
    }
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    written.append(target.relative_to(repo_root).as_posix())

    return written


def _load_proposal(proposal_dir: Path) -> ParadigmProposal:
    return ParadigmProposal.model_validate(
        json.loads((proposal_dir / "proposal.json").read_text(encoding="utf-8"))
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("proposal_dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--update-existing", action="store_true")
    parser.add_argument("--no-refresh-generated", action="store_true")
    args = parser.parse_args(argv)

    try:
        proposal = _load_proposal(args.proposal_dir)
        written = apply_proposal(
            args.proposal_dir,
            args.repo_root,
            update_existing=args.update_existing,
            refresh_generated=not args.no_refresh_generated,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"apply FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"applied proposal {proposal.proposal_id} -> {PROPOSED_PACKS_DIR}/{proposal.proposal_id}.yaml")
    for path in written:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
