"""Resolve paths and prepare a fresh run workspace for a paper.

Stage 0 of the pipeline. Replaces the inline `mkdir + cp` block that lived in
the orchestrator's Step 0 prompt — per architecture principle 1 (stages are
deterministic scripts; the orchestrator is a thin sequencer).

Computes the canonical paths derived from a paper's filename, creates the run
directory (with its `<RUN_DIR>/.pipeline/` working dir), and copies the source
paper to the run-dir root so the input sits next to the deliverables it
produces (easy to match a PDF against its generated code, including across
iterations like our_method_v1.pdf / our_method_v2.pdf). For markdown inputs it
also copies the paper to `<PIPELINE_DIR>/paper.md`; PDF inputs are passed
through and Stage 0 parses `paper.md` from them.

Usage:
    python scripts/setup_pipeline_dirs.py <input> [--repo-root <path>]
    python scripts/setup_pipeline_dirs.py <input> --resolve-only [--repo-root <path>]

`<input>` may be:
  - a filename or slug ("deep-batch-active-learning", "BADGE.pdf") — resolved
    under <repo_root>/input_papers/, trying .pdf before .md;
  - an absolute or relative path to an existing .md or .pdf file.

Stdout: a single JSON object with the resolved paths and metadata. Unless
`--resolve-only` is passed, the same JSON is also written to
`<PIPELINE_DIR>/setup_result.json` for downstream stages to read directly.

Exit codes:
  0  setup completed (or was idempotent — paper.md already matches input)
  1  input not found, wrong type, or existing paper.md conflicts with input
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


SUPPORTED_SUFFIXES = {".md": "markdown", ".pdf": "pdf"}


def _resolve_input(raw: str, repo_root: Path) -> Path | None:
    """Resolve a user-supplied input string to an existing file path.

    Tries, in order:
      1. As an absolute or already-resolvable path.
      2. As a filename under <repo_root>/input_papers/.
      3. As a slug under <repo_root>/input_papers/, trying .pdf then .md.
    """
    candidate = Path(raw)
    if candidate.is_file():
        return candidate.resolve()

    if not candidate.is_absolute():
        rooted = (repo_root / raw).resolve()
        if rooted.is_file():
            return rooted

    papers_dir = repo_root / "input_papers"
    direct = papers_dir / raw
    if direct.is_file():
        return direct.resolve()

    if Path(raw).suffix == "":
        # Markdown-first (maintainer decision 2026-07-07, superseding 388b2d576's
        # PDF-first): a pre-parsed .md skips the Marker parse entirely —
        # faster, cheaper, and immune to the parse service being down.
        # Passing an explicit "<slug>.pdf" still forces the PDF when a
        # fresh parse is wanted.
        for suffix in (".md", ".pdf"):
            with_suffix = papers_dir / f"{raw}{suffix}"
            if with_suffix.is_file():
                return with_suffix.resolve()

    return None


def _list_input_papers(repo_root: Path) -> list[str]:
    papers_dir = repo_root / "input_papers"
    if not papers_dir.is_dir():
        return []
    return sorted(p.name for p in papers_dir.iterdir() if p.is_file())


def resolve_setup(raw_input: str, repo_root: Path) -> tuple[dict | None, str | None]:
    input_path = _resolve_input(raw_input, repo_root)

    if input_path is None:
        available = ", ".join(_list_input_papers(repo_root)) or "(none)"
        return None, (
            f"error: could not resolve input {raw_input!r}. "
            f"Looked under {repo_root / 'input_papers'} and as a direct path. "
            f"Available papers: {available}"
        )

    suffix = input_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return None, f"error: input must be .md or .pdf (got {suffix!r}: {input_path})"
    input_kind = SUPPORTED_SUFFIXES[suffix]

    slug = input_path.stem
    run_dir = repo_root / "r2c_runs" / slug
    pipeline_dir = run_dir / ".pipeline"
    paper_md_path = pipeline_dir / "paper.md"

    return {
        "repo_root": str(repo_root),
        "input_path": str(input_path),
        "input_kind": input_kind,
        "slug": slug,
        "run_dir": str(run_dir),
        "pipeline_dir": str(pipeline_dir),
        "paper_md_path": str(paper_md_path),
        "paper_md_present": paper_md_path.is_file(),
    }, None


def materialize_setup(result: dict) -> tuple[dict | None, str | None]:
    input_path = Path(result["input_path"])
    run_dir = Path(result["run_dir"])
    pipeline_dir = Path(result["pipeline_dir"])
    paper_md_path = Path(result["paper_md_path"])

    pipeline_dir.mkdir(parents=True, exist_ok=True)

    # Keep a copy of the source paper with the run (provenance across
    # iterations like our_method_v1.pdf / our_method_v2.pdf; original filename
    # preserved), plus its sibling in the other supported format when one
    # sits next to it (researcher request 2026-07-07: an .md-driven run
    # should still carry the .pdf it was exported from, and vice versa).
    # Copies live under .pipeline/ — the run-dir top level is reserved for
    # the four deliverables (see run_layout.py). Idempotent: skips the
    # rewrite when an identical copy is already there, refreshes a stale
    # one, never copies onto itself.
    sources = [input_path]
    other_suffix = ".pdf" if input_path.suffix.lower() == ".md" else ".md"
    sibling = input_path.with_suffix(other_suffix)
    if sibling.is_file():
        sources.append(sibling)
    for src in sources:
        dest = pipeline_dir / src.name
        if dest.resolve() != src.resolve() and (
            not dest.exists() or dest.read_bytes() != src.read_bytes()
        ):
            shutil.copy2(src, dest)
    result["source_copy_path"] = str(pipeline_dir / input_path.name)

    if result["input_kind"] == "markdown":
        if paper_md_path.exists():
            if paper_md_path.read_bytes() != input_path.read_bytes():
                return None, (
                    f"error: {paper_md_path} already exists with different content "
                    f"than the input. Refusing to overwrite — delete it manually if "
                    f"you want to re-run setup against a different source."
                )
        else:
            shutil.copy2(input_path, paper_md_path)
        result["paper_md_present"] = True
    else:
        result["paper_md_present"] = paper_md_path.exists()

    setup_result_path = pipeline_dir / "setup_result.json"
    setup_result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", help="Paper filename, slug, or path.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repo root for resolving inputs and outputs (default: cwd).",
    )
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Only resolve paths and print setup metadata; do not create or write run files.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    result, error = resolve_setup(args.input, repo_root)
    if error is not None:
        print(error, file=sys.stderr)
        return 1
    assert result is not None
    if not args.resolve_only:
        result, error = materialize_setup(result)
        if error is not None:
            print(error, file=sys.stderr)
            return 1
        assert result is not None

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
