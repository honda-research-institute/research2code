"""Explainer dispatch builder + sidecar merge (slice 2.1, redesigned).

The first explainer dispatch failed in an instructive way: given file
PATHS, the agent read nothing and spent 45 minutes inventing a full
sidecar from the element ids alone. The redesign is structural, not
exhortative:

- **Inline, don't point.** Every dispatch prompt carries each element's
  paper statement (source_text), decomposition description and
  pseudocode verbatim, plus the paper's front matter (title/abstract
  region) once for novelty positioning. The ground truth is in-context;
  fabrication-from-name has nothing to feed on.
- **Chunk the work.** A few elements per dispatch keeps each turn at
  minutes on a shared endpoint, and a dead turn loses one part, not the
  whole layer.
- **Merge then validate.** Parts are merged deterministically and the
  MERGED sidecar goes through validate_method_explanations — coverage is
  judged on the whole, so a silently lost part is a hard error.

Usage (scratch experiments now, driver wiring in the idle window):
    python3 scripts/explainer_dispatch.py --run-dir r2c_runs/<slug> --show
    python3 scripts/explainer_dispatch.py --merge part*.json --output out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from generate_method_md import _explained_equations

SIDECAR_SCHEMA_VERSION = "1.0.0"
# 1, was 4 until 2026-08-24 (same decision as the per-element test
# dispatch): Qwen 3.8 explanation turns run 5-9 minutes PER EQUATION, so
# a four-equation part cannot fit the 900s dispatch budget — three A8
# rolls timed out mid-part and METHOD.md shipped pending markers. One
# equation per part fits the budget, and the stage's existing per-part
# resume then re-attempts exactly the missing equations.
DEFAULT_CHUNK_SIZE = 1
FRONT_MATTER_CHARS = 4000

_PROMPT_HEADER = """\
Write the explanation sidecar part for the elements below. Everything \
you need is IN THIS PROMPT: each element's verbatim paper statement, \
its decomposition description, and its pseudocode, plus the paper's \
front matter for the authors' own positioning. Ground every claim in \
this inlined material. The full paper is at {paper_path} if the front \
matter leaves a novelty claim genuinely ambiguous; do not re-derive \
what is already inlined.

Write EXACTLY ONE file at: {output_path}

It must contain explanations for exactly these element ids, no others:
{id_list}

Schema (schema_version "{schema_version}"):
{{"schema_version": "{schema_version}",
 "explanations": {{"<element_id>": {{"what": ..., "why_novel": ..., "intuition": ...}}}}}}
"""


def _paper_front(paper_text: str, max_chars: int = FRONT_MATTER_CHARS) -> str:
    """The title/abstract/intro region — cut at a paragraph boundary."""
    text = (paper_text or "").strip()
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n\n", 0, max_chars)
    if cut < max_chars // 2:
        cut = max_chars
    return text[:cut].rstrip()


def chunk_elements(elements: list[dict],
                   chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[list[dict]]:
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    return [elements[i:i + chunk_size]
            for i in range(0, len(elements), chunk_size)]


def _element_block(element: dict) -> str:
    parts = [
        f"--- element: {element.get('id', '?')} ---",
        f"name: {element.get('name', '?')}",
        f"paper location: {element.get('section', '?')}",
        f"role: {element.get('code_role', '?')}",
        "the paper states (verbatim):",
        str(element.get("source_text") or "(no source_text recorded)").strip(),
        "decomposition description:",
        str(element.get("description") or "(none)").strip(),
    ]
    pseudo = element.get("pseudocode")
    if pseudo:
        parts += ["pseudocode:", str(pseudo).strip()]
    deps = element.get("dependencies") or []
    if deps:
        parts.append(f"builds on: {', '.join(deps)}")
    return "\n".join(parts)


def build_dispatches(paper_map: dict, paper_text: str, *,
                     paper_path: str,
                     output_dir: str,
                     chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[dict]:
    """One dispatch dict per chunk of key equations.

    Returns [{"part", "element_ids", "output_path", "prompt"}, ...] —
    the driver (or a scratch runner) sends each prompt to the explainer
    agent and collects the part files for merge_sidecar_parts.
    """
    chunks = chunk_elements(_explained_equations(paper_map), chunk_size)
    front = _paper_front(paper_text)
    title = paper_map.get("title", "")
    dispatches = []
    for n, chunk in enumerate(chunks, start=1):
        ids = [e.get("id", "?") for e in chunk]
        output_path = f"{output_dir}/method_explanations.part-{n}.json"
        prompt = "\n".join([
            _PROMPT_HEADER.format(
                paper_path=paper_path,
                output_path=output_path,
                id_list="\n".join(f"- {i}" for i in ids),
                schema_version=SIDECAR_SCHEMA_VERSION),
            f"=== paper front matter: {title} ===",
            front,
            "",
            "=== elements to explain ===",
            "\n\n".join(_element_block(e) for e in chunk),
        ])
        dispatches.append({
            "part": n,
            "element_ids": ids,
            "output_path": output_path,
            "prompt": prompt,
        })
    return dispatches


_REJECTION_BLOCK_HEADER = """\
=== previous attempt rejected ===
A previous attempt at this part was rejected by the validator. Each note
below names the entry and its specific defect (with a numeric
counterexample where one exists). Rewrite the named entries so the
defect is gone — fix the explanation or drop the refuted claim; do not
restate it in different words. Entries not named below were fine."""

# Sized so a fabricated-quote note can carry its candidate paper passage
# (item 23 part 3: up to 600 chars of passage plus the message and the
# conditional-use wording) without truncating the passage mid-quote.
_MAX_NOTE_CHARS = 1400


def augment_prompt_with_rejections(prompt: str, notes: list[str]) -> str:
    """Append validator rejection notes to a retry prompt.

    The stage-1x retry is informed, not a blind re-roll: a refuted math
    claim's counterexample and a fabricated quote's defect both ride the
    retry prompt, so the explainer can correct the specific failure
    instead of probabilistically repeating it. No notes → the prompt is
    returned unchanged (the initial dispatch and missing-part retries)."""
    if not notes:
        return prompt
    return "\n".join([prompt, "", _REJECTION_BLOCK_HEADER]
                     + [f"- {n[:_MAX_NOTE_CHARS]}" for n in notes])


def reconcile_landed_parts(dispatches: list[dict]) -> list[Path]:
    """Delete landed part files whose element set disagrees with the plan.

    Part files are numbered positionally, so a chunk-size change between a
    run and its resume renumbers every part: a part-2 landed under the
    four-equation layout carries eq 5-8 where the current plan's part-2
    means one equation. Left in place, the stale file is skipped by the
    missing-part dispatcher and then collides with freshly dispatched
    parts at merge time (duplicate element id — the 2026-08 A8 rolls).

    A landed part survives only if its explanation keys are exactly the
    plan's element ids for its part number; a part file with no planned
    counterpart is stale by definition. Unparseable files are left alone —
    the merge's existing skip/requeue machinery owns those. Returns the
    deleted paths so the caller can log the reconciliation.
    """
    if not dispatches:
        return []
    planned = {d["output_path"]: set(d["element_ids"]) for d in dispatches}
    part_dir = Path(dispatches[0]["output_path"]).parent
    deleted: list[Path] = []
    for path in sorted(part_dir.glob("method_explanations.part-*.json")):
        expected = planned.get(str(path))
        if expected is not None:
            try:
                part = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if set(part.get("explanations") or {}) == expected:
                continue
        path.unlink()
        deleted.append(path)
    return deleted


def merge_sidecar_parts(part_paths: list[Path]) -> tuple[dict, list[Path]]:
    """Merge part sidecars into one, resilient to bad parts.

    A missing or unparseable part is SKIPPED — its element-ids are simply
    absent from the merge — and returned in the skipped list, rather than
    aborting the whole merge. One malformed part (e.g. an unescaped inner quote
    breaking JSON) must not wipe every other part's explanations (audit
    2026-06-25 E1); the caller renders the skipped ids as pending and queues
    those parts for resume. A DUPLICATE element-id across parts is still a hard
    error — the disjoint-chunk invariant is a build_dispatches logic bug, not a
    transient parse failure.

    Returns (merged_sidecar, skipped_parts). Coverage (every required id
    present) is NOT checked here — validate_method_explanations owns that.
    """
    merged: dict = {}
    skipped: list[Path] = []
    for path in part_paths:
        if not path.is_file():
            skipped.append(path)
            continue
        try:
            part = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            skipped.append(path)
            continue
        for element_id, entry in (part.get("explanations") or {}).items():
            if element_id in merged:
                raise ValueError(
                    f"element {element_id} explained by two parts "
                    f"(second: {path}) — chunks must be disjoint")
            merged[element_id] = entry
    return ({"schema_version": SIDECAR_SCHEMA_VERSION,
             "explanations": merged}, skipped)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--show", action="store_true",
                        help="print the dispatch prompts and exit")
    parser.add_argument("--merge", nargs="+", type=Path, default=None,
                        help="merge part sidecars into --output")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.merge:
        if not args.output:
            parser.error("--merge requires --output")
        try:
            sidecar, skipped = merge_sidecar_parts(args.merge)
        except ValueError as e:
            print(f"FAIL: {e}", file=sys.stderr)
            return 1
        args.output.write_text(json.dumps(sidecar, indent=2) + "\n",
                               encoding="utf-8")
        msg = (f"wrote {args.output} "
               f"({len(sidecar['explanations'])} elements)")
        if skipped:
            msg += (f"; skipped {len(skipped)} unparseable/missing part(s): "
                    f"{', '.join(str(p) for p in skipped)}")
        print(msg)
        return 0

    if not args.run_dir:
        parser.error("--run-dir required unless merging")
    pipeline = args.run_dir / ".pipeline"
    paper_map = json.loads((pipeline / "paper_map.json").read_text())
    # Canonical home (what the probes read); top-level globs would match
    # the run README and silently feed the wrong "paper".
    paper_file = pipeline / "paper.md"
    paper_text = paper_file.read_text() if paper_file.is_file() else ""
    dispatches = build_dispatches(
        paper_map, paper_text,
        paper_path=str(paper_file) if paper_file.is_file()
        else "(paper not found)",
        output_dir=str(pipeline), chunk_size=args.chunk_size)
    for d in dispatches:
        print(f"=== part {d['part']} -> {d['output_path']} "
              f"({len(d['element_ids'])} elements) ===")
        if args.show:
            print(d["prompt"])
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
