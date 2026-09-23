"""Fuzzy passage locator for verbatim-quote retry findings (item 23 part 3).

When a verbatim-quote floor rejects an agent's quote, the retry finding
should carry the paper's own best-matching passage so the fixer copies it
instead of re-reading the whole paper — the re-read plus transcription
rehearsal is what burned the per-step output cap on the 2026-07-08
ACC2021_MPC_CBF run. Design (approved 2026-07-10, k=1 / same-block
explainer half / 0.5 floor):
the passage locator design note (internal, not shipped).

Two consumers, one locator:
- stage 1's equation quote floor (validate_paper_map.check_equation_quotes),
  via the driver's finding enricher on the validator retry loop;
- the explainer's fabricated-quote check (validate_method_explanations),
  which attaches a `candidate_passage` field to the finding.

Honesty guardrails: every candidate is a raw byte slice of the paper, so
it passes the floor by construction — which also means a WRONG-but-real
passage would pass while being semantically wrong. Candidates are
therefore advisory (the carrier wording makes adoption conditional), and
anything scoring below `min_score` is dropped entirely: a confidently
wrong passage is worse than none.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from validate_params_provenance import _tok_match  # noqa: E402

# Matching tolerates Marker artifacts the same way the provenance matcher
# does: emphasis runs and backslashes are stripped inside tokens, so a
# JSON-corrupted quote (a `\t` that became a real tab, leaving "heta")
# still shares enough tokens with the paper to locate its passage.
_RAW_TOKEN_RE = re.compile(r"[A-Za-z0-9.%\\*_`]+")


def _tokens_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """The provenance matcher's token family, with character offsets kept
    so a matched window maps back to raw paper bytes."""
    out: list[tuple[str, int, int]] = []
    for m in _RAW_TOKEN_RE.finditer(text):
        cleaned = re.sub(r"[*_`\\]", "", m.group(0)).strip(".").lower()
        if cleaned and (len(cleaned) >= 3 or any(c.isdigit() for c in cleaned)):
            out.append((cleaned, m.start(), m.end()))
    return out


def _block_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Expand [start, end) to the enclosing blank-line-delimited block(s).
    Marker renders display math as its own block, so this returns whole
    equations, delimiters included."""
    bs = text.rfind("\n\n", 0, start)
    bs = bs + 2 if bs != -1 else 0
    be = text.find("\n\n", end)
    be = be if be != -1 else len(text)
    return bs, be


def _adjacent_math_block(text: str, bs: int, be: int) -> tuple[int, int] | None:
    """The nearest adjacent block containing display math, or None.

    The one prototype miss (ACC eq-barrier-quartic): the quote's prose
    matched a block that sits NEXT to the equation, separated from it by
    an interleaved figure caption, so the expanded block had no `$$`.
    One block before / one block after covers that shape."""
    prev_end = bs - 2
    if prev_end > 0:
        pbs, pbe = _block_bounds(text, prev_end - 1, prev_end)
        if "$$" in text[pbs:pbe]:
            return pbs, pbe
    next_start = be + 2
    if next_start < len(text):
        nbs, nbe = _block_bounds(text, next_start, next_start + 1)
        if "$$" in text[nbs:nbe]:
            return nbs, nbe
    return None


def locate_passages(
    quote: str,
    paper_text: str,
    *,
    element_type: str | None = None,
    k: int = 1,
    max_passage_chars: int = 1500,
    min_score: float = 0.5,
) -> list[dict]:
    """Top-k candidate raw passages for a rejected quote.

    Returns [{"score": float, "passage": str}, ...] best-first, where
    `passage` is a verbatim slice of `paper_text`. Empty list when nothing
    clears `min_score` — the caller then leaves the finding unchanged.

    Scoring is windowed token overlap (fraction of the quote's tokens
    stem-lite-matched inside a sliding window of paper tokens), the same
    family the provenance matcher uses, prototype-validated on the real
    ACC halt artifacts: 11 of 11 rejected quotes located, top-1 scores
    0.71–1.0, first-to-second gaps 0.14+. For `element_type="equation"`,
    a matched block without display math pulls in the nearest adjacent
    `$$` block (the stitched-quote shape)."""
    q_tokens = [t for t, _, _ in _tokens_with_offsets(quote)]
    if not q_tokens or not paper_text:
        return []
    p_tokens = _tokens_with_offsets(paper_text)
    if not p_tokens:
        return []

    win = max(20, int(1.5 * len(q_tokens)))
    stride = max(1, win // 3)
    scored: list[tuple[float, int, int]] = []
    for start in range(0, max(1, len(p_tokens) - win + stride), stride):
        window = p_tokens[start:start + win]
        wtoks = [w[0] for w in window]
        hits = sum(1 for q in q_tokens if any(_tok_match(q, w) for w in wtoks))
        score = hits / len(q_tokens)
        if score >= min_score:
            scored.append((score, window[0][1], window[-1][2]))
    scored.sort(key=lambda s: (-s[0], s[1]))

    candidates: list[dict] = []
    taken: list[tuple[int, int]] = []
    for score, cs, ce in scored:
        if any(not (ce <= a or cs >= b) for a, b in taken):
            continue  # overlaps a better window already taken
        bs, be = _block_bounds(paper_text, cs, ce)
        span_parts = [(bs, be)]
        if element_type == "equation" and "$$" not in paper_text[bs:be]:
            adjacent = _adjacent_math_block(paper_text, bs, be)
            if adjacent is not None:
                span_parts.append(adjacent)
                span_parts.sort()
        passage = "\n\n".join(paper_text[s:e] for s, e in span_parts)
        if len(passage) > max_passage_chars and element_type == "equation":
            # Tight budget (the SRL 2026-07-21 starvation: 8 of 15
            # failing equations lost their candidates to the block-size
            # cap, so per-element budgets shrank): the math itself is
            # what the fixer must copy, so prefer the $$-delimited
            # blocks alone over prose context before truncating.
            math_only = "\n\n".join(
                m.group(0)
                for s, e in span_parts
                for m in re.finditer(r"\$\$.*?\$\$", paper_text[s:e],
                                     re.DOTALL))
            if math_only:
                passage = math_only
        passage = passage[:max_passage_chars]
        candidates.append({"score": round(score, 3), "passage": passage})
        taken.append((bs, be))
        if len(candidates) >= k:
            break
    return candidates
