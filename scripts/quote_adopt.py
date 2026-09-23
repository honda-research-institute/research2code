"""Accept-the-candidate adoption for verbatim-quote failures (R2C-030).

Design maintainer-approved 2026-07-28 (task record
the R2C-030 work item (internal, not shipped)).

The stage 1 fix loop has no deterministic path for a quote defect the
producer cannot perceive: a one-character transcription slip (a closing
\\big) where the paper writes \\big]) makes the producer return the same
bytes every retry, the fixation detector fires, and the run halts even
though the paper's own passage is known at near-perfect similarity. Both
recorded cases returned the identical error with the exact bytes in the
fix prompt: SRL on one apostrophe (2026-07-27, that class is folded now)
and bayesian-active-learning on one closing bracket at candidate
similarity 1.0 (2026-07-28).

Render-equivalent differences already resolve deterministically
(quote_reanchor); this module extends the same principle — the quote is
a locator, the source's bytes are the authority — from render-equivalent
matches to NEAR-matches, under a strict uniqueness rule so an adoption
can never silently retarget which passage an element points at.

Adoption rule (every condition, or no adoption):
- the best candidate's similarity on the render-equivalence-folded
  sequences reaches ADOPTION_FLOOR;
- exactly one candidate at or above the floor, and it sits inside the
  element's own anchored region when that region resolves;
- the best other candidate anywhere in the paper sits below the floor
  by ADOPTION_MARGIN — the retargeting protection, so two similar
  equations in one section block adoption and keep the honest halt;
- when the element's region cannot be resolved, the same uniqueness and
  margin conditions apply over the whole paper, which is strictly
  stronger on the ambiguity axis.

Every adoption is the caller's to record: a structured logged assumption
carrying the element id, the producer's original quote, the adopted
paper bytes, the similarity score, and the runner-up score as uniqueness
evidence. Ambiguity of any kind keeps the honest halt path.

Floor calibration (2026-07-28, real artifacts): the recorded bracket
slip scores 0.9978 and adopts; a paraphrase of the same equation scores
0.15; twin equations differing in one subscript both score 0.97 and are
refused by uniqueness. See tests/test_quote_adoption.py.
"""

from __future__ import annotations

import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from quote_reanchor import (MIN_FOLDED_LEN,  # noqa: E402
                            _expand_math_delimiters, _fold_with_offsets)

# Floor on the folded-sequence similarity ratio (see calibration above).
ADOPTION_FLOOR = 0.95
# The best candidate other than the winner must sit at least this far
# below the floor.
ADOPTION_MARGIN = 0.05

# Discovery scan: windows scoring below this on the cheap prefilters are
# never refined. Well under the floor so near-misses still surface as
# runner-up evidence for the margin check.
_PREFILTER_FLOOR = 0.70


def folded_similarity(a: str, b: str) -> float:
    """Similarity ratio between the render-equivalence folds of a and b."""
    fa, _ = _fold_with_offsets(a)
    fb, _ = _fold_with_offsets(b)
    if not fa or not fb:
        return 0.0
    return SequenceMatcher(None, fa, fb, autojunk=False).ratio()


# ---------------------------------------------------------------------------
# Math transliteration (2026-08-06, the forecasting stage-1 halt)
#
# A producer copying a paper sentence that contains inline math often rewrites
# the math into plain text and leaves every word of prose alone: the paper's
# `$\mathcal{N}_i^t = (y_i^{t-P+1}, \dots, y_i^t)$` comes back as
# `N_i^t = (y_i^{t-P+1}, ..., y_i^t)`. The render-equivalence fold DELIBERATELY
# does not fold LaTeX command spelling, because the exact re-anchor path needs
# twin equations differing in one symbol to stay distinguishable. So the fold
# stays as it is and adoption gains a SECOND scoring pass over both sides,
# under the same floor, margin, and uniqueness rules.
#
# Conservative on purpose. Only KNOWN transliteration pairs normalize, and an
# unrecognized command is left standing, so `\alpha` and `\beta` never collapse
# into each other. Measured on the real halt: the two transliterated quotes go
# from 0.922 and 0.927 to an exact 1.0, an otherwise-identical alpha/beta pair
# stays at 0.837 (nowhere near adopting), and the appendix table cell that
# should NOT adopt is unmoved at 0.941 with a tied runner-up, so uniqueness
# still refuses it.
# ---------------------------------------------------------------------------

# Wrappers whose braces render as their contents: \mathcal{N} reads as N.
_MATH_WRAPPER_RE = re.compile(
    r"\\(?:mathcal|mathbb|mathrm|mathbf|mathit|mathsf|boldsymbol"
    r"|text|textit|textbf|textrm|operatorname)\s*\{([^{}]*)\}")

# Commands with an unambiguous plain-text spelling. Anything not listed here
# is left alone: an unknown command is a symbol we cannot claim to know the
# transliteration of, and guessing would erode the twin-equation protection.
_MATH_LITERALS = (
    (re.compile(r"\\(?:dots|ldots|cdots|dotsc|dotsb)\b"), "..."),
    (re.compile(r"\\times\b"), "x"),
    (re.compile(r"\\cdot\b"), "."),
    (re.compile(r"\\in\b"), "in"),
    (re.compile(r"\\leq\b|\\le\b"), "<="),
    (re.compile(r"\\geq\b|\\ge\b"), ">="),
    (re.compile(r"\\neq\b|\\ne\b"), "!="),
    (re.compile(r"\\(?:left|right|,|;|:|!)"), ""),
    (re.compile(r"\$+"), ""),
)

# Named operators (2026-08-07, the forecasting stage-1 halt's second shape).
# These belong to the same "unambiguous plain-text spelling" family as the
# literals above, but they need no table: the spelling IS the command name, so
# dropping the backslash PRESERVES the symbol instead of guessing at it. That
# is what keeps the twin-equation protection intact — `\cos` can only ever
# become `cos`, never collide with a different operator.
#
# Longest-first so `\arccos` and `\cosh` win over `\cos`, and `\limsup` over
# `\lim`.
_MATH_OPERATOR_NAMES = (
    "arccos", "arcsin", "arctan", "argmax", "argmin", "arg", "cos", "cosh",
    "cot", "coth", "csc", "deg", "det", "dim", "exp", "gcd", "hom", "inf",
    "ker", "lg", "lim", "liminf", "limsup", "ln", "log", "max", "min", "Pr",
    "sec", "sin", "sinh", "sup", "tan", "tanh",
)
#
# The trailing boundary is "not a letter" rather than `\b`, because a subscript
# follows an operator constantly in real math (`\max_i`, `\inf_x`,
# `\limsup_n`). `_` is a word character, so `\b` does NOT match between `p` and
# `_`, and every subscripted operator was silently left standing. Caught by the
# `\limsup_n` case. A letter still blocks the match, which is what keeps
# `\cosh` from being read as `\cos` plus a stray h.
_MATH_OPERATOR_RE = re.compile(
    r"\\(" + "|".join(sorted(_MATH_OPERATOR_NAMES, key=len, reverse=True))
    + r")(?![A-Za-z])"
)

# Fractions. The one STRUCTURAL rule here, and it earns its place for the same
# reason the others do: it is symbol-preserving. `\frac{A}{B}` becomes
# `(A) / (B)` with A and B carried through verbatim, so no two distinct
# equations can fold into each other. A producer retyping a fraction in plain
# text writes exactly this.
#
# `[^{}]*` matches only brace-free groups, so nested fractions unwind
# innermost-first over repeated passes, sharing the wrapper unwind's guard.
_MATH_FRAC_RE = re.compile(
    r"\\(?:frac|dfrac|tfrac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}")

# Runaway guard on the nested-wrapper unwind. Real papers nest two or three
# deep; a pathological input must not spin.
_MAX_UNWRAP_PASSES = 8


def transliterate_math(text: str) -> str:
    """The paper's inline math as a producer would retype it in plain text."""
    out = text or ""
    for _ in range(_MAX_UNWRAP_PASSES):
        unwrapped = _MATH_WRAPPER_RE.sub(r"\1", out)
        if unwrapped == out:
            break
        out = unwrapped
    for _ in range(_MAX_UNWRAP_PASSES):
        unfracked = _MATH_FRAC_RE.sub(r"(\1) / (\2)", out)
        if unfracked == out:
            break
        out = unfracked
    # Operators before the literal table: `\inf` must not be reached by the
    # literals' `\in` rule, and its own `\b` already prevents that, but
    # ordering it first keeps the two independent of each other.
    out = _MATH_OPERATOR_RE.sub(r"\1", out)
    for pattern, replacement in _MATH_LITERALS:
        out = pattern.sub(replacement, out)
    return out


def transliterated_similarity(a: str, b: str) -> float:
    """Similarity with both sides' inline math transliterated first."""
    return folded_similarity(transliterate_math(a), transliterate_math(b))


def _refine(folded_source: str, offsets: list[int], source: str,
            folded_quote: str, peak_start: int) -> tuple[float, int, int]:
    """Best-aligned raw span around a peak window: take a generous folded
    slice around the peak, align it against the folded quote, trim the
    slice to the aligned subrange, and score the trimmed slice. Returns
    (ratio, raw_start, raw_end)."""
    win = len(folded_quote)
    lo = max(0, peak_start - win // 2)
    hi = min(len(folded_source), peak_start + win + win // 2)
    sm = SequenceMatcher(None, folded_source[lo:hi], folded_quote,
                         autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    if not blocks:
        return 0.0, 0, 0
    # The outermost matching blocks can overrun the true passage when
    # similar tokens repeat just past it (bayesian eq-entropy-integral,
    # 2026-07-29: trailing prose reusing the equation's own math tokens
    # dragged the trim 135 characters wide and sank a 0.978 bracket-slip
    # near-match to 0.811, under the adoption floor). Evaluate the
    # block-boundary trims and keep the one that scores best.
    starts = sorted({lo + b.a for b in blocks})[:4]
    ends = sorted({lo + b.a + b.size for b in blocks})[-4:]
    best_ratio, best_start, best_end = 0.0, 0, 0
    for a_start in starts:
        for a_end in ends:
            if a_end <= a_start:
                continue
            trimmed = folded_source[a_start:a_end]
            ratio = SequenceMatcher(None, trimmed, folded_quote,
                                    autojunk=False).ratio()
            if ratio > best_ratio:
                best_ratio, best_start, best_end = ratio, a_start, a_end
    if best_ratio == 0.0:
        return 0.0, 0, 0
    raw_start = offsets[best_start]
    raw_end = offsets[best_end - 1] + 1
    raw_start, raw_end = _expand_math_delimiters(source, raw_start, raw_end)
    return best_ratio, raw_start, raw_end


def scan_candidates(quote: str, source: str) -> list[dict]:
    """All non-overlapping near-match candidates for `quote` in `source`,
    best-first: [{"score": float, "start": int, "end": int}, ...] with
    raw character offsets into `source`."""
    folded_quote, _ = _fold_with_offsets(quote)
    if len(folded_quote) < MIN_FOLDED_LEN:
        return []
    folded_source, offsets = _fold_with_offsets(source)
    if not folded_source:
        return []
    win = len(folded_quote)
    stride = max(1, win // 4)
    sm = SequenceMatcher(autojunk=False)
    sm.set_seq2(folded_quote)
    peaks: list[tuple[float, int]] = []
    for start in range(0, max(1, len(folded_source) - win + stride), stride):
        sm.set_seq1(folded_source[start:start + win])
        if sm.real_quick_ratio() < _PREFILTER_FLOOR:
            continue
        if sm.quick_ratio() < _PREFILTER_FLOOR:
            continue
        ratio = sm.ratio()
        if ratio >= _PREFILTER_FLOOR:
            peaks.append((ratio, start))
    peaks.sort(key=lambda p: (-p[0], p[1]))

    candidates: list[dict] = []
    taken: list[tuple[int, int]] = []
    for _, peak_start in peaks:
        if any(not (peak_start + win <= a or peak_start >= b)
               for a, b in taken):
            continue  # overlaps a better peak already refined
        ratio, raw_start, raw_end = _refine(
            folded_source, offsets, source, folded_quote, peak_start)
        if raw_end <= raw_start:
            continue
        if any(not (raw_end <= c["start"] or raw_start >= c["end"])
               for c in candidates):
            continue  # refined into a span another peak already claimed
        # The transliteration pass scores the SAME span a second way and the
        # better reading wins. Uniqueness and margin then run over the winning
        # scores unchanged, so a second candidate that also improves still
        # blocks the adoption (the appendix-table case).
        translit = transliterated_similarity(quote, source[raw_start:raw_end])
        candidates.append(
            {"score": max(ratio, translit), "start": raw_start,
             "end": raw_end, "folded_score": ratio,
             "transliterated_score": translit})
        taken.append((peak_start, peak_start + win))
    candidates.sort(key=lambda c: (-c["score"], c["start"]))
    return candidates


_SECTION_NUMBER_RE = re.compile(r"\b(\d+(?:\.\d+)*)\b")
# Reference numbers inside a section label ("Section II-A (Equation 5)",
# "Fig. 3") are not section numbers: parsing one resolves a WRONG region
# that refuses a legitimate unique candidate (the SRL eq-value-function
# holdout, 2026-07-28 overnight batch). Strip them before looking for
# the section number.
_LABEL_REF_RE = re.compile(
    r"(?i)\(?\s*(?:eq(?:uation)?|fig(?:ure)?|table)s?\.?"
    r"\s*\(?\s*\d+\s*\)?\)?")


# Numbers explicitly introduced by a section token. Only these may
# contribute EXTRA regions for a multi-section label: a bare trailing
# number ("Section 6.3, Theorem 3") must never resolve a spurious
# region, because any-region containment WIDENS adoption and the
# refusal guarantee only holds when every resolved region is real.
_SECTION_TOKEN_RE = re.compile(
    r"(?i)\bsec(?:tion)?s?\.?\s*(\d+(?:\.\d+)*)")


def _heading_region(number: str, source: str) -> tuple[int, int] | None:
    """Span of the markdown section numbered `number`, or None."""
    heading_re = re.compile(
        r"^(#{1,6})\s+(?:[^\n]*?\b)?" + re.escape(number) + r"(?:\b|\.)",
        re.MULTILINE)
    match = heading_re.search(source)
    if match is None:
        return None
    level = len(match.group(1))
    next_re = re.compile(r"^#{1," + str(level) + r"}\s", re.MULTILINE)
    nxt = next_re.search(source, match.end())
    return match.start(), (nxt.start() if nxt else len(source))


def resolve_section_regions(section_label: str,
                            source: str) -> list[tuple[int, int]]:
    """Character spans of every markdown section a paper-map element
    names; empty when no label part matches a heading.

    The label is producer-authored free text and may name SEVERAL
    sections ("Section 4.3, Section 7" — the bayesian mc-dropout
    element, 2026-07-29: the equation is introduced in 4.3 and its
    quoted passage lives in 7, and resolving only the first refused a
    legitimate 0.95 unique candidate). Every explicitly section-tokened
    number resolves to its own region and containment may pass in any.
    Labels without a section token keep the single first-number
    reading. Resolution is by section NUMBER against markdown headings;
    a region runs from the matched heading to the next heading at the
    same or a shallower level, which keeps subsections inside their
    parent. An unresolvable or wrongly-resolved label can only REFUSE
    an adoption (containment fails), never widen one."""
    cleaned = _LABEL_REF_RE.sub(" ", section_label or "")
    numbers = [m.group(1) for m in _SECTION_TOKEN_RE.finditer(cleaned)]
    if not numbers:
        m = _SECTION_NUMBER_RE.search(cleaned)
        numbers = [m.group(1)] if m else []
    regions = []
    for number in dict.fromkeys(numbers):
        region = _heading_region(number, source)
        if region is not None:
            regions.append(region)
    return regions


def resolve_section_region(section_label: str,
                           source: str) -> tuple[int, int] | None:
    """First resolvable region of `section_label`, or None. Single-region
    reading kept for callers that anchor to one section; multi-section
    labels should use resolve_section_regions."""
    regions = resolve_section_regions(section_label, source)
    return regions[0] if regions else None


def find_adoption(
    quote: str,
    source: str,
    *,
    region: tuple[int, int] | list[tuple[int, int]] | None = None,
    floor: float = ADOPTION_FLOOR,
    margin: float = ADOPTION_MARGIN,
) -> dict | None:
    """The unique adoptable paper passage for a rejected quote, or None.

    Returns {"passage": str, "start": int, "end": int, "score": float,
    "runner_up": float | None, "region_resolved": bool} only when the
    approved adoption rule holds. Any ambiguity returns None — the
    caller stays on the honest fix-loop path."""
    candidates = scan_candidates(quote, source)
    if not candidates:
        return None
    above = [c for c in candidates if c["score"] >= floor]
    if len(above) != 1:
        return None  # zero above the floor, or ambiguous — honest halt
    best = above[0]
    runner_up = max((c["score"] for c in candidates if c is not best),
                    default=None)
    if runner_up is not None and runner_up > floor - margin:
        return None
    region_resolved = False
    regions: list[tuple[int, int]]
    if region is None:
        regions = []
    elif isinstance(region, tuple):
        regions = [region]
    else:
        regions = list(region)
    if regions:
        if not any(start <= best["start"] and best["end"] <= end
                   for start, end in regions):
            # The unique candidate sits outside every region the
            # element's own label names — refuse the retarget.
            return None
        region_resolved = True
    return {
        "passage": source[best["start"]:best["end"]],
        "start": best["start"],
        "end": best["end"],
        "score": best["score"],
        "runner_up": runner_up,
        "region_resolved": region_resolved,
    }
