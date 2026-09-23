"""Render-equivalent quote re-anchoring (queue item 11, 2026-07-21).

Evidence (the overnight-0720 batch log (internal, not shipped), three runs):
Think-tier producers emit quotes that are RENDER-equivalent to the
source but byte-different, and demonstrably cannot transcribe the
difference even with the exact bytes in the fix prompt (SRL burned all
3 retries that way). The observed delta classes:

- A. combining vs spacing diacritics (U+0303 vs U+02DC tilde, U+0304 vs
  U+00AF macron — identical when rendered), em dash vs "--", and
  typographic vs ASCII quotation marks (the paper's "agent’s" U+2019
  against the producer's "agent's" U+0027 — SRL halted stage 1 on
  exactly one such apostrophe on 2026-07-27);
- B. math-delimiter substitution ($...$ / $$...$$ vs \\(...\\) / \\[...\\]);
- C. Markdown emphasis stripped (**bold** quoted as plain prose);
- D. whitespace inserted or collapsed mid-token (subscript splits,
  double-space collapse around math).

The honest fix is the primitive the generic-insight contract canonized:
the producer's quote is a LOCATOR, the source's own bytes are the
authority. `find_unique_span` folds both sides with a render-equivalence
normalization and returns the raw source span ONLY when exactly one
match exists (a normalized form is never used to choose among ambiguous
occurrences). The caller replaces the quote with the source's bytes —
meaning is never edited, and the deterministic validator remains the
terminal gate.
"""

from __future__ import annotations

import unicodedata

# Multi-char sequences dropped by the fold: math delimiters in both the
# source's form ($$, $) and the MathJax form producers substitute.
_DELIMITERS = ("$$", "\\(", "\\)", "\\[", "\\]")
_EMPHASIS = set("*_`$")
_DASHES = {"—", "–", "−", "-"}
# Quotation marks, folded the same way dashes are: one canonical form per
# class. Producers reproduce a PDF's typographic quotes as their ASCII
# equivalents and cannot see the difference in a fix prompt, so without
# this the fix loop exhausts on a character that renders identically.
# The backtick stays out — `_EMPHASIS` already drops it, and folding it
# here instead would change established behavior.
_SINGLE_QUOTES = {"’", "‘", "‛", "′", "´", "'"}
_DOUBLE_QUOTES = {"“", "”", "‟", "″", "\""}

# A folded needle shorter than this is too little context to trust a
# "unique" match; callers keep such quotes on the honest failure path.
MIN_FOLDED_LEN = 12


def _fold_with_offsets(text: str) -> tuple[str, list[int]]:
    """The render-equivalence fold, keeping a per-folded-char map back to
    the original character index so a folded match recovers raw bytes.

    Folds: math delimiters and emphasis markers dropped; every dash
    variant collapsed to a single '-' run; every single- and double-quote
    variant collapsed to ASCII ' and "; ALL whitespace dropped; NFKD
    per character (which converges spacing modifiers and their combining
    counterparts: U+02DC decomposes to space + U+0303) with decomposed
    whitespace dropped; lowercased. Deliberately NOT folded: backslashes
    and LaTeX command spelling (inner LaTeX must still match), and
    letter/digit identity."""
    chars: list[str] = []
    offsets: list[int] = []
    i = 0
    n = len(text)
    while i < n:
        matched_delim = False
        for delim in _DELIMITERS:
            if text.startswith(delim, i):
                i += len(delim)
                matched_delim = True
                break
        if matched_delim:
            continue
        ch = text[i]
        if ch in _EMPHASIS:
            i += 1
            continue
        if ch in _DASHES:
            if not chars or chars[-1] != "-":
                chars.append("-")
                offsets.append(i)
            i += 1
            continue
        if ch in _SINGLE_QUOTES:
            chars.append("'")
            offsets.append(i)
            i += 1
            continue
        if ch in _DOUBLE_QUOTES:
            chars.append("\"")
            offsets.append(i)
            i += 1
            continue
        if ch.isspace():
            i += 1
            continue
        for decomposed in unicodedata.normalize("NFKD", ch):
            if decomposed.isspace():
                continue
            chars.append(decomposed.lower())
            offsets.append(i)
        i += 1
    return "".join(chars), offsets


def _expand_math_delimiters(source: str, start: int, end: int) -> tuple[int, int]:
    """Absorb '$' runs that tightly enclose the span on BOTH sides, so a
    re-anchored equation keeps the paper's own delimiters for display."""
    left = start
    while left > 0 and source[left - 1] == "$":
        left -= 1
    right = end
    while right < len(source) and source[right] == "$":
        right += 1
    if left < start and right > end:
        return left, right
    return start, end


def find_unique_span(
    quote: str,
    source: str,
    *,
    region: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """The unique render-equivalent source span for `quote`, or None.

    Returns raw character offsets [start, end) into `source` when the
    folded quote occurs EXACTLY ONCE in the folded source (searched
    within `region` when given). Zero matches or more than one both
    return None — ambiguity is honest failure, never a guess."""
    folded_quote, _ = _fold_with_offsets(quote)
    if len(folded_quote) < MIN_FOLDED_LEN:
        return None
    search_base = 0
    haystack = source
    if region is not None:
        search_base = max(0, region[0])
        haystack = source[search_base:max(search_base, region[1])]
    folded_source, offsets = _fold_with_offsets(haystack)
    first = folded_source.find(folded_quote)
    if first == -1:
        return None
    if folded_source.find(folded_quote, first + 1) != -1:
        return None
    start = offsets[first] + search_base
    last_char_idx = offsets[first + len(folded_quote) - 1] + search_base
    end = last_char_idx + 1
    return _expand_math_delimiters(source, start, end)


def reanchor_quote(
    quote: str,
    source: str,
    *,
    region: tuple[int, int] | None = None,
) -> str | None:
    """The source's own bytes for a render-equivalent quote, or None."""
    span = find_unique_span(quote, source, region=region)
    if span is None:
        return None
    return source[span[0]:span[1]]
