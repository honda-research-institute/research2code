"""Static provenance/plausibility probes over params.json (recentering Tier i).

Implements three probe families from the probe catalog note (internal, not shipped):

- US-1 `param-plausibility-ranges` — values with universal physical semantics
  (learning rates, dropout, accuracies, probabilities, momentum) must sit in
  canonical bounds. Names without universal semantics are skipped, never
  guessed. Fail-side evidence: lr=7.4 (the paper's section number) shipped in
  a notebook that trains to chance (tests/fixtures/zoo/gbald-lr74-never-learns).

- US-2 `provenance-internal-consistency` —
  (a) a `source: "paper"` entry's `paper_section` must look like a paper
      locator and must not itself admit a non-paper origin ("Field-guide
      convention ..." — the A-001 class);
  (b) a reasoning that cites a named convention must match the value it
      justifies ("Adam-default convention" cannot justify lr=7.4).

- US-3 `paper-provenance-quote-match` — for `source: "paper"` entries, any
  explicit quote in note/reasoning ("Paper: <claim>") must appear in the
  paper text, and the value itself must be findable (with rendering
  equivalences: 0.99 ↔ "99%"). Fail-side evidence: train_until_accuracy=0.99
  labeled paper-sourced with the invented quote "train until training
  accuracy exceeds 99%" — zero occurrences in that run's own paper.md
  (tests/fixtures/evidence/june9-gbald-run). Value-presence is only enforced
  for values with >=3 significant characters: small integers ("6") match any
  paper trivially, so they are the quote arm's job, not the value arm's.

Standalone and read-only: no run_pipeline imports, no field-guide access, no
network. Exit 0 = no errors (warnings allowed), exit 1 = errors, exit 2 = bad
invocation. `--report` writes the probe-verdict JSON (chat-readable, per the
recentering verdict schema).

Usage:
    python3 scripts/validate_params_provenance.py --params <params.json> \
        [--paper <paper.md>] [--report <verdicts.json>]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# One alias vocabulary with the anchored extractor (role-suffixed generated
# names like alpha_loss anchor on the paper's bare `alpha`), so the deriver
# never emits a paper claim this validator's US-3b arm then rejects. The
# vocabulary lives in the dependency-free schemas/param_text.py; the bare
# fallback import serves the vendored harness layout, where this validator
# and param_text.py sit side by side with no schemas package (R2C-019).
try:
    from schemas.param_text import param_text_aliases  # noqa: E402
except ImportError:  # pragma: no cover - vendored harness layout
    from param_text import param_text_aliases  # noqa: E402

# ---------------------------------------------------------------------------
# US-1: canonical ranges for names with universal semantics
# ---------------------------------------------------------------------------

# (pattern, lo, hi, rationale) — inclusive bounds. Conservative by design:
# only names whose meaning is paradigm-independent. R_0/eta/temperature etc.
# are scale- or method-dependent and belong to US-4/spec checks, not here.
RANGE_REGISTRY: list[tuple[str, float, float, str]] = [
    (r"^(learning_rate|lr)$", 1e-6, 1.0, "optimizer learning rate"),
    (r"^dropout(_rate|_p)?$", 0.0, 0.95, "dropout probability"),
    (r"^momentum$", 0.0, 1.0, "optimizer momentum"),
    (r"(^|_)(accuracy|acc)$", 0.0, 1.0, "accuracy fraction"),
    (r"^train_until_accuracy$", 0.0, 1.0, "accuracy fraction"),
    (r"^weight_decay$", 0.0, 1.0, "weight decay coefficient"),
]

# US-2b: named conventions and the value families they can justify.
CONVENTION_REGISTRY: list[tuple[str, float, float, str]] = [
    # (reasoning-pattern, lo, hi, label)
    (r"adam[\s-]default", 1e-4, 1e-2, "Adam default learning rate (~1e-3)"),
]

# US-2a: a paper locator should reference a place in the paper...
_LOCATOR_HINT = re.compile(
    r"\d|section|table|eq\.?|equation|fig\.?|figure|appendix|abstract|page|§",
    re.IGNORECASE,
)
# ...and must not admit a non-paper origin.
_NON_PAPER_ORIGIN = re.compile(
    r"convention|field[\s-]guide|default|typical|heuristic", re.IGNORECASE
)

_QUOTE_PATTERNS = [
    re.compile(r"paper:\s*(.{12,}?)(?:$|\.\s|\")", re.IGNORECASE),
    re.compile(r"\"([^\"]{12,})\""),
]

# US-3b: inline "Paper states name=value" claims. These appear on entries of
# ANY source (the pdwa fresh run put "Paper states beta_1=0.1" in
# system_default reasoning with paper_value=0.1 — the paper never states any
# beta value). Caught 2026-06-10 by cross-checking the stage-4 LLM review
# against this probe; see the zoo pdwa-jterms-fresh scenario.
_PAPER_CLAIM_RE = re.compile(
    r"paper\s+(?:states|uses|sets|specifies|reports|defines|gives)\s+"
    r"([A-Za-z_][\w]*)\s*=\s*([0-9][0-9.eE+-]*)",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    """Lowercase, collapse whitespace, and strip markdown emphasis artifacts.

    PDF→markdown conversion mangles numerals with emphasis runs (the BADGE
    paper renders 0.001 as `0*.*001`), so `*`, `_`, backticks, and
    backslashes are stripped before matching. Digit-grouping commas fold
    too ("3,000 episodes" ↔ 3000 — the SRL paper states the episode count
    only in comma-grouped form, so the value arm called the paper's own
    number unfindable); the lookahead requires a full three-digit group so
    a decimal comma ("0,5") is never collapsed. Leniency-only in both
    cases: more paper renderings match, never fewer.
    """
    text = re.sub(r"[*_`\\]", "", text)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


_TOKEN_RE = re.compile(r"[a-z0-9.%]+")


def _tokens(normalized: str) -> list[str]:
    return [t.strip(".") for t in _TOKEN_RE.findall(normalized) if len(t.strip(".")) >= 3 or any(c.isdigit() for c in t)]


def _tok_match(quote_tok: str, window_tok: str) -> bool:
    """Stem-lite equality: train ↔ training, 99% ↔ 99%."""
    if quote_tok == window_tok:
        return True
    if len(quote_tok) >= 4 and window_tok.startswith(quote_tok):
        return True
    return len(window_tok) >= 4 and quote_tok.startswith(window_tok)


def _quote_in_paper(quote: str, paper_tokens: list[str]) -> bool:
    """Windowed token-overlap match for a claimed paper quote.

    Provenance notes legitimately paraphrase ("Paper: train until..." vs the
    paper's "...SGD until training accuracy exceeds 99%"), so exact substring
    is wrong. A quote counts as found when some window of the paper contains
    (a) EVERY numeric token of the quote — numbers are the claim's payload
    and are exactly what fabricated quotes get wrong — and (b) >= 70% of its
    word tokens (stem-lite matched).
    """
    q_tokens = _tokens(_norm(quote))
    if not q_tokens:
        return True  # nothing checkable
    q_numeric = [t for t in q_tokens if any(c.isdigit() for c in t)]
    q_words = [t for t in q_tokens if t not in q_numeric]

    win = max(30, 4 * len(q_tokens))
    stride = max(1, win // 2)
    for start in range(0, max(1, len(paper_tokens) - win + stride), stride):
        window = paper_tokens[start:start + win]
        if not all(any(_tok_match(n, w) for w in window) for n in q_numeric):
            continue
        if not q_words:
            return True
        hits = sum(1 for q in q_words if any(_tok_match(q, w) for w in window))
        if hits / len(q_words) >= 0.7:
            return True
    return False


_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _scientific_renderings(value) -> list[str]:
    """Scientific / power-of-ten equivalents: 1e-3, 1e-03, 10^-3, 10^{-3}, 10⁻³.

    A paper may state a learning rate as `1e-3` or `10^{-3}` rather than the
    decimal `0.001` — and PDF→markdown OCR mangles the decimal form (the GBALD
    paper rendered 0.001 as "10K3"/"103"). Emitting these equivalents makes the
    US-3 value arm more lenient (fewer false 'value not in paper' over-claims),
    never stricter. Case is normalised at match time, so only lowercase-relevant
    forms are emitted. Conservative: finite nonzero numerics only; power-of-ten
    word forms (10^-3) are emitted only when the mantissa is ~1.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return []
    if value == 0 or not math.isfinite(value):
        return []
    exp = math.floor(math.log10(abs(value)))
    mant = round(value / (10.0 ** exp), 6)
    mant_str = f"{mant:g}"
    forms = [f"{mant_str}e{exp}", f"{mant_str}e{exp:+03d}"]  # 1e-3, 1e-03
    sup = str(exp).translate(_SUPERSCRIPT)
    plain = f"10^{exp}"
    # LaTeX table math survives Marker as e.g. `$5 \times 10^{-3}$`; `_norm`
    # strips the backslash, leaving `5 times 10^{-3}` — so the braced
    # exponent AND the times/cdot WORD forms must be emitted for every
    # mantissa. The pdfgnn 2026-08-05 overnight halt: the paper stated the
    # learning rate exactly this way, the analyzer truthfully claimed
    # source=paper, and the probe declared the value unfindable — a probe
    # false negative that first laundered a correct claim to
    # system_inferred (afternoon roll) and then halted the run (night
    # roll). Leniency-only: more paper renderings match, never fewer.
    braced = f"10^{{{exp}}}"
    if abs(mant - 1.0) < 1e-9:
        forms += [plain, braced, f"10{sup}",
                  f"1×10{sup}", f"1x10^{exp}",
                  f"1 times {plain}", f"1 times {braced}",
                  f"1 cdot {plain}", f"1 cdot {braced}"]
    else:
        forms += [f"{mant_str}×10^{exp}", f"{mant_str}x10^{exp}",
                  f"{mant_str}×10{sup}",
                  f"{mant_str}×{braced}", f"{mant_str}x{braced}",
                  f"{mant_str} times {plain}", f"{mant_str} times {braced}",
                  f"{mant_str} cdot {plain}", f"{mant_str} cdot {braced}"]
    return forms


def _value_renderings(value) -> list[str]:
    """Forms under which a paper might state this value (decimal, integer,
    percent, and scientific/exponent equivalents)."""
    forms: list[str] = []
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return forms
    forms.append(f"{value:g}")
    if isinstance(value, float) and value == int(value):
        forms.append(str(int(value)))
    if isinstance(value, float) and 0 < value < 1:
        pct = value * 100
        forms.append(f"{pct:g}%")
        forms.append(f"{pct:g} %")
    forms.extend(_scientific_renderings(value))
    # de-dup, preserve order
    seen: set[str] = set()
    return [f for f in forms if not (f in seen or seen.add(f))]


def _significant_chars(value) -> int:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return 0
    return len(re.sub(r"[^0-9]", "", f"{value:g}"))


def _value_findable_in_norm(value, norm_paper: str | None) -> bool:
    """US-3 value-arm predicate over an already-normalised paper string.

    Low-significance values (<3 significant digits) match paper text trivially
    and are not value-checkable, so they count as findable. When no paper is
    available the value cannot be disproven, so it counts as findable. Otherwise
    at least one `_value_renderings` form must appear in the normalised paper.
    """
    if norm_paper is None or _significant_chars(value) < 3:
        return True
    forms = _value_renderings(value)
    return bool(forms) and any(f.lower() in norm_paper for f in forms)


def value_findable_in_paper(value, paper_text: str | None) -> bool:
    """Public wrapper: normalise the paper once, then apply the US-3 value arm.

    derive_params.py calls this before stamping source="paper" so it never
    manufactures a claim this validator's US-3 value arm would later reject —
    deriver and validator share one definition of "findable in the paper".
    """
    return _value_findable_in_norm(
        value, _norm(paper_text) if paper_text is not None else None)


def unsatisfiable_paper_claims(
    entry: dict, paper_text: str | None,
    *, paper_tokens: list[str] | None = None,
) -> list[tuple[str, str]]:
    """The inline "paper states name=value" claims in ONE params entry that
    US-3b will reject, as (claimed_name, claimed_value) pairs.

    The US-3b arm below calls this, and so does the driver's
    `relabel_param_source` auto-resolve applier before it declares a
    resolution successful. One definition, two callers: relabeling a param's
    `source` does NOT clear an inline claim (US-3b fires on every source by
    design), so an applier that only checked the pydantic gate reported
    success on a resolution the validator then halted on — the
    bayesian-active-learning "internal inconsistency between applier and
    validator" halt on 2026-07-27, where a relabel to `system_inferred` left
    "The paper uses batch_returns=300" standing in `reasoning`.
    """
    if paper_text is None and paper_tokens is None:
        return []
    if paper_tokens is None:
        paper_tokens = _tokens(_norm(paper_text))
    blob = " ".join(
        str(entry.get(k, "")) for k in ("reasoning", "note", "paper_section")
    )
    unsatisfiable: list[tuple[str, str]] = []
    for m in _PAPER_CLAIM_RE.finditer(blob):
        claimed_name, claimed_value = m.group(1), m.group(2)
        if not _claim_findable(claimed_name, claimed_value, paper_tokens):
            unsatisfiable.append((claimed_name, claimed_value))
    return unsatisfiable


def _claim_findable(name: str, value_repr: str,
                    paper_tokens: list[str]) -> bool:
    """One claim-findability predicate for BOTH the deriver's stamping gate
    (`paper_claim_findable`) and the US-3b validator arm.

    Before 2026-09-01 the two disagreed: the deriver's gate tried every
    `param_text_aliases` form of the name while the validator arm checked
    the claimed name only, so a claim the gate blessed could still be
    rejected by the probe (bayesian-active-learning `batch_returns`, whose
    paper symbol is `b` — six analyzer fix dispatches on an unfixable
    finding, converging only by splitting one knob into two entries).

    Try every text alias of the name and both integral renderings of the
    value (an extracted "1.0" coerces to int 1, while the paper says "1.0"
    — the token match must accept either direction). Findable under any
    combination counts."""
    renderings = [value_repr]
    try:
        as_float = float(value_repr)
    except (TypeError, ValueError):
        as_float = None
    if as_float is not None and as_float.is_integer():
        renderings.extend([str(int(as_float)), f"{as_float:.1f}"])
    for alias in param_text_aliases(str(name)):
        name_words = " ".join(
            p for p in re.split(r"[_\W]+", alias) if len(p) >= 3
        )
        for rendering in dict.fromkeys(renderings):
            pseudo_quote = f"{name_words} {rendering}".strip()
            if _quote_in_paper(pseudo_quote, paper_tokens):
                return True
    return False


def paper_claim_findable(name, value, paper_text: str | None) -> bool:
    """Public wrapper for the US-3b arm: would "Paper states name=value" survive?

    US-3b (below) rejects an inline "Paper states name=value" claim when the
    name words and the value do not co-occur in a window of the paper text.
    derive_params.py calls this before stamping such a claim from a HEURISTIC
    text scan, so the deriver never manufactures a claim this validator would
    then halt on — the same deriver/validator agreement the value_findable arm
    gives source="paper". Non-numeric values and a missing paper are treated as
    findable (nothing to disprove), matching US-3b, which only checks numeric
    claims against available paper text.
    """
    if paper_text is None:
        return True
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return True
    return _claim_findable(str(name), repr(value),
                           _tokens(_norm(paper_text)))


# One-paragraph acceptance contract per probe, keyed by the probe id the
# stderr bullets carry. The stage-2x fix loop inlines these verbatim into the
# analyzer's fix finding so the retry receives the complete rule WITHOUT
# reading this file (agent dispatch optimization 2026-09-01: fix dispatches
# burned most of their turns reverse-engineering these predicates from
# source). Keep each contract beside the arm it describes when editing.
PROBE_ACCEPTANCE_CONTRACTS: dict[str, str] = {
    "US-1": (
        "US-1 plausibility: the parameter's VALUE must lie inside the "
        "[lo, hi] range the finding names (the range registry keys on the "
        "parameter's name and its universal physical semantics). The probe "
        "re-checks the re-derived value against exactly that range."
    ),
    "US-2a": (
        "US-2a locator: a source=paper entry's paper_section must read as a "
        "place in the paper (a section/table/equation/figure/appendix/page "
        "reference) and must NOT contain any of: 'convention', 'field "
        "guide', 'default', 'typical', 'heuristic'. If the value is not "
        "actually paper-stated, change the provenance honestly instead of "
        "rewording the locator."
    ),
    "US-2b": (
        "US-2b convention consistency: when an entry's reasoning cites the "
        "named convention, the value must lie inside that convention's "
        "value family (the finding names the [lo, hi] family). Either the "
        "value genuinely follows the convention, or the reasoning must stop "
        "citing it."
    ),
    "US-3": (
        "US-3 paper findability (source=paper entries only), two arms. "
        "Quote arm: any quoted text in reasoning/note ('Paper: ...' or a "
        "double-quoted span of 12+ chars) must match the paper — some "
        "window of the normalized paper text (lowercased; '*', '_', "
        "backticks, backslashes stripped; digit-grouping commas folded; "
        "whitespace collapsed) must contain EVERY numeric token of the "
        "quote and at least 70% of its word tokens (3+ chars, prefix/stem "
        "matched). Value arm: a value with 3+ significant digits must "
        "appear in the normalized paper under SOME standard rendering — "
        "decimal (0.001), integer (300), percent for values in (0,1) "
        "(30%), or scientific/power-of-ten forms (1e-3, 10^-3, 10^{-3}, "
        "5x10^{-3}, '5 times 10^{-3}'). If no rendering appears, the paper "
        "does not state the value: relabel the provenance honestly (demo "
        "assumption or derived statistic) rather than rewording the quote."
    ),
    "US-3b": (
        "US-3b inline paper claims (checked on EVERY source, not just "
        "source=paper): any sentence in reasoning/note/paper_section "
        "matching 'paper states|uses|sets|specifies|reports|defines|gives "
        "<name>=<number>' is verified against the paper — some window of "
        "the normalized paper text must contain the claimed number AND at "
        "least 70% of the claimed name's word pieces (3+ char pieces of "
        "the name split on underscores/punctuation, prefix/stem matched). "
        "Relabeling the entry's source does NOT clear this probe: the "
        "claim sentence itself must either be true in the paper or be "
        "removed/reworded so it no longer asserts 'paper <verb> "
        "name=value'. A claim about a quantity the paper names by a "
        "different symbol must quote the paper's own symbol and words."
    ),
}


def check_params(params: dict, paper_text: str | None) -> list[dict]:
    """Run US-1/US-2/US-3 over a loaded params mapping.

    Returns verdict dicts: {probe, param, severity: error|warn, message}.
    """
    findings: list[dict] = []
    norm_paper = _norm(paper_text) if paper_text is not None else None
    paper_tokens = _tokens(norm_paper) if norm_paper is not None else []

    for name, entry in params.items():
        if not isinstance(entry, dict):
            continue
        value = entry.get("value")
        source = entry.get("source", "")
        reasoning_blob = " ".join(
            str(entry.get(k, "")) for k in ("reasoning", "note", "paper_section")
        )

        # --- US-1: plausibility ranges -----------------------------------
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            for pattern, lo, hi, label in RANGE_REGISTRY:
                if re.search(pattern, name):
                    if not (lo <= float(value) <= hi):
                        findings.append({
                            "probe": "US-1",
                            "param": name,
                            "severity": "error",
                            "message": (
                                f"{name}={value!r} outside plausible range "
                                f"[{lo:g}, {hi:g}] for {label}"
                            ),
                        })
                    break

        # --- US-2b: convention-value consistency --------------------------
        for pattern, lo, hi, label in CONVENTION_REGISTRY:
            if re.search(pattern, reasoning_blob, re.IGNORECASE) and isinstance(
                value, (int, float)
            ) and not isinstance(value, bool):
                if not (lo <= float(value) <= hi):
                    findings.append({
                        "probe": "US-2b",
                        "param": name,
                        "severity": "error",
                        "message": (
                            f"{name}={value!r} cites '{label}' but is outside "
                            f"that convention's value family [{lo:g}, {hi:g}]"
                        ),
                    })

        # --- US-3b: inline paper claims, regardless of source --------------
        if norm_paper is not None:
            for claimed_name, claimed_value in unsatisfiable_paper_claims(
                entry, None, paper_tokens=paper_tokens
            ):
                findings.append({
                    "probe": "US-3b",
                    "param": name,
                    "severity": "error",
                    "message": (
                        f"{name} claims the paper states "
                        f"{claimed_name}={claimed_value}, but no such "
                        f"value appears near {claimed_name!r} in the "
                        f"paper text (fabricated paper claim)"
                    ),
                })

        if source == "paper":
            # --- US-2a: locator shape -------------------------------------
            section = str(entry.get("paper_section", "") or "")
            if _NON_PAPER_ORIGIN.search(section):
                findings.append({
                    "probe": "US-2a",
                    "param": name,
                    "severity": "error",
                    "message": (
                        f"{name} is source=paper but its paper_section admits a "
                        f"non-paper origin: {section!r} (A-001 class)"
                    ),
                })
            elif not section or not _LOCATOR_HINT.search(section):
                findings.append({
                    "probe": "US-2a",
                    "param": name,
                    "severity": "warn",
                    "message": (
                        f"{name} is source=paper but paper_section "
                        f"{section!r} does not look like a paper locator"
                    ),
                })

            # --- US-3: quote + value findable in the paper ----------------
            if norm_paper is not None:
                for qp in _QUOTE_PATTERNS:
                    m = qp.search(reasoning_blob)
                    if m and not _quote_in_paper(m.group(1), paper_tokens):
                        findings.append({
                            "probe": "US-3",
                            "param": name,
                            "severity": "error",
                            "message": (
                                f"{name} cites a paper quote not found in the "
                                f"paper text: {m.group(1)!r}"
                            ),
                        })
                        break
                if not _value_findable_in_norm(value, norm_paper):
                    forms = _value_renderings(value)
                    if forms:
                        findings.append({
                            "probe": "US-3",
                            "param": name,
                            "severity": "error",
                            "message": (
                                f"{name}={value!r} is source=paper but no "
                                f"rendering of the value ({', '.join(forms)}) "
                                f"appears in the paper text"
                            ),
                        })

    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--params", required=True, help="path to params.json")
    parser.add_argument("--paper", help="path to paper.md (enables US-3)")
    parser.add_argument("--report", help="write probe verdicts JSON here")
    args = parser.parse_args(argv)

    params_path = Path(args.params)
    if not params_path.is_file():
        print(f"FAIL: params file not found: {params_path}", file=sys.stderr)
        return 2
    try:
        loaded = json.loads(params_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"FAIL: params file is not valid JSON: {e}", file=sys.stderr)
        return 2
    params = loaded.get("params", loaded)

    paper_text: str | None = None
    if args.paper:
        paper_path = Path(args.paper)
        if not paper_path.is_file():
            print(f"FAIL: paper file not found: {paper_path}", file=sys.stderr)
            return 2
        paper_text = paper_path.read_text(encoding="utf-8")

    findings = check_params(params, paper_text)
    if args.report:
        Path(args.report).write_text(
            json.dumps({"validator": "validate_params_provenance",
                        "findings": findings}, indent=2) + "\n",
            encoding="utf-8",
        )

    errors = [f for f in findings if f["severity"] == "error"]
    warns = [f for f in findings if f["severity"] == "warn"]
    for f in warns:
        print(f"WARN [{f['probe']}] {f['message']}")
    if errors:
        print(f"FAIL: {len(errors)} provenance/plausibility error(s):")
        for f in errors:
            print(f"  - [{f['probe']}] {f['message']}")
        return 1
    skipped = " (US-3 skipped: no --paper)" if paper_text is None else ""
    print(f"ok: params provenance probes passed "
          f"({len(warns)} warning(s)){skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
