"""Static notebook narrative probes.

US-8 checks markdown claims of the form ``param=value`` against the delivered
``params.json`` table. It is intentionally narrow: it catches stale notebook
prose after deterministic parameter fixes without trying to interpret every
number in the narrative.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from probes import ProbeVerdict
from probes.catalogs.narrative import PROBE_CATALOG as _PROBE_CATALOG

PROBE_CATALOG = _PROBE_CATALOG

_NUMBER_CORE_RE = r"(?:\d{1,3}(?:[ ,]\d{3})+|\d+(?:\.\d*)?|\.\d+)"
# Magnitude suffixes count as multipliers only with no-space adjacency and a
# word boundary, and lowercase `m` is not a multiplier at all: meters collide
# endemically in motion planning ("0.15 m" read as 150,000 on pdwa; RCA
# 2026-08-03 finding 3a). A space-separated k/K/M is handled as ambiguous at
# match time (skipped), never as a millionfold/thousandfold multiplier.
_NUMBER_RE = rf"[-+]?{_NUMBER_CORE_RE}(?:[eE][-+]?\d+)?(?:[kKM]\b)?"
_AMBIGUOUS_SUFFIX_RE = re.compile(r"[ \t][kKM]\b")
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_PAPER_ORIGINAL_RE = re.compile(
    r"\b(?:paper(?:'s)?|published|reported|original|rescaled\s+from|from\s+the\s+paper)\b",
    re.IGNORECASE,
)
# Sentence boundaries that do not split inside "Section 7.1" or "0.25": a
# period counts only when not BETWEEN digits (the bayesian table cell left a
# two-character context window; RCA 2026-08-03 finding 3a).
_SENTENCE_SPLIT_RE = re.compile(r"[;\n]|(?<=\d)\.(?!\d)|(?<!\d)\.")
_IMPERATIVE_LEAD_RE = re.compile(
    r"^(?:set|use|try|change|increase|decrease|adjust|replace|tune|pass|"
    r"edit|modify|lower|raise|choose|pick|override|reduce|swap|switch|"
    r"drop|bump|run|re-?run|experiment)\b",
    re.IGNORECASE,
)
_COUNTERFACTUAL_RE = re.compile(
    r"\b(?:you can|if you|to reproduce)\b", re.IGNORECASE)
# The machine-rendered provenance table places paper values beside run
# values by construction; its exact header row identifies it.
_RENDERED_TABLE_HEADER = (
    "| Parameter | Variable from paper | Value from paper | Paper value | "
    "System value | Where in paper | Used? | Notes |"
)


def _params_map(run_dir: Path) -> dict[str, object] | None:
    path = run_dir / ".pipeline" / "params.json"
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    params = raw.get("params", raw) if isinstance(raw, dict) else None
    if not isinstance(params, dict):
        return None
    out: dict[str, object] = {}
    for name, entry in params.items():
        if isinstance(entry, dict) and "value" in entry:
            out[str(name)] = entry.get("value")
    return out


def _markdown_text(nb_path: Path) -> str | None:
    if not nb_path.is_file():
        return None
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cells = nb.get("cells") if isinstance(nb, dict) else None
    if not isinstance(cells, list):
        return None
    chunks: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(str(part) for part in source)
        chunks.append(str(source))
    return "\n\n".join(chunks)


def _numeric_value(raw: str, percent: str | None) -> float | None:
    compact = raw.strip()
    multiplier = 1.0
    if compact and compact[-1] in {"k", "K", "M"}:
        suffix = compact[-1]
        compact = compact[:-1].strip()
        multiplier = 1_000.0 if suffix in {"k", "K"} else 1_000_000.0
    if re.fullmatch(r"[-+]?\d{1,3}(?:[ ,]\d{3})+(?:[eE][-+]?\d+)?", compact):
        compact = re.sub(r"[ ,]", "", compact)
    try:
        value = float(compact)
    except ValueError:
        return None
    value *= multiplier
    return value / 100.0 if percent else value


def _literal_tolerance(raw: str, percent: str | None) -> float:
    normalized = raw.strip()
    if normalized and normalized[-1] in {"k", "K", "M"}:
        normalized = normalized[:-1].strip()
    normalized = re.sub(r"[ ,]", "", normalized)
    mantissa = normalized.lower().split("e", 1)[0]
    if "." not in mantissa:
        return 0.0
    decimals = len(mantissa.rsplit(".", 1)[1])
    tolerance = 0.5 * (10 ** -decimals)
    return tolerance / 100.0 if percent else tolerance


def _numbers_match(claimed: float, actual: object, raw: str, percent: str | None) -> bool:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return True
    actual_f = float(actual)
    tolerance = max(1e-9, abs(actual_f) * 1e-6, _literal_tolerance(raw, percent))
    return abs(claimed - actual_f) <= tolerance


def _claim_pattern(param_name: str) -> re.Pattern[str] | None:
    aliases = [re.escape(param_name)]
    stripped = param_name.replace("_", "")
    if stripped != param_name:
        aliases.append(re.escape(stripped))
    alternatives = "|".join(dict.fromkeys(aliases))
    if not alternatives:
        return None
    return re.compile(
        rf"(?<![A-Za-z0-9_])`?(?:{alternatives})`?\s*=\s*({_NUMBER_RE})(\s*%)?",
        re.IGNORECASE,
    )


def _strip_rendered_params_table(text: str) -> str:
    """Remove the machine-rendered params table (exact header row through
    the last consecutive |-prefixed line)."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == _RENDERED_TABLE_HEADER:
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return "".join(out)


def _sentence_parts(text: str, start: int, end: int) -> tuple[str, str]:
    """(before, after) within the sentence containing [start, end), each
    capped at 160 characters, digit-safe at sentence boundaries."""
    window_start = max(0, start - 160)
    window_end = min(len(text), end + 160)
    before = _SENTENCE_SPLIT_RE.split(text[window_start:start])[-1]
    after = _SENTENCE_SPLIT_RE.split(text[end:window_end])[0]
    return before, after


def _paper_original_context(before: str, after: str) -> bool:
    return bool(_PAPER_ORIGINAL_RE.search(before)
                or _PAPER_ORIGINAL_RE.search(after))


def _speech_act_skip(before: str, after: str) -> bool:
    """Instructions and counterfactuals are not claims about the shipped
    run ('set qn=0.0 to train a norm-free policy'; SRL, RCA finding 3a)."""
    lead = re.sub(r"^[\s>#*+\-\d.)\]`]*", "", before)
    if _IMPERATIVE_LEAD_RE.match(lead):
        return True
    sentence = f"{before} {after}"
    return bool(_COUNTERFACTUAL_RE.search(sentence))


def _arithmetic_tail(text: str, start: int) -> bool:
    """A name token that is the tail of an arithmetic expression is not a
    standalone claim ('r_robot + r_obs = 0.25' says nothing about r_obs;
    pdwa, RCA finding 3a). The operator must itself follow an operand, so a
    markdown '-' bullet does not count."""
    j = start - 1
    while j >= 0 and text[j] in " \t":
        j -= 1
    if j < 0 or text[j] not in "+-*/":
        return False
    k = j - 1
    while k >= 0 and text[k] in " \t":
        k -= 1
    return k >= 0 and (text[k].isalnum() or text[k] in "_)`")


def probe_narrative_vs_params(run_dir: Path) -> ProbeVerdict:
    """US-8: markdown ``param=value`` claims must match params.json."""
    params = _params_map(run_dir)
    if params is None:
        return ProbeVerdict(
            "US-8", "unprobeable", "no readable params.json for prose comparison",
            tier="static",
        )
    markdown = _markdown_text(run_dir / "notebook.ipynb")
    if markdown is None:
        return ProbeVerdict(
            "US-8", "unprobeable", "no readable notebook markdown for prose comparison",
            tier="static",
        )

    text = _strip_rendered_params_table(_FENCE_RE.sub("", markdown))
    checked = 0
    mismatches: list[str] = []
    for name in sorted(params, key=len, reverse=True):
        actual = params[name]
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            continue
        pattern = _claim_pattern(name)
        if pattern is None:
            continue
        for match in pattern.finditer(text):
            before, after = _sentence_parts(text, match.start(), match.end())
            if _paper_original_context(before, after):
                continue
            if _speech_act_skip(before, after):
                continue
            if _arithmetic_tail(text, match.start()):
                continue
            if _AMBIGUOUS_SUFFIX_RE.match(text, match.end()):
                # A space-separated magnitude letter is undecidable between
                # unit and multiplier; a claim that cannot be read with
                # confidence is skipped, never guessed.
                continue
            percent = match.group(2)
            claimed = _numeric_value(match.group(1), percent)
            if claimed is None:
                continue
            checked += 1
            if not _numbers_match(claimed, actual, match.group(1), percent):
                mismatches.append(
                    f"{name} is described as {match.group(1)}"
                    f"{'%' if percent else ''} but params.json has {actual!r}"
                )

    if mismatches:
        return ProbeVerdict(
            "US-8",
            "flag_for_researcher",
            "; ".join(mismatches[:3]),
            tier="static",
            evidence=f"{len(mismatches)} stale markdown parameter claim(s)",
        )
    return ProbeVerdict(
        "US-8",
        "pass",
        f"notebook markdown has {checked} checked parameter claim(s), all matching params.json",
        tier="static",
    )
