"""Deterministic acquisition of a paper's OWN cited public datasets.

R2C generates a runnable demo, and a demo for a data-hungry paradigm needs
data. Three canonical paradigm templates already download theirs (MNIST /
CIFAR through torchvision, cached, behind `R2C_OFFLINE`), but the
gap-path template is file-only: a brand-new family's data source is unknown
when the template is authored, so it cannot hardcode a fetch. The
consequence, unintended, is that a new-territory paper inherits a loader
that refuses to obtain data even when its own paper names a concrete public
dataset (R2C-052, the probabilistic-demand-forecasting run).

This module closes that gap WITHOUT adding a reasoning-tier surface. The
paper text already contains the URLs, so nothing has to be invented:

  1. Scan the parsed paper for cited URLs.
  2. Keep only the ones that match a known DATA HOST (the whitelist).
  3. Refuse, by rule, any host that needs credentials.
  4. Probe what survives, anonymously, and fetch the first usable answer
     under a byte ceiling, recording a checksum and full provenance.

## Why extraction beats asking a model for the URL

A hallucinated URL that happens to resolve is a fidelity disaster: the
demo would train on data the paper never used while the report cited the
paper's dataset. Extracting from the paper makes that impossible by
construction, since every candidate is a substring of the paper itself.
The trade is that extraction cannot tell a dataset link from a citation
link, which is what the host whitelist is for: arXiv, DOI, and vendor
marketing links do not match a data host, so they never become candidates.

## Why every candidate is probed rather than trusted

PDF-to-markdown conversion mangles footnotes. The motivating paper's
footnote 4 repeats one Kaggle URL twenty-odd times, corrupts the tail into
`...retail-sales-data-data-data-data`, and splices a second URL's fragment
onto the end. A cited string is therefore a CLAIM about a source, not a
source. Probing separates the three outcomes that need different handling:
usable, needs-credentials (a Kaggle competition answers 401), and
unavailable (the mangled tail 404s). Repairing a mangled URL is
deliberately NOT attempted: guessing what a corrupted string meant is how
you silently fetch the wrong dataset.

Network access is injected (`opener`), never imported ambiently, so the
unit tests exercise every branch with zero network traffic. `R2C_OFFLINE`
short-circuits the whole module, matching the guard the canonical
templates already honor, so validation harnesses and restricted-network
runs behave predictably.

This module acquires and records. It does NOT decide demo scale: a byte
ceiling and a row cap keep a 1 GB table out of a delivery, and the
paradigm's own calibration owns what a demo-sized sample means.

Usage:

    python scripts/dataset_acquisition.py --paper <paper.md> --dest <dir>
    python scripts/dataset_acquisition.py --paper <paper.md> --probe-only

Exit codes:
  0  a dataset was acquired (or, with --probe-only, at least one usable
     source was found)
  1  no usable source (every candidate refused, unavailable, or absent)
  2  setup error (missing paper file, unreadable destination)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parent.parent

OFFLINE_ENV = "R2C_OFFLINE"

# Default byte ceiling for a single fetch. The motivating dataset is 90.7 MB
# zipped, so 250 MB leaves headroom without letting a multi-gigabyte archive
# land in a run directory.
DEFAULT_MAX_BYTES = 250 * 1024 * 1024

# Content types that mean "you were handed a web page, not a dataset". A
# whitelisted host answering 200 with HTML is a login wall or an error page.
_HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")

_URL_RE = re.compile(r"https?://[^\s<>\"'\\]+", re.IGNORECASE)

# Trailing characters that belong to the prose or markup around a URL rather
# than to the URL itself. LaTeX-converted papers routinely end a footnote URL
# with a comma, a dollar sign, or a backslash.
_URL_TRAILING_JUNK = ".,;:!?)]}>\"'`$\\|"


@dataclass(frozen=True)
class HostRule:
    """One whitelisted data host.

    `credentialed` marks a host that cannot be fetched anonymously. Such a
    host is still recognized, so the report can say WHY the paper's own
    source was not used instead of silently ignoring it.
    """

    host: str
    pattern: re.Pattern[str]
    credentialed: bool = False
    download_template: str | None = None

    def resolve(self, url: str, match: re.Match[str]) -> str:
        if self.download_template is None:
            return url
        return self.download_template.format(**match.groupdict())


# Ordered: the first matching rule wins. Kaggle's COMPETITION rule must
# precede its dataset rule, since both live under kaggle.com and only the
# dataset endpoint is anonymous (verified 2026-08-04: the dataset download
# answers 200 with 90.7 MB of application/zip, the competition answers 401).
DATA_HOST_RULES: tuple[HostRule, ...] = (
    HostRule(
        host="kaggle_competition",
        pattern=re.compile(
            r"^https?://(?:www\.)?kaggle\.com/(?:c|competitions)/(?P<slug>[^/?#]+)",
            re.IGNORECASE,
        ),
        credentialed=True,
    ),
    HostRule(
        host="kaggle_dataset",
        pattern=re.compile(
            r"^https?://(?:www\.)?kaggle\.com/datasets/(?P<owner>[^/?#]+)/(?P<slug>[^/?#]+)",
            re.IGNORECASE,
        ),
        download_template=(
            "https://www.kaggle.com/api/v1/datasets/download/{owner}/{slug}"
        ),
    ),
    HostRule(
        host="zenodo",
        pattern=re.compile(
            r"^https?://(?:www\.)?zenodo\.org/(?:record|records)/(?P<record>\d+)",
            re.IGNORECASE,
        ),
    ),
    HostRule(
        host="figshare",
        pattern=re.compile(r"^https?://(?:\w+\.)?figshare\.com/\S+", re.IGNORECASE),
    ),
    HostRule(
        host="uci",
        pattern=re.compile(
            r"^https?://archive\.ics\.uci\.edu/\S+", re.IGNORECASE
        ),
    ),
    HostRule(
        host="huggingface_dataset",
        pattern=re.compile(
            r"^https?://huggingface\.co/datasets/(?P<owner>[^/?#]+)/(?P<slug>[^/?#]+)",
            re.IGNORECASE,
        ),
    ),
    HostRule(
        host="github_asset",
        pattern=re.compile(
            r"^https?://(?:raw\.githubusercontent\.com/\S+"
            r"|github\.com/\S+?/(?:releases/download|archive)/\S+)",
            re.IGNORECASE,
        ),
    ),
    HostRule(
        host="openml",
        pattern=re.compile(r"^https?://(?:www\.)?openml\.org/\S+", re.IGNORECASE),
    ),
)


@dataclass(frozen=True)
class DatasetSource:
    """A citation-derived candidate: which host, as cited, and how to fetch."""

    host: str
    cited_url: str
    download_url: str
    credentialed: bool

    @property
    def identity(self) -> str:
        return f"{self.host}:{self.cited_url}"


@dataclass(frozen=True)
class ProbeResult:
    """What an anonymous probe of one candidate established."""

    source: DatasetSource
    status: str  # ok | needs_credentials | unavailable | refused_host | refused_offline
    reason: str
    http_code: int | None = None
    content_type: str | None = None
    content_length: int | None = None

    @property
    def usable(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class Acquisition:
    """A dataset that was actually written to disk."""

    source: DatasetSource
    path: Path
    sha256: str
    bytes_written: int
    truncated: bool


@dataclass
class AcquisitionReport:
    """The full, disclosable story of one acquisition attempt.

    Every candidate's outcome is retained, because "we used synthetic data"
    is only honest when it can say what was tried and why each source was
    rejected."""

    acquired: Acquisition | None = None
    probes: list[ProbeResult] = field(default_factory=list)
    candidates_found: int = 0
    offline: bool = False

    @property
    def succeeded(self) -> bool:
        return self.acquired is not None

    def as_provenance(self) -> dict:
        """A JSON-safe record for the run's provenance and the report."""
        return {
            "status": "acquired" if self.succeeded else "not_acquired",
            "offline": self.offline,
            "candidates_found": self.candidates_found,
            "acquired": (
                None
                if self.acquired is None
                else {
                    "host": self.acquired.source.host,
                    "cited_url": self.acquired.source.cited_url,
                    "download_url": self.acquired.source.download_url,
                    "path": self.acquired.path.name,
                    "sha256": self.acquired.sha256,
                    "bytes": self.acquired.bytes_written,
                    "truncated": self.acquired.truncated,
                }
            ),
            "attempts": [
                {
                    "host": p.source.host,
                    "cited_url": p.source.cited_url,
                    "status": p.status,
                    "reason": p.reason,
                    "http_code": p.http_code,
                }
                for p in self.probes
            ],
        }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _normalize_url(raw: str) -> str:
    """Trim prose and markup that a converter glued onto a URL.

    Trailing punctuation only. A mangled PATH is left exactly as found, so a
    corrupted candidate fails its probe instead of being silently "fixed"
    into some other real dataset."""
    url = raw.strip()
    while url and url[-1] in _URL_TRAILING_JUNK:
        url = url[:-1]
    return url


def extract_cited_urls(text: str) -> list[str]:
    """Every URL in the paper text, normalized, de-duplicated, in order."""
    seen: set[str] = set()
    urls: list[str] = []
    for match in _URL_RE.finditer(text):
        url = _normalize_url(match.group(0))
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def classify_source(url: str) -> DatasetSource | None:
    """Match one URL against the data-host whitelist. None means "not a data
    host", which is the correct answer for an arXiv or DOI citation."""
    for rule in DATA_HOST_RULES:
        match = rule.pattern.match(url)
        if match is None:
            continue
        return DatasetSource(
            host=rule.host,
            cited_url=url,
            download_url=rule.resolve(url, match),
            credentialed=rule.credentialed,
        )
    return None


def candidate_sources(text: str) -> list[DatasetSource]:
    """Whitelisted data sources cited by the paper, in citation order.

    De-duplicated by resolved download target, so the motivating paper's
    twenty-odd repetitions of one footnote URL collapse to a single
    candidate."""
    out: list[DatasetSource] = []
    seen: set[str] = set()
    for url in extract_cited_urls(text):
        source = classify_source(url)
        if source is None or source.download_url in seen:
            continue
        seen.add(source.download_url)
        out.append(source)
    return out


# ---------------------------------------------------------------------------
# Probing and fetching
# ---------------------------------------------------------------------------

Opener = Callable[[str], object]
"""Injected network transport: url -> a file-like response exposing
`status`/`getcode()`, `headers`, and `read(n)`. Injected so tests never
touch the network."""


def _default_opener(url: str):  # pragma: no cover - exercised live, not in tests
    request = urllib.request.Request(
        url, headers={"User-Agent": "r2c-dataset-acquisition/1.0"}
    )
    return urllib.request.urlopen(request, timeout=60)


def is_offline() -> bool:
    return os.environ.get(OFFLINE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _header(response: object, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    return getter(name) if callable(getter) else None


def _classify_http_failure(code: int) -> tuple[str, str]:
    """Map an HTTP failure to a status plus an honest detail clause.

    Only 401 is read as "needs credentials". A 403 is deliberately NOT: Kaggle
    answers 403 for a dataset that does not exist, which is exactly what a
    mangled citation resolves to (verified against the real
    probabilistic-demand-forecasting footnote, whose corrupted tail 403s). The
    two causes are indistinguishable from the response, so the reason states
    the ambiguity rather than asserting the wrong one in a delivered report."""
    if code == 401:
        return "needs_credentials", ", which requires authentication"
    if code == 403:
        return (
            "unavailable",
            ", so it is not anonymously accessible. The dataset may be "
            "private or withdrawn, or the cited URL may be wrong (a citation "
            "mangled by PDF conversion resolves this way)",
        )
    return "unavailable", ""


def _status_of(response: object) -> int | None:
    status = getattr(response, "status", None)
    if isinstance(status, int):
        return status
    getcode = getattr(response, "getcode", None)
    if callable(getcode):
        code = getcode()
        return code if isinstance(code, int) else None
    return None


def probe_source(
    source: DatasetSource,
    *,
    opener: Opener | None = None,
    offline: bool | None = None,
) -> ProbeResult:
    """Establish anonymously whether one candidate is actually fetchable."""
    if offline if offline is not None else is_offline():
        return ProbeResult(
            source=source,
            status="refused_offline",
            reason=(
                f"{OFFLINE_ENV} is set, so no source was contacted. Provide a "
                f"local dataset file, or unset {OFFLINE_ENV} to allow the "
                f"paper's own cited source to be fetched."
            ),
        )
    if source.credentialed:
        return ProbeResult(
            source=source,
            status="refused_host",
            reason=(
                f"{source.host} requires an account and, for competitions, "
                f"rule acceptance, so it cannot be fetched unattended. The "
                f"paper cites it at {source.cited_url}; a researcher with "
                f"credentials can download it manually."
            ),
        )

    open_url = opener or _default_opener
    try:
        response = open_url(source.download_url)
    except urllib.error.HTTPError as exc:
        code = int(getattr(exc, "code", 0) or 0)
        status, detail = _classify_http_failure(code)
        return ProbeResult(
            source=source,
            status=status,
            reason=f"{source.download_url} answered HTTP {code}{detail}",
            http_code=code,
        )
    except Exception as exc:  # URLError, socket timeouts, DNS failures
        return ProbeResult(
            source=source,
            status="unavailable",
            reason=f"{source.download_url} could not be reached: {exc}",
        )

    code = _status_of(response)
    content_type = (_header(response, "Content-Type") or "").split(";")[0].strip()
    raw_length = _header(response, "Content-Length")
    try:
        content_length = int(raw_length) if raw_length is not None else None
    except ValueError:
        content_length = None
    closer = getattr(response, "close", None)
    if callable(closer):
        closer()

    if code is not None and code >= 400:
        status, detail = _classify_http_failure(code)
        return ProbeResult(
            source=source,
            status=status,
            reason=f"{source.download_url} answered HTTP {code}{detail}",
            http_code=code,
            content_type=content_type or None,
        )
    if content_type in _HTML_CONTENT_TYPES:
        return ProbeResult(
            source=source,
            status="unavailable",
            reason=(
                f"{source.download_url} returned {content_type}, which is a "
                f"web page rather than a dataset (a login wall or an error "
                f"page answering 200)"
            ),
            http_code=code,
            content_type=content_type,
        )
    return ProbeResult(
        source=source,
        status="ok",
        reason=f"{source.download_url} is anonymously fetchable",
        http_code=code,
        content_type=content_type or None,
        content_length=content_length,
    )


def _filename_for(source: DatasetSource, response: object) -> str:
    disposition = _header(response, "Content-Disposition") or ""
    match = re.search(r'filename="?([^";]+)"?', disposition)
    if match:
        candidate = Path(match.group(1)).name
        if candidate:
            return candidate
    content_type = (_header(response, "Content-Type") or "").split(";")[0].strip()
    suffix = {
        "application/zip": ".zip",
        "application/gzip": ".gz",
        "application/x-gzip": ".gz",
        "text/csv": ".csv",
        "application/json": ".json",
    }.get(content_type, ".bin")
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.cited_url.rstrip("/").split("/")[-1])
    return f"{stem or 'dataset'}{suffix}"


def fetch_source(
    source: DatasetSource,
    dest_dir: Path,
    *,
    opener: Opener | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    chunk_size: int = 1024 * 1024,
) -> Acquisition:
    """Stream one source to `dest_dir` under a byte ceiling, with a checksum.

    Streaming with a hard ceiling matters: the size a server promises and the
    size it sends are different facts, and a run directory is not the place to
    discover a multi-gigabyte archive. Hitting the ceiling is recorded as
    `truncated` rather than raised, so the caller can disclose a partial
    fetch instead of losing the work."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    open_url = opener or _default_opener
    response = open_url(source.download_url)
    path = dest_dir / _filename_for(source, response)

    digest = hashlib.sha256()
    written = 0
    truncated = False
    try:
        with path.open("wb") as handle:
            while True:
                remaining = max_bytes - written
                if remaining <= 0:
                    truncated = True
                    break
                chunk = response.read(min(chunk_size, remaining))
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                written += len(chunk)
    finally:
        closer = getattr(response, "close", None)
        if callable(closer):
            closer()

    return Acquisition(
        source=source,
        path=path,
        sha256=digest.hexdigest(),
        bytes_written=written,
        truncated=truncated,
    )


def acquire_for_paper(
    paper_text: str,
    dest_dir: Path | None,
    *,
    opener: Opener | None = None,
    offline: bool | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    probe_only: bool = False,
) -> AcquisitionReport:
    """Try the paper's own cited data sources in citation order.

    Returns as soon as one is fetched. Every rejection is retained in the
    report so the delivery can state what was attempted, which is what makes
    a synthetic fallback honest rather than silent."""
    sources = candidate_sources(paper_text)
    resolved_offline = offline if offline is not None else is_offline()
    report = AcquisitionReport(
        candidates_found=len(sources), offline=resolved_offline
    )

    for source in sources:
        probe = probe_source(source, opener=opener, offline=resolved_offline)
        report.probes.append(probe)
        if not probe.usable or probe_only:
            continue
        if dest_dir is None:
            continue
        report.acquired = fetch_source(
            source, dest_dir, opener=opener, max_bytes=max_bytes
        )
        break

    return report


# ---------------------------------------------------------------------------
# Demo-bundle materialization (R2C-052 part 2)
# ---------------------------------------------------------------------------

# Tabular member types worth carrying into a demo bundle. Everything else in
# an archive (images, models, docs) is skipped and named in the provenance,
# so a partial bundle never silently reads as the whole dataset.
_TABULAR_SUFFIXES = (".csv", ".tsv", ".json")
# A fetched single-file bundle must be consumable by the file-only provisional
# and forecasting loaders. Unknown binary payloads are acquisition attempts,
# not usable demo data merely because they fit under the byte ceiling.
_SINGLE_FILE_SUFFIXES = _TABULAR_SUFFIXES + (".npz",)

PROVENANCE_FILENAME = "PROVENANCE.json"
PUBLIC_TRANSACTION_FILENAME = ".R2C_PUBLIC_BUNDLE_TRANSACTION.json"
PUBLIC_TRANSACTION_SCHEMA_VERSION = "1.0.0"
PUBLIC_MATERIALIZER_ID = "paper_cited_public_bundle"
PUBLIC_MATERIALIZER_VERSION = "1.0.0"

DEFAULT_MAX_ROWS_PER_TABLE = 50_000

# A table with a time axis has to fund TWO dimensions out of one budget: the
# time span the protocol consumes and enough entities for the method to be
# worth demonstrating. The row-exchangeable budget cannot do both. At the
# motivating table's density (roughly 12,500 rows per date across 1,036
# dates) 50,000 rows buys the full span at four articles, which runs the
# protocol and makes the paper's article graph trivial. Ten times that buys
# the same span at a few dozen articles for about 27 MB on disk, which is
# the trade worth making: the span cannot be substituted for, and the
# entity count is a disclosed demo-scale reduction.
AXIS_ROW_BUDGET_MULTIPLIER = 10

DEFAULT_MAX_MEMBER_BYTES = 64 * 1024 * 1024
_BUNDLE_METADATA_NAMES = frozenset({
    "provenance.json",
    "readme.md",
    PUBLIC_TRANSACTION_FILENAME.casefold(),
})


class PublicBundleRecoveryError(ValueError):
    """An interrupted public publication cannot be changed safely."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _safe_bundle_name(member_name: str, taken: set[str]) -> str | None:
    """Flatten an archive member path to a safe basename.

    Archives nest freely (the motivating dataset stores its main table at
    `sales.csv/sales.csv`), and member paths are attacker-shaped input, so
    only the basename is ever used. Absolute paths and traversal segments
    are refused outright; a basename collision gets a numeric prefix."""
    if member_name.startswith(("/", "\\")) or ".." in member_name.split("/"):
        return None
    base = member_name.rstrip("/").split("/")[-1]
    if (
        not base
        or base.startswith(".")
        or base.casefold() in _BUNDLE_METADATA_NAMES
    ):
        return None
    candidate = base
    counter = 2
    while candidate in taken:
        candidate = f"{counter}_{base}"
        counter += 1
    taken.add(candidate)
    return candidate


def _row_capped_copy(reader, dest: Path, *, max_rows: int) -> tuple[int, bool]:
    """Stream text lines from `reader` to `dest`, keeping the header plus at
    most `max_rows` data rows. Returns (rows_written, truncated). Streaming
    matters: the motivating table is 1.06 GB uncompressed and must never be
    loaded whole."""
    rows = 0
    truncated = False
    with dest.open("w", encoding="utf-8", newline="") as out:
        for i, line in enumerate(reader):
            if i > max_rows:  # header + max_rows data lines
                truncated = True
                break
            out.write(line)
            if i > 0:
                rows += 1
    return rows, truncated


# ---------------------------------------------------------------------------
# Time-axis-aware subsampling (R2C-065)
# ---------------------------------------------------------------------------

# Header tokens that name a table's time axis, weakest rank wins. Panel
# exports routinely carry component columns beside a real timestamp
# (`year`, `month` next to `date`), so the ranking makes the real axis win
# rather than whichever column happens to sit leftmost.
_TIME_AXIS_TOKENS = {
    "date": 0, "datetime": 0, "timestamp": 0, "ds": 0,
    "time": 1, "dt": 1,
    "week": 2, "yearweek": 2, "weekofyear": 2,
    "month": 3, "period": 3, "quarter": 3, "epoch": 3,
    "year": 4,
}

_TIME_VALUE_RE = re.compile(
    r"""^\s*(
        \d{4}[-/]\d{1,2}([-/]\d{1,2})?      # 2017-01-02, 2017/1, 2017-01
      | \d{1,2}[-/]\d{1,2}[-/]\d{2,4}       # 02-01-2017
      | \d{4}(0[1-9]|1[0-2])(\d{2})?        # 201701, 20170102
      | \d{1,2}:\d{2}(:\d{2})?              # 13:45:00
      | \d{9,13}                            # unix seconds / milliseconds
    )""",
    re.VERBOSE,
)

# A time column whose distinct-value count runs past this is not a protocol
# axis at a useful granularity (a free-text note column that happens to be
# named `time`, or microsecond timestamps). Scanning stops and the table
# falls back to the plain row cap rather than holding an unbounded set.
_MAX_SCANNED_STEPS = 500_000


@dataclass
class AxisFacts:
    """What the time axis of one bundled table actually turned out to be.

    Recorded in provenance and read by the dispatch briefings, because every
    consumer downstream of the bundle needs the realized axis rather than the
    axis the paper describes: the 2026-08-05 pdfgnn roll shipped 4 daily
    steps against a protocol needing 36-plus, and the notebook called them
    "weekly demand" because nothing had ever stated otherwise."""

    column: str
    steps_kept: int
    steps_in_source: int
    first_step: str
    last_step: str
    rows_per_step_cap: int | None
    # Distinct values of the table's leading column among the kept rows —
    # the entity breadth the demo actually has. Thinning the entity
    # dimension is how the time span survives, so how MUCH was thinned is a
    # fact the delivery has to be able to state.
    entity_column: str = ""
    entities_kept: int = 0
    entities_in_source: int = 0
    # Exact sets are materializer-internal.  Counts enter provenance, while
    # the values themselves exist only long enough to propagate an entity
    # subsample to the other tables that carry the same named key (R2C-078).
    retained_entities: frozenset[str] = field(default_factory=frozenset,
                                               repr=False)
    source_entities: frozenset[str] = field(default_factory=frozenset,
                                             repr=False)
    # The LIVE extent (R2C-069). A long axis is not the same as a usable one:
    # a public forecasting dataset normally withholds its target over the
    # final period, because that period is the competition's own prediction
    # window. The 2026-08-06 pdfgnn bundle had 1,092 daily steps of which the
    # last 59 carry no `sales` value, the demo held out its last four weeks
    # inside that region, and every reported metric printed nan.
    #
    # `live_steps` / `last_live_step` are the axis a protocol can actually
    # use: the earliest point at which some column stops carrying values. When
    # no column has a dead tail they equal steps_kept / last_step.
    live_steps: int = 0
    last_live_step: str = ""
    # Per-column detail, so a consumer that knows WHICH column it treats as
    # the target can reason precisely instead of taking the table-wide
    # minimum. A column empty everywhere is not listed here — it has no data
    # at all, which the missing-count briefing already states.
    dead_tail_columns: list[dict] = field(default_factory=list)
    # Exact cadence of the SOURCE axis when every distinct timestamp parses
    # and the sorted intervals are regular.  Absence is meaningful: later
    # protocol checks must report an unresolved unit join instead of treating
    # a raw row count as though it were already on the paper's axis.
    source_cadence: dict[str, int | float | str] | None = None

    def as_dict(self) -> dict:
        facts = {
            "column": self.column,
            "steps_kept": self.steps_kept,
            "steps_in_source": self.steps_in_source,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "rows_per_step_cap": self.rows_per_step_cap,
            "entity_column": self.entity_column,
            "entities_kept": self.entities_kept,
            "entities_in_source": self.entities_in_source,
            "live_steps": self.live_steps,
            "last_live_step": self.last_live_step,
            "dead_tail_columns": list(self.dead_tail_columns),
        }
        if self.source_cadence is not None:
            facts["source_cadence"] = dict(self.source_cadence)
        return facts


def _header_tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", name.strip().lower()) if t}


def _looks_like_time_point(value: str) -> bool:
    return bool(value) and bool(_TIME_VALUE_RE.match(value))


def detect_time_axis_column(header: list[str], first_row: list[str]) -> int | None:
    """Index of the column that carries this table's time axis, or None.

    Both halves are required: the NAME must carry a time token (on token
    boundaries, so `promo_discount_type_2` never matches `date`) and the
    first data VALUE must parse as a time point (so a free-text column
    named `update_time` is refused). Ties break on the token rank, then
    leftmost."""
    best: tuple[int, int] | None = None
    for index, name in enumerate(header):
        ranks = [_TIME_AXIS_TOKENS[t] for t in _header_tokens(name)
                 if t in _TIME_AXIS_TOKENS]
        if not ranks:
            continue
        if index >= len(first_row) or not _looks_like_time_point(first_row[index]):
            continue
        candidate = (min(ranks), index)
        if best is None or candidate < best:
            best = candidate
    return None if best is None else best[1]


def _csv_dialect_delimiter(suffix: str) -> str:
    return "\t" if suffix.lower() == ".tsv" else ","


@dataclass
class _AxisScan:
    column: int
    column_name: str
    steps: int
    data_rows: int
    entity_column: int
    entity_column_name: str
    entities: frozenset[str]
    source_cadence: dict[str, int | float | str] | None


def _parse_axis_timestamp(raw: str) -> datetime | None:
    """Parse one timestamp conservatively for exact cadence inference.

    This is deliberately narrower than a general date parser.  A guessed
    timestamp would manufacture unit compatibility, while an unrecognized
    timestamp merely leaves the later feasibility verdict unresolved.
    """
    value = raw.strip()
    if not value:
        return None
    if re.fullmatch(r"(?:\d{10}|\d{13})", value):
        try:
            seconds = int(value)
            if len(value) == 13:
                seconds /= 1000
            return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(
                tzinfo=None
            )
        except (OverflowError, OSError, ValueError):
            return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except ValueError:
        pass
    # Deliberately exclude day/month-order forms and month-only values.  Both
    # can be parsed, but neither identifies an exact fixed source cadence.
    for fmt in ("%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        parsed_time = time.fromisoformat(value)
    except ValueError:
        return None
    parsed = datetime.combine(datetime(1970, 1, 1).date(), parsed_time)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _regular_source_cadence(
    values: Iterable[str],
) -> dict[str, int | float | str] | None:
    """Return an exact fixed cadence, or None for ambiguous/irregular axes."""
    parsed: set[datetime] = set()
    for raw in values:
        timestamp = _parse_axis_timestamp(raw)
        if timestamp is None:
            return None
        parsed.add(timestamp)
    ordered = sorted(parsed)
    if len(ordered) < 2:
        return None
    deltas = [
        (right - left).total_seconds()
        for left, right in zip(ordered, ordered[1:])
    ]
    if any(delta <= 0 for delta in deltas) or any(
        delta != deltas[0] for delta in deltas[1:]
    ):
        return None
    seconds = deltas[0]
    for unit, scale in (
        ("week", 7 * 24 * 60 * 60),
        ("day", 24 * 60 * 60),
        ("hour", 60 * 60),
        ("minute", 60),
        ("second", 1),
    ):
        granularity = seconds / scale
        if granularity.is_integer():
            return {"unit": unit, "granularity": int(granularity)}
    return {"unit": "second", "granularity": seconds}


def _scan_time_axis(reader, delimiter: str) -> _AxisScan | None:
    """First pass: locate the time column and measure the axis.

    Returns None when the table has no usable time axis. Costs one extra
    decompression of the member, which buys the two numbers no single pass
    can know: how many time steps the source actually has, and therefore how
    many rows per step fit the budget while keeping the WHOLE axis. The
    2026-08-05 roll is why that trade is worth making — a cheaper fixed
    per-step cap would have kept 57 weeks of a 148-week paper."""
    rows = csv.reader(reader, delimiter=delimiter)
    try:
        header = next(rows)
        first = next(rows)
    except StopIteration:
        return None
    column = detect_time_axis_column(header, first)
    if column is None:
        return None
    entity_column = 1 if column == 0 else 0
    seen: set[str] = {first[column] if column < len(first) else ""}
    entities: set[str] = set()
    if entity_column < len(first) and first[entity_column]:
        entities.add(first[entity_column])
    total = 1
    for row in rows:
        total += 1
        if column < len(row):
            seen.add(row[column])
            if len(seen) > _MAX_SCANNED_STEPS:
                return None
        if entity_column < len(row) and row[entity_column]:
            entities.add(row[entity_column])
    name = header[column] if column < len(header) else f"column {column}"
    entity_name = (header[entity_column]
                   if entity_column < len(header) else f"column {entity_column}")
    return _AxisScan(column=column, column_name=name, steps=len(seen),
                     data_rows=total, entity_column=entity_column,
                     entity_column_name=entity_name,
                     entities=frozenset(entities),
                     source_cadence=_regular_source_cadence(seen))


def _axis_capped_copy(
    reader,
    dest: Path,
    *,
    delimiter: str,
    scan: _AxisScan,
    rows_per_step: int | None,
    max_steps: int | None,
) -> tuple[int, bool, AxisFacts]:
    """Second pass: write the header plus a time-axis-preserving subsample.

    The row cap this replaces treated every table as row-exchangeable, which
    is exactly wrong for a long-format panel sorted by date: its first N rows
    are its first few dates, so volume survives and the axis the method
    consumes is destroyed. Here the ENTITY dimension is thinned instead
    (`rows_per_step` rows per time step) so the full time span survives,
    which is the dimension a forecasting protocol cannot substitute for.

    `rows_per_step` of None keeps every row; `max_steps` of None keeps every
    step. Rows are kept by position within their step, which holds the entity
    set roughly stable because panel exports sort consistently inside a step,
    and gaps that remain are reported by the missingness briefing rather than
    silently repaired."""
    kept = 0
    truncated = False
    per_step: dict[str, int] = {}
    order: list[str] = []
    entity_column = scan.entity_column
    entities: set[str] = set()
    header: list[str] = []
    # Last axis ordinal at which each column carried a non-blank value, over
    # the rows actually WRITTEN. Measured here rather than by a later sampling
    # pass because this loop already visits every kept row, and because a head
    # sample cannot see a withheld tail (R2C-069). Keyed by the row's OWN step
    # rather than by insertion position, so an unsorted export measures the
    # same as a sorted one.
    last_live: dict[int, int] = {}
    ordinals: dict[str, int] = {}
    with dest.open("w", encoding="utf-8", newline="") as fh:
        out = csv.writer(fh, delimiter=delimiter, lineterminator="\n")
        rows = csv.reader(reader, delimiter=delimiter)
        try:
            header = next(rows)
            out.writerow(header)
        except StopIteration:  # pragma: no cover — the scan already read two rows
            return 0, False, AxisFacts(scan.column_name, 0, scan.steps, "", "", None)
        for row in rows:
            step = row[scan.column] if scan.column < len(row) else ""
            if step not in per_step:
                if max_steps is not None and len(per_step) >= max_steps:
                    truncated = True
                    continue
                per_step[step] = 0
                ordinals[step] = len(order)
                order.append(step)
            if rows_per_step is not None and per_step[step] >= rows_per_step:
                truncated = True
                continue
            per_step[step] += 1
            if entity_column < len(row) and row[entity_column]:
                entities.add(row[entity_column])
            ordinal = ordinals[step]
            for index, cell in enumerate(row):
                if cell.strip() and last_live.get(index, -1) < ordinal:
                    last_live[index] = ordinal
            out.writerow(row)
            kept += 1
    live_steps, last_live_step, dead_tails = _live_extent(
        header, order, last_live, axis_column=scan.column)
    return kept, truncated, AxisFacts(
        column=scan.column_name,
        steps_kept=len(order),
        steps_in_source=scan.steps,
        first_step=order[0] if order else "",
        last_step=order[-1] if order else "",
        rows_per_step_cap=rows_per_step,
        entity_column=scan.entity_column_name,
        entities_kept=len(entities),
        entities_in_source=len(scan.entities),
        retained_entities=frozenset(entities),
        source_entities=scan.entities,
        live_steps=live_steps,
        last_live_step=last_live_step,
        dead_tail_columns=dead_tails,
        source_cadence=scan.source_cadence,
    )


def _live_extent(
    header: list[str], order: list[str], last_live: dict[int, int],
    *, axis_column: int,
) -> tuple[int, str, list[dict]]:
    """The bundled table's usable axis, plus the columns that end early.

    A column has a DEAD TAIL when it carries values somewhere and then stops
    before the axis does. That is the withheld-target signature of a public
    forecasting dataset, and it is structural rather than random: the tail is
    a contiguous suffix by construction, since the boundary is the column's
    own last live step.

    A column absent from `last_live` is empty everywhere. It has no dead tail
    because it has no data at all, and the briefing's missing-count line
    already says so, so it must not drag the table's usable extent to zero.

    The table-wide answer is the EARLIEST dead-tail boundary, because the
    materializer does not know which column the method will treat as its
    target and any of them could be it. Consumers that do know read
    `dead_tail_columns` instead of taking this minimum.

    "Tail" means tail of `order`, which is the order the steps appear in the
    file. That is the same assumption `first_step` and `last_step` already
    make, and it holds for a panel export. When the observed order is NOT
    non-decreasing the file is not chronological, position tells us nothing
    about lateness, and we report no dead tail at all rather than a boundary
    that would read as fact. Fail-safe: a missed withheld tail costs the
    briefing a warning, an invented one would send a producer to trim live
    data.
    """
    if not order:
        return 0, "", []
    last_ordinal = len(order) - 1
    if any(b < a for a, b in zip(order, order[1:])):
        return len(order), order[last_ordinal], []
    dead_tails: list[dict] = []
    for index, name in enumerate(header):
        if index == axis_column:
            continue
        ordinal = last_live.get(index)
        if ordinal is None or ordinal >= last_ordinal:
            continue
        dead_tails.append({
            "column": name,
            "last_live_step": order[ordinal],
            "live_steps": ordinal + 1,
            "dead_tail_steps": last_ordinal - ordinal,
        })
    if not dead_tails:
        return len(order), order[last_ordinal], []
    earliest = min(d["live_steps"] for d in dead_tails)
    return earliest, order[earliest - 1], dead_tails


def _axis_aware_copy(
    open_member: Callable[[], Iterable[str]],
    dest: Path,
    *,
    suffix: str,
    max_rows: int,
    axis_max_rows: int | None = None,
) -> tuple[int, bool, AxisFacts | None]:
    """Copy one tabular member, preserving its time axis when it has one.

    `open_member` returns a fresh text stream each call, because the scan and
    the write are two passes over the same member. Falls back to the plain
    row cap (returning axis facts of None) for a table with no time axis OR
    with a single time step, which keeps every non-time-series paradigm
    byte-identical: a table whose rows all share one timestamp has no axis to
    preserve and is row-exchangeable like any other.

    A real axis table gets `axis_max_rows` (default: the row budget times
    `AXIS_ROW_BUDGET_MULTIPLIER`), because it has to fund the time span and
    the entity breadth at once."""
    delimiter = _csv_dialect_delimiter(suffix)
    scan = _scan_time_axis(open_member(), delimiter)
    if scan is None or scan.steps <= 1:
        return (*_row_capped_copy(open_member(), dest, max_rows=max_rows), None)

    budget = (max_rows * AXIS_ROW_BUDGET_MULTIPLIER if axis_max_rows is None
              else axis_max_rows)
    if scan.data_rows <= budget:
        rows_per_step: int | None = None
        max_steps: int | None = None
    elif scan.steps > budget:
        # More time steps than the whole budget: one row each, first N steps.
        rows_per_step, max_steps = 1, budget
    else:
        rows_per_step, max_steps = max(1, budget // scan.steps), None

    return _axis_capped_copy(
        open_member(), dest, delimiter=delimiter, scan=scan,
        rows_per_step=rows_per_step, max_steps=max_steps,
    )


def _subsample_rule(files: list[dict], max_rows: int) -> str:
    """One sentence stating how each table was cut, for the provenance and
    every researcher-facing surface that quotes it."""
    axis_tables = [f for f in files if f.get("time_axis")]
    rule = (f"tabular members only, header plus at most {max_rows} data rows "
            f"per table")
    if not axis_tables:
        return rule
    detail = "; ".join(
        f"{f['file']} keeps its full time axis "
        f"({f['time_axis']['steps_kept']} steps on `{f['time_axis']['column']}`"
        + (f", at most {f['time_axis']['rows_per_step_cap']} rows per step"
           if f["time_axis"]["rows_per_step_cap"] else "")
        + (f", {f['time_axis']['entities_kept']} distinct "
           f"`{f['time_axis']['entity_column']}` value(s)"
           if f["time_axis"].get("entities_kept") else "")
        + ")"
        for f in axis_tables
    )
    return (f"{rule}, EXCEPT a table with a time axis, which gets "
            f"{max_rows * AXIS_ROW_BUDGET_MULTIPLIER} rows and is thinned "
            f"along its ENTITY dimension so the whole time span survives "
            f"(the span is what a temporal protocol cannot substitute for; "
            f"the entity count is a demo-scale reduction) — {detail}")


def _usable_extent_note(files: list[dict]) -> str:
    """Where the bundled data stops being usable, in prose (R2C-069).

    The structured facts live on each file's `time_axis`; this is the line a
    researcher reads, since `example_data/README.md` sends them to
    PROVENANCE.json for exactly this kind of fact. Empty when every column
    runs to the end of its axis, which is the ordinary case."""
    parts: list[str] = []
    for entry in files:
        axis = entry.get("time_axis")
        if not isinstance(axis, dict):
            continue
        dead = axis.get("dead_tail_columns") or []
        if not dead:
            continue
        detail = ", ".join(
            f"`{d.get('column')}` after {d.get('last_live_step')} "
            f"({d.get('dead_tail_steps')} step(s))"
            for d in dead if isinstance(d, dict)
        )
        parts.append(
            f"{entry.get('file')} spans {axis.get('steps_kept')} steps "
            f"({axis.get('first_step')} through {axis.get('last_step')}) but "
            f"carries no values for {detail}"
        )
    if not parts:
        return ""
    return (
        "; ".join(parts)
        + ". A public forecasting dataset normally withholds its target over "
          "the final period, because that period is the source's own "
          "prediction window. Metrics must be computed inside the live "
          "extent: scoring against the withheld tail compares predictions "
          "with missing values and yields nan."
    )


def _member_opener(archive, member) -> Callable[[], Iterable[str]]:
    """A re-openable text stream over one archive member (two-pass reads)."""
    return lambda: io.TextIOWrapper(
        archive.open(member), encoding="utf-8", errors="replace")


def _file_opener(path: Path) -> Callable[[], Iterable[str]]:
    return lambda: path.open("r", encoding="utf-8", errors="replace")


def _bundle_entry(name: str, source_member: str, rows: int, truncated: bool,
                  axis: AxisFacts | None) -> dict:
    entry = {"file": name, "source_member": source_member,
             "rows_kept": rows, "truncated": truncated}
    if axis is not None:
        entry["time_axis"] = axis.as_dict()
        if axis.entity_column:
            entry.update({
                "entity_column": axis.entity_column,
                "entities_kept": axis.entities_kept,
                "entities_in_source": axis.entities_in_source,
                "entity_filter_applied": (
                    axis.entities_kept < axis.entities_in_source),
            })
    return entry


class BundleReferentialIntegrityError(ValueError):
    """A materializer-created entity mismatch that makes joins unsafe."""


@dataclass
class _MaterializedTable:
    """One CSV/TSV while its original source stream is still available."""

    entry: dict
    target: Path
    delimiter: str
    open_source: Callable[[], Iterable[str]]
    axis: AxisFacts | None


def _column_values(reader, *, delimiter: str,
                   column: str) -> set[str] | None:
    """Distinct non-blank values for a named column, or None if absent."""
    rows = csv.reader(reader, delimiter=delimiter)
    try:
        header = next(rows)
    except StopIteration:
        return None
    try:
        index = header.index(column)
    except ValueError:
        return None
    return {
        row[index] for row in rows
        if index < len(row) and row[index]
    }


def _path_column_values(path: Path, *, column: str) -> set[str] | None:
    if path.suffix.lower() not in (".csv", ".tsv") or not path.is_file():
        return None
    delimiter = _csv_dialect_delimiter(path.suffix)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return _column_values(fh, delimiter=delimiter, column=column)


def _filter_materialized_table(
    table: _MaterializedTable, *, column: str, retained: frozenset[str],
) -> bool:
    """Filter one bundled table to an entity set and record what changed.

    The source stream supplies the pre-materialization count.  The already
    bounded destination supplies the rows to keep, so propagation never
    defeats the table's existing row budget.  If that earlier cap omitted a
    required entity, the referential floor below refuses the bundle instead
    of silently inventing rows or widening the budget.
    """
    if table.axis is not None and table.axis.entity_column == column:
        source_values = set(table.axis.source_entities)
    else:
        source_reader = table.open_source()
        try:
            source_values = _column_values(
                source_reader, delimiter=table.delimiter, column=column)
        finally:
            close = getattr(source_reader, "close", None)
            if close is not None:
                close()
        if source_values is None:
            return False

    temp = table.target.with_name(f".{table.target.name}.entity-filter")
    temp.unlink(missing_ok=True)
    kept_values: set[str] = set()
    kept_count = 0
    try:
        with table.target.open(
            "r", encoding="utf-8", errors="replace", newline="",
        ) as source, temp.open(
            "w", encoding="utf-8", newline="",
        ) as destination:
            rows = csv.reader(source, delimiter=table.delimiter)
            try:
                header = next(rows)
            except StopIteration:
                return False
            try:
                index = header.index(column)
            except ValueError:
                return False
            writer = csv.writer(
                destination, delimiter=table.delimiter, lineterminator="\n")
            writer.writerow(header)
            for row in rows:
                if index >= len(row) or row[index] not in retained:
                    continue
                writer.writerow(row)
                kept_count += 1
                if row[index]:
                    kept_values.add(row[index])
        temp.replace(table.target)
    finally:
        temp.unlink(missing_ok=True)

    previous_rows = table.entry.get("rows_kept")
    table.entry.update({
        "rows_kept": kept_count,
        "truncated": bool(table.entry.get("truncated"))
                     or previous_rows != kept_count,
        "entity_column": column,
        "entities_kept": len(kept_values),
        "entities_in_source": len(source_values),
        "entity_filter_applied": True,
    })
    axis = table.entry.get("time_axis")
    if isinstance(axis, dict) and axis.get("entity_column") == column:
        axis["entities_kept"] = len(kept_values)
        axis["entities_in_source"] = len(source_values)
    return True


def bundle_referential_integrity_error(
    files: list[dict], dest_dir: Path,
) -> str | None:
    """Return the first materializer-created shared-key mismatch.

    Equality is required only for an entity key the materializer actually
    subsampled and propagated.  A source dimension table may legitimately
    be broader than a fact table, so source relationships untouched by this
    materialization are outside this floor.  Once the materializer thins an
    entity axis, every bundled table carrying that named key must expose the
    same retained set or the smaller artifact is no longer joinable.
    """
    by_column: dict[str, list[tuple[str, set[str]]]] = {}
    active_columns = {
        str(entry.get("entity_column"))
        for entry in files
        if entry.get("entity_filter_applied") and entry.get("entity_column")
    }
    for entry in files:
        column = entry.get("entity_column")
        name = entry.get("file")
        if column not in active_columns or not name:
            continue
        values = _path_column_values(dest_dir / name, column=column)
        if values is not None:
            by_column.setdefault(column, []).append((name, values))

    for column in sorted(by_column):
        tables = sorted(by_column[column], key=lambda item: item[0])
        if len(tables) < 2:
            continue
        baseline_name, baseline_values = tables[0]
        for name, values in tables[1:]:
            if values == baseline_values:
                continue
            return (
                "bundle referential integrity failed after entity "
                f"subsampling: shared key `{column}` has "
                f"{len(baseline_values)} distinct value(s) in "
                f"`{baseline_name}` and {len(values)} distinct value(s) in "
                f"`{name}`; "
                "the acquisition is refused"
            )
    return None


def _propagate_entity_subsamples(
    tables: list[_MaterializedTable], dest_dir: Path,
) -> None:
    """Propagate every actual entity reduction, then enforce the floor."""
    retained_by_column: dict[str, tuple[str, frozenset[str]]] = {}
    for table in tables:
        axis = table.axis
        if (axis is None or not axis.entity_column
                or axis.entities_kept >= axis.entities_in_source):
            continue
        previous = retained_by_column.get(axis.entity_column)
        if previous is not None and previous[1] != axis.retained_entities:
            raise BundleReferentialIntegrityError(
                "bundle referential integrity failed before propagation: "
                f"entity-subsampled tables `{previous[0]}` and "
                f"`{table.entry['file']}` retained different "
                f"`{axis.entity_column}` sets "
                f"({len(previous[1])} and {axis.entities_kept} distinct "
                "value(s)); the acquisition is refused"
            )
        retained_by_column[axis.entity_column] = (
            table.entry["file"], axis.retained_entities)

    for column, (_, retained) in sorted(retained_by_column.items()):
        for table in tables:
            _filter_materialized_table(
                table, column=column, retained=retained)

    mismatch = bundle_referential_integrity_error(
        [table.entry for table in tables], dest_dir)
    if mismatch:
        raise BundleReferentialIntegrityError(mismatch)


def _remove_materialized_files(files: list[dict], dest_dir: Path) -> None:
    """Keep a refused acquisition atomic from downstream consumers."""
    for entry in files:
        name = entry.get("file")
        if name:
            (dest_dir / name).unlink(missing_ok=True)
    (dest_dir / PROVENANCE_FILENAME).unlink(missing_ok=True)


def _public_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _public_bundle_names(manifest: dict) -> list[str]:
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("public bundle manifest files must be a list")
    names: list[str] = []
    seen: set[str] = set()
    for entry in raw_files:
        name = entry.get("file") if isinstance(entry, dict) else None
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name.startswith(".")
            or name.casefold() in _BUNDLE_METADATA_NAMES
            or Path(name).suffix.lower() not in _SINGLE_FILE_SUFFIXES
            or name.casefold() in seen
        ):
            raise ValueError(f"unsafe or duplicate public bundle file {name!r}")
        seen.add(name.casefold())
        names.append(name)
    return names


def _public_transaction_record(stage: Path, manifest: dict) -> dict:
    names = _public_bundle_names(manifest) + [PROVENANCE_FILENAME]
    return {
        "schema_version": PUBLIC_TRANSACTION_SCHEMA_VERSION,
        "materializer_id": PUBLIC_MATERIALIZER_ID,
        "materializer_version": PUBLIC_MATERIALIZER_VERSION,
        "files": [
            {"file": name, "sha256": _public_file_sha256(stage / name)}
            for name in names
        ],
    }


def recover_interrupted_public_bundle(destination: Path) -> str | None:
    """Recover only hash-matched output from an interrupted publication.

    The marker is installed after every output is complete in a private stage
    and before the first canonical rename. A full hash-matched publication is
    complete and only needs its stale marker removed. A matching subset is
    materializer-owned and is removed before retry. Changed output is
    preserved and refused so a later synthetic tier cannot mix with it.
    """

    destination = Path(destination)
    marker = destination / PUBLIC_TRANSACTION_FILENAME
    if not marker.exists():
        return None
    if not marker.is_file():
        raise PublicBundleRecoveryError(
            "public_bundle_transaction_marker",
            "interrupted public-bundle marker is not a regular file",
        )
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicBundleRecoveryError(
            "public_bundle_transaction_marker",
            f"interrupted public-bundle marker is unreadable: {exc}",
        ) from exc
    if (
        not isinstance(record, dict)
        or set(record) != {
            "schema_version", "materializer_id", "materializer_version",
            "files",
        }
        or record.get("schema_version") != PUBLIC_TRANSACTION_SCHEMA_VERSION
        or record.get("materializer_id") != PUBLIC_MATERIALIZER_ID
        or record.get("materializer_version") != PUBLIC_MATERIALIZER_VERSION
        or not isinstance(record.get("files"), list)
    ):
        raise PublicBundleRecoveryError(
            "public_bundle_transaction_marker",
            "interrupted public-bundle marker does not match the closed schema",
        )
    expected: dict[str, str] = {}
    seen_casefold: set[str] = set()
    ordered_names: list[str] = []
    for item in record["files"]:
        name = item.get("file") if isinstance(item, dict) else None
        digest = item.get("sha256") if isinstance(item, dict) else None
        if (
            not isinstance(item, dict)
            or set(item) != {"file", "sha256"}
            or not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name.startswith(".")
            or name == PUBLIC_TRANSACTION_FILENAME
            or name.casefold() in seen_casefold
            or (
                name != PROVENANCE_FILENAME
                and (
                    name.casefold() in _BUNDLE_METADATA_NAMES
                    or Path(name).suffix.lower() not in _SINGLE_FILE_SUFFIXES
                )
            )
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise PublicBundleRecoveryError(
                "public_bundle_transaction_marker",
                "interrupted public-bundle marker has an invalid file record",
            )
        expected[name] = digest
        seen_casefold.add(name.casefold())
        ordered_names.append(name)
    if not ordered_names or ordered_names[-1] != PROVENANCE_FILENAME:
        raise PublicBundleRecoveryError(
            "public_bundle_transaction_marker",
            "interrupted public-bundle marker must publish provenance last",
        )
    present = []
    for name, digest in expected.items():
        target = destination / name
        if not target.exists():
            continue
        present.append(name)
        if not target.is_file() or _public_file_sha256(target) != digest:
            raise PublicBundleRecoveryError(
                "public_bundle_transaction_conflict",
                f"refusing to remove changed interrupted file {name!r}",
            )
    try:
        extra_local_data = [
            path for path in destination.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and path.name not in expected
            and path.name != PROVENANCE_FILENAME
            and path.suffix.lower() in _SINGLE_FILE_SUFFIXES
        ]
    except OSError as exc:
        raise PublicBundleRecoveryError(
            "public_bundle_transaction_marker",
            f"interrupted public-bundle directory is unreadable: {exc}",
        ) from exc
    if (
        len(present) == len(expected)
        and len(expected) > 1
        and not extra_local_data
    ):
        marker.unlink()
        return "completed"
    # A provenance-only publication has no loader-consumable public data. If
    # local data appeared during a complete publication, remove the unchanged
    # lower-priority owned bundle so the local tier wins without mixing.
    for name in present:
        (destination / name).unlink()
    marker.unlink()
    return "removed_partial"


def _publish_public_bundle(stage: Path, destination: Path, manifest: dict) -> None:
    names = _public_bundle_names(manifest) + [PROVENANCE_FILENAME]
    conflicts = [name for name in names if (destination / name).exists()]
    if conflicts:
        raise FileExistsError(
            f"refusing to overwrite existing bundle files {conflicts!r}"
        )
    if (destination / PUBLIC_TRANSACTION_FILENAME).exists():
        raise PublicBundleRecoveryError(
            "public_bundle_transaction_conflict",
            "an interrupted public-bundle marker must be recovered first",
        )
    created = not destination.exists()
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / PUBLIC_TRANSACTION_FILENAME
    staged_marker = stage / PUBLIC_TRANSACTION_FILENAME
    published: list[Path] = []
    try:
        staged_marker.write_text(
            json.dumps(
                _public_transaction_record(stage, manifest),
                sort_keys=True,
                separators=(",", ":"),
            ) + "\n",
            encoding="utf-8",
        )
        os.replace(staged_marker, marker)
        for name in names:
            target = destination / name
            if target.exists():
                raise FileExistsError(
                    f"destination appeared during public publish: {target}"
                )
            source = stage / name
            try:
                os.replace(source, target)
            except Exception:
                # Some filesystem failures can be reported after the rename
                # took effect. The vanished private source distinguishes that
                # side effect from a pre-existing destination race.
                if not source.exists() and target.exists():
                    published.append(target)
                raise
            else:
                published.append(target)
        marker.unlink()
    except Exception:
        for target in reversed(published):
            target.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
        if created:
            try:
                destination.rmdir()
            except OSError:
                pass
        raise


def _materialize_demo_bundle_staged(
    acquisition: Acquisition,
    dest_dir: Path,
    *,
    report: AcquisitionReport | None = None,
    max_rows_per_table: int = DEFAULT_MAX_ROWS_PER_TABLE,
    max_rows_per_axis_table: int | None = None,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
) -> dict:
    """Turn a fetched artifact into a validated private staged bundle.

    A fetched archive is not a usable demo dataset: the motivating fetch is a
    90.7 MB zip whose main table is 1.06 GB uncompressed, far past the demo
    calibration target. This extracts only tabular members, row-caps each one
    deterministically (header plus the first N data rows — documented,
    reproducible, no sampling randomness to explain), flattens nested member
    paths, skips everything else by name, and deletes the archive afterward
    so a delivery never carries a gigabyte of raw source.

    Writes `PROVENANCE.json` beside the data: where the data came from (the
    paper's own citation), the original fetch checksum, exactly what was kept
    and what was cut, and the honesty note for downstream report surfaces —
    demo numbers computed on a subsample demonstrate the mechanism and are
    not the paper's benchmark results.

    Returns the manifest dict (the same content as PROVENANCE.json)."""
    import zipfile  # noqa: PLC0415

    dest_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict] = []
    tables: list[_MaterializedTable] = []
    skipped: list[str] = []
    taken: set[str] = set()

    acquisition_suffix = acquisition.path.suffix.lower()
    if acquisition_suffix == ".npz":
        if (
            not acquisition.path.name.startswith(".")
            and acquisition.path.name.casefold() not in _BUNDLE_METADATA_NAMES
            and acquisition.path.stat().st_size <= max_member_bytes
        ):
            target = dest_dir / acquisition.path.name
            files.append({
                "file": target.name,
                "source_member": acquisition.path.name,
                "rows_kept": None,
                "truncated": False,
            })
            try:
                if target != acquisition.path:
                    if target.exists():
                        raise FileExistsError(
                            f"refusing to overwrite existing bundle file {target}"
                        )
                    acquisition.path.replace(target)
            except Exception:
                _remove_materialized_files(files, dest_dir)
                acquisition.path.unlink(missing_ok=True)
                raise
        else:
            skipped.append(acquisition.path.name)
            acquisition.path.unlink()
    elif zipfile.is_zipfile(acquisition.path):
        try:
            with zipfile.ZipFile(acquisition.path) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    suffix = Path(member.filename).suffix.lower()
                    if suffix not in _TABULAR_SUFFIXES:
                        skipped.append(member.filename)
                        continue
                    name = _safe_bundle_name(member.filename, taken)
                    if name is None:
                        skipped.append(member.filename)
                        continue
                    target = dest_dir / name
                    if target.exists():
                        raise FileExistsError(
                            f"refusing to overwrite existing bundle file {target}"
                        )
                    if suffix == ".json":
                        if member.file_size > max_member_bytes:
                            skipped.append(member.filename)
                            continue
                        entry = {
                            "file": name,
                            "source_member": member.filename,
                            "rows_kept": None,
                            "truncated": False,
                        }
                        files.append(entry)
                        target.write_bytes(archive.read(member))
                        continue
                    open_member = _member_opener(archive, member)
                    files.append({"file": name})
                    rows, truncated, axis = _axis_aware_copy(
                        open_member, target, suffix=suffix,
                        max_rows=max_rows_per_table,
                        axis_max_rows=max_rows_per_axis_table)
                    entry = _bundle_entry(
                        name, member.filename, rows, truncated, axis)
                    files[-1] = entry
                    tables.append(_MaterializedTable(
                        entry=entry,
                        target=target,
                        delimiter=_csv_dialect_delimiter(suffix),
                        open_source=open_member,
                        axis=axis,
                    ))
                # Archive order is arbitrary.  Propagation happens only after
                # every retained entity set and every companion header is
                # known, while the original member streams remain available.
                _propagate_entity_subsamples(tables, dest_dir)
        except Exception:
            _remove_materialized_files(files, dest_dir)
            acquisition.path.unlink(missing_ok=True)
            raise
        acquisition.path.unlink(missing_ok=True)
    elif acquisition.path.suffix.lower() in (".csv", ".tsv"):
        target = dest_dir / acquisition.path.name
        source = acquisition.path
        if target == source:
            # Two passes read the source while the write truncates the
            # destination, so an in-place bundle has to move out of the way.
            source = source.with_name(source.name + ".fetched")
            target.rename(source)
        try:
            if target.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing bundle file {target}"
                )
            files.append({"file": target.name})
            open_source = _file_opener(source)
            rows, truncated, axis = _axis_aware_copy(
                open_source, target, suffix=source.suffix,
                max_rows=max_rows_per_table,
                axis_max_rows=max_rows_per_axis_table)
            entry = _bundle_entry(
                target.name, acquisition.path.name, rows, truncated, axis)
            files[-1] = entry
            tables.append(_MaterializedTable(
                entry=entry,
                target=target,
                delimiter=_csv_dialect_delimiter(source.suffix),
                open_source=open_source,
                axis=axis,
            ))
            _propagate_entity_subsamples(tables, dest_dir)
        except Exception:
            _remove_materialized_files(files, dest_dir)
            source.unlink(missing_ok=True)
            raise
        source.unlink(missing_ok=True)
    else:
        # Already a supported single-file artifact (a .json, a small .npz):
        # keep it whole when it fits. Unknown binary suffixes are not readable
        # by the package loaders and therefore cannot suppress the fallback.
        suffix = acquisition.path.suffix.lower()
        if (
            suffix in _SINGLE_FILE_SUFFIXES
            and not acquisition.path.name.startswith(".")
            and acquisition.path.name.casefold() not in _BUNDLE_METADATA_NAMES
            and acquisition.path.stat().st_size <= max_member_bytes
        ):
            target = dest_dir / acquisition.path.name
            files.append({
                "file": target.name,
                "source_member": acquisition.path.name,
                "rows_kept": None,
                "truncated": False,
            })
            try:
                if target != acquisition.path:
                    if target.exists():
                        raise FileExistsError(
                            f"refusing to overwrite existing bundle file {target}"
                        )
                    acquisition.path.replace(target)
            except Exception:
                _remove_materialized_files(files, dest_dir)
                acquisition.path.unlink(missing_ok=True)
                raise
        else:
            skipped.append(acquisition.path.name)
            acquisition.path.unlink()

    try:
        manifest = {
            "tier": "paper_cited_public",
            "source": {
                "cited_url": acquisition.source.cited_url,
                "download_url": acquisition.source.download_url,
                "host": acquisition.source.host,
                "fetch_sha256": acquisition.sha256,
                "fetch_bytes": acquisition.bytes_written,
                "fetch_truncated": acquisition.truncated,
            },
            "files": files,
            "skipped_members": skipped,
            "subsample_rule": _subsample_rule(files, max_rows_per_table),
            "usable_extent_note": _usable_extent_note(files),
            "honesty_note": (
                "This is a demo-scale extract of the dataset the paper itself "
                "cites, fetched anonymously at build time. Numbers computed "
                "on it demonstrate the method's mechanism; they are NOT the "
                "paper's benchmark results."
            ),
            "acquisition_attempts": (
                report.as_provenance()["attempts"] if report is not None else []
            ),
        }
        (dest_dir / PROVENANCE_FILENAME).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    except Exception:
        # PROVENANCE.json is the completion marker. A bundle whose manifest
        # could not be constructed or written must leave no public tables for
        # the synthetic tier to mix with.
        _remove_materialized_files(files, dest_dir)
        raise
    return manifest


def materialize_demo_bundle(
    acquisition: Acquisition,
    dest_dir: Path,
    *,
    report: AcquisitionReport | None = None,
    max_rows_per_table: int = DEFAULT_MAX_ROWS_PER_TABLE,
    max_rows_per_axis_table: int | None = None,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
) -> dict:
    """Stage, validate, and transactionally publish a demo-scale bundle.

    `PROVENANCE.json` is published last. A hash-bound hidden marker makes a
    process interruption distinguishable from researcher-owned local data on
    the next Stage 2a invocation.
    """

    destination = Path(dest_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage_parent = acquisition.path.parent
    stage_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".r2c-public-bundle-", dir=stage_parent,
    ) as temporary:
        stage = Path(temporary)
        manifest = _materialize_demo_bundle_staged(
            acquisition,
            stage,
            report=report,
            max_rows_per_table=max_rows_per_table,
            max_rows_per_axis_table=max_rows_per_axis_table,
            max_member_bytes=max_member_bytes,
        )
        _publish_public_bundle(stage, destination, manifest)
    return manifest


def first_usable(report: AcquisitionReport) -> ProbeResult | None:
    for probe in report.probes:
        if probe.usable:
            return probe
    return None


def refusal_summary(report: AcquisitionReport) -> str:
    """One human-readable line per rejected candidate, for disclosure."""
    if not report.probes:
        return "the paper cites no dataset on a recognized public data host"
    return "; ".join(
        f"{p.source.host} ({p.status}): {p.reason}"
        for p in report.probes
        if not p.usable
    ) or "all cited sources were usable"


def _iter_report_lines(report: AcquisitionReport) -> Iterable[str]:
    yield f"candidates found: {report.candidates_found}"
    for probe in report.probes:
        yield f"  [{probe.status}] {probe.source.host} {probe.source.cited_url}"
        yield f"      {probe.reason}"
    if report.acquired is not None:
        acq = report.acquired
        yield (
            f"acquired {acq.path.name} ({acq.bytes_written} bytes, "
            f"sha256={acq.sha256[:16]}..., truncated={acq.truncated})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument(
        "--provenance-json",
        type=Path,
        default=None,
        help="write the acquisition provenance record to this path",
    )
    args = parser.parse_args()

    if not args.paper.is_file():
        print(f"error: paper not found: {args.paper}", file=sys.stderr)
        return 2
    if not args.probe_only and args.dest is None:
        print("error: --dest is required unless --probe-only", file=sys.stderr)
        return 2

    text = args.paper.read_text(encoding="utf-8", errors="replace")
    report = acquire_for_paper(
        text,
        args.dest,
        max_bytes=args.max_bytes,
        probe_only=args.probe_only,
    )

    for line in _iter_report_lines(report):
        print(line)
    if args.provenance_json is not None:
        args.provenance_json.write_text(
            json.dumps(report.as_provenance(), indent=2) + "\n", encoding="utf-8"
        )

    if args.probe_only:
        return 0 if first_usable(report) is not None else 1
    return 0 if report.succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
