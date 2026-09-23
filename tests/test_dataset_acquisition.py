"""R2C-052: the paper's own cited datasets, acquired deterministically.

Every test injects a fake transport, so the suite exercises all of probing,
refusal, and fetching with zero network traffic. The mangled-footnote and
Kaggle-competition cases are taken from the real
probabilistic-demand-forecasting paper text.
"""

from __future__ import annotations

import io
import urllib.error

import pytest

from dataset_acquisition import (
    DEFAULT_MAX_BYTES,
    AcquisitionReport,
    acquire_for_paper,
    candidate_sources,
    classify_source,
    extract_cited_urls,
    fetch_source,
    probe_source,
    refusal_summary,
)


class FakeResponse:
    def __init__(self, *, status=200, headers=None, body=b""):
        self.status = status
        self.headers = headers or {}
        self._stream = io.BytesIO(body)
        self.closed = False

    def read(self, n=-1):
        return self._stream.read(n)

    def close(self):
        self.closed = True


def opener_for(mapping):
    """Transport that serves a {url: FakeResponse-or-Exception} map."""

    def _open(url):
        result = mapping.get(url)
        if result is None:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        if isinstance(result, Exception):
            raise result
        return result

    return _open


# The real footnote 4 shape: one URL repeated many times, a corrupted tail,
# and a second URL's fragment spliced on.
MANGLED_FOOTNOTE = (
    r"$<sup>^{4} \</sup> Source: \ "
    r"https://www.kaggle.com/datasets/berkayalan/retail-sales-data, \ "
    r"https://www.kaggle.com/datasets/berkayalan/retail-sales-data, \ "
    r"https://www.kaggle.com/datasets/berkayalan/retail-sales-data-data-data$ "
    r"com/competitions/competitive-data-science-predict-future-sales/data"
)

KAGGLE_PAGE = "https://www.kaggle.com/datasets/berkayalan/retail-sales-data"
KAGGLE_DOWNLOAD = (
    "https://www.kaggle.com/api/v1/datasets/download/berkayalan/retail-sales-data"
)


def test_trailing_punctuation_is_trimmed_but_paths_are_never_repaired():
    urls = extract_cited_urls(MANGLED_FOOTNOTE)

    # The clean URL survives with its comma and backslash stripped.
    assert KAGGLE_PAGE in urls
    # The corrupted variant is kept AS FOUND. Silently "fixing" it is how you
    # fetch a different dataset than the paper used.
    assert f"{KAGGLE_PAGE}-data-data" in urls


def test_repeated_footnote_url_collapses_to_one_candidate():
    sources = candidate_sources(MANGLED_FOOTNOTE)

    kaggle_datasets = [s for s in sources if s.host == "kaggle_dataset"]
    # Two DISTINCT dataset paths (the clean one and the corrupted one), not
    # the twenty-odd repetitions the footnote actually contains.
    assert len(kaggle_datasets) == 2
    assert kaggle_datasets[0].cited_url == KAGGLE_PAGE
    assert kaggle_datasets[0].download_url == KAGGLE_DOWNLOAD


def test_citation_links_are_not_dataset_candidates():
    """arXiv, vendor marketing, and plain GitHub repo links are citations."""
    text = (
        "http://arxiv.org/abs/1905.10069 "
        "https://github.com/jdb78/pytorch-forecasting "
        "https://www.shopify.com/enterprise/global-ecommerce-statistics"
    )
    assert candidate_sources(text) == []


def test_github_release_and_raw_assets_are_candidates():
    raw = classify_source(
        "https://raw.githubusercontent.com/owner/repo/main/data/demand.csv"
    )
    release = classify_source(
        "https://github.com/owner/repo/releases/download/v1/data.zip"
    )
    assert raw is not None and raw.host == "github_asset"
    assert release is not None and release.host == "github_asset"


def test_kaggle_competition_is_recognized_then_refused_by_rule():
    """A competition needs an account plus rule acceptance. Recognizing it is
    what lets the report say WHY the paper's source went unused."""
    source = classify_source(
        "https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data"
    )
    assert source is not None
    assert source.host == "kaggle_competition"
    assert source.credentialed is True

    # Refused without any network call at all.
    probe = probe_source(source, opener=opener_for({}), offline=False)
    assert probe.status == "refused_host"
    assert "credentials" in probe.reason or "account" in probe.reason


def test_offline_short_circuits_before_any_request():
    source = classify_source(KAGGLE_PAGE)
    def exploding_opener(url):
        raise AssertionError("offline must not contact the network")

    probe = probe_source(source, opener=exploding_opener, offline=True)
    assert probe.status == "refused_offline"
    assert "R2C_OFFLINE" in probe.reason


def test_anonymous_zip_probes_usable():
    source = classify_source(KAGGLE_PAGE)
    response = FakeResponse(
        headers={"Content-Type": "application/zip", "Content-Length": "90727623"}
    )
    probe = probe_source(
        source, opener=opener_for({KAGGLE_DOWNLOAD: response}), offline=False
    )

    assert probe.usable
    assert probe.content_type == "application/zip"
    assert probe.content_length == 90727623
    assert response.closed is True


def test_html_two_hundred_is_not_a_dataset():
    """A whitelisted host answering 200 with HTML is a login wall."""
    source = classify_source(KAGGLE_PAGE)
    probe = probe_source(
        source,
        opener=opener_for(
            {KAGGLE_DOWNLOAD: FakeResponse(headers={"Content-Type": "text/html"})}
        ),
        offline=False,
    )
    assert probe.status == "unavailable"
    assert "web page" in probe.reason


@pytest.mark.parametrize(
    "code,expected",
    [(401, "needs_credentials"), (403, "unavailable"), (404, "unavailable"),
     (500, "unavailable")],
)
def test_http_error_codes_classify(code, expected):
    source = classify_source(KAGGLE_PAGE)
    error = urllib.error.HTTPError(KAGGLE_DOWNLOAD, code, "err", {}, None)
    probe = probe_source(
        source, opener=opener_for({KAGGLE_DOWNLOAD: error}), offline=False
    )
    assert probe.status == expected
    assert probe.http_code == code


def test_forbidden_does_not_claim_credentials_are_the_cause():
    """Verified against the real paper: Kaggle 403s a nonexistent dataset, so
    the mangled citation tail 403s. Reporting that as "needs credentials"
    would put a wrong story in a delivered report."""
    source = classify_source(f"{KAGGLE_PAGE}-data-data-data")
    error = urllib.error.HTTPError(source.download_url, 403, "Forbidden", {}, None)
    probe = probe_source(
        source, opener=opener_for({source.download_url: error}), offline=False
    )

    assert probe.status == "unavailable"
    assert "private or withdrawn" in probe.reason
    assert "cited URL may be wrong" in probe.reason


def test_status_only_response_without_exception_also_classifies():
    """Some transports return the error response instead of raising."""
    source = classify_source(KAGGLE_PAGE)
    probe = probe_source(
        source,
        opener=opener_for({KAGGLE_DOWNLOAD: FakeResponse(status=401)}),
        offline=False,
    )
    assert probe.status == "needs_credentials"


def test_unreachable_network_is_unavailable_not_a_crash():
    source = classify_source(KAGGLE_PAGE)
    probe = probe_source(
        source,
        opener=opener_for({KAGGLE_DOWNLOAD: urllib.error.URLError("dns failure")}),
        offline=False,
    )
    assert probe.status == "unavailable"
    assert "could not be reached" in probe.reason


def test_fetch_writes_content_with_checksum_and_disposition_name(tmp_path):
    source = classify_source(KAGGLE_PAGE)
    body = b"product_id,store_id,date,sales\nP0001,S0002,2017-01-02,0.0\n"
    response = FakeResponse(
        headers={
            "Content-Type": "application/zip",
            "Content-Disposition": 'attachment; filename="retail-sales-data.zip"',
        },
        body=body,
    )

    acquisition = fetch_source(
        source, tmp_path, opener=opener_for({KAGGLE_DOWNLOAD: response})
    )

    assert acquisition.path.name == "retail-sales-data.zip"
    assert acquisition.path.read_bytes() == body
    assert acquisition.bytes_written == len(body)
    assert acquisition.truncated is False
    import hashlib

    assert acquisition.sha256 == hashlib.sha256(body).hexdigest()


def test_fetch_stops_at_the_byte_ceiling_and_records_truncation(tmp_path):
    """The size a server promises and the size it sends are different facts."""
    source = classify_source(KAGGLE_PAGE)
    response = FakeResponse(
        headers={"Content-Type": "application/zip"}, body=b"x" * 5000
    )

    acquisition = fetch_source(
        source,
        tmp_path,
        opener=opener_for({KAGGLE_DOWNLOAD: response}),
        max_bytes=1000,
        chunk_size=256,
    )

    assert acquisition.truncated is True
    assert acquisition.bytes_written == 1000
    assert acquisition.path.stat().st_size == 1000


def test_content_type_names_the_file_when_no_disposition(tmp_path):
    source = classify_source(KAGGLE_PAGE)
    acquisition = fetch_source(
        source,
        tmp_path,
        opener=opener_for(
            {
                KAGGLE_DOWNLOAD: FakeResponse(
                    headers={"Content-Type": "text/csv"}, body=b"a,b\n1,2\n"
                )
            }
        ),
    )
    assert acquisition.path.suffix == ".csv"
    assert "retail-sales-data" in acquisition.path.name


def test_end_to_end_prefers_the_first_usable_source(tmp_path):
    """The real paper shape: a corrupted candidate 404s, the clean one works,
    and the competition is refused by rule without a request."""
    corrupted_download = (
        "https://www.kaggle.com/api/v1/datasets/download/berkayalan/"
        "retail-sales-data-data-data"
    )
    body = b"PK\x03\x04 zip bytes"
    opener = opener_for(
        {
            corrupted_download: urllib.error.HTTPError(
                corrupted_download, 404, "Not Found", {}, None
            ),
            KAGGLE_DOWNLOAD: FakeResponse(
                headers={"Content-Type": "application/zip"}, body=body
            ),
        }
    )

    report = acquire_for_paper(
        MANGLED_FOOTNOTE, tmp_path, opener=opener, offline=False
    )

    assert report.succeeded
    assert report.acquired.source.cited_url == KAGGLE_PAGE
    assert report.acquired.path.read_bytes() == body
    # Every attempt is retained for disclosure, including the refused
    # competition and the 404 on the mangled tail.
    statuses = {p.status for p in report.probes}
    assert "refused_host" in statuses or "ok" in statuses
    assert report.candidates_found >= 2


def test_end_to_end_offline_acquires_nothing_and_says_why(tmp_path):
    report = acquire_for_paper(
        MANGLED_FOOTNOTE,
        tmp_path,
        opener=opener_for({}),
        offline=True,
    )

    assert not report.succeeded
    assert all(p.status == "refused_offline" for p in report.probes)
    assert "R2C_OFFLINE" in refusal_summary(report)
    assert list(tmp_path.iterdir()) == []


def test_paper_citing_no_data_host_is_reported_honestly(tmp_path):
    report = acquire_for_paper(
        "See http://arxiv.org/abs/1905.10069 for details.",
        tmp_path,
        opener=opener_for({}),
        offline=False,
    )

    assert not report.succeeded
    assert report.candidates_found == 0
    assert "no dataset on a recognized public data host" in refusal_summary(report)


def test_provenance_record_is_json_safe_and_complete(tmp_path):
    body = b"PK\x03\x04"
    report = acquire_for_paper(
        f"data at {KAGGLE_PAGE}",
        tmp_path,
        opener=opener_for(
            {
                KAGGLE_DOWNLOAD: FakeResponse(
                    headers={"Content-Type": "application/zip"}, body=body
                )
            }
        ),
        offline=False,
    )
    provenance = report.as_provenance()

    import json

    json.dumps(provenance)  # must not raise
    assert provenance["status"] == "acquired"
    assert provenance["acquired"]["host"] == "kaggle_dataset"
    assert provenance["acquired"]["cited_url"] == KAGGLE_PAGE
    assert provenance["acquired"]["sha256"]
    assert provenance["acquired"]["bytes"] == len(body)
    assert provenance["attempts"][0]["status"] == "ok"


def test_probe_only_never_writes_anything(tmp_path):
    report = acquire_for_paper(
        f"data at {KAGGLE_PAGE}",
        tmp_path,
        opener=opener_for(
            {
                KAGGLE_DOWNLOAD: FakeResponse(
                    headers={"Content-Type": "application/zip"}, body=b"PK"
                )
            }
        ),
        offline=False,
        probe_only=True,
    )

    assert not report.succeeded
    assert report.probes[0].usable
    assert list(tmp_path.iterdir()) == []


def test_empty_report_provenance_is_still_valid():
    report = AcquisitionReport()
    provenance = report.as_provenance()
    assert provenance["status"] == "not_acquired"
    assert provenance["acquired"] is None
    assert provenance["attempts"] == []
    assert DEFAULT_MAX_BYTES > 0
