"""R2C-052 part 2 — the demo-bundle materializer and the stage 2.a wiring.

The materializer turns a fetched archive into a demo-scale bundle
(tabular members only, deterministic row caps, provenance beside the
data). The wiring runs after a clean scaffold on gap-path runs only, and
must never be able to kill a run.

All network is faked; the zip fixtures are built in memory.
"""

from __future__ import annotations

import io
import hashlib
import importlib.util
import json
import zipfile
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pytest

from tests.helpers.assertions import assert_stage_completed, assert_stage_halted
from tests.helpers.inject import minimal_method_spec
from tests.helpers.state import make_state


def _make_acquisition(tmp_path: Path, payload: bytes, filename: str):
    from dataset_acquisition import Acquisition, classify_source

    path = tmp_path / filename
    path.write_bytes(payload)
    source = classify_source(
        "https://www.kaggle.com/datasets/berkayalan/retail-sales-data")
    return Acquisition(source=source, path=path, sha256="ab" * 32,
                       bytes_written=len(payload), truncated=False)


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in members.items():
            z.writestr(name, content)
    return buf.getvalue()


def _csv_rows(n: int) -> bytes:
    lines = ["product_id,store_id,date,sales"]
    lines += [f"P{i:04d},S0001,2017-01-02,{i}.0" for i in range(n)]
    return ("\n".join(lines) + "\n").encode()


def test_bundle_extracts_tabular_members_with_row_caps(tmp_path):
    """The motivating shape: a nested big table (sales.csv/sales.csv in the
    real dataset), a small table kept whole, and a binary member skipped.
    The archive itself is deleted so a delivery never carries the raw
    gigabytes."""
    from dataset_acquisition import materialize_demo_bundle

    payload = _zip_bytes({
        "sales.csv/sales.csv": _csv_rows(500),
        "product_hierarchy.csv": _csv_rows(10),
        "assets/logo.png": b"\x89PNG fake",
    })
    acq = _make_acquisition(tmp_path, payload, "retail.zip")
    dest = tmp_path / "example_data"

    manifest = materialize_demo_bundle(acq, dest, max_rows_per_table=100)

    big = dest / "sales.csv"
    small = dest / "product_hierarchy.csv"
    assert big.read_text().splitlines()[0].startswith("product_id,")
    assert len(big.read_text().splitlines()) == 101  # header + 100 rows
    assert len(small.read_text().splitlines()) == 11  # kept whole
    assert not (tmp_path / "retail.zip").exists()
    assert "assets/logo.png" in manifest["skipped_members"]

    by_name = {f["file"]: f for f in manifest["files"]}
    assert by_name["sales.csv"]["truncated"] is True
    assert by_name["sales.csv"]["rows_kept"] == 100
    assert by_name["product_hierarchy.csv"]["truncated"] is False

    provenance = json.loads((dest / "PROVENANCE.json").read_text())
    assert provenance["tier"] == "paper_cited_public"
    assert provenance["source"]["cited_url"].endswith("retail-sales-data")
    assert "NOT the paper's benchmark results" in provenance["honesty_note"]
    from dataset_acquisition import PUBLIC_TRANSACTION_FILENAME
    assert not (dest / PUBLIC_TRANSACTION_FILENAME).exists()


def test_bundle_refuses_traversal_members_and_resolves_collisions(tmp_path):
    from dataset_acquisition import materialize_demo_bundle

    payload = _zip_bytes({
        "../evil.csv": _csv_rows(2),
        "metadata/PROVENANCE.json": b'{"tier":"forged"}',
        "a/data.csv": _csv_rows(2),
        "b/data.csv": _csv_rows(3),
    })
    acq = _make_acquisition(tmp_path, payload, "d.zip")
    dest = tmp_path / "example_data"

    manifest = materialize_demo_bundle(acq, dest, max_rows_per_table=100)

    assert not (tmp_path / "evil.csv").exists()
    assert "../evil.csv" in manifest["skipped_members"]
    assert "metadata/PROVENANCE.json" in manifest["skipped_members"]
    names = sorted(f["file"] for f in manifest["files"])
    assert names == ["2_data.csv", "data.csv"]
    assert (dest / "data.csv").is_file()
    assert (dest / "2_data.csv").is_file()


def test_bundle_row_caps_a_bare_csv_fetch(tmp_path):
    from dataset_acquisition import materialize_demo_bundle

    acq = _make_acquisition(tmp_path, _csv_rows(50), "demand.csv")
    dest = tmp_path / "example_data"

    manifest = materialize_demo_bundle(acq, dest, max_rows_per_table=20)

    assert len((dest / "demand.csv").read_text().splitlines()) == 21
    assert manifest["files"][0]["truncated"] is True


def test_bundle_refuses_an_unknown_single_file_as_non_consumable(tmp_path):
    from dataset_acquisition import materialize_demo_bundle

    acq = _make_acquisition(tmp_path, b"opaque payload", "dataset.bin")
    dest = tmp_path / "example_data"

    manifest = materialize_demo_bundle(acq, dest)

    assert manifest["files"] == []
    assert manifest["skipped_members"] == ["dataset.bin"]
    assert not (dest / "dataset.bin").exists()


@pytest.mark.parametrize(
    "template_relative",
    [
        "paradigms/time_series_forecasting/templates/method/data.py.template",
        "paradigms/_provisional/templates/method/data.py.template",
    ],
)
@pytest.mark.parametrize("suffix", [".tsv", ".npz"])
def test_file_only_templates_consume_every_public_single_file_format(
    tmp_path, template_relative, suffix,
):
    root = Path(__file__).resolve().parents[1]
    template = root / template_relative
    loader = SourceFileLoader(
        f"r2c_loader_{template.parent.parent.name}_{suffix[1:]}",
        str(template),
    )
    module_spec = importlib.util.spec_from_loader(loader.name, loader)
    assert module_spec is not None
    module = importlib.util.module_from_spec(module_spec)
    loader.exec_module(module)

    path = tmp_path / f"table{suffix}"
    if suffix == ".tsv":
        path.write_text("series_id\ttarget\na\t1.5\n", encoding="utf-8")
        loaded = module.load_data(path=path)
        assert loaded["table"]["series_id"].tolist() == ["a"]
        assert loaded["table"]["target"].tolist() == [1.5]
    else:
        np.savez(path, target=np.asarray([1.5]))
        loaded = module.load_data(path=path)
        assert loaded["target"].tolist() == [1.5]


@pytest.mark.parametrize("filename", ["DATA.CSV", "table.tsv", "table.npz"])
def test_materializer_output_is_consumable_by_the_forecasting_loader(
    tmp_path, filename,
):
    from dataset_acquisition import materialize_demo_bundle

    if filename.lower().endswith(".csv"):
        payload = _csv_rows(3)
    elif filename.lower().endswith(".tsv"):
        payload = b"series_id\ttarget\na\t1.5\n"
    else:
        stream = io.BytesIO()
        np.savez(stream, target=np.asarray([1.5]))
        payload = stream.getvalue()
    acq = _make_acquisition(tmp_path, payload, filename)
    destination = tmp_path / "example_data"
    manifest = materialize_demo_bundle(acq, destination)
    assert [item["file"] for item in manifest["files"]] == [filename]

    root = Path(__file__).resolve().parents[1]
    template = (
        root / "paradigms/time_series_forecasting/templates/method/data.py.template"
    )
    loader = SourceFileLoader(f"r2c_materialized_{filename}", str(template))
    module_spec = importlib.util.spec_from_loader(loader.name, loader)
    assert module_spec is not None
    module = importlib.util.module_from_spec(module_spec)
    loader.exec_module(module)
    loaded = module.load_data(path=destination)

    if filename.lower().endswith(".npz"):
        assert loaded["target"].tolist() == [1.5]
    else:
        table = loaded[Path(filename).stem]
        assert table


def test_public_manifest_write_failure_rolls_back_materialized_tables(
    tmp_path, monkeypatch,
):
    from dataset_acquisition import materialize_demo_bundle

    payload = _zip_bytes({"sales.csv": _csv_rows(10)})
    acq = _make_acquisition(tmp_path, payload, "retail.zip")
    dest = tmp_path / "example_data"
    real_write_text = Path.write_text

    def _fail_public_manifest(path, data, *args, **kwargs):
        if (
            path.name == "PROVENANCE.json"
            and path.parent.name.startswith(".r2c-public-bundle-")
        ):
            raise OSError("injected public manifest failure")
        return real_write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _fail_public_manifest)

    with pytest.raises(OSError, match="injected public manifest failure"):
        materialize_demo_bundle(acq, dest)

    assert not (dest / "sales.csv").exists()
    assert not (dest / "PROVENANCE.json").exists()
    assert not any(dest.parent.glob(".r2c-public-bundle-*"))


def test_public_mid_copy_failure_publishes_no_partial_tables(
    tmp_path, monkeypatch,
):
    import dataset_acquisition

    payload = _zip_bytes({"sales.csv": _csv_rows(10)})
    acq = _make_acquisition(tmp_path, payload, "retail.zip")
    dest = tmp_path / "example_data"

    def _partial_then_fail(open_source, target, **kwargs):
        target.write_text("partial public table\n", encoding="utf-8")
        raise OSError("injected mid-copy failure")

    monkeypatch.setattr(
        dataset_acquisition, "_axis_aware_copy", _partial_then_fail,
    )

    with pytest.raises(OSError, match="injected mid-copy failure"):
        dataset_acquisition.materialize_demo_bundle(acq, dest)

    assert not (dest / "sales.csv").exists()
    assert not (dest / "PROVENANCE.json").exists()
    assert not (dest / dataset_acquisition.PUBLIC_TRANSACTION_FILENAME).exists()


def test_interrupted_public_publish_recovers_only_hash_matched_output(
    tmp_path, monkeypatch,
):
    import dataset_acquisition

    payload = _zip_bytes({"sales.csv": _csv_rows(10)})
    acq = _make_acquisition(tmp_path, payload, "retail.zip")
    dest = tmp_path / "example_data"
    real_replace = dataset_acquisition.os.replace

    def _interrupt_after_replace(source, target):
        real_replace(source, target)
        if Path(target).name == "sales.csv":
            raise KeyboardInterrupt("injected process interruption")

    monkeypatch.setattr(
        dataset_acquisition.os, "replace", _interrupt_after_replace,
    )
    with pytest.raises(KeyboardInterrupt, match="process interruption"):
        dataset_acquisition.materialize_demo_bundle(acq, dest)

    marker = dest / dataset_acquisition.PUBLIC_TRANSACTION_FILENAME
    assert marker.is_file()
    assert (dest / "sales.csv").is_file()
    assert not (dest / "PROVENANCE.json").exists()

    monkeypatch.setattr(dataset_acquisition.os, "replace", real_replace)
    outcome = dataset_acquisition.recover_interrupted_public_bundle(dest)

    assert outcome == "removed_partial"
    assert not marker.exists()
    assert not (dest / "sales.csv").exists()


def test_complete_public_publish_with_stale_marker_is_accepted(
    tmp_path, monkeypatch,
):
    import dataset_acquisition

    payload = _zip_bytes({"sales.csv": _csv_rows(10)})
    acq = _make_acquisition(tmp_path, payload, "retail.zip")
    dest = tmp_path / "example_data"
    real_unlink = Path.unlink
    interrupted = False

    def _interrupt_marker_unlink(path, *args, **kwargs):
        nonlocal interrupted
        if (
            path == dest / dataset_acquisition.PUBLIC_TRANSACTION_FILENAME
            and not interrupted
        ):
            interrupted = True
            raise KeyboardInterrupt("injected marker-removal interruption")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _interrupt_marker_unlink)
    with pytest.raises(KeyboardInterrupt, match="marker-removal interruption"):
        dataset_acquisition.materialize_demo_bundle(acq, dest)

    assert (dest / "sales.csv").is_file()
    assert (dest / "PROVENANCE.json").is_file()
    assert (dest / dataset_acquisition.PUBLIC_TRANSACTION_FILENAME).is_file()

    monkeypatch.setattr(Path, "unlink", real_unlink)
    outcome = dataset_acquisition.recover_interrupted_public_bundle(dest)

    assert outcome == "completed"
    assert (dest / "sales.csv").is_file()
    assert (dest / "PROVENANCE.json").is_file()
    assert not (dest / dataset_acquisition.PUBLIC_TRANSACTION_FILENAME).exists()


def test_provenance_only_public_publication_is_removed_for_fallback(tmp_path):
    import dataset_acquisition

    dest = tmp_path / "example_data"
    dest.mkdir()
    provenance = dest / "PROVENANCE.json"
    provenance.write_text('{"tier":"paper_cited_public","files":[]}\n')
    marker = dest / dataset_acquisition.PUBLIC_TRANSACTION_FILENAME
    marker.write_text(json.dumps({
        "schema_version": dataset_acquisition.PUBLIC_TRANSACTION_SCHEMA_VERSION,
        "materializer_id": dataset_acquisition.PUBLIC_MATERIALIZER_ID,
        "materializer_version": dataset_acquisition.PUBLIC_MATERIALIZER_VERSION,
        "files": [{
            "file": "PROVENANCE.json",
            "sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
        }],
    }), encoding="utf-8")

    outcome = dataset_acquisition.recover_interrupted_public_bundle(dest)

    assert outcome == "removed_partial"
    assert not provenance.exists()
    assert not marker.exists()


def test_complete_public_publication_yields_to_new_local_data(tmp_path):
    import dataset_acquisition

    dest = tmp_path / "example_data"
    dest.mkdir()
    public = dest / "sales.csv"
    public.write_text("product_id,target\np1,1\n", encoding="utf-8")
    provenance = dest / "PROVENANCE.json"
    provenance.write_text('{"tier":"paper_cited_public"}\n', encoding="utf-8")
    local = dest / "researcher.csv"
    local.write_text("series_id,target\nowned,3\n", encoding="utf-8")
    marker = dest / dataset_acquisition.PUBLIC_TRANSACTION_FILENAME
    marker.write_text(json.dumps({
        "schema_version": dataset_acquisition.PUBLIC_TRANSACTION_SCHEMA_VERSION,
        "materializer_id": dataset_acquisition.PUBLIC_MATERIALIZER_ID,
        "materializer_version": dataset_acquisition.PUBLIC_MATERIALIZER_VERSION,
        "files": [
            {
                "file": "sales.csv",
                "sha256": hashlib.sha256(public.read_bytes()).hexdigest(),
            },
            {
                "file": "PROVENANCE.json",
                "sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
            },
        ],
    }), encoding="utf-8")

    outcome = dataset_acquisition.recover_interrupted_public_bundle(dest)

    assert outcome == "removed_partial"
    assert not public.exists()
    assert not provenance.exists()
    assert not marker.exists()
    assert local.read_text(encoding="utf-8").startswith("series_id,target")


def test_catchable_public_publish_error_rolls_back_targets_and_marker(
    tmp_path, monkeypatch,
):
    import dataset_acquisition

    payload = _zip_bytes({"sales.csv": _csv_rows(10)})
    acq = _make_acquisition(tmp_path, payload, "retail.zip")
    dest = tmp_path / "example_data"
    real_replace = dataset_acquisition.os.replace

    def _fail_after_provenance_replace(source, target):
        real_replace(source, target)
        if Path(target).name == "PROVENANCE.json":
            raise OSError("injected canonical publication failure")

    monkeypatch.setattr(
        dataset_acquisition.os, "replace", _fail_after_provenance_replace,
    )
    with pytest.raises(OSError, match="canonical publication failure"):
        dataset_acquisition.materialize_demo_bundle(acq, dest)

    assert not (dest / "sales.csv").exists()
    assert not (dest / "PROVENANCE.json").exists()
    assert not (dest / dataset_acquisition.PUBLIC_TRANSACTION_FILENAME).exists()


def test_interrupted_public_recovery_preserves_changed_output(
    tmp_path, monkeypatch,
):
    import dataset_acquisition

    payload = _zip_bytes({"sales.csv": _csv_rows(10)})
    acq = _make_acquisition(tmp_path, payload, "retail.zip")
    dest = tmp_path / "example_data"
    real_replace = dataset_acquisition.os.replace

    def _interrupt_after_replace(source, target):
        real_replace(source, target)
        if Path(target).name == "sales.csv":
            raise KeyboardInterrupt("injected process interruption")

    monkeypatch.setattr(
        dataset_acquisition.os, "replace", _interrupt_after_replace,
    )
    with pytest.raises(KeyboardInterrupt):
        dataset_acquisition.materialize_demo_bundle(acq, dest)
    changed = dest / "sales.csv"
    changed.write_text("researcher changed this file\n", encoding="utf-8")
    monkeypatch.setattr(dataset_acquisition.os, "replace", real_replace)

    with pytest.raises(
        dataset_acquisition.PublicBundleRecoveryError,
        match="refusing to remove changed interrupted file",
    ):
        dataset_acquisition.recover_interrupted_public_bundle(dest)

    assert changed.read_text(encoding="utf-8") == "researcher changed this file\n"
    assert (dest / dataset_acquisition.PUBLIC_TRANSACTION_FILENAME).is_file()


def test_public_recovery_rejects_marker_that_names_non_data_output(tmp_path):
    import dataset_acquisition

    dest = tmp_path / "example_data"
    dest.mkdir()
    readme = dest / "README.md"
    readme.write_text("researcher-owned instructions\n", encoding="utf-8")
    marker = dest / dataset_acquisition.PUBLIC_TRANSACTION_FILENAME
    marker.write_text(json.dumps({
        "schema_version": dataset_acquisition.PUBLIC_TRANSACTION_SCHEMA_VERSION,
        "materializer_id": dataset_acquisition.PUBLIC_MATERIALIZER_ID,
        "materializer_version": dataset_acquisition.PUBLIC_MATERIALIZER_VERSION,
        "files": [
            {
                "file": "README.md",
                "sha256": hashlib.sha256(readme.read_bytes()).hexdigest(),
            },
            {"file": "PROVENANCE.json", "sha256": "0" * 64},
        ],
    }), encoding="utf-8")

    with pytest.raises(
        dataset_acquisition.PublicBundleRecoveryError,
        match="invalid file record",
    ):
        dataset_acquisition.recover_interrupted_public_bundle(dest)

    assert readme.read_text(encoding="utf-8") == "researcher-owned instructions\n"
    assert marker.is_file()


# ---------------------------------------------------------------------------
# Stage 2.a wiring
# ---------------------------------------------------------------------------

_KAGGLE_PAPER = (
    "We evaluate on public data: "
    "https://www.kaggle.com/datasets/berkayalan/retail-sales-data\n"
)
_KAGGLE_DOWNLOAD = (
    "https://www.kaggle.com/api/v1/datasets/download/berkayalan/retail-sales-data"
)


class _FakeResponse:
    def __init__(self, body: bytes, content_type: str = "application/zip"):
        self.status = 200
        self.headers = {"Content-Type": content_type,
                        "Content-Length": str(len(body))}
        self._stream = io.BytesIO(body)

    def read(self, n=-1):
        return self._stream.read(n)

    def close(self):
        pass


def _seed_stage_2a(run_dir: Path, *, gap_run: bool):
    state = make_state(run_dir)
    spec = minimal_method_spec(paradigm_id="active_learning")
    state.paths.method_spec.write_text(json.dumps(spec), encoding="utf-8")
    state.paths.paper_md.write_text(_KAGGLE_PAPER, encoding="utf-8")
    if gap_run:
        (run_dir / ".pipeline" / "provisional_packs" / "x").mkdir(
            parents=True, exist_ok=True)
    return state


def _typed_tsf_spec() -> dict:
    spec = minimal_method_spec(
        paradigm_id="time_series_forecasting", pluggable_name="forecast",
    )
    spec["data_requirements"] = {"synthetic_feasible": True}

    def quantity(role: str, value: int | None) -> dict:
        return {
            "role": role,
            "parameter_name": (
                "context_length" if role == "context_length"
                else "forecast_horizon"
                if role == "forecast_call_horizon" else None
            ),
            "value": value,
            "unit": "week",
            "granularity": 1,
            "paper_value_status": (
                "paper_stated" if value is not None else "paper_unspecified"
            ),
            "paper_element_ids": [f"value-{role}"],
            "axis_paper_element_ids": ["weekly-axis"],
        }

    spec["comparison"]["evaluation_protocol"] = {
        "scheme": {
            "kind": "single_holdout",
            "paper_value_status": "paper_stated",
            "paper_element_ids": ["chronological-split"],
        },
        "quantities": [
            quantity("context_length", 10),
            quantity("forecast_call_horizon", None),
            quantity("validation_span", 13),
            quantity("test_span", 26),
        ],
    }
    return spec


def test_stage_2a_gap_run_bundles_the_cited_dataset(
    fake_subprocess, run_dir, monkeypatch
):
    """End to end through the wiring with a faked transport: the paper's
    citation is probed, fetched, subsampled into example_data/, the
    provenance lands in .pipeline/, and the run event is appended."""
    import dataset_acquisition
    from run_pipeline import run_stage_2a

    state = _seed_stage_2a(run_dir, gap_run=True)
    payload = _zip_bytes({"sales.csv": _csv_rows(30)})
    monkeypatch.setattr(
        dataset_acquisition, "_default_opener",
        lambda url: _FakeResponse(payload) if url == _KAGGLE_DOWNLOAD
        else (_ for _ in ()).throw(AssertionError(f"unexpected url {url}")))
    monkeypatch.delenv("R2C_OFFLINE", raising=False)

    fake_subprocess.expect_script(returncode=0)   # scaffold_package.py
    fake_subprocess.expect_stage2_script(ok=True)  # validate_scaffolder_output

    result = run_stage_2a(state)

    assert_stage_completed(result, "stage_2a")
    example_data = run_dir / "method" / "example_data"
    assert (example_data / "sales.csv").is_file()
    assert (example_data / "PROVENANCE.json").is_file()
    record = json.loads(
        (run_dir / ".pipeline" / "dataset_acquisition.json").read_text())
    assert record["status"] == "acquired"
    assert "offline_fallback" not in record
    assert record["bundle"]["files"][0]["file"] == "sales.csv"
    events = (run_dir / ".pipeline" / "run_events.jsonl").read_text()
    assert "demo_dataset_acquired" in events


def test_stage_2a_non_gap_run_never_attempts_acquisition(
    fake_subprocess, run_dir, monkeypatch
):
    import dataset_acquisition
    from run_pipeline import run_stage_2a

    state = _seed_stage_2a(run_dir, gap_run=False)

    def _explode(url):
        raise AssertionError("non-gap runs must not touch the network")

    monkeypatch.setattr(dataset_acquisition, "_default_opener", _explode)

    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_stage2_script(ok=True)

    result = run_stage_2a(state)

    assert_stage_completed(result, "stage_2a")
    assert not (run_dir / ".pipeline" / "dataset_acquisition.json").exists()


def test_stage_2a_acquisition_failure_is_never_fatal(
    fake_subprocess, run_dir, monkeypatch
):
    """The seam's hard rule: whatever breaks inside acquisition, the stage
    completes and the error is recorded for the report."""
    import dataset_acquisition
    from run_pipeline import run_stage_2a

    state = _seed_stage_2a(run_dir, gap_run=True)

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic acquisition explosion")

    monkeypatch.setattr(dataset_acquisition, "acquire_for_paper", _boom)

    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_stage2_script(ok=True)

    result = run_stage_2a(state)

    assert_stage_completed(result, "stage_2a")
    record = json.loads(
        (run_dir / ".pipeline" / "dataset_acquisition.json").read_text())
    assert record["status"] == "fallback_refused"
    assert "synthetic acquisition explosion" in record["error"]


def test_stage_2a_offline_records_the_refusals_and_continues(
    fake_subprocess, run_dir, monkeypatch
):
    from run_pipeline import run_stage_2a

    state = _seed_stage_2a(run_dir, gap_run=True)
    monkeypatch.setenv("R2C_OFFLINE", "1")

    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_stage2_script(ok=True)

    result = run_stage_2a(state)

    assert_stage_completed(result, "stage_2a")
    record = json.loads(
        (run_dir / ".pipeline" / "dataset_acquisition.json").read_text())
    assert record["status"] == "fallback_refused"
    assert record["offline"] is True
    assert all(a["status"] == "refused_offline" for a in record["attempts"])
    assert record["offline_fallback"]["status"] == "refused"
    assert (
        record["offline_fallback"]["code"]
        == "offline_fallback_family_unsupported"
    )
    assert not (run_dir / "method" / "example_data" / "PROVENANCE.json").exists()
    events = (run_dir / ".pipeline" / "run_events.jsonl").read_text()
    assert "demo_dataset_fallback_refused" in events
    fallback_event = next(
        json.loads(line)
        for line in events.splitlines()
        if "demo_dataset_fallback_refused" in line
    )
    assert fallback_event["details"]["producer_retry_consumed"] is False


def test_stage_2a_offline_tsf_generates_the_family_owned_third_tier(
    fake_subprocess, run_dir, monkeypatch,
):
    from run_pipeline import run_stage_2a

    state = make_state(run_dir)
    state.paths.method_spec.write_text(
        json.dumps(_typed_tsf_spec()), encoding="utf-8",
    )
    state.paths.paper_md.write_text(_KAGGLE_PAPER, encoding="utf-8")
    monkeypatch.setenv("R2C_OFFLINE", "1")
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_stage2_script(ok=True)

    result = run_stage_2a(state)

    assert_stage_completed(result, "stage_2a")
    example_data = run_dir / "method" / "example_data"
    assert (example_data / "series.csv").is_file()
    assert (example_data / "observations.csv").is_file()
    manifest = json.loads((example_data / "PROVENANCE.json").read_text())
    assert manifest["tier"] == "family_owned_synthetic"
    assert manifest["protocol_capacity"]["usable_steps"] == 73
    assert manifest["protocol_capacity"]["static_axis_floor_steps"] == 49
    record = json.loads(
        (run_dir / ".pipeline" / "dataset_acquisition.json").read_text()
    )
    assert record["status"] == "fallback_generated"
    assert record["offline"] is True
    assert record["offline_fallback"]["status"] == "generated"
    statuses = [
        attempt["status"] for attempt in manifest["acquisition_attempts"]
    ]
    assert statuses == ["refused_offline", "fallback_trigger"]
    events = (run_dir / ".pipeline" / "run_events.jsonl").read_text()
    assert "demo_dataset_fallback_generated" in events
    assert "demo_dataset_acquired" not in events


def test_stage_2a_public_acquisition_error_triggers_tsf_fallback(
    fake_subprocess, run_dir, monkeypatch,
):
    import dataset_acquisition
    from run_pipeline import run_stage_2a

    state = make_state(run_dir)
    state.paths.method_spec.write_text(
        json.dumps(_typed_tsf_spec()), encoding="utf-8",
    )
    state.paths.paper_md.write_text(_KAGGLE_PAPER, encoding="utf-8")

    def _unreachable(*args, **kwargs):
        raise OSError("injected public source failure")

    monkeypatch.setattr(dataset_acquisition, "acquire_for_paper", _unreachable)
    monkeypatch.delenv("R2C_OFFLINE", raising=False)
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_stage2_script(ok=True)

    result = run_stage_2a(state)

    assert_stage_completed(result, "stage_2a")
    example_data = run_dir / "method" / "example_data"
    manifest = json.loads((example_data / "PROVENANCE.json").read_text())
    assert manifest["tier"] == "family_owned_synthetic"
    assert manifest["acquisition_attempts"] == [{
        "host": None,
        "cited_url": None,
        "status": "fallback_trigger",
        "reason": (
            "public dataset acquisition failed: injected public source failure"
        ),
        "http_code": None,
    }]
    record = json.loads(
        (run_dir / ".pipeline" / "dataset_acquisition.json").read_text()
    )
    assert record["status"] == "fallback_generated"
    assert record["error"] == "injected public source failure"
    assert record["offline_fallback"]["status"] == "generated"
    events = (run_dir / ".pipeline" / "run_events.jsonl").read_text()
    assert "demo_dataset_fallback_generated" in events
    assert "demo_dataset_acquired" not in events


def test_stage_2a_public_artifact_without_supported_files_triggers_fallback(
    fake_subprocess, run_dir, monkeypatch,
):
    import dataset_acquisition
    from run_pipeline import run_stage_2a

    state = make_state(run_dir)
    state.paths.method_spec.write_text(
        json.dumps(_typed_tsf_spec()), encoding="utf-8",
    )
    state.paths.paper_md.write_text(_KAGGLE_PAPER, encoding="utf-8")
    payload = _zip_bytes({"assets/readme.txt": b"not a supported table\n"})
    monkeypatch.setattr(
        dataset_acquisition,
        "_default_opener",
        lambda url: _FakeResponse(payload),
    )
    monkeypatch.delenv("R2C_OFFLINE", raising=False)
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_stage2_script(ok=True)

    result = run_stage_2a(state)

    assert_stage_completed(result, "stage_2a")
    example_data = run_dir / "method" / "example_data"
    manifest = json.loads((example_data / "PROVENANCE.json").read_text())
    assert manifest["tier"] == "family_owned_synthetic"
    record = json.loads(
        (run_dir / ".pipeline" / "dataset_acquisition.json").read_text()
    )
    assert record["status"] == "fallback_generated"
    assert record["bundle"]["files"] == []
    assert record["bundle"]["refused"]["status"] == "no_supported_files"
    assert record["offline_fallback"]["status"] == "generated"
    events = (run_dir / ".pipeline" / "run_events.jsonl").read_text()
    assert "demo_dataset_refused" in events
    assert "demo_dataset_fallback_generated" in events
    assert "demo_dataset_acquired" not in events


def test_stage_2a_materialization_error_retains_public_attempt_provenance(
    fake_subprocess, run_dir, monkeypatch,
):
    import dataset_acquisition
    from run_pipeline import run_stage_2a

    state = make_state(run_dir)
    state.paths.method_spec.write_text(
        json.dumps(_typed_tsf_spec()), encoding="utf-8",
    )
    state.paths.paper_md.write_text(_KAGGLE_PAPER, encoding="utf-8")
    payload = _zip_bytes({"sales.csv": _csv_rows(30)})
    monkeypatch.setattr(
        dataset_acquisition,
        "_default_opener",
        lambda url: _FakeResponse(payload),
    )

    def _materialization_error(*args, **kwargs):
        raise RuntimeError("injected materialization failure")

    monkeypatch.setattr(
        dataset_acquisition, "materialize_demo_bundle", _materialization_error,
    )
    monkeypatch.delenv("R2C_OFFLINE", raising=False)
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_stage2_script(ok=True)

    result = run_stage_2a(state)

    assert_stage_completed(result, "stage_2a")
    example_data = run_dir / "method" / "example_data"
    manifest = json.loads((example_data / "PROVENANCE.json").read_text())
    record = json.loads(
        (run_dir / ".pipeline" / "dataset_acquisition.json").read_text()
    )
    assert record["status"] == "fallback_generated"
    assert record["error"] == "injected materialization failure"
    assert record["acquired"]["cited_url"] == (
        "https://www.kaggle.com/datasets/berkayalan/retail-sales-data"
    )
    assert record["attempts"]
    assert manifest["acquisition_attempts"][0]["cited_url"] == (
        "https://www.kaggle.com/datasets/berkayalan/retail-sales-data"
    )
    assert manifest["acquisition_attempts"][-1]["status"] == "fallback_trigger"


def test_stage_2a_local_tsf_data_preempts_public_and_synthetic_tiers(
    fake_subprocess, run_dir, monkeypatch,
):
    import dataset_acquisition
    import time_series_offline_fallback
    from run_pipeline import run_stage_2a

    state = make_state(run_dir)
    state.paths.method_spec.write_text(
        json.dumps(_typed_tsf_spec()), encoding="utf-8",
    )
    state.paths.paper_md.write_text(_KAGGLE_PAPER, encoding="utf-8")
    example_data = run_dir / "method" / "example_data"
    example_data.mkdir(parents=True)
    local = example_data / "researcher.csv"
    local.write_text("series_id,target\nowned,3\n", encoding="utf-8")

    def _unexpected(*args, **kwargs):
        raise AssertionError("local data must preempt both provisioning tiers")

    monkeypatch.setattr(dataset_acquisition, "acquire_for_paper", _unexpected)
    monkeypatch.setattr(
        time_series_offline_fallback,
        "generate_time_series_offline_fallback",
        _unexpected,
    )
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_stage2_script(ok=True)

    result = run_stage_2a(state)

    assert_stage_completed(result, "stage_2a")
    assert local.read_text(encoding="utf-8") == "series_id,target\nowned,3\n"
    assert not (run_dir / ".pipeline" / "dataset_acquisition.json").exists()


def test_stage_2a_unreadable_local_pt_does_not_suppress_tsf_fallback(
    fake_subprocess, run_dir, monkeypatch,
):
    from run_pipeline import run_stage_2a

    state = make_state(run_dir)
    state.paths.method_spec.write_text(
        json.dumps(_typed_tsf_spec()), encoding="utf-8",
    )
    state.paths.paper_md.write_text(_KAGGLE_PAPER, encoding="utf-8")
    example_data = run_dir / "method" / "example_data"
    example_data.mkdir(parents=True)
    unsupported = example_data / "researcher.pt"
    unsupported.write_bytes(b"not consumable by the family loader")
    monkeypatch.setenv("R2C_OFFLINE", "1")
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_stage2_script(ok=True)

    result = run_stage_2a(state)

    assert_stage_completed(result, "stage_2a")
    assert unsupported.read_bytes() == b"not consumable by the family loader"
    assert (example_data / "series.csv").is_file()
    assert (example_data / "observations.csv").is_file()
    manifest = json.loads((example_data / "PROVENANCE.json").read_text())
    assert manifest["tier"] == "family_owned_synthetic"


def test_stage_2a_hidden_json_does_not_preempt_visible_fallback_data(
    fake_subprocess, run_dir, monkeypatch,
):
    from run_pipeline import run_stage_2a

    state = make_state(run_dir)
    state.paths.method_spec.write_text(
        json.dumps(_typed_tsf_spec()), encoding="utf-8",
    )
    state.paths.paper_md.write_text(_KAGGLE_PAPER, encoding="utf-8")
    example_data = run_dir / "method" / "example_data"
    example_data.mkdir(parents=True)
    hidden = example_data / ".researcher.json"
    hidden.write_text('{"note":"not loader-visible"}\n', encoding="utf-8")
    monkeypatch.setenv("R2C_OFFLINE", "1")
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_stage2_script(ok=True)

    result = run_stage_2a(state)

    assert_stage_completed(result, "stage_2a")
    assert hidden.is_file()
    assert (example_data / "series.csv").is_file()
    assert (example_data / "observations.csv").is_file()
    manifest = json.loads((example_data / "PROVENANCE.json").read_text())
    assert manifest["tier"] == "family_owned_synthetic"


def test_stage_2a_metadata_only_provenance_does_not_preempt_or_get_overwritten(
    fake_subprocess, run_dir, monkeypatch,
):
    from run_pipeline import run_stage_2a

    state = make_state(run_dir)
    state.paths.method_spec.write_text(
        json.dumps(_typed_tsf_spec()), encoding="utf-8",
    )
    state.paths.paper_md.write_text(_KAGGLE_PAPER, encoding="utf-8")
    example_data = run_dir / "method" / "example_data"
    example_data.mkdir(parents=True)
    provenance = example_data / "PROVENANCE.json"
    original = '{"tier":"researcher_metadata_only"}\n'
    provenance.write_text(original, encoding="utf-8")
    monkeypatch.setenv("R2C_OFFLINE", "1")
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_stage2_script(ok=True)

    result = run_stage_2a(state)

    assert_stage_completed(result, "stage_2a")
    assert provenance.read_text(encoding="utf-8") == original
    assert not (example_data / "series.csv").exists()
    assert not (example_data / "observations.csv").exists()
    record = json.loads(
        (run_dir / ".pipeline" / "dataset_acquisition.json").read_text()
    )
    assert record["status"] == "fallback_refused"
    assert (
        record["offline_fallback"]["code"]
        == "offline_fallback_destination_conflict"
    )


def test_stage_2a_halts_on_changed_interrupted_public_output_without_fallback(
    fake_subprocess, run_dir, monkeypatch,
):
    import dataset_acquisition
    import time_series_offline_fallback
    from run_pipeline import run_stage_2a

    state = make_state(run_dir)
    state.paths.method_spec.write_text(
        json.dumps(_typed_tsf_spec()), encoding="utf-8",
    )
    state.paths.paper_md.write_text(_KAGGLE_PAPER, encoding="utf-8")
    example_data = run_dir / "method" / "example_data"
    example_data.mkdir(parents=True)
    changed = example_data / "sales.csv"
    changed.write_text("researcher changed this interrupted file\n", encoding="utf-8")
    marker = example_data / dataset_acquisition.PUBLIC_TRANSACTION_FILENAME
    marker.write_text(json.dumps({
        "schema_version": dataset_acquisition.PUBLIC_TRANSACTION_SCHEMA_VERSION,
        "materializer_id": dataset_acquisition.PUBLIC_MATERIALIZER_ID,
        "materializer_version": dataset_acquisition.PUBLIC_MATERIALIZER_VERSION,
        "files": [
            {"file": "sales.csv", "sha256": "0" * 64},
            {"file": "PROVENANCE.json", "sha256": "1" * 64},
        ],
    }), encoding="utf-8")

    def _unexpected(*args, **kwargs):
        raise AssertionError("unsafe public recovery must not attempt a tier")

    monkeypatch.setattr(dataset_acquisition, "acquire_for_paper", _unexpected)
    monkeypatch.setattr(
        time_series_offline_fallback,
        "generate_time_series_offline_fallback",
        _unexpected,
    )
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_stage2_script(ok=True)

    result = run_stage_2a(state)

    assert_stage_halted(
        result,
        stage_id="stage_2a",
        reason_contains="unsafe interrupted demo-data publication",
    )
    assert changed.read_text(encoding="utf-8").startswith("researcher changed")
    assert marker.is_file()
    record = json.loads(
        (run_dir / ".pipeline" / "dataset_acquisition.json").read_text()
    )
    assert record["status"] == "public_bundle_recovery_refused"
    assert record["public_bundle_recovery"]["producer_retry_consumed"] is False
    events = (run_dir / ".pipeline" / "run_events.jsonl").read_text()
    assert "demo_dataset_public_recovery_refused" in events
    assert "demo_dataset_fallback_generated" not in events


def test_stage_2a_recovers_hash_matched_interrupted_fallback_before_preemption(
    fake_subprocess, run_dir, monkeypatch,
):
    from run_pipeline import run_stage_2a
    from time_series_offline_fallback import TRANSACTION_FILENAME

    state = make_state(run_dir)
    state.paths.method_spec.write_text(
        json.dumps(_typed_tsf_spec()), encoding="utf-8",
    )
    state.paths.paper_md.write_text(_KAGGLE_PAPER, encoding="utf-8")
    example_data = run_dir / "method" / "example_data"
    example_data.mkdir(parents=True)
    partial = b"generator-owned partial series bytes\n"
    (example_data / "series.csv").write_bytes(partial)
    files = []
    for name in ("series.csv", "observations.csv", "PROVENANCE.json"):
        digest = (
            hashlib.sha256(partial).hexdigest()
            if name == "series.csv" else "0" * 64
        )
        files.append({"file": name, "sha256": digest})
    (example_data / TRANSACTION_FILENAME).write_text(json.dumps({
        "schema_version": "1.0.0",
        "generator_id": "time_series_offline_fallback",
        "generator_version": "1.0.0",
        "files": files,
    }), encoding="utf-8")
    monkeypatch.setenv("R2C_OFFLINE", "1")
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_stage2_script(ok=True)

    result = run_stage_2a(state)

    assert_stage_completed(result, "stage_2a")
    assert not (example_data / TRANSACTION_FILENAME).exists()
    manifest = json.loads((example_data / "PROVENANCE.json").read_text())
    assert manifest["tier"] == "family_owned_synthetic"
    assert (example_data / "series.csv").read_bytes() != partial


def test_stage_2a_retries_after_provenance_only_public_publication(
    fake_subprocess, run_dir, monkeypatch,
):
    import dataset_acquisition
    from run_pipeline import run_stage_2a

    state = make_state(run_dir)
    state.paths.method_spec.write_text(
        json.dumps(_typed_tsf_spec()), encoding="utf-8",
    )
    state.paths.paper_md.write_text(_KAGGLE_PAPER, encoding="utf-8")
    example_data = run_dir / "method" / "example_data"
    example_data.mkdir(parents=True)
    provenance = example_data / "PROVENANCE.json"
    provenance.write_text(
        '{"tier":"paper_cited_public","files":[]}\n', encoding="utf-8",
    )
    marker = example_data / dataset_acquisition.PUBLIC_TRANSACTION_FILENAME
    marker.write_text(json.dumps({
        "schema_version": dataset_acquisition.PUBLIC_TRANSACTION_SCHEMA_VERSION,
        "materializer_id": dataset_acquisition.PUBLIC_MATERIALIZER_ID,
        "materializer_version": dataset_acquisition.PUBLIC_MATERIALIZER_VERSION,
        "files": [{
            "file": "PROVENANCE.json",
            "sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
        }],
    }), encoding="utf-8")
    monkeypatch.setenv("R2C_OFFLINE", "1")
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_stage2_script(ok=True)

    result = run_stage_2a(state)

    assert_stage_completed(result, "stage_2a")
    assert not marker.exists()
    manifest = json.loads(provenance.read_text(encoding="utf-8"))
    assert manifest["tier"] == "family_owned_synthetic"
    record = json.loads(
        (run_dir / ".pipeline" / "dataset_acquisition.json").read_text()
    )
    assert record["status"] == "fallback_generated"
    events = (run_dir / ".pipeline" / "run_events.jsonl").read_text()
    assert "demo_dataset_public_recovered" in events
    assert "demo_dataset_fallback_generated" in events


def test_stage_2a_halts_on_changed_interrupted_synthetic_output(
    fake_subprocess, run_dir, monkeypatch,
):
    import dataset_acquisition
    import time_series_offline_fallback
    from run_pipeline import run_stage_2a
    from time_series_offline_fallback import TRANSACTION_FILENAME

    state = make_state(run_dir)
    state.paths.method_spec.write_text(
        json.dumps(_typed_tsf_spec()), encoding="utf-8",
    )
    state.paths.paper_md.write_text(_KAGGLE_PAPER, encoding="utf-8")
    example_data = run_dir / "method" / "example_data"
    example_data.mkdir(parents=True)
    changed = example_data / "series.csv"
    changed.write_text("researcher changed this interrupted file\n", encoding="utf-8")
    files = [
        {"file": name, "sha256": "0" * 64}
        for name in ("series.csv", "observations.csv", "PROVENANCE.json")
    ]
    marker = example_data / TRANSACTION_FILENAME
    marker.write_text(json.dumps({
        "schema_version": "1.0.0",
        "generator_id": "time_series_offline_fallback",
        "generator_version": "1.0.0",
        "files": files,
    }), encoding="utf-8")

    def _unexpected(*args, **kwargs):
        raise AssertionError("unsafe fallback recovery must not attempt a tier")

    monkeypatch.setattr(dataset_acquisition, "acquire_for_paper", _unexpected)
    monkeypatch.setattr(
        time_series_offline_fallback,
        "generate_time_series_offline_fallback",
        _unexpected,
    )
    fake_subprocess.expect_script(returncode=0)
    fake_subprocess.expect_stage2_script(ok=True)

    result = run_stage_2a(state)

    assert_stage_halted(
        result,
        stage_id="stage_2a",
        reason_contains="unsafe interrupted demo-data publication",
    )
    assert changed.read_text(encoding="utf-8").startswith("researcher changed")
    assert marker.is_file()
    record = json.loads(
        (run_dir / ".pipeline" / "dataset_acquisition.json").read_text()
    )
    assert record["status"] == "fallback_recovery_refused"
    assert record["offline_fallback"]["code"] == (
        "offline_fallback_transaction_conflict"
    )


def test_synthetic_bundle_briefings_preserve_tier_and_require_the_exact_files(
    tmp_path,
):
    from run_pipeline import _bundled_demo_data_section, _notebook_bundle_section

    example_data = tmp_path / "method" / "example_data"
    example_data.mkdir(parents=True)
    (example_data / "series.csv").write_text(
        "series_id,static_level,static_amplitude\n"
        "series_0000,10,2\n",
        encoding="utf-8",
    )
    (example_data / "observations.csv").write_text(
        "series_id,timestamp,target,season_sin,season_cos,time_fraction\n"
        "series_0000,2000-01-03T00:00:00Z,10,0,1,0\n",
        encoding="utf-8",
    )
    (example_data / "PROVENANCE.json").write_text(
        json.dumps({
            "tier": "family_owned_synthetic",
            "files": [
                {"file": "series.csv", "rows_kept": 1, "truncated": False},
                {
                    "file": "observations.csv",
                    "rows_kept": 1,
                    "truncated": False,
                    "time_axis": {
                        "column": "timestamp",
                        "steps_kept": 1,
                        "first_step": "2000-01-03T00:00:00Z",
                        "last_step": "2000-01-03T00:00:00Z",
                    },
                },
            ],
        }),
        encoding="utf-8",
    )

    class _Paths:
        run_dir = tmp_path

    coder = _bundled_demo_data_section(_Paths())
    notebook = _notebook_bundle_section(_Paths())

    assert coder is not None and notebook is not None
    for section in (coder, notebook):
        assert "family-owned synthetic" in section.lower()
        assert "not the paper's dataset" in section.lower()
        assert "series.csv" in section and "observations.csv" in section
        assert "REAL, from the paper's own cited source" not in section
    assert "mechanics only" in coder
    assert "not paper-comparable or benchmark evidence" in notebook
    assert "do NOT generate a second stand-in" in notebook
