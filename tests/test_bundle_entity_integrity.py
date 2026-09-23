"""R2C-078: entity subsampling keeps every bundled table joinable."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import pytest


def _zip(members: dict[str, bytes]) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for name, body in members.items():
            archive.writestr(name, body)
    return payload.getvalue()


def _panel(*, products: int, dates: int = 3) -> bytes:
    lines = ["product_id,date,sales"]
    for day in range(dates):
        for product in range(products):
            lines.append(f"P{product},2017-01-{day + 2:02d},{product}.0")
    return ("\n".join(lines) + "\n").encode()


def _hierarchy(*, products: int) -> bytes:
    lines = ["product_id,cluster_id"]
    lines += [f"P{product},C{product % 2}" for product in range(products)]
    return ("\n".join(lines) + "\n").encode()


def _acquisition(tmp_path: Path, payload: bytes):
    from dataset_acquisition import Acquisition, classify_source

    source_path = tmp_path / "retail.zip"
    source_path.write_bytes(payload)
    source = classify_source(
        "https://www.kaggle.com/datasets/berkayalan/retail-sales-data")
    return Acquisition(
        source=source,
        path=source_path,
        sha256="de" * 32,
        bytes_written=len(payload),
        truncated=False,
    )


def _values(path: Path, column: str) -> set[str]:
    with path.open(encoding="utf-8", newline="") as source:
        return {row[column] for row in csv.DictReader(source)}


def test_the_2026_08_07_mismatch_fails_with_both_counts(tmp_path):
    from dataset_acquisition import bundle_referential_integrity_error

    sales = tmp_path / "sales.csv"
    hierarchy = tmp_path / "product_hierarchy.csv"
    sales.write_bytes(_hierarchy(products=30))
    hierarchy.write_bytes(_hierarchy(products=699))
    files = [
        {"file": sales.name, "entity_column": "product_id",
         "entities_kept": 30, "entities_in_source": 699,
         "entity_filter_applied": True},
        {"file": hierarchy.name, "entity_column": "product_id",
         "entities_kept": 699, "entities_in_source": 699,
         "entity_filter_applied": True},
    ]

    error = bundle_referential_integrity_error(files, tmp_path)

    assert error is not None
    assert "`product_id`" in error
    assert "`sales.csv`" in error
    assert "`product_hierarchy.csv`" in error
    assert "30 distinct" in error and "699" in error


def test_a_broader_source_dimension_is_filtered_to_the_retained_set(tmp_path):
    """The dimension is legitimately broader before subsampling.

    The floor judges the two post-materialization sets.  It does not reject
    the unequal source populations of four fact products and six dimension
    products.
    """
    from dataset_acquisition import materialize_demo_bundle
    from run_pipeline import _bundled_demo_data_section, _notebook_bundle_section

    # Dimension first proves propagation does not depend on archive order.
    payload = _zip({
        "product_hierarchy.csv": _hierarchy(products=6),
        "sales.csv/sales.csv": _panel(products=4),
    })
    example_data = tmp_path / "method" / "example_data"
    manifest = materialize_demo_bundle(
        _acquisition(tmp_path, payload),
        example_data,
        max_rows_per_table=100,
        max_rows_per_axis_table=6,
    )

    assert _values(example_data / "sales.csv", "product_id") == {"P0", "P1"}
    assert _values(
        example_data / "product_hierarchy.csv", "product_id") == {"P0", "P1"}
    by_name = {entry["file"]: entry for entry in manifest["files"]}
    assert (by_name["sales.csv"]["entities_in_source"],
            by_name["sales.csv"]["entities_kept"]) == (4, 2)
    assert (by_name["product_hierarchy.csv"]["entities_in_source"],
            by_name["product_hierarchy.csv"]["entities_kept"]) == (6, 2)

    class _Paths:
        run_dir = tmp_path

    for section in (
        _bundled_demo_data_section(_Paths()),
        _notebook_bundle_section(_Paths()),
    ):
        assert section is not None
        assert section.count(
            "ENTITY SET: column `product_id` carries 2 distinct") == 2
        assert "join by the named key rather than row position" in section


def test_a_broader_dimension_is_unchanged_when_no_entity_was_subsampled(
    tmp_path,
):
    from dataset_acquisition import materialize_demo_bundle

    payload = _zip({
        "sales.csv": _panel(products=2),
        "product_hierarchy.csv": _hierarchy(products=3),
    })
    example_data = tmp_path / "bundle"
    manifest = materialize_demo_bundle(
        _acquisition(tmp_path, payload),
        example_data,
        max_rows_per_table=100,
        max_rows_per_axis_table=100,
    )

    assert _values(
        example_data / "product_hierarchy.csv", "product_id") == {
            "P0", "P1", "P2"}
    hierarchy = next(
        entry for entry in manifest["files"]
        if entry["file"] == "product_hierarchy.csv")
    assert "entity_column" not in hierarchy


def test_a_missing_retained_entity_refuses_atomically(tmp_path):
    from dataset_acquisition import (
        BundleReferentialIntegrityError,
        materialize_demo_bundle,
    )

    payload = _zip({
        "sales.csv": _panel(products=4),
        "product_hierarchy.csv": _hierarchy(products=1),
    })
    acquisition = _acquisition(tmp_path, payload)
    example_data = tmp_path / "bundle"

    with pytest.raises(BundleReferentialIntegrityError) as exc_info:
        materialize_demo_bundle(
            acquisition,
            example_data,
            max_rows_per_table=100,
            max_rows_per_axis_table=6,
        )

    message = str(exc_info.value)
    assert "`product_id`" in message
    assert "1 distinct" in message and "2" in message
    assert not acquisition.path.exists()
    assert not list(example_data.glob("*.csv"))
    assert not (example_data / "PROVENANCE.json").exists()
