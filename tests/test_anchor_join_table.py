"""Tests for scripts/build_anchor_join_table.py — the anchor join table
(shared foundation for Japan-feedback notes 3, 4, and 5).

The fixture package models the anchor shapes the design review verified on
the real ms3d run: multi-anchor ids, an orchestrator function carrying many
ids at call sites, methods inside classes, docstring-embedded anchors, a
leading-comment anchor beating a call-site anchor, and the fewest-ids
tie-break. A verbatim anchor sample extracted from the real ms3d delivery
is pinned as a literal below, and an opt-in integration test exercises the
builder against the live run dir when present.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from build_anchor_join_table import (
    ARTIFACT_REL_PATH,
    JoinTableError,
    JoinTableSetupError,
    build_join_table,
    main,
    write_join_table,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FIXTURE_METHOD_PY = '''\
"""Fixture package — anchor shapes modeled on the real ms3d delivery."""


def kde_mode_1d(values):
    """Weighted KDE mode (the dedicated implementation)."""
    # paper-element: eq-kde
    return values[0]


def kbf_fuse(sources):
    """KBF fusion — leading anchors both in and after the docstring.

    # paper-element: eq-kde
    # paper-element: concept-kbf
    """
    # paper-element: eq-kde
    # paper-element: concept-kbf
    return [kde_mode_1d(s) for s in sources]


def refine_static(tracks):
    """Static refinement — mid-body anchors only."""
    out = []
    for t in tracks:
        # paper-element: eq-kde (applied temporally)
        out.append(kde_mode_1d(t))
    # paper-element: concept-static_refinement
    return out


def generate_pipeline(data):
    """Orchestrator carrying many ids at the call sites of the
    dedicated implementations.

    # paper-element: alg-pipeline
    """
    # paper-element: alg-pipeline
    fused = kbf_fuse(data)
    # paper-element: concept-kbf
    refined = refine_static(fused)
    # paper-element: concept-static_refinement
    # paper-element: concept-tracking
    return refined
'''

_FIXTURE_MODEL_PY = '''\
"""Model fixture — methods in classes, class-body and module anchors."""

# paper-element: concept-module_level


class MultiFrameDetector:
    """Detector."""

    # paper-element: concept-detector

    def predict(self, frame):
        # paper-element: concept-detector
        return frame

    class Inner:
        def helper(self):
            # paper-element: concept-inner
            return 0
'''

_FIXTURE_IDS = [
    "eq-kde",
    "concept-kbf",
    "concept-static_refinement",
    "concept-tracking",
    "alg-pipeline",
    "concept-detector",
    "concept-module_level",
    "concept-inner",
    "concept-unanchored",  # in the paper map, never anchored
]


def _mk_run(tmp_path: Path, files: dict[str, str],
            ids: list[str] = _FIXTURE_IDS) -> Path:
    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)
    (run / ".pipeline" / "paper_map.json").write_text(json.dumps({
        "elements": [{"id": i, "type": "concept", "name": i} for i in ids],
    }), encoding="utf-8")
    for rel, content in files.items():
        target = run / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return run


@pytest.fixture
def fixture_run(tmp_path: Path) -> Path:
    return _mk_run(tmp_path, {
        "method/method.py": _FIXTURE_METHOD_PY,
        "method/model.py": _FIXTURE_MODEL_PY,
    })


def _anchor_lines(text: str, element_id: str) -> list[int]:
    """1-based lines carrying an anchor for `element_id` in fixture text."""
    return [i for i, ln in enumerate(text.split("\n"), start=1)
            if f"# paper-element: {element_id}" in ln]


# ---------------------------------------------------------------------------
# The many-to-many table
# ---------------------------------------------------------------------------


def test_all_sites_recorded_for_multi_anchor_id(fixture_run):
    table = build_join_table(fixture_run)
    sites = table["elements"]["eq-kde"]["sites"]
    assert [s["qualname"] for s in sites] == [
        "kde_mode_1d", "kbf_fuse", "kbf_fuse", "refine_static"]
    assert [s["line"] for s in sites] == \
        _anchor_lines(_FIXTURE_METHOD_PY, "eq-kde")
    assert all(s["file"] == "method/method.py" for s in sites)


def test_placements_leading_docstring_and_body(fixture_run):
    table = build_join_table(fixture_run)
    sites = table["elements"]["eq-kde"]["sites"]
    # kde_mode_1d leading comment; kbf_fuse docstring-embedded anchor and
    # post-docstring comment both count as leading; refine_static's
    # call-site anchor is mid-body.
    assert [s["placement"] for s in sites] == [
        "leading", "leading", "leading", "body"]


def test_leading_anchor_beats_call_site(fixture_run):
    # concept-kbf: leading in kbf_fuse, call-site (body) in the
    # orchestrator — the dedicated implementation wins.
    table = build_join_table(fixture_run)
    entry = table["elements"]["concept-kbf"]
    assert {s["qualname"] for s in entry["sites"]} == {
        "kbf_fuse", "generate_pipeline"}
    assert entry["primary"]["qualname"] == "kbf_fuse"
    assert entry["primary"]["placement"] == "leading"


def test_fewest_ids_tie_break(fixture_run):
    # eq-kde has leading anchors in BOTH kde_mode_1d (1 id) and kbf_fuse
    # (2 ids) — the function carrying the fewest ids wins the tie.
    table = build_join_table(fixture_run)
    primary = table["elements"]["eq-kde"]["primary"]
    assert primary["qualname"] == "kde_mode_1d"
    assert primary["placement"] == "leading"


def test_body_vs_body_tie_break_avoids_orchestrator(fixture_run):
    # concept-static_refinement is anchored mid-body in refine_static
    # (2 ids) and mid-body at the orchestrator call site (4 ids).
    table = build_join_table(fixture_run)
    primary = table["elements"]["concept-static_refinement"]["primary"]
    assert primary["qualname"] == "refine_static"


def test_call_site_only_id_maps_to_orchestrator(fixture_run):
    # concept-tracking exists ONLY as an orchestrator call-site anchor —
    # a body site is still an honest primary when it is the only site.
    table = build_join_table(fixture_run)
    entry = table["elements"]["concept-tracking"]
    assert len(entry["sites"]) == 1
    assert entry["primary"]["qualname"] == "generate_pipeline"
    assert entry["primary"]["placement"] == "body"


def test_orchestrator_docstring_anchor_is_leading(fixture_run):
    table = build_join_table(fixture_run)
    entry = table["elements"]["alg-pipeline"]
    assert entry["primary"]["qualname"] == "generate_pipeline"
    assert entry["primary"]["placement"] == "leading"
    # Both sites (docstring + leading comment) attribute to the same
    # function; the earliest line is the deterministic pick.
    assert entry["primary"]["line"] == min(s["line"] for s in entry["sites"])


# ---------------------------------------------------------------------------
# Methods, class bodies, module level
# ---------------------------------------------------------------------------


def test_method_qualname_and_class_body_placement(fixture_run):
    table = build_join_table(fixture_run)
    entry = table["elements"]["concept-detector"]
    placements = {s["placement"]: s for s in entry["sites"]}
    assert placements["class_body"]["qualname"] == "MultiFrameDetector"
    assert placements["leading"]["qualname"] == "MultiFrameDetector.predict"
    # The in-method anchor beats the class-body one.
    assert entry["primary"]["qualname"] == "MultiFrameDetector.predict"


def test_nested_class_method_qualname(fixture_run):
    table = build_join_table(fixture_run)
    primary = table["elements"]["concept-inner"]["primary"]
    assert primary["qualname"] == "MultiFrameDetector.Inner.helper"
    assert primary["placement"] == "leading"


def test_module_level_anchor(fixture_run):
    table = build_join_table(fixture_run)
    entry = table["elements"]["concept-module_level"]
    assert len(entry["sites"]) == 1
    assert entry["primary"]["qualname"] is None
    assert entry["primary"]["function_line"] is None
    assert entry["primary"]["placement"] == "module"


def test_unanchored_paper_map_id_absent(fixture_run):
    table = build_join_table(fixture_run)
    assert "concept-unanchored" not in table["elements"]


# ---------------------------------------------------------------------------
# Honest failures + empty package
# ---------------------------------------------------------------------------


def test_unknown_anchor_id_fails_honestly(tmp_path):
    run = _mk_run(tmp_path, {
        "method/data.py": "# paper-element: eq-nonexistent\nX = 1\n",
    })
    with pytest.raises(JoinTableError) as exc:
        build_join_table(run)
    msg = str(exc.value)
    assert "eq-nonexistent" in msg
    assert "method/data.py:1" in msg


def test_unknown_id_in_unvalidated_file_fails_at_table_build(tmp_path):
    # data.py is validated by no coder gate — the table build is where its
    # anchors get their id-set validation (the design note's scope rule).
    run = _mk_run(tmp_path, {
        "method/method.py": _FIXTURE_METHOD_PY,
        "method/data.py": (
            "def load():\n    # paper-element: eq-made_up\n    return 1\n"),
    })
    with pytest.raises(JoinTableError, match="eq-made_up"):
        build_join_table(run)


def test_no_anchors_produces_empty_table_not_error(tmp_path):
    run = _mk_run(tmp_path, {
        "method/data.py": "def load():\n    return 1\n",
    })
    table = build_join_table(run)
    assert table["elements"] == {}
    assert "method/data.py" in table["files_scanned"]


def test_missing_paper_map_is_setup_error(tmp_path):
    run = tmp_path / "run"
    (run / "method").mkdir(parents=True)
    (run / "method" / "data.py").write_text("X = 1\n", encoding="utf-8")
    with pytest.raises(JoinTableSetupError, match="paper_map.json"):
        build_join_table(run)


def test_missing_method_dir_is_setup_error(tmp_path):
    run = tmp_path / "run"
    (run / ".pipeline").mkdir(parents=True)
    (run / ".pipeline" / "paper_map.json").write_text(
        '{"elements": []}', encoding="utf-8")
    with pytest.raises(JoinTableSetupError, match="method"):
        build_join_table(run)


def test_unparseable_anchored_file_fails_honestly(tmp_path):
    run = _mk_run(tmp_path, {
        "method/method.py": "def broken(:\n    # paper-element: eq-kde\n",
    })
    with pytest.raises(JoinTableError, match="fails to parse"):
        build_join_table(run)


# ---------------------------------------------------------------------------
# CLI + determinism
# ---------------------------------------------------------------------------


def test_cli_round_trip(fixture_run, capsys):
    assert main(["--run-dir", str(fixture_run)]) == 0
    artifact = fixture_run / ARTIFACT_REL_PATH
    assert artifact.is_file()
    assert json.loads(artifact.read_text(encoding="utf-8")) == \
        build_join_table(fixture_run)
    assert "anchor site(s)" in capsys.readouterr().out


def test_cli_empty_package_exits_zero(tmp_path):
    run = _mk_run(tmp_path, {"method/data.py": "X = 1\n"})
    assert main(["--run-dir", str(run)]) == 0
    table = json.loads(
        (run / ARTIFACT_REL_PATH).read_text(encoding="utf-8"))
    assert table["elements"] == {}


def test_cli_unknown_id_exits_one(tmp_path, capsys):
    run = _mk_run(tmp_path, {
        "method/data.py": "# paper-element: eq-nonexistent\n",
    })
    assert main(["--run-dir", str(run)]) == 1
    assert "eq-nonexistent" in capsys.readouterr().err
    assert not (run / ARTIFACT_REL_PATH).is_file()


def test_cli_missing_run_dir_exits_two(tmp_path, capsys):
    assert main(["--run-dir", str(tmp_path / "nope")]) == 2
    assert "error:" in capsys.readouterr().err


def test_output_is_deterministic(fixture_run, tmp_path):
    out_a = write_join_table(fixture_run, tmp_path / "a.json")
    out_b = write_join_table(fixture_run, tmp_path / "b.json")
    assert out_a.read_bytes() == out_b.read_bytes()


# ---------------------------------------------------------------------------
# Real anchor sample (verbatim from the ms3d 0721-overnight delivery) + the
# opt-in integration test against the live run dir.
# ---------------------------------------------------------------------------

# Anchor placements copied verbatim from r2c_runs/ms3d/method/method.py
# (kde_mode_1d + kbf_fuse; bodies trimmed, anchor lines and docstring
# structure preserved exactly — including the docstring-embedded anchors).
_MS3D_ANCHOR_SAMPLE = '''\
def kde_mode_1d(
    values,
    weights,
    bandwidth: float = 0.5,
) -> float:
    """Return the mode of a weighted KDE over 1-D values (Eq. 1, Section 4.2).

    Args:
        values: (N,) attribute values from N box predictions.
        weights: (N,) per-detector weights w_i.
        bandwidth: KDE smoothing bandwidth h.

    Returns:
        The argument x maximising the KDE density.
    """
    # paper-element: eq-kde
    if len(values) == 0:
        return 0.0
    return float(values[0])


def kbf_fuse(
    boxes_per_source,
    weights=None,
    kde_bandwidth: float = 0.5,
):
    """Kernel Density Estimation Box Fusion (KBF) — the core fusion of MS3D++.

    # paper-element: eq-kde
    # paper-element: concept-kbf
    """
    # paper-element: eq-kde
    # paper-element: concept-kbf

    if weights is None:
        weights = [1.0] * len(boxes_per_source)
    return boxes_per_source
'''


def test_real_ms3d_anchor_sample(tmp_path):
    run = _mk_run(tmp_path, {"method/method.py": _MS3D_ANCHOR_SAMPLE},
                  ids=["eq-kde", "concept-kbf"])
    table = build_join_table(run)
    # The heuristic outcome the design review validated on the real run:
    # both functions carry a leading eq-kde anchor; kde_mode_1d carries
    # fewer ids and wins.
    eq_kde = table["elements"]["eq-kde"]
    assert len(eq_kde["sites"]) == 3
    assert eq_kde["primary"]["qualname"] == "kde_mode_1d"
    kbf = table["elements"]["concept-kbf"]
    assert kbf["primary"]["qualname"] == "kbf_fuse"
    assert all(s["placement"] == "leading" for s in kbf["sites"])


_REAL_MS3D = Path(__file__).resolve().parent.parent / "r2c_runs" / "ms3d"


@pytest.mark.manual_only
@pytest.mark.skipif(
    os.environ.get("R2C_LIVE_RUN_TESTS") != "1" or not _REAL_MS3D.is_dir(),
    reason="integration against a real green run dir — opt in with "
           "R2C_LIVE_RUN_TESTS=1 (needs r2c_runs/ms3d present)",
)
def test_integration_real_ms3d_run():
    # Read-only: builds in memory, never writes into the run dir.
    table = build_join_table(_REAL_MS3D)
    assert table["elements"], "the ms3d delivery carries anchors"
    paper_map = json.loads(
        (_REAL_MS3D / ".pipeline" / "paper_map.json").read_text(
            encoding="utf-8"))
    known = {e["id"] for e in paper_map["elements"]}
    assert set(table["elements"]) <= known
    for entry in table["elements"].values():
        assert entry["sites"]
        assert entry["primary"] in entry["sites"]
    # The design review's validated case on this run.
    assert table["elements"]["eq-kde"]["primary"]["qualname"] == \
        "kde_mode_1d"
