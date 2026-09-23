"""The `# paper-element:` scan, and the reverse lookup the probe binding needs.

The scan itself is long-standing behavior that moved here from the architecture
validator so the portable harness can reach it (R2C-072). The reverse lookup is
new: given the callables a behavioral check exercised, which parts of the paper
do they implement. That is the last hop from a probe finding to the methodology
contract obligation it bears on.
"""

from __future__ import annotations

from pathlib import Path

from paper_element_anchors import (
    element_ids_by_callable,
    element_ids_for_callables,
    extract_paper_element_ids,
)

SOURCE = '''\
"""Module docstring."""
import numpy as np

CONSTANT = 1  # paper-element: concept-module-level


def select_batch(x, batch_size):
    """Pick a batch.

    # paper-element: alg-acquisition
    """
    scores = score(x)  # paper-element: eq-score
    return scores[:batch_size]


def score(x):
    # paper-element: eq-score
    return x.sum(axis=1)


class Selector:
    def rank(self, x):
        # paper-element: eq-ranking
        return x


def unannotated(x):
    return x
'''


def _package(tmp_path: Path, **files: str) -> Path:
    method = tmp_path / "method"
    method.mkdir(parents=True, exist_ok=True)
    for name, body in (files or {"method.py": SOURCE}).items():
        (method / name).write_text(body, encoding="utf-8")
    return method


def test_extract_finds_every_annotation_with_its_line(tmp_path):
    method = _package(tmp_path, **{"method.py": SOURCE})
    found = extract_paper_element_ids(method / "method.py")
    assert [i for _, i in found] == [
        "concept-module-level", "alg-acquisition", "eq-score", "eq-score",
        "eq-ranking",
    ]
    assert all(line > 0 for line, _ in found)


def test_extract_on_a_missing_file_is_empty_not_an_error(tmp_path):
    assert extract_paper_element_ids(tmp_path / "nope.py") == []


def test_ids_are_grouped_by_enclosing_callable(tmp_path):
    table = element_ids_by_callable(_package(tmp_path, **{"method.py": SOURCE}))
    assert table["select_batch"] == ("alg-acquisition", "eq-score")
    assert table["score"] == ("eq-score",)
    assert table["Selector.rank"] == ("eq-ranking",)


def test_a_module_level_anchor_belongs_to_no_callable(tmp_path):
    # It describes the file, so attributing it to whatever a probe happened to
    # exercise would bind findings to code the check never touched.
    table = element_ids_by_callable(_package(tmp_path, **{"method.py": SOURCE}))
    assert "concept-module-level" not in {i for ids in table.values() for i in ids}


def test_an_unannotated_callable_appears_nowhere(tmp_path):
    table = element_ids_by_callable(_package(tmp_path, **{"method.py": SOURCE}))
    assert "unannotated" not in table


def test_lookup_merges_and_deduplicates_across_callables(tmp_path):
    method = _package(tmp_path, **{"method.py": SOURCE})
    assert element_ids_for_callables(method, ["select_batch", "score"]) == (
        "alg-acquisition", "eq-score")


def test_a_bare_method_name_finds_its_dotted_form(tmp_path):
    # A probe that invoked a method through an instance knows `rank`, not
    # `Selector.rank`.
    method = _package(tmp_path, **{"method.py": SOURCE})
    assert element_ids_for_callables(method, ["rank"]) == ("eq-ranking",)


def test_an_unknown_callable_binds_nothing(tmp_path):
    method = _package(tmp_path, **{"method.py": SOURCE})
    assert element_ids_for_callables(method, ["unannotated", "missing"]) == ()
    assert element_ids_for_callables(method, []) == ()


def test_anchors_are_collected_across_every_module_in_the_package(tmp_path):
    method = _package(tmp_path, **{
        "method.py": "def a():\n    # paper-element: eq-one\n    pass\n",
        "model.py": "def b():\n    # paper-element: eq-two\n    pass\n",
    })
    table = element_ids_by_callable(method)
    assert table == {"a": ("eq-one",), "b": ("eq-two",)}


def test_one_qualname_in_two_files_carries_the_union(tmp_path):
    # The probe cannot tell the two apart either, so neither does the lookup.
    method = _package(tmp_path, **{
        "method.py": "def train():\n    # paper-element: eq-one\n    pass\n",
        "training.py": "def train():\n    # paper-element: eq-two\n    pass\n",
    })
    assert element_ids_for_callables(method, ["train"]) == ("eq-one", "eq-two")


def test_an_unparseable_module_contributes_nothing_rather_than_crashing(tmp_path):
    method = _package(tmp_path, **{
        "method.py": "def a(:\n    # paper-element: eq-broken\n",
        "model.py": "def b():\n    # paper-element: eq-two\n    pass\n",
    })
    assert element_ids_by_callable(method) == {"b": ("eq-two",)}


def test_a_missing_package_dir_is_empty(tmp_path):
    assert element_ids_by_callable(tmp_path / "nope") == {}
    assert element_ids_for_callables(tmp_path / "nope", ["a"]) == ()


def test_the_architecture_validator_still_exports_the_historical_names():
    # Several modules and tests import the private names; the move must not
    # break them.
    import validate_architecture_coder_output as vaco

    assert vaco._extract_paper_element_ids is extract_paper_element_ids
    assert vaco._PAPER_ELEMENT_PATTERN.search("# paper-element: alg-x")


def test_the_scan_is_vendored_into_the_portable_harness():
    # The harness cannot import the architecture validator, so if the scan were
    # still only there a researcher's re-run would adjudicate differently from
    # the delivered report.
    from build_probe_harness import _VENDORED_FILES

    assert _VENDORED_FILES["scripts/paper_element_anchors.py"] == \
        "paper_element_anchors.py"
