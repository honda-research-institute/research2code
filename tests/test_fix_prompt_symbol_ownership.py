"""Fix-mode prompts carry the naming-bridge symbol-ownership map.

Motivating case (fedavg 2026-07-21, overnight-0721 fix proposal 3): during
stage-5 findings routing, a fix dispatch to the architecture coder ADDED
`federated_train` to method/training.py while method/method.py had owned
that symbol since stage 2.c. The naming-bridge ownership re-check then
failed with "defined in 2 files — ambiguous ownership" and the run halted.
The fix producer could not know it should import rather than re-define,
because fix-mode prompts did not carry the ownership map the pipeline
already derives.

Covers:
  - the map + import-don't-redefine instruction render when bridge data is
    derivable (the fedavg shape: symbol owned by one module, fix dispatch
    targeting another module's producer);
  - runs WITHOUT derivable bridge data keep byte-identical prompts (the
    block is omitted entirely, not rendered empty);
  - ownership resolution mirrors the gates: unique public owner wins,
    imports are never definition sites, collisions and unresolved symbols
    are omitted, the private-underscore alias remedy attributes ownership.
"""

from __future__ import annotations

import json
from pathlib import Path

import dispatch_templates
from dispatch_templates import (
    WRITEABLE_PATHS,
    DispatchPaths,
    build_fix_mode_prompt,
    format_symbol_ownership_block,
)
from symbol_ownership import load_bridge_symbol_ownership

# The fedavg mirror: `federated_train` is spec-promised and owned by
# method/method.py; the fix dispatch under test targets the ARCHITECTURE
# coder (owner of model.py/training.py), the producer that re-defined the
# symbol in the motivating run.
_FEDAVG_SPEC = {
    "try_it_out": {
        "system_provides": [
            {
                "name": "Federated training loop",
                "symbol": "federated_train",
                "type": "training",
                "symbol_kind": "function",
            },
        ],
    },
}

_FEDAVG_METHOD_FILES = {
    "method.py": (
        "def federated_train(model, client_loaders, rounds):\n"
        "    return model\n"
    ),
    "training.py": (
        "def train_from_scratch(model, loader):\n"
        "    return model\n"
    ),
    "model.py": "class FedAvgNet:\n    pass\n",
    "data.py": "def load_data():\n    return []\n",
}

_SAMPLE_FINDINGS = [
    {
        "id": "F001",
        "severity": "critical",
        "description": "Training loop drifts from the paper's aggregation.",
        "proposed_fix": "Re-align the aggregation step with Eq. 3.",
    },
]


def _write_run(tmp_path: Path, *, spec: dict | None, method_files: dict[str, str]) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / ".pipeline").mkdir(parents=True)
    if spec is not None:
        (run_dir / ".pipeline" / "method_spec.json").write_text(
            json.dumps(spec), encoding="utf-8")
    if method_files:
        (run_dir / "method").mkdir()
        for name, content in method_files.items():
            (run_dir / "method" / name).write_text(content, encoding="utf-8")
    return run_dir


def _paths(run_dir: Path) -> DispatchPaths:
    return DispatchPaths(
        spec=str(run_dir / ".pipeline" / "method_spec.json"),
        paper=str(run_dir / ".pipeline" / "paper.md"),
        paper_map=str(run_dir / ".pipeline" / "paper_map.json"),
        run_dir=str(run_dir),
        taxonomy_source="taxonomy:federated_learning",
    )


def _fix_prompt(run_dir: Path) -> str:
    return build_fix_mode_prompt(
        target_agent="r2c-architecture-coder",
        findings=_SAMPLE_FINDINGS,
        paths=_paths(run_dir),
        writeable_paths=WRITEABLE_PATHS["r2c-architecture-coder"],
    )


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def test_fix_prompt_carries_ownership_map_when_bridge_data_present(tmp_path):
    """The fedavg shape: the symbol's owner and the import-don't-redefine
    instruction must both reach the fix producer's prompt."""
    run_dir = _write_run(
        tmp_path, spec=_FEDAVG_SPEC, method_files=_FEDAVG_METHOD_FILES)
    prompt = _fix_prompt(run_dir)

    assert "**Symbol-ownership map**" in prompt
    assert "`federated_train` — defined in `method/method.py`" in prompt
    assert "IMPORT the symbol from its owning module" in prompt
    assert "do NOT write a new definition" in prompt
    assert (
        "Creating a second definition of an owned symbol is itself a defect"
        in prompt
    )
    # Placement: package-state context sits after the run-paths block and
    # ahead of the writeable-paths scope constraint.
    assert prompt.index("**Run paths:**") < prompt.index("**Symbol-ownership map**")
    assert prompt.index("**Symbol-ownership map**") < prompt.index("**Writeable paths**")


def test_fix_prompt_is_byte_identical_without_bridge_data(tmp_path, monkeypatch):
    """A run whose spec declares no bridge symbols must produce a prompt
    byte-identical to one assembled with the ownership feature inert."""
    spec_without_symbols = {"try_it_out": {"system_provides": []}}
    run_dir = _write_run(
        tmp_path, spec=spec_without_symbols, method_files=_FEDAVG_METHOD_FILES)

    prompt = _fix_prompt(run_dir)
    assert "Symbol-ownership map" not in prompt

    monkeypatch.setattr(
        dispatch_templates, "_bridge_symbol_ownership", lambda paths: {})
    inert_prompt = _fix_prompt(run_dir)
    assert prompt == inert_prompt


def test_fix_prompt_omits_block_when_spec_or_package_missing(tmp_path):
    """The shapes every pre-package fix dispatch sees (stage-1 fixes, and
    all existing tests' nonexistent sample paths) stay block-free."""
    no_spec = _write_run(
        tmp_path / "a", spec=None, method_files=_FEDAVG_METHOD_FILES)
    assert "Symbol-ownership map" not in _fix_prompt(no_spec)

    no_package = _write_run(tmp_path / "b", spec=_FEDAVG_SPEC, method_files={})
    assert "Symbol-ownership map" not in _fix_prompt(no_package)


def test_ownership_block_renders_every_mapped_symbol():
    block = format_symbol_ownership_block({
        "federated_train": "method/method.py",
        "FedAvgNet": "method/model.py",
    })
    assert "  - `federated_train` — defined in `method/method.py`" in block
    assert "  - `FedAvgNet` — defined in `method/model.py`" in block


# ---------------------------------------------------------------------------
# Ownership resolution (must mirror the gates)
# ---------------------------------------------------------------------------


def _resolve(tmp_path: Path, spec: dict, method_files: dict[str, str]) -> dict[str, str]:
    run_dir = _write_run(tmp_path, spec=spec, method_files=method_files)
    return load_bridge_symbol_ownership(
        run_dir / ".pipeline" / "method_spec.json", run_dir)


def test_unique_public_definition_site_owns_the_symbol(tmp_path):
    ownership = _resolve(tmp_path, _FEDAVG_SPEC, _FEDAVG_METHOD_FILES)
    assert ownership == {"federated_train": "method/method.py"}


def test_import_bindings_are_never_definition_sites(tmp_path):
    """Importing the symbol is the sanctioned remedy — a `from .method
    import` in another module must NOT dilute ownership."""
    files = dict(_FEDAVG_METHOD_FILES)
    files["training.py"] = (
        "from .method import federated_train\n\n"
        "def train_from_scratch(model, loader):\n"
        "    return federated_train(model, [loader], 1)\n"
    )
    ownership = _resolve(tmp_path, _FEDAVG_SPEC, files)
    assert ownership == {"federated_train": "method/method.py"}


def test_collided_symbol_is_omitted_not_arbitrated(tmp_path):
    """Two public definition sites = the exact state the gates halt on; the
    prompt map must not pick a winner."""
    files = dict(_FEDAVG_METHOD_FILES)
    files["training.py"] = (
        "def train_from_scratch(model, loader):\n"
        "    return model\n\n"
        "def federated_train(model, client_loaders, rounds):\n"
        "    return model\n"
    )
    assert _resolve(tmp_path, _FEDAVG_SPEC, files) == {}


def test_unresolved_symbol_is_omitted(tmp_path):
    """A symbol deferred to a producer that has not run yet has no owner to
    state (2.b-time fix dispatches see this shape)."""
    files = dict(_FEDAVG_METHOD_FILES)
    files.pop("method.py")  # owner not generated yet
    assert _resolve(tmp_path, _FEDAVG_SPEC, files) == {}


def test_private_underscore_variant_attributes_ownership(tmp_path):
    """The finalizer's alias-re-export remedy: a single `_symbol` definition
    site still owns the promised symbol."""
    files = dict(_FEDAVG_METHOD_FILES)
    files["method.py"] = (
        "def _federated_train(model, client_loaders, rounds):\n"
        "    return model\n"
    )
    ownership = _resolve(tmp_path, _FEDAVG_SPEC, files)
    assert ownership == {"federated_train": "method/method.py"}


def test_unparseable_module_is_skipped_like_the_gate(tmp_path):
    """`_package_module_trees` skips modules that do not parse; ownership
    still resolves from the parseable ones (never a crash)."""
    files = dict(_FEDAVG_METHOD_FILES)
    files["training.py"] = "def train_from_scratch(:\n"  # syntax error
    ownership = _resolve(tmp_path, _FEDAVG_SPEC, files)
    assert ownership == {"federated_train": "method/method.py"}


def test_no_bridge_declarations_yield_empty_map(tmp_path):
    assert _resolve(tmp_path, {"try_it_out": {}}, _FEDAVG_METHOD_FILES) == {}


def test_malformed_spec_yields_empty_map(tmp_path):
    run_dir = _write_run(
        tmp_path, spec=None, method_files=_FEDAVG_METHOD_FILES)
    spec_path = run_dir / ".pipeline" / "method_spec.json"
    spec_path.write_text("{not json", encoding="utf-8")
    assert load_bridge_symbol_ownership(spec_path, run_dir) == {}


def test_missing_spec_yields_empty_map(tmp_path):
    run_dir = _write_run(
        tmp_path, spec=None, method_files=_FEDAVG_METHOD_FILES)
    assert load_bridge_symbol_ownership(
        run_dir / ".pipeline" / "method_spec.json", run_dir) == {}
