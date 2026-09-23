"""Naming-bridge symbol ownership, derived for dispatch prompts.

The pipeline already enforces single-module ownership of every spec-promised
importable symbol (`try_it_out.system_provides` entries carrying a `symbol`,
see the naming bridge design note (internal, not shipped)): the 2.b gate
(scripts/validate_architecture_coder_output.py) and the 2.d finalizer
(scripts/finalize_package_init.py) both resolve each declared symbol against
the method/ package's top-level definition sites and fail loud on a
collision ("defined in N files — ambiguous ownership"). Neither gate
persists the resolved map — it is re-derived from the spec + package files
on every check.

This module derives the same map READ-ONLY so a fix-mode dispatch prompt
can carry it. Motivating case (fedavg 2026-07-21, overnight-0721-log.md
20:47 entry): a stage-5 fix dispatch to the architecture coder ADDED
`federated_train` to method/training.py while method/method.py had owned
that symbol since stage 2.c; the ownership re-check then halted the run on
ambiguous ownership. The fix producer could not know it should import
rather than re-define, because fix-mode prompts did not carry the ownership
map the pipeline already derives.

Resolution reuses the 2.b gate's own helpers (imported below), so the map
can never drift from what that gate enforces:

  - modules scanned = `_package_module_trees`: every parseable flat
    method/*.py, `__init__.py` excluded (it is the re-export surface, not a
    definition site); unparseable modules are skipped, exactly as the gate
    skips them;
  - a symbol's owner is the ONE module that defines it PUBLIC at top level
    (imports never count as definition sites — same rule as the gates);
  - failing that, the ONE module defining a private underscore variant
    (`_symbol`) owns it — the finalizer's deterministic alias-re-export
    remedy makes that module the delivery site;
  - ambiguous symbols (multiple definition sites) and unresolved symbols
    (no definition anywhere, e.g. deferred to a producer that has not run
    yet) are OMITTED — the gates own failing those states; the prompt map
    only states ownership facts that are true right now.

No LLM calls, no writes. Any missing/malformed input yields an empty map,
which callers render as "no block at all" (byte-identical prompts for runs
without bridge data).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Shared naming-bridge helpers — the SAME functions the 2.b gate resolves
# ownership with (and the finalizer imports two of them too), so this map
# can never drift from what the gates enforce.
from scripts.validate_architecture_coder_output import (  # noqa: E402
    _package_module_trees,
    _spec_bridge_entries,
    _top_level_defined_names,
    _top_level_public_definitions,
)


def _load_spec(spec_path: Path) -> dict | None:
    try:
        loaded = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def load_bridge_symbol_ownership(
    spec_path: str | Path, run_dir: str | Path,
) -> dict[str, str]:
    """Symbol → owning module (relative path, e.g. "method/method.py") for
    every spec-promised naming-bridge symbol that resolves to exactly one
    owner right now.

    Returns {} whenever the map is not derivable (missing/malformed spec,
    no declared symbols, or no parseable method/ modules on disk yet).
    Deterministic: entries follow the spec's declaration order.
    """
    spec = _load_spec(Path(spec_path))
    if spec is None:
        return {}
    bridge_entries = _spec_bridge_entries(spec)
    if not bridge_entries:
        return {}

    module_trees = _package_module_trees(Path(run_dir), None, None)
    if not module_trees:
        return {}
    public_defs = {
        rel: _top_level_public_definitions(tree)
        for rel, tree in module_trees.items()
    }
    all_defs = {
        rel: _top_level_defined_names(tree)
        for rel, tree in module_trees.items()
    }

    ownership: dict[str, str] = {}
    for entry in bridge_entries:
        symbol = entry["symbol"]
        if symbol in ownership:
            continue
        definition_sites = sorted(
            rel for rel, defined in public_defs.items() if symbol in defined
        )
        if len(definition_sites) == 1:
            ownership[symbol] = definition_sites[0]
            continue
        if definition_sites:
            # Already-ambiguous ownership — the gates own failing this
            # state; a prompt must not pick a winner.
            continue
        # Private-variant remedy (mirrors the finalizer): a single
        # underscore-private definition site still owns the symbol — the
        # finalizer re-exports it under the promised name. Duplicates are
        # kept (two variants in ONE module are still ambiguous).
        variant_sites = sorted(
            rel
            for rel, names in all_defs.items()
            for name in names
            if name != symbol and name.startswith("_")
            and name.lstrip("_") == symbol
        )
        if len(variant_sites) == 1:
            ownership[symbol] = variant_sites[0]
    return ownership
