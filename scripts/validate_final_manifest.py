#!/usr/bin/env python3
"""Validate `<RUN_DIR>/final_manifest.json` schema and artifact hashes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from final_manifest import FinalManifestError, validate_final_manifest_file  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory to resolve manifest artifact paths against.",
    )
    args = parser.parse_args(argv)

    try:
        diagnostics = validate_final_manifest_file(args.manifest, run_dir=args.run_dir)
    except FinalManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if diagnostics:
        for diagnostic in diagnostics:
            print(f"error: {diagnostic}", file=sys.stderr)
        return 1
    print(f"ok: final manifest validates at {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
