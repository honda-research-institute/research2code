"""Repo-root .env loader — opt-in configuration without shell-profile edits.

The Python entry points (run_pipeline, parse_pdf) call load_env_file()
at the top of main(), and r2c-start.sh sources the same file before
starting opencode, so one .env beside the repo README configures a
fresh machine. Two rules
keep it unsurprising:

- The shell always wins: a variable already present in the environment
  is never overridden by the file.
- The file is bash-compatible: "export KEY=VALUE" lines, # comments,
  blank lines, and single/double quotes around values all parse, so the
  same file can be `source`d directly.

The opencode server itself reads only what r2c-start.sh exports from
this file at launch (R2C_MODEL and the provider API keys); everything
the driver consumes — the Marker URL included — is loaded here.
"""

from __future__ import annotations

import os
from pathlib import Path


def default_env_path() -> Path:
    """<repo-root>/.env — one directory above scripts/."""
    return Path(__file__).resolve().parent.parent / ".env"


def load_env_file(path: Path | None = None) -> dict[str, str]:
    """Apply KEY=VALUE lines from .env for keys not already in the
    environment. Returns what was applied. Missing file is a no-op;
    malformed lines are skipped, never fatal."""
    if path is None:
        path = default_env_path()
    if not path.is_file():
        return {}
    applied: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key or any(ch.isspace() for ch in key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied
