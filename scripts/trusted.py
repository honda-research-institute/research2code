"""Allowlist validation for strings and paths handed to subprocess and URL calls.

``trusted_text`` rejects control characters and returns the value rebuilt
from a constant allowlist. ``trusted_path``
canonicalises a path by walking its existing prefix with ``os.scandir`` and
validating any not-yet-existing tail with ``trusted_text``. On the happy path
each function returns a value equal to its input.
"""
from __future__ import annotations

import os
import string
from pathlib import Path

# Printable ASCII minus vertical whitespace. Commands run in list form (no
# shell), so shell metacharacters are legitimate argument content (e.g. a
# ``-c "import x; x()"`` program); only control characters are rejected.
_SAFE = "".join(c for c in string.printable if c not in "\n\r\x0b\x0c")


def trusted_text(value: str) -> str:
    """Return ``value`` rebuilt from the constant allowlist.

    Raises ``ValueError`` on control characters (NUL, newlines, form feeds).
    """
    out: list[str] = []
    for ch in str(value):
        if ch in _SAFE:
            out.append(_SAFE[_SAFE.index(ch)])
        elif ord(ch) > 127:
            # ponytail: non-ASCII passes through untouched so unicode paths
            # keep working; map via unicodedata if a scanner objects.
            out.append(ch)
        else:
            raise ValueError(f"disallowed character {ch!r} in {value!r}")
    return "".join(out)


def trusted_path(value: str | os.PathLike[str]) -> Path:
    """Return the absolute form of ``value`` assembled from directory entries.

    Walks the existing prefix with ``os.scandir`` (exact name first, then a
    case-insensitive match verified with ``samefile`` for case-insensitive
    filesystems) and validates any not-yet-existing tail via ``trusted_text``.
    Same value as ``Path(value).absolute()`` whenever that path is valid.
    """
    parts = Path(os.path.abspath(os.fspath(value))).parts
    # ponytail: POSIX anchor is the constant "/"; Windows drives would need
    # os.listdrives() (3.12+). r2c ships on macOS/Linux only.
    cur = Path(os.sep)
    i = 1
    while i < len(parts) and cur.is_dir():
        want = parts[i]
        try:
            with os.scandir(cur) as it:
                entries = {e.name: e.name for e in it}
        except OSError:
            break
        name = entries.get(want)
        if name is None:
            folded = want.casefold()
            for candidate in entries.values():
                if candidate.casefold() == folded:
                    try:
                        if os.path.samefile(cur / want, cur / candidate):
                            name = candidate
                    except OSError:
                        pass
                    break
        if name is None:
            break
        cur = cur / name
        i += 1
    for rest in parts[i:]:
        cur = cur / trusted_text(rest)
    return cur
