"""Repo-scope dispatch write boundary (mechanical stray-write detection).

The run-dir ownership scan in run_pipeline only walks the ACTIVE run
directory, so an agent write that lands outside it entirely — a
hallucinated sibling of the runs root, an audit-style report dropped at
the repo root — used to escape silently. This module gives the dispatch
wrapper a cheap repo-level diff that makes those writes loud:

  Layer 1 (porcelain): `git status --porcelain` before/after a dispatch,
  compared as sets of (status, path). New untracked paths, newly modified
  tracked paths, and newly deleted tracked paths that are outside the run
  dir and outside the exemptions are violations.

  Layer 2 (listdir): the top-level entry names of the repo root AND of the
  runs root, before/after. This catches gitignore-invisible strays — a new
  sibling directory of the runs root, or a foreign run-slug directory
  under the (gitignored) runs root. Only NEW entries count.

Violation handling is dispatch-scoped and never run-killing: NEW stray
files/trees are quarantined into
`<run_dir>/.pipeline/quarantine/<dispatch-label>/<original-relative-path>`
(evidence preserved inside the run dir, repo left clean); modified or
deleted TRACKED files are listed but never restored or touched
(halt-and-investigate). The caller raises the existing
OutOfScopeWritesError so halt-class resolution and judge routing engage
unchanged.

Exemptions (nothing else — no agent has a sanctioned write outside the
run dir; every WRITEABLE_PATHS grant is run-dir-relative): the active run
dir itself, `.opencode/scratch/` (the sanctioned gitignored escape valve,
see .gitignore), the driver's `_history/` ledger sibling under the runs
root, and whatever the caller's `path_exempt` predicate covers (the
run-dir scan's bytecode/cache helpers are passed through here so both
scans share one notion of interpreter side traffic).

Honest limitations:
  - A git-level diff cannot distinguish an agent's write from a concurrent
    session's write to the same tree; the no-writes-during-live-runs
    discipline stays load-bearing.
  - Deep writes inside pre-existing gitignored directories other than the
    runs root are not covered (neither layer sees them).
  - A tracked file that was ALREADY dirty before the dispatch keeps the
    same porcelain entry when edited again, so a further edit to it during
    the dispatch is invisible to layer 1.

Performance: one porcelain call plus two listdirs per capture, one capture
before and one after each dispatch; no recursive walks of the repo root.
If git is unavailable or errors, the check degrades to the listdir layer
(the caller logs the warning carried on the snapshot); it never crashes a
dispatch.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

_GIT_TIMEOUT_S = 30

# The sanctioned gitignored escape valve: agents may drop draft material
# under .opencode/scratch/ mid-dispatch (see .gitignore). Everything else
# outside the run dir is a violation.
_SCRATCH_PREFIX = ".opencode/scratch"

# Driver-owned entries beside the run dirs, never agent write targets:
# run_history's ledger lives at <runs_root>/_history (see
# run_history.resolve_history_dir).
_RUNS_ROOT_DRIVER_ENTRIES = frozenset({"_history"})

# repo_root (resolved, str) -> None when the porcelain layer is usable,
# else the degradation warning. The gate result cannot change mid-run, so
# it is probed once per repo root per process.
_git_gate_cache: dict[str, str | None] = {}


def _git_gate(repo_root: Path) -> str | None:
    """None when `git status --porcelain -C repo_root` will produce paths
    aligned with repo_root; otherwise a human-readable degradation warning.
    Cached per resolved repo root."""
    key = str(Path(repo_root).resolve())
    if key in _git_gate_cache:
        return _git_gate_cache[key]
    warning: str | None
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as e:
        warning = (f"git unavailable ({type(e).__name__}: {e}); "
                   "repo-scope check degraded to the listing layer")
        _git_gate_cache[key] = warning
        return warning
    if proc.returncode != 0:
        warning = ("repo root is not inside a git work tree "
                   f"({proc.stderr.strip()[:200]}); "
                   "repo-scope check degraded to the listing layer")
    else:
        toplevel = Path(proc.stdout.strip())
        if str(toplevel.resolve()) == key:
            warning = None
        else:
            # Porcelain paths are toplevel-relative; if repo_root is merely
            # NESTED in some other repo, the paths would not align with it
            # (and the enclosing repo is not ours to diff).
            warning = (f"{repo_root} is not the git toplevel ({toplevel}); "
                       "repo-scope check degraded to the listing layer")
    _git_gate_cache[key] = warning
    return warning


def _unquote(path: str) -> str:
    """Undo git's C-style quoting of unusual paths (good-enough unescape)."""
    if len(path) >= 2 and path.startswith('"') and path.endswith('"'):
        try:
            return path[1:-1].encode("utf-8").decode("unicode_escape")
        except UnicodeDecodeError:
            return path[1:-1]
    return path


def _porcelain_entries(repo_root: Path) -> frozenset[tuple[str, str]]:
    """`git status --porcelain` as a set of (2-char status, toplevel-relative
    path). Rename/copy entries keep only their destination path."""
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        capture_output=True, text=True, timeout=_GIT_TIMEOUT_S,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git status --porcelain failed: {proc.stderr.strip()[:200]}")
    entries: set[tuple[str, str]] = set()
    for line in proc.stdout.splitlines():
        if len(line) < 4:
            continue
        status, rest = line[:2], line[3:]
        if " -> " in rest and ("R" in status or "C" in status):
            rest = rest.split(" -> ", 1)[1]
        entries.add((status, _unquote(rest)))
    return frozenset(entries)


def _listdir_names(directory: Path) -> frozenset[str]:
    try:
        return frozenset(os.listdir(directory))
    except OSError:
        return frozenset()


@dataclass(frozen=True)
class RepoScopeSnapshot:
    """One capture of repo-level state around a dispatch."""

    porcelain: frozenset[tuple[str, str]] | None  # None: git layer degraded
    repo_top: frozenset[str]
    runs_top: frozenset[str]
    git_warning: str | None = None


def capture(repo_root: Path, runs_root: Path) -> RepoScopeSnapshot:
    """Snapshot porcelain + top-level listings. Never raises for git
    problems — those degrade to porcelain=None with `git_warning` set."""
    warning = _git_gate(repo_root)
    porcelain: frozenset[tuple[str, str]] | None = None
    if warning is None:
        try:
            porcelain = _porcelain_entries(repo_root)
        except (OSError, RuntimeError, subprocess.SubprocessError) as e:
            warning = (f"git status failed ({e}); "
                       "repo-scope check degraded to the listing layer")
    return RepoScopeSnapshot(
        porcelain=porcelain,
        repo_top=_listdir_names(repo_root),
        runs_top=_listdir_names(runs_root),
        git_warning=warning,
    )


@dataclass
class RepoScopeViolations:
    """Out-of-run-dir writes detected between two snapshots. All paths are
    repo-root-relative POSIX strings."""

    new_untracked: list[str] = field(default_factory=list)
    modified_tracked: list[str] = field(default_factory=list)
    deleted_tracked: list[str] = field(default_factory=list)
    new_repo_top_entries: list[str] = field(default_factory=list)
    new_runs_top_entries: list[str] = field(default_factory=list)
    git_degraded: bool = False

    def any_violation(self) -> bool:
        return bool(
            self.new_untracked or self.modified_tracked
            or self.deleted_tracked or self.new_repo_top_entries
            or self.new_runs_top_entries
        )

    def new_stray_paths(self) -> list[str]:
        """NEW paths eligible for quarantine (never the modified/deleted
        tracked files), deduped across layers, with descendants collapsed
        into their listed ancestor so a tree is moved once."""
        seen = sorted(
            set(self.new_untracked)
            | set(self.new_repo_top_entries)
            | set(self.new_runs_top_entries)
        )
        collapsed: list[str] = []
        for rel in seen:
            if any(rel == kept or rel.startswith(kept + "/")
                   for kept in collapsed):
                continue
            collapsed.append(rel)
        return collapsed


def detect_violations(
    before: RepoScopeSnapshot,
    after: RepoScopeSnapshot,
    *,
    repo_root: Path,
    runs_root: Path,
    run_dir: Path,
    path_exempt: Callable[[str], bool] | None = None,
) -> RepoScopeViolations:
    """Diff two snapshots into violations. Only entries that appear AFTER
    the dispatch count; anything already dirty or already present before
    the dispatch is background state, not an agent write."""
    extra_exempt = path_exempt or (lambda _rel: False)

    def _rel_or_none(p: Path) -> str | None:
        try:
            return Path(p).relative_to(repo_root).as_posix()
        except ValueError:
            return None

    run_rel = _rel_or_none(run_dir)
    runs_rel = _rel_or_none(runs_root)

    def exempt(rel: str) -> bool:
        rel = rel.rstrip("/")
        if not rel or rel == ".git" or rel.startswith(".git/"):
            return True
        if run_rel is not None and (
                rel == run_rel or rel.startswith(run_rel + "/")):
            return True  # the active run dir: the run-dir scan owns it
        if rel == _SCRATCH_PREFIX or rel.startswith(_SCRATCH_PREFIX + "/"):
            return True  # sanctioned gitignored escape valve
        return extra_exempt(rel)

    v = RepoScopeViolations()
    v.git_degraded = before.porcelain is None or after.porcelain is None

    if not v.git_degraded:
        for status, path in sorted(after.porcelain - before.porcelain):
            rel = path.rstrip("/")
            if exempt(rel):
                continue
            if status == "??":
                v.new_untracked.append(rel)
            elif "D" in status:
                v.deleted_tracked.append(rel)
            else:
                v.modified_tracked.append(rel)

    for name in sorted(after.repo_top - before.repo_top):
        if runs_rel is not None and name == runs_rel:
            continue  # the runs root itself is driver infrastructure
        if name == ".opencode":
            # The scratch valve lives under it. Non-scratch content under a
            # fresh .opencode is still caught by the porcelain layer.
            continue
        if exempt(name):
            continue
        v.new_repo_top_entries.append(name)

    if Path(runs_root).resolve() != Path(repo_root).resolve():
        for name in sorted(after.runs_top - before.runs_top):
            if name == Path(run_dir).name:
                continue  # the active run's own dir
            if name in _RUNS_ROOT_DRIVER_ENTRIES:
                continue  # driver ledger, never an agent target
            if extra_exempt(name):
                continue
            rel = (f"{runs_rel}/{name}" if runs_rel is not None
                   else (Path(runs_root) / name).as_posix())
            if exempt(rel):
                continue
            v.new_runs_top_entries.append(rel)

    return v


def quarantine_new_strays(
    strays: list[str],
    *,
    repo_root: Path,
    run_dir: Path,
    label: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Move each NEW stray (file or whole directory tree) into
    `<run_dir>/.pipeline/quarantine/<label>/<original-relative-path>`.

    Returns (quarantined {orig_rel: dest rel to run_dir}, leftovers
    {orig_rel: reason}). Never raises — a failed move is reported in
    leftovers, and the caller raises OutOfScopeWritesError either way."""
    quarantined: dict[str, str] = {}
    leftovers: dict[str, str] = {}
    qroot = Path(run_dir) / ".pipeline" / "quarantine" / label
    for rel in strays:
        src = Path(repo_root) / rel
        if not src.exists():
            leftovers[rel] = "already gone before quarantine"
            continue
        dest = qroot / rel
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            suffix = 0
            while dest.exists():
                suffix += 1
                dest = qroot / f"{rel}.{suffix}"
            shutil.move(str(src), str(dest))
            quarantined[rel] = dest.relative_to(run_dir).as_posix()
        except OSError as e:
            leftovers[rel] = f"quarantine failed: {e}"
    return quarantined, leftovers


def describe_violations(
    violations: RepoScopeViolations,
    quarantined: dict[str, str],
    leftovers: dict[str, str],
) -> list[str]:
    """One human-readable line per violating path, naming what happened to
    it. These become OutOfScopeWritesError's `violations` list, so the halt
    message and the judge both see path + disposition."""
    lines: list[str] = []
    for rel in violations.new_stray_paths():
        if rel in quarantined:
            lines.append(
                f"{rel} (new path outside the run dir; quarantined to "
                f"{quarantined[rel]})")
        else:
            lines.append(
                f"{rel} (new path outside the run dir; "
                f"{leftovers.get(rel, 'left in place')})")
    for rel in violations.modified_tracked:
        lines.append(
            f"{rel} (tracked file modified during the dispatch; left in "
            f"place for investigation, not restored)")
    for rel in violations.deleted_tracked:
        lines.append(
            f"{rel} (tracked file deleted during the dispatch; not restored)")
    return lines
