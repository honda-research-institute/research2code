"""Read-only checks for a generated run's locally installed dependencies.

The pipeline must be able to answer "is this environment already prepared?"
without invoking pip or contacting a package index.  That is load-bearing for
restricted-network users who install a run's requirements while temporarily on
another network, then return and resume the same run.

Generated requirements currently use the standard requirement grammar but are
deliberately simple: distribution names, with optional lower bounds.  Parsing
is delegated to ``packaging`` so version comparison follows PEP 440 rather than
an ad-hoc tuple comparison; installed versions come from ``importlib.metadata``.
Neither operation executes package code, starts a package manager, or performs
network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class LocalRequirementsCheck:
    """Result of inspecting a requirements file against local metadata."""

    unsatisfied: tuple[str, ...] = ()
    invalid: tuple[str, ...] = ()
    checker_error: str | None = None

    @property
    def satisfied(self) -> bool:
        return not self.unsatisfied and not self.invalid and self.checker_error is None

    @property
    def problem_descriptions(self) -> tuple[str, ...]:
        problems = [*self.unsatisfied, *self.invalid]
        if self.checker_error:
            problems.append(self.checker_error)
        return tuple(problems)


def check_local_requirements(
    requirements_path: Path,
    *,
    version_for: Callable[[str], str] | None = None,
) -> LocalRequirementsCheck:
    """Compare a requirements file with installed distribution metadata.

    This function is intentionally read-only.  It never imports the generated
    package, invokes pip, or probes a network.  Unsupported/invalid requirement
    lines fail closed so callers never mistake an uncheckable environment for a
    prepared one.
    """

    if not requirements_path.is_file():
        return LocalRequirementsCheck(
            checker_error=f"requirements file is missing: {requirements_path}"
        )

    try:
        from packaging.requirements import InvalidRequirement, Requirement
    except ImportError:
        return LocalRequirementsCheck(
            checker_error=(
                "local dependency checker is unavailable because the R2C runtime "
                "dependency 'packaging' is not installed"
            )
        )

    lookup = version_for or metadata.version
    unsatisfied: list[str] = []
    invalid: list[str] = []

    try:
        lines = requirements_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return LocalRequirementsCheck(
            checker_error=f"could not read {requirements_path}: {exc}"
        )

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Standard requirements files allow an inline comment after
        # whitespace.  Generated manifests do not currently emit comments,
        # but accepting them makes the read-only checker unsurprising.
        line = line.split(" #", 1)[0].rstrip()
        try:
            requirement = Requirement(line)
        except InvalidRequirement as exc:
            invalid.append(f"line {line_number} {line!r} is invalid: {exc}")
            continue

        if requirement.url is not None:
            invalid.append(
                f"line {line_number} {line!r} uses a direct URL, which the "
                "local metadata check cannot verify"
            )
            continue
        if requirement.extras:
            invalid.append(
                f"line {line_number} {line!r} uses extras, which generated R2C "
                "requirements do not support"
            )
            continue
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue

        try:
            installed_version = lookup(requirement.name)
        except metadata.PackageNotFoundError:
            unsatisfied.append(f"{line} (not installed)")
            continue
        except Exception as exc:  # noqa: BLE001 - metadata backends are pluggable
            return LocalRequirementsCheck(
                unsatisfied=tuple(unsatisfied),
                invalid=tuple(invalid),
                checker_error=(
                    f"could not inspect installed distribution {requirement.name!r}: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        if requirement.specifier and installed_version not in requirement.specifier:
            unsatisfied.append(f"{line} (installed: {installed_version})")

    return LocalRequirementsCheck(
        unsatisfied=tuple(unsatisfied),
        invalid=tuple(invalid),
    )
