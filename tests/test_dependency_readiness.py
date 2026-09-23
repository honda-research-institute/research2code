"""Read-only local dependency readiness checks.

These tests pin the restricted-network contract: inspecting a prepared Python
environment uses installed distribution metadata only.  It never invokes pip,
imports generated code, or needs network access.
"""

from __future__ import annotations

from importlib import metadata

from dependency_readiness import check_local_requirements


def _write_requirements(tmp_path, text: str):
    path = tmp_path / "requirements.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_plain_and_lower_bounded_requirements_are_satisfied_locally(tmp_path):
    requirements = _write_requirements(
        tmp_path,
        "numpy>=1.24\nscipy\ntorch>=2.0\n",
    )
    versions = {"numpy": "1.26.4", "scipy": "1.11.0", "torch": "2.1.0+cpu"}

    result = check_local_requirements(requirements, version_for=versions.__getitem__)

    assert result.satisfied is True
    assert result.problem_descriptions == ()


def test_missing_and_too_old_distributions_are_reported(tmp_path):
    requirements = _write_requirements(tmp_path, "numpy>=1.24\nscipy\n")

    def version_for(name: str) -> str:
        if name == "numpy":
            return "1.23.5"
        raise metadata.PackageNotFoundError(name)

    result = check_local_requirements(requirements, version_for=version_for)

    assert result.satisfied is False
    assert result.unsatisfied == (
        "numpy>=1.24 (installed: 1.23.5)",
        "scipy (not installed)",
    )


def test_hyphenated_distribution_names_use_distribution_metadata(tmp_path):
    requirements = _write_requirements(tmp_path, "scikit-learn\n")
    looked_up: list[str] = []

    def version_for(name: str) -> str:
        looked_up.append(name)
        return "1.5.0"

    result = check_local_requirements(requirements, version_for=version_for)

    assert result.satisfied is True
    assert looked_up == ["scikit-learn"]


def test_blank_lines_and_comments_are_ignored(tmp_path):
    requirements = _write_requirements(
        tmp_path,
        "\n# generated dependency list\nnumpy>=1.24  # numeric runtime\n",
    )

    result = check_local_requirements(
        requirements,
        version_for=lambda name: "2.0.0",
    )

    assert result.satisfied is True


def test_unsupported_requirement_shapes_fail_closed(tmp_path):
    requirements = _write_requirements(
        tmp_path,
        "-r shared-requirements.txt\nexample[extra]>=1.0\n",
    )

    result = check_local_requirements(
        requirements,
        version_for=lambda name: "9.0",
    )

    assert result.satisfied is False
    assert len(result.invalid) == 2
    assert "invalid" in result.invalid[0]
    assert "uses extras" in result.invalid[1]


def test_missing_requirements_file_fails_closed(tmp_path):
    result = check_local_requirements(tmp_path / "requirements.txt")

    assert result.satisfied is False
    assert "requirements file is missing" in (result.checker_error or "")
