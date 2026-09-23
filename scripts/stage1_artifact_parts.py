"""Assemble chunked Stage 1 artifacts into canonical JSON files.

Think-class agents may be capped below the size needed for one large JSON
tool-call payload. These helpers let Stage 1 agents emit bounded JSON parts
while preserving the existing downstream interface:

  - .pipeline/paper_map.json
  - .pipeline/method_spec.json

The assembler is intentionally mechanical. It does not repair or infer schema
content; the existing validators remain the source of truth for correctness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from schemas.method_spec import SCHEMA_VERSION as METHOD_SPEC_SCHEMA_VERSION
from schemas.method_spec import MethodSpec

# The non-Optional MethodSpec top-level fields, derived from the schema so the
# assembler's completeness contract cannot drift from schemas/method_spec.py.
# Optional fields (schema_version, scenario assumptions, and the three
# methodology-fidelity fields) are intentionally excluded. A complete spec
# may omit them.
REQUIRED_METHOD_SPEC_KEYS = frozenset(
    name for name, f in MethodSpec.model_fields.items() if f.is_required()
)


@dataclass(frozen=True)
class AssemblyResult:
    assembled: bool
    output_path: Path
    source_dir: Path
    message: str
    part_paths: list[Path] = field(default_factory=list)


METHOD_SPEC_TOP_LEVEL_KEYS = {
    "paper",
    "core_method",
    "paper_claims",
    "try_it_out",
    "data_requirements",
    "dependencies",
    "repo",
    "comparison",
    "critical_requirements",
    "scenario_assumptions",
    "methodology_replication_contract",
    "methodology_contract_pack",
    "replication_feasibility",
    "feasibility",
}


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _failure(output_path: Path, source_dir: Path, message: str) -> AssemblyResult:
    return AssemblyResult(
        assembled=False,
        output_path=output_path,
        source_dir=source_dir,
        message=message,
    )


def _safe_child_path(source_dir: Path, rel_path: str) -> Path:
    if not isinstance(rel_path, str) or not rel_path.strip():
        raise ValueError(f"part path must be a non-empty string, got {rel_path!r}")
    if Path(rel_path).is_absolute():
        raise ValueError(f"part path must be relative to {source_dir}: {rel_path!r}")
    candidate = (source_dir / rel_path).resolve()
    source_root = source_dir.resolve()
    if candidate != source_root and source_root not in candidate.parents:
        raise ValueError(f"part path escapes {source_dir}: {rel_path!r}")
    return candidate


def _load_manifest(
    source_dir: Path,
    names: tuple[str, ...],
) -> tuple[dict[str, Any], Path | None]:
    for name in names:
        path = source_dir / name
        if not path.is_file():
            continue
        data = _load_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"{path.name} must be a JSON object")
        return data, path
    return {}, None


def _default_part_files(source_dir: Path, *, exclude: set[Path]) -> list[Path]:
    files = []
    for path in sorted(source_dir.rglob("*.json")):
        if path in exclude:
            continue
        if not path.is_file():
            continue
        files.append(path)
    return files


def _manifest_part_files(
    source_dir: Path,
    manifest: dict[str, Any],
    field_name: str,
    *,
    exclude: set[Path],
) -> list[Path]:
    raw = manifest.get(field_name)
    if raw is None:
        return _default_part_files(source_dir, exclude=exclude)
    if not isinstance(raw, list):
        raise ValueError(
            f"manifest field {field_name!r} must be a list of relative paths"
        )
    return [_safe_child_path(source_dir, entry) for entry in raw]


def parts_newer_than_output(source_dir: Path, output_path: Path) -> bool:
    """True when any JSON part is newer than the canonical output."""
    if not source_dir.is_dir():
        return False
    try:
        output_mtime = output_path.stat().st_mtime
    except OSError:
        return True
    for path in source_dir.rglob("*.json"):
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime > output_mtime:
                return True
        except OSError:
            continue
    return False


def assemble_paper_map_parts(
    pipeline_dir: Path,
    output_path: Path | None = None,
) -> AssemblyResult:
    """Assemble `.pipeline/paper_map_parts/` into `paper_map.json`.

    Expected manifest shape:

        {
          "schema_version": "1.0.0",
          "title": "Paper title",
          "element_files": ["elements/001-core.json", "..."]
        }

    Each element file must be the element object itself. Full paper-map
    envelopes and wrapper objects are rejected so the producer has one
    mechanical chunk shape.
    """
    output = output_path or (pipeline_dir / "paper_map.json")
    source_dir = pipeline_dir / "paper_map_parts"
    if not source_dir.is_dir():
        return _failure(output, source_dir, "paper_map_parts directory is absent")

    try:
        manifest, manifest_path = _load_manifest(
            source_dir,
            ("manifest.json", "outline.json"),
        )
        exclude = {manifest_path} if manifest_path is not None else set()
        part_paths = _manifest_part_files(
            source_dir, manifest, "element_files", exclude=exclude,
        )
        title = manifest.get("title")
        if not isinstance(title, str) or not title.strip():
            return _failure(
                output,
                source_dir,
                "paper_map_parts manifest must include a non-empty title",
            )
        schema_version = manifest.get("schema_version", "1.0.0")
        elements: list[dict[str, Any]] = []
        for path in part_paths:
            if not path.is_file():
                raise ValueError(f"element part not found: {path}")
            data = _load_json(path)
            if not isinstance(data, dict):
                raise ValueError(f"element part must be a JSON object: {path}")
            if "elements" in data or "title" in data:
                raise ValueError(
                    f"element part must contain exactly one paper-map element, "
                    f"not a full paper_map envelope: {path}"
                )
            if set(data) == {"element"} and isinstance(data.get("element"), dict):
                raise ValueError(
                    f"element part must contain the element object directly, "
                    f"not an {{'element': ...}} wrapper: {path}"
                )
            elements.append(data)
        if not elements:
            return _failure(
                output,
                source_dir,
                "paper_map_parts contains no element JSON files",
            )
        assembled = {
            "schema_version": schema_version,
            "title": title,
            "elements": elements,
        }
        _write_json(output, assembled)
        return AssemblyResult(
            assembled=True,
            output_path=output,
            source_dir=source_dir,
            message=f"assembled paper_map.json from {len(part_paths)} part file(s)",
            part_paths=part_paths,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _failure(
            output,
            source_dir,
            f"failed to assemble paper_map_parts: {exc}",
        )


def assemble_method_spec_parts(
    pipeline_dir: Path,
    output_path: Path | None = None,
) -> AssemblyResult:
    """Assemble `.pipeline/method_spec_parts/` into `method_spec.json`.

    Expected manifest shape:

        {
          "schema_version": "1.0.0",
          "part_files": ["paper.json", "core_method.json", "..."]
        }

    Each part file must be a wrapper object with exactly one MethodSpec
    top-level key, e.g. `{"paper": {...}}` or
    `{"feasibility": "reproducible"}`. Direct section values are rejected so
    producers do not have to choose between equivalent chunk shapes.
    """
    output = output_path or (pipeline_dir / "method_spec.json")
    source_dir = pipeline_dir / "method_spec_parts"
    if not source_dir.is_dir():
        return _failure(output, source_dir, "method_spec_parts directory is absent")

    try:
        manifest, manifest_path = _load_manifest(source_dir, ("manifest.json",))
        if manifest_path is None:
            # Require an explicit manifest. Globbing whatever parts happen to be
            # on disk silently assembles an INCOMPLETE spec — the 3-of-13 partial
            # that reached --strict as the cryptic "9 fields missing" failure
            # (bayesian-active-learning, 2026-06-24). Without a manifest we cannot
            # know the intended part set, so fail (writing nothing) and let the
            # driver's missing-output retry re-dispatch the analyzer with the
            # full-parts prompt. (DRV-C1)
            return _failure(
                output, source_dir,
                "method_spec_parts present but manifest.json is missing; cannot "
                "determine the intended part set. Re-dispatch must write "
                "manifest.json listing every part_file AND all parts in one turn.",
            )
        exclude = {manifest_path}
        part_paths = _manifest_part_files(
            source_dir, manifest, "part_files", exclude=exclude,
        )
        # schema_version is the schema's/driver's concern, not the producer's:
        # stamp the canonical value rather than trusting the manifest, whose
        # value has drifted in prompt examples ("1.0.0"/"1.3.0") from the schema.
        # (DRV-C2)
        assembled: dict[str, Any] = {
            "schema_version": METHOD_SPEC_SCHEMA_VERSION,
        }
        for path in part_paths:
            if not path.is_file():
                raise ValueError(f"method spec part not found: {path}")
            data = _load_json(path)
            if not isinstance(data, dict):
                raise ValueError(
                    f"method spec part must be a wrapper object with exactly "
                    f"one MethodSpec top-level key: {path}"
                )
            keys = set(data)
            if len(keys) != 1:
                raise ValueError(
                    f"method spec part must wrap exactly one MethodSpec "
                    f"top-level key: {path}; got keys {sorted(keys)}"
                )
            key = next(iter(keys))
            value = data[key]
            if key == "schema_version":
                raise ValueError(
                    f"schema_version belongs in manifest.json, not section "
                    f"part {path}"
                )
            fragment = {key: value}
            for key, value in fragment.items():
                if key not in METHOD_SPEC_TOP_LEVEL_KEYS:
                    raise ValueError(
                        f"unknown MethodSpec top-level key {key!r} in {path}"
                    )
                if key in assembled:
                    raise ValueError(
                        f"duplicate MethodSpec top-level key {key!r} in {path}"
                    )
                assembled[key] = value
        if len(assembled) == 1:
            return _failure(
                output,
                source_dir,
                "method_spec_parts contains no section JSON files",
            )
        missing_required = sorted(REQUIRED_METHOD_SPEC_KEYS - set(assembled))
        if missing_required:
            # Completeness contract: never assemble (or write) a spec missing a
            # REQUIRED MethodSpec field. This surfaces an actionable, EARLY
            # diagnostic — "wrote N of M required parts; missing: [...]" — instead
            # of the opaque pydantic "Field required" errors a full validate+judge
            # cycle later, and leaves method_spec.json unwritten so the driver's
            # missing-output retry fires with the missing-part names. (DRV-C1)
            return _failure(
                output,
                source_dir,
                f"assembled method_spec is incomplete: missing required "
                f"section(s) {missing_required}. The analyzer wrote "
                f"{len(assembled) - 1} of {len(REQUIRED_METHOD_SPEC_KEYS)} required "
                f"parts; re-dispatch must write ALL parts (and manifest.json) in "
                f"one turn.",
            )
        _write_json(output, assembled)
        return AssemblyResult(
            assembled=True,
            output_path=output,
            source_dir=source_dir,
            message=f"assembled method_spec.json from {len(part_paths)} part file(s)",
            part_paths=part_paths,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _failure(
            output,
            source_dir,
            f"failed to assemble method_spec_parts: {exc}",
        )
