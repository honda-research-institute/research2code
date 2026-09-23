#!/usr/bin/env python3
"""Refresh generated documentation blocks from docs/ssot.

This is a small, main-local adaptation of v3's docs SSOT pattern. It is
maintainer tooling only: the R2C runtime does not import or read this SSOT.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import taxonomy_role_views  # noqa: E402

SSOT_DIR = Path("docs/ssot")
INTEGRATION_SSOT = SSOT_DIR / "integration.yaml"
DOC_TARGETS = SSOT_DIR / "doc_targets.yaml"

_BEGIN_RE = re.compile(r"<!--\s*BEGIN\s+AUTO:\s*([A-Za-z0-9_.\-/]+)\s*-->")
_END_RE = re.compile(r"<!--\s*END\s+AUTO:\s*([A-Za-z0-9_.\-/]+)\s*-->")
_SKIP_MARKER_SCAN_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "r2c_runs",
}


class SSOTDocsError(RuntimeError):
    """Raised when SSOT docs cannot be validated or refreshed."""


@dataclass(frozen=True)
class MarkerSpan:
    tag: str
    begin_end: int
    end_start: int


@dataclass(frozen=True)
class TargetSpec:
    id: str
    renderer: str
    target: str
    tag: str
    description: str = ""
    mode: str = "block"


def _repo_path(repo_root: Path, rel_path: str | Path) -> Path:
    return repo_root / Path(rel_path)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SSOTDocsError(f"cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise SSOTDocsError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SSOTDocsError(f"{path} must contain a YAML mapping")
    return payload


def load_integration_ssot(repo_root: Path = ROOT) -> dict[str, Any]:
    return load_yaml(_repo_path(repo_root, INTEGRATION_SSOT))


def load_targets(repo_root: Path = ROOT) -> list[TargetSpec]:
    data = load_yaml(_repo_path(repo_root, DOC_TARGETS))
    if data.get("schema_version") != "1.0":
        raise SSOTDocsError("docs/ssot/doc_targets.yaml schema_version must be '1.0'")
    raw_targets = data.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise SSOTDocsError("docs/ssot/doc_targets.yaml must define non-empty targets")

    targets: list[TargetSpec] = []
    for index, raw in enumerate(raw_targets):
        if not isinstance(raw, dict):
            raise SSOTDocsError(f"target #{index + 1} must be a mapping")
        missing = [
            key for key in ("id", "renderer", "target", "tag") if not raw.get(key)
        ]
        if missing:
            raise SSOTDocsError(
                f"target #{index + 1} is missing required field(s): {missing}"
            )
        targets.append(
            TargetSpec(
                id=str(raw["id"]),
                renderer=str(raw["renderer"]),
                target=str(raw["target"]),
                tag=str(raw["tag"]),
                description=str(raw.get("description") or ""),
                mode=str(raw.get("mode") or "block"),
            )
        )
    _validate_unique([target.id for target in targets], "doc target id")
    return targets


def _validate_unique(values: list[str], label: str) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            errors.append(f"duplicate {label}: {value}")
        seen.add(value)
    return errors


def _list(data: dict[str, Any], key: str, errors: list[str]) -> list[dict[str, Any]]:
    value = data.get(key)
    if not isinstance(value, list):
        errors.append(f"{key} must be a list")
        return []
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            errors.append(f"{key}[{index}] must be a mapping")
        else:
            rows.append(row)
    return rows


def _ids(rows: list[dict[str, Any]], key: str = "id") -> list[str]:
    return [str(row.get(key) or "") for row in rows]


def validate_ssot_data(data: dict[str, Any], repo_root: Path = ROOT) -> list[str]:
    """Validate SSOT shape plus cheap cross-checks against runtime constants."""

    errors: list[str] = []
    if data.get("schema_version") != "1.0":
        errors.append("docs/ssot/integration.yaml schema_version must be '1.0'")

    stages = _list(data, "stages", errors)
    agents = _list(data, "agents", errors)
    validators = _list(data, "validators", errors)
    artifact_types = _list(data, "artifact_types", errors)
    candidate_artifacts = _list(data, "candidate_artifacts", errors)
    integration_status = _list(data, "integration_status", errors)

    errors.extend(_validate_unique(_ids(stages), "stage id"))
    errors.extend(_validate_unique(_ids(agents), "agent id"))
    errors.extend(_validate_unique(_ids(validators), "validator id"))
    errors.extend(_validate_unique(_ids(artifact_types), "artifact type id"))
    errors.extend(_validate_unique(_ids(integration_status), "integration status id"))

    stage_ids = set(_ids(stages))
    artifact_ids = set(_ids(artifact_types))

    for stage in stages:
        for key in ("id", "display_name", "category", "active_form"):
            if not isinstance(stage.get(key), str) or not stage.get(key):
                errors.append(f"stage {stage.get('id')!r} missing string field {key}")

    for agent in agents:
        agent_id = str(agent.get("id") or "")
        rel_file = str(agent.get("file") or "")
        if not rel_file:
            errors.append(f"agent {agent_id!r} missing file")
        elif not _repo_path(repo_root, rel_file).is_file():
            errors.append(f"agent {agent_id}: file does not exist: {rel_file}")
        if agent.get("model_tier") not in {"Think", "Code"}:
            errors.append(f"agent {agent_id}: model_tier must be Think or Code")
        for stage_id in agent.get("stages") or []:
            if stage_id not in stage_ids:
                errors.append(f"agent {agent_id}: unknown stage {stage_id!r}")

    for validator in validators:
        validator_id = str(validator.get("id") or "")
        rel_file = str(validator.get("file") or "")
        if not rel_file:
            errors.append(f"validator {validator_id!r} missing file")
        elif not _repo_path(repo_root, rel_file).is_file():
            errors.append(f"validator {validator_id}: file does not exist: {rel_file}")
        stage_id = validator.get("stage_id")
        if stage_id not in stage_ids:
            errors.append(f"validator {validator_id}: unknown stage {stage_id!r}")
        artifact = validator.get("artifact")
        if artifact != "final_manifest" and artifact not in artifact_ids:
            errors.append(f"validator {validator_id}: unknown artifact {artifact!r}")

    for artifact in artifact_types:
        artifact_id = str(artifact.get("id") or "")
        if artifact.get("source") != "final_manifest":
            errors.append(f"artifact_type {artifact_id}: source must be final_manifest")
        if not artifact.get("description"):
            errors.append(f"artifact_type {artifact_id}: description is required")

    for candidate in candidate_artifacts:
        candidate_label = f"{candidate.get('stage_id')}/{candidate.get('artifact_id')}"
        if candidate.get("stage_id") not in stage_ids:
            errors.append(f"candidate artifact {candidate_label}: unknown stage")
        if candidate.get("artifact_id") not in artifact_ids:
            errors.append(f"candidate artifact {candidate_label}: unknown artifact type")

    for item in integration_status:
        item_id = str(item.get("id") or "")
        for key in ("display_name", "status", "notes"):
            if not isinstance(item.get(key), str) or not item.get(key):
                errors.append(f"integration_status {item_id!r} missing string field {key}")
        mapping = item.get("v3_mapping")
        if not isinstance(mapping, list) or not all(isinstance(v, str) for v in mapping):
            errors.append(f"integration_status {item_id!r} v3_mapping must be string list")

    errors.extend(_validate_against_runtime(data, repo_root))
    return errors


def _validate_against_runtime(data: dict[str, Any], repo_root: Path) -> list[str]:
    errors: list[str] = []
    scripts_dir = repo_root / "scripts"
    added_paths: list[str] = []
    for path in (repo_root, scripts_dir):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
            added_paths.append(text)
    try:
        import artifact_candidates  # noqa: PLC0415
        import final_manifest  # noqa: PLC0415
        import run_pipeline  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return [f"runtime import failed during SSOT validation: {exc}"]
    finally:
        for text in added_paths:
            try:
                sys.path.remove(text)
            except ValueError:
                pass

    ssot_stages = {
        str(row.get("id")): row for row in data.get("stages", []) if isinstance(row, dict)
    }
    todo_rows = {
        stage_id: {
            "display_name": display_name,
            "active_form": active_form,
            "category": run_pipeline.STAGE_CATEGORIES.get(stage_id),
        }
        for stage_id, display_name, active_form in run_pipeline.PIPELINE_TODOS
    }
    if list(ssot_stages) != list(todo_rows):
        errors.append(
            "stages must match run_pipeline.PIPELINE_TODOS order: "
            f"ssot={list(ssot_stages)} runtime={list(todo_rows)}"
        )
    for stage_id, runtime_row in todo_rows.items():
        ssot_row = ssot_stages.get(stage_id)
        if not ssot_row:
            continue
        for key, expected in runtime_row.items():
            if ssot_row.get(key) != expected:
                errors.append(
                    f"stage {stage_id} {key} mismatch: "
                    f"ssot={ssot_row.get(key)!r} runtime={expected!r}"
                )

    if list(final_manifest.REQUIRED_STAGES) != list(todo_rows):
        errors.append("final_manifest.REQUIRED_STAGES does not match run_pipeline stages")
    for stage_id, runtime_row in todo_rows.items():
        if final_manifest.STAGE_LABELS.get(stage_id) != runtime_row["display_name"]:
            errors.append(f"final_manifest.STAGE_LABELS drift for {stage_id}")
        if final_manifest.STAGE_CATEGORIES.get(stage_id) != runtime_row["category"]:
            errors.append(f"final_manifest.STAGE_CATEGORIES drift for {stage_id}")

    ssot_artifact_types = {
        str(row.get("id")) for row in data.get("artifact_types", []) if isinstance(row, dict)
    }
    manifest_artifact_types = {spec.kind for spec in final_manifest.ARTIFACT_SPECS}
    if ssot_artifact_types != manifest_artifact_types:
        errors.append(
            "artifact_types must match final_manifest.ARTIFACT_SPECS kinds: "
            f"ssot={sorted(ssot_artifact_types)} runtime={sorted(manifest_artifact_types)}"
        )

    ssot_candidates = {
        (str(row.get("stage_id")), str(row.get("artifact_id"))): row
        for row in data.get("candidate_artifacts", [])
        if isinstance(row, dict)
    }
    runtime_candidates = artifact_candidates.ARTIFACT_REGISTRY
    if set(ssot_candidates) != set(runtime_candidates):
        errors.append(
            "candidate_artifacts must match artifact_candidates.ARTIFACT_REGISTRY: "
            f"ssot={sorted(ssot_candidates)} runtime={sorted(runtime_candidates)}"
        )
    for key, runtime_entry in runtime_candidates.items():
        ssot_row = ssot_candidates.get(key)
        if not ssot_row:
            continue
        checks = {
            "display_name": runtime_entry.display_name,
            "canonical_rel_path": runtime_entry.canonical_rel_path,
            "validator": runtime_entry.validator_label,
        }
        for field, expected in checks.items():
            if ssot_row.get(field) != expected:
                errors.append(
                    f"candidate artifact {key} {field} mismatch: "
                    f"ssot={ssot_row.get(field)!r} runtime={expected!r}"
                )
    return errors


def validate_all(repo_root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    try:
        ssot = load_integration_ssot(repo_root)
        targets = load_targets(repo_root)
    except SSOTDocsError as exc:
        return [str(exc)]
    errors.extend(validate_ssot_data(ssot, repo_root))
    errors.extend(validate_targets(repo_root, targets))
    errors.extend(audit_auto_markers(repo_root, targets))
    return errors


def validate_targets(repo_root: Path, targets: list[TargetSpec]) -> list[str]:
    errors: list[str] = []
    for target in targets:
        if target.mode not in {"block", "file_set"}:
            errors.append(f"target {target.id}: unknown mode {target.mode!r}")
            continue
        known_renderers = FILE_SET_RENDERERS if target.mode == "file_set" else RENDERERS
        if target.renderer not in known_renderers:
            errors.append(f"target {target.id}: unknown renderer {target.renderer!r}")
            continue
        target_path = _repo_path(repo_root, target.target)
        if target.mode == "file_set":
            if target_path.exists() and not target_path.is_dir():
                errors.append(f"target {target.id}: file_set target is not a directory: {target.target}")
            continue
        if not target_path.is_file():
            errors.append(f"target {target.id}: target file missing: {target.target}")
            continue
        try:
            find_marker(target_path.read_text(encoding="utf-8"), target.tag)
        except (OSError, SSOTDocsError) as exc:
            errors.append(f"target {target.id}: {exc}")
    return errors


def find_markers(text: str) -> list[MarkerSpan]:
    events: list[tuple[int, str, re.Match[str]]] = []
    for match in _BEGIN_RE.finditer(text):
        events.append((match.start(), "begin", match))
    for match in _END_RE.finditer(text):
        events.append((match.start(), "end", match))
    events.sort(key=lambda item: item[0])

    spans: list[MarkerSpan] = []
    stack: list[tuple[str, re.Match[str]]] = []
    for _, kind, match in events:
        tag = match.group(1)
        if kind == "begin":
            if any(open_tag == tag for open_tag, _ in stack):
                raise SSOTDocsError(f"nested AUTO marker for tag {tag!r}")
            stack.append((tag, match))
            continue
        if not stack or stack[-1][0] != tag:
            expected = stack[-1][0] if stack else "<none>"
            raise SSOTDocsError(
                f"END AUTO marker for tag {tag!r} without matching BEGIN "
                f"(innermost open tag is {expected!r})"
            )
        _, begin = stack.pop()
        spans.append(MarkerSpan(tag=tag, begin_end=begin.end(), end_start=match.start()))
    if stack:
        tags = ", ".join(tag for tag, _ in stack)
        raise SSOTDocsError(f"unclosed BEGIN AUTO marker(s): {tags}")
    return spans


def find_marker(text: str, tag: str) -> MarkerSpan:
    matches = [span for span in find_markers(text) if span.tag == tag]
    if not matches:
        raise SSOTDocsError(f"target text contains no AUTO marker for tag {tag!r}")
    if len(matches) > 1:
        raise SSOTDocsError(f"target text contains multiple AUTO markers for tag {tag!r}")
    return matches[0]


def splice(text: str, tag: str, body: str) -> str:
    span = find_marker(text, tag)
    normalized = body
    if not normalized.startswith("\n"):
        normalized = "\n" + normalized
    if not normalized.endswith("\n"):
        normalized += "\n"
    return text[: span.begin_end] + normalized + text[span.end_start :]


def diff_target(path: Path, tag: str, body: str) -> str:
    text = path.read_text(encoding="utf-8")
    refreshed = splice(text, tag, body)
    if text == refreshed:
        return ""
    return "".join(
        difflib.unified_diff(
            text.splitlines(keepends=True),
            refreshed.splitlines(keepends=True),
            fromfile=f"{path} (current)",
            tofile=f"{path} (refreshed)",
        )
    )


def write_target(path: Path, tag: str, body: str) -> bool:
    text = path.read_text(encoding="utf-8")
    refreshed = splice(text, tag, body)
    if text == refreshed:
        return False
    path.write_text(refreshed, encoding="utf-8")
    return True


def audit_auto_markers(repo_root: Path, targets: list[TargetSpec]) -> list[str]:
    errors: list[str] = []
    managed = {(target.target, target.tag) for target in targets if target.mode == "block"}
    seen: dict[tuple[str, str], int] = {}
    for path in _iter_doc_files(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        try:
            text = _strip_inline_code_spans(
                _strip_fenced_blocks(path.read_text(encoding="utf-8"))
            )
            spans = find_markers(text)
        except (OSError, SSOTDocsError) as exc:
            errors.append(f"{rel}: malformed AUTO marker: {exc}")
            continue
        for span in spans:
            key = (rel, span.tag)
            seen[key] = seen.get(key, 0) + 1
            if key not in managed:
                errors.append(f"{rel}: AUTO marker {span.tag!r} is not managed")
    for key, count in sorted(seen.items()):
        if count > 1:
            errors.append(f"{key[0]}: AUTO marker {key[1]!r} appears {count} times")
    for target in targets:
        if target.mode != "block":
            continue
        key = (target.target, target.tag)
        if key not in seen:
            errors.append(f"{target.target}: missing AUTO marker {target.tag!r}")
    return errors


def _iter_doc_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in repo_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".md", ".html"}:
            continue
        rel_parts = path.relative_to(repo_root).parts
        if any(part in _SKIP_MARKER_SCAN_DIRS for part in rel_parts):
            continue
        files.append(path)
    return sorted(files)


def _strip_fenced_blocks(text: str) -> str:
    out: list[str] = []
    in_fence = False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append("\n" if line.endswith("\n") else "")
        elif in_fence:
            out.append("\n" if line.endswith("\n") else "")
        else:
            out.append(line)
    return "".join(out)


def _strip_inline_code_spans(text: str) -> str:
    """Blank single-backtick inline code spans before marker auditing."""

    return re.sub(r"`[^`\n]*`", "``", text)


def _md(value: Any) -> str:
    text = str(value if value is not None else "")
    text = text.replace("|", "\\|")
    return text.replace("\n", "<br>")


def _join(values: list[Any]) -> str:
    return "<br>".join(_md(value) for value in values)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_md(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_integration_status(data: dict[str, Any]) -> str:
    lines: list[str] = [
        "<!-- Generated by scripts/refresh_ssot_docs.py from docs/ssot/integration.yaml. -->",
        "",
        "## Runtime Stage Inventory",
        "",
        _table(
            ["Stage ID", "Display Name", "Category", "Active Form"],
            [
                [
                    row["id"],
                    row["display_name"],
                    row["category"],
                    row["active_form"],
                ]
                for row in data["stages"]
            ],
        ),
        "",
        "## Agent Inventory",
        "",
        _table(
            ["Agent", "Tier", "Stage(s)", "Role", "File"],
            [
                [
                    row["id"],
                    row["model_tier"],
                    _join(row.get("stages") or []),
                    row["role"],
                    row["file"],
                ]
                for row in data["agents"]
            ],
        ),
        "",
        "## Validator Inventory",
        "",
        _table(
            ["Validator", "Stage", "Artifact", "File"],
            [
                [row["id"], row["stage_id"], row["artifact"], row["file"]]
                for row in data["validators"]
            ],
        ),
        "",
        "## Artifact Types",
        "",
        _table(
            ["Artifact Type", "Source", "Description"],
            [
                [row["id"], row["source"], row["description"]]
                for row in data["artifact_types"]
            ],
        ),
        "",
        "## Candidate Artifact Registry",
        "",
        _table(
            ["Stage", "Artifact", "Canonical Path", "Current Projection", "Validator"],
            [
                [
                    row["stage_id"],
                    row["artifact_id"],
                    row["canonical_rel_path"],
                    row["current_projection"],
                    row["validator"],
                ]
                for row in data["candidate_artifacts"]
            ],
        ),
        "",
        "## Integration Status And v3 Mapping",
        "",
        _table(
            ["Step", "Status", "v3 Mapping", "Notes"],
            [
                [
                    f"{row['id']} - {row['display_name']}",
                    row["status"],
                    _join(row["v3_mapping"]),
                    row["notes"],
                ]
                for row in data["integration_status"]
            ],
        ),
        "",
    ]
    return "\n".join(lines)


RENDERERS = {
    "integration_status": render_integration_status,
}

FILE_SET_RENDERERS = {
    "taxonomy_role_views": taxonomy_role_views.render_role_views,
}


def render_target(
    target: TargetSpec,
    data: dict[str, Any],
    repo_root: Path = ROOT,
) -> str | dict[Path, str]:
    if target.mode == "file_set":
        try:
            renderer = FILE_SET_RENDERERS[target.renderer]
        except KeyError as exc:
            raise SSOTDocsError(f"unknown renderer {target.renderer!r}") from exc
        return renderer(repo_root, Path(target.target))
    try:
        renderer = RENDERERS[target.renderer]
    except KeyError as exc:
        raise SSOTDocsError(f"unknown renderer {target.renderer!r}") from exc
    return renderer(data)


def diff_file_set(
    repo_root: Path,
    target: TargetSpec,
    rendered: dict[Path, str],
) -> str:
    lines: list[str] = []
    for rel_path, expected_text in sorted(rendered.items(), key=lambda item: item[0].as_posix()):
        path = repo_root / rel_path
        if not path.is_file():
            lines.append(f"missing: {rel_path.as_posix()}\n")
            continue
        actual_text = path.read_text(encoding="utf-8")
        if actual_text == expected_text:
            continue
        lines.append(
            "".join(
                difflib.unified_diff(
                    actual_text.splitlines(keepends=True),
                    expected_text.splitlines(keepends=True),
                    fromfile=f"{rel_path.as_posix()} (current)",
                    tofile=f"{rel_path.as_posix()} (expected)",
                )
            )
        )
    expected_abs = {(repo_root / rel_path).resolve() for rel_path in rendered}
    for path in sorted(taxonomy_role_views.generated_role_files(repo_root / target.target)):
        if path.resolve() not in expected_abs:
            rel_path = path.relative_to(repo_root).as_posix()
            lines.append(f"unexpected generated file: {rel_path}\n")
    return "".join(lines)


def write_file_set(
    repo_root: Path,
    target: TargetSpec,
    rendered: dict[Path, str],
) -> int:
    if target.renderer == "taxonomy_role_views":
        return taxonomy_role_views.write_role_views(rendered, repo_root, Path(target.target))
    raise SSOTDocsError(f"unknown file_set renderer {target.renderer!r}")


def cmd_validate(args: argparse.Namespace) -> int:
    errors = validate_all(args.repo_root)
    if errors:
        print("validate FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("validate OK: SSOT shape, targets, markers, and runtime mirrors match.")
    return 0


def cmd_refresh(args: argparse.Namespace) -> int:
    errors = validate_all(args.repo_root)
    if errors:
        print("refresh FAILED during validation:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    data = load_integration_ssot(args.repo_root)
    targets = load_targets(args.repo_root)
    changed_targets = 0
    changed_files = 0
    for target in targets:
        body = render_target(target, data, args.repo_root)
        if target.mode == "file_set":
            file_count = write_file_set(args.repo_root, target, body)
            changed_files += file_count
            if file_count:
                changed_targets += 1
                if args.verbose:
                    print(f"wrote {file_count} file(s) for {target.target}")
        elif write_target(_repo_path(args.repo_root, target.target), target.tag, body):
            changed_targets += 1
            changed_files += 1
            if args.verbose:
                print(f"wrote {target.target} :: {target.tag}")
    print(
        f"refresh OK: {changed_targets} target(s) updated, "
        f"{len(targets) - changed_targets} unchanged, {changed_files} file(s) written."
    )
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    errors = validate_all(args.repo_root)
    if errors:
        print("check FAILED during validation:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    data = load_integration_ssot(args.repo_root)
    targets = load_targets(args.repo_root)
    drift: list[tuple[TargetSpec, str]] = []
    for target in targets:
        body = render_target(target, data, args.repo_root)
        if target.mode == "file_set":
            diff = diff_file_set(args.repo_root, target, body)
        else:
            diff = diff_target(_repo_path(args.repo_root, target.target), target.tag, body)
        if diff:
            drift.append((target, diff))
    if drift:
        for target, diff in drift:
            print(f"\nDRIFT in {target.target} :: {target.tag}")
            print(diff)
        print(
            f"\ncheck FAILED: {len(drift)} target(s) out of date. "
            "Run `python3 scripts/refresh_ssot_docs.py refresh`.",
            file=sys.stderr,
        )
        return 1
    print(f"check OK: all {len(targets)} target(s) match the SSOT.")
    return 0


def cmd_list_targets(args: argparse.Namespace) -> int:
    try:
        targets = load_targets(args.repo_root)
    except SSOTDocsError as exc:
        print(f"list-targets FAILED: {exc}", file=sys.stderr)
        return 1
    for target in targets:
        description = f" - {target.description}" if target.description else ""
        print(f"{target.id}: {target.target} :: {target.tag} ({target.renderer}){description}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help="Repository root. Defaults to the current checkout.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="Validate SSOT YAML and wiring.")
    p_validate.set_defaults(func=cmd_validate)

    p_refresh = sub.add_parser("refresh", help="Refresh generated doc blocks.")
    p_refresh.add_argument("-v", "--verbose", action="store_true")
    p_refresh.set_defaults(func=cmd_refresh)

    p_check = sub.add_parser("check", help="Fail if generated docs are stale.")
    p_check.set_defaults(func=cmd_check)

    p_list = sub.add_parser("list-targets", help="List managed generated blocks.")
    p_list.set_defaults(func=cmd_list_targets)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.repo_root = args.repo_root.resolve()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
