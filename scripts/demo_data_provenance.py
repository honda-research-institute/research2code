"""Shared interpretation and disclosure of bundled demo-data provenance.

Dataset acquisition predates the explicit provenance-tier field, so public
bundles already in delivered runs need a narrow compatibility rule.  Every
other origin claim is fail-closed: an unrecognized manifest is never described
as real data or as data from the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


PUBLIC_TIER = "paper_cited_public"
SYNTHETIC_TIER = "family_owned_synthetic"
UNKNOWN_TIER = "unknown"


@dataclass(frozen=True)
class DemoDataDisclosure:
    """Researcher-facing facts which callers may safely render verbatim."""

    tier: str
    tier_label: str
    origin_statement: str
    honesty_statement: str
    is_public: bool = False
    is_synthetic: bool = False


def provenance_path(run_dir: Path) -> Path:
    return Path(run_dir) / "method" / "example_data" / "PROVENANCE.json"


def read_provenance(run_dir: Path) -> dict[str, Any] | None:
    """Return a provenance object, or ``None`` when absent or malformed."""

    path = provenance_path(run_dir)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def provenance_tier(manifest: Mapping[str, Any]) -> str:
    """Classify explicit schema-2 tiers plus legacy public manifests.

    The legacy inference is intentionally exact: the old acquisition manifest
    identifies the paper-cited public source at ``source.cited_url``.  Mere
    presence of files or a free-text honesty note grants no origin authority.
    """

    explicit = manifest.get("tier")
    if explicit in {PUBLIC_TIER, SYNTHETIC_TIER}:
        return str(explicit)
    if explicit is None:
        compatibility_tier = manifest.get("provenance_tier")
        if compatibility_tier in {PUBLIC_TIER, SYNTHETIC_TIER}:
            return str(compatibility_tier)
    source = manifest.get("source")
    if explicit is None and isinstance(source, Mapping):
        cited_url = source.get("cited_url")
        if isinstance(cited_url, str) and cited_url.strip():
            return PUBLIC_TIER
    return UNKNOWN_TIER


def disclosure_for(manifest: Mapping[str, Any]) -> DemoDataDisclosure:
    """Build tier-specific copy with a non-overridable synthetic disclaimer."""

    tier = provenance_tier(manifest)
    if tier == SYNTHETIC_TIER:
        return DemoDataDisclosure(
            tier=tier,
            tier_label="Family-owned synthetic fallback",
            origin_statement=(
                "R2C generated this deterministic forecasting fixture locally "
                "so the package remains runnable without a public dataset."
            ),
            honesty_statement=(
                "This is synthetic data, not the paper's dataset. Results on "
                "it demonstrate package mechanics only; they are not "
                "paper-comparable or benchmark evidence."
            ),
            is_synthetic=True,
        )
    if tier == PUBLIC_TIER:
        source = manifest.get("source")
        source = source if isinstance(source, Mapping) else {}
        cited_url = str(source.get("cited_url") or "").strip()
        source_phrase = (
            f"the paper-cited public source {cited_url}"
            if cited_url
            else "a paper-cited public source"
        )
        honesty_note = str(manifest.get("honesty_note") or "").strip()
        return DemoDataDisclosure(
            tier=tier,
            tier_label="Paper-cited public data",
            origin_statement=(
                f"R2C fetched and bundled a deterministic demo-scale extract "
                f"from {source_phrase}."
            ),
            honesty_statement=honesty_note or (
                "This is real source data, but results on the demo-scale "
                "extract are not the paper's benchmark results."
            ),
            is_public=True,
        )
    return DemoDataDisclosure(
        tier=UNKNOWN_TIER,
        tier_label="Unrecognized provenance tier",
        origin_statement=(
            "A demo-data provenance manifest is present, but it does not "
            "identify a supported source tier."
        ),
        honesty_statement=(
            "No claim is made that these files are real data or data from the "
            "paper. Inspect PROVENANCE.json before interpreting results."
        ),
    )


def compact_value(value: Any) -> str:
    """Stable one-line rendering for structured provenance metadata."""

    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
