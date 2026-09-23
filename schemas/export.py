"""Regenerate JSON Schema exports from the pydantic models.

Run any time any `schemas/*.py` model changes:

    python schemas/export.py

The exported JSON Schemas are the portable form of each contract for non-Python
consumers (CI lint, IDE tooling, future TypeScript readers). The pydantic
models remain the source of truth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from schemas.arch_contract_v2 import ArchContractV2  # noqa: E402
from schemas.arch_contract_v2 import SCHEMA_VERSION as ARCH_CONTRACT_V2_VERSION  # noqa: E402
from schemas.method_spec import MethodSpec  # noqa: E402
from schemas.method_spec import SCHEMA_VERSION as METHOD_SPEC_VERSION  # noqa: E402
from schemas.paradigm_gap import ParadigmGapReport  # noqa: E402
from schemas.paradigm_gap import SCHEMA_VERSION as PARADIGM_GAP_VERSION  # noqa: E402
from schemas.paradigm_proposal import ProposalValidationResult  # noqa: E402
from schemas.paradigm_proposal import SCHEMA_VERSION as PARADIGM_PROPOSAL_VERSION  # noqa: E402
from schemas.paper_map import PaperMap  # noqa: E402
from schemas.paper_map import SCHEMA_VERSION as PAPER_MAP_VERSION  # noqa: E402
from schemas.final_manifest import FinalManifest  # noqa: E402
from schemas.final_manifest import SCHEMA_VERSION as FINAL_MANIFEST_VERSION  # noqa: E402


def _export(model_cls, version: str, out_name: str, schema_id_slug: str) -> None:
    schema = model_cls.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = f"https://r2c.local/schemas/{schema_id_slug}/v{version}"

    out_path = ROOT / "schemas" / out_name
    out_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path.relative_to(ROOT)} (schema_version={version})")


def main() -> None:
    _export(
        ArchContractV2,
        ARCH_CONTRACT_V2_VERSION,
        "arch_contract.schema.json",
        "arch_contract",
    )
    _export(MethodSpec, METHOD_SPEC_VERSION, "method_spec.schema.json", "method_spec")
    _export(PaperMap, PAPER_MAP_VERSION, "paper_map.schema.json", "paper_map")
    _export(FinalManifest, FINAL_MANIFEST_VERSION, "final_manifest.schema.json", "final_manifest")
    _export(ParadigmGapReport, PARADIGM_GAP_VERSION, "paradigm_gap.schema.json", "paradigm_gap")
    _export(
        ProposalValidationResult,
        PARADIGM_PROPOSAL_VERSION,
        "paradigm_proposal.schema.json",
        "paradigm_proposal",
    )


if __name__ == "__main__":
    main()
