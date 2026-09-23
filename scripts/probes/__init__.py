"""Behavioral verification harness — core (recentering slice 1.2).

This package holds the probe infrastructure from the probe catalog note (internal, not shipped):
seeded synthetic fixtures (fixtures.py), method-package loading + recording
stubs (package_loader.py), and the shared verdict schema below. Probes
themselves (UB-*/AL-*/MP-*/KD-*/TSF-*) build on these in slice 1.3.

Verdict semantics (catalog rules):
- `fail` — defect demonstrated; cites evidence.
- `flag_for_researcher` — faithful-but-degenerate (M-005 class): routes to
  assumptions.md, never auto-fix.
- `warn` — advisory; never gates.
- `not_applicable` — the conditional probe does not apply to this declared
  method shape; this is a completed, non-demoting result, not missing evidence.
- `unprobeable` — the probe could not run (missing dep, unloadable package,
  unextractable loop). First-class outcome routed to LLM review — NEVER a
  silent pass (v3's skip-collapses-to-pass lesson).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

VERDICTS = (
    "pass",
    "fail",
    "flag_for_researcher",
    "warn",
    "unprobeable",
    "not_applicable",
)


@dataclass
class ProbeVerdict:
    probe_id: str            # catalog id, e.g. "UB-2", "AL-1"
    verdict: str             # one of VERDICTS
    message: str
    tier: str = "behavioral"  # static | behavioral | paradigm | claims
    evidence: str = ""        # file/cell/line or concrete value trace
    finding_class: str = ""   # e.g. "M-003" when tied to a tracked class
    # The path join key (claims-ledger CT-3, P0 instrumentation §1.1). The
    # paper_map element ids this verdict bears on, so the claims ledger can ask
    # "is this claim's code path fidelity-clean?" against the SAME elements a
    # reviewer Finding keys on (Finding.related_elements). Empty by default and
    # left empty until the verdict machinery populates it — an empty key means
    # the fidelity gate cannot certify any path clean, which is the safe
    # fail-closed degenerate state, never a silent pass.
    element_ids: list[str] = field(default_factory=list)
    # The callables this check actually exercised (R2C-072). A probe knows this
    # and nothing else does, so the probe declares it and the battery resolves
    # it to `element_ids` through the generated code's own `# paper-element:`
    # annotations. Declaring nothing leaves the verdict unbound, which is the
    # honest state: inferring a binding from "probes usually exercise the
    # pluggable" would attach findings to elements the check never touched, and
    # a confidently wrong binding is worse than a neutral one.
    bound_callables: list[str] = field(default_factory=list)
    # Outcome arm selected by the emitter.  Additive and optional so probe
    # reports written before verification explainability remain readable.
    reason: str = ""
    # Family-specific taxonomy context is stamped by the battery at dispatch
    # time.  The renderer therefore stays dependency-free and never loads YAML.
    pack_check_id: str = ""
    pack_check: str = ""
    pack_why: str = ""
    # Which way the run's own methodology contract adjudicates this finding
    # (R2C-047): contract_contradicts | contract_consistent | contract_silent.
    # Stamped by the battery after every probe has run, from the element ids
    # above. Empty on a verdict that never went through the battery's
    # spec-conditioning pass, and "contract_silent" whenever the join found
    # nothing — the two are deliberately distinct, so a consumer can tell
    # "not adjudicated" from "adjudicated as silent".
    spec_branch: str = ""
    # Stable executor identity from the effective family's taxonomy. This is
    # distinct from the researcher-facing catalog id: a shared callable or
    # paper-map anchor may ground several obligations, so new family evidence
    # binds only where the contract deliberately declares this exact ref.
    # Empty is retained for archived probe-report compatibility; it yields an
    # id-only legacy join and can never be reference-qualified certification.
    # Appended after every pre-existing field to preserve positional callers.
    probe_ref: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"unknown verdict {self.verdict!r}")
        if not isinstance(self.probe_ref, str):
            raise ValueError("probe_ref must be a string")
        self.probe_ref = self.probe_ref.strip()
        # One registry is the structural completeness gate.  A new emitter
        # cannot silently fall back to "verification check": it must add its
        # authored explanation in the same change.
        from probes.catalog import PROBE_CATALOG  # noqa: PLC0415
        if self.probe_id not in PROBE_CATALOG:
            raise ValueError(
                f"probe id {self.probe_id!r} has no authored catalog entry"
            )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProbeReport:
    """A run of one or more probes over one artifact set."""

    target: str
    verdicts: list[ProbeVerdict] = field(default_factory=list)

    def add(self, verdict: ProbeVerdict) -> None:
        self.verdicts.append(verdict)

    def counts(self) -> dict[str, int]:
        out = {v: 0 for v in VERDICTS}
        for v in self.verdicts:
            out[v.verdict] += 1
        return out

    @property
    def gating_failures(self) -> list[ProbeVerdict]:
        return [v for v in self.verdicts if v.verdict == "fail"]

    @property
    def needs_review(self) -> list[ProbeVerdict]:
        return [v for v in self.verdicts
                if v.verdict in ("flag_for_researcher", "unprobeable")]

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "counts": self.counts(),
            "verdicts": [v.to_dict() for v in self.verdicts],
        }
