"""Catalog text for homogeneous graph-mechanism probes."""

from __future__ import annotations

from probes.catalogs._shared import entry


PROBE_CATALOG = {
    "HG-1": entry(
        "graph mechanism has certified relational alignment",
        "Topology interventions are uninterpretable when node-bearing values, "
        "stable entity ids, graph endpoints, degrees, or outputs refer to "
        "different entity positions.",
    ),
    "HG-2": entry(
        "graph parameters agree across carrier, params, and runtime",
        "A constructor can appear paper-faithful only because an internal "
        "literal happens to equal the paper value while the auditable params "
        "path is missing or ignored.",
    ),
    "HG-3": entry(
        "graph construction follows the declared topology semantics",
        "A wrong threshold, cap, self-loop rule, or direction convention "
        "changes the graph before message passing while leaving tensor shapes "
        "and training execution apparently valid.",
    ),
    "HG-4": entry(
        "declared output responds to graph topology",
        "A model may accept and forward a graph argument while its output is "
        "independent of every edge, leaving the paper's graph mechanism inert.",
    ),
    "HG-5": entry(
        "declared output responds to connected neighbor signal",
        "Topology sensitivity alone cannot show that information from a "
        "neighbor reaches the destination rather than only self features or a "
        "graph-size side effect.",
    ),
    "HG-6": entry(
        "coherent entity relabeling preserves canonical graph behavior",
        "Permuting only some entity-bearing inputs can create order-dependent "
        "behavior that looks like graph sensitivity while violating the "
        "method's declared relational identity contract.",
    ),
    "HG-7": entry(
        "paper-justified graph ablation distinguishes contribution",
        "Mechanism liveness is not evidence of contribution unless the real "
        "graph route passes a discriminating check that the paper-justified "
        "null fails under the same seeded inputs.",
    ),
}
