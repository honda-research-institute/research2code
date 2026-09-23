"""Parameter name-anchor vocabulary — dependency-free (stdlib only).

Extracted from schemas/method_spec.py (R2C-019 block 2): the params
provenance validator ships inside the portable check harness, and its
US-3b findability arm needs this vocabulary without dragging the pydantic
spec models along. method_spec re-imports everything here, so there is
exactly one implementation and the deriver, the extractor, and the
validator keep agreeing on what counts as a name anchor.
"""

from __future__ import annotations

_PARAM_TEXT_ALIASES: dict[str, tuple[str, ...]] = {
    "mc_samples": (
        "mc_samples",
        "mc samples",
        "mc sample",
        "mc dropout samples",
        "mc dropout sample",
        "monte carlo dropout samples",
        "monte carlo dropout sample",
    ),
    "eta": ("eta", "η", r"\eta"),
    "core_set_size": (
        "core_set_size",
        "core set size",
        "core-set size",
        "N_M",
        r"N_{\mathcal{M}}",
    ),
    "R_0": ("R_0", "R0", r"R_0", r"R_{0}"),
}


# Generated pluggable signatures disambiguate short paper names with a role
# suffix (the paper's `alpha`/`beta` become `alpha_loss`/`beta_loss`), so a
# name-anchored scan of paper-derived text never binds (bev-distill 2026-07-02
# F004). When a param name's trailing token is one of these role words, the
# suffix-stripped head is also a valid anchor. The head must keep >= 3 chars
# so nothing degenerate ("l_loss" -> "l") becomes an anchor.
_PARAM_ROLE_SUFFIXES = frozenset({
    "loss", "weight", "weights", "coef", "coeff", "coefficient",
    "factor", "term",
})


def param_text_aliases(param_name: str) -> tuple[str, ...]:
    """Text forms under which paper-derived prose may name this param.

    Explicit `_PARAM_TEXT_ALIASES` entries win; otherwise the name, its
    spaced form, and (for role-suffixed names) the suffix-stripped head.
    Shared by the anchored extractor in method_spec and the US-3b
    findability gate in validate_params_provenance, so the deriver and
    validator agree on what counts as a name anchor."""
    explicit = _PARAM_TEXT_ALIASES.get(param_name)
    if explicit is not None:
        return explicit
    aliases = [param_name, param_name.replace("_", " ")]
    tokens = param_name.split("_")
    if len(tokens) >= 2 and tokens[-1].lower() in _PARAM_ROLE_SUFFIXES:
        head = "_".join(tokens[:-1])
        if len(head) >= 3:
            aliases.extend([head, head.replace("_", " ")])
    return tuple(dict.fromkeys(aliases))
