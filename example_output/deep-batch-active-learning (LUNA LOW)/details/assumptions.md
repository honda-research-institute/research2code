# Pipeline assumptions

Decisions the system made automatically during this run. **🔴 entries need your decision; 🟢 entries were applied for you** (FYI — you can override any of them via the "To override" line in the entry). Scan the summary, then jump to the 🔴 entries first.

_1 need your attention · 1 applied automatically._

## Summary

- 🔴 A002 — NEEDS USER ATTENTION — finding F001
- 🟢 A001 — max_epochs raised from 8 to 200 for training sufficiency

---

## A002 — NEEDS USER ATTENTION — finding F001

**Detected.** Finding F001: The contract still references `AL-kmeanspp-incremental-distances`, but the matched batch_acquisition taxonomy declares no probe with that name. This leaves the core k-MEANS++ incremental-distance requirement unverifiable through the declared integration surface even though the generated implementation contains the corresponding live logic.

**Action taken.** No automatic action taken; the reviewer flagged this finding as having no safe expert default for this run's configuration.

**Reasoning.** When the reviewer cannot construct a reliable default reconciliation (e.g., exotic data preprocessing, contradictory spec entry, regen convergence failure), it surfaces here for human judgment rather than guessing.

**Alternative an expert might prefer.** Re-emit the full analyzer spec with this element bound to an exact applicable taxonomy probe, most likely `al_loop.microharness`, while preserving the paper-grounded requirements and feasibility fields.

**To override.** Inspect .pipeline/method_spec.json at methodology_replication_contract.elements[kmeanspp-gradient-selection].verification_probe_refs, apply a fix, and re-run the pipeline.

---

## A001 — max_epochs raised from 8 to 200 for training sufficiency

**Detected.** The derived training budget (max_epochs=8, batch=100) could not fit a separable toy set sized to the round-one labeled set (100 samples): train accuracy 0.3333 vs chance+margin 0.5833. Cross-stage incoherence — the budget was set without reference to the architecture's convergence behavior (dropout 0.5).

**Action taken.** Set params.json `params.max_epochs.value` to 200, the smallest value on the search grid that fits the toy set with batch and learning rate unchanged. batch_size was left alone — it is the acquisition batch, not a training knob.

**Reasoning.** A high-dropout model on a few-hundred-example labeled set needs more epochs than a low smoke-scale cap allows, or training stays near chance and the acquisition signal collapses with it.

**Alternative an expert might prefer.** Reduce the model's dropout, or have the training function mini-batch the labeled set so each epoch does more gradient steps, instead of raising max_epochs.

**To override.** Edit `<run_dir>/.pipeline/params.json` to set `params.max_epochs.value` to your preferred value, or revert to 8 if your training protocol differs.

---

