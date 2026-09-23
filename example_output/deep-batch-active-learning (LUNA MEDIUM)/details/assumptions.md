# Pipeline assumptions

Decisions the system made automatically during this run. **🔴 entries need your decision; 🟢 entries were applied for you** (FYI — you can override any of them via the "To override" line in the entry). Scan the summary, then jump to the 🔴 entries first.

_0 need your attention · 1 applied automatically._

## Summary

- 🟢 A001 — training config cannot fit a toy set at any budget on the grid

---

## A001 — training config cannot fit a toy set at any budget on the grid

**Detected.** the live config cannot fit a separable 100-sample fixture even at max_epochs=400 (acc stays below 0.583). This is not just a budget shortfall — the learning rate (0.001) or the architecture (dropout 0.5) needs attention.

**Action taken.** No change made. max_epochs was left as derived because raising it alone does not make the configuration learn.

**Reasoning.** When the live config cannot fit a separable toy set even at the epoch ceiling, the cause is the learning rate or the architecture (e.g. dropout too high for the labeled-set size), not the epoch budget. The executed-notebook sanity probe will also flag the degenerate curve downstream.

**Alternative an expert might prefer.** Lower the model dropout, lower the learning rate, or mini-batch training.

**To override.** Inspect `params.json` learning_rate and the model in `method/model.py`; adjust and re-run.

---

