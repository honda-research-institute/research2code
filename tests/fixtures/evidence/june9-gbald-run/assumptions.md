# Pipeline assumptions

Decisions the system made automatically during this run. **🔴 entries need your decision; 🟢 entries were applied for you** (FYI — you can override any of them via the "To override" line in the entry). Scan the summary, then jump to the 🔴 entries first.

_0 need your attention · 1 applied automatically._

## Summary

- 🟢 A001 — R_0 rescaled from 2000.0 to 7.843

---

## A001 — R_0 rescaled from 2000.0 to 7.843

**Detected.** R_0=2000.0 is calibrated for raw [0,255] pixel distances (per spec.critical_requirements.scale_dependent_hyperparameters) but data.py applies ToTensor() which normalizes MNIST images to [0,1]. Under this normalization, pixel distances shrink by ~255×, so R_0 becomes effectively infinite and the geometric prior p(y|x,theta) = 1 if ||x-Dj||<=R_0 else R_0/||x-Dj|| loses its intended thresholding behavior.

**Action taken.** Set params.json `params.R_0.source` to 'system_default', preserved `paper_value` as 2000.0, and set runtime `value` to 7.843. Derivation: 2000.0 * (1/255) = 7.843.

**Reasoning.** The geometric prior p(y|x,theta) = R_0/||x-D_j|| uses raw L2 distances calibrated on MNIST's [0,255] pixel scale. Under ToTensor() normalization, pixel distances shrink by ~255×, so R_0 must shrink proportionally to preserve the same probability threshold the paper calibrated. This is a scale-dependent hyperparameter mismatch that breaks the algorithm's intended behavior.

**Alternative an expert might prefer.** Skip ToTensor() in data.py to preserve raw [0,255] pixel values; keeps R_0=2000.0 paper-exact but diverges from standard preprocessing conventions.

**To override.** Edit `<run_dir>/.pipeline/params.json` to set `params.R_0.value` to your preferred value (or change the data preprocessing in `method/data.py` to match the paper's scale), then re-run.

---

