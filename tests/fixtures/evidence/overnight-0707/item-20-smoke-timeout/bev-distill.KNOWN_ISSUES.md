# Known issues

This package was produced end-to-end, but the items below are quality gates R2C could not fully satisfy automatically within its retry budget. **The artifacts are delivered as best-effort** — review these before relying on the output. Each entry says what failed, where, and what to try (including handing the notebook + this file to an AI assistant like Claude to finish). R2C never edits your code to force a gate to pass; it surfaces issues here honestly instead.

---

## stage_3c — the notebook does not run end-to-end

<!-- issue_signature: stage_3c | the notebook does not run end-to-end | notebook.ipynb — cell 44 (section 5) | smoke gate still failed after cap=3 + one judge-blessed extra iteration (judge picked r2c-notebook-generator). stderr: osses = [] curve_teacher.eval() for p in curve_teacher.parameters(): p.requires_grad = false optimizer = torch.optim.adamw(curve_student.parameters(), lr=cfg["learning_rate"]) curve_num_epochs = 3 # smoke-scale: 3 epochs stays within the 180s per-cell timeout (spec-approved range: 3-5) for epoch in range(curve_num_epochs): total_loss_sum = 0.0 kd_loss_sum = 0.0 sup_loss_sum = 0.0 n_batches = 0 n = len(student_train) indices = list(range(n)) np.random.shuffle(indices) for start in range(0, n, cfg["batch_size"]): ... (truncated; full length 3798) -->

**Where.** notebook.ipynb — cell 44 (section 5)

**What to do.** The notebook was generated and runs up to the failing cell, but that cell raises an error. Open the notebook and fix the cell against method/, or hand the notebook + this file to an AI assistant (Claude/ChatGPT) and ask it to fix the failing cell. The traceback is in the technical detail below.

**Technical detail.** smoke gate STILL failed after cap=3 + one judge-blessed extra iteration (judge picked r2c-notebook-generator). stderr: osses = []

curve_teacher.eval()
for p in curve_teacher.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(curve_student.parameters(), lr=cfg["learning_rate"])

curve_num_epochs = 3  # smoke-scale: 3 epochs stays within the 180s per-cell timeout (spec-approved range: 3-5)
for epoch in range(curve_num_epochs):
    total_loss_sum = 0.0
    kd_loss_sum = 0.0
    sup_loss_sum = 0.0
    n_batches = 0

    N = len(student_train)
    indices = list(range(N))
    np.random.shuffle(indices)

    for start in range(0, N, cfg["batch_size"]):
      
... (truncated; full length 3798)


---

