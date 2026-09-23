# %% [markdown]
# # DEEP BATCH ACTIVE LEARNING BY DIVERSE, UNCERTAIN GRADIENT LOWER BOUNDS
#
# **Jordan T. Ash, Chicheng Zhang, Akshay Krishnamurthy, John Langford, and Alekh Agarwal (2020), International Conference on Learning Representations.**
#
# BADGE represents every unlabeled example with a predicted-label, final-layer gradient embedding, then uses seeded k-MEANS++ to acquire a batch that is both uncertain and diverse.
#
# ### What this notebook gives you
# - A single-seed, end-to-end BADGE acquisition tutorial on MNIST.
# - Direct demonstrations of gradient embeddings and k-MEANS++ seeding.
# - A reusable `select_batch` call for a trained multiclass classifier and an unlabeled tensor.
#
# ### Two ways to use this
# - Run the notebook as-is to see one BADGE learning curve.
# - Import `select_batch` and the package helpers in your own active-learning loop.
#
# ### What this notebook does **not** do
# - It is a tutorial, not a benchmark reproduction or a comparison of methods.
# - It uses one seed, while the paper repeats experiments five times.
# - It uses five acquisition rounds rather than the paper-scale 349-round condition.
# - It uses a 12,000-example smoke pool rather than each paper benchmark's full training pool.
# - It uses the bundled two-layer MLP rather than reproducing all paper architectures and benchmark conditions.
# - It does not reproduce the paper's 231 experiments, multiple datasets, architectures, or batch-size conditions.

# %% [markdown]
# ## 0. Install dependencies (first run only)
#
# Run this once in a fresh environment. `%pip` installs into this notebook's kernel.

# %%
%pip install -r requirements.txt

# %% [markdown]
# ## 1. Setup

# %%
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from method import (
    GradientEmbeddingMLP,
    build_model,
    compute_gradient_embeddings,
    kmeans_plus_plus_seeding,
    load_data,
    select_batch,
    train_from_scratch,
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# %% [markdown]
# ## 2. Parameters
#
# The table records provenance: `paper` values are paper-stated, `system_default` values make this laptop-sized tutorial practical, and `system_inferred` values fill a paper-unspecified implementation choice.

# %% [markdown] PLACEHOLDER: params_table

# %% PLACEHOLDER: params_dict

# %% [markdown]
# ### Optional: scale up to paper-faithful values
#
# The system-supplied round count, pool size, epoch cap, and MLP width are demo choices. The paper's image experiments use a 256-dimensional MLP embedding; full paper-scale reproduction also requires the paper datasets, repeated runs, and appropriate architecture conditions.

# %%
# cfg.update({
#     "num_rounds": 349,
#     "pool_size": 60000,
#     "max_epochs": 50,
#     "hidden_dim": 256,
# })
# True paper-faithfulness also requires matching the dataset and architecture condition.

# %% [markdown]
# ## 3. The setup pieces
#
# BADGE operates on ordinary supervised classification: labeled data, a multiclass model, and fresh training after each acquisition. These are paper protocol, **not** the paper's contribution; skim them and move to §4. The model must expose raw multiclass logits and `forward_with_embedding`, whose second output is the real penultimate activation.

# %% [markdown]
# ### 3.1 Data
#
# `load_data` reads `method/example_data/` and downloads a flattened MNIST subset on first use. The runtime loader returns float32 features shaped `(N_pool, D)` and `(N_test, D)`, and int64 labels shaped `(N_pool,)` and `(N_test,)`; the architecture contract's smoke descriptors use `D=784` and `K=10`.

# %%
x_pool, y_pool, x_test, y_test = load_data(
    pool_size=cfg["pool_size"], n_test=1000, seed=SEED
)
input_dim = int(x_pool.shape[1])
n_classes = max(int(y_pool.max().item()), int(y_test.max().item())) + 1
print(f"pool: {tuple(x_pool.shape)} / {y_pool.dtype}")
print(f"test: {tuple(x_test.shape)} / {y_test.dtype}; classes: {n_classes}")

# %% [markdown]
# ### 3.2 Model
#
# The package supplies `GradientEmbeddingMLP`, a two-affine-layer MLP with hidden width `cfg["hidden_dim"]`. The paper evaluates MLP, ResNet-18, and VGG-11 conditions; this bundled MLP preserves the essential penultimate-layer hook and multiclass output, but does not reproduce those architecture results. Its contract forward input is float32 `(B, D)` and output is float32 `(B, K)`.

# %%
model = build_model(
    input_dim=input_dim, n_classes=n_classes, hidden_dim=cfg["hidden_dim"]
)
print(model)

# %% [markdown]
# ### 3.3 Training
#
# Each checkpoint uses Adam and cross-entropy, then retrains a **fresh** model after labels are acquired (**Algorithm 1, Section 4**). Training is performed in §3.4 and §5.1, so the checkpoint model always corresponds to the displayed labeled count.

# %% [markdown]
# ### 3.4 Bootstrap labeled set
#
# BADGE starts from uniformly sampled labels, trains a warmup classifier, and uses that fixed snapshot for the component demonstrations below (**Algorithm 1, Section 3**). This is random initialization; BADGE has no separate method-specific core-set bootstrap.

# %%
bootstrap_rng = np.random.default_rng(SEED)
bootstrap_labeled_idx = np.sort(
    bootstrap_rng.choice(len(x_pool), size=cfg["initial_labeled"], replace=False)
)
warmup_model = build_model(
    input_dim=input_dim, n_classes=n_classes, hidden_dim=cfg["hidden_dim"]
)
warmup_model = train_from_scratch(
    warmup_model,
    x_pool[bootstrap_labeled_idx],
    y_pool[bootstrap_labeled_idx],
    learning_rate=cfg["learning_rate"],
    max_epochs=cfg["max_epochs"],
    train_until_accuracy=cfg["train_until_accuracy"],
    seed=SEED,
)
warmup_model.eval()
with torch.no_grad():
    warmup_train_accuracy = (
        warmup_model(x_pool[bootstrap_labeled_idx]).argmax(dim=1)
        == y_pool[bootstrap_labeled_idx]
    ).float().mean().item()
    warmup_train_loss = F.cross_entropy(
        warmup_model(x_pool[bootstrap_labeled_idx]), y_pool[bootstrap_labeled_idx]
    ).item()
print(
    f"warmup training loss: {warmup_train_loss:.3f}; "
    f"accuracy: {warmup_train_accuracy:.3f}"
)

# %% [markdown]
# ## 4. The BADGE: Batch Active Learning by Diverse Gradient Embeddings method ⭐
#
# This is the paper's contribution: predicted-label gradient embeddings followed by seeded k-MEANS++ sampling (**Algorithm 1, Section 3**). The demo preserves both core elements; it does not substitute scalar uncertainty scores or a different batch sampler.

# %% [markdown]
# ### 4.1 Intuition
#
# Selecting only uncertain points can make a batch redundant, while selecting only diverse points can ignore likely model updates. BADGE uses the predicted label to form a last-layer gradient: its magnitude reflects uncertainty and its direction includes the penultimate representation. k-MEANS++ then favors separated high-impact embeddings without a hand-tuned trade-off (**Section 3**, **Algorithm 1**).

# %% [markdown]
# ### 4.2 Hallucinated gradient embeddings ⭐
#
# `compute_gradient_embeddings` uses the model prediction as the hypothetical label and returns one flattened final-layer gradient per candidate (**Algorithm 1, Section 3**). For class `i`, it computes $(p_i - \mathbb{1}\{\hat y=i\})z(x;V)$, so the implementation uses the actual penultimate activation rather than logits or input features (**Eq. 1, Section 3**).

# %%
warmup_model.eval()
demo_candidates = x_pool[:cfg["batch_size"]]
gradient_embeddings = compute_gradient_embeddings(warmup_model, demo_candidates)
gradient_norms = gradient_embeddings.norm(dim=1)
print("gradient embedding shape:", tuple(gradient_embeddings.shape))
print("mean gradient norm:", float(gradient_norms.mean().item()))
plt.figure(figsize=(7, 3))
plt.hist(gradient_norms.detach().numpy(), bins=20)
plt.xlabel("gradient embedding norm")
plt.ylabel("candidate count")
plt.title("BADGE predicted-label gradient magnitudes")
plt.show()

# %% [markdown]
# ### 4.3 k-MEANS++ gradient-batch seeding
#
# `kmeans_plus_plus_seeding` first chooses one gradient embedding uniformly, then samples each later center in proportion to its squared distance from the nearest selected center (**Appendix A, Algorithm 2**). This incremental nearest-distance update makes selections diverse in BADGE's gradient-embedding space (**Section 3**, **Appendix A, Algorithm 2**).

# %%
demo_embedding_array = gradient_embeddings.detach().numpy()
demo_rng = np.random.default_rng(SEED)
demo_selected_positions = kmeans_plus_plus_seeding(
    demo_embedding_array, cfg["batch_size"], demo_rng
)
print(f"selected {len(demo_selected_positions)} unique positions")
print("first ten positions:", demo_selected_positions[:10])

# %% [markdown]
# ### 4.4 Putting it together
#
# `select_batch(model, x_unlabeled, batch_size, seed)` composes the two helpers: it computes predicted-label gradient embeddings for the exact candidate tensor, then performs seeded k-MEANS++ and returns positions into that same tensor (**Algorithm 1**, **Section 3**).
#
# ```python
# positions = select_batch(model, x_unlabeled, batch_size, seed)
# ```

# %% [markdown]
# ## 5. Running active learning end-to-end
#
# The tutorial continues from the bootstrap state. It evaluates the initial model once, then performs one BADGE acquisition, label merge, fresh retraining, and evaluation for each acquisition round (**Algorithm 1, Section 3**). Each printed round metric is test accuracy, the fraction of correct predicted test labels.

# %% [markdown]
# ### 5.1 The acquisition loop
#
# The loop maps pool-relative positions returned by `select_batch` back through `unlabeled_idx`. It rebuilds the model after every merge, so no checkpoint warm-starts a later round (**Algorithm 1, Section 4**).

# %%
def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        return (model(x).argmax(dim=1) == y).float().mean().item()


def training_loss(model, x, y):
    model.eval()
    with torch.no_grad():
        return F.cross_entropy(model(x), y).item()


labeled_idx = bootstrap_labeled_idx.copy()
unlabeled_idx = np.setdiff1d(np.arange(len(x_pool)), labeled_idx)
learning_curve = []

initial_model = build_model(
    input_dim=input_dim, n_classes=n_classes, hidden_dim=cfg["hidden_dim"]
)
initial_model = train_from_scratch(
    initial_model,
    x_pool[labeled_idx],
    y_pool[labeled_idx],
    learning_rate=cfg["learning_rate"],
    max_epochs=cfg["max_epochs"],
    train_until_accuracy=cfg["train_until_accuracy"],
    seed=SEED,
)
initial_accuracy = evaluate(initial_model, x_test, y_test)
initial_loss = training_loss(initial_model, x_pool[labeled_idx], y_pool[labeled_idx])
learning_curve.append((len(labeled_idx), initial_accuracy))
print(
    f"round 0: labels={len(labeled_idx)}, training loss={initial_loss:.3f}, "
    f"test accuracy={initial_accuracy:.3f}"
)

current_model = initial_model
for round_index in range(cfg["num_rounds"]):
    current_model.eval()
    positions = select_batch(
        model=current_model,
        x_unlabeled=x_pool[unlabeled_idx],
        batch_size=cfg["batch_size"],
        seed=SEED + round_index,
    )
    chosen = unlabeled_idx[np.asarray(positions, dtype=int)]
    labeled_idx = np.union1d(labeled_idx, chosen)
    unlabeled_idx = np.setdiff1d(unlabeled_idx, chosen)

    current_model = build_model(
        input_dim=input_dim, n_classes=n_classes, hidden_dim=cfg["hidden_dim"]
    )
    current_model = train_from_scratch(
        current_model,
        x_pool[labeled_idx],
        y_pool[labeled_idx],
        learning_rate=cfg["learning_rate"],
        max_epochs=cfg["max_epochs"],
        train_until_accuracy=cfg["train_until_accuracy"],
        seed=SEED + round_index + 1,
    )
    round_accuracy = evaluate(current_model, x_test, y_test)
    round_loss = training_loss(current_model, x_pool[labeled_idx], y_pool[labeled_idx])
    learning_curve.append((len(labeled_idx), round_accuracy))
    print(
        f"round {round_index + 1}: labels={len(labeled_idx)}, "
        f"selected={len(chosen)}, training loss={round_loss:.3f}, "
        f"test accuracy={round_accuracy:.3f}"
    )

# %% [markdown]
# ### 5.2 Learning curve
#
# This is one BADGE run on one smoke-sized MNIST condition, not an estimate of the paper's multi-run, multi-condition results (**Section 4**).

# %%
labels_acquired, test_accuracies = zip(*learning_curve)
plt.figure(figsize=(7, 4))
plt.plot(labels_acquired, test_accuracies, marker="o", label="BADGE")
plt.xlabel("total labeled examples")
plt.ylabel("test accuracy")
plt.title("BADGE on smoke-sized MNIST (single seed)")
plt.ylim(0.0, 1.0)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# %% [markdown]
# ## 6. Use your own data
#
# **Option A:** pass a `.pt` file or directory to `load_data(path=...)`. A `.pt` file stores `x_pool`, `y_pool`, `x_test`, and `y_test`; features are float tensors and labels are long tensors.
#
# **Option B:** place data in `method/example_data/`. The loader accepts the same four-key `.json` structure, or paired `pool.csv` (or `train.csv`) and `test.csv` files with float features followed by an integer label in the final column.
#
# Inputs must be flattened 2D `(N, input_dim)` tensors for the bundled MLP. For image-shaped inputs, replace the architecture with a compatible CNN that returns raw multiclass logits and the real penultimate activation through `forward_with_embedding`.
