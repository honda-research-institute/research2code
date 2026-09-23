# %% [markdown]
# # Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds
#
# Ash et al. (ICLR 2020).
#
# **What this notebook gives you**
# - A small, runnable BADGE package walkthrough.
# - Direct demonstrations of gradient embeddings and k-MEANS++ seeding.
# - One single-method active-learning loop with a test-accuracy curve.
#
# **Two ways to use this**
# - Run the cells as-is to inspect one smoke-sized BADGE experiment.
# - Import the package and replace the loader or classifier for your own pool.
#
# **What this notebook does NOT do**
# - It is not a reproduction of the paper's benchmark or its empirical claims.
# - It runs 5 acquisition rounds instead of the paper's 349-round SVHN protocol.
# - It uses a 12,000-example pool instead of the paper's full training pool.
# - It uses one seed instead of the paper's five independent repetitions.
# - It uses the bundled flattened MLP instead of the paper's benchmark-specific CNN/ResNet/VGG architectures.
# - It uses the package's SGD implementation; the paper describes the Adam variant of SGD.
# - Its inferred learning rate, epoch cap, and MLP hidden width are system choices, not paper-stated values.
# - It does not implement baselines, comparisons, or multi-method benchmark tables.

# %% [markdown]
# ## 0. Install dependencies (first run only)
#
# Run this cell once in a new environment. `%pip` installs into the active notebook kernel.

# %%
%pip install -r requirements.txt

# %% [markdown]
# ## 1. Setup

# %%
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import torch

from method import (
    compute_gradient_embeddings,
    kmeans_plus_plus_seeding,
    select_batch,
    build_model,
    train_from_scratch,
    load_data,
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# %% [markdown]
# ## 2. Parameters
#
# The rendered table separates values stated by the paper from smoke-scale system defaults and inferred runtime choices. The paper's batch size and initial labeled count are retained; the loop and pool are intentionally smaller.

# %% [markdown] PLACEHOLDER: params_table

# %% PLACEHOLDER: params_dict

# %% [markdown]
# ### Optional: scale up toward the paper protocol
#
# The following is intentionally commented out. It shows the main runtime changes needed for a larger run; paper-faithful results also require the paper's benchmark data, architecture, optimizer protocol, and repeated seeds.

# %%
# cfg.update({
#     "num_rounds": 349,
#     "pool_size": 60000,  # example full-scale image-pool order of magnitude
#     "learning_rate": 0.001,  # paper's image-data value; verify for your benchmark
#     "max_epochs": 50,
#     "hidden_dim": 256,
# })

# %% [markdown]
# ## 3. The setup pieces
#
# BADGE sits on standard pool-based supervised classification: labeled data, a differentiable classifier, and retraining between queries. These pieces are protocol rather than the paper's contribution; the contribution is the uncertainty-aware final-layer gradient embedding and batch k-MEANS++ selection in §4. The classifier must expose `forward_with_embedding`, returning logits and the real penultimate features used by the final linear layer.

# %% [markdown]
# ### 3.1 Data
#
# `load_data` returns float32 feature tensors and int64 class-id tensors. The v2 runtime contract declares pool examples as `(N_pool, D)` and test examples as `(N_test, D)`; this loader supplies flattened MNIST features, with `D=784` for the bundled smoke data.

# %%
x_pool, y_pool, x_test, y_test = load_data(
    pool_size=cfg["pool_size"], n_test=1000, seed=SEED
)
n_classes = int(y_pool.max().item()) + 1
input_dim = int(x_pool.shape[1])
print("pool:", tuple(x_pool.shape), tuple(y_pool.shape))
print("test:", tuple(x_test.shape), tuple(y_test.shape))
print("input_dim:", input_dim, "classes:", n_classes)

# %% [markdown]
# ### 3.2 Model
#
# The bundled `GradientEmbeddingClassifier` is a compact MLP with a final linear classifier and an explicit penultimate-feature hook. The paper evaluates several architectures; this flattened MLP is a runnable demo choice. Replacing it is safe only when the replacement preserves the declared `(batch, D) -> (batch, C)` logits path and `forward_with_embedding` contract.

# %%
model = build_model(
    input_dim=input_dim,
    n_classes=n_classes,
    hidden_dim=cfg["hidden_dim"],
)
print(model)

# %% [markdown]
# ### 3.3 Training
#
# Algorithm 1 retrains from scratch after each acquisition. The paper describes cross-entropy training until training accuracy exceeds 99% (**Algorithm 1, Section 4**). This package's generated training function uses the supplied SGD protocol and the configured smoke epoch cap; the paper's Adam wording is listed as a departure in the title cell.

# %% [markdown]
# ### 3.4 Bootstrap labeled set
#
# BADGE begins with a uniformly random labeled set (**Algorithm 1**). We retain the paper's `initial_labeled` count, train a warmup model from scratch, and reuse that fixed model snapshot only for the §4 helper demonstrations. The end-to-end loop rebuilds a fresh model for every round.

# %%
rng = np.random.default_rng(SEED)
bootstrap_labeled_idx = rng.choice(
    len(x_pool), size=cfg["initial_labeled"], replace=False
).astype(np.int64)
bootstrap_model = build_model(
    input_dim=input_dim,
    n_classes=n_classes,
    hidden_dim=cfg["hidden_dim"],
)
bootstrap_model = train_from_scratch(
    bootstrap_model,
    x_pool[bootstrap_labeled_idx],
    y_pool[bootstrap_labeled_idx],
    learning_rate=cfg["learning_rate"],
    max_epochs=cfg["max_epochs"],
    train_until_accuracy=cfg["train_until_accuracy"],
    seed=SEED,
)
bootstrap_model.eval()
with torch.no_grad():
    bootstrap_loss = torch.nn.functional.cross_entropy(
        bootstrap_model(x_pool[bootstrap_labeled_idx]),
        y_pool[bootstrap_labeled_idx],
    ).item()
print(f"bootstrap training loss: {bootstrap_loss:.4f}")

# %% [markdown]
# ## 4. The BADGE method ⭐
#
# BADGE is the paper's contribution (**Algorithm 1, Section 3**). It computes one hallucinated final-layer gradient embedding per unlabeled example, then samples a diverse batch in that gradient space.

# %% [markdown]
# ### 4.1 Intuition
#
# A label is unavailable when the pool is scored, so BADGE uses the model's predicted class as a hallucinated label. The resulting last-layer gradient is small for confident examples and larger when a possible label would induce a larger update (**Section 3, Proposition 1**). Its direction also retains penultimate representation information. k-MEANS++ then favors points far from already selected centers, combining uncertainty and diversity without replacing the gradient embedding with entropy, margins, raw probabilities, or input-space features (**Section 3, Equation 1**).

# %% [markdown]
# ### 4.2 Hallucinated gradient embeddings ⭐
#
# `compute_gradient_embeddings` implements the per-example representation from **Eq. 1, Section 3** (`eq-gradient-embedding`, `eq-gradient-block`). For class block `i`, it computes `(p_i - I(yhat=i)) z(x; V)`, where `p` is the softmax output and `z` is the penultimate feature. The code below inspects embedding magnitudes; it does not substitute those magnitudes for the embeddings used by acquisition.

# %%
bootstrap_embeddings = compute_gradient_embeddings(
    bootstrap_model, x_pool
)
embedding_norms = np.linalg.norm(bootstrap_embeddings, axis=1)
print("embedding matrix:", bootstrap_embeddings.shape)
print("mean embedding norm:", float(embedding_norms.mean()))
plt.figure(figsize=(6, 3))
plt.hist(embedding_norms, bins=30)
plt.xlabel("final-layer gradient-embedding norm")
plt.ylabel("pool examples")
plt.title("BADGE gradient-space magnitudes")
plt.show()

# %% [markdown]
# ### 4.3 Diverse batch seeding
#
# `kmeans_plus_plus_seeding` chooses the first center uniformly and subsequent centers with probability proportional to squared distance from the nearest selected center (**Appendix A, Algorithm 2**, `alg-kmeans-plus-plus`). This is the diversity step applied to the actual gradient embeddings.

# %%
demo_positions = kmeans_plus_plus_seeding(
    bootstrap_embeddings, cfg["batch_size"], seed=SEED
)
demo_norms = embedding_norms[np.asarray(demo_positions, dtype=np.int64)]
print("selected unique positions:", len(set(demo_positions)))
print("selected batch size:", len(demo_positions))
print("selected mean embedding norm:", float(demo_norms.mean()))

# %% [markdown]
# ### 4.4 Putting it together
#
# The public `select_batch` composition first calls the gradient-embedding helper and then calls k-MEANS++ with the requested batch size (**Algorithm 1, Section 3**). The end-to-end cell below keeps the original unlabeled tensor and maps returned positions back to global pool indices.

# %% [markdown]
# ## 5. Running active learning end-to-end
#
# This is one BADGE run on one dataset and one seed. It follows the paper's retrain-from-scratch round structure, but uses the explicit smoke-scale defaults shown in the rendered provenance table. No baseline or comparison method is run.

# %% [markdown]
# ### 5.1 The acquisition loop

# %%
def evaluate_accuracy(eval_model, x, y):
    eval_model.eval()
    with torch.no_grad():
        predictions = eval_model(x).argmax(dim=1)
    return float((predictions == y).float().mean().item())


def train_and_evaluate(current_indices, round_seed):
    round_model = build_model(
        input_dim=input_dim,
        n_classes=n_classes,
        hidden_dim=cfg["hidden_dim"],
    )
    round_model = train_from_scratch(
        round_model,
        x_pool[current_indices],
        y_pool[current_indices],
        learning_rate=cfg["learning_rate"],
        max_epochs=cfg["max_epochs"],
        train_until_accuracy=cfg["train_until_accuracy"],
        seed=round_seed,
    )
    round_model.eval()
    with torch.no_grad():
        loss_value = torch.nn.functional.cross_entropy(
            round_model(x_pool[current_indices]), y_pool[current_indices]
        ).item()
    accuracy = evaluate_accuracy(round_model, x_test, y_test)
    print(
        f"round seed={round_seed}: labels={len(current_indices)}, "
        f"training_loss={loss_value:.4f}, test_accuracy={accuracy:.4f}"
    )
    return round_model, accuracy


# Rebuild all loop state from the stable bootstrap output on every execution.
labeled_idx = np.asarray(bootstrap_labeled_idx, dtype=np.int64).copy()
unlabeled_idx = np.setdiff1d(
    np.arange(len(x_pool), dtype=np.int64), labeled_idx
)
learning_curve = []
label_counts = []

model, accuracy = train_and_evaluate(labeled_idx, SEED)
label_counts.append(len(labeled_idx))
learning_curve.append(accuracy)

for round_index in range(cfg["num_rounds"]):
    model.eval()
    with torch.no_grad():
        selected_positions = select_batch(
            model,
            x_pool[unlabeled_idx],
            cfg["batch_size"],
            SEED + round_index,
        )
    selected_positions = np.asarray(selected_positions, dtype=np.int64)
    chosen_global = unlabeled_idx[selected_positions]
    labeled_idx = np.union1d(labeled_idx, chosen_global)
    unlabeled_idx = np.setdiff1d(unlabeled_idx, chosen_global)
    model, accuracy = train_and_evaluate(
        labeled_idx, SEED + round_index + 1
    )
    label_counts.append(len(labeled_idx))
    learning_curve.append(accuracy)

print("label counts:", label_counts)
print("evaluated points:", len(learning_curve))

# %% [markdown]
# ### 5.2 Learning curve
#
# Each point is evaluated after training on the labeled set shown on the x-axis: the initial bootstrap point, followed by one post-acquisition point for each configured round. This is a single BADGE curve, not a comparison overlay.

# %%
plt.figure(figsize=(7, 4))
plt.plot(label_counts, learning_curve, marker="o", label="BADGE")
plt.xlabel("number of labeled examples")
plt.ylabel("test accuracy")
plt.title("BADGE active learning on the smoke dataset")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# %% [markdown]
# ## 6. Use your own data
#
# `load_data` accepts a `.pt` file or directory containing `{"x_pool", "y_pool", "x_test", "y_test"}`, a `.json` file with the same keys as nested lists, or a directory containing `pool.csv` (or `train.csv`) and `test.csv` with the integer label in the last column. Pass the path explicitly with `load_data(path="...")`, or place a file in `method/example_data/` and call `load_data()`.
#
# The bundled classifier expects flattened 2-D feature tensors. For image-shaped inputs, swap in a differentiable architecture whose logits and `forward_with_embedding` penultimate features satisfy the model contract. Keep BADGE's final-layer gradient embeddings and k-MEANS++ selection unchanged.
