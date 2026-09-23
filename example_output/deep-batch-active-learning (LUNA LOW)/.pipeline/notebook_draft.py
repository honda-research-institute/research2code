# %% [markdown]
# # DEEP BATCH ACTIVE LEARNING BY DIVERSE, UNCERTAIN GRADIENT LOWER BOUNDS
#
# Jordan T. Ash, Chicheng Zhang, Akshay Krishnamurthy, John Langford, and Alekh Agarwal.
#
# This notebook gives a smoke-sized, single-method tutorial for BADGE.
#
# **What this notebook gives you:** a runnable package walkthrough, a small supervised setup, and one active-learning run.
#
# **Two ways to use this:** run the bundled example, or replace the loader inputs with your own data.
#
# **What this notebook does NOT do:** it is not a benchmark and does not implement baselines. It uses system-default `pool_size` and `num_rounds` rather than the paper's full training set and 349 rounds; it uses inferred `max_epochs` and `hidden_dim` for the smoke training budget/model width; and it runs one seed rather than repeated experiments. The bundled flattened MLP is a demo architecture, not the paper's full image-architecture benchmark.

# %% [markdown]
# ## 0. Install dependencies (first run only)
#
# Install the package requirements into the active notebook kernel.

# %%
%pip install -r requirements.txt

# %% [markdown]
# ## 1. Setup

# %%
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import torch
from method import (select_batch, compute_gradient_embeddings,
                    kmeans_plus_plus_seeding, BADGEClassifier,
                    build_model, train_from_scratch, load_data)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# %% [markdown]
# ## 2. Parameters
#
# The table distinguishes values taken from the paper from values inferred or selected for this demo.

# %% [markdown] PLACEHOLDER: params_table

# %% PLACEHOLDER: params_dict

# %% [markdown]
# ### Optional: scale up to paper-faithful values
#
# The paper-faithful run also requires the full dataset, round budget, and paper benchmark architectures.

# %%
# cfg.update({
#     "pool_size": "full training set",
#     "num_rounds": 349,
#     "max_epochs": "train until accuracy exceeds 0.99",
# })
# True paper-faithfulness for image data also requires swapping the bundled flattened MLP for the paper's MLP, ResNet-18, or VGG architecture.

# %% [markdown]
# ## 3. The setup pieces

# %% [markdown]
# ## 3.1 Data

# %%
x_pool, y_pool, x_test, y_test = load_data(pool_size=cfg["pool_size"], n_test=1000, seed=SEED)
print("pool", tuple(x_pool.shape), "test", tuple(x_test.shape), "classes", int(torch.unique(y_pool).numel()))

# %% [markdown]
# ## 3.2 Model

# %%
model = build_model(input_dim=x_pool.shape[1], n_classes=int(torch.unique(y_pool).numel()), hidden_dim=cfg["hidden_dim"])
print(model)

# %% [markdown]
# ## 3.3 Training

# %%
# BADGE follows the paper's cross-entropy + Adam, retrain-from-scratch protocol (**Section 4**).
print("Training is performed after bootstrap and after every acquisition round.")

# %% [markdown]
# ## 3.4 Bootstrap

# %%
rng = np.random.default_rng(SEED)
bootstrap_labeled_idx = rng.choice(len(x_pool), size=min(100, len(x_pool)), replace=False)
labeled_idx = set(int(i) for i in bootstrap_labeled_idx)
model = build_model(input_dim=x_pool.shape[1], n_classes=int(torch.unique(y_pool).numel()), hidden_dim=cfg["hidden_dim"])
model = train_from_scratch(model, x_pool[bootstrap_labeled_idx], y_pool[bootstrap_labeled_idx],
                           learning_rate=cfg["learning_rate"], max_epochs=cfg["max_epochs"],
                           train_until_accuracy=cfg["train_until_accuracy"], seed=SEED)
model.eval()
with torch.no_grad():
    bootstrap_accuracy = (model(x_test).argmax(1) == y_test).float().mean().item()
print(f"bootstrap accuracy: {bootstrap_accuracy:.3f}")

# %% [markdown]
# ## 4. The method ⭐

# %% [markdown]
# ### 4.1 Intuition
#
# BADGE uses the model's predicted label to form a usable gradient before querying the true label. The resulting last-layer gradient embedding is large for uncertain predictions and retains the penultimate representation, so geometry can express both influence and diversity (**Section 3, Equation (1)**; `concept-hallucinated-label`, `eq-gradient-embedding`). Sequential k-MEANS++ then favors candidates far from already selected embeddings (**Appendix A, Algorithm 2**). These are the two core methodology elements; this notebook does not replace either with confidence or top-K scoring.

# %% [markdown]
# ### 4.2 Hallucinated gradient embeddings ⭐
#
# For a predicted class, BADGE forms each output-class block as `(p_i - I[hat y=i]) z(x; V)` (**Equation (1), Section 3**; `eq-gradient-embedding`).

# %%
model.eval()
with torch.no_grad():
    demo_embeddings = compute_gradient_embeddings(model, x_pool[:100])
print("gradient embedding shape:", tuple(demo_embeddings.shape))
print("embedding norms:", demo_embeddings.detach().norm(dim=1)[:5].numpy())

# %% [markdown]
# ### 4.3 Sequential k-MEANS++ selection ⭐
#
# The first center is uniform; each later center is sampled in proportion to squared distance from its nearest selected center (**Appendix A, Algorithm 2**; `alg-kmeans-plus-plus`).

# %%
demo_centers = kmeans_plus_plus_seeding(demo_embeddings.detach().cpu().numpy(), 10,
                                        rng=np.random.default_rng(SEED))
print("selected center positions:", demo_centers)

# %% [markdown]
# ### 4.4 Putting it together
#
# `select_batch` composes the predicted-label gradient construction and sequential k-MEANS++ sampler into the paper's acquisition rule (**Algorithm 1, Section 3**; `alg-badge`).

# %% [markdown]
# ## 5. Running active learning end-to-end

# %% [markdown]
# ### 5.1 Acquisition loop

# %%
labels_history = []
accuracy_history = []
bootstrap_source = np.asarray(bootstrap_labeled_idx, dtype=np.int64)
labeled_idx = set(int(i) for i in bootstrap_source)
model = build_model(input_dim=x_pool.shape[1], n_classes=int(torch.unique(y_pool).numel()), hidden_dim=cfg["hidden_dim"])
model = train_from_scratch(model, x_pool[bootstrap_source], y_pool[bootstrap_source],
                           learning_rate=cfg["learning_rate"], max_epochs=cfg["max_epochs"],
                           train_until_accuracy=cfg["train_until_accuracy"], seed=SEED)
model.eval()
with torch.no_grad():
    bootstrap_accuracy = (model(x_test).argmax(1) == y_test).float().mean().item()
labels_history.append(len(labeled_idx)); accuracy_history.append(bootstrap_accuracy)
print(f"bootstrap: labels={len(labeled_idx)} test_accuracy={bootstrap_accuracy:.3f}")
for round_id in range(int(cfg["num_rounds"])):
    unlabeled_idx = np.asarray(sorted(set(range(len(x_pool))) - labeled_idx), dtype=np.int64)
    if len(unlabeled_idx) == 0:
        break
    chosen_positions = select_batch(model, x_pool[unlabeled_idx], int(cfg["batch_size"]), seed=SEED + round_id)
    chosen = unlabeled_idx[chosen_positions.detach().cpu().numpy()]
    labeled_idx.update(int(i) for i in chosen)
    model = build_model(input_dim=x_pool.shape[1], n_classes=int(torch.unique(y_pool).numel()), hidden_dim=cfg["hidden_dim"])
    model = train_from_scratch(model, x_pool[sorted(labeled_idx)], y_pool[sorted(labeled_idx)],
                               learning_rate=cfg["learning_rate"], max_epochs=cfg["max_epochs"],
                               train_until_accuracy=cfg["train_until_accuracy"], seed=SEED + round_id + 1)
    model.eval()
    with torch.no_grad():
        post_accuracy = (model(x_test).argmax(1) == y_test).float().mean().item()
    labels_history.append(len(labeled_idx)); accuracy_history.append(post_accuracy)
    print(f"round {round_id + 1}: labels={len(labeled_idx)} test_accuracy={post_accuracy:.3f}")

# %% [markdown]
# ### 5.2 Learning curve

# %%
plt.figure(figsize=(6, 4))
plt.plot(labels_history, accuracy_history, marker="o")
plt.xlabel("Labels acquired")
plt.ylabel("Test accuracy")
plt.title("BADGE learning curve")
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## 6. Use your own data
#
# Replace the bundled loader inputs with your own supported `.pt`, `.json`, or `.csv` data. Keep the feature and label shapes expected by the package model.
