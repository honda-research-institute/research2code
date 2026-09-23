# %% [markdown]
# # Bayesian Active Learning by Disagreements: A Geometric Perspective
#
# **Xiaofeng Cao and Ivor W. Tsang** (2022)
#
# ## What this notebook gives you
#
# - A runnable reference implementation of GBALD (Geometric Bayesian Active Learning by Disagreements)
# - A step-by-step walkthrough of the two-stage algorithm: ellipsoid core-set construction (Stage 1) and geometric BALD ranking (Stage 2)
# - A smoke-scale demonstration on MNIST with ~5-minute runtime
#
# ## Two ways to use this
#
# 1. **Run as-is**: Execute all cells to see GBALD in action on MNIST.
# 2. **Import from your own code**: Use `from method import select_batch, ...` to integrate GBALD into your pipeline.
#
# ## What this notebook does NOT do
#
# - This is a **single-method tutorial**, not a benchmark. It runs only GBALD, not baselines (BALD, BatchBALD, Entropy, etc.).
# - The demo uses **smoke-scale parameters** to fit a ~5-minute runtime budget. Key departures from the paper:
#   - `mc_samples=20` vs. paper's 2000 (field-guide minimum for meaningful BALD posterior estimate)
#   - `pool_size=800` vs. paper's full MNIST training set (60,000)
#   - `core_set_size=100` vs. paper's 5000
#   - `batch_returns=300` (signature default, matches paper's b=300)
#   - `R_0=2000.0` calibrated for raw [0,255] pixel scale; MNIST is normalized to [0,1] here, so the geometric prior is approximate
#   - MLP architecture vs. paper's CNN-like structure (3 blocks of conv/dropout/maxpool/relu)
# - Single-seed results are for **mechanical verification**, not method ranking claims.
#
# For paper-faithful results, scale up parameters per the "Optional: scale up to paper-faithful values" subsection in §2.

# %% [markdown]
# ## 0. Install dependencies (first run only)
#
# Run this cell once to install required packages. Subsequent runs are no-ops if packages are already installed.

# %%
%pip install -r requirements.txt

# %% [markdown]
# ## 1. Setup

# %%
import matplotlib
%matplotlib inline

import numpy as np
import torch
import matplotlib.pyplot as plt

# Import the method package public API
from method import (
    select_batch,
    compute_bald_scores,
    compute_geometric_prior,
    construct_ellipsoid_core_set,
    geometric_ranking,
    MCDropoutMLP,
    build_model,
    train_from_scratch,
    load_data,
)

# Master seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# %% [markdown]
# ## 2. Parameters
#
# Parameters are attributed to their source:
# - `paper`: Value explicitly stated in the paper.
# - `system_default`: Runtime-smoke default that deviates from the paper value (with reasoning).
# - `system_inferred`: Not stated in the paper; inferred from conventions or related work.
# - `spec_default`: Default from the pluggable component signature.
#
# The table below shows the full provenance. The code cell after it defines `params` and `cfg`.

# %% [markdown] PLACEHOLDER: params_table

# %% PLACEHOLDER: params_dict

# %% [markdown]
# ### Optional: scale up to paper-faithful values
#
# Uncomment the block below to use paper-faithful parameter values. Note: this will significantly increase runtime (potentially hours) and may require more memory.
#
# ```python
# # cfg.update({
# #     "mc_samples": 2000,      # paper value (Section 7.1, 7.9)
# #     "pool_size": 60000,      # full MNIST training set
# #     "core_set_size": 5000,   # paper value (Section 7.9, Table 6)
# # })
# # # Note: true paper-faithfulness for image data also requires swapping the bundled
# # # MLP for the paper's CNN-like architecture (3 blocks of conv/dropout/maxpool/relu).
# # # See method/model.py for swap instructions.
# ```

# %% [markdown]
# ## 3. The setup pieces
#
# GBALD works on top of standard supervised classification — model + training + labeled data. The pieces below are **paper protocol but not the paper's contribution**; skim and move on to §4 for the algorithm itself.
#
# **Key constraint for Bayesian AL**: the model must have dropout layers active at inference time. Without active dropout, MC samples are identical, BALD scores collapse to zero, and the algorithm degenerates to random sampling — the canonical silent-failure mode for Bayesian AL.

# %% [markdown]
# ### 3.1 Data

# %%
# Load MNIST data
x_pool, y_pool, x_test, y_test = load_data(
    pool_size=cfg["pool_size"],
    n_test=1000,
    seed=SEED,
)

print(f"x_pool shape: {x_pool.shape}  (N_pool={len(x_pool)}, n_features={x_pool.shape[1]})")
print(f"y_pool shape: {y_pool.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"Unique labels in pool: {torch.unique(y_pool).tolist()}")
print(f"Number of classes: {len(torch.unique(y_pool))}")

# %% [markdown]
# ### 3.2 Model

# %%
# Build the MCDropoutMLP architecture
n_classes = len(torch.unique(y_pool))
input_dim = x_pool.shape[1]

model = build_model(
    input_dim=input_dim,
    n_classes=n_classes,
    hidden_dim=cfg.get("hidden_dim", 256),
    dropout_rate=cfg.get("dropout_rate", 0.5),
)

print(model)

# %% [markdown]
# ### 3.3 Training
#
# Protocol: Adam optimizer, retrain from scratch each round, train until training accuracy ≥ 99% or max_epochs exhausted. The paper follows BatchBALD [5] which uses Adam with dataset-dependent learning rates. **No code here** — training happens per-round in §3.4 (bootstrap) and §5.1 (the acquisition loop).

# %% [markdown]
# ### 3.4 Bootstrap labeled set
#
# GBALD's Stage 1 constructs a core-set via ellipsoid geodesic search to initialize the labeled set. This is **not** random sampling — the core-set provides representative samples that cover the input distribution, relieving the uninformative prior problem.
#
# We call `construct_ellipsoid_core_set` with `core_set_size=cfg["core_set_size"]` to get the initial labeled indices, then train a warmup model on them. This warmup model is used for §4's per-component demonstrations.

# %%
# Stage 1: Construct the initial core-set
core_set_indices = construct_ellipsoid_core_set(
    x_pool=x_pool,
    core_set_size=cfg["core_set_size"],
    R_0=cfg["R_0"],
    eta=cfg["eta"],
    seed=SEED,
    kmeans_centers=10,
)

print(f"Core-set size: {len(core_set_indices)}")

# Extract labeled and unlabeled sets
x_labeled = x_pool[core_set_indices]
y_labeled = y_pool[core_set_indices]

# Remaining unlabeled pool
remaining_indices = [i for i in range(len(x_pool)) if i not in core_set_indices]
x_unlabeled = x_pool[remaining_indices]
y_unlabeled = y_pool[remaining_indices]

print(f"Labeled set: {x_labeled.shape}, Unlabeled pool: {x_unlabeled.shape}")

# Train warmup model on the core-set
model = build_model(
    input_dim=input_dim,
    n_classes=n_classes,
    hidden_dim=cfg.get("hidden_dim", 256),
    dropout_rate=cfg.get("dropout_rate", 0.5),
)

model = train_from_scratch(
    model,
    x_labeled,
    y_labeled,
    learning_rate=cfg["learning_rate"],
    max_epochs=cfg["max_epochs"],
    train_until_accuracy=cfg["train_until_accuracy"],
    seed=SEED,
)

print("Warmup model trained on core-set.")

# %% [markdown]
# ## 4. The GBALD method ⭐
#
# This is the paper's contribution (**Algorithm 1, Section 5**). GBALD is a two-stage framework:
#
# - **Stage 1 (core-set construction)**: Constructs an initial core-set on an ellipsoid geometry using a distance-based geometric prior (**Eq. 5, Section 4.2**) and ellipsoid geodesic rescaling (**Eq. 11, Section 4.2**). This provides representative samples that cover the input distribution.
# - **Stage 2 (model uncertainty estimation)**: Uses MC dropout to score unlabeled candidates by BALD mutual information (**Eq. 12, Section 4.3**), then ranks those candidates by their geometric representativeness (**Eq. 13-14, Section 4.3**) to avoid redundant selections.
#
# The two stages together relieve both the uninformative prior problem and redundant information issue that plague standard BALD.

# %% [markdown]
# ### 4.1 Intuition
#
# Standard BALD selects data that maximizes the decrease in expected posterior entropy (**Eq. 1, Section 3.1**). However, when the initial labeled set has insufficient or unbalanced labels, the prior becomes uninformative, causing BALD to produce biased, redundant acquisitions concentrated in few classes (**Section 3.2, Section 7.2**).
#
# GBALD's insight is to use **core-set construction on an ellipsoid** rather than a sphere. The ellipsoid geodesic rescaling (affine factor `eta ∈ (0,1)`) prevents updates from converging to boundary regions where data is less representative (**concept-ellipsoid-core-set, Section 4.1**).
#
# After Stage 1 provides a representative core-set, Stage 2 ranks BALD-scored candidates by their geometric representativeness (distance to the labeled set), which further reduces redundancy. The result: faster convergence with fewer labels.

# %% [markdown]
# ### 4.2 Geometric prior probability (Eq. 5) ⭐
#
# The geometric prior `p(y|x,theta)` provides an informative prior independent of a trained DNN:
#
# $$p(y_i|x_i,\theta) = \begin{cases} 1, & \exists j, \|x_i - D_j\| \le R_0 \\ \max\{R_0 / \|x_i - D_j\|\}, & \text{otherwise} \end{cases}$$
#
# where `D_j` are the labeled samples and `R_0` is a distance threshold. Samples within `R_0` of any labeled point get probability 1; others get `R_0 / distance` (inverse distance, so closer = higher probability).
#
# **paper-element: eq-geometric-prior**

# %%
# Demonstrate geometric prior on a subset of unlabeled data
n_demo = 100
x_demo = x_unlabeled[:n_demo]

prior_probs = compute_geometric_prior(
    x=x_demo,
    x_labeled=x_labeled,
    R_0=cfg["R_0"],
)

print(f"Geometric prior probabilities (first 10): {prior_probs[:10].tolist()}")
print(f"Min: {prior_probs.min().item():.4f}, Max: {prior_probs.max().item():.4f}, Mean: {prior_probs.mean().item():.4f}")

# Visualize the distribution
plt.hist(prior_probs.tolist(), bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Geometric prior probability p(y|x,theta)')
plt.ylabel('Count')
plt.title(f'Distribution of geometric prior probabilities (n={n_demo})')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.3 BALD mutual information scoring (Eq. 1, Eq. 12) ⭐
#
# BALD selects `x*` that maximizes the decrease in expected posterior entropy:
#
# $$x^* = \arg\max_{x \in D_u} H[\theta|D_0] - \mathbb{E}_{y \sim p(y|x,D_0)} [H[\theta|x,y,D_0]]$$
#
# This is the mutual information `I[y; θ | x, D_0]`. We estimate it via MC dropout:
# - `p_mean = (1/T) Σ_t p(y|x, θ_t)` — averaged predictive probability over T MC samples
# - `prior_entropy = H[p_mean]` — entropy of the averaged prediction
# - `expected_posterior_entropy = E_θ[H[p_θ]]` — average entropy of per-sample predictions
# - `BALD = prior_entropy - expected_posterior_entropy`
#
# **paper-element: eq-bald-score**
# **paper-element: eq-batch-returns**
# **paper-element: concept-mc-dropout-inference**

# %%
# Demonstrate BALD scoring on a subset of unlabeled data
n_demo = 50
x_demo = x_unlabeled[:n_demo]

bald_scores = compute_bald_scores(
    model=model,
    x_unlabeled=x_demo,
    mc_samples=cfg["mc_samples"],
)

print(f"BALD scores (first 10): {bald_scores[:10].tolist()}")
print(f"Min: {bald_scores.min().item():.4f}, Max: {bald_scores.max().item():.4f}, Mean: {bald_scores.mean().item():.4f}")

# Visualize the distribution
plt.hist(bald_scores.tolist(), bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('BALD score (mutual information)')
plt.ylabel('Count')
plt.title(f'Distribution of BALD scores (n={n_demo}, mc_samples={cfg["mc_samples"]})')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.4 Geometric ranking (Eq. 13-14) ⭐
#
# Among `b` BALD-ranked candidates, select the one with highest geometric representativeness:
#
# $$x_t^* = \arg\max_{x_i^* \in \{x_1^*, ..., x_b^*\}} \max_{D_j \in D_0} p(y_i | x_i^*, \theta) := \frac{R_0}{\|x_i^* - D_j\|}$$
#
# This extends to batch output by selecting the top `b'` candidates by representativeness. The geometric prior ensures we pick samples that are far from the current labeled set (high `R_0 / distance`), reducing redundancy.
#
# **paper-element: eq-ranking-criterion**
# **paper-element: eq-batch-output**

# %%
# Demonstrate geometric ranking
# First, get top batch_returns BALD candidates
n_candidates = min(cfg["batch_returns"], len(x_unlabeled))
bald_scores_all = compute_bald_scores(model, x_unlabeled, cfg["mc_samples"])
top_candidate_indices = torch.topk(bald_scores_all, n_candidates).indices
x_candidates = x_unlabeled[top_candidate_indices]

print(f"Number of BALD candidates: {len(x_candidates)}")

# Rank by geometric representativeness and select top batch_size
selected_local_indices = geometric_ranking(
    x_candidates=x_candidates,
    x_labeled=x_labeled,
    R_0=cfg["R_0"],
    batch_outputs=min(cfg["batch_size"], len(x_candidates)),
)

print(f"Selected {len(selected_local_indices)} samples by geometric ranking")
print(f"Local indices (in candidates): {selected_local_indices[:10]}")

# Map back to original unlabeled indices
selected_global_indices = [top_candidate_indices[i].item() for i in selected_local_indices]
print(f"Global indices (in x_unlabeled): {selected_global_indices[:10]}")

# %% [markdown]
# ### 4.5 Ellipsoid core-set construction (Eq. 10, Eq. 11) ⭐
#
# Stage 1 constructs the initial core-set using max-min optimization with ellipsoid geodesic rescaling:
#
# $$x^* = \arg\max_{x_j \in D_u} \min_{D_j \in D_0} \{ \|\mathcal{L}_0 - \mathcal{L}\|^2 + \log p(y_j | x_j, \theta) \}$$
#
# where `L_0` is the unbiased full likelihood over k-means centers, and `L` is the log-likelihood over the current labeled set. After selecting `x*`, apply ellipsoid geodesic rescaling:
#
# $$x_e^* = x_i + \eta(x^* - x_i)$$
#
# then snap to the nearest neighbor in the unlabeled pool. The affine factor `eta ∈ (0,1)` prevents updates from converging to boundary regions.
#
# **paper-element: eq-coreset-acquisition**
# **paper-element: eq-ellipsoid-geodesic**
# **paper-element: alg-gbald (Stage 1)**

# %%
# Demonstrate core-set construction on a fresh subset
# Use a small pool for demonstration
n_demo_pool = 200
x_demo_pool = x_pool[:n_demo_pool]

demo_core_set = construct_ellipsoid_core_set(
    x_pool=x_demo_pool,
    core_set_size=20,
    R_0=cfg["R_0"],
    eta=cfg["eta"],
    seed=SEED + 1,  # Different seed for demo
    kmeans_centers=5,
)

print(f"Demo core-set indices (in x_demo_pool): {demo_core_set}")
print(f"Core-set size: {len(demo_core_set)}")

# Visualize the core-set distribution (in 2D via PCA-like projection)
from sklearn.decomposition import PCA

demo_labeled = x_demo_pool[demo_core_set]
pca = PCA(n_components=2)
pca_full = pca.fit_transform(x_demo_pool.cpu().numpy())
pca_core = pca.transform(demo_labeled.cpu().numpy())

plt.scatter(pca_full[:, 0], pca_full[:, 1], c='lightblue', s=10, alpha=0.5, label='Pool')
plt.scatter(pca_core[:, 0], pca_core[:, 1], c='red', s=50, marker='*', label='Core-set')
plt.xlabel('PCA dim 1')
plt.ylabel('PCA dim 2')
plt.title(f'Ellipsoid core-set (N_M={len(demo_core_set)}) in 2D projection')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.6 Putting it together
#
# The `select_batch` function composes all the components above into the full two-stage GBALD algorithm:
#
# ```python
# def select_batch(model, x_unlabeled, x_labeled, batch_size, seed, ...):
#     # Stage 1: Construct ellipsoid core-set
#     x_pool = torch.cat([x_labeled, x_unlabeled], dim=0)
#     core_set_indices = construct_ellipsoid_core_set(x_pool, core_set_size, R_0, eta, seed)
#     # Update x_labeled with core-set acquisitions
#     # Update x_unlabeled to exclude core-set
#
#     # Stage 2: BALD scoring + geometric ranking
#     bald_scores = compute_bald_scores(model, x_unlabeled, mc_samples)
#     top_candidates = topk(bald_scores, batch_returns)
#     selected = geometric_ranking(top_candidates, x_labeled, R_0, batch_size)
#
#     return selected  # indices into original x_unlabeled
# ```
#
# **paper-element: alg-gbald**
# **paper-element: concept-gbald-framework**

# %% [markdown]
# ## 5. Running active learning end-to-end

# %% [markdown]
# ### 5.1 The acquisition loop
#
# We run the GBALD acquisition loop for `num_rounds` rounds. Each round:
# 1. Call `select_batch` to acquire `batch_size` new samples.
# 2. Add them to the labeled set.
# 3. **Retrain from scratch** (build a fresh model, train on the current labeled set).
# 4. Evaluate on the test set and record accuracy.
#
# **Important**: We retrain from scratch each round (not warm-start), per the paper's protocol. The model is rebuilt at the top of every round.

# %%
# Initialize labeled and unlabeled sets from the bootstrap core-set
labeled_idx = np.array(core_set_indices)
unlabeled_idx = np.array([i for i in range(len(x_pool)) if i not in core_set_indices])

learning_curve = []

def train_eval_current(round_seed):
    """Train from scratch and evaluate on test set."""
    model = build_model(
        input_dim=input_dim,
        n_classes=n_classes,
        hidden_dim=cfg.get("hidden_dim", 256),
        dropout_rate=cfg.get("dropout_rate", 0.5),
    )
    model = train_from_scratch(
        model,
        x_pool[labeled_idx],
        y_pool[labeled_idx],
        learning_rate=cfg["learning_rate"],
        max_epochs=cfg["max_epochs"],
        train_until_accuracy=cfg["train_until_accuracy"],
        seed=round_seed,
    )
    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(x_test).argmax(dim=1)
        acc = (preds == y_test).float().mean().item()
    return model, acc

# Initial evaluation (before any acquisition)
model, acc = train_eval_current(SEED)
learning_curve.append((len(labeled_idx), acc))
print(f"Initial: {len(labeled_idx)} labeled, test accuracy = {acc:.4f}")

# Acquisition loop
for r in range(cfg["num_rounds"]):
    # Get current unlabeled set
    x_unlabeled = x_pool[unlabeled_idx]
    x_labeled = x_pool[labeled_idx]
    
    # Acquire batch_size new samples using GBALD
    selected_positions = select_batch(
        model=model,
        x_unlabeled=x_unlabeled,
        x_labeled=x_labeled,
        batch_size=cfg["batch_size"],
        seed=SEED + r,
        mc_samples=cfg["mc_samples"],
        core_set_size=cfg["core_set_size"],
        R_0=cfg["R_0"],
        eta=cfg["eta"],
        batch_returns=cfg["batch_returns"],
    )
    
    # Map positions to global indices
    chosen = unlabeled_idx[np.array(selected_positions)]
    
    # Update labeled/unlabeled sets
    labeled_idx = np.union1d(labeled_idx, chosen)
    unlabeled_idx = np.setdiff1d(unlabeled_idx, chosen)
    
    # Retrain from scratch and evaluate
    model, acc = train_eval_current(SEED + r + 1)
    learning_curve.append((len(labeled_idx), acc))
    
    print(f"Round {r+1}: {len(labeled_idx)} labeled, test accuracy = {acc:.4f}, acquired {len(chosen)}")

# %% [markdown]
# ### 5.2 Learning curve

# %%
# Plot learning curve
labels_acquired = [lc[0] for lc in learning_curve]
test_accuracies = [lc[1] for lc in learning_curve]

plt.figure(figsize=(10, 6))
plt.plot(labels_acquired, test_accuracies, 'b-o', linewidth=2, markersize=8, label='GBALD')
plt.xlabel('Number of labeled samples')
plt.ylabel('Test accuracy')
plt.title(f'GBALD learning curve (MNIST, smoke-scale, seed={SEED})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nFinal: {labels_acquired[-1]} labeled samples, test accuracy = {test_accuracies[-1]:.4f}")
print(f"Total rounds: {cfg['num_rounds']}, batch_size: {cfg['batch_size']}")
print(f"Expected labeled count: {len(core_set_indices)} + {cfg['num_rounds']} × {cfg['batch_size']} = {len(core_set_indices) + cfg['num_rounds'] * cfg['batch_size']}")

# %% [markdown]
# ## 6. Use your own data
#
# To use GBALD on your own dataset, you have two options:
#
# **Option A: Pass an explicit path**
# ```python
# x_pool, y_pool, x_test, y_test = load_data(
#     path="/path/to/your/data.pt",
#     pool_size=...,
#     n_test=...,
#     seed=...,
# )
# ```
#
# **Option B: Drop file in `method/example_data/`**
# Place your data file in `method/example_data/` and call `load_data()` without a path (it defaults to that directory).
#
# **Supported formats**:
# - `.pt`: Torch tensor tuple `(x_pool, y_pool, x_test, y_test)`
# - `.json`: Dict with keys `"x_pool"`, `"y_pool"`, `"x_test"`, `"y_test"` (lists or nested lists)
# - `.csv`: CSV with feature columns and a `"label"` column
#
# **Architecture swap note**: If your data is image-shaped (e.g., CIFAR-10, custom images), you may want to swap the bundled MLP for a CNN. See `method/model.py` for swap instructions. The essential contract is: the model must have dropout layers active at inference for MC dropout to work.
