<!-- method-md {"explained_equations": ["eq-gradient-embedding", "eq-kmeanspp-distance", "eq-kmeanspp-probability", "eq-classifier-prediction", "eq-cross-entropy", "eq-softmax-gradient-block", "eq-softmax-cross-entropy", "eq-gradient-norm"], "inputs": {"explanations": "method_explanations.json", "method_spec": true, "paper_map": true, "params": true}, "schema_version": "1.0.0"} -->
# DEEP BATCH ACTIVE LEARNING BY DIVERSE, UNCERTAIN GRADIENT LOWER BOUNDS — the method, explained

This document explains the paper's method on its own: what the key equations do, why they're novel, and the intuition behind them, with every claim anchored to the paper. It is generated from the pipeline's paper decomposition and survives any downstream failure — code or no code, this explanation stands alone.

**How to read this document.** The equation sections follow the paper's own order. If you prefer to start from the algorithm and follow the math from there, jump to the [algorithm walkthrough](#algorithm-walkthrough) — its steps link back to each equation's explanation.

## What this method does

BADGE represents each unlabeled example by the cross-entropy gradient at the final layer under its predicted label, then applies k-MEANS++ seeding to those embeddings to select an uncertain and diverse batch.

## The key equations

Each section below covers one equation the pipeline implements or demonstrates in code. The **The paper states** block quotes the paper's own verbatim statement of the equation, so you can check the explanation against the paper's words (a marked paraphrase appears when the verbatim quote was not captured). Equations with any other role are not explained here; they are indexed in the [source map](#source-map) at the end.

### <a id="eq-gradient-embedding"></a>BADGE gradient embedding

*Algorithm 1 · role: implement*

<!-- derived-block: begin {"kind": "implemented_by", "spec": {"element_id": "eq-gradient-embedding"}} -->
**Implemented by:** [`compute_gradient_embeddings`](method/method.py) (`method/method.py:20`) · demonstrated in [the notebook](notebook.ipynb#sec-42-hallucinated-gradient-embeddings)
<!-- derived-block: end -->

**The paper states:**

$g_x = \frac{\partial}{\partial \theta_{\text{out}}} \ell_{\text{CE}}(f(x;\theta), \hat{y}(x))|_{\theta=\theta_t}$

```text
g_x = d/d(theta_out) cross_entropy(f(x; theta), y_hat(x)) evaluated at theta_t
```

**What it does:** For an unlabeled input x, this computes g_x, the gradient of the cross-entropy loss with respect to the output-layer parameters theta_out. The loss is evaluated using the network's current predicted label y_hat(x) as a stand-in for the unavailable true label, with all parameters evaluated at the current model theta_t. This vector is the point's representation in BADGE's hallucinated gradient space, which is subsequently used to choose a batch.

**Why it's novel:** Cross-entropy and a last-layer gradient are standard machinery; the distinctive use here is to form that gradient with the hallucinated label y_hat(x) for each unlabeled point. BADGE then selects points that are both high magnitude and disparate in this gradient representation, rather than using predictive uncertainty or input-space diversity alone.

**The intuition:** Treat the model's current guess as a provisional label and ask how the output layer would need to move to support that guess. A confident prediction produces a gradient consistent with little need to change, whereas uncertainty can yield a larger or more informative update direction. Points whose g_x vectors point in different directions represent potentially different ways the model could need to change, so selecting separated vectors avoids filling a batch with nearly identical uncertain examples.

*Builds on: [eq-classifier-prediction](#eq-classifier-prediction), [eq-cross-entropy](#eq-cross-entropy)*

### <a id="eq-kmeanspp-distance"></a>Nearest-center distance

*Appendix A, Algorithm 2 · role: implement*

<!-- derived-block: begin {"kind": "implemented_by", "spec": {"element_id": "eq-kmeanspp-distance"}} -->
**Implemented by:** [`kmeans_plus_plus_seeding`](method/method.py) (`method/method.py:61`) · demonstrated in [the notebook](notebook.ipynb#sec-43-k-means-gradient-batch-seeding)
<!-- derived-block: end -->

**The paper states:**

> D_t(x) := \min_{c \in C_{t-1}} \|x - c\|_2.

```text
D_t(x) = min(norm(x - c, 2) for c in C)
```

**What it does:** For a candidate point x, D_t(x) is the Euclidean distance from x to its nearest already selected center in C_{t-1}. It produces one nonnegative distance score used to characterize how close the candidate is to the current set of centers.

**Why it's novel:** This is standard nearest-center distance machinery rather than a novel equation by itself. In BADGE's broader selection strategy, it supplies the diversity component: a point far from every existing center is distinct in the representation space where the method performs selection.

**The intuition:** Imagine that the selected centers already mark regions represented in the batch. A candidate with small D_t(x) lies near one of those marked regions and is therefore redundant under this distance measure; a candidate with large D_t(x) reaches into a region not yet covered. The equation does not itself measure uncertainty—it only says how separated a candidate is from the current centers.

### <a id="eq-kmeanspp-probability"></a>k-MEANS++ sampling probability

*Appendix A, Algorithm 2 · role: implement*

<!-- derived-block: begin {"kind": "implemented_by", "spec": {"element_id": "eq-kmeanspp-probability"}} -->
**Implemented by:** [`kmeans_plus_plus_seeding`](method/method.py) (`method/method.py:61`) · demonstrated in [the notebook](notebook.ipynb#sec-43-k-means-gradient-batch-seeding)
<!-- derived-block: end -->

**The paper states:**

> c_t \leftarrow \text{Sample } x \text{ from } G \text{ with probability } \frac{D_t(x)^2}{\sum_{x \in G} D_t(x)^2}.

```text
sample x from G with weights D_t(x)^2
```

**What it does:** This rule draws the next center c_t from the gradient-embedding set G. Each candidate x receives weight D_t(x)^2, where D_t(x) is its distance to the nearest center already selected; normalizing these weights by their sum turns them into sampling probabilities. The selected center is therefore one part of the batch-construction procedure rather than a deterministic highest-distance choice.

**Why it's novel:** The squared-distance sampling rule is standard k-MEANS++ machinery, not a new equation by itself. In BADGE, it is used to construct a batch from the paper's hallucinated gradient embeddings, so the standard rule supplies the diversity component alongside the embedding magnitudes that encode predictive uncertainty.

**The intuition:** A point already close to a chosen center represents much the same region of G, so it gets little chance of being selected again. A point far from every chosen center has a much larger squared-distance weight and is more likely to represent a new region. Squaring the distance accentuates that preference while retaining randomness, rather than forcing every next choice to be the single farthest point.

*Builds on: [eq-kmeanspp-distance](#eq-kmeanspp-distance)*

### <a id="eq-classifier-prediction"></a>Neural classifier prediction

*Section 2 · role: implement*

<!-- derived-block: begin {"kind": "implemented_by", "spec": {"element_id": "eq-classifier-prediction"}} -->
**Implemented by:** [`GradientEmbeddingMLP.forward`](method/model.py) (`method/model.py:35`) · also used in `compute_gradient_embeddings` (`method/method.py:20`)
<!-- derived-block: end -->

**The paper states:**

$h_{\theta}(x) = \operatorname{argmax}_{y \in [K]} f(x; \theta)_y$

```text
y_hat = argmax(f(x; theta))
```

**What it does:** For an input x, the neural network f(x; θ) produces one score or probability f(x; θ)_y for each of K classes. The classifier h_θ(x) returns the class index y with the largest such output. In BADGE, this predicted class supplies the hallucinated label used when forming a gradient embedding for an unlabeled point.

**Why it's novel:** This argmax prediction rule is standard multiclass-classification machinery, not a new contribution by itself. Its role here is to provide the predicted label that BADGE uses in its hallucinated gradient construction, which the paper positions as a way to combine uncertainty and diversity in batch selection.

**The intuition:** The network is making its best current guess: among all class outputs, take the largest one. That guess lets BADGE ask, in effect, how the model's parameters would be pushed if this unlabeled example really had the class the model currently favors; the later gradient representation can then distinguish confident, redundant guesses from informative ones.

*Used in the [algorithm walkthrough](#algorithm-walkthrough): [Hallucinated gradient embedding computation](#alg-gradient-embedding)*

### <a id="eq-cross-entropy"></a>Cross-entropy training loss

*Section 2 · role: implement*

<!-- derived-block: begin {"kind": "implemented_by", "spec": {"element_id": "eq-cross-entropy"}} -->
**Implemented by:** [`train_from_scratch`](method/training.py) (`method/training.py:22`) · demonstrated in [the notebook](notebook.ipynb#sec-34-bootstrap-labeled-set)
<!-- derived-block: end -->

**The paper states:**

$\ell_{\mathrm{CE}}(p, y) = \sum_{i=1}^K I(y=i) \ln 1/p_i = \ln 1/p_y$ .

```text
cross_entropy(p, y) = -log(p[y])
```

**What it does:** This loss takes the classifier's predicted class-probability vector p and the true class y. The indicator I(y=i) selects only the probability assigned to the true class, so the K-class sum reduces to ln(1/p_y), equivalently -log(p_y); training minimizes this quantity over labeled examples.

**Why it's novel:** This is standard cross-entropy training loss, not a novel component of BADGE. It supplies the ordinary supervised objective for the neural classifier rather than introducing a separate uncertainty-diversity trade-off.

**The intuition:** The loss asks one simple question: how much probability did the model give the answer that turned out to be correct? Assigning the true class probability close to one gives a loss near zero, whereas assigning it a very small probability produces a large penalty, so minimizing the loss pushes probability toward the observed label.

*Builds on: [eq-classifier-prediction](#eq-classifier-prediction)*

*Used in the [algorithm walkthrough](#algorithm-walkthrough): [BADGE batch active learning](#alg-badge), [Hallucinated gradient embedding computation](#alg-gradient-embedding)*

### <a id="eq-softmax-gradient-block"></a>Per-class gradient embedding block

*Section 3, Equation (1) · role: implement*

<!-- derived-block: begin {"kind": "implemented_by", "spec": {"element_id": "eq-softmax-gradient-block"}} -->
**Implemented by:** [`GradientEmbeddingMLP.forward_with_embedding`](method/model.py) (`method/model.py:40`) · also used in `compute_gradient_embeddings` (`method/method.py:20`)
<!-- derived-block: end -->

**The paper states:**

$$(g_x)_i = \frac{\partial}{\partial W_i} \ell_{CE}(f(x;\theta), \hat{y}) = (p_i - I(\hat{y} = i))z(x; V). \tag{1}$$

```text
g_x[i] = (p[i] - indicator(y_hat == i)) * z(x; V) for each class i
```

**What it does:** For each output class i, this computes that class's block of the gradient of cross-entropy loss with respect to the final-layer weights W_i. The block is the penultimate-layer feature vector z(x; V), scaled by p_i minus an indicator that the hallucinated label y_hat equals i. Concatenating these class blocks gives the gradient embedding used to represent an unlabeled example for batch selection.

**Why it's novel:** The derivative itself is standard softmax cross-entropy machinery. In BADGE, its distinctive use is to form a gradient embedding with the hallucinated predicted label y_hat: the residual p_i - I(y_hat = i) combines the model's class probabilities with z(x; V), so examples are compared in this gradient space rather than by uncertainty scores or feature-space diversity alone.

**The intuition:** Think of each block as the update the final classifier would receive if its current most likely class were treated as the label. A confident prediction makes p_i close to the indicator for its predicted class and leaves a small gradient, whereas uncertainty leaves larger residuals across classes. The feature vector then says which kind of input would cause that update, allowing BADGE to seek examples whose potential updates are both substantial and different from one another.

*Builds on: [eq-gradient-embedding](#eq-gradient-embedding), [eq-softmax-cross-entropy](#eq-softmax-cross-entropy)*

### <a id="eq-softmax-cross-entropy"></a>Softmax cross-entropy at the final layer

*Section 3, multiclass softmax example · role: implement*

<!-- derived-block: begin {"kind": "implemented_by", "spec": {"element_id": "eq-softmax-cross-entropy"}} -->
**Implemented by:** [`compute_gradient_embeddings`](method/method.py) (`method/method.py:20`) · demonstrated in [the notebook](notebook.ipynb#sec-42-hallucinated-gradient-embeddings)
<!-- derived-block: end -->

**The paper states:**

$$\ell_{\text{CE}}(f(x;\theta), y) = \ln \left( \sum_{j=1}^{K} e^{W_j \cdot z(x;V)} \right) - W_y \cdot z(x;V).$$

```text
loss = log(sum_j(exp(dot(W_j, z)))) - dot(W_y, z)
```

**What it does:** This is the multiclass softmax cross-entropy loss for an input x with true class y. It takes the penultimate-layer representation z(x;V), scores it with each final-layer class weight vector W_j, and returns the log of the summed exponentiated scores minus the score W_y dot z(x;V) of the true class. Minimizing it raises the true class's score relative to the other K class scores.

**Why it's novel:** This is standard softmax cross-entropy machinery, not the paper's novelty. It is included because writing the loss in terms of the final-layer weights W_j and representation z(x;V) exposes the quantities from which the method's gradient-based representation is built.

**The intuition:** Think of the final layer as holding one prototype direction W_j per class, while z(x;V) describes the input just before classification. The loss is small when the true class direction W_y aligns much more strongly with that representation than the competing class directions. If a dog image receives a high cat score as well as a dog score, the log-sum-exp term remains large, so the loss indicates that the classifier still needs to separate those alternatives.

*Builds on: [eq-cross-entropy](#eq-cross-entropy)*

### <a id="eq-gradient-norm"></a>Last-layer gradient norm

*Section 3, Proposition 1 · role: demonstrate*

**The paper states:**

$$||g_x^y||^2 = \Big(\sum_{i=1}^K p_i^2 + 1 - 2p_y\Big)||z(x;V)||^2.$$

```text
gradient_norm_sq(y) = (sum_i(p[i]^2) + 1 - 2*p[y]) * norm(z)^2
```

**What it does:** This equation computes the squared Euclidean norm of the last-layer gradient embedding g_x^y for input x under a chosen label y. It multiplies the squared norm of the penultimate representation z(x;V) by a probability-dependent factor: the sum of squared class probabilities p_i^2, plus 1, minus twice the probability p_y assigned to y. Thus both the feature scale and how the model scores the selected label determine the gradient's magnitude.

**Why it's novel:** The norm identity itself is standard algebra from the last-layer softmax-gradient blocks, rather than a new loss or optimization objective. Its role in BADGE is to make explicit why a hallucinated-label gradient can encode uncertainty: choosing a label y with low p_y increases the displayed factor, while the gradient embedding can also be used for batch diversity.

**The intuition:** Think of the gradient magnitude as how strongly the final classifier would need to move if this point were assigned label y. If the network already gives y high probability, the negative term -2p_y makes that hypothetical correction smaller; if y has low probability, the correction is larger. The representation norm then scales this signal by how large the point's last hidden-layer feature vector is.

*Builds on: [eq-softmax-gradient-block](#eq-softmax-gradient-block)*

## <a id="algorithm-walkthrough"></a>Algorithm walkthrough

### <a id="alg-badge"></a>BADGE batch active learning

*Algorithm 1*

<!-- derived-block: begin {"kind": "implemented_by", "spec": {"element_id": "alg-badge"}} -->
**Implemented by:** [`select_batch`](method/method.py) (`method/method.py:118`) · demonstrated in [the notebook](notebook.ipynb#sec-51-the-acquisition-loop)
<!-- derived-block: end -->

BADGE repeatedly embeds unlabeled points with hallucinated last-layer gradients, selects a diverse batch, queries labels, and retrains.

```text
Initialize S with M random labeled pool examples. Train theta on S. For each round: predict y_hat and compute g_x for each unlabeled x; choose B embeddings with k-means++ seeding; query their labels; add them to S; retrain theta from scratch. Return theta.
```

*Equations in this algorithm, explained above: [Cross-entropy training loss](#eq-cross-entropy)*

### <a id="alg-gradient-embedding"></a>Hallucinated gradient embedding computation

*Algorithm 1 and Section 3*

<!-- derived-block: begin {"kind": "implemented_by", "spec": {"element_id": "alg-gradient-embedding"}} -->
**Implemented by:** [`GradientEmbeddingMLP.forward_with_embedding`](method/model.py) (`method/model.py:40`) · also used in `compute_gradient_embeddings` (`method/method.py:20`)
<!-- derived-block: end -->

For every unlabeled point, use the model's predicted label as a surrogate label and compute the cross-entropy gradient with respect to the output layer.

```text
p = model(x); y_hat = argmax(p); g_x = gradient(cross_entropy(p, y_hat), output_layer_parameters)
```

*Equations in this algorithm, explained above: [Neural classifier prediction](#eq-classifier-prediction), [Cross-entropy training loss](#eq-cross-entropy)*

### <a id="alg-kmeanspp-seeding"></a>k-MEANS++ seeding over gradient embeddings

*Section 3 and Appendix A, Algorithm 2*

<!-- derived-block: begin {"kind": "implemented_by", "spec": {"element_id": "alg-kmeanspp-seeding"}} -->
**Implemented by:** [`kmeans_plus_plus_seeding`](method/method.py) (`method/method.py:61`) · demonstrated in [the notebook](notebook.ipynb#sec-43-k-means-gradient-batch-seeding)
<!-- derived-block: end -->

Select the first gradient embedding uniformly, then repeatedly sample candidates proportional to squared distance from the nearest selected embedding.

```text
Choose first center uniformly. Until B centers: D(x) = min distance from x to chosen centers; sample x with probability D(x)^2 / sum_u D(u)^2; add x.
```

### Where the method plugs in

The paper's contribution is packaged as `select_batch(model, x_unlabeled, batch_size, seed) -> List[int]` — Compute predicted-label last-layer gradient embeddings for x_unlabeled and use seeded k-MEANS++ sampling to return exactly batch_size positions into that same tensor.

## Parameters and provenance

_**4 of 8 parameters** are taken directly from the paper. 2 use a runtime value that differs from the paper's stated value, with the paper value preserved for comparison. 2 were supplemented from field conventions where the paper does not specify them, and are the values to scrutinise most when judging how closely the code follows the paper._

| parameter | value | source | paper says |
|---|---|---|---|
| batch_size | 100 | paper | Algorithm 1 and Section 4; "batch size B varying from  $\{100,1000,10000\}$ ." |
| hidden_dim | 256 | system_inferred |  |
| initial_labeled | 100 | paper | Algorithm 1 and Section 4; "M=100 being the number of initial random labeled examples" |
| learning_rate | 0.001 | paper | Section 4 |
| max_epochs | 50 | system_inferred |  |
| num_rounds | 5 | system_default | paper value: 349 |
| pool_size | 12000 | system_default | paper value: full training set |
| train_until_accuracy | 0.99 | paper | Section 4 EXPERIMENTS |

## <a id="source-map"></a>Source map

This table is the completeness index and citation trail for the whole document: one row for every element the decomposition extracted from the paper — including the ones that did not get a full section above — with the paper location it came from and the role the pipeline assigned it. Use it to confirm nothing the decomposition found was silently dropped, and to jump from any element to where it lives in the paper.

<!-- derived-block: begin {"kind": "source_map", "spec": {}} -->
| element | type | paper location | role | implementation |
|---|---|---|---|---|
| <a id="concept-pool-based-active-learning"></a>Pool-based active learning setting | concept | Section 2 | implement | `select_batch` — `method/method.py:118` |
| <a id="eq-classifier-prediction"></a>Neural classifier prediction | equation | Section 2 | implement | `GradientEmbeddingMLP.forward` — `method/model.py:35` |
| <a id="eq-cross-entropy"></a>Cross-entropy training loss | equation | Section 2 | implement | `train_from_scratch` — `method/training.py:22` |
| <a id="alg-badge"></a>BADGE batch active learning | algorithm | Algorithm 1 | implement | `select_batch` — `method/method.py:118` |
| <a id="alg-gradient-embedding"></a>Hallucinated gradient embedding computation | algorithm | Algorithm 1 and Section 3 | implement | `GradientEmbeddingMLP.forward_with_embedding` — `method/model.py:40` |
| <a id="eq-gradient-embedding"></a>BADGE gradient embedding | equation | Algorithm 1 | implement | `compute_gradient_embeddings` — `method/method.py:20` |
| <a id="concept-gradient-uncertainty"></a>Gradient magnitude as uncertainty | concept | Section 3 | implement | — |
| <a id="eq-softmax-cross-entropy"></a>Softmax cross-entropy at the final layer | equation | Section 3, multiclass softmax example | implement | `compute_gradient_embeddings` — `method/method.py:20` |
| <a id="eq-softmax-gradient-block"></a>Per-class gradient embedding block | equation | Section 3, Equation (1) | implement | `GradientEmbeddingMLP.forward_with_embedding` — `method/model.py:40` |
| <a id="property-hallucinated-gradient-lower-bound"></a>Predicted-label gradient is a lower bound | property | Section 3, Proposition 1 | demonstrate | — |
| <a id="eq-gradient-norm"></a>Last-layer gradient norm | equation | Section 3, Proposition 1 | demonstrate | — |
| <a id="alg-kmeanspp-seeding"></a>k-MEANS++ seeding over gradient embeddings | algorithm | Section 3 and Appendix A, Algorithm 2 | implement | `kmeans_plus_plus_seeding` — `method/method.py:61` |
| <a id="eq-kmeanspp-distance"></a>Nearest-center distance | equation | Appendix A, Algorithm 2 | implement | `kmeans_plus_plus_seeding` — `method/method.py:61` |
| <a id="eq-kmeanspp-probability"></a>k-MEANS++ sampling probability | equation | Appendix A, Algorithm 2 | implement | `kmeans_plus_plus_seeding` — `method/method.py:61` |
| <a id="concept-diverse-uncertain-batches"></a>Joint diversity and uncertainty acquisition | concept | Section 3 | implement | `select_batch` — `method/method.py:118` |
| <a id="property-kmeanspp-diversity-guarantee"></a>Expected k-means approximation diversity guarantee | property | Appendix A | demonstrate | — |
| <a id="hyp-initial-labeled-examples"></a>Initial labeled pool size M | hyperparameter | Algorithm 1 and Section 4 | implement | — |
| <a id="hyp-batch-size"></a>Acquisition batch size B | hyperparameter | Algorithm 1 and Section 4 | implement | — |
| <a id="concept-retrain-from-scratch"></a>Retraining after acquisition | concept | Section 4 | implement | — |
| <a id="experiment-badge-robustness"></a>BADGE robustness evaluation | experiment | Section 4 | implement | — |
| <a id="hyp-training-protocol"></a>Experimental training protocol | hyperparameter | Section 4 | implement | — |
| <a id="property-badge-empirical-robustness"></a>Empirical robustness across conditions | property | Abstract and Section 6 | demonstrate | — |
<!-- derived-block: end -->
