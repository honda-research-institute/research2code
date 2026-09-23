<!-- method-md {"explained_equations": ["eq-classifier-cross-entropy", "eq-softmax-cross-entropy", "eq-gradient-embedding"], "inputs": {"explanations": "method_explanations.json", "method_spec": true, "paper_map": true, "params": true}, "schema_version": "1.0.0"} -->
# DEEP BATCH ACTIVE LEARNING BY DIVERSE, UNCERTAIN GRADIENT LOWER BOUNDS — the method, explained

This document explains the paper's method on its own: what the key equations do, why they're novel, and the intuition behind them, with every claim anchored to the paper. It is generated from the pipeline's paper decomposition and survives any downstream failure — code or no code, this explanation stands alone.

**How to read this document.** The equation sections follow the paper's own order. If you prefer to start from the algorithm and follow the math from there, jump to the [algorithm walkthrough](#algorithm-walkthrough) — its steps link back to each equation's explanation.

## What this method does

Selects informative and diverse batches using hallucinated last-layer gradient embeddings and k-MEANS++ seeding.

## The key equations

Each section below covers one equation the pipeline implements or demonstrates in code. The **The paper states** block quotes the paper's own verbatim statement of the equation, so you can check the explanation against the paper's words (a marked paraphrase appears when the verbatim quote was not captured). Equations with any other role are not explained here; they are indexed in the [source map](#source-map) at the end.

### <a id="eq-classifier-cross-entropy"></a>Neural classifier and cross-entropy training objective

*Section 2 · role: implement*

<!-- derived-block: begin {"kind": "implemented_by", "spec": {"element_id": "eq-classifier-cross-entropy"}} -->
**Implemented by:** [`BADGEClassifier.forward`](method/model.py) (`method/model.py:37`)
<!-- derived-block: end -->

**The paper states:**

We optimize the parameters by minimizing the cross-entropy loss  $\mathbb{E}_S[\ell_{\mathrm{CE}}(f(x; \theta), y)]$  over the labeled examples, where  $\ell_{\mathrm{CE}}(p, y) = \sum_{i=1}^K I(y=i) \ln 1/p_i = \ln 1/p_y$ .

```text
loss = mean(cross_entropy(model(x), y) for x, y in labeled_set); update model parameters to minimize loss
```

**What it does:** The neural classifier f(x; theta) produces a probability p_i for each of the K classes when given input x. For a labeled example whose true class is y, the cross-entropy loss is ln(1/p_y), so training minimizes the mean of this loss over the labeled set by updating the network parameters theta. The classifier's prediction is the class with the highest score or probability.

**Why it's novel:** This is standard multiclass cross-entropy training, not the novel part of BADGE. It supplies the trained classifier and the per-example loss from which the later hallucinated gradient representation is constructed; BADGE's contribution is to use that representation to combine uncertainty and diversity when selecting a batch.

**The intuition:** The loss is small when the network assigns high probability to the observed class y and large when it is wrong or doubtful about y. Repeating this over labeled examples teaches the network to put its probability mass on the correct class, producing the current model whose uncertainty and gradients can then guide active selection.

*Used in the [algorithm walkthrough](#algorithm-walkthrough): [BADGE batch active learning](#alg-badge)*

### <a id="eq-softmax-cross-entropy"></a>Softmax cross-entropy in last-layer parameterization

*Section 3 · role: implement*

<!-- derived-block: begin {"kind": "implemented_by", "spec": {"element_id": "eq-softmax-cross-entropy"}} -->
**Implemented by:** [`compute_gradient_embeddings`](method/method.py) (`method/method.py:18`) · demonstrated in [the notebook](notebook.ipynb#sec-42-hallucinated-gradient-embeddings)
<!-- derived-block: end -->

**The paper states:**

$$\ell_{\text{CE}}(f(x;\theta), y) = \ln \left( \sum_{j=1}^{K} e^{W_j \cdot z(x;V)} \right) - W_y \cdot z(x;V).$$

```text
loss = log(sum(exp(W[j] dot z) for j in classes)) - W[y] dot z
```

**What it does:** This equation computes the cross-entropy loss for an input x with observed class label y. The penultimate network representation is z(x;V), and W_j is the output-layer weight vector for class j; the log-sum-exp term aggregates the logits W_j · z over all K classes, while the second term subtracts the logit of the observed class. Its role is to express the training loss directly in the last-layer parameters W and the representation parameters V.

**Why it's novel:** This is standard softmax cross-entropy, not the novel part of BADGE; it supplies the loss whose last-layer gradient is used to form the method's hallucinated gradient embeddings. The parameterization makes the dependence on W_j and z(x;V) explicit, which is important for that gradient-space construction.

**The intuition:** The loss compares the observed class's logit with the log-sum-exp score of all classes: increasing W_y · z lowers the loss, while competing class logits increase the aggregate term. In BADGE, this ordinary classification loss is the starting point for asking how an example would change the model if its label were assigned to each possible class.

### <a id="eq-gradient-embedding"></a>Hallucinated final-layer gradient embedding

*Section 3, Equation (1) · role: implement*

<!-- derived-block: begin {"kind": "implemented_by", "spec": {"element_id": "eq-gradient-embedding"}} -->
**Implemented by:** [`compute_gradient_embeddings`](method/method.py) (`method/method.py:18`) · also used in `BADGEClassifier.forward_with_embedding` (`method/model.py:31`) · demonstrated in [the notebook](notebook.ipynb#sec-42-hallucinated-gradient-embeddings)
<!-- derived-block: end -->

**The paper states:**

$$(g_x)_i = \frac{\partial}{\partial W_i} \ell_{CE}(f(x;\theta), \hat{y}) = (p_i - I(\hat{y} = i))z(x; V). \tag{1}$$

```text
for class i: g_block[i] = (p[i] - indicator(y_hat == i)) * penultimate_features
```

**What it does:** For an unlabeled input x, this equation forms an embedding g_x by differentiating the cross-entropy loss ell_CE of the network f with respect to each final-layer weight block W_i. It uses the model's predicted class y_hat as if it were the label: for class i, the block is the probability residual p_i - I(y_hat = i) multiplied by the penultimate feature vector z(x; V), where p_i is the predicted probability and I is an indicator. The resulting collection of class-specific blocks is the point's hallucinated gradient representation for acquisition.

**Why it's novel:** The cross-entropy gradient itself is standard machinery, but BADGE uses this particular predicted-label, or hallucinated-label, final-layer gradient as the acquisition embedding. Unlike selecting only low-confidence points or only diverse points, the embedding exposes both the probability residuals and the shared feature vector, so its geometry can support the paper's stated combination of uncertainty and diversity without a hand-tuned trade-off hyperparameter.

**The intuition:** Pretend that the network's own prediction y_hat is the missing label, then ask which final-layer weights this example would push and in what direction. A point with a large probability residual produces a larger gradient signal, while two points with similar residuals but different penultimate features point in different directions; thus magnitude reflects how much the current model could be changed and direction reflects how the examples differ. This is the paper's core idea of representing candidates in a hallucinated gradient space so a batch can be both uncertain and disparate.

*Builds on: [concept-hallucinated-label](#concept-hallucinated-label)*

*Used in the [algorithm walkthrough](#algorithm-walkthrough): [BADGE batch active learning](#alg-badge), [k-MEANS++ seeding sampler](#alg-kmeans-plus-plus)*

## <a id="algorithm-walkthrough"></a>Algorithm walkthrough

### <a id="alg-badge"></a>BADGE batch active learning

*Algorithm 1*

<!-- derived-block: begin {"kind": "implemented_by", "spec": {"element_id": "alg-badge"}} -->
**Implemented by:** [`select_batch`](method/method.py) (`method/method.py:89`) · demonstrated in [the notebook](notebook.ipynb#sec-51-acquisition-loop)
<!-- derived-block: end -->

BADGE initializes a labeled set, repeatedly computes hallucinated last-layer gradient embeddings for the remaining pool, samples a diverse batch with k-MEANS++, queries labels, retrains from the accumulated labeled data, and returns the final model.

```text
S = uniform_sample(U, M); query labels(S); theta = train_cross_entropy(S); for t in 1..T: candidates = U - S; for x in candidates: y_hat[x] = argmax(model(theta, x)); g[x] = gradient_last_layer(cross_entropy(model(theta, x), y_hat[x])); S_t = kmeans_plus_plus(g, B); query labels(S_t); S = S union S_t; theta = train_cross_entropy(S); return theta
```

*Equations in this algorithm, explained above: [Neural classifier and cross-entropy training objective](#eq-classifier-cross-entropy), [Hallucinated final-layer gradient embedding](#eq-gradient-embedding)*

### <a id="alg-kmeans-plus-plus"></a>k-MEANS++ seeding sampler

*Appendix A, Algorithm 2*

<!-- derived-block: begin {"kind": "implemented_by", "spec": {"element_id": "alg-kmeans-plus-plus"}} -->
**Implemented by:** [`kmeans_plus_plus_seeding`](method/method.py) (`method/method.py:49`) · demonstrated in [the notebook](notebook.ipynb#sec-43-sequential-k-means-selection)
<!-- derived-block: end -->

Sequentially selects the first embedding uniformly and each later embedding with probability proportional to squared distance from its nearest selected center.

```text
C = {uniform_sample(G)}; while len(C) < k: distance[x] = min squared distance from x to C; choose x with probability distance[x] / sum(distance); C.add(x); return C
```

*Equations in this algorithm, explained above: [Hallucinated final-layer gradient embedding](#eq-gradient-embedding)*

### Where the method plugs in

The paper's contribution is packaged as `select_batch(model, x_unlabeled, batch_size, seed) -> List[int]` — Compute predicted-label final-layer gradient embeddings and select a diverse batch with randomized k-MEANS++ seeding.

## Parameters and provenance

_**4 of 8 parameters** are taken directly from the paper. 2 use a runtime value that differs from the paper's stated value, with the paper value preserved for comparison. 2 were supplemented from field conventions where the paper does not specify them, and are the values to scrutinise most when judging how closely the code follows the paper._

| parameter | value | source | paper says |
|---|---|---|---|
| batch_size | 100 | paper | Algorithm 1 / Section 4 |
| hidden_dim | 256 | system_inferred |  |
| initial_labeled | 100 | paper | Algorithm 1 / Section 4 |
| learning_rate | 0.001 | paper | Section 4 |
| max_epochs | 200 | system_inferred |  |
| num_rounds | 5 | system_default | paper value: 349 |
| pool_size | 12000 | system_default | paper value: full training set |
| train_until_accuracy | 0.99 | paper | Section 4 EXPERIMENTS |

## <a id="source-map"></a>Source map

This table is the completeness index and citation trail for the whole document: one row for every element the decomposition extracted from the paper — including the ones that did not get a full section above — with the paper location it came from and the role the pipeline assigned it. Use it to confirm nothing the decomposition found was silently dropped, and to jump from any element to where it lives in the paper.

<!-- derived-block: begin {"kind": "source_map", "spec": {}} -->
| element | type | paper location | role | implementation |
|---|---|---|---|---|
| <a id="concept-pool-active-learning"></a>Pool-based batch active learning setting | concept | Section 2 | implement | — |
| <a id="eq-classifier-cross-entropy"></a>Neural classifier and cross-entropy training objective | equation | Section 2 | implement | `BADGEClassifier.forward` — `method/model.py:37` |
| <a id="alg-badge"></a>BADGE batch active learning | algorithm | Algorithm 1 | implement | `select_batch` — `method/method.py:89` |
| <a id="concept-hallucinated-label"></a>Hallucinated label | concept | Algorithm 1 / Section 3 | implement | — |
| <a id="eq-gradient-embedding"></a>Hallucinated final-layer gradient embedding | equation | Section 3, Equation (1) | implement | `compute_gradient_embeddings` — `method/method.py:18` |
| <a id="eq-softmax-cross-entropy"></a>Softmax cross-entropy in last-layer parameterization | equation | Section 3 | implement | `compute_gradient_embeddings` — `method/method.py:18` |
| <a id="alg-kmeans-plus-plus"></a>k-MEANS++ seeding sampler | algorithm | Appendix A, Algorithm 2 | implement | `kmeans_plus_plus_seeding` — `method/method.py:49` |
| <a id="prop-gradient-lower-bound"></a>Predicted-label gradient is a lower bound | property | Section 3 | demonstrate | — |
| <a id="eq-true-label-gradient-norm"></a>True-label gradient norm | equation | Proposition 1 | theoretical | — |
| <a id="prop-uncertainty-magnitude"></a>Embedding magnitude captures predictive uncertainty | property | Section 3 | demonstrate | — |
| <a id="prop-diversity-uncertainty-tradeoff"></a>Joint diversity and uncertainty selection | property | Section 3 / Section 6 | demonstrate | — |
| <a id="exp-badge-robustness"></a>BADGE robustness across datasets, architectures, and batch sizes | experiment | Section 4 | implement | — |
| <a id="hyp-active-learning-budget"></a>Initial set, rounds, and batch size | hyperparameter | Algorithm 1 / Section 4 | implement | — |
| <a id="hyp-training-protocol"></a>Training protocol | hyperparameter | Section 4 | implement | — |
<!-- derived-block: end -->
