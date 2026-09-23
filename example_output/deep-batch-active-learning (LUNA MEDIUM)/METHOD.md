<!-- method-md {"explained_equations": ["eq-gradient-embedding", "eq-classifier-argmax", "eq-cross-entropy", "eq-softmax-cross-entropy", "eq-softmax-definition", "eq-softmax-model", "eq-gradient-block"], "inputs": {"explanations": "method_explanations.json", "method_spec": true, "paper_map": true, "params": true}, "schema_version": "1.0.0"} -->
# DEEP BATCH ACTIVE LEARNING BY DIVERSE, UNCERTAIN GRADIENT LOWER BOUNDS — the method, explained

This document explains the paper's method on its own: what the key equations do, why they're novel, and the intuition behind them, with every claim anchored to the paper. It is generated from the pipeline's paper decomposition and survives any downstream failure — code or no code, this explanation stands alone.

**How to read this document.** The equation sections follow the paper's own order. If you prefer to start from the algorithm and follow the math from there, jump to the [algorithm walkthrough](#algorithm-walkthrough) — its steps link back to each equation's explanation.

## What this method does

BADGE selects a batch of unlabeled examples by clustering uncertainty-aware gradient embeddings with k-means++ seeding.

## The key equations

Each section below covers one equation the pipeline implements or demonstrates in code. The **The paper states** block quotes the paper's own verbatim statement of the equation, so you can check the explanation against the paper's words (a marked paraphrase appears when the verbatim quote was not captured). Equations with any other role are not explained here; they are indexed in the [source map](#source-map) at the end.

### <a id="eq-gradient-embedding"></a>Hallucinated last-layer gradient embedding

*Algorithm 1 and Section 3 · role: implement*

<!-- derived-block: begin {"kind": "implemented_by", "spec": {"element_id": "eq-gradient-embedding"}} -->
**Implemented by:** [`compute_gradient_embeddings`](method/method.py) (`method/method.py:21`) · also used in `GradientEmbeddingClassifier` (`method/model.py:28`) · demonstrated in [the notebook](notebook.ipynb#sec-42-hallucinated-gradient-embeddings)
<!-- derived-block: end -->

**The paper states:**

$g_x = \frac{\partial}{\partial \theta_{\text{out}}} \ell_{\text{CE}}(f(x;\theta), \hat{y}(x))|_{\theta=\theta_t}$

```text
g_x = gradient_theta_out(cross_entropy(f(x, theta_t), yhat(x)))
```

**What it does:** For an unlabeled input x, this computes the gradient of the cross-entropy loss with respect to the network's output-layer parameters theta_out, using the model's current parameters theta_t and its own predicted label yhat(x) as the temporary target. The resulting vector g_x is the point's representation in a gradient space, and its direction and magnitude are then available for selecting a batch.

**Why it's novel:** The gradient operation itself is standard machinery; the distinctive change is to evaluate it with the predicted label yhat(x), rather than a known ground-truth label that is unavailable for an unlabeled pool. This creates the paper's hallucinated gradient embedding, whose magnitude can reflect predictive uncertainty while the vectors' relationships can support diversity in the same batch without a hand-tuned uncertainty-versus-diversity weight.

**The intuition:** Treat each unlabeled example as if the network had just received its current guess as the label, and record which output-layer parameters that hypothetical training step would change and by how much. An uncertain example produces a larger hypothetical correction, while examples that would correct the model in different directions are separated in this space; BADGE can therefore seek points that are both informative and non-redundant.

### <a id="eq-classifier-argmax"></a>Neural classifier prediction

*Section 2 · role: implement*

**The paper states:**

$h_{\theta}(x) = \operatorname{argmax}_{y \in [K]} f(x; \theta)_y$

```text
prediction = argmax_y f(x, theta)[y]
```

**What it does:** For an input x, the neural network with parameters theta produces one score f(x; theta)_y for each class y in the K-class set [K]. The argmax selects the class whose score is largest, so h_theta(x) is the classifier's predicted label. This is the basic prediction rule used to turn the network's scores into a discrete class decision.

**Why it's novel:** This is standard classifier machinery, not a BADGE-specific contribution. BADGE builds its active-learning selection strategy from the model's predictions and a hallucinated gradient representation, while this equation simply defines the predicted class from the network scores.

**The intuition:** At inference time, treat each class score as a vote from the network and choose the class with the strongest vote. The equation makes that choice explicit: compare all K scores for x, and return the index of the largest one.

### <a id="eq-cross-entropy"></a>Cross-entropy training objective

*Section 2 · role: implement*

**The paper states:**

$\ell_{\mathrm{CE}}(p, y) = \sum_{i=1}^K I(y=i) \ln 1/p_i = \ln 1/p_y$

```text
loss = mean(-log(predicted_probability_of_true_label))
```

**What it does:** This loss takes the network's predicted class probabilities p and the observed label y, then returns the negative log-probability assigned to the true class. The indicator I(y=i) selects the term for the observed class from the K possible classes, so the sum reduces to ln(1/p_y); training minimizes the average of this quantity over the labeled set.

**Why it's novel:** This is standard cross-entropy training, not a BADGE-specific novelty. It supplies the usual neural-network objective whose predicted probabilities and gradients are later used by the active-learning method.

**The intuition:** The network is penalized when it gives the correct class a small probability: assigning probability 0.9 to the true class gives a small loss, while assigning 0.1 gives a much larger one. Minimizing the average loss therefore trains the model to put more probability on the labels already known, providing the predictions used to judge unlabeled examples.

### <a id="eq-softmax-cross-entropy"></a>Softmax cross-entropy in output weights

*Section 3 · role: implement*

**The paper states:**

$$\ell_{\text{CE}}(f(x;\theta), y) = \ln \left( \sum_{j=1}^{K} e^{W_j \cdot z(x;V)} \right) - W_y \cdot z(x;V).$$

```text
loss = log(sum_j exp(W_j dot z)) - W_y dot z
```

**What it does:** For an input x with observed class y, this computes the softmax cross-entropy loss from the output-layer weights W_j and the penultimate representation z(x;V). The sum runs over the K classes: the log-sum-exp term accounts for all class logits formed by the dot product of W_j with z(x;V), and subtracting the true-class logit formed by W_y with z(x;V) gives a scalar measuring how poorly the model scores the observed label.

**Why it's novel:** This is standard softmax cross-entropy, not the novel part of BADGE; it supplies the loss whose gradient is used to form the hallucinated gradient embedding. The paper's contribution is the later use of those embeddings to select batches that combine uncertainty and diversity without a hand-tuned trade-off parameter, rather than a change to this loss itself.

**The intuition:** The model assigns one logit to each class using the representation z and the corresponding output weights W_j. The log-sum-exp term normalizes these competing scores, while the subtraction rewards a high score for class y, so the loss is small when the observed class dominates and large when competing classes collectively outrank it. BADGE then uses how this loss would change with the model parameters as an uncertainty-and-representation signal for batch selection.

### <a id="eq-softmax-definition"></a>Softmax activation

*Section 3 · role: implement*

<!-- derived-block: begin {"kind": "implemented_by", "spec": {"element_id": "eq-softmax-definition"}} -->
**Implemented by:** [`compute_gradient_embeddings`](method/method.py) (`method/method.py:21`) · demonstrated in [the notebook](notebook.ipynb#sec-42-hallucinated-gradient-embeddings)
<!-- derived-block: end -->

**The paper states:**

$\sigma(z)_i = {e^{z_i}}/{\sum_{j=1}^K e^{z_j}}$

```text
p_i = exp(z_i) / sum_j exp(z_j)
```

**What it does:** For an input logit vector z, where z_i is the score for class i, this equation produces a probability p_i for each of the K classes. It exponentiates each class score and normalizes by the sum of all exponentiated scores, so the resulting probabilities sum to one. In BADGE, these class probabilities provide the predictive quantities used when representing an unlabeled point in the hallucinated gradient space.

**Why it's novel:** This is standard softmax machinery, not a novel contribution of BADGE. Its role is to turn the network's output scores into probabilities that can support BADGE's combination of predictive uncertainty and diversity in gradient embeddings.

**The intuition:** Exponentiation makes larger logits receive more weight, while division by the total converts those weights into a competition among the K classes. Thus a point with one dominant class probability is confident, whereas a point with probabilities spread across classes is uncertain; BADGE can use that predictive information together with representation geometry when selecting a batch.

### <a id="eq-softmax-model"></a>Softmax output model

*Section 3 · role: implement*

<!-- derived-block: begin {"kind": "implemented_by", "spec": {"element_id": "eq-softmax-model"}} -->
**Implemented by:** [`compute_gradient_embeddings`](method/method.py) (`method/method.py:21`) · demonstrated in [the notebook](notebook.ipynb#sec-42-hallucinated-gradient-embeddings)
<!-- derived-block: end -->

**The paper states:**

$f(x;\theta) = \sigma(W \cdot z(x;V))$

```text
probabilities = softmax(W dot z(x, V))
```

**What it does:** For an input x, the network uses the penultimate representation z(x; V), formed using parameters V, and combines it with output-layer weights W. The softmax function converts the resulting class scores into a vector of class probabilities; theta denotes the model parameters as a whole. This is the predictive model whose outputs are used when deciding which unlabeled examples to query.

**Why it's novel:** This is standard multiclass neural-network output machinery, not the novel part of BADGE. Its role is to provide the class probabilities and the representation needed to construct the later hallucinated gradient embedding; the paper's distinctive contribution is selecting batches using those embeddings to combine uncertainty and diversity without a hand-tuned trade-off parameter.

**The intuition:** The network first summarizes the input in z(x; V), then W turns that summary into one score per class, and softmax normalizes the scores into competing probabilities. Thus an input that the model finds ambiguous produces a less concentrated probability vector, while a confident prediction produces one dominant class probability. BADGE can use this uncertainty together with the representation when choosing informative, non-redundant points.

### <a id="eq-gradient-block"></a>Gradient embedding block formula

*Section 3, Equation 1 · role: implement*

<!-- derived-block: begin {"kind": "implemented_by", "spec": {"element_id": "eq-gradient-block"}} -->
**Implemented by:** [`compute_gradient_embeddings`](method/method.py) (`method/method.py:21`) · also used in `GradientEmbeddingClassifier` (`method/model.py:28`) · demonstrated in [the notebook](notebook.ipynb#sec-42-hallucinated-gradient-embeddings)
<!-- derived-block: end -->

**The paper states:**

$$(g_x)_i = \frac{\partial}{\partial W_i} \ell_{CE}(f(x;\theta), \hat{y}) = (p_i - I(\hat{y} = i))z(x; V). \tag{1}$$

```text
gradient_block_i = (p_i - indicator(yhat == i)) * z
```

**What it does:** For class i, this computes one block of the example x's hallucinated gradient embedding. p_i is the network's probability for class i, I(ŷ = i) is one when the provisional label ŷ equals i and zero otherwise, and z(x; V) is the penultimate-layer representation of x. The residual p_i - I(ŷ = i) scales that representation, giving the derivative of the cross-entropy loss with respect to the class-specific weights W_i.

**Why it's novel:** The derivative itself is standard cross-entropy gradient machinery. The paper's methodological change is to use these class-wise gradient blocks, evaluated with a hallucinated label ŷ, as the representation on which BADGE seeks points that are both high-magnitude and disparate; this is the mechanism that puts predictive uncertainty and diversity into the same selection space without a hand-tuned trade-off parameter.

**The intuition:** Think of each class block as the feature vector z(x; V) multiplied by how much the current prediction disagrees with the provisional label for that class. A large residual makes the example's gradient embedding large, while the shared penultimate representation makes examples with similar content point in similar directions; selecting disparate, high-magnitude embeddings therefore favors informative uncertainty without simply filling a batch with near-duplicates.

## <a id="algorithm-walkthrough"></a>Algorithm walkthrough

### <a id="alg-badge"></a>BADGE batch active learning

*Algorithm 1*

<!-- derived-block: begin {"kind": "implemented_by", "spec": {"element_id": "alg-badge"}} -->
**Implemented by:** [`select_batch`](method/method.py) (`method/method.py:101`) · demonstrated in [the notebook](notebook.ipynb#sec-51-the-acquisition-loop)
<!-- derived-block: end -->

BADGE initializes a labeled set, repeatedly computes hallucinated last-layer gradient embeddings for the unlabeled pool, selects a batch with k-MEANS++ seeding, queries labels, retrains, and returns the final model.

```text
sample M labeled points; train theta; repeat T times: predict yhat for each unlabeled x; compute last-layer gradient embedding; select B embeddings with k-means++; query labels; add them; retrain from labeled data; return theta
```

### <a id="alg-kmeans-plus-plus"></a>k-MEANS++ seeding sampler

*Appendix A, Algorithm 2*

<!-- derived-block: begin {"kind": "implemented_by", "spec": {"element_id": "alg-kmeans-plus-plus"}} -->
**Implemented by:** [`kmeans_plus_plus_seeding`](method/method.py) (`method/method.py:57`) · demonstrated in [the notebook](notebook.ipynb#sec-43-diverse-batch-seeding)
<!-- derived-block: end -->

The sampler chooses the first center uniformly and then samples each subsequent center with probability proportional to squared distance from the nearest selected center.

```text
choose first center uniformly; for each next center compute nearest-center distance; sample point proportional to squared distance; add it; return k centers
```

### Where the method plugs in

The paper's contribution is packaged as `select_batch(model, x_unlabeled, batch_size, seed) -> List[int]` — Select a diverse uncertain batch using last-layer gradient embeddings and k-means++ seeding.

## Parameters and provenance

_**3 of 8 parameters** are taken directly from the paper. 2 use a runtime value that differs from the paper's stated value, with the paper value preserved for comparison. 3 were supplemented from field conventions where the paper does not specify them, and are the values to scrutinise most when judging how closely the code follows the paper._

| parameter | value | source | paper says |
|---|---|---|---|
| batch_size | 100 | paper | Experimental setup |
| hidden_dim | 256 | system_inferred |  |
| initial_labeled | 100 | paper | Experimental setup |
| learning_rate | 0.001 | system_inferred |  |
| max_epochs | 8 | system_inferred |  |
| num_rounds | 5 | system_default | paper value: 349 |
| pool_size | 12000 | system_default | paper value: full training set |
| train_until_accuracy | 0.99 | paper | Section 4 EXPERIMENTS |

## <a id="source-map"></a>Source map

This table is the completeness index and citation trail for the whole document: one row for every element the decomposition extracted from the paper — including the ones that did not get a full section above — with the paper location it came from and the role the pipeline assigned it. Use it to confirm nothing the decomposition found was silently dropped, and to jump from any element to where it lives in the paper.

<!-- derived-block: begin {"kind": "source_map", "spec": {}} -->
| element | type | paper location | role | implementation |
|---|---|---|---|---|
| <a id="concept-pool-active-learning"></a>Pool-based deep batch active learning | concept | Section 2 | implement | — |
| <a id="eq-classifier-argmax"></a>Neural classifier prediction | equation | Section 2 | implement | — |
| <a id="eq-cross-entropy"></a>Cross-entropy training objective | equation | Section 2 | implement | — |
| <a id="alg-badge"></a>BADGE batch active learning | algorithm | Algorithm 1 | implement | `select_batch` — `method/method.py:101` |
| <a id="concept-hallucinated-label"></a>Hallucinated-label uncertainty | concept | Section 3 | implement | `compute_gradient_embeddings` — `method/method.py:21` |
| <a id="eq-gradient-embedding"></a>Hallucinated last-layer gradient embedding | equation | Algorithm 1 and Section 3 | implement | `compute_gradient_embeddings` — `method/method.py:21` |
| <a id="concept-gradient-uncertainty"></a>Gradient magnitude as uncertainty | concept | Section 3 | demonstrate | `compute_gradient_embeddings` — `method/method.py:21` |
| <a id="concept-gradient-diversity"></a>Diversity in gradient space | concept | Section 3 | demonstrate | `compute_gradient_embeddings` — `method/method.py:21` |
| <a id="concept-quality-diversity-tradeoff"></a>Hyperparameter-free quality and diversity tradeoff | concept | Abstract and Section 3 | demonstrate | `compute_gradient_embeddings` — `method/method.py:21` |
| <a id="eq-softmax-model"></a>Softmax output model | equation | Section 3 | implement | `compute_gradient_embeddings` — `method/method.py:21` |
| <a id="eq-softmax-definition"></a>Softmax activation | equation | Section 3 | implement | `compute_gradient_embeddings` — `method/method.py:21` |
| <a id="eq-softmax-cross-entropy"></a>Softmax cross-entropy in output weights | equation | Section 3 | implement | — |
| <a id="eq-gradient-block"></a>Gradient embedding block formula | equation | Section 3, Equation 1 | implement | `compute_gradient_embeddings` — `method/method.py:21` |
| <a id="prop-gradient-lower-bound"></a>Hallucinated gradient is a lower bound | property | Section 3, Proposition 1 | demonstrate | — |
| <a id="eq-true-label-gradient-norm"></a>True-label gradient norm | equation | Section 3, Proposition 1 | theoretical | — |
| <a id="prop-predicted-label-minimizes-gradient"></a>Predicted label minimizes gradient norm | property | Section 3, Proposition 1 | theoretical | — |
| <a id="alg-kmeans-plus-plus"></a>k-MEANS++ seeding sampler | algorithm | Appendix A, Algorithm 2 | implement | `kmeans_plus_plus_seeding` — `method/method.py:57` |
| <a id="prop-kmeans-diversity"></a>k-MEANS++ expected diversity guarantee | property | Appendix A | theoretical | — |
| <a id="exp-badge-robustness"></a>BADGE robustness across settings | experiment | Section 4 | implement | — |
| <a id="exp-gradient-space-diagnostics"></a>Gradient-space uncertainty and diversity diagnostics | experiment | Section 4 and Appendix F | implement | — |
| <a id="exp-kmeans-runtime"></a>k-MEANS++ sampling efficiency | experiment | Section 3 and Appendix G | demonstrate | — |
| <a id="hyp-initial-label-count"></a>Initial labeled examples M | hyperparameter | Algorithm 1 and Section 4 | implement | — |
| <a id="hyp-batch-size"></a>Acquisition batch size B | hyperparameter | Algorithm 1 and Section 4 | implement | — |
| <a id="hyp-training-optimizer"></a>Cross-entropy and Adam training protocol | hyperparameter | Section 4 | implement | — |
| <a id="hyp-learning-rates"></a>Learning rates | hyperparameter | Section 4 | implement | — |
| <a id="hyp-repetitions"></a>Independent experiment repetitions | hyperparameter | Section 4 | implement | — |
| <a id="prop-robustness"></a>Empirical robustness claim | property | Section 6 | demonstrate | — |
<!-- derived-block: end -->
