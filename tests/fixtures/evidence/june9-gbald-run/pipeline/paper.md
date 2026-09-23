# Bayesian Active Learning by Disagreements: A Geometric Perspective

Xiaofeng Cao and Ivor W. Tsang

**Abstract**—We present geometric Bayesian active learning by disagreements (GBALD), a framework that performs BALD on its core-set construction interacting with model uncertainty estimation. Technically, GBALD constructs core-set on ellipsoid, not typical sphere, preventing low-representative elements from spherical boundaries. The improvements are twofold: 1) relieve uninformative prior and 2) reduce redundant estimations. Theoretically, geodesic search with ellipsoid can derive tighter lower bound on error and easier to achieve zero error than with sphere. Experiments show that GBALD has slight perturbations to noisy and repeated samples, and outperforms BALD, BatchBALD and other existing deep active learning approaches.

F

**Index Terms**—Geometric Bayesian, deep active learning, core-set, model uncertainty, ellipsoid.

## **1 INTRODUCTION**

Deep neural networks (DNNs) lack the ability of learning from limited (insufficient) labels, which degenerates its generalizations to new tasks. Recently, leveraging the abundance of unlabeled data has become a potential solution to relieve this bottleneck whereby the expert knowledge is involved to perform annotations. In such setting, the deep learning researchers introduced the active learning (AL) [1], which solicit experts' annotations from those informative or representative unlabeled data, by maximizing the model uncertainty [2], [3] of the current learning model. During this AL process, the learning model tries to achieve a desired accuracy performance using the minimal data labeling. Recent shift of model uncertainty in many fields shows that deep Bayesian AL [4], [5] contributes the Bayesian neural networks training [6], Monte-Carlo (MC) dropout [7], and Bayesian core-set construction [8], etc.

Bayesian AL [9], [10] presents an expressive probabilistic interpretation on the model uncertainty estimation [7]. Theoretically, for a simple regression model such as linear, logistic, and probit, AL can derive their closed-forms on updating one sparse subset, which maximally reduces the uncertainty of posteriors over regression parameters [4]. However, for a DNN model, optimizing massive training parameters is not easily tractable. It is thus that the Bayesian approximation provides alternatives, including the importance sampling [11] and the Frank-Wolfe optimization [12]. With importance sampling, a typical Bayesian AL approach can be expressed as to maximizing the information gain in terms of predictive entropy over the model, and it is called Bayesian active learning by disagreements (BALD) [13].

BALD has two interpretations: model uncertainty estimation and core-set construction. To estimate the uncertainty of a model, a greedy strategy is usually applied. The criterion is to select those data that maximize the parameter disagreements between the current training model and its following updates as [1].

● X. Cao and I. W. Tsang are with the Australian Artificial Intelligence Council under Grant DP180100106 and DP200101328.

Institute, University of Technology Sydney, NSW 2008, Australia. E-mail: ivor.tsang@uts.edu.au. This work was supported by Australian Research

Image /page/0/Figure/12 description: A schematic flow diagram illustrating an active learning pipeline comparing BALD and GBALD. At the top center is a green rounded rectangle labeled “BALD.” Two black arrows descend from it: one to the left leading to a block titled “Model uncertainty estimation” and one to the right leading to a block titled “Core-set construction.” The left block is drawn as a stylized fully connected neural network icon (three vertical columns of nodes with interconnecting lines). Between that neural-net icon and the center is a small 3D-looking probability density / mountain plot with a red dashed horizontal line and a red arrow labeled “Input” pointing left; the dashed line area is annotated “Stage ①.” A black arrow runs from the neural-net block down toward a red rounded rectangle labeled “GBALD” at the bottom center; that downward connection is annotated “Stage ②.” From the right “Core-set construction” area there are two small scatter-cloud illustrations (blue points) with a few red-highlighted selected points interspersed. To the right of those clouds are two large purple shapes stacked vertically: top is a circular “Sphere” with a red X to indicate rejection; bottom is an elongated “Ellipsoid” with a red check mark to indicate acceptance. Black arrows form a loop from the bottom red “GBALD” box back up to the top green “BALD” box, illustrating a closed cycle. Overall color cues: BALD in green, GBALD in red, selected points in red, candidate points in blue, and the final shapes in purple. Text labels visible in the figure include: BALD, GBALD, Model uncertainty estimation, Core-set construction, Input, Stage ①, Stage ②, Sphere, and Ellipsoid.

Figure 1: Illustration of two-stage GBALD framework, which integrates model uncertainty estimation and core-set construction into a uniform framework. Stage 1 : core-set construction is with ellipsoid, not typical sphere, representing the original distribution to initialize the input features of DNNs. Stage 2 : model uncertainty estimation with those initial acquisitions then explores highly informative and representative samples for DNN.

However, naively interacting with BALD using an uninformative prior [14], [15] leads to unstable biased acquisitions [16], for example, insufficient or unbalanced prior labels. Under this setting, an uninformative prior can be constructed to reflect a balanced state among different Bayesian outcomes, if there is no available information. Moreover, the similarity or consistency of those acquisitions to their previous, brings some redundant information to the model, and may decelerate the subsequent training.

Core-set construction [17] avoids the greedy interaction to learning model via capturing the characteristics of data distributions. By approximating complete data posterior over model parameters, optimization of BALD can be deemed as a core-set construction process on a sphere [5], which seamlessly solicits a compact subset to draw the input data distribution, and efficiently mitigates the sensitivity to uninformative prior and redundant information.

From a geometric perspective, updates of core-set construction are usually optimized with spherical geodesic as [18], [19]. Once the core-set is obtained, deep AL algorithm immediately seeks annotations from knowledgeable experts and starts the training. However, the data located at the boundary regions of distributions, usually with a uniform manner, could not be highly-representative elements of core-set. Therefore, constructing core-set on sphere may not be the optimal choice for deep AL.

This paper presents a novel AL framework, namely Geometric BALD (GBALD), over the geometric interpretation of BALD that, interpreting BALD with the core-set construction on ellipsoid, initializes effective representations to estimate the model uncertainty. The goal is to seek for significant accuracy improvements against an uninformative prior and the redundant information. Figure 1 describes this two-stage framework. In the first stage, geometric core-set construction on ellipsoid [20] initializes a set of effective acquisitions to start a DNN model regardless of an uninformative prior. Taking the core-set as the inputs, the next stage ranks the batch acquisitions of model uncertainty according to their geometric representativeness, and then solicits some highly-representative examples from the batch. With representation constraints, the ranked acquisitions reduce the probability of sampling those nearby samples of the previous acquisitions, preventing redundant information. In order to explore these improvements, following the typical approximately linear perceptron analysis [21], our generalization analysis shows that, the lower bound of generalization errors of geodesic search with ellipsoid, is proven to be tighter than that of geodesic search with sphere. Achieving a nearly zero error by geodesic search with ellipsoid, is also proven to have a higher probability than that of

Contributions of this paper can be summarized from the geometric, algorithmic, and theoretical perspectives.

- Geometrically, our key technical innovation is to construct core-set on ellipsoid, not typical sphere, preventing lowrepresentative elements from boundary distribution.
- In term of the algorithm design, our work proposes a
  two-stage framework from a Bayesian perspective that sequentially introduces the core-set representation and model\nuncertainty estimation, strengthening their performance
  "independently". Moreover, different to the typical BALD
  optimizations, we present geometric solvers to construct
  a core-set and estimate model uncertainty using it, which
  result in a different perspective for Bayesian AL.
- Theoretically, to guarantee those improvements, our generalization analysis proves that, compared to the typical Bayesian spherical interpretation, the geodesic search with ellipsoid can derive a tighter lower error bound and achieve a higher probability to obtain a nearly zero error.

The rest of this paper is organized as follows. In Section 2, we first review the related work. Secondly, we elaborate BALD and GBALD in Sections 3 and 4, respectively. Experimental results are presented in Section 5. Finally, we conclude this paper in Section 6.

## 2 RELATED WORK

AL. The early probability support vector machine (SVM) proposed the concept of AL [22], [23] that acquires the data with minimum margin to effectively update the support vectors. Many AL acquisition algorithms were then proposed to relieve the training bottleneck of SVM, which results in unsatisfied predictions, for example, uncertainty sampling [24], margin sampling [25], MC estimation of error reduction [26], transductive experimental design [27], etc. Given a learning model without sufficient training labels, those effective acquisitions reduce the expensive cost of human

annotations in many scenarios, e.g. multiple correct outputs [10], cost-sensitive classification [28], adversarial training [29], etc. In theory, the researchers studied the label complexity bound [30] (label demand before achieving a desired error threshold) and its noisy performance [31] of an AL algorithm.

Model uncertainty. In deep learning setting, AL was introduced to improve the training of DNNs by annotating a batch of unlabeled data, where the data which maximize the model uncertainty [3] are the primary acquisitions. For example, in ensemble deep learning [2], out-of-domain uncertainty estimation selects those data, which do not follow the same distribution as the input training data; in-domain uncertainty draws the data from the original input distribution, producing reliable probability estimates. Gal *et al.* [7] use MC dropout to estimate the predictive uncertainty via approximating a Bayesian convolutional neural network. Lakshminarayanan *et al.* [3] evaluate the uncertainty of unlabeled data using a proper scoring rule, deriving the sampling criteria of AL to fed the DNNs.

Bayesian AL. Taking a Bayesian perspective [9], AL can be deemed as minimizing the Bayesian posterior risk with multiple label acquisitions over the input unlabeled data. A potential informative approach is to reduce the uncertainty about model parameters using Shannon's entropy [32]. This can be interpreted as seeking the acquisitions for which the Bayesian parameters under the posterior disagree about the outcome the most, so this acquisition algorithm is referred to as Bayesian active learning by disagreement (BALD) [13]. In applications, BALD was introduced into natural language processing [33], text classification [34], decision making [31], data augmentation [35], etc.

**Deep AL.** Recently, Gal *et al.* [1] proposed to cooperate BALD with DNNs to improve its acquisition performance. The unlabeled data which maximize the model uncertainty of DNNs provide positive feedback. However, it needs to repeatedly update the model until the acquisition budget is exhausted. To improve the acquisition efficiency, batch sampling with BALD is applied as [5], [4]. In BatchBALD, Kirsch et al. [5] developed a tractable approximation to the mutual information of one batch of unlabeled data and the current model parameters. However, those uncertainty evaluations of Bayesian AL whether in single or batch acquisitions all take a greedy strategy, which leads to computationally infeasible, or excursive parameter estimations. Pinsler et al. [4] thus approximated the posterior over the model parameters by a sparse subset, i.e. a core-set. Applying the Frank-Wolfe optimization [12], the batch acquisitions of a large-scale dataset can be efficiently derived, thereby interpreting closed-form solutions for the coreset construction on a linear or probit regression function. As a consequence, the non-deep models obtained the theoretical guarantees from this optimization solver due to their tractable parameters. However, for deep AL, being short of interactions to DNNs is not able to maximally drive their model performance.

## 3 BALD

BALD has two different interpretations: model uncertainty estima-tion and core-set construction. We simply introduce them in this section.

### 3.1 Model uncertainty estimation

We consider a discriminative model  $p(y|x, \theta)$  parameterized by  $\theta$  that maps  $x \in \mathcal{X}$  into an output distribution over a set of  $y \in \mathcal{Y}$ . Given an initial labeled (training) set  $\mathcal{D}_0 \in \mathcal{X} \times \mathcal{Y}$ , the Bayesian

inference over this parameterized model is to estimate the posterior  $p(\theta|\mathcal{D}_0)$ , i.e. estimate  $\theta$  by repeatedly updating  $\mathcal{D}_0$ . AL adopts this setting from a Bayesian perspective.

With AL, the learner can choose a set of unlabeled data from  $\mathcal{D}_u = \{x_i\}_{j=1}^N \in \mathcal{X}$  via maximizing the uncertainty of the model parameters. Houlsby *et al.* 13 proposed a greedy strategy termed BALD to update  $\mathcal{D}_0$  by estimating a desired data  $x^*$  that maximizes the decrease in expected posterior entropy:

$$x^* = \arg \max_{x \in \mathcal{D}_n} H[\theta|\mathcal{D}_0] - \mathbb{E}_{y \sim p(y|x,\mathcal{D}_0)} [H[\theta|x,y,\mathcal{D}_0]], \quad (1)$$

where the labeled and unlabeled sets are updated by  $\mathcal{D}_0 = \mathcal{D}_0 \cup \{x^*, y^*\}, \mathcal{D}_u = \mathcal{D}_u \backslash x^*$ , and  $y^*$  denotes the output of  $x^*$ . In deep AL,  $y^*$  can be annotated as a label from experts and  $\theta$  yields a DNN model.

### 3.2 Core-set construction

Let  $p(\theta|\mathcal{D}_0)$  be updated by its log posterior  $\log p(\theta|\mathcal{D}_0, x^*)$ ,  $y^* \in \{y_i\}_{i=1}^N$ , assume the outputs are conditional independent of the inputs, i.e.  $p(y^*|x^*, D_0) = \int_{\theta} p(y^*|x^*, \theta) p(\theta|D_0) d\theta$ , then we have the *complete data log posterior following*  $\boxed{4}$ :

$$\mathbb{E}_{y^{*}}\big[\log p(\theta\mid\mathcal{D}_0,\,x^{*},\,y^{*})\big]$$

$$= \mathbb{E}_{y^{*}}\big[\log p(\theta\mid\mathcal{D}_0) + \log p(y^{*}\mid x^{*},\theta) - \log p(y^{*}\mid x^{*},\mathcal{D}_0)\big]$$

$$= \log p(\theta\mid\mathcal{D}_0) + \mathbb{E}_{y^{*}}\big[\log p(y^{*}\mid x^{*},\theta) + H\big[y^{*}\mid x^{*},\mathcal{D}_0\big]\big]$$

$$= \log p(\theta\mid\mathcal{D}_0) + \sum_{i=1}^{N}\Big(\mathbb{E}_{y_{i}}\big[\log p(y_{i}\mid x_{i},\theta) + H\big[y_{i}\mid x_{i},\mathcal{D}_0\big]\big]\Big).$$
 $(2)$ 

The key idea of the core-set construction is to approximate the log posterior of Eq. (2) by a subset of  $D'_u \subseteq D_u$  such that:  $\mathbb{E}_{\mathcal{Y}_u}[\log p(\theta|\mathcal{D}_0,\mathcal{D}_u,\mathcal{Y}_u)] \approx \mathbb{E}_{\mathcal{Y}_u'}[\log p(\theta|\mathcal{D}_0,\mathcal{D}_u',\mathcal{Y}_u')],$  where  $\mathcal{Y}_u$  and  $\mathcal{Y}_u'$  denote the predictive labels of  $\mathcal{D}_u$  and  $\mathcal{D}_u'$  respectively by the Bayesian discriminative model, that is,  $p(\mathcal{Y}_u|\mathcal{D}_u,D_0)=\int_{\theta}p(\mathcal{Y}_u'|\mathcal{D}_u,\theta)p(\theta|D_0)\mathrm{d}\theta,$  and  $p(\mathcal{Y}_u'|\mathcal{D}_u',D_0)=\int_{\theta}p(\mathcal{Y}_u'|\mathcal{D}_u',\theta)p(\theta|D_0)\mathrm{d}\theta.$  Here  $D'_u$  can be indicated by a core-set  $\Phi$  that highly represents  $\Phi$ . The optimization tricks such as the Frank-Wolfe optimization  $\Phi$  then can be adopted to solve this problem.

**Motivations.** Eqs. (1) and (2) provide the Bayesian rules of BALD over model uncertainty and core-set construction respectively, which further attract the attention of deep learning researchers. However, the two interpretations of BALD are limited by: 1) the redundant information and 2) an uninformative prior, where one major reason which causes these two issues is the poor initialization on the prior, i.e.  $p(\mathcal{D}_0|\theta)$ . For example, an unbalanced label initialization on  $\mathcal{D}_0$  usually leads to an uninformative prior, which further conducts the acquisitions of AL to select those unlabeled data from one or some fixed classes; highly-biased results 16 with redundant information are inevitable. Therefore, these two limitations affect each other.

## 4 GBALD

GBALD consists of two components: 1) initial acquisitions based on core-set construction and 2) model uncertainty estimation with those initial acquisitions.

Image /page/2/Picture/13 description: A stylized blue wireframe sphere (latitude/longitude grid) showing a geodesic path on its surface. The sphere has blue meridian and parallel grid lines. Several small solid black circular markers are placed on the surface: one at the north pole, one on the left mid-latitude, one on the lower hemisphere, and one just above the red marker on the right. A red star-shaped marker appears on the right-hand mid-latitude of the sphere. Black dashed curved lines on the sphere trace a path: a dashed arc runs from the left black marker across the front to the red star and then continues as a dashed arc down to the lower black marker; a second dashed/curved arrowed segment connects the north-pole marker down toward the black marker near the red star. Small black arrowheads on the dashed curves indicate direction along the geodesic. The figure is labeled beneath as “(a) Sphere geodesic.”

Image /page/2/Picture/14 description: A schematic 3‑D ellipsoid (globe) drawn as a blue wireframe of latitude and longitude lines. A dashed black curved path with arrowheads traces a trajectory across the surface: it begins at a black oval marker on the left equatorial region, curves down to a second black oval in the lower (southern) hemisphere, then rises to a small red circular marker on the right-hand mid-latitude, and continues upward to a final black oval near the top of the ellipsoid. The dashed route and arrowheads indicate direction of travel; most surface markers are solid black ovals except the single red point.

esic (b) Ellipsoid geodesic

Figure 2: Optimizing BALD with sphere and ellipsoid geodesics. Ellipsoid geodesic rescales the sphere geodesic to prevent the updates of core-set towards the boundary regions of the sphere where characteristics of spherical distribution cannot be properly captured. Note that the black points denote the feasible updates of the red points and the dash lines denote the geodesics.

### 4.1 Geometric interpretation of core-set

Modeling the complete data posterior over parameter distribution can relieve the two limitations of BALD, which has been stated at the end of Section 3.2. Typically, optimizing the acquisitions of Bayesian AL is equivalent to approximating a core-set centered with the spherical embeddings [S]. Let  $w_i$  be the sampling weight of  $x_i$ ,  $||w_i||_0 \le N$ , the core-set construction is to optimize:

$$\min_{w} \left\| \underbrace{\sum_{i=1}^{N} \mathbb{E}_{y_{i}} \left[ \log p(y_{i}|x_{i}, \theta) + H[y_{i}|x_{i}, \mathcal{D}_{0}] \right]}_{\mathcal{L}} - \underbrace{\sum_{i=1}^{N} w_{i} \mathbb{E}_{y_{i}} \left[ \log p(y_{i}|x_{i}, \theta) + H[y_{i}|x_{i}, \mathcal{D}_{0}] \right]}_{\mathcal{L}(w)} \right\|^{2}, \tag{3}$$

where  $\mathcal{L}$  and  $\mathcal{L}(w)$  denote the full and expected (weighted) log-likelihoods, respectively [17], [36]. Specifically,  $\sum_{i=1}^{N} H[y_i|x_i, \mathcal{D}_0] = -\sum_{y_i} p(y_i|x_i, \mathcal{D}_0) \log(p(y_i|x_i, \mathcal{D}_0))$ , where  $p(y_i|x_i, \mathcal{D}_0) = \int_{\theta} p(y_i|x_i, \theta) p(\theta|\mathcal{D}_0) d\theta$ . Note  $\|\cdot\|$  denotes the  $\ell^2$  norm.

The approximation of Eq. (3) implicitly requires that the complete data log posterior of Eq. (2) w.r.t.  $\mathcal L$  must be close to an expected posterior w.r.t.  $\mathcal{L}(w)$  such that approximating a sparse subset for the original inputs by sphere geodesic search is feasible (see Figure 2(a)). Generally, solving this optimization is intractable due to the cardinality constraint [4]. Campbell et al. [36] proposed to relax the constraint in Frank–Wolfe optimization, in which mapping  $\mathcal{X}$  is usually performed in a Hilbert space (HS) with a bounded inner product operation. In this solution, the sphere embedded in the HS replaces the cardinality constraint with a polynomial constraint. However, the initialization on  $\mathcal{D}_0$  affects the iterative approximation to  $\mathcal{D}_{u}$  at the beginning of the geodesic search. Moreover, the posterior of  $p(\theta|\mathcal{D}_0)$  is uninformative, if the initialized  $\mathcal{D}_0$  is empty or not correct. Therefore, the typical Bayesian core-set construction of BALD cannot ideally fit an uninformative prior. The another geometric interpretation of coreset construction, such as k-centers [8], is not restricted to this setting. We thus follow the construction of k-centers to find the core-set.

k-centers. Sener  $et\ al.\ [8]$  proposed a core-set representation approach for deep AL based on k-centers. This approach can be adopted in the core-set construction of BALD without the help of a

discriminative (training) model. Therefore, the uninformative prior has no further influence to the core-set. Typically, the k-centers approach uses a greedy strategy to search the data  $\widetilde{x}$  whose nearest distance to the elements of  $\mathcal{D}_0$  is the maximal:

$$\widetilde{x} = \underset{x_i \in \mathcal{D}_u}{\arg \max} \min_{c_i \in \mathcal{D}_0} \|x_i - c_i\|, \tag{4}$$

then  $\mathcal{D}_0$  is updated by  $\mathcal{D}_0 \cup \{\widetilde{x}, \widetilde{y}\}, \mathcal{D}_u$  is updated by  $\mathcal{D}_u \setminus \widetilde{x}$ , where  $\widetilde{y}$  denotes the output of  $\widetilde{x}$ . This max-min operation usually performs k times to construct the centers.

From a geometric perspective, the k-centers can be deemed as the core-set construction via the spherical geodesic search as [37], [38]. Specifically, the max-min optimization guides  $\mathcal{D}_0$  to be updated into one data which draws the longest geodesic from  $x_i, \forall i$  across the sphere center. The iterative update on  $\widetilde{x}$  is then along its unique diameter through the sphere center. However, this greedy optimization has a large probability that leads the core-set to fall into the boundary regions of the sphere, which is not able to capture the characteristics of the distributions.

### 4.2 Initial acquisitions based on core-set construction

We present a novel greedy search which rescales the geodesic of a sphere into an ellipsoid following Eq. (4), in which the iterative update on the geodesic search is rescaled (see Figure 2(b)). We follow the importance sampling strategy to begin the search.

Initial prior on geometry. Initializing  $p(\mathcal{D}_0|\theta)$  is performed with a group of internal spheres centered with  $D_j$ ,  $\forall j$ , subjected to  $D_j \in \mathcal{D}_0$ , in which the geodesic between  $\mathcal{D}_0$  and the unlabeled data is over those spheres. Since  $\mathcal{D}_0$  is known, the specification of  $\theta$  then plays a key role on initializing  $p(\mathcal{D}_0|\theta)$ . Given a radius  $R_0$  for any observed internal sphere,  $p(y_i|x_i,\theta)$  is firstly defined by

$$p(y_i|x_i,\theta) = \begin{cases} 1, & \exists j, ||x_i - D_j|| \le R_0, \\ \max\left\{\frac{R_0}{\|x_i - D_j\|}\right\}, & \forall j, ||x_i - D_i|| > R_0, \end{cases}$$
(5)

thereby  $\theta$  yields the parameter  $R_0$ . When the data is enclosed with a ball, the probability of Eq. (5) is 1. The data near the ball, is given a probability of  $\max\left\{\frac{R_0}{\|x_i-D_j\|}\right\}$  constrained by  $\min\|x_i-D_j\|, \, \forall j$ , i.e. the probability is assigned by the nearest ball to  $x_i$ , which is centered with  $D_j$ . From Eq. (3), the information entropy of  $y_i \sim \{y_1, y_2, ..., y_N\}$  over  $x_i \sim \{x_1, x_2, ..., x_N\}$  can be expressed as the integral regarding  $p(y_i|x_i, \theta)$ :

$$\sum_{i=1}^{N} H\left(y_i \mid x_i, \mathcal{D}_0\right)$$

$$= -\sum_{i=1}^{N} \int_{\theta} p\left(y_i\mid x_i,\theta\right)\, p\left(\theta\mid \mathcal{D}_0\right)\, d\theta \; \log\Big(\int_{\theta} p\left(y_i\mid x_i,\theta\right)\, p\left(\theta\mid \mathcal{D}_0\right)\, d\theta\Big)$$
(6)

which can be approximated by  $-\sum_{i=1}^{N} p(y_i|x_i,\theta) \log \left(p(y_i|x_i,\theta)\right)$  following the details of Eq. (3). In short, this indicates an approximation to the entropy over the entire outputs on  $\mathcal{D}_u$  that assumes the prior  $p(D_0|\theta)$  w.r.t.  $p(y_i|x_i,\theta)$  is already known from Eq. (5).

**Max-min optimization.** Recalling the max-min optimization trick of k-centers in the core-set construction of [8], the minimizer of Eq. (3) then can be divided into two parts:  $\min_{x^*} \mathcal{L}$  and  $\max_w \mathcal{L}(w)$ , where  $\mathcal{D}_0$  is updated by acquiring  $x^*$ . However, the updates of  $\mathcal{D}_0$  decide the minimizer of  $\mathcal{L}$  with regard to the internal spheres centered with  $D_i$ ,  $\forall i$ . Therefore, minimizing

 $\mathcal L$  should be constrained by an unbiased full likelihood over  $\mathcal X$  to alleviate the potential biases from the initialization of  $\mathcal D_0$ . Let  $\mathcal L_0$  denote the unbiased full likelihood over  $\mathcal X$  that particularly stipulates  $\mathcal D_0$  as the k-means centers written as  $\mathcal U$  of  $\mathcal X$  which jointly draw the input distribution. We define  $\mathcal L_0 = |\sum_{i=1}^N \mathbb E_{y_i}[\log p(y_i|x_i,\theta) + \mathrm H[y_i|x_i,\mathcal U]]|$  to regulate  $\mathcal L$ , that is

$$\min_{x^*} \| \mathcal{L}_0 - \mathcal{L} \|^2, \text{ s.t. } \mathcal{D}_0 = \mathcal{D}_0 \cup \{x^*, y^*\}, \mathcal{D}_u = \mathcal{D}_u \backslash x^*.$$
 (7)

The other sub optimizer is  $\max_{w} \mathcal{L}(w)$ . We present a greedy strategy following Eq. (1):

$$\max_{1 \le i \le N} \min_{w_i} \sum_{i=1}^{N} w_i \mathbb{E}_{y_i}\left[\log p(y_i\mid x_i,\theta) + H\left[y_i\mid x_i,\mathcal{D}_0\right]\right]$$

$$= \sum_{i=1}^{N} w_i \log p(y_i\mid x_i,\theta) - \sum_{i=1}^{N} w_i\, p(y_i\mid x_i,\theta)\, \log p(y_i\mid x_i,\theta),$$
(8)

which can be further written as:  $\sum_{i=1}^{N} w_i \log p(y_i|x_i,\theta)(1 - \log p(y_i|x_i,\theta))$ . Let  $w_i = 1, \forall i$  for unbiased estimation of the likelihood  $\mathcal{L}(w)$ , Eq. (8) can be simplified as

$$\max_{x_i \in \mathcal{D}_u} \min_{D_i \in \mathcal{D}_0} \log p(y_i | x_i, \theta), \tag{9}$$

where  $p(y_i|x_i,\theta)$  follows Eq. (5). Combining Eqs. (7) and (9), the optimization of Eq. (3) is then transformed as

$$x^* = \underset{x_j \in \mathcal{D}_u}{\operatorname{arg max}} \min_{D_j \in \mathcal{D}_0} \left\{ \|\mathcal{L}_0 - \mathcal{L}\|^2 + \log p(y_j | x_j, \theta) \right\}, \quad (10)$$

where  $\mathcal{D}_0$  is updated by acquiring  $x^*$ , i.e.  $\mathcal{D}_0 = \mathcal{D}_0 \cup \{x^*, y^*\}$ .

**Geodesic line.** For a metric geometry M, a geodesic line is a curve  $\gamma$  which projects its interval I to  $M\colon I\to M$ , maintaining everywhere locally a distance minimizer 39. Given a constant  $\nu>0$  such that for any  $a,b\in I$  there exists a geodesic distance  $d(\gamma(a),\gamma(b))\coloneqq\int_a^b\sqrt{g_{\gamma(t)}(\gamma'(t),\gamma'(t))}dt$ , where  $\gamma'(t)$  denotes the geodesic curvature, and g denotes the metric tensor over M. Here, we define  $\gamma'(t)=0$ , then  $g_{\gamma(t)}(0,0)=1$  such that  $d(\gamma(a),\gamma(b))$  can be generalized as a segment of a straight line:  $d(\gamma(a),\gamma(b))=\|a-b\|$ .

Ellipsoid geodesic distance. For any observation points  $p,q \in M$ , if the spherical geodesic distance is defined as  $d(\gamma(p),\gamma(q)) = \|p-q\|$ . The affine projection obtains its ellipsoid interpretation:  $d(\gamma(p),\gamma(q)) = \|\eta(p-q)\|$ , where  $\eta$  denotes the affine factor subjected to  $0 < \eta < 1$ .

Optimizing with ellipsoid geodesic search. The max-min optimization of Eq. (10) is performed on an ellipsoid geometry to prevent the updates of the core-set towards the boundary regions, where the ellipsoid geodesic line scales the original update on the sphere. Assume  $x_i$  is the previous acquisition and  $x^*$  is the next desired acquisition, the ellipsoid geodesic rescales the position of  $x^*$  as  $x^*_e = x_i + \eta(x^* - x_i)$ . Then, we update this position of  $x^*_e$  to its nearest neighbor  $x_j$  in the unlabeled data pool, i.e. arg  $\min_{x_j \in \mathcal{D}_u} \|x_j - x^*_e\|$ , also can be written as

$$\underset{x_j \in \mathcal{D}_u}{\arg \min} \left\| x_j - \left[ x_i + \eta (x^* - x_i) \right] \right\|. \tag{11}$$

To study the advantage of ellipsoid geodesic search, Section 6 presents our generalization analysis.

### 4.3 Model uncertainty estimation with core-set

GBALD starts the model uncertainty estimation with those initial core-set acquisitions, in which it introduces a ranking scheme to derive both informative and representative acquisitions.

**Single acquisition.** We follow **1** and use MC dropout to perform Bayesian inference on the neural network model. It then leads to ranking the informative acquisitions with batch sequences is with high efficiency. We first present the ranking criterion by rewriting Eq. (1) as the batch returns:

$$\{x_{1}^{*}, x_{2}^{*}, \ldots, x_{b}^{*}\} = \underset{\{\hat{x}_{1},\hat{x}_{2},\ldots,\hat{x}_{b}\} \in \mathcal{D}_{u}}{\arg\max}$$

$$\left\{ \mathrm{H}[\theta \mid \mathcal{D}_{0}] - \mathbb{E}_{\hat{y}_{1:b} \sim p(\hat{y}_{1:b} \mid \hat{x}_{1:b}, \mathcal{D}_{0})}\left[ \mathrm{H}[\theta \mid \hat{x}_{1:b},\hat{y}_{1:b},\mathcal{D}_{0}] \right] \right\},$$
(12)

where  $\hat{x}_{1:b} = \{\hat{x}_1, \hat{x}_2, ..., \hat{x}_b\}$ ,  $\hat{y}_{1:b} = \{\hat{y}_1, \hat{y}_2, ..., \hat{y}_b\}$ ,  $\hat{y}_i$  denotes the output of  $\hat{x}_i$ . The informative acquisition  $x_t^*$  is then selected from the ranked batch acquisitions  $\hat{x}_{1:b}$  due to the highest (most) representation for the unlabeled data:

$$x_{t}^{*} = \underset{x_{i}^{*} \in \{x_{1}^{*}, x_{2}^{*}, \dots, x_{b}^{*}\}}{\arg \max} \left\{ \underset{D_{j} \in \mathcal{D}_{0}}{\max} \ p(y_{i} | x_{i}^{*}, \theta) \coloneqq \frac{R_{0}}{\|x_{i}^{*} - D_{j}\|} \right\}, (13)$$

where t denotes the index of the final acquisition, subjected to  $1 \le t \le b$ . This also adopts the max-min optimization of k-centers in Eq. (4), i.e.  $x_t^* = \arg\max_{x_i^* \in \{x_1^*, x_2^*, \dots, x_b^*\}} \min_{D_j \in \mathcal{D}_0} \|x_i^* - D_j\|$ . Batch acquisitions. The greedy strategy of Eq. (13) can be

**Batch acquisitions.** The greedy strategy of Eq. (13) can be written as a batch of acquisitions by controlling its output as a batch set, i.e.

$$\{x_{t_1}^*, ..., x_{t_{b'}}^*\} = \underset{x_{t_1:t_{b'}}^* \subseteq \{x_1^*, x_2^*, ..., x_b^*\}}{\arg\max} p(y_{t_1:t_{b'}}^* | x_{t_1:t_{b'}}^*, \theta), \quad (14)$$

where  $x_{t_1:t_{b'}}^* = \{x_{t_1}^*,...,x_{t_{b'}}^*\}, \ y_{t_1:t_{b'}}^* = \{y_{t_1}^*,...,y_{t_{b'}}^*\}, \ y_{t_i}^*$  denotes the output of  $x_{t_i}^*, \ 1 \leq i \leq b', \ \text{and} \ 1 \leq b' \leq b.$  This setting can be used to accelerate the acquisitions of AL in a large dataset.

## 5 Two-stage GBALD Algorithm

The GBALD algorithm has two stages: 1) construct a core-set on ellipsoid (Lines 3 to 13), and 2) estimate model uncertainty with a deep learning model (Lines 14 to 21).

Algorithmically, core-set construction is derived from the maxmin optimization of Eq. (10), then updated with ellipsoid geodesic w.r.t. Eq. (11), where  $\theta$  yields a geometric probability model w.r.t. Eq. (5). Importing the core-set into  $\mathcal{D}_0$  derives the deep learning model to return b informative acquisitions one time, where  $\theta$  yields a deep learning model. Ranking those samples, we select b' samples with the highest representations as the batch outputs w.r.t. Eq. (14). The iterations of batch acquisitions stop until its budget is exhaust. The final update on  $\mathcal{D}_0$  is our acquisition set of AL.

## 6 GENERALIZATION ERRORS OF GEODESIC SEARCH WITH SPHERE AND ELLIPSOID

Optimizing with ellipsoid geodesic linearly rescales the spherical search, which draws core-set on a tighter geometric object. The inherent motivation is that, geodesic search with ellipsoid can prevent the redundant updates of core-set, avoiding those elements from spherical boundaries. Following the approximately perceptron analysis of [21], this section presents generalization error analysis from geometry, which provides feasible guarantees for geodesic search with ellipsoid. The proofs are presented in Appendix.

### 6.1 Assumptions of generalization analysis

Let  $\Pr[err(h,k)=0]_{\mathrm{Sphere}}$  and  $\Pr[err(h,k)=0]_{\mathrm{Ellipsoid}}$  be the probabilities of achieving a zero error by geodesic search with sphere and ellipsoid, respectively, we study their inequality relationship. The assumptions are inspired from  $\gamma$ -tube manifold, which characterizes the probability mass of decision boundaries.

Given  $S_A$  be the sphere that tightly covers class A where  $S_A$  is with a center  $c_a$  and radius  $R_a$ , the assumption on sphere is as follows.

**Assumption 1.** Ben-David et al. [40] proposed that the  $\gamma$ -tube manifold [41] can characterize the probability mass of an optimal version space-based hypothesis. From geometry, we here assume that the probability mass of achieving a zero error by geodesic search with sphere, is roughly defined as the volume ratio of the  $\gamma$ -tube and sphere, that is,  $\Pr[err(h,k) = 0]_{\text{Sphere}} := \frac{\text{Vol}(\text{Tube})}{\text{Vol}(\text{Sphere})}$ . Given  $\frac{\pi}{\omega} = \arcsin \frac{R_a - d_a}{R_s}$ , the assumption is formalized as

$$\Pr[\mathrm{err}(h,k)=0]_{\text{Sphere}} = 1 - \frac{t_k^{3}}{R_a^{3}},$$

where

$$t_k = \frac{R_a^{2}}{3} + \sqrt[3]{-\dfrac{\mu_k}{2\pi} + \sqrt{\dfrac{\mu_k^{2}}{4\pi^{2}} - \dfrac{\pi^{3}R_a^{3}}{27}}} \; + \; \sqrt[3]{-\dfrac{\mu_k}{2\pi} - \sqrt{\dfrac{\mu_k^{2}}{4\pi^{2}} - \dfrac{\pi^{3}R_a^{3}}{27}}},$$

and

$$\mu_k = \left(\dfrac{2k-4}{3k} - \dfrac{1}{\varphi}\cos\dfrac{\pi}{\varphi}\right)\pi R_a^{3} - \dfrac{4\pi R_b^{3}}{3k}.$$

Given class A is tightly covered by ellipsoid  $E_a$ , let  $R_{a_1}$  be the polar radius of  $E_a$ , and  $R_{a_2}$ ,  $R_{a_3}$  be the equatorial radii of  $E_a$ , the assumption on ellipsoid is as follows.

**Assumption 2.** Following Assumption I the probability mass of achieving a zero error by geodesic search with ellipsoid, can be assumed as the volume ratio of the  $\gamma$ -tube and ellipsoid, that is  $\Pr[err(h,k)=0]_{\text{Ellipsoid}} \coloneqq \frac{\operatorname{Vol}(\operatorname{Tube})}{\operatorname{Vol}(\operatorname{Ellipsoid})}$ . Then, the assumption is formalized as

The following equation:

$$\Pr\{\mathrm{err}(h,k)=0\}_{\mathrm{Ellipsoid}} = 1 - \frac{\lambda_{k_1}\,\lambda_{k_2}\,\lambda_{k_3}}{R_{a_1}\,R_{a_2}\,R_{a_3}}$$
where

$$\begin{aligned}
\lambda_{k_i} &= \frac{R_{a_i}^2}{3} \\
&\quad + \sqrt[3]{-\frac{\sigma_{k_i}}{2\pi} + \sqrt{\frac{\sigma_{k_i}^2}{4\pi^2} - \frac{R_{a_i}^3}{27}}} \\
&\quad + \sqrt[3]{-\frac{\sigma_{k_i}}{2\pi} - \sqrt{\frac{\sigma_{k_i}^2}{4\pi^2} - \frac{R_{a_i}^3}{27}}}, \\
\\[6pt]
\sigma_{k_i} &= \left(\frac{2k-4}{3k} - \frac{\pi^2 R_{a_i}}{2\varphi}\right)\pi\,R_{a_1}R_{a_2}R_{a_3} - \frac{4\pi\,R_{b_1}R_{b_2}R_{b_3}}{3k}, \qquad i=1,2,3.
\end{aligned}$$

The specifications of of Assumptions 1 and 2 are presented in Appendix A.2 and A.3, respectively. We next present our low-dimensional generalization analysis.

### 6.2 Low-dimensional generalizations

Our generalization analysis begins from low-dimensional (3-D) sphere/ellipsoid to high-dimensional hypersphere/hyperellipsoid, where the high-dimensional settings can be extended into infinite dimensions.

#### 6.2.1 Our settings

**Geodesic search with sphere.** With Assumptions 1 and 2, given a perceptron function  $h := w_1x_1 + w_2x_2 + w_3$ , the task is to classify the two classes A and B embedded in a 3-D space. Let  $S_A$  and  $S_B$  be the spheres that tightly cover A and B, respectively, where  $S_A$  is with a center  $c_a$  and radius  $R_a$ , and  $S_B$  is with a center  $c_b$  and radius  $R_b$ . Under this setting, our generalization analysis is presented as follows.

#### Algorithm 1: Two-stage GBALD Algorithm

```
1 Input: Data set \mathcal{X}, core-set size N_{\mathcal{M}}, batch returns b, batch output b', iteration budget \mathcal{A}.
 2 Initialization: \alpha \leftarrow 0, core-set \mathcal{M} \leftarrow \emptyset.
 3 Stage (1) begins:
 4 Initialize \theta to yield a geometric probability model w.r.t. Eq. (5).
 5 Perform k-means to initialize \mathcal{U} to \mathcal{D}_0.
 6 Core-set construction begins by acquiring x_i^*,
 7 for i \leftarrow 1, 2, ..., N_{\mathcal{M}} do
          x_i^* \leftarrow \arg\max_{x_i \in \mathcal{D}_u} \min_{D_i \in \mathcal{D}_0} \left\{ \left\| \mathcal{L}_0 - \mathcal{L} \right\|^2 + \log p(y_i | x_i, \theta) \right\}, \text{ where } \mathcal{L}_0 \leftarrow \left| \sum_{i=1}^N \mathbb{E}_{y_i} [\log p(y_i | x_i, \theta) + H[y_i | x_i, \mathcal{U}]] \right|.
          Ellipsoid geodesic line scales x_i^*: x_i^* \leftarrow \arg\min_{x_i \in \mathcal{D}_n} ||x_j - [x_i + \eta(x^* - x_i)]||
           Update x_i^* into core-set \mathcal{M}: \mathcal{M} \leftarrow x_i^* \cup \mathcal{M}.
10
          Update N \leftarrow N - 1.
11
12 end
    Import core-set to update \mathcal{D}_0: \mathcal{D}_0 \leftarrow \mathcal{M} \cup \mathcal{U}', where \mathcal{U}' updates each element of \mathcal{U} into their nearest samples in \mathcal{X}.
14 Stage (2) begins:
15 Initialize \theta to yield a deep learning model.
    while \alpha < A do
           Return b informative deep learning acquisitions in one budget:
17
             \{x_1^*, x_2^*, ..., x_b^*\} \leftarrow \arg\max_{x \in \mathcal{D}_n} H[\theta|\mathcal{D}_0] - \mathbb{E}_{y \sim p(y|x, \mathcal{D}_0)} |H[\theta|x, y, \mathcal{D}_0]|.
          Rank b' informative acquisitions with the highest geometric representativeness:
18
           \{x_{t_1}^*, ..., x_{t_{b'}}^*\} \leftarrow \arg\max_{x_i^* \in \{x_1^*, x_2^*, ..., x_b^*\}} p(y_i | x_i^*, \theta).
          Update \{x_{t_1}^*, ..., x_{t_{b'}}^*\} into \mathcal{D}_0 : \mathcal{D}_0 \leftarrow \mathcal{D}_0 \cup \{x_{t_1}^*, ..., x_{t_{b'}}^*\}
19
20
21 end
```

**Theorem 1.** With Assumptions 1 and 2, given a perceptron function  $h = w_1x_1 + w_2x_2 + w_3$  that classifies A and B, and a sampling budget k. By drawing core-set on  $S_A$  and  $S_B$ , the minimum distances to the boundaries of that core-set elements of  $S_A$  and  $S_B$ , are defined as  $d_a$  and  $d_b$ , respectively. Let err(h,k) be the classification error rate with respect to h and k, given  $\frac{\pi}{\varphi} = \arcsin \frac{R_a - d_a}{R_a}$ , we then have an inequality of error:

22 Output: final update on  $\mathcal{D}_0$ .

$$\min\left\{ \dfrac{4R_a^{3}-(2R_a+t_k)(R_a-t_k)^{2}}{4R_a^{3}+4R_b^{3}},\; \dfrac{4R_b^{3}-(2R_b+t_k')(R_b-t_k')^{2}}{4R_b^{3}+4R_a^{3}} \right\} < \mathrm{err}(h,k) < \dfrac{1}{k}$$
where

$$t_k = \dfrac{R_a^{2}}{3} + \sqrt[3]{-\dfrac{\mu_k}{2\pi} + \sqrt{\dfrac{\mu_k^{2}}{4\pi^{2}} - \dfrac{R_a^{3}}{27}}} + \sqrt[3]{-\dfrac{\mu_k}{2\pi} - \sqrt{\dfrac{\mu_k^{2}}{4\pi^{2}} - \dfrac{R_a^{3}}{27}}}$$

$$\mu_k = \left(\dfrac{2k-4}{3k} - \dfrac{1}{\varphi}\cos\dfrac{\pi}{\varphi}\right)\pi R_a^{3} - \dfrac{4\pi R_b^{3}}{3k}$$

$$t_k' = \dfrac{R_b^{2}}{3} + \sqrt[3]{-\dfrac{\mu_k'}{2\pi} + \sqrt{\dfrac{\mu_k'^{2}}{4\pi^{2}} - \dfrac{R_b^{3}}{27}}} + \sqrt[3]{-\dfrac{\mu_k'}{2\pi} - \sqrt{\dfrac{\mu_k'^{2}}{4\pi^{2}} - \dfrac{R_b^{3}}{27}}}$$
and

$$\mu_k' = \left(\dfrac{2k-4}{3k} - \dfrac{1}{\varphi}\cos\dfrac{\pi}{\varphi}\right)\pi R_b^{3} - \dfrac{4\pi R_a^{3}}{3k}$$

Geodesic search with ellipsoid. With Assumptions 1 and 2, given class A and B are tightly covered by ellipsoid  $E_a$  and  $E_b$  in a 3-D space. Let  $R_{a_1}$  be the polar radius of  $E_a$ , and  $R_{a_2}$ ,  $R_{a_3}$  be the equatorial radii of  $E_a$ ,  $R_{b_1}$  be the polar radius of  $E_b$ , and  $R_{b_2}$ ,  $R_{b_3}$  be the equatorial radii of  $E_b$ , the generalization analysis is ready to present following these settings.

**Theorem 2.** With Assumptions 1 and 2, given a perceptron function  $h = w_1x_1 + w_2x_2 + w_3$  that classifies A and B, and a sampling budget k. By drawing core-set on  $E_a$  and  $E_b$ , the

minimum distances to the boundaries of that core-set elements of  $S_A$  and  $S_B$ , are defined as  $d_a$  and  $d_b$ , respectively. Let err(h,k) be the classification error rate with respect to h and k, given  $\frac{\pi}{\omega} = \arcsin \frac{R_a - d_a}{R_a}$ , we then have an inequality of error:

$$\min \left\{ \frac{4 \prod_{i} R_{a_{i}} - (2R_{a_{1}} + \lambda_{k})(R_{a_{1}} - \lambda_{k})^{2}}{4 \prod_{i} R_{a_{i}} + 4 \prod_{i} R_{b_{i}}}, \frac{4 \prod_{i} R_{b_{i}} - (2R_{b_{1}} + \lambda'_{k})(R_{b_{1}} - \lambda'_{k})^{2}}{4 \prod_{i} R_{b_{i}} + 4 \prod_{i} R_{a_{i}}} \right\}$$

$$\mathrm{err}(h,k) < \dfrac{1}{k},$$

where  $i=1,2,3,$  and

$$\lambda_k = \dfrac{R_{a_1}^2}{3} 
+ \sqrt[3]{ -\dfrac{\sigma_k}{2\pi} + \sqrt{\dfrac{\sigma_k^2}{4\pi^2} - \dfrac{\pi^3 R_{a_1}^3}{27}} } 
+ \sqrt[3]{ -\dfrac{\sigma_k}{2\pi} - \sqrt{\dfrac{\sigma_k^2}{4\pi^2} - \dfrac{\pi^3 R_{a_1}^3}{27}} }$$

$$\sigma_k = \Big(\dfrac{2k-4}{3k} - \dfrac{\pi R_{a_1}}{2\varphi}\Big)\pi\prod_{i} R_{a_i} 
- \dfrac{4\pi\prod_{i} R_{b_i}}{3k}$$

$$\lambda_k' = \dfrac{R_{b_1}^2}{3} 
+ \sqrt[3]{ -\dfrac{\sigma_k'}{2\pi} + \sqrt{\dfrac{{\sigma_k'}^{2}}{4\pi^2} - \dfrac{\pi^3 R_{b_1}^3}{27}} } 
+ \sqrt[3]{ -\dfrac{\sigma_k'}{2\pi} - \sqrt{\dfrac{{\sigma_k'}^{2}}{4\pi^2} - \dfrac{\pi^3 R_{b_1}^3}{27}} }$$

$$\sigma_k' = \Big(\dfrac{2k-4}{3k} - \dfrac{\pi R_{b_1}}{2\varphi}\Big)\pi\prod_{i} R_{b_i} 
- \dfrac{4\pi\prod_{i} R_{a_i}}{3k}$$

#### 6.2.2 Our insights

Insight ①: tighter lower error bound. Let  $lower[err(h,k)]_{Sphere}$  and  $lower[err(h,k)]_{Ellipsoid}$  be the lower bounds of the generalization errors by geodesic search with sphere and ellipsoid, respectively. With  $R_{a_1} < R_a$ , compare Theorems 1 and 2 we have the following proposition.

**Proposition 1.** Given a perceptron function  $h = w_1x_1 + w_2x_2 + w_3$  that classifies A and B, and a sampling budget k. By drawing core-set on  $S_a$  and  $S_b$ , let err(h,k) be the classification error rate with respect

to h and k, with Theorems  $\boxed{1}$  and  $\boxed{2}$  the lower bounds of geodesic search with sphere and ellipsoid satisfy: lower $[err(h,k)]_{Ellipsoid} < lower[err(h,k)]_{Sphere}$ .

Insight ②: higher probability of achieving a zero error. Let  $\Pr[err(h,k)=0]_{\mathrm{Sphere}}$  and  $\Pr[err(h,k)=0]_{\mathrm{Ellipsoid}}$  be the probabilities of achieving a zero error of geodesic search with sphere and ellipsoid, respectively. Their relationship is presented in Proposition 2.

**Proposition 2.** Given a perceptron function  $h = w_1x_1 + w_2x_2 + w_3$  that classifies A and B, and a sampling budget k. By drawing core-set on  $E_a$  and  $E_b$ , let err(h,k) be the classification error rate with respect to h and k, with Assumptions 1 and 2 the probabilities of geodesic search with sphere and ellipsoid satisfy:  $Pr[err(h,k) = 0]_{Ellipsoid} > Pr[err(h,k) = 0]_{Sphere}$ .

Overall, geodesic search with ellipsoid is more effective than with sphere, due to 1) tighter lower error bound, and 2) higher probability to achieve a zero error.

### 6.3 High-dimensional generalizations

With the above insights, we next present a connection between 3-D sphere/ellipsoid and d-dimensional hyperesphere/hyperellipsoid, where d > 3. The major technique is to prove that the volume of the 3-D sphere and ellipsoid are lower dimensional generalization of the d-dimensional hypheresphere and hyperellipsoid, respectively. References can refer to n-sphere [42], [43], and volume prototypes of hyperellipsoids [44]. With volume generalization analysis, all proofs from Theorems 1 to 2 and Propositions 1 and 2 can hold in d-dimensional geometry.

In the following, Theorems 3 and 4 then present a high-dimensional generalization for the above theoretical results, in terms of the volume functions of sphere and ellipsoid.

**Theorem 3.** Let  $\operatorname{Vol}_{\mathbf{d}}(S_a)$  or  $\operatorname{Vol}_{\mathbf{d}}(r_a)$  be the volume of  $\mathbf{d}$ -dimensional hypersphere  $S_a$  with a radius  $r_a$ , given  $\vartheta \in [0,\pi]$ , by performing integral operation on any  $(\mathbf{d}\text{-}1)$ -dimensional hypersphere, there exists  $\operatorname{Vol}_{\mathbf{d}}$  can be approximated as  $\operatorname{Vol}_{\mathbf{m}}(S_a) = \int_0^{\pi/2} 2\operatorname{Vol}_{\mathbf{m}\cdot 1}(r_a \cos(\vartheta))r_a(\cos(\vartheta))d\vartheta$ , and we define this operation as  $\operatorname{Vol}_{\mathbf{m}}(S_a) \bowtie \operatorname{Vol}_{\mathbf{m}\cdot 1}(S_a)$ . Then, we know  $\operatorname{Vol}_{\mathbf{m}\cdot 1}(S_a) \bowtie \operatorname{Vol}_{\mathbf{m}\cdot 2}(S_a)$ ,  $\operatorname{Vol}_{\mathbf{m}\cdot 2}(S_a) \bowtie \operatorname{Vol}_{\mathbf{m}\cdot 3}(S_a)$ , ...,  $\operatorname{Vol}_{\mathbf{d}}(S_a) \bowtie \operatorname{Vol}_{\mathbf{3}}(S_a)$  is a low-dimensional generalization of  $\operatorname{Vol}_{\mathbf{d}}(S_a)$ .

The proof skills of Theorem 3 can refer to a mathematical perspective Appendix A.8 also presents a machine learning proof skill. Moreover, the proof of Theorem 3 can be adopted in the generalization of 3-D ellipsoid to *d*-D hyperellipsoid.

**Theorem 4.** Let  $\operatorname{Vol}_{\boldsymbol{d}}(E_a)$  be the volume of a  $\boldsymbol{d}$ -dimensional hyperellipsoid, given  $\vartheta \in [0,\pi]$ , by performing integral operation on any  $(\boldsymbol{d}\text{-}1)$ -dimensional hyperellipsoid, there exists  $\operatorname{Vol}_{\boldsymbol{m}-1}(S_a) \bowtie \operatorname{Vol}_{\boldsymbol{m}-2}(S_a)$ ,  $\operatorname{Vol}_{\boldsymbol{m}-2}(S_a) \bowtie \operatorname{Vol}_{\boldsymbol{m}-3}(S_a)$ , ...,  $\operatorname{Vol}_{\boldsymbol{d}}(S_a) \bowtie \operatorname{Vol}_{\boldsymbol{3}}(S_a)$ . With this progressive relationship, we can say  $\operatorname{Vol}_{\boldsymbol{3}}(E_a)$  is a low-dimensional generalization of  $\operatorname{Vol}_{\boldsymbol{d}}(E_a)$ .

The volume of a hyperellipsoid can also be generalized by replacing the operation  $\prod_i R_a$  by the polar radius  $\prod_i R_{a_i}$ , for 0 < i < d + 1, in the formula for the volume of a hypersphere. Proofs also can refer to a mathematical skill<sup>2</sup>

## 7 EXPERIMENTS

In experiments, we start by showing how BALD degenerates its performance with an uninformative prior and the redundant information, and show that how our proposed GBALD relieves theses limitations.

Our experiments discuss three questions: 1) is GBALD using core-set of Eq. (11) competitive with an uninformative prior? 2) can GBALD using the ranking of Eq. (14) improve the informative acquisitions of model uncertainty? and 3) can GBALD outperform the state-of-the-art acquisition approaches? Following the experiment settings of [1], [5], we use MC dropout to implement the Bayesian approximation of DNNs. Three benchmark datasets are selected: MNIST, SVHN, and CIFAR10.

### 7.1 Baselines

To evaluate the performance of GBALD, several typical baselines from the latest deep AL literature are selected.

- Bayesian active learning by disagreement (BALD) 13. It has been introduced in Section 3.
- Maximize variation ratio (Var) 1. The algorithm chooses the unlabeled data that maximizes its variation ratio of the probability:

$$x^* = \arg\max_{x \in \mathcal{D}_n} \left\{ 1 - \max_{y \in \mathcal{Y}} \Pr(y|, x, \mathcal{D}_0)) \right\}. \tag{15}$$

• Maximize entropy (Entropy) [1]. The algorithm chooses the unlabeled data that maximizes the predictive entropy:

$$x^* = \underset{x \in \mathcal{D}_u}{\operatorname{arg max}} \Big\{ - \sum_{y \in \mathcal{Y}} \Pr(y|x, \mathcal{D}_0)) \log \Big( \Pr(y|x, \mathcal{D}_0) \Big) \Big\}.$$
(16)

• k-modoids [45]. A classical unsupervised algorithm that represents the input distribution by k clustering centers:

$$\{x_1^*, x_2^*, ..., x_k^*\} = \underset{z_1, z_2, ..., z_k}{\arg\min} \left\{ \sum_{i=1}^k \sum_{z_i \in \mathcal{X}^k} \|x_i - z_i\| \right\},$$
 (17)

where  $\mathcal{X}^k$  denotes the k-th subcluster centered with  $z_i$ , and  $z_i \in \mathcal{X}, \forall i$ .

- Greedy *k*-centers (*k*-centers) [8]. A geometric core-set interpretation on sphere. See Eq. (4).
- BatchBALD 5. A batch extension of BALD which incorporates the diversity, not maximal entropy as BALD, to rank the acquisitions:

$$\{x_{t_1}^*, ..., x_{t_b}^*\}$$

$$= \underset{x_{t_1}, ..., x_{t_b}}{\operatorname{arg}} \max_{x_{t_1}, ..., x_{t_b}} H(y_{t_1}, ..., y_{t_b}) - \operatorname{E}_{p(\theta \mid \mathcal{D}_0)} [H(y_{t_1}, ..., y_{t_b} \mid \theta)],$$

$$(18)$$

where  $H(y_{t_1},...,y_{t_b})$  denote the expected entropy over all possible labels from  $y_{t_1}$  to  $y_{t_b}$  such that  $H(y_{t_1},...,y_{t_b}) = E_p(y_{t_1},...,y_{t_b})[-\log p(y_{t_1},...,y_{t_b})]$ , and  $E_{p(\theta|\mathcal{D}_0)}[H(y_{t_1},...,y_{t_b}|\theta)]$  is estimated by MC sampling [26] [46] a subset from  $\mathcal{X}$  which approximates the parameter distributions of  $\theta$ .

**Parameters of GBALD.** The parameter settings of Eq. (5) are  $R_0 = 2.0e + 3$  and  $\eta = 0.9$ . Accuracy of each acquired dataset of the experiments are averaged over 3 runs.

<sup>1.</sup> https://www.sjsu.edu/faculty/watkins/ndim.htm

<sup>2.</sup> https://www.sjsu.edu/faculty/watkins/ellipsoid.htm

### 7.2 Uninformative priors

As discussed in the introduction, BALD is sensitive to an uninformative prior, i.e.  $p(\mathcal{D}_0|\theta)$ . We thus initialize  $\mathcal{D}_0$  from a fixed class of the training data of datasets to observe its acquisition performance. In this way,  $p(\mathcal{D}_0|\theta)$  is uninformative due to extreme label category.

Figure 3 presents the prediction accuracies of BALD with an acquisition budget of 130 over the training set of MNIST. Based on the uninformative setting of  $p(\mathcal{D}_0|\theta)$ , we randomly select 20 samples from digit '0' and '1' to initialize  $\mathcal{D}_0$ , respectively. The classification model of AL follows a convolutional neural network (CNN) with one block of [convolution, dropout, max-pooling, relu], with 32, 3x3 convolution filters, 5x5 max pooling, and 0.5 dropout rate. In the AL loops, we use 2,000 MC dropout samples from the unlabeled data pool to fit the training of the network as with  $\boxed{5}$ .

As the figure shown, BALD can slowly accelerate the training model due to the biased initial acquisitions, which cannot uniformly cover all the label categories. Moreover, the uninformative prior guides BALD to unstable acquisition results. Specifically, in Figure 3(b), BALD with Bathsize = 10 shows better performance than that of Batchsize =1; while BALD in Figure 3(a) keeps stable performance. This is because the initial labeled data does not cover all classes and BALD with Batchsize =1 may further be misled to select those samples from one or a few fixed classes at the first acquisitions. However, Batchsize >1 may result in a random acquisition process that possibly covers more diverse labels at its first acquisitions. Another excursive result of BALD is that the increasing batch size cannot degenerate its acquisition performance in Figure 3(b). For example, Batchsize =10 > Batchsize =1 > Batchsize =20,40 > Batchsize =30, where '>' denotes 'better' performance; Batchsize = 20 achieves similar results as with Batchsize =40. This undermines the acquisition policy of BALD: its performance would be degenerated when the batch size increases, and sometimes worse than random sampling. This also is the reason why we utilize a core-set to start BALD in our framework.

Different to BALD, core-set construction of GBALD using Eq. (11) provides a **complete label matching against all classes**. Therefore, it outperforms BALD with the batch sizes of 1, 10, 20, 30, and 40. As the shown learning curves in Figure 3, GBALD with a batch size of 1 and sequence size of 10 (i.e. breakpoints of acquired size are 10, 20, ..., 130) achieves significantly higher accuracies than BALD using different batch sizes since BALD misguides the network updating using a poor prior.

### 7.3 Improved informative acquisitions

For BALD, repeated or similar acquisitions can easily delay the acceleration or improvement of the model training. Following the experiment settings of Section 7.1, we compare the best performance of BALD with a batch size of 1 and GBALD with different batch size parameters. Following Eq. (14), we set  $b = \{3, 5, 7\}$  and b'=1, respectively, that means, we output the most representative data from a batch of highly-informative acquisitions. Different settings on b and b' are used to observe the parameter perturbations of GBALD.

Training by the same parameterized CNN model as in Section 7.2, Figure 4 presents the acquisition performance of parameterized BALD and GBALD. As the learning curves shown, BALD cannot accelerate the model as fast as GBALD due to the repeated information over the acquisitions. For GBALD, it ranks

the batch acquisitions of the highly-informative samples and selects the most representative ones. By employing this special ranking strategy, GBALD can **reduce the probability of sampling those nearby data of the previous acquisitions**. It is thus GBALD significantly outperforms BALD, even if we progressively increase the ranked batch size b.

### 7.4 Active acquisitions

GBALD using Eqs. (11) and (14) has been demonstrated to achieve successful improvements over BALD. We thus combine these two components into a uniform framework. Figure 5 presents the AL accuracies using different acquisition algorithms on the three image datasets. The selected baselines follow [1] including 1) maximizing the variation ratios (Var), 2) BALD, 3) maximizing the entropy (Entropy), 4) k-medoids, and one greedy 5) k-centers approach [8]. The network architecture is a three-layer multi-layer perceptron (MLP) with three blocks of [convolution, dropout, maxpooling, relu], with 32, 64, and 128 3x3 convolution filters, 5x5 max pooling, and 0.5 dropout rate. In the AL loops, the MC dropout still randomly samples 2,000 data from the unlabeled data pool to approximate the training of the network architecture following [5]. The initial labeled data of MNIST, SVHN and CIFAR-10 are 20, 1000, 1000 random samples from their full training sets, respectively.

The batch size of the compared baselines is 100, where GBALD ranks 300 acquisitions to select 100 data for the training, i.e. b = 300, b' = 100. As the learning curves shown in Figure 5, 1) k-centers algorithm performs more poorly than the other compared baselines because the representation optimization with the sphere geodesic usually falls into the selection of the boundary data; 2) Var, Entropy, and BALD algorithms cannot accelerate the network model rapidly due to those highly-skewed acquisitions towards few fixed classes at its first acquisitions (start states); 3) k-medoids approach does not interact with the neural network model while directly imports the clustering centers into its training set and the results are not strong; 4) the accuracies of the acquisitions of GBALD achieve better performance at the beginning than the Var, Entropy, and BALD approaches which fed the training set of the network model via acquisition loops. In short, the network is improved faster after drawing the distribution characteristics of the input dataset with sufficient labels. GBALD thus consists of the representative and informative acquisitions in its uniform framework. The advantages of these two acquisition paradigms are integrated to present higher accuracies than any single paradigm.

Table 1 reports the mean±std values of the test accuracies of the breakpoints of the learning curves in Figure 5, where the breakpoints of MNIST are  $\{0,10,20,30,...,600\}$ , the breakpoints of SVHN are  $\{0,100,200,...,10000\}$ , and the breakpoints of CIFAR10 are  $\{0,100,200,...,20000\}$ . We then calculate their average accuracies and std values over these acquisition points. As the shown in Table 1, all std values around 0.1, yielding a norm value. Usually, an average accuracy on the same acquisition size with different random seeds of DNNs, will result a small std value. Our mean accuracy spans across the whole learning curve.

The results show that 1) GBALD achieves the highest average accuracies; 2)k-medoids is ranked the second amongst the compared baselines; 3) k-centers has ranked the worst accuracies amongst these approaches; 4) the others, which iteratively update the training model are ranked at the middle including BALD, Var and Entropy algorithms. Table 2 shows the acquisition numbers

Image /page/8/Figure/2 description: A two-panel figure showing line plots of classification accuracy versus acquired dataset size for two handwritten-digit classes. Both panels share the same axes: x-axis labeled "Acquired dataset size" from 0 to about 130 (tick marks at 0, 20, 40, 60, 80, 100, 120) and y-axis labeled "Accuracy" ranging from 0.1 to 1.0. Panel (a) on the left is labeled "(a) Digit '0'" and panel (b) on the right is labeled "(b) Digit '1'". Each panel contains a legend (boxed) listing seven methods with colored lines: "BALD,Batchsize=1" (light blue), "Random Sampling" (purple), "BALD,Batchsize=10" (green), "BALD,Batchsize=20" (dark blue), "BALD,Batchsize=30" (magenta), "BALD,Batchsize=40" (navy), and "GBALD,Sequence" (red). In both plots all methods start near accuracy 0.1 at dataset size 0 and increase as more data is acquired. The red GBALD sequence line rises most steeply and attains the highest accuracy (around 0.85–0.92) by mid to large dataset sizes (roughly 40–120). The BALD with batch size 1 (light blue) shows a noisy but consistently strong upward trend reaching accuracies close to 0.8–0.9 by the largest dataset sizes. Random sampling (purple) and BALD with small batch sizes (green for 10) give moderate performance, while larger BALD batch sizes (20, 30, 40; dark blue, magenta, navy) improve more slowly and generally achieve lower accuracy at a given acquired size. Overall the figure compares how acquisition strategy and batch size affect accuracy growth for digits '0' and '1', with GBALD and batchsize=1 performing best and larger batch sizes lagging behind.

Figure 3: Acquisitions with uninformative priors from digit '0' and '1'.

Image /page/8/Figure/4 description: Two side-by-side line plots comparing accuracy vs. acquired dataset size for active learning on two digits. Left plot labeled “(a) Digit ‘0’” and right plot labeled “(b) Digit ‘1’”. Both plots have x-axes labeled “Acquired dataset size” (tick marks roughly 0, 20, 40, 60, 80, 100, 120) and y-axes labeled “Accuracy” with values from about 0.1 up to 0.9. Each plot shows four time-series: a black dashed line (legend: “BALD, Batchsize=1”), a green solid line (“GBALD, Batchsize=3”), a blue solid line (“GBALD, Batchsize=5”), and a magenta/pink solid line (“GBALD, Batchsize=7”). Gridlines are shown. In both plots all methods start near accuracy ≈0.1 at the smallest acquired sizes and generally rise as acquired dataset size increases, with fluctuations: by dataset sizes ~60–80 accuracies are in the 0.6–0.8 range and by ~120 they reach ≈0.8–0.9. The left (Digit ‘0’) curves are closely clustered with occasional small peaks (one curve near 0.63 around size ~40) and the blue/magenta lines end slightly higher (~0.85–0.9) than the dashed black. The right (Digit ‘1’) plot shows somewhat larger early variability (one method briefly reaching ~0.5–0.6 around sizes 10–30) but the four methods converge toward ≈0.85–0.9 at the largest acquired dataset sizes. A legend box in the lower-right of each subplot identifies the four line styles and batch sizes.

Figure 4: GBALD outperforms BALD using ranked informative acquisitions which cooperate with representation constraints.

Image /page/8/Figure/6 description: Three-panel figure comparing active learning sampling methods on three image classification datasets. Each panel is a line plot of Accuracy (y-axis) vs Acquired dataset size (x-axis) with a legend showing six methods: Var (black), BALD (green), Entropy (blue), k-medoids (purple), k-centers (magenta/pink), and GBALD (red). Grid lines are shown on all plots. (a) MNIST (left): x-axis labeled 0–600 acquired samples, y-axis from ~0.1 to 1.0. Curves rise quickly; GBALD (red, batchsize=3) reaches the highest accuracy earliest (≈0.95 by ~100 samples and ≈0.99 by 600). Var, BALD, and Entropy (black/green/blue, batchsize=1) follow closely and converge near the top. k-medoids and k-centers lag slightly and show more variability, peaking lower (~0.9–0.95). Legend text in this panel includes “Var,Batchsize=1”, “BALD,Batchsize=1”, “Entropy,Batchsize=1”, and “GBALD,Batchsize=3”. (b) SVHN (center): x-axis 0–10,000 acquired samples, y-axis ~0.1–1.0. GBALD (red, batchsize=300) attains the highest accuracy throughout (≈0.90–0.95), while Var/BALD/Entropy (black/green/blue, batchsize=100) climb to about 0.86–0.91 and then flatten. k-medoids and k-centers remain slightly lower and show more fluctuations. Legend labels include “Var,Batchsize=100”, “BALD,Batchsize=100”, “Entropy,Batchsize=100”, and “GBALD,Batchsize=300”. (c) CIFAR10 (right): x-axis 0–20,000 (noted as ×10^4 scale on the axis), y-axis ~0.1–0.9. GBALD (red, batchsize=300) again yields the best final accuracy (≈0.84–0.87 at the right end). Var/BALD/Entropy (black/green/blue, batchsize=100) cluster below GBALD, ending around ≈0.70–0.78. k-medoids (purple) shows the largest variability and the lowest accuracies (~0.60–0.75). Overall the red GBALD curve consistently outperforms the other methods across all three datasets.

Figure 5: Active acquisitions on MNIST, SVHN, and CIFAR10 datasets.

Table 1: Mean±std of the test accuracies of the breakpoints of the learning curves on MNIST, SVHN, and CIFAR-10.

| Datasets | Algorithms          |                     |                     |                      |                     |                                       |
|----------|---------------------|---------------------|---------------------|----------------------|---------------------|---------------------------------------|
|          | Var                 | BALD                | Entropy             | k-medoids            | k-centers           | GBALD                                 |
| MNIST    | $0.8419 \pm 0.1721$ | $0.8645 \pm 0.1909$ | $0.8498 \pm 0.2098$ | $0.8785 \pm 0.1433$  | $0.8052 \pm 0.1838$ | <b><math>0.9106 \pm 0.1296</math></b> |
| SVHN     | $0.8535 \pm 0.1098$ | $0.8510 \pm 0.1160$ | $0.8294 \pm 0.1415$ | $0.8498 \pm 0.1294$  | $0.7909 \pm 0.1235$ | <b><math>0.8885 \pm 0.1054</math></b> |
| CIFAR-10 | $0.7122 \pm 0.1034$ | $0.6760 \pm 0.1023$ | $0.6536 \pm 0.1038$ | $0.71837 \pm 0.1245$ | $0.5890 \pm 0.1758$ | <b><math>0.7440 \pm 0.1087</math></b> |

Table 2: Number of acquisitions on MNIST, SVHN and CIFAR10 until 70%, 80%, and 90% accuracies are reached.

| Algorithms       | Accuracies            |                        |                             |
|------------------|-----------------------|------------------------|-----------------------------|
|                  | 70%                   | 80%                    | 90%                         |
| Var              | 140/1,700/5,700       | 150/2,200/>20,000      | 210/>10,000/>6,100          |
| BALD             | 110/1,700/8,800       | 120/2,300/>20,000      | 190/7,100/>20,000           |
| Entropy          | 110/1,900/11,200      | 150/2,400/>20,000      | 200/8,600/>20,000           |
| <i>k-modoids</i> | 70/1,700/5,900        | 90/2,200/16,000        | <b>170</b> /6,200/>20,000   |
| <i>k-centers</i> | 110/2,000/10,100      | 150/3,800/>20,000      | 280/>10,000/>20,000         |
| GBALD            | <b>50/1,400/4,800</b> | <b>70/1,900/12,200</b> | <b>170/3,900/&gt;20,000</b> |

Table 3: Mean±std of active acquisitions on SVHN with 5,000 and 10,000 repeated samples.

| Algorithms   | Accuracies                        |                                   |                                   |
|--------------|-----------------------------------|-----------------------------------|-----------------------------------|
|              | 0 repeats                         | 5,000 repeats                     | 10,000 repeats                    |
| Var          | $0.8535±0.1098$                   | $0.8478±0.1074$                   | $0.8281±0.1082$                   |
| BALD         | $0.8510±0.1160$                   | $0.8119±0.1216$                   | $0.7689±0.1288$                   |
| <b>GBALD</b> | <b><math>0.8885±0.1054</math></b> | <b><math>0.8694±0.1032</math></b> | <b><math>0.8630±0.1002</math></b> |

of achieving the accuracies of 70%, 80%, and 90% on the three datasets. The three numbers of each cell are the acquisition numbers over MNIST, SVHN, and CIFAR10, respectively. The results show that GBALD can use fewer acquisitions to achieve a desired accuracy than the other algorithms.

### 7.5 Active acquisitions with repeated samples

Repeatedly collecting samples in the establishment of a database is very common. Those repeated samples may be continuously evaluated as the primary acquisitions of AL due to the lack of one or more categories of class labels. Meanwhile, this situation may lead the evaluation of the model uncertainty to fall into repeated acquisitions. To respond this collecting situation, we compare the acquisition performance of BALD, Var, and GBALD using 5,000 and 10,000 repeated samples from the first 5,000 and 10,000 unlabeled data of SVHN, respectively. In addition, the unsupervised algorithms which do not interact with the network architecture, such as k-medoids and k-centers, have been shown that they cannot accelerate the training in terms of the experiment results of Section 7.3. Thus, we are no longer studying their performance. The network architecture still follows the settings of Section 7.3.

The acquisition results over the repeated SVHN datasets are presented in Figure 7. The batch sizes of the compared baselines are 100, where GBALD ranks 300 acquisitions to select 100 data for the training, i.e. b = 300, b' = 100. The mean±std values of these baselines of the breakpoints (i.e.  $\{0, 100, 200, ..., 10000\}$ ) are reported in Table 3. The results demonstrate that GBALD shows slighter perturbations on the repeated samples than Var and BALD because it draws the core-set from the input distribution as the initial acquisitions, leading a small probability to sample from one or more fixed class. In GBALD, the informative acquisitions constrained with geometric representations further scatter the acquisitions spread in different classes. However, the Var and BALD algorithms have no particular schemes against the repeated acquisitions. The maximizer on the model uncertainty may be repeatedly produced by those repeated samples. In additional, the unsupervised algorithms such as k-medoids and k-centers don not have these limitations, but cannot accelerate the training since there has no interactions with the network architecture.

Table 4: Mean±std of active noisy acquisitions on SVHN with 5,000 and 10,000 noises.

| Algorithms   | Accuracies                            |                                       |                                       |
|--------------|---------------------------------------|---------------------------------------|---------------------------------------|
|              | 0 noises                              | 5,000 noises                          | 10,000 noises                         |
| Var          | $0.8535 \pm 0.1098$                   | $0.7980 \pm 0.1203$                   | $0.7702 \pm 0.1238$                   |
| BALD         | $0.8510 \pm 0.1160$                   | $0.8205 \pm 0.1185$                   | $0.7849 \pm 0.1239$                   |
| <b>GBALD</b> | <b><math>0.8885 \pm 0.1054</math></b> | <b><math>0.8622 \pm 0.0991</math></b> | <b><math>0.8301 \pm 0.0916</math></b> |

### 7.6 Active acquisitions with noisy samples

Noisy labels [9], [47] are inevitable due to human errors in data annotation. Training on noisy labels, the neural network model will degenerate its inherent properties. To assess the perturbations of the above acquisition algorithms against noisy labels, we organize the following experiment scenarios: we select the first 5,000 and 10,000 samples respectively from the unlabeled data pool of the MNIST dataset and reset their labels by shifting {'0', '1',...,'8'} to {'1', '2',...,'9'}, respectively. The network architecture follows the MLP of Section 7.3. The selected baselines are Var and BALD.

Figure 7 presents the acquisition results of those baseline with noisy labels. The batch sizes of the compared baselines are 100, where GBALD ranks 300 acquisitions to select 100 data for the training, i.e. b = 300, b' = 100. Table 4 presents the mean±std values of the breakpoints (i.e.  $\{0, 100, 200, ..., 10000\}$ ) over learning curves of Figure 7. The results further show that GBALD has smaller noisy perturbations than the other baselines. For Var and BALD, model uncertainty leads high probabilities to sample those noisy data due to their greatly updating on the model.

### 7.7 GBALD vs. BatchBALD

Batch deep AL was recently proposed to accelerate the training of a DNN model. In recent literature, BatchBALD [5] extended BALD with a batch acquisition setting to converge the network using fewer iteration loops. Different to BALD, BathBALD introduces the diversity to avoid the repeated or similar acquisitions.

How to set the batch size of the acquisitions attracted our eyes before starting the experiments. It involves with whether our experiment settings are fair and reasonable. From a theoretical view, the larger the batch size, the worse the batch acquisitions will be. Experimental results of [5] also demonstrated this phenomenon. We thus set different batch sizes to run BatchBALD. Figure 8 presents the comparison results of BALD, BatchBALD, and our proposed GBALD as with the experiment settings of Section 7.3. As the shown in this figure, BatchBALD degenerates the test accuracies if we progressively increase the bath sizes, where BatchBALD with a batch size of 10 keeps similar learning curves as BALD. It shows that BatchBALD actually can accelerate BALD with a similar acquisition result if the batch size is not large. This means, if the batch size is between 2 to 10, BatchBALD will degenerate into BALD and maintains highly-consistent results.

Also because of this, BatchBALD also has the same sensitivity to an uninformative prior. For our GBALD, the core-set solicits sufficient data which properly matches the input distribution (w.r.t. acquired data set size  $\leq 100$ ), providing expressive input features to start the DNN model (w.r.t. acquired data set size > 100). Table 5 then presents the mean $\pm$ std of the breakpoints ( $\{0,10,20,...,600\}$ ) of active acquisitions on MNIST with batch settings. The statistical results show that GBALD has much higher mean accuracy than BatchBALD with different bath sizes. Therefore, evaluating the

Image /page/10/Figure/2 description: Three horizontally arranged line plots comparing active learning strategies labeled (a) Var, (b) BALD, and (c) GBALD. Each subplot has Accuracy on the y-axis (range ~0.1 to 1.0) and Acquired dataset size on the x-axis (0 to 10,000). Each plot contains three colored curves: red = 0 Repeats, green = 5,000 Repeats, blue = 100,000 Repeats. Legends in each plot specify the method name and batch size: (a) Var, Batchsize=100 (red/green/blue for 0/5,000/100,000 repeats); (b) BALD, Batchsize=100 (same repeat colors); (c) GBALD, Batchsize=300 (same repeat colors). In all three plots the curves start at low accuracy (~0.15–0.25) for small acquired dataset sizes, rise steeply up to ~0.75–0.85 by about 1,000–2,000 examples, then gradually increase and plateau near 0.85–0.95 as the acquired dataset size approaches 10,000. In (b) BALD and (c) GBALD the red (0 repeats) curve tends to achieve the highest final accuracy, green is intermediate, and blue (100,000 repeats) is lowest; a similar ordering is visible in (a) Var though differences are small. Each subplot shows grid lines and the subplot captions centered below: “(a) Var”, “(b) BALD”, and “(c) GBALD.” The legend boxes are placed in the lower-right area of each subplot.

Figure 6: Active acquisitions on SVHN with 5,000 and 10,000 repeated samples.

Image /page/10/Figure/4 description: A three-panel figure of line plots comparing active learning methods. Each panel plots Accuracy (y-axis, range ~0.1–1.0) versus Acquired dataset size (x-axis, 0–10,000) with gridlines. (a) Var (left): three curves (red, green, blue) labeled in the legend as "Var,Batchsize=100, 0 Noises", "Var,Batchsize=100, 5000 Noises", "Var,Batchsize=100, 100000 Noises". All three rise steeply from ~0.2 at small dataset sizes to ~0.7–0.8 by ~1,000–2,000 samples, then gradually increase and asymptote near 0.85–0.92 by 10,000 samples. The red (0 noises) is highest across the range, green (5,000 noises) is intermediate, blue (100,000 noises) lowest. (b) BALD (center): three curves with legend "BALD,Batchsize=100, 0 Noises", "BALD,Batchsize=100, 5000 Noises", "BALD,Batchsize=100, 100000 Noises". Similar pattern to Var — rapid gain early, then slower improvement — red highest, green middle, blue lowest, ending around 0.86–0.93 by 10,000. (c) GBALD (right): three curves with legend "GBALD,Batchsize=300, 0 Noises", "GBALD,Batchsize=300, 5000 Noises", "GBALD,Batchsize=300, 100000 Noises". Curves climb quickly to ~0.8–0.9 within the first few thousand samples and then level off near 0.88–0.95 by 10,000. Across all panels higher label noise (5,000 and 100,000 noisy labels) reduces accuracy (green and especially blue curves lie below the red curve). Each subplot is captioned below as (a) Var, (b) BALD, (c) GBALD.

Figure 7: Active noisy acquisitions on SVHN with 5,000 and 10,000 noisy labels.

Image /page/10/Figure/6 description: A line plot showing classification accuracy versus acquired dataset size for several active learning methods. The x-axis is labeled “Acquired dataset size” and runs from 0 to 600 (major ticks at ~0, 100, 200, 300, 400, 500, 600). The y-axis is labeled “Accuracy” and ranges from about 0.1 up to 1.0 (gridlines visible at roughly 0.1 increments). Five colored lines are plotted with a legend in a boxed inset near the center-right: green labeled “BALD,Batchsize=1”; black labeled “BatchBALD,Batchsize=10”; blue labeled “BatchBALD,Batchsize=40”; purple labeled “BatchBALD,Batchsize=100”; and red labeled “GBALD,Batchsize=3”. All methods start near accuracy 0.1 at dataset size 0, then rapidly increase between 0 and ~100 samples. The red GBALD curve rises fastest (reaching ~0.85–0.9 around 60–100 samples), while the others (green, black, blue, purple) climb more gradually but similarly, reaching roughly 0.8–0.9 by ~100–150 samples. After about 150–200 acquired samples the lines oscillate slightly but converge and plateau toward high accuracy (around 0.95–0.99) as the acquired dataset size approaches 600. A light grid is shown behind the curves.

Figure 8: Comparisons of BALD, BatchBALD, and GBALD of active acquisitions on MNIST with bath settings.

model uncertainty of DNN using those highly-representative coreset samples can improve the performance of the neural network.

### 7.8 Acceleration of accuracy

Accelerations of accuracy i.e. the first-orders of the breakpoints of the learning curve, describe the efficiency of the active acquisition loops. Different to the accuracy curves, the acceleration curve

Table 5: Mean±std of BALD, BatchBALD, and GBALD of active acquisitions on MNIST with batch settings.

| Algorithms | Batch sizes | Accuracies          |
|------------|-------------|---------------------|
| BALD       | 1           | $0.8654 \pm 0.0354$ |
| BatchBALD  | 10          | $0.8645 \pm 0.0365$ |
| BatchBALD  | 40          | $0.8273 \pm 0.0545$ |
| BatchBALD  | 100         | $0.7902 \pm 0.0951$ |
| GBALD      | 3           | $0.9106 \pm 0.1296$ |

reflects how active acquisitions help the convergence of the interacting DNN model.

We thus firstly present the acceleration curves of different baselines on MNIST, SVHN, and CIFAR10 datasets as with the experiments of Section 7.3. The acceleration curves of active acquisitions are drawn in Figure 9. Observing those acceleration curves of different algorithms clearly finds that, GBALD always keeps **higher accelerations of accuracy** than the other baselines against the three benchmark datasets. This revels the reason of why GBALD can derive more informative and representative data to maximally update the DNN model.

The acceleration curves of active acquisitions with repeated samples are presented in Figure 10. As the shown in this figure, GBALD presents **slighter perturbations** to the number of repeated samples than that of Var and BALD due to its effective ranking scheme on optimizing model uncertainty of DNNs. The acceleration curves of active noisy acquisitions are drawn in Figure 11. Compared to Figure 7, it presents more intuitive descriptions for

Table 6: Relationship of accuracies and sizes of core-set on SVHN.

| Size of core-set          | Accuracies     |                   |                      |
|---------------------------|----------------|-------------------|----------------------|
|                           | Start accuracy | Ultimate accuracy | Mean±std accuracy    |
| $N_{\mathcal{M}} = 1,000$ | 0.8790         | 0.9344            | 0.9134±0.0169        |
| $N_{\mathcal{M}} = 2,000$ | 0.8898         | 0.9212            | 0.9151±0.0148        |
| $N_{\mathcal{M}} = 3,000$ | 0.8848         | <b>0.9364</b>     | 0.9173±0.0138        |
| $N_{\mathcal{M}} = 4,000$ | 0.8811         | 0.9271            | 0.9146±0.0165        |
| $N_{\mathcal{M}} = 5,000$ | <b>0.8959</b>  | 0.9342            | <b>0.9197±0.0117</b> |

the noisy perturbations to different baselines. With horizontal comparisons to the acceleration curves of Var and BALD, our proposed GBALD has smaller noisy perturbations due to 1) the powerful core-set which properly captures the input distribution, and 2) both the highly representative and informative acquisitions of the model uncertainty.

### 7.9 Hyperparameter settings

What is the proper time to start active acquisitions using Eq. (14) in GBALD framework? Does the size ratio of the core-set and model uncertainty acquisitions affect the performance of GBALD?

We discuss the key hyperparameter of GBALD here: the core-set size  $N_{\mathcal{M}}$ . Table 6 presents the relationship of accuracies and size of core-set, where start accuracy denotes the test accuracy over the initial core-set, and ultimate accuracy denotes the test accuracy over up to Q=20,000 training data. Let b=1000,b'=500 in GBALD,  $\mathcal{N}_{\mathcal{M}}$  be the number of the core-set size, the iteration budget  $\mathcal{A}$  of GBALD can then be defined as  $\mathcal{A}=(Q-\mathcal{N}_{\mathcal{M}})/b'$ . For example, if the number of the initial core-set labels are set as  $\mathcal{N}_{\mathcal{M}}=1,000$ , we have  $\mathcal{A}=(Q-\mathcal{N}_{\mathcal{M}})/b'\approx38$ ; if  $\mathcal{N}_{\mathcal{M}}=2,000$ , then  $\mathcal{A}=(Q-\mathcal{N}_{\mathcal{M}})/b'\approx36$ .

From Table 6, the GBALD algorithm keeps stable accuracies over the start, ultimate, and mean±std accuracies when there inputs more than 1,000 core-set labels. Therefore, drawing sufficient coreset labels using Eq. (10) to start the model uncertainty of Eq. (14) can maximize the performance of our GBALD framework.

**Hyperparameter settings on batch returns** b **and bath outputs** b'. Experiments of Sections 7.1 and 7.2 used different b and b' to observe the parameter perturbations. No matter what the settings of b' and b are, GBALD still outperforms BALD. For single acquisition of GBALD, we suggest b=3 and b'=1. For bath acquisitions, the settings on b' and b are user-defined according the time cost and hardware resources.

Hyperparameter setting on iteration budget  $\mathcal{A}$ . Given the acquisition budget Q, let b' be the number of the output returns at each loop,  $\mathcal{N}_{\mathcal{M}}$  be the number of the core-set size, the iteration budget  $\mathcal{A}$  of GBLAD then can be defined as  $\mathcal{A} = (Q - \mathcal{N}_{\mathcal{M}})/b'$ .

Other hyperparameter settings. Eq. (5) has one parameter  $R_0$  which describes the geometric prior from probability. The default radius of the intern balls  $R_0$  is used to legalize the prior and has no further influences on Eq. (10). It is set as  $R_0 = 2.0e + 3$  for those three image datasets. Ellipsoid geodesic is adjusted by  $\eta$  which controls how far of the updates of core-set to the boundaries of distributions. It is set as  $\eta = 0.9$  in this paper.

### 7.10 Two-sided t-test

We present two-sided (two-tailed) t-test [48] [49] for the learning curves of Figure 5. Different to the mean $\pm$  std of Table 1, t-test can enlarge the significant difference of those baselines. In the typical t-test, the two groups of observations usually require a

degree of freedom smaller than 30. However, the numbers of the breakpoints of MNIST, SVHN, and CIFAR10 are 61, 101, and 201, respectively, thereby holding a degree of freedom of 60, 100, 200, respectively. It is thus we introduce t-test score to directly compare the significant difference of the pairwise baselines.

t-test score between any pair group of breakpoints are defined as follows. Let  $B_1 = \{\alpha_1, \alpha_2, ..., \alpha_n\}$  and  $B_2 = \{\beta_1, \beta_2, ..., \beta_n\}$ , there exists t-score of

$$t$$
 - score =  $\sqrt{n} \frac{\mu}{\sigma}$ ,

where  $\mu = \frac{1}{n} \sum_{i=1}^{n} (\alpha_i - \beta_i)$ , and  $\sigma = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (\alpha_i - \beta_i - \mu)^2}$ .

In two-sided t-test,  $B_1$  beats  $B_2$  on breakpoints  $\alpha_i$  and  $\beta_i$  satisfying a condition of t-score  $> \nu$ ;  $B_2$  beats  $B_1$  on breakpoints  $\alpha_i$  and  $\beta_i$  satisfying a condition of t-score  $< -\nu$ , where  $\nu$  denotes the hypothesized criterion with a given confidence risk. Following 50, we add a penalty of  $\frac{1}{e}$  to each pair of breakpoints, which further enlarges their differences in the aggregated penalty matrix, where e denotes the number of  $B_1$  beats  $B_2$  on all breakpoints. All penalty values finally calculate their  $L_1$  expressions.

Figure 12 presents the penalty matrix over the learning curves of Figure 5. Column-wise values at the bottom of each matrix show the overall performance of the compared baselines. As the shown results, GBALD has significant performance than that of the other baselines over the three datasets. Especially for SVHN, it has superior performance.

## 8 Conclusion

We have introduced a novel Bayesian AL framework termed GBALD from the perspective of geometry, which seamlessly incorporates the representative (core-set) and informative (model uncertainty estimation) acquisitions to accelerate the training of a DNN model. Our GBALD yields significant improvements over BALD, flexibly resolving the limitations of an uninformative prior and the redundant information by optimizing the acquisition on an ellipsoid. Generalization analysis has asserted that, geodesic search with ellipsoid has tighter lower error bound and higher probability to achieve a zero error, than that of geodesic search with sphere. Compared to the representative or informative acquisition algorithms, experiments show that our GBALD spends much fewer acquisitions to accelerate the convergence of training model. Moreover, it keeps slighter accuracy reduction than other baselines against repeated and noisy acquisitions. Leveraging the acquisition sizes of the geometric core-set decides how our framework interacts with the representative and informative acquisitions. It will be a future work of Auto-AL that derives an advanced AL pipeline.

## REFERENCES

- Y. Gal, R. Islam, and Z. Ghahramani, "Deep bayesian active learning with image data," in *Proceedings of the 34th International Conference on Machine Learning-Volume 70*. JMLR. org, 2017, pp. 1183–1192.
- [2] A. Ashukha, A. Lyzhov, D. Molchanov, and D. Vetrov, "Pitfalls of indomain uncertainty estimation and ensembling in deep learning," in *International Conference on Learning Representations*, 2019.
- [3] B. Lakshminarayanan, A. Pritzel, and C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles," in *Advances in neural information processing systems*, 2017, pp. 6402–6413.
- [4] R. Pinsler, J. Gordon, E. Nalisnick, and J. M. Hernández-Lobato, "Bayesian batch active learning as sparse subset approximation," in Advances in Neural Information Processing Systems, 2019, pp. 6356– 6367.

Image /page/12/Figure/2 description: Three-panel line-plot figure comparing active learning acquisition methods. Each panel plots "Acceleration of accuracy" (y-axis) versus "Acquired dataset size" (x-axis) and includes an upper-right legend. (a) MNIST (left): x-axis from 0 to 600 (ticks at 0,100,...,600), y-axis from -0.02 to 0.12. Curves shown in the legend: Var, Batchsize=1 (black), BALD, Batchsize=1 (green), Entropy, Batchsize=1 (blue), k-medoids (purple/magenta), k-centers (violet), GBALD, Batchsize=3 (red). All methods show a sharp positive spike at very small acquired size that rapidly decays toward zero; GBALD (red) shows the largest early positive acceleration, BALD (green) dips slightly negative then returns, and the k-medoids/k-centers/Entropy/Var curves oscillate near zero afterward. (b) SVHN (center): x-axis from 1,000 to 10,000 (ticks at 1000,2000,...,10000), y-axis from -0.02 to 0.16. Legend lists Var, BALD, Entropy with Batchsize=100 and GBALD Batchsize=300 plus k-medoids and k-centers. All methods show a small initial positive spike then quickly settle close to zero; GBALD (red) has a slightly larger early peak than others but all lines converge near zero with small fluctuations. (c) CIFAR10 (right): x-axis labeled 0 to 2 (×10^4) with ticks at 0, 0.5, 1, 1.5, 2 (i.e., 0 to 20,000), y-axis from -0.02 to ~0.1. Legend similar to SVHN (Batchsize=100 for Var/BALD/Entropy, GBALD Batchsize=300). Curves are noisy near zero with small positive and negative fluctuations; no method sustains large acceleration, and GBALD again shows slightly larger early variability but all traces hover around zero across the acquired dataset sizes. Each subplot is labeled below: (a) MNIST, (b) SVHN, (c) CIFAR10.

Figure 9: Accelerations of accuracy of different baselines on MNIST, SVHN, and CIFAR10 datasets.

Image /page/12/Figure/4 description: Three small-multiple line plots arranged left-to-right comparing the "acceleration of accuracy" versus acquired dataset size for three acquisition criteria. Each subplot has x-axis labeled "Acquired dataset size" with ticks 0, 2000, 4000, 6000, 8000, 10000 and y-axis labeled "Acceleration of accuracy" with a multiplier shown as ×10^-3. (a) Left: titled "(a) Var". A red, green and blue line (legend entries: "Var,Batchsize=100, 0 Repeats", "Var,Batchsize=100, 5000 Repeats", "Var,Batchsize=100, 100000 Repeats") all start with a positive peak around ~1.0–1.5 ×10^-3 at small acquired sizes, then decay quickly and fluctuate near zero for larger dataset sizes, with typical oscillations roughly between −0.5×10^-3 and +0.5×10^-3. (b) Center: titled "(b) BALD". Three lines with legend entries "BALD,Batchsize=100, 0 Repeats", "BALD,Batchsize=100, 5000 Repeats", "BALD,Batchsize=100, 100000 Repeats" show a similar pattern: initial positive spike near ~1.5×10^-3, a rapid drop and oscillation about zero with small noise (≈±0.5×10^-3) as acquired size approaches 10,000. (c) Right: titled "(c) GBALD". Legend entries are "GBALD,Batchsize=100, 0 Repeats", "GBALD,Batchsize=100, 5000 Repeats", "GBALD,Batchsize=100, 100000 Repeats". These curves also show a large early positive peak (up to ≈2.0–2.5×10^-3), a brief deeper negative dip (approaching −1×10^-3) early on, then settle into small oscillations around zero for larger dataset sizes. All three panels indicate that acceleration is largest at small acquired dataset sizes and decays toward near-zero fluctuations as the acquired dataset size increases; batch size is 100 in all experiments and the three colored traces correspond to 0, 5,000 and 100,000 repeats respectively.

Figure 10: Accelerations of accuracy of active acquisitions on SVHN with 5,000 and 10,000 repeated samples.

Image /page/12/Figure/6 description: A three-panel line-plot figure comparing the "acceleration of accuracy" vs. acquired dataset size for three acquisition methods: (a) Var, (b) BALD, and (c) GBALD. Each panel shares an x-axis labeled "Acquired dataset size" with tick marks at 0, 2000, 4000, 6000, 8000 and 10000. Each panel has a legend with three colored traces representing different noise levels: red = "... , 0 Noises", green = "... , 5000 Noises", and blue = "... , 100000 Noises". Specific legend entries are: (a) "Var, Batchsize=100, 0 Noises" / "Var, Batchsize=100, 5000 Noises" / "Var, Batchsize=100, 100000 Noises"; (b) "BALD, Batchsize=100, 0 Noises" / "BALD, Batchsize=100, 5000 Noises" / "BALD, Batchsize=100, 100000 Noises"; (c) "GBALD, Batchsize=300, 0 Noises" / "GBALD, Batchsize=300, 5000 Noises" / "GBALD, Batchsize=300, 100000 Noises".

Panel (a) labeled "(a) Var": y-axis labeled "Acceleration of Accuracy" with a multiplier printed as "10^-4". The y-axis tick labels shown include approximately -5, 15 and 20 (meaning values in the range roughly -5×10^-4 to 20×10^-4). All three traces (red, green, blue) show a large positive spike at very small acquired sizes (near 0) reaching about 20×10^-4 (~0.002), then rapidly decay over the first ~2000 acquired samples and thereafter oscillate closely around zero with small amplitude fluctuations (on the order of a few ×10^-4) out to 10000.

Panel (b) labeled "(b) BALD": y-axis labeled "Acceleration of Accuracy" with multiplier "10^-3" and ticks showing roughly -1 and 2 (i.e., values ~-1×10^-3 to 2×10^-3). The three colored traces again show an early peak (up to ~2×10^-3) and then decay within the first ~2000 acquisitions to small oscillations around zero (amplitudes on the order of 10^-4 to 10^-3) through 10000. The red trace shows a pronounced negative dip immediately after the initial peak before settling; green and blue tracks follow similar damped noisy behavior.

Panel (c) labeled "(c) GBALD": y-axis labeled "Acceleration of accuracy" with multiplier "10^-3" and ticks around -0.5 and 1.5 (i.e., roughly -0.5×10^-3 to 1.5×10^-3). The three colored traces again peak at the start (the highest initial values of the three panels, up to ~2.5×10^-3), then rapidly decay and fluctuate near zero with small noise-like oscillations for acquired sizes beyond ~2000. Overall, across all three panels the three noise-level curves (red/green/blue) have very similar shapes and magnitudes: a large early positive acceleration, fast decay within the first few thousand acquired samples, and small zero-centered oscillations thereafter. Light grid lines are present in all subplots.

Figure 11: Accelerations of accuracy of active noisy acquisitions on SVHN with 5,000 and 10,000 noisy labels.

- [5] A. Kirsch, J. van Amersfoort, and Y. Gal, "Batchbald: Efficient and diverse batch acquisition for deep bayesian active learning," in *Advances in Neural Information Processing Systems*, 2019, pp. 7024–7035.
- [6] C. Blundell, J. Cornebise, K. Kavukcuoglu, and D. Wierstra, "Weight uncertainty in neural network," in *International Conference on Machine Learning*, 2015, pp. 1613–1622.
- [7] Y. Gal and Z. Ghahramani, "Dropout as a bayesian approximation: Representing model uncertainty in deep learning," in *international conference on machine learning*, 2016, pp. 1050–1059.
- [8] O. Sener and S. Savarese, "Active learning for convolutional neural networks: A core-set approach," in *6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings*. OpenReview.net, 2018. [Online]. Available:<https://openreview.net/forum?id=H1aIuk-RW>
- [9] D. Golovin, A. Krause, and D. Ray, "Near-optimal bayesian active learning

- with noisy observations," in *Advances in Neural Information Processing Systems*, 2010, pp. 766–774.
- [10] K. Jedoui, R. Krishna, M. Bernstein, and L. Fei-Fei, "Deep bayesian active learning for multiple correct outputs," *arXiv preprint arXiv:1912.01119*, 2019.
- [11] A. Doucet, S. Godsill, and C. Andrieu, "On sequential monte carlo sampling methods for bayesian filtering," *Statistics and computing*, vol. 10, no. 3, pp. 197–208, 2000.
- [12] S. A. Vavasis, "Approximation algorithms for indefinite quadratic programming," *Mathematical Programming*, vol. 57, no. 1-3, pp. 279–311, 1992.
- [13] N. Houlsby, F. Huszár, Z. Ghahramani, and M. Lengyel, "Bayesian active learning for classification and preference learning," *arXiv preprint arXiv:1112.5745*, 2011.
- [14] R. W. Strachan and H. K. Van Dijk, "Bayesian model selection with an

Image /page/13/Figure/2 description: The figure contains three small-multipanel heatmaps (colored matrices) side-by-side labeled (a) MNIST, (b) SVHN, and (c) CIFAR10. Each panel is a 6×6 matrix of pairwise numeric values between six selection methods (row and column labels, in red): Var, BALD, Entropy, k-medoids, k-centers, GBALD. Cells are colored from dark blue (small) to yellow (large) and each cell is annotated with a red numeric value. Diagonal cells are 0. Each panel also shows a row of red numbers beneath the matrix (column totals) and a column of red numbers at the right of the matrix (row totals). The numeric values printed in the figure (as extracted) are as follows.

(a) MNIST (methods in order Var, BALD, Entropy, k-medoids, k-centers, GBALD):
- Matrix rows (annotated cell values):
 Row Var: 0.0246 0 0.2131 0.0757 0.1860 1.1148
 Row BALD: 0.0706 0.0988 0 0.0357 0.2646 0.4736
 Row Entropy: 0.0964 0.1167 0.1005 0 0.1336 1.7922
 Row k-medoids: 0.1112 0.0782 0.0715 0.1258 0.1456 0
 (the figure shows the remaining rows/values visually; the annotated column totals across bottom are:) 
- Column totals (bottom, red): 2.5053 3.0837 4.8024 0.5782 0.8843 18.7898

(b) SVHN (same method order):
- Matrix rows (annotated cell values):
 Row Var: 0.4668 0 0.4910 0.9680 0.1289 11.2565
 Row BALD: 0.0339 0.0670 0 0.0138 0.2069 16.9627
 Row Entropy: 0.0276 0.0721 0.0111 0 0.1925 0
 Row k-medoids: 0 2.0198 0 6.2881 0 26.2869
 Row k-centers: 0.2273 0.1126 0.1696 0.1052 0.2629 0
- Column totals (bottom, red): 0.7556 2.3351 0.7023 7.4051 0.9958 77.2318

(c) CIFAR10 (same method order):
- Matrix rows (annotated cell values):
 Row Var: 1.8673 0 1.5031 2.8668 0.0627 2.3121
 Row BALD: 0.7040 0.0959 0 1.4250 0.0709 3.112
 Row Entropy: 0.0205 0.1669 0.1155 0 0.1234 1.4611
 Row k-medoids: 0 0.2433 2.2930 0.01251 0
 Row k-centers: 0.1292 0.3696 0.2713 0.0769 0.1268 0
- Column totals (bottom, red): 2.721 1.1112 4.3284 4.4366 0.4728 9.9326

Visual details: each panel has method labels on both the left (rows) and top (columns) in red; each cell is annotated with its numeric value in red; diagonal entries are zero; color intensity increases with larger values (yellow indicates largest inter-method distances such as large GBALD-related cells in MNIST/SVHN). Panels are titled below as (a) MNIST, (b) SVHN, (c) CIFAR10.

Figure 12: A pairwise penalty matrix over active acquisitions on MNIST, SVHN, and CIFAR10. Column-wise values at the bottom of each matrix show the overall performance of the compared baselines (larger value has more significant superior performance).

- uninformative prior," *Oxford Bulletin of Economics and Statistics*, vol. 65, pp. 863–876, 2003.
- [15] H. J. Price and A. R. Manson, "Uninformative priors for bayes' theorem," in *AIP Conference Proceedings*, vol. 617, no. 1, 2002, pp. 379–391.
- [16] M. Gao, Z. Zhang, G. Yu, S. O. Arik, L. S. Davis, and T. Pfister, "Consistency-based semi-supervised active learning: Towards minimizing labeling cost," *ECCV*, 2020.
- [17] T. Campbell and T. Broderick, "Bayesian coreset construction via greedy iterative geodesic ascent," in *International Conference on Machine Learning*, 2018, pp. 698–706.
- [18] F. Nie, H. Wang, H. Huang, and C. Ding, "Early active learning via robust representation and structured sparsity," in *Twenty-Third International Joint Conference on Artificial Intelligence*, 2013.
- [19] Z. Wang, B. Du, W. Tu, L. Zhang, and D. Tao, "Incorporating distribution matching into uncertainty for multiple kernel active learning," *IEEE Transactions on Knowledge and Data Engineering*, 2019.
- [20] V. Perrone, H. Shen, M. W. Seeger, C. Archambeau, and R. Jenatton, "Learning search spaces for bayesian optimization: Another view of hyperparameter transfer learning," in *Advances in Neural Information Processing Systems*, 2019, pp. 12 771–12 781.
- [21] M. Sugiyama, "Active learning in approximately linear regression based on conditional expectation of generalization error," *Journal of Machine Learning Research*, vol. 7, no. Jan, pp. 141–166, 2006.
- [22] D. Cohn, L. Atlas, and R. Ladner, "Improving generalization with active learning," *Machine learning*, vol. 15, no. 2, pp. 201–221, 1994.
- [23] G. Schohn and D. Cohn, "Less is more: Active learning with support vector machines," in *ICML*, vol. 2, no. 4. Citeseer, 2000, p. 6.
- [24] D. D. Lewis and W. A. Gale, "A sequential algorithm for training text classifiers," in *SIGIR'94*. Springer, 1994, pp. 3–12.
- [25] T. Scheffer, C. Decomain, and S. Wrobel, "Active hidden markov models for information extraction," in *International Symposium on Intelligent Data Analysis*. Springer, 2001, pp. 309–318.
- [26] N. Roy and A. McCallum, "Toward optimal active learning through monte carlo estimation of error reduction," *ICML, Williamstown*, pp. 441–448, 2001.
- [27] K. Yu, J. Bi, and V. Tresp, "Active learning via transductive experimental design," in *Proceedings of the 23rd international conference on Machine learning*, 2006, pp. 1081–1088.
- [28] A. Krishnamurthy, A. Agarwal, T.-K. Huang, H. Daumé III, and J. Langford, "Active learning for cost-sensitive classification." *Journal of Machine Learning Research*, vol. 20, no. 65, pp. 1–50, 2019.
- [29] S. Sinha, S. Ebrahimi, and T. Darrell, "Variational adversarial active learning," in *Proceedings of the IEEE International Conference on Computer Vision*, 2019, pp. 5972–5981.
- [30] S. Hanneke, "A bound on the label complexity of agnostic active learning," in *Proceedings of the 24th international conference on Machine learning*, 2007, pp. 353–360.
- [31] S. Javdani, Y. Chen, A. Karbasi, A. Krause, D. Bagnell, and S. S. Srinivasa, "Near optimal bayesian active learning for decision making." in *AISTATS*, vol. 14, 2014, pp. 430–438.
- [32] M. Tang, X. Luo, and S. Roukos, "Active learning for statistical natural language parsing," in *Proceedings of the 40th Annual Meeting on Association for Computational Linguistics*. Association for Computational Linguistics, 2002, pp. 120–127.

- [33] A. Siddhant and Z. C. Lipton, "Deep bayesian active learning for natural language processing: Results of a large-scale empirical study," in *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, 2018, pp. 2904–2909.
- [34] S. Burkhardt, J. Siekiera, and S. Kramer, "Semisupervised bayesian active learning for text classification," in *Bayesian Deep Learning Workshop at NeurIPS*, 2018.
- [35] T. Tran, T.-T. Do, I. Reid, and G. Carneiro, "Bayesian generative active deep learning," in *International Conference on Machine Learning*, 2019, pp. 6295–6304.
- [36] T. Campbell and T. Broderick, "Automated scalable bayesian inference via hilbert coresets," *The Journal of Machine Learning Research*, vol. 20, no. 1, pp. 551–588, 2019.
- [37] M. Badoiu, S. Har-Peled, and P. Indyk, "Approximate clustering via ¯ core-sets," in *Proceedings of the thiry-fourth annual ACM symposium on Theory of computing*, 2002, pp. 250–257.
- [38] S. Har-Peled and S. Mazumdar, "On coresets for k-means and k-median clustering," in *Proceedings of the thirty-sixth annual ACM symposium on Theory of computing*, 2004, pp. 291–300.
- [39] A. Lou, I. Katsman, Q. Jiang, S. Belongie, S.-N. Lim, and C. De Sa, "Differentiating through the frechet mean," *ICML*, 2020.
- [40] S. Ben-David and U. Von Luxburg, "Relating clustering stability to properties of cluster boundaries," in *21st Annual Conference on Learning Theory (COLT 2008)*. Omnipress, 2008, pp. 379–390.
- [41] W. Li, G. Dasarathy, K. Natesan Ramamurthy, and V. Berisha, "Finding the homology of decision boundaries with active learning," *Advances in Neural Information Processing Systems*, vol. 33, 2020.
- [42] N. Barnea, "Hyperspherical functions with arbitrary permutational symmetry: Reverse construction," *Physical Review A*, vol. 59, no. 2, p. 1135, 1999.
- [43] L. Blumenson, "A derivation of n-dimensional spherical coordinates," *The American Mathematical Monthly*, vol. 67, no. 1, pp. 63–66, 1960.
- [44] U. Kaymak and M. Setnes, "Fuzzy clustering with volume prototypes and adaptive cluster merging," *IEEE Transactions on Fuzzy Systems*, vol. 10, no. 6, pp. 705–712, 2002.
- [45] H.-S. Park and C.-H. Jun, "A simple and fast algorithm for k-medoids clustering," *Expert systems with applications*, vol. 36, no. 2, pp. 3336– 3341, 2009.
- [46] M. Osborne, R. Garnett, Z. Ghahramani, D. K. Duvenaud, S. J. Roberts, and C. E. Rasmussen, "Active learning of model evidence using bayesian quadrature," in *Advances in neural information processing systems*, 2012, pp. 46–54.
- [47] B. Han, Q. Yao, X. Yu, G. Niu, M. Xu, W. Hu, I. Tsang, and M. Sugiyama, "Co-teaching: Robust training of deep neural networks with extremely noisy labels," in *Advances in neural information processing systems*, 2018, pp. 8527–8537.
- [48] J. L. Hodges, E. L. Lehmann *et al.*, "The efficiency of some nonparametric competitors of the *t*-test," *The Annals of Mathematical Statistics*, vol. 27, no. 2, pp. 324–335, 1956.
- [49] P. Donmez, J. G. Carbonell, and P. N. Bennett, "Dual strategy active learning," in *European Conference on Machine Learning*. Springer, 2007, pp. 116–127.
- [50] J. T. Ash, C. Zhang, A. Krishnamurthy, J. Langford, and A. Agarwal, "Deep batch active learning by diverse, uncertain gradient lower bounds," in *International Conference on Learning Representations*, 2019.

Image /page/14/Picture/2 description: Figure showing two black-outlined spheres aligned vertically with a slanted purple plane labeled h between them. At top left is a smaller sphere (unshaded) with a dashed horizontal ellipse drawn inside and a small red dot near its lower interior. Below the plane is a larger sphere whose lower half (a spherical cap) is filled solid red. Inside the large sphere a dashed horizontal ellipse marks the internal equatorial plane; a small dark circular marker sits at the center of the red cap region and a small square marker (colored dark/blue) appears in the upper interior of the large sphere, labeled q\*. To the right of the image a short purple horizontal tick and a downward arrow mark a vertical distance labeled d\_a measured toward the bottom of the large sphere. A legend in the upper right shows markers with labels: q\_a as a black dot, q\_b as a red dot, q\* as a small square, and a red semicircular icon labeled “cut C.” The overall composition illustrates the plane h cutting the larger sphere to produce the red cut C (a spherical cap) and shows marker positions and the distance d\_a.

Figure 13: Assumption of the generalization analysis. The ball above h denotes  $S_b$ , the ball below h denotes  $S_a$ , and  $R_b < R_a$ . Spherical cap<sup>1</sup> of the half-sphere of  $S_a$  is denoted as  $\mathcal{D}$ .

## A.1 Case study of generalization analysis of err(h,3) of geodesic search with sphere

**Theorem 5.** Given a perceptron function  $h = w_1x_1 + w_2x_2 + w_3$  that classifies A and B, and a sampling budget k. By drawing core-set on  $S_a$  and  $S_b$ , the minimum distance to the boundaries of that core-set elements of  $S_A$  and  $S_B$ , are defined as  $d_a$  and  $d_b$ , respectively. Let err(h,k) be the classification error rate with respect to h and k, given  $\frac{\pi}{\varphi} = \arcsin \frac{R_a - d_a}{R_a}$ , we have an inequality of error:

$$\min \left\{ \frac{(2R_a + t)(R_a - t)^2}{4R_a^3 + 4R_b^3}, \frac{(2R_b + t')(R_b - t')^2}{4R_b^3 + 4R_a^3} \right\} < err(h, 3) < 0.3334,$$

*where*

$$t = \frac{R_a^{2}}{3} + \sqrt[3]{-\frac{\mu}{2\pi} + \sqrt{\frac{\mu^{2}}{4\pi^{2}} - \frac{R_a^{3}}{27}}} + \sqrt[3]{-\frac{\mu}{2\pi} - \sqrt{\frac{\mu^{2}}{4\pi^{2}} - \frac{R_a^{3}}{27}}}$$

$$\mu = \left(\frac{2}{9} - \frac{1}{\varphi}\cos\frac{\pi}{\varphi}\right)\pi R_a^{3} - \frac{4\pi R_b^{3}}{9}$$

$$t' = \frac{R_b^{2}}{3} + \sqrt[3]{-\frac{\mu'}{2\pi} + \sqrt{\frac{{\mu'}^{2}}{4\pi^{2}} - \frac{R_b^{3}}{27}}} + \sqrt[3]{-\frac{\mu'}{2\pi} - \sqrt{\frac{{\mu'}^{2}}{4\pi^{2}} - \frac{R_b^{3}}{27}}}$$

$$\mu' = \left(\frac{2}{9} - \frac{1}{\varphi}\cos\frac{\pi}{\varphi}\right)\pi R_b^{3} - \frac{4\pi R_a^{3}}{9}$$

*Proof.* Given the unseen acquisitions of  $\{q_a, q_b, q^*\}$ , where  $q_a \in A$ ,  $q_b \in B$ , and  $q^* \in A$  or  $q^* \in B$  is uncertain. However, the position of  $q^*$  largely decides h. Therefore, the proof studies the error bounds highly related to  $q^*$  in terms of two cases:  $R_a \ge R_b$  and  $R_a < R_b$ .

1) If  $R_a \ge R_b$ ,  $q^* \in A$ . Estimating the position of  $q^*$  starts from the analysis on  $q_a$ . Given the volume function  $\operatorname{Vol}(\cdot)$  over the 3-D geometry, we know:  $\operatorname{Vol}(A) = \frac{4\pi}{3} R_a^3$  and  $\operatorname{Vol}(B) = \frac{4\pi}{3} R_b^3$ . Given k = 3 over  $S_a$  and  $S_b$ , we define the minimum distance of  $q_a$  to the boundary of A as  $d_a$ . Let  $S_a$  be cut off by a cross section h', where C be the cut and D be the *spherical cap* of the half-sphere (see Figure 13) that satisfy

$$Vol(C) = \frac{2}{3}\pi R_a^3 - Vol(D) = \frac{4\pi (R_a^3 + R_b^3)}{9},$$
(19)

and the volume of  $\mathcal{D}$  is

$$\operatorname{Vol}(\mathcal{D}) = \pi\left(\sqrt{R_a^2 - (R_a - d_a)^2}\right)^2 (R_a - d_a) + \int_{0}^{2\pi\sqrt{R_a^2 - (R_a - d_a)^2}} \frac{\arcsin\left(\frac{R_a - d_a}{R_a}\right)}{2\pi}\,\pi R_a^2\, dx$$

$$= \pi\left(R_a^2 - (R_a - d_a)^2\right)(R_a - d_a) + \frac{\arcsin\left(\frac{R_a - d_a}{R_a}\right)}{2\pi}\,\pi R_a^2\left(2\pi\sqrt{R_a^2 - (R_a - d_a)^2}\right)$$

$$= \pi\left(R_a^2 - (R_a - d_a)^2\right)(R_a - d_a) + \pi R_a^2\,\arcsin\left(\frac{R_a - d_a}{R_a}\right)\sqrt{R_a^2 - (R_a - d_a)^2}.$$
(20)

Let  $\frac{\pi}{\varphi} = \arcsin \frac{R_a - d_a}{R_a}$ , Eq. (20) can be written as

$$\operatorname{Vol}(\mathcal{D}) = \pi (R_a^2 - (R_a - d_a)^2)(R_a - d_a) + \frac{\pi R_a^3}{\varphi} \cos \frac{\pi}{\varphi}. \tag{21}$$

Introducing Eq. (21) to Eq. (19), we have

$$\left(\frac{2}{3} - \frac{1}{\varphi}\cos\frac{\pi}{\varphi}\right)\pi R_a^3 - \pi (R_a^2 - (R_a - d_a)^2)(R_a - d_a) = \frac{4\pi (R_a^3 + R_b^3)}{9}.$$
 (22)

Let  $t = R_a - d_a$ , Eq. (22) can be rewritten as

$$\pi t^3 - \pi R_a^2 t + \left(\frac{2}{9} - \frac{1}{\varphi} \cos \frac{\pi}{\varphi}\right) \pi R_a^3 - \frac{4\pi R_b^3}{9} = 0.$$
 (23)

To simplify Eq. (23), let  $\mu$  =  $(\frac{2}{9} - \frac{1}{\varphi}cos\frac{\pi}{\varphi})\pi R_a^3 - \frac{4\pi R_b^3}{9}$ , Eq. (23) then can be written as

$$\pi t^3 - \pi R_a^2 t + \mu = 0. (24)$$

The positive solution of t can be

$$t = \frac{R_a^2}{3} + \sqrt[3]{-\frac{\mu}{2\pi} + \sqrt{\frac{\mu^2}{4\pi^2} - \frac{\pi^3 R_a^3}{27\pi^3}}} + \sqrt[3]{-\frac{\mu}{2\pi} - \sqrt{\frac{\mu^2}{4\pi^2} - \frac{\pi^3 R_a^3}{27\pi^3}}}.$$
 (25)

Based on Eq. (19), we know

$$\operatorname{Vol}(\mathcal{D}) = \frac{2}{3}\pi R_a^3 - \frac{4\pi (R_a^3 + R_b^3)}{9}$$
$$= \frac{2}{9}\pi R_a^3 - \frac{4}{9}\pi R_b^3 > 0.$$
 (26)

Thus,  $\sqrt[3]{2}R_b < R_a$ . We next prove  $q^* \in A$ . Based on Eq. (26), we know

$$\pi R_b^3 < \frac{1}{2}\pi R_a^3. \tag{27}$$

Then, the following inequalities hold: 1) $2\pi R_b^3 < \pi R_a^3$ , 2)  $\frac{2}{3}\pi R_b^3 < \frac{1}{3}\pi R_a^3$ , and 3)  $\frac{2}{3}\pi R_b^3 + \frac{1}{3}\pi R_b^3 < \frac{1}{3}\pi R_a^3 + \frac{1}{3}\pi R_b^3$ . Finally, we have

$$\pi R_b^3 < \frac{1}{3} (\pi R_a^3 + \pi R_b^3). \tag{28}$$

Therefore,  $Vol(B) < \frac{1}{3}(Vol(A) + Vol(B))$ . We thus know: 1)  $q_a \in A$  and it is with a minimum distance  $d_a$  to the boundary of  $S_a$ , 2)  $q_b \in B$ , and 3)  $q^* \in A$ . Therefore, class B can be deemed as having a very high probability to achieve a nearly zero generalization error and the position of  $q^*$  largely decides the upper bound of the generalization error of h.

In  $S_a$  that covers class A, the nearly optimal error region can be bounded as the spherical cap of  $S_a$  with a volume constraint of Vol(A) - Vol(C). We thence have an inequality of

$$\frac{\operatorname{Vol}(A) - \operatorname{Vol}(C)}{\operatorname{Vol}(A) + \operatorname{Vol}(B)} < err(h, 3) < \frac{1}{3}.$$
(29)

We next calculate the volume of the spherical cap:

$$\mathrm{Vol}(A)-\mathrm{Vol}(C)=\int_{-R_a}^{\,R_a-d_a} \pi x^{2}\,dy$$

$$= \pi \int_{R_a-d_a}^{\,R_a} \left(R_a^{2}-y^{2}\right)\,dy$$

$$= \frac{4}{3}\pi R_a^{3} - \pi d_a^{2}\left(R_a - \frac{d_a}{3}\right)$$
(30)

Eq. (29) then is rewritten as

$$\frac{\frac{4}{3}\pi R_a^3 - \frac{\pi}{3}(3R_a - d_a)d_a^2}{\frac{4}{3}\pi R_a^3 + \frac{4}{3}\pi R_b^3} < err(h,3) < 0.3334,\tag{31}$$

Then, we have the error bound of

$$\frac{4R_a^3 - (3R_a - d_a)d_a^2}{4R_a^3 + 4R_b^3} < err(h,3) < 0.3334.$$
(32)

Introducing  $d_a = R_a - t$ , Eq. (32) is written as

$$\frac{4R_a^3 - (2R_a + t)(R_a - t)^2}{4R_a^3 + 4R_b^3} < err(h, 3) < 0.3334.$$
(33)

2) With another assumption of  $R_a < R_b$ , we follow the same proof skills of  $R_a \ge R_b$  and know  $\frac{4R_b^3 - (3R_b - d_b)d_b^2}{4R_b^3 + 4R_a^3} < err(h, 3) < \frac{4R_b^3 - (3R_b - d_b)d_b^2}{4R_b^3 + 4R_a^3} < err(h, 3) < \frac{4R_b^3 - (3R_b - d_b)d_b^2}{4R_b^3 + 4R_a^3} < err(h, 3) < \frac{4R_b^3 - (3R_b - d_b)d_b^2}{4R_b^3 + 4R_a^3} < err(h, 3) < \frac{4R_b^3 - (3R_b - d_b)d_b^2}{4R_b^3 + 4R_a^3} < err(h, 3) < \frac{4R_b^3 - (3R_b - d_b)d_b^2}{4R_b^3 + 4R_a^3} < err(h, 3) < \frac{4R_b^3 - (3R_b - d_b)d_b^2}{4R_b^3 + 4R_a^3} < err(h, 3) < \frac{4R_b^3 - (3R_b - d_b)d_b^2}{4R_b^3 + 4R_a^3} < err(h, 3) < \frac{4R_b^3 - (3R_b - d_b)d_b^2}{4R_b^3 + 4R_a^3} < err(h, 3) < \frac{4R_b^3 - (3R_b - d_b)d_b^2}{4R_b^3 + 4R_b^3} < err(h, 3) < \frac{4R_b^3 - (3R_b - d_b)d_b^2}{4R_b^3 + 4R_b^3} < err(h, 3) < \frac{4R_b^3 - (3R_b - d_b)d_b^2}{4R_b^3 + 4R_b^3} < err(h, 3) < \frac{4R_b^3 - (3R_b - d_b)d_b^2}{4R_b^3 + 4R_b^3} < err(h, 3) < \frac{4R_b^3 - (3R_b - d_b)d_b^2}{4R_b^3 + 4R_b^3} < err(h, 3) < \frac{4R_b^3 - (3R_b - d_b)d_b^2}{4R_b^3 + 4R_b^3} < err(h, 3) < \frac{4R_b^3 - (3R_b - d_b)d_b^2}{4R_b^3 + 4R_b^3} < err(h, 3) < \frac{4R_b^3 - (3R_b - d_b)d_b^2}{4R_b^3 + 4R_b^3} < err(h, 3) < \frac{4R_b^3 - (3R_b - d_b)d_b^2}{4R_b^3 + 4R_b^3} < err(h, 3) < \frac{4R_b^3 - (3R_b - d_b)d_b^2}{4R_b^3 + 4R_b^3} < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h, 3) < err(h,$ 

0.3334, i.e. where 
$$d_b = R_b - t'$$
 and  $t' = \frac{R_b^2}{3} + \sqrt[3]{-\frac{\mu'}{2\pi} + \sqrt{\frac{\mu'^2}{4\pi^2} - \frac{\pi^3 R_b^3}{27\pi^3}}} + \sqrt[3]{-\frac{\mu'}{2\pi} - \sqrt{\frac{\mu'^2}{4\pi^2} - \frac{\pi^3 R_b^3}{27\pi^3}}}$ , and  $\mu' = (\frac{2}{9} - \frac{1}{\varphi} \cos \frac{\pi}{\varphi})\pi R_b^3 - \frac{4\pi R_a^3}{9}$ .

We thus conclude that 
$$\min \left\{ \frac{4R_a^3 - (2R_a + t)(R_a - t)^2}{4R_a^3 + 4R_b^3}, \frac{4R_b^3 - (2R_b + t')(R_b - t')^2}{4R_b^3 + 4R_a^3} \right\} < err(h, 3) < 0.3334.$$

## A.2 Specification of Assumption 1

In clustering stability,  $\gamma$ -tube structure that surrounds the cluster boundary, largely decides the performance of a learning algorithm. Definition of  $\gamma$ -tube is as follows.

**Definition 1.**  $\gamma$ -tube Tube $_{\gamma}(f)$  is a set of points distributed in the boundary of the cluster.

Tube<sub>$$\gamma$$</sub> $(f) := \{x \in X | \ell(x, B(f)) \le \gamma\},$  (34)

where X is a noise-free cluster with n samples,  $B(f) := \{x \in X, f \text{ is discontinuous at } x \}$ , f is a clustering function, and  $\ell(\cdot, \cdot)$  denotes the distance function.

Following this conclusion, representation data can achieve the optimal generalization error if they are spread over the tube structure. Let  $\gamma = d_a$ , the probability of achieving a nearly zero generalization error can be expressed as the volume ration of  $\gamma$ -tube and  $S_a$ :

$$\Pr[\mathrm{err}(h,k)=0]_{\mathrm{Sphere}} = \frac{\mathrm{Vol}(\mathrm{Tube}_{\gamma})}{\mathrm{Vol}(S_a)}$$

$$= \frac{\frac{4}{3}\pi R_a^{3} - \frac{4}{3}\pi (R_a - d_a)^{3}}{\frac{4}{3}\pi R_a^{3}}$$

$$= 1 - \frac{t_k^{3}}{R_a^{3}},$$
(35)

where  $t_k$  keeps consistent with Eq. (40). With the initial sampling from the tube structure of class A, the subsequent acquisitions of AL would be updated from the tube structure of class B. If the initial sampling comes from the tube structure of B, the next acquisition must be updated from the tube structure of A. With the updated acquisitions spread over the tube structures of both classes, h is easy to achieve a nearly zero error.

## A.3 Specification of Assumption 2

Following the specification of Assumption 1, volume of the tube is redefined as  $Vol(Tube_{\gamma}) = \frac{4}{3}\pi R_{a_1}R_{a_2}R_{a_3}$ . Then, we know

$$\Pr[\mathrm{err}(h,k)=0]_{\mathrm{Ellipsoid}} = \frac{\operatorname{Vol}(\mathrm{Tube}_{\gamma})}{\operatorname{Vol}(E_{a})}$$

$$= \frac{\frac{4}{3}\pi R_{a_{1}}R_{a_{2}}R_{a_{3}} - \frac{4}{3}\bigl(R_{a_{1}}-d_{a}\bigr)\bigl(R_{a_{2}}-d_{a}\bigr)\bigl(R_{a_{3}}-d_{a}\bigr)}{\frac{4}{3}\pi R_{a_{1}}R_{a_{2}}R_{a_{3}}}$$

$$= 1 - \frac{\lambda_{k_{1}}\,\lambda_{k_{2}}\,\lambda_{k_{3}}}{R_{a_{1}}R_{a_{2}}R_{a_{3}}},$$

where  $\lambda_{k_{i}}$  $=\dfrac{R_{a_{i}}^{2}}{3} + \sqrt[3]{-\dfrac{\sigma_{k_{i}}}{2\pi} + \sqrt{\dfrac{\sigma_{k_{i}}^{2}}{4\pi^{2}} - \dfrac{\pi^{3}R_{a_{i}}^{3}}{27}}} + \sqrt[3]{-\dfrac{\sigma_{k_{i}}}{2\pi} - \sqrt{\dfrac{\sigma_{k_{i}}^{2}}{4\pi^{2}} - \dfrac{\pi^{3}R_{a_{i}}^{3}}{27}}}},$   $\text{and}\right.$ 

$$\sigma_{k_{i}} = \left(\dfrac{2k-4}{3k} - \dfrac{\pi R_{a_{i}}}{2\varphi}\right)\pi\,R_{a_{1}}R_{a_{2}}R_{a_{3}} - \dfrac{4\pi\,R_{b_{1}}R_{b_{2}}R_{b_{3}}}{3k}, \quad i=1,2,3.$$

## A.4 Proof of Theorem 1

We next present the generalization errors against an agnostic sampling budget k following the above proof technique.

*Proof.* The proof studies two cases:  $R_a \ge R_b$  and  $R_a < R_b$ . 1) If  $R_a \ge R_b$ , we estimate the optimal position of  $q_a$  that satisfies  $q_a \in A$ . Given the volume function  $\operatorname{Vol}(\cdot)$  over the 3-D geometry, we know:  $\operatorname{Vol}(A) = \frac{4\pi}{3}R_a^3$  and  $\operatorname{Vol}(B) = \frac{4\pi}{3}R_b^3$ . Assume  $q_a$  be the nearest representative data to the boundary of  $S_a$ ,  $q_b$  be the nearest representative data to the boundary of  $S_b$ , and  $q^*$  be the nearest representative data to h either in  $S_a$  or  $S_b$ . Given the minimum distance of  $q_a$  to the boundary of A as  $d_a$ . Let  $S_a$  be cut off by a cross section h', where C be the cut and D be the spherical cap of the half-sphere that satisfy

$$Vol(C) = \frac{2}{3}\pi R_a^3 - Vol(D) = \frac{4\pi (R_a^3 + R_b^3)}{3k},$$
(37)

and the volume of  $\mathcal{D}$  is

$$\operatorname{Vol}(\mathcal{D}) = \pi\left(\sqrt{R_a^2 - (R_a - d_a)^2}\right)^2 (R_a - d_a) + \int_0^{2\pi\sqrt{R_a^2 - (R_a - d_a)^2}} \frac{\arcsin\left(\frac{R_a - d_a}{R_a}\right)}{2\pi} \,\pi R_a^2 \,dx$$

$$= \pi\left(R_a^2 - (R_a - d_a)^2\right)(R_a - d_a) + \frac{\arcsin\left(\frac{R_a - d_a}{R_a}\right)}{2\pi} \pi R_a^2 \left(2\pi\sqrt{R_a^2 - (R_a - d_a)^2}\right)$$

$$= \pi\left(R_a^2 - (R_a - d_a)^2\right)(R_a - d_a) + \pi R_a^2 \arcsin\left(\frac{R_a - d_a}{R_a}\right) \sqrt{R_a^2 - (R_a - d_a)^2}.$$
(38)

Let  $\frac{\pi}{\varphi} = \arcsin \frac{R_a - d_a}{R_a}$ , Eq. (36) can be written as

$$\operatorname{Vol}(\mathcal{D}) = \pi (R_a^2 - (R_a - d_a)^2)(R_a - d_a) + \frac{\pi R_a^3}{\varphi} \cos \frac{\pi}{\varphi}. \tag{39}$$

Introducing Eq. (36) to Eq. (34), we have

$$\left(\frac{2}{3} - \frac{1}{\varphi}\cos\frac{\pi}{\varphi}\right)\pi R_a^3 - \pi (R_a^2 - (R_a - d_a)^2)(R_a - d_a) = \frac{4\pi (R_a^3 + R_b^3)}{3k}.$$
 (40)

Let  $t_k = R_a - d_a$ , we know

$$\pi t^3 - \pi R_a^2 t + \left(\frac{2k-4}{3k} - \frac{1}{\varphi} \cos\frac{\pi}{\varphi}\right) \pi R_a^3 - \frac{4\pi R_b^3}{3k} = 0.$$
 (41)

To simplify Eq. (38), let  $\mu_k = \left(\frac{2k-4}{3k} - \frac{1}{\varphi}\cos\frac{\pi}{\varphi}\right)\pi R_a^3 - \frac{4\pi R_b^3}{3k}$ , Eq. (38) then can be written as

$$\pi t^3 - \pi R_a^2 t + \mu_k = 0. (42)$$

The positive solution of  $t_k$  can be

$$t_k = \frac{R_a^2}{3} + \sqrt[3]{-\frac{\mu_k}{2\pi} + \sqrt{\frac{\mu_k^2}{4\pi^2} - \frac{\pi^3 R_a^3}{27\pi^3}}} + \sqrt[3]{-\frac{\mu_k}{2\pi} - \sqrt{\frac{\mu_k^2}{4\pi^2} - \frac{\pi^3 R_a^3}{27\pi^3}}}.$$
 (43)

Based on Eq. (35), we know

$$\operatorname{Vol}(\mathcal{D}) = \frac{2}{3}\pi R_a^3 - \frac{4\pi (R_a^3 + R_b^3)}{3k}$$
$$= \frac{2k - 4}{3k}\pi R_a^3 - \frac{4}{3k}\pi R_b^3 > 0. \tag{44}$$

Thus,  $\sqrt[3]{\frac{2}{k-2}}R_b < R_a$ . We next prove  $q^* \in A$ . According to Eq. (41), we know

$$\pi R_b^3 < \frac{k-2}{2} \pi R_a^3. \tag{45}$$

Then, the following inequalities hold:  $1)\frac{2}{k-2}\pi R_b^3 < \pi R_a^3$ ,  $2)\frac{2}{(k-2)k}\pi R_b^3 < \frac{1}{k}\pi R_a^3$ , and  $3)\frac{2}{(k-2)k}\pi R_b^3 + \frac{k^2-2k-2}{(k-2)k}\pi R_b^3 < \frac{1}{k}\pi R_a^3 + \frac{k^2-2k-2}{(k-2)k}\pi R_b^3$ . Finally, we have:

$$\pi R_b^{3}$$

$$< \frac{1}{k}\pi R_a^{3} + \frac{k^{2}-2k-2}{(k-2)k}\pi R_b^{3}$$

$$= \frac{1}{k}\pi\left(R_a^{3} + R_b^{3}\right) - \frac{2}{(k-2)k}\pi R_a^{3}$$

$$< \frac{1}{k}\pi\left(R_a^{3} + R_b^{3}\right).$$
(46)

Therefore,  $Vol(B) < \frac{1}{k}(Vol(A) + Vol(B))$ . We thus know: 1)  $q_a \in A$  and it is with a minimum distance  $d_a$  to the boundary of  $S_a$ , 2)  $q_b \in B$ , and 3)  $q^* \in A$ . Therefore, class B can be deemed as having a very high probability to achieve a zero generalization error and the position of  $q^*$  largely decides the upper bound of the generalization error of h.

In  $S_a$  that covers class A, the nearly optimal error region can be bounded as Vol(A) - Vol(C). We then have the inequality of

$$\frac{\operatorname{Vol}(A) - \operatorname{Vol}(C)}{\operatorname{Vol}(A) + \operatorname{Vol}(B)} < err(h, k) < \frac{1}{k}.$$
(47)

Based on the volume equation of the spherical cap in Eq. (30), we have

$$\frac{\frac{4}{3}\pi R_a^3 - \frac{\pi}{3}(3R_a - d_a)d_a^2}{\frac{4}{3}\pi R_a^3 + \frac{4}{3}\pi R_b^3} < err(h, k) < \frac{1}{k}.$$
(48)

Then, we have the error bound of

$$\frac{4R_a^3 - (3R_a - d_a)d_a^2}{4R_a^3 + 4R_b^3} < err(h, k) < \frac{1}{k}.$$
(49)

Introducing  $d_a = R_a - t_k$ , Eq. (46) is written as

$$\frac{4R_a^3 - (2R_a + t)(R_a - t_k)^2}{4R_a^3 + 4R_b^3} < err(h, k) < \frac{1}{k}.$$
 (50)

2) With another assumption of  $R_a < R_b$ , we follow the same proof skills of  $R_a \ge R_b$  and know  $\frac{4R_b^3 - (3R_b - d_b)d_b^2}{4R_b^3 + 4R_a^3} < err(h, k) < \frac{1}{k}$ , where  $d_b = R_b - t_k'$  and  $t_k' = \frac{R_b^2}{3} + \sqrt[3]{-\frac{\mu_k'}{2\pi} + \sqrt{\frac{\mu_k'^2}{4\pi^2} - \frac{\pi^3 R_b^3}{27\pi^3}}} + \sqrt[3]{-\frac{\mu_k'}{2\pi} - \sqrt{\frac{\mu_k'^2}{4\pi^2} - \frac{\pi^3 R_b^3}{27\pi^3}}}$ , and  $\mu_k' = (\frac{2k-4}{3k} - \frac{1}{\varphi}cos\frac{\pi}{\varphi})\pi R_b^3 - \frac{4\pi R_a^3}{3k}$ .

ere 
$$d_b = R_b - t_k'$$
 and  $t_k' = \frac{R_b^2}{3} + \sqrt[3]{-\frac{\mu_k'}{2\pi}} + \sqrt{\frac{\mu_k'^2}{4\pi^2} - \frac{\pi^3 R_b^3}{27\pi^3}} + \sqrt[3]{-\frac{\mu_k'}{2\pi}} - \sqrt{\frac{\mu_k'^2}{4\pi^2} - \frac{\pi^3 R_b^3}{27\pi^3}}$ , and  $\mu_k' = (\frac{2k-4}{3k} - \frac{1}{\varphi} cos \frac{\pi}{\varphi})\pi R_b^3 - \frac{4\pi R_a^3}{3k}$ .

We thus conclude that  $\min \left\{ \frac{4R_a^3 - (2R_a + t_k)(R_a - t_k)^2}{4R_a^3 + 4R_b^3}, \frac{4R_b^3(2R_b + t_k')(R_b - t_k')^2}{4R_b^3 + 4R_a^3} \right\} < err(h, k) < \frac{1}{k}$ .

## A.5 Proof of Theorem 2

*Proof.* Given class A and B are tightly covered by ellipsoid  $E_a$  and  $E_b$  in a three-dimensional geometry. Let  $R_{a_1}$  be the polar radius of  $E_a$ ,  $\{R_{a_2}, R_{a_3}\}$  be the equatorial radii of  $E_a$ ,  $R_{b_1}$  be polar radius of  $E_b$ , and  $\{R_{b_2}, R_{b_3}\}$  be the equatorial radii of  $E_b$ . Based on Eq. (10), we know  $R_{a_i} < R_a, R_{b_i} < R_b, \forall i$ , where  $R_a$  and  $R_b$  are the radii of the spheres over the class A and B, respectively. We follow the same proof technique of Theorem 1 to present the generalization errors of AL with ellipsoid.

The proof studies two cases:  $R_{a_1} \ge R_b$  and  $R_{a_1} < R_{b_1}$ . 1) If  $R_{a_1} \ge R_{b_1}$ ,  $q^* \in A$ . Given the volume function  $\operatorname{Vol}(\cdot)$  over the 3-D geometry, we know:  $\operatorname{Vol}(A) = \frac{4\pi}{3} R_{a_1} R_{a_2} R_{a_3}$  and  $\operatorname{Vol}(B) = \frac{4\pi}{3} R_{b_1} R_{b_2} R_{b_3}$ . Given the minimum distance of  $q_a$  to the boundary of A as  $d_a$ . Let  $E_a$  by cut off by a cross section h', where  $\mathcal C$  be the cut and  $\mathcal D$  be the ellipsoid cap of the half-ellipsoid that satisfy

$$Vol(\mathcal{C}) = \frac{2}{3}\pi R_a^1 R_a^2 R_a^3 - Vol(\mathcal{D}) = \frac{4\pi (R_{a_1} R_{a_2} R_{a_3} + R_{b_1} R_{b_2} R_{b_3})}{3k},$$
(51)

and the volume of  $\mathcal{D}$  is approximated as

$$\operatorname{Vol}(\mathcal{D}) \approx \pi \left(\sqrt{R_{a_{1}}^{2} - (R_{a_{1}} - d_{a})^{2}}\right)^{2} (R_{a_{1}} - d_{a}) + \int_{0}^{\pi R_{a_{2}} R_{a_{3}}} \frac{\arcsin\left(\frac{R_{a_{1}} - d_{a}}{R_{a_{1}}}\right)}{2\pi} \pi R_{a_{1}}^{2} \, dx$$

$$= \pi \left(R_{a_{1}}^{2} - (R_{a_{1}} - d_{a})^{2}\right) (R_{a_{1}} - d_{a}) + \frac{\arcsin\left(\frac{R_{a_{1}} - d_{a}}{R_{a_{1}}}\right)}{2\pi} \pi R_{a_{1}}^{2} \left(\pi R_{a_{2}} R_{a_{3}}\right)$$

$$= \pi \left(R_{a_{1}}^{2} - (R_{a_{1}} - d_{a})^{2}\right) (R_{a_{1}} - d_{a}) + \frac{1}{2} \pi R_{a_{1}}^{2} R_{a_{2}} R_{a_{3}} \arcsin\left(\frac{R_{a_{1}} - d_{a}}{R_{a_{1}}}\right).$$

$$(52)$$

Let  $\frac{\pi}{\varphi} = \arcsin \frac{R_{a_1} - d_a}{R_{a_1}}$ , Eq. (49) can be written as

$$Vol(\mathcal{D}) = \pi (R_{a_1}^2 - (R_{a_1} - d_a)^2)(R_{a_1} - d_a) + \frac{\pi^2}{2\omega} R_{a_1}^2 R_{a_2} R_{a_3}.$$
 (53)

Introducing Eq. (50) to Eq. (48), we have

$$\left(\frac{2}{3} - \frac{\pi R_{a_1}}{2\varphi}\right)\pi R_{a_1}R_{a_2}R_{a_3} - \pi \left(R_{a_1}^2 - \left(R_{a_1} - d_a\right)^2\right)\left(R_{a_1} - d_a\right) = \frac{4\pi \left(R_{a_1}R_{a_2}R_{a_3} + R_{b_1}R_{b_2}R_{b_3}\right)}{3k}.$$
 (54)

Let  $\lambda_k = R_{a_1} - d_a$ , we know

$$\pi \lambda_k^3 - \pi R_a^2 \lambda_k + \left(\frac{2k-4}{3k} - \frac{\pi R_{a_1}}{2\omega}\right) \pi R_{a_1} R_{a_2} R_{a_3} - \frac{4\pi R_{b_1} R_{b_2} R_{b_3}}{3k} = 0.$$
 (55)

To simplify Eq. (52), let  $\sigma_k = \left(\frac{2k-4}{3k} - \frac{\pi R_{a_1}}{2\varphi}\right)\pi R_{a_1}R_{a_2}R_{a_3} - \frac{4\pi R_{b_1}R_{b_2}R_{b_3}}{3k}$ , Eq. (52) then can be written as

$$\pi \lambda_k^3 - \pi R_{a_1}^2 \lambda_k + \sigma_k = 0. \tag{56}$$

The positive solution of  $\lambda_k$  can be

$$\lambda_k = \frac{R_{a_1}^2}{3} + \sqrt[3]{-\frac{\sigma_k}{2\pi} + \sqrt{\frac{\sigma_k^2}{4\pi^2} - \frac{\pi^3 R_{a_1}^3}{27\pi^3}}} + \sqrt[3]{-\frac{\sigma_k}{2\pi} - \sqrt{\frac{\sigma_k^2}{4\pi^2} - \frac{\pi^3 R_{a_1}^3}{27\pi^3}}}.$$
 (57)

The remaining proof process follows Eq. (40) to Eq. (46) of Theorem 1. We thus conclude that

$$\min \left\{ \frac{4\prod_{i} R_{a_{i}} - (2R_{a_{1}} + \lambda_{k})(R_{a_{1}} - \lambda_{k})^{2}}{4\prod_{i} R_{a_{i}} + 4\prod_{i} R_{b_{i}}}, \frac{4\prod_{i} R_{b_{i}} - (2R_{b_{1}} + \lambda'_{k})(R_{b_{1}} - \lambda'_{k})^{2}}{4\prod_{i} R_{b_{i}} + 4\prod_{i} R_{a_{i}}} \right\} < err(h, k) < \frac{1}{k},$$

$$(58)$$

where  $i=1,2,3$ ,

$$\lambda_k^{\prime}=\frac{R_{b_1}^{2}}{3}
+\sqrt[3]{-\frac{\sigma_k}{2\pi}+\sqrt{\frac{\sigma_k^{\prime 2}}{4\pi^{2}}-\frac{R_{b_1}^{3}}{27}}}
+\sqrt[3]{-\frac{\sigma_k}{2\pi}-\sqrt{\frac{\sigma_k^{2}}{4\pi^{2}}-\frac{R_{a_1}^{3}}{27}}}$$

$$\sigma_k^{\prime}=\Big(\frac{2k-4}{3k}-\frac{\pi R_{b_1}}{2\varphi}\Big)\pi R_{b_1}R_{b_2}R_{b_3}-\frac{4\pi R_{a_1}R_{a_2}R_{a_3}}{3k}.$$
In a simple way,  $R_{a_1}R_{a_2}R_{a_3}$  and  $R_{b_1}R_{b_2}R_{b_3}$  can be written as  $\prod_{i}R_{a_i}$  and  $\prod_{i}R_{b_i}$ ,  $i=1,2,3$ , respectively.  $\square$ 

## A.6 Proof of Proposition 1

*Proof.* Let Cube<sub>a</sub> tightly covers  $S_a$  with a side length of  $2R_a$ , and Cube'<sub>a</sub> tightly covers the cut C, following theorem 1, we know

$$err(h,k) > \frac{\operatorname{Vol}(A) - \operatorname{Vol}(C)}{\operatorname{Vol}(A)} > \frac{\operatorname{Cube_a} - \operatorname{Cube'_a}}{\operatorname{Cube_a}}.$$
 (59)

Then, we know

$$\mathrm{err}(h,k) > \frac{\pi R_{a}^{3} - \pi R_{a}^{2} d_{a}}{\pi R_{a}^{3}}$$

$$= 1 - \frac{d_{a}}{R_{a}}.$$
(60)

Meanwhile, let Cube<sub>e</sub> tightly covers  $E_a$  with a side length of  $2R_{a_1}$ , Cube<sub>e</sub>' tightly covers C, following theorem 1, we know

$$err(h,k) > \frac{\operatorname{Vol}(A) - \operatorname{Vol}(C)}{\operatorname{Vol}(A)} > \frac{\operatorname{Cube_e} - \operatorname{Cube'_e}}{\operatorname{Cube_e}}.$$
 (61)

Then, we know

$$\operatorname{err}(h,k) > \frac{\pi R_{a_1} R_{a_2} R_{a_3} - \pi d_a R_{a_2} R_{a_3}}{\pi R_{a_1} R_{a_2} R_{a_3}}$$

$$= 1 - \frac{d_a}{R_{a_1}}.$$
(62)

Since  $R_{a_1} < R_a$ , we know  $1 - \frac{d_a}{R_a} > 1 - \frac{d_a}{R_{a_1}}$ . It is thus the lower bound of AL with ellipsoid is tighter than AL with sphere. Then, Proposition 1 holds.

## A.7 Proof of Proposition 2

*Proof.* Following the proofs of Theorem 3:

$$\begin{aligned}\Pr\big[\mathrm{err}(h,k)=0\big]_{\text{sphere}} &= 1 - \frac{t_k^{3}}{R_a^{3}}\\ &= 1 - \frac{(R_a - d_a)^{3}}{R_a^{3}}\\ &= 1 - \left(1 - \frac{d_a}{R_a}\right)^{3}.
\end{aligned}$$
(63)

Following the proofs Theorem 4:

$$\Pr\left[\mathrm{err}(h,k)=0\right]_{\text{Ellipsoid}} = 1 - \frac{\lambda_{k_1}\lambda_{k_2}\lambda_{k_3}}{R_{a_1}^3}$$

$$= 1 - \frac{(R_{a_1}-d_a)(R_{a_2}-d_a)(R_{a_3}-d_a)}{R_{a_1}^3}$$

$$> 1 - \left(1 - \frac{d_a}{R_{a_1}}\right)^3 .$$
(64)

Based on Proposition 1,  $1 - \frac{d_a}{R_a} > 1 - \frac{d_a}{R_{a_1}}$ , therefore  $\Pr[err(h,k) = 0]_{\text{Sphere}} < \Pr[err(h,k) = 0]_{\text{Ellipsoid}}$ . Then, Proposition 2 is as stated.

## A.8 Proof of Theorems 3 and 4

Proof of Theorems 3.

*Proof.* Given  $S_a$  over class A is defined with  $\mathbf{x_1^2} + \mathbf{x_2^2} + \mathbf{x_3^2} = R_a^2$ . Let  $\mathbf{x_1^2} + \mathbf{x_2^2} = r_a^2$  be its 2-D generalization of  $S_a$ , assume that  $x_2$  be a variable parameter in this 2-D generalization formula, the "volume" (2-D volume is the area of the geometry object) of it can be expressed as

$$Vol_{2}(S_{a}) = \int_{-r_{a}}^{r_{a}} 2\sqrt{r_{a}^{2} - \mathbf{x}_{2}^{2}} d\mathbf{x}_{2}.$$
 (65)

Let  $\theta$  be an angle variable that satisfies  $x_2 = r_a sin(\theta)$ , we know  $d\mathbf{x_2} = r_a cos(\theta) d\theta$ . Then, Eq. (65) is rewritten as

$$\operatorname{Vol}_{\mathbf{2}}(S_a) = \int_{-\pi/2}^{\pi/2} 2r_a^2 \cos^2(\vartheta) d\vartheta$$
$$= \int_0^{\pi/2} 4r_a^2 \cos^2(\vartheta) d\vartheta. \tag{66}$$

For a 3-D geometry, for the variable  $\mathbf{x_3}$ , it is over a cross-section which is a **2**-dimensional ball (circle), where the radius of the ball can be expressed as  $r_a cos(\vartheta)$ , s.t.  $\vartheta \in [0, \pi]$ . Particularly, let  $\operatorname{Vol}_3(S_a)$  be the volume of  $S_a$  with 3 dimensions, the volume of this 3-dimensional sphere then can be written as

$$\operatorname{Vol}_{\mathbf{3}}(S_a) = \int_0^{\pi/2} 2\operatorname{Vol}_{\mathbf{2}}(r_a cos(\vartheta)) r_a(cos(\vartheta)) d\vartheta. \tag{67}$$

With Eq. (67), volume of a d-dimensional geometry can be expressed as the integral over the (d-1)-dimensional cross-section of  $S_a$ 

$$Vol_{\boldsymbol{m}}(S_a) = \int_0^{\pi/2} 2Vol_{\boldsymbol{m}-1}(r_a cos(\vartheta)) r_a(cos(\vartheta)) d\vartheta, \tag{68}$$

where  $Vol_{m-1}$  denotes the volume of (m-1)-dimensional generalization geometry of  $S_a$ .

Based on Eq. (68), we know Vol<sub>3</sub> can be written as

$$Vol_{\mathbf{3}}(S_a) = \int_0^{\pi/2} 2Vol_{\mathbf{2}}(R_a cos(\vartheta)) r_a(cos(\vartheta')) d\vartheta$$
(69)

Introducing Eq. (66) into Eq. (69), we have

$$\mathrm{Vol}_{3}(S_{a})=\int_{0}^{\pi/2}2\,\mathrm{Vol}_{2}\big(R_{a}\cos\vartheta'\big)\;r_{a}\big(\cos\vartheta'\big)\,d\vartheta'$$

$$=\int_{0}^{\pi/2}\int_{0}^{\pi/2}8\big(r_{a}\cos\vartheta'\big)^{2}\cos^{2}\vartheta'\;r_{a}\big(\cos\vartheta\big)\,d\vartheta'\,d\vartheta$$

$$=\frac{4}{3}\pi R_{a}^{2}\quad\text{s.t. }R_{a}=r_{a}.$$

(70)

Therefore, the generalization analysis results of the 3-D geometry still can hold in the high dimensional geometry. Then,  $\Box$ 

Proof of Theorems 4.

*Proof.* The integral of Eq. (67) also can be adopted into the volume of  $Vol_3(E_a)$  by transforming the area i.e.  $Vol_2(S_a)$  into  $Vol_2(E_a)$ . Then, Eq. (68) follows this transform.

Image /page/20/Picture/9 description: A head-and-shoulders selfie of a person with short dark hair and a slight smile, wearing a navy-blue and white horizontally striped shirt. The photo is taken outdoors in daylight; a multi-story brick building with rectangular windows appears in the left background and green-leaved trees are visible on the right. The image is slightly tilted and the subject is turned a little to the viewer’s left. Bright sunlight illuminates the face and casts mild shadows; the framing cuts off around the upper chest.

Xiaofeng Cao completed his PhD study at Australian Artificial Intelligence Institute (AAII), University of Technology Sydney. He is working as a Research Assistant at AAII. His research interests include PAC learning theory, agnostic learning algorithm, generalization analysis, and hyperbolic geometry.

Image /page/20/Picture/11 description: Passport-style head-and-shoulders portrait of an adult Asian man photographed against a light gray to white gradient background. He has short, straight black hair neatly combed to the side and is wearing thin rectangular eyeglasses. His expression is neutral, mouth closed, looking directly at the camera. He is dressed in a light-colored collared dress shirt with subtle vertical stripes and a dark necktie. The image is centered on his face and upper torso with even lighting and no visible text or additional objects.

Ivor W. Tsang is Professor of Artificial Intelligence, at University of Technology Sydney. He is also the Research Director of the Australian Artificial Intelligence Institute. In 2019, his paper titled "Towards ultrahigh dimensional feature selection for big data" received the International Consortium of Chinese Mathematicians Best Paper Award. In 2020, Prof Tsang was recognized as the AI 2000 AAAI/IJCAI Most Influential Scholar in Australia for his outstanding contributions to the field of Artificial Intelligence between 2009 and 2019. His works on transfer learning granted him the Best Student Paper Award International Conference on Computer Vision and Pattern Recognition 2010 and the 2014 IEEE Transactions on Multimedia Prize Paper Award. In addition, he had received the prestigious IEEE Transactions on Neural Networks Outstanding 2004 Paper Award in 2007.

Prof. Tsang serves as a Senior Area Chair for Neural Information Processing Systems and Area Chair for International Conference on Machine Learning, and the Editorial Board for Journal Machine Learning Research, Machine Learning, Journal of Artificial Intelligence Research, and IEEE Transactions on Pattern Analysis and Machine Intelligence.