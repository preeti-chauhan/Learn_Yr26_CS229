# CS229: Machine Learning

**Instructor:** Tengyu Ma · **Offering:** Spring 2026

A rigorous ground-up treatment of machine learning — from linear models and probabilistic inference to deep learning, generative models, and reinforcement learning. The mathematical backbone behind every modern ML system.

---

[Course Website](https://cs229.stanford.edu/) · [Course Playlist](https://www.youtube.com/playlist?list=PLaqpC4kq8Gpw) · [Official Notes](https://cs229.stanford.edu/notes/)

---

## What CS229 Is

CS221 teaches you the agent lifecycle — search, decisions, probabilistic reasoning, language. CS229 goes deep on one part of that lifecycle: **how machines learn from data**.

Where CS221 gives you the ML pipeline in 4 lectures, CS229 gives you 17. The depth is in the math: MLE derivations from scratch, the full EM algorithm, kernel methods, the bias-variance decomposition proved formally, diffusion model score matching, policy gradient theorem. Every result you use in practice has a derivation here.

The 2026 edition is meaningfully different from older CS229. It has dropped SVMs as a primary topic and added three modern units — **Diffusion Models** (L11), **Representation Learning** (L12), and **LLMs + Transformers** (L13–L14) — reflecting where the field has moved. The mathematical rigour stays the same; the content now covers what you will actually build.

---

## Study Workflow

1. **Watch the lecture** — get the flow and intuition first
2. **Build the source notebook** — derive every equation from scratch and implement it in code; if you can't code it, you don't understand it
3. **Write a blog post** — explaining it publicly forces precision

---

## Concept Map

```
CS229
│
├── THE FRAMEWORK (cuts across everything)
│   ├── ERM — minimize loss over training data
│   ├── MLE — choose θ to maximize P(data | θ)
│   └── Loss = -log likelihood → squared, cross-entropy, etc.
│
├── SUPERVISED LEARNING
│   │
│   ├── Linear Models
│   │   ├── Linear Regression       → Gaussian noise → squared loss    [L02]
│   │   ├── Logistic Regression     → Bernoulli → cross-entropy         [L03]
│   │   └── GLMs / Exp. Family      → generalizes both of the above     [L04]
│   │
│   ├── Neural Networks
│   │   ├── Architecture            → layers, activations, depth        [L07]
│   │   ├── Backpropagation         → chain rule at scale               [L08]
│   │   └── Training tricks         → dropout, batch norm, regularize   [L06]
│   │
│   └── Optimization
│       ├── Gradient Descent        → batch, full data                  [L02]
│       ├── SGD / Mini-batch        → workhorse of ML                   [L02]
│       ├── Normal Equations        → closed form, small data only      [L02]
│       └── Newton's Method         → second order, classical stats     [L03]
│
├── UNSUPERVISED LEARNING
│   ├── K-Means                     → cluster by distance               [L09]
│   ├── GMM + EM Algorithm          → soft clusters, probabilistic      [L09-10]
│   └── PCA                         → dimensionality reduction          [L10]
│
├── GENERATIVE MODELS               (model p(x), not just p(y|x))
│   ├── VAE                         → encode → latent → decode          [L12]
│   ├── GANs                        → generator vs discriminator        [L12]
│   └── Diffusion Models            → denoise step by step              [L11]
│
├── REPRESENTATION LEARNING         (learn features without labels)
│   ├── Self-Supervised Learning    → CLIP, DINO-style contrastive      [L12]
│   └── Transfer Learning           → pretrain → finetune               [L12]
│
├── SEQUENCE + LANGUAGE MODELS
│   ├── Transformers                → attention, Q/K/V, positional enc  [L14]
│   └── LLMs                        → next-token prediction at scale    [L13]
│
├── REINFORCEMENT LEARNING
│   ├── MDPs                        → state, action, reward, policy     [L15]
│   ├── Policy Gradient             → ∇J = E[∇log π · R]               [L15]
│   └── RLHF                        → bridge to LLMs                   [L01]
│
└── THEORY
    ├── Bias-Variance Tradeoff      → underfitting vs overfitting       [L06]
    ├── Generalization Bounds       → why it works on new data          [L06]
    └── Regularization              → L1 (sparse), L2 (small weights)   [L06]
```

**The connective thread:** MLE sits at the center. The choice of probability distribution for your labels determines the loss function. Gaussian → squared loss. Bernoulli → cross-entropy. Categorical → softmax. Every loss function in the course derives from MLE under a different distribution assumption.

---

## Course Map

CS229 divides into five blocks:

| Block | Lectures | Theme |
|---|---|---|
| **Supervised Learning** | L01–L06 | Linear models → generative models → GLMs → practical ML advice |
| **Neural Networks** | L07–L08 | Architecture → backpropagation — the engine of modern ML |
| **Unsupervised Learning** | L09–L10 | K-Means · GMM · EM algorithm · PCA |
| **Generative + Representation** | L11–L12 | Diffusion models · representation learning · self-supervised learning |
| **Language + RL** | L13–L16 | LLMs · Transformers · reinforcement learning · policy gradient |
| **EM / PCA Deep Dive** | L18, L20 | GMM EM algorithm · PCA — extended treatment |

---

## Lectures

### Block 1 — Supervised Learning

| # | Title | YouTube | Notebook |
|---|---|---|---|
| L01 | Introduction | [▶](https://www.youtube.com/watch?v=DATnpGoGhM8) | [notebook](Source-Notebooks/L01-Introduction.ipynb) |
| L02 | Supervised Learning Setup | [▶](https://www.youtube.com/watch?v=cmNIMjPYdgM) | [notebook](Source-Notebooks/L02-SupervisedLearning.ipynb) |
| L03 | Weighted Least Squares | [▶](https://www.youtube.com/watch?v=uJF_gL3jhxI) | [notebook](Source-Notebooks/L03-WeightedLeastSquares.ipynb) |
| L04 | Exponential Family, GLMs Classification | [▶](https://www.youtube.com/watch?v=8gVi4Rk21Eg) | [notebook](Source-Notebooks/L04-GLM.ipynb) |
| L05 | Gaussian Discriminant Analysis | [▶](https://www.youtube.com/watch?v=zRdE8A4UZes) | [notebook](Source-Notebooks/L05-GDA.ipynb) |
| L06 | Dataset Split, ML Advice | [▶](https://www.youtube.com/watch?v=llnEgyyuYkQ) | [notebook](Source-Notebooks/L06-MLAdvice.ipynb) |

### Block 2 — Neural Networks

| # | Title | YouTube | Notebook |
|---|---|---|---|
| L07 | Neural Networks 1 — Architecture | [▶](https://www.youtube.com/watch?v=fRM41w9jzQo) | [notebook](Source-Notebooks/L07-NeuralNetworks-Architecture.ipynb) |
| L08 | Neural Networks 2 — Backprop | [▶](https://www.youtube.com/watch?v=ne2ngVAoMG8) | [notebook](Source-Notebooks/L08-NeuralNetworks-Backprop.ipynb) |

### Block 3 — Unsupervised Learning

| # | Title | YouTube | Notebook |
|---|---|---|---|
| L09 | K-Means and GMM | [▶](https://www.youtube.com/watch?v=bSmIGBCoffA) | [notebook](Source-Notebooks/L09-KMeans-GMM.ipynb) |
| L10 | GMM (EM), PCA | [▶](https://www.youtube.com/watch?v=sUS-eTa0l6s) | [notebook](Source-Notebooks/L10-EM-PCA.ipynb) |

### Block 4 — Generative + Representation

| # | Title | YouTube | Notebook |
|---|---|---|---|
| L11 | Diffusion Models | [▶](https://www.youtube.com/watch?v=dqUMCzWjZSI) | [notebook](Source-Notebooks/L11-DiffusionModels.ipynb) |
| L12 | Representation Learning | [▶](https://www.youtube.com/watch?v=_kREM2UAiJ8) | [notebook](Source-Notebooks/L12-RepresentationLearning.ipynb) |

### Block 5 — Language + RL

| # | Title | YouTube | Notebook |
|---|---|---|---|
| L13 | LLMs, Next-Word Prediction Loss | [▶](https://www.youtube.com/watch?v=lNTajqxxOn4) | [notebook](Source-Notebooks/L13-ContrastiveLearning.ipynb) |
| L14 | Transformers, In-Context Learning | [▶](https://www.youtube.com/watch?v=pwQ0l4hFCVI) | [notebook](Source-Notebooks/L14-Transformers.ipynb) |
| L15 | *(not on YouTube)* | — | — |
| L16 | Basic Concepts in RL, Policy Gradient | [▶](https://www.youtube.com/watch?v=hHC-SF3utxg) | [notebook](Source-Notebooks/L16-EfficientTransformers.ipynb) |

### Block 6 — EM / PCA Deep Dive

| # | Title | YouTube | Notebook |
|---|---|---|---|
| L17 | *(not on YouTube)* | — | — |
| L18 | GMM (EM), PCA — continued | [▶](https://www.youtube.com/watch?v=xveNBYVTrqw) | [notebook](Source-Notebooks/L18-ReinforcementLearning-MDP-PolicyGradient.ipynb) |
| L19 | *(not on YouTube)* | — | — |
| L20 | GMM (EM), PCA — continued | [▶](https://www.youtube.com/watch?v=J7CossjMvEg) | [notebook](Source-Notebooks/L20-PolicyGradient-PPO-RLforLLMs.ipynb) |

---

## Source Notebooks

Each notebook is a one-stop learning and teaching guide for that lecture:

- `> 📌 *Lecture:*` — instructor's exact words from the YouTube transcript
- `> 🎯 **Interview:**` — Q&A blocks for interview articulation
- LaTeX math — full derivations, not just results
- External resources — best explanations from papers, blogs, and videos

| # | Topic | Colab |
|---|---|---|
| L01 | Introduction | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/Learn_Yr26_CS229/blob/main/Source-Notebooks/L01-Introduction.ipynb) |
| L02 | Supervised Learning Setup | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/Learn_Yr26_CS229/blob/main/Source-Notebooks/L02-SupervisedLearning.ipynb) |
| L03 | Weighted Least Squares | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/Learn_Yr26_CS229/blob/main/Source-Notebooks/L03-WeightedLeastSquares.ipynb) |
| L04 | Exponential Family, GLMs | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/Learn_Yr26_CS229/blob/main/Source-Notebooks/L04-GLM.ipynb) |
| L05 | Gaussian Discriminant Analysis | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/Learn_Yr26_CS229/blob/main/Source-Notebooks/L05-GDA.ipynb) |
| L06 | Dataset Split, ML Advice | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/Learn_Yr26_CS229/blob/main/Source-Notebooks/L06-MLAdvice.ipynb) |
| L07 | Neural Networks — Architecture | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/Learn_Yr26_CS229/blob/main/Source-Notebooks/L07-NeuralNetworks-Architecture.ipynb) |
| L08 | Neural Networks — Backprop | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/Learn_Yr26_CS229/blob/main/Source-Notebooks/L08-NeuralNetworks-Backprop.ipynb) |
| L09 | K-Means and GMM | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/Learn_Yr26_CS229/blob/main/Source-Notebooks/L09-KMeans-GMM.ipynb) |
| L10 | GMM (EM), PCA | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/Learn_Yr26_CS229/blob/main/Source-Notebooks/L10-EM-PCA.ipynb) |
| L11 | Diffusion Models | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/Learn_Yr26_CS229/blob/main/Source-Notebooks/L11-DiffusionModels.ipynb) |
| L12 | Representation Learning | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/Learn_Yr26_CS229/blob/main/Source-Notebooks/L12-RepresentationLearning.ipynb) |
| L13 | LLMs, Next-Word Prediction | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/Learn_Yr26_CS229/blob/main/Source-Notebooks/L13-ContrastiveLearning.ipynb) |
| L14 | Transformers, In-Context Learning | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/Learn_Yr26_CS229/blob/main/Source-Notebooks/L14-Transformers.ipynb) |
| L16 | Basic Concepts in RL, Policy Gradient | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/Learn_Yr26_CS229/blob/main/Source-Notebooks/L16-EfficientTransformers.ipynb) |
| L18 | RL, MDP, Policy Gradient | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/Learn_Yr26_CS229/blob/main/Source-Notebooks/L18-ReinforcementLearning-MDP-PolicyGradient.ipynb) |
| L20 | Policy Gradient, PPO, RL for LLMs | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/Learn_Yr26_CS229/blob/main/Source-Notebooks/L20-PolicyGradient-PPO-RLforLLMs.ipynb) |
| — | Probability Review | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/preeti-chauhan/Learn_Yr26_CS229/blob/main/Source-Notebooks/Probability-Review.ipynb) |

---

## Skills Tracker

Check off as you complete each topic.

---

### Block 1: Supervised Learning

**L01 — Introduction**

- [ ] What is machine learning — supervised, unsupervised, RL taxonomy
- [ ] The ML pipeline: data → model → loss → optimization → evaluation
- [ ] Course roadmap and motivation

**L02 — Supervised Learning Setup**

- [ ] Hypothesis class · loss function · optimization algorithm
- [ ] Linear regression: `f(x) = wᵀx + b`
- [ ] Squared loss: `L = (y - ŷ)²`
- [ ] Normal equations: closed-form solution `w = (XᵀX)⁻¹Xᵀy`
- [ ] Gradient descent derivation for linear regression
- [ ] Probabilistic interpretation: MLE → squared loss under Gaussian noise

**L03 — Weighted Least Squares**

- [ ] Locally weighted regression (LWR): weight each training point by distance to query
- [ ] Kernel `w(i) = exp(-||x(i) - x||² / 2τ²)` — bandwidth τ controls locality
- [ ] Bandwidth tradeoff: small τ → high variance · large τ → high bias
- [ ] Parametric vs non-parametric models
- [ ] Logistic regression: sigmoid · cross-entropy loss · MLE derivation
- [ ] Newton's method: quadratic convergence vs gradient descent linear convergence

**L04 — Exponential Family, GLMs**

- [ ] Exponential family: `p(y; η) = b(y) exp(ηᵀT(y) - a(η))`
- [ ] Natural parameter η · sufficient statistic T(y) · log-partition a(η)
- [ ] Bernoulli → logistic regression · Gaussian → linear regression as GLM special cases
- [ ] GLM recipe: choose exponential family → link function → predictor
- [ ] Softmax regression as multiclass GLM

**L05 — Gaussian Discriminant Analysis**

- [ ] Discriminative vs generative models — what each models
- [ ] GDA: `p(x|y) = Gaussian`, `p(y) = Bernoulli` → joint → posterior
- [ ] GDA decision boundary is linear (when Σ shared across classes)
- [ ] GDA vs logistic regression: GDA stronger assumptions, fewer samples needed
- [ ] Naive Bayes: `p(x|y) = ∏ p(xⱼ|y)` — feature independence assumption
- [ ] Naive Bayes for text classification · Laplace smoothing

**L06 — Dataset Split, ML Advice**

- [ ] Train / dev / test split rationale — contamination risk
- [ ] Bias-variance tradeoff: `Error = Bias² + Variance + Irreducible`
- [ ] High bias → underfitting → bigger model · more features · less regularization
- [ ] High variance → overfitting → more data · regularization · smaller model
- [ ] Regularization: L2 (Ridge) `λ||w||²` · L1 (Lasso) `λ||w||₁`
- [ ] L1 sparsity: why L1 produces sparse solutions (geometry argument)
- [ ] Error analysis workflow: ceiling analysis · ablation studies
- [ ] Learning curves — diagnosing bias vs variance from training curve shape

---

### Block 2: Neural Networks

**L07 — Neural Networks 1: Architecture**

- [ ] MLP: `h = σ(Wx + b)` stacked layers
- [ ] Activation functions: sigmoid · tanh · ReLU · GeLU — why ReLU dominates
- [ ] Universal approximation theorem — what it says and what it doesn't
- [ ] Depth vs width tradeoff — why depth is preferred
- [ ] Forward pass as function composition

**L08 — Neural Networks 2: Backpropagation**

- [ ] Computation graph — forward pass builds the graph
- [ ] Chain rule for vectors: Jacobian accumulation
- [ ] Backprop algorithm: topological sort → reverse pass
- [ ] Weight sharing gradient accumulation (`+=` not `=`)
- [ ] Vanishing gradient problem — why sigmoid/tanh fail for deep nets
- [ ] Xavier / He initialization — variance scaling derivation
- [ ] Full training loop: forward → loss → backward → update

---

### Block 3: Unsupervised Learning

**L09 — K-Means and GMM**

- [ ] K-Means algorithm: assign → update → convergence
- [ ] K-Means objective: `∑ᵢ ||x(i) - μ_{c(i)}||²`
- [ ] K-Means as special case of EM (hard assignments)
- [ ] Gaussian Mixture Model: `p(x) = ∑ₖ πₖ N(x; μₖ, Σₖ)`
- [ ] Soft assignments vs hard assignments
- [ ] GMM likelihood — why it's not convex

**L10 — GMM (EM), PCA**

- [ ] EM algorithm: E-step (compute responsibilities) → M-step (update parameters)
- [ ] ELBO derivation: `log p(x) ≥ E_q[log p(x,z)] - E_q[log q(z)]`
- [ ] EM monotonically increases log-likelihood (convergence guarantee)
- [ ] PCA: directions of maximum variance = eigenvectors of covariance matrix
- [ ] PCA derivation via SVD: `X = UΣVᵀ` → top-k components
- [ ] Reconstruction error and explained variance ratio
- [ ] PCA vs autoencoders — linear vs nonlinear dimensionality reduction

---

### Block 4: Generative + Representation

**L11 — Diffusion Models**

- [ ] Forward process: `q(xₜ|xₜ₋₁) = N(xₜ; √(1-β)xₜ₋₁, βI)` — adding noise
- [ ] Reparameterization: `xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε` — sample any t directly
- [ ] Reverse process: learn `p_θ(xₜ₋₁|xₜ)` — denoising
- [ ] Score matching: `s_θ(x) ≈ ∇_x log p(x)`
- [ ] Training objective: predict noise ε from noisy image xₜ
- [ ] DDPM vs DDIM — stochastic vs deterministic sampling
- [ ] Classifier-free guidance — conditioning without a classifier

**L12 — Representation Learning**

- [ ] Self-supervised learning: contrastive · predictive · generative approaches
- [ ] Contrastive loss (SimCLR): pull positives together, push negatives apart
- [ ] InfoNCE loss derivation
- [ ] CLIP: image-text contrastive pretraining at scale
- [ ] Autoencoders: encoder-decoder · bottleneck · reconstruction loss
- [ ] VAE: reparameterization trick · ELBO = reconstruction + KL divergence
- [ ] Transfer learning: freeze backbone → finetune head vs full finetune

---

### Block 5: Language + RL

**L13 — LLMs, Next-Word Prediction**

- [ ] Language modeling as next-token prediction: `p(xₜ|x₁,...,xₜ₋₁)`
- [ ] Cross-entropy loss = negative log-likelihood of target token
- [ ] Perplexity: `exp(average cross-entropy)` — lower is better
- [ ] Scaling laws: loss ∝ (data, compute, parameters) power laws
- [ ] Pre-training → fine-tuning → RLHF pipeline

**L14 — Transformers, In-Context Learning**

- [ ] Self-attention: `Attention(Q,K,V) = softmax(QKᵀ/√d)V`
- [ ] Multi-head attention: project into h subspaces, concatenate
- [ ] Positional encoding — why attention is permutation invariant without it
- [ ] Transformer block: attention → LayerNorm → FFN → LayerNorm (residual)
- [ ] In-context learning: few-shot prompting as implicit Bayesian inference
- [ ] KV cache — why it matters for inference efficiency

**L16 — Basic Concepts in RL, Policy Gradient**

- [ ] MDP: `(S, A, P, R, γ)` — state, action, transition, reward, discount
- [ ] Policy `π(a|s)` · value function `V^π(s)` · Q-function `Q^π(s,a)`
- [ ] Policy gradient theorem: `∇_θ J(θ) = E[∇_θ log π_θ(a|s) · Q(s,a)]`
- [ ] REINFORCE algorithm
- [ ] Baseline subtraction to reduce variance
- [ ] Connection to RLHF: reward model + policy gradient = alignment

---

### Block 6: EM / PCA Deep Dive

**L18 — GMM (EM), PCA — continued**

- [ ] EM algorithm: convergence proof, ELBO bound tightening
- [ ] GMM parameter updates: responsibilities, μ, Σ, π M-step derivations
- [ ] PCA: maximum variance direction = first eigenvector of XᵀX
- [ ] Whitening and ZCA preprocessing

**L20 — GMM (EM), PCA — continued**

- [ ] PCA via SVD: `X = UΣVᵀ` — principal components as right singular vectors
- [ ] Choosing k: scree plot, explained variance ratio
- [ ] Kernel PCA — nonlinear dimensionality reduction
- [ ] Factor analysis vs PCA — latent variable interpretation

---

## Reference Materials

| Resource | What it covers |
|---|---|
| [CS229 Official Notes](https://cs229.stanford.edu/notes/) | Full derivations per topic — authoritative reference |
| [CS229 Handouts (Archive)](https://cs229.stanford.edu/lectures/) | Per-lecture PDFs: supervised learning, GDA, EM, PCA, RL |
| [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) | Matrix derivatives, identities — open during derivations |
| [Pattern Recognition and Machine Learning — Bishop](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) | Probabilistic ML — EM, GMMs, Bayesian methods |
| [Understanding Deep Learning — Prince](https://udlbook.github.io/udlbook/) | Modern DL with diffusion, VAEs, transformers |
| [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) | Transformer code walkthrough — L14 companion |
| [Lilian Weng — Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | Best diffusion model explainer — L11 companion |
| [Stanford CS229 2018 Lectures (Ng)](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) | Older version — Andrew Ng's explanations, still excellent for intuition |
| [Prior CS229 Notes (2024)](https://github.com/preeti-chauhan/Stanford-Artificial-Intelligence-Professional-Program/tree/master/Stanford-CS229-Course) | Previous pass through CS229 — reference for comparison |
