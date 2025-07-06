---
myst:
 substitutions:
  date: "2025-07-06"
---
```{post} 2025-07-06
:excerpt: 0
```

```{math}
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
```

# One change at a time: how to compare SGD, signSGD, spectral descent, LARS, etc?
_Posted on {{ date }}_

<!-- 
Two commonly used methods in ML are:

$$
\begin{align*}
 x^{k+1}&=x^k - \gamma_k \nabla f(x^k,\xi_k) && \text{(SGD)} \\
x^{k+1}&=x^k - \gamma_k\operatorname{sign}(\nabla f(x^k,\xi_k)) && \text{(signSGD)}
\end{align*}
$$ -->

When comparing SGD and signSGD we are making two changes at once (possibly without realizing): 

- we are changing the geometry 
- _and_ we are normalizing the update.

To see this, it is useful to write (normalized) steepest descent in terms of the linear minimization oracle (LMO):

$$\operatorname{lmo}(d) \in \argmin_{x \in \mathcal D} \langle d,x\rangle$$

where the constraint is the norm ball, $\mathcal D = \{x \mid \Vert x \Vert \leq 1\}$. 
We have the following special cases:

|  Method | Norm | Steepest descent | Normalized steepest descent |
|---|---|---|----|
| General | $\Vert \cdot\Vert$ | $x^{k+1}=x^k + \gamma_k\Vert d^k\Vert_* \operatorname{lmo}(d^k)$ | $x^{k+1}=x^k + \gamma_k  \operatorname{lmo}(d^k)$ |
| Gradient descent | $\Vert \cdot\Vert_2$  | $\color{red}x^{k+1}=x^k - \gamma_k d^k$ | $x^{k+1}=x^k - \gamma_k \tfrac{d^k}{\Vert d^k\Vert_2}$ |
| Sign descent  | $\Vert \cdot\Vert_\infty$ | $x^{k+1}=x^k - \gamma_k\Vert d^k\Vert_1\operatorname{sign}(d^k)$ | $\color{red}x^{k+1}=x^k - \gamma_k\operatorname{sign}(d^k)$ |
| Spectral descent  | $\Vert \cdot\Vert_{\mathcal{S}_\infty}$ | $x^{k+1}=x^k - \gamma_k\Vert d^k\Vert_1 U^k(V^k)^\top$ | $x^{k+1}=x^k - \gamma_k U^k(V^k)^\top$ |

where the reduced SVD is given as $d^k=U^k\operatorname{diag}(\sigma^k)(V^k)^\top$ and $d^k$ is some gradient feedback.

SGD and signSGD are highlighted in <span style="color:red;">red</span>.
We argue that a more natural comparison to signSGD is normalized SGD.
In a similar vein, Muon should be compared with normalized GD, like spectral descent should be compared with GD.

Why do we care about disentangling the changes?
Normalized methods have very different properties from their non-normalized variants:

- one benefit of normalization is that the initial stepsize does not have to be taken small enough to ensure convergence (although we do need $\gamma_k\rightarrow 0$). In comparison steepest descent requires a small enough stepsize $\gamma_k <2/L$, otherwise the method might diverge. Intuitively, the benefit comes from the normalization making the update invariant to the gradient magnitude.

- one downside is that, due to the same invariance, normalization does not adapt to strong convexity/PL. In contrast, steepest descent enjoys linear convergence—intuitively, because steps are larger when farther from the solution, due to the updates dependency on the gradient magnitude. 

  _For instance, for a simple quadratic problem like $\min_x \Vert x\Vert^2_2$ we should expect steepest descent methods (GD, signSGD with dual norm scaling) to converge much faster than their normalized counterparts (NGD, signSGD) when stepsizes are set appropriately._

The fact that normalization works much better in practice for deep learning suggests that we are not in the strongly convex/PL case (or that we are at least currently prevented from exploiting such structure even if present).


## Layerwise/block normalization

Normalized updates alone have been observed to work very well for large batch sizes if updates are normalized layerwise (see e.g. LARS {cite:p}`you2017large` and other block-normalization schemes {cite:p}`ginsburg2019stochastic,yu2017block`).

Consider the case where the parameter $x$ decomposes into $n$ blocks, i.e., $x=(W_1,...,W_n)$ and the gradient estimator $d=(D_1,...,D_n)$.
The block-normalized update implicitly chooses $\Vert x\Vert = \max_l \Vert W_l \Vert_F$.
If using global $\ell_2$ normalization instead we are implicitly picking $\ell_2$ over the layers, i.e. $\Vert x\Vert_2 = \Vert (\Vert W_l\Vert_F)_l \Vert_2$.

Some popular norm choices:

| Name | Norm | Normalized steepest descent |
|---|---|----|
| Euclidean | $\Vert x\Vert_2 = \Vert (\Vert W_l\Vert_F)_l \Vert_2 $  | $x^{k+1}=x^k - \gamma_k \tfrac{d^k}{\Vert d^k\Vert_2}$ |
| Layerwise Frobenius | $\max_l \Vert W_l\Vert_F$ | $W_l^{k+1}=W_l^k - \gamma_k \tfrac{D^k_i}{\Vert D^k_l\Vert_F}$ |
|  Layerwise Sign | $\max_l \Vert W_l\Vert_{1 \rightarrow \infty}=\max_l \max_{i,j} \|[W_l]_{i,j}\|$ | $x^{k+1}=x^k - \gamma_k\operatorname{sign}(d^k)$ |
|  Layerwise RowNorm | $\max_l \Vert W_l\Vert_{2 \rightarrow \infty}=\max_l \max_i \Vert [W_l]_{i,\cdot}\Vert_2$ | $[W_l^{k+1}]_{i,\cdot}=[W_l^k]_{i,\cdot} - \gamma_k \tfrac{[D^k_i]_{i,\cdot}}{\Vert [D^k_l]_{i,\cdot}\Vert_2}$ |
|  Layerwise Spectral | $\max_l \Vert W_l\Vert_{2\rightarrow 2}$ | $W_l^{k+1}=W_l^k - \gamma_k U^k_l(V^k_l)^\top$ |

There are a couple of choices being made simultaneously here:

- There is an implicit choice of the norm over layers.
 The global $\ell_2$ implicitly picks $\ell_2$ over layers while the remaining uses the max-norm over layers. One could similarly consider e.g., the spectral norm with $\ell_2$ over layers, i.e., $\Vert x\Vert = \Vert (\Vert W_l\Vert_{2\rightarrow 2})_l \Vert_2$.
 A comparison should ideally make one change at a time and for example compare layerwise Frobenius with layerwise spectral rather than with global $\ell_2$ normalization.
- There is an implicit choice of what _type_ of matrix norm. Is it an operator norm? Is it a Schatten norm? This has implications in the stochastic case where Sign and Spectral should be expected to behave differently (More on this later).

One thing that can be gauged from above table is that RowNorm is a closer approximation to Spectral than Sign is.
This might explain the popularity of row normalization in practice, e.g., in nGPT {cite:p}`loshchilov2024ngpt`, optimizers for diffusion models {cite:p}`karras2024analyzing` and DASGO {cite:p}`an2025asgo` (see the recent {cite:t}`kovalev2025` for a connection with RowNorm).

## Momentum based gradient estimator


Another element in these methods is the choice of the gradient estimator $d^k$, which turns out to be crucial for establishing convergence in the stochastic case.

For SGD, $d^k$ is simply picked as $d^k=\nabla f(x^k,\xi_k)$ for some randomly sampled minibatch $\xi_k$.
Both SGD and normalized SGD converges under standard assumptions (unbiased and bounded variance) {cite:p}`hazan2015beyond`.

In the non-Euclidean case, convergence is unfortunately not guaranteed[^batch], since the LMO can be non-linear, so the update $\operatorname{lmo}(d^k)$ can be biased as soon as $d^k$ has variance, regardless of whether $d^k$ itself is unbiased.

[^batch]: That is, without trivially killing the noise with an increasing batchsize.

The issue can be resolved with the following estimator, which is sometimes referred to as a momentum estimator due to its connection with Polyak momentum in the Euclidean case:

$$
d^k = (1-\alpha_k)d^{k-1} + \alpha_k \nabla f(x^k,\xi_k)
$$ (mom)

**Single sample**
There is an interesting exception where convergence can be established without the momentum estimator: the gradient of a weight matrix in an MLP computed over a single sample is rank-1 (see e.g., {cite:p}`yang2023spectral`).

In that case, all Schatten norms are equivalent (so Frobenius = Spectral = Nuclear), which means that spectral methods become equivalent to their $\ell_2$ counterpart (e.g., Muon=normalized SGD with momentum).
So spectral descent converges in the single-sample case, since SGD does![^singlsample]

[^singlsample]: Strictly speaking, the equivalence holds when $\ell_2$ is picked as the norm-choice over layers, i.e. $\Vert (\Vert W_1\Vert_{\mathcal S_{\infty}}, ..., \Vert W_D\Vert_{\mathcal S_{\infty}})\Vert _2$.

In contrast, we do not have that equivalence between SignSGD and SGD under low-rank structure (the equivalence rather hold when the gradients are as sparse as possible, namely a 1-hot vector).

_Why is this interesting?
We should expect signSGD to _not_ converge for `batchsize=1`, whereas spectral descent (with or without normalization) should in fact converge---and be equivalent to SGD._

**The difference between averaging gradients and iterates**
Coming back to the momentum estimator, for non-Euclidean methods, the momentum in {eq}`mom` (i.e., averaging past gradient) is crucial to establish convergence by reducing the variance.
Averaging the iterates instead would not lead to a convergent method.[^ALMOND]

[^ALMOND]: See e.g., the analysis of ALMOND in {cite:t}`pethick2025training`.

In the Euclidean case, averaging the gradients (dual feedback) as in {eq}`mom` and averaging the iterates (primal variables) can be equivalent.
E.g., Polyak momentum, can be written both in terms of averaging dual feedback ($x^{k+1} = x^k - \gamma v^{k+1}$ where $v^{k+1} = \sum_{k=0}^l \beta^k \nabla f(x^{l-k})$) and in terms of primal averaging (see e.g. {cite:t}`defazio2020momentum,tao2018primal`).
This leads to two additional properties:

- for the Euclidean and deterministic case, gradient descent with momentum applied to a convex quadratic the convergence rate is accelerated.
- for the Euclidean and stochastic case, stochastic gradient descent with momentum enjoys tight last iterate rates {cite:p}`defazio2020momentum,tao2018primal`. 

Even though the two mechanism, primal averaging and averaging dual feedback, can sometimes be equivalent, they generally have different properties.

Overall, to develop a better empirical understanding we should probably be more careful about _changing one thing at a time_.

---
```{bibliography}
:filter: docname in docnames
```
