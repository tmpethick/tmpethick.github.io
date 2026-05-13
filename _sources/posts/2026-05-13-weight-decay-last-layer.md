---
myst:
 substitutions:
  date: "2026-05-13"
---
```{post} 2026-05-13
:excerpt: 0
```

# Why not use weight decay for the last layer?
_Posted on {{ date }}_

It is common to exclude the unembedding layer from using weight decay, but why?

**TL;DR**: Applying weight decay to the last layer will constrain the logits of the model and consequently prevent high-confident output.

To make this statement precise, let $h \in \mathbb{R}^d$ denote the hidden state before the final linear layer.
Modern architectures often apply normalization before the last layer.
Ignoring the small numerical $\epsilon$ and any learned gain, the RMS-normalized state $\hat h := \operatorname{RMSNorm}(h)$ satisfies

$$
\|\hat h\|_2 = \sqrt d.
$$

Let $m$ be the output dimension, e.g., the number of classes or vocabulary size.
The output matrix is $W \in \mathbb{R}^{m \times d}$, with rows $w_1, \dots, w_m$, and logits

$$
z_i = \langle w_i, \hat h \rangle.
$$

If we take a Frank-Wolfe perspective on weight decay {cite:p}`pethick2025training`, the effect is to keep $W$ in a norm ball of radius $\rho>0$:

$$
S^k \in \operatorname*{arg\,min}_{S : \|S\|\leq \rho} \langle \nabla f(W^k), S \rangle,
\qquad
W^{k+1} = (1 - \gamma_k) W^k + \gamma_k S^k.
$$

For simplicity, choose $\|\cdot\|$ to be the $\mathrm{RMS} \to \infty$ operator norm,

$$
\|W\|_{\mathrm{RMS}\to\infty}
:=
\sup_{\|x\|_2/\sqrt d \leq 1} \|Wx\|_\infty.
$$

The radius-$\rho$ ball for this norm is equivalent to the row-wise constraints $\|w_i\|_2 \leq \rho/\sqrt d$.
Hence the linear minimization oracle separates across rows and reduces to

$$
s_i^\star
\in
\operatorname*{arg\,min}_{\|s\|_2 \le \rho/\sqrt d} \langle g_i, s \rangle
=
- \frac{\rho}{\sqrt d} \frac{g_i}{\|g_i\|_2},
$$

where $g_i$ is the $i$-th row of $\nabla f(W^k)$.
Thus, each update direction lies on the Euclidean sphere of radius $\rho/\sqrt d$.

For any normalized hidden state, the logits satisfy

$$
z_i = \langle w_i, \hat h \rangle
= \|w_i\|_2 \|\hat h\|_2 \cos \theta_i
= \sqrt d\, \|w_i\|_2 \cos \theta_i,
$$

where $\theta_i$ is the angle between $w_i$ and $\hat h$.
Since $\|w_i\|_2 \le \rho/\sqrt d$ under the norm constraint,

$$
|z_i| \le \rho,
\qquad
z_i \in [-\rho,\rho].
$$

Equivalently, the logits are bounded, $\rho$-scaled cosine similarities.
A similar argument can be made for other norm choices.

This boundedness is restrictive for classification.
Cross-entropy does not just require the correct class to have the largest logit, it also rewards increasing the margin.
If $z_i \in [-\rho,\rho]$, then the largest possible gap between the correct logit and any incorrect logit is $2\rho$.
Consequently, even in the best case the correct-class probability is bounded by

$$
p_y
\le
\frac{e^\rho}{e^\rho + (m-1)e^{-\rho}}
=
\frac{e^{2\rho}}{e^{2\rho} + m - 1}.
$$

For large vocabularies this upper bound is small unless $\rho$ grows on the order of $\log m$.
Thus, the final layer needs sufficiently large row norms to express highly confident predictions.

From this perspective, the conclusion is not especially surprising.
After all, in separable logistic regression, the unregularized cross-entropy objective has no finite minimizer: the direction of the classifier converges, while its norm grows without bound in order to keep increasing the margin.
Some form of control is usually still needed in practice for numerical stability, e.g., through explicit regularization.
However, it is worth being aware that the constraint radius in Frank-Wolfe will directly dictate the permitted confidence of the model.

**Note**: This observation came out of a conversation with Frederik Kunstner at the [focus group in Lund](https://elliit.se/focus-period-lund-2026/).

---
```{bibliography}
:filter: docname in docnames
:labelprefix: weight-decay-
```
