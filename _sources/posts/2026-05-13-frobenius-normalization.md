---
myst:
 substitutions:
  date: "2026-05-13"
---
```{post} 2026-05-13
:excerpt: 0
```

# Why Frobenius weight normalization is ok
_Posted on {{ date }}_

It has been proposed to normalize weights by the Frobenius norm even when the underlying optimizer is a non-Euclidean method such as Muon.
Is this justified? Do we preserve convergence guarantees?

**TL;DR**: If a matrix weight is immediately followed by RMSNorm, then globally rescaling that matrix does not change the function value. Frobenius normalization only performs such a global rescaling, so the descent inequality is preserved.

To make this precise, consider a matrix weight $W$ followed by RMSNorm, so the layer output depends on
$
\operatorname{RMSNorm}(Wx).
$
For simplicity, take the idealized $\epsilon=0$ version of RMSNorm and ignore any learned gain.

Now consider a steepest descent update for $W$ based on a norm $\|\cdot\|$.
For a matrix $G$, define the sharp operator by

$$
G^\#
\in
\operatorname*{arg\,max}_{S}
\left\{
\langle G, S\rangle
-
\tfrac{1}{2}\|S\|^2
\right\}
$$

and define the projection onto the Frobenius sphere $\mathcal S_F(\rho)=\{S:\|S\|_F=\rho\}$ by

$$
\Pi_{\mathcal S_F(\rho)}(W)
:=
\operatorname*{arg\,min}_{S\in\mathcal S_F(\rho)}
\|S-W\|_F^2
=
\tfrac{\rho}{\|W\|_F}W,
$$
where the last line holds for $W\neq0$.

Then we can write Frobenius normalization followed by steepest descent in the possibly non-Euclidean norm $\|\cdot\|$ as:

$$
\begin{aligned}
\widetilde W^k
&=
\Pi_{\mathcal S_F(\rho)}(W^k)\\
W^{k+1}
&=
W^k - \eta_k \left[\nabla_W f(\widetilde W^k)\right]^\#
\end{aligned}
$$

For any positive scalar $c>0$,
$
\operatorname{RMSNorm}(c\,Wx)
= \operatorname{RMSNorm}(Wx).
$
Thus Frobenius normalization leaves the normalized layer output unchanged. [^scale-invariance] 
This property propagates to the loss $f$:

[^scale-invariance]: The same invariance can also hold for weights that are not immediately followed by RMSNorm. For example, in an MLP block of the form $\operatorname{RMSNorm}(W_2 \sigma(W_1x))$, scaling $W_1$ by a positive constant leaves the mapping unchanged as long as the activation function $\sigma$ is positively homogeneous, as is the case for ReLU and ReLU$^2$.

**Scale invariance.** For every $a>0$,

$$
f(aW)=f(W).
$$

Also assume smoothness in the norm used to define the sharp operator:

**Smoothness.** The objective $f$ is $L$-smooth with respect to $\|\cdot\|$, meaning, for all $X,Y$,

$$
f(Y)
\le
f(X)
+
\langle \nabla f(X),Y-X\rangle
+
\tfrac{L}{2}\|Y-X\|^2
$$

```{prf:theorem}
Suppose scale invariance and $L$-smoothness hold.
Let $\bar\eta_k=\eta_k\tfrac{\rho}{\|W^k\|_F}$.
If $0<\bar\eta_k<2/L$, then

$$
f(\widetilde W^{k+1})
\le
f(\widetilde W^k)
-
\bar\eta_k
\left(1-\tfrac{L\bar\eta_k}{2}\right)
\left\|
\nabla_W f(\widetilde W^k)
\right\|_*^2.
$$
```

```{prf:proof}
We can write $W^k=\alpha_k\widetilde W^k$ with $\alpha_k=\|W^k\|_F/\rho>0$.
The update rule can be factored as

$$
W^{k+1}
=
\alpha_k\left(
\widetilde W^k
-
\bar\eta_k
\left[\nabla_W f(\widetilde W^k)\right]^\#
\right),
\qquad
\bar\eta_k=\tfrac{\eta_k}{\alpha_k}.
$$

This is only an algebraic rewriting of the actual update, not a different algorithm.
By scale invariance,

$$
f(W^{k+1})
=
f\left(
\widetilde W^k
-
\bar\eta_k
\left[\nabla_W f(\widetilde W^k)\right]^\#
\right).
$$

Apply $L$-smoothness with
$X=\widetilde W^k$ and
$Y=\widetilde W^k-\bar\eta_k[\nabla_W f(\widetilde W^k)]^\#$.
Writing $G_k=\nabla_W f(\widetilde W^k)$, we get

$$
f(W^{k+1})
\le
f(\widetilde W^k)
-
\bar\eta_k
\left(1-\tfrac{L\bar\eta_k}{2}\right)
\left\|
\nabla_W f(\widetilde W^k)
\right\|_*^2,
$$

where we used
$\langle G_k,G_k^\#\rangle=\|G_k^\#\|^2=\|G_k\|_*^2$, which follows from the optimality condition defining the sharp operator.
Finally, scale invariance also gives
$f(\widetilde W^{k+1})=f(W^{k+1})$, and the result follows.
```

The theorem gives a descent inequality in the normalized iterates, which can then be telescoped in the usual way, whenever the effective stepsize satisfies $0<\bar\eta_k<2/L$.

So, at least in the scale-invariant setting due to layer normalization, Frobenius normalization does not break the descent lemma.
It just chooses a representative of the same function with controlled Frobenius norm.
The argument is not specific to the Frobenius norm, but it does rely on the normalization being a global positive rescaling, so that $W^k=\alpha_k\widetilde W^k$ and the actual update can be compared to a step from $\widetilde W^k$ by changing only the effective stepsize.

Interestingly, the stepsize condition is on the effective stepsize $\bar\eta_k$ which normalizes the stepsize $\eta_k$ by $\|W^k\|_F$ instead of using $\|\text{BaseUpdate}\|_F$ as in e.g., MuonH.
Let $G_k=\nabla_W f(\widetilde W^k)$ and let
$\operatorname{lmo}(G_k)\in\operatorname*{arg\,min}_{\|S\|\le1}\langle G_k,S\rangle$, so that $G_k^\#=-\|G_k\|_*\operatorname{lmo}(G_k)$. 
Keeping the effective stepsize fixed as $\bar\eta_k\equiv \bar\eta$ corresponds to running the descent step with

$$
W^{k+1}
=
W^k
+
\bar\eta
\|W^k\|_F
\|G_k\|_*
\operatorname{lmo}(G_k),
$$
followed by the projection.
