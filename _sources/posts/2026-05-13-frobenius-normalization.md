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

It has been proposed to normalize weights by the Frobenious norm even when the underlying optimizer is a non-Euclidean methods such as Muon.
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
\in
\operatorname*{arg\,min}_{S\in\mathcal S_F(\rho)}
\|S-W\|_F^2
=
\tfrac{\rho}{\|W\|_F}W.
$$
for $W\neq0$.

Then we can write steepest descent (in the possibly non-Euclidean norm $\|\cdot\|$) followed by an Euclidean projection as:

$$
\begin{aligned}
\widetilde W^{k+1}
&=
W^k - \eta_k \left[\nabla_W f(W^k)\right]^\#,\\
W^{k+1}
&=
\Pi_{\mathcal S_F(\rho)}(\widetilde W^{k+1}).
\end{aligned}
$$

For any positive scalar $c>0$,
$
\operatorname{RMSNorm}(c Wx)
= \operatorname{RMSNorm}(Wx).
$
Taking $c=\rho/\|\widetilde W^{k+1}\|_F$, the normalized layer output is thus unchanged:

$$
\operatorname{RMSNorm}(W^{k+1}x)
=
\operatorname{RMSNorm}(\widetilde W^{k+1}x).
$$

So the Frobenius projection preserves the function value of a normalized layer:

$$
f(W^{k+1}) = f(\widetilde W^{k+1}).
$$

The descent analyses of steepest descent reason through function values by showing that

$$
f(\widetilde W^{k+1}) < f(W^k).
$$

Since the subsequent Frobenius projection does not change the function value, then the projected iterate $W^{k+1}=\Pi_{\mathcal S_F(\rho)}(\widetilde W^{k+1})$ satisfies

$$
f(W^{k+1})
=
f(\widetilde W^{k+1})
<
f(W^k).
$$

So, at least in the scale-invariant setting due to layer normalization, the Frobenius projection does not break the descent lemma.
It just chooses a representative of the same function with controlled Frobenius norm.
