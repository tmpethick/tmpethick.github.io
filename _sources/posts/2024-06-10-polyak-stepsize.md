---
myst:
 substitutions:
  date: "2024-06-10"
---
```{post} 2024-06-10
:tags: adaptivity
:excerpt: 0
```

```{math}
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
```

# Polyak stepsize through a hyperplane projection interpretation
_Posted on {{ date }}_

The gradient method can obtain a $\mathcal O(1/\sqrt{T})$ rate for nonsmooth problems
and a $\mathcal O(1/T)$ for smooth problems.
However, the stepsize needs to be set differently (decreasing and constant respectively) and requires knowledge of the problem constants.
In this blog post we will look at a very simple adaptive stepsize choice that circumvents these problems:

```{prf:algorithm} Gradient method with Polyak stepsize
:label: alg-polyak

1. Compute $g^k \in \partial f(x^k)$
2. Update $x^{k+1} = x^k - \gamma_k g^k \quad \text{with}\quad 
\gamma_k = \tfrac{f(x^k) - f(x^\star)}{\| g^k\|^2}$
```

The scheme only requires knowledge of the optimal function value $f(x^\star)$.
The adaptive stepsize choice $\gamma_k$ might look mysterious at first, but as we will see promptly, it appears very naturally when constructing a halfspace containing our solutions.


## Problem formulation 

We will work under the following assumption, 
which includes all convex functions.

```{prf:definition} Star-convex
:label: star-convex

A function $f: \mathbb R^d \rightarrow \mathbb R$ is star-convex with respect to all $x^\star \in \mathcal X^\star \subseteq \argmin_x f(x)$, if for all $g \in \partial f(x)$ and $x \in \mathbb R^d$

$$
\braket{g, x-x^\star} \geq f(x) - f(x^\star).
$$
```


## Method

Say we are given an iterate $x^k \in \mathbb R^d$ and an associated subgradient $g^k \in \partial f(x^k)$. 
We can construct a halfspace that contains the solution set $\mathcal X^\star$:

$$
\mathcal D(x^k) = \set{w\mid \braket{g^k, x^k-w} \geq f(x^k) - f(x^\star)}.
$$ (halfspace)

It is easy to verify that the solution set is contained by observing that {eq}`halfspace` reduces to {prf:ref}`star-convex` when taking $w=x^\star$.

If we can project onto $\mathcal D(x^k)$ we know that we would make progress towards a solution (as long as $\operatorname{fix} \boldsymbol \Pi_{\mathcal{D}(x)}(x) \subseteq \argmin_x f(x)$).
Fortunately, the projection unto a halfspace is simple.

```{prf:lemma}
The projection $\boldsymbol \Pi_{\mathcal{C}}(x):=\argmin_{z \in \mathcal{C}} \|z-x\|^2$ onto the set $\mathcal{C} = \{z \mid \braket{a,z} \geq b\}$ of $x \not \in \mathcal{C}$ is given as,

$$
\boldsymbol \Pi_{\mathcal{C}}(x) = x - \frac{\braket{a,x}-b}{\|a\|^2} a.
$$
```

Applying the above lemma to our particular case, by taking $\mathcal C= \mathcal D(x^k)$ such that $a=- g^k$ and $b=-\braket{ g^k,x^k} + f(x^k) - f(x^\star)$, leads to:

$$
P(x^k):=\boldsymbol \Pi_{\mathcal{D}(x^k)}(x^k) = x^k - \gamma_k g^k \quad \text{with}\quad 
\gamma_k = \tfrac{f(x^k) - f(x^\star)}{\| g^k\|^2}
$$

when $x^k \notin \mathcal D(x^k)$ and otherwise $P(x^k)=x^k$.
We can apply the projection iteratively, i.e. $x^{k+1}=P(x^k)$.
This is the (sub)gradient method with an adaptive stepsize provided in {prf:ref}`alg-polyak`. 
In other words we do not need to specify a stepsize!


## Analysis

Convergence analysis is straightforward, since we can rely on firmly quasi-nonexpansiveness of the projection to ensure convergence, i.e. 

$$
\|P(x^k) - w^\star\|^2 \leq \|x^k - w^\star\|^2 - \|x^k - P(x^k)\|^2
$$

where $w^\star\in \operatorname{fix} P$.
We just need to convince ourselves that $w^\star$ also provides a solution to our original problem, i.e. $\operatorname{fix} P \subseteq \argmin_x f(x)$.
If $x^k \notin \mathcal D(x^k)$ then it is easy to see that the the subgradient $g^k=0$ (which implies $x^k$ being a minimizer).
On the other hand, if $x^k \in \mathcal D(x^k)$ then we know $0 \geq f(x^k) - f(x^\star)$, which also implies $x^k \in \argmin_x f(x)$.
Thus, under star-convexity alone, nonasymptotic convergence is established.

Rates follows directly from lower bounding the stepsize $\gamma_k$ as made precise in the following theorem.

```{prf:theorem}
Suppose {prf:ref}`star-convex` holds for $f: \mathbb R^d \rightarrow \mathbb R$.
Then $(x^k)_{k\in \mathbb N}$ generated by {prf:ref}`alg-polyak` satisfies the following for all $x^\star \in \mathcal X^\star$

1. If $f$ has bounded subgradients, i.e. $\|g^k\| \leq G$ for all $k\in \mathbb N$, then

    $$
    \min_{k=0,...,K-1} f(x^k) - f(x^\star)
        \leq \tfrac{\| x^0 - x^\star \|}{G \sqrt{K}}
    $$

2. If $f$ has $L$-Lipschitz continuous gradients, then

    $$
    \min_{k=0,...,K-1} f(x^k) - f(x^\star)
        \leq \tfrac{\| x^0 - x^\star \|^2}{2L K}
    $$
```

```{prf:proof}
By simply expanding the update rule we have

$$
\begin{align*}
\| P(x^k) - x^\star \|^2
    &= \| x^k - \gamma_k  g^k - x^\star \|^2 \\
    &= \| x^k - x^\star \|^2 + \gamma_k^2\| g^k\|^2 - 2\gamma_k \braket{ g^k, x^k - x^\star} \\
    &\leq \| x^k - x^\star \|^2 + \gamma_k^2\| g^k\|^2 - 2\gamma_k (f(x^k) - f(x^\star))
\end{align*}
$$
where the last line uses star-convexity ({prf:ref}`star-convex`).
Using the definition of $\gamma_k$ we establish the following descent inequality

$$
\| P(x^k) - x^\star \|^2
    \leq \| x^k - x^\star \|^2 - \tfrac{(f(x^k) - f(x^\star))^2}{\| g^k\|^2}.
$$ (fejer)

What remains is to argue that the denominator $\| g^k\|^2$ remains bounded. 
We proceed case by case.

**For the nonsmooth case**: 
By assumption, $\|g^k\|\leq G$, so by telescoping {eq}`fejer` we immediately establish

$$
\begin{align*}
\tfrac{1}{K} \textstyle \sum_{k=0}^{K-1} \mathbb (f(x^k) - f(x^\star))^2
    &\leq \tfrac{\| x^0 - x^\star \|^2}{G^2 K}
\end{align*}
$$

The first claim follows by noting that the minimum is smaller than the average and that the square root can be brought inside the minimum due to monotonicity.

**For the smooth case**:
By assumption $\nabla f$ is $L$-Lipschitz continuous, which provides a better upper bound on the gradient norm, namely

$$
\|\nabla f(x)\|^2 \leq 2L(f(x) - \inf_{x \in \mathbb R^d} f(x))
$$

Consequently, proceeding from {eq}`fejer` we have

$$
\begin{align*}
\tfrac{1}{K} \textstyle \sum_{k=0}^{K-1}(f(x^k) - f(x^\star))
    &\leq \tfrac{\| x^0 - x^\star \|^2}{2L K}
\end{align*}
$$

Observing that the minimum is smaller than the average completes the proof.
```

## Conclusion

There are a couple of things we did not cover:

- The Polyak stepsize can also adapt to strongly convexity, which is treated in {cite:t}`hazan2019revisiting`.
- The method requires knowledge of the optimal value, $f(x^\star)$. {cite:t}`hazan2019revisiting` removes this requirement by paying a logarithmic factor in the complexity (provided we have a lower bound on $f(x^\star)$).
- The stochastic case {cite:p}`gower2021stochastic,garrigos2023function`, which also uses a hyperplane projection approach.
- See Ch. 5 in {cite:t}`boyd2003subgradient` for an example of the subgradient method with Polyak stepsize ({prf:ref}`alg-polyak`) applied to finding a point in the intersection of convex sets.

<!-- p. 142 of Polyak's original book simply states the stepsize, Ch. 4.1 of Boyd's book motived the stepsize intuitively through analysis  -->

---
```{bibliography}
:filter: docname in docnames
```
