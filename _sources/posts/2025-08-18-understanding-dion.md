---
myst:
 substitutions:
  date: "2025-08-18"
---
```{post} 2025-08-18
:excerpt: 0
```

```{math}
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
```

# Understanding Dion
_Posted on {{ date }}_ 

Here are some notes for better understanding Dion in terms of the changes made from PowerSGD.

Let us focus on the centralized setting and ignore momentum for simplicity. 
PowerSGD has the following structure:

$$
\begin{aligned}
\Delta &\leftarrow g + e  \\
e &\leftarrow \Delta - \sum_{i=1}^r \sigma_i u_iv^\top_i \\
x &\leftarrow x - \gamma \sum_{i=1}^r \sigma_i u_iv^\top_i
\end{aligned}
$$
where $\gamma > 0$ and $\sum_{i=1}^r \sigma_i u_iv^\top_i$ is the rank-$r$ truncated SVD of $\Delta \in \mathbb{R}^{n \times m}$.

Let us break the steps down:

1. $\Delta$ constitutes both of the gradient $g$ and the remaining residual $e$ (the unprocessed parts of previous gradients)
2. the residual $e$ is updated by subtracting the current low rank approximation from the gradient
3. the iterate $x$ is updated by using the low rank approximation

Instead Dion proceeds as follows:

$$
\begin{aligned}
\Delta &\leftarrow g + e  \\
e &\leftarrow \Delta - \sum_{i=1}^r \sigma_i u_iv^\top_i \\
x &\leftarrow x - \gamma \sum_{i=1}^r u_iv^\top_i
\end{aligned}
$$
where $\sum_{i=1}^r \sigma_i u_iv^\top_i$ is still the truncated SVD of $\Delta$.
When $r = \min\{n,m\}$, then $\sum_{i=1}^r u_iv^\top_i$ is simply the solution to the linear minimization oracle over the spectral norm ball.

The only difference with PowerSGD is the fact that Dion _sets all singular values to 1_ in the update of $x$. 
Importantly, the update of $e$ still involves the singular values $\sigma_i$.
This ensures that the residual indeed goes to zero if the same gradient $g$ was fed repeatedly over multiple iterations.
Specifically, in the above case where the SVD is exact, it will take $\operatorname{ceil}(r/n)$ iterations where $n$ is the rank of $\Delta$ for the residual to become exactly zero.

## SVD approximation
Dion adopts the _power iteration_ used in PowerSGD in order to approximate the low-rank approximation.


```{prf:algorithm} PowerIter1($B;Q$)
$$\begin{aligned}
& P \leftarrow BQ \\
& P \leftarrow \operatorname{orthogonalize}(P) && \text{with QR decomposition} \\ 
& R \leftarrow B^\top P \\
& Q \leftarrow \operatorname{ColumnNormalize}(R) \\
& \textbf{return } P,R,Q
\end{aligned}$$
```

The above procedure computes one step of power iteration on the matrix $B$ using $Q$ as the initial guess for the orthonormal basis.
The output of the algorithm provides a way to compute the two quantities that Dion relies on:

- $PR^\top = \sum_{i=1}^r \tilde{\sigma}_i \tilde{u}_i\tilde{v}^\top_i$
- $PQ^\top = \sum_{i=1}^r \tilde{u}_i\tilde{v}^\top_i$

where the tilde notation is simply to indicate that the truncated SVD is not exact but rather an approximation.

In both PowerSGD and Dion the power iteration is warmstarted with the $Q$ from the previous iteration, so the update for e.g., Dion, becomes

$$
\begin{aligned}
\Delta &\leftarrow g + e  \\
P,R,Q &\leftarrow \operatorname{PowerIter1}(\Delta;Q) \\
e &\leftarrow \Delta - PR^\top \\
x &\leftarrow x - \gamma PQ^\top
\end{aligned}
$$

