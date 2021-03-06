---
title: Content
---

## Acceleration

- [Various ways of writing Nesterov's acceleration](/posts/2020-06-04-acceleration-perspectives/)
- [Acceleration convergence using a potential function](/posts/2020-06-04-acceleration-with-potential-function/)

## Online Convex Optimization

- [Follow the Regularized Leader](/posts/2019-11-02-FTRL/)
- [Hedge and Bandits](/posts/2020-01-06-hedge-and-bandit/)
- [Gaussian Processes and Hedge for infinite armed bandits](/posts/2020-01-07-gp-mw/)

## Talks

- [Provably Beneficial Artificial Intelligence by Stuart Russell](/posts/2019-11-03-russell-talk/)
- [From Causal Inference to Autoencoders and Gene Regulation by Caroline Uhler](/posts/2019-11-16-caroline-uhler/)

## Algorithms for Massive Datasets

- [The I/O model](/posts/2018-05-22-io-model/)

## Tidbits

- [Gradient Boosting](/posts/2020-05-15-gradientboosting/)
- [Bayesian Logistic Regression](/posts/2020-05-26-bayesian-logistic-regression/)

You can also find all the posts in chronological order in the [archive](/archive/).

<hr>

## Open Source

- **[Scalable Gaussian Processes for Economic Models][master]**.
  This codebase can be used to run high-dimensional scalable Gaussian Processes on Economic Models on a High Performance Computing cluster.
- **[Ensembled Deep Network for Global Optimization][6]**.
  This project explores the behavior of an ensembled variant of the architecture proposed by [(Snoek et al 2015)][7] on various Bayesian Optimization benchmark problems.
- **[Prolog code generation from Isabelle's inner syntax][1]**. 
  This project compiles a theorem prover written and proven with Isabelle and compiles it into Prolog. It does so in Haskell through several catamorphism that changes the Isabelle AST into a Prolog AST.
- **[CampusNet Sync][2]**. A Dropbox like inspired app to sync your computer with the filesystem used at the Technical University of Denmark.
- **[Anki Onenote importer][3]**. 
  Allows one to import `.mht` files exported from OneNote into Anki.
<!-- - Haskell Spanning Tree -->

... and more on [Github][4] including [this site][5] which is build by <a href="http://jaspervdj.be/hakyll">Hakyll</a> with some added $\text{\LaTeX}$ goods.

[1]: https://github.com/tmpethick/simple-prover-pl
[2]: http://pethick.dk/campusnet-electron/
[3]: https://github.com/tmpethick/anki-onenote-importer
[4]: https://github.com/tmpethick/
[5]: https://github.com/tmpethick/tmpethick.github.io
[6]: https://github.com/tmpethick/ensembled-dngo
[7]: https://arxiv.org/abs/1502.05700
[master]: https://github.com/tmpethick/thesis-code
