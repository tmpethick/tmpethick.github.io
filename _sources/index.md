# Short bio

I am interested in understanding the structure underlying deep learning models and use this to build algorithms that learn efficiently and robustly. Of particular importance to me is how to make these systems learn continuously under changed environments and in multi-agent settings. In my work, I am typically on the look-out for where and why training instabilities occur, since these often help identify the barriers to both efficiency and robustness.

I strongly believe that deep learning does not need to be an art, but that it can be made into a scientific discipline through both theoretical and empirical work. Interestingly, in recent years we have seen a closing of the theory-practice gap in optimization for deep learning that is slowly leading to a unification across both training paradigms and model families such as RNNs, GANs, Transformers, and diffusion models. These developments leave many opportunities, which I am excited to explore and exploit.

I received my PhD from EPFL under Prof. Volkan Cevher, where I was broadly interested in optimization for machine learning with a focus on stable training of deep learning models.
During my studies, I interned with Amazon and ETH Zürich.


[lions]: https://lions.epfl.ch/


<!-- ### News -->


### Selected publications

See [publications](/publications/) for other publications and [Google Scholar](https://scholar.google.com/citations?user=l99LRFkAAAAJ&hl=en) for the most up to date version.

```{div} full-width
```{jinja} publications_ctx

<div class="disable-hyphens">
{% for p in publications %}
{% if p.selected %}
<a href="{{p.paper}}"><b>{{p.title}}</b></a><br/>
{{p.authors}}<br/>
<em>{{p.conference}}</em><br/>
{% if p.paper %}<a href="{{p.paper}}">paper</a>{% endif %} {% if p.code %}<a href="{{p.code}}">code</a>{% endif %} {% if p.tweet %}<a href="{{p.tweet}}">tweet</a>{% endif %}
{% if not loop.last %}<hr/>{% endif %}
{% endif %}
{% endfor %}
</div>
```

### Content

**A geometric view on optimization**

- [Polyak stepsize through a hyperplane projection interpretation](/posts/2024-06-10-polyak-stepsize/)

**Online learning**

- [Follow the regularized leader](/posts/2019-11-02-FTRL/)
- [Hedge and bandits](/posts/2020-01-06-hedge-and-bandit/)
- [Gaussian processes and Hedge for infinite armed bandits](/posts/2020-01-07-gp-mw/)

**Talks**

- [Provably beneficial artificial intelligence by Stuart Russell](/posts/2019-11-03-russell-talk/)
- [From causal inference to autoencoders and gene regulation by Caroline Uhler](/posts/2019-11-16-caroline-uhler/)

**Tidbits**

- [Various ways of writing Nesterov's acceleration](/posts/2020-06-04-acceleration-perspectives/)
- [Acceleration convergence using a potential function](/posts/2020-06-04-acceleration-with-potential-function/)
- [Gradient boosting](/posts/2020-05-15-gradientboosting/)
- [Bayesian logistic Regression](/posts/2020-05-26-bayesian-logistic-regression/)
- [The I/O model](/posts/2018-05-22-io-model/)


All the posts can also be found in chronological order in the [archive](/posts/).

<hr>

### Open source

Some of the projects I worked on prior to the PhD:

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

... and more on [Github][4] including [this site][5] which was [originally][8] build by <a href="http://jaspervdj.be/hakyll">Hakyll</a> with some added $\text{\LaTeX}$ goods. 
I have since moved to <a href="https://ebp.jupyterbook.org">the Executable Book Project</a> for a well-maintained codebase with many of the same features.

[1]: https://github.com/tmpethick/simple-prover-pl
[2]: http://pethick.dk/campusnet-electron/
[3]: https://github.com/tmpethick/anki-onenote-importer
[4]: https://github.com/tmpethick/
[5]: https://github.com/tmpethick/pethick-site
[6]: https://github.com/tmpethick/ensembled-dngo
[7]: https://arxiv.org/abs/1502.05700
[8]: https://github.com/tmpethick/pethick-site-hakyll
[master]: https://github.com/tmpethick/thesis-code

<!--

### External links

Prior to the PhD I have written on a variety of areas outside of this blog:

- My Master thesis on [Scalable Gaussian Processes for Economic Models][master].
- Bayesian Optimization using an [Ensembled Deep Network for Global Optimization][ensembled-dngo] which explores the behavior of an ensembled variant of the architecture proposed by [(Snoek et al 2015)][snoek2015] on various Bayesian Optimization benchmark problems.
- Environmental sound classification using [convolutional autoencoders][CAE] with an custom built unpooling layer in keras.
- [Bachelor][mcmc] on markov chain monte carlo and probabilistic programming.
- [Compilation][prover] of a theorem prover written in Isabelle into Prolog using catamorphism in Haskell.
- *A Process Calculus for Design and Modeling of Retro-synthesis* (published and presented at [EJC 2018 conference][ejc2018]).
- A [summary][DEL] of dynamic epistemic logic and game theory.

-->


[DEL]: https://pethick.dk/epistemic-planning-and-games.pdf
[mcmc]: https://github.com/tmpethick/mcmc
[ejc2018]: http://www.ejc-conference.org/
[prover]: https://github.com/tmpethick/simple-prover-pl
[CAE]: https://github.com/Rasmusafj/02456-deep-learning-project12/blob/master/paper/Arpethick-CAE.pdf
[02405]: http://kurser.dtu.dk/course/02405
[02180]: http://kurser.dtu.dk/course/02180
[birdback]: https://www.crunchbase.com/organization/birdback
[campusnet]: http://pethick.dk/campusnet-electron/
[ensembled-dngo]: https://github.com/tmpethick/ensembled-dngo
[snoek2015]: https://arxiv.org/abs/1502.05700
[master]: https://github.com/tmpethick/thesis-code
[lions]: https://lions.epfl.ch/
[MICRO-455]: https://edu.epfl.ch/coursebook/en/applied-machine-learning-MICRO-455
[EE-559]: https://edu.epfl.ch/coursebook/en/deep-learning-EE-559
[EE-618]: https://edu.epfl.ch/coursebook/en/theory-and-methods-for-reinforcement-learning-EE-618
[EE-556]: https://edu.epfl.ch/coursebook/en/mathematics-of-data-from-theory-to-computation-EE-556


<!--
I have had the joy of being a teaching assistant in the following courses:

- [02405 Probability theory][02405] (fall 2015)
- [02180 Introduction to artificial intelligence][02180] (Spring 2018)
- [MICRO-455 Applied machine learning][MICRO-455] (spring 2020)
- [EE-559 Deep learning][EE-559] (spring 2021)
- [EE-618 Theory and methods for reinforcement Learning][EE-618] (spring 2022)
- [EE-556 Mathematics of data: from theory to computation][EE-556] (fall 2020, fall 2021, fall 2022)
--> 

<!-- Before I went into academia I worked on [card linked loyalty][birdback]
- app development in react
- [desktop development][campusnet] with electron -->
