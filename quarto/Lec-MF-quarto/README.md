# STAT3009 Matrix Factorization · Quarto edition

This RevealJS deck adapts `slides/Lec-MF/note.tex` into the course's Quarto
style. It moves from baseline limitations to latent factors, states the four
model questions before introducing the estimator, and connects ALS updates to
the Ridge regression and cross-validation workflow taught in ML II. Repeated
overfitting illustrations and the long coordinate-descent derivation from the
original have been condensed into one conceptual bridge and two readable
Ridge subproblems.

Slides 4–6 use one two-factor film example throughout: shared user/film
profiles, two hand-calculated dot products, and the resulting full prediction
matrix. The action/romance labels are explicitly an illustration; actual MF
factors are learned from ratings and may not have simple names.

The deck uses the **sum of squared errors** convention in its fitting
criterion. Consequently, each ALS subproblem maps directly to
`Ridge(alpha=lambda_, fit_intercept=False)`. Validation and test performance
are still reported with RMSE on held-out observed ratings.

Render with `quarto render S.qmd`, then open `S.html`. The HTML is
self-contained. Speaker notes contain teaching prompts and technical caveats.
