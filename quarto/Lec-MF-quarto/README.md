# STAT3009 Matrix Factorization · Quarto edition

This RevealJS deck adapts `slides/Lec-MF/note.tex` into the course's Quarto
style. A four-rating counterexample shows why global/user/item means miss
user–item interactions. Biased MF keeps a global mean and learned user/item
biases, then adds a factor dot product for pair-specific effects. The deck
then moves to latent factors, states the four
model questions before introducing the estimator, and connects ALS updates to
the Ridge regression and cross-validation workflow taught in ML II. Repeated
overfitting illustrations and the long coordinate-descent derivation from the
original have been condensed into one conceptual bridge and two readable
Ridge subproblems.

Immediately after the model definition, a parameter diagram shows the selected
user position in $a$ and row in $P$, alongside the selected item position in
$b$ and row in $Q$. A three-click RevealJS build then traces one $(u,i)$ input
through parameter lookup, baseline/interaction calculation, and the final
rating prediction. It uses the same illustrative numbers as the earlier slides.

Slides 4–7 use numeric user/item IDs in one two-factor film example throughout: shared user/film
profiles, an explicit distinction between observed ratings and learned latent
factors, and pairwise predictions for two users and two films. The
action/romance labels are explicitly an illustration; actual MF
factors are learned from ratings and may not have simple names.

The deck uses the **sum of squared errors** convention in its fitting
criterion. The global mean comes from fitting ratings and stays fixed during
ALS. The regularization parameter penalizes the latent factors. Each ALS
subproblem fits a user or item bias as an unpenalized Ridge intercept and its
factor vector as Ridge coefficients via `Ridge(alpha=lambda_, fit_intercept=True)`.
Validation and test performance
are still reported with RMSE on held-out observed ratings.

Render with `quarto render S.qmd`, then open `S.html`. The HTML is
self-contained. Speaker notes contain teaching prompts and technical caveats.
