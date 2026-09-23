# STAT3009 Machine Learning I · Quarto edition

This directory contains a redesigned Quarto / RevealJS version of
`slides/Lec-ML/`. The original Beamer source remains unchanged.

The lecture develops supervised prediction through the California housing
regression example, then introduces the scikit-learn estimator contract through
a detailed `LinearRegression` walkthrough. A reusable four-question template—
model, learned parameters, hyperparameters, and loss or fitting criterion—is
applied to `LinearRegression` and the global-, user-, and item-mean baselines
before their estimator implementations appear. The lecture also separates
constructor settings, methods, and fitted attributes before implementing
global-mean and user-mean
recommenders with `BaseEstimator`, `RegressorMixin`, `fit`, `predict`, and
explicit RMSE evaluation.

Generalization, validation, the Netflix Prize split, and cross-validation now
live in the separate deck at `slides/Lec-ML-II-quarto/`.

Most teaching visuals are native HTML/CSS diagrams. The staged RevealJS
animations and code remain fully embedded in the generated HTML. The CUHK emblem
and the shared purple-and-gold visual system are reused from
`quarto/Lec-overview/`.

## Render

```bash
quarto render S.qmd
```

Open `S.html` in a browser. Use the arrow keys to advance through builds, `S` for
speaker view, and `B` to pause the screen. The generated HTML is self-contained.
