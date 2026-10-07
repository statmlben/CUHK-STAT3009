# STAT3009 Machine Learning I · Quarto edition

This directory contains a redesigned Quarto / RevealJS version of
`slides/Lec-ML/`. The original Beamer source remains unchanged.

The lecture moves from the input/target split and four-question template to ML
principles illustrated with California housing: model family, least-squares
fitting, and train/test roles. It then introduces the scikit-learn class and
estimator contract before showing `LinearRegression` code and RMSE on the same
housing dataset. Global-, user-, and item-mean recommenders follow. The notebook
retains its tiny array exercise for hands-on practice. For each recommender
baseline, the model and fitting criterion appear before the learned parameters
and estimator code. The item-mean implementation is left for the notebook exercise.

The deck has three planned notebook pauses. It omits repeated explanations of
the estimator lifecycle so the live coding carries those details.

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
