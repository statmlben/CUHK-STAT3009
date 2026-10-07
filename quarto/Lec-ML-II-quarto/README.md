# STAT3009 Machine Learning II · Quarto edition

This deck continues from `quarto/Lec-ML-quarto/`. Familiar recommender baselines,
memorization, and a schematic of fitting noisy data motivate generalization.
Ridge regression then provides one
continuous model-selection example: its four questions, `alpha`, holdout,
the limits of one split, K-fold cross-validation, final refitting, and a
`GridSearchCV` implementation for selecting Ridge's `alpha`.
An empirical train-versus-CV curve on a 200-row housing diagnostic sample
shows how the best fitting score can differ from the best validation score;
the full-data search remains the final selection.
The last section transfers the same data roles to rating prediction through
the Netflix Prize and fold-safe recommender baselines. It contrasts pair-level,
whole-user, and whole-item validation questions, then uses actual fold counts
to show that an overall pair-level RMSE is dominated by warm pairs. One held-out 0/1 rating
shows how a candidate weight triple becomes a prediction and squared error;
the five-fold CV table then compares four weight candidates. A
sklearn-ecosystem recap connects ID encoding, the estimator interface, and
`GridSearchCV`. The deck closes with the complete validation workflow and
connects it to the next topic, matrix factorization.

The deck reuses the shared STAT3009 theme and ML-specific styles from
`quarto/Lec-ML-quarto/ml.scss`, with local additions in `ml-ii.scss`.

## Render

```bash
quarto render S.qmd
```

Open `S.html` in a browser. Use the arrow keys to advance through builds, `S` for
speaker view, and `B` to pause the screen. The generated HTML is self-contained.
