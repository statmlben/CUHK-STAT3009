# STAT3009 Baseline Methods · Quarto edition

This directory contains the Quarto / RevealJS redesign of `slides/Lec-baseline/`.
The original Beamer source remains unchanged.

The deck reuses the CUHK visual system from `slides/Lec-overview-quarto/`, including
the atmospheric title and section backgrounds, restrained teaching-slide canvas,
code treatment, footer, and lower-right CUHK emblem. Original course figures remain
in `slides/Lec-baseline/figs/` and are embedded into the generated HTML.

The revised sequence assumes students have completed the overview lecture and avoids
repeating its feedback-loop, production-pipeline, data-schema, and ranking setup.
Instead, it begins with the public STAT3009 Netflix course subset and lets the data
motivate the models. Students inspect the rating distribution, sparsity, interaction
counts, train/test pair structure, and warm/cold-start cases before implementing the
global, user, item, and additive baselines.

Displayed benchmark values are calculated from the course files. The global, user,
item, and additive methods obtain test RMSE values of 1.085, 1.017, 1.052, and 0.976,
respectively. The final slides connect these prediction scores back to ranking and
use the remaining interaction error to motivate collaborative filtering.

## Render

```bash
quarto render S.qmd
```

Open `S.html` in a browser. Use the arrow keys to navigate, `S` for speaker view,
and `B` to pause the screen. The generated HTML is self-contained.
