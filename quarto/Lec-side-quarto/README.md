# Recommendation with Side Information · Quarto edition

Adapted from `slides/Lec-side/S.tex`. The lecture follows cold start → usable
side features → feature embeddings → linear side-aware factors → two towers.
It explicitly separates features available at prediction time from summaries
derived from ratings, which must be computed inside each fitting fold.

The dense source `linearRS.png` diagram is redrawn as two readable branches;
`two-tower.png` is copied into `figs/`.
Render with `quarto render S.qmd`; `S.html` is self-contained.
