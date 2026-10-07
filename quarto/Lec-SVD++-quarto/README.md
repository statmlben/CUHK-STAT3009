# SVD++ and history-derived factors

Quarto conversion of `slides/Lec-SVD++/note.tex`. The classroom sequence emphasizes the history-derived user vector, why it can fail to distinguish equal item sets, and how SVD++ adds a free user vector. The original extended algebraic derivation is condensed to the ridge/block-update connection; NMF, group MF, and smooth MF remain as a short comparison.

Render with `quarto render S.qmd` from this directory. `S.html` is a self-contained Reveal.js presentation. `figs/pop_counts.png` is copied from the original slides.

The deck uses a sum of squared errors plus `lambda` times a penalty. This is equivalent in form to using mean squared error, but the numerical interpretation of `lambda` changes with the normalization.
