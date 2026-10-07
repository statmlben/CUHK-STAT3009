# Neural Collaborative Filtering · Quarto edition

Adapted from `slides/Lec-ncf/S.tex`. The main path goes from MF to embedding
lookups, then to a nonlinear interaction network and an additive MF + neural
extension. A small optional slide retains the original's more general
coordinate-wise interaction. The original NCF research used implicit feedback;
the course deck explicitly labels its rating-regression adaptation.

The source figures `MF2NN.png` and `NCF.png` are copied from the original deck
into `figs/`. Render with `quarto render S.qmd`; `S.html` is self-contained.
