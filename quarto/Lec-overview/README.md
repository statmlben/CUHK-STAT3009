# STAT3009 Lec-overview · Quarto edition

This is a redesigned 16:9 Quarto/RevealJS version of the original Beamer deck.

The theme integrates the user-provided CUHK emblem and the University's classic purple-and-gold visual identity. The original emblem file is copied to `figs/cu_logo.webp`. The title, section dividers, and closing slide retain the atmospheric grid-and-glow treatment; teaching slides use a quieter charcoal canvas and restrained brand accents.

The deck follows five stages: course purpose, course administration, software preparation, a Python refresher, and recommendation setup. The AI discussion begins with a knowledge landscape connecting linear algebra to matrix factorization, statistics to evaluation, and coding to computational experiments. A familiar-versus-new-questions analogy introduces evaluation before any model code appears. The training-error code example returns after the feature/target arrays and RMSE have been introduced.

Software preparation links directly to the verified course Jupyter notebook in Colab and explains how to download an `.ipynb` for VS Code. Markdown uses one source-versus-preview slide and one technical-notation slide. The programming refresher covers Python, NumPy, and Pandas. Code examples use larger type with manually wrapped long lines.

The recommendation section uses two concise pages for the online/offline system and the course's modeling scope. Netflix examples distinguish observed pairs from cold-start users and movies, separate training/validation/test roles, and rank candidates that the example user has not rated. Numbers and candidate eligibility were checked against the local Netflix CSV files. Chinese speaker notes support the new conceptual explanations.

Assessment weights are homework 15%, open-book in-class Kaggle 40% approximately mid-semester, and the final coding quiz 45% in the last class. The repository's grading page uses the same weights and timing.

Brand-colour reference: CUHK visual-identity materials list purple as RGB 117/15/109 and gold as RGB 221/163/0: https://www.50.cuhk.edu.hk/en/logo

## Render

```bash
quarto render S.qmd
```

Open `S.html` in a browser. Use the arrow keys to navigate, `S` for speaker view, and `B` to pause the screen.

The generated HTML is self-contained (`embed-resources: true`). Equations use native MathML so they remain visible without loading a math library from the network. The source images used by the deck are copied into `figs/` so the project can be moved independently.
