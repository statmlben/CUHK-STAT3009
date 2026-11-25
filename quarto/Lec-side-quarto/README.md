# Recommendation with Side Information · Quarto edition

A full-length lecture on side information for rating prediction. The deck follows
one spine: **encode → fuse → evaluate**.

- **Part 0** motivates cold start with the Netflix split and the canteen dataset.
- **Parts 1–2** cover the required tabular encoders: continuous preprocessing and
  categorical embeddings, including multi-valued and high-cardinality features.
- **Part 3** compares fusion architectures: feature-only, additive MF, side-aware
  factors, “features as prior / ID as residual”, and two-tower models.
- **Part 4** covers training, leakage-safe cross-validation, warm-vs-cold
  evaluation, and the canteen ablation table.
- **Parts 5–6** are optional extensions: text via LLM embeddings and images via
  CNN/ViT encoders. They keep the same fusion interface.
- **Part 7** connects the pattern to YouTube, eBay, Alibaba, Wide & Deep, and
  DeepFM using the examples in `slides/Lec-side/S.tex`.
- **Part 8** is the coding break, practice, exit ticket, and summary.

Theme-aware SVGs in `figs/`:

- `feature-fusion.svg` — continuous and categorical features enter one MLP.
- `linear-rs.svg` — ID rows, categorical embeddings, and numeric maps build
  user and item factors.
- `two-tower.svg` — separate user and item towers with a final interaction.
- `cold-start-split.svg` — random split versus cold-start split.
- `text-image-pipeline.svg` — text/image encoders produce vectors for fusion.

The notebook `nb/STAT3009_Side.ipynb` uses `dataset/cuhk-food` to reproduce the
cold-user comparison. It uses the ALS `BiasedMF` from the MF notebook for the ID
baseline and a small PyTorch two-tower model for the side-aware model.

Render with `quarto render S.qmd`; `S.html` is self-contained.
