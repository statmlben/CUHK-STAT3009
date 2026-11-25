# Recommendation with Side Information · Quarto edition

A 41-slide lecture built around one spine: **encode → fuse → evaluate**.

- **Part 0** motivates side information with the Amazon Reviews 2023 Video
  Games subset: 2,000 products, 39,796 ratings, 29,209 users, 300 images.
- **Part 1** covers continuous features: `price` and `user_count`, training-only
  preprocessing, and three fusion diagrams (early concat, additive MF, two
  towers).
- **Part 2** covers categorical features: store and category embeddings, then
  a continuous + categorical fusion diagram.
- **Part 3** compares fusion architectures: feature-only, additive MF,
  side-aware factors, and two-tower models, each paired with its equation.
- **Part 4** covers the four questions, MSE, the named-input dict, a compact
  PyTorch sketch, random-row evaluation, leakage, and the Amazon ablation.
  On this split, text is the only clear gain; continuous and categorical are
  flat, and image is neutral on its 300-product subset.
- **Part 5** is optional text: product title/description → a MiniLM
  `feature-extraction` pipeline → mean-pooled 384-d vectors → a small additive
  block. Review text is not used because it is written with the rating.
- **Part 6** is optional image: product photos → a Hugging Face
  `image-feature-extraction` pipeline with `microsoft/resnet-18` → flattened
  512-d vectors → a small additive block. The image ablation is honest about
  the 300-image subset.
- **Part 7** is optional: network diagrams from the original LaTeX lecture for
  YouTube, eBay, and Alibaba, plus the Wide & Deep / DeepFM connection.
- The deck closes with **Summary** and **References**.

Theme-aware SVGs in `figs/`:

- `shape-contract.svg` — the ID-only network and its shape contract.
- `modality-loading.svg` — how continuous, categorical, text, and image blocks
  enter one recommender head.
- `categorical-id.svg` — user ID and a general categorical feature share the
  same embedding lookup.
- `feature-fusion.svg` — continuous and categorical features enter one MLP.
- `continuous-fusion.svg` — early concat, additive MF, and two towers side by
  side.
- `feature-only-additive.svg` — feature-only model versus additive MF.
- `side-aware-factors.svg` — ID residual + feature map + category prior, then a
  dot product.
- `two-tower.svg` — separate user and item towers with a final interaction.
- `youtube.png`, `ebay.png`, `alibaba.png` — industry network diagrams from
  `slides/Lec-side/figs`.
- `amazon-product.jpg` — a real product image from the demo subset.

The notebook `nb/STAT3009_Side.ipynb` starts from the NCF `MFNet` and adds one
block at a time with `skorch.NeuralNetRegressor`: continuous features,
categorical embeddings, a MiniLM text projection, and a ResNet18 image
ablation. The target is the continuous 1–5 star rating; models are evaluated
with validation and test RMSE on a random 80/20 rating-row split.

Render with `quarto render S.qmd`; `S.html` is self-contained.
