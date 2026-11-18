# Neural Collaborative Filtering

This deck is the NCF half of the lecture. The remaining Neural Networks material
(data, model, loss, pure PyTorch loop, validation, and `GridSearchCV`) stays in
`quarto/Lec-NN-quarto/S.html`; use that deck first, then continue here.

## Slides

The deck is NCF-only:

1. **Where we are** — the training workflow is already in the NN deck.
2. **Embedding lookups** — an ID selects a learned row; lookup equals a one-hot
   matrix product.
3. **MF as a network** — two embedding branches and a dot product.
4. **NCF** — concatenate the two embeddings and replace the dot product with a
   small MLP; the shape contract is `(B,) → (B,K) → (B,2K) → (B,H) → (B,1)`.
5. **Variants** — biased NCF and additive NCF (MF path + neural correction).
6. **Training and comparison** — the same skorch wrapper, a fixed 50-epoch
   budget, validation curves, and a small `GridSearchCV`.
7. **Practice and wrap-up** — tuning exercise, debrief, exit ticket, and the
   transition to side information.

There is an explicit **coding break** before the MF/NCF comparison: it tells the
instructor to pause the slides, open `nb/STAT3009_NCF.ipynb`, and run Sections
2–6. The slide includes a three-step checklist and a suggested 8–10 minute
window.

A **CV snapshot** slide shows the `GridSearchCV` table over `K` and hidden
width, including the selected setting and its test RMSE. An **extra practice**
slide lists GMF, two-tower, and NeuMF as optional extensions; Section 8 of the
notebook contains the skeletons and solutions.

Two closing perspective slides connect the same embedding lookup to NLP and
LLMs: token embeddings at the Transformer input, and query/document embeddings
for retrieval-augmented generation (RAG).

The deck uses vanilla SGD throughout. It does not use Adam, early stopping,
`torch.no_grad()`, GPU/device material, or an L2/Gaussian-prior framing.

## Notebook

`nb/STAT3009_NCF.ipynb` is the NCF-only notebook:

- **Sections 1–2** load and encode the Netflix 1–5 star split.
- **Sections 3–6** build `MFNet`, `BiasedNCF`, and `ANCF`, train them for a
  fixed 50-epoch budget, and compare validation/test RMSE.
- **Section 7** is the in-class practice, including a small `GridSearchCV`.

The Neural Networks warm-up lives in `nb/STAT3009_NN.ipynb`. `MFNet`,
`BiasedNCF`, and `ANCF` pin `torch.manual_seed(0)` inside `__init__`, so CV
folds and the final refit start from the same initialization.

## Figures

Theme-aware SVGs in `figs/`:

- `embedding-lookup.svg` — a user-ID table with one row selected.
- `embedding-batch.svg` — a batch of IDs gathers several rows in input order.
- `embedding-size.svg` — the `n_users × K` table; `K` is chosen, the entries are learned.
- `side-info.svg` — a known ID uses a learned row; a new ID has no row, so side
  features go through an MLP to build a vector.
- `one-hot-product.svg` — a one-hot row times the table selects one row.
- `mf-two-branch.svg` — MF drawn as two embedding branches and a dot product.
- `ncf-shapes.svg` — the `(B,) → (B,K) → (B,2K) → (B,H) → (B,1)` contract.
- `ncf-flow.svg` — the end-to-end NCF graph: two lookups, concatenation,
  hidden layers, and one score.
- `cv-bars.svg` — the cross-validation comparison over `K` and hidden width.
- `validation-curve.svg` — copied from the Neural Networks deck for the
  overfitting slide.
- `token-pipeline.svg` — text → token IDs → token embeddings → Transformer.
- `rag-pipeline.svg` — query and documents → embeddings → cosine similarity →
  top-k → prompt + LLM.

Local dense-code rules are in `ncf.scss`. The shared course theme and lesson
components remain in `../Lec-overview/stat3009.scss` and
`../lecture-decks.scss`.

## Render

```bash
quarto render S.qmd
```

`S.html` is self-contained. Speaker notes include teaching cues and API links.
