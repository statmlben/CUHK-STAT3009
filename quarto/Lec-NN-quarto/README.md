# Neural Networks

Adapted from `slides/Lec-NN/note.tex`. The deck is organized around four
implementation modules:

1. **Data** — `TensorDataset` + `DataLoader`
2. **Model** — `nn.Linear`, `nn.ReLU`, `nn.Sequential`, `nn.Module`
3. **Loss** — `nn.MSELoss`
4. **Optimization** — the SGD update and its mathematics

The deck focuses on the **mathematics and intuition of SGD**: full-batch
gradient descent, one step, higher dimensions, mini-batches, and SGD terminology.
It shows the five-line pure-PyTorch training loop once, then introduces skorch
as a thin wrapper around that loop. Evaluation internals stay in the notebook.

The model-building section uses a LEGO-style brick metaphor: each layer is a
brick with an input and output shape, `nn.Sequential` is the baseplate, and
`nn.Module` is a custom build for branched models. The deck shows the explicit `MLP` layout and the branch-and-merge
`ParallelMLP` example. A shape-mismatch slide shows why `Linear(8,32)` cannot
connect directly to `Linear(16,1)`.

The evaluation section covers validation, RMSE/MAE, and the validation curve.
Cross-validation is presented through K-fold, holdout, and `GridSearchCV`;
`refit=True` selects the tuned model for the final test report.

The companion notebook `nb/STAT3009_NN.ipynb` uses skorch as a thin
sklearn-style wrapper (`NeuralNetRegressor`, `.fit`, `.predict`, `.history`).
It first trains a fixed-epoch baseline and inspects the validation curve, then
uses `GridSearchCV(refit=True)` and `gs.best_estimator_` for the final test.
`MLP.__init__` pins `torch.manual_seed(0)` so CV folds and the refit start from
the same weights.

The notebook first trains five epochs with a pure PyTorch loop
(`zero_grad → forward → loss → backward → step`), then repeats the workflow
with skorch. It also demonstrates model construction with a deeper
`nn.Sequential` stack and a `ConcatMLP` that concatenates two 4-dimensional
branches into an 8-dimensional feature vector. These are the brick-by-brick
examples that prepare the NCF pattern.

Visual explanation slides include a full-gradient vs batch-gradient sampling diagram, a batch/epoch strip, a K-fold cross-validation grid, and a layer-by-layer backpropagation diagram.

Theme-aware figures live in `figs/`: `nn-architecture.svg`, `relu.svg`,
`gradient-step.svg`, `validation-curve.svg`, and `backprop.svg`. Local layout rules, including
the brick components and dense-code classes, are in `nn.scss`; the shared course
theme and lesson components remain in `../Lec-overview/stat3009.scss` and
`../lecture-decks.scss`.

Render with `quarto render S.qmd`; `S.html` is self-contained. Speaker notes
include teaching cues and PyTorch API links.
