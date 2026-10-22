# Midterm rehearsal (mini)

Open `STAT3009_Midterm_Rehearsal.ipynb` from this folder. Suggested time: 35–40 minutes.

This is **not** the real midterm and does **not** use the real exam data. It is a short
synthetic rehearsal with a related but deliberately different workflow:

1. read `user, item, rating` data;
2. write an `encode_ratings()` helper;
3. write a `fit_and_score()` helper and compare baselines;
4. fit the provided `BiasedMF`;
5. tune `K` and `lambda_` with **4-fold** `GridSearchCV`;
6. compare local RMSE and wrap up.

`test_labels.csv` is included only so students can compute local RMSE. The real
exam hides test labels and uses `submit()`.

Files:

- `raw_ratings.csv` — 1,598 synthetic training ratings.
- `test_pairs.csv` — 402 held-out pairs.
- `test_labels.csv` — ratings for local scoring.
- `TabRS.py` — supplied models (`GlobalMeanRS`, `UserMeanRS`, `ItemMeanRS`, `BiasedMF`).
- `STAT3009_Midterm_Rehearsal_solution.ipynb` — fully executed instructor solution.
