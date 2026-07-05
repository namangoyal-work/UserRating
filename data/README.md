# Data

Two datasets ship with the repo. Both are for research/education use.

## train.csv (~39 MB, 158,433 rows)

The training corpus: headerless CSV, column 0 = review text, column 1 =
rating (integer 1–5). Originates from the COL772 (IIT Delhi) assignment 1
distribution of clothing-rental reviews.

Know its shape before you trust your metrics on it: the label distribution is
heavily skewed toward 5 stars (roughly 70% of rows), which is why the project
reports **both** micro and macro F1 and why `userating cv` prints a confusion
matrix. A model can look great on micro-F1 while never predicting a 1.

## modcloth_final_data.json (~39 MB)

The ModCloth clothing-fit dataset (Misra, Wan & McAuley, RecSys 2018 —
"Decomposing fit semantics for product size recommendation in metric spaces").
One JSON object per line: item/user measurements, `quality` (1–5), and `fit`.
Used in `notebooks/extern_data_analysis.ipynb` to sanity-check how rating
behavior generalizes beyond the assignment corpus. Not used in training.

## Rules

- Nothing else goes in this directory; git history is forever and data blobs
  are already at the comfortable limit.
- Trained models never belong in git at all (`.gitignore` enforces;
  SECURITY.md explains).
