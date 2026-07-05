# UserRating

[![CI](https://github.com/namangoyal-work/UserRating/actions/workflows/ci.yml/badge.svg)](https://github.com/namangoyal-work/UserRating/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)

Turn free-text user feedback into 1–5 star ratings — with a fully classical,
fully interpretable NLP stack. No GPU, no transformer, no API key: the whole
model trains on a laptop in minutes and every stage of it can be inspected.

```text
review text
   │  tokenize → PoS-tag → negate → de-stopword → lemmatize      (preprocess.py)
   ▼
(polarity, word, tag) triples
   │
   ├── TF-IDF → SVD → LogisticRegression          ┐
   ├── TF-IDF → SVD → LR (class-balanced)         │ stacked
   ├── selected bigrams → TF-IDF → SVD → LR       │ log-probabilities
   └── per-word log-likelihood-ratio scores       ┘
   ▼
XGBoost meta-classifier  →  rating ∈ {1..5} (+ confidence)      (model.py)
```

The interesting parts: **PoS-aware negation** ("not good" becomes a `NEG_good`
token, so the unigram models see negation instead of losing it), a
**log-likelihood-ratio word model** that scores each word by how sharply it
discriminates between star levels, and **bigrams mined only around those
high-signal words** instead of exploding the feature space with every bigram.
## Quickstart

```bash
git clone https://github.com/namangoyal-work/UserRating.git
cd UserRating
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Train on the bundled 158k-review dataset and score some text:

```bash
userating train data/train.csv model.pkl
userating predict model.pkl "runs long but the color is stunning, compliments all night"
# 5 ***** (confidence 0.83)  runs long but the color is stunning, compliments all night
```

Batch-predict a CSV (one review per line) and cross-validate:

```bash
userating test model.pkl reviews.csv predictions.txt
userating cv data/train.csv          # 5-fold, with per-class F1 + confusion matrix
```

For a fast smoke run: `userating train data/train.csv model.pkl --limit 1000 --dim 200`.

As a library:

```python
from userating import SentimentClassifier, save, load

model = SentimentClassifier()
model.fit(reviews, ratings)          # list[str], list[int 1..5]
model.predict(["never buying again"])          # array([1])
model.predict_proba(["it was fine i guess"])   # (n, 5) distribution
```

## Results

5-fold cross-validated on the 158k-review dataset (`userating cv data/train.csv`),
measured on this exact codebase (Apple M4, 16 min wall clock, ~10 GB peak RAM):

| Metric | Score |
|---|---|
| F1 micro | 0.7024 |
| F1 macro | 0.3753 |
| Final (mean) | 0.5389 |
| MAE | 0.34 stars |

Per-class F1: 5★ ≈ 0.83, 4★ ≈ 0.46, 3★ ≈ 0.28, 2★ ≈ 0.19, 1★ ≈ 0.12. The
original writeup reported 0.7036 / 0.3812 / 0.5424 for the same architecture
on the library versions of its day; the small drift (−0.0035 final) is years
of scikit-learn/XGBoost releases, not a code change — per-fold finals in this
run spanned 0.536–0.541.

The micro/macro gap is the dataset talking: ~70% of reviews are 5-star, so the
rare 1–3 star classes dominate the macro penalty. The class-balanced LR head
and the LL word model exist precisely to claw back macro points.

## Repository map

```
src/userating/     the package: preprocess.py, model.py, metrics.py, cli.py
tests/             fast deterministic suite (synthetic corpus, no downloads needed beyond NLTK data)
data/              train.csv (158k labeled reviews) + modcloth_final_data.json (see data/README.md)
notebooks/         exploratory analysis the design decisions came from
scripts/           train.sh / test.sh — the original two-command interface
2024EE30913/       the original course submission, frozen verbatim (never modified)
```

## Quality gates

CI runs `ruff` and the full `pytest` suite (with coverage) on Python 3.9–3.12
on every push and PR. Determinism is a tested guarantee: two fits on the same
data produce identical predictions. See [CONTRIBUTING.md](CONTRIBUTING.md) for
the complete gate list and how to run it locally.

## Governance & security

- [CONTRIBUTING.md](CONTRIBUTING.md) — dev setup, quality gates, style, PR conventions
- [SECURITY.md](SECURITY.md) — trust model (read it before deploying; model pickles are code)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) — Contributor Covenant 2.1
- [.github/CODEOWNERS](.github/CODEOWNERS) — review ownership

## License

[MIT](LICENSE) © Naman Goyal
