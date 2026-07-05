# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/).

## [1.0.0] — 2026-07-04

The "flagship" release: the course submission grew into an installable,
tested, CI-gated package. The model architecture is unchanged.

### Added
- `userating` package (`src/` layout) with `pip install -e .` and a
  `userating` console command (`train` / `test` / `cv` / `predict`).
- `predict` verb: score ad-hoc text from the shell with confidence
  (`predict_proba` via `multi:softprob` — same argmax as before, now with a
  distribution).
- `eval_report`: per-class F1, MAE, and a confusion matrix alongside the
  course metric.
- Parallel preprocessing: `transform(X, n_jobs=...)` fans out across cpu cores
  above 16k reviews; output is identical to the serial path (asserted by test).
- `fit`/`predict`/`predict_proba` accept `pretransformed=True`; `cv` now
  preprocesses the corpus once and shares it across folds (stateless, no
  leakage) instead of re-preprocessing per fold.
- Deterministic training: `random_state` threaded through SVD and XGBoost;
  guarded by a test.
- Test suite (13 tests, seconds to run, synthetic corpus) and GitHub Actions
  CI (ruff + pytest + coverage on Python 3.9/3.11/3.12).
- Governance: CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md, CODEOWNERS,
  issue/PR templates, Dependabot.

### Changed
- `--limit` replaces the hard-coded `.iloc[:1000]` debug truncation — training
  now defaults to the full dataset.
- The `C`, `dim`, and `max_iter` constructor parameters are actually wired to
  the estimators (previously accepted and ignored).
- SVD dimensionality is capped at the vocabulary size, so small datasets train
  instead of crashing.
- `test` handles blank/NaN reviews (one prediction per input row, always).
- Targeted NLTK downloads (5 resources) instead of `nltk.download('popular')`.

### Unchanged, on purpose
- `2024EE30913/` — the original submission, frozen verbatim.
- The model architecture and its cross-validated scores.

## [0.1.0] — original course submission

`2024EE30913/main.py`: the stacked classifier, `train.sh`/`test.sh`, and the
writeup. Preserved in-tree.
