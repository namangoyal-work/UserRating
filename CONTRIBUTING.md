# Contributing to UserRating

Thanks for considering a contribution. This document is the contract: if your PR
follows it, review is fast; if it doesn't, the first review round will be about
process instead of your idea, which wastes everyone's time.

## Ground rules

- **Be kind.** See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
- **Open an issue before large changes.** Bug fixes and doc fixes can go straight
  to a PR; new features, new dependencies, or anything touching the model
  architecture should be discussed in an issue first so you don't build something
  that won't be merged.
- **`2024EE30913/` is frozen.** It is the original course submission, kept
  verbatim as a historical artifact. PRs that modify it will be closed — the
  living code is in `src/userating/`.
- **Do not commit data or models.** `data/` ships two reference datasets; nothing
  else belongs in git. Trained models (`*.pkl`) are ignored by `.gitignore` and
  must stay that way (see [SECURITY.md](SECURITY.md) for why pickles are a
  trust boundary).

## Development setup

```bash
git clone https://github.com/namangoyal-work/UserRating.git
cd UserRating
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

NLTK resources download automatically on first use (only the five resources the
pipeline needs, not `nltk.download('popular')`).

## Quality gates

Every PR must pass all of these. CI enforces them on every push and pull
request; run them locally first:

| Gate | Command | What it catches |
|---|---|---|
| Lint | `ruff check .` | style drift, unused imports, real bugs (`B` rules) |
| Tests | `pytest` | behavioral regressions; the suite runs in seconds on a synthetic corpus |
| Coverage | `pytest --cov=userating` | untested new code — don't lower the number you inherit |

Additional gates that are on you, not CI:

- **Determinism.** `SentimentClassifier` is seeded end-to-end
  (`random_state`); two fits on the same data must produce identical
  predictions (`tests/test_model.py::test_deterministic_across_fits` guards
  this). If your change introduces nondeterminism, it needs a very good reason
  and a parameter to control it.
- **Metric honesty.** If your change claims to improve the model, include
  `userating cv data/train.csv` numbers (before/after) in the PR description.
  Cross-validated, on the full data — not a lucky split. One good fold is not
  evidence.
- **No silent behavior changes.** The `train`/`test` CLI verbs and their
  argument order are a stable interface (they mirror the original
  `train.sh`/`test.sh` contract). Breaking them is a major-version event.

## Style

Match the house style you see in `src/userating/` rather than reformatting it:

- comments are lowercase, terse, and explain *why*, not *what*
- compact code is preferred over defensive boilerplate; list comprehensions are
  idiomatic here
- keep the preprocessing pipeline stateless (functions in `preprocess.py`);
  anything with fitted state lives on the model class
- `ruff` is configured (`pyproject.toml`) to accept this style — if you're
  fighting the linter, you're probably fighting the style

## Commit and PR conventions

- Small, focused commits with imperative subjects ("add MAE to eval report",
  not "updated stuff").
- One logical change per PR. Refactors and behavior changes go in separate PRs.
- Fill in the PR template. "What / Why / How verified" — the *verified* section
  is the one reviewers read first.
## Reporting bugs / requesting features

Use the issue templates. For security problems, **do not open a public issue** —
follow [SECURITY.md](SECURITY.md).
