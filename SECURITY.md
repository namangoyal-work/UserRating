# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 1.x (main) | ✅ |
| the frozen `2024EE30913/` submission | ❌ historical artifact, never patched |

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Instead, use
GitHub's [private vulnerability reporting](https://github.com/namangoyal-work/UserRating/security/advisories/new)
for this repository. You can expect an acknowledgment within a week. If the
report is valid, a fix ships before the details are published, and you get
credit in the advisory unless you'd rather not.

## The trust model — read this before deploying

This is a text-classification library. Its security surface is small but real,
and it is documented here rather than left for you to discover:

### 1. Model files are code (pickle)

Models are persisted with `pickle`. **Unpickling a file executes arbitrary code
contained in it.** This is an inherent property of pickle, not a bug here.
The rule:

> Only load model files **you trained yourself** or received from a source you
> trust as much as you trust any installed package.

Never load a model file from an untrusted user, an email attachment, or a
public bucket. This is why `.gitignore` excludes `*.pkl` and why the repo will
never ship a pre-trained binary blob — a poisoned "convenience model" is the
most realistic attack on a project like this.

### 2. Untrusted input text is in scope

`predict`/`test` are designed to consume arbitrary user reviews, so hostile
*text* must be safe. The pipeline (tokenize → tag → negate → lemmatize →
TF-IDF) does no `eval`, no format-string expansion, and no filesystem access
derived from input content. Pathological inputs (very long reviews, exotic
Unicode) cost CPU time but do not change the trust boundary. If you find an
input that does more than waste time, that's a vulnerability — report it.

### 3. Training data poisoning is out of scope for the library, in scope for you

The model learns whatever the training CSV says. If an attacker can write to
your training data, they control your ratings — no library can fix that.
Treat the training CSV with the same integrity controls as the model file.

### 4. Dependencies

Runtime dependencies are the standard scientific stack (numpy, pandas,
scikit-learn, nltk, xgboost). Dependabot watches them (`.github/dependabot.yml`)
and CI runs against multiple Python versions, so a vulnerable pin doesn't
linger silently. NLTK resources are downloaded from the official NLTK index at
first use; if your deployment forbids that, pre-provision `~/nltk_data` and
nothing will be fetched.

## Out of scope

- Denial of service via absurdly large inputs to a process you chose to expose
  publicly (put a length limit at your API layer).
- Vulnerabilities in the frozen `2024EE30913/` coursework directory.
- The confidentiality of your own training data or models at rest.
