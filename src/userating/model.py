"""The stacked rating model: TF-IDF/SVD unigrams + bigrams + LL word scores -> XGBoost.

Same architecture as the original submission (2024EE30913/main.py), packaged: the
base-learner log-probabilities are stacked and fitted with an XGBoost tree.
"""

import pickle

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .preprocess import create_feature, dummy, transform

N_CLASSES = 5  # ratings 1..5


class SentimentClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1.0, dim=2000, max_iter=100, shortlist=200, random_state=0):
        self.C = C
        self.dim = dim
        self.max_iter = max_iter
        self.shortlist = shortlist
        self.random_state = random_state

        self.tfidf = TfidfVectorizer(analyzer='word', tokenizer=dummy,
            preprocessor=dummy, token_pattern=None)
        self.tfidf_bigram = TfidfVectorizer(analyzer='word', tokenizer=dummy,
            preprocessor=dummy, token_pattern=None)

        self.svd = TruncatedSVD(n_components=dim, random_state=random_state)
        self.svd_bigram = TruncatedSVD(n_components=dim, random_state=random_state)

        self.lr     = LogisticRegression(solver='lbfgs', C=C, max_iter=max_iter)
        self.lr_bal = LogisticRegression(solver='lbfgs', class_weight='balanced', C=C, max_iter=max_iter)
        self.lr_bg  = LogisticRegression(solver='lbfgs', C=C, max_iter=max_iter)

        self.xgb = XGBClassifier(
            max_depth=6,
            colsample_bytree=0.2,
            colsample_bynode=0.8,
            n_estimators=20,
            objective='multi:softprob',
            learning_rate=0.3,
            random_state=random_state,
        )

    def fit_ll_scores(self, X, y):
        # Zipf law encoding and dropping extraneous words
        tok_set = set()
        for review in X:
            for tok in review:
                tok_set.add(tok)

        toks = list(tok_set)
        rev_toks = {tok: i for i, tok in enumerate(toks)}
        tok_freq = np.zeros((N_CLASSES, len(toks)))

        for i, review in zip(y, X):
            for tok in review:
                tok_freq[i - 1, rev_toks[tok]] += 1

        tok_freq_lists = [sorted([(f, i) for i, f in enumerate(tok_freq[c])])[::-1][:1000] for c in range(N_CLASSES)]
        tok_freqs = set()
        for tfl in tok_freq_lists:
            tok_freqs = tok_freqs.union(set([i for f, i in tfl]))

        tok_freq_w = tok_freq.sum(axis=0)
        tok_freq_c = tok_freq.sum(axis=1)
        tot_tok = tok_freq.sum()

        def P_w(w):
            return (tok_freq_w[w] + 5) / (tot_tok + 5 * len(toks))

        def P_w_c(w, c):
            return (tok_freq[c, w] + 1) / (tok_freq_c[c] + len(toks))

        tok_lls = [([P_w_c(i, c) / P_w(i) for c in range(N_CLASSES)], i) for i in tok_freqs]
        tok_ll_ratios = [(max(l) / min(l), i) for l, i in tok_lls]
        top_ratios = sorted(tok_ll_ratios)[::-1]

        top_ratios_shortlisted = top_ratios[:self.shortlist]

        self.bigram_toks = set(toks[i] for r, i in top_ratios_shortlisted)

        # feature goodness calc
        self.tok_ll_dict = {toks[b]: np.log(np.array(a)) for a, b in tok_lls}

    def predict_ll_scores(self, X):
        ll_scores = []
        for pos_tokens in X:
            tok_lls = np.zeros(N_CLASSES)
            for tok in pos_tokens:
                if tok in self.tok_ll_dict:
                    tok_lls += self.tok_ll_dict[tok]
            ll_scores.append(tok_lls)

        return np.array(ll_scores)

    def extract_bigrams(self, pos_arr):
        bigrams = []
        n = len(pos_arr)
        tok_arr = [r[1] for r in pos_arr]

        for i in range(n - 1):
            if pos_arr[i] in self.bigram_toks or \
               (i == n - 2 and pos_arr[i + 1] in self.bigram_toks):
                bigrams.append("-".join(tok_arr[i:i + 2]))

        return bigrams

    def fit(self, X, y, pretransformed=False):

        if not pretransformed:
            X = transform(X)

        self.fit_ll_scores(X, y)

        bigrams = create_feature(X, self.extract_bigrams)

        X_str = [[f"{t[0]}_{t[1]}_{t[2]}" for t in x] for x in X]

        train_tfidf = self.tfidf.fit_transform(X_str)
        train_bigram_tfidf = self.tfidf_bigram.fit_transform(bigrams)

        # dim is a cap, not a promise: SVD rank cannot exceed the vocabulary size,
        # so small corpora train instead of crashing
        self.svd.n_components = max(1, min(self.dim, train_tfidf.shape[1] - 1))
        self.svd_bigram.n_components = max(1, min(self.dim, train_bigram_tfidf.shape[1] - 1))

        train_svd = self.svd.fit_transform(train_tfidf)
        train_bigram_svd = self.svd_bigram.fit_transform(train_bigram_tfidf)

        self.train_svd_mean = train_svd.mean(axis=0)
        self.train_svd_std = train_svd.std(axis=0)
        train_svd_white = (train_svd - self.train_svd_mean) / self.train_svd_std

        self.train_bigram_svd_mean = train_bigram_svd.mean(axis=0)
        self.train_bigram_svd_std = train_bigram_svd.std(axis=0)
        train_bigram_svd_white = (train_bigram_svd - self.train_bigram_svd_mean) / self.train_bigram_svd_std

        self.lr.fit(train_svd_white, y)
        self.lr_bal.fit(train_svd_white, y)
        self.lr_bg.fit(train_bigram_svd_white, y)

        train_log_probs = np.hstack([
            self.lr.predict_log_proba(train_svd_white),
            self.lr_bal.predict_log_proba(train_svd_white),
            self.lr_bg.predict_log_proba(train_bigram_svd_white),
            self.predict_ll_scores(X)
        ])

        self.xgb.fit(train_log_probs, np.array(y) - 1)

        return self

    def stack_features(self, X):
        """Preprocessed reviews -> the stacked log-prob matrix the XGB head consumes."""

        bigrams = create_feature(X, self.extract_bigrams)

        X_str = [[f"{t[0]}_{t[1]}_{t[2]}" for t in x] for x in X]

        X_tfidf = self.tfidf.transform(X_str)
        X_svd = self.svd.transform(X_tfidf)

        X_bigram_tfidf = self.tfidf_bigram.transform(bigrams)
        X_bigram_svd = self.svd_bigram.transform(X_bigram_tfidf)

        X_svd_white = (X_svd - self.train_svd_mean) / self.train_svd_std
        X_bigram_svd_white = (X_bigram_svd - self.train_bigram_svd_mean) / self.train_bigram_svd_std

        return np.hstack([
            self.lr.predict_log_proba(X_svd_white),
            self.lr_bal.predict_log_proba(X_svd_white),
            self.lr_bg.predict_log_proba(X_bigram_svd_white),
            self.predict_ll_scores(X)
        ])

    def predict(self, X, pretransformed=False):
        return self.predict_proba(X, pretransformed).argmax(axis=1) + 1

    def predict_proba(self, X, pretransformed=False):
        if not pretransformed:
            X = transform(X)
        return self.xgb.predict_proba(self.stack_features(X))


def save(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load(path):
    # pickle runs arbitrary code on load -- only load models you trained yourself
    # or obtained from a source you trust (see SECURITY.md)
    with open(path, 'rb') as f:
        return pickle.load(f)
