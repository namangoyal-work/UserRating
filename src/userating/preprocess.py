"""Text preprocessing pipeline: tokenize -> PoS tag -> negate -> stopword removal -> lemmatize.

Every function here is stateless: a review goes in as a string and comes out as a list of
(polarity, word, tag) triples. All model state lives in model.py.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# resources word_tokenize / pos_tag / stopwords / WordNetLemmatizer actually need,
# instead of nltk.download('popular') which pulls ~10x more
NLTK_RESOURCES = [
    ('tokenizers/punkt_tab', 'punkt_tab'),
    ('tokenizers/punkt', 'punkt'),
    ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
    ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
    ('corpora/stopwords', 'stopwords'),
    ('corpora/wordnet', 'wordnet'),
    ('corpora/omw-1.4', 'omw-1.4'),
]


def ensure_nltk_data():
    for path, name in NLTK_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass  # older/newer nltk splits resources differently; missing ones fail loudly at use


def dummy(doc):
    return doc


def create_feature(l, f):
    return [f(e) for e in l]


def negate(pos_arr):
    # negation via looking at PoS: negate all words from a not/n't to the next
    # clause boundary (punctuation, preposition, conjunction)
    neg_pos_arr = []
    negating = False
    for word in pos_arr:
        w = word[0]
        if (w == 'not' or w == "n't"):
            negating = True
            neg_pos_arr.append(("POS", w, word[1]))
            continue
        elif (word[1] == '.' or word[1] == ':' or word[1] == 'IN' or word[1] == 'CC'):
            negating = False
            neg_pos_arr.append(("POS", w, word[1]))
            continue

        if negating:
            neg_pos_arr.append(("NEG", w, word[1]))
        else:
            neg_pos_arr.append(("POS", w, word[1]))

    return neg_pos_arr


def make_stopword_removal():
    sw = set(stopwords.words('english'))

    def stopword_removal(pos_arr):
        return [word for word in pos_arr if word[1] not in sw]

    return stopword_removal


def make_lemmatize():
    wnl = WordNetLemmatizer()

    # PoS tag conversion: Penn Treebank first letter -> wordnet PoS
    pos_tag_map = {
        'J': 'a',  # adjective
        'N': 'n',  # noun
        'V': 'v',  # verb
        'R': 'r',  # adverb
    }

    def lemmatize(pos_arr):
        lemmatized_list = []
        for word in pos_arr:
            if word[2][0] in pos_tag_map:
                lemmatized_list.append((word[0], wnl.lemmatize(word[1], pos=pos_tag_map[word[2][0]]), word[2]))
            else:
                lemmatized_list.append(word)
        return lemmatized_list

    return lemmatize


def _pipeline(X):
    # tokenize
    X = create_feature(X, word_tokenize)

    # lowercase
    X = create_feature(X, lambda x: [w.lower() for w in x])

    # PoS tagging
    X = create_feature(X, nltk.pos_tag)

    # negation
    X = create_feature(X, negate)

    # stopword and punctuation removal
    X = create_feature(X, make_stopword_removal())

    # lemmatization
    X = create_feature(X, make_lemmatize())

    return X


# below this many reviews, process-pool spawn (worker startup + tagger reload)
# costs more than it saves -- measured break-even on an M4 is ~15-20k reviews
PARALLEL_THRESHOLD = 16384


def transform(X, n_jobs=None):
    """Full preprocessing pipeline over a list of raw review strings.

    Stateless per review, so large inputs are fanned out across cpu cores;
    results are identical to the serial path. n_jobs=1 forces serial,
    n_jobs>1 forces that many workers, None decides by input size.
    """
    ensure_nltk_data()

    if n_jobs is None:
        n_jobs = os.cpu_count() if len(X) >= PARALLEL_THRESHOLD else 1
    if n_jobs == 1 or len(X) < 2:
        return _pipeline(X)

    # chunks small enough to load-balance, big enough to amortize pickling
    n_chunks = min(n_jobs * 4, len(X))
    step = (len(X) + n_chunks - 1) // n_chunks
    chunks = [X[i:i + step] for i in range(0, len(X), step)]

    with ProcessPoolExecutor(max_workers=n_jobs, initializer=ensure_nltk_data) as pool:
        out = []
        for part in pool.map(_pipeline, chunks):
            out.extend(part)
    return out
