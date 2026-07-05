from userating.preprocess import make_stopword_removal, negate, transform


def test_negate_flips_until_boundary():
    tagged = [('this', 'DT'), ('is', 'VBZ'), ('not', 'RB'), ('good', 'JJ'),
              ('quality', 'NN'), ('.', '.'), ('nice', 'JJ')]
    out = negate(tagged)

    polarity = {w: p for p, w, t in out}
    assert polarity['good'] == 'NEG'
    assert polarity['quality'] == 'NEG'
    # negation stops at the punctuation boundary
    assert polarity['nice'] == 'POS'
    assert polarity['this'] == 'POS'
    # the negator itself and the boundary token stay POS
    assert polarity['not'] == 'POS'
    assert polarity['.'] == 'POS'


def test_negate_handles_nt():
    tagged = [('does', 'VBZ'), ("n't", 'RB'), ('fit', 'VB')]
    out = negate(tagged)
    assert out[-1][0] == 'NEG'


def test_stopword_removal_keeps_content_words():
    remove = make_stopword_removal()
    pos_arr = [('POS', 'the', 'DT'), ('POS', 'dress', 'NN'), ('POS', 'is', 'VBZ'),
               ('POS', 'gorgeous', 'JJ')]
    kept = [w for p, w, t in remove(pos_arr)]
    assert 'dress' in kept and 'gorgeous' in kept
    assert 'the' not in kept and 'is' not in kept


def test_transform_end_to_end():
    out = transform(["The dresses were not fitting well."])
    assert len(out) == 1
    # every token is a (polarity, word, tag) triple
    assert all(len(t) == 3 and t[0] in ('POS', 'NEG') for t in out[0])
    words = [t[1] for t in out[0]]
    # lowercased and lemmatized
    assert 'dress' in words or 'dresses' in words
    assert all(w == w.lower() for w in words)


def test_transform_parallel_matches_serial():
    # parallelism must never change results: same reviews, same triples, same order
    reviews = [
        "This dress is absolutely lovely, not too long and fits great.",
        "Terrible quality. The fabric ripped on day one!",
        "It was okay, nothing special but not bad either.",
    ] * 30
    serial = transform(reviews, n_jobs=1)
    parallel = transform(reviews, n_jobs=2)
    assert serial == parallel
