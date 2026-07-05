import numpy as np

from userating.model import SentimentClassifier, load, save


def test_predictions_are_valid_ratings(trained_model, corpus):
    texts, labels = corpus
    preds = trained_model.predict(texts)
    assert set(np.unique(preds)) <= {1, 2, 3, 4, 5}
    assert len(preds) == len(texts)


def test_model_learns_the_training_set(trained_model, corpus):
    texts, labels = corpus
    preds = trained_model.predict(texts)
    acc = (preds == np.array(labels)).mean()
    assert acc > 0.6, f"train accuracy {acc} barely above chance -- the stack is broken"


def test_predict_proba_is_a_distribution(trained_model, corpus):
    texts, _ = corpus
    probs = trained_model.predict_proba(texts[:10])
    assert probs.shape == (10, 5)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    # predict is exactly argmax of predict_proba
    assert (probs.argmax(axis=1) + 1 == trained_model.predict(texts[:10])).all()


def test_save_load_roundtrip(trained_model, corpus, tmp_path):
    texts, _ = corpus
    path = tmp_path / 'model.pkl'
    save(trained_model, path)
    loaded = load(path)
    assert (loaded.predict(texts[:10]) == trained_model.predict(texts[:10])).all()


def test_deterministic_across_fits(corpus):
    texts, labels = corpus
    a = SentimentClassifier(dim=4, shortlist=50).fit(texts, labels).predict(texts)
    b = SentimentClassifier(dim=4, shortlist=50).fit(texts, labels).predict(texts)
    assert (a == b).all()
