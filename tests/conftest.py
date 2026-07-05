import pytest

from userating.model import SentimentClassifier

SUBJECTS = ['dress', 'shirt', 'jacket', 'skirt', 'sweater']

TEMPLATES = {
    5: ["absolutely love this {s} , stunning quality and gorgeous fit",
        "this {s} is perfect , amazing fabric and beautiful color"],
    4: ["really nice {s} , good quality and comfortable fit",
        "solid {s} overall , pretty fabric and pleasant color"],
    3: ["the {s} is okay , average quality and decent fit",
        "this {s} is fine , acceptable fabric and plain color"],
    2: ["disappointing {s} , poor quality and awkward fit",
        "not a good {s} , cheap fabric and dull color"],
    1: ["terrible {s} , awful quality and horrible fit",
        "did not like this {s} , dreadful fabric and ugly color"],
}


@pytest.fixture(scope='session')
def corpus():
    texts, labels = [], []
    for rating, templates in TEMPLATES.items():
        for template in templates:
            for s in SUBJECTS:
                texts.append(template.format(s=s))
                labels.append(rating)
    return texts, labels


@pytest.fixture(scope='session')
def trained_model(corpus):
    texts, labels = corpus
    model = SentimentClassifier(dim=4, shortlist=50)
    model.fit(texts, labels)
    return model
