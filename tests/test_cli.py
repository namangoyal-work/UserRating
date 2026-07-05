import numpy as np
import pandas as pd

from userating.cli import main


def test_train_test_predict_workflow(corpus, tmp_path, capsys):
    texts, labels = corpus
    train_csv = tmp_path / 'train.csv'
    pd.DataFrame({0: texts, 1: labels}).to_csv(train_csv, header=False, index=False)

    model_path = tmp_path / 'model.pkl'
    main(['train', str(train_csv), str(model_path), '--limit', '50'])
    assert model_path.exists()

    # test csv with a blank review -- must still get one prediction per row
    input_csv = tmp_path / 'input.csv'
    pd.DataFrame({0: [texts[0], None, texts[-1]]}).to_csv(input_csv, header=False, index=False)
    output_path = tmp_path / 'preds.txt'
    main(['test', str(model_path), str(input_csv), str(output_path)])

    preds = np.loadtxt(output_path)
    assert preds.shape == (3,)
    assert all(1 <= p <= 5 for p in preds)

    main(['predict', str(model_path), 'absolutely gorgeous dress, stunning quality'])
    out = capsys.readouterr().out
    assert 'confidence' in out
