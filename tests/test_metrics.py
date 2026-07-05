import numpy as np

from userating.metrics import eval_metrics, eval_report


def test_eval_metrics_perfect_prediction():
    true = [1, 2, 3, 4, 5]
    f1_micro, f1_macro = eval_metrics(true, true)
    assert f1_micro == 1.0 and f1_macro == 1.0


def test_eval_report_shapes():
    true = [1, 2, 3, 4, 5, 5, 4, 3, 2, 1]
    preds = [1, 2, 3, 4, 5, 4, 4, 3, 1, 1]
    report = eval_report(true, preds)
    assert report['confusion'].shape == (5, 5)
    assert report['per_class_f1'].shape == (5,)
    assert 0 <= report['final'] <= 1
    assert np.isclose(report['final'], (report['f1_micro'] + report['f1_macro']) / 2)
