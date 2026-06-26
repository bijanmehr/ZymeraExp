import numpy as np
from fidler import metrics


def test_accuracy_perfect_is_one():
    y = np.array([0.5, 1.0, 2.0])
    assert metrics.accuracy(y, y) == 1.0

def test_accuracy_is_one_minus_median_rel_error():
    true = np.array([1.0, 1.0, 1.0])
    pred = np.array([1.1, 0.9, 1.2])          # rel errs 0.1, 0.1, 0.2 -> median 0.1
    assert abs(metrics.accuracy(pred, true) - 0.9) < 1e-9

def test_within_pct_fraction():
    true = np.array([1.0, 1.0, 1.0, 1.0])
    pred = np.array([1.01, 1.04, 1.10, 0.96])  # within 5%: T,T,F,T -> 0.75
    assert metrics.within_pct(pred, true, pct=0.05) == 0.75

def test_r2_perfect():
    y = np.array([0.1, 0.5, 0.9])
    assert abs(metrics.r2(y, y) - 1.0) < 1e-9

def test_connected_accuracy():
    pred_flag = np.array([True, True, False, True])
    true_flag = np.array([True, False, False, True])
    assert metrics.connected_accuracy(pred_flag, true_flag) == 0.75
