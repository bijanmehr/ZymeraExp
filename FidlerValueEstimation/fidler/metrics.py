"""Scoring for Fiedler-value predictions (numpy, eval-side)."""
import numpy as np

EPS = 1e-6


def accuracy(pred, true):
    """1 - median relative error (clamped to [0,1])."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    rel = np.abs(pred - true) / np.maximum(np.abs(true), EPS)
    return float(np.clip(1.0 - np.median(rel), 0.0, 1.0))


def within_pct(pred, true, pct=0.05):
    """Fraction of predictions within `pct` relative error."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    rel = np.abs(pred - true) / np.maximum(np.abs(true), EPS)
    return float(np.mean(rel <= pct))


def r2(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2) + EPS
    return float(1.0 - ss_res / ss_tot)


def connected_accuracy(pred_flag, true_flag):
    return float(np.mean(np.asarray(pred_flag, bool) == np.asarray(true_flag, bool)))
