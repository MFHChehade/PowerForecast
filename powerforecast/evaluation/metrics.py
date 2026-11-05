"""Forecast-accuracy metrics.

``score``/``smape`` are the model-zoo metrics (aggregate over all horizon steps);
``bootstrap_metrics`` is the pilot-benchmark variant with a bootstrap MAE CI
(historical ``foundation_benchmark.metrics``, SEED=3 preserved).
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

BOOTSTRAP_SEED = 3


def smape(y, p):
    return float(np.mean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p) + 1e-9)) * 100)


def score(y, p, scale):
    """y,p shape [n, horizon]; returns aggregate metrics over all steps."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    m = np.isfinite(p).all(axis=1)
    y, p = y[m], p[m]
    e = y - p
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "sMAPE": smape(y, p), "MASE": float(np.mean(np.abs(e)) / scale) if scale > 0 else np.nan,
            "n": int(m.sum())}


def bootstrap_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    window_mae = np.mean(np.abs(y_true - y_pred), axis=1)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boots = rng.choice(window_mae, size=(2000, len(window_mae)), replace=True).mean(axis=1)
    return {
        "mae": float(mean_absolute_error(y_true.ravel(), y_pred.ravel())),
        "mae_ci95_low": float(np.percentile(boots, 2.5)),
        "mae_ci95_high": float(np.percentile(boots, 97.5)),
        "n_test_windows": int(len(window_mae)),
        "rmse": float(math.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel()))),
        "mape": float(np.mean(np.abs((y_true.ravel() - y_pred.ravel()) / np.maximum(np.abs(y_true.ravel()), 1e-6))) * 100),
    }
