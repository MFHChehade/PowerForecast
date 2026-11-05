"""Classical / statistical forecasters.

Simple baselines (persistence, exponential smoothing) are plain functions so
pilot scripts can call them directly; :class:`ClassicalForecaster` wraps all
seven classical zoo models behind the :class:`~powerforecast.models.base.Forecaster`
interface. Prediction logic is verbatim from the historical ``alcf_model_zoo.py``.
"""
from __future__ import annotations

import numpy as np

from .base import Forecaster


def persistence(x, horizon):
    return np.repeat(x[:, -1:], horizon, axis=1)


def exp_smoothing(x, horizon, alpha=0.3):
    level = x[:, 0].copy()
    for i in range(1, x.shape[1]):
        level = alpha * x[:, i] + (1 - alpha) * level
    return np.repeat(level[:, None], horizon, axis=1)


class ClassicalForecaster(Forecaster):
    """Persistence, SeasonalNaive24, MovingAvg6, ExpSmoothing, AutoETS, Theta, AutoARIMA."""

    family = "classical"

    def predict(self, split, horizon):
        Xte = split["Xte"]
        name = self.name
        if name == "Persistence":
            return np.repeat(Xte[:, -1:], horizon, axis=1)
        if name == "SeasonalNaive24":
            # repeat the value 24h ago, tiled across the horizon
            base = Xte[:, -24] if Xte.shape[1] >= 24 else Xte[:, -1]
            return np.repeat(base[:, None], horizon, axis=1)
        if name == "MovingAvg6":
            base = Xte[:, -6:].mean(axis=1) if Xte.shape[1] >= 6 else Xte.mean(axis=1)
            return np.repeat(base[:, None], horizon, axis=1)
        if name == "ExpSmoothing":
            return exp_smoothing(Xte, horizon)
        # statsforecast per-window models
        from statsforecast.models import AutoETS, AutoARIMA, AutoTheta
        if name == "AutoETS":
            model = AutoETS(season_length=24)
        elif name == "Theta":
            model = AutoTheta(season_length=24)
        elif name == "AutoARIMA":
            # non-seasonal: per-window seasonal stepwise search is ~165 min/machine;
            # daily structure is already represented by AutoETS/Theta/SeasonalNaive24.
            model = AutoARIMA(season_length=1, max_p=3, max_q=3, stepwise=True)
        else:
            raise ValueError(name)
        # ARIMA is slow; cap fit history to the last 512 points per window.
        cap = 512 if name == "AutoARIMA" else Xte.shape[1]
        preds = np.empty((len(Xte), horizon), np.float32)
        for i, row in enumerate(Xte):
            hist = row[-cap:].astype(np.float64)
            try:
                f = model.forecast(y=hist, h=horizon)["mean"]
            except Exception:
                f = np.repeat(row[-1], horizon)
            preds[i] = f
        return preds
