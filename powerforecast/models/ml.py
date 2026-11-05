"""Lag-feature machine-learning forecasters (Ridge + gradient-boosted trees).

Prediction logic is verbatim from the historical ``alcf_model_zoo.py`` (SEED=42).
"""
from __future__ import annotations

import numpy as np

from .base import Forecaster

SEED = 42


class MLForecaster(Forecaster):
    """RidgeLag, HistGBM, LightGBM, XGBoost — fit on train windows, per-horizon heads."""

    family = "ml"

    def predict(self, split, horizon):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        Xtr, Ytr, Xte = split["Xtr"], split["Ytr"], split["Xte"]
        name = self.name
        if name == "RidgeLag":
            mdl = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            mdl.fit(Xtr, Ytr)
            return mdl.predict(Xte).reshape(len(Xte), horizon)
        # gradient-boosted per-horizon
        preds = np.empty((len(Xte), horizon), np.float32)
        if name == "HistGBM":
            from sklearn.ensemble import HistGradientBoostingRegressor as M
            mk = lambda: M(random_state=SEED, max_iter=200)
        elif name == "LightGBM":
            import lightgbm as lgb
            mk = lambda: lgb.LGBMRegressor(n_estimators=200, random_state=SEED, verbose=-1)
        elif name == "XGBoost":
            import xgboost as xgb
            mk = lambda: xgb.XGBRegressor(n_estimators=200, random_state=SEED, verbosity=0, tree_method="hist")
        else:
            raise ValueError(name)
        for hstep in range(horizon):
            m = mk(); m.fit(Xtr, Ytr[:, hstep]); preds[:, hstep] = m.predict(Xte)
        return preds
