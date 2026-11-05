"""Chronological 60/15/25 split into fixed context/target windows (ALCF zoo protocol)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from powerforecast import config

FEATURES_DIR = config.OUTPUTS_ROOT / "outputs_alcf_power_forecast" / "features"


def load_series(machine):
    h = pd.read_parquet(FEATURES_DIR / f"hourly_{machine}.parquet")
    return h["gpu_node_active"].to_numpy(float)


def make_split(y, context, horizon):
    """Return train/val/test windowed tensors plus the naive-1 scale.

    Test windows are the fixed evaluation set: every t in the final 25% gets a
    context window y[t-context:t] and target y[t:t+horizon]."""
    n = len(y)
    tr_end, va_end = int(n * 0.60), int(n * 0.75)
    scale = np.mean(np.abs(np.diff(y[context:tr_end]))) + 1e-9

    def windows(lo, hi):
        ts = [t for t in range(max(lo, context), hi - horizon + 1)]
        if not ts:
            return np.empty((0, context), np.float32), np.empty((0, horizon), np.float32), np.array([], int)
        X = np.stack([y[t - context:t] for t in ts]).astype(np.float32)
        Y = np.stack([y[t:t + horizon] for t in ts]).astype(np.float32)
        return X, Y, np.array(ts, int)

    Xtr, Ytr, _ = windows(0, tr_end)
    Xva, Yva, _ = windows(tr_end, va_end)
    Xte, Yte, tte = windows(va_end, n)
    return dict(Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva, Xte=Xte, Yte=Yte, tte=tte, scale=scale)
