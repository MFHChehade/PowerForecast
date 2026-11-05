"""Raw-trace loading and sliding-window construction for the MIT/SURF benchmarks.

Extracted verbatim from the historical ``foundation_benchmark.py``; the SEED=3
test-sampling RNG is preserved so window selections are reproducible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from powerforecast import config

SEED = 3
ROOT = config.DATA_ROOT


def prepare_surf(days=30, cadence_steps=1):
    target = "total_observed_power_watts"
    df = pd.read_parquet(ROOT / "power_sample.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    start = df["timestamp"].min()
    df = df[df["timestamp"] < start + pd.Timedelta(days=days)].copy()
    df = df.rename(columns={"node": "series"}).sort_values(["series", "timestamp"])
    if cadence_steps > 1:
        df["_ix"] = df.groupby("series").cumcount()
        df = df[df["_ix"].mod(cadence_steps).eq(0)].drop(columns=["_ix"])
    return df[["series", "timestamp", target]].dropna().sort_values(["series", "timestamp"]), target, 300 * cadence_steps


def prepare_mit(cadence_steps=1):
    target = "power_draw_W"
    df = pd.read_parquet(ROOT / "mit_power_sample.parquet")
    df = df.sort_values(["series", "timestamp"]).copy()
    if cadence_steps > 1:
        df["_ix"] = df.groupby("series").cumcount()
        df = df[df["_ix"].mod(cadence_steps).eq(0)].drop(columns=["_ix"])
    return df[["series", "timestamp", "t_rel_s", target]].dropna().sort_values(["series", "timestamp"]), target, None


def make_windows(df, target, context, horizon, expected_seconds=None):
    xs, ys, meta = [], [], []
    for series, g in df.groupby("series"):
        g = g.sort_values("timestamp")
        if expected_seconds is None:
            breaks = pd.Series(np.zeros(len(g), dtype=int), index=g.index)
        else:
            breaks = g["timestamp"].diff().dt.total_seconds().fillna(expected_seconds).ne(expected_seconds).cumsum()
        for _, run in g.groupby(breaks):
            values = run[target].to_numpy(np.float32)
            times = run["timestamp"].to_numpy()
            n = len(values)
            for i in range(max(0, n - context - horizon + 1)):
                xs.append(values[i : i + context])
                ys.append(values[i + context : i + context + horizon])
                meta.append({"series": series, "forecast_start": pd.Timestamp(times[i + context]).isoformat()})
    return np.stack(xs), np.stack(ys), pd.DataFrame(meta)


def sample_test(x, y, meta, max_test=240):
    test = []
    for _, idxs in meta.groupby("series").groups.items():
        idxs = np.array(list(idxs))
        idxs = idxs[np.argsort(meta.loc[idxs, "forecast_start"].to_numpy())]
        test.extend(idxs[int(len(idxs) * 0.85) :])
    rng = np.random.default_rng(SEED)
    test = rng.choice(test, min(max_test, len(test)), replace=False)
    return x[test], y[test], meta.iloc[test].reset_index(drop=True)
