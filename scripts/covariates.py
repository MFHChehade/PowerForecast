"""Exogenous covariates for the PowerForecast traces, aligned sample-for-sample with run_fm_sweep.

Two kinds, kept strictly apart so nothing from the forecast window leaks in:
  * past covariates  - telemetry recorded alongside the target (GPU utilisation, queue length, ...).
    Only their values up to the forecast origin are ever used; nobody knows them for the future.
  * calendar         - hour-of-day / day-of-week (sin, cos). Deterministic, so the values for the
    forecast window are legitimately known in advance ("future-known").

Traces without either (LNN, NLR: short 1-s runs with only the power column) have no covariate arm.
Target arrays are rebuilt by the same code path as run_fm_sweep.load_trace and asserted identical, so
every covariate model is scored on the identical test windows (same test_id).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_fm_sweep as sw  # noqa: E402
from powerforecast.data import load_dataset  # noqa: E402

LOG = {"memory_used_MiB", "nvidia_gpu_memory_used_bytes_sum", "job_arrivals", "job_completions",
       "mean_queue_wait_s", "mean_runtime_s", "nodes_down"}

# Past covariates per trace. Columns that are a rescaled copy of the target (Polaris/Aurora
# nodes_used_active, cores_used_active, node_util_frac == gpu_node_active / 4, ...) are left out:
# their past is already the target's past, so they add nothing but collinearity.
PAST = {
    "mit": ["utilization_gpu_pct", "utilization_memory_pct", "memory_used_MiB", "temperature_gpu"],
    "surf": ["cpu_power_watts", "gpu_power_watts", "nvidia_gpu_duty_cycle_mean",
             "nvidia_gpu_memory_used_bytes_sum", "node_load1", "node_hwmon_temp_celsius_mean", "active_jobs"],
    "polaris": ["nodes_requested_active", "running_jobs", "job_arrivals", "job_completions",
                "mean_queue_wait_s", "mean_runtime_s", "nodes_down", "is_all_machine_down"],
    "aurora": ["nodes_requested_active", "running_jobs", "job_arrivals", "job_completions",
               "mean_queue_wait_s", "mean_runtime_s", "nodes_down", "is_all_machine_down"],
}
# calendar only where timestamps are wall-clock and the trace spans days (MIT jobs are minutes long)
CALENDAR = {"surf", "google", "bitbrains", "polaris", "aurora"}
COV_TRACES = [t for t in sw.TRACES if t in PAST or t in CALENDAR]
CAL_NAMES = ["hod_sin", "hod_cos", "dow_sin", "dow_cos"]


def calendar(ts):
    ts = pd.DatetimeIndex(ts)
    h = (ts.hour + ts.minute / 60.0) / 24.0
    d = (ts.dayofweek + h) / 7.0
    return np.stack([np.sin(2 * np.pi * h), np.cos(2 * np.pi * h),
                     np.sin(2 * np.pi * d), np.cos(2 * np.pi * d)], 1).astype(np.float32)


def _segments(df, tgt, cols, gap_s):
    """Same segmentation as run_fm_sweep._series_from_df, carrying covariates and timestamps along."""
    key = "series" if "series" in df.columns else ("series_id" if "series_id" in df.columns else None)
    groups = [g for _, g in df.groupby(key)] if key else [df]
    out = []
    for g in groups:
        order = "t_rel_s" if "t_rel_s" in g.columns else "timestamp"
        g = g.sort_values(order)
        if gap_s is not None:
            f = g.set_index(pd.to_datetime(g["timestamp"]))[[tgt] + cols].astype(float)
            f = f[~f.index.duplicated()].resample(f"{gap_s}s").mean()
            missing = f[tgt].isna()
            run_id = (missing != missing.shift()).cumsum()
            run_len = missing.groupby(run_id).transform("sum")
            long_gap = missing & (run_len > sw.MAX_FILL)
            f[tgt] = f[tgt].interpolate(limit=sw.MAX_FILL, limit_area="inside")
            f[cols] = f[cols].interpolate(limit=sw.MAX_FILL, limit_area="inside")
            seg_id = long_gap.cumsum()
            for _, seg in f[~long_gap].groupby(seg_id[~long_gap]):
                out.append((seg[tgt].to_numpy(float), seg[cols].to_numpy(float), seg.index))
        else:
            ts = pd.to_datetime(g["timestamp"]) if np.issubdtype(g["timestamp"].dtype, np.datetime64) else None
            out.append((g[tgt].to_numpy(float), g[cols].to_numpy(float), ts))
    res = []
    for y, c, ts in out:
        ok = np.isfinite(y)
        res.append((y[ok], c[ok], None if ts is None else pd.DatetimeIndex(ts)[ok]))
    return res


def _raw(loader):
    """load_dataset() for the trace, but keeping the telemetry columns the windows loader drops.
    Row filtering mirrors powerforecast.data.windows.prepare_mit / prepare_surf exactly."""
    from powerforecast.data import windows as W
    if loader == "mit_supercloud":
        df = pd.read_parquet(W.ROOT / "mit_power_sample.parquet").sort_values(["series", "timestamp"])
        tgt = "power_draw_W"
        return df.dropna(subset=["series", "timestamp", "t_rel_s", tgt]), tgt
    if loader == "surf_lisa":
        df = pd.read_parquet(W.ROOT / "power_sample.parquet")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[df["timestamp"] < df["timestamp"].min() + pd.Timedelta(days=30)].copy()
        df = df.rename(columns={"node": "series"}).sort_values(["series", "timestamp"])
        tgt = "total_observed_power_watts"
        return df.dropna(subset=["series", "timestamp", tgt]), tgt
    df, tgt, _ = load_dataset(loader)
    return df, tgt


def load_trace_cov(name):
    """Return (series, past, cal): lists aligned 1:1 with sw.load_trace(name)."""
    spec = sw.TRACES[name]
    cols = PAST.get(name, [])
    segs = []
    for ld in spec["loaders"]:
        df, tgt = _raw(ld)
        segs += _segments(df, tgt, cols, spec.get("gap_s"))
    L_max = spec["L_max"]
    segs = [s for s in segs if int(0.6 * len(s[0])) >= L_max // 2 and int(0.75 * len(s[0])) >= L_max + 1]
    series = [s[0] for s in segs]
    ref = sw.load_trace(name)
    assert len(ref) == len(series) and all(np.array_equal(a, b) for a, b in zip(ref, series)), name
    past, cal = [], []
    for y, c, ts in segs:
        c = c.copy()
        for j, col in enumerate(cols):
            if col in LOG:
                c[:, j] = np.log1p(np.maximum(c[:, j], 0))
        c = pd.DataFrame(c).ffill().bfill().fillna(0.0).to_numpy(np.float32)
        past.append(c)
        cal.append(calendar(ts) if (name in CALENDAR and ts is not None) else np.zeros((len(y), 0), np.float32))
    # z-score each past covariate with statistics of the TRAIN regions only
    if cols:
        tr = np.vstack([c[: int(0.6 * len(c))] for c in past])
        mu, sd = tr.mean(0), tr.std(0)
        sd[sd < 1e-8] = 1.0
        past = [((c - mu) / sd).astype(np.float32) for c in past]
    return series, past, cal


class CovCell(sw.Cell):
    """sw.Cell plus covariate windows cut at exactly the same positions.

    P*: past covariates over the context, [N, L, k]   (normalised; raw copies in P*_raw are not kept)
    F*: calendar over context + horizon,  [N, L+H, 4] (future part is known in advance)
    """

    def __init__(self, trace, series, past, cal, H, L):
        super().__init__(trace, series, H, L)
        self.n_past, self.n_cal = past[0].shape[1], cal[0].shape[1]
        for k, pos in self.pos.items():
            P = np.stack([past[s][t - L:t] for s, t in pos]).astype(np.float32)
            F = np.stack([cal[s][t - L:t + H] for s, t in pos]).astype(np.float32)
            setattr(self, "P" + k, P)
            setattr(self, "F" + k, F)
        self.cov_id = hashlib.md5(repr((trace, PAST.get(trace), trace in CALENDAR)).encode()).hexdigest()[:8]


_CACHE = {}


def cov_cell(trace, H, L):
    if trace not in _CACHE:
        _CACHE.clear()  # one trace in memory at a time
        _CACHE[trace] = (load_trace_cov(trace), {})
    (series, past, cal), cells = _CACHE[trace]
    if (H, L) not in cells:
        if len(cells) > 6:
            cells.clear()
        cells[(H, L)] = CovCell(trace, series, past, cal, H, L)
    return cells[(H, L)]


if __name__ == "__main__":
    for t in COV_TRACES:
        c = cov_cell(t, 24, 128)
        print(f"{t:10s} series={len(_CACHE[t][0][0]):4d} past={c.n_past} cal={c.n_cal} "
              f"Ptr={c.Ptr.shape} Fte={c.Fte.shape} test_id={c.test_id}")
