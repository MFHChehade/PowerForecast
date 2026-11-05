"""ALCF workload -> power/GPU-usage proxy forecasting pipeline (Tasks 2-9).

Inputs (streamed): DJC-POLARIS/AURORA (per-job records) + MACHINESTATUS (downtime).
No real power/GPU telemetry exists in the downloads, so the forecast target is a
workload-derived GPU/power proxy. Telemetry ingestion is pluggable (see detect_telemetry).

Outputs in outputs_alcf_power_forecast/: hourly feature parquet per machine, a metrics
summary CSV, actual-vs-forecast plots, and a markdown report.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
for _p in (str(_REPO), str(_REPO / "scripts")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from powerforecast import config

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
warnings.filterwarnings("ignore")

DL = config.ALCF_RAW_ROOT
OUT = config.OUTPUTS_ROOT / "outputs_alcf_power_forecast"
FEAT, MET, FIG = OUT / "features", OUT / "metrics", OUT / "figures"
SEED = 3

# machine -> (DJC file, MACHINESTATUS file, GPUs per node, total nodes)
MACHINES = {
    "polaris": ("ANL-ALCF-DJC-POLARIS_20250101_20251231.csv.gz",
                "ANL-ALCF-MACHINESTATUS-POLARIS_20250101_20251231.csv.gz", 4, 560),
    "aurora": ("ANL-ALCF-DJC-AURORA_20250127_20251231.csv.gz",
               "ANL-ALCF-MACHINESTATUS-AURORA_20250127_20251231.csv.gz", 6, 10624),
}
USECOLS = ["MACHINE_NAME", "QUEUED_TIMESTAMP", "START_TIMESTAMP", "END_TIMESTAMP",
           "NODES_USED", "NODES_REQUESTED", "CORES_USED", "USED_CORE_HOURS",
           "RUNTIME_SECONDS", "QUEUED_WAIT_SECONDS", "GPUS_REQUESTED", "QUEUE_NAME", "EXIT_STATUS"]
LAGS = [1, 2, 3, 6, 12, 24, 48, 168]


# --------------------------------------------------------------- telemetry hook
TELEMETRY_HINTS = ["POWER", "WATT", "GPU_UTIL", "GPU_METRIC", "ENERGY", "JOULE", "UTILIZATION"]


def detect_telemetry(columns):
    """Return real power/GPU-utilization columns if a future telemetry file provides them."""
    return [c for c in columns if any(h in c.upper() for h in TELEMETRY_HINTS)]


# --------------------------------------------------------------- hourly builder
def build_hourly(machine):
    djc, status, gpn, total_nodes = MACHINES[machine]
    secs = lambda s: pd.to_datetime(s, errors="coerce").astype("int64").to_numpy() / 1e9
    parts = []
    for ch in pd.read_csv(DL / djc, compression="gzip", usecols=USECOLS, chunksize=200_000, low_memory=False):
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    print(f"  [{machine}] {len(df):,} job records", flush=True)
    start = secs(df["START_TIMESTAMP"]); end = secs(df["END_TIMESTAMP"]); queued = secs(df["QUEUED_TIMESTAMP"])
    nodes = pd.to_numeric(df["NODES_USED"], errors="coerce").fillna(0).to_numpy()
    nodes_req = pd.to_numeric(df["NODES_REQUESTED"], errors="coerce").fillna(0).to_numpy()
    cores = pd.to_numeric(df["CORES_USED"], errors="coerce").fillna(0).to_numpy()
    wait = pd.to_numeric(df["QUEUED_WAIT_SECONDS"], errors="coerce").to_numpy()
    runtime = pd.to_numeric(df["RUNTIME_SECONDS"], errors="coerce").to_numpy()
    ok = np.isfinite(start) & np.isfinite(end) & (end > start)
    start, end, queued, nodes, nodes_req, cores, wait, runtime = (a[ok] for a in
        (start, end, queued, nodes, nodes_req, cores, wait, runtime))

    t0 = np.floor(np.nanmin(start) / 3600) * 3600
    t1 = np.ceil(np.nanmax(end) / 3600) * 3600
    H = int((t1 - t0) / 3600) + 1
    grid = t0 + np.arange(H) * 3600

    def occ(weight):  # instantaneous occupancy at each hour boundary (delta + cumsum)
        d = np.zeros(H + 1)
        si = np.clip(np.ceil((start - t0) / 3600).astype(int), 0, H)
        ei = np.clip(np.ceil((end - t0) / 3600).astype(int), 0, H)
        np.add.at(d, si, weight); np.add.at(d, ei, -weight)
        return np.cumsum(d)[:H]

    def per_hour_count(ts):
        idx = np.clip(np.floor((ts - t0) / 3600).astype(int), 0, H - 1)
        c = np.zeros(H); np.add.at(c, idx, 1); return c

    def per_hour_mean(ts, val):
        idx = np.clip(np.floor((ts - t0) / 3600).astype(int), 0, H - 1)
        s = np.zeros(H); c = np.zeros(H)
        m = np.isfinite(val)
        np.add.at(s, idx[m], val[m]); np.add.at(c, idx[m], 1)
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)

    hourly = pd.DataFrame({
        "timestamp": pd.to_datetime(grid, unit="s"),
        "machine": machine,
        "nodes_used_active": occ(nodes),
        "nodes_requested_active": occ(nodes_req),
        "cores_used_active": occ(cores),
        "running_jobs": occ(np.ones_like(nodes)),
        "job_arrivals": per_hour_count(queued),
        "job_completions": per_hour_count(end),
        "mean_queue_wait_s": per_hour_mean(end, wait),
        "mean_runtime_s": per_hour_mean(end, runtime),
    })
    hourly["gpu_node_active"] = hourly["nodes_used_active"] * gpn          # primary GPU/power proxy
    hourly["node_util_frac"] = (hourly["nodes_used_active"] / total_nodes).clip(0, 1)
    hourly["mean_queue_wait_s"] = hourly["mean_queue_wait_s"].fillna(0)
    hourly["mean_runtime_s"] = hourly["mean_runtime_s"].fillna(0)

    hourly = add_downtime(hourly, status, machine)
    hourly["gpus_per_node"] = gpn
    return hourly


def add_downtime(hourly, status_file, machine):
    s = pd.read_csv(DL / status_file, compression="gzip", low_memory=False)
    st = pd.to_datetime(s["START_TIMESTAMP"], errors="coerce")
    en = pd.to_datetime(s["END_TIMESTAMP"], errors="coerce")
    ts = hourly["timestamp"]
    is_down = np.zeros(len(hourly), bool); nodes_down = np.zeros(len(hourly))
    unsched = np.zeros(len(hourly), bool)
    for i in range(len(s)):
        if pd.isna(st.iloc[i]) or pd.isna(en.iloc[i]):
            continue
        m = (ts >= st.iloc[i]) & (ts < en.iloc[i])
        is_down |= m.to_numpy()
        nodes_down = np.maximum(nodes_down, m.to_numpy() * float(s["NUMBER_NODES_DOWN"].iloc[i] if pd.notna(s["NUMBER_NODES_DOWN"].iloc[i]) else 0))
        if int(s.get("IS_ALL_MACHINE_DOWN_UNSCHEDULED", pd.Series([0]*len(s))).iloc[i] or 0):
            unsched |= m.to_numpy()
    hourly["is_all_machine_down"] = is_down.astype(int)
    hourly["downtime_unscheduled"] = unsched.astype(int)
    hourly["nodes_down"] = nodes_down
    return hourly


# --------------------------------------------------------------- forecasting
def supervised(y, exog, idx):
    X, names = [], []
    for L in LAGS:
        X.append(np.roll(y, L)); names.append(f"lag{L}")
    X.append(pd.Series(y).shift(1).rolling(6).mean().to_numpy()); names.append("roll6")
    X.append(pd.Series(y).shift(1).rolling(24).mean().to_numpy()); names.append("roll24")
    X.append(idx.hour.to_numpy()); names.append("hour")
    X.append(idx.dayofweek.to_numpy()); names.append("dow")
    for c, v in exog.items():
        X.append(np.roll(v, 1)); names.append(f"{c}_lag1")
    return np.column_stack(X), names


def smape(y, p):
    return float(np.mean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p) + 1e-9)) * 100)


def metrics(y, p, scale):
    e = y - p
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "sMAPE": smape(y, p), "MASE": float(np.mean(np.abs(e)) / scale) if scale > 0 else np.nan}


def forecast_machine(hourly, machine):
    y = hourly["gpu_node_active"].to_numpy(float)
    idx = pd.DatetimeIndex(hourly["timestamp"])
    exog = {c: hourly[c].to_numpy(float) for c in ["running_jobs", "job_arrivals", "nodes_requested_active"]}
    X, names = supervised(y, exog, idx)
    n = len(y)
    tr_end, va_end = int(n * 0.60), int(n * 0.75)
    warm = max(LAGS)
    train = slice(warm, tr_end); val = slice(tr_end, va_end); test = slice(va_end, n)
    scale = np.mean(np.abs(np.diff(y[warm:tr_end]))) + 1e-9  # in-sample naive(1) MAE

    preds = {}
    preds["Persistence"] = np.roll(y, 1)
    preds["SeasonalNaive24"] = np.roll(y, 24)
    preds["MovingAvg6"] = pd.Series(y).shift(1).rolling(6).mean().to_numpy()
    # exp smoothing
    es = np.zeros(n); a = 0.3; es[0] = y[0]
    for t in range(1, n):
        es[t] = a * y[t - 1] + (1 - a) * es[t - 1]
    preds["ExpSmoothing"] = es
    # ML models fit on train+val, predict test
    fit_idx = np.arange(warm, va_end); Xtr, ytr = X[fit_idx], y[fit_idx]
    good = np.all(np.isfinite(Xtr), axis=1)
    for name, mk in [("HistGBM", lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=6, random_state=SEED)),
                     ("LightGBM", _lgbm)]:
        try:
            m = mk(); m.fit(Xtr[good], ytr[good])
            full = np.full(n, np.nan); fin = np.all(np.isfinite(X), axis=1)
            full[fin] = m.predict(X[fin]); preds[name] = full
        except Exception as exc:
            print(f"  [{machine}] {name} skipped: {exc}", flush=True)

    rows, test_curves = [], {}
    yt = y[test]; down = hourly["is_all_machine_down"].to_numpy()[test].astype(bool)
    for name, p in preds.items():
        pt = p[test]
        m = np.isfinite(pt)
        rows.append({"machine": machine, "model": name, "segment": "all", **metrics(yt[m], pt[m], scale), "n": int(m.sum())})
        for seg, mask in [("normal", ~down), ("downtime", down)]:
            mm = m & mask
            if mm.sum() >= 5:
                rows.append({"machine": machine, "model": name, "segment": seg, **metrics(yt[mm], pt[mm], scale), "n": int(mm.sum())})
        test_curves[name] = pt
    return pd.DataFrame(rows), idx[test], yt, test_curves, down


def _lgbm():
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=400, max_depth=7, learning_rate=0.05, random_state=SEED, verbose=-1)


def plot_machine(machine, tidx, yt, curves, down):
    # show last 14 days of the test period for readability
    k = min(len(yt), 24 * 14)
    sl = slice(len(yt) - k, len(yt))
    best = min([m for m in curves if m in ("HistGBM", "LightGBM")],
               key=lambda m: np.nanmean(np.abs(yt - curves[m])), default="Persistence")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(tidx[sl], yt[sl], color="black", lw=1.6, label="actual (GPU-node proxy)")
    ax.plot(tidx[sl], curves["Persistence"][sl], color="#999", lw=1.0, ls="--", label="Persistence")
    ax.plot(tidx[sl], curves[best][sl], color="#d95f02", lw=1.2, label=best)
    dd = down[sl]
    if dd.any():
        ax.fill_between(tidx[sl], 0, yt[sl].max() * 1.05, where=dd, color="red", alpha=0.10, label="downtime")
    ax.set_title(f"{machine.capitalize()}: actual vs forecast — next-hour GPU/power proxy (last 14 test days)")
    ax.set_ylabel("active GPUs (proxy = nodes_used × GPUs/node)"); ax.set_xlabel("time")
    ax.legend(loc="upper right", fontsize=9, ncol=2, frameon=False)
    fig.autofmt_xdate(); fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"forecast_{machine}.png", dpi=200); fig.savefig(FIG / f"forecast_{machine}.pdf")
    plt.close(fig)


def main():
    for d in (FEAT, MET, FIG):
        d.mkdir(parents=True, exist_ok=True)
    all_metrics, summaries = [], []
    for machine in MACHINES:
        print(f"== {machine} ==", flush=True)
        hourly = build_hourly(machine)
        hourly.to_parquet(FEAT / f"hourly_{machine}.parquet", index=False)
        # daily aggregate (next-day core-hours) saved for completeness
        daily = hourly.set_index("timestamp").resample("1D").agg(
            gpu_node_active_mean=("gpu_node_active", "mean"),
            used_core_hours=("cores_used_active", "sum"),
            job_arrivals=("job_arrivals", "sum"), down_hours=("is_all_machine_down", "sum")).reset_index()
        daily.insert(1, "machine", machine)
        daily.to_parquet(FEAT / f"daily_{machine}.parquet", index=False)
        mdf, tidx, yt, curves, down = forecast_machine(hourly, machine)
        all_metrics.append(mdf)
        plot_machine(machine, tidx, yt, curves, down)
        span = f"{hourly['timestamp'].min():%Y-%m-%d} → {hourly['timestamp'].max():%Y-%m-%d}"
        summaries.append({"machine": machine, "hours": len(hourly), "span": span,
                          "downtime_hours": int(hourly["is_all_machine_down"].sum()),
                          "mean_active_gpus": round(hourly["gpu_node_active"].mean(), 1),
                          "peak_active_gpus": round(hourly["gpu_node_active"].max(), 1)})
        print(f"  [{machine}] hourly={len(hourly)} downtime_hrs={int(hourly['is_all_machine_down'].sum())}", flush=True)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_df.to_csv(MET / "alcf_forecast_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(MET / "alcf_series_summary.csv", index=False)
    write_report(metrics_df, pd.DataFrame(summaries))
    # surface key results
    print("\n=== TEST METRICS (all-segment, by machine/model) ===")
    print(metrics_df[metrics_df.segment.eq("all")][["machine", "model", "MAE", "RMSE", "sMAPE", "MASE"]].round(3).to_string(index=False))
    print("\nwrote:", [p.name for p in sorted(OUT.rglob('*')) if p.is_file()][:30])


def write_report(metrics_df, summary):
    allseg = metrics_df[metrics_df.segment.eq("all")]
    best = allseg.loc[allseg.groupby("machine")["MASE"].idxmin()][["machine", "model", "MAE", "RMSE", "sMAPE", "MASE"]]
    lines = [
        "# ALCF Workload → Power/GPU-Usage Proxy Forecasting — Report", "",
        "## What this pipeline does",
        "Builds **hourly utilization time series** from ALCF `DIM_JOB_COMPOSITE` (per-job records) for Polaris and "
        "Aurora (2025), flags **downtime** from `DIM_MACHINE_STATUS`, and forecasts a **next-hour GPU/power demand "
        "proxy** = `nodes_used_active × GPUs_per_node` (Polaris 4, Aurora 6). No real power/GPU telemetry exists in "
        "the current downloads; `detect_telemetry()` will switch to true power/GPU-utilization columns automatically "
        "if a GPU/Power/Node-Metric-History file is added later.", "",
        "## Series built", "```csv", summary.to_csv(index=False).strip(), "```", "",
        "## Best model per machine (lowest MASE on the held-out final period)", "```csv", best.to_csv(index=False).strip(), "```", "",
        "## Full test metrics (chronological split: 60% train / 15% val / 25% test; no leakage)", "```csv",
        allseg[["machine", "model", "MAE", "RMSE", "sMAPE", "MASE", "n"]].round(3).to_csv(index=False).strip(), "```", "",
        "## Normal vs downtime windows", "```csv",
        metrics_df[metrics_df.segment.isin(["normal", "downtime"])][["machine", "model", "segment", "MAE", "MASE", "n"]]
        .round(3).to_csv(index=False).strip(), "```", "",
        "![polaris](figures/forecast_polaris.png)", "![aurora](figures/forecast_aurora.png)", "",
        "## Which datasets were useful",
        "- **DJC-POLARIS / DJC-AURORA** — primary: yield node/core occupancy, arrivals, completions, queue-wait & runtime, GPU proxy.",
        "- **MACHINESTATUS-POLARIS / AURORA** — downtime windows → availability features and normal-vs-downtime evaluation split.",
        "",
        "## What is still missing for TRUE electric-power / GPU-usage forecasting",
        "- **GPU Metric History, Power History, Node/Machine/Rack/Chassis Metric History** (ALCF internal telemetry) — "
        "not in the downloads. These provide measured watts and GPU utilization; the current target is a workload PROXY.",
        "- **Temporal-overlap covariates:** GRIDFTP (2024) and Darshan (Theta 2023 / Mira 2020) do not overlap the 2025 "
        "Polaris/Aurora window, so they are inventoried but not merged. With same-period GRIDFTP/Darshan they could be added.",
        "",
        "## Caveats",
        "- The target is a utilization-derived proxy, not measured power; absolute watts require the telemetry files.",
        "- Occupancy uses instantaneous hour-boundary counts (delta+cumsum over jobs); a good hourly utilization proxy.",
        "- Downtime hours are reported separately because the proxy is near-zero there (trivially predictable).",
    ]
    OUT.joinpath("alcf_forecast_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
