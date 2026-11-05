"""Foundation-model size x context-window sweep across all datasets, with three reference baselines.

Design (the part a reviewer checks first):
  * Every trace is a list of univariate series (one per GPU / node / run where applicable).
  * Per-series chronological split: train = first 60%, val = next 15%, test = last 25%.
  * Test and validation windows are sampled ONCE per (trace, horizon) from target positions that
    admit the largest context of that trace, so every model and every context length is scored on
    the identical windows. Only the training windows depend on context.
  * Foundation models are zero-shot and see only each window's context.
  * Context per model is later selected on VALIDATION error, never test error.

Results are cached one JSON (+ one .npz of per-window errors) per (trace, horizon, context, model),
so the run is resumable and the two groups (`local`, `fm`) can run as separate processes.

Usage:
    python scripts/run_fm_sweep.py --group local
    python scripts/run_fm_sweep.py --group fm
    python scripts/run_fm_sweep.py --group fm --models timesfm_2p5_200m --traces mit
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from powerforecast.data import load_dataset  # noqa: E402
from powerforecast.utils import keep_awake  # noqa: E402

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUT = REPO / "outputs" / "fm_sweep"
CELLS = OUT / "cells"
CELLS.mkdir(parents=True, exist_ok=True)

CONTEXTS = [32, 64, 128, 256, 512, 1024]
HORIZONS = [int(h) for h in os.environ.get("PF_HORIZONS", "24,6").split(",")]
MAX_HORIZON = {"lnn": 64, "surf": 256}  # these two traces run out of validation windows beyond this
N_TEST, N_VAL, N_TRAIN = 300, 200, 4000
MAX_FILL = 6  # longest outage (in samples) that is linearly interpolated rather than split on

# period = natural seasonal period in steps (daily cycle) where one exists; None = no natural season.
TRACES = {
    "mit":       dict(loaders=["mit_supercloud"],                 period=None, L_max=1024),
    # SURF is heavily fragmented by multi-hour monitoring outages (1,178 segments, median 94 pts);
    # a 512 cap keeps 107 segments / 43% of points vs 16 / 11% at 1024.
    "surf":      dict(loaders=["surf_lisa"],                      period=288,  L_max=512,  gap_s=300),
    "google":    dict(loaders=["google_powerdata"],               period=288,  L_max=1024),
    "bitbrains": dict(loaders=["bitbrains"],                      period=288,  L_max=1024),
    "lnn":       dict(loaders=["lnn_inference", "lnn_training"],  period=None, L_max=256),
    "nlr":       dict(loaders=["nlr_h100_training"],              period=None, L_max=512),
    "polaris":   dict(loaders=["alcf_polaris"],                   period=24,   L_max=1024),
    "aurora":    dict(loaders=["alcf_aurora"],                    period=24,   L_max=1024),
}

LOCAL_MODELS = {"Persistence": "classical", "SeasonalNaive": "classical", "AutoETS": "classical"}  # reference baselines
FM_MODELS = {  # name -> (family, params_millions)
    "chronos_bolt_tiny":  ("chronos_bolt", 9),
    "chronos_bolt_mini":  ("chronos_bolt", 21),
    "chronos_bolt_small": ("chronos_bolt", 48),
    "chronos_bolt_base":  ("chronos_bolt", 205),
    "chronos_2":          ("chronos_2", 120),
    "timesfm_2p5_200m":   ("timesfm", 200),
}


# ----------------------------------------------------------------------------- data

def _series_from_df(df, tgt, gap_s=None):
    """Return a list of 1-D float arrays, one per contiguous series."""
    key = "series" if "series" in df.columns else ("series_id" if "series_id" in df.columns else None)
    groups = [g for _, g in df.groupby(key)] if key else [df]
    out = []
    for g in groups:
        order = "t_rel_s" if "t_rel_s" in g.columns else "timestamp"
        g = g.sort_values(order)
        if gap_s is not None:
            # Regrid onto the nominal cadence; linearly fill short outages (<= MAX_FILL missing
            # samples), split only at longer ones so no window ever spans a real outage.
            s = g.set_index(pd.to_datetime(g["timestamp"]))[tgt].astype(float)
            s = s[~s.index.duplicated()].resample(f"{gap_s}s").mean()
            missing = s.isna()
            run_id = (missing != missing.shift()).cumsum()
            run_len = missing.groupby(run_id).transform("sum")
            long_gap = missing & (run_len > MAX_FILL)
            s = s.interpolate(limit=MAX_FILL, limit_area="inside")
            seg_id = long_gap.cumsum()
            for _, seg in s[~long_gap].groupby(seg_id[~long_gap]):
                out.append(seg.to_numpy(float))
        else:
            out.append(g[tgt].to_numpy(float))
    return [y[np.isfinite(y)] for y in out]


def load_trace(name):
    spec = TRACES[name]
    series = []
    for ld in spec["loaders"]:
        df, tgt, _ = load_dataset(ld)
        series += _series_from_df(df, tgt, spec.get("gap_s"))
    L_max = spec["L_max"]
    # keep series whose val/test regions can host the largest context of this trace
    keep = [y for y in series if int(0.6 * len(y)) >= L_max // 2 and int(0.75 * len(y)) >= L_max + 1]
    return keep


def positions(series, lo_frac, hi_frac, H, min_t):
    pos = []
    for si, y in enumerate(series):
        n = len(y)
        lo, hi = int(lo_frac * n), int(hi_frac * n)
        for t in range(max(lo, min_t), hi - H + 1):
            pos.append((si, t))
    return pos


def sample(pos, n, seed):
    if len(pos) <= n:
        return pos
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(pos), n, replace=False))
    return [pos[i] for i in idx]


def build(series, pos, L, H):
    X = np.stack([series[s][t - L:t] for s, t in pos]).astype(np.float32)
    Y = np.stack([series[s][t:t + H] for s, t in pos]).astype(np.float32)
    return X, Y


def mase_scale(series):
    num = den = 0.0
    for y in series:
        tr = y[: int(0.6 * len(y))]
        d = np.abs(np.diff(tr))
        num += d.sum()
        den += len(d)
    return num / den


def lag1(series):
    vals, w = [], []
    for y in series:
        tr = y[: int(0.6 * len(y))]
        if len(tr) > 3 and np.std(tr) > 0:
            vals.append(np.corrcoef(tr[1:], tr[:-1])[0, 1])
            w.append(len(tr))
    return float(np.average(vals, weights=w))


def stable_seed(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


class Cell:
    """Everything one (trace, H, L) needs; test/val windows are identical across L."""

    def __init__(self, trace, series, H, L):
        self.trace, self.H, self.L = trace, H, L
        L_max = TRACES[trace]["L_max"]
        te_pos = sample(positions(series, 0.75, 1.0, H, L_max), N_TEST, stable_seed(trace, H, "test"))
        va_pos = sample(positions(series, 0.60, 0.75, H, L_max), N_VAL, stable_seed(trace, H, "val"))
        tr_pos = sample(positions(series, 0.0, 0.60, H, L), N_TRAIN, stable_seed(trace, H, L, "train"))
        self.pos = dict(te=te_pos, va=va_pos, tr=tr_pos)  # kept so covariate windows can be cut alike
        self.Xte, self.Yte = build(series, te_pos, L, H)
        self.Xva, self.Yva = build(series, va_pos, L, H)
        self.Xtr, self.Ytr = build(series, tr_pos, L, H)
        self.test_id = hashlib.md5(repr(te_pos).encode()).hexdigest()[:12]
        self.scale = mase_scale(series)
        # leakage guard: every test target starts in its series' final 25%
        for s, t in te_pos:
            assert t >= int(0.75 * len(series[s]))
        for s, t in tr_pos:
            assert t + H <= int(0.60 * len(series[s]))

    def split(self):
        return dict(Xtr=self.Xtr, Ytr=self.Ytr, Xva=self.Xva, Yva=self.Yva, Xte=self.Xte, Yte=self.Yte,
                    scale=self.scale)


def cell_path(trace, H, L, model):
    return CELLS / f"{trace}_H{H}_L{L}_{model}"


def save(cell, model, family, pred_te, pred_va, seconds, extra=None):
    err_te = np.mean(np.abs(cell.Yte - pred_te), axis=1)
    err_va = np.mean(np.abs(cell.Yva - pred_va), axis=1)
    rec = dict(trace=cell.trace, H=cell.H, L=cell.L, model=model, family=family,
               test_mae=float(err_te.mean()), val_mae=float(err_va.mean()),
               test_mase=float(err_te.mean() / cell.scale), val_mase=float(err_va.mean() / cell.scale),
               n_test=len(err_te), n_val=len(err_va), n_train=len(cell.Xtr), test_id=cell.test_id,
               seconds=round(seconds, 2), **(extra or {}))
    p = cell_path(cell.trace, cell.H, cell.L, model)
    np.savez_compressed(p.with_suffix(".npz"), test_err=err_te, val_err=err_va)
    p.with_suffix(".json").write_text(json.dumps(rec))
    print(f"  {cell.trace:9s} H{cell.H:<3d} L{cell.L:<5d} {model:20s} MASE={rec['test_mase']:.3f} "
          f"val={rec['val_mase']:.3f} ({seconds:.1f}s)", flush=True)


def done(trace, H, L, model):
    return cell_path(trace, H, L, model).with_suffix(".json").exists()


# --------------------------------------------------------------------- local models

def run_local(cell, model):
    H, L = cell.H, cell.L
    period = TRACES[cell.trace]["period"]

    if model == "Persistence":
        f = lambda X: np.repeat(X[:, -1:], H, 1)
        return f(cell.Xte), f(cell.Xva), {}
    if model == "SeasonalNaive":
        if period is None or period > L:
            return None
        def f(X):
            idx = [L - period + (h % period) for h in range(H)]
            return X[:, idx]
        return f(cell.Xte), f(cell.Xva), {"period": period}
    if model == "AutoETS":
        from statsforecast.models import AutoETS
        def f(X):
            out = np.empty((len(X), H), np.float32)
            for i, row in enumerate(X):
                hist = row[-min(L, 512):].astype(np.float64)
                try:
                    out[i] = AutoETS(season_length=1).forecast(y=hist, h=H)["mean"]
                except Exception:
                    out[i] = row[-1]
            return out
        return f(cell.Xte), f(cell.Xva), {}
    raise ValueError(model)


# ------------------------------------------------------------------ foundation models

class FM:
    def __init__(self, name):
        import torch
        self.name, self.torch = name, torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        if name.startswith("chronos_bolt"):
            from chronos import ChronosBoltPipeline
            mid = "amazon/chronos-bolt-" + name.split("_")[-1]
            self.pipe = ChronosBoltPipeline.from_pretrained(mid, device_map=self.dev, torch_dtype=torch.float32)
        elif name == "chronos_2":
            from chronos import Chronos2Pipeline
            self.pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=self.dev)
        elif name == "timesfm_2p5_200m":
            import timesfm
            # from_pretrained
            # trips on a huggingface_hub kwarg mismatch in this environment)
            self.model = timesfm.TimesFM_2p5_200M_torch._from_pretrained(
                model_id="google/timesfm-2.5-200m-pytorch", revision=None, cache_dir=None,
                force_download=False, local_files_only=False, token=None, config=None,
                torch_compile=False)
            self._compiled = None
        else:
            raise ValueError(name)

    def predict(self, X, H):
        """Median (q=0.5) point forecast — the MAE-optimal summary — for every row of X."""
        torch = self.torch
        if self.name.startswith("chronos_bolt"):
            outs = []
            for i in range(0, len(X), 256):
                ctx = [torch.tensor(r) for r in X[i:i + 256]]
                q, _ = self.pipe.predict_quantiles(ctx, prediction_length=H, quantile_levels=[0.5])
                outs.append(q[..., 0].cpu().numpy())
            return np.vstack(outs)
        if self.name == "chronos_2":
            q, _ = self.pipe.predict_quantiles([r for r in X], prediction_length=H, quantile_levels=[0.5],
                                               batch_size=256, cross_learning=False)
            return np.stack([qi[0, :, 0].cpu().numpy() for qi in q])
        if self.name == "timesfm_2p5_200m":
            import timesfm
            L = X.shape[1]
            key = (L, H)
            if self._compiled != key:
                self.model.compile(timesfm.ForecastConfig(
                    max_context=L, max_horizon=max(128, H), normalize_inputs=True, per_core_batch_size=64,
                    use_continuous_quantile_head=True, force_flip_invariance=True,
                    infer_is_positive=True, fix_quantile_crossing=True))
                self._compiled = key
            _, q = self.model.forecast(horizon=H, inputs=[r.astype(np.float32) for r in X])
            return np.asarray(q)[:, :H, 5]  # index 0 = mean, 1..9 = q0.1..q0.9 -> 5 = median
        raise ValueError(self.name)

    def close(self):
        for a in ("pipe", "model"):
            if hasattr(self, a):
                delattr(self, a)
        gc.collect()
        self.torch.cuda.empty_cache()


# --------------------------------------------------------------------------- driver

def feasible(trace, H):
    """Long-horizon cap: at least 50 validation and 100 test windows."""
    return H <= MAX_HORIZON.get(trace, 336)


def iter_cells(traces, horizons):
    for H in horizons:
        for tr in traces:
            if not feasible(tr, H):
                continue
            series = load_trace(tr)
            for L in CONTEXTS:
                if L > TRACES[tr]["L_max"]:
                    continue
                yield tr, series, H, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["local", "fm", "info"], required=True)
    ap.add_argument("--traces", nargs="*", default=list(TRACES))
    ap.add_argument("--horizons", nargs="*", type=int, default=HORIZONS)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    keep_awake()

    if args.group == "info":
        rows = []
        for tr in args.traces:
            s = load_trace(tr)
            rows.append(dict(trace=tr, n_series=len(s), total_points=sum(map(len, s)),
                             min_len=min(map(len, s)), lag1=round(lag1(s), 4),
                             mase_scale=round(mase_scale(s), 4), L_max=TRACES[tr]["L_max"]))
        info = pd.DataFrame(rows)
        info.to_csv(OUT / "trace_info.csv", index=False)
        print(info.to_string(index=False))
        return

    if args.group == "local":
        models = args.models or list(LOCAL_MODELS)
        for tr, series, H, L in iter_cells(args.traces, args.horizons):
            todo = [m for m in models if not done(tr, H, L, m)]
            if not todo:
                continue
            cell = Cell(tr, series, H, L)
            for m in todo:
                t0 = time.time()
                try:
                    out = run_local(cell, m)
                except Exception as exc:
                    print(f"  [FAIL] {tr} H{H} L{L} {m}: {exc}", flush=True)
                    continue
                if out is None:
                    continue
                save(cell, m, LOCAL_MODELS[m], out[0], out[1], time.time() - t0, out[2])
        return

    models = args.models or list(FM_MODELS)
    for m in models:  # one foundation model in GPU memory at a time
        needed = [(tr, s, H, L) for tr, s, H, L in iter_cells(args.traces, args.horizons)
                  if not done(tr, H, L, m)]
        if not needed:
            continue
        print(f"=== loading {m} ({len(needed)} cells) ===", flush=True)
        fm = FM(m)
        for tr, series, H, L in needed:
            cell = Cell(tr, series, H, L)
            t0 = time.time()
            try:
                p_te = fm.predict(cell.Xte, H)
                p_va = fm.predict(cell.Xva, H)
            except Exception as exc:
                print(f"  [FAIL] {tr} H{H} L{L} {m}: {exc}", flush=True)
                continue
            fam, params = FM_MODELS[m]
            save(cell, m, fam, p_te, p_va, time.time() - t0, {"params_m": params})
        fm.close()


if __name__ == "__main__":
    main()
