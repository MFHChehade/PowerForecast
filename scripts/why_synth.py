"""H5  Controlled synthetic study: zero-shot FMs vs tuned local models on series with a known best forecast.

Each synthetic "trace" is one series of T = 9000 steps generated from a process whose optimal
forecaster is known, so every model can be scored as EXCESS error over the Bayes oracle:
  ar_<phi>     AR(1), lag-1 autocorrelation = phi           (smoothness sweep)
  seas         AR(1) phi=.9 + daily seasonality (period 24)
  ll_<q>       local level (random walk + noise), signal-to-noise q (non-stationary)
  shift_<k>    AR(1) phi=.9 whose mean jumps by k std devs at 70% (train/test distribution shift)
  onoff_<p>    two-level Markov switching (GPU job on/off) with switch prob p, + AR(.5) noise;
               its "oracle" knows the hidden state, so it is a lower bound, not an achievable target
The protocol is exactly the real-data one (same splits, windows, MASE, tuned local models chosen on
validation, FMs zero-shot with context chosen on validation), at horizon 24.

    python scripts/why_synth.py work --worker 1
    python scripts/why_synth.py analyze
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("PF_STUDY_SPLIT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import run_dl_hpo as dl  # noqa: E402
import run_ml_hpo as ml  # noqa: E402
import run_fm_sweep as sw  # noqa: E402
import why_local as wl  # noqa: E402

OUT = wl.OUT / "synth"
for d in (OUT, OUT / "final", OUT / "studies" / "split", OUT / "fm"):
    d.mkdir(parents=True, exist_ok=True)
dl.FINAL, dl.STUDIES = OUT / "final", OUT / "studies"  # keep synthetic results out of the real-data folders

T, H = 9000, 24
FM_L = [64, 256, 512, 1024]
LOCAL = ["NHITS", "PatchTST", "NBEATS", "LightGBM", "Ridge"]
FMS = ["timesfm_3p0_330m", "chronos_2", "timesfm_2p5_200m", "chronos_bolt_small"]
N_TRIALS = 10


def gen(name, seed=0, T=T):
    """Return (y, oracle_fn) where oracle_fn(t) gives the optimal H-step forecast from origin t."""
    rng = np.random.default_rng(sw.stable_seed("synth", name, seed) if T == 9000 else sw.stable_seed("synth", name, seed, T))
    t = np.arange(T)

    def ar(phi, sd):
        e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), T)
        x = np.zeros(T)
        for i in range(1, T):
            x[i] = phi * x[i - 1] + e[i]
        return x

    hs = np.arange(1, H + 1)
    kind, _, par = name.partition("_")
    if kind == "ar":
        phi = float(par); x = ar(phi, 10.0); y = 100 + x
        return y, lambda o: 100 + phi ** hs * x[o - 1]
    if kind == "seas":
        x = ar(0.9, 5.0); s = 10 * np.sin(2 * np.pi * t / 24); y = 100 + s + x
        return y, lambda o: 100 + 10 * np.sin(2 * np.pi * (o - 1 + hs) / 24) + 0.9 ** hs * x[o - 1]
    if kind == "ll":
        q = float(par); sd_e = 5.0; sd_n = sd_e * np.sqrt(q)
        mu = 3000 + np.cumsum(rng.normal(0, sd_n, T)); y = mu + rng.normal(0, sd_e, T)
        # steady-state Kalman gain of the local-level model; forecast = filtered level (flat)
        P = (q + np.sqrt(q * q + 4 * q)) / 2 * sd_e ** 2
        K = P / (P + sd_e ** 2)
        lev = np.empty(T); lev[0] = y[0]
        for i in range(1, T):
            lev[i] = lev[i - 1] + K * (y[i] - lev[i - 1])
        return y, lambda o: np.full(H, lev[o - 1])
    if kind == "shift":
        k = float(par); x = ar(0.9, 10.0); jump = int(0.70 * T)
        m = 100 + 10 * k * (t >= jump); y = m + x
        return y, lambda o: 100 + 10 * k * ((o - 1 + hs) >= jump) + 0.9 ** hs * x[o - 1]
    if kind == "onoff":
        p = float(par); st = np.zeros(T, int)
        for i in range(1, T):
            st[i] = 1 - st[i - 1] if rng.random() < p else st[i - 1]
        lv = np.array([50.0, 250.0]); x = ar(0.5, 10.0); y = lv[st] + x
        # MAE-optimal = MEDIAN of the predictive distribution, a two-component Gaussian mixture:
        # P(same state after h) = (1 + (1-2p)^h) / 2, noise mean .5^h x, var 10^2 (1 - .25^h)
        same = (1 + (1 - 2 * p) ** hs) / 2
        sd = 10 * np.sqrt(1 - 0.25 ** hs)

        def median(o):
            from scipy.optimize import brentq
            from scipy.stats import norm
            mu_same = lv[st[o - 1]] + 0.5 ** hs * x[o - 1]
            mu_flip = lv[1 - st[o - 1]] + 0.5 ** hs * x[o - 1]
            cdf = lambda z, i: same[i] * norm.cdf(z, mu_same[i], sd[i]) + (1 - same[i]) * norm.cdf(z, mu_flip[i], sd[i]) - .5
            return np.array([brentq(cdf, -200, 500, args=(i,)) for i in range(H)])
        return y, median
    raise ValueError(name)


TRACES = ["ar_0.3", "ar_0.6", "ar_0.9", "ar_0.97", "ar_0.995", "seas", "ll_0.01", "ll_0.1", "ll_1",
          "shift_0", "shift_3", "shift_6", "onoff_0.02", "onoff_0.1"]


def register(name):
    """Make a synthetic series look like a real trace to run_fm_sweep / run_dl_hpo."""
    key = "syn_" + name
    if key not in sw.TRACES:
        sw.TRACES[key] = dict(loaders=[], period=24 if name == "seas" else None, L_max=1024)
        y, _ = gen(name)
        dl._SERIES[key] = [y]
    return key


def oracle_mase(name):
    key = register(name)
    y, f = gen(name)
    c = sw.Cell(key, dl._SERIES[key], H, 64)
    P = np.stack([f(o) for _, o in c.pos["te"]])
    return float(np.mean(np.abs(c.Yte - P)) / c.scale), float(sw.lag1(dl._SERIES[key]))


# ------------------------------------------------------------------------------- tasks

def tasks():
    return [("local", m, n) for n in TRACES for m in LOCAL] + [("fm", m, n) for n in TRACES for m in FMS]


def tag(t):
    return "__".join(t)


def run(kind, model, name):
    key = register(name)
    t0 = time.time()
    if kind == "local":
        if model in wl.ML_MODELS:
            ml.TRIALS[model] = N_TRIALS
            ml.run_hpo(model, key, H)
            ml.run_final(model, key, H)
        else:
            dl.run_hpo(model, key, H, N_TRIALS)
            dl.run_final(model, key, H)
        rec = json.loads((dl.FINAL / f"{key}_H{H}_{model}.json").read_text())
        rec["errs"] = str(dl.FINAL / f"{key}_H{H}_{model}.npz")
    else:
        if model == "timesfm_3p0_330m":
            sys.path.insert(0, os.environ.get("PF_TIMESFM3_PKGS", ""))
            import run_fm_sweep_timesfm3 as t3
            fm = t3.TimesFM3(); pred = fm.predict
        else:
            fm = sw.FM(model); pred = fm.predict
        rows = []
        for L in FM_L:
            c = sw.Cell(key, dl._SERIES[key], H, L)
            pt, pv = pred(c.Xte, H), pred(c.Xva, H)
            rows.append(dict(L=L, test_mase=float(np.mean(np.abs(c.Yte - pt)) / c.scale),
                             val_mase=float(np.mean(np.abs(c.Yva - pv)) / c.scale),
                             err=np.mean(np.abs(c.Yte - pt), 1)))
        best = min(rows, key=lambda r: r["val_mase"])
        np.savez_compressed(OUT / "fm" / f"{key}_{model}.npz", test_err=best["err"])
        rec = dict(trace=key, model=model, L=best["L"], test_mase=best["test_mase"], val_mase=best["val_mase"],
                   by_L={r["L"]: r["test_mase"] for r in rows}, test_mase_std=0.0)
    rec.update(kind=kind, synth=name, seconds=round(time.time() - t0, 1))
    return rec


def work(worker, only=None):
    job = os.environ.get("SLURM_JOB_ID", "local")
    for t in tasks():
        out = OUT / f"{tag(t)}.json"
        if out.exists() or (only and t[1] not in only):
            continue
        d = wl.CLAIMS / ("synth__" + tag(t))
        try:
            d.mkdir()
        except FileExistsError:
            continue
        (d / "job").write_text(job)
        try:
            rec = run(*t)
            out.write_text(json.dumps(rec))
            print(f"[w{worker}] {tag(t)} MASE={rec['test_mase']:.3f}", flush=True)
        except Exception:
            (d / "failed").write_text(traceback.format_exc())
            print(f"[w{worker}] FAILED {tag(t)}\n{traceback.format_exc()}", flush=True)


def analyze():
    import pandas as pd
    rows = []
    for n in TRACES:
        o, lag = oracle_mase(n)
        for kind, m, _ in [t for t in tasks() if t[2] == n]:
            p = OUT / f"{tag((kind, m, n))}.json"
            if p.exists():
                r = json.loads(p.read_text())
                rows.append(dict(synth=n, lag1=lag, oracle=o, model=m, kind=kind, test_mase=r["test_mase"],
                                 val_mase=r["val_mase"], excess_pct=(r["test_mase"] / o - 1) * 100, L=r.get("L")))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "synthetic_results.csv", index=False)
    best = d.loc[d.groupby(["synth", "kind"]).val_mase.idxmin()]
    w = best.pivot(index="synth", columns="kind", values=["model", "test_mase", "excess_pct"])
    w["oracle"] = d.groupby("synth").oracle.first(); w["lag1"] = d.groupby("synth").lag1.first()
    w["fm_gain_pct"] = (w[("test_mase", "local")] - w[("test_mase", "fm")]) / w[("test_mase", "local")] * 100
    pd.set_option("display.width", 220)
    print(w.loc[TRACES].round(3).to_string())
    print(d.pivot_table(index="synth", columns="model", values="excess_pct").loc[TRACES].round(1).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["work", "analyze", "oracle", "one"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--task", nargs=3)
    a = ap.parse_args()
    if a.mode == "work":
        work(a.worker, a.models)
    elif a.mode == "analyze":
        analyze()
    elif a.mode == "one":
        print(json.dumps(run(*a.task))[:400])
    else:
        for n in TRACES:
            o, lag = oracle_mase(n)
            print(f"{n:12s} lag1={lag:.3f} oracle MASE={o:.3f}")


if __name__ == "__main__":
    main()
