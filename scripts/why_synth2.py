"""H1 x H5  Estimation error vs adaptation: does MORE training data let local models catch up?

Long synthetic series (T = 120,000) for five processes; local models are trained on N = 1k, 4k, 16k
and 64k windows from the first 60%, with the configuration Optuna chose for that process in the
short-series study (why_synth). Zero-shot FMs are scored on the same test windows.
  * if the FM advantage is ESTIMATION error (local models fit an easy process imperfectly from
    finite data), it should shrink toward zero as N grows;
  * if it is ADAPTATION (the test regime differs from training, e.g. shift_*), more data from the
    old regime should not help.

    python scripts/why_synth2.py work --worker 1
    python scripts/why_synth2.py analyze
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
import optuna  # noqa: E402

import run_dl_hpo as dl  # noqa: E402
import run_fm_sweep as sw  # noqa: E402
import why_local as wl  # noqa: E402
import why_synth as ws  # noqa: E402  (sets dl.STUDIES to the synthetic study folder)

OUT = wl.OUT / "synth2"
OUT.mkdir(parents=True, exist_ok=True)
T_LONG, H = 120_000, 24
PROCS = ["ar_0.9", "ar_0.97", "seas", "shift_3", "onoff_0.02"]
NS = [1000, 4000, 16000, 64000]
LOCAL = ["NHITS", "PatchTST", "LightGBM", "Ridge"]
FMS = ["timesfm_3p0_330m", "chronos_2"]


def register(name):
    key = f"syl_{name}"
    if key not in sw.TRACES:
        sw.TRACES[key] = dict(loaders=[], period=24 if name == "seas" else None, L_max=1024)
        dl._SERIES[key] = [ws.gen(name, T=T_LONG)[0]]
    return key


def oracle(name):
    key = register(name)
    _, f = ws.gen(name, T=T_LONG)
    c = sw.Cell(key, dl._SERIES[key], H, 64)
    return float(np.mean(np.abs(c.Yte - np.stack([f(o) for _, o in c.pos["te"]]))) / c.scale)


def cfg(model, name):
    """Configuration tuned on the SHORT version of the same process (validation only)."""
    short = ws.register(name)  # the short series must be known to the search-space code
    st = optuna.load_study(study_name=dl.study_name(model, short, H), storage=dl.storage(model, short, H))
    fixed = optuna.trial.FixedTrial(st.best_trial.params)
    if model in wl.ML_MODELS:
        import run_ml_hpo as ml
        L, hp = ml.space(fixed, model, short)
        return L, hp, None
    return dl.space(fixed, model, short, H)


def tasks():
    return [("local", m, p, n) for p in PROCS for m in LOCAL for n in NS] + [("fm", m, p, 0) for p in PROCS for m in FMS]


def tag(t):
    return "__".join(map(str, t))


def run(kind, model, name, n):
    key = register(name)
    t0 = time.time()
    if kind == "local":
        L, kw, tr = cfg(model, name)
        c = wl.make_cell(key, H, L, n=n)
        pv, pt = zip(*[wl.fit_predict(c, model, (L, kw, tr), s) for s in dl.FINAL_SEEDS])
        te = [float(np.mean(np.abs(c.Yte - p)) / c.scale) for p in pt]
        va = [float(np.mean(np.abs(c.Yva - p)) / c.scale) for p in pv]
        rec = dict(L=L, n_train=len(c.Xtr), test_mase=float(np.mean(te)), test_mase_std=float(np.std(te, ddof=1)),
                   val_mase=float(np.mean(va)))
    else:
        if model == "timesfm_3p0_330m":
            sys.path.insert(0, os.environ.get("PF_TIMESFM3_PKGS", ""))
            import run_fm_sweep_timesfm3 as t3
            pred = t3.TimesFM3().predict
        else:
            pred = sw.FM(model).predict
        best = None
        for L in ws.FM_L:
            c = sw.Cell(key, dl._SERIES[key], H, L)
            r = dict(L=L, test_mase=float(np.mean(np.abs(c.Yte - pred(c.Xte, H))) / c.scale),
                     val_mase=float(np.mean(np.abs(c.Yva - pred(c.Xva, H))) / c.scale))
            best = r if best is None or r["val_mase"] < best["val_mase"] else best
        rec = dict(best, test_mase_std=0.0)
    rec.update(kind=kind, model=model, proc=name, N=n, seconds=round(time.time() - t0, 1))
    return rec


def work(worker):
    job = os.environ.get("SLURM_JOB_ID", "local")
    for t in tasks():
        out = OUT / f"{tag(t)}.json"
        if out.exists():
            continue
        d = wl.CLAIMS / ("syn2__" + tag(t))
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
    d = pd.DataFrame([json.loads(p.read_text()) for p in OUT.glob("*.json")])
    rows = []
    for p in PROCS:
        o = oracle(p)
        g = d[d.proc == p]
        fm = g[g.kind == "fm"].sort_values("val_mase").iloc[0]
        row = dict(proc=p, oracle=o, fm=fm.model, fm_excess=(fm.test_mase / o - 1) * 100)
        for n in NS:
            loc = g[(g.kind == "local") & (g.N == n)].sort_values("val_mase").iloc[0]
            row[f"local_excess_N{n}"] = (loc.test_mase / o - 1) * 100
        rows.append(row)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "synth2_results.csv", index=False)
    pd.set_option("display.width", 200)
    print(r.round(2).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["work", "analyze"])
    ap.add_argument("--worker", type=int, default=0)
    a = ap.parse_args()
    work(a.worker) if a.mode == "work" else analyze()


if __name__ == "__main__":
    main()
