"""Zero-shot foundation models vs tuned local models: local-model experiments.

Every run reuses the configuration Optuna selected for that (model, trace, horizon) in the tuned-
baseline study (validation only), retrains it with 3 seeds, and is scored on the shared test windows.
Only the TRAINING windows change between experiments:

  lc      H1 data scarcity  - nested training subsets of N windows (250 ... 64000 or all available)
  recent  H2 shift/recency  - training windows drawn from 15-75% (same N, more recent) or 0-75%
                              (more and more recent) of each series instead of 0-60%
  preds   H7/H9 baseline    - the standard 0-60% / 4000-window cell, saving predictions so seed and
                              model ensembles and error decompositions can be computed afterwards

    python scripts/why_local.py work --worker 1        # cluster: claim tasks until none are left
    python scripts/why_local.py status
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("PF_STUDY_SPLIT", "1")  # tuned configurations live in one SQLite file per study
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import optuna  # noqa: E402

import run_dl_hpo as dl  # noqa: E402
import run_ml_hpo as ml  # noqa: E402
import run_fm_sweep as sw  # noqa: E402

OUT = sw.OUT.parent / "why"
CLAIMS = OUT / "claims"
for d in (OUT, CLAIMS, OUT / "logs"):
    d.mkdir(parents=True, exist_ok=True)

ML_MODELS = {"Ridge", "LightGBM"}
LC_N = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
EXPERIMENTS = {
    "lc": dict(models=["PatchTST", "NHITS", "NBEATS", "LightGBM", "Ridge"],
               variants=[str(n) for n in LC_N] + ["all"]),
    "recent": dict(models=["PatchTST", "NHITS", "NBEATS", "LightGBM", "Ridge"],
                   variants=["0.15-0.75", "0.00-0.75"]),
    # H6 x H1: can local models USE long context if forced to it, with the usual or with all data?
    "longctx": dict(models=["PatchTST", "NHITS", "NBEATS"],
                    variants=[f"{L}-{n}" for n in ("4000", "all") for L in (256, 512, 1024)]),
    "preds": dict(models=["PatchTST", "NHITS", "NBEATS", "BiTCN", "TSMixer", "MLP", "TCN", "LSTM",
                          "LightGBM", "Ridge"], variants=["base"]),
}


# ------------------------------------------------------------------ configurations and cells

def best_cfg(model, trace, H):
    """The tuned configuration (selected on validation in the tuned-baseline study)."""
    st = optuna.load_study(study_name=dl.study_name(model, trace, H), storage=dl.storage(model, trace, H))
    fixed = optuna.trial.FixedTrial(st.best_trial.params)
    if model in ML_MODELS:
        L, hp = ml.space(fixed, model, trace)
        return L, hp, None
    return dl.space(fixed, model, trace, H)


def make_cell(trace, H, L, n=None, region=(0.0, 0.60)):
    """Standard cell (identical test/val windows); training windows optionally re-drawn.

    n=None and region=(0, .6) gives exactly the tuned-baseline cell. Otherwise training windows are a
    NESTED random subset (fixed permutation, so N=1000 contains N=500) of all positions in `region`.
    """
    series = dl.series(trace)
    c = sw.Cell(trace, series, H, L)
    c.n_avail = len(sw.positions(series, region[0], region[1], H, L))
    if n is None and region == (0.0, 0.60):
        return c
    pos = sw.positions(series, region[0], region[1], H, L)
    rng = np.random.default_rng(sw.stable_seed(trace, H, L, "why", *region))
    keep = np.sort(rng.permutation(len(pos))[: (len(pos) if n is None else min(n, len(pos)))])
    tr_pos = [pos[i] for i in keep]
    for s, t in tr_pos:  # no training target ever reaches into the test region
        assert t + H <= int(0.75 * len(series[s]))
    c.Xtr, c.Ytr = sw.build(series, tr_pos, L, H)
    return c


def fit_predict(cell, model, cfg, seed):
    L, kw, tr = cfg
    if model in ML_MODELS:
        return ml.fit_predict(cell, model, kw, seed, return_pred=True)
    return dl.train_eval(cell, model, kw, tr, seed, return_pred=True)


# ------------------------------------------------------------------------------- tasks

def tasks(exps=None):
    out = []
    for e in exps or EXPERIMENTS:
        spec = EXPERIMENTS[e]
        for m in spec["models"]:
            for H in sw.HORIZONS:
                for tr in sw.TRACES:
                    for v in spec["variants"]:
                        out.append((e, m, tr, H, v))
    return out


def tag(e, m, tr, H, v):
    return f"{e}__{m}__{tr}__H{H}__{v}"


def path(t):
    return OUT / t[0] / f"{tag(*t)}.json"


def run_task(e, m, tr, H, v):
    (OUT / e).mkdir(parents=True, exist_ok=True)
    L, kw, trn = best_cfg(m, tr, H)
    if e == "lc":
        probe = make_cell(tr, H, L)
        n = min(probe.n_avail, 64000) if v == "all" else int(v)
        if v != "all" and n > probe.n_avail:
            return dict(skipped=True, reason=f"only {probe.n_avail} positions", n_avail=probe.n_avail)
        cell = make_cell(tr, H, L, n=n)
    elif e == "longctx":
        Ls, ns = v.split("-")
        L = int(Ls)
        if L > sw.TRACES[tr]["L_max"] or (m in dl.HEAVY and L > 512):
            return dict(skipped=True, reason=f"L={L} not allowed for this trace/model")
        if m == "PatchTST":  # patch length was tuned for the tuned context; keep it, but within L/2
            kw = dict(kw, patch_len=min(kw["patch_len"], L // 2), stride=max(1, min(kw["patch_len"], L // 2) // 2))
        probe = make_cell(tr, H, L)
        cell = make_cell(tr, H, L, n=(min(probe.n_avail, 64000) if ns == "all" else int(ns)))
    elif e == "recent":
        lo, hi = (float(x) for x in v.split("-"))
        cell = make_cell(tr, H, L, n=sw.N_TRAIN, region=(lo, hi))
    else:
        cell = make_cell(tr, H, L)
    t0 = time.time()
    pv, pt = zip(*[fit_predict(cell, m, (L, kw, trn), s) for s in dl.FINAL_SEEDS])
    te = [float(np.mean(np.abs(cell.Yte - p)) / cell.scale) for p in pt]
    va = [float(np.mean(np.abs(cell.Yva - p)) / cell.scale) for p in pv]
    np.savez_compressed(path((e, m, tr, H, v)).with_suffix(".npz"),
                        pred_te=np.stack(pt).astype(np.float32), pred_va=np.stack(pv).astype(np.float32))
    return dict(exp=e, model=m, trace=tr, H=H, variant=v, L=L, n_train=len(cell.Xtr), n_avail=cell.n_avail,
                test_mase_seeds=te, val_mase_seeds=va, test_mase=float(np.mean(te)),
                test_mase_std=float(np.std(te, ddof=1)), val_mase=float(np.mean(va)), test_id=cell.test_id,
                seconds=round(time.time() - t0, 1))


def release():
    try:
        live = set(subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i"],
                                  capture_output=True, text=True).stdout.split())
    except FileNotFoundError:
        live = set()
    n = 0
    for d in CLAIMS.iterdir():
        job = (d / "job").read_text().strip() if (d / "job").exists() else ""
        t = d.name.split("__")
        done = (OUT / t[0] / f"{d.name}.json").exists()
        if not done and job not in live and not (d / "failed").exists():
            shutil.rmtree(d, ignore_errors=True); n += 1
    print(f"released {n} stale claims")


def work(worker, exps=None, only=None):
    job = os.environ.get("SLURM_JOB_ID", "local")
    for t in tasks(exps):
        if path(t).exists() or (only and t[1] not in only):
            continue
        d = CLAIMS / tag(*t)
        try:
            d.mkdir()
        except FileExistsError:
            continue
        (d / "job").write_text(job)
        path(t).parent.mkdir(parents=True, exist_ok=True)
        try:
            rec = run_task(*t)
            path(t).write_text(json.dumps(rec))
            msg = "skip" if rec.get("skipped") else f"MASE={rec['test_mase']:.3f}±{rec['test_mase_std']:.3f}"
            print(f"[w{worker}] {tag(*t)} {msg}", flush=True)
        except Exception:
            (d / "failed").write_text(traceback.format_exc())
            print(f"[w{worker}] FAILED {tag(*t)}\n{traceback.format_exc()}", flush=True)


def status(exps=None):
    ts = tasks(exps)
    done = sum(path(t).exists() for t in ts)
    failed = [t for t in ts if (CLAIMS / tag(*t) / "failed").exists()]
    claimed = sum((CLAIMS / tag(*t)).exists() and not path(t).exists() for t in ts)
    print(f"tasks {len(ts)}  done {done}  in progress {claimed - len(failed)}  failed {len(failed)}  "
          f"queued {len(ts) - done - claimed}")
    for t in failed[:20]:
        print("  failed:", tag(*t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["work", "status", "release", "list", "one"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--exps", nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--task", nargs=5)
    a = ap.parse_args()
    if a.mode == "work":
        work(a.worker, a.exps, a.models)
    elif a.mode == "status":
        status(a.exps)
    elif a.mode == "release":
        release()
    elif a.mode == "one":
        e, m, tr, H, v = a.task
        print(json.dumps(run_task(e, m, tr, int(H), v))[:600])
    else:
        for t in tasks(a.exps):
            print(tag(*t), "done" if path(t).exists() else "")


if __name__ == "__main__":
    main()
