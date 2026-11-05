"""Which family forecasts outliers better: foundation models or tuned site models?

Scored on up to 2000 test windows per (trace, H), far more than the 300 of the headline, because outliers
are rare. Every model uses the context and settings already chosen on validation, so nothing here is
tuned for outliers. Site models = the 3 best tuned models by validation MASE, retrained with their
stored settings and 3 seeds.

A target step is an outlier under three definitions, each fixed before looking at any forecast:
    surprise  |y - median(context)| > 5 * max(1.4826 * MAD(context), MASE scale), context = last 256 readings at most
    tail      y outside the 0.5% / 99.5% quantiles of the trace's training region
    jump      |y - previous y| at or above the 99.5% quantile of one-step changes in the training region

    python scripts/why_outliers.py fm --family tfm3
    python scripts/why_outliers.py site --worker 1
    python scripts/why_outliers.py analyze
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run_fm_sweep as sw  # noqa: E402

OUT = sw.OUT.parent / "why" / "outliers"
PRED = OUT / "preds"
CLAIMS = OUT / "claims"
for d in (OUT, PRED, CLAIMS):
    d.mkdir(parents=True, exist_ok=True)
N_WINDOWS, HORIZONS, N_BOOT = 2000, [24, 6], 2000
Z, TOP = 5.0, 0.995
FMS = {"bolt": ["chronos_bolt_base"], "chronos2": ["chronos_2"], "tfm25": ["timesfm_2p5_200m"],
       "tfm3": ["timesfm_3p0_330m"], "timestral": ["timestral_209m"]}


def windows(series, trace, H):
    pos = sw.positions(series, 0.75, 1.0, H, sw.TRACES[trace]["L_max"])
    return sw.sample(pos, N_WINDOWS, sw.stable_seed(trace, H, "outliers"))


def best_context(model, trace, H):
    """The context chosen on validation in the FM sweep."""
    recs = [json.loads(p.read_text()) for p in sw.CELLS.glob(f"{trace}_H{H}_L*_{model}.json")]
    return min(recs, key=lambda r: r["val_mase"])["L"]


def predictor(model):
    if model == "timesfm_3p0_330m":
        import run_fm_sweep_timesfm3 as t3
        return t3.TimesFM3().predict
    if model == "timestral_209m":
        import run_fm_sweep_timestral as ts
        return ts.Timestral("209M").predict
    return sw.FM(model).predict


def run_fm(family):
    for model in FMS[family]:
        predict = None
        for trace in sw.TRACES:
            series = sw.load_trace(trace)
            for H in HORIZONS:
                p = PRED / f"{trace}_H{H}_{model}.npz"
                if p.exists():
                    continue
                predict = predict or predictor(model)
                pos = windows(series, trace, H)
                L = best_context(model, trace, H)
                X, _ = sw.build(series, pos, L, H)
                np.savez_compressed(p, pred=np.asarray(predict(X, H)).astype(np.float32)[None], L=L)
                print(f"{model} {trace} H{H} L{L} n={len(pos)}", flush=True)


def top_site(trace, H, k=3):
    import run_dl_hpo as dl
    recs = [json.loads(p.read_text()) for p in dl.FINAL.glob(f"{trace}_H{H}_*.json")]
    return sorted(recs, key=lambda r: r["val_mase"])[:k]


def run_site_task(trace, H):
    import optuna

    import run_dl_hpo as dl
    import run_ml_hpo as ml
    series = dl.series(trace)
    pos = windows(series, trace, H)
    for rec in top_site(trace, H):
        model = rec["model"]
        p = PRED / f"{trace}_H{H}_site_{model}.npz"
        if p.exists():
            continue
        fixed = optuna.trial.FixedTrial(rec["params"])
        is_ml = model in ml.ML_MODELS
        if is_ml:
            L, hp = ml.space(fixed, model, trace)
        else:
            L, kw, tr = dl.space(fixed, model, trace, H)
        cell = sw.Cell(trace, series, H, L)
        cell.Xte, cell.Yte = sw.build(series, pos, L, H)  # same training and validation, more test windows
        preds = []
        for sd in dl.FINAL_SEEDS:
            _, p_te = ml.fit_predict(cell, model, hp, sd, return_pred=True) if is_ml else \
                dl.train_eval(cell, model, kw, tr, sd, return_pred=True)
            preds.append(p_te)
        np.savez_compressed(p, pred=np.stack(preds).astype(np.float32), L=L, val_mase=rec["val_mase"])
        print(f"site {model} {trace} H{H} L{L} n={len(pos)}", flush=True)


def work_site(worker):
    job = os.environ.get("SLURM_JOB_ID", "local")
    for H in HORIZONS:
        for trace in sw.TRACES:
            d = CLAIMS / f"{trace}__H{H}"
            if (d / "done").exists():
                continue
            try:
                d.mkdir()
            except FileExistsError:
                continue
            (d / "job").write_text(job)
            try:
                run_site_task(trace, H)
                (d / "done").write_text("")
            except Exception:
                (d / "failed").write_text(traceback.format_exc())
                print(f"[w{worker}] FAILED {trace} H{H}\n{traceback.format_exc()}", flush=True)


# ------------------------------------------------------------------------------- scoring

def masks(series, trace, H):
    """Boolean [N, H] outlier masks per definition, the target array, and the reference level per window."""
    pos = windows(series, trace, H)
    L_ref = min(256, sw.TRACES[trace]["L_max"])
    scale = sw.mase_scale(series)
    train = np.concatenate([y[: int(0.6 * len(y))] for y in series])
    lo, hi = np.quantile(train, [1 - TOP, TOP])
    jump = np.quantile(np.concatenate([np.abs(np.diff(y[: int(0.6 * len(y))])) for y in series]), TOP)
    X, Y = sw.build(series, pos, L_ref, H)
    med = np.median(X, 1, keepdims=True)
    width = Z * np.maximum(1.4826 * np.median(np.abs(X - med), 1, keepdims=True), scale)
    prev = np.concatenate([X[:, -1:], Y[:, :-1]], 1)
    return dict(surprise=np.abs(Y - med) > width, tail=(Y < lo) | (Y > hi), jump=np.abs(Y - prev) >= jump), \
        Y, med, width, prev, scale


def score(pred, Y, mask, med, width, scale, name):
    """Scores averaged over seeds; pred is [seeds, N, H]."""
    err = np.abs(pred - Y[None]).mean(0)  # [N, H]
    out = dict(mase_out=float(err[mask].mean() / scale), mase_rest=float(err[~mask].mean() / scale))
    gap = Y - med
    sign = np.sign(gap)
    # share of the excursion from the context median that the forecast reached (0 = none, 1 = all)
    cap = np.clip(((pred - med[None]) * sign[None]).mean(0)[mask] / np.maximum(np.abs(gap[mask]), 1e-9), -1, 2)
    out["capture"] = float(cap.mean())
    # event detection under the surprise rule: does the forecast itself leave the band, same window, same side?
    if name == "surprise":
        fc = pred.mean(0)
        truth = {s: (s * gap > width).any(1) for s in (1, -1)}
        hit = {s: (s * (fc - med) > width).any(1) for s in (1, -1)}
        tp = sum((truth[s] & hit[s]).sum() for s in truth)
        fn = sum((truth[s] & ~hit[s]).sum() for s in truth)
        fp = sum((~truth[s] & hit[s]).sum() for s in truth)
        out.update(recall=float(tp / max(tp + fn, 1)), precision=float(tp / max(tp + fp, 1)),
                   f1=float(2 * tp / max(2 * tp + fp + fn, 1)))
    return out, err


def boot_gain(e_site, e_fm, mask, seed=0):
    """Gain of FM over site on outlier steps, % ; paired bootstrap over windows."""
    rng = np.random.default_rng(seed)
    s = (e_site * mask).sum(1); f = (e_fm * mask).sum(1)
    keep = mask.sum(1) > 0
    s, f = s[keep], f[keep]
    idx = rng.integers(0, len(s), (N_BOOT, len(s)))
    g = 100 * (s[idx].sum(1) - f[idx].sum(1)) / np.maximum(s[idx].sum(1), 1e-12)
    return float(100 * (s.sum() - f.sum()) / s.sum()), *np.percentile(g, [2.5, 97.5]).tolist()


def analyze():
    rows, heads = [], []
    for trace in sw.TRACES:
        series = sw.load_trace(trace)
        for H in HORIZONS:
            ms, Y, med, width, prev, scale = masks(series, trace, H)
            persist = np.repeat(sw.build(series, windows(series, trace, H), 1, H)[0], H, 1)[None]
            models = {"Persistence": (persist, None, "reference")}
            for fam, names in FMS.items():
                for m in names:
                    p = PRED / f"{trace}_H{H}_{m}.npz"
                    if p.exists():
                        r = min((json.loads(q.read_text()) for q in sw.CELLS.glob(f"{trace}_H{H}_L*_{m}.json")),
                                key=lambda x: x["val_mase"])
                        models[m] = (np.load(p)["pred"], r["val_mase"], "fm")
            for p in PRED.glob(f"{trace}_H{H}_site_*.npz"):
                z = np.load(p)
                models[p.stem.split("_site_")[1]] = (z["pred"], float(z["val_mase"]), "site")
            errs = {}
            for name, mask in ms.items():
                for m, (pred, val, fam) in models.items():
                    if mask.sum() == 0:
                        continue
                    s, err = score(pred, Y, mask, med, width, scale, name)
                    errs[m] = err
                    rows.append(dict(trace=trace, H=H, definition=name, model=m, family=fam, val_mase=val,
                                     n_windows=len(Y), n_out=int(mask.sum()), n_out_windows=int(mask.any(1).sum()),
                                     rate=float(mask.mean()), **s))
                fm = [m for m, v in models.items() if v[2] == "fm"]
                site = [m for m, v in models.items() if v[2] == "site"]
                if not fm or not site or mask.sum() == 0:
                    continue
                bf = min(fm, key=lambda m: models[m][1])     # best of each family, chosen on overall validation error
                bs = min(site, key=lambda m: models[m][1])
                g, lo, hi = boot_gain(errs[bs], errs[bf], mask)
                gr, lor, hir = boot_gain(errs[bs], errs[bf], ~mask)
                heads.append(dict(trace=trace, H=H, definition=name, best_fm=bf, best_site=bs, n_out=int(mask.sum()),
                                  n_out_windows=int(mask.any(1).sum()), gain_out=g, lo=lo, hi=hi,
                                  verdict="fm" if lo > 0 else ("site" if hi < 0 else "tie"), gain_rest=gr,
                                  lo_rest=lor, hi_rest=hir))
    pd.DataFrame(rows).to_csv(OUT / "outliers_all.csv", index=False)
    h = pd.DataFrame(heads)
    h.to_csv(OUT / "outliers_headline.csv", index=False)
    print(h.round(2).to_string(index=False))
    for (name, H), g in h.groupby(["definition", "H"]):
        print(name, H, g.verdict.value_counts().to_dict(), "median gain", round(g.gain_out.median(), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fm", "site", "analyze"])
    ap.add_argument("--family")
    ap.add_argument("--worker", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "fm":
        run_fm(a.family)
    elif a.mode == "site":
        work_site(a.worker)
    else:
        analyze()


if __name__ == "__main__":
    main()
