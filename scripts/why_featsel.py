"""Forward feature selection on the 6 datasets that have extra inputs, for both families.

One rule for everyone: start with no inputs; on each round try adding each remaining feature group, keep
the one with the lowest VALIDATION MASE if it improves by at least MIN_GAIN, otherwise stop. The test set
is only scored after selection. Groups = one per telemetry column, plus hour-of-day and day-of-week (each a
sin/cos pair) where calendar inputs exist.

Foundation models (TimesFM-3.0, Timestral-209M, Chronos-2) stay untrained; context = the one chosen on
validation in the covariate arm. Site model = the best tuned model with inputs by validation MASE, its
settings (tuned with all inputs) held fixed, 1 seed per candidate, then 3 seeds for the final set.

    python scripts/why_featsel.py fm --family tfm3
    python scripts/why_featsel.py site --worker 1
    python scripts/why_featsel.py analyze
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import covariates as cv  # noqa: E402
import run_fm_sweep as sw  # noqa: E402

OUT = sw.OUT.parent / "why" / "featsel"
CLAIMS = OUT / "claims"
for d in (OUT, CLAIMS):
    d.mkdir(parents=True, exist_ok=True)
FM_CELLS = sw.OUT.parent / "cov_hpo" / "fm_cells"
HORIZONS = [24, 6]
MIN_GAIN = 0.005  # at least 0.5% lower validation error to keep a feature
N_BOOT = 2000


def groups(trace):
    g = {c: ("past", [i]) for i, c in enumerate(cv.PAST.get(trace, []))}
    if trace in cv.CALENDAR:
        g.update(hour_of_day=("cal", [0, 1]), day_of_week=("cal", [2, 3]))
    return g


def subset(cell, trace, names):
    """A copy of the cell that carries only the chosen feature groups."""
    g = groups(trace)
    pi = [i for n in names if g[n][0] == "past" for i in g[n][1]]
    ci = sorted(i for n in names if g[n][0] == "cal" for i in g[n][1])
    c = copy.copy(cell)
    for k in ("tr", "va", "te"):
        setattr(c, "P" + k, getattr(cell, "P" + k)[:, :, pi])
        setattr(c, "F" + k, getattr(cell, "F" + k)[:, :, ci])
    c.n_past, c.n_cal = len(pi), len(ci)
    return c


def greedy(trace, evaluate):
    """evaluate(names) -> validation MASE. Returns the chosen names and the path taken."""
    all_names = list(groups(trace))
    chosen, best = [], evaluate([])
    path = [dict(round=0, added=None, val_mase=best, tried={})]
    while len(chosen) < len(all_names):
        tried = {n: evaluate(chosen + [n]) for n in all_names if n not in chosen}
        n = min(tried, key=tried.get)
        if not tried[n] < best * (1 - MIN_GAIN):
            path.append(dict(round=len(chosen) + 1, added=None, val_mase=best, tried=tried))
            break
        chosen.append(n)
        best = tried[n]
        path.append(dict(round=len(chosen), added=n, val_mase=best, tried=tried))
    return chosen, path


# ------------------------------------------------------------------------- foundation models

def fm_model(family):
    if family == "tfm3":
        import run_fm_sweep_timesfm3 as t3
        m = t3.TimesFM3()
        return "timesfm_3p0_cov", lambda c, k: m.predict(getattr(c, "X" + k), c.H, getattr(c, "P" + k), getattr(c, "F" + k))
    if family == "timestral":
        import run_fm_sweep_timestral as ts
        m = ts.Timestral("209M")
        return "timestral_209m_cov", lambda c, k: m.predict(getattr(c, "X" + k), c.H, getattr(c, "P" + k), getattr(c, "F" + k))
    if family == "chronos2":
        import torch
        from chronos import Chronos2Pipeline

        import run_cov_fm as cf
        pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda" if torch.cuda.is_available() else "cpu")

        def f(c, k):
            d = cf.inputs(c, k)
            for r in d:
                if not r["past_covariates"]:
                    del r["past_covariates"]
            q, _ = pipe.predict_quantiles(d, prediction_length=c.H, quantile_levels=[0.5], batch_size=64, cross_learning=False)
            return np.stack([qi[0, :, 0].cpu().numpy() for qi in q])
        return "chronos_2_cov", f
    raise ValueError(family)


def run_fm(family):
    model, f = fm_model(family)
    for H in HORIZONS:
        for trace in cv.COV_TRACES:
            out = OUT / f"{trace}_H{H}_{model}.json"
            if out.exists():
                continue
            recs = [json.loads(p.read_text()) for p in FM_CELLS.glob(f"{trace}_H{H}_L*_{model}.json")]
            L = min(recs, key=lambda r: r["val_mase"])["L"]
            c = cv.cov_cell(trace, H, L)
            t0 = time.time()
            val = lambda names: float(np.mean(np.abs(c.Yva - f(subset(c, trace, names), "va"))) / c.scale)
            chosen, path = greedy(trace, val)
            errs = {n: np.mean(np.abs(c.Yte - f(subset(c, trace, s), "te")), 1)
                    for n, s in dict(none=[], selected=chosen, all=list(groups(trace))).items()}
            np.savez_compressed(out.with_suffix(".npz"), **errs)
            rec = dict(trace=trace, H=H, L=L, model=model, family="fm", chosen=chosen, path=path, n_groups=len(groups(trace)),
                       test_id=c.test_id, seconds=round(time.time() - t0, 1),
                       **{f"test_{n}": float(e.mean() / c.scale) for n, e in errs.items()})
            out.write_text(json.dumps(rec))
            print(f"{model} {trace} H{H} L{L}: {chosen} none={rec['test_none']:.3f} sel={rec['test_selected']:.3f} "
                  f"all={rec['test_all']:.3f} ({rec['seconds']}s)", flush=True)


# ------------------------------------------------------------------------------- site models

def best_site(trace, H):
    import run_cov_hpo as ch
    recs = [json.loads(p.read_text()) for p in ch.FINAL.glob(f"{trace}_H{H}_*.json")]
    return min(recs, key=lambda r: r["val_mase"])


def run_site(trace, H):
    import optuna

    import run_cov_hpo as ch
    import run_dl_hpo as dl
    rec = best_site(trace, H)
    model = rec["model"]
    fixed = optuna.trial.FixedTrial(rec["params"])
    if model in ch.ML:
        L, hp = ch.ml_space(fixed, model, trace)
        fit = lambda c, s: ch.ml_fit_predict(c, model, hp, seed=s)
    else:
        L, kw, tr = ch.space(fixed, model, trace, H)
        fit = lambda c, s: ch.train_eval(c, model, kw, tr, seed=s)
    c = cv.cov_cell(trace, H, L)
    t0 = time.time()

    def val(names):
        try:
            return float(fit(subset(c, trace, names), 0)[0].mean() / c.scale)
        except Exception as ex:  # a model that cannot run on this input set (e.g. TimeXer with no inputs)
            print(f"  {model} {trace} H{H} {names}: cannot run ({ex!r})", flush=True)
            return float("inf")

    chosen, path = greedy(trace, val)
    errs = {}
    for n, s in dict(none=[], selected=chosen, all=list(groups(trace))).items():
        try:
            errs[n] = np.mean([fit(subset(c, trace, s), sd)[1] for sd in dl.FINAL_SEEDS], axis=0)
        except Exception:
            errs[n] = np.full(len(c.Yte), np.nan)
    out = OUT / f"{trace}_H{H}_site.json"
    np.savez_compressed(out.with_suffix(".npz"), **errs)
    r = dict(trace=trace, H=H, L=L, model=model, family="site", chosen=chosen, path=path, n_groups=len(groups(trace)),
             test_id=c.test_id, seconds=round(time.time() - t0, 1),
             **{f"test_{n}": float(e.mean() / c.scale) for n, e in errs.items()})
    out.write_text(json.dumps(r))
    print(f"site {model} {trace} H{H} L{L}: {chosen} none={r['test_none']:.3f} sel={r['test_selected']:.3f} "
          f"all={r['test_all']:.3f} ({r['seconds']}s)", flush=True)


def work_site(worker):
    job = os.environ.get("SLURM_JOB_ID", "local")
    for H in HORIZONS:
        for trace in cv.COV_TRACES:
            if (OUT / f"{trace}_H{H}_site.json").exists():
                continue
            d = CLAIMS / f"{trace}__H{H}"
            try:
                d.mkdir()
            except FileExistsError:
                continue
            (d / "job").write_text(job)
            try:
                run_site(trace, H)
            except Exception:
                (d / "failed").write_text(traceback.format_exc())
                print(f"[w{worker}] FAILED {trace} H{H}\n{traceback.format_exc()}", flush=True)


# ------------------------------------------------------------------------------------ results

def boot(ref, new, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ref), (N_BOOT, len(ref)))
    g = 100 * (ref[idx].mean(1) - new[idx].mean(1)) / ref[idx].mean(1)
    return float(100 * (ref.mean() - new.mean()) / ref.mean()), *np.percentile(g, [2.5, 97.5]).tolist()


def analyze():
    rows = []
    for p in sorted(OUT.glob("*.json")):
        r = json.loads(p.read_text())
        z = np.load(p.with_suffix(".npz"))
        row = dict(trace=r["trace"], H=r["H"], family=r["family"], model=r["model"], L=r["L"],
                   chosen="+".join(r["chosen"]) or "(none)", n_chosen=len(r["chosen"]), n_groups=r["n_groups"],
                   test_none=r["test_none"], test_selected=r["test_selected"], test_all=r["test_all"])
        for name, a, b in (("sel_vs_none", "none", "selected"), ("sel_vs_all", "all", "selected")):
            if np.isfinite(z[a]).all():
                g, lo, hi = boot(z[a], z[b])
                row.update({name: g, name + "_lo": lo, name + "_hi": hi})
        rows.append(row)
    d = pd.DataFrame(rows)
    # the two families head to head, each with its own selected inputs
    head = []
    for (tr, H), g in d.groupby(["trace", "H"]):
        s = g[g.family == "site"]
        if not len(s):
            continue
        es = np.load(OUT / f"{tr}_H{H}_site.npz")["selected"]
        for _, f in g[g.family == "fm"].iterrows():
            ef = np.load(OUT / f"{tr}_H{H}_{f.model}.npz")["selected"]
            x, lo, hi = boot(es, ef)
            head.append(dict(trace=tr, H=H, fm=f.model, gain=x, lo=lo, hi=hi,
                             verdict="fm" if lo > 0 else ("site" if hi < 0 else "tie")))
    d.to_csv(OUT / "featsel.csv", index=False)
    h = pd.DataFrame(head)
    h.to_csv(OUT / "featsel_headtohead.csv", index=False)
    print(d.round(3).to_string(index=False))
    print(h.round(2).to_string(index=False))


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
