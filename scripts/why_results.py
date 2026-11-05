"""Summarise the follow-up experiments (why_local / why_ft / why_pretrain / why_synth).

Reference points on every (trace, horizon), all selected on VALIDATION:
  FM       best zero-shot foundation model (any of 8, context chosen on val)   - fm_sweep cells
  tuned    best of the 22 tuned univariate local models                        - dl_hpo/final
Gains are % by which the FM's test MASE is below the other model's (positive = FM better).

    python scripts/why_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_fm_sweep as afs  # noqa: E402
import run_dl_hpo as dl  # noqa: E402
import run_fm_sweep as sw  # noqa: E402

W = sw.OUT.parent / "why"
R = W / "results"
R.mkdir(exist_ok=True)
pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 40)


def jl(folder):
    return pd.DataFrame([json.loads(p.read_text()) for p in (W / folder).glob("*.json")])


def references():
    cells = afs.load_cells()
    fm = cells[cells.model.isin(afs.FM)]
    sel = fm.loc[fm.groupby(["trace", "H", "model"]).val_mase.idxmin()]
    best_fm = sel.loc[sel.groupby(["trace", "H"]).val_mase.idxmin()].set_index(["trace", "H"])
    tuned = pd.DataFrame([json.loads(p.read_text()) for p in dl.FINAL.glob("*.json")])
    best_t = tuned.loc[tuned.groupby(["trace", "H"]).val_mase.idxmin()].set_index(["trace", "H"])
    return fm, best_fm, best_t


def gain(ref, other):
    return (other - ref) / other * 100


def h1_learning_curves(best_fm):
    d = jl("lc")
    d = d[~d.get("skipped", False).fillna(False).astype(bool)] if "skipped" in d else d
    d["fm"] = [best_fm.loc[(r.trace, r.H)].test_mase for r in d.itertuples()]
    # best local at each N, chosen on validation among the 5 models
    b = d.loc[d.groupby(["trace", "H", "n_train"]).val_mase.idxmin()].copy()
    b["fm_gain"] = gain(b.fm, b.test_mase)
    b.to_csv(R / "h1_learning_curves_best.csv", index=False)
    d.to_csv(R / "h1_learning_curves_all.csv", index=False)
    print("\n=== H1 data scarcity: FM gain over the best local model trained on N windows ===")
    t = b.pivot_table(index=["trace", "H"], columns="n_train", values="fm_gain").round(1)
    print(t.to_string())
    big = b[b.trace.isin(["mit", "surf", "nlr"])]
    print("\nper-model MASE vs N on the three large traces (median over trace x H, relative to N=4000):")
    d4 = d[d.n_train == 4000].set_index(["trace", "H", "model"]).test_mase
    dd = d[d.trace.isin(["mit", "surf", "nlr"])].copy()
    dd["rel"] = [r.test_mase / d4.get((r.trace, r.H, r.model), np.nan) for r in dd.itertuples()]
    print(dd.pivot_table(index="model", columns="n_train", values="rel", aggfunc="median").round(3).to_string())
    return b


def h2_recent(best_fm):
    rc, base = jl("recent"), jl("preds")
    base = base.set_index(["model", "trace", "H"]).test_mase
    rc["base"] = [base.get((r.model, r.trace, r.H), np.nan) for r in rc.itertuples()]
    rc["change_pct"] = (rc.base - rc.test_mase) / rc.base * 100  # + = recent training helps
    rc.to_csv(R / "h2_recent.csv", index=False)
    print("\n=== H2 recency: % improvement from training on more recent windows (vs 0-60%) ===")
    print(rc.pivot_table(index="model", columns="variant", values="change_pct", aggfunc="median").round(2).to_string())
    print(rc.pivot_table(index=["trace", "H"], columns="variant", values="change_pct", aggfunc="median").round(1)
          .to_string())
    # does the gap to the FM close?
    best = rc.loc[rc.groupby(["trace", "H", "variant"]).val_mase.idxmin()].copy()
    best["fm_gain"] = [gain(best_fm.loc[(r.trace, r.H)].test_mase, r.test_mase) for r in best.itertuples()]
    print("FM gain over best recency-trained local (median):", best.groupby("variant").fm_gain.median().round(1).to_dict())


def h6_long_context(fm, best_fm):
    d = jl("longctx")
    d = d[~d.get("skipped", False).fillna(False).astype(bool)]
    base = jl("preds").set_index(["model", "trace", "H"])
    d["base_tuned_L"] = [base.test_mase.get((r.model, r.trace, r.H), np.nan) for r in d.itertuples()]
    d["vs_tunedL_pct"] = (d.base_tuned_L - d.test_mase) / d.base_tuned_L * 100
    L = d.variant.str.split("-").str[0].astype(int); N = d.variant.str.split("-").str[1]
    d["L"], d["N"] = L, N
    # FM restricted to the same context L
    fmL = fm.loc[fm.groupby(["trace", "H", "L"]).val_mase.idxmin()].set_index(["trace", "H", "L"]).test_mase
    d["fm_sameL_gain"] = [gain(fmL.get((r.trace, r.H, r.L), np.nan), r.test_mase) for r in d.itertuples()]
    d.to_csv(R / "h6_long_context.csv", index=False)
    print("\n=== H6 x H1: local models forced to long context (median over traces, H) ===")
    print("improvement over the same model at its tuned context (%):")
    print(d.pivot_table(index=["model", "N"], columns="L", values="vs_tunedL_pct", aggfunc="median").round(1).to_string())
    print("FM gain when FM uses the SAME context L (%):")
    print(d.pivot_table(index=["model", "N"], columns="L", values="fm_sameL_gain", aggfunc="median").round(1).to_string())


def h7_ensembles(best_fm, best_t):
    d = jl("preds")
    rows = []
    for (tr, H), g in d.groupby(["trace", "H"]):
        c = sw.Cell(tr, dl.series(tr), H, 32)
        Y, sc = c.Yte, c.scale
        P = {r.model: np.load(W / "preds" / f"preds__{r.model}__{tr}__H{H}__base.npz")["pred_te"] for r in g.itertuples()}
        g = g.sort_values("val_mase")
        m = lambda p: float(np.mean(np.abs(Y - p)) / sc)
        best = g.iloc[0].model
        row = dict(trace=tr, H=H, best_single=m(P[best][0]), best_single_seedavg=float(np.mean([m(p) for p in P[best]])),
                   seed_ensemble=m(P[best].mean(0)))
        for k in (3, 5, 10):
            top = list(g.model[:k])
            row[f"top{k}_mean"] = m(np.mean([P[x].mean(0) for x in top], 0))
            row[f"top{k}_median"] = m(np.median([P[x].mean(0) for x in top], 0))
        row["fm"] = best_fm.loc[(tr, H)].test_mase
        rows.append(row)
    e = pd.DataFrame(rows)
    for col in ["best_single_seedavg", "seed_ensemble", "top3_mean", "top5_mean", "top5_median", "top10_median"]:
        e[f"fmgain_vs_{col}"] = gain(e.fm, e[col])
    e.to_csv(R / "h7_ensembles.csv", index=False)
    print("\n=== H7 variance: do ensembles of tuned local models close the gap? FM gain (%) ===")
    print(e[["trace", "H"] + [c for c in e if c.startswith("fmgain")]].round(1).to_string(index=False))
    print("median:", e[[c for c in e if c.startswith("fmgain")]].median().round(1).to_dict())


def h3_finetune(best_fm, best_t):
    d = jl("ft")
    p = d.pivot_table(index=["trace", "H"], columns=["model", "mode"], values="test_mase")
    p[("tuned", "best")] = [best_t.loc[i].test_mase for i in p.index]
    p.to_csv(R / "h3_finetune.csv")
    print("\n=== H3 pretraining vs architecture: test MASE (L* = min(512, L_max)) ===")
    print(p.round(3).to_string())
    rel = pd.DataFrame(index=p.index)
    for mdl in ("bolt_small", "bolt_tiny"):
        rel[f"{mdl}: finetune vs zeroshot %"] = gain(p[(mdl, "finetune")], p[(mdl, "zeroshot")])
        rel[f"{mdl}: zeroshot vs scratch %"] = gain(p[(mdl, "zeroshot")], p[(mdl, "scratch")])
        rel[f"{mdl}: zeroshot vs tuned %"] = gain(p[(mdl, "zeroshot")], p[("tuned", "best")])
        rel[f"{mdl}: scratch vs tuned %"] = gain(p[(mdl, "scratch")], p[("tuned", "best")])
    rel["chronos2: finetune vs zeroshot %"] = gain(p[("chronos2", "finetune")], p[("chronos2", "zeroshot")])
    print(rel.round(1).to_string())
    print("median:", rel.median().round(1).to_dict())


def h4_pretrain(best_fm, best_t):
    rows = []
    for p in (W / "pretrain").glob("*.json"):
        for r in json.loads(p.read_text())["recs"]:
            rows.append(r)
    if not rows:
        return
    d = pd.DataFrame(rows)
    p = d.pivot_table(index=["trace", "H"], columns=["arch", "corpus"], values="test_mase")
    p[("FM", "zero-shot")] = [best_fm.loc[i].test_mase for i in p.index]
    p[("tuned", "best")] = [best_t.loc[i].test_mase for i in p.index]
    p.to_csv(R / "h4_pretrain.csv")
    print("\n=== H4 data diversity: test MASE by training corpus (L = 256) ===")
    print(p.round(3).to_string())
    for arch in ("NHITS", "PatchTST"):
        for k in [c for c in ("loo", "pool", "public", "public+ft") if (arch, c) in p]:
            g = gain(p[(arch, k)], p[(arch, "target")])
            print(f"  {arch:8s} {k:9s} vs target-only: median {g.median():+.1f}%  (better on {(g > 0).sum()}/{len(g)})")


def h5_synth():
    import why_synth as ws
    rows = []
    for n in ws.TRACES:
        o, lag = ws.oracle_mase(n)
        for p in (W / "synth").glob(f"*__{n}.json"):
            r = json.loads(p.read_text())
            rows.append(dict(synth=n, lag1=lag, oracle=o, model=r["model"], kind=r["kind"], test_mase=r["test_mase"],
                             val_mase=r["val_mase"], excess=(r["test_mase"] / o - 1) * 100))
    d = pd.DataFrame(rows)
    d.to_csv(R / "h5_synthetic_all.csv", index=False)
    b = d.loc[d.groupby(["synth", "kind"]).val_mase.idxmin()]
    w = b.pivot(index="synth", columns="kind", values=["model", "excess", "test_mase"]).loc[ws.TRACES]
    w["lag1"] = d.groupby("synth").lag1.first()
    w["fm_gain"] = gain(w[("test_mase", "fm")].astype(float), w[("test_mase", "local")].astype(float))
    w.to_csv(R / "h5_synthetic_best.csv")
    print("\n=== H5 synthetic: excess error over the Bayes oracle (%) of the val-selected best of each group ===")
    print(w.round(2).to_string())
    print(d.pivot_table(index="synth", columns="model", values="excess").loc[ws.TRACES].round(1).to_string())
    from scipy.stats import spearmanr
    ar = w.loc[[t for t in ws.TRACES if t.startswith("ar_")]]
    print("AR family: Spearman(lag1, fm_gain) =", round(spearmanr(ar.lag1, ar.fm_gain)[0], 2))


def main():
    fm, best_fm, best_t = references()
    h1_learning_curves(best_fm)
    h2_recent(best_fm)
    h6_long_context(fm, best_fm)
    h7_ensembles(best_fm, best_t)
    h3_finetune(best_fm, best_t)
    h4_pretrain(best_fm, best_t)
    h5_synth()


if __name__ == "__main__":
    main()
