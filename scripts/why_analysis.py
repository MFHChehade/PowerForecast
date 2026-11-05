"""Analysis-only hypothesis tests on results already computed (no new training).

  A1  H6 context    - FM advantage when FMs are restricted to each context length
  A2  H2 shift      - how different is each trace's test period from its training period, and does
                      that predict the FM advantage?
  A3  H9 windows    - how FM and site errors differ across test windows (volatile contexts, big moves, levels outside
                      the training range)?
  A4  H9 selection  - validation-to-test degradation of the selected tuned model vs the selected FM
  A5  H8 leakage    - ALCF test windows dated against each FM's release date

    python scripts/why_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_fm_sweep as afs  # noqa: E402
import run_dl_hpo as dl  # noqa: E402
import run_fm_sweep as sw  # noqa: E402

OUT = sw.OUT.parent / "why" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
FM = list(afs.FM)
pd.set_option("display.width", 220)


def load():
    cells = afs.load_cells()
    fm = cells[cells.model.isin(FM)]
    tuned = pd.DataFrame([json.loads(p.read_text()) for p in dl.FINAL.glob("*.json")])
    return fm, tuned


def fm_err(r):
    return np.load(sw.cell_path(r.trace, r.H, r.L, r.model).with_suffix(".npz"))["test_err"]


def tuned_err(r):
    return np.load(dl.FINAL / f"{r.trace}_H{r.H}_{r.model}.npz")["test_err"]


def best_tuned(tuned, tr, H):
    return tuned[(tuned.trace == tr) & (tuned.H == H)].sort_values("val_mase").iloc[0]


def a1_context(fm, tuned):
    rows = []
    for (tr, H), g in fm.groupby(["trace", "H"]):
        t = best_tuned(tuned, tr, H)
        for L, gl in g.groupby("L"):
            f = gl.sort_values("val_mase").iloc[0]
            rows.append(dict(trace=tr, H=H, L=L, tuned=t.test_mase, tuned_L=t.L, fm=f.model, fm_mase=f.test_mase,
                             gain=(t.test_mase - f.test_mase) / t.test_mase * 100))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "a1_context.csv", index=False)
    print("\n=== A1: FM gain over best tuned model, FMs restricted to context L (median over traces) ===")
    print(d.pivot_table(index="H", columns="L", values="gain", aggfunc="median").round(1).to_string())
    print(d.pivot_table(index=["trace", "H"], columns="L", values="gain").round(1).to_string())
    print("tuned winners' chosen context:", tuned.groupby("model").L.median().to_dict() if False else
          d.drop_duplicates(["trace", "H"]).tuned_L.value_counts().to_dict())
    return d


def shift_metrics(trace):
    rows = []
    for y in dl.series(trace):
        n = len(y)
        tr, te = y[: int(0.6 * n)], y[int(0.75 * n):]
        if len(tr) < 20 or len(te) < 20:
            continue
        sd = tr.std() + 1e-12
        lag = lambda a: np.corrcoef(a[1:], a[:-1])[0, 1] if a.std() > 0 else np.nan
        rows.append(dict(w=len(y), mean_shift=abs(te.mean() - tr.mean()) / sd, sd_ratio=te.std() / sd,
                         ks=ks_2samp(tr, te).statistic, lag_change=abs(lag(te) - lag(tr)),
                         out_of_range=float(np.mean((te > tr.max()) | (te < tr.min())))))
    d = pd.DataFrame(rows)
    w = d.pop("w")
    return {k: float(np.nansum(d[k] * w) / w[d[k].notna()].sum()) for k in d}


def a2_shift(fm, tuned, hl):
    info = pd.read_csv(sw.OUT / "trace_info.csv").set_index("trace")
    rows = [dict(trace=t, lag1=info.lag1[t], **shift_metrics(t)) for t in sw.TRACES]
    s = pd.DataFrame(rows).set_index("trace")
    s.to_csv(OUT / "a2_shift.csv")
    print("\n=== A2: train -> test distribution shift per trace ===")
    print(s.round(3).to_string())
    m = hl.join(s, on="trace")
    print("Spearman with FM gain (best FM vs best tuned), n=16 (trace x horizon):")
    for k in ["lag1", "mean_shift", "sd_ratio", "ks", "lag_change", "out_of_range"]:
        rho, p = spearmanr(m[k], m.gain)
        print(f"  {k:13s} rho={rho:+.2f}  p={p:.3f}")
    return s


def a3_windows(fm, tuned, hl):
    rows = []
    for r in hl.itertuples():
        series = dl.series(r.trace)
        c = sw.Cell(r.trace, series, r.H, sw.TRACES[r.trace]["L_max"])
        e_f, e_t = fm_err(r.fm_row), tuned_err(r.t_row)
        X, Y, sc = c.Xte, c.Yte, c.scale
        tr_all = np.concatenate([y[: int(0.6 * len(y))] for y in series])
        lo, hi = np.percentile(tr_all, [1, 99])
        ctx = X[:, -64:]
        feats = dict(volatility=np.std(np.diff(ctx, axis=1), 1) / sc,
                     move=np.abs(Y.mean(1) - X[:, -1]) / sc,
                     out_of_range=((Y > hi) | (Y < lo)).mean(1))
        for k, v in feats.items():
            if k == "out_of_range":
                grp = np.where(v > 0, "any", "none")
            else:
                q = np.quantile(v, [1 / 3, 2 / 3]); grp = np.where(v <= q[0], "low", np.where(v <= q[1], "mid", "high"))
            for gname in np.unique(grp):
                mk = grp == gname
                rows.append(dict(trace=r.trace, H=r.H, feature=k, group=gname, n=int(mk.sum()),
                                 gain=(e_t[mk].mean() - e_f[mk].mean()) / e_t[mk].mean() * 100))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "a3_windows.csv", index=False)
    print("\n=== A3: FM gain (%) by window type (median over the 16 trace x horizon settings) ===")
    print(d.groupby(["feature", "group"]).gain.median().round(1).to_string())
    print(d[d.feature == "out_of_range"].pivot_table(index=["trace", "H"], columns="group", values=["gain", "n"])
          .round(1).to_string())
    return d


def a4_selection(hl):
    print("\n=== A4: validation -> test degradation of the selected model (test/val MASE) ===")
    d = hl.assign(tuned_ratio=hl.t_test / hl.t_val, fm_ratio=hl.f_test / hl.f_val)
    print(d[["trace", "H", "tuned_ratio", "fm_ratio"]].round(3).to_string(index=False))
    print(f"median tuned {d.tuned_ratio.median():.3f}   median FM {d.fm_ratio.median():.3f}")
    d.to_csv(OUT / "a4_selection.csv", index=False)


def a5_leakage(fm, tuned):
    from powerforecast.data import load_dataset
    release = {"chronos_bolt_base": "2024-11-26", "timesfm_2p5_200m": "2025-09-15", "chronos_2": "2025-10-20",
               "timesfm_3p0_330m": "2026-08-31"}
    print("\n=== A5: ALCF test periods vs FM release dates ===")
    for tr, ld in (("polaris", "alcf_polaris"), ("aurora", "alcf_aurora")):
        df, tgt, _ = load_dataset(ld)
        ts = pd.to_datetime(df.sort_values("timestamp").timestamp.values)
        n = len(ts)
        print(f"{tr}: train {ts[0].date()}..{ts[int(.6 * n)].date()}  test {ts[int(.75 * n)].date()}..{ts[-1].date()}")
        for H in sw.HORIZONS:
            c = sw.Cell(tr, dl.series(tr), H, 1024)
            t_end = np.array([ts[t] for _, t in c.pos["te"]])
            t_row = best_tuned(tuned, tr, H)
            e_t = tuned_err(t_row)
            for m, rel in release.items():
                g = fm[(fm.trace == tr) & (fm.H == H) & (fm.model == m)].sort_values("val_mase")
                if g.empty:
                    continue
                e_f = fm_err(g.iloc[0])
                post = t_end >= pd.Timestamp(rel)
                out = []
                for nm, mk in (("before", ~post), ("after", post)):
                    out.append(f"{nm} n={mk.sum():3d} gain={((e_t[mk].mean() - e_f[mk].mean()) / e_t[mk].mean() * 100) if mk.any() else float('nan'):+.1f}%")
                print(f"  H={H:2d} {m:18s} released {rel}:  " + "   ".join(out))


def main():
    fm, tuned = load()
    sel = fm.loc[fm.groupby(["trace", "H", "model"]).val_mase.idxmin()]
    rows = []
    for (tr, H), g in sel.groupby(["trace", "H"]):
        f = g.sort_values("val_mase").iloc[0]
        t = best_tuned(tuned, tr, H)
        rows.append(dict(trace=tr, H=H, gain=(t.test_mase - f.test_mase) / t.test_mase * 100, fm_row=f, t_row=t,
                         t_test=t.test_mase, t_val=t.val_mase, f_test=f.test_mase, f_val=f.val_mase))
    hl = pd.DataFrame(rows)
    a1_context(fm, tuned)
    a2_shift(fm, tuned, hl)
    a3_windows(fm, tuned, hl)
    a4_selection(hl)
    a5_leakage(fm, tuned)


if __name__ == "__main__":
    main()
