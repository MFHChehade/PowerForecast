"""Do exogenous covariates help — the trained baselines, and the foundation models?

Arms per (trace, horizon), every one chosen on VALIDATION and scored on the shared test windows:
  tuned_uni   best of the 22 tuned univariate models            (dl_hpo/final)
  tuned_cov   best of the 14 tuned covariate models             (cov_hpo/final)
  <fm>_uni    Chronos-2 / TimesFM-3 univariate, context selected on val   (fm_sweep/cells)
  <fm>_cov    the same model with covariates                     (cov_hpo/fm_cells)
  best_fm_uni best univariate foundation model (val-selected among all 8)

    python scripts/analyze_cov.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_fm_sweep as afs  # noqa: E402
import covariates as cv  # noqa: E402
import run_fm_sweep as sw  # noqa: E402

DLF = sw.OUT.parent / "dl_hpo" / "final"
COV = sw.OUT.parent / "cov_hpo"
FM_PAIRS = {"chronos_2": "chronos_2_cov", "timesfm_3p0_330m": "timesfm_3p0_cov"}
# univariate counterpart of each covariate model (exogenous variants map to their base model)
BASE = {"NBEATSx": "NBEATS", "TSMixerx": "TSMixer"}


def finals(folder):
    df = pd.DataFrame([json.loads(p.read_text()) for p in folder.glob("*.json")])
    df["err_path"] = [str(folder / f"{r.trace}_H{r.H}_{r.model}.npz") for r in df.itertuples()]
    return df


def fm_cells(folder, models):
    rows = [json.loads(p.read_text()) for p in folder.glob("*.json")]
    df = pd.DataFrame([r for r in rows if r["model"] in models])
    df["err_path"] = [str(folder / f"{r.trace}_H{r.H}_L{r.L}_{r.model}.npz") for r in df.itertuples()]
    return df.loc[df.groupby(["trace", "H", "model"]).val_mase.idxmin()]


def boot_gain(a, b, n=2000):
    """% by which b (new) beats a (reference), with a paired bootstrap 95% CI over test windows."""
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(a), (n, len(a)))
    g = (a[idx].mean(1) - b[idx].mean(1)) / a[idx].mean(1) * 100
    return (a.mean() - b.mean()) / a.mean() * 100, np.percentile(g, 2.5), np.percentile(g, 97.5)


def main():
    uni, cov = finals(DLF), finals(COV / "final")
    fmu = fm_cells(sw.CELLS, set(afs.FM))
    fmc = fm_cells(COV / "fm_cells", set(FM_PAIRS.values()))
    ref = fmu.groupby(["trace", "H"]).test_id.first()
    for name, d in (("tuned covariate", cov), ("FM covariate", fmc)):
        bad = d[d.apply(lambda r: r.test_id != ref[(r.trace, r.H)], axis=1)]
        print(f"{name} rows: {len(d)}  test-set mismatches: {len(bad)}")
    E = lambda r: np.load(r.err_path)["test_err"]

    rows, pairs = [], []
    for tr in cv.COV_TRACES:
        for H in sw.HORIZONS:
            u = uni[(uni.trace == tr) & (uni.H == H)].sort_values("val_mase").iloc[0]
            c = cov[(cov.trace == tr) & (cov.H == H)].sort_values("val_mase").iloc[0]
            f = fmu[(fmu.trace == tr) & (fmu.H == H)].sort_values("val_mase").iloc[0]
            g, lo, hi = boot_gain(E(u), E(c))
            row = dict(trace=tr, H=H, covariates=("+".join(["telemetry"] * bool(cv.PAST.get(tr)) +
                                                          ["calendar"] * (tr in cv.CALENDAR))),
                       tuned_uni=f"{u.model}", tuned_uni_mase=u.test_mase, tuned_uni_std=u.test_mase_std,
                       tuned_cov=f"{c.model}", tuned_cov_mase=c.test_mase, tuned_cov_std=c.test_mase_std,
                       tuned_cov_gain=g, tuned_cov_lo=lo, tuned_cov_hi=hi,
                       best_fm_uni=f.model, best_fm_uni_mase=f.test_mase)
            for m_u, m_c in FM_PAIRS.items():
                a = fmu[(fmu.trace == tr) & (fmu.H == H) & (fmu.model == m_u)]
                b = fmc[(fmc.trace == tr) & (fmc.H == H) & (fmc.model == m_c)]
                if a.empty or b.empty:
                    continue
                a, b = a.iloc[0], b.iloc[0]
                g, lo, hi = boot_gain(E(a), E(b))
                row.update({f"{m_u}_uni": a.test_mase, f"{m_u}_cov": b.test_mase, f"{m_u}_cov_gain": g,
                            f"{m_u}_cov_lo": lo, f"{m_u}_cov_hi": hi})
            # best FM-with-covariates vs best tuned-with-covariates (both val-selected)
            fc = fmc[(fmc.trace == tr) & (fmc.H == H)]
            if not fc.empty:
                fbest = fc.sort_values("val_mase").iloc[0]
                g, lo, hi = boot_gain(E(c), E(fbest))
                row.update(best_fm_cov=fbest.model, best_fm_cov_mase=fbest.test_mase,
                           fmcov_vs_tunedcov=g, fmcov_vs_tunedcov_lo=lo, fmcov_vs_tunedcov_hi=hi)
            rows.append(row)
            # same model with vs without covariates
            for r in cov[(cov.trace == tr) & (cov.H == H)].itertuples():
                base = uni[(uni.trace == tr) & (uni.H == H) & (uni.model == BASE.get(r.model, r.model))]
                if base.empty:
                    continue
                g, lo, hi = boot_gain(E(base.iloc[0]), E(r))
                pairs.append(dict(trace=tr, H=H, model=r.model, uni_mase=base.iloc[0].test_mase,
                                  cov_mase=r.test_mase, gain=g, lo=lo, hi=hi))
    t = pd.DataFrame(rows)
    p = pd.DataFrame(pairs)
    t.to_csv(COV / "covariate_headline.csv", index=False)
    p.to_csv(COV / "covariate_same_model.csv", index=False)
    pd.set_option("display.width", 250)
    show = ["trace", "H", "covariates", "tuned_uni", "tuned_uni_mase", "tuned_cov", "tuned_cov_mase", "tuned_cov_gain",
            "chronos_2_uni", "chronos_2_cov", "chronos_2_cov_gain", "timesfm_3p0_330m_uni", "timesfm_3p0_330m_cov",
            "timesfm_3p0_330m_cov_gain", "best_fm_cov", "fmcov_vs_tunedcov", "fmcov_vs_tunedcov_lo", "fmcov_vs_tunedcov_hi"]
    print(t[[c for c in show if c in t]].round(3).to_string(index=False))
    print("\n=== same model, with vs without covariates (median gain %, share improved) ===")
    s = p.groupby("model").agg(median_gain=("gain", "median"), improved=("gain", lambda g: (g > 0).mean()),
                               sig_better=("lo", lambda x: (x > 0).sum()), sig_worse=("hi", lambda x: (x < 0).sum()),
                               n=("gain", "size"))
    print(s.sort_values("median_gain", ascending=False).round(2).to_string())
    print("\n=== by trace (median over models) ===")
    print(p.groupby(["trace", "H"]).gain.median().unstack().round(2).to_string())


if __name__ == "__main__":
    main()
