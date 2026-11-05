"""Mistral's Timestral (4 sizes) against TimesFM-3.0 and the best tuned site model.

Same rules as analyze_tfm3.py: each FM's context chosen on validation, best tuned model chosen on
validation, gain = (MASE_ref - MASE_new) / MASE_ref x 100 with a paired bootstrap 95% CI over the
300 test windows. Also the covariate run (timestral_209m_cov vs timesfm_3p0_cov).

    python scripts/analyze_timestral.py  -> outputs/why/timestral.csv, outputs/why/timestral_cov.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_cov as ac  # noqa: E402
import analyze_dl_hpo as ad  # noqa: E402
import run_fm_sweep as sw  # noqa: E402

SIZES = ["timestral_5m", "timestral_28m", "timestral_123m", "timestral_209m"]
TF = "timesfm_3p0_330m"
E = lambda r: np.load(r.err_path)["test_err"]


def verdict(lo, hi):
    return "better" if lo > 0 else ("worse" if hi < 0 else "tie")


def main():
    tuned = ad.load_tuned()
    fm = ac.fm_cells(sw.CELLS, set(SIZES) | {TF})
    rows = []
    for (tr, H), g in fm.groupby(["trace", "H"]):
        t = tuned[(tuned.trace == tr) & (tuned.H == H)].sort_values("val_mase").iloc[0]
        et = ad.errs(t)
        tf = g[g.model == TF].iloc[0]
        assert len(set(g.test_id)) == 1, (tr, H)
        for m in SIZES:
            r = g[g.model == m].iloc[0]
            gs, los, his = ac.boot_gain(et, E(r))
            gf, lof, hif = ac.boot_gain(E(tf), E(r))
            rows.append(dict(trace=tr, H=H, model=m, L=r.L, mase=r.test_mase, tfm3_mase=tf.test_mase,
                             best_site=t.model, site_mase=t.test_mase, vs_site=gs, vs_site_lo=los, vs_site_hi=his,
                             vs_site_v=verdict(los, his), vs_tfm3=gf, vs_tfm3_lo=lof, vs_tfm3_hi=hif,
                             vs_tfm3_v=verdict(lof, hif)))
    d = pd.DataFrame(rows)
    out = sw.OUT.parent / "why"
    d.to_csv(out / "timestral.csv", index=False)
    print(d[d.model == "timestral_209m"][["trace", "H", "L", "mase", "tfm3_mase", "site_mase", "vs_site",
                                          "vs_site_v", "vs_tfm3", "vs_tfm3_v"]].round(3).to_string(index=False))
    for m in SIZES:
        s = d[d.model == m]
        print(f"\n{m}: median vs site {s.vs_site.median():+.1f}%, vs TimesFM-3.0 {s.vs_tfm3.median():+.1f}%")
        for H in sw.HORIZONS:
            h = s[s.H == H]
            print(f"  H={H}: vs site {h.vs_site_v.value_counts().to_dict()}  vs TFM3 {h.vs_tfm3_v.value_counts().to_dict()}")

    fc = ac.fm_cells(ac.COV / "fm_cells", {"timestral_209m_cov", "timesfm_3p0_cov"})
    crow = []
    for (tr, H), g in fc.groupby(["trace", "H"]):
        if set(g.model) != {"timestral_209m_cov", "timesfm_3p0_cov"}:
            continue
        a, b = g[g.model == "timesfm_3p0_cov"].iloc[0], g[g.model == "timestral_209m_cov"].iloc[0]
        u = d[(d.trace == tr) & (d.H == H) & (d.model == "timestral_209m")].iloc[0]
        gn, lo, hi = ac.boot_gain(E(a), E(b))
        crow.append(dict(trace=tr, H=H, ts_uni=u.mase, ts_cov=b.test_mase, tfm3_cov=a.test_mase,
                         ts_cov_gain=(u.mase - b.test_mase) / u.mase * 100, vs_tfm3cov=gn, lo=lo, hi=hi,
                         v=verdict(lo, hi)))
    c = pd.DataFrame(crow)
    c.to_csv(out / "timestral_cov.csv", index=False)
    print("\nCovariates (Timestral-209M):\n", c.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
