"""Covariates, TimesFM-3.0 only, for the talk.

For the 6 traces that have covariates and both horizons:
  * tuned site model: best WITH covariates vs best WITHOUT (both chosen on validation)
  * TimesFM-3.0: with covariates vs without (context chosen on validation for each)
  * TimesFM-3.0 with covariates vs the best tuned site model with covariates
Gains in % with paired-bootstrap 95% CIs over the test windows (analyze_cov.boot_gain).

    python scripts/analyze_cov_tfm3.py  -> outputs/cov_hpo/covariates_tfm3.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_cov as ac  # noqa: E402
import run_fm_sweep as sw  # noqa: E402


def main():
    uni, cov = ac.finals(ac.DLF), ac.finals(ac.COV / "final")
    fmu = ac.fm_cells(sw.CELLS, {"timesfm_3p0_330m"})
    fmc = ac.fm_cells(ac.COV / "fm_cells", {"timesfm_3p0_cov"})
    E = lambda r: np.load(r.err_path)["test_err"]
    rows = []
    for tr in ac.cv.COV_TRACES:
        for H in sw.HORIZONS:
            u = uni[(uni.trace == tr) & (uni.H == H)].sort_values("val_mase").iloc[0]
            c = cov[(cov.trace == tr) & (cov.H == H)].sort_values("val_mase").iloc[0]
            a = fmu[(fmu.trace == tr) & (fmu.H == H)].iloc[0]
            b = fmc[(fmc.trace == tr) & (fmc.H == H)].iloc[0]
            row = dict(trace=tr, H=H,
                       covariates="+".join(["telemetry"] * bool(ac.cv.PAST.get(tr)) + ["calendar"] * (tr in ac.cv.CALENDAR)),
                       site_uni=u.model, site_cov=c.model)
            for key, (ref, new) in {"site_cov_gain": (u, c), "tfm3_cov_gain": (a, b), "tfm3cov_vs_sitecov": (c, b),
                                    "tfm3uni_vs_sitecov": (c, a)}.items():
                g, lo, hi = ac.boot_gain(E(ref), E(new))
                row.update({key: g, key + "_lo": lo, key + "_hi": hi})
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(ac.COV / "covariates_tfm3.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
