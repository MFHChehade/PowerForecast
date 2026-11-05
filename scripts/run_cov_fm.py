"""Chronos-2 with exogenous covariates, zero-shot, on the same windows as everything else.

Chronos-2 takes covariates natively (group attention over target + covariate variates):
  * past covariates  -> `past_covariates` (context window only)
  * calendar         -> `past_covariates` over the context AND `future_covariates` over the horizon
Context length is a sweep dimension exactly as in run_fm_sweep.py, and is selected later on
validation. The univariate Chronos-2 cells of the FM sweep are the matching "without covariates" arm.

    python scripts/run_cov_fm.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import covariates as cv  # noqa: E402
import run_fm_sweep as sw  # noqa: E402

OUT = sw.OUT.parent / "cov_hpo" / "fm_cells"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "chronos_2_cov"


def inputs(cell, k):
    X, P, F = getattr(cell, "X" + k), getattr(cell, "P" + k), getattr(cell, "F" + k)
    L = cell.L
    out = []
    for i in range(len(X)):
        past = {f"x{j}": P[i, :, j] for j in range(cell.n_past)}
        past.update({n: F[i, :L, j] for j, n in enumerate(cv.CAL_NAMES[:cell.n_cal])})
        d = {"target": X[i], "past_covariates": past}
        if cell.n_cal:
            d["future_covariates"] = {n: F[i, L:, j] for j, n in enumerate(cv.CAL_NAMES[:cell.n_cal])}
        out.append(d)
    return out


def main():
    import torch
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2",
                                            device_map="cuda" if torch.cuda.is_available() else "cpu")
    for H in sw.HORIZONS:
        for tr in cv.COV_TRACES:
            for L in sw.CONTEXTS:
                if L > sw.TRACES[tr]["L_max"]:
                    continue
                p = OUT / f"{tr}_H{H}_L{L}_{MODEL}"
                if p.with_suffix(".json").exists():
                    continue
                c = cv.cov_cell(tr, H, L)
                t0 = time.time()
                pred = {}
                for k in ("te", "va"):
                    q, _ = pipe.predict_quantiles(inputs(c, k), prediction_length=H, quantile_levels=[0.5],
                                                  batch_size=64, cross_learning=False)
                    pred[k] = np.stack([qi[0, :, 0].cpu().numpy() for qi in q])
                e_te = np.mean(np.abs(c.Yte - pred["te"]), 1)
                e_va = np.mean(np.abs(c.Yva - pred["va"]), 1)
                rec = dict(trace=tr, H=H, L=L, model=MODEL, family="chronos_2", arm="covariates",
                           past_covariates=cv.PAST.get(tr, []), calendar=tr in cv.CALENDAR,
                           test_mase=float(e_te.mean() / c.scale), val_mase=float(e_va.mean() / c.scale),
                           test_id=c.test_id, seconds=round(time.time() - t0, 1))
                np.savez_compressed(p.with_suffix(".npz"), test_err=e_te, val_err=e_va)
                p.with_suffix(".json").write_text(json.dumps(rec))
                print(f"  {tr:9s} H{H:<3d} L{L:<5d} {MODEL} MASE={rec['test_mase']:.3f} "
                      f"val={rec['val_mase']:.3f} ({rec['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
