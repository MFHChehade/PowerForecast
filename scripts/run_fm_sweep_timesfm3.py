"""TimesFM-3.0 (330M, Aug 2026) arm of the FM sweep: univariate on all traces, plus with covariates.

Kept separate because timesfm 3.0.2 is installed in an isolated folder (PF_TIMESFM3_PKGS) so the
TimesFM-2.5 environment of the main sweep is untouched. Uses the package's benchmark evaluator
(TimesFM3Evaluator: symmetric averaging, non-negative outputs, sorted quantiles) and the MEDIAN
quantile as the point forecast, exactly like the other foundation models.

    python scripts/run_fm_sweep_timesfm3.py            # univariate -> outputs/fm_sweep/cells
    python scripts/run_fm_sweep_timesfm3.py --cov      # covariates -> outputs/cov_hpo/fm_cells
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ.get("PF_TIMESFM3_PKGS", "vendor/timesfm3_pkgs"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import run_fm_sweep as sw  # noqa: E402

NAME, PARAMS = "timesfm_3p0_330m", 330
MEDIAN = 4  # deciles 0.1..0.9 -> index 4 = 0.5


class TimesFM3:
    def __init__(self):
        from timesfm3 import TimesFM3Evaluator
        self.m = TimesFM3Evaluator.from_pretrained("google/timesfm-3.0-pytorch", per_core_batch_size=64)

    def predict(self, X, H, past=None, futr=None):
        """X [N,L] raw context; past [N,L,k] past-only covariates; futr [N,L+H,f] known-future covariates."""
        po = None if past is None or past.shape[-1] == 0 else [p.T for p in past]
        pf = None if futr is None or futr.shape[-1] == 0 else [f.T for f in futr]
        outs = self.m.predict_batch(contexts=[x.astype(np.float32) for x in X], horizon=H,
                                    past_only_covariates=po, past_future_covariates=pf,
                                    return_quantiles=True)
        return np.stack([np.asarray(o.quantiles).reshape(H, -1)[:, MEDIAN] for o in outs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov", action="store_true")
    ap.add_argument("--traces", nargs="*")
    a = ap.parse_args()
    fm = TimesFM3()
    if not a.cov:
        for tr, series, H, L in sw.iter_cells(a.traces or list(sw.TRACES), sw.HORIZONS):
            if sw.done(tr, H, L, NAME):
                continue
            c = sw.Cell(tr, series, H, L)
            t0 = time.time()
            p_te, p_va = fm.predict(c.Xte, H), fm.predict(c.Xva, H)
            sw.save(c, NAME, "timesfm", p_te, p_va, time.time() - t0, {"params_m": PARAMS})
        return
    import covariates as cv
    out = sw.OUT.parent / "cov_hpo" / "fm_cells"
    out.mkdir(parents=True, exist_ok=True)
    model = "timesfm_3p0_cov"
    for H in sw.HORIZONS:
        for tr in a.traces or cv.COV_TRACES:
            for L in sw.CONTEXTS:
                if L > sw.TRACES[tr]["L_max"]:
                    continue
                p = out / f"{tr}_H{H}_L{L}_{model}"
                if p.with_suffix(".json").exists():
                    continue
                c = cv.cov_cell(tr, H, L)
                t0 = time.time()
                pr = {k: fm.predict(getattr(c, "X" + k), H, getattr(c, "P" + k), getattr(c, "F" + k))
                      for k in ("te", "va")}
                e_te = np.mean(np.abs(c.Yte - pr["te"]), 1)
                e_va = np.mean(np.abs(c.Yva - pr["va"]), 1)
                rec = dict(trace=tr, H=H, L=L, model=model, family="timesfm", arm="covariates",
                           past_covariates=cv.PAST.get(tr, []), calendar=tr in cv.CALENDAR,
                           test_mase=float(e_te.mean() / c.scale), val_mase=float(e_va.mean() / c.scale),
                           test_id=c.test_id, seconds=round(time.time() - t0, 1))
                np.savez_compressed(p.with_suffix(".npz"), test_err=e_te, val_err=e_va)
                p.with_suffix(".json").write_text(json.dumps(rec))
                print(f"  {tr:9s} H{H:<3d} L{L:<5d} {model} MASE={rec['test_mase']:.3f} "
                      f"val={rec['val_mase']:.3f} ({rec['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
