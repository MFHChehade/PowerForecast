"""Timestral (Mistral AI, Oct 2026) arm of the FM sweep: univariate on all traces, plus with covariates.

Same protocol as run_fm_sweep_timesfm3.py: every (trace, horizon, context L) cell, the same test /
validation windows, MEDIAN quantile as the point forecast, context chosen later on validation.
Package installed in an isolated folder (PF_TIMESTRAL_PKGS) because it needs torch >= 2.4.
Defaults of the package's forecast() are kept (symmetric averaging, non-negative clipping), as for
TimesFM-3.0's evaluator.

    python scripts/run_fm_sweep_timestral.py --sizes 209M 123M 28M 5M     # univariate
    python scripts/run_fm_sweep_timestral.py --cov --sizes 209M           # covariates
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ.get("PF_TIMESTRAL_PKGS", ""))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_fm_sweep as sw  # noqa: E402

PARAMS = {"5M": 5, "28M": 28, "123M": 123, "209M": 209}


class Timestral:
    def __init__(self, size):
        from timestral import TimestralForecaster
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.m = TimestralForecaster.from_pretrained(f"mistralai/Timestral-{size}", device=dev)

    def predict(self, X, H, past=None, futr=None, bs=128):
        """X [N,L] raw context; past [N,L,k] past-only covariates; futr [N,L+H,f] known-ahead covariates."""
        out = []
        for i in range(0, len(X), bs):
            kw = {}
            if past is not None and past.shape[-1]:
                kw["past_covariates"] = np.transpose(past[i:i + bs], (0, 2, 1)).astype(np.float32)
            if futr is not None and futr.shape[-1]:
                kw["known_covariates"] = np.transpose(futr[i:i + bs], (0, 2, 1)).astype(np.float32)
            f = self.m.forecast(X[i:i + bs].astype(np.float32), prediction_length=H, **kw)
            out.append(f.median[:, 0, :].float().cpu().numpy())
        return np.vstack(out)


def name(size, cov=False):
    return f"timestral_{size.lower()}" + ("_cov" if cov else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov", action="store_true")
    ap.add_argument("--sizes", nargs="*", default=["209M"])
    ap.add_argument("--traces", nargs="*")
    a = ap.parse_args()
    for size in a.sizes:
        fm = Timestral(size)
        if not a.cov:
            model = name(size)
            for tr, series, H, L in sw.iter_cells(a.traces or list(sw.TRACES), sw.HORIZONS):
                if sw.done(tr, H, L, model):
                    continue
                c = sw.Cell(tr, series, H, L)
                t0 = time.time()
                p_te, p_va = fm.predict(c.Xte, H), fm.predict(c.Xva, H)
                sw.save(c, model, "timestral", p_te, p_va, time.time() - t0, {"params_m": PARAMS[size]})
            continue
        import covariates as cv
        out = sw.OUT.parent / "cov_hpo" / "fm_cells"
        out.mkdir(parents=True, exist_ok=True)
        model = name(size, cov=True)
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
                    rec = dict(trace=tr, H=H, L=L, model=model, family="timestral", arm="covariates",
                               past_covariates=cv.PAST.get(tr, []), calendar=tr in cv.CALENDAR,
                               test_mase=float(e_te.mean() / c.scale), val_mase=float(e_va.mean() / c.scale),
                               test_id=c.test_id, seconds=round(time.time() - t0, 1))
                    np.savez_compressed(p.with_suffix(".npz"), test_err=e_te, val_err=e_va)
                    p.with_suffix(".json").write_text(json.dumps(rec))
                    print(f"  {tr:9s} H{H:<3d} L{L:<5d} {model} MASE={rec['test_mase']:.3f} "
                          f"val={rec['val_mase']:.3f} ({rec['seconds']}s)", flush=True)
        del fm
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
