"""TimesFM-2.0-500M arm of the foundation-model sweep.

The 500M checkpoint needs the legacy timesfm 1.x API, which conflicts with the installed 2.x
package used for TimesFM-2.5. The 1.x package lives in an isolated folder that is put first on
sys.path for this process only; everything else (windows, splits, caching) is imported from
run_fm_sweep so results are scored on exactly the same windows as every other model.

    python scripts/run_fm_sweep_timesfm2.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

TIMESFM1 = os.environ.get("PF_TIMESFM1_PKGS", "vendor/timesfm1_pkgs")
sys.path.insert(0, TIMESFM1)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import run_fm_sweep as sw  # noqa: E402

NAME, FAMILY, PARAMS = "timesfm_2p0_500m", "timesfm", 500


def main():
    import timesfm
    assert "timesfm1_pkgs" in timesfm.__file__, f"wrong timesfm on path: {timesfm.__file__}"
    sw.keep_awake()
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(backend="gpu", per_core_batch_size=64, horizon_len=128,
                                       num_layers=50, use_positional_embedding=False, context_len=2048),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch"))

    def predict(X, H):
        # freq=0 is TimesFM's "high frequency (<= daily)" token: correct for every trace here.
        _, q = tfm.forecast([r.astype(np.float32) for r in X], freq=[0] * len(X))
        return np.asarray(q)[:, :H, 5]  # index 0 = mean, 1..9 = q0.1..q0.9 -> 5 = median

    for tr, series, H, L in sw.iter_cells(list(sw.TRACES), sw.HORIZONS):
        if sw.done(tr, H, L, NAME):
            continue
        cell = sw.Cell(tr, series, H, L)
        t0 = time.time()
        try:
            p_te, p_va = predict(cell.Xte, H), predict(cell.Xva, H)
        except Exception as exc:
            print(f"  [FAIL] {tr} H{H} L{L} {NAME}: {exc}", flush=True)
            continue
        sw.save(cell, NAME, FAMILY, p_te, p_va, time.time() - t0, {"params_m": PARAMS})


if __name__ == "__main__":
    main()
