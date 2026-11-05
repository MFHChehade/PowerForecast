"""Does a foundation model's forecast depend on the horizon you ask for?

For one fixed set of contexts per trace (the H=24 test windows at L* = min(512, L_max)), each model is
asked for every horizon in HS. The first k steps of each longer forecast are compared with the k-step
forecast: if the model only reads off a fixed output, they match exactly. H=6 is asked twice, so the
model's own run-to-run noise is measured too. All differences are in MASE units of the trace.

One model family per process, because the TimesFM packages conflict:
    python scripts/horizon_consistency.py --family bolt
    python scripts/horizon_consistency.py --family chronos2
    python scripts/horizon_consistency.py --family tfm25
    python scripts/horizon_consistency.py --family tfm3      (needs PF_TIMESFM3_PKGS)
    python scripts/horizon_consistency.py --family timestral (needs PF_TIMESTRAL_PKGS)
    python scripts/horizon_consistency.py --feasible         window counts per trace and horizon
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import run_fm_sweep as sw  # noqa: E402

OUT = sw.OUT.parent / "why" / "horizon_consistency"
OUT.mkdir(parents=True, exist_ok=True)
HS = [6, 12, 24, 32, 48, 64, 96, 128]
FEASIBLE = [6, 24, 32, 48, 64, 96, 128, 192, 256, 336, 512]


def models(family):
    if family == "bolt":
        return {m: sw.FM(m).predict for m in ("chronos_bolt_tiny", "chronos_bolt_mini", "chronos_bolt_small",
                                              "chronos_bolt_base")}
    if family == "chronos2":
        return {"chronos_2": sw.FM("chronos_2").predict}
    if family == "tfm25":
        return {"timesfm_2p5_200m": sw.FM("timesfm_2p5_200m").predict}
    if family == "tfm3":
        import run_fm_sweep_timesfm3 as t3
        return {t3.NAME: t3.TimesFM3().predict}
    if family == "timestral":
        import run_fm_sweep_timestral as ts
        return {ts.name(s): ts.Timestral(s).predict for s in ("5M", "28M", "123M", "209M")}
    raise ValueError(family)


def compare(preds, again, Y, scale):
    """One row per (k, H): how far the first k steps of the H-step forecast are from the k-step forecast."""
    rows = []
    noise = float(np.mean(np.abs(preds[6] - again)) / scale)
    for k in (6, 24):
        base = preds[k]
        base_mase = float(np.mean(np.abs(Y[:, :k] - base)) / scale)
        for H in HS:
            if H <= k:
                continue
            d = np.abs(preds[H][:, :k] - base) / scale
            rows.append(dict(k=k, H=H, diff_mase=float(d.mean()), diff_p95=float(np.percentile(d.mean(1), 95)),
                             diff_max=float(d.max()), exact_frac=float(np.mean(d.max(1) < 1e-4)),
                             base_test_mase=base_mase,
                             long_test_mase=float(np.mean(np.abs(Y[:, :k] - preds[H][:, :k])) / scale),
                             rerun_noise=noise))
    return rows


def feasible():
    rows = []
    for tr in sw.TRACES:
        s = sw.load_trace(tr)
        for H in FEASIBLE:
            rows.append(dict(trace=tr, H=H, L_max=sw.TRACES[tr]["L_max"], n_series=len(s),
                             n_test=len(sw.positions(s, 0.75, 1.0, H, sw.TRACES[tr]["L_max"])),
                             n_val=len(sw.positions(s, 0.60, 0.75, H, sw.TRACES[tr]["L_max"])),
                             n_train_L64=len(sw.positions(s, 0.0, 0.60, H, 64)),
                             n_series_test=len({a for a, _ in sw.positions(s, 0.75, 1.0, H, sw.TRACES[tr]["L_max"])})))
    p = OUT / "feasible.json"
    p.write_text(json.dumps(rows))
    for r in rows:
        print(r, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family")
    ap.add_argument("--feasible", action="store_true")
    a = ap.parse_args()
    if a.feasible:
        return feasible()
    for name, predict in models(a.family).items():
        for tr in sw.TRACES:
            p = OUT / f"{name}__{tr}.json"
            if p.exists():
                continue
            L = min(512, sw.TRACES[tr]["L_max"])
            c = sw.Cell(tr, sw.load_trace(tr), 24, L)
            preds = {H: np.asarray(predict(c.Xte, H)) for H in HS}
            again = np.asarray(predict(c.Xte, 6))
            rows = compare(preds, again, c.Yte, c.scale)
            for r in rows:
                r.update(model=name, trace=tr, L=L, test_id=c.test_id)
            np.savez_compressed(p.with_suffix(".npz"), **{f"H{h}": v.astype(np.float32) for h, v in preds.items()})
            p.write_text(json.dumps(rows))
            r = rows[1]  # first 6 of the 24-step forecast against the 6-step forecast
            print(f"{name:22s} {tr:10s} first-6: diff={r['diff_mase']:.4f} exact={r['exact_frac']:.2f} "
                  f"noise={r['rerun_noise']:.5f} mase6={r['base_test_mase']:.3f} from24={r['long_test_mase']:.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
