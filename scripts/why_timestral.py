"""H3 for Timestral: is it PRETRAINING or the ARCHITECTURE?  Same design as why_tfm3.py and why_ft.py.

For each (size, trace, horizon), at one context L* = min(512, L_max), Timestral is run three ways:
    zeroshot  pretrained weights, no training
    finetune  pretrained weights, trained on the trace's training windows
    scratch   the SAME architecture and configuration with random weights, trained on the same windows

Training data = the tuned local models' training windows (up to 4000 from the first 60%). Loss = the
pinball loss over the model's 21 quantiles, in units of the trace's MASE scale. Early stopping on
validation MASE of the median. The learning rate is chosen on validation from the same grids as the
other two families, then the chosen rate is rerun with 3 seeds. All modes forecast through the same
differentiable `_forecast_raw` path (no mirrored averaging), so they are compared like for like.

    python scripts/why_timestral.py work --worker 1
    python scripts/why_timestral.py status
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.environ.get("PF_TIMESTRAL_PKGS", ""))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_fm_sweep as sw  # noqa: E402
import why_ft as ft  # noqa: E402  (shared cell / MASE helpers and output folder)
import why_local as wl  # noqa: E402

OUT = ft.OUT
DEV = ft.DEV
GRID = ft.GRID            # finetune [1e-5, 3e-5, 1e-4]; scratch [1e-4, 3e-4, 1e-3]
STEPS = ft.STEPS          # finetune 2000; scratch 5000
if os.environ.get("PF_SMOKE"):  # a few steps only, to check the code path
    STEPS = {"finetune": 3, "scratch": 3}
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
SEEDS = ft.SEEDS
SIZES = ["5M", "209M", "28M", "123M"]  # the two ends first; the middle sizes fill in


def load(size, init):
    from timestral.forecaster import TimestralForecaster
    from timestral.loading import load_model
    from timestral.model import TimestralModel
    m = load_model(f"mistralai/Timestral-{size}", device=DEV)
    if init == "scratch":
        m = TimestralModel(copy.deepcopy(m.config)).to(DEV)  # same architecture and config, random weights
        m.transformer.rope.rebuild_inv_freq()
    return TimestralForecaster(m)


def decode(fc, X, H):
    """Differentiable forecast: (B, L) -> (B, H, Q) quantiles in the data's own units."""
    inputs, mask = fc._as_context(X, None)
    return fc._forecast_raw(inputs, mask, H)[:, 0]


def predict(fc, X, H):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            q = decode(fc, torch.tensor(X[i:i + 256], device=DEV, dtype=torch.float32), H)
            out.append(q[..., fc.median_index].float().cpu().numpy())
    return np.vstack(out)


def pinball(q, y, levels, scale):
    e = (y.unsqueeze(-1) - q) / scale
    return torch.mean(torch.maximum(levels * e, (levels - 1) * e))


def train(cell, size, mode, lr, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    fc = load(size, "scratch" if mode == "scratch" else "pretrained")
    m = fc.model
    levels = torch.tensor(fc.quantile_levels, device=DEV, dtype=torch.float32)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2 if mode == "scratch" else 0.0)
    X = torch.tensor(cell.Xtr, device=DEV, dtype=torch.float32)
    Y = torch.tensor(cell.Ytr, device=DEV, dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)
    best, state, bad, step = np.inf, None, 0, 0
    patience = 8 if mode == "scratch" else 5
    while step < STEPS[mode]:
        for idx in torch.randperm(len(X), generator=gen).split(64):
            idx = idx.to(DEV)
            q = decode(fc, X[idx], cell.H)
            loss = pinball(q.float(), Y[idx], levels, cell.scale)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); step += 1
            if step % 100 == 0 or STEPS[mode] < 100:
                v = ft.mase(cell, predict(fc, cell.Xva, cell.H), "va")
                if v < best - 1e-5:
                    best, bad, state = v, 0, copy.deepcopy(m.state_dict())
                else:
                    bad += 1
                if bad >= patience:
                    step = STEPS[mode]
            if step >= STEPS[mode]:
                break
    if state is not None:
        m.load_state_dict(state)
    pv, pt = predict(fc, cell.Xva, cell.H), predict(fc, cell.Xte, cell.H)
    del fc, m, opt
    torch.cuda.empty_cache()
    return pv, pt


def name(size):
    return f"timestral_{size.lower()}"


def tasks():
    # zero-shot first, then 24-step runs of the smallest and largest size; the rest resume from there
    ts = [(s, mode, tr, H) for s in SIZES for H in sw.HORIZONS for tr in sw.TRACES
          for mode in ("zeroshot", "finetune", "scratch")]
    rank = lambda t: (0 if t[1] == "zeroshot" else 1, SIZES.index(t[0]) // 2, 0 if t[3] == 24 else 1)
    return sorted(ts, key=rank)


def label(t):
    return ft.tag((name(t[0]), t[1], t[2], t[3]))


def run(size, mode, tr, H):
    cell = ft.cell_for(tr, H)
    t0 = time.time()
    rec = dict(model=name(size), mode=mode, trace=tr, H=H, L=cell.L, test_id=cell.test_id)
    if mode == "zeroshot":
        fc = load(size, "pretrained")
        preds_v, preds_t, grid = [predict(fc, cell.Xva, H)], [predict(fc, cell.Xte, H)], {}
        del fc
    else:
        lrs = GRID[mode]
        cache = OUT / ("cache_smoke" if os.environ.get("PF_SMOKE") else "cache")
        cache.mkdir(exist_ok=True)

        def cached(lr, seed):
            # each (lr, seed) run is saved as it finishes, so a job killed at its time limit resumes mid-cell
            p = cache / f"{label((size, mode, tr, H))}__lr{lr}__s{seed}.npz"
            if p.exists():
                z = np.load(p)
                return z["pv"], z["pt"]
            pv, pt = train(cell, size, mode, lr, seed)
            np.savez(p, pv=pv, pt=pt)
            return pv, pt

        first = {lr: cached(lr, SEEDS[0]) for lr in lrs}
        grid = {str(lr): ft.mase(cell, first[lr][0], "va") for lr in lrs}
        lr = min(lrs, key=lambda x: grid[str(x)])           # chosen on VALIDATION
        rest = [cached(lr, s) for s in SEEDS[1:]]
        preds_v = [first[lr][0]] + [r[0] for r in rest]
        preds_t = [first[lr][1]] + [r[1] for r in rest]
        rec.update(lr=lr)
    te = [ft.mase(cell, p) for p in preds_t]
    va = [ft.mase(cell, p, "va") for p in preds_v]
    rec.update(grid_val_mase=grid, test_mase_seeds=te, val_mase_seeds=va, test_mase=float(np.mean(te)),
               test_mase_std=float(np.std(te, ddof=1)) if len(te) > 1 else 0.0, val_mase=float(np.mean(va)),
               seconds=round(time.time() - t0, 1))
    np.savez_compressed(OUT / f"{label((size, mode, tr, H))}.npz", pred_te=np.stack(preds_t).astype(np.float32),
                        pred_va=np.stack(preds_v).astype(np.float32))
    return rec


def work(worker):
    job = os.environ.get("SLURM_JOB_ID", "local")
    for t in tasks():
        out = OUT / f"{label(t)}.json"
        if out.exists():
            continue
        d = wl.CLAIMS / ("ft__" + label(t))
        try:
            d.mkdir()
        except FileExistsError:
            continue
        (d / "job").write_text(job)
        try:
            rec = run(*t)
            out.write_text(json.dumps(rec))
            print(f"[w{worker}] {label(t)} MASE={rec['test_mase']:.3f}+-{rec['test_mase_std']:.3f} "
                  f"({rec['seconds']:.0f}s)", flush=True)
        except Exception:
            (d / "failed").write_text(traceback.format_exc())
            print(f"[w{worker}] FAILED {label(t)}\n{traceback.format_exc()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["work", "status", "one"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--task", nargs=4, help="SIZE MODE TRACE H")
    a = ap.parse_args()
    if a.mode == "work":
        work(a.worker)
    elif a.mode == "one":
        size, mode, tr, H = a.task
        print(json.dumps(run(size, mode, tr, int(H)))[:600])
    else:
        ts = tasks()
        done = sum((OUT / f"{label(t)}.json").exists() for t in ts)
        failed = sum((wl.CLAIMS / ("ft__" + label(t)) / "failed").exists() for t in ts)
        print(f"tasks {len(ts)}  done {done}  failed {failed}")


if __name__ == "__main__":
    main()
