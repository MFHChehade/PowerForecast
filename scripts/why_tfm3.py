"""H3 for TimesFM-3.0: is it PRETRAINING or the ARCHITECTURE?  Same design as why_ft.py (Chronos-Bolt).

For each (trace, horizon), at one context L* = min(512, L_max), TimesFM-3.0 (330M) is run three ways:
    zeroshot  pretrained weights, no training
    finetune  pretrained weights, trained on the trace's training windows
    scratch   the SAME architecture and configuration with random weights, trained on the same windows

Training data = exactly the tuned local models' training windows (up to 4000 from the first 60%). Loss =
the quantile (pinball) loss over the model's 9 output quantiles, in units of the trace's MASE scale.
Early stopping on validation MASE of the median. The learning rate is chosen on validation from the
same 3-point grids as Chronos-Bolt, then the chosen rate is rerun with 3 seeds. All three modes forecast
through the same differentiable `decode` path (median quantile), so they are compared like for like.

    python scripts/why_tfm3.py work --worker 1
    python scripts/why_tfm3.py status
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

sys.path.insert(0, os.environ.get("PF_TIMESFM3_PKGS", ""))
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
NAME = "tfm3"
MEDIAN = 4


class _SafeSqrt(torch.autograd.Function):
    """sqrt whose gradient is 0 (not inf) at 0. TimesFM-3.0 rescales each patch by its std; a flat patch has
    std 0, and d sqrt(x)/dx = inf there turns every weight into NaN after one step. Forward values are unchanged."""

    @staticmethod
    def forward(ctx, x):
        y = torch.sqrt(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, g):
        (y,) = ctx.saved_tensors
        return torch.where(y > 0, g / (2 * y.clamp_min(1e-30)), torch.zeros_like(g))


class _TorchShim:
    sqrt = staticmethod(_SafeSqrt.apply)

    def __getattr__(self, name):
        return getattr(torch, name)


def _patch_sqrt():
    from timesfm3.torch import util
    if not isinstance(util.torch, _TorchShim):
        util.torch = _TorchShim()


def load(init):
    from timesfm3.torch import model as M
    _patch_sqrt()
    m = M.TimesFM3Torch.from_pretrained("google/timesfm-3.0-pytorch")
    if init == "scratch":
        m = M.TimesFM3Torch(**m._hub_mixin_config)  # same architecture and config, random weights
    return m.to(DEV)


def decode(m, X, H):
    """Differentiable forecast: (B, L) -> (B, H, 9) quantiles in the data's own units."""
    from timesfm3.torch import model as M
    out = M.TimesFM3Torch.decode.__wrapped__(m, X.unsqueeze(1), horizon=H)  # (B, 1, H, 9)
    return out[:, 0]


def predict(m, X, H):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            q = decode(m, torch.tensor(X[i:i + 256], device=DEV, dtype=torch.float32), H)
            out.append(q[..., MEDIAN].float().cpu().numpy())
    m.train()
    return np.vstack(out)


def pinball(q, y, levels, scale):
    e = (y.unsqueeze(-1) - q) / scale
    return torch.mean(torch.maximum(levels * e, (levels - 1) * e))


def train(cell, mode, lr, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = load("scratch" if mode == "scratch" else "pretrained")
    m.train()
    levels = torch.tensor(m.quantiles, device=DEV, dtype=torch.float32)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2 if mode == "scratch" else 0.0)
    X = torch.tensor(cell.Xtr, device=DEV, dtype=torch.float32)
    Y = torch.tensor(cell.Ytr, device=DEV, dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)
    best, state, bad, step = np.inf, None, 0, 0
    patience = 8 if mode == "scratch" else 5
    while step < STEPS[mode]:
        for idx in torch.randperm(len(X), generator=gen).split(64):
            idx = idx.to(DEV)
            q = decode(m, X[idx], cell.H)
            loss = pinball(q.float(), Y[idx], levels, cell.scale)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); step += 1
            if step % 100 == 0:
                v = ft.mase(cell, predict(m, cell.Xva, cell.H), "va")
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
    pv, pt = predict(m, cell.Xva, cell.H), predict(m, cell.Xte, cell.H)
    del m, opt
    torch.cuda.empty_cache()
    return pv, pt


def tasks():
    return [(NAME, mode, tr, H) for H in sw.HORIZONS for tr in sw.TRACES for mode in ("zeroshot", "finetune", "scratch")]


def run(name, mode, tr, H):
    cell = ft.cell_for(tr, H)
    t0 = time.time()
    rec = dict(model=name, mode=mode, trace=tr, H=H, L=cell.L, test_id=cell.test_id)
    if mode == "zeroshot":
        m = load("pretrained")
        preds_v, preds_t, grid = [predict(m, cell.Xva, H)], [predict(m, cell.Xte, H)], {}
        del m
    else:
        lrs = GRID[mode]
        cache = OUT / "cache"
        cache.mkdir(exist_ok=True)

        def cached(cell, mode, lr, seed):
            # each (lr, seed) run is saved as it finishes, so a job killed at its time limit resumes mid-cell
            p = cache / f"{ft.tag((name, mode, tr, H))}__lr{lr}__s{seed}.npz"
            if p.exists():
                z = np.load(p)
                return z["pv"], z["pt"]
            pv, pt = train(cell, mode, lr, seed)
            np.savez(p, pv=pv, pt=pt)
            return pv, pt

        first = {lr: cached(cell, mode, lr, SEEDS[0]) for lr in lrs}
        grid = {str(lr): ft.mase(cell, first[lr][0], "va") for lr in lrs}
        lr = min(lrs, key=lambda x: grid[str(x)])           # chosen on VALIDATION
        rest = [cached(cell, mode, lr, s) for s in SEEDS[1:]]
        preds_v = [first[lr][0]] + [r[0] for r in rest]
        preds_t = [first[lr][1]] + [r[1] for r in rest]
        rec.update(lr=lr)
    te = [ft.mase(cell, p) for p in preds_t]
    va = [ft.mase(cell, p, "va") for p in preds_v]
    rec.update(grid_val_mase=grid, test_mase_seeds=te, val_mase_seeds=va, test_mase=float(np.mean(te)),
               test_mase_std=float(np.std(te, ddof=1)) if len(te) > 1 else 0.0, val_mase=float(np.mean(va)),
               seconds=round(time.time() - t0, 1))
    np.savez_compressed(OUT / f"{ft.tag((name, mode, tr, H))}.npz", pred_te=np.stack(preds_t).astype(np.float32),
                        pred_va=np.stack(preds_v).astype(np.float32))
    return rec


def work(worker):
    job = os.environ.get("SLURM_JOB_ID", "local")
    # zero-shot first, then 24-step from-scratch runs (the comparison the talk needs), then the rest
    rank = lambda t: 0 if t[1] == "zeroshot" else (1 if (t[1], t[3]) == ("scratch", 24) else
                                                   (2 if t[1] == "finetune" else 3))
    order = sorted(tasks(), key=rank)
    for t in order:
        out = OUT / f"{ft.tag(t)}.json"
        if out.exists():
            if np.isfinite(json.loads(out.read_text())["test_mase"]):
                continue
            out.unlink(missing_ok=True)  # a NaN result from before the sqrt fix: redo it
            import shutil
            shutil.rmtree(wl.CLAIMS / ("ft__" + ft.tag(t)), ignore_errors=True)
        d = wl.CLAIMS / ("ft__" + ft.tag(t))
        try:
            d.mkdir()
        except FileExistsError:
            continue
        (d / "job").write_text(job)
        try:
            rec = run(*t)
            out.write_text(json.dumps(rec))
            print(f"[w{worker}] {ft.tag(t)} MASE={rec['test_mase']:.3f}+-{rec['test_mase_std']:.3f} "
                  f"({rec['seconds']:.0f}s)", flush=True)
        except Exception:
            (d / "failed").write_text(traceback.format_exc())
            print(f"[w{worker}] FAILED {ft.tag(t)}\n{traceback.format_exc()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["work", "status", "one"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--task", nargs=3, help="MODE TRACE H")
    a = ap.parse_args()
    if a.mode == "work":
        work(a.worker)
    elif a.mode == "one":
        mode, tr, H = a.task
        print(json.dumps(run(NAME, mode, tr, int(H)))[:600])
    else:
        ts = tasks()
        done = sum((OUT / f"{ft.tag(t)}.json").exists() for t in ts)
        failed = sum((wl.CLAIMS / ("ft__" + ft.tag(t)) / "failed").exists() for t in ts)
        print(f"tasks {len(ts)}  done {done}  failed {failed}")


if __name__ == "__main__":
    main()
