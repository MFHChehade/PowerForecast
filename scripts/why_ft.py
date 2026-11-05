"""H3  Is it PRETRAINING or the ARCHITECTURE?  Foundation models zero-shot vs fine-tuned vs from scratch.

For each (trace, horizon), at one context L* = min(512, L_max):
  bolt_small / bolt_tiny  (Chronos-Bolt, 48M / 9M, T5 encoder-decoder)
      zeroshot  pretrained weights, no training
      finetune  pretrained weights, trained on the trace's training windows
      scratch   SAME architecture and config, random init, trained on the same windows
  chronos2 (120M)
      zeroshot / finetune   (Chronos2Pipeline.fit, full fine-tuning)

Training data = exactly the tuned local models' training windows (4000 windows from the first 60%),
the quantile loss the models were pretrained with, early stopping on validation MASE of the median.
Learning rate is chosen on validation from a small grid; the chosen one is rerun with 3 seeds.

    python scripts/why_ft.py work --worker 1
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import run_dl_hpo as dl  # noqa: E402
import run_fm_sweep as sw  # noqa: E402
import why_local as wl  # noqa: E402

OUT = wl.OUT / "ft"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BOLT = {"bolt_small": "amazon/chronos-bolt-small", "bolt_tiny": "amazon/chronos-bolt-tiny"}
GRID = {"finetune": [1e-5, 3e-5, 1e-4], "scratch": [1e-4, 3e-4, 1e-3]}
STEPS = {"finetune": 2000, "scratch": 5000}
C2_GRID = [1e-6, 1e-5]
SEEDS = [0, 1, 2]


def lstar(trace):
    return min(512, sw.TRACES[trace]["L_max"])


def cell_for(trace, H):
    return wl.make_cell(trace, H, lstar(trace))


def mase(cell, p, k="te"):
    Y = cell.Yte if k == "te" else cell.Yva
    return float(np.mean(np.abs(Y - p)) / cell.scale)


# ------------------------------------------------------------------------- Chronos-Bolt

def bolt_model(name, init):
    from chronos import ChronosBoltPipeline
    from chronos.chronos_bolt import ChronosBoltModelForForecasting
    pipe = ChronosBoltPipeline.from_pretrained(BOLT[name], device_map=DEV, torch_dtype=torch.float32)
    m = pipe.model
    if init == "scratch":
        m = ChronosBoltModelForForecasting(copy.deepcopy(m.config)).to(DEV)  # same architecture, random weights
    return m


def bolt_predict(m, X, H):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            q = m(context=torch.tensor(X[i:i + 256], device=DEV)).quantile_preds  # [B, 9, 64]
            out.append(q[:, 4, :H].float().cpu().numpy())                           # median
    m.train()
    return np.vstack(out)


def bolt_train(cell, name, mode, lr, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = bolt_model(name, "scratch" if mode == "scratch" else "pretrained")
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2 if mode == "scratch" else 0.0)
    X = torch.tensor(cell.Xtr, device=DEV); Y = torch.tensor(cell.Ytr, device=DEV)
    gen = torch.Generator().manual_seed(seed)
    best, state, bad, step = np.inf, None, 0, 0
    patience = 8 if mode == "scratch" else 5
    while step < STEPS[mode]:
        for idx in torch.randperm(len(X), generator=gen).split(64):
            idx = idx.to(DEV)
            loss = m(context=X[idx], target=Y[idx]).loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); step += 1
            if step % 100 == 0:
                v = mase(cell, bolt_predict(m, cell.Xva, cell.H), "va")
                if v < best - 1e-5:
                    best, bad, state = v, 0, copy.deepcopy(m.state_dict())
                else:
                    bad += 1
                if bad >= patience:
                    step = STEPS[mode]
            if step >= STEPS[mode]:
                break
    m.load_state_dict(state)
    pv, pt = bolt_predict(m, cell.Xva, cell.H), bolt_predict(m, cell.Xte, cell.H)
    del m, opt
    torch.cuda.empty_cache()
    return pv, pt


# ---------------------------------------------------------------------------- Chronos-2

def c2_pipe():
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DEV)


def c2_predict(pipe, X, H):
    q, _ = pipe.predict_quantiles([r for r in X], prediction_length=H, quantile_levels=[0.5], batch_size=256,
                                  cross_learning=False)
    return np.stack([qi[0, :, 0].cpu().numpy() for qi in q])


def c2_finetune(cell, lr, seed, workdir):
    torch.manual_seed(seed); np.random.seed(seed)
    L, H = cell.L, cell.H
    pipe = c2_pipe()
    windows = [np.concatenate([x, y]).astype(np.float32) for x, y in zip(cell.Xtr, cell.Ytr)]
    # fixed budget, no internal validation (its load-best-at-end needs checkpointing); the learning rate is
    # chosen on OUR validation windows by the caller, as for every other model
    ft = pipe.fit(inputs=windows, prediction_length=H, context_length=L, min_past=L,
                  learning_rate=lr, num_steps=1000, batch_size=128, output_dir=str(workdir), seed=seed,
                  remove_printer_callback=True, report_to="none", save_strategy="no")
    pv, pt = c2_predict(ft, cell.Xva, H), c2_predict(ft, cell.Xte, H)
    del pipe, ft
    torch.cuda.empty_cache()
    return pv, pt


# ------------------------------------------------------------------------------- tasks

def tasks():
    out = []
    for H in sw.HORIZONS:
        for tr in sw.TRACES:
            for name in ("chronos2", "bolt_small", "bolt_tiny"):
                modes = ["zeroshot", "finetune"] + (["scratch"] if name in BOLT else [])
                out += [(name, mode, tr, H) for mode in modes]
    return out


def tag(t):
    return "__".join(map(str, t[:3])) + f"__H{t[3]}"


def run(name, mode, tr, H):
    cell = cell_for(tr, H)
    t0 = time.time()
    rec = dict(model=name, mode=mode, trace=tr, H=H, L=cell.L, test_id=cell.test_id)
    if mode == "zeroshot":
        if name in BOLT:
            m = bolt_model(name, "pretrained")
            pv, pt = bolt_predict(m, cell.Xva, H), bolt_predict(m, cell.Xte, H)
        else:
            pipe = c2_pipe()
            pv, pt = c2_predict(pipe, cell.Xva, H), c2_predict(pipe, cell.Xte, H)
        preds_v, preds_t, grid = [pv], [pt], {}
    else:
        trainer = (lambda lr, s: bolt_train(cell, name, mode, lr, s)) if name in BOLT else \
            (lambda lr, s: c2_finetune(cell, lr, s, OUT / "tmp" / f"{tag((name, mode, tr, H))}_{lr}_{s}"))
        lrs = GRID[mode] if name in BOLT else C2_GRID
        first = {lr: trainer(lr, SEEDS[0]) for lr in lrs}            # seed 0 for every grid point
        grid = {str(lr): mase(cell, first[lr][0], "va") for lr in lrs}
        lr = min(lrs, key=lambda x: grid[str(x)])                     # chosen on VALIDATION
        rest = [trainer(lr, s) for s in SEEDS[1:]]
        preds_v = [first[lr][0]] + [r[0] for r in rest]
        preds_t = [first[lr][1]] + [r[1] for r in rest]
        rec.update(lr=lr)
    te = [mase(cell, p) for p in preds_t]
    va = [mase(cell, p, "va") for p in preds_v]
    rec.update(grid_val_mase=grid, test_mase_seeds=te, val_mase_seeds=va, test_mase=float(np.mean(te)),
               test_mase_std=float(np.std(te, ddof=1)) if len(te) > 1 else 0.0, val_mase=float(np.mean(va)),
               seconds=round(time.time() - t0, 1))
    np.savez_compressed(OUT / f"{tag((name, mode, tr, H))}.npz", pred_te=np.stack(preds_t).astype(np.float32),
                        pred_va=np.stack(preds_v).astype(np.float32))
    return rec


def work(worker, only=None):
    job = os.environ.get("SLURM_JOB_ID", "local")
    for t in tasks():
        out = OUT / f"{tag(t)}.json"
        if out.exists() or (only and t[0] not in only):
            continue
        d = wl.CLAIMS / ("ft__" + tag(t))
        try:
            d.mkdir()
        except FileExistsError:
            continue
        (d / "job").write_text(job)
        try:
            rec = run(*t)
            out.write_text(json.dumps(rec))
            print(f"[w{worker}] {tag(t)} MASE={rec['test_mase']:.3f}±{rec['test_mase_std']:.3f}", flush=True)
        except Exception:
            (d / "failed").write_text(traceback.format_exc())
            print(f"[w{worker}] FAILED {tag(t)}\n{traceback.format_exc()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["work", "status", "one"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--task", nargs=4)
    a = ap.parse_args()
    if a.mode == "work":
        work(a.worker, a.models)
    elif a.mode == "one":
        n, m, tr, H = a.task
        print(json.dumps(run(n, m, tr, int(H)))[:500])
    else:
        ts = tasks()
        done = sum((OUT / f"{tag(t)}.json").exists() for t in ts)
        failed = sum((wl.CLAIMS / ("ft__" + tag(t)) / "failed").exists() for t in ts)
        print(f"tasks {len(ts)}  done {done}  failed {failed}")


if __name__ == "__main__":
    main()
