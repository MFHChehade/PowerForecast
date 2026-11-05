"""H4  Is it DATA DIVERSITY?  Pretrain a standard architecture on other data, then test zero-shot.

Two standard architectures (NHITS, PatchTST; fixed mid-size configs, context L = 256) trained on:
  target   only the target trace's own 4000 training windows        (= a local model)
  loo      the training regions of the 7 OTHER data-center traces   (zero-shot on the target)
  pool     the training regions of all 8 traces                     (one global model)
  public   11 public non-data-center datasets (electricity, traffic, solar, taxi, ...), zero-shot
  public+ft  the public model fine-tuned on the target's 4000 training windows
Everything else is the shared protocol: per-window standardisation, MAE loss, AdamW, test windows
identical to all other experiments; checkpoints and fine-tuning are early-stopped on VALIDATION.
Each configuration is trained with 3 seeds.

    python scripts/why_pretrain.py fetch            # login node: download the public corpus
    python scripts/why_pretrain.py work --worker 1
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

OUT = wl.OUT / "pretrain"
CORPUS = Path(os.environ.get("PF_CORPUS", sw.OUT.parent / "why" / "corpus"))
for d in (OUT, CORPUS):
    d.mkdir(parents=True, exist_ok=True)
DEV = dl.DEV
L = 256
PUBLIC = ["electricity_15min", "monash_electricity_hourly", "monash_traffic", "solar_1h", "taxi_30min",
          "uber_tlc_hourly", "m4_hourly", "monash_kdd_cup_2018", "monash_pedestrian_counts", "wind_farms_hourly",
          "ercot", "monash_london_smart_meters"]
ARCH = {
    "NHITS": dict(mlp_units=[[512, 512]] * 3, n_blocks=[2, 2, 2], n_pool_kernel_size=[8, 4, 1],
                  n_freq_downsample=[8, 4, 1], dropout_prob_theta=0.1),
    "PatchTST": dict(encoder_layers=3, n_heads=16, hidden_size=128, linear_hidden_size=256, patch_len=16,
                     stride=8, dropout=0.1, fc_dropout=0.1),
}
STEPS = {"public": 20000, "loo": 10000, "pool": 10000, "target": 2000, "ft": 2000}
LR = {"public": 1e-3, "loo": 1e-3, "pool": 1e-3, "target": 1e-3, "ft": 1e-4}
SEEDS = [0, 1, 2]
BATCH = 256


# ------------------------------------------------------------------------------ corpora

def fetch():
    from huggingface_hub import list_repo_files, hf_hub_download
    files = list_repo_files("autogluon/chronos_datasets", repo_type="dataset")
    for name in PUBLIC:
        for f in [f for f in files if f.startswith(name + "/") and f.endswith(".parquet")]:
            hf_hub_download("autogluon/chronos_datasets", f, repo_type="dataset", local_dir=CORPUS)
        print("fetched", name, flush=True)


class Pool:
    """Sampler of (L+H) windows from a list of datasets; each dataset is a list of 1-D arrays.
    Datasets are sampled uniformly, series proportional to their number of windows."""

    def __init__(self, datasets, H, max_series=2000, seed=0):
        rng = np.random.default_rng(seed)
        self.H, self.sets = H, []
        for ds in datasets:
            ds = [np.asarray(y, np.float32) for y in ds if len(y) >= L + H + 1]
            if len(ds) > max_series:
                ds = [ds[i] for i in rng.choice(len(ds), max_series, replace=False)]
            if not ds:
                continue
            n = np.array([len(y) - L - H + 1 for y in ds], float)
            sc = np.array([max(float(np.mean(np.abs(np.diff(y)))), 1e-6) for y in ds], np.float32)
            self.sets.append((ds, n / n.sum(), sc))

    def sample(self, B, rng):
        X = np.empty((B, L), np.float32); Y = np.empty((B, self.H), np.float32); F = np.empty((B, 1), np.float32)
        which = rng.integers(0, len(self.sets), B)
        for i, k in enumerate(which):
            ds, p, sc = self.sets[k]
            j = rng.choice(len(ds), p=p)
            t = rng.integers(L, len(ds[j]) - self.H + 1)
            X[i], Y[i], F[i] = ds[j][t - L:t], ds[j][t:t + self.H], sc[j]
        return X, Y, F


_PUBLIC = None


def public_sets():
    global _PUBLIC
    if _PUBLIC is not None:
        return _PUBLIC
    import pandas as pd
    out = []
    for name in PUBLIC:
        arrs = []
        for f in sorted((CORPUS / name).rglob("*.parquet")):
            df = pd.read_parquet(f)
            col = next(c for c in ("target", "consumption_kW", "power_mw") if c in df.columns)
            for v in df[col]:
                a = np.asarray(v, np.float64)
                a = a[np.isfinite(a)]
                if len(a) > L + 30:
                    arrs.append(a[-20000:])  # cap very long series
        out.append(arrs)
    _PUBLIC = out
    return out


def trace_train(trace):
    return [y[: int(0.6 * len(y))] for y in dl.series(trace)]


def corpus(kind, target, H, seed):
    if kind == "public":
        return Pool(public_sets(), H, seed=seed)
    if kind == "loo":
        return Pool([trace_train(t) for t in sw.TRACES if t != target], H, seed=seed)
    if kind == "pool":
        return Pool([trace_train(t) for t in sw.TRACES], H, seed=seed)
    raise ValueError(kind)


# ------------------------------------------------------------------------- train / eval

def cells(H, traces):
    return {t: sw.Cell(t, dl.series(t), H, L) for t in traces}


def evaluate(net, c, k="va"):
    X = c.Xva if k == "va" else c.Xte
    mu, sd = dl.norm_stats(X, c.scale)
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            out.append(dl.forward(net, torch.tensor((X[i:i + 512] - mu[i:i + 512]) / sd[i:i + 512],
                                                    device=DEV)).cpu().numpy())
    net.train()
    return np.vstack(out) * sd + mu


def vmase(net, c):
    return float(np.mean(np.abs(c.Yva - evaluate(net, c))) / c.scale)


def train(arch, H, kind, seed, val_cells, target=None, init=None):
    """kind in {public, loo, pool}: sample from a Pool; {target, ft}: the target cell's 4000 windows."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = dl.build(arch, H, L, ARCH[arch]) if init is None else copy.deepcopy(init)
    opt = torch.optim.AdamW(net.parameters(), lr=LR[kind], weight_decay=1e-4)
    if kind in ("target", "ft"):
        c = val_cells[target]
        mu, sd = dl.norm_stats(c.Xtr, c.scale)
        Xa, Ya = (c.Xtr - mu) / sd, (c.Ytr - mu) / sd
        draw = lambda: (lambda i: (Xa[i], Ya[i]))(rng.integers(0, len(Xa), min(BATCH, len(Xa))))
    else:
        pool = corpus(kind, target, H, seed)

        def draw():
            X, Y, F = pool.sample(BATCH, rng)
            m = X.mean(1, keepdims=True); s = np.maximum(X.std(1, keepdims=True), F)
            return (X - m) / s, (Y - m) / s
    every = 100 if kind in ("target", "ft") else 1000
    best, state, bad = np.inf, None, 0
    for step in range(1, STEPS[kind] + 1):
        x, y = draw()
        loss = torch.nn.functional.l1_loss(dl.forward(net, torch.tensor(x, device=DEV)), torch.tensor(y, device=DEV))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % every == 0:
            v = np.mean([vmase(net, c) for c in val_cells.values()]) if target is None else vmase(net, val_cells[target])
            if v < best - 1e-5:
                best, bad, state = v, 0, copy.deepcopy(net.state_dict())
            else:
                bad += 1
            if bad >= (5 if kind in ("target", "ft") else 4):
                break
    net.load_state_dict(state)
    return net


# ------------------------------------------------------------------------------- tasks

def tasks():
    """public / pool: one model shared by all targets (checkpoint = best MEAN val MASE over the 8 traces);
    loo / target / public+ft: one model per target."""
    out = []
    for H in sw.HORIZONS:
        for arch in ARCH:
            out.append((arch, H, "public", "all"))
            out.append((arch, H, "pool", "all"))
            for tr in sw.TRACES:
                out += [(arch, H, k, tr) for k in ("target", "loo", "public+ft")]
    return out


def tag(t):
    return f"{t[0]}__H{t[1]}__{t[2]}__{t[3]}"


def ckpt(arch, H, kind, seed):
    return OUT / "ckpt" / f"{arch}__H{H}__{kind}__s{seed}.pt"


def run(arch, H, kind, target):
    vc = cells(H, list(sw.TRACES))
    res = {}
    t0 = time.time()
    for seed in SEEDS:
        if kind in ("public", "pool"):
            ck = ckpt(arch, H, kind, seed)
            if ck.exists():  # resume: a seed finished by an earlier (killed) job is not retrained
                net = dl.build(arch, H, L, ARCH[arch])
                net.load_state_dict(torch.load(ck, map_location=DEV))
            else:
                net = train(arch, H, kind, seed, vc)
                ck.parent.mkdir(exist_ok=True)
                torch.save(net.state_dict(), ck)
            evals = {t: evaluate(net, c, "te") for t, c in vc.items()}
        elif kind == "public+ft":
            base = dl.build(arch, H, L, ARCH[arch])
            p = ckpt(arch, H, "public", seed)
            if not p.exists():
                raise FileNotFoundError(f"{p} (public pretraining must finish first)")
            base.load_state_dict(torch.load(p, map_location=DEV))
            net = train(arch, H, "ft", seed, vc, target=target, init=base)
            evals = {target: evaluate(net, vc[target], "te")}
        else:  # target-only or leave-one-out (early stopped on the TARGET's validation windows)
            net = train(arch, H, kind if kind == "target" else "loo", seed, vc, target=target)
            evals = {target: evaluate(net, vc[target], "te")}
        for t, p in evals.items():
            res.setdefault(t, []).append(p)
        del net
        torch.cuda.empty_cache()
    recs = []
    for t, ps in res.items():
        c = vc[t]
        te = [float(np.mean(np.abs(c.Yte - p)) / c.scale) for p in ps]
        np.savez_compressed(OUT / f"{arch}__H{H}__{kind}__{t}.npz", pred_te=np.stack(ps).astype(np.float32))
        recs.append(dict(arch=arch, H=H, corpus=kind, trace=t, test_mase_seeds=te, test_mase=float(np.mean(te)),
                         test_mase_std=float(np.std(te, ddof=1)), test_id=c.test_id, L=L))
    return dict(recs=recs, seconds=round(time.time() - t0, 1))


def work(worker, kinds=None):
    job = os.environ.get("SLURM_JOB_ID", "local")
    order = sorted(tasks(), key=lambda t: {"public": 0, "pool": 1, "loo": 2, "target": 3, "public+ft": 4}[t[2]])
    for t in order:
        out = OUT / f"{tag(t)}.json"
        if out.exists() or (kinds and t[2] not in kinds):
            continue
        if t[2] == "public+ft" and not all(ckpt(t[0], t[1], "public", s).exists() for s in SEEDS):
            continue  # its pretrained checkpoints are not ready yet; a later pass picks it up
        d = wl.CLAIMS / ("pre__" + tag(t))
        try:
            d.mkdir()
        except FileExistsError:
            continue
        (d / "job").write_text(job)
        try:
            rec = run(*t)
            out.write_text(json.dumps(rec))
            print(f"[w{worker}] {tag(t)} " + " ".join(f"{r['trace']}={r['test_mase']:.3f}" for r in rec["recs"]),
                  flush=True)
        except Exception:
            (d / "failed").write_text(traceback.format_exc())
            print(f"[w{worker}] FAILED {tag(t)}\n{traceback.format_exc()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fetch", "work", "status", "corpus_stats"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--kinds", nargs="*")
    a = ap.parse_args()
    if a.mode == "fetch":
        fetch()
    elif a.mode == "work":
        for _ in range(3):  # later passes pick up fine-tuning once pretraining has finished
            work(a.worker, a.kinds)
    elif a.mode == "corpus_stats":
        for n, s in zip(PUBLIC, public_sets()):
            print(f"{n:28s} series={len(s):6d} points={sum(map(len, s)):>11,d}")
    else:
        ts = tasks()
        print(f"tasks {len(ts)}  done {sum((OUT / f'{tag(t)}.json').exists() for t in ts)}")


if __name__ == "__main__":
    main()
