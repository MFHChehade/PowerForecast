"""Hyperparameter-optimised deep-learning (and ML) baselines on the PowerForecast sweep windows.

Architectures come from neuralforecast (Nixtla) — maintained, citable implementations — but they are
driven by THIS script's training loop on the exact windows of run_fm_sweep.Cell, so every model is
scored on the same test windows as the foundation-model sweep.

Protocol
  * Search: Optuna TPE per (model, trace, horizon), objective = VALIDATION MASE. Context length L is a
    hyperparameter. A median pruner stops clearly-bad trials early (multi-fidelity).
  * Final:  the best configuration is retrained with 3 seeds; test MASE reported as mean +- std.
  * Inputs: per-window standardisation (RevIN-style, std floored at the trace's MASE scale), applied
    identically to every model; MAE loss; AdamW; early stopping on validation MAE.

    python scripts/run_dl_hpo.py hpo   --models NHITS LSTM --trials 20
    python scripts/run_dl_hpo.py final --models NHITS LSTM
    python scripts/run_dl_hpo.py smoke
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import warnings
from pathlib import Path

NF_PKGS = os.environ.get("PF_NF_PKGS", "vendor/nf_pkgs")
sys.path.insert(0, NF_PKGS)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import optuna  # noqa: E402
import torch  # noqa: E402

import run_fm_sweep as sw  # noqa: E402

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUT = sw.OUT.parent / "dl_hpo"
STUDIES = OUT / "studies"
FINAL = OUT / "final"
for d in (OUT, STUDIES, STUDIES / "split", FINAL):
    d.mkdir(parents=True, exist_ok=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAX_STEPS, EVAL_EVERY, PATIENCE = 2000, 100, 5
FINAL_SEEDS = [0, 1, 2]
# quadratic-attention / 2-D-conv models cannot fit L=1024 x batch on an 8 GB GPU
HEAVY = {"VanillaTransformer", "Informer", "Autoformer", "FEDformer", "TimesNet"}
# reduced budget (documented in the report): context <= 256, 10 trials, <= 1000 steps
SLOWEST = {"Informer", "Autoformer", "FEDformer", "TimesNet"}


# ------------------------------------------------------------------ search spaces

def _ctx(trial, trace, model):
    cap = sw.TRACES[trace]["L_max"]
    if model in HEAVY:
        cap = min(cap, 512)
    if model in SLOWEST:  # 5-17 min per trial at long context on an 8 GB GPU; papers used L ~ 96
        cap = min(cap, 256)
    choices = [L for L in sw.CONTEXTS if L <= cap]
    return trial.suggest_categorical("L", choices)


def space(trial, model, trace, H):
    """Return (L, constructor kwargs, train kwargs) for one trial."""
    L = _ctx(trial, trace, model)
    tr = dict(lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
              batch=trial.suggest_categorical("batch", [64, 256]),
              wd=trial.suggest_float("wd", 1e-6, 1e-2, log=True))
    s = trial.suggest_categorical
    f = trial.suggest_float
    if model == "NLinear":
        kw = {}
    elif model == "DLinear":
        kw = dict(moving_avg_window=s("moving_avg_window", [5, 13, 25, 49]))
    elif model == "MLP":
        kw = dict(num_layers=s("num_layers", [1, 2, 3]), hidden_size=s("hidden_size", [128, 256, 512, 1024]))
    elif model in ("LSTM", "GRU"):
        kw = dict(encoder_n_layers=s("enc_layers", [1, 2, 3]),
                  encoder_hidden_size=s("enc_hidden", [32, 64, 128, 256]),
                  encoder_dropout=f("enc_dropout", 0.0, 0.3),
                  decoder_hidden_size=s("dec_hidden", [64, 128]), decoder_layers=s("dec_layers", [1, 2]))
    elif model == "TCN":
        kw = dict(kernel_size=s("kernel_size", [2, 3]), encoder_hidden_size=s("enc_hidden", [32, 64, 128]),
                  dilations=[1, 2, 4, 8, 16, 32], decoder_hidden_size=s("dec_hidden", [64, 128]))
    elif model == "BiTCN":
        kw = dict(hidden_size=s("hidden_size", [16, 32, 64]), dropout=f("dropout", 0.0, 0.5))
    elif model == "NBEATS":
        stacks = s("stacks", ["generic", "interpretable"])
        w = s("width", [256, 512])
        nb = s("n_blocks", [1, 2, 3])
        kw = dict(stack_types=["identity"] * 3 if stacks == "generic" else ["identity", "trend", "seasonality"],
                  n_blocks=[nb] * 3, mlp_units=[[w, w]] * 3)  # neuralforecast's N-BEATS has no dropout
    elif model == "NHITS":
        w = s("width", [256, 512])
        pools = s("pools", ["2-2-1", "4-2-1", "8-4-1", "16-8-1"])
        freqs = s("freqs", ["4-2-1", "8-4-1", "12-4-1", "24-12-1"])
        kw = dict(mlp_units=[[w, w]] * 3, n_blocks=[s("n_blocks", [1, 2])] * 3,
                  n_pool_kernel_size=[int(x) for x in pools.split("-")],
                  n_freq_downsample=[min(int(x), H) for x in freqs.split("-")],
                  dropout_prob_theta=f("dropout", 0.0, 0.3))
    elif model == "TiDE":
        kw = dict(hidden_size=s("hidden_size", [128, 256, 512]), num_encoder_layers=s("enc_layers", [1, 2]),
                  num_decoder_layers=s("dec_layers", [1, 2]), decoder_output_dim=s("dec_out", [8, 16, 32]),
                  temporal_decoder_dim=s("temp_dec", [32, 64, 128]), dropout=f("dropout", 0.0, 0.5))
    elif model == "TSMixer":
        kw = dict(n_series=1, n_block=s("n_block", [1, 2, 4]), ff_dim=s("ff_dim", [32, 64, 128]),
                  dropout=f("dropout", 0.0, 0.9), revin=True)
    elif model == "TimeMixer":
        kw = dict(n_series=1, d_model=s("d_model", [16, 32, 64]), d_ff=s("d_ff", [32, 64, 128]),
                  e_layers=s("e_layers", [1, 2, 4]), dropout=f("dropout", 0.0, 0.3),
                  down_sampling_layers=s("ds_layers", [1, 2, 3]))
    elif model == "KAN":
        kw = dict(hidden_size=s("hidden_size", [64, 128, 256, 512]), grid_size=s("grid_size", [5, 10]),
                  n_hidden_layers=s("n_hidden_layers", [1, 2]))
    elif model == "PatchTST":
        p = s("patch_len", [8, 16, 32])
        p = min(p, L // 2)
        kw = dict(encoder_layers=s("enc_layers", [1, 2, 3]), n_heads=s("n_heads", [4, 8, 16]),
                  hidden_size=s("hidden_size", [32, 64, 128]), linear_hidden_size=s("ffn", [64, 128, 256]),
                  patch_len=p, stride=max(1, p // 2), dropout=f("dropout", 0.0, 0.3),
                  fc_dropout=f("fc_dropout", 0.0, 0.3))
    elif model == "iTransformer":
        kw = dict(n_series=1, hidden_size=s("hidden_size", [64, 128, 256, 512]), n_heads=s("n_heads", [4, 8]),
                  e_layers=s("e_layers", [1, 2, 3]), d_ff=s("d_ff", [128, 512, 2048]),
                  dropout=f("dropout", 0.0, 0.3))
    elif model in ("VanillaTransformer", "Informer", "Autoformer", "FEDformer"):
        kw = dict(hidden_size=s("hidden_size", [32, 64, 128]), encoder_layers=s("enc_layers", [1, 2]),
                  dropout=f("dropout", 0.0, 0.3), conv_hidden_size=s("conv_hidden", [32, 64]))
        if model == "FEDformer":
            kw.update(n_head=8, modes=min(s("modes", [16, 32, 64]), L // 2))
        else:
            kw.update(n_head=s("n_head", [4, 8]))
    elif model == "TimesNet":
        kw = dict(hidden_size=s("hidden_size", [16, 32, 64]), conv_hidden_size=s("conv_hidden", [16, 32, 64]),
                  top_k=s("top_k", [2, 3, 5]), encoder_layers=s("enc_layers", [1, 2]),
                  dropout=f("dropout", 0.0, 0.3))
    else:
        raise ValueError(model)
    return L, kw, tr


DL_MODELS = ["NLinear", "DLinear", "MLP", "NBEATS", "NHITS", "TiDE", "TSMixer", "TimeMixer", "KAN",
             "LSTM", "GRU", "TCN", "BiTCN", "TimesNet",
             "PatchTST", "iTransformer", "VanillaTransformer", "Informer", "Autoformer", "FEDformer"]
FAMILY = {"NLinear": "linear", "DLinear": "linear",
          "MLP": "mlp", "NBEATS": "mlp", "NHITS": "mlp", "TiDE": "mlp", "TSMixer": "mlp", "TimeMixer": "mlp",
          "KAN": "mlp", "LSTM": "rnn", "GRU": "rnn", "TCN": "cnn", "BiTCN": "cnn", "TimesNet": "cnn",
          "PatchTST": "transformer", "iTransformer": "transformer", "VanillaTransformer": "transformer",
          "Informer": "transformer", "Autoformer": "transformer", "FEDformer": "transformer"}


# --------------------------------------------------------------------- model glue

def build(model, H, L, kw):
    import neuralforecast.models as M
    cls = getattr(M, model)
    net = cls(h=H, input_size=L, max_steps=1, scaler_type="identity", **kw)
    return net.to(DEV)


def forward(net, x):
    """x: [B, L] normalised windows -> [B, H] forecasts, via neuralforecast's own forward."""
    B, L = x.shape
    multi = bool(getattr(net, "MULTIVARIATE", False))
    y = x.unsqueeze(-1)  # [B, L, 1] (univariate) or [B, L, N=1] (multivariate)
    batch = dict(insample_y=y, insample_mask=torch.ones_like(y), futr_exog=None, hist_exog=None,
                 stat_exog=None)
    out = net(batch)
    if isinstance(out, dict):
        out = out.get("mean", next(iter(out.values())))
    out = out.reshape(B, net.h, -1)[..., 0] if out.dim() == 3 else out.reshape(B, net.h)
    return out


def norm_stats(X, floor):
    mu = X.mean(1, keepdims=True)
    sd = np.maximum(X.std(1, keepdims=True), floor)
    return mu, sd


def train_eval(cell, model, kw, tr, seed, trial=None, return_pred=False):
    """Train on cell.Xtr/Ytr, early-stop on validation; return (val_err, test_err) per window
    (or the (val, test) predictions themselves with return_pred=True)."""
    torch.manual_seed(seed); np.random.seed(seed)
    H, scale = cell.H, cell.scale
    m_tr, s_tr = norm_stats(cell.Xtr, scale)
    m_va, s_va = norm_stats(cell.Xva, scale)
    m_te, s_te = norm_stats(cell.Xte, scale)
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)
    xtr, ytr = T((cell.Xtr - m_tr) / s_tr), T((cell.Ytr - m_tr) / s_tr)
    xva = T((cell.Xva - m_va) / s_va)
    xte = T((cell.Xte - m_te) / s_te)

    net = build(model, H, cell.L, kw)
    opt = torch.optim.AdamW(net.parameters(), lr=tr["lr"], weight_decay=tr["wd"])
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def predict(x, mu, sd):
        net.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(x), 512):
                outs.append(forward(net, x[i:i + 512]).float().cpu().numpy())
        net.train()
        return np.vstack(outs) * sd + mu

    max_steps = 1000 if model in SLOWEST else MAX_STEPS
    best, best_state, bad, step = np.inf, None, 0, 0
    n = len(xtr)
    while step < max_steps:
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, tr["batch"]):
            idx = perm[i:i + tr["batch"]].to(DEV)
            loss = torch.nn.functional.l1_loss(forward(net, xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); step += 1
            if step % EVAL_EVERY == 0:
                p_va = predict(xva, m_va, s_va)
                v = float(np.mean(np.abs(cell.Yva - p_va)) / scale)
                if not np.isfinite(v):
                    v = 1e9
                if v < best - 1e-5:
                    best, bad = v, 0
                    best_state = copy.deepcopy(net.state_dict())
                else:
                    bad += 1
                if trial is not None:
                    trial.report(v, step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                if bad >= PATIENCE:
                    step = max_steps
            if step >= max_steps:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    p_va = predict(xva, m_va, s_va)
    p_te = predict(xte, m_te, s_te)
    if return_pred:
        del net, opt
        torch.cuda.empty_cache()
        return p_va, p_te
    err_va = np.mean(np.abs(cell.Yva - p_va), axis=1)
    err_te = np.mean(np.abs(cell.Yte - p_te), axis=1)
    del net, opt
    torch.cuda.empty_cache()
    return err_va, err_te


# ------------------------------------------------------------------- HPO + final

_SERIES = {}


def series(trace):
    if trace not in _SERIES:
        _SERIES[trace] = sw.load_trace(trace)
    return _SERIES[trace]


_CELLS = {}


def cell_for(trace, H, L):
    key = (trace, H, L)
    if key not in _CELLS:
        if len(_CELLS) > 6:
            _CELLS.clear()
        _CELLS[key] = sw.Cell(trace, series(trace), H, L)
    return _CELLS[key]


def study_name(model, trace, H):
    return f"{model}__{trace}__H{H}"


def storage(model, trace=None, H=None):
    # one database per model: parallel workers never write to the same SQLite file.
    # PF_STUDY_SPLIT=1 (cluster runs): one database per (model, trace, horizon) study instead.
    if os.environ.get("PF_STUDY_SPLIT") == "1" and trace is not None:
        return f"sqlite:///{(STUDIES / 'split' / f'{study_name(model, trace, H)}.db').as_posix()}"
    return f"sqlite:///{(STUDIES / f'{model}.db').as_posix()}"


def n_trials(model, default=20):
    # laptop budget: 10 trials for the slowest models; PF_SLOW_TRIALS raises it on the cluster
    return int(os.environ.get("PF_SLOW_TRIALS", 10)) if model in SLOWEST else default


def run_hpo(model, trace, H, n_trials):
    st = optuna.create_study(study_name=study_name(model, trace, H), storage=storage(model, trace, H), load_if_exists=True,
                             direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=sw.stable_seed(model, trace, H) % 2**31,
                                                                n_startup_trials=6, multivariate=True),
                             pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=300))
    done = len([t for t in st.trials if t.state in (optuna.trial.TrialState.COMPLETE,
                                                     optuna.trial.TrialState.PRUNED,
                                                     optuna.trial.TrialState.FAIL)])
    remaining = n_trials - done
    if remaining <= 0:
        return st

    def objective(trial):
        L, kw, tr = space(trial, model, trace, H)
        t0 = time.time()
        try:
            err_va, _ = train_eval(cell_for(trace, H, L), model, kw, tr, seed=0, trial=trial)
        except optuna.TrialPruned:
            raise
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()
        v = float(err_va.mean() / cell_for(trace, H, L).scale)
        trial.set_user_attr("seconds", round(time.time() - t0, 1))
        return v

    st.optimize(objective, n_trials=remaining, catch=(RuntimeError, ValueError))
    try:
        b = st.best_trial
        print(f"  [HPO] {model:18s} {trace:9s} H{H:<3d} best val MASE={b.value:.3f}  L={b.params['L']}  "
              f"({len(st.trials)} trials)", flush=True)
    except ValueError:
        print(f"  [HPO] {model:18s} {trace:9s} H{H:<3d} no successful trial", flush=True)
    return st


def run_final(model, trace, H):
    out = FINAL / f"{trace}_H{H}_{model}.json"
    if out.exists():
        return
    st = optuna.load_study(study_name=study_name(model, trace, H), storage=storage(model, trace, H))
    best = st.best_trial
    fixed = optuna.trial.FixedTrial(best.params)
    L, kw, tr = space(fixed, model, trace, H)
    cell = cell_for(trace, H, L)
    t0 = time.time()
    va, te = [], []
    for sd in FINAL_SEEDS:
        e_va, e_te = train_eval(cell, model, kw, tr, seed=sd)
        va.append(e_va); te.append(e_te)
    te_mase = [float(e.mean() / cell.scale) for e in te]
    va_mase = [float(e.mean() / cell.scale) for e in va]
    rec = dict(trace=trace, H=H, L=L, model=model, family=FAMILY.get(model, "ml"), params=best.params,
               test_mase_seeds=te_mase, val_mase_seeds=va_mase,
               test_mase=float(np.mean(te_mase)), test_mase_std=float(np.std(te_mase, ddof=1)),
               val_mase=float(np.mean(va_mase)), n_trials=len(st.trials), test_id=cell.test_id,
               seconds=round(time.time() - t0, 1))
    np.savez_compressed(out.with_suffix(".npz"), test_err=np.mean(te, axis=0), val_err=np.mean(va, axis=0),
                        test_err_seeds=np.stack(te))
    out.write_text(json.dumps(rec))
    print(f"  [FINAL] {model:18s} {trace:9s} H{H:<3d} L={L:<5d} test MASE={rec['test_mase']:.3f} "
          f"± {rec['test_mase_std']:.3f}  seeds={[round(x, 3) for x in te_mase]}", flush=True)


def smoke():
    """One short training run per model on a small trace: catches API / shape errors before the sweep."""
    global MAX_STEPS
    MAX_STEPS = 100
    cell = sw.Cell("aurora", series("aurora"), 24, 128)
    for m in DL_MODELS:
        trial = optuna.trial.FixedTrial({})
        try:
            st = optuna.create_study(direction="minimize")
            tr = st.ask()
            L, kw, trk = space(tr, m, "aurora", 24)
            c = cell_for("aurora", 24, L)
            t0 = time.time()
            e_va, e_te = train_eval(c, m, kw, trk, seed=0)
            n_par = sum(p.numel() for p in build(m, 24, L, kw).parameters())
            print(f"  OK   {m:18s} L={L:<5d} params={n_par:>9,d} val={e_va.mean() / c.scale:.3f} "
                  f"({time.time() - t0:.1f}s / 100 steps)", flush=True)
        except Exception as exc:
            print(f"  FAIL {m:18s} {type(exc).__name__}: {str(exc)[:160]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "hpo", "final"])
    ap.add_argument("--models", nargs="*", default=DL_MODELS)
    ap.add_argument("--traces", nargs="*", default=list(sw.TRACES))
    ap.add_argument("--horizons", nargs="*", type=int, default=sw.HORIZONS)
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()
    sw.keep_awake()
    if args.mode == "smoke":
        smoke()
        return
    for H in args.horizons:
        for tr in args.traces:
            for m in args.models:
                try:
                    if args.mode == "hpo":
                        run_hpo(m, tr, H, n_trials(m, args.trials))
                    else:
                        run_final(m, tr, H)
                except Exception as exc:
                    print(f"  [ERROR] {args.mode} {m} {tr} H{H}: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
