"""Tuned baselines WITH exogenous covariates (the covariate arm of the tuned-baseline study).

Identical protocol to run_dl_hpo.py / run_ml_hpo.py (same windows, Optuna TPE on validation MASE,
20 trials, median pruning, 3 final seeds), but every model also receives
  * past covariates (telemetry) over the context window -> neuralforecast `hist_exog`, and
  * calendar features over context + horizon            -> neuralforecast `futr_exog`
(see covariates.py). Only models whose neuralforecast implementation accepts past covariates are in
this arm; models that accept only future inputs (the Transformer family, TimesNet, iTransformer) and
models with no exogenous support (NLinear, DLinear, N-BEATS, TSMixer, TimeMixer, PatchTST) are not.
Exogenous variants stand in for the latter where they exist: NBEATSx for N-BEATS, TSMixerx for TSMixer.
TimeXer and TFT are added as the two Transformers designed for covariates.

    python scripts/run_cov_hpo.py smoke
    python scripts/run_cov_hpo.py work --worker 1      # cluster: claim (model, trace, H) tasks
    python scripts/run_cov_hpo.py status
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import optuna  # noqa: E402
import torch  # noqa: E402

import covariates as cv  # noqa: E402
import run_dl_hpo as dl  # noqa: E402
import run_fm_sweep as sw  # noqa: E402

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT = sw.OUT.parent / "cov_hpo"
STUDIES, FINAL, CLAIMS = OUT / "studies", OUT / "final", OUT / "claims"
for d in (OUT, STUDIES, FINAL, CLAIMS, OUT / "logs"):
    d.mkdir(parents=True, exist_ok=True)

DEV = dl.DEV
N_TRIALS = {"LightGBM": 15}  # everything else: 20, as in the univariate arm
ML = ["Ridge", "LightGBM"]
DL = ["MLP", "NHITS", "TiDE", "KAN", "LSTM", "GRU", "TCN", "BiTCN", "NBEATSx", "TSMixerx", "TimeXer", "TFT"]
COV_MODELS = ML + DL
HEAVY = {"TFT", "TimeXer"}  # attention over the context: context capped at 512, like the univariate arm


def caps(model):
    import neuralforecast.models as M
    c = getattr(M, model)
    return bool(c.EXOGENOUS_HIST), bool(c.EXOGENOUS_FUTR), bool(getattr(c, "MULTIVARIATE", False))


# ------------------------------------------------------------------ search spaces

def space(trial, model, trace, H):
    if model in ("NBEATSx", "TSMixerx"):
        L, kw, tr = dl.space(trial, model[:-1], trace, H)  # same search space as the base model
        return L, kw, tr
    if model in dl.DL_MODELS:
        return dl.space(trial, model, trace, H)
    cap = min(sw.TRACES[trace]["L_max"], 512 if model in HEAVY else 10 ** 9)
    L = trial.suggest_categorical("L", [x for x in sw.CONTEXTS if x <= cap])
    tr = dict(lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
              batch=trial.suggest_categorical("batch", [64, 256]),
              wd=trial.suggest_float("wd", 1e-6, 1e-2, log=True))
    s, f = trial.suggest_categorical, trial.suggest_float
    if model == "TimeXer":
        p = min(s("patch_len", [8, 16, 32]), L // 2)
        kw = dict(n_series=1, patch_len=p, hidden_size=s("hidden_size", [64, 128, 256]),
                  n_heads=s("n_heads", [4, 8]), e_layers=s("e_layers", [1, 2]), d_ff=s("d_ff", [128, 256, 512]),
                  dropout=f("dropout", 0.0, 0.3))
    elif model == "TFT":
        kw = dict(hidden_size=s("hidden_size", [32, 64, 128]), n_head=s("n_head", [1, 4]),
                  dropout=f("dropout", 0.0, 0.3), attn_dropout=f("attn_dropout", 0.0, 0.2),
                  n_rnn_layers=s("n_rnn_layers", [1, 2]))
    else:
        raise ValueError(model)
    return L, kw, tr


def ml_space(trial, model, trace):
    import run_ml_hpo as ml
    return ml.space(trial, model, trace)


# --------------------------------------------------------------------- model glue

def exog_plan(model, cell):
    """Which covariates go where for this model on this trace: (n_hist_from_past, cal_as_hist, cal_as_futr)."""
    hist_ok, futr_ok, _ = caps(model)
    use_cal_futr = futr_ok and cell.n_cal > 0
    use_cal_hist = (not futr_ok) and hist_ok and cell.n_cal > 0  # past part of the calendar only
    n_past = cell.n_past if hist_ok else 0
    return n_past, use_cal_hist, use_cal_futr


def build(model, H, L, kw, n_hist, n_futr):
    import neuralforecast.models as M
    extra = {}
    if n_hist:
        extra["hist_exog_list"] = [f"x{i}" for i in range(n_hist)]
    if n_futr:
        extra["futr_exog_list"] = [f"f{i}" for i in range(n_futr)]
    net = getattr(M, model)(h=H, input_size=L, max_steps=1, scaler_type="identity", **kw, **extra)
    return net.to(DEV)


def forward(net, x, hist, futr):
    """x [B,L]; hist [B,L,X] or None; futr [B,L+H,F] or None -> [B,H]."""
    B = x.shape[0]
    multi = bool(getattr(net, "MULTIVARIATE", False))
    y = x.unsqueeze(-1)
    if multi:  # neuralforecast multivariate layout: exog [B, C, T, N]
        hist = None if hist is None else hist.permute(0, 2, 1).unsqueeze(-1)
        futr = None if futr is None else futr.permute(0, 2, 1).unsqueeze(-1)
    batch = dict(insample_y=y, insample_mask=torch.ones_like(y), hist_exog=hist, futr_exog=futr, stat_exog=None)
    out = net(batch)
    if isinstance(out, dict):
        out = out.get("mean", next(iter(out.values())))
    return out.reshape(B, net.h, -1)[..., 0] if out.dim() == 3 else out.reshape(B, net.h)


def train_eval(cell, model, kw, tr, seed, trial=None):
    torch.manual_seed(seed); np.random.seed(seed)
    H, scale = cell.H, cell.scale
    n_past, cal_hist, cal_futr = exog_plan(model, cell)
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=DEV)

    def pack(k):
        X = getattr(cell, "X" + k)
        mu, sd = dl.norm_stats(X, scale)
        P, F = getattr(cell, "P" + k), getattr(cell, "F" + k)
        hist = []
        if n_past:
            hist.append(P)
        if cal_hist:
            hist.append(F[:, :cell.L])
        hist = T(np.concatenate(hist, -1)) if hist else None
        futr = T(F) if cal_futr else None
        return T((X - mu) / sd), hist, futr, mu, sd

    xtr, htr, ftr, m_tr, s_tr = pack("tr")
    ytr = T((cell.Ytr - m_tr) / s_tr)
    xva, hva, fva, m_va, s_va = pack("va")
    xte, hte, fte, m_te, s_te = pack("te")
    n_hist = (n_past + (cell.n_cal if cal_hist else 0))
    net = build(model, H, cell.L, kw, n_hist, cell.n_cal if cal_futr else 0)
    opt = torch.optim.AdamW(net.parameters(), lr=tr["lr"], weight_decay=tr["wd"])
    gen = torch.Generator(device="cpu").manual_seed(seed)
    sub = lambda a, i: None if a is None else a[i]

    def predict(x, h, f, mu, sd):
        net.eval(); outs = []
        with torch.no_grad():
            for i in range(0, len(x), 512):
                sl = slice(i, i + 512)
                outs.append(forward(net, x[sl], sub(h, sl), sub(f, sl)).float().cpu().numpy())
        net.train()
        return np.vstack(outs) * sd + mu

    best, best_state, bad, step, n = np.inf, None, 0, 0, len(xtr)
    while step < dl.MAX_STEPS:
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, tr["batch"]):
            idx = perm[i:i + tr["batch"]].to(DEV)
            loss = torch.nn.functional.l1_loss(forward(net, xtr[idx], sub(htr, idx), sub(ftr, idx)), ytr[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); step += 1
            if step % dl.EVAL_EVERY == 0:
                v = float(np.mean(np.abs(cell.Yva - predict(xva, hva, fva, m_va, s_va))) / scale)
                v = v if np.isfinite(v) else 1e9
                if v < best - 1e-5:
                    best, bad, best_state = v, 0, copy.deepcopy(net.state_dict())
                else:
                    bad += 1
                if trial is not None:
                    trial.report(v, step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                if bad >= dl.PATIENCE:
                    step = dl.MAX_STEPS
            if step >= dl.MAX_STEPS:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    e_va = np.mean(np.abs(cell.Yva - predict(xva, hva, fva, m_va, s_va)), 1)
    e_te = np.mean(np.abs(cell.Yte - predict(xte, hte, fte, m_te, s_te)), 1)
    del net, opt
    torch.cuda.empty_cache()
    return e_va, e_te


def ml_features(cell, k, lags=24):
    """Normalised target context + last `lags` steps and context mean of every past covariate."""
    X = getattr(cell, "X" + k)
    mu, sd = dl.norm_stats(X, cell.scale)
    P = getattr(cell, "P" + k)
    feats = [(X - mu) / sd]
    if cell.n_past:
        w = min(lags, cell.L)
        feats += [P[:, -w:].reshape(len(P), -1), P.mean(1)]
    return np.concatenate(feats, 1), mu, sd


def ml_fit_predict(cell, model, hp, seed):
    F = {k: getattr(cell, "F" + k) for k in ("tr", "va", "te")}
    (Xtr, mtr, str_), (Xva, mva, sva), (Xte, mte, ste) = (ml_features(cell, k) for k in ("tr", "va", "te"))
    Ytr = (cell.Ytr - mtr) / str_
    L, H = cell.L, cell.H
    if model == "Ridge":
        from sklearn.linear_model import Ridge
        cal = (lambda k: F[k][:, L]) if cell.n_cal else (lambda k: np.zeros((len(F[k]), 0)))  # calendar at origin
        m = Ridge(alpha=hp["alpha"]).fit(np.concatenate([Xtr, cal("tr")], 1), Ytr)
        p_va = m.predict(np.concatenate([Xva, cal("va")], 1))
        p_te = m.predict(np.concatenate([Xte, cal("te")], 1))
    else:
        import lightgbm as lgb
        p_va, p_te = np.empty((len(Xva), H)), np.empty((len(Xte), H))
        for h in range(H):  # one model per step, each with the calendar of ITS target step
            cal = (lambda k: F[k][:, L + h]) if cell.n_cal else (lambda k: np.zeros((len(F[k]), 0)))
            m = lgb.LGBMRegressor(random_state=seed, subsample=0.8, subsample_freq=1, verbose=-1,
                                  n_jobs=int(os.environ.get("PF_NJOBS", -1)), **hp)
            m.fit(np.concatenate([Xtr, cal("tr")], 1), Ytr[:, h])
            p_va[:, h] = m.predict(np.concatenate([Xva, cal("va")], 1))
            p_te[:, h] = m.predict(np.concatenate([Xte, cal("te")], 1))
    p_va, p_te = p_va * sva + mva, p_te * ste + mte
    return np.mean(np.abs(cell.Yva - p_va), 1), np.mean(np.abs(cell.Yte - p_te), 1)


# ------------------------------------------------------------------- HPO + final

def storage(model, trace, H):
    return f"sqlite:///{(STUDIES / f'{dl.study_name(model, trace, H)}.db').as_posix()}"


def eligible(model, trace):
    cell_past = bool(cv.PAST.get(trace)); cell_cal = trace in cv.CALENDAR
    if model in ML:
        return cell_past or cell_cal
    hist_ok, futr_ok, _ = caps(model)
    return (hist_ok and (cell_past or cell_cal)) or (futr_ok and cell_cal)


def run_hpo(model, trace, H):
    n_trials = N_TRIALS.get(model, 20)
    st = optuna.create_study(study_name=dl.study_name(model, trace, H), storage=storage(model, trace, H),
                             load_if_exists=True, direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=sw.stable_seed(model, trace, H, "cov") % 2**31,
                                                                n_startup_trials=6, multivariate=True),
                             pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=300))
    done = [t for t in st.trials if t.state.name in ("COMPLETE", "PRUNED", "FAIL")]
    if n_trials - len(done) <= 0:
        return

    def objective(trial):
        t0 = time.time()
        if model in ML:
            L, hp = ml_space(trial, model, trace)
            c = cv.cov_cell(trace, H, L)
            e_va, _ = ml_fit_predict(c, model, hp, seed=0)
        else:
            L, kw, tr = space(trial, model, trace, H)
            c = cv.cov_cell(trace, H, L)
            try:
                e_va, _ = train_eval(c, model, kw, tr, seed=0, trial=trial)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()
        trial.set_user_attr("seconds", round(time.time() - t0, 1))
        return float(e_va.mean() / c.scale)

    st.optimize(objective, n_trials=n_trials - len(done), catch=(RuntimeError, ValueError))
    b = st.best_trial
    print(f"  [HPO] {model:10s} {trace:9s} H{H:<3d} best val MASE={b.value:.3f} L={b.params['L']}", flush=True)


def run_final(model, trace, H):
    out = FINAL / f"{trace}_H{H}_{model}.json"
    if out.exists():
        return
    st = optuna.load_study(study_name=dl.study_name(model, trace, H), storage=storage(model, trace, H))
    best = st.best_trial
    fixed = optuna.trial.FixedTrial(best.params)
    t0 = time.time()
    if model in ML:
        L, hp = ml_space(fixed, model, trace)
        c = cv.cov_cell(trace, H, L)
        va, te = zip(*[ml_fit_predict(c, model, hp, seed=s) for s in dl.FINAL_SEEDS])
    else:
        L, kw, tr = space(fixed, model, trace, H)
        c = cv.cov_cell(trace, H, L)
        va, te = zip(*[train_eval(c, model, kw, tr, seed=s) for s in dl.FINAL_SEEDS])
    te_m = [float(e.mean() / c.scale) for e in te]
    va_m = [float(e.mean() / c.scale) for e in va]
    plan = ("ml", c.n_past, c.n_cal) if model in ML else exog_plan(model, c)
    rec = dict(trace=trace, H=H, L=L, model=model, arm="covariates", params=best.params, exog_plan=list(plan),
               past_covariates=cv.PAST.get(trace, []), calendar=trace in cv.CALENDAR,
               test_mase_seeds=te_m, val_mase_seeds=va_m, test_mase=float(np.mean(te_m)),
               test_mase_std=float(np.std(te_m, ddof=1)), val_mase=float(np.mean(va_m)),
               n_trials=len([t for t in st.trials if t.state.name in ("COMPLETE", "PRUNED", "FAIL")]),
               test_id=c.test_id, seconds=round(time.time() - t0, 1))
    np.savez_compressed(out.with_suffix(".npz"), test_err=np.mean(te, axis=0), val_err=np.mean(va, axis=0),
                        test_err_seeds=np.stack(te))
    out.write_text(json.dumps(rec))
    print(f"  [FINAL] {model:10s} {trace:9s} H{H:<3d} L={L:<5d} test MASE={rec['test_mase']:.3f} "
          f"± {rec['test_mase_std']:.3f}", flush=True)


# ------------------------------------------------------------------ cluster runner

def tasks():
    order = ["TFT", "TimeXer", "LSTM", "GRU", "TiDE", "NHITS", "NBEATSx", "TSMixerx", "TCN", "BiTCN", "KAN", "MLP",
             "LightGBM", "Ridge"]
    return [(m, tr, H) for m in order for H in sw.HORIZONS for tr in cv.COV_TRACES if eligible(m, tr)]


def finished(m, tr, H):
    return (FINAL / f"{tr}_H{H}_{m}.json").exists()


def release():
    try:
        live = set(subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i"],
                                  capture_output=True, text=True).stdout.split())
    except FileNotFoundError:
        live = set()
    n = 0
    for d in CLAIMS.iterdir():
        job = (d / "job").read_text().strip() if (d / "job").exists() else ""
        m, tr, H = d.name.split("__")
        if not finished(m, tr, int(H[1:])) and job not in live and not (d / "failed").exists():
            shutil.rmtree(d, ignore_errors=True); n += 1
    print(f"released {n} stale claims")


def work(worker, only=None):
    job = os.environ.get("SLURM_JOB_ID", "local")
    for m, tr, H in tasks():
        if finished(m, tr, H) or (only and m not in only):
            continue
        d = CLAIMS / dl.study_name(m, tr, H)
        try:
            d.mkdir()
        except FileExistsError:
            continue
        (d / "job").write_text(job)
        t0 = time.time()
        print(f"[w{worker}] start {d.name}", flush=True)
        try:
            run_hpo(m, tr, H)
            run_final(m, tr, H)
            print(f"[w{worker}] done  {d.name} in {(time.time() - t0) / 60:.1f} min", flush=True)
        except Exception:
            (d / "failed").write_text(traceback.format_exc())
            print(f"[w{worker}] FAILED {d.name}\n{traceback.format_exc()}", flush=True)


def status():
    ts = tasks()
    done = sum(finished(*t) for t in ts)
    failed = [t for t in ts if (CLAIMS / dl.study_name(*t) / "failed").exists()]
    claimed = sum((CLAIMS / dl.study_name(*t)).exists() and not finished(*t) for t in ts)
    print(f"tasks {len(ts)}  done {done}  in progress {claimed - len(failed)}  failed {len(failed)}  "
          f"queued {len(ts) - done - claimed}")
    for t in failed:
        print("  failed:", dl.study_name(*t))


def smoke():
    """One short run per (model, plan) on Polaris (past + calendar) and Google (calendar only)."""
    dl.MAX_STEPS = 100
    for trace in ("polaris", "google"):
        for m in COV_MODELS:
            if not eligible(m, trace):
                print(f"  skip {m:10s} {trace}: no usable covariates"); continue
            try:
                t = optuna.create_study().ask()
                t0 = time.time()
                if m in ML:
                    L, hp = ml_space(t, m, trace)
                    c = cv.cov_cell(trace, 24, L)
                    e_va, _ = ml_fit_predict(c, m, hp, 0); plan = "ml"
                else:
                    L, kw, tr = space(t, m, trace, 24)
                    c = cv.cov_cell(trace, 24, L)
                    e_va, _ = train_eval(c, m, kw, tr, 0); plan = exog_plan(m, c)
                print(f"  OK   {m:10s} {trace:8s} L={L:<5d} plan={plan} val={e_va.mean() / c.scale:.3f} "
                      f"({time.time() - t0:.1f}s)", flush=True)
            except Exception as exc:
                print(f"  FAIL {m:10s} {trace:8s} {type(exc).__name__}: {str(exc)[:200]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "work", "status", "release", "list"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--models", nargs="*")
    a = ap.parse_args()
    if a.mode == "smoke":
        smoke()
    elif a.mode == "work":
        work(a.worker, a.models)
    elif a.mode == "status":
        status()
    elif a.mode == "release":
        release()
    else:
        for t in tasks():
            print(dl.study_name(*t), "done" if finished(*t) else "")


if __name__ == "__main__":
    main()
