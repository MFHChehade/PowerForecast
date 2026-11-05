"""Hyperparameter-optimised ML baselines (Ridge, LightGBM) on the PowerForecast sweep windows.

Same protocol as run_dl_hpo.py: Optuna TPE per (model, trace, horizon) on VALIDATION MASE, context L
tuned, per-window standardisation, then 3 final seeds scored on the shared test windows. Runs on CPU,
so it can run alongside the GPU deep-learning search.

    python scripts/run_ml_hpo.py hpo   --trials 20
    python scripts/run_ml_hpo.py final
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import optuna  # noqa: E402

import run_dl_hpo as dl  # noqa: E402  (shared storage, cells, normalisation)
import run_fm_sweep as sw  # noqa: E402

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ML_MODELS = ["Ridge", "LightGBM"]
TRIALS = {"Ridge": 20, "LightGBM": 15}  # LightGBM fits one model per horizon step (24x cost at H=24)


def space(trial, model, trace):
    L = trial.suggest_categorical("L", [L for L in sw.CONTEXTS if L <= sw.TRACES[trace]["L_max"]])
    if model == "Ridge":
        return L, dict(alpha=trial.suggest_float("alpha", 1e-3, 1e3, log=True))
    return L, dict(n_estimators=trial.suggest_categorical("n_estimators", [100, 300, 600]),
                   learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                   num_leaves=trial.suggest_categorical("num_leaves", [15, 31, 63, 127]),
                   min_child_samples=trial.suggest_categorical("min_child_samples", [10, 20, 50]),
                   colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                   reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True))


def fit_predict(cell, model, hp, seed, return_pred=False):
    """Per-window standardised inputs/targets; returns (val_err, test_err) per window."""
    scale = cell.scale
    m_tr, s_tr = dl.norm_stats(cell.Xtr, scale)
    m_va, s_va = dl.norm_stats(cell.Xva, scale)
    m_te, s_te = dl.norm_stats(cell.Xte, scale)
    Xtr, Ytr = (cell.Xtr - m_tr) / s_tr, (cell.Ytr - m_tr) / s_tr
    Xva, Xte = (cell.Xva - m_va) / s_va, (cell.Xte - m_te) / s_te
    if model == "Ridge":
        from sklearn.linear_model import Ridge
        mdl = Ridge(alpha=hp["alpha"]).fit(Xtr, Ytr)
        p_va, p_te = mdl.predict(Xva), mdl.predict(Xte)
    else:
        import lightgbm as lgb
        p_va = np.empty((len(Xva), cell.H)); p_te = np.empty((len(Xte), cell.H))
        for h in range(cell.H):
            mdl = lgb.LGBMRegressor(random_state=seed, subsample=0.8, subsample_freq=1, verbose=-1,
                                    n_jobs=int(os.environ.get("PF_NJOBS", -1)), **hp).fit(Xtr, Ytr[:, h])
            p_va[:, h], p_te[:, h] = mdl.predict(Xva), mdl.predict(Xte)
    p_va, p_te = p_va * s_va + m_va, p_te * s_te + m_te
    if return_pred:
        return p_va, p_te
    return np.mean(np.abs(cell.Yva - p_va), 1), np.mean(np.abs(cell.Yte - p_te), 1)


def run_hpo(model, trace, H):
    st = optuna.create_study(study_name=dl.study_name(model, trace, H), storage=dl.storage(model, trace, H),
                             load_if_exists=True, direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=sw.stable_seed(model, trace, H) % 2**31,
                                                                n_startup_trials=6, multivariate=True))
    # count only finished trials: a trial left RUNNING by a killed process must not use up budget
    finished = [t for t in st.trials if t.state.name in ("COMPLETE", "PRUNED", "FAIL")]
    remaining = TRIALS[model] - len(finished)
    if remaining <= 0:
        return

    def objective(trial):
        L, hp = space(trial, model, trace)
        c = dl.cell_for(trace, H, L)
        t0 = time.time()
        e_va, _ = fit_predict(c, model, hp, seed=0)
        trial.set_user_attr("seconds", round(time.time() - t0, 1))
        return float(e_va.mean() / c.scale)

    st.optimize(objective, n_trials=remaining, catch=(RuntimeError, ValueError))
    b = st.best_trial
    print(f"  [HPO] {model:18s} {trace:9s} H{H:<3d} best val MASE={b.value:.3f}  L={b.params['L']}", flush=True)


def run_final(model, trace, H):
    out = dl.FINAL / f"{trace}_H{H}_{model}.json"
    if out.exists():
        return
    st = optuna.load_study(study_name=dl.study_name(model, trace, H), storage=dl.storage(model, trace, H))
    best = st.best_trial
    L, hp = space(optuna.trial.FixedTrial(best.params), model, trace)
    c = dl.cell_for(trace, H, L)
    t0 = time.time()
    va, te = zip(*[fit_predict(c, model, hp, seed=s) for s in dl.FINAL_SEEDS])
    te_m = [float(e.mean() / c.scale) for e in te]
    va_m = [float(e.mean() / c.scale) for e in va]
    rec = dict(trace=trace, H=H, L=L, model=model, family="ml", params=best.params, test_mase_seeds=te_m,
               val_mase_seeds=va_m, test_mase=float(np.mean(te_m)), test_mase_std=float(np.std(te_m, ddof=1)),
               val_mase=float(np.mean(va_m)), n_trials=len(st.trials), test_id=c.test_id,
               seconds=round(time.time() - t0, 1))
    np.savez_compressed(out.with_suffix(".npz"), test_err=np.mean(te, axis=0), val_err=np.mean(va, axis=0),
                        test_err_seeds=np.stack(te))
    out.write_text(json.dumps(rec))
    print(f"  [FINAL] {model:18s} {trace:9s} H{H:<3d} L={L:<5d} test MASE={rec['test_mase']:.3f} "
          f"± {rec['test_mase_std']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["hpo", "final"])
    ap.add_argument("--models", nargs="*", default=ML_MODELS)
    ap.add_argument("--traces", nargs="*", default=list(sw.TRACES))
    ap.add_argument("--horizons", nargs="*", type=int, default=sw.HORIZONS)
    args = ap.parse_args()
    sw.keep_awake()
    for H in args.horizons:
        for tr in args.traces:
            for m in args.models:
                try:
                    (run_hpo if args.mode == "hpo" else run_final)(m, tr, H)
                except Exception as exc:
                    print(f"  [ERROR] {args.mode} {m} {tr} H{H}: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
