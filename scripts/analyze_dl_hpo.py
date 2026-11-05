"""Combine tuned DL/ML baselines (3 seeds) with the foundation-model sweep and produce tables/figures.

Selection rules (no test peeking):
  * tuned models  — configuration chosen by Optuna on validation MASE (run_dl_hpo / run_ml_hpo).
  * foundation    — context chosen on validation MASE (from the FM sweep); deterministic, so std = 0.
  * classical     — Persistence / Seasonal-naive / AutoETS from the FM sweep, context chosen on validation.
  * "best tuned" and "best foundation" are each chosen by validation MASE, then scored on test.

    python scripts/analyze_dl_hpo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_fm_sweep as sw  # noqa: E402
import analyze_fm_sweep as afs  # noqa: E402

OUT = sw.OUT.parent / "dl_hpo"
FINAL = OUT / "final"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

FM_NAMES = set(afs.FM)
CLASSICAL = {"Persistence", "SeasonalNaive", "AutoETS"}


def load_tuned():
    rows = [json.loads(p.read_text()) for p in FINAL.glob("*.json")]
    df = pd.DataFrame(rows)
    df["group"] = np.where(df.model.isin(["Ridge", "LightGBM"]), "ml", "dl")
    return df


def load_reference():
    """FM + classical rows from the earlier sweep, context selected on validation."""
    sel = pd.read_csv(sw.OUT / "context_selected.csv")
    sel = sel[sel.model.isin(FM_NAMES | CLASSICAL)].copy()
    sel["test_mase_std"] = 0.0
    sel["group"] = np.where(sel.model.isin(FM_NAMES), "foundation", "classical")
    return sel


def errs(row, which="test_err"):
    if row["group"] in ("dl", "ml"):
        return np.load(FINAL / f"{row['trace']}_H{row['H']}_{row['model']}.npz")[which]
    return np.load(sw.cell_path(row["trace"], row["H"], row["L"], row["model"]).with_suffix(".npz"))[which]


def seed_errs(row):
    return np.load(FINAL / f"{row['trace']}_H{row['H']}_{row['model']}.npz")["test_err_seeds"]


def main():
    tuned = load_tuned()
    ref = load_reference()
    # integrity: every tuned model scored on the same test windows as the FM sweep
    ref_ids = ref.groupby(["trace", "H"]).test_id.first()
    bad = tuned[tuned.apply(lambda r: r.test_id != ref_ids[(r.trace, r.H)], axis=1)]
    print(f"tuned final rows: {len(tuned)}  | test-set mismatches vs FM sweep: {len(bad)}")
    cols = ["trace", "H", "model", "group", "L", "val_mase", "test_mase", "test_mase_std"]
    allm = pd.concat([tuned[cols], ref[cols]], ignore_index=True)
    allm.to_csv(OUT / "leaderboard_all.csv", index=False)

    # ---- headline: best tuned trained model vs best FM (both val-selected) ----
    rows = []
    for (tr, H), g in allm.groupby(["trace", "H"]):
        t = g[g.group.isin(["dl", "ml"])].sort_values("val_mase").iloc[0]
        f = g[g.group == "foundation"].sort_values("val_mase").iloc[0]
        e_f = errs(f)
        per_seed = [(es.mean() - e_f.mean()) / es.mean() * 100 for es in seed_errs(t)]
        e_t = errs(t)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(e_f), (2000, len(e_f)))
        boots = (e_t[idx].mean(1) - e_f[idx].mean(1)) / e_t[idx].mean(1) * 100
        rows.append(dict(trace=tr, H=H, best_tuned=t.model, tuned_L=t.L, tuned_mase=t.test_mase,
                         tuned_std=t.test_mase_std, best_fm=f.model, fm_L=f.L, fm_mase=f.test_mase,
                         gain_mean=float(np.mean(per_seed)), gain_std=float(np.std(per_seed, ddof=1)),
                         ci_lo=float(np.percentile(boots, 2.5)), ci_hi=float(np.percentile(boots, 97.5))))
    hl = pd.DataFrame(rows)
    hl.to_csv(OUT / "headline_tuned.csv", index=False)
    pd.set_option("display.width", 220)
    print(hl.round(3).to_string(index=False))

    # ---- average rank (by test MASE) across the 16 trace x horizon settings ----
    allm["rank"] = allm.groupby(["trace", "H"]).test_mase.rank()
    rk = allm.groupby(["model", "group"]).agg(mean_rank=("rank", "mean"), n=("rank", "size"),
                                              gm_vs_pers=("test_mase", lambda s: np.nan)).reset_index()
    pers = allm[allm.model == "Persistence"].set_index(["trace", "H"]).test_mase
    allm["rel_pers"] = allm.apply(lambda r: r.test_mase / pers[(r.trace, r.H)], axis=1)
    gm = allm.groupby("model").rel_pers.apply(lambda x: float(np.exp(np.log(x).mean())))
    rk["gm_vs_pers"] = rk.model.map(gm)
    rk = rk.sort_values("mean_rank")
    rk.to_csv(OUT / "average_rank.csv", index=False)
    print("\n=== average rank (lower = better) ===")
    print(rk.round(3).to_string(index=False))

    # ---- smoothness law vs tuned baselines ----
    from scipy.stats import spearmanr
    lag = pd.read_csv(sw.OUT / "trace_info.csv").set_index("trace").lag1
    for H in sw.HORIZONS:
        h = hl[hl.H == H]
        rho, p = spearmanr(h.trace.map(lag), h.gain_mean)
        print(f"H={H}: smoothness vs tuned baselines  Spearman rho={rho:+.3f} p={p:.3f}")

    fig_rank(rk)
    for H in sw.HORIZONS:
        fig_headline(hl, H)


def fig_rank(rk):
    plt = afs.fig_setup()
    colors = {"foundation": "#d62728", "dl": "#1f77b4", "ml": "#e377c2", "classical": "#7f7f7f"}
    r = rk.sort_values("mean_rank", ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 8.4))
    ax.barh(range(len(r)), r.mean_rank, color=[colors[g] for g in r.group])
    ax.set_yticks(range(len(r)))
    ax.set_yticklabels([afs.NICE.get(m, m).split(" (")[0] for m in r.model], fontsize=8)
    ax.set_xlabel("average rank across 16 trace × horizon settings (lower is better)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="s", ls="", color=c, label=g) for g, c in colors.items()],
              frameon=False, loc="lower right")
    ax.set_title("All models after hyperparameter tuning")
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"average_rank.{e}")
    plt.close(fig)


def fig_headline(hl, H):
    plt = afs.fig_setup()
    h = hl[hl.H == H].sort_values("gain_mean")
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    y = np.arange(len(h))
    ax.barh(y, h.gain_mean, xerr=h.gain_std, color=["#d62728" if g > 0 else "#7f7f7f" for g in h.gain_mean],
            capsize=3)
    ax.set_yticks(y); ax.set_yticklabels([afs.TRACE_NICE[t] for t in h.trace])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("best foundation vs best TUNED trained model (%)\n(mean ± std over 3 seeds; >0 = foundation better)")
    ax.set_title(f"Foundation models vs hyperparameter-tuned baselines (H = {H})")
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"headline_tuned_H{H}.{e}")
    plt.close(fig)


if __name__ == "__main__":
    main()
