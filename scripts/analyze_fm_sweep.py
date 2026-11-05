"""Aggregate the foundation-model size x context sweep into tables, figures and significance tests.

Selection rule (applied everywhere unless a table says "oracle"): each model's context window is
chosen by VALIDATION MASE; "best local" / "best foundation" are then chosen by validation MASE among
the context-selected models; only after selection is anything scored on test.

    python scripts/analyze_fm_sweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_fm_sweep as sw  # noqa: E402

OUT = sw.OUT
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

FM = {**{k: v for k, v in sw.FM_MODELS.items()}, "timesfm_2p0_500m": ("timesfm", 500),
      "timesfm_3p0_330m": ("timesfm", 330)}
LOCAL = sw.LOCAL_MODELS
NICE = {
    "chronos_bolt_tiny": "Chronos-Bolt tiny (9M)", "chronos_bolt_mini": "Chronos-Bolt mini (21M)",
    "chronos_bolt_small": "Chronos-Bolt small (48M)", "chronos_bolt_base": "Chronos-Bolt base (205M)",
    "chronos_2": "Chronos-2 (120M)", "timesfm_2p5_200m": "TimesFM-2.5 (200M)",
    "timesfm_2p0_500m": "TimesFM-2.0 (500M)", "timesfm_3p0_330m": "TimesFM-3.0 (330M)",
}
TRACE_NICE = {"mit": "MIT Supercloud", "surf": "SURF Lisa", "google": "Google PowerData'19",
              "bitbrains": "Bitbrains", "lnn": "LNN AI", "nlr": "NLR H100",
              "polaris": "ALCF Polaris", "aurora": "ALCF Aurora"}


def load_cells():
    rows = [json.loads(p.read_text()) for p in sw.CELLS.glob("*.json")]
    df = pd.DataFrame(rows)
    df["group"] = np.where(df.model.isin(list(FM)), "foundation", "local")
    return df


def errs(r, which="test_err"):
    p = sw.cell_path(r["trace"], r["H"], r["L"], r["model"]).with_suffix(".npz")
    return np.load(p)[which]


def integrity(df):
    issues = []
    for (tr, H), g in df.groupby(["trace", "H"]):
        if g.test_id.nunique() != 1:
            issues.append(f"{tr} H{H}: {g.test_id.nunique()} different test sets")
    expected_L = {tr: [L for L in sw.CONTEXTS if L <= sw.TRACES[tr]["L_max"]] for tr in sw.TRACES}
    missing = []
    for tr in sw.TRACES:
        for H in sw.HORIZONS:
            for m in list(FM) + list(LOCAL):
                for L in expected_L[tr]:
                    if m == "SeasonalNaive" and (sw.TRACES[tr]["period"] is None or sw.TRACES[tr]["period"] > L):
                        continue
                    if df[(df.trace == tr) & (df.H == H) & (df.model == m) & (df.L == L)].empty:
                        missing.append((tr, H, m, L))
    return issues, missing


def context_selected(df):
    """One row per (trace, H, model): the context with the lowest validation MASE."""
    idx = df.groupby(["trace", "H", "model"]).val_mase.idxmin()
    return df.loc[idx].reset_index(drop=True)


def paired(r_fm, r_loc, n_boot=2000, seed=0):
    from scipy.stats import wilcoxon
    a, b = errs(r_fm), errs(r_loc)
    gain = (b.mean() - a.mean()) / b.mean() * 100
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    boots = (b[idx].mean(1) - a[idx].mean(1)) / b[idx].mean(1) * 100
    try:
        p = wilcoxon(a, b).pvalue
    except ValueError:
        p = np.nan
    return gain, np.percentile(boots, 2.5), np.percentile(boots, 97.5), p


def headline(sel):
    rows = []
    for (tr, H), g in sel.groupby(["trace", "H"]):
        loc = g[g.group == "local"].sort_values("val_mase").iloc[0]
        fm = g[g.group == "foundation"].sort_values("val_mase").iloc[0]
        gain, lo, hi, p = paired(fm, loc)
        oracle_loc = g[g.group == "local"].test_mase.min()
        oracle_fm = g[g.group == "foundation"].test_mase.min()
        rows.append(dict(trace=tr, H=H,
                         best_local=f"{loc.model} (L={loc.L})", local_mase=loc.test_mase,
                         best_fm=f"{fm.model} (L={fm.L})", fm_mase=fm.test_mase,
                         gain_pct=gain, ci_lo=lo, ci_hi=hi, wilcoxon_p=p,
                         oracle_gain_pct=(oracle_loc - oracle_fm) / oracle_loc * 100))
    return pd.DataFrame(rows)


def family_winners(sel):
    rows = []
    for (tr, H), g in sel.groupby(["trace", "H"]):
        g = g.copy()
        g["fam"] = np.where(g.group == "foundation", "foundation", g.family)
        best = g.loc[g.groupby("fam").val_mase.idxmin()].set_index("fam").test_mase
        rows.append(dict(trace=tr, H=H, **{f"{k}": v for k, v in best.items()}))
    return pd.DataFrame(rows)


def size_table(sel):
    """Test MASE of every foundation model at its validation-selected context, relative to the
    trace's best local model (val-selected): < 1 means lower error than that model."""
    rows = []
    for (tr, H), g in sel.groupby(["trace", "H"]):
        loc = g[g.group == "local"].sort_values("val_mase").iloc[0].test_mase
        for _, r in g[g.group == "foundation"].iterrows():
            rows.append(dict(trace=tr, H=H, model=r.model, L=r.L, test_mase=r.test_mase,
                             rel_to_best_local=r.test_mase / loc))
    return pd.DataFrame(rows)


def context_curves(df):
    """Geometric-mean (over traces) of test MASE / Persistence MASE, per model and context."""
    pers = df[df.model == "Persistence"].groupby(["trace", "H"]).test_mase.first()
    d = df.copy()
    d["rel_pers"] = d.apply(lambda r: r.test_mase / pers[(r.trace, r.H)], axis=1)
    return d


def lag1_table():
    info = pd.read_csv(OUT / "trace_info.csv")
    return info.set_index("trace").lag1


# --------------------------------------------------------------------------- figures

def fig_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
                         "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 200})
    return plt


FM_COLORS = {"chronos_bolt_tiny": "#c6dbef", "chronos_bolt_mini": "#6baed6",
             "chronos_bolt_small": "#2171b5", "chronos_bolt_base": "#08306b",
             "chronos_2": "#9467bd", "timesfm_2p5_200m": "#d62728", "timesfm_2p0_500m": "#ff9896",
             "timesfm_3p0_330m": "#ff7f0e"}
LOC_COLORS = {"Persistence": "#7f7f7f", "AutoETS": "#bcbd22", "SeasonalNaive": "#c7c7c7",
              "RidgeLag": "#8c564b", "LightGBM": "#e377c2",
              "NLinear": "#2ca02c", "DLinear": "#98df8a", "PatchTST": "#17becf"}


def fig_size_scaling(sz, H):
    plt = fig_setup()
    s = sz[sz.H == H]
    gm = s.groupby("model").rel_to_best_local.apply(lambda x: float(np.exp(np.log(x).mean())))
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    bolt = ["chronos_bolt_tiny", "chronos_bolt_mini", "chronos_bolt_small", "chronos_bolt_base"]
    ax.plot([FM[m][1] for m in bolt], [gm[m] for m in bolt], "-o", color="#2171b5",
            label="Chronos-Bolt (tiny→base)")
    for m in ["chronos_2", "timesfm_2p5_200m", "timesfm_2p0_500m", "timesfm_3p0_330m"]:
        if m in gm:
            ax.plot(FM[m][1], gm[m], "s", ms=8, color=FM_COLORS[m], label=NICE[m])
    ax.axhline(1.0, color="0.4", ls="--", lw=1)
    ax.set_xscale("log")
    ax.text(600, 1.0, " best local\n baseline = 1", fontsize=8, color="0.35", va="center", ha="left")
    ax.set_xlim(6, 1500)
    ax.set_xlabel("parameters (millions, log scale)")
    ax.set_ylabel("test MASE ÷ best local MASE\n(geo-mean over 8 traces; lower is better)")
    ax.set_title(f"Does a bigger foundation model help?  (horizon = {H} steps)")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"size_scaling_H{H}.{e}")
    plt.close(fig)


def fig_context_scaling(cc, H):
    plt = fig_setup()
    full = [t for t in sw.TRACES if sw.TRACES[t]["L_max"] == 1024]
    d = cc[(cc.H == H) & (cc.trace.isin(full))]
    show = ["chronos_bolt_base", "chronos_bolt_tiny", "chronos_2", "timesfm_2p5_200m", "timesfm_2p0_500m",
            "timesfm_3p0_330m", "PatchTST", "NLinear", "LightGBM", "RidgeLag", "AutoETS"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for m in show:
        g = d[d.model == m].groupby("L").rel_pers.apply(lambda x: float(np.exp(np.log(x).mean())))
        if g.empty:
            continue
        is_fm = m in FM
        ax.plot(g.index, g.values, "-o" if is_fm else "--^", ms=4, lw=1.6 if is_fm else 1.1,
                color=FM_COLORS.get(m, LOC_COLORS.get(m)), label=NICE.get(m, m))
    ax.axhline(1.0, color="0.4", ls=":", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sw.CONTEXTS); ax.set_xticklabels(sw.CONTEXTS)
    ax.set_xlabel("context window L (steps)")
    ax.set_ylabel("test MASE ÷ Persistence MASE\n(geo-mean; lower is better)")
    ax.set_title(f"How much history does each model use?  (H = {H}; {len(full)} traces with L up to 1024)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"context_scaling_H{H}.{e}")
    plt.close(fig)


def fig_smoothness(hl, lag, H):
    from scipy.stats import spearmanr
    plt = fig_setup()
    h = hl[hl.H == H].copy()
    h["lag1"] = h.trace.map(lag)
    rho, p = spearmanr(h.lag1, h.gain_pct)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.errorbar(h.lag1, h.gain_pct, yerr=[h.gain_pct - h.ci_lo, h.ci_hi - h.gain_pct], fmt="o",
                color="#08306b", ecolor="0.6", capsize=3)
    offsets = {"mit": (6, -14), "lnn": (6, 6), "surf": (-6, -14), "polaris": (-8, 4),
               "bitbrains": (6, 4), "google": (-10, 8), "nlr": (6, 4), "aurora": (6, 4)}
    if H == 6:  # Google, SURF and Polaris sit close together at H = 6
        offsets.update({"surf": (-8, -14), "google": (6, -14), "polaris": (-8, 8)})
        ax.set_xlim(right=1.10)  # room for Google's label to the right of its point
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # autocorrelation cannot exceed 1
    for _, r in h.iterrows():
        dx, dy = offsets.get(r.trace, (5, 4))
        ax.annotate(TRACE_NICE[r.trace], (r.lag1, r.gain_pct), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8, color="0.3", ha="right" if dx < 0 else "left")
    ax.axhline(0, color="0.5", lw=0.8, ls=":")
    ax.set_xlabel("lag-1 autocorrelation of training history (smoother →)")
    ax.set_ylabel("best foundation vs. best local (%)\n(validation-selected; 95% bootstrap CI)")
    ax.set_title(f"Smoothness law re-check (H = {H}):  Spearman ρ = {rho:+.2f}, p = {p:.2f}, n = {len(h)}")
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"smoothness_H{H}.{e}")
    plt.close(fig)
    return rho, p


def fig_heatmaps(df, H):
    plt = fig_setup()
    order = ["Persistence", "SeasonalNaive", "AutoETS", "RidgeLag", "LightGBM", "NLinear", "DLinear",
             "PatchTST"] + list(FM)
    traces = list(sw.TRACES)
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 8.6), gridspec_kw=dict(wspace=0.08, hspace=0.28))
    for k, (ax, tr) in enumerate(zip(axes.ravel(), traces)):
        d = df[(df.trace == tr) & (df.H == H)]
        # identical row order on every panel (missing rows stay blank) so labels can be shared
        piv = d.pivot_table(index="model", columns="L", values="test_mase").reindex(order)
        best = np.nanmin(piv.values)
        rel = piv / best
        im = ax.imshow(rel.values, cmap="viridis_r", vmin=1, vmax=2, aspect="auto")
        ax.set_xticks(range(len(rel.columns))); ax.set_xticklabels(rel.columns, fontsize=7)
        ax.set_yticks(range(len(rel.index)))
        if k % 4 == 0:
            ax.set_yticklabels([NICE.get(m, m).split(" (")[0] for m in rel.index], fontsize=7)
        else:
            ax.set_yticklabels([])
        for i in range(rel.shape[0]):
            for j in range(rel.shape[1]):
                v = rel.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.2,
                            color="white" if v > 1.5 else "black")
        ax.set_title(TRACE_NICE[tr], fontsize=9)
        ax.set_xlabel("context L", fontsize=7.5)
    fig.colorbar(im, ax=axes, shrink=0.6, label="test MASE ÷ best cell on that trace (lower is better; capped at 2)")
    fig.suptitle(f"Every model × every context, every trace (H = {H})", fontsize=11)
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"heatmaps_H{H}.{e}", bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_cells()
    df.to_csv(OUT / "all_cells.csv", index=False)
    issues, missing = integrity(df)
    print(f"cells: {len(df)}   test-set mismatches: {issues or 'none'}   missing cells: {len(missing)}")
    if missing:
        print("  first missing:", missing[:8])
    sel = context_selected(df)
    sel.to_csv(OUT / "context_selected.csv", index=False)
    hl = headline(sel)
    hl.to_csv(OUT / "headline.csv", index=False)
    fw = family_winners(sel)
    fw.to_csv(OUT / "family_winners.csv", index=False)
    sz = size_table(sel)
    sz.to_csv(OUT / "size_table.csv", index=False)
    cc = context_curves(df)
    lag = lag1_table()
    pd.set_option("display.width", 200)
    print("\n=== headline (validation-selected) ===")
    print(hl.round(3).to_string(index=False))
    for H in sw.HORIZONS:
        if (df.H == H).any():
            fig_size_scaling(sz, H)
            fig_context_scaling(cc, H)
            rho, p = fig_smoothness(hl, lag, H)
            fig_heatmaps(df, H)
            print(f"H={H}: smoothness Spearman rho={rho:+.3f} p={p:.3f}")


if __name__ == "__main__":
    main()
