"""Figures for the follow-up experiments (reads the CSVs written by why_analysis / why_results /
why_synth2). Each figure is saved as .pdf (for the report) and .png (for inspection).

    python scripts/why_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_fm_sweep as afs  # noqa: E402
import run_fm_sweep as sw  # noqa: E402

W = sw.OUT.parent / "why"
R, A = W / "results", W / "analysis"
FIG = W / "figures"
FIG.mkdir(exist_ok=True)
plt = afs.fig_setup()
FMC, LOCC = "#d62728", "#1f77b4"
TN = afs.TRACE_NICE
SHORT = {"mit": "MIT", "surf": "SURF", "google": "Google", "bitbrains": "Bitbrains", "lnn": "LNN", "nlr": "NLR",
         "polaris": "Polaris", "aurora": "Aurora"}


def save(fig, name):
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{e}")
    plt.close(fig)


def f_context():
    d = pd.read_csv(A / "a1_context.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, H in zip(axes, (24, 6)):
        g = d[d.H == H]
        for tr, gt in g.groupby("trace"):
            ax.plot(gt.L, gt.gain, "-", color="0.75", lw=0.9)
        med = g.groupby("L").gain.median()
        ax.plot(med.index, med.values, "-o", color=FMC, lw=2, label="median over 8 traces")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log", base=2); ax.set_xticks(sw.CONTEXTS); ax.set_xticklabels(sw.CONTEXTS)
        ax.set_xlabel("context length the FMs may use (steps)")
        ax.set_title(f"H = {H}")
    axes[0].set_ylabel("FM gain over best tuned model (%)\n(> 0: FM better)")
    axes[0].legend(frameon=False, loc="upper left")
    save(fig, "context")


def f_scratch():
    p = pd.read_csv(R / "h3_finetune.csv", header=[0, 1], index_col=[0, 1])
    p = p.xs(24, level="H")
    order = [t for t in sw.TRACES if t in p.index]
    x = np.arange(len(order)); w = 0.2
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    tuned = p[("tuned", "best")].loc[order].values
    for i, (col, lab, c) in enumerate([(("bolt_small", "zeroshot"), "Chronos-Bolt-small, pretrained, zero-shot", FMC),
                                       (("bolt_small", "finetune"), "… pretrained + fine-tuned", "#ff9896"),
                                       (("bolt_small", "scratch"), "… same architecture, trained from scratch", "#7f7f7f")]):
        ax.bar(x + (i - 1) * w, p[col].loc[order].values / tuned, w, color=c, label=lab)
    ax.axhline(1, color="k", lw=0.8)
    ax.text(-0.45, 1.03, "best tuned local model = 1", ha="left", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[t] for t in order])
    ax.set_ylabel("test MASE ÷ best tuned MASE\n(lower is better)")
    ax.set_ylim(0, 3.2)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Same architecture, with vs without pretraining (H = 24)")
    save(fig, "pretraining_vs_scratch")


def f_synth():
    d = pd.read_csv(R / "h5_synthetic_all.csv")
    import why_synth as ws
    b = d.loc[d.groupby(["synth", "kind"]).val_mase.idxmin()].pivot(index="synth", columns="kind", values="excess")
    b = b.loc[ws.TRACES]
    labels = {"ar_0.3": "AR φ=.3", "ar_0.6": "AR φ=.6", "ar_0.9": "AR φ=.9", "ar_0.97": "AR φ=.97",
              "ar_0.995": "AR φ=.995", "seas": "AR+daily", "ll_0.01": "RW q=.01", "ll_0.1": "RW q=.1",
              "ll_1": "RW q=1", "shift_0": "jump 0σ", "shift_3": "jump 3σ", "shift_6": "jump 6σ",
              "onoff_0.02": "on/off p=.02", "onoff_0.1": "on/off p=.1"}
    x = np.arange(len(b)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 2.85))
    ax.bar(x - w / 2, b.fm, w, color=FMC, label="best FM (zero-shot)")
    ax.bar(x + w / 2, b.local, w, color=LOCC, label="best tuned local model (4k windows)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([labels[t] for t in b.index], rotation=45, ha="right")
    ax.set_ylabel("excess over optimum (%)\n(lower is better)")
    ax.legend(frameon=False)
    ax.axvspan(8.5, 11.5, color="0.94", zorder=0)  # the distribution-shift block
    top = ax.get_ylim()[1]
    ax.set_ylim(top=top * 1.22)  # head-room so the block label sits above the tallest bar
    ax.text(10, top * 1.2, "mean jump just before\nthe test period", ha="center", va="top", fontsize=7.5)
    save(fig, "synthetic")


def f_synth2():
    d = pd.read_csv(W / "synth2" / "synth2_results.csv")
    names = {"ar_0.9": "AR φ=0.9", "ar_0.97": "AR φ=0.97", "seas": "AR + daily\ncycle",
             "shift_3": "mean jump\nbefore test", "onoff_0.02": "rare on/off\nswitching"}
    fig, axes = plt.subplots(1, 5, figsize=(7.4, 2.9), sharey=False)
    Ns = [1000, 4000, 16000, 64000]
    for ax, (_, r) in zip(axes, d.iterrows()):
        ax.plot(Ns, [r[f"local_excess_N{n}"] for n in Ns], "-o", color=LOCC, ms=3.5, label="tuned local")
        ax.axhline(r.fm_excess, color=FMC, lw=1.6, label="FM zero-shot")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xscale("log"); ax.set_xticks([1e3, 1e4, 6.4e4]); ax.set_xticklabels(["1k", "10k", "64k"])
        ax.set_title(names[r.proc], fontsize=8.5)
        ax.set_xlabel("training windows", fontsize=8)
    axes[0].set_ylabel("excess over optimum (%)\n(lower is better)")
    axes[0].legend(frameon=False, fontsize=7.5)
    save(fig, "more_data_synthetic")


def f_lc():
    b = pd.read_csv(R / "h1_learning_curves_best.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    for ax, tr in zip(axes, ["mit", "surf", "nlr"]):
        for H, ls in ((24, "-o"), (6, "--s")):
            g = b[(b.trace == tr) & (b.H == H)].sort_values("n_train")
            g = g[g.n_train.isin([250, 500, 1000, 2000, 4000, 8000, 16000, 32000]) | (g.n_train == g.n_train.max())]
            ax.plot(g.n_train, g.fm_gain, ls, ms=3.5, label=f"H = {H}")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log"); ax.set_title(TN[tr], fontsize=9.5); ax.set_xlabel("training windows")
    axes[0].set_ylabel("FM gain over best local (%)\n(> 0: FM better)")
    axes[0].legend(frameon=False)
    save(fig, "learning_curves")


def f_ens():
    e = pd.read_csv(R / "h7_ensembles.csv")
    cols = [("fmgain_vs_best_single_seedavg", "single best\nmodel"), ("fmgain_vs_seed_ensemble", "3-seed\nensemble"),
            ("fmgain_vs_top3_mean", "top-3\nmodels"), ("fmgain_vs_top5_mean", "top-5\nmodels"),
            ("fmgain_vs_top10_median", "top-10\n(median)")]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    data = [e[c].values for c, _ in cols]
    ax.boxplot(data, widths=0.5, medianprops=dict(color=FMC, lw=2))
    for i, v in enumerate(data):
        ax.scatter(np.full(len(v), i + 1) + np.random.default_rng(i).uniform(-.12, .12, len(v)), v, s=9, color="0.4", zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(1, len(cols) + 1)); ax.set_xticklabels([l for _, l in cols])
    ax.set_ylabel("FM gain over the local ensemble (%)\n(> 0: FM better)")
    save(fig, "ensembles")


def f_corpus():
    p = pd.read_csv(R / "h4_pretrain.csv", header=[0, 1], index_col=[0, 1])
    corp = [("target", "target only"), ("loo", "7 other traces"), ("pool", "all 8 traces"),
            ("public", "public corpus"), ("public+ft", "public + fine-tune")]
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    fm = p[("FM", "zero-shot")]
    x = np.arange(len(corp)); w = 0.36
    for i, (arch, c) in enumerate((("NHITS", "#2ca02c"), ("PatchTST", "#17becf"))):
        rel = [float(np.exp(np.log(p[(arch, k)] / fm).mean())) for k, _ in corp]
        ax.bar(x + (i - 0.5) * w, rel, w, color=c, label=arch)
    ax.axhline(1, color=FMC, lw=1.6, label="real FMs, zero-shot (= 1)")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in corp])
    ax.set_ylabel("test MASE ÷ FM MASE\n(geo-mean, 16 settings;\nlower is better)")
    ax.set_ylim(0.9, max(1.4, ax.get_ylim()[1]))
    ax.legend(frameon=False)
    save(fig, "pretraining_corpus")


if __name__ == "__main__":
    for f in (f_context, f_scratch, f_synth, f_synth2, f_lc, f_ens, f_corpus):
        f(); print("ok", f.__name__)
