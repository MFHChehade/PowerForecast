from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED = 3


def _optional_tests(diff: np.ndarray) -> dict:
    out = {
        "test_name": "sign_or_wilcoxon",
        "test_statistic": float(np.sum(diff < 0)),
        "p_value": np.nan,
        "p_value_fm_better": np.nan,
        "p_value_local_better": np.nan,
        "p_value_two_sided": np.nan,
        "scipy_available": False,
    }
    try:
        from scipy import stats

        nonzero = diff[np.abs(diff) > 1e-12]
        if len(nonzero) > 0:
            if len(nonzero) >= 2:
                wilcoxon_less = stats.wilcoxon(nonzero, alternative="less", zero_method="wilcox")
                wilcoxon_greater = stats.wilcoxon(nonzero, alternative="greater", zero_method="wilcox")
                wilcoxon_two = stats.wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
                out.update({
                    "test_name": "wilcoxon_signed_rank",
                    "test_statistic": float(wilcoxon_two.statistic),
                    "p_value": float(wilcoxon_two.pvalue),
                    "p_value_fm_better": float(wilcoxon_less.pvalue),
                    "p_value_local_better": float(wilcoxon_greater.pvalue),
                    "p_value_two_sided": float(wilcoxon_two.pvalue),
                    "scipy_available": True,
                })
            else:
                fm_sign = stats.binomtest(int(np.sum(nonzero < 0)), len(nonzero), 0.5, alternative="greater")
                local_sign = stats.binomtest(int(np.sum(nonzero > 0)), len(nonzero), 0.5, alternative="greater")
                two_sign = stats.binomtest(int(np.sum(nonzero < 0)), len(nonzero), 0.5, alternative="two-sided")
                out.update({
                    "test_name": "sign_test",
                    "test_statistic": float(np.sum(nonzero < 0)),
                    "p_value": float(two_sign.pvalue),
                    "p_value_fm_better": float(fm_sign.pvalue),
                    "p_value_local_better": float(local_sign.pvalue),
                    "p_value_two_sided": float(two_sign.pvalue),
                    "scipy_available": True,
                })
    except Exception:
        pass
    return out


def optional_dependency_status() -> dict:
    status = {}
    for name in ["scipy", "lightgbm"]:
        try:
            __import__(name)
            status[name] = "available"
        except Exception as exc:
            status[name] = f"skipped: {exc.__class__.__name__}"
    return status


def paired_window_frame(
    y_true: np.ndarray,
    fm_pred: np.ndarray,
    local_pred: np.ndarray,
    *,
    dataset: str,
    scenario: str,
    context: int,
    horizon: int,
    local_model: str,
    fm_model: str,
    local_feature_mode: str = "target_history_only",
    fm_feature_mode: str = "target_history_only",
) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    fm_pred = np.asarray(fm_pred)
    local_pred = np.asarray(local_pred)
    if y_true.shape != fm_pred.shape or y_true.shape != local_pred.shape:
        raise ValueError(f"Prediction shapes must match: y={y_true.shape}, fm={fm_pred.shape}, local={local_pred.shape}")
    local_mae = np.mean(np.abs(y_true - local_pred), axis=1)
    fm_mae = np.mean(np.abs(y_true - fm_pred), axis=1)
    abs_gap = fm_mae - local_mae
    pct_gap = abs_gap / np.maximum(local_mae, 1e-9) * 100.0
    return pd.DataFrame({
        "dataset": dataset,
        "scenario": scenario,
        "context": context,
        "horizon": horizon,
        "window_id": np.arange(len(local_mae), dtype=int),
        "local_model": local_model,
        "fm_model": fm_model,
        "local_feature_mode": local_feature_mode,
        "fm_feature_mode": fm_feature_mode,
        "local_mae": local_mae,
        "fm_mae": fm_mae,
        "absolute_gap_fm_minus_local": abs_gap,
        "percent_gap_fm_minus_local": pct_gap,
        "foundation_window_win": fm_mae < local_mae,
    })


def summarize_paired_windows(by_window: pd.DataFrame, n_boot: int = 3000, seed: int = SEED) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset", "scenario", "context", "horizon", "local_model", "fm_model", "local_feature_mode", "fm_feature_mode"]
    rng = np.random.default_rng(seed)
    for keys, g in by_window.groupby(group_cols, dropna=False):
        local_mae = g["local_mae"].to_numpy(float)
        fm_mae = g["fm_mae"].to_numpy(float)
        abs_gap = g["absolute_gap_fm_minus_local"].to_numpy(float)
        pct_gap = g["percent_gap_fm_minus_local"].to_numpy(float)
        idx = rng.integers(0, len(g), size=(n_boot, len(g)))
        boot_pct = 100.0 * (fm_mae[idx].mean(axis=1) - local_mae[idx].mean(axis=1)) / np.maximum(local_mae[idx].mean(axis=1), 1e-9)
        boot_abs = abs_gap[idx].mean(axis=1)
        tests = _optional_tests(abs_gap)
        p_fm = np.nan_to_num(tests.get("p_value_fm_better"), nan=1.0)
        p_local = np.nan_to_num(tests.get("p_value_local_better"), nan=1.0)
        scenario_pct_gap = float(100.0 * (fm_mae.mean() - local_mae.mean()) / max(local_mae.mean(), 1e-9))
        significant_fm = bool(p_fm < 0.05 and scenario_pct_gap < 0)
        significant_local = bool(p_local < 0.05 and scenario_pct_gap > 0)
        rows.append({
            **dict(zip(group_cols, keys)),
            "n_windows": int(len(g)),
            "local_mae_mean": float(local_mae.mean()),
            "local_mae_median": float(np.median(local_mae)),
            "fm_mae_mean": float(fm_mae.mean()),
            "fm_mae_median": float(np.median(fm_mae)),
            "absolute_gap_mean": float(abs_gap.mean()),
            "absolute_gap_median": float(np.median(abs_gap)),
            "percent_gap_mean": scenario_pct_gap,
            "percent_gap_median": float(np.median(pct_gap)),
            "percent_gap_ci95_low": float(np.percentile(boot_pct, 2.5)),
            "percent_gap_ci95_high": float(np.percentile(boot_pct, 97.5)),
            "absolute_gap_ci95_low": float(np.percentile(boot_abs, 2.5)),
            "absolute_gap_ci95_high": float(np.percentile(boot_abs, 97.5)),
            "foundation_window_win_rate": float(g["foundation_window_win"].mean()),
            "foundation_scenario_win": bool(abs_gap.mean() < 0),
            "local_scenario_win": bool(abs_gap.mean() > 0),
            "significant_foundation_win": significant_fm,
            "significant_local_win": significant_local,
            **tests,
        })
    return pd.DataFrame(rows)


def aggregate_summary(stats_df: pd.DataFrame, model_results: pd.DataFrame | None = None) -> pd.DataFrame:
    valid = stats_df.dropna(subset=["percent_gap_mean"]).copy()
    if valid.empty:
        return pd.DataFrame([{
            "n_scenarios": 0,
            "foundation_wins": 0,
            "local_wins": 0,
            "significant_foundation_wins": 0,
            "significant_local_wins": 0,
            "average_percent_gap": np.nan,
            "average_percent_gap_by_dataset": "",
            "average_percent_gap_by_cadence": "",
            "average_foundation_window_win_rate": np.nan,
            "average_rank_by_model_family": "",
        }])
    rank_text = ""
    if model_results is not None and {"dataset", "scenario", "context", "horizon", "family", "mae"}.issubset(model_results.columns):
        ranks = []
        for _, g in model_results.dropna(subset=["mae"]).groupby(["dataset", "scenario", "context", "horizon"]):
            fam = g.groupby("family")["mae"].min().rank(method="average").reset_index(name="rank")
            ranks.append(fam)
        if ranks:
            rank_df = pd.concat(ranks, ignore_index=True).groupby("family")["rank"].mean().sort_values()
            rank_text = "; ".join(f"{fam}: {rank:.2f}" for fam, rank in rank_df.items())
    by_dataset = valid.groupby("dataset")["percent_gap_mean"].mean().sort_index()
    dataset_text = "; ".join(f"{k}: {v:.2f}%" for k, v in by_dataset.items())
    cadence_col = "cadence" if "cadence" in valid.columns else None
    cadence_text = ""
    if cadence_col:
        by_cadence = valid.groupby(cadence_col)["percent_gap_mean"].mean().sort_index()
        cadence_text = "; ".join(f"{k}: {v:.2f}%" for k, v in by_cadence.items())
    return pd.DataFrame([{
        "n_scenarios": int(len(valid)),
        "foundation_wins": int(valid["foundation_scenario_win"].sum()),
        "local_wins": int(valid.get("local_scenario_win", ~valid["foundation_scenario_win"]).sum()),
        "significant_foundation_wins": int(valid["significant_foundation_win"].sum()),
        "significant_local_wins": int(valid.get("significant_local_win", pd.Series(False, index=valid.index)).sum()),
        "average_percent_gap": float(valid["percent_gap_mean"].mean()),
        "average_percent_gap_by_dataset": dataset_text,
        "average_percent_gap_by_cadence": cadence_text,
        "average_foundation_window_win_rate": float(valid["foundation_window_win_rate"].mean()),
        "average_rank_by_model_family": rank_text,
    }])


def write_figures(stats_df: pd.DataFrame, model_results: pd.DataFrame, fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 16, "axes.labelsize": 14})
    ok = stats_df.dropna(subset=["percent_gap_mean"]).copy()
    if not ok.empty:
        ok["label"] = ok["dataset"].astype(str) + "\n" + ok["scenario"].astype(str)
        xerr = np.vstack([
            ok["percent_gap_mean"] - ok["percent_gap_ci95_low"],
            ok["percent_gap_ci95_high"] - ok["percent_gap_mean"],
        ])
        fig, ax = plt.subplots(figsize=(11, max(5, 0.65 * len(ok))))
        colors = ["#2F6F9F" if v < 0 else "#B75D69" for v in ok["percent_gap_mean"]]
        ax.barh(ok["label"], ok["percent_gap_mean"], xerr=xerr, color=colors, alpha=0.9, capsize=4)
        ax.axvline(0, color="#333333", linewidth=1)
        ax.set_xlabel("Percent MAE gap: FM minus local (negative favors FM)")
        ax.set_title("Paired Percent Gap with Bootstrap 95% CI")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        path = fig_dir / "paired_percent_gap_ci_v2.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    if not model_results.empty and {"dataset", "context", "horizon", "family", "mae"}.issubset(model_results.columns):
        best = model_results.dropna(subset=["mae"]).groupby(["dataset", "context", "horizon", "family"])["mae"].min().reset_index()
        for dataset, g in best.groupby("dataset"):
            pivot = g.groupby(["context", "horizon"])["mae"].min().unstack("horizon")
            if pivot.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 5.5))
            im = ax.imshow(pivot.to_numpy(float), aspect="auto", cmap="viridis")
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([str(c) for c in pivot.columns])
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels([str(c) for c in pivot.index])
            ax.set_xlabel("Horizon")
            ax.set_ylabel("Context")
            ax.set_title(f"{dataset}: Best MAE by Context/Horizon")
            fig.colorbar(im, ax=ax, label="MAE")
            fig.tight_layout()
            path = fig_dir / f"context_horizon_heatmap_{safe_name(dataset)}.png"
            fig.savefig(path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
    if "foundation_window_win_rate" in stats_df and not ok.empty:
        fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(ok))))
        ax.barh(ok["label"], ok["foundation_window_win_rate"], color="#54A24B")
        ax.axvline(0.5, color="#333333", linewidth=1, linestyle="--")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Per-window FM win rate")
        ax.set_title("Foundation Win Rate by Paired Window")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        path = fig_dir / "paired_window_win_rates_v2.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def safe_name(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text)).strip("_")


def markdown_report(stats_df: pd.DataFrame, aggregate: pd.DataFrame, generated_files: Iterable[Path]) -> str:
    agg = aggregate.iloc[0].to_dict()
    lines = [
        "# Full Paired Benchmark Report V2",
        "",
        "This v2 layer adds paired per-window statistical analysis, confidence intervals, significance tests when available, and ablation-ready context/horizon outputs.",
        "",
        "## Aggregate",
        "",
        f"- Scenarios analyzed: {int(agg.get('n_scenarios', 0))}",
        f"- Foundation wins: {int(agg.get('foundation_wins', 0))}",
        f"- Significant foundation wins: {int(agg.get('significant_foundation_wins', 0))}",
        f"- Average percent gap, FM minus local: {float(agg.get('average_percent_gap', np.nan)):.2f}%",
        f"- Average per-window FM win rate: {float(agg.get('average_foundation_window_win_rate', np.nan)):.2f}",
        "",
        "## Scenario Statistics",
        "",
    ]
    cols = ["dataset", "scenario", "context", "horizon", "percent_gap_mean", "percent_gap_ci95_low", "percent_gap_ci95_high", "foundation_window_win_rate", "p_value", "significant_foundation_win"]
    if stats_df.empty:
        lines.append("No paired per-window predictions were available.")
    else:
        lines.append(stats_df[cols].to_markdown(index=False) if _has_tabulate() else stats_df[cols].to_csv(index=False))
    lines.extend(["", "## Generated Files", ""])
    lines.extend(f"- `{p}`" for p in generated_files)
    return "\n".join(lines) + "\n"


def _has_tabulate() -> bool:
    import importlib.util
    return importlib.util.find_spec("tabulate") is not None
