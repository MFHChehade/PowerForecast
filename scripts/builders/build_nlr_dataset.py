"""Build a compact NLR H100 GenAI power-profile dataset + suitability analysis.

Source: NLR GenAI Power Profiles (Vercellino et al., arXiv:2604.07345; OSTI 3025227),
downloaded zip at extension_data/ai_power_profiles/raw/nlr_dataset.zip.
Each parquet under 01_aggregated_datasets/<workload>/results/*.parquet is ONE measured
power run (column power[W], measured Watts, ~0.1-0.2s cadence). We keep each run as a
separate series_id (no blind concatenation of unrelated runs).

Outputs:
  extension_data/ai_power_profiles/nlr_h100_power_profiles.parquet
  outputs_ai_power_profile_extension/ai_power_dataset_suitability.md
  outputs_ai_power_profile_extension/ai_power_dataset_metadata.csv
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
for _p in (str(_REPO), str(_REPO / "scripts")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from powerforecast import config

import numpy as np
import pandas as pd

EXT = config.EXTENSION_ROOT / "ai_power_profiles"
ZIP = EXT / "raw" / "nlr_dataset.zip"
OUTDIR = config.OUTPUTS_ROOT / "outputs_ai_power_profile_extension"
PARQUET = EXT / "nlr_h100_power_profiles.parquet"
CADENCE_S = 0.1                       # README: aggregated to 0.1/0.2 s; use 0.1 s nominal
GRID = [(24, 6), (84, 12), (168, 24), (336, 48)]
ORIGIN = pd.Timestamp("2026-01-01")

# Per workload: (max runs to include, max rows per run cap) — keeps file/pilot bounded.
SELECT = {
    "training": (41, 40000, "training"),
    "inference_online_rate_llama3_70b": (20, 40000, "inference"),
    "inference_online_finite_llama3_70b": (40, 40000, "inference"),
    "inference_offline_llama3_70b": (40, 40000, "inference"),
}


def load_runs():
    z = zipfile.ZipFile(ZIP)
    agg = [n for n in z.namelist() if n.startswith("01_aggregated_datasets") and n.endswith(".parquet")]
    by_wl = {}
    for n in agg:
        by_wl.setdefault(n.split("/")[1], []).append(n)
    rows = []
    for wl, (maxruns, maxrows, wtype) in SELECT.items():
        files = sorted(by_wl.get(wl, []))[:maxruns]
        for i, n in enumerate(files):
            d = pd.read_parquet(io.BytesIO(z.read(n)))
            pcol = [c for c in d.columns if "power" in c.lower()][0]
            p = pd.to_numeric(d[pcol], errors="coerce").dropna().to_numpy()[:maxrows]
            if len(p) < 30:
                continue
            sid = f"{wl}__run{i:04d}"
            # Distinct, far-apart start per series so windows never cross runs.
            start = ORIGIN + pd.Timedelta(days=i) + pd.Timedelta(hours=hash(wl) % 24)
            ts = start + pd.to_timedelta(np.arange(len(p)) * CADENCE_S, unit="s")
            rows.append(pd.DataFrame({
                "timestamp": ts, "power_watts": p.astype(np.float32),
                "series_id": sid, "workload_name": wl, "workload_type": wtype,
                "source_dataset": "NLR_H100_GenAI_PowerProfiles",
                "sampling_interval_seconds": CADENCE_S,
            }))
    return pd.concat(rows, ignore_index=True)


def series_stats(df):
    recs = []
    for sid, g in df.groupby("series_id"):
        v = g["power_watts"].to_numpy(float)
        n = len(v)
        ac1 = float(np.corrcoef(v[:-1], v[1:])[0, 1]) if n > 2 and v.std() > 0 else np.nan
        vol = float(np.abs(np.diff(v)).mean() / v.std()) if v.std() > 0 else 0.0
        recs.append({"series_id": sid, "workload_name": g["workload_name"].iloc[0],
                     "workload_type": g["workload_type"].iloc[0], "length": n,
                     "mean_w": float(v.mean()), "std_w": float(v.std()),
                     "lag1_autocorr": ac1, "volatility_meanabsdiff_over_std": vol,
                     **{f"supports_{c}_{h}": bool(n >= c + h + 1) for c, h in GRID}})
    return pd.DataFrame(recs)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = load_runs()
    df.to_parquet(PARQUET, index=False)
    st = series_stats(df)
    st.to_csv(OUTDIR / "ai_power_dataset_metadata.csv", index=False)

    # Per-workload roll-up.
    lines = ["# NLR H100 GenAI Power Profiles — Suitability Analysis", "",
             f"Source: NLR GenAI Power Profiles (Vercellino et al., arXiv:2604.07345; OSTI 3025227).",
             f"Measured node power in Watts (WattAMeter), nominal cadence {CADENCE_S}s; each run kept as a separate series.",
             f"Total series included: {df['series_id'].nunique()} | total rows: {len(df):,}", "",
             "## Per-workload summary", ""]
    head = ("workload | type | n_series | len_min | len_med | len_max | mean_W | lag1_ac | volatility | "
            + " | ".join(f"sup{c}/{h}" for c, h in GRID))
    lines += [head, "|".join(["---"] * (head.count("|") + 1))]
    rollup = []
    for wl, g in st.groupby("workload_name"):
        sup = {f"{c}_{h}": int(g[f"supports_{c}_{h}"].sum()) for c, h in GRID}
        row = {"workload": wl, "type": g["workload_type"].iloc[0], "n_series": len(g),
               "len_min": int(g.length.min()), "len_med": int(g.length.median()), "len_max": int(g.length.max()),
               "mean_W": round(g.mean_w.mean(), 1), "lag1_ac": round(g.lag1_autocorr.mean(), 4),
               "volatility": round(g.volatility_meanabsdiff_over_std.mean(), 4), **sup}
        rollup.append(row)
        lines.append(f"{wl} | {row['type']} | {row['n_series']} | {row['len_min']} | {row['len_med']} | {row['len_max']} | {row['mean_W']} | {row['lag1_ac']} | {row['volatility']} | "
                     + " | ".join(str(sup[f'{c}_{h}']) for c, h in GRID))
    rollup = pd.DataFrame(rollup)

    lines += ["", "## Classification", ""]
    for _, r in rollup.iterrows():
        supports_full = r["336_48"] > 0
        supports_any = any(r[f"{c}_{h}"] > 0 for c, h in GRID)
        if r["len_med"] >= 384 and supports_full:
            cls = "**suitable for main forecasting benchmark** (long runs, supports all context/horizon pairs)"
        elif supports_any:
            cls = "**suitable only for short-profile forecasting** (short runs; only short context/horizon pairs)"
        else:
            cls = "**not usable for forecasting** (runs too short)"
        lines.append(f"- `{r['workload']}` ({r['type']}, median len {r['len_med']}, lag1-ac {r['lag1_ac']}, "
                     f"volatility {r['volatility']}): {cls}")
    lines += ["", "## Notes",
              "- Volatility = mean|Δ| / std. Higher = more dynamic; persistence-friendly series have low volatility and lag1-ac≈1.",
              "- Cadence is nominal 0.1s per the dataset README (some runs 0.2s); window grids are in steps, so cadence does not affect window counts.",
              "- Runs are NOT concatenated; each run is an independent series_id."]
    (OUTDIR / "ai_power_dataset_suitability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(rollup.to_string(index=False))
    print(f"\nwrote {PARQUET} ({PARQUET.stat().st_size/1e6:.1f} MB), series={df['series_id'].nunique()}, rows={len(df):,}")
    print(f"wrote {OUTDIR/'ai_power_dataset_suitability.md'}")


if __name__ == "__main__":
    main()
