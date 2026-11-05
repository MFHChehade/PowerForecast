"""Resample NLR training runs to a 1 s cadence.

Source: NLR parquet with native 0.1 s cadence, training runs only. Each run is resampled to the mean power in
each 1-second bin and kept as a separate series_id (no concatenation).

Output: extension_data/ai_power_profiles/nlr_training_1s_resampled.parquet
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

from powerforecast import config

import numpy as np
import pandas as pd

EXT = config.EXTENSION_ROOT / "ai_power_profiles"
PARQUET_IN = EXT / "nlr_h100_power_profiles.parquet"
PARQUET_OUT = EXT / "nlr_training_1s_resampled.parquet"


def resample_to_1s():
    """Load NLR, filter training, resample each run to 1 s."""
    df = pd.read_parquet(PARQUET_IN)
    train_df = df[df.workload_name == "training"].copy()

    rows = []
    for sid, g in train_df.groupby("series_id"):
        v = g.set_index("timestamp")["power_watts"]
        # Resample to 1 s mean.
        v_1s = v.resample("1s").mean()
        # Drop NaN (only at boundaries if timestamps don't align to second boundary).
        v_1s = v_1s.dropna()
        if len(v_1s) < 30:
            continue
        rows.append(pd.DataFrame({
            "timestamp": v_1s.index,
            "power_watts": v_1s.values.astype(np.float32),
            "series_id": sid,
            "workload_name": "training",
            "workload_type": "training",
            "source_dataset": "NLR_H100_GenAI_PowerProfiles_1s_resample",
            "sampling_interval_seconds": 1.0,
        }))
    result = pd.concat(rows, ignore_index=True)
    result.to_parquet(PARQUET_OUT, index=False)
    return result


def main():
    df = resample_to_1s()
    print(f"Resampled: {df['series_id'].nunique()} runs, {len(df):,} rows")


if __name__ == "__main__":
    main()
