"""Unified access to every dataset.

One function, :func:`load_dataset`, returns every data center's demand trace in the same
shape -- a long dataframe, the name of the target column, and the sampling cadence in
seconds -- so all eight datasets are loaded the *same way* regardless of how many raw or
build files each one needs internally.

This is a thin façade over the existing per-source loaders (``powerforecast.data.windows`` for
MIT/SURF, ``powerforecast.data.extension`` for Google/Bitbrains, and the pre-built parquets for the
AI-power and ALCF traces). It does not change any computation or result; it exists so there
is a single, consistent entry point for loading data.

Example
-------
>>> from powerforecast.data import load_dataset, DATASETS
>>> df, target, cadence_s = load_dataset("bitbrains")
>>> df.columns, target, cadence_s
"""
from __future__ import annotations

import pandas as pd

from powerforecast import config

# name -> (kind, spec).  kind: "fb" (windows loader) | "ext" (extension loader) | "parquet"
DATASETS = {
    "mit_supercloud":    ("fb",  "prepare_mit"),
    "surf_lisa":         ("fb",  "prepare_surf"),
    "google_powerdata":  ("ext", "prepare_google"),
    "bitbrains":         ("ext", "prepare_bitbrains"),
    "lnn_inference":     ("parquet",
                          (config.EXTENSION_ROOT / "lnn_ai_datacenter" / "lnn_inference_power.parquet",
                           "power_watts", 1.0)),
    "lnn_training":      ("parquet",
                          (config.EXTENSION_ROOT / "lnn_ai_datacenter" / "lnn_training_power.parquet",
                           "power_watts", 1.0)),
    "nlr_h100_training": ("parquet",
                          (config.EXTENSION_ROOT / "ai_power_profiles" / "nlr_training_1s_resampled.parquet",
                           "power_watts", 1.0)),
    "alcf_polaris":      ("parquet",
                          (config.OUTPUTS_ROOT / "outputs_alcf_power_forecast" / "features" / "hourly_polaris.parquet",
                           "gpu_node_active", 3600.0)),
    "alcf_aurora":       ("parquet",
                          (config.OUTPUTS_ROOT / "outputs_alcf_power_forecast" / "features" / "hourly_aurora.parquet",
                           "gpu_node_active", 3600.0)),
}


def load_dataset(name: str):
    """Return ``(df, target_column, cadence_seconds)`` for a dataset in :data:`DATASETS`."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choices: {sorted(DATASETS)}")
    kind, spec = DATASETS[name]
    if kind == "fb":
        from powerforecast.data import windows
        df, target, seconds = getattr(windows, spec)()[:3]
        return df, target, seconds
    if kind == "ext":
        from powerforecast.data import extension as ext
        res = getattr(ext, spec)()      # (df, target, seconds, meta)
        return res[0], res[1], res[2]
    if kind == "parquet":
        path, target, cadence = spec
        return pd.read_parquet(path), target, cadence
    raise ValueError(kind)


if __name__ == "__main__":
    for n in DATASETS:
        try:
            df, tgt, c = load_dataset(n)
            print(f"{n:20s} rows={len(df):>9,}  target={tgt:<26s} cadence={c}s")
        except Exception as exc:
            print(f"{n:20s} (not loaded: {type(exc).__name__}: {exc})")
