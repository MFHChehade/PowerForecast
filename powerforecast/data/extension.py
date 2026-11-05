"""Loaders for the three additional data-center / cloud workload datasets used in
the extension pilot: Alibaba Cluster Trace, Google Cluster Trace, Bitbrains.

IMPORTANT FRAMING
-----------------
- MIT Supercloud and SURF Lisa are *measured* GPU/HPC power traces (handled by
  ``powerforecast.data.windows``, not here).
- Alibaba / Google / Bitbrains are *workload / resource-utilization* traces.
  They are NOT measured power. We build transparent aggregate demand series and
  an optional "proxy power" series. Never call these measured power.

Each loader returns a standardized tuple:
    (df, target, expected_seconds, meta)
where
    df   : DataFrame with columns ["series", "timestamp", target]
    target            : name of the target column
    expected_seconds  : nominal cadence in seconds (for window break detection)
    meta              : dict of metadata for datacenter_extension_dataset_metadata.csv

EASIEST ON-RAMP
---------------
Drop a pre-aggregated file at:
    <EXTENSION_ROOT>/<name>/aggregated_<name>.parquet   (or .csv)
with columns:  timestamp, cpu_util_percent[, mem_util_percent]
and the loaders will use it directly. Otherwise they parse the raw trace files
documented in expected_files(); if neither is present they raise
DatasetUnavailable with placement instructions.

Proxy power (optional, transparent):
    proxy_power = ALPHA * cpu_util_percent + BETA * mem_util_percent
Defaults ALPHA=0.7, BETA=0.3 (unitless 0..100 proxy). Sensitivity note: results
should be checked against ALPHA in {0.5, 0.7, 0.9}; the qualitative ordering of
foundation vs local models is not expected to depend on the exact split because
cpu and mem utilization are strongly correlated in these traces.
"""
from __future__ import annotations

from powerforecast import config

import numpy as np
import pandas as pd

# Co-located with the existing MIT/SURF parquet files.
DC_ROOT = config.DATA_ROOT
EXTENSION_ROOT = DC_ROOT / "extension_data"

ALPHA = 0.7  # cpu weight for proxy power
BETA = 0.3   # mem weight for proxy power

PRIMARY_TARGET = "cpu_util_percent"


class DatasetUnavailable(Exception):
    """Raised when neither an aggregated file nor raw trace files are present."""


def expected_files() -> dict:
    """Exact paths/files the user should provide for each dataset."""
    return {
        "alibaba": {
            "aggregated": str(EXTENSION_ROOT / "alibaba" / "aggregated_alibaba.parquet"),
            "raw": str(EXTENSION_ROOT / "alibaba" / "machine_usage.csv"),
            "source": "Alibaba cluster-trace-v2018 (https://github.com/alibaba/clusterdata)",
            "raw_schema": "headerless CSV: machine_id,time_stamp,cpu_util_percent,mem_util_percent,mem_gps,mkpi,net_in,net_out,disk_io_percent",
            "note": "time_stamp is seconds from trace start. cpu_util_percent/mem_util_percent already in 0..100.",
        },
        "google": {
            "aggregated": str(EXTENSION_ROOT / "google" / "aggregated_google.parquet"),
            "raw": str(EXTENSION_ROOT / "google" / "task_usage" / "part-*.csv"),
            "source": "Google ClusterData 2011 task_usage (gs://clusterdata-2011-2)",
            "raw_schema": "headerless CSV part files; col5=mean CPU usage rate, col6=canonical memory usage, col0=start_time(microseconds)",
            "note": "Aggregated per 5-min bucket (records are 5-min aggregates). Utilization normalized to 0..100 of observed max.",
        },
        "bitbrains": {
            "aggregated": str(EXTENSION_ROOT / "bitbrains" / "aggregated_bitbrains.parquet"),
            "raw": str(EXTENSION_ROOT / "bitbrains" / "fastStorage" / "*.csv"),
            "source": "Bitbrains GWA-T-12 (http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains)",
            "raw_schema": "per-VM ';'-separated CSVs with header incl. 'Timestamp [ms]','CPU usage [%]','Memory usage [KB]','Memory capacity provisioned [KB]'",
            "note": "Timestamp is epoch seconds (despite the [ms] label in some mirrors); cadence ~300s.",
        },
    }


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _finalize(agg: pd.DataFrame, target: str, expected_seconds: int, name: str,
              cadence_steps: int, source: str, raw_or_agg: str, n_series: int | None) -> tuple:
    """agg has columns: timestamp, cpu_util_percent[, mem_util_percent]."""
    agg = agg.sort_values("timestamp").reset_index(drop=True)
    if "mem_util_percent" in agg.columns:
        agg["proxy_power"] = ALPHA * agg["cpu_util_percent"] + BETA * agg["mem_util_percent"]
    else:
        agg["proxy_power"] = agg["cpu_util_percent"]
    if cadence_steps > 1:
        agg = agg.iloc[::cadence_steps].reset_index(drop=True)
        expected_seconds = expected_seconds * cadence_steps
    agg["series"] = "aggregate"
    df = agg[["series", "timestamp", target]].dropna().reset_index(drop=True)
    meta = {
        "dataset": name,
        "trace_type": "workload/resource-utilization (proxy-demand)",
        "measured_power": False,
        "target_variable": target,
        "aggregation_method": "mean across machines/VMs/tasks per timestamp bucket",
        "sampling_interval_seconds": expected_seconds,
        "n_time_steps": int(len(df)),
        "n_series_machines_vms": n_series if n_series is not None else "n/a",
        "missing_data_handling": "dropna after aggregation; gaps split windows via cadence break detection",
        "normalization": "none at load; per-window target-history-only models see raw utilization (0..100)",
        "proxy_power_formula": f"{ALPHA}*cpu_util_percent + {BETA}*mem_util_percent",
        "source": source,
        "loaded_from": raw_or_agg,
        "caveats": "Proxy-demand, NOT measured power. Do not merge into the measured-power (MIT/SURF) claim without qualification.",
    }
    return df, target, expected_seconds, meta


def _try_aggregated(name: str, target: str, default_seconds: int, cadence_steps: int) -> tuple | None:
    folder = EXTENSION_ROOT / name
    for ext in (".parquet", ".csv"):
        p = folder / f"aggregated_{name}{ext}"
        if p.exists():
            agg = pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
            agg["timestamp"] = pd.to_datetime(agg["timestamp"])
            agg = agg.sort_values("timestamp")
            # Infer cadence from the data rather than trusting a hardcoded default.
            diffs = agg["timestamp"].diff().dt.total_seconds().dropna()
            inferred = int(diffs.median()) if len(diffs) and diffs.median() > 0 else default_seconds
            if target not in agg.columns and target == "proxy_power":
                pass  # computed in _finalize
            n_series = int(agg["n_series"].iloc[0]) if "n_series" in agg.columns else None
            return _finalize(agg, target, inferred, name, cadence_steps,
                             expected_files()[name]["source"], str(p), n_series)
    return None


def _unavailable(name: str) -> "DatasetUnavailable":
    info = expected_files()[name]
    msg = (
        f"[{name}] no data found.\n"
        f"  Easiest: provide aggregated file -> {info['aggregated']}\n"
        f"           columns: timestamp, cpu_util_percent[, mem_util_percent][, n_series]\n"
        f"  Or raw trace files -> {info['raw']}\n"
        f"  Source: {info['source']}\n"
        f"  Raw schema: {info['raw_schema']}\n"
        f"  Note: {info['note']}"
    )
    return DatasetUnavailable(msg)


# --------------------------------------------------------------------------- #
# Alibaba cluster-trace-v2018
# --------------------------------------------------------------------------- #
def prepare_alibaba(cadence_steps: int = 1, target: str = PRIMARY_TARGET) -> tuple:
    ready = _try_aggregated("alibaba", target, default_seconds=60, cadence_steps=cadence_steps)
    if ready is not None:
        return ready
    raw = EXTENSION_ROOT / "alibaba" / "machine_usage.csv"
    if not raw.exists():
        raise _unavailable("alibaba")
    cols = ["machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent",
            "mem_gps", "mkpi", "net_in", "net_out", "disk_io_percent"]
    df = pd.read_csv(raw, header=None, names=cols, usecols=[0, 1, 2, 3])
    n_series = int(df["machine_id"].nunique())
    # Records sampled ~ every 10s; bucket to 60s for a regular cadence.
    df["bucket"] = (df["time_stamp"] // 60) * 60
    agg = df.groupby("bucket").agg(cpu_util_percent=("cpu_util_percent", "mean"),
                                   mem_util_percent=("mem_util_percent", "mean")).reset_index()
    origin = pd.Timestamp("2018-01-01")
    agg["timestamp"] = origin + pd.to_timedelta(agg["bucket"], unit="s")
    return _finalize(agg, target, 60, "alibaba", cadence_steps,
                     expected_files()["alibaba"]["source"], str(raw), n_series)


# --------------------------------------------------------------------------- #
# Google ClusterData 2011 task_usage
# --------------------------------------------------------------------------- #
def prepare_google(cadence_steps: int = 1, target: str = PRIMARY_TARGET) -> tuple:
    # Preferred path: PowerData 2019 MEASURED power utilization from the public GCS
    # bucket (column power_utilization). This is measured power, not workload proxy.
    for ext_ in (".parquet", ".csv"):
        p = EXTENSION_ROOT / "google" / f"aggregated_google{ext_}"
        if p.exists():
            agg = pd.read_parquet(p) if ext_ == ".parquet" else pd.read_csv(p)
            if "power_utilization" in agg.columns:
                agg["timestamp"] = pd.to_datetime(agg["timestamp"])
                agg = agg.sort_values("timestamp").reset_index(drop=True)
                diffs = agg["timestamp"].diff().dt.total_seconds().dropna()
                inferred = int(diffs.median()) if len(diffs) and diffs.median() > 0 else 300
                if cadence_steps > 1:
                    agg = agg.iloc[::cadence_steps].reset_index(drop=True)
                    inferred *= cadence_steps
                agg["series"] = "aggregate"
                tgt = "power_utilization"
                df = agg[["series", "timestamp", tgt]].dropna().reset_index(drop=True)
                n_dom = int(agg["n_domains"].median()) if "n_domains" in agg.columns else None
                meta = {
                    "dataset": "google", "trace_type": "measured data-center power utilization",
                    "measured_power": True, "target_variable": tgt,
                    "aggregation_method": "mean measured_power_util across power domains per 300s bucket; bad_measurement_data dropped",
                    "sampling_interval_seconds": inferred, "n_time_steps": int(len(df)),
                    "n_series_machines_vms": n_dom if n_dom is not None else "n/a",
                    "missing_data_handling": "rows flagged bad_measurement_data removed; dropna",
                    "normalization": "native 0..1 power-utilization fraction; target-history-only models",
                    "proxy_power_formula": "n/a (measured power)",
                    "source": "gs://powerdata_2019 (public GCS, anonymous) — Google PowerData 2019",
                    "loaded_from": str(p),
                    "caveats": "Measured data-center power utilization (May 2019, 57 power domains). Counts as a measured-power dataset.",
                }
                return df, tgt, inferred, meta
    # Fallback: legacy ClusterData 2011 task_usage workload proxy (if someone drops raw parts).
    ready = _try_aggregated("google", target, default_seconds=300, cadence_steps=cadence_steps)
    if ready is not None:
        return ready
    parts = sorted((EXTENSION_ROOT / "google" / "task_usage").glob("part-*.csv")) if (EXTENSION_ROOT / "google" / "task_usage").exists() else []
    if not parts:
        raise _unavailable("google")
    frames = []
    for p in parts:
        # col0=start_time(us), col5=mean CPU usage rate, col6=canonical memory usage
        d = pd.read_csv(p, header=None, usecols=[0, 5, 6], names=["start_us", "cpu_rate", "canon_mem"])
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["bucket"] = (df["start_us"] // (300 * 1_000_000)) * 300  # 5-min buckets (seconds)
    agg = df.groupby("bucket").agg(cpu_rate=("cpu_rate", "sum"), canon_mem=("canon_mem", "sum")).reset_index()
    # Normalize the summed demand to 0..100 of observed max (transparent proxy utilization).
    agg["cpu_util_percent"] = 100.0 * agg["cpu_rate"] / max(agg["cpu_rate"].max(), 1e-9)
    agg["mem_util_percent"] = 100.0 * agg["canon_mem"] / max(agg["canon_mem"].max(), 1e-9)
    origin = pd.Timestamp("2011-05-01")
    agg["timestamp"] = origin + pd.to_timedelta(agg["bucket"], unit="s")
    return _finalize(agg[["timestamp", "cpu_util_percent", "mem_util_percent"]],
                     target, 300, "google", cadence_steps,
                     expected_files()["google"]["source"], str(parts[0].parent), None)


# --------------------------------------------------------------------------- #
# Bitbrains GWA-T-12
# --------------------------------------------------------------------------- #
def prepare_bitbrains(cadence_steps: int = 1, target: str = PRIMARY_TARGET) -> tuple:
    ready = _try_aggregated("bitbrains", target, default_seconds=300, cadence_steps=cadence_steps)
    if ready is not None:
        return ready
    folder = EXTENSION_ROOT / "bitbrains" / "fastStorage"
    files = sorted(folder.glob("*.csv")) if folder.exists() else []
    if not files:
        raise _unavailable("bitbrains")
    frames = []
    for p in files:
        d = pd.read_csv(p, sep=";")
        d.columns = [c.strip() for c in d.columns]
        ts = d["Timestamp [ms]"].astype(float)
        cpu = d["CPU usage [%]"].astype(float)
        mem_cap = d.get("Memory capacity provisioned [KB]")
        mem_use = d.get("Memory usage [KB]")
        mem_util = (mem_use.astype(float) / mem_cap.replace(0, np.nan).astype(float) * 100.0) if mem_cap is not None and mem_use is not None else np.nan
        frames.append(pd.DataFrame({"ts": ts, "cpu_util_percent": cpu, "mem_util_percent": mem_util}))
    n_series = len(files)
    df = pd.concat(frames, ignore_index=True)
    df["bucket"] = (df["ts"] // 300) * 300
    agg = df.groupby("bucket").agg(cpu_util_percent=("cpu_util_percent", "mean"),
                                   mem_util_percent=("mem_util_percent", "mean")).reset_index()
    agg["timestamp"] = pd.to_datetime(agg["bucket"], unit="s")
    return _finalize(agg[["timestamp", "cpu_util_percent", "mem_util_percent"]],
                     target, 300, "bitbrains", cadence_steps,
                     expected_files()["bitbrains"]["source"], str(folder), n_series)


PREPARERS = {
    "alibaba": prepare_alibaba,
    "google": prepare_google,
    "bitbrains": prepare_bitbrains,
}
