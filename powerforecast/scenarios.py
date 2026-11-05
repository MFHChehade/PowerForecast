"""Pilot-benchmark scenario construction (MIT / SURF windows + simple baselines).

Verbatim from the historical ``foundation_benchmark.make_scenario``; used by the
foundation pilot and the paired v2 benchmark scripts.
"""
from __future__ import annotations

from powerforecast.data.windows import make_windows, prepare_mit, prepare_surf, sample_test
from powerforecast.evaluation.metrics import bootstrap_metrics as metrics
from powerforecast.models.classical import exp_smoothing, persistence


def make_scenario(name, dataset, context, horizon, cadence_steps=1, days=None, max_test=240):
    if dataset == "surf":
        df, target, expected_seconds = prepare_surf(days=days or 30, cadence_steps=cadence_steps)
        cadence_label_seconds = expected_seconds
        context_hours = context * cadence_label_seconds / 3600
        horizon_hours = horizon * cadence_label_seconds / 3600
    elif dataset == "mit":
        df, target, expected_seconds = prepare_mit(cadence_steps=cadence_steps)
        cadence_label_seconds = float(df.sort_values(["series", "timestamp"]).groupby("series")["timestamp"].diff().median() * cadence_steps)
        context_hours = context * cadence_label_seconds / 3600
        horizon_hours = horizon * cadence_label_seconds / 3600
    else:
        raise ValueError(f"unknown dataset {dataset}")
    x, y, meta = make_windows(df, target=target, context=context, horizon=horizon, expected_seconds=expected_seconds)
    xte, yte, test_meta = sample_test(x, y, meta, max_test=max_test)
    rows = []
    for model, pred, seconds, family in [
        ("Persistence", persistence(xte, horizon), 0.0, "baseline"),
        ("ExpSmoothing", exp_smoothing(xte, horizon), 0.0, "baseline"),
    ]:
        rows.append({"model": model, "family": family, "seconds": seconds, **metrics(yte, pred)})
    return {
        "scenario": name,
        "dataset": dataset,
        "target": target,
        "days": days,
        "context": context,
        "horizon": horizon,
        "cadence_seconds_est": cadence_label_seconds,
        "cadence_minutes": cadence_label_seconds / 60,
        "context_hours": context_hours,
        "horizon_hours": horizon_hours,
        "windows_total": int(len(x)),
        "test_windows": int(len(xte)),
        "results": rows,
        "test_examples": test_meta.head(10).to_dict(orient="records"),
        "_xte": xte,
        "_yte": yte,
    }
