"""Data-dependent tests: skip cleanly when the bundled parquets are not present
(they are gitignored), so a code-only clone still passes the rest of the suite.
"""
from __future__ import annotations

import pytest

EXPECTED_TARGETS = {
    "mit_supercloud": "power_draw_W",
    "surf_lisa": "total_observed_power_watts",
    "google_powerdata": "power_utilization",
    "bitbrains": "cpu_util_percent",
    "lnn_inference": "power_watts",
    "lnn_training": "power_watts",
    "nlr_h100_training": "power_watts",
    "alcf_polaris": "gpu_node_active",
    "alcf_aurora": "gpu_node_active",
}


@pytest.mark.parametrize("name,target", sorted(EXPECTED_TARGETS.items()))
def test_load_dataset(name, target):
    from powerforecast.data import load_dataset
    try:
        df, tgt, cadence = load_dataset(name)
    except (FileNotFoundError, KeyError, Exception) as exc:  # noqa: BLE001
        if "not" in str(exc).lower() or "no such" in str(exc).lower() or isinstance(exc, FileNotFoundError):
            pytest.skip(f"dataset {name} not present locally: {exc}")
        raise
    assert tgt == target
    assert len(df) > 0
    assert target in df.columns
