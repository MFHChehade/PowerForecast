"""Behavior-preservation guard: the deterministic core must match the committed golden.

If any of these fail, a change altered a numeric result — investigate before
regenerating the golden file (see ``tests/golden.py``).
"""
from __future__ import annotations

import json

from tests.golden import GOLDEN_PATH, compute_golden


def _same_within_platform_noise(key, expected, actual):
    """RidgeLag solves a linear system, so its last bits differ between math libraries (Windows vs Linux).
    Its hash may change, but its error may not move by more than 1e-5 relative (measured: 1.1e-6)."""
    return key.startswith("pred:RidgeLag:") and abs(actual["MAE"] - expected["MAE"]) <= 1e-5 * abs(expected["MAE"])


def test_golden_matches_committed_reference():
    expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    actual = compute_golden()
    assert set(actual) == set(expected), "golden key set changed"
    mismatches = {k: (expected[k], actual[k]) for k in expected
                  if actual[k] != expected[k] and not _same_within_platform_noise(k, expected[k], actual[k])}
    assert not mismatches, f"golden drift in {list(mismatches)}"
