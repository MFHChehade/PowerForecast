"""Checks of the evaluation protocol on made-up data. No GPU or dataset needed."""
from __future__ import annotations

import importlib

import numpy as np
import pytest

import run_fm_sweep as sw


@pytest.mark.parametrize("module", ["powerforecast", "powerforecast.config", "powerforecast.data", "run_fm_sweep"])
def test_module_imports(module):
    importlib.import_module(module)


def _cell(H=24, L=64):
    rng = np.random.RandomState(0)
    series = [50 + np.cumsum(rng.randn(4000)) for _ in range(2)]
    trace = "mit"
    return sw.Cell(trace, series, H, L), series


def test_targets_stay_in_their_split():
    """Train targets end inside the first 60%; test targets start in the final 25%."""
    cell, series = _cell()
    assert all(t + cell.H <= int(0.60 * len(series[s])) for s, t in cell.pos["tr"])
    assert all(t >= int(0.75 * len(series[s])) for s, t in cell.pos["te"])


def test_test_windows_do_not_depend_on_context():
    """Every context length is scored on the same test windows."""
    short, _ = _cell(L=32)
    long, _ = _cell(L=256)
    assert short.test_id == long.test_id
    np.testing.assert_array_equal(short.Yte, long.Yte)


def test_persistence_repeats_the_last_reading():
    cell, _ = _cell()
    pred, _, _ = sw.run_local(cell, "Persistence")
    np.testing.assert_array_equal(pred, np.repeat(cell.Xte[:, -1:], cell.H, 1))


def test_windows_are_cut_correctly():
    cell, series = _cell()
    s, t = cell.pos["te"][0]
    np.testing.assert_array_equal(cell.Xte[0], series[s][t - cell.L:t].astype(np.float32))
    np.testing.assert_array_equal(cell.Yte[0], series[s][t:t + cell.H].astype(np.float32))
