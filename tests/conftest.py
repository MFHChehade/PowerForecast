"""Shared test fixtures: make the repo importable and build a deterministic series.

The golden tests below depend only on synthetic, fixed-seed inputs, so the whole
suite runs on a fresh clone with no datasets present. Tests that do need the
bundled parquets skip themselves when the data is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def synthetic_series() -> np.ndarray:
    """A fixed daily-seasonal + noise trace; identical on every machine and run."""
    rng = np.random.RandomState(0)
    return 50 + 10 * np.sin(np.arange(4000) / 24 * 2 * np.pi) + rng.randn(4000) * 3
