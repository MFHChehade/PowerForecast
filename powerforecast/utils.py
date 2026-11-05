"""Small shared utilities: seeding and Windows keep-awake."""
from __future__ import annotations

import random

import numpy as np

SEED = 3


def set_seed():
    """Seed python + numpy (seed 3). """
    random.seed(SEED)
    np.random.seed(SEED)


def set_seed_torch():
    """Seed python + numpy + torch and cap torch threads (historical foundation_benchmark.set_seed)."""
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)


def keep_awake():
    """Prevent Windows from sleeping during a long benchmark run (no-op elsewhere)."""
    try:
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass
