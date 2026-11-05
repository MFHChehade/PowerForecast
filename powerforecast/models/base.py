"""Common interface every forecaster in the zoo implements.

A forecaster receives the split dict produced by
:func:`powerforecast.evaluation.splits.make_split` and returns point forecasts of
shape ``[n_test, horizon]``. Local models (classical / ML / DL) fit inside
``predict`` on the train region only; foundation models are zero-shot and expose
an explicit ``load``/``unload`` lifecycle so one fits on an 8 GB GPU at a time.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Forecaster(ABC):
    """One model in the zoo: ``predict(split, horizon) -> [n_test, horizon]``."""

    family: str = ""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def predict(self, split: dict, horizon: int) -> np.ndarray:
        """Forecast every test window in ``split`` for ``horizon`` steps."""

    def load(self, context: int, horizon: int) -> None:
        """Acquire heavyweight resources (pre-trained weights). No-op for local models."""

    def unload(self) -> None:
        """Release heavyweight resources. No-op for local models."""

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"
