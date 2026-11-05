"""PowerForecast: benchmarking classical, ML, deep-learning, and time-series
foundation models for short-horizon data-center power/demand forecasting.

Package layout
--------------
config       central, env-overridable paths
data         dataset loading: unified facade, window builders, extension sources
models       one class per model family behind the :class:`~powerforecast.models.base.Forecaster` ABC
evaluation   metrics, chronological splits, and the :class:`~powerforecast.evaluation.runner.ZooBenchmark` runner
analysis     paired-window statistics
"""
from __future__ import annotations

__version__ = "1.0.0"
