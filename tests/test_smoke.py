"""Structural smoke tests: everything imports, the registry is complete, every
forecaster produces a well-formed prediction. No external data required.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

PACKAGE_MODULES = [
    "powerforecast", "powerforecast.config", "powerforecast.utils", "powerforecast.scenarios",
    "powerforecast.data", "powerforecast.data.datasets", "powerforecast.data.windows",
    "powerforecast.data.extension",
    "powerforecast.models", "powerforecast.models.base", "powerforecast.models.classical",
    "powerforecast.models.ml", "powerforecast.models.deep", "powerforecast.models.foundation",
    "powerforecast.models.registry",
    "powerforecast.evaluation", "powerforecast.evaluation.metrics",
    "powerforecast.evaluation.splits", "powerforecast.evaluation.runner",
    "powerforecast.analysis", "powerforecast.analysis.paired_stats",
]


@pytest.mark.parametrize("module", PACKAGE_MODULES)
def test_module_imports(module):
    importlib.import_module(module)


def test_registry_builds_every_model():
    from powerforecast.models import FAMILIES, FAMILY_OF, create
    for family, names in FAMILIES.items():
        for name in names:
            fc = create(name)
            assert fc.family == family
            assert FAMILY_OF[name] == family


def test_registry_rejects_unknown_model():
    from powerforecast.models import create
    with pytest.raises(KeyError):
        create("NoSuchModel")


@pytest.mark.parametrize("model", ["Persistence", "SeasonalNaive24", "MovingAvg6",
                                   "ExpSmoothing", "RidgeLag", "NLinear", "DLinear", "MLP"])
def test_forecaster_prediction_shape(model, synthetic_series):
    from powerforecast.evaluation.splits import make_split
    from powerforecast.models import create
    horizon = 6
    split = make_split(synthetic_series, context=48, horizon=horizon)
    pred = create(model).predict(split, horizon)
    assert pred.shape == (len(split["Xte"]), horizon)
    assert np.isfinite(pred).all()
