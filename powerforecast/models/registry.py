"""Model registry: family membership and the factory that builds a Forecaster by name."""
from __future__ import annotations

from .base import Forecaster
from .classical import ClassicalForecaster
from .deep import DeepForecaster
from .foundation import FoundationForecaster
from .ml import MLForecaster

FAMILIES = {
    "classical": ["Persistence", "SeasonalNaive24", "MovingAvg6", "ExpSmoothing", "AutoETS", "Theta", "AutoARIMA"],
    "ml": ["RidgeLag", "HistGBM", "LightGBM", "XGBoost"],
    "dl": ["NLinear", "DLinear", "MLP", "PatchTST", "LSTM", "NBEATS"],
    "fm": ["TimesFM_2p5", "ChronosBoltTiny", "ChronosBoltMini", "ChronosBoltSmall", "ChronosBoltBase", "MOMENT", "Moirai"],
}
FAMILY_OF = {m: f for f, ms in FAMILIES.items() for m in ms}

_CLASS_OF = {
    "classical": ClassicalForecaster,
    "ml": MLForecaster,
    "dl": DeepForecaster,
    "fm": FoundationForecaster,
}


def create(name: str) -> Forecaster:
    """Build the Forecaster for a zoo model name, e.g. ``create('PatchTST')``."""
    if name not in FAMILY_OF:
        raise KeyError(f"unknown model {name!r}; choices: {sorted(FAMILY_OF)}")
    return _CLASS_OF[FAMILY_OF[name]](name)
