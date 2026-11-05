"""Generate / hold the golden reference for the deterministic core.

The golden file pins the exact outputs of every deterministic code path — windowed
splits, and the classical + ML forecasters — on the fixed synthetic series from
``conftest``. ``test_golden.py`` recomputes these and asserts they are unchanged;
any drift means a refactor altered behavior.

Regenerate intentionally (only when a numeric change is expected and reviewed):

    python -m tests.golden            # writes tests/golden_reference.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_reference.json"

# Deterministic forecasters only: the statsforecast models (AutoETS/Theta/AutoARIMA)
# fit per-window and are slow; the trained DL/FM models are stochastic. Those are
# covered by separate smoke tests, not the bit-exact golden.
DETERMINISTIC_MODELS = ["Persistence", "SeasonalNaive24", "MovingAvg6", "ExpSmoothing",
                        "RidgeLag", "HistGBM", "LightGBM", "XGBoost"]
GRID = [(336, 1), (168, 24)]


def _hash(arr) -> str:
    a = np.ascontiguousarray(np.asarray(arr, np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def _synthetic() -> np.ndarray:
    rng = np.random.RandomState(0)
    return 50 + 10 * np.sin(np.arange(4000) / 24 * 2 * np.pi) + rng.randn(4000) * 3


def compute_golden() -> dict:
    """Compute the full golden dict from the deterministic core."""
    from powerforecast.evaluation.metrics import bootstrap_metrics, score
    from powerforecast.evaluation.splits import make_split
    from powerforecast.models import create

    y = _synthetic()
    out: dict = {}
    for ctx, hzn in GRID:
        S = make_split(y, ctx, hzn)
        out[f"split:{ctx}:{hzn}"] = {"Xte": _hash(S["Xte"]), "Yte": _hash(S["Yte"]),
                                     "scale": float(S["scale"])}
        for m in DETERMINISTIC_MODELS:
            p = create(m).predict(S, hzn)
            out[f"pred:{m}:{ctx}:{hzn}"] = {"hash": _hash(p),
                                            "MAE": score(S["Yte"], p, S["scale"])["MAE"]}
    # metric functions on fixed inputs
    rng = np.random.RandomState(7)
    yt = rng.randn(50, 6); pp = yt + rng.randn(50, 6) * 0.1
    out["metric:score"] = score(yt, pp, 1.0)
    out["metric:bootstrap"] = bootstrap_metrics(yt, pp)
    return out


def main() -> None:
    GOLDEN_PATH.write_text(json.dumps(compute_golden(), indent=1), encoding="utf-8")
    print(f"wrote {GOLDEN_PATH} ({len(json.loads(GOLDEN_PATH.read_text()))} keys)")


if __name__ == "__main__":
    main()
