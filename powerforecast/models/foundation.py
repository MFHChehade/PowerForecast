"""Pre-trained time-series foundation models, evaluated zero-shot.

The ``load_*``/``run_*`` functions are verbatim from the historical
``foundation_benchmark.py`` (Chronos-Bolt, TimesFM 2.5, MOMENT) and
``alcf_model_zoo.py`` (Moirai); :class:`FoundationForecaster` wraps them behind
the :class:`~powerforecast.models.base.Forecaster` interface with an explicit
load/unload lifecycle (one model at a time on an 8 GB GPU).
"""
from __future__ import annotations

import gc
import time

import numpy as np

from .base import Forecaster

CHRONOS_MODELS = [
    ("ChronosBoltTiny", "amazon/chronos-bolt-tiny"),
    ("ChronosBoltMini", "amazon/chronos-bolt-mini"),
    ("ChronosBoltSmall", "amazon/chronos-bolt-small"),
    ("ChronosBoltBase", "amazon/chronos-bolt-base"),
]
TIMESFM_MODEL = ("TimesFM_2p5_200M", "google/timesfm-2.5-200m-pytorch")
MOMENT_MODEL = ("MOMENT_1_small", "AutonLab/MOMENT-1-small")

CHRONOS_IDS = dict(CHRONOS_MODELS)


def load_chronos_bolt(model_id="amazon/chronos-bolt-tiny"):
    import torch
    from chronos import ChronosBoltPipeline

    return ChronosBoltPipeline.from_pretrained(model_id, device_map="cpu", dtype=torch.float32)


def run_chronos_bolt(pipe, x, horizon, batch_size=24):
    import torch
    start = time.time()
    preds = []
    for i in range(0, len(x), batch_size):
        batch = [torch.tensor(row, dtype=torch.float32) for row in x[i : i + batch_size]]
        with torch.no_grad():
            q = pipe.predict(batch, prediction_length=horizon)
        # Official Bolt models return 0.1 ... 0.9 quantiles; index 4 is median.
        preds.append(q[:, 4, :].detach().cpu().numpy())
    return np.vstack(preds), time.time() - start


def load_timesfm(max_context, max_horizon):
    import timesfm

    model = timesfm.TimesFM_2p5_200M_torch._from_pretrained(
        model_id=TIMESFM_MODEL[1],
        revision=None,
        cache_dir=None,
        force_download=False,
        local_files_only=False,
        token=None,
        config=None,
        torch_compile=False,
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=max_horizon,
            normalize_inputs=False,
            per_core_batch_size=8,
        )
    )
    return model


def run_timesfm(model, x, horizon, batch_size=16):
    start = time.time()
    preds = []
    fallback_windows = 0
    for i in range(0, len(x), batch_size):
        inputs = [row.astype(np.float32) for row in x[i : i + batch_size]]
        point, _ = model.forecast(horizon=horizon, inputs=inputs)
        point = np.asarray(point, dtype=np.float32)
        if not np.isfinite(point).all():
            fallback = model.model.forecast_naive(horizon, inputs)
            point = np.stack([row[:horizon, 5] if row.ndim == 2 else row[:horizon] for row in fallback]).astype(np.float32)
            fallback_windows += len(inputs)
        preds.append(point)
    model._last_fallback_windows = fallback_windows
    return np.vstack(preds), time.time() - start


def load_moment(max_horizon):
    from momentfm import MOMENTPipeline

    model = MOMENTPipeline.from_pretrained(
        MOMENT_MODEL[1],
        task_name="forecasting",
        forecast_horizon=max_horizon,
    )
    model.init()
    model.eval()
    return model


def run_moment(model, x, horizon, seq_len=512, batch_size=24):
    import torch
    start = time.time()
    preds = []
    for i in range(0, len(x), batch_size):
        batch = x[i : i + batch_size].astype(np.float32)
        padded = np.zeros((len(batch), seq_len), dtype=np.float32)
        mask = np.zeros((len(batch), seq_len), dtype=np.float32)
        take = min(batch.shape[1], seq_len)
        padded[:, -take:] = batch[:, -take:]
        mask[:, -take:] = 1.0
        xt = torch.tensor(padded[:, None, :], dtype=torch.float32)
        mt = torch.tensor(mask, dtype=torch.float32)
        with torch.no_grad():
            out = model.short_forecast(x_enc=xt, input_mask=mt, forecast_horizon=horizon)
        preds.append(out.forecast[:, 0, :].detach().cpu().numpy())
    return np.vstack(preds), time.time() - start


def load_moirai(context, horizon):
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    module = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small")
    return MoiraiForecast(module=module, prediction_length=horizon, context_length=context,
                          patch_size=32, num_samples=20, target_dim=1,
                          feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)


def run_moirai(model, Xte, horizon):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev); preds = []
    for i in range(0, len(Xte), 64):
        b = Xte[i:i + 64]
        past = torch.tensor(b[:, :, None], dtype=torch.float32, device=dev)   # [B, T, 1]
        obs = torch.ones_like(past, dtype=torch.bool)                          # [B, T, 1]
        pad = torch.zeros(past.shape[:2], dtype=torch.bool, device=dev)        # [B, T]
        with torch.no_grad():
            out = model(past_target=past, past_observed_target=obs, past_is_pad=pad)
        arr = out.cpu().numpy()                       # [B, sample, future_time, *tgt]
        med = np.median(arr, axis=1)                  # [B, future_time, *tgt]
        med = med.reshape(med.shape[0], med.shape[1], -1)[:, :horizon, 0]
        preds.append(med)
    return np.vstack(preds)


class FoundationForecaster(Forecaster):
    """TimesFM 2.5, Chronos-Bolt (tiny/mini/small/base), MOMENT, Moirai — zero-shot.

    ``load`` pulls the pre-trained weights for the given (context, horizon);
    ``unload`` frees them so the next model fits in GPU memory.
    """

    family = "fm"

    def __init__(self, name):
        super().__init__(name)
        self._obj = None

    def load(self, context, horizon):
        name = self.name
        if name == "TimesFM_2p5":
            self._obj = load_timesfm(context, horizon)
        elif name == "MOMENT":
            self._obj = load_moment(horizon)
        elif name == "Moirai":
            self._obj = load_moirai(context, horizon)
        elif name.startswith("ChronosBolt"):
            self._obj = load_chronos_bolt(CHRONOS_IDS[name])
        else:
            raise ValueError(name)

    def predict(self, split, horizon):
        if self._obj is None:
            raise RuntimeError(f"{self.name}: call load(context, horizon) before predict")
        Xte = split["Xte"]
        name = self.name
        if name == "TimesFM_2p5":
            p, _ = run_timesfm(self._obj, Xte, horizon, batch_size=32)
        elif name == "MOMENT":
            p, _ = run_moment(self._obj, Xte, horizon, batch_size=32)
        elif name == "Moirai":
            p = run_moirai(self._obj, Xte, horizon)
        else:
            p, _ = run_chronos_bolt(self._obj, Xte, horizon, batch_size=64)
        return np.asarray(p)[:, :horizon]

    def unload(self):
        self._obj = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
