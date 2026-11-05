# PowerForecast

Benchmark code comparing pretrained time-series foundation models with models trained and tuned on each site's
own data, for data-center power and GPU-demand forecasting.

> This repository holds code only. Results are not included yet, and datasets and model weights are not
> redistributed. The work has not been peer reviewed, and some experiments are still being extended.

## What it does

- **Datasets:** 8 traces: MIT Supercloud, SURF Lisa, Google, Bitbrains, LNN, NLR H100, ALCF Polaris and ALCF Aurora.
- **Foundation models, run with no training:** TimesFM 2.0, 2.5 and 3.0; Chronos-Bolt Tiny, Mini, Small and Base;
  Chronos-2; Timestral 5M, 28M, 123M and 209M.
- **Site models, tuned per dataset (22):** Ridge, LightGBM, NLinear, DLinear, MLP, N-BEATS, NHITS, TiDE, TSMixer,
  TimeMixer, KAN, LSTM, GRU, TCN, BiTCN, PatchTST, iTransformer, TimesNet, VanillaTransformer, Informer, Autoformer and
  FEDformer. Each gets 20 Optuna TPE trials (LightGBM 15) and final runs with 3 seeds.
- **Follow-up experiments:** extra inputs, fine-tuning versus training from scratch, context length, training-set
  size, simulated series with a known best forecast, whether short forecasts match the start of long ones, horizons up
  to 336 steps, feature selection and outlier forecasting.

## Evaluation protocol

- Chronological 60/15/25 train, validation and test split, with identical test windows for every model.
- All choices, including context length and hyperparameters, are made on validation data. Test data is scored afterwards.
- Error is MASE. Comparisons use a paired bootstrap 95% confidence interval over test windows.

## Repository layout

```
powerforecast/   library: data loading, splits, metrics, forecaster classes
scripts/         experiments, analysis and Slurm files; see scripts/README.md
tests/           import, registry and golden-value tests (no GPU or data needed)
data/            where datasets go, with their sources; see data/README.md
outputs/         run output, not committed
```

## Install and test

Developed on Python 3.11 with PyTorch 2.6.

```bash
pip install -r requirements.txt
pip install -e ".[test]"
pytest
```

Foundation models and some site models need extra packages; see the notes in `requirements.txt`. Point the
package variables `PF_NF_PKGS`, `PF_TIMESFM3_PKGS` and `PF_TIMESTRAL_PKGS` at wherever they are installed.

## Running experiments

Get the datasets as described in [`data/README.md`](data/README.md), then follow the experiment order in
[`scripts/README.md`](scripts/README.md). Long runs save each finished cell, so they resume when resubmitted.

## Citation and contact

See [`CITATION.cff`](CITATION.cff). Contact: Mohamad Chehade, chehade@utexas.edu.

Code is under the [MIT License](LICENSE). Datasets and pretrained models keep their own licenses.
