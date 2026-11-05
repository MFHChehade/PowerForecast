# PowerForecast

Code to compare pretrained time-series foundation models with models trained and tuned on each site's own data, for
data-center power and GPU-demand forecasting. It contains only what is needed to produce the main results.

> Results, datasets and model weights are not included. The work has not been peer reviewed.

## What it produces

| Main result | Output (`outputs/`) |
|---|---|
| Best foundation model vs best tuned site model, 8 datasets, 6 and 24 steps ahead | `dl_hpo/headline_tuned.csv` |
| All models ranked | `dl_hpo/average_rank.csv` |
| TimesFM-3.0 alone vs best tuned site model; effect of context length | `why/tfm3/headline_tfm3.csv`, `why/tfm3/context_tfm3.csv` |
| TimesFM-3.0 as is, fine-tuned and from scratch | `why/tfm3/threeway_tfm3.csv` |
| What extra inputs add for both families | `cov_hpo/covariate_headline.csv`, `cov_hpo/covariates_tfm3.csv` |
| All Timestral sizes vs site models and TimesFM-3.0 | `why/timestral.csv`, `why/timestral_cov.csv` |

## Setup

Use Python 3.11 and a GPU. The commands below assume a Slurm cluster, and we ran them on one TACC Vista GH200 node.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps neuralforecast==3.3.0                  # newer torch is requested, but the results used 2.6.0
pip install timesfm==2.0.0                                   # TimesFM-2.5
pip install --target "$PWD/vendor/timesfm3_pkgs" timesfm==3.0.2
git clone https://github.com/mistralai/timestral vendor/timestral_src   # commit a534bdb
export PF_TIMESFM3_PKGS=$PWD/vendor/timesfm3_pkgs
export PF_TIMESTRAL_PKGS=$PWD/vendor/timestral_src/src
mkdir -p outputs/fm_sweep/logs outputs/dl_hpo/logs outputs/cov_hpo/logs outputs/why/logs
```

Put the datasets where [`data/README.md`](data/README.md) says. LNN, NLR and the ALCF machines are built from raw files
with `scripts/builders/`; those builds were not timed. Model weights download from Hugging Face on first use, so
download them first if your compute nodes have no internet.

## What to run

Run these in order from the repository root, adding your allocation with `-A`. Every step resumes if you resubmit it.

| Step | Command | Time | What it does |
|---|---|---|---|
| 1 | `sbatch -A <acct> scripts/fm_sweep.slurm` | About 4–5 hours (one node) | Runs reference baselines and all foundation models at every context length, with and without extra inputs. No training. |
| 2 | `sbatch -A <acct> scripts/tacc_hpo.slurm` | At least 6–7 hours (one node), estimated | Tunes the 22 site models: 352 tasks, 20 trials each (LightGBM 15), then 3 seeds. |
| 3 | `sbatch -A <acct> scripts/tacc_cov.slurm` | About 3 hours (one node) | Tunes the 14 site models that accept extra inputs: 168 tasks. |
| 4 | `sbatch -A <acct> -J pf_tfm3 -p gh -t 06:00:00 --export=ALL,SCRIPT=why_tfm3.py,WORKERS=4 scripts/why.slurm` | About 15 hours (one node), so resubmit it twice | Runs TimesFM-3.0 as is, fine-tuned and from scratch: 48 runs. |
| 5 | The analysis commands below | A few minutes | Builds the tables above. |

```bash
python scripts/analyze_fm_sweep.py     # run this first: it chooses each model's context on validation
python scripts/analyze_dl_hpo.py
python scripts/analyze_tfm3.py
python scripts/analyze_threeway.py
python scripts/analyze_cov.py
python scripts/analyze_cov_tfm3.py
python scripts/analyze_timestral.py
```

**About the times:**
- Steps 3 and 4 are measured from our job records.
- Step 1 adds up the measured time of each model. TimesFM-3.0 with extra inputs takes about 4 hours of it, and everything
  else runs alongside it.
- Step 2 is an estimate scaled from step 3, because part of our search ran on another machine. It may take longer because
  it includes heavier transformer models.
- Step 2 and 3 use 16 workers sharing one GPU, and the worker count can be changed with `WORKERS`.

**TimesFM-2.0** is not in `fm_sweep.slurm`, because its older package was not available on Vista. We ran it separately with
`python scripts/run_fm_sweep_timesfm2.py`, with its package in a folder named by `PF_TIMESFM1_PKGS`. Without it, the "best
foundation model" table can change.

## How models are evaluated

- **Split:** chronological 60/15/25 train, validation and test, with the same test windows for every model.
- **Selection:** context length and hyperparameters are chosen on validation. Test data is scored afterwards.
- **Score:** MASE on 300 test windows, with a paired bootstrap 95% confidence interval.

## Which scripts do what

| Scripts | Role |
|---|---|
| `run_fm_sweep*.py`, `run_cov_fm.py` | Foundation models and reference baselines |
| `run_dl_hpo.py`, `run_ml_hpo.py`, `tacc_worker.py` | Site-model tuning |
| `covariates.py`, `run_cov_hpo.py` | Extra inputs |
| `why_tfm3.py` | Three-way test; `why_ft.py` and `why_local.py` are helpers it imports |
| `analyze_*.py` | Tables and figures |
| `powerforecast/` | Dataset loading |

Tests need no GPU or data: `pip install pytest && pytest`.

## Citation and contact

See [`CITATION.cff`](CITATION.cff). Contact: Mohamad Chehade, chehade@utexas.edu.

Code is under the [MIT License](LICENSE). Datasets and pretrained models keep their own licenses.
