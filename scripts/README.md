# Experiments

Run from the repository root. Long runs save each finished cell, so a job can be resubmitted and will resume.
Parallel workers claim tasks through folders under `outputs/`. Slurm files are written for TACC Vista: pass your
allocation with `sbatch -A <account>`, and change the paths for another cluster.

| Step | Purpose | Scripts |
|---|---|---|
| 1 | Foundation and local model sweep over size and context | `run_fm_sweep.py`, `run_fm_sweep_timesfm2.py`, `run_fm_sweep_timesfm3.py`, `run_fm_sweep_timestral.py`, `analyze_fm_sweep.py` |
| 2 | Tune the 22 site models | `run_ml_hpo.py`, `run_dl_hpo.py`, `tacc_worker.py`, `tacc_hpo.slurm`, `split_studies.py`, `analyze_dl_hpo.py` |
| 3 | Extra inputs for both families | `covariates.py`, `run_cov_fm.py`, `run_cov_hpo.py`, `tacc_cov.slurm`, `analyze_cov.py`, `analyze_cov_tfm3.py` |
| 4 | Context, training data, recency, pretraining and ensembles | `why_local.py`, `why_ft.py`, `why_pretrain.py`, `why_synth.py`, `why_synth2.py`, `why_analysis.py`, `why_results.py`, `why_figures.py`, `why.slurm` |
| 5 | The same tests for TimesFM-3.0 and Timestral | `why_tfm3.py`, `analyze_tfm3.py`, `why_timestral.py`, `analyze_timestral.py` |
| 6 | Horizon consistency | `horizon_consistency.py`, `analyze_horizon_consistency.py`, `hconsist.slurm` |
| 7 | Horizons 32 and longer: set `PF_HORIZONS` | `long_fm.slurm`, `tacc_hpo.slurm` |
| 8 | Feature selection and outliers | `why_featsel.py`, `why_outliers.py`, `why_fm_inputs.slurm`, `why_site_inputs.slurm` |

`dataset_sizes.py` counts readings and `tfm3_check.py` checks the TimesFM-3.0 install. `builders/` builds the LNN, NLR
and ALCF datasets from their raw files.
