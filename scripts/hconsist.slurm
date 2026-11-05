#!/bin/bash
#SBATCH -J pf_hcons
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:40:00
#SBATCH -o outputs/why/logs/slurm_%x_%j.out
# Horizon-consistency check for every foundation model, plus Timestral's other sizes with covariates.
cd "$SLURM_SUBMIT_DIR"
source "$WORK/pf_tacc/venv/bin/activate"
unset TRANSFORMERS_CACHE
export HF_HOME=$WORK/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
python scripts/horizon_consistency.py --feasible
python scripts/horizon_consistency.py --family bolt
python scripts/horizon_consistency.py --family chronos2
python scripts/horizon_consistency.py --family tfm25
PF_TIMESFM3_PKGS=$WORK/pf_tacc/timesfm3_pkgs python scripts/horizon_consistency.py --family tfm3
PF_TIMESTRAL_PKGS=$WORK/pf_tacc/timestral_pkgs python scripts/horizon_consistency.py --family timestral
PF_TIMESTRAL_PKGS=$WORK/pf_tacc/timestral_pkgs python scripts/run_fm_sweep_timestral.py --cov --sizes 5M 28M 123M
