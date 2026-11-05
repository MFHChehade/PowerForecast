#!/bin/bash
#SBATCH -J pf_fm
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -o outputs/fm_sweep/logs/slurm_%x_%j.out
# Step 1: reference baselines and all foundation models, without and with extra inputs (no training).
# Finished cells are skipped, so resubmitting resumes. Set PF_TIMESFM3_PKGS and PF_TIMESTRAL_PKGS first.
cd "$SLURM_SUBMIT_DIR"
source "${PF_VENV:-.venv}/bin/activate"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
L=outputs/fm_sweep/logs/fm_${SLURM_JOB_ID}
python scripts/run_fm_sweep.py --group info
(python scripts/run_fm_sweep.py --group local; python scripts/run_fm_sweep.py --group fm --models chronos_2 chronos_bolt_base chronos_bolt_small chronos_bolt_mini chronos_bolt_tiny; python scripts/run_cov_fm.py) > ${L}_chronos.log 2>&1 &
python scripts/run_fm_sweep.py --group fm --models timesfm_2p5_200m > ${L}_tfm25.log 2>&1 &
(python scripts/run_fm_sweep_timesfm3.py; python scripts/run_fm_sweep_timesfm3.py --cov) > ${L}_tfm3.log 2>&1 &
(python scripts/run_fm_sweep_timestral.py --sizes 5M 28M 123M 209M; python scripts/run_fm_sweep_timestral.py --cov --sizes 5M 28M 123M 209M) > ${L}_timestral.log 2>&1 &
wait
