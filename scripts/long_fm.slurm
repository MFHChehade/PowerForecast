#!/bin/bash
#SBATCH -J pf_longfm
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o outputs/fm_sweep/logs/slurm_%x_%j.out
# Foundation models at horizons of 32 and longer (inference only), one process per package set.
# Finished cells are skipped, so resubmitting resumes.
cd "$SLURM_SUBMIT_DIR"
source "$WORK/pf_tacc/venv/bin/activate"
unset TRANSFORMERS_CACHE
export HF_HOME=$WORK/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 PF_NF_PKGS=/nonexistent
export PF_HORIZONS=32,48,64,96,128,192,256,336
TM=$WORK/pf_tacc/timestral_pkgs; T3=$WORK/pf_tacc/timesfm3_pkgs
L=outputs/fm_sweep/logs/longfm_${SLURM_JOB_ID}
(python scripts/horizon_consistency.py --family bolt; python scripts/run_fm_sweep.py --group local --models Persistence; python scripts/run_fm_sweep.py --group fm --models chronos_2 chronos_bolt_base chronos_bolt_small chronos_bolt_mini chronos_bolt_tiny) > ${L}_chronos.log 2>&1 &
python scripts/run_fm_sweep.py --group fm --models timesfm_2p5_200m > ${L}_tfm25.log 2>&1 &
PF_TIMESFM3_PKGS=$T3 python scripts/run_fm_sweep_timesfm3.py > ${L}_tfm3.log 2>&1 &
PF_TIMESTRAL_PKGS=$TM python scripts/run_fm_sweep_timestral.py --sizes 5M 209M 28M 123M > ${L}_timestral.log 2>&1 &
wait
