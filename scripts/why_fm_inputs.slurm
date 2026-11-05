#!/bin/bash
#SBATCH -J pf_fmpart
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:15:00
#SBATCH -o outputs/why/logs/slurm_%x_%j.out
# Foundation-model halves of the feature-selection and outlier studies (inference only), run side by side,
# plus a 3-step check that Timestral's training path works before the long job.
cd "$SLURM_SUBMIT_DIR"
source "$WORK/pf_tacc/venv/bin/activate"
unset TRANSFORMERS_CACHE
export HF_HOME=$WORK/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 PF_NF_PKGS=/nonexistent
TM=$WORK/pf_tacc/timestral_pkgs; T3=$WORK/pf_tacc/timesfm3_pkgs
L=outputs/why/logs/fmpart_${SLURM_JOB_ID}
PF_SMOKE=1 PF_TIMESTRAL_PKGS=$TM python scripts/why_timestral.py one --task 5M finetune mit 24
PF_SMOKE=1 PF_TIMESTRAL_PKGS=$TM python scripts/why_timestral.py one --task 5M scratch aurora 24
(PF_TIMESFM3_PKGS=$T3 python scripts/why_featsel.py fm --family tfm3; PF_TIMESFM3_PKGS=$T3 python scripts/why_outliers.py fm --family tfm3) > ${L}_tfm3.log 2>&1 &
(PF_TIMESTRAL_PKGS=$TM python scripts/why_featsel.py fm --family timestral; PF_TIMESTRAL_PKGS=$TM python scripts/why_outliers.py fm --family timestral) > ${L}_ts.log 2>&1 &
(python scripts/why_featsel.py fm --family chronos2; python scripts/why_outliers.py fm --family chronos2; python scripts/why_outliers.py fm --family bolt; python scripts/why_outliers.py fm --family tfm25) > ${L}_other.log 2>&1 &
wait
