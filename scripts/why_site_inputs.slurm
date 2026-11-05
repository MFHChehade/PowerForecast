#!/bin/bash
#SBATCH -J pf_sitepart
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -o outputs/why/logs/slurm_%x_%j.out
# Site-model halves of the outlier and feature-selection studies: workers claim (trace, H) tasks.
cd "$SLURM_SUBMIT_DIR"
source "$WORK/pf_tacc/venv/bin/activate"
unset TRANSFORMERS_CACHE
export HF_HOME=$WORK/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PF_NF_PKGS=/nonexistent PF_STUDY_SPLIT=1 PF_NJOBS=4
L=outputs/why/logs/sitepart_${SLURM_JOB_ID}
for w in 1 2 3 4; do python scripts/why_outliers.py site --worker $w > ${L}_o$w.log 2>&1 & sleep 2; done
for w in 1 2 3 4 5 6 7 8; do python scripts/why_featsel.py site --worker $w > ${L}_f$w.log 2>&1 & sleep 2; done
wait
