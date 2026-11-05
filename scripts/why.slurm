#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o outputs/why/logs/slurm_%x_%j.out
# Runs WORKERS copies of `python scripts/$SCRIPT work $ARGS` on one node; each claims tasks, and resubmitting resumes.
cd "$SLURM_SUBMIT_DIR"
source "${PF_VENV:-.venv}/bin/activate"
unset TRANSFORMERS_CACHE
export PF_STUDY_SPLIT=1 PF_NJOBS=4
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
python scripts/why_local.py release
for w in $(seq 1 "${WORKERS:-8}"); do
  python scripts/$SCRIPT work --worker "$w" $ARGS > "outputs/why/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_w${w}.log" 2>&1 &
  sleep 3
done
wait
