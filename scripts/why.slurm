#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o outputs/why/logs/slurm_%x_%j.out
# Generic runner for the follow-up experiments: WORKERS copies of `python scripts/$SCRIPT work $ARGS`
# share one GH200 node, each claiming tasks; resubmitting resumes.
cd "$SLURM_SUBMIT_DIR"
source "$WORK/pf_tacc/venv/bin/activate"
unset TRANSFORMERS_CACHE
export HF_HOME=$WORK/hf_cache HF_HUB_OFFLINE=1 PF_STUDY_SPLIT=1 PF_NF_PKGS=/nonexistent PF_NJOBS=4
export PF_TIMESTRAL_PKGS=$WORK/pf_tacc/timestral_pkgs PF_TIMESFM3_PKGS=$WORK/pf_tacc/timesfm3_pkgs PF_CORPUS=$WORK/pf_tacc/corpus
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
python scripts/why_local.py release
for w in $(seq 1 "${WORKERS:-8}"); do
  python scripts/$SCRIPT work --worker "$w" $ARGS > "outputs/why/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_w${w}.log" 2>&1 &
  sleep 3
done
wait
