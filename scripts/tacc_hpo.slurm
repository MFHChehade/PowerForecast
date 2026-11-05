#!/bin/bash
#SBATCH -J pf_hpo
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -o outputs/dl_hpo/logs/slurm_%j.out
# One GH200 node: WORKERS processes share the GPU (the models are small and launch-bound), each
# claiming (model, trace, horizon) tasks from scripts/tacc_worker.py. Resubmit to resume.
WORKERS=${WORKERS:-16}
cd "$SLURM_SUBMIT_DIR"
source "${PF_VENV:-.venv}/bin/activate"
export PF_STUDY_SPLIT=1 PF_SLOW_TRIALS=20 PF_NJOBS=4
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/tacc_worker.py release
for w in $(seq 1 "$WORKERS"); do
  python scripts/tacc_worker.py work --worker "$w" ${MODELS:+--models $MODELS} > "outputs/dl_hpo/logs/job${SLURM_JOB_ID}_w${w}.log" 2>&1 &
  sleep 5
done
wait
python scripts/tacc_worker.py status
