#!/bin/bash
#SBATCH --job-name=mia_parallel
#SBATCH --output=${HOME}/slurm_logs/mia_%A_%a.out
#SBATCH --error=${HOME}/slurm_logs/mia_%A_%a.err
#SBATCH --partition=paula
#SBATCH --array=0-35%10            # 36 runs total; up to 10 run in parallel
#SBATCH --nodes=1                  # one node per run
#SBATCH --ntasks=1                 # one task per run
#SBATCH --cpus-per-task=8          # adjust to your Python workload
#SBATCH --gres=gpu:1               # one GPU per run
#SBATCH --mem=16G                  # memory per run
#SBATCH --time=04:00:00

set -euo pipefail
mkdir -p logs
LOG_DIR="logs/mia_$SLURM_JOB_ID"
mkdir -p "$LOG_DIR"
echo "Logs will be saved in: $LOG_DIR"
PROJECT_DIR="$HOME/masterarbeit/Flower2"

RUN_ID=$SLURM_ARRAY_TASK_ID

echo "🚀 Starting MIA run${RUN_ID} on node $(hostname)"
echo "SLURM JobID: ${SLURM_JOB_ID}, ArrayTaskID: ${SLURM_ARRAY_TASK_ID}"

cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }

# --- Execute the run ---
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --gres=gpu:1 --kill-on-bad-exit=1 bash -lc "
    python -u run_mia_experiment.py 'mia=main_mia_experiments/run${RUN_ID}'
  " 2>&1 | tee "${LOG_DIR}/run${RUN_ID}.log"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ Finished MIA run${RUN_ID} successfully."
else
  echo "❌ MIA run${RUN_ID} failed with exit code $EXIT_CODE."
fi
