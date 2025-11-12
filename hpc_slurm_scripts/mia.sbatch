#!/bin/bash
#SBATCH --job-name=mia_parallel
#SBATCH --output=logs/mia_%A_%a.out
#SBATCH --error=logs/mia_%A_%a.err
#SBATCH --partition=paula
#SBATCH --nodes=1                  # one node per run
#SBATCH --ntasks=1                 # one task per run
#SBATCH --cpus-per-task=8          # adjust to your Python workload
#SBATCH --gres=gpu:1               # one GPU per run
#SBATCH --mem=16G                  # memory per run

# 0-35: main experiments
# 36-41: clients experiments (same save folder + change wandb run)
# 42-46: shadow experiments (same save folder + change wandb run)

set -euo pipefail
mkdir -p logs
LOG_DIR="logs/mia_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$LOG_DIR"
echo "Logs will be saved in: $LOG_DIR"

MIA_RUN_CONFIG_FOLDER="${MIA_RUN_CONFIG_FOLDER:-main_mia_experiments}"
RUN_ID=$SLURM_ARRAY_TASK_ID

# activate conda environment
module purge
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate ~/conda/envs/flwr

echo "🚀 Starting MIA run${RUN_ID} on node $(hostname)"
echo "SLURM JobID: ${SLURM_JOB_ID}, ArrayTaskID: ${SLURM_ARRAY_TASK_ID}"

PROJECT_DIR="$HOME/masterarbeit/Flower2"
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }

# --- Execute the run ---
python -u run_mia_experiment.py "mia=${MIA_RUN_CONFIG_FOLDER}/run${RUN_ID}" \
  2>&1 | tee "${LOG_DIR}/run${RUN_ID}.log"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ Finished MIA run${RUN_ID} successfully."
else
  echo "❌ MIA run${RUN_ID} failed with exit code $EXIT_CODE."
fi
