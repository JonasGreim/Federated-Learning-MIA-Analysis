#!/bin/bash
#SBATCH --job-name=mia_parallel
#SBATCH --output=logs/mia_%A_%a_%N.out
#SBATCH --error=logs/mia_%A_%a_%N.err
#SBATCH --partition=paula
#SBATCH --array=0-34%5            # 36 runs total; up to 10 run in parallel
#SBATCH --nodes=1                  # one node per run
#SBATCH --ntasks=1                 # one task per run
#SBATCH --cpus-per-task=8          # adjust to your Python workload
#SBATCH --gres=gpu:1               # one GPU per run
#SBATCH --mem=16G                  # memory per run
#SBATCH --time=04:00:00

set -euo pipefail
mkdir -p logs
LOG_DIR="logs/mia_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$LOG_DIR"
echo "Logs will be saved in: $LOG_DIR"
PROJECT_DIR="$HOME/masterarbeit/Flower2"
RUN_ID=$SLURM_ARRAY_TASK_ID

# activate conda environment
module purge
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate ~/conda/envs/flwr

echo "🚀 Starting MIA run${RUN_ID} on node $(hostname)"
echo "SLURM JobID: ${SLURM_JOB_ID}, ArrayTaskID: ${SLURM_ARRAY_TASK_ID}"

cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }

# --- Execute the run ---
python -u run_mia_experiment.py "mia=main_mia_experiments/run${RUN_ID}" \
  2>&1 | tee "${LOG_DIR}/run${RUN_ID}.log"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ Finished MIA run${RUN_ID} successfully."
else
  echo "❌ MIA run${RUN_ID} failed with exit code $EXIT_CODE."
fi


echo "Copying results to shared storage..."

RESULTS_SRC="$PROJECT_DIR/metrics/mia"   # or an absolute path if it’s outside the project
RESULTS_DEST="/work/hi85udaj-flower/results_mia/${SLURM_JOB_ID}"

mkdir -p "$RESULTS_DEST"

if [ -d "$RESULTS_SRC" ]; then
  echo "📦 Copying from $RESULTS_SRC to $RESULTS_DEST ..."
  cp -r "$RESULTS_SRC"/* "$RESULTS_DEST"/ || echo "⚠️ Some files could not be copied"
else
  echo "⚠️ Source folder not found: $RESULTS_SRC"
fi

echo "✅ Results saved to: $RESULTS_DEST"