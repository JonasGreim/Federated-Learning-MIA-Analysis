#!/bin/bash
#SBATCH --job-name=flower-mia
#SBATCH --output=logs/mia_%j.out
#SBATCH --error=logs/mia_%j.err
#SBATCH --partition=paula
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1

# --- Environment Setup ---
set -euo pipefail
module purge
mkdir -p logs

# activate conda environment
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate ~/conda/envs/flwr

# check python
which python
python -V

# check wandb login
python - <<'PY'
import wandb
print("wandb version:", wandb.__version__)
print("login ok:", wandb.login(relogin=False))  # uses saved key; returns True if configured
PY

# --- MIA ---
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Running on nodes:"
scontrol show hostnames $SLURM_JOB_NODELIST
PROJECT_DIR="$HOME/masterarbeit/Flower2"


echo start MIA
srun python -u run_mia_experiment.py "$@"


echo "✅ All MIA processes finished successfully."



