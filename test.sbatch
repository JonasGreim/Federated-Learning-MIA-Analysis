#!/bin/bash
#SBATCH --job-name=flower-cluster
#SBATCH --output=logs/flower_%j_%N.out
#SBATCH --error=logs/flower_%j_%N.err
#SBATCH --partition=paula
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:01:00
#SBATCH --gres=gpu:1

# --- Environment Setup ---
set -euo pipefail
mkdir -p logs

module purge
module load Anaconda3/2024.02-1
module load CUDA/11.6.0

eval "$(conda shell.bash hook)"
conda activate ~/conda/envs/flwr

which python
python -V

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Running on nodes:"
scontrol show hostnames $SLURM_JOB_NODELIST

# --- Distribute Flower ---
# Node 0 = Server, Nodes 1-9 = Clients

# Liste aller zugewiesenen Nodes
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
SERVER_NODE=${NODES[0]}

echo "Server Node: $SERVER_NODE"
echo "Client Nodes: ${NODES[@]:1}"

conda deactivate
echo "✅ All Flower processes finished successfully."
