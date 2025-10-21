#!/bin/bash
#SBATCH --job-name=flower-cluster
#SBATCH --output=logs/flower_%j_%N.out
#SBATCH --error=logs/flower_%j_%N.err
#SBATCH --partition=paula
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:03:00
#SBATCH --gres=gpu:1

# --- Environment Setup ---
set -euo pipefail
mkdir -p logs

module purge
module load Anaconda3/2024.02-1

eval "$(conda shell.bash hook)"
conda activate ~/conda/envs/flwr

which python
python -V
nvidia-smi  # shows Driver Version and “CUDA Version”

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


python - <<'PY'
import ray, torch, os
ray.init(num_cpus=4, num_gpus=1)
@ray.remote(num_gpus=1)
def gpu_task():
    import torch
    return torch.cuda.is_available(), torch.cuda.get_device_name(0)
print(ray.get(gpu_task.remote()))
PY


conda deactivate
echo "✅ All Flower processes finished successfully."
