#!/bin/bash
#SBATCH --job-name=flower-cluster
#SBATCH --output=logs/flower_%j_%N.out
#SBATCH --error=logs/flower_%j_%N.err
#SBATCH --partition=paula
#SBATCH --nodes=3
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


# nvidia-smi  # shows Driver Version and “CUDA Version”
# check gpu
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
PY


# --- Distribute Flower ---
# Node 0 = Server, Nodes 1-9 = Clients
PROJECT_DIR="$HOME/masterarbeit/Flower2"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Running on nodes:"
scontrol show hostnames $SLURM_JOB_NODELIST

# Liste aller zugewiesenen Nodes
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
SERVER_NODE=${NODES[0]}
PORT_FLEET=9092   # SuperNodes -> SuperLink (Fleet API)
PORT_EXEC=9093    # flwr run -> SuperLink (Exec API)

echo "SuperLink: ${SERVER_NODE} (fleet:${PORT_FLEET}, exec:${PORT_EXEC})"
echo "SuperNodes: ${NODES[@]:1}"


# 1) Start SuperLink on server node (background)
srun --nodes=1 --nodelist="$SERVER_NODE" --exclusive bash -lc "
  cd '$PROJECT_DIR'
  mkdir -p logs
  flower-superlink \
    --insecure \
    --fleet-api-address 0.0.0.0:${PORT_FLEET} \
    --exec-api-address 0.0.0.0:${PORT_EXEC} \
    > logs/superlink_${SLURM_JOB_ID}.log 2>&1
" &
SL_PID=$!
sleep 3  # give it time to bind

# 2) Start SuperNodes (one per remaining node)
NUM_CLIENTS=$(( ${#NODES[@]} - 1 ))
for i in $(seq 1 $(( ${#NODES[@]} - 1 ))); do
  NODE=${NODES[$i]}
  PARTITION_ID=$((i-1))
  srun --nodes=1 --nodelist="$NODE" --exclusive bash -lc "
    cd '$PROJECT_DIR'
    mkdir -p logs
    flower-supernode \
      --insecure \
      --superlink ${SERVER_NODE}:${PORT_FLEET} \
      --node-config \"partition-id=${PARTITION_ID} num-partitions=${NUM_CLIENTS}\"
  " > "logs/supernode_${i}_${SLURM_JOB_ID}.log" 2>&1 &
done

# 3) Wait until SuperLink Exec API is ready (no netcat needed)
python - <<PY
import socket, time, sys
host="${SERVER_NODE}"
port=${PORT_EXEC}
for _ in range(120):
    try:
        with socket.create_connection((host, port), timeout=2):
            print("SuperLink Exec API reachable"); sys.exit(0)
    except OSError:
        time.sleep(1)
print("SuperLink Exec API NOT reachable"); sys.exit(1)
PY

# 4) Trigger the run from the server node (non-interactive)
# Make sure pyproject.toml has [tool.flwr.federations.hpc-deploy] address="${SERVER_NODE}:${PORT_EXEC}" insecure=true
srun --nodes=1 --nodelist="$SERVER_NODE" --exclusive bash -lc "
  cd '$PROJECT_DIR'
  export FLWR_FEDERATION__HPC_DEPLOY__ADDRESS=${SERVER_NODE}:${PORT_EXEC}
  flwr run . hpc-deploy --stream
" | tee "logs/flwr_run_${SLURM_JOB_ID}.log"

wait
echo "✅ All Flower processes finished successfully."
