#!/bin/bash
#SBATCH --job-name=flower-cluster
#SBATCH --output=logs/flower_%j_%N.out
#SBATCH --error=logs/flower_%j_%N.err
#SBATCH --partition=paula
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1

# --- Environment Setup ---
# --nodes=3 -> set client number also in the experiment_config.flower.base.yaml

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
CLIENTS=("${NODES[@]:1}")
NUM_CLIENTS=${#CLIENTS[@]}
CLIENT_HOSTS=$(IFS=, ; echo "${CLIENTS[*]}")
PORT_FLEET=9092   # SuperNodes -> SuperLink (Fleet API)
PORT_EXEC=9093    # flwr run -> SuperLink (Exec API)

echo "Server node: $SERVER_NODE  Fleet:${PORT_FLEET}  Exec:${PORT_EXEC}"
echo "Client nodes: ${CLIENTS[*]}"
echo "Project dir: $PROJECT_DIR"


# 1) Start SuperLink on server node (background)
srun --nodes=1 --nodelist="$SERVER_NODE" bash -lc "
  cd '$PROJECT_DIR'
  mkdir -p logs
  flower-superlink \
    --insecure \
    --fleet-api-address 0.0.0.0:${PORT_FLEET} \
    --exec-api-address 0.0.0.0:${PORT_EXEC} \
    > logs/superlink_${SLURM_JOB_ID}.log 2>&1
" &
sleep 3  # give it time to bind

# 2) Start ALL SuperNodes in ONE srun (one task per client node)
#    Each task gets a unique PARTITION_ID via SLURM_PROCID (0..NUM_CLIENTS-1)
srun --nodes="$NUM_CLIENTS" --ntasks="$NUM_CLIENTS" --ntasks-per-node=1 --nodelist="$CLIENT_HOSTS" --exclusive bash -lc "
  cd '$PROJECT_DIR' && mkdir -p logs
  PARTITION_ID=\$SLURM_PROCID
  flower-supernode \
    --insecure \
    --superlink ${SERVER_NODE}:${PORT_FLEET} \
    --node-config \"partition-id=\${PARTITION_ID} num-partitions=${NUM_CLIENTS}\"
" &
# Note: omit --clientappio-api-address because it's one SuperNode per machine.

# 3) Wait until SuperLink Exec API is reachable
python - <<PY
import socket, time, sys
host="${SERVER_NODE}"; port=${PORT_EXEC}
for _ in range(180):
    try:
        with socket.create_connection((host, port), timeout=2):
            print("SuperLink Exec API reachable at %s:%s" % (host, port)); sys.exit(0)
    except OSError:
        time.sleep(1)
print("ERROR: SuperLink Exec API NOT reachable"); sys.exit(1)
PY


# 4) Trigger the run from the server node (non-interactive)
SERVER_ADDRESS="${SERVER_NODE}:${PORT_EXEC}" # Define the final address here

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --nodelist="$SERVER_NODE" --overlap bash -lc "
  cd '$PROJECT_DIR'

  # Inject server address to flower .toml (server address defined only at runtime)
  sed -i '/^\[tool\.flwr\.federations\.hpc-deploy\]/,/^\[/{
    s/^address *= *.*/address = \"${SERVER_ADDRESS}\"/
  }' pyproject.toml

  # Run Flower
  python -u run_flower_hpc.py flower=run1

  # revert back to placeholder address for future runs
  sed -i '/^\[tool\.flwr\.federations\.hpc-deploy\]/,/^\[/{
    s/^address *= *.*/address = \"SERVER_NODE:9093\"/
  }' pyproject.toml
" | tee "logs/flwr_run_${SLURM_JOB_ID}.log"

wait

echo "✅ All Flower processes finished successfully."
