#!/bin/bash
#SBATCH --job-name=flower-cluster
#SBATCH --output=logs/flower_%j.out
#SBATCH --error=logs/flower_%j.err
#SBATCH --partition=paula
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# --- Environment Setup ---

set -euo pipefail
module purge
mkdir -p logs
LOG_DIR="logs/flower_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"
echo "Logs will be saved in: $LOG_DIR"

# --- Conda / env ---
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate ~/conda/envs/flwr

which python
python -V

python - <<'PY'
import wandb
print("wandb version:", wandb.__version__)
print("login ok:", wandb.login(relogin=False))
PY

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Running on nodes:"
scontrol show hostnames "$SLURM_JOB_NODELIST"

# --- Cluster layout ---
PROJECT_DIR="$HOME/masterarbeit/Flower2"
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
SERVER_NODE=${NODES[0]}
CLIENTS=("${NODES[@]:1}")
NUM_CLIENTS=${#CLIENTS[@]}
CLIENT_HOSTS=$(IFS=, ; echo "${CLIENTS[*]}")
PORT_FLEET=9092   # SuperNodes -> SuperLink (Fleet API)
PORT_CTRL=9093    # flwr run -> SuperLink (CTRL API)
SEQ_START="${SEQ_START:-0}"
SEQ_END="${SEQ_END:-35}"
FLOWER_RUN_CONFIG_FOLDER="${FLOWER_RUN_CONFIG_FOLDER:-main_mia_experiments}"

echo "Server node: $SERVER_NODE  Fleet:${PORT_FLEET}  CTRL:${PORT_CTRL}"
echo "Client nodes: ${CLIENTS[*]}"
echo "Project dir: $PROJECT_DIR"

# --- Prepare Data Splits (once per experiment) ---
SHARED_SPLITS_PATH="/work/hi85udaj-flower/dataset_splits"
mkdir -p "$SHARED_SPLITS_PATH"

echo "Creating Splits: $SHARED_SPLITS_PATH"
if [ ! -d "$SHARED_SPLITS_PATH/D1" ]; then
  python - <<PY
from pathlib import Path
from flower.utils.split_cifar10_mia import split_cifar10_for_target_and_shadow
target = Path("$SHARED_SPLITS_PATH")
print("Writing CIFAR-10 splits to:", target)
split_cifar10_for_target_and_shadow(split_save_dir=target)
print("✅ Wrote splits to:", target)
PY
fi

############################################
# Helper functions: SuperLink + port waits
############################################

start_superlink() {
  echo "🔧 Starting SuperLink on ${SERVER_NODE}..."

  srun --nodes=1 --nodelist="$SERVER_NODE" --ntasks=1 bash -lc "
    set -e
    cd '$PROJECT_DIR'
    echo 'Using python: ' \$(which python)
    echo 'Starting flower-superlink...'
    flower-superlink \
      --insecure \
      --fleet-api-address 0.0.0.0:${PORT_FLEET} \
      --control-api-address 0.0.0.0:${PORT_CTRL} \
      --fleet-api-num-workers 8 \
      > ${LOG_DIR}/superlink.log 2>&1
  " &
}

wait_for_port() {
  local host="$1"
  local port="$2"
  local name="$3"
  local seconds="${4:-180}"

  echo "⏳ Waiting for $name on ${host}:${port} (up to ${seconds}s)..."
  python - <<PY
import socket, time, sys
host = "${host}"
port = ${port}
seconds = ${seconds}
for _ in range(seconds):
    try:
        with socket.create_connection((host, port), timeout=2):
            print(f"✅ {name} reachable at {host}:{port}")
            sys.exit(0)
    except OSError:
        time.sleep(1)
print(f"ERROR: {name} NOT reachable at {host}:{port}")
sys.exit(1)
PY
}

############################################
# 1) Start SuperLink with retry logic
############################################

MAX_ATTEMPTS=3
attempt=1
success=0

while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  echo "🔁 SuperLink start attempt ${attempt}/${MAX_ATTEMPTS}..."
  # just in case something leftover is running
  pkill -u "$USER" -f "flower-superlink" || true

  start_superlink

  if wait_for_port "$SERVER_NODE" "$PORT_FLEET" "SuperLink Fleet API" 60; then
    success=1
    break
  else
    echo "⚠️ SuperLink Fleet API not reachable on attempt $attempt."
    echo "   See ${LOG_DIR}/superlink.log for details."
  fi

  attempt=$((attempt + 1))
  sleep 5
done

if [ "$success" -ne 1 ]; then
  echo "❌ SuperLink failed to start after ${MAX_ATTEMPTS} attempts. Exiting."
  exit 1
fi

echo "SuperLink started successfully on ${SERVER_NODE}."

############################################
# 2) Start ALL SuperNodes in ONE srun (bg)
############################################

if [ "$NUM_CLIENTS" -eq 0 ]; then
  echo "❌ No client nodes available (NUM_CLIENTS=0). Did you request at least 2 nodes?"
  exit 1
fi

srun --nodes="$NUM_CLIENTS" --ntasks="$NUM_CLIENTS" --ntasks-per-node=1 \
     --nodelist="$CLIENT_HOSTS" --exclusive --kill-on-bad-exit=1 bash -lc "
  set -e
  cd '$PROJECT_DIR'
  PARTITION_ID=\$SLURM_PROCID

  echo 'Starting supernode on' \${HOSTNAME} 'with PARTITION_ID=' \${PARTITION_ID}
  flower-supernode \
    --insecure \
    --superlink ${SERVER_NODE}:${PORT_FLEET} \
    --max-retries 2 \
    --node-config \"partition-id=\${PARTITION_ID} num-partitions=${NUM_CLIENTS}\" \
    > ${LOG_DIR}/supernode_\${SLURMD_NODENAME}.log 2>&1
" &
echo "SuperNodes started on client nodes: ${CLIENTS[*]}"

############################################
# 3) Wait until SuperLink Exec API (9093)
############################################

wait_for_port "$SERVER_NODE" "$PORT_CTRL" "SuperLink Exec API" 180

############################################
# 4) Trigger the Flower runs from server
############################################

SERVER_ADDRESS="${SERVER_NODE}:${PORT_CTRL}"

echo "🌼 Starting multiple Flower runs sequentially on $SERVER_NODE"

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --nodelist="$SERVER_NODE" --overlap bash -lc "
  set -e
  cd '$PROJECT_DIR'

  # Inject correct server address into pyproject.toml
  sed -i '/^\[tool\.flwr\.federations\.hpc-deploy\]/,/^\[/{
    s/^address *= *.*/address = \"${SERVER_ADDRESS}\"/
  }' pyproject.toml

  # Sequentially run multiple Flower experiments
  for run_id in \$(seq ${SEQ_START} ${SEQ_END}); do
    echo \"🚀 Starting Flower run\${run_id}\"
    python -u run_flower_hpc.py \"flower=${FLOWER_RUN_CONFIG_FOLDER}/run\${run_id}\" 2>&1 | tee \"${LOG_DIR}/flwr_run\${run_id}.log\"
    echo \"✅ Finished Flower run\${run_id}\"
    echo
    sleep 10
  done

  # Revert .toml so it's clean for next job
  sed -i '/^\[tool\.flwr\.federations\.hpc-deploy\]/,/^\[/{
    s/^address *= *.*/address = \"SERVER_NODE:9093\"/
  }' pyproject.toml
"

echo "✅ All Flower processes finished successfully."
