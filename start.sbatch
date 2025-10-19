#!/bin/bash
#SBATCH --job-name=flower-cluster
#SBATCH --output=logs/flower_%j_%N.out
#SBATCH --error=logs/flower_%j_%N.err
#SBATCH --partition=paula
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

# --- Environment Setup ---
set -eo pipefail
module load python/3.12

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

# Start Flower Server auf dem ersten Node
srun --nodes=1 --ntasks=1 --nodelist=$SERVER_NODE \
    bash -c 'echo "Starting Flower Server on $(hostname)";
             python3 run_flower_experiment.py flower=run1' &

# Kleine Pause, damit Server starten kann
sleep 60

# Starte Clients auf den restlichen 9 Nodes
for CLIENT_NODE in "${NODES[@]:1}"; do
    echo "Starting Client on ${CLIENT_NODE}"
    srun --nodes=1 --ntasks=1 --nodelist=$CLIENT_NODE \
        bash -c "echo Running Flower Client on \$(hostname);
                 python3 src/client.py --server-address ${SERVER_NODE}:8080" &
done

wait
echo "✅ All Flower processes finished successfully."
