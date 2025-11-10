#!/bin/bash
# -------------------------------------------------------------
# This script submits a series of SLURM jobs that run in order.
#
# /// SIDE EXPERIMENTS ///
# The first three jobs run `flower2.sbatch` with different client counts.
# The fourth job runs `mia2.sbatch` as an array job (tasks 36–38),
#     executing MIA analysis on the three corresponding Flower runs.
# The fifth job runs `mia2.sbatch` as an array job (tasks 39–43),
#     executing MIA analysis on the first Flower run but with varying
#     numbers of shadow models.
#
# Notes:
#   -N <n>          → number of nodes to allocate (for Flower: 1 server + n−1 clients)
#   -t <hh:mm:ss>   → time limit for the job
#   --array=36-38%5 → array job with 3 tasks (36–38), max 5 running concurrently
#   --export        → pass environment variables into the job
#       - SEQ_START, SEQ_END: run indices for Flower runs
#       - FLOWER_RUN_CONFIG_FOLDER: config folder for Flower runs
#       - MIA_RUN_CONFIG_FOLDER: config folder for MIA runs
#       - ALL: pass all current environment variables
# -------------------------------------------------------------

set -euo pipefail

# Get the directory of the current script
SCRIPT_DIR="$(dirname "$0")"

jid1=$(sbatch --parsable -N 3  -t 00:45:00 --export=ALL,SEQ_START=36,SEQ_END=36,FLOWER_RUN_CONFIG_FOLDER=client_number_mia_experiments "$SCRIPT_DIR/flower.sbatch")
jid2=$(sbatch --parsable --dependency=afterok:${jid1} -N 6  -t 00:45:00 --export=ALL,SEQ_START=37,SEQ_END=37,FLOWER_RUN_CONFIG_FOLDER=client_number_mia_experiments "$SCRIPT_DIR/flower.sbatch")
jid3=$(sbatch --parsable --dependency=afterok:${jid2} -N 11 -t 00:45:00 --export=ALL,SEQ_START=38,SEQ_END=38,FLOWER_RUN_CONFIG_FOLDER=client_number_mia_experiments "$SCRIPT_DIR/flower.sbatch")
jid4=$(sbatch --parsable --dependency=afterok:${jid3} -t 00:10:00 --array=36-38%3 --export=ALL,MIA_RUN_CONFIG_FOLDER=client_number_mia_experiments "$SCRIPT_DIR/mia.sbatch")
jid5=$(sbatch --parsable --dependency=afterok:${jid4} -t 00:30:00 --array=39-43%5 --export=ALL,MIA_RUN_CONFIG_FOLDER=shadow_models_experiments "$SCRIPT_DIR/mia.sbatch")

echo "Submitted: $jid1 -> $jid2 -> $jid3 -> $jid4 -> $jid5"

