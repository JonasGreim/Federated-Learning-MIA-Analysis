#!/bin/bash
# -------------------------------------------------------------
# This script submits a series of SLURM jobs that run in order.
#
# /// MAIN EXPERIMENTS ///
# The first job runs `flower2.sbatch` — it performs all 36 Flower runs
#     sequentially, using 5 nodes (1 server + 4 clients per run).
# When that job finishes successfully, the second job runs `mia2.sbatch`
#     as an array job (tasks 0–35), launching the MIA analysis for each run.
#
# Notes:
#   -N <n>        → number of nodes to allocate (for Flower: 1 server + n−1 clients)
#   -t <hh:mm:ss> → time limit for the job
#   --export        → pass environment variables into the job
#       - SEQ_START, SEQ_END: run indices for Flower runs
#       - ALL: pass all current environment variables
#   --array=0-35%5 → array job with 36 tasks (0–35), max 5 running concurrently
#
# -------------------------------------------------------------

set -euo pipefail

# Get the directory of the current script
SCRIPT_DIR="$(dirname "$0")"

jid1=$(sbatch --parsable -N 6  -t 11:00:00 --export=ALL,SEQ_START=0,SEQ_END=35 "$SCRIPT_DIR/flower.sbatch")
jid2=$(sbatch --parsable --dependency=afterok:${jid1} -t 00:10:00 --array=0-35%5 "$SCRIPT_DIR/mia.sbatch")

echo "Submitted: $jid1 -> $jid2"

