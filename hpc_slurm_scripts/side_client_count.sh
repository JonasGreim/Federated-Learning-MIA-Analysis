#!/bin/bash
# -------------------------------------------------------------
# This script submits a series of SLURM jobs that run in order.
#
# /// SIDE EXPERIMENTS ///
# The first three jobs run `flower.sbatch` with different client counts.
# The fourth job runs `mia.sbatch` as an array job (tasks 36–38),
#     executing MIA analysis on the three corresponding Flower runs.
#
# Notes:
#   -N <n>          → number of nodes to allocate (for Flower: 1 server + n−1 clients)
#   -t <hh:mm:ss>   → time limit for the job
#   --array=36-38%3 → array job with 3 tasks (run36–38), max 3 running concurrently
#   --export        → pass environment variables into the job
#       - SEQ_START, SEQ_END: run indices for Flower runs
#       - FLOWER_RUN_CONFIG_FOLDER: config folder for Flower runs
#       - MIA_RUN_CONFIG_FOLDER: config folder for MIA runs
#       - ALL: pass all current environment variables
#
#   Set client number parameter also in experiments_conf for plot generation
# -------------------------------------------------------------

set -euo pipefail

# Get the directory of the current script
SCRIPT_DIR="$(dirname "$0")"

jid1_simple_model=$(sbatch --parsable -N 3  -t 01:00:00 --export=ALL,SEQ_START=36,SEQ_END=36,FLOWER_RUN_CONFIG_FOLDER=client_number_mia_experiments "$SCRIPT_DIR/flower.sbatch")
jid2_simple_model=$(sbatch --parsable --dependency=afterok:${jid1_simple_model} -N 6  -t 01:00:00 --export=ALL,SEQ_START=37,SEQ_END=37,FLOWER_RUN_CONFIG_FOLDER=client_number_mia_experiments "$SCRIPT_DIR/flower.sbatch")
jid3_simple_model=$(sbatch --parsable --dependency=afterok:${jid2_simple_model} -N 11 -t 01:00:00 --export=ALL,SEQ_START=38,SEQ_END=38,FLOWER_RUN_CONFIG_FOLDER=client_number_mia_experiments "$SCRIPT_DIR/flower.sbatch")
jid4_simple_model_mia=$(sbatch --parsable --dependency=afterok:${jid3_simple_model} -t 00:15:00 --array=36-38%3 --export=ALL,MIA_RUN_CONFIG_FOLDER=client_number_mia_experiments "$SCRIPT_DIR/mia.sbatch")

jid1_complex_model=$(sbatch --parsable --dependency=afterok:${jid4_simple_model_mia} -N 3  -t 01:00:00 --export=ALL,SEQ_START=39,SEQ_END=39,FLOWER_RUN_CONFIG_FOLDER=client_number_mia_experiments "$SCRIPT_DIR/flower.sbatch")
jid2_complex_model=$(sbatch --parsable --dependency=afterok:${jid1_complex_model} -N 6  -t 01:00:00 --export=ALL,SEQ_START=40,SEQ_END=40,FLOWER_RUN_CONFIG_FOLDER=client_number_mia_experiments "$SCRIPT_DIR/flower.sbatch")
jid3_complex_model=$(sbatch --parsable --dependency=afterok:${jid2_complex_model} -N 11 -t 01:10:00 --export=ALL,SEQ_START=41,SEQ_END=41,FLOWER_RUN_CONFIG_FOLDER=client_number_mia_experiments "$SCRIPT_DIR/flower.sbatch")
jid4_complex_model_mia=$(sbatch --parsable --dependency=afterok:${jid3_complex_model} -t 00:15:00 --array=39-41%3 --export=ALL,MIA_RUN_CONFIG_FOLDER=client_number_mia_experiments "$SCRIPT_DIR/mia.sbatch")



echo "Submitted: simple model: $jid1_simple_model -> $jid2_simple_model -> $jid3_simple_model -> $jid4_simple_model_mia"
echo "Submitted: complex model: $jid1_complex_model -> $jid2_complex_model -> $jid3_complex_model -> $jid4_complex_model_mia"
