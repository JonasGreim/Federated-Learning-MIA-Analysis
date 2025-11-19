#!/bin/bash
# -------------------------------------------------------------
# This script submits a series of SLURM jobs that run in order.
#
# /// SIDE EXPERIMENTS ///
# The first three jobs run `flower.sbatch` with different client counts.
# The fourth job runs `mia.sbatch` as an array job (tasks 36–38),
#     executing MIA analysis on the three corresponding Flower runs.
# The fifth job runs `mia.sbatch` as an array job (tasks 39–43),
#     executing MIA analysis on the first Flower run but with varying
#     numbers of shadow models.
#
# Notes:
#   -t <hh:mm:ss>   → time limit for the job
#   --array=42-57%5 → array job with 5 tasks (run42–57), max 5 running concurrently
#   --export        → pass environment variables into the job
#       - MIA_RUN_CONFIG_FOLDER: config folder for MIA runs
#       - ALL: pass all current environment variables
# -------------------------------------------------------------

# Note use main experiment flower checkpoints folder

set -euo pipefail

# Get the directory of the current script
SCRIPT_DIR="$(dirname "$0")"

#jid1_shadow_models=$(sbatch --parsable -t 01:20:00 --array=42-57%5 --export=ALL,MIA_RUN_CONFIG_FOLDER=shadow_models_experiments "$SCRIPT_DIR/mia.sbatch")

jid1_shadow_models=$(sbatch --parsable -t 01:20:00 --array=53-53%5 --export=ALL,MIA_RUN_CONFIG_FOLDER=shadow_models_experiments "$SCRIPT_DIR/mia.sbatch")
jid2_shadow_models=$(sbatch --parsable -t 01:20:00 --array=57-57%5 --export=ALL,MIA_RUN_CONFIG_FOLDER=shadow_models_experiments "$SCRIPT_DIR/mia.sbatch")


#echo "Submitted shadow model experiment: $jid1_shadow_models"
echo "Submitted shadow model experiment: $jid1_shadow_models, $jid2_shadow_models"
