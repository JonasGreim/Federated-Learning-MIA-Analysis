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
#   --export        → pass environment variables into the job (e.g. run indices)
#   --array=36-38%5 → array job with 3 tasks (36–38), max 5 running concurrently
# -------------------------------------------------------------

jid1=$(sbatch --parsable -N 3  -t 00:45:00 --export=ALL,SEQ_START=36,SEQ_END=36 flower2.sbatch)
jid2=$(sbatch --parsable --dependency=afterok:${jid1} -N 5  -t 00:45:00 --export=ALL,SEQ_START=37,SEQ_END=37 flower2.sbatch)
jid3=$(sbatch --parsable --dependency=afterok:${jid2} -N 11 -t 00:45:00 --export=ALL,SEQ_START=38,SEQ_END=38 flower2.sbatch)
jid4=$(sbatch --parsable --dependency=afterok:${jid3} -t 00:10:00 --array=36-38%3 mia2.sbatch)
jid5=$(sbatch --parsable --dependency=afterok:${jid4} -t 00:30:00 --array=39-43%5 mia2.sbatch)

echo "Submitted: $jid1 -> $jid2 -> $jid3 -> $jid4 -> $jid5"

