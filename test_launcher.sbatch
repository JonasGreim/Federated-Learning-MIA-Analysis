#!/usr/bin/env bash
set -euo pipefail

jid_flower=$(sbatch --parsable test.sbatch)
echo "Flower job: $jid_flower"

jid_mia=$(sbatch --parsable --dependency=afterok:${jid_flower} test_mia.sbatch)
echo "MIA job (afterok on ${jid_flower}): $jid_mia"