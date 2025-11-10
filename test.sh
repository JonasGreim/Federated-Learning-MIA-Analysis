jid1=$(sbatch --parsable -N 2  -t 00:45:00 --export=SEQ_START=36,SEQ_END=36 flower2.sbatch)
jid2=$(sbatch --parsable --dependency=afterok:${jid1} -N 5  -t 00:45:00 --export=SEQ_START=37,SEQ_END=37 flower2.sbatch)
jid3=$(sbatch --parsable --dependency=afterok:${jid2} -N 10 -t 00:45:00 --export=SEQ_START=38,SEQ_END=38 flower2.sbatch)
jid4=$(sbatch --parsable --dependency=afterok:${jid3} -t 00:10:00 --array=36-38%3 mia2.sbatch)
jid5=$(sbatch --parsable --dependency=afterok:${jid4} -t 00:30:00 --array=39-43%5 mia2.sbatch)

echo "Submitted: $jid1 -> $jid2 -> $jid3 -> $jid4 -> $jid5"

