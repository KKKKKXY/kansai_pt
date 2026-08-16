#!/bin/bash
# Submit all three experiments to Slurm in the correct order.
# A and B are independent; C needs B's CLIP cache, so C waits for B to finish.
# Usage:  cd script && ./run.sh
set -e

JOB_A=$(sbatch --parsable run_A.slurm)
echo "submitted A: job $JOB_A"

JOB_B=$(sbatch --parsable run_B.slurm)
echo "submitted B: job $JOB_B"

# C starts only after B completes successfully (afterok dependency)
JOB_C=$(sbatch --parsable --dependency=afterok:$JOB_B run_C_and_Combos.slurm)
echo "submitted C: job $JOB_C  (waits for B=$JOB_B)"

echo
echo "watch progress:  squeue -u \$USER"
echo "results appear in ../result/exp_A , ../result/exp_B , ../result/exp_C"