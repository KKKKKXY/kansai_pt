#!/bin/bash
#SBATCH --job-name=expC_combos
#SBATCH --partition=cluster_short
#SBATCH --cpus-per-task=48
#SBATCH --mem=7G
#SBATCH --time=01:00:00
#SBATCH --output=../result/exp_C/expC_%j.out
#SBATCH --error=../result/exp_C/expC_%j.err

mkdir -p ../result/exp_C
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}
export DATA_ROOT=${DATA_ROOT:-../data}
export RESULT_DIR=${RESULT_DIR:-../result/exp_C}
# B+C needs the CLIP cache produced by exp_B
export CLIP_CACHE=${CLIP_CACHE:-../result/exp_B/clip_emb.npz}

echo "==== Experiment C, A+C, B+C ===="
echo "job=$SLURM_JOB_ID  node=$SLURMD_NODENAME  cpus=$SLURM_CPUS_PER_TASK"
srun python3 exp_C_and_combos.py
