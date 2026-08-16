#!/bin/bash
#SBATCH --job-name=expB_clip
#SBATCH --partition=cluster_short
#SBATCH --cpus-per-task=48
#SBATCH --mem=7G
#SBATCH --time=04:00:00
#SBATCH --output=../result/exp_B/expB_%j.out
#SBATCH --error=../result/exp_B/expB_%j.err
# If you later get a GPU partition, change --partition and add:
# #SBATCH --gres=gpu:1

mkdir -p ../result/exp_B
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}
export DATA_ROOT=${DATA_ROOT:-../data}
export RESULT_DIR=${RESULT_DIR:-../result/exp_B}

echo "==== Experiment B (CLIP image) ===="
echo "job=$SLURM_JOB_ID  node=$SLURMD_NODENAME  cpus=$SLURM_CPUS_PER_TASK"
srun python3 exp_B_clip.py
