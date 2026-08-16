#!/bin/bash
#SBATCH --job-name=expA_lgbm
#SBATCH --partition=cluster_short
#SBATCH --cpus-per-task=48
#SBATCH --mem=7G
#SBATCH --time=01:00:00
#SBATCH --output=../result/exp_A/expA_%j.out    # stdout log
#SBATCH --error=../result/exp_A/expA_%j.err     # stderr log

# make sure the result folder exists before Slurm writes logs into it
mkdir -p ../result/exp_A

# use all allocated cores for numpy/lightgbm
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-48}

# data path (../data relative to script/, matches common.py default)
export DATA_ROOT=${DATA_ROOT:-../data}
# where exp_A_lgbm.py writes its results CSV
export RESULT_DIR=${RESULT_DIR:-../result/exp_A}

echo "==== Experiment A (LightGBM, no image) ===="
echo "job=$SLURM_JOB_ID  node=$SLURMD_NODENAME  cpus=$SLURM_CPUS_PER_TASK"
srun python3 exp_A_lgbm.py
