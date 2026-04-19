#!/bin/bash
#SBATCH --job-name=causal_forest
#SBATCH --account=statdept
#SBATCH --partition=common
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=cf_%j.out
#SBATCH --error=cf_%j.err

echo "================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Memory:    $SLURM_MEM_PER_NODE MB"
echo "Start:     $(date)"
echo "================================================"

cd ~/pa_fmd

# Activate venv
source venv/bin/activate

# Run causal forest
python src/05_causal_forest.py

echo "================================================"
echo "End:       $(date)"
echo "================================================"
