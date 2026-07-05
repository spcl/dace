#!/bin/bash
#SBATCH --job-name=tsvc2-5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=__CPUS__   # <-- FILL IN: this node's CPU count (nproc --all)
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=tsvc2_5_%j.out
#SBATCH --error=tsvc2_5_%j.err

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=1

srun python3 tsvc2_5_perf.py --reps 100
python3 tsvc2_5_perf.py --tables-only
