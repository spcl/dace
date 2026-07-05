#!/bin/bash
#SBATCH --job-name=npbench-polybench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=__CPUS__   # <-- FILL IN: this node's CPU count (nproc --all)
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=npbench_polybench_%j.out
#SBATCH --error=npbench_polybench_%j.err

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=1

srun python3 npbench_polybench_perf.py --reps 100
python3 npbench_polybench_perf.py --tables-only
