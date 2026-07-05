#!/bin/bash
#SBATCH --job-name=dace-tsvc2-5-perf
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4      # 4 ranks/node by default -- see note below on sizing this
#SBATCH --cpus-per-task=8        # cores-per-node / ntasks-per-node -- check your node's core
                                 # count first (see "checking cpus per node" below) and adjust
#SBATCH --time=04:00:00
#SBATCH --output=tsvc2_5_%j.out
#SBATCH --error=tsvc2_5_%j.err
#
# Example SLURM job: TSVC2.5 canonicalize performance regression, distributed
# over nodes * ntasks-per-node ranks total. Same isolation/timing/crash
# handling as tsvc2 (see example_slurm_tsvc2.sh for the full rationale) --
# this file only differs in which script it srun's.
#
# Submit with:  sbatch example_slurm_tsvc2_5.sh
# Adjust --nodes / --ntasks-per-node for however many ranks (X) you want.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# --cxx <name-or-abs-path> pins a specific compiler for DaCe's own codegen
# (default: latest LLVM/clang++ on PATH, else latest GCC/g++); DaCe-lane
# results are namespaced by compiler+hostname automatically. Native lanes
# (native-gcc/-clang/-icpx/-nvc and their autopar variants) are unaffected by
# --cxx -- each finds its own vendor's compiler independently, and a vendor
# with no compiler installed is just skipped for that lane (see
# example_slurm_tsvc2.sh for the full rationale).
srun python3 tsvc2_5_perf.py --reps 100

python3 tsvc2_5_perf.py --tables-only
