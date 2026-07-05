#!/bin/bash
#SBATCH --job-name=dace-npbench-polybench-perf
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4      # 4 ranks/node by default -- see note below on sizing this
#SBATCH --cpus-per-task=8        # cores-per-node / ntasks-per-node -- check your node's core
                                 # count first (see "checking cpus per node" below) and adjust
#SBATCH --time=08:00:00          # paper-preset kernels (e.g. gemm) run real-world sized problems
#SBATCH --output=npbench_polybench_%j.out
#SBATCH --error=npbench_polybench_%j.err
#
# Example SLURM job: NPBench+PolyBench (vendored corpus + paper-preset data +
# vendored numpy reference) canonicalize performance regression -- 5 lanes
# (baseline, auto-opt, canon, fast-canon, numpy), distributed over
# nodes * ntasks-per-node ranks total. Same isolation/timing/crash handling
# as the TSVC scripts (see example_slurm_tsvc2.sh for the full rationale).
#
# The paper preset runs real-world problem sizes (e.g. gemm at its full
# published size), so this is the slowest of the 4 job types per-kernel --
# --time is set generously above; tune down for a smaller smoke test with
# --reps.
#
# Submit with:  sbatch example_slurm_npbench_polybench.sh
# Adjust --nodes / --ntasks-per-node for however many ranks (X) you want.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# --cxx <name-or-abs-path> pins a specific compiler for DaCe codegen
# (default: latest LLVM/clang++ on PATH, else latest GCC/g++). Results are
# namespaced by compiler+hostname automatically.
srun python3 npbench_polybench_perf.py --reps 100

python3 npbench_polybench_perf.py --tables-only
