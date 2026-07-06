#!/bin/bash
#SBATCH --job-name=dace-npbench-polybench-perf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4      # 4 ranks/node by default -- see note below on sizing this
#SBATCH --cpus-per-task=72       # cores-per-node / ntasks-per-node -- run `nproc --all` on a
                                 # compute node to check its core count first and adjust
#SBATCH --time=08:00:00          # paper-preset kernels (e.g. gemm) run real-world sized problems
#SBATCH --partition=normal
#SBATCH --account=g34
#SBATCH --output=npbench_polybench_%j.out
#SBATCH --error=npbench_polybench_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs
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
cd /capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs

export OMP_NUM_THREADS="72"
export PYTHONUNBUFFERED=1  # otherwise stdout is fully buffered (not a tty), so progress prints
                           # don't show up in the log until a buffer fills -- looks like a hang

export PYTHONUSERBASE=/capstor/scratch/cscs/$USER/aarch64/python
export PATH=$PYTHONUSERBASE/bin:$PATH
#python3.11 -m venv /capstor/scratch/cscs/$USER/aarch64/venvs/myenv  # one-time setup; scratch can get purged
source /capstor/scratch/cscs/$USER/aarch64/venvs/myenv/bin/activate
alias python=python3.11

spack load gcc@16.1.0
spack load llvm@22.1.5

# --cxx <name-or-abs-path> pins a specific compiler for DaCe codegen
# (default: clang++ on PATH, else g++ -- plain PATH lookup). Results are
# namespaced by compiler+hostname automatically.
#
# --cpu-bind=cores keeps each rank pinned to its own allocated cores instead
# of letting the OS scheduler migrate/share them across ranks.
srun --cpu-bind=cores python3 npbench_polybench_perf.py --reps 100 --cxx=clang++

python3 npbench_polybench_perf.py --tables-only
