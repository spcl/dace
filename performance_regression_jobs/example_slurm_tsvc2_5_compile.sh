#!/bin/bash
#SBATCH --job-name=dace-tsvc2_5-compile-perf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4      # 4 ranks/node by default -- see note below on sizing this
#SBATCH --cpus-per-task=72       # cores-per-node / ntasks-per-node -- run `nproc --all` on a
                                 # compute node to check its core count first and adjust
#SBATCH --time=06:00:00
#SBATCH --partition=normal
#SBATCH --account=g34
#SBATCH --output=tsvc2_5_compile_%j.out
#SBATCH --error=tsvc2_5_compile_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs
#
# Example SLURM job: the compile-speed + post-compile-performance comparison of
# the 4 DaCe pipelines (baseline = simplify+loop2map+mapfusion, auto-opt, canon,
# fast-canon) on TSVC2.5, distributed over nodes * ntasks-per-node ranks total.
#
# ONE job, TWO metrics into the SAME results tree:
#   1. tsvc2_5_perf.py          -> post-compile RUNTIME  (speedup.md, correctness.md)
#   2. tsvc2_5_compile_perf.py  -> COMPILE speed         (compile_total.md,
#                                  compile_codegen.md, compile_cxx.md)
# Both self-partition kernels by SLURM_PROCID/SLURM_NTASKS; the final
# --tables-only passes aggregate across ranks (see engine.py). Drop the first
# srun if you have already run the runtime sweep.
#
# Submit with:  sbatch example_slurm_tsvc2_5_compile.sh
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

srun --cpu-bind=cores python3 tsvc2_5_perf.py --reps 100 --cxx=clang++
srun --cpu-bind=cores python3 tsvc2_5_compile_perf.py --compile-reps 5 --cxx=clang++

python3 tsvc2_5_perf.py --tables-only
python3 tsvc2_5_compile_perf.py --tables-only
