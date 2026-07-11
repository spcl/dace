#!/bin/bash
#SBATCH --job-name=dace-npbench-polybench-compile-perf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4      # 4 ranks/node by default -- see note below on sizing this
#SBATCH --cpus-per-task=72       # cores-per-node / ntasks-per-node -- run `nproc --all` on a
                                 # compute node to check its core count first and adjust
#SBATCH --time=08:00:00          # paper-preset kernels (e.g. gemm) build real-world sized SDFGs
#SBATCH --partition=normal
#SBATCH --account=g34
#SBATCH --output=npbench_polybench_compile_%j.out
#SBATCH --error=npbench_polybench_compile_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs
#
# Example SLURM job: the compile-speed + post-compile-performance comparison of
# the 4 DaCe pipelines (baseline = simplify+loop2map+mapfusion, auto-opt, canon,
# fast-canon) on NPBench+PolyBench at the paper preset, distributed over
# nodes * ntasks-per-node ranks total.
#
# ONE job, TWO metrics into the SAME results tree:
#   1. npbench_polybench_perf.py          -> post-compile RUNTIME (speedup.md vs
#                                            baseline + a numpy reference lane)
#   2. npbench_polybench_compile_perf.py  -> COMPILE speed        (compile_total.md,
#                                            compile_codegen.md, compile_cxx.md;
#                                            numpy has no compile lane)
# Both self-partition kernels by SLURM_PROCID/SLURM_NTASKS; the final
# --tables-only passes aggregate across ranks. Drop the first srun if you have
# already run the runtime sweep.
#
# Submit with:  sbatch example_slurm_npbench_polybench_compile.sh
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

srun --cpu-bind=cores python3 npbench_polybench_perf.py --reps 100 --cxx=clang++
srun --cpu-bind=cores python3 npbench_polybench_compile_perf.py --compile-reps 5 --cxx=clang++

python3 npbench_polybench_perf.py --tables-only
python3 npbench_polybench_compile_perf.py --tables-only
