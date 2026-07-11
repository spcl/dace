#!/bin/bash
#SBATCH --job-name=dace-tsvc2_5-compile-perf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4      # 4 ranks/node by default -- see note below on sizing this
#SBATCH --cpus-per-task=72       # cores-per-node / ntasks-per-node -- run `nproc --all` on a
                                 # compute node to check its core count first and adjust
#SBATCH --time=06:00:00          # single-compiler default; cross-compiler = this x len(CXXES)
#SBATCH --partition=normal
#SBATCH --account=g34
#SBATCH --output=tsvc2_5_compile_%j.out
#SBATCH --error=tsvc2_5_compile_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs
#
# Compile-speed + post-compile-performance comparison of the 4 DaCe pipelines
# (auto_opt, parallel = simplify+loop2map+mapfusion, canon, fast-canon) on
# TSVC2.5, distributed over nodes * ntasks-per-node ranks total.
#
# ONE job, TWO metrics into the SAME results tree:
#   1. tsvc2_5_perf.py           -> post-compile RUNTIME  (speedup.md, correctness.md)
#   2. tsvc2_5_compile_perf.py   -> COMPILE speed         (compile_total.md,
#                                   compile_codegen.md, compile_cxx.md)
#
# Sweeps every compiler in $CXXES. The DEFAULT is a single compiler (clang++) --
# that is the plain single-compiler run. Set CXXES to more than one to turn this
# into the CROSS-COMPILER sweep: each compiler builds every DaCe lane once and is
# timed for runtime + cold-compile, results namespaced per compiler
# (engine.compiler_host_tag = <compiler>_<host>_<preset>) so each kernel gets one
# row per compiler -- read a lane DOWN the rows to compare it ACROSS compilers.
# An absent compiler is skipped, not fatal. Both drivers self-partition kernels
# by SLURM_PROCID/SLURM_NTASKS; the final --tables-only passes are the cross-rank
# (and cross-compiler) aggregation step.
#
# Submit with:
#   sbatch slurm_tsvc2_5_compile.sh                                   # clang++ only
#   CXXES="g++ clang++ icpx nvc++" sbatch slurm_tsvc2_5_compile.sh    # cross-compiler
# Adjust --nodes / --ntasks-per-node for however many ranks you want.

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
# For icpx add:  source /opt/intel/oneapi/setvars.sh
# For nvc++ add: add the nvhpc compilers/bin to PATH (or load its modulefile)

# Compilers to sweep. Default = single clang++ (plain single-compiler run);
# override with more than one for the cross-compiler comparison.
CXXES="${CXXES:-clang++}"
REPS="${REPS:-25}"
COMPILE_REPS="${COMPILE_REPS:-5}"

for CXX in $CXXES; do
  if ! command -v "$CXX" >/dev/null 2>&1; then
    echo "[compile] skip '$CXX' (not on PATH)"; continue
  fi
  echo "[compile] === $CXX ==="
  # || echo so one compiler's failure never aborts the rest or the table passes.
  srun --cpu-bind=cores python3 tsvc2_5_perf.py --reps "$REPS" --cxx="$CXX" || echo "[compile] runtime sweep failed for $CXX"
  srun --cpu-bind=cores python3 tsvc2_5_compile_perf.py --compile-reps "$COMPILE_REPS" --cxx="$CXX" || echo "[compile] compile sweep failed for $CXX"
done

# Cross-rank (and cross-compiler) aggregation: re-scans the whole results tree.
python3 tsvc2_5_perf.py --tables-only
python3 tsvc2_5_compile_perf.py --tables-only
