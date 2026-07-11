#!/bin/bash
#SBATCH --job-name=dace-tsvc2-crosscompiler
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4      # 4 ranks/node by default -- see note below on sizing this
#SBATCH --cpus-per-task=72       # cores-per-node / ntasks-per-node -- run `nproc --all` on a
                                 # compute node to check its core count first and adjust
#SBATCH --time=12:00:00          # cross-compiler = the single-compiler time x len(CXXES)
#SBATCH --partition=normal
#SBATCH --account=g34
#SBATCH --output=tsvc2_crosscompiler_%j.out
#SBATCH --error=tsvc2_crosscompiler_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs
#
# CROSS-COMPILER performance + compile-time comparison on TSVC2.
#
# Sweeps a set of C++ compilers ($CXXES). For EACH compiler, every DaCe lane is
# built ONCE (a single build) with that compiler and then:
#   - timed for post-compile RUNTIME over many reps  (tsvc2_perf.py)
#   - cold-compiled a few times for COMPILE speed     (tsvc2_compile_perf.py)
# on the corpus's seeded-random input (tsvc.allocate + stable_seed -- reproducible
# random arrays, identical across compilers so the comparison is apples-to-apples).
#
# Results are namespaced per compiler automatically (engine.compiler_host_tag =
# <compiler>_<host>_<preset>), so the SAME kernel gets one row per compiler in
# the final tables -- that IS the cross-compiler comparison, with each compiler's
# own DaCe `baseline` lane (plain simplify+loop2map+mapfusion) as its single-build
# reference. speedup.md compares lanes within a compiler; read a lane DOWN the
# rows to compare that lane ACROSS compilers.
#
# Tables (written once at the end, aggregating all ranks AND all compilers):
#   runtime : correctness.md, speedup.md
#   compile : compile_total.md, compile_codegen.md, compile_cxx.md
#
# Submit with:  sbatch example_slurm_tsvc2_crosscompiler.sh
#   CXXES="g++ clang++ icpx nvc++" sbatch example_slurm_tsvc2_crosscompiler.sh   # add vendor compilers
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

# The compilers to sweep. Each must be on PATH (an absent one is skipped, not fatal).
# The default keeps the two the spack loads above provide; override via the env var.
CXXES="${CXXES:-g++ clang++}"
REPS="${REPS:-100}"
COMPILE_REPS="${COMPILE_REPS:-5}"

for CXX in $CXXES; do
  if ! command -v "$CXX" >/dev/null 2>&1; then
    echo "[crosscompiler] skip '$CXX' (not on PATH)"; continue
  fi
  echo "[crosscompiler] === $CXX ==="
  # || true so one compiler's failure (e.g. a missing vendor runtime) never aborts
  # the rest of the sweep or the table passes below.
  srun --cpu-bind=cores python3 tsvc2_perf.py --reps "$REPS" --cxx="$CXX" || echo "[crosscompiler] runtime sweep failed for $CXX"
  srun --cpu-bind=cores python3 tsvc2_compile_perf.py --compile-reps "$COMPILE_REPS" --cxx="$CXX" || echo "[crosscompiler] compile sweep failed for $CXX"
done

# Cross-rank AND cross-compiler aggregation (re-scans the whole results tree).
python3 tsvc2_perf.py --tables-only
python3 tsvc2_compile_perf.py --tables-only
