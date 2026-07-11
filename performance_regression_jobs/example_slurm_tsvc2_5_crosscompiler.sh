#!/bin/bash
#SBATCH --job-name=dace-tsvc2_5-crosscompiler
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4      # 4 ranks/node by default -- see note below on sizing this
#SBATCH --cpus-per-task=72       # cores-per-node / ntasks-per-node -- run `nproc --all` on a
                                 # compute node to check its core count first and adjust
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --account=g34
#SBATCH --output=tsvc2_5_crosscompiler_%j.out
#SBATCH --error=tsvc2_5_crosscompiler_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs
#
# CROSS-COMPILER performance + compile-time comparison on TSVC2.5.
#
# Sweeps $CXXES; for each compiler every DaCe lane is built ONCE (single build)
# with it, then timed for RUNTIME (tsvc2_5_perf.py) and cold-compiled for
# COMPILE speed (tsvc2_5_compile_perf.py), on the corpus's seeded-random input
# (make_inputs + stable seed, identical across compilers). Results namespace per
# compiler (engine.compiler_host_tag), so each kernel gets one row per compiler --
# each compiler's DaCe `baseline` lane (simplify+loop2map+mapfusion) is its
# single-build reference. See example_slurm_tsvc2_crosscompiler.sh for the full
# rationale + how to read the tables.
#
# Submit with:  sbatch example_slurm_tsvc2_5_crosscompiler.sh
#   CXXES="g++ clang++ icpx nvc++" sbatch example_slurm_tsvc2_5_crosscompiler.sh

set -euo pipefail
cd /capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs

export OMP_NUM_THREADS="72"
export PYTHONUNBUFFERED=1

export PYTHONUSERBASE=/capstor/scratch/cscs/$USER/aarch64/python
export PATH=$PYTHONUSERBASE/bin:$PATH
#python3.11 -m venv /capstor/scratch/cscs/$USER/aarch64/venvs/myenv  # one-time setup; scratch can get purged
source /capstor/scratch/cscs/$USER/aarch64/venvs/myenv/bin/activate
alias python=python3.11

spack load gcc@16.1.0
spack load llvm@22.1.5
# For icpx add:  source /opt/intel/oneapi/setvars.sh
# For nvc++ add: add the nvhpc compilers/bin to PATH

CXXES="${CXXES:-g++ clang++}"
REPS="${REPS:-100}"
COMPILE_REPS="${COMPILE_REPS:-5}"

for CXX in $CXXES; do
  if ! command -v "$CXX" >/dev/null 2>&1; then
    echo "[crosscompiler] skip '$CXX' (not on PATH)"; continue
  fi
  echo "[crosscompiler] === $CXX ==="
  srun --cpu-bind=cores python3 tsvc2_5_perf.py --reps "$REPS" --cxx="$CXX" || echo "[crosscompiler] runtime sweep failed for $CXX"
  srun --cpu-bind=cores python3 tsvc2_5_compile_perf.py --compile-reps "$COMPILE_REPS" --cxx="$CXX" || echo "[crosscompiler] compile sweep failed for $CXX"
done

python3 tsvc2_5_perf.py --tables-only
python3 tsvc2_5_compile_perf.py --tables-only
