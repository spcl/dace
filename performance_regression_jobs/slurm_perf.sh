#!/bin/bash
#SBATCH --job-name=dace-perf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4      # 4 ranks/node = one per Grace CPU (GH200 node = 4 Grace sockets);
                                 # kernels self-partition across ranks (engine.my_slice)
#SBATCH --cpus-per-task=72       # 72 cores per Grace CPU -> 4 x 72 = 288 cores = the whole node
#SBATCH --hint=nomultithread     # Grace Neoverse-V2 has no SMT (1 thread/core); keep it explicit
#SBATCH --time=08:00:00          # 8h: a compile-heavy 4-lane sweep needs the headroom
#SBATCH --partition=normal
#SBATCH --account=g34
#SBATCH --output=perf_%j.out
#SBATCH --error=perf_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs
#
# ONE generic sbatch for all 8 jobs = {canon_vs, vector_vs} x {npbench, polybench,
# tsvc2, tsvc2_5}. Parameterized by environment (submit_perf_jobs.sh sets these
# per job via `sbatch --export`, and overrides --job-name/--output/--error):
#
#   EXPERIMENT   canon_vs | vector_vs
#   CORPUS       npbench | polybench | tsvc2 | tsvc2_5
#   CXX          C++ compiler for DaCe codegen (default: clang++)
#   RESULTS_DIR  results root for this job (default: results/$EXPERIMENT)
#
# Kernels are distributed across the job's 4 ranks automatically.

set -euo pipefail
cd /capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs

EXPERIMENT="${EXPERIMENT:-canon_vs}"
CORPUS="${CORPUS:-tsvc2}"
CXX="${CXX:-clang++}"
RESULTS_DIR="${RESULTS_DIR:-results/$EXPERIMENT}"

# MPI anti-hang (must match run_perf.py's own os.environ.setdefault block) +
# threading. Exported so every rank / spawned subprocess inherits them.
export OMP_NUM_THREADS="72"        # one Grace CPU's worth of cores per rank
export OMP_PROC_BIND="close"       # pin OpenMP threads, packed within the rank's socket
export OMP_PLACES="cores"          # one OpenMP place per physical core
export MPI4PY_RC_INITIALIZE="0"
export OMPI_MCA_pml="ob1"
export OMPI_MCA_btl="self,vader"
export UCX_VFS_ENABLE="n"
export PYTHONUNBUFFERED=1  # otherwise stdout is fully buffered (not a tty) and progress
                          # prints don't appear until a buffer fills -- looks like a hang

export PYTHONUSERBASE=/capstor/scratch/cscs/$USER/aarch64/python
export PATH=$PYTHONUSERBASE/bin:$PATH
#python3.11 -m venv /capstor/scratch/cscs/$USER/aarch64/venvs/myenv  # one-time setup; scratch can get purged
source /capstor/scratch/cscs/$USER/aarch64/venvs/myenv/bin/activate
alias python=python3.11

spack load gcc@16.1.0
spack load llvm@22.1.5

echo "EXPERIMENT=$EXPERIMENT CORPUS=$CORPUS CXX=$CXX RESULTS_DIR=$RESULTS_DIR"

# --distribution=block:block hands rank i a contiguous 72-core block = Grace socket i, so the 4
# ranks split 72-72-72-72 across the node's 4 Grace CPUs; --cpu-bind=cores then pins each rank's
# threads to its own cores (verbose = log the CPU mask to the job output to confirm the split).
srun --cpu-bind=verbose,cores --distribution=block:block python3 run_perf.py \
    --experiment "$EXPERIMENT" --corpus "$CORPUS" --reps 25 --cxx="$CXX" --results-dir="$RESULTS_DIR"

# One single-rank pass to (re)build the aggregate tables across every rank's output.
python3 run_perf.py --experiment "$EXPERIMENT" --corpus "$CORPUS" --tables-only --results-dir="$RESULTS_DIR"
