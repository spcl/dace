#!/bin/bash
#SBATCH --job-name=dace-cloudsc-cache
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --partition=normal
#SBATCH --account=g34
#SBATCH --output=cloudsc_cache_%j.out
#SBATCH --error=cloudsc_cache_%j.err
#SBATCH --chdir=/capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs
#
# Phase A ONLY: build the CloudSC SDFG (parse + simplify + ShortLoopUnroll + LoopToMap chain --
# the expensive loop2map+unroll the user flagged) and cache it as
# cloudsc_variants/cache/cloudsc_parallel.sdfgz. Runs on the NORMAL partition with a 1.5h window
# because this exceeds the 30-min debug cap (per the user: "for sdfgz submit a 1.5 hour job if it
# times out on normal node"). The measurement (Phase B) then loads this cache inside a debug job.
set -euo pipefail
cd /capstor/scratch/cscs/ybudanaz/aarch64/dace/performance_regression_jobs
export OMP_NUM_THREADS="72" OPENBLAS_NUM_THREADS="72" OMP_PROC_BIND="close" OMP_PLACES="cores" PYTHONUNBUFFERED=1
export PYTHONUSERBASE=/capstor/scratch/cscs/$USER/aarch64/python
export PATH=$PYTHONUSERBASE/bin:$PATH
source /capstor/scratch/cscs/$USER/aarch64/venvs/myenv/bin/activate
export PYTHONPATH=/capstor/scratch/cscs/ybudanaz/aarch64/dace:${PYTHONPATH:-}
spack load gcc@16.1.0
spack load cmake
spack load openblas threads=pthreads
export OPENBLAS_DIR="$(spack location -i openblas threads=pthreads 2>/dev/null || echo "${OPENBLAS_DIR:-}")"
if [ -n "$OPENBLAS_DIR" ]; then
    for _d in "$OPENBLAS_DIR"/lib "$OPENBLAS_DIR"/lib64; do
        [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d:${LD_LIBRARY_PATH:-}" LIBRARY_PATH="$_d:${LIBRARY_PATH:-}"
    done
    [ -d "$OPENBLAS_DIR/include" ] && export CPATH="$OPENBLAS_DIR/include:${CPATH:-}"
fi
echo "### CloudSC cache build (Phase A) starting $(date)"
/usr/bin/time -v python3 cloudsc_variants/cloudsc_variants_perf.py --build-cache-only --rebuild-cache
echo "### cache file:"; ls -la cloudsc_variants/cache/ 2>/dev/null
echo CLOUDSC_CACHE_DONE
