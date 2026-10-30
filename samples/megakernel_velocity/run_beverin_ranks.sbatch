#!/bin/bash
#SBATCH --job-name=megakernel-velocity-ranks
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --time=08:00:00
# From this directory: sbatch -A <account> run_beverin_ranks.sbatch
# One node does the work of four run_beverin.sbatch jobs: see run_rank.sh for what each rank runs.
. "${SPACK_ROOT:?a spack with gcc and llvm}/share/spack/setup-env.sh"
spack load gcc@16.1.0 llvm@22.1.7
export CC=$(command -v gcc) CXX=$(command -v g++) FC=$(command -v gfortran)
export ROCM_PATH=/opt/rocm CMAKE_PREFIX_PATH=/opt/rocm${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CHIPLETS=6 OMP_NUM_THREADS=24 BUILD_PARALLEL=${BUILD_PARALLEL:-4} MACHINE=${MACHINE:-beverin}
# ROCm also reads CUDA_VISIBLE_DEVICES, on top of this: setting both leaves ranks 2 and 3 without a GPU
export GPU_SELECT=ROCR_VISIBLE_DEVICES
export HERE=$SLURM_SUBMIT_DIR
export ROOT=${OUT:-$HERE/results/beverin-$SLURM_JOB_ID}
mkdir -p "$ROOT"
# A rank that fails leaves the other three running
srun --kill-on-bad-exit=0 --cpu-bind=none bash "$HERE/run_rank.sh"
cat "$ROOT"/rank*/verify.txt >"$ROOT/verify.txt"
awk 'FNR > 1 || NR == 1' "$ROOT"/rank*/times.csv >"$ROOT/times.csv"
