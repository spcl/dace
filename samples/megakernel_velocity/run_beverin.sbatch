#!/bin/bash
#SBATCH --job-name=megakernel-velocity
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
# From this directory: sbatch -A <account> run_beverin.sbatch
# Needs SPACK_ROOT with gcc@16.1.0 and llvm@22, and the python environment of the README active.
. "${SPACK_ROOT:?a spack with gcc and llvm}/share/spack/setup-env.sh"
spack load gcc@16.1.0 llvm@22.1.7
export CC=$(command -v gcc) CXX=$(command -v g++) FC=$(command -v gfortran)
export ROCM_PATH=/opt/rocm CMAKE_PREFIX_PATH=/opt/rocm${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# One MI300A socket, as one GPU of Daint takes the 72 cores next to it
export ROCR_VISIBLE_DEVICES=0 OMP_NUM_THREADS=${OMP_NUM_THREADS:-24}
HERE=$SLURM_SUBMIT_DIR
MACHINE=beverin
OUT=${OUT:-$HERE/results/beverin-$SLURM_JOB_ID}
. "$HERE/run_experiment.sh"
