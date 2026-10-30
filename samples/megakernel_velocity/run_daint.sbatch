#!/bin/bash
#SBATCH --job-name=megakernel-velocity
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
# From this directory: sbatch -A <project> run_daint.sbatch
# Needs SPACK_ROOT with gcc and llvm 22 for aarch64 and cuda, and the python environment of the README active.
. "${SPACK_ROOT:?a spack with gcc, llvm and cuda}/share/spack/setup-env.sh"
spack load gcc@16.1.0 llvm@22.1.7 cuda
export CC=$(command -v gcc) CXX=$(command -v g++) FC=$(command -v gfortran)
# One GH200 module: one GPU and its 72 Grace cores
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=${OMP_NUM_THREADS:-72}
HERE=$SLURM_SUBMIT_DIR
MACHINE=${MACHINE:-daint}
OUT=${OUT:-$HERE/results/daint-$SLURM_JOB_ID}
. "$HERE/run_experiment.sh"
