#!/bin/bash
#SBATCH --job-name=megakernel-velocity-ranks
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=04:00:00
# From this directory: sbatch -A <project> run_daint_ranks.sbatch
# One GH200 node does the work of four run_daint.sbatch jobs: see run_rank.sh for what each rank runs.
# Needs SPACK_ROOT, the python environment of the README active, and BASELINES, the SDFGs of save_baselines.py: Daint
# has no flang, so the frontend ran elsewhere. CPU_CODEGEN=legacy times the old CPU code generator, as its own machine
# in the CSV; it also emits the tasklets of GPU kernels.
. "${SPACK_ROOT:?a spack with gcc, llvm and cuda}/share/spack/setup-env.sh"
spack load gcc@16.2.0 target=neoverse_v2
spack load cuda@13.2.1
export CC=$(command -v gcc) CXX=$(command -v g++) FC=$(command -v gfortran)
# spack bakes no rpath into what this gcc links: its libstdc++ and libgomp are newer than the system's
export LD_LIBRARY_PATH=$(dirname "$($CXX -print-file-name=libstdc++.so)")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# nvcc cannot parse the libstdc++ of gcc 16, so GPU libraries take the system gcc for their host side; nvcc builds the kernels
export GPU_CXX=/usr/bin/g++-14
export CHIPLETS=1 OMP_NUM_THREADS=72 BUILD_PARALLEL=${BUILD_PARALLEL:-4}
export CPU_CODEGEN=${CPU_CODEGEN:-experimental_readable} DACE_compiler_cpu_implementation=${CPU_CODEGEN:-experimental_readable}
export MACHINE=daint
[ "$CPU_CODEGEN" = legacy ] && MACHINE=daint-legacy
export PYTHON=${PYTHON:-$VIRTUAL_ENV/bin/python} BASELINES=${BASELINES:?the SDFGs of save_baselines.py}
export HERE=$SLURM_SUBMIT_DIR
export ROOT=${OUT:-$HERE/results/$MACHINE-$SLURM_JOB_ID}
mkdir -p "$ROOT"
# A rank that fails leaves the other three running
srun --kill-on-bad-exit=0 --cpu-bind=none bash "$HERE/run_rank.sh"
cat "$ROOT"/rank*/verify.txt >"$ROOT/verify.txt"
awk 'FNR > 1 || NR == 1' "$ROOT"/rank*/times.csv >"$ROOT/times.csv"
