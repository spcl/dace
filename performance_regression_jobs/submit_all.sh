#!/bin/bash
# Submit every runtime perf-regression job as an independent SLURM job. sbatch
# returns immediately, so they queue and run concurrently (subject to the
# scheduler having free nodes). Each job's node/rank/GPU topology and account/
# partition live in its own #SBATCH header -- edit those there, not here.
#
# These are the runtime sweeps. The compile-speed sweeps live in the
# slurm_*_compile.sh jobs (each defaults to a single compiler and turns into a
# cross-compiler sweep when CXXES lists more than one) -- add them below if you
# also want compile-time numbers, e.g.:
#     sbatch slurm_tsvc2_compile.sh
#     CXXES="g++ clang++" sbatch slurm_npbench_polybench_compile.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

sbatch slurm_tsvc2.sh
sbatch slurm_tsvc2_5.sh
sbatch slurm_tsvc2_canon_vectorize.sh
sbatch slurm_tsvc2_5_canon_vectorize.sh
sbatch slurm_npbench_polybench.sh       # CPU
sbatch slurm_npbench_polybench_gpu.sh   # GPU (x nodes x 4 ranks x 1 GPU/rank)
