#!/bin/bash
# ONE command to submit EVERY perf-regression job -- runtime AND compile speed for all
# three corpora, plus the GPU and canon-vectorize sweeps. sbatch returns immediately, so
# these all queue and run concurrently (subject to free nodes). Each job's node/rank/GPU
# topology and account/partition live in its own #SBATCH header -- edit those there.
#
# Every job pins --cxx=clang++ and times --reps 25 (buffers are reused across reps and
# reset in place, never reallocated -- see engine.time_sdfg).
#
# The three *_compile.sh jobs each run BOTH the runtime sweep and the compile-speed sweep
# into the same results tree, so they SUBSUME the plain slurm_{tsvc2,tsvc2_5,
# npbench_polybench}.sh runtime jobs -- those are intentionally NOT submitted here (that
# would double-allocate nodes to redo the same runtime measurement). Use submit_all.sh
# instead if you want runtime numbers ONLY (no compile timing).
#
# Default is a single compiler (clang++). For the cross-compiler comparison, export CXXES
# with more than one before running, e.g.:  CXXES="g++ clang++" ./submit_main.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# runtime + compile speed (each compile job also does the runtime sweep)
sbatch slurm_tsvc2_compile.sh
sbatch slurm_tsvc2_5_compile.sh
sbatch slurm_npbench_polybench_compile.sh   # CPU

# GPU runtime (separate job: X nodes x 4 ranks x 1 GPU/rank)
sbatch slurm_npbench_polybench_gpu.sh

# canon-vectorize sweeps (TSVC2 / TSVC2.5)
sbatch slurm_tsvc2_canon_vectorize.sh
sbatch slurm_tsvc2_5_canon_vectorize.sh
