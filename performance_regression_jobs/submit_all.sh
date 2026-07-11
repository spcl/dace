#!/bin/bash
# Submit every runtime perf-regression job as an independent SLURM job. sbatch
# returns immediately, so they queue and run concurrently (subject to the
# scheduler having free nodes). Each job's node/rank/GPU topology and account/
# partition live in its own #SBATCH header -- edit those there, not here.
#
# These are the example_slurm_*.sh jobs (the older submit_tsvc2*.sh template
# stubs never existed). Comment out any arm you don't want; add the *_compile.sh
# / *_crosscompiler.sh variants if you also want the compile-time sweeps.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

sbatch example_slurm_tsvc2.sh
sbatch example_slurm_tsvc2_5.sh
sbatch example_slurm_tsvc2_canon_vectorize.sh
sbatch example_slurm_tsvc2_5_canon_vectorize.sh
sbatch example_slurm_npbench_polybench.sh       # CPU
sbatch example_slurm_npbench_polybench_gpu.sh   # GPU (x nodes x 4 ranks x 1 GPU/rank)
