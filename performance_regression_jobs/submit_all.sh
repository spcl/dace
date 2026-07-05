#!/bin/bash
# Submits all 5 jobs as independent SLURM jobs -- sbatch returns immediately,
# so they queue and run concurrently (subject to the scheduler having enough
# free nodes). Fill in __CPUS__ in each submit_*.sh before running this.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

sbatch submit_tsvc2.sh
sbatch submit_tsvc2_5.sh
sbatch submit_tsvc2_canon_vectorize.sh
sbatch submit_tsvc2_5_canon_vectorize.sh
sbatch submit_npbench_polybench.sh
