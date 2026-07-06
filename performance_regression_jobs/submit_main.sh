#!/bin/bash
# Submits the 3 general-regression-suite jobs (skips the canon_vectorize
# add-on experiments) as independent SLURM jobs -- sbatch returns
# immediately, so they queue and run concurrently.

sbatch example_slurm_npbench_polybench.sh
sbatch example_slurm_tsvc2.sh
sbatch example_slurm_tsvc2_5.sh
